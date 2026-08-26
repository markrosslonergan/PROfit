/**
 * @file PROAdaptiveFCmesh.cxx
 * @brief Adaptive Feldman-Cousins: throw generation, Wilks AMR prepass, and
 *        meta-mesh aggregation.
 * @author PROfit Collaboration
 *
 * @details Part of the adaptive FC pipeline (see inc/PROAdaptiveFC.h and
 * src/PROAdaptiveFCinternal.h for the file layout). The AMR per-point fit
 * body is shared with PROsurf::FillSurfaceAMR via PROmesh::pinned_scan_eval
 * (inc/PROmeshEval.h).
 */
#include "PROAdaptiveFCinternal.h"

#include "PROlog.h"
#include "PROmeshEval.h"
#include "PROmetrics/PROchi.h"
#include "PROmetrics/PROpearson.h"
#include "PROmetrics/PROCNP.h"
#include "PROmetrics/PROpoisson.h"
#include "PROmetric.h"
#include "PROspec.h"
#include "PROcess.h"
#include "PROtocall.h"

#include <Eigen/Eigen>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace PROfit {
namespace afc {

// Truncated-Gaussian nuisance throws use the shared, bounded, OOB-safe
// ThrowRestrictedSplinePull from PROcess.h (pattern from commit 000b3d0).
static float throw_restricted_spline(const PROsyst &systs, size_t i,
                                     std::mt19937 &rng,
                                     std::normal_distribution<float> &d)
{
    return ThrowRestrictedSplinePull(systs, i, rng, d);
}

// ====================================================================
//  Section 1a — run_throw_global_fit
//
//  Minimal global fit per throw, distilled from the core of do_a_fit in
//  bin/PROfit.cxx (lines 3323-3357). Strips out progress bar, MCMC covariance,
//  and error-band side-effects — we only need the best-fit parameter vector
//  to use as a warm-start seed for the AMR level-0 evaluations.
//
//  Doing this once per throw anchors the per-throw gmin at the actual global
//  best fit and warm-starts level-0 cell evaluations, removing the cold-start
//  fitter noise that would otherwise drive false-straddle decisions.
// ====================================================================

struct ThrowGlobalFit {
    float chi2 = 0.0f;
    Eigen::VectorXf best_fit;
    bool valid = false;
};

static ThrowGlobalFit run_throw_global_fit(
    const PROconfig &config,
    const PROpeller &prop,
    const PROsyst   &systs,
    const PROmodel  &model,
    const PROdata   &data,
    const PROfitterConfig &fitconfig,
    const std::string &chi2_kind,
    bool binned,
    uint32_t seed)
{
    ThrowGlobalFit res;
    const size_t nphys   = model.nparams;
    const size_t nspline = systs.GetNSplines();
    const size_t nparams = nphys + nspline;

    PROmetric::EvalStrategy strat = binned ? PROmetric::BinnedChi2 : PROmetric::EventByEvent;
    std::unique_ptr<PROmetric> metric;
    if      (chi2_kind == "PROchi")    metric.reset(new PROchi   ("", config, prop, &systs, model, data, strat));
    else if (chi2_kind == "PROpearson") metric.reset(new PROpearson("", config, prop, &systs, model, data, strat));
    else if (chi2_kind == "PROCNP")    metric.reset(new PROCNP   ("", config, prop, &systs, model, data, strat));
    else if (chi2_kind == "Poisson")   metric.reset(new PROpoisson("", config, prop, &systs, model, data, strat));
    else {
        log<LOG_ERROR>(L"%1% || run_throw_global_fit: unknown chi2 kind '%2%'.") % __func__ % chi2_kind.c_str();
        return res;
    }

    // Full-floating bounds: physics within model lb/ub, splines within their restrict ranges.
    Eigen::VectorXf lb((int)nparams), ub((int)nparams);
    for (size_t j = 0; j < nphys; ++j) {
        lb((int)j) = model.lb(j);
        ub((int)j) = model.ub(j);
    }
    for (size_t j = nphys; j < nparams; ++j) {
        const size_t si = j - nphys;
        const float lo = systs.spline_has_restrict[si] ? systs.spline_restrict_lo[si] : systs.spline_lo[si];
        const float hi = systs.spline_has_restrict[si] ? systs.spline_restrict_hi[si] : systs.spline_hi[si];
        lb((int)j) = lo; ub((int)j) = hi;
    }
    metric->setBounds(lb, ub);

    // Initial seed at the model's per-param default for physics + zero for splines.
    Eigen::VectorXf seed_pt = Eigen::VectorXf::Zero((int)nparams);
    for (size_t j = 0; j < nphys; ++j) seed_pt((int)j) = model.default_val(j);

    PROfitter fitter(ub, lb, fitconfig, seed);
    float best_chi2 = fitter.Fit(*metric, seed_pt);
    Eigen::VectorXf best_fit = fitter.best_fit;

    // Sweep the frequency-domain harmonic seed alternatives — same trick as do_a_fit.
    fitter.calcFreqSeedPoints(*metric);
    for (size_t i = 0; i < fitter.freq_seed_points.size(); ++i) {
        const float c = fitter.freq_seed_values.at(i);
        if (c < best_chi2) {
            best_chi2 = c;
            best_fit  = fitter.freq_seed_points.at(i);
        }
    }

    res.chi2     = best_chi2;
    res.best_fit = std::move(best_fit);
    res.valid    = true;
    return res;
}

// ====================================================================
//  Section 1 — run_wilks_prepass
//
//  Shares the per-point fit body with PROsurf::FillSurfaceAMR via
//  PROmesh::pinned_scan_eval (inc/PROmeshEval.h). Only metric acquisition
//  differs: PROsurf clones a pre-built metric, while this path closes over
//  a *PROdata* (the per-throw fake dataset) and constructs a fresh
//  PROmetric per thread from (config, prop, systs, model, data).
//
//  `caller_seeds`: warm-start seeds for level-0 cell evaluations. Mirrors
//  PROsurf::FillSurfaceAMR's caller_seeds parameter — typically the per-throw
//  global-fit best_fit, so cold-start fitter noise is eliminated.
// ====================================================================

static PROmesh::AMRResult run_wilks_prepass(
    const PROconfig &config,
    const PROpeller &prop,
    const PROsyst   &systs,
    const PROmodel  &model,
    const PROdata   &data,
    const PROfitterConfig &fitconfig,
    const std::string &chi2_kind,
    bool eventbyevent,
    size_t xaxis_idx,
    size_t yaxis_idx,
    float x_lo, float x_hi,
    float y_lo, float y_hi,
    const PROmesh::AMROptions &opts_in,
    int nthreads,
    const std::vector<Eigen::VectorXf> &caller_seeds = {})
{
    PROmesh::AMROptions opts = opts_in;
    opts.nthreads = nthreads;
    // Plumb caller seeds into AMR's level-0 warm-start, matching PROsurf::FillSurfaceAMR.
    if (opts.initial_seed_points.empty() && !caller_seeds.empty()) {
        opts.initial_seed_points = caller_seeds;
    }

    // Thread-local metrics. The first call on each thread allocates;
    // subsequent calls reuse. Mirrors the acquisition in PROsurf::FillSurfaceAMR.
    auto eval_fn = [&config, &prop, &systs, &model, &data, &fitconfig,
                    chi2_kind, eventbyevent, xaxis_idx, yaxis_idx]
        (const PROmesh::EvalRequest &req) -> PROmesh::EvalResult
    {
        thread_local std::unique_ptr<PROmetric> tls_metric;
        if (!tls_metric) {
            // Construct a per-thread metric matching the chi2_kind choice.
            // The lifetime of `data`, `systs`, etc. is the lifetime of
            // run_wilks_prepass — fine because the AMR scheduler joins all
            // workers before run_amr returns.
            PROmetric::EvalStrategy strat = eventbyevent
                ? PROmetric::EventByEvent : PROmetric::BinnedChi2;
            if (chi2_kind == "PROchi") {
                tls_metric.reset(new PROchi("", config, prop, &systs, model, data, strat));
            } else if (chi2_kind == "PROpearson") {
                tls_metric.reset(new PROpearson("", config, prop, &systs, model, data, strat));
            } else if (chi2_kind == "PROCNP") {
                tls_metric.reset(new PROCNP("", config, prop, &systs, model, data, strat));
            } else if (chi2_kind == "Poisson") {
                tls_metric.reset(new PROpoisson("", config, prop, &systs, model, data, strat));
            } else {
                log<LOG_ERROR>(L"%1% || run_wilks_prepass: unknown chi2 kind '%2%'.")
                    % __func__ % chi2_kind.c_str();
                abort();
            }
        }
        return PROmesh::pinned_scan_eval(*tls_metric, fitconfig, xaxis_idx, yaxis_idx, req);
    };

    log<LOG_INFO>(L"%1% || run_wilks_prepass starting AMR on [%2%, %3%] x [%4%, %5%], "
                  L"initial=%6%x%7%, levels=%8%, nthreads=%9%.")
        % __func__ % x_lo % x_hi % y_lo % y_hi
        % opts.initial_nx % opts.initial_ny % opts.max_levels % opts.nthreads;

    PROmesh::AMRResult amr = PROmesh::run_amr(eval_fn, x_lo, x_hi, y_lo, y_hi, opts);

    log<LOG_INFO>(L"%1% || AMR done: %2% fits across %3% levels, min_chi2=%4%, %5% leaves, %6% contour levels.")
        % __func__ % amr.total_fits % opts.max_levels % amr.min_chi2
        % (int)amr.leaves.size() % (int)amr.polylines.size();

    return amr;
}

// ====================================================================
//  Section 2 — generate_throws
//
//  Mirrors the brazil-band throw loop at bin/PROfit.cxx:1582-1620 but writes
//  each throw's result into our own vector of AMRResults instead of into
//  PROsurf::surface. Honours `stat_only_throws`.
// ====================================================================

std::vector<PROmesh::AMRResult> generate_throws(
    const PROconfig &config,
    const PROpeller &prop,
    const PROsyst   &systs,
    const PROmodel  &model,
    const PROfitterConfig &fitconfig,
    PROseed &proseed,
    const Eigen::VectorXf &fakeDataParams,
    const AdaptiveFCConfig &acfg,
    int nthreads,
    size_t xaxis_idx, size_t yaxis_idx,
    MultiPROgressBar &progress)
{
    const size_t N_phys_params = model.nparams;
    const size_t nbins_collapsed = (size_t)config.m_num_variable_bins_total_collapsed[config.i_prime];

    // Build the CV spectrum and the Cholesky factor once — both are throw-independent.
    PROspec cv = FillSpectra(config, prop, systs, model, fakeDataParams, !acfg.binned ? false : true, config.i_prime);
    PROspec collapsed_cv = PROspec(CollapseMatrix(config, cv.Spec()),
                                   CollapseMatrix(config, cv.Error()));
    Eigen::MatrixXf L = systs.DecomposeFractionalCovariance(config, cv.Spec());

    // Convert acfg axis bounds back to *transformed* (log/lin) space — that's
    // what PROmesh expects and what PROsurf::FillSurfaceAMR uses (see line
    // 747-750 of PROsurf.cxx).
    const float x_lo_t = acfg.logx ? std::log10(acfg.x_lo) : acfg.x_lo;
    const float x_hi_t = acfg.logx ? std::log10(acfg.x_hi) : acfg.x_hi;
    const float y_lo_t = acfg.logy ? std::log10(acfg.y_lo) : acfg.y_lo;
    const float y_hi_t = acfg.logy ? std::log10(acfg.y_hi) : acfg.y_hi;

    PROmesh::AMROptions opts;
    opts.initial_nx     = acfg.prepass_amr_initial_x;
    opts.initial_ny     = acfg.prepass_amr_initial_y;
    opts.max_levels     = acfg.prepass_amr_levels;
    opts.delta_widen    = acfg.prepass_delta_widen;
    opts.contour_levels = acfg.prepass_contour_levels;
    opts.produce_dense  = true;

    std::vector<PROmesh::AMRResult> results;
    results.reserve(acfg.n_throws);

    // RNG plumbing — mirrors fc_worker's pattern (PROfc.cxx:36-37).
    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());

    std::normal_distribution<float> d;

    for (int t = 0; t < acfg.n_throws; ++t) {
        // Build this throw's parameter vector and stat-throw vector.
        Eigen::VectorXf throwp = fakeDataParams;
        Eigen::VectorXf throwC = Eigen::VectorXf::Constant((int)nbins_collapsed, 0);

        if (!acfg.stat_only_throws) {
            // Truncated to each spline's allowed range — keeps the mesh throws
            // statistically consistent with the PE throws in run_one_pe /
            // generate_pseudo_experiment_data.
            for (size_t i = 0; i < systs.GetNSplines(); ++i) {
                throwp((int)(i + N_phys_params)) = throw_restricted_spline(systs, i, PROseed::global_rng, d);
            }
        }
        for (size_t i = 0; i < nbins_collapsed; ++i) {
            throwC((int)i) = d(PROseed::global_rng);
        }

        const bool binned_flag = acfg.binned;
        PROspec shifted = FillSpectra(config, prop, systs, model, throwp, binned_flag, config.i_prime);

        PROspec newSpec = acfg.stat_only_throws
            ? PROspec::PoissonVariation(collapsed_cv, dseed(proseed.global_rng))
            : PROspec::PoissonVariation(
                  PROspec(CollapseMatrix(config, shifted.Spec()) + L * throwC,
                          CollapseMatrix(config, shifted.Error())),
                  dseed(proseed.global_rng));
        PROdata data(newSpec.Spec(), newSpec.Error());

        log<LOG_INFO>(L"%1% || throw %2%/%3%: stat_only=%4%, n_splines_thrown=%5%, n_stat_bins=%6%.")
            % __func__ % t % acfg.n_throws % (int)acfg.stat_only_throws
            % (int)(acfg.stat_only_throws ? 0 : systs.GetNSplines()) % (int)nbins_collapsed;

        // Per-throw global fit — used to (a) anchor gmin at the actual best
        // fit instead of letting it drift to the lucky-seed minimum over the
        // AMR cells, and (b) warm-start level-0 AMR fits to reduce cold-start
        // fitter noise. Mirrors the surface --surface-amr path
        // (bin/PROfit.cxx:1399-1506).
        const uint32_t gf_seed = dseed(proseed.global_rng);
        ThrowGlobalFit gf = run_throw_global_fit(
            config, prop, systs, model, data, fitconfig,
            acfg.chi2, acfg.binned, gf_seed);
        std::vector<Eigen::VectorXf> caller_seeds;
        if (gf.valid) {
            caller_seeds.push_back(gf.best_fit);
            log<LOG_INFO>(L"%1% || throw %2%: per-throw global fit chi2=%3%, seeding AMR.")
                % __func__ % t % gf.chi2;
        } else {
            log<LOG_WARNING>(L"%1% || throw %2%: per-throw global fit failed; AMR will cold-start.")
                % __func__ % t;
        }

        PROmesh::AMRResult amr = run_wilks_prepass(
            config, prop, systs, model, data, fitconfig,
            acfg.chi2, !acfg.binned ? true : false,
            xaxis_idx, yaxis_idx,
            x_lo_t, x_hi_t, y_lo_t, y_hi_t,
            opts, nthreads, caller_seeds);

        // ----- Per-throw chi^2 landscape diagnostics -----
        // Buckets are Δχ² = (point chi² - amr.gmin), bucket edges chosen so the
        // contour level (5.99 for 2σ at 2 dof) falls cleanly in [4, 6]. If the
        // deep basin sits in the [4, 8] bucket, the AMR is correctly seeing the
        // null hypothesis disfavoured at ~2σ — over-refinement there is genuine
        // FC structure, NOT fitter noise.
        if (!amr.chi2_map.empty()) {
            const float gmin = amr.min_chi2;
            float global_fit_chi2 = gf.valid ? gf.chi2 : std::numeric_limits<float>::quiet_NaN();
            float landscape_min =  std::numeric_limits<float>::infinity();
            float landscape_max = -std::numeric_limits<float>::infinity();
            double sum = 0.0, sum_sq = 0.0;
            int n_pts = 0;
            int buckets[6] = {0,0,0,0,0,0}; // [0,2), [2,4), [4,6), [6,8), [8,12), [12,inf)
            for (const auto &kv : amr.chi2_map) {
                const float c = kv.second;
                landscape_min = std::min(landscape_min, c);
                landscape_max = std::max(landscape_max, c);
                sum += c; sum_sq += (double)c * c;
                ++n_pts;
                const float d = c - gmin;
                int b = 0;
                if      (d <  2.0f) b = 0;
                else if (d <  4.0f) b = 1;
                else if (d <  6.0f) b = 2;
                else if (d <  8.0f) b = 3;
                else if (d < 12.0f) b = 4;
                else                b = 5;
                ++buckets[b];
            }
            const double mean = sum / (double)n_pts;
            const double var  = (sum_sq / (double)n_pts) - mean * mean;
            const double stddev = std::sqrt(std::max(0.0, var));
            log<LOG_INFO>(L"%1% || throw %2% landscape: global_fit_chi2=%3%, amr_gmin=%4%, gmax=%5%, mean=%6%, stddev=%7%, n_pts=%8%.")
                % __func__ % t % global_fit_chi2 % landscape_min % landscape_max
                % (float)mean % (float)stddev % n_pts;
            log<LOG_INFO>(L"%1% || throw %2% chi2 buckets relative to gmin: "
                          L"[0,2)=%3%, [2,4)=%4%, [4,6)=%5%, [6,8)=%6%, [8,12)=%7%, [12+)=%8%.")
                % __func__ % t % buckets[0] % buckets[1] % buckets[2] % buckets[3] % buckets[4] % buckets[5];
            log<LOG_INFO>(L"%1% || throw %2% leaves=%3%, fits_by_level=[%4%, %5%, %6%, %7%].")
                % __func__ % t % (int)amr.leaves.size()
                % amr.fits_by_level[0] % amr.fits_by_level[1]
                % amr.fits_by_level[2] % amr.fits_by_level[3];
        }

        // bestfit_map is warm-start scratch used only DURING the AMR run; it is
        // by far the largest piece of an AMRResult (one full parameter vector
        // per evaluated point) and nothing downstream (meta-mesh tally,
        // diagnostics) reads it. Free it so memory doesn't grow linearly with
        // n_throws.
        amr.bestfit_map.clear();

        results.push_back(std::move(amr));
        progress.increment_bar(0);
    }

    return results;
}

// ====================================================================
//  Section 3 — build_meta_mesh
//
//  Aggregates per-throw AMRResults into a MetaMesh by counting, at each
//  finest-grid coordinate, how many throws produced a leaf containing that
//  coordinate at refinement depth ≥ L for each L.
//
//  A cell at level L is "kept" in the meta-mesh if either
//    (a) L < baseline_level (always kept — uniform baseline), OR
//    (b) refine_count[L] / n_throws ≥ p_thresh.
// ====================================================================

MetaMesh build_meta_mesh(const std::vector<PROmesh::AMRResult> &throws,
                                float p_thresh,
                                int baseline_level)
{
    MetaMesh mm;
    if (throws.empty()) {
        log<LOG_WARNING>(L"%1% || build_meta_mesh: empty throw list.") % __func__;
        return mm;
    }

    // Sanity: all throws must share the same finest grid + AMR depth + bounds.
    mm.finest_nx  = throws.front().finest_nx;
    mm.finest_ny  = throws.front().finest_ny;
    mm.x_lo = throws.front().x_lo;  mm.x_hi = throws.front().x_hi;
    mm.y_lo = throws.front().y_lo;  mm.y_hi = throws.front().y_hi;
    for (const auto &amr : throws) {
        if (amr.finest_nx != mm.finest_nx || amr.finest_ny != mm.finest_ny ||
            amr.max_levels != throws.front().max_levels) {
            log<LOG_ERROR>(L"%1% || build_meta_mesh: throws disagree on finest grid / AMR depth (%2%x%3% L%4% vs %5%x%6% L%7%).")
                % __func__ % amr.finest_nx % amr.finest_ny % amr.max_levels
                % mm.finest_nx % mm.finest_ny % throws.front().max_levels;
            return mm;
        }
    }
    // Use the AMR configuration depth, NOT the deepest level any throw
    // happened to reach: the finest grid is always initial × 2^max_levels, so
    // deriving max_levels from observed leaves mis-sizes every cell whenever
    // no throw refined to full depth.
    mm.max_levels = throws.front().max_levels;
    if (mm.finest_nx % (1 << mm.max_levels) != 0 || mm.finest_ny % (1 << mm.max_levels) != 0) {
        log<LOG_ERROR>(L"%1% || build_meta_mesh: finest grid %2%x%3% is not a multiple of 2^max_levels (L%4%). This is a bug.")
            % __func__ % mm.finest_nx % mm.finest_ny % mm.max_levels;
        return mm;
    }
    const int n_levels = mm.max_levels + 1;
    const int n_throws = (int)throws.size();

    // For each finest-grid point (i, j), count refinement-depth tallies across throws.
    // Storage: dense grid (finest_nx * finest_ny) of vector<int> with n_levels slots.
    // For SBN-class grids (typical finest=80x80 with max_levels=3, n_levels=4) this is
    // 80*80*4 = 25600 ints = ~100 KB. Acceptable.
    const size_t W = (size_t)mm.finest_nx;
    const size_t H = (size_t)mm.finest_ny;
    std::vector<std::vector<int>> tally(W * H, std::vector<int>(n_levels, 0));

    // Each leaf in a throw covers a square region in finest-integer coords from
    // (i_bl, j_bl) to (i_bl + step, j_bl + step). For depth tallying, we credit
    // every finest-grid point inside that square with "refined to depth = leaf.level".
    // A cell at depth L was produced because the throw's AMR refined depth (L-1),
    // (L-2), ..., 0 to reach this point — so we credit all levels ≤ leaf.level.
    for (const auto &amr : throws) {
        for (const auto &leaf : amr.leaves) {
            const int i0 = std::max(0, leaf.i_bl);
            const int j0 = std::max(0, leaf.j_bl);
            const int i1 = std::min((int)W, leaf.i_bl + leaf.step);
            const int j1 = std::min((int)H, leaf.j_bl + leaf.step);
            for (int i = i0; i < i1; ++i) {
                for (int j = j0; j < j1; ++j) {
                    auto &v = tally[(size_t)i * H + (size_t)j];
                    for (int L = 0; L <= leaf.level && L < n_levels; ++L) v[L] += 1;
                }
            }
        }
    }

    // Build MetaCells by top-down quadtree descent so the result is an EXACT
    // tiling of the finest grid — no holes, no overlaps. (The previous
    // two-pass anchor-based emission left uncovered finest points inside
    // partially-refined baseline blocks — regions with no PE bank cell,
    // typically right along contours — and could also emit a coarser cell
    // overlapping an already-emitted deeper one.)
    //
    // Descent rule at a block (i, j, level): subdivide iff any finest point in
    // the block was refined to depth ≥ level+1 by at least threshold_count
    // throws; otherwise emit the block as a leaf. Leaves at level ≥
    // baseline_level are "refined" cells, coarser leaves are baseline cells.
    const int threshold_count = std::max(1, (int)std::ceil(p_thresh * (float)n_throws));

    std::vector<MetaCell> emitted;
    emitted.reserve(W * H / 4);
    std::vector<uint8_t> covered(W * H, 0); // tiling sanity check

    // Peak tally over a block footprint at a given level.
    auto block_peak_at_level = [&](int i, int j, int step, int L) -> int {
        int peak = 0;
        for (int ii = i; ii < i + step && ii < (int)W; ++ii)
            for (int jj = j; jj < j + step && jj < (int)H; ++jj)
                peak = std::max(peak, tally[(size_t)ii * H + (size_t)jj][L]);
        return peak;
    };

    std::function<void(int, int, int)> descend = [&](int i, int j, int level) {
        const int step = 1 << std::max(0, mm.max_levels - level);
        const bool can_refine = level < mm.max_levels && step >= 2 &&
                                (level + 1) < n_levels &&
                                block_peak_at_level(i, j, step, level + 1) >= threshold_count;
        if (can_refine) {
            const int half = step / 2;
            descend(i,        j,        level + 1);
            descend(i + half, j,        level + 1);
            descend(i,        j + half, level + 1);
            descend(i + half, j + half, level + 1);
            return;
        }

        MetaCell c;
        c.i_bl = i; c.j_bl = j; c.step = step; c.level = level;
        c.per_level_refine_count.assign(n_levels, 0);
        for (int L = 0; L < n_levels; ++L)
            c.per_level_refine_count[L] = block_peak_at_level(i, j, step, L);
        emitted.push_back(std::move(c));
        if (level >= baseline_level) mm.n_refined_cells++;
        else                         mm.n_baseline_cells++;
        for (int ii = i; ii < i + step && ii < (int)W; ++ii)
            for (int jj = j; jj < j + step && jj < (int)H; ++jj)
                covered[(size_t)ii * H + (size_t)jj]++;
    };

    // Top-level sweep at the baseline tiling level. W and H are multiples of
    // every power-of-two step ≤ 2^max_levels, so the blocks tile exactly.
    {
        const int level_b = std::max(0, baseline_level - 1);
        const int step_b = 1 << std::max(0, mm.max_levels - level_b);
        for (int i = 0; i < (int)W; i += step_b)
            for (int j = 0; j < (int)H; j += step_b)
                descend(i, j, level_b);
    }

    // Tiling assertion: every finest point covered exactly once.
    size_t n_holes = 0, n_overlaps = 0;
    for (size_t p = 0; p < covered.size(); ++p) {
        if (covered[p] == 0) ++n_holes;
        else if (covered[p] > 1) ++n_overlaps;
    }
    if (n_holes || n_overlaps) {
        log<LOG_ERROR>(L"%1% || build_meta_mesh: tiling invariant violated (%2% holes, %3% overlaps of %4% finest points). This is a bug.")
            % __func__ % (int)n_holes % (int)n_overlaps % (int)covered.size();
    }

    mm.cells = std::move(emitted);

    log<LOG_INFO>(L"%1% || build_meta_mesh: aggregated %2% throws, finest=%3%x%4%, "
                  L"meta_cells=%5% (baseline=%6%, refined=%7%), threshold_count=%8%/%9%.")
        % __func__ % n_throws % mm.finest_nx % mm.finest_ny
        % (int)mm.cells.size() % mm.n_baseline_cells % mm.n_refined_cells
        % threshold_count % n_throws;

    return mm;
}

// Union-merge N meta-meshes (same finest grid + bounds, caller-validated)
// into the coarsest tiling that refines every input. Aligned power-of-two
// cells from one finest grid can only be identical, nested, or disjoint, so
// the union is canonical: a block is a leaf iff NO input subdivides it.
// Because a block's containing leaf in each input covers the whole block
// whenever its step >= the block's (alignment), the check at the block's
// bottom-left bin is sufficient.
//
// max_levels of the output = max over inputs; per_level_refine_count is
// carried as the element-wise max of the inputs' covering-leaf counters
// (index-shifted where inputs had smaller max_levels) — diagnostics only.
// baseline_level is used only for the baseline/refined log counters.
MetaMesh merge_meta_meshes(const std::vector<MetaMesh> &inputs, int baseline_level)
{
    MetaMesh out;
    const MetaMesh &ref = inputs.front();
    out.finest_nx = ref.finest_nx;
    out.finest_ny = ref.finest_ny;
    out.x_lo = ref.x_lo; out.x_hi = ref.x_hi;
    out.y_lo = ref.y_lo; out.y_hi = ref.y_hi;
    out.max_levels = 0;
    for (const auto &mm : inputs) out.max_levels = std::max(out.max_levels, mm.max_levels);

    const int W = out.finest_nx, H = out.finest_ny;
    const int n_levels_out = out.max_levels + 1;

    // Per-input finest-bin → cell-index lookup (same layout as build_meta_mesh).
    std::vector<std::vector<int>> id_at(inputs.size(),
                                        std::vector<int>((size_t)W * (size_t)H, -1));
    for (size_t m = 0; m < inputs.size(); ++m) {
        const auto &cells = inputs[m].cells;
        for (size_t c = 0; c < cells.size(); ++c) {
            const auto &mc = cells[c];
            for (int ii = mc.i_bl; ii < mc.i_bl + mc.step && ii < W; ++ii)
                for (int jj = mc.j_bl; jj < mc.j_bl + mc.step && jj < H; ++jj)
                    id_at[m][(size_t)ii * (size_t)H + (size_t)jj] = (int)c;
        }
    }

    std::vector<uint8_t> covered((size_t)W * (size_t)H, 0);
    bool hole_in_input = false;

    std::function<void(int, int, int)> descend = [&](int i, int j, int level) {
        const int step = 1 << std::max(0, out.max_levels - level);
        bool any_finer = false;
        for (size_t m = 0; m < inputs.size(); ++m) {
            const int cid = id_at[m][(size_t)i * (size_t)H + (size_t)j];
            if (cid < 0) { hole_in_input = true; continue; }
            if (inputs[m].cells[(size_t)cid].step < step) { any_finer = true; break; }
        }
        if (any_finer && level < out.max_levels && step >= 2) {
            const int half = step / 2;
            descend(i,        j,        level + 1);
            descend(i + half, j,        level + 1);
            descend(i,        j + half, level + 1);
            descend(i + half, j + half, level + 1);
            return;
        }

        MetaCell c;
        c.i_bl = i; c.j_bl = j; c.step = step; c.level = level;
        c.per_level_refine_count.assign((size_t)n_levels_out, 0);
        for (size_t m = 0; m < inputs.size(); ++m) {
            const int cid = id_at[m][(size_t)i * (size_t)H + (size_t)j];
            if (cid < 0) continue;
            const auto &counts = inputs[m].cells[(size_t)cid].per_level_refine_count;
            const int shift = out.max_levels - inputs[m].max_levels;
            for (size_t L = 0; L < counts.size(); ++L) {
                const size_t Lo = L + (size_t)shift;
                if (Lo < (size_t)n_levels_out)
                    c.per_level_refine_count[Lo] =
                        std::max(c.per_level_refine_count[Lo], counts[L]);
            }
        }
        out.cells.push_back(std::move(c));
        if (level >= baseline_level) out.n_refined_cells++;
        else                         out.n_baseline_cells++;
        for (int ii = i; ii < i + step && ii < W; ++ii)
            for (int jj = j; jj < j + step && jj < H; ++jj)
                covered[(size_t)ii * (size_t)H + (size_t)jj]++;
    };

    const int S0 = 1 << out.max_levels;
    for (int i = 0; i < W; i += S0)
        for (int j = 0; j < H; j += S0)
            descend(i, j, 0);

    size_t n_holes = 0, n_overlaps = 0;
    for (size_t p = 0; p < covered.size(); ++p) {
        if (covered[p] == 0) ++n_holes;
        else if (covered[p] > 1) ++n_overlaps;
    }
    if (n_holes || n_overlaps || hole_in_input) {
        log<LOG_ERROR>(L"%1% || merge_meta_meshes: tiling invariant violated "
                       L"(%2% holes, %3% overlaps, input_hole=%4%). This is a bug or a corrupt input mesh.")
            % __func__ % (int)n_holes % (int)n_overlaps % (int)hole_in_input;
        out.cells.clear();
        return out;
    }

    log<LOG_INFO>(L"%1% || merge_meta_meshes: %2% inputs -> %3% cells "
                  L"(baseline=%4%, refined=%5%), finest=%6%x%7%, max_levels=%8%.")
        % __func__ % (int)inputs.size() % (int)out.cells.size()
        % out.n_baseline_cells % out.n_refined_cells
        % W % H % out.max_levels;
    return out;
}

// Build a meta-mesh from a finest-grid flag map (row-major [i * H + j], the
// shared bin-index convention): any block containing >= 1 flagged bin is
// subdivided down to the finest level, so step-1 cells appear exactly over
// the flagged bins (plus their enclosing quadtree path) and everything else
// collapses to the coarsest tiling. Same finest grid + bounds contract as
// merge_meta_meshes, so the result union-merges with any sibling mesh of the
// same footprint. Used by --mode brazil-cleanup to densify around the Brazil
// ±2σ contours. per_level_refine_count is left zeroed (no throw tallies
// exist for a derived mesh); baseline_level only affects the log counters.
MetaMesh build_mesh_from_flags(int finest_nx, int finest_ny, int max_levels,
                               float x_lo, float x_hi, float y_lo, float y_hi,
                               const std::vector<uint8_t> &flags,
                               int baseline_level)
{
    MetaMesh out;
    out.finest_nx = finest_nx;
    out.finest_ny = finest_ny;
    out.max_levels = max_levels;
    out.x_lo = x_lo; out.x_hi = x_hi;
    out.y_lo = y_lo; out.y_hi = y_hi;

    const int W = finest_nx, H = finest_ny;
    const int n_levels_out = max_levels + 1;

    auto any_flag = [&](int i, int j, int step) -> bool {
        for (int ii = i; ii < i + step && ii < W; ++ii)
            for (int jj = j; jj < j + step && jj < H; ++jj)
                if (flags[(size_t)ii * (size_t)H + (size_t)jj]) return true;
        return false;
    };

    std::function<void(int, int, int)> descend = [&](int i, int j, int level) {
        const int step = 1 << std::max(0, max_levels - level);
        if (level < max_levels && step >= 2 && any_flag(i, j, step)) {
            const int half = step / 2;
            descend(i,        j,        level + 1);
            descend(i + half, j,        level + 1);
            descend(i,        j + half, level + 1);
            descend(i + half, j + half, level + 1);
            return;
        }
        MetaCell c;
        c.i_bl = i; c.j_bl = j; c.step = step; c.level = level;
        c.per_level_refine_count.assign((size_t)n_levels_out, 0);
        out.cells.push_back(std::move(c));
        if (level >= baseline_level) out.n_refined_cells++;
        else                         out.n_baseline_cells++;
    };

    const int S0 = 1 << max_levels;
    for (int i = 0; i < W; i += S0)
        for (int j = 0; j < H; j += S0)
            descend(i, j, 0);

    log<LOG_INFO>(L"%1% || build_mesh_from_flags: %2% flagged bins -> %3% cells "
                  L"(baseline=%4%, refined=%5%), finest=%6%x%7%, max_levels=%8%.")
        % __func__ % (int)std::count(flags.begin(), flags.end(), (uint8_t)1)
        % (int)out.cells.size() % out.n_baseline_cells % out.n_refined_cells
        % W % H % max_levels;
    return out;
}

// Compute per-cell (x, y) coords for every MetaCell — Option A: PE bank at
// each cell center.
//
// Returns values in *model space* — i.e. log10(physical) for log-axis params
// and linear for lin-axis params. This matches the convention used by
// FillSpectra / fc_worker / PROsurf::FillSurfaceAMR's EvalFn: the model
// always wants log10(sinsq2thmm) for a log-axis sinsq2thmm. Display code
// (slice-1 diagnostic plots) applies pow(10) when needed for ROOT axes.
//
// The xlog/ylog arguments are unused now but kept in the signature for
// readability — if we ever need physical-coord output again it goes here.
void compute_cell_centers(const MetaMesh &mm,
                                 bool /*xlog*/, bool /*ylog*/,
                                 std::vector<float> &cx_out,
                                 std::vector<float> &cy_out)
{
    const int W = mm.finest_nx;
    const int H = mm.finest_ny;
    cx_out.clear(); cy_out.clear();
    cx_out.reserve(mm.cells.size());
    cy_out.reserve(mm.cells.size());
    for (const auto &c : mm.cells) {
        const float fi = (float)c.i_bl + 0.5f * (float)c.step;
        const float fj = (float)c.j_bl + 0.5f * (float)c.step;
        cx_out.push_back(mm.x_lo + fi / (float)W * (mm.x_hi - mm.x_lo));
        cy_out.push_back(mm.y_lo + fj / (float)H * (mm.y_hi - mm.y_lo));
    }
}

// --------------------------------------------------------------------
//  Pseudo-experiment data generator.
//
//  One FC-style realisation of the data: spline Gaussian pulls (rejection-
//  sampled within restrict bounds) + covariance-systematic bin shifts via
//  Cholesky factor + Poisson stats variation. Mirrors the per-PE body of
//  src/PROfc.cxx::fc_worker and the --pseudo-experiment branch in
//  bin/PROfit.cxx:986-1024.
//
//  Used by --mode brazil to generate one fake-data realisation per throw.
// --------------------------------------------------------------------
// The CV spectrum and its covariance decomposition (an SVD) are
// throw-INVARIANT: the caller computes them once and passes L_chol in,
// instead of redoing the factorization inside every brazil throw.
PROdata generate_pseudo_experiment_data(
    const PROconfig &config,
    const PROpeller &prop,
    const PROsyst   &systs,
    const PROmodel  &model,
    const Eigen::VectorXf &fakeDataParams,
    bool binned,
    const Eigen::MatrixXf &L_chol,
    PROseed &proseed)
{
    const size_t nphys   = model.nparams;
    const size_t nspline = systs.GetNSplines();

    std::normal_distribution<float> d;
    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());

    Eigen::VectorXf throws = fakeDataParams;
    for (size_t i = 0; i < nspline; ++i) {
        throws((int)(i + nphys)) = throw_restricted_spline(systs, i, PROseed::global_rng, d);
    }

    const int nbins_coll = config.m_num_variable_bins_total_collapsed[config.i_prime];
    Eigen::VectorXf throwC(nbins_coll);
    for (int i = 0; i < nbins_coll; ++i) throwC(i) = d(PROseed::global_rng);

    PROspec shifted = FillSpectra(config, prop, systs, model, throws, binned, config.i_prime);
    PROspec newSpec = PROspec::PoissonVariation(
        PROspec(CollapseMatrix(config, shifted.Spec(), config.i_prime) + L_chol * throwC,
                CollapseMatrix(config, shifted.Error(), config.i_prime)),
        dseed(proseed.global_rng));
    return PROdata(newSpec.Spec(), newSpec.Error());
}

} // namespace afc
} // namespace PROfit
