#include "PRObench.h"

#include "PROlog.h"
#include "PROdata.h"
#include "PROmetric.h"
#include "PROmetrics/PROchi.h"
#include "PROmodel.h"
#include "PROsyst.h"
#include "PROpeller.h"
#include "PROcess.h"
#include "PROspec.h"
#include "PROtocall.h"
#include "PROMCMC.h"

#include <Eigen/Eigen>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <cmath>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace PROfit {
namespace PRObench {

namespace {

// Build a random parameter vector. `vary_phys` and `vary_nuis` independently
// control which slice is randomised; the other slice is held at the
// model/spline central value (CV = model.default_val for physics, 0 for
// splines).  Physics is sampled uniform within the model's [lb, ub] range
// and nuisance is sampled N(0, 1) which is the natural pull range for splines.
//
// Some PROmodel definitions use ±infinity sentinels for lb/ub (e.g.
// PROsterileMuMu sets `lb = -infinity` for sin²2θ). Feeding ±∞ to
// std::uniform_real_distribution is undefined behaviour and silently
// produces NaN samples which then propagate to FillSpectra and blow up the
// metric. We clamp to a wide but finite window around default_val so the
// bench is robust to those sentinel bounds without asking the model
// definition to change.
Eigen::VectorXf draw_params(const PROmodel &model, const PROsyst &syst,
                            std::mt19937 &rng,
                            bool vary_phys, bool vary_nuis)
{
    const size_t nphys = model.nparams;
    const size_t nnuis = syst.GetNSplines();
    Eigen::VectorXf p = Eigen::VectorXf::Zero(nphys + nnuis);

    constexpr float kFiniteWindow = 4.0f; // ± of default_val when bound is ±inf
    for (size_t i = 0; i < nphys; ++i) {
        const float dv = model.default_val(i);
        float lb = model.lb(i);
        float ub = model.ub(i);
        if (!std::isfinite(lb)) lb = dv - kFiniteWindow;
        if (!std::isfinite(ub)) ub = dv + kFiniteWindow;
        if (vary_phys && ub > lb) {
            std::uniform_real_distribution<float> u(lb, ub);
            p(i) = u(rng);
        } else {
            p(i) = dv;
        }
    }
    if (vary_nuis) {
        std::normal_distribution<float> g(0.0f, 1.0f);
        for (size_t i = 0; i < nnuis; ++i) p(nphys + i) = g(rng);
    }
    return p;
}

// Pre-generate `n` parameter vectors so the timed loop only measures the hot
// path, not RNG/allocation noise.
std::vector<Eigen::VectorXf> draw_param_set(const PROmodel &model, const PROsyst &syst,
                                            int n, uint32_t seed,
                                            bool vary_phys, bool vary_nuis)
{
    std::mt19937 rng(seed);
    std::vector<Eigen::VectorXf> out;
    out.reserve(n);
    for (int k = 0; k < n; ++k) out.push_back(draw_params(model, syst, rng, vary_phys, vary_nuis));
    return out;
}

// One greppable LOG line. Bash wrappers should grep for the literal
// "[SCALETEST]" prefix. Microseconds suit FillSpectra/metric/collapse/pseudo
// because per-call costs are O(10²–10⁴) µs; the fit benchmark uses seconds
// (per-call cost is O(10⁵–10⁶) µs ≈ tenths-of-a-second to several seconds)
// and switches the column suffix to `_s` so a parsing wrapper can
// distinguish without a unit-conversion guess.
void emit_result(const BenchResult &r, bool seconds = false) {
    if (seconds) {
        const double total_s = r.total_us    * 1e-6;
        const double per_s   = r.per_call_us * 1e-6;
        log<LOG_INFO>(L"[SCALETEST] tag=%1% N=%2% nbins=%3% nphys=%4% nnuis=%5% total_s=%6% per_call_s=%7%")
            % r.tag.c_str() % r.n_calls % r.nbins % r.nphys % r.nnuis
            % total_s % per_s;
    } else {
        log<LOG_INFO>(L"[SCALETEST] tag=%1% N=%2% nbins=%3% nphys=%4% nnuis=%5% total_us=%6% per_call_us=%7%")
            % r.tag.c_str() % r.n_calls % r.nbins % r.nphys % r.nnuis
            % static_cast<long long>(r.total_us)
            % r.per_call_us;
    }
}

BenchResult time_fillspectra(const std::string &tag,
                             const PROconfig &config,
                             const PROpeller &prop,
                             const PROsyst &syst,
                             const PROmodel &model,
                             const std::vector<Eigen::VectorXf> &params,
                             bool binned)
{
    BenchResult r;
    r.tag = tag;
    r.n_calls = static_cast<int>(params.size());
    r.nphys = static_cast<int>(model.nparams);
    r.nnuis = static_cast<int>(syst.GetNSplines());
    r.nbins = static_cast<int>(config.m_num_variable_bins_total[config.i_prime]);

    // Use the cached FillSpectra overload — same path PROchi takes internally
    // via its `fs_cache` member. Without this, repeated calls with fixed
    // nuisance (the vary_phys mode) would re-do the full systw factor build
    // each call and the cache-hit speedup would be invisible. With caching,
    // vary_phys hits the "phys_changed only" branch where systw is reused
    // and only the physics-result is recomputed.
    FillSpectraCache cache;

    double sink = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (const auto &p : params) {
        PROspec s = FillSpectra(config, prop, syst, model, p, cache, binned, config.i_prime);
        sink += static_cast<double>(s.Spec().sum());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.total_us    = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    r.per_call_us = (r.n_calls > 0) ? r.total_us / r.n_calls : 0.0;

    log<LOG_DEBUG>(L"%1% || (sink=%2% -- anti-DCE only)") % __func__ % sink;
    emit_result(r);
    return r;
}

BenchResult time_metric(const std::string &tag,
                        const PROconfig &config,
                        PROmetric &metric,
                        const std::vector<Eigen::VectorXf> &params,
                        bool with_gradient)
{
    const PROmodel &model = metric.GetModel();
    const PROsyst  &syst  = metric.GetSysts();

    BenchResult r;
    r.tag = tag;
    r.n_calls = static_cast<int>(params.size());
    r.nphys = static_cast<int>(model.nparams);
    r.nnuis = static_cast<int>(syst.GetNSplines());
    r.nbins = static_cast<int>(config.m_num_variable_bins_total[config.i_prime]);

    // The gradient path reads metric.lb / metric.ub for boundary handling
    // (see the FD loop in PROcovariance::operator()). Without setBounds() those vectors are
    // size 0 and the index access asserts under Eigen. Mirror the standard
    // PROfit chain pattern (proglobal / profile / surface).
    if (with_gradient) {
        metric.setBounds(metric.LowerBound(), metric.UpperBound());
    }

    Eigen::VectorXf grad = Eigen::VectorXf::Zero(metric.nParams());

    double sink = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (const auto &p : params) {
        sink += static_cast<double>(metric(p, grad, with_gradient));
        if (with_gradient) sink += static_cast<double>(grad.sum());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.total_us    = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    r.per_call_us = (r.n_calls > 0) ? r.total_us / r.n_calls : 0.0;

    log<LOG_DEBUG>(L"%1% || (sink=%2% -- anti-DCE only)") % __func__ % sink;
    emit_result(r);
    return r;
}

BenchResult time_fit(const std::string &tag,
                     const PROconfig &config,
                     PROmetric &metric,
                     const std::vector<Eigen::VectorXf> &seed_params,
                     const PROfitterConfig &fitconfig,
                     uint32_t base_seed)
{
    const PROmodel &model = metric.GetModel();
    const PROsyst  &syst  = metric.GetSysts();

    BenchResult r;
    r.tag = tag;
    r.n_calls = static_cast<int>(seed_params.size());
    r.nphys = static_cast<int>(model.nparams);
    r.nnuis = static_cast<int>(syst.GetNSplines());
    r.nbins = static_cast<int>(config.m_num_variable_bins_total[config.i_prime]);

    // Fit() drives the LBFGS gradient path which reads metric.lb / metric.ub /
    // metric.is_fixed for boundary-aware finite-difference logic. Those arrays
    // are populated only by setBounds() — without it, the Eigen vectors are
    // size 0 and lb(i)/ub(i) inside PROchi::operator() trips the index
    // assertion. Mirror the standard PROfit.cxx pattern (e.g. proglobal /
    // profile / surface): construct the fitter AND call setBounds.
    const Eigen::VectorXf bench_ub = metric.UpperBound();
    const Eigen::VectorXf bench_lb = metric.LowerBound();
    metric.setBounds(bench_lb, bench_ub);

    double sink = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < r.n_calls; ++k) {
        PROfitter fitter(bench_ub, bench_lb, fitconfig,
                         base_seed + static_cast<uint32_t>(k));
        sink += static_cast<double>(fitter.Fit(metric, seed_params[k]));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.total_us    = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    r.per_call_us = (r.n_calls > 0) ? r.total_us / r.n_calls : 0.0;

    log<LOG_DEBUG>(L"%1% || (sink=%2% -- anti-DCE only)") % __func__ % sink;
    emit_result(r, /*seconds=*/true);
    return r;
}

// (h) Pseudo-universe throw — mirrors the canonical PROfc / brazil-band loop:
// random spline shifts (with rejection sampling against per-spline ranges) +
// MVN systematic noise via the Cholesky factor L of the fractional covariance
// + per-bin Poisson stat fluctuation. The Cholesky decompose is done ONCE
// outside the timed loop, matching production code where L is amortised
// across all throws.
BenchResult time_pseudo_universe(const std::string &tag,
                                 const PROconfig &config,
                                 const PROpeller &prop,
                                 const PROsyst &syst,
                                 const PROmodel &model,
                                 int n_throws,
                                 uint32_t seed,
                                 bool binned)
{
    BenchResult r;
    r.tag = tag;
    r.n_calls = n_throws;
    r.nphys = static_cast<int>(model.nparams);
    r.nnuis = static_cast<int>(syst.GetNSplines());
    r.nbins = static_cast<int>(config.m_num_variable_bins_total[config.i_prime]);

    const size_t nphys = model.nparams;
    const size_t nnuis = syst.GetNSplines();
    const size_t ncoll = config.m_num_variable_bins_total_collapsed[config.i_prime];

    // CV spectrum + Cholesky factor — one-shot setup (NOT inside the timed loop).
    Eigen::VectorXf cv_params = Eigen::VectorXf::Zero(nphys + nnuis);
    for (size_t i = 0; i < nphys; ++i) cv_params(i) = model.default_val(i);
    PROspec cv = FillSpectra(config, prop, syst, model, cv_params, binned, config.i_prime);
    Eigen::MatrixXf L = syst.DecomposeFractionalCovariance(config, cv.Spec());

    // Pre-build all throw vectors so RNG cost isn't on the timed hot path.
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());

    std::vector<Eigen::VectorXf> throwp_set(n_throws);
    std::vector<Eigen::VectorXf> throwC_set(n_throws);
    std::vector<uint32_t>        poisson_seeds(n_throws);
    for (int u = 0; u < n_throws; ++u) {
        Eigen::VectorXf throwp(nphys + nnuis);
        for (size_t i = 0; i < nphys; ++i) throwp(i) = model.default_val(i);
        // Rejection sample each spline shift inside its allowed range — same
        // policy as PROfc::fc_worker.
        for (size_t i = 0; i < nnuis; ++i) {
            const float lo = syst.spline_has_restrict[i] ? syst.spline_restrict_lo[i] : syst.spline_lo[i];
            const float hi = syst.spline_has_restrict[i] ? syst.spline_restrict_hi[i] : syst.spline_hi[i];
            float v;
            do { v = d(rng); } while (v < lo || v > hi);
            throwp(nphys + i) = v;
        }
        Eigen::VectorXf throwC(ncoll);
        for (size_t i = 0; i < ncoll; ++i) throwC(i) = d(rng);
        throwp_set[u]    = std::move(throwp);
        throwC_set[u]    = std::move(throwC);
        poisson_seeds[u] = dseed(rng);
    }

    double sink = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int u = 0; u < n_throws; ++u) {
        PROspec shifted = FillSpectra(config, prop, syst, model, throwp_set[u], binned, config.i_prime);
        PROspec newSpec = PROspec::PoissonVariation(
            PROspec(CollapseMatrix(config, shifted.Spec()) + L * throwC_set[u],
                    CollapseMatrix(config, shifted.Error())),
            poisson_seeds[u]);
        sink += static_cast<double>(newSpec.Spec().sum());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.total_us    = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    r.per_call_us = (r.n_calls > 0) ? r.total_us / r.n_calls : 0.0;

    log<LOG_DEBUG>(L"%1% || (sink=%2% -- anti-DCE only)") % __func__ % sink;
    emit_result(r);
    return r;
}

// (m, n) Metropolis MCMC step — proposal + target eval + accept/reject.
// Two flavours:
//   burnin_mode=true  → tune_mode=true on the adaptive_proposal; every step()
//                        recomputes the proposal Cholesky (sub_L) inside
//                        operator() and we call proposal.tune(accepted) which
//                        updates running mean/cov + adapts the scale every
//                        adapt_window steps. Mirrors lines 64–69 of PROMCMC.h.
//   burnin_mode=false → tune_mode=false; the cached sub_L from warmup is
//                        reused, tune() is not called. Mirrors lines 73–77.
// We use the same (target, proposal) pair production runs use (simple_target +
// adaptive_proposal), so the metric inside is the live one — its fs_cache is
// active. Initial point = CV physics + zero nuisance.
BenchResult time_mcmc_step(const std::string &tag,
                           const PROconfig &config,
                           PROmetric &metric,
                           int n_steps,
                           bool burnin_mode,
                           uint32_t seed)
{
    const PROmodel &model = metric.GetModel();
    const PROsyst  &syst  = metric.GetSysts();

    BenchResult r;
    r.tag = tag;
    r.n_calls = n_steps;
    r.nphys = static_cast<int>(model.nparams);
    r.nnuis = static_cast<int>(syst.GetNSplines());
    r.nbins = static_cast<int>(config.m_num_variable_bins_total[config.i_prime]);

    // The metric's lb/ub are read by PROchi gradient code; not reached here
    // (target uses rungradient=false), but harmless and matches production.
    metric.setBounds(metric.LowerBound(), metric.UpperBound());

    const size_t nphys = model.nparams;
    const size_t nnuis = syst.GetNSplines();
    Eigen::VectorXf initial = Eigen::VectorXf::Zero(nphys + nnuis);
    for (size_t i = 0; i < nphys; ++i) initial(i) = model.default_val(i);

    Metropolis<simple_target, adaptive_proposal> mh(
        simple_target{metric},
        adaptive_proposal(metric, seed, /*fixed=*/{}),
        initial, seed,
        /*save_chain=*/false);

    if (!burnin_mode) {
        // Warm up the proposal so sub_L is populated and cov is non-trivial,
        // then freeze tune_mode. 200 warmup steps is enough to populate
        // adaptive state without dominating bench setup time.
        constexpr int kWarmupSteps = 200;
        for (int i = 0; i < kWarmupSteps; ++i) {
            bool acc = mh.step();
            mh.proposal.tune(acc);
        }
        mh.proposal.tune_mode = false;
    }

    int naccept = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_steps; ++i) {
        bool acc = mh.step();
        if (burnin_mode) mh.proposal.tune(acc);
        if (acc) ++naccept;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.total_us    = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    r.per_call_us = (r.n_calls > 0) ? r.total_us / r.n_calls : 0.0;

    log<LOG_DEBUG>(L"%1% || (acceptance=%2%/%3% -- anti-DCE info)") % __func__ % naccept % n_steps;
    emit_result(r);
    return r;
}

// (i) CollapseMatrix() — bin-collapsing hot path. Times the vector overload
// on a fixed CV spectrum; isolates the collapse cost from FillSpectra and
// the chi² covariance solve.
BenchResult time_collapse(const std::string &tag,
                          const PROconfig &config,
                          const PROpeller &prop,
                          const PROsyst &syst,
                          const PROmodel &model,
                          int n_calls,
                          bool binned)
{
    BenchResult r;
    r.tag = tag;
    r.n_calls = n_calls;
    r.nphys = static_cast<int>(model.nparams);
    r.nnuis = static_cast<int>(syst.GetNSplines());
    r.nbins = static_cast<int>(config.m_num_variable_bins_total[config.i_prime]);

    const size_t nphys = model.nparams;
    const size_t nnuis = syst.GetNSplines();
    Eigen::VectorXf cv_params = Eigen::VectorXf::Zero(nphys + nnuis);
    for (size_t i = 0; i < nphys; ++i) cv_params(i) = model.default_val(i);
    PROspec cv = FillSpectra(config, prop, syst, model, cv_params, binned, config.i_prime);
    Eigen::VectorXf full_spec = cv.Spec();

    double sink = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < n_calls; ++k) {
        Eigen::VectorXf collapsed = CollapseMatrix(config, full_spec);
        sink += static_cast<double>(collapsed.sum());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.total_us    = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    r.per_call_us = (r.n_calls > 0) ? r.total_us / r.n_calls : 0.0;

    log<LOG_DEBUG>(L"%1% || (sink=%2% -- anti-DCE only)") % __func__ % sink;
    emit_result(r);
    return r;
}

// (o) Gradient cross-validation. Evaluates the gradient at random parameter
// points under every mode and compares against the central-full FD reference.
// Points rejected by the model constraint (chi² plateau, zero gradient) are
// skipped. Emits one [GRADCHECK] line per mode.
void run_grad_check(PROmetric &metric, const std::vector<Eigen::VectorXf> &params)
{
    metric.setBounds(metric.LowerBound(), metric.UpperBound());
    const size_t np = metric.nParams();

    const std::vector<PROmetric::GradientMode> modes = {
        PROmetric::GradientOneSidedFull,
        PROmetric::GradientCentralLin,
        PROmetric::GradientOneSidedLin,
        PROmetric::GradientAnalytic,
    };

    // Reference gradients (central FD on the full chi²) at every usable point.
    std::vector<Eigen::VectorXf> refs;
    std::vector<size_t> used_points;
    Eigen::VectorXf gref = Eigen::VectorXf::Zero(np);
    metric.setGradientMode(PROmetric::GradientCentralFull);
    for (size_t k = 0; k < params.size(); ++k) {
        metric.reset();
        float chi2 = metric(params[k], gref, true);
        if (chi2 >= 1e9f) continue; // constraint-rejected plateau: all modes return zero gradient
        refs.push_back(gref);
        used_points.push_back(k);
    }
    log<LOG_INFO>(L"[GRADCHECK] reference=central-full points_used=%1% of %2%")
        % refs.size() % params.size();

    Eigen::VectorXf g = Eigen::VectorXf::Zero(np);
    for (auto mode : modes) {
        metric.setGradientMode(mode);
        double max_abs = 0, sum_abs = 0, max_rel = 0, sum_rel = 0;
        long   n_comp = 0;
        for (size_t r = 0; r < refs.size(); ++r) {
            metric.reset();
            metric(params[used_points[r]], g, true);
            for (size_t i = 0; i < np; ++i) {
                const double a = std::abs((double)g(i) - (double)refs[r](i));
                // Guard tiny denominators: FD reference noise dominates below ~1e-3.
                const double rel = a / std::max(std::abs((double)refs[r](i)), 1e-3);
                max_abs = std::max(max_abs, a);   sum_abs += a;
                max_rel = std::max(max_rel, rel); sum_rel += rel;
                ++n_comp;
            }
        }
        const double denom = std::max(1L, n_comp);
        log<LOG_INFO>(L"[GRADCHECK] mode=%1% points=%2% max_abs=%3% mean_abs=%4% max_rel=%5% mean_rel=%6%")
            % PROmetric::gradientModeName(mode) % refs.size()
            % max_abs % (sum_abs / denom) % max_rel % (sum_rel / denom);
    }
}

// (p) Repeated full fits per (fit preset × gradient mode) on a SHARED set of
// Poisson-fluctuated pseudo-data universes:
//   - The base data spectrum (whatever the main chain built — Asimov at the CLI
//     injected point unless real data was supplied) is Poisson-fluctuated once
//     per universe with fixed seeds; every (preset, mode) cell fits the SAME
//     universes, and for a given (preset, universe) every gradient mode uses
//     the same PROfitter seed. LHS/PSO evaluate chi² only, so L-BFGS-B starts
//     from identical points across modes — differences in iterations / calls /
//     wall time / recovered parameters are due to the gradient alone.
//   - With throw_systs, universes are FC-style pseudo-experiments instead:
//     thrown spline pulls + Cholesky covariance shift + Poisson (same recipe
//     as the --pseudo-experiment path in bin/PROfit.cxx). With throw_phys the
//     truth physics point is additionally drawn uniformly within the fit
//     bounds per universe (per-universe truth columns are added to the CSV).
//   - Fits run in parallel over universes (opts.nthreads workers); results are
//     logged after each cell completes so [GRADBENCH] lines never interleave.
//   - Fitted physics parameters are logged per fit for parameter-recovery
//     plots; the injected truth is logged once as [GRADBENCH-TRUTH].
// Metrics are constructed fresh per universe as PROchi/BinnedChi2 regardless of
// the CLI --chi2 choice (the analytic mode is PROchi-only so far, and a fresh
// metric is needed anyway to bind the per-universe data).
void run_grad_mode_fits(const PROconfig &config, const PROpeller &prop,
                        PROmetric &base_metric, const PROdata &base_data,
                        int n_universes, uint32_t base_seed, int nthreads,
                        const Eigen::VectorXf &truth_params,
                        const std::string &csv_path,
                        const std::string &preset_filter,
                        bool throw_systs, bool throw_phys)
{
    const PROmodel &model = base_metric.GetModel();
    const PROsyst  &syst  = base_metric.GetSysts();
    const size_t nphys = model.nparams;

    const Eigen::VectorXf bench_ub = base_metric.UpperBound();
    const Eigen::VectorXf bench_lb = base_metric.LowerBound();

    // ---- Shared universes ----
    std::vector<PROdata> universes;
    universes.reserve(n_universes);
    std::vector<Eigen::VectorXf> universe_truth;   // filled iff throw_phys
    const size_t nspline = syst.GetNSplines();
    Eigen::VectorXf nominal = Eigen::VectorXf::Zero(nphys + nspline);
    nominal.head(std::min(truth_params.size(), nominal.size())) =
        truth_params.head(std::min(truth_params.size(), nominal.size()));
    if (throw_systs || throw_phys) {
        const size_t io = config.i_prime;
        std::mt19937 rng(base_seed ^ 0x715c0deu);
        std::normal_distribution<float> nd;
        // Cholesky factor of the covariance at the CV point (splines at
        // nominal); recomputed per universe when the physics point is thrown.
        Eigen::MatrixXf L;
        if (throw_systs && !throw_phys) {
            PROspec cv = FillSpectra(config, prop, syst, model, nominal, true, io);
            L = syst.DecomposeFractionalCovariance(config, cv.Spec());
        }
        for (int u = 0; u < n_universes; ++u) {
            Eigen::VectorXf tp = nominal;
            if (throw_phys) {
                // Rejection-sample against the model's physical-region
                // constraint (e.g. 3+1 unitarity Ue4^2+Um4^2 < 1): the metric
                // returns its invalid sentinel outside it, so a truth thrown
                // there is unreachable by any fitter and only measures how
                // fitters cope with the sentinel wall — not what we test here.
                for (int attempt = 0; attempt < 10000; ++attempt) {
                    for (size_t j = 0; j < nphys; ++j) {
                        float lo = bench_lb((Eigen::Index)j), hi = bench_ub((Eigen::Index)j);
                        if (!std::isfinite(lo)) lo = std::isfinite(model.lb((Eigen::Index)j)) ? model.lb((Eigen::Index)j) : -2.0f;
                        if (!std::isfinite(hi)) hi = std::isfinite(model.ub((Eigen::Index)j)) ? model.ub((Eigen::Index)j) :  2.0f;
                        tp((Eigen::Index)j) = std::uniform_real_distribution<float>(lo, hi)(rng);
                    }
                    if (!model.model_constraint || model.model_constraint(tp.head(nphys))) break;
                }
                if (throw_systs) {
                    PROspec cv = FillSpectra(config, prop, syst, model, tp, true, io);
                    L = syst.DecomposeFractionalCovariance(config, cv.Spec());
                }
            }
            if (throw_systs)
                for (size_t i = 0; i < nspline; ++i)
                    tp((Eigen::Index)(nphys + i)) = ThrowRestrictedSplinePull(syst, i, rng, nd);
            PROspec shifted = FillSpectra(config, prop, syst, model, tp, true, io);
            Eigen::VectorXf coll  = CollapseMatrix(config, shifted.Spec(), io);
            Eigen::VectorXf collE = CollapseMatrix(config, shifted.Error(), io);
            if (throw_systs && L.size() > 0) {
                Eigen::VectorXf throwC(coll.size());
                for (Eigen::Index i = 0; i < throwC.size(); ++i) throwC(i) = nd(rng);
                // Clamp: PoissonVariation zeroes negative bins with a warning
                // per bin; do it silently here instead.
                coll = (coll + L * throwC).cwiseMax(0.0f);
            }
            PROspec pe = PROspec::PoissonVariation(PROspec(coll, collE),
                                                   base_seed + (uint32_t)u);
            universes.emplace_back(pe.Spec(), pe.Error());
            if (throw_phys) universe_truth.push_back(tp.head(nphys));
        }
    } else {
        // Poisson fluctuations of the base data spectrum.
        for (int u = 0; u < n_universes; ++u) {
            PROspec fluc = PROspec::PoissonVariation(
                PROspec(base_data.Spec(), base_data.Error()),
                base_seed + (uint32_t)u);
            universes.emplace_back(fluc.Spec(), fluc.Error());
        }
    }

    // Per-fit records go to a CSV: PROlog suppresses repeated-format lines
    // after ~1000 emissions, which silently drops per-fit lines in a
    // 4-preset × 5-mode × 100-universe sweep. Only summaries are logged.
    std::ofstream csv(csv_path);
    if (!csv) {
        log<LOG_ERROR>(L"[GRADBENCH] cannot open output CSV '%1%'; aborting fit benchmark.")
            % csv_path.c_str();
        return;
    }
    if (truth_params.size() >= (Eigen::Index)nphys) {
        std::string tstr;
        for (size_t i = 0; i < nphys; ++i)
            tstr += (i ? " " : "") + model.param_names[i] + "=" + std::to_string(truth_params(i));
        log<LOG_INFO>(L"[GRADBENCH-TRUTH] universes=%1% phys: %2%")
            % n_universes % tstr.c_str();
        csv << "# truth " << tstr << "\n";
    }
    csv << "preset,mode,fit,wall_s,cpu_s,lbfgs_iters,metric_calls,chi2";
    for (size_t i = 0; i < nphys; ++i) csv << ",phys" << i;
    if (throw_phys)
        for (size_t i = 0; i < nphys; ++i) csv << ",truth" << i;
    csv << "\n";
    log<LOG_INFO>(L"[GRADBENCH] per-fit records -> %1%") % csv_path.c_str();
    {
        std::string umode = "poisson";
        if (throw_systs) umode += " + FC-style syst throws";
        if (throw_phys)  umode += " + thrown truth physics (uniform in fit bounds, inside the model's physical region)";
        log<LOG_INFO>(L"[GRADBENCH] universe generation: %1%") % umode.c_str();
    }

    // Standard presets are compared across every gradient mode; the grad-*
    // presets are designed around GradientAnalytic (their PSO budget is cut in
    // favour of L-BFGS-B multistarts, which only pays off with a cheap exact
    // gradient), so they run analytic-only.
    const std::vector<PROmetric::GradientMode> all_modes = {
        PROmetric::GradientCentralFull,
        PROmetric::GradientOneSidedFull,
        PROmetric::GradientCentralLin,
        PROmetric::GradientOneSidedLin,
        PROmetric::GradientAnalytic,
    };
    const std::vector<PROmetric::GradientMode> analytic_only = {
        PROmetric::GradientAnalytic,
    };
    struct GradCell { std::string preset; const std::vector<PROmetric::GradientMode> *modes; };
    std::vector<GradCell> cells = {
        {"sensitivity", &all_modes}, {"fast", &all_modes},
        {"good", &all_modes}, {"overkill", &all_modes},
        {"grad-fast", &analytic_only}, {"grad-good", &analytic_only},
        {"grad-deep", &analytic_only}, {"grad-overkill", &analytic_only},
    };
    if (!preset_filter.empty()) {
        std::set<std::string> keep;
        std::stringstream ss(preset_filter);
        for (std::string tok; std::getline(ss, tok, ',');)
            if (!tok.empty()) keep.insert(tok);
        std::vector<GradCell> filtered;
        for (auto &c : cells)
            if (keep.count(c.preset)) filtered.push_back(c);
        if (filtered.empty()) {
            log<LOG_ERROR>(L"[GRADBENCH] --grad-presets '%1%' matches no known preset; aborting fit benchmark.")
                % preset_filter.c_str();
            return;
        }
        cells = std::move(filtered);
    }

    struct FitRec {
        double wall_s = 0;
        // Per-thread CPU time (CLOCK_THREAD_CPUTIME_ID): immune to other load
        // on the node, unlike wall time. This is the cost metric to compare.
        double cpu_s = 0;
        size_t iters = 0, calls = 0;
        float  chi2 = 0;
        Eigen::VectorXf phys;
    };
    auto thread_cpu_s = []() {
        timespec ts;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
        return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
    };

    for (const auto &cell : cells) {
        const std::string &preset = cell.preset;
        // Preset defaults only — CLI --fit-options are deliberately NOT applied
        // here so the presets are compared as-shipped.
        PROfitterConfig pcfg({}, preset, /*isScan=*/false);
        for (auto mode : *cell.modes) {
            pcfg.gradient_mode = mode;

            std::vector<FitRec> recs(n_universes);
            std::atomic<int> next{0};
            auto worker = [&]() {
                for (int u = next.fetch_add(1); u < n_universes; u = next.fetch_add(1)) {
                    PROchi m("", config, prop, &syst, model, universes[u],
                             PROmetric::BinnedChi2, /*shape_only=*/false);
                    m.setBounds(bench_lb, bench_ub);
                    // Same fitter seed per universe across all (preset, mode)
                    // cells → identical LHS/PSO trajectories per universe.
                    PROfitter fitter(bench_ub, bench_lb, pcfg,
                                     base_seed ^ 0x5eedu ^ (uint32_t)u);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    const double c0 = thread_cpu_s();
                    const float chi2 = fitter.Fit(m);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    FitRec &r = recs[u];
                    r.wall_s = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() * 1e-6;
                    r.cpu_s  = thread_cpu_s() - c0;
                    r.iters  = fitter.total_lbfgs_iterations;
                    r.calls  = m.getCallCount();
                    r.chi2   = chi2;
                    r.phys   = fitter.best_fit.head(nphys);
                }
            };
            std::vector<std::thread> pool;
            const int nw = std::max(1, std::min(nthreads, n_universes));
            for (int t = 0; t < nw; ++t) pool.emplace_back(worker);
            for (auto &t : pool) t.join();

            double sum_s = 0, sum_iters = 0, sum_calls = 0, sum_chi = 0;
            double best_chi = std::numeric_limits<double>::infinity();
            for (int u = 0; u < n_universes; ++u) {
                const FitRec &r = recs[u];
                csv << preset << ',' << PROmetric::gradientModeName(mode) << ','
                    << u << ',' << r.wall_s << ',' << r.cpu_s << ',' << r.iters << ','
                    << r.calls << ',' << r.chi2;
                for (size_t i = 0; i < nphys; ++i) csv << ',' << r.phys(i);
                if (throw_phys)
                    for (size_t i = 0; i < nphys; ++i)
                        csv << ',' << universe_truth[u]((Eigen::Index)i);
                csv << "\n";
                sum_s += r.wall_s; sum_iters += (double)r.iters;
                sum_calls += (double)r.calls; sum_chi += (double)r.chi2;
                best_chi = std::min(best_chi, (double)r.chi2);
            }
            csv.flush();
            const double dn = std::max(1, n_universes);
            log<LOG_INFO>(L"[GRADBENCH-SUMMARY] preset=%1% mode=%2% fits=%3% mean_wall_s=%4% mean_lbfgs_iters=%5% mean_metric_calls=%6% mean_chi2=%7% best_chi2=%8%")
                % preset.c_str() % PROmetric::gradientModeName(mode) % n_universes
                % (sum_s / dn) % (sum_iters / dn) % (sum_calls / dn)
                % (sum_chi / dn) % best_chi;
        }
    }
}

}  // namespace

std::vector<BenchResult> run_scale_test(
    const PROconfig &config,
    const PROpeller &prop,
    PROmetric &metric,
    const PROdata &data,
    const PROfitterConfig &fitconfig,
    const BenchOptions &opts)
{
    const PROmodel &model = metric.GetModel();
    const PROsyst  &syst  = metric.GetSysts();

    std::vector<BenchResult> results;

    const int N_fill   = std::max(1, opts.N);
    const int N_metric = std::max(1, opts.N / 10);
    const int N_fit    = std::max(1, opts.N / 100);

    // Honour the gradient mode the user selected via --grad-mode. The fit
    // benchmark already gets this through PROfitter::Fit(), but the
    // metric_grad_* tests call metric() directly and would otherwise see the
    // metric's default mode regardless of CLI selection.
    metric.setGradientMode(fitconfig.gradient_mode);

    log<LOG_INFO>(L"%1% || ===== PRObench scale-test starting =====") % __func__;
    log<LOG_INFO>(L"%1% || N_fillspectra=%2%  N_metric=%3%  N_fit=%4%  grad_mode=%5%")
        % __func__ % N_fill % N_metric % N_fit
        % PROmetric::gradientModeName(fitconfig.gradient_mode);
    log<LOG_INFO>(L"%1% || nphys=%2%  nnuis=%3%  nbins=%4%")
        % __func__ % model.nparams % syst.GetNSplines() % config.m_num_variable_bins_total[config.i_prime];

    // Distinct sub-seeds so the three variation modes draw distinct sequences
    // while keeping the whole bench reproducible from opts.rng_seed.
    const uint32_t s_all  = opts.rng_seed ^ 0x1u;
    const uint32_t s_phys = opts.rng_seed ^ 0x2u;
    const uint32_t s_nuis = opts.rng_seed ^ 0x4u;

    // ---- (a–c) FillSpectra ----
    if (opts.tests & Bench_FillSpectra_All) {
        auto p = draw_param_set(model, syst, N_fill, s_all,  /*vary_phys*/true,  /*vary_nuis*/true);
        results.push_back(time_fillspectra("fillspectra_vary_all",  config, prop, syst, model, p, opts.binned));
    }
    if (opts.tests & Bench_FillSpectra_Phys) {
        auto p = draw_param_set(model, syst, N_fill, s_phys, true,  false);
        results.push_back(time_fillspectra("fillspectra_vary_phys", config, prop, syst, model, p, opts.binned));
    }
    if (opts.tests & Bench_FillSpectra_Nuis) {
        auto p = draw_param_set(model, syst, N_fill, s_nuis, false, true);
        results.push_back(time_fillspectra("fillspectra_vary_nuis", config, prop, syst, model, p, opts.binned));
    }

    // ---- (d–f) PROmetric() chi² only ----
    if (opts.tests & Bench_Metric_All) {
        auto p = draw_param_set(model, syst, N_metric, s_all,  true,  true);
        results.push_back(time_metric("metric_vary_all",  config, metric, p, /*grad=*/false));
    }
    if (opts.tests & Bench_Metric_Phys) {
        auto p = draw_param_set(model, syst, N_metric, s_phys, true,  false);
        results.push_back(time_metric("metric_vary_phys", config, metric, p, /*grad=*/false));
    }
    if (opts.tests & Bench_Metric_Nuis) {
        auto p = draw_param_set(model, syst, N_metric, s_nuis, false, true);
        results.push_back(time_metric("metric_vary_nuis", config, metric, p, /*grad=*/false));
    }

    // ---- (j–l) PROmetric() chi² + finite-difference gradient ----
    // Each gradient call triggers ~2*(nphys+nnuis) extra FillSpectra calls
    // for two-sided finite differences (one-sided at boundaries). Per-call
    // cost is roughly (nparams) × the no-grad metric cost.
    if (opts.tests & Bench_MetricGrad_All) {
        auto p = draw_param_set(model, syst, N_metric, s_all,  true,  true);
        results.push_back(time_metric("metric_grad_vary_all",  config, metric, p, /*grad=*/true));
    }
    if (opts.tests & Bench_MetricGrad_Phys) {
        auto p = draw_param_set(model, syst, N_metric, s_phys, true,  false);
        results.push_back(time_metric("metric_grad_vary_phys", config, metric, p, /*grad=*/true));
    }
    if (opts.tests & Bench_MetricGrad_Nuis) {
        auto p = draw_param_set(model, syst, N_metric, s_nuis, false, true);
        results.push_back(time_metric("metric_grad_vary_nuis", config, metric, p, /*grad=*/true));
    }

    // ---- (g) PROfitter::Fit ----
    // Single test only: each Fit() internally drives Latin + Swarm + LBFGS
    // over the whole parameter space, so the seed-mode (vary_phys vs
    // vary_nuis) is not observable in the timing — measured directly and
    // collapsed in 2026-05.
    if (opts.tests & Bench_Fit) {
        auto p = draw_param_set(model, syst, N_fit, s_all, true, true);
        results.push_back(time_fit("fit", config, metric, p, fitconfig, s_all));
    }

    // ---- (h) Pseudo-universe throw ----
    // Same N as FillSpectra: each throw is roughly one FillSpectra + one
    // collapse + one matrix-vector + one Poisson sweep, dominated by FillSpectra.
    if (opts.tests & Bench_PseudoUniverse) {
        results.push_back(time_pseudo_universe("pseudo_full", config, prop, syst, model,
                                               N_fill, opts.rng_seed ^ 0x8u, opts.binned));
    }

    // ---- (i) CollapseMatrix ----
    // Cheap operation; reuses N_fill so it's directly comparable per-call to
    // (a–c). Useful for spotting collapse-cost regressions independent of
    // FillSpectra changes.
    if (opts.tests & Bench_Collapse) {
        results.push_back(time_collapse("collapse_matrix", config, prop, syst, model,
                                        N_fill, opts.binned));
    }

    // ---- (m, n) MCMC step ----
    // Use N_fill (same as fillspectra) — each step is one metric eval +
    // proposal overhead, comparable per-call to (a). Burnin includes adaptive
    // covariance Cholesky each step; post-burnin reuses the cached factor.
    if (opts.tests & Bench_MCMC_Burnin) {
        results.push_back(time_mcmc_step("mcmc_step_burnin", config, metric,
                                         N_fill, /*burnin=*/true,
                                         opts.rng_seed ^ 0x10u));
    }
    if (opts.tests & Bench_MCMC_Post) {
        results.push_back(time_mcmc_step("mcmc_step_post", config, metric,
                                         N_fill, /*burnin=*/false,
                                         opts.rng_seed ^ 0x20u));
    }

    // ---- (o) Gradient cross-validation ----
    // N_metric random points; every mode compared against central-full FD.
    if (opts.tests & Bench_GradCheck) {
        auto p = draw_param_set(model, syst, N_metric, opts.rng_seed ^ 0x40u, true, true);
        run_grad_check(metric, p);
        // Restore the CLI-selected mode for any tests that follow.
        metric.setGradientMode(fitconfig.gradient_mode);
    }

    // ---- (p) Fit benchmark: preset × gradient-mode sweep ----
    // N_fit shared Poisson-fluctuated universes fitted per (preset, mode) with
    // matched seeds, in parallel over opts.nthreads workers. This is the
    // headline comparison for the analytic-gradient work; results go to
    // [GRADBENCH] / [GRADBENCH-SUMMARY] lines rather than BenchResult rows.
    if (opts.tests & Bench_GradModeFit) {
        run_grad_mode_fits(config, prop, metric, data, N_fit,
                           opts.rng_seed ^ 0x80u, opts.nthreads, opts.truth_params,
                           opts.grad_csv, opts.grad_presets,
                           opts.throw_systs, opts.throw_phys);
    }

    log<LOG_INFO>(L"%1% || ===== PRObench scale-test complete: %2% tests =====")
        % __func__ % results.size();
    return results;
}

}  // namespace PRObench
}  // namespace PROfit
