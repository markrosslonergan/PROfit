#include "PRObench.h"

#include "PROlog.h"
#include "PROmetric.h"
#include "PROmodel.h"
#include "PROsyst.h"
#include "PROpeller.h"
#include "PROcess.h"
#include "PROspec.h"
#include "PROtocall.h"
#include "PROMCMC.h"

#include <Eigen/Eigen>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <string>
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

}  // namespace

std::vector<BenchResult> run_scale_test(
    const PROconfig &config,
    const PROpeller &prop,
    PROmetric &metric,
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

    log<LOG_INFO>(L"%1% || ===== PRObench scale-test complete: %2% tests =====")
        % __func__ % results.size();
    return results;
}

}  // namespace PRObench
}  // namespace PROfit
