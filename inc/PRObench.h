/**
 * @file PRObench.h
 * @brief Internal scaling / timing benchmarks for PROfit hot paths.
 * @author PROfit Collaboration
 *
 * @details Drives a fixed catalogue of micro-benchmarks covering the three
 * dominant hot paths exercised across a typical analysis:
 *   - FillSpectra evaluation (default N calls)
 *   - PROmetric operator() evaluation (default N/10 calls)
 *   - PROfitter::Fit global fits (default N/100 calls)
 *
 * For each path, three variation modes are timed:
 *   (vary_all) physics + nuisance both randomised per call
 *   (vary_phys) physics randomised, nuisance held at central value
 *   (vary_nuis) nuisance randomised, physics held at central value
 *
 * The benchmark uses the live PROmetric instance constructed by the main
 * PROfit chain (PROchi / PROCNP / PROpoisson — chosen by the existing
 * `--chi2` style argument). It does not build its own metric.
 *
 * Results are emitted via the standard PROlog `log` facility on a single
 * greppable line per test, e.g.
 *
 *   [SCALETEST] tag=fillspectra_vary_all N=1000 nbins=42 nphys=4 nnuis=87 total_us=12345 per_call_us=12.345
 *
 * A bash wrapper that varies binning/systematics/etc. across runs can
 * `grep "\[SCALETEST\]"` and parse the columns to plot scaling.
 */
#ifndef PROBENCH_H
#define PROBENCH_H

#include "PROconfig.h"
#include "PROdata.h"
#include "PROmetric.h"
#include "PROpeller.h"
#include "PROfitter.h"

#include <Eigen/Eigen>

#include <cstdint>
#include <string>
#include <vector>

namespace PROfit {
namespace PRObench {

    /**
     * @brief Bitmask selector for which benchmarks to run.
     * @details Combine with bitwise OR.
     * The fit benchmark has no vary_phys / vary_nuis variants because each
     * Fit() internally explores the whole parameter space (Latin + Swarm +
     * LBFGS) regardless of the seed point — the seed-mode variation is not
     * observable in the timing.
     */
    enum BenchTest : unsigned int {
        Bench_None             = 0,
        Bench_FillSpectra_All  = 1u << 0,  ///< (a) FillSpectra, vary phys + nuis.
        Bench_FillSpectra_Phys = 1u << 1,  ///< (b) FillSpectra, vary phys only.
        Bench_FillSpectra_Nuis = 1u << 2,  ///< (c) FillSpectra, vary nuis only.
        Bench_Metric_All       = 1u << 3,  ///< (d) PROmetric() chi² only, vary phys + nuis.
        Bench_Metric_Phys      = 1u << 4,  ///< (e) PROmetric() chi² only, vary phys only.
        Bench_Metric_Nuis      = 1u << 5,  ///< (f) PROmetric() chi² only, vary nuis only.
        Bench_Fit              = 1u << 6,  ///< (g) PROfitter::Fit(), random phys+nuis seeds.
        Bench_PseudoUniverse   = 1u << 7,  ///< (h) Pseudo-universe throw: random splines + MVN syst + Poisson stat (matches PROfc / brazil-band pattern).
        Bench_Collapse         = 1u << 8,  ///< (i) CollapseMatrix() on a CV vector — bin-collapsing hot path.
        Bench_MetricGrad_All   = 1u << 9,  ///< (j) PROmetric() chi² + finite-diff gradient, vary phys + nuis.
        Bench_MetricGrad_Phys  = 1u << 10, ///< (k) PROmetric() chi² + finite-diff gradient, vary phys only.
        Bench_MetricGrad_Nuis  = 1u << 11, ///< (l) PROmetric() chi² + finite-diff gradient, vary nuis only.
        Bench_MCMC_Burnin      = 1u << 12, ///< (m) Metropolis::step() during burnin (tune_mode=true; proposal Cholesky rebuilt every step + tune()).
        Bench_MCMC_Post        = 1u << 13, ///< (n) Metropolis::step() post-burnin (tune_mode=false; cached proposal Cholesky, no tune()).
        Bench_GradCheck        = 1u << 14, ///< (o) Gradient cross-validation: every mode (incl. analytic) vs the central-full FD reference at random points. [GRADCHECK] lines.
        Bench_GradModeFit      = 1u << 15, ///< (p) Repeated full PROfitter::Fit per gradient mode with matched seeds; logs wall time, LBFGS iterations, metric calls, best chi². [GRADBENCH] lines.
        Bench_FillSpectra_Group = Bench_FillSpectra_All | Bench_FillSpectra_Phys | Bench_FillSpectra_Nuis,
        Bench_Metric_Group      = Bench_Metric_All       | Bench_Metric_Phys       | Bench_Metric_Nuis,
        Bench_MetricGrad_Group  = Bench_MetricGrad_All   | Bench_MetricGrad_Phys   | Bench_MetricGrad_Nuis,
        Bench_Fit_Group         = Bench_Fit,
        Bench_MCMC_Group        = Bench_MCMC_Burnin      | Bench_MCMC_Post,
        Bench_Grad_Group        = Bench_GradCheck        | Bench_GradModeFit,  ///< NOT in Bench_All: opt-in via --tests gradcheck/gradmodes/grad.
        Bench_All               = Bench_FillSpectra_Group | Bench_Metric_Group | Bench_MetricGrad_Group | Bench_Fit_Group | Bench_PseudoUniverse | Bench_Collapse | Bench_MCMC_Group,
    };

    /**
     * @brief Configuration knobs for run_scale_test.
     */
    struct BenchOptions {
        int      N        = 1000;          ///< Base call count: FillSpectra=N, Metric=N/10, Fit=N/100. All clamped to ≥1.
        unsigned tests    = Bench_All;     ///< Bitmask of benchmarks to run.
        uint32_t rng_seed = 0xBEEFCAFE;    ///< Fixed seed for reproducibility.
        bool     binned   = true;          ///< FillSpectra binned vs event-by-event.
        int      nthreads = 1;             ///< Worker threads for the (p) gradmodes fit benchmark (universes fitted in parallel).
        /// Injected fake-data parameter vector (physics + splines) the chain built its
        /// data spectrum from; used by the (p) benchmark to report truth for
        /// parameter-recovery plots ([GRADBENCH-TRUTH] line). May be empty.
        Eigen::VectorXf truth_params;
        /// Output CSV for the (p) benchmark's per-fit records. PROlog suppresses
        /// repeated-format lines after ~1000 emissions, so per-fit data goes to a
        /// file; only the per-cell [GRADBENCH-SUMMARY] lines are logged.
        std::string grad_csv = "gradbench_fits.csv";
        /// Comma-separated preset names restricting the (p) benchmark's grid
        /// (e.g. "grad-fast,grad-good"). Empty = run every preset. Universes and
        /// per-universe fitter seeds depend only on rng_seed, so a filtered run's
        /// CSV rows can be concatenated with an earlier full run's.
        std::string grad_presets;
        /// (p) benchmark: generate universes as FC-style pseudo-experiments
        /// (thrown spline pulls + Cholesky covariance shift + Poisson) instead
        /// of Poisson-only fluctuations of the base data spectrum.
        bool throw_systs = false;
        /// (p) benchmark: draw each universe's truth physics point uniformly
        /// within the fit bounds; per-universe truth columns are added to the
        /// CSV for parameter-recovery plots.
        bool throw_phys = false;
    };

    /**
     * @brief One row of timing output.
     */
    struct BenchResult {
        std::string tag;          ///< Short greppable label, e.g. "fillspectra_vary_all".
        int         n_calls = 0;
        int         nbins   = 0;
        int         nphys   = 0;
        int         nnuis   = 0;
        double      total_us    = 0.0;
        double      per_call_us = 0.0;
    };

    /**
     * @brief Run the requested scaling benchmarks and emit greppable LOG lines.
     * @details Uses the caller-provided PROmetric instance as-is — same
     * concrete class (PROchi / PROCNP / PROpoisson) and same configuration
     * the rest of the PROfit chain is using. The bench mutates internal
     * metric state (last_param, last_value, fs_cache, call_count) so the
     * caller should not rely on those after run_scale_test returns.
     * @param config     Analysis configuration (used for nbins, i_prime).
     * @param prop       MC event store (needed for FillSpectra free function).
     * @param metric     Live PROmetric — its GetSysts(), GetModel(), and
     *                   operator() drive the (a–c) and (d–i) benchmarks.
     * @param fitconfig  PROfitter configuration for the (g–i) fit benchmarks.
     * @param opts       Knobs; see BenchOptions.
     * @return Per-test BenchResult records (also logged at LOG_INFO).
     */
    std::vector<BenchResult> run_scale_test(
        const PROconfig &config,
        const PROpeller &prop,
        PROmetric &metric,
        const PROdata &data,
        const PROfitterConfig &fitconfig,
        const BenchOptions &opts);

}  // namespace PRObench
}  // namespace PROfit

#endif
