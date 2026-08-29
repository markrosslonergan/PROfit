/**
 * @file PROfitter.h
 * @brief Multi-start global optimiser and configuration for PROfit chi-squared minimisation.
 * @author PROfit Collaboration
 *
 * @details Defines PROfitterConfig (parameter struct for the fitting strategy) and PROfitter
 * (the optimiser class).  The fitting pipeline is:
 *  1. Latin hypercube sampling to seed the parameter space broadly.
 *  2. Particle Swarm Optimisation (PSO) starting from the best LHS points.
 *  3. Multiple L-BFGS-B local refinements from the best PSO result(s) and any
 *     frequency-domain harmonic seed points.
 *
 * The L-BFGS-B solver is provided by the LBFGSpp library.  All solver and
 * PROfit-specific hyperparameters are collected in PROfitterConfig.
 */
#ifndef PROFITTER_H
#define PROFITTER_H

#include "PROmetric.h"
#include "PROgress.h"

#include <Eigen/Eigen>
#include "LBFGSB.h"

#include <atomic>
#include <cstdint>
#include <random>

namespace PROfit {

    /**
     * @brief Thread-safe accumulator for PROfile/PROsurf scan-mode timing diagnostics.
     * @details PROfitter::Fit() reports per-phase microseconds (latin / PSO / LBFGS)
     * and total fit count when GetScanTimingEnabled() is true. The dispatcher (PROfile
     * constructor) resets these at the start of a scan and reads them at the end to
     * print a parallelism / cost-breakdown report. When the flag is false, all the
     * Fit() instrumentation collapses to a few inlined comparisons — zero allocation,
     * negligible runtime impact.
     */
    struct ScanTimingStats {
        std::atomic<uint64_t> n_fits{0};        ///< Number of Fit() calls completed since reset.
        std::atomic<uint64_t> total_fit_us{0};  ///< Sum of wall time spent inside Fit() across all threads.
        std::atomic<uint64_t> latin_us{0};      ///< Sum of wall time spent in the LHS evaluation loop.
        std::atomic<uint64_t> pso_us{0};        ///< Sum of wall time spent in PSO.runSwarm().
        std::atomic<uint64_t> lbfgs_us{0};      ///< Sum of wall time spent in all LBFGS local-fit phases.

        /// Reset all counters to zero. Should be called from a single thread before dispatching workers.
        void reset() {
            n_fits.store(0);
            total_fit_us.store(0);
            latin_us.store(0);
            pso_us.store(0);
            lbfgs_us.store(0);
        }
    };

    /// Global scan-timing accumulator. PROfitter::Fit reads-then-updates this when
    /// GetScanTimingEnabled() returns true.
    ScanTimingStats& GetScanTimingStats();

    /// Global toggle gating the timing instrumentation in PROfitter::Fit.
    /// Returns a reference so callers can flip it on/off (e.g. PROfile constructor).
    bool& GetScanTimingEnabled();

    /**
     * @brief Configuration parameters for the PROfitter multi-start optimisation pipeline.
     * @details Collects all tuning knobs for the three-stage optimiser:
     * Latin hypercube sampling, Particle Swarm Optimisation, and L-BFGS-B local refinement.
     * Also includes MCMC and harmonic seed-search parameters.  Parameters can be set via
     * a named map (e.g. from command-line options) or by selecting a named preset
     * ("fast", "good", "overkill", "sensitivity", or the multistart presets
     * "grad-fast", "grad-good", "grad-deep", "grad-overkill": physics-only LHS,
     * no PSO, many L-BFGS-B multistarts — designed around the analytic gradient).
     */
    struct PROfitterConfig {

        /// L-BFGS-B solver parameters (convergence tolerances, maximum iterations, line-search settings).
        LBFGSpp::LBFGSBParam<float> param;

        /***PROfit fitter parameters**/

        int n_latin_points = 1500;         ///< Number of Latin hypercube points sampled across all parameters.
        float latin_diversity_factor = 0.5; ///< Distance-weighting factor for LHS: 0 = no weighting, 1 = maximally diverse.

        int n_swarm_particles = 1;                   ///< Number of PSO particles initialised from the best LHS points.
        int n_swarm_iterations = 1;                  ///< Maximum number of PSO iterations.
        int n_swarm_max_stagnent_iterations = 50;    ///< PSO early-stopping: halt after this many iterations without improvement.
        float swarm_inertia_start = 0.9;             ///< Initial PSO inertia weight (linearly decreased to swarm_inertia_end).
        float swarm_inertia_end = 0.6;               ///< Final PSO inertia weight.
        float swarm_cognitive_score = 2.0;           ///< PSO cognitive (personal-best) weight.
        float swarm_social_score = 2.0;              ///< PSO social (global-best) weight.
        float swarm_convergence_theshold = 1e-4;     ///< PSO convergence threshold: stop if improvement < this value.

        int n_localfit = 10;                         ///< Number of L-BFGS-B local refinement fits run after PSO.
        /// Sample the LHS in the physics subspace only (nuisance parameters at nominal 0).
        /// A full-space LHS ranks start points by a chi2 dominated by random nuisance
        /// pulls; a physics-only LHS ranks them by the oscillation landscape instead.
        bool latin_phys_only = false;
        /// Start the nuisance block of every LHS multistart descent from the best fit so
        /// far (physics from the LHS point). Nuisance minima are nearly common across
        /// oscillation basins, so descents converge in fewer iterations.
        bool localfit_warm_nuisance = false;
        size_t n_max_local_retries = 1;              ///< Maximum retries if an L-BFGS-B fit throws an exception.

        size_t MCMCiter = 20'000;  ///< Number of MCMC iterations (after burn-in) for posterior sampling.
        size_t MCMCburn = 25'000;  ///< Number of MCMC burn-in iterations discarded before sampling.

        size_t harmonic_min_num_seeds = 3;               ///< Minimum number of seed points from the harmonic frequency search.
        size_t harmonic_max_num_seeds = 5;               ///< Maximum number of seed points from the harmonic frequency search.
        size_t harmonic_num_test_points = 125;           ///< Number of test points in physics-parameter frequency space 125.
        size_t harmonic_raw_max_tests = 65;              ///< Maximum iterations to find significant minima in the harmonic scan 65
        float harmonic_prominence_threshold = 0.5;       ///< Absolute cap on the basin-persistence significance threshold (chi2 units).
        float harmonic_persistence_rel = 0.15;           ///< Relative significance: a basin is significant if its persistence exceeds this fraction of the scan curve's full range.
        float harmonic_persistence_floor = 1e-4;         ///< Noise floor on the persistence threshold (chi2 units); guards against pure float noise on flat curves.
        float harmonic_prominence_threshold_shift = 0.2; ///< DEPRECATED: unused since the persistence-based minima finder (kept for option-file compatibility).
        float harmonic_min_spacing_log = 0.0225;         ///< Minimum log-space separation between selected harmonic seed peaks 0.0225
        float harmonic_prominence_threshold_minimum = 1e-5; ///< DEPRECATED: unused since the persistence-based minima finder (kept for option-file compatibility).
        float harmonic_seed_norm_tolerance = 1e-4;       ///< Tolerance for seed-point norm convergence in the harmonic search.
        float harmonic_seed_chi_tolerence = 1e-6;        ///< Tolerance for chi-squared convergence in the harmonic seed search.
        int harmonic_scan_mode = 1;                      ///< 0: single chi2 eval per scan point (slice at BF). 1: fit non-frequency physics, splines pinned at BF. 2: full profile, ALL params free except the pinned frequency.
        size_t harmonic_phys_ladder = 4;                 ///< Trial values per non-frequency physics parameter evaluated at every scan point (min taken). Makes the scan amplitude-aware: a basin whose depth only appears away from the BF amplitude is invisible to the plain BF slice. 0/1 disables.
        float harmonic_dense_lo = 0.0f;                  ///< Lower edge of the densely sampled log10(dm2) window (clamped into the model range at runtime).
        float harmonic_dense_hi = 1.5f;                  ///< Upper edge of the densely sampled log10(dm2) window (clamped into the model range at runtime).
        size_t harmonic_refine_rounds = 1;               ///< Adaptive refinement rounds: insert scan midpoints where adjacent chi2 values jump by more than harmonic_refine_dchi.
        float harmonic_refine_dchi = 3.0f;               ///< Adjacent-point chi2 jump that triggers a refinement midpoint.
        float harmonic_refit_window = 25.0f;             ///< Skip STEP-3 refits of scan minima more than this many chi2 units above the best scan point.

        bool progress_bar = false; ///< If true, display a progress bar during fitting.

        /// If true (default), fixed seeds passed to Fit() are used — currently the
        /// background-only seed (physics pinned at the model defaults, refined as a
        /// nuisances-only fit and recorded as a candidate). Set to 0 via
        /// `--fit-options use_bkg_seed 0` (global fit) or `--scan-fit-options
        /// use_bkg_seed 0` (profile/surface scan fits) to disable at runtime.
        bool use_bkg_seed = true;

        /// Gradient evaluation strategy applied to the PROmetric used by this fitter.
        /// Default mirrors the historical behaviour (central FD on full chi²); set to
        /// GradientOneSidedFull for ~2× speedup, or to one of the *Lin variants for
        /// the Gauss-Newton-style linearised gradient (5–20× speedup, exact at minimum).
        /// See PROmetric::GradientMode for the full mode matrix.
        PROmetric::GradientMode gradient_mode = PROmetric::GradientAnalytic; ///< Gradient mode PROfitter::Fit sets on the metric (default: exact analytic; see PROmetric::GradientMode for the fallback).

        /** @brief Default constructor — leaves all parameters at their default values. */
        PROfitterConfig(){};

        /**
         * @brief Construct from a named preset and an optional map of overrides.
         * @param input_fit_options  Map of parameter-name to value overrides applied after the preset.
         * @param fit_preset         Preset name: "fast", "good", "overkill", "sensitivity", or the
         *                           multistart presets "grad-fast", "grad-good", "grad-deep", "grad-overkill".
         * @param isScan             If true, apply reduced settings appropriate for a parameter scan.
         */
        PROfitterConfig(std::map<std::string, float> input_fit_options, std::string fit_preset, bool isScan){

            if(fit_preset == "good"){
                param.epsilon = 1e-5;
                param.epsilon_rel = 1e-5;
                param.max_iterations = 150;
                param.max_linesearch = 20;
                param.delta = 1e-6;
                param.wolfe = 0.90;
                param.ftol = 1e-4;
                param.m = 6;
                param.max_submin =10;
                param.min_step = std::numeric_limits<float>::epsilon();

                n_latin_points = 2000;
                n_swarm_particles = 25;
                n_swarm_iterations = 100;
                n_localfit=8;
                 
            }else if (fit_preset == "fast"){
                //typically used for scans
                param.epsilon = 1e-5;
                param.epsilon_rel = 1e-5;
                param.max_iterations = 100;
                param.max_linesearch = 20;
                param.delta = 1e-6;
                param.wolfe = 0.90;
                param.ftol = 1e-4;
                param.m = 6;
                param.max_submin =10;
                param.min_step = std::numeric_limits<float>::epsilon();

                n_latin_points = 750;
                n_swarm_particles = 15;
                n_swarm_iterations = 100;
                n_localfit=4;

            }else if(fit_preset == "overkill"){
                param.epsilon = 1e-5;
                param.epsilon_rel = 1e-5;
                param.max_iterations = 250;
                param.max_linesearch = 30;
                param.delta = 1e-6;
                param.wolfe = 0.90;
                param.ftol = 1e-4;
                param.m = 6;
                param.max_submin =10;
                param.min_step = std::numeric_limits<float>::epsilon();

                n_latin_points = 5000;
                n_swarm_particles = 100;
                n_swarm_iterations = 300;
                n_localfit=15;

                harmonic_min_num_seeds = 10;//4
                harmonic_max_num_seeds = 20;
                harmonic_num_test_points = 250;
                harmonic_scan_mode = 2;
            }

            else if(fit_preset == "sensitivity"){
                
                param.epsilon = 1e-5;
                param.epsilon_rel = 1e-5;
                param.max_iterations = 100;
                param.max_linesearch = 20;
                param.delta = 1e-6;
                param.wolfe = 0.90;
                param.ftol = 1e-4;
                param.m = 6;
                param.max_submin =10;
                param.min_step = std::numeric_limits<float>::epsilon();

                n_latin_points = 100;
                n_swarm_particles = 10;
                n_swarm_iterations = 10;
                n_localfit=2;
            }




            // ---- grad-* presets ----
            // Same L-BFGS-B pipeline and (default analytic) gradient, but: LHS sampled in
            // physics space only, no PSO stage (n_swarm_iterations = 1 degenerates the
            // swarm to its start points, so the descents start from the best-ranked
            // LHS points themselves), and the budget spent on many multistart descents
            // instead. Benchmarked on pseudo-experiments with thrown systematics,
            // thrown oscillation parameters and Poisson fluctuations (nueapp and 3+1):
            // each tier reaches deeper minima than the same-cost standard preset, and
            // grad-deep beats "overkill" in every dchi2 quantile at ~1/3 the time.
            // n_swarm_particles also sizes the ranked-LHS start pool, so it equals
            // n_localfit here.
            else if(fit_preset == "grad-fast" || fit_preset == "grad-good" ||
                    fit_preset == "grad-deep" || fit_preset == "grad-overkill"){
                param.epsilon = 1e-5;
                param.epsilon_rel = 1e-6;
                param.max_iterations = 200;
                param.max_linesearch = 25;
                // NOTE: LBFGSpp's default past=1 keeps the relative-stagnation test
                // (delta) active; it is how descents terminate cleanly once float noise
                // stalls the line search. Do NOT set past=0 or delta below ~1e-6 (the
                // float ulp scale of a typical chi2): the test then never fires, every
                // descent grinds until the More-Thuente line search throws, and thrown
                // descents are discarded by the fitter.
                param.delta = 1e-6;
                param.wolfe = 0.90;
                param.ftol = 1e-4;
                param.m = 10;
                param.max_submin =10;
                param.min_step = std::numeric_limits<float>::epsilon();

                n_swarm_iterations = 1;
                latin_phys_only = true;

                if(fit_preset == "grad-fast"){          //  ~0.8 s (3+1) / 0.2 s (nueapp) per fit
                    n_latin_points = 300;   n_swarm_particles = 8;   n_localfit = 8;
                } else if(fit_preset == "grad-good"){   //  ~1.5 s / 0.5 s
                    n_latin_points = 1000;  n_swarm_particles = 16;  n_localfit = 16;
                } else if(fit_preset == "grad-deep"){   //  ~4 s / 1 s
                    n_latin_points = 2000;  n_swarm_particles = 50;  n_localfit = 50;
                    localfit_warm_nuisance = true;
                } else {                                //  grad-overkill: ~9 s / 2.3 s
                    n_latin_points = 4000;  n_swarm_particles = 100; n_localfit = 100;
                    localfit_warm_nuisance = true;
                }
            }

            std::string whichFit = ( isScan? "Simplier Scan" : "Detailed Global");
            log<LOG_INFO>(L"%1% ||Fit and  L-BFGS-B parameters for the %2% minimia finder.  ") % __func__ % whichFit.c_str();
            for(const auto &[param_name, value]: input_fit_options) {
                log<LOG_WARNING>(L"%1% || L-BFGS-B %2% set to %3% ") % __func__% param_name.c_str() % value ;
                
                // L-BFGS-B parameters
                if(param_name == "epsilon") {
                    param.epsilon = value;
                } else if(param_name == "delta") {
                    param.delta = value;
                } else if(param_name == "m") {
                    param.m = value;
                    if(value < 3) {
                        log<LOG_WARNING>(L"%1% || Number of corrections to approximate inverse Hessian in"
                                L" L-BFGS-B is recommended to be at least 3, provided value is %2%."
                                L" Note: this is controlled via --fit-options m.")
                            % __func__ % value;
                    }
                } else if(param_name == "epsilon_rel") {
                    param.epsilon_rel = value;
                } else if(param_name == "past") {
                    param.past = value;
                    if(value == 0) {
                        log<LOG_WARNING>(L"%1% || L-BFGS-B 'past' parameter set to 0. This will disable delta convergence test")
                            % __func__;
                    }
                } else if(param_name == "max_iterations") {
                    param.max_iterations = value;
                } else if(param_name == "max_submin") {
                    param.max_submin = value;
                } else if(param_name == "max_linesearch") {
                    param.max_linesearch = value;
                } else if(param_name == "min_step") {
                    param.min_step = value;
                    log<LOG_WARNING>(L"%1% || Modifying the minimum step size in the line search to be %2%."
                            L" This is not usually needed according to the LBFGSpp documentation.")
                        % __func__ % value;
                } else if(param_name == "max_step") {
                    param.max_step = value;
                    log<LOG_WARNING>(L"%1% || Modifying the maximum step size in the line search to be %2%."
                            L" This is not usually needed according to the LBFGSpp documentation.")
                        % __func__ % value;
                } else if(param_name == "ftol") {
                    param.ftol = value;
                } else if(param_name == "wolfe") {
                    param.wolfe = value;
                    
                // PROfitter specific parameters
                } else if(param_name == "n_latin_points") {
                    n_latin_points = value;
                    if(n_latin_points < 1) {
                        log<LOG_ERROR>(L"%1% || Expected to run at least 1 multistart point. Provided value is %2%.")
                            % __func__ % value;
                        exit(EXIT_FAILURE);
                    }
                } else if(param_name == "latin_diversity_factor") {
                    latin_diversity_factor = value;
                    if(latin_diversity_factor > 1 || latin_diversity_factor<0) {
                        log<LOG_ERROR>(L"%1% || Latin Diversity Factor must be between 0 and 1.  Provided value is %2%.")
                            % __func__ % value;
                        exit(EXIT_FAILURE);
                    }

                } else if(param_name == "latin_phys_only") {
                    latin_phys_only = (value != 0);
                } else if(param_name == "localfit_warm_nuisance") {
                    localfit_warm_nuisance = (value != 0);
                } else if(param_name == "n_localfit") {
                    n_localfit = value;
                    if(n_localfit < 1) {
                        log<LOG_ERROR>(L"%1% || Expected to run at least 1 local fit point. Provided value is %2%.")
                            % __func__ % value;
                        exit(EXIT_FAILURE);
                    }
                } else if(param_name == "n_max_local_retries") {
                    n_max_local_retries = value;
                } else if(param_name == "use_bkg_seed") {
                    use_bkg_seed = (value != 0);

                // Particle Swarm Optimization parameters
                } else if(param_name == "n_swarm_particles") {
                    n_swarm_particles = value;
                    if(n_swarm_particles < 1) {
                        log<LOG_ERROR>(L"%1% || Expected to run at least 1 PSO swarm particle point. Provided value is %2%.")
                            % __func__ % value;
                        exit(EXIT_FAILURE);
                    }
                } else if(param_name == "n_swarm_iterations") {
                    n_swarm_iterations = value;
                    if(n_swarm_iterations < 1) {
                        log<LOG_ERROR>(L"%1% || Expected to run at least 1 swarm_iterations point. Provided value is %2%.")
                            % __func__ % value;
                        exit(EXIT_FAILURE);
                    }
                } else if(param_name == "n_swarm_max_stagnent_iterations") {
                    n_swarm_max_stagnent_iterations = value;
                } else if(param_name == "swarm_inertia_start") {
                    swarm_inertia_start = value;
                } else if(param_name == "swarm_inertia_end") {
                    swarm_inertia_end = value;
                } else if(param_name == "swarm_cognitive_score") {
                    swarm_cognitive_score = value;
                } else if(param_name == "swarm_social_score") {
                    swarm_social_score = value;
                } else if(param_name == "swarm_convergence_threshold") {
                    swarm_convergence_theshold = value;
                    
                // MCMC parameters
                } else if(param_name == "MCMC-Burnin") {
                    MCMCburn = value;
                    if(MCMCburn < 1) {
                        log<LOG_WARNING>(L"%1% || Warning: Running without any burnin for MCMC.") % __func__ ;
                    }
                } else if(param_name == "MCMC-Iterations") {
                    MCMCiter = value;
                    if(MCMCiter < 1) {
                        log<LOG_ERROR>(L"%1% || Requested to run MCMC with no iterations.") % __func__ ;
                    }
                    
                // Harmonic seed search parameters
                } else if(param_name == "harmonic_min_num_seeds") {
                    harmonic_min_num_seeds = value;
                } else if(param_name == "harmonic_max_num_seeds") {
                    harmonic_max_num_seeds = value;
                } else if(param_name == "harmonic_num_test_points") {
                    harmonic_num_test_points = value;
                } else if(param_name == "harmonic_raw_max_tests") {
                    harmonic_raw_max_tests = value;
                } else if(param_name == "harmonic_prominence_threshold") {
                    harmonic_prominence_threshold = value;
                } else if(param_name == "harmonic_persistence_rel") {
                    harmonic_persistence_rel = value;
                } else if(param_name == "harmonic_persistence_floor") {
                    harmonic_persistence_floor = value;
                } else if(param_name == "harmonic_prominence_threshold_shift") {
                    harmonic_prominence_threshold_shift = value;
                    log<LOG_WARNING>(L"%1% || harmonic_prominence_threshold_shift is deprecated and unused (persistence-based minima finder).") % __func__;
                } else if(param_name == "harmonic_min_spacing_log") {
                    harmonic_min_spacing_log = value;
                } else if(param_name == "harmonic_prominence_threshold_minimum") {
                    harmonic_prominence_threshold_minimum = value;
                } else if(param_name == "harmonic_seed_norm_tolerance") {
                    harmonic_seed_norm_tolerance = value;
                } else if(param_name == "harmonic_seed_chi_tolerence") {
                    harmonic_seed_chi_tolerence = value;
                } else if(param_name == "harmonic_scan_fit") {
                    // Deprecated alias: maps onto harmonic_scan_mode 0/1.
                    harmonic_scan_mode = value ? 1 : 0;
                } else if(param_name == "harmonic_scan_mode") {
                    harmonic_scan_mode = int(value);
                    if(harmonic_scan_mode < 0 || harmonic_scan_mode > 2) {
                        log<LOG_WARNING>(L"%1% || harmonic_scan_mode must be 0, 1 or 2; got %2%. Clamping.") % __func__ % harmonic_scan_mode;
                        harmonic_scan_mode = std::min(std::max(harmonic_scan_mode, 0), 2);
                    }
                } else if(param_name == "harmonic_dense_lo") {
                    harmonic_dense_lo = value;
                } else if(param_name == "harmonic_dense_hi") {
                    harmonic_dense_hi = value;
                } else if(param_name == "harmonic_refine_rounds") {
                    harmonic_refine_rounds = value;
                } else if(param_name == "harmonic_refine_dchi") {
                    harmonic_refine_dchi = value;
                } else if(param_name == "harmonic_refit_window") {
                    harmonic_refit_window = value;
                } else if(param_name == "harmonic_phys_ladder") {
                    harmonic_phys_ladder = value;
                } else {
                    log<LOG_WARNING>(L"%1% || Unrecognized parameter %2%. Will ignore.") 
                        % __func__ % param_name.c_str();
                }
            }
            try {
                print();
            } catch(std::invalid_argument &except) {
                log<LOG_ERROR>(L"%1% || Invalid L-BFGS-B parameters: %2%") % __func__ % except.what();
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }

        }

        void print(){
            log<LOG_INFO>(L"%1% || Printing PROfitterConfig Values.") % __func__;
            
            log<LOG_INFO>(L"%1% || ------------ PROfitter specific -------------- ") % __func__ ;
            log<LOG_INFO>(L"%1% || n_latin_points: %2%  ") % __func__ % n_latin_points;
            log<LOG_INFO>(L"%1% || latin_diversity_factor: %2%  ") % __func__ % latin_diversity_factor;
            log<LOG_INFO>(L"%1% || latin_phys_only: %2%  ") % __func__ % latin_phys_only;
            log<LOG_INFO>(L"%1% || localfit_warm_nuisance: %2%  ") % __func__ % localfit_warm_nuisance;
            log<LOG_INFO>(L"%1% || n_localfit: %2%  ") % __func__ % n_localfit;
            log<LOG_INFO>(L"%1% || n_max_local_retries: %2%  ") % __func__ % n_max_local_retries;
            log<LOG_INFO>(L"%1% || use_bkg_seed: %2%  ") % __func__ % use_bkg_seed;
            
            log<LOG_INFO>(L"%1% || ------------ Particle Swarm Optimization -------------- ") % __func__ ;
            log<LOG_INFO>(L"%1% || n_swarm_particles: %2%  ") % __func__ % n_swarm_particles;
            log<LOG_INFO>(L"%1% || n_swarm_iterations: %2%  ") % __func__ % n_swarm_iterations;
            log<LOG_INFO>(L"%1% || n_swarm_max_stagnent_iterations: %2%  ") % __func__ % n_swarm_max_stagnent_iterations;
            log<LOG_INFO>(L"%1% || swarm_inertia_start: %2%  ") % __func__ % swarm_inertia_start;
            log<LOG_INFO>(L"%1% || swarm_inertia_end: %2%  ") % __func__ % swarm_inertia_end;
            log<LOG_INFO>(L"%1% || swarm_cognitive_score: %2%  ") % __func__ % swarm_cognitive_score;
            log<LOG_INFO>(L"%1% || swarm_social_score: %2%  ") % __func__ % swarm_social_score;
            log<LOG_INFO>(L"%1% || swarm_convergence_threshold: %2%  ") % __func__ % swarm_convergence_theshold;
            
            log<LOG_INFO>(L"%1% || ------------ MCMC Parameters -------------- ") % __func__ ;
            log<LOG_INFO>(L"%1% || MCMCiter: %2%  ") % __func__ % MCMCiter;
            log<LOG_INFO>(L"%1% || MCMCburn: %2%  ") % __func__ % MCMCburn;
            
            log<LOG_INFO>(L"%1% || ------------ Harmonic Seed Search -------------- ") % __func__ ;
            log<LOG_INFO>(L"%1% || harmonic_min_num_seeds: %2%  ") % __func__ % harmonic_min_num_seeds;
            log<LOG_INFO>(L"%1% || harmonic_max_num_seeds: %2%  ") % __func__ % harmonic_max_num_seeds;
            log<LOG_INFO>(L"%1% || harmonic_num_test_points: %2%  ") % __func__ % harmonic_num_test_points;
            log<LOG_INFO>(L"%1% || harmonic_raw_max_tests: %2%  ") % __func__ % harmonic_raw_max_tests;
            log<LOG_INFO>(L"%1% || harmonic_prominence_threshold: %2%  ") % __func__ % harmonic_prominence_threshold;
            log<LOG_INFO>(L"%1% || harmonic_persistence_rel: %2%  ") % __func__ % harmonic_persistence_rel;
            log<LOG_INFO>(L"%1% || harmonic_persistence_floor: %2%  ") % __func__ % harmonic_persistence_floor;
            log<LOG_INFO>(L"%1% || harmonic_min_spacing_log: %2%  ") % __func__ % harmonic_min_spacing_log;
            log<LOG_INFO>(L"%1% || harmonic_seed_norm_tolerance: %2%  ") % __func__ % harmonic_seed_norm_tolerance;
            log<LOG_INFO>(L"%1% || harmonic_seed_chi_tolerence: %2%  ") % __func__ % harmonic_seed_chi_tolerence;
            log<LOG_INFO>(L"%1% || harmonic_scan_mode: %2%  ") % __func__ % harmonic_scan_mode;
            log<LOG_INFO>(L"%1% || harmonic_dense_lo: %2%  ") % __func__ % harmonic_dense_lo;
            log<LOG_INFO>(L"%1% || harmonic_dense_hi: %2%  ") % __func__ % harmonic_dense_hi;
            log<LOG_INFO>(L"%1% || harmonic_refine_rounds: %2%  ") % __func__ % harmonic_refine_rounds;
            log<LOG_INFO>(L"%1% || harmonic_refine_dchi: %2%  ") % __func__ % harmonic_refine_dchi;
            log<LOG_INFO>(L"%1% || harmonic_refit_window: %2%  ") % __func__ % harmonic_refit_window;
            log<LOG_INFO>(L"%1% || harmonic_phys_ladder: %2%  ") % __func__ % harmonic_phys_ladder;
            
            log<LOG_INFO>(L"%1% || ------------ LBFGSBParam -------------- ") % __func__ ;
            log<LOG_INFO>(L"%1% || m: %2%   ") % __func__ % param.m ;
            log<LOG_INFO>(L"%1% || epsilon: %2%  ") % __func__ % param.epsilon ;
            log<LOG_INFO>(L"%1% || epsilon_rel: %2%  ") % __func__ % param.epsilon_rel;
            log<LOG_INFO>(L"%1% || past: %2%   ") % __func__ % param.past ;
            log<LOG_INFO>(L"%1% || delta: %2%  ") % __func__ % param.delta ;
            log<LOG_INFO>(L"%1% || max_iterations: %2%  ") % __func__ % param.max_iterations ;
            log<LOG_INFO>(L"%1% || max_submin: %2%   ") % __func__ % param.max_submin ;
            log<LOG_INFO>(L"%1% || max_linesearch: %2%   ") % __func__ % param.max_linesearch ;
            log<LOG_INFO>(L"%1% || min_step: %2%  ) ") % __func__ % param.min_step ;
            log<LOG_INFO>(L"%1% || max_step: %2%   ") % __func__ % param.max_step ;
            log<LOG_INFO>(L"%1% || ftol: %2%   ") % __func__ % param.ftol ;
            log<LOG_INFO>(L"%1% || wolfe: %2%   ") % __func__ % param.wolfe ;
            log<LOG_INFO>(L"%1% || --------------------------------------- ") % __func__ ;
            param.check_param();

        }
        
        static void PrintHelp(){
            log<LOG_INFO>(L"==================================================================");
            log<LOG_INFO>(L"                  PROfitter Configuration Help                    ");
            log<LOG_INFO>(L"==================================================================");
            
            log<LOG_INFO>(L"");
            log<LOG_INFO>(L"------ PROfitter Specific Parameters ------");
            log<LOG_INFO>(L"  n_latin_points                       : Number of Latin hypercube points to sample across all parameters");
            log<LOG_INFO>(L"  latin_diversity_factor               : Diversity of latin points, 0: no distance weighting, 1: select most diverse far away points");
            log<LOG_INFO>(L"  latin_phys_only                      : 1 = sample the latin hypercube in physics-parameter space only (nuisances at nominal); 0 = all parameters");
            log<LOG_INFO>(L"  localfit_warm_nuisance               : 1 = start each latin multistart descent's nuisance parameters at the best fit so far; 0 = at the latin point");
            log<LOG_INFO>(L"  n_localfit                           : Total number of L-BFGS-B fits to do after PSO");
            log<LOG_INFO>(L"  n_max_local_retries                  : Maximum retries if L-BFGS-B throws an exception");
            log<LOG_INFO>(L"  use_bkg_seed                         : 1 (default) fits the background-only fixed seed (physics pinned at model defaults) as a candidate; 0 disables");
            
            log<LOG_INFO>(L"");
            log<LOG_INFO>(L"------ Particle Swarm Optimization (PSO) Parameters ------");
            log<LOG_INFO>(L"  n_swarm_particles                    : Number of particles in the swarm (taken from best hypercube points)");
            log<LOG_INFO>(L"  n_swarm_iterations                   : Maximum number of PSO iterations");
            log<LOG_INFO>(L"  n_swarm_max_stagnent_iterations      : Stop PSO after this many iterations without improvement");
            log<LOG_INFO>(L"  swarm_inertia_start                  : Initial inertia weight for PSO velocity update");
            log<LOG_INFO>(L"  swarm_inertia_end                    : Final inertia weight (linearly decreased during optimization)");
            log<LOG_INFO>(L"  swarm_cognitive_score                : Weight for particle's personal best in velocity update");
            log<LOG_INFO>(L"  swarm_social_score                   : Weight for global best in velocity update");
            log<LOG_INFO>(L"  swarm_convergence_threshold          : Convergence criterion for PSO");
            
            log<LOG_INFO>(L"");
            log<LOG_INFO>(L"------ MCMC Parameters ------");
            log<LOG_INFO>(L"  MCMC-Iterations                      : Number of MCMC iterations to run");
            log<LOG_INFO>(L"  MCMC-Burnin                          : Number of burn-in iterations for MCMC");
            
            log<LOG_INFO>(L"");
            log<LOG_INFO>(L"------ Harmonic Seed Search Parameters ------");
            log<LOG_INFO>(L"  harmonic_min_num_seeds               : Minimum number of seed points for harmonic search");
            log<LOG_INFO>(L"  harmonic_max_num_seeds               : Maximum number of seed points for harmonic search");
            log<LOG_INFO>(L"  harmonic_num_test_points             : Number of test points in frequency space");
            log<LOG_INFO>(L"  harmonic_raw_max_tests               : Max number of iterations to find significant minima.");
            log<LOG_INFO>(L"  harmonic_prominence_threshold        : Absolute cap on the basin-persistence significance threshold (chi2)");
            log<LOG_INFO>(L"  harmonic_persistence_rel             : Relative significance: persistence must exceed this fraction of the curve range");
            log<LOG_INFO>(L"  harmonic_persistence_floor           : Noise floor on the persistence threshold (chi2)");
            log<LOG_INFO>(L"  harmonic_min_spacing_log             : Minimum spacing between selected seeds in scan space");
            log<LOG_INFO>(L"  harmonic_seed_norm_tolerance         : Tolerance for seed point norm convergence");
            log<LOG_INFO>(L"  harmonic_seed_chi_tolerence          : Tolerance for chi-squared convergence in seed search");
            log<LOG_INFO>(L"  harmonic_scan_mode                   : 0: eval-only slice at BF (default). 1: fit non-freq physics per point. 2: full profile, all params free except pinned freq");
            log<LOG_INFO>(L"  harmonic_scan_fit                    : Deprecated alias for harmonic_scan_mode 0/1");
            log<LOG_INFO>(L"  harmonic_dense_lo / harmonic_dense_hi: Densely sampled log10(dm2) window (clamped into model bounds)");
            log<LOG_INFO>(L"  harmonic_refine_rounds               : Adaptive midpoint-refinement rounds over the scan curve");
            log<LOG_INFO>(L"  harmonic_refine_dchi                 : Adjacent-point chi2 jump that triggers a refinement midpoint");
            log<LOG_INFO>(L"  harmonic_refit_window                : Skip refits of scan minima more than this chi2 above the best scan point");
            log<LOG_INFO>(L"  harmonic_phys_ladder                 : Trial values per non-freq physics param evaluated at each scan point (min taken); 0/1 disables");
            
            log<LOG_INFO>(L"");
            log<LOG_INFO>(L"------ L-BFGS-B Parameters ------");
            log<LOG_INFO>(L"  m                                    : Number of corrections to approximate inverse Hessian (recommend >= 3)");
            log<LOG_INFO>(L"  epsilon                              : Tolerance for gradient norm convergence test");
            log<LOG_INFO>(L"  epsilon_rel                          : Relative tolerance for gradient norm");
            log<LOG_INFO>(L"  past                                 : Number of past iterations for delta convergence test (0 disables)");
            log<LOG_INFO>(L"  delta                                : Tolerance for delta convergence test");
            log<LOG_INFO>(L"  max_iterations                       : Maximum number of L-BFGS-B iterations (0 = unlimited)");
            log<LOG_INFO>(L"  max_submin                           : Maximum number of trials for subspace minimization");
            log<LOG_INFO>(L"  max_linesearch                       : Maximum number of line search iterations");
            log<LOG_INFO>(L"  min_step                             : Minimum step size in line search");
            log<LOG_INFO>(L"  max_step                             : Maximum step size in line search");
            log<LOG_INFO>(L"  ftol                                 : Tolerance for sufficient decrease condition in line search");
            log<LOG_INFO>(L"  wolfe                                : Coefficient for Wolfe condition (0 < wolfe < 1)");
            
            log<LOG_INFO>(L"");
            log<LOG_INFO>(L"==================================================================");
            log<LOG_INFO>(L"Note: Use --fit-options <param_name> <value> to set parameters");
            log<LOG_INFO>(L"Example: --fit-options n_latin_points 2000 max_iterations 5000");
            log<LOG_INFO>(L"==================================================================");
        }
    };

    /**
     * @brief A seed point that carries per-parameter pins honored during its refinement.
     * @details Unlike a plain seed (a start point from which ALL free parameters float),
     * a FixedSeed holds every flagged parameter at its seed value during the L-BFGS-B
     * refinement of THIS seed only (via zero-width solver bounds); the rest of the fit
     * pipeline is untouched. The canonical use is the background-only seed: physics
     * parameters pinned at the model defaults so the null hypothesis is always fit
     * (nuisances only) and recorded as a candidate for the global minimum.
     *
     * Conflict rule: if any pinned value falls outside the fit's own [lb, ub] for that
     * parameter — e.g. a --fix'd parameter, or a profile scan pinning the same axis at
     * a different value — the whole seed is skipped (logged, not an error).
     */
    struct FixedSeed {
        Eigen::VectorXf point;   ///< Full parameter vector (physics + splines) start point.
        std::vector<int> fixed;  ///< Same length as point; 1 = hold point(i) fixed during this seed's refinement.
    };

    /**
     * @brief Multi-start global optimiser for PROfit chi-squared minimisation.
     * @details Implements the three-stage pipeline: Latin hypercube sampling,
     * Particle Swarm Optimisation, and L-BFGS-B local refinement.  The Fit() method
     * executes the full pipeline and stores the best-fit parameter vector in best_fit.
     * After fitting, the approximate Hessian and inverse Hessian (parameter covariance)
     * are accessible via the L-BFGS-B solver.
     */
    class PROfitter {
        public:
            Eigen::VectorXf ub;       ///< Upper bounds for all parameters (physics + splines).
            Eigen::VectorXf lb;       ///< Lower bounds for all parameters.
            Eigen::VectorXf best_fit; ///< Best-fit parameter vector found by the optimiser.
            PROfitterConfig fitconfig;              ///< Configuration struct controlling the optimisation pipeline.
            LBFGSpp::LBFGSBSolver<float> solver;   ///< Underlying L-BFGS-B solver instance.
            uint32_t seed;                         ///< Random seed for reproducible LHS and PSO initialisations.
            std::map<std::string,size_t> exception_string_map; ///< Counts of L-BFGS-B exception messages for diagnostics.
            MultiPROgressBar * progress;           ///< Pointer to progress bar (non-owning); null when not used.
            bool run_progress;                     ///< True if a progress bar should be updated during fitting.

            std::vector<Eigen::VectorXf> freq_seed_points; ///< Seed points found by the harmonic frequency scan.
            size_t total_lbfgs_iterations = 0; ///< L-BFGS-B iterations summed over all local refinements of the most recent Fit() (benchmark diagnostic).
            std::vector<float> freq_seed_values;           ///< Chi-squared values at the harmonic seed points.
            std::vector<float> harmonic_scan_pos;          ///< Diagnostic: frequency positions of the last harmonic scan curve (sorted, finite points only).
            std::vector<float> harmonic_scan_chi;          ///< Diagnostic: chi-squared values of the last harmonic scan curve, parallel to harmonic_scan_pos.

            /**
             * @brief Construct a PROfitter with given bounds and configuration.
             * @param ub        Upper bounds for all parameters.
             * @param lb        Lower bounds for all parameters.
             * @param fitconfig_ Fitting configuration (default-constructed if not supplied).
             * @param inseed    Random seed (default 0).
             */
            PROfitter(const Eigen::VectorXf ub, const Eigen::VectorXf lb, PROfitterConfig fitconfig_ = {}, uint32_t inseed = 0)
                : ub(ub), lb(lb), fitconfig(fitconfig_), solver(fitconfig.param), seed(inseed) {run_progress=false;}

            /**
             * @brief Run the full multi-start optimisation pipeline and return the minimum chi-squared.
             * @param metric   The PROmetric to minimise (provides chi-squared and gradient).
             * @param seed_pt  Optional single seed point; if empty, LHS is used for seeding.
             * @return Minimum chi-squared value found.
             */
            float Fit(PROmetric &metric, const Eigen::VectorXf &seed_pt = Eigen::VectorXf());

            /**
             * @brief Run the optimisation pipeline from a provided list of seed points.
             * @param metric       The PROmetric to minimise.
             * @param seed_points  List of starting parameter vectors; augments or replaces LHS seeding.
             * @param fixed_seeds  Optional seeds refined with their flagged parameters held fixed
             *                     (see FixedSeed). A seed whose pinned values conflict with this
             *                     fitter's [lb, ub] is skipped.
             * @return Minimum chi-squared value found.
             */
            float Fit(PROmetric &metric, const std::vector<Eigen::VectorXf> &seed_points, const std::vector<FixedSeed> &fixed_seeds = {});

            /**
             * @brief Compute harmonic frequency-domain seed points for the physics parameter space.
             * @details Scans chi-squared as a function of physics parameters in frequency space,
             * identifies peaks, and stores the corresponding parameter-space seeds in freq_seed_points.
             * @param metric  The PROmetric used for evaluation.
             * @return Number of seed points found.
             */
            int calcFreqSeedPoints(PROmetric &metric);

            /**
             * @brief Identify significant local minima in a 1D scan (x, y) profile.
             * @param x_values        Vector of x values (parameter values or frequencies).
             * @param y_values        Vector of corresponding y values (chi-squared or amplitude).
             * @param use_log_spacing If true, apply log-space peak-finding logic.
             * @return Vector of (x, y) pairs for each identified significant minimum.
             */
            std::vector<std::pair<float, float>> findSignificantMinima(  const std::vector<float>& x_values,const std::vector<float>& y_values, bool use_log_spacing = true);
            

            /**
             * @brief Register a progress bar to be updated during fitting.
             * @param pin  Non-owning pointer to a MultiPROgressBar instance.
             */
            void setProgressBar(MultiPROgressBar* pin){
                    run_progress = true;
                    progress = pin;
                    return;
            }
            /** @brief Return the gradient vector at the final L-BFGS-B solution. */
            Eigen::VectorXf FinalGradient() const {return solver.final_grad();}
            /** @brief Return the L2 norm of the gradient at the final solution. */
            float FinalGradientNorm() const {return solver.final_grad_norm();}
            /** @brief Return the approximate Hessian matrix at the final solution (from L-BFGS-B). */
            Eigen::MatrixXf Hessian() const {return solver.final_approx_hessian();}
            /** @brief Return the approximate inverse Hessian (parameter covariance) at the final solution. */
            Eigen::MatrixXf InverseHessian() const {return solver.final_approx_inverse_hessian();}
            /** @brief Return the parameter covariance matrix (alias for InverseHessian()). */
            Eigen::MatrixXf Covariance() const {return InverseHessian();}
            /** @brief Return the best-fit parameter vector. */
            Eigen::VectorXf BestFit() const {return best_fit;}

            /**
             * @brief Return a covariance matrix rescaled by the goodness-of-fit chi-squared.
             * @details Useful when the L-BFGS-B Hessian-based uncertainties are not trusted;
             * scales the covariance by chi2 / (n_datapoint - nparams) (reduced chi-squared).
             * @param chi2        Best-fit chi-squared value.
             * @param n_datapoint Number of data bins used in the fit.
             * @return Rescaled parameter covariance matrix.
             */
            Eigen::MatrixXf ScaledCovariance(float chi2, int n_datapoint) const {
                const long ndof = (long)n_datapoint - (long)best_fit.size();
                if(ndof <= 0) {
                    log<LOG_WARNING>(L"%1% || ScaledCovariance: n_datapoint (%2%) <= nparams (%3%); returning unscaled covariance.")
                        % __func__ % n_datapoint % best_fit.size();
                    return Covariance();
                }
                return Covariance()*chi2/float(ndof);
            }


    };
    
    /**
     * @brief Generate a Latin hypercube sample in the unit hypercube.
     * @param num_samples  Number of sample points to generate.
     * @param dimensions   Dimensionality of the parameter space.
     * @param dis          Uniform real distribution used for randomisation.
     * @param gen          Mersenne Twister random number generator.
     * @return A vector of @p num_samples points, each a vector of @p dimensions values in [0, 1].
     */
    std::vector<std::vector<float>> latin_hypercube_sampling(size_t num_samples, size_t dimensions, std::uniform_real_distribution<float>&dis, std::mt19937 &gen);

    /**
     * @brief Rescale unit-hypercube Latin hypercube samples to the physical parameter bounds.
     * @param samples  The samples to rescale in-place (each element is one sample point).
     * @param ub       Upper bounds for each dimension.
     * @param lb       Lower bounds for each dimension.
     */
    void recenter_latin_samples(std::vector<std::vector<float>> &samples, const Eigen::VectorXf &ub, const Eigen::VectorXf &lb);

}

#endif
