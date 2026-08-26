/**
 * @file PROmetric.h
 * @brief Abstract base class defining the chi-squared metric interface for PROfit optimisers.
 * @author PROfit Collaboration
 *
 * @details PROmetric is the pure-virtual interface that connects the physics chi-squared
 * calculation to the PROfitter multi-start optimiser.  Concrete implementations
 * (PROchi, PROCNP, PROpoisson) compute different chi-squared statistics while sharing
 * the common bounds, fixed-parameter, and call-counting infrastructure defined here.
 */
#ifndef PROMETRIC_H
#define PROMETRIC_H

#include "PROsyst.h"
#include "PROmodel.h"
#include "PROconfig.h"
#include "PROdata.h"
#include "PROcess.h"

#include <Eigen/Eigen>
#include <atomic>
#include <cctype>
#include <string>

namespace PROfit {

    /**
     * @brief Abstract base class for PROfit chi-squared metrics passed to the optimiser.
     * @details Defines the interface required by PROfitter: parameter bounds, fixed-parameter
     * masking, call counting, and the functor operator() that returns chi-squared and gradient.
     * All concrete metrics (PROchi, PROCNP, PROpoisson) derive from this class.
     */
    class PROmetric {
        public:
            /**
             * @brief Strategy for evaluating the chi-squared (and its gradient).
             */
            enum EvalStrategy {
                EventByEvent, ///< Evaluate event-by-event (slowest but most accurate for oscillation weights).
                BinnedGrad,   ///< Use pre-binned histograms with gradient calculation.
                BinnedChi2    ///< Use pre-binned histograms, chi-squared only (no analytic gradient).
            };

            /**
             * @brief Strategy for computing the finite-difference gradient.
             * @details Two orthogonal axes: the FD stencil (central vs one-sided)
             * and the chi² treatment (full vs linearised).
             *
             *   - Full: each FD perturbation re-computes the entire chi²
             *     including the covariance build, collapse, and Cholesky solve
             *     (M = stat + reduced collapsed full covariance, with M built
             *     from the perturbed spectrum).
             *   - Linearised (Gauss-Newton style): freezes M at the base point
             *     and uses the chain rule
             *         dchi²/dθ_i ≈ 2 (M⁻¹ δ_b)^T (dδ/dθ_i) + dP/dθ_i
             *     which drops the (M⁻¹δ)^T (dM/dθ) (M⁻¹δ) term. That term is
             *     second-order in δ and vanishes at the minimum, so the
             *     approximation is exact at convergence and very small far
             *     from it. The pull derivative is computed analytically.
             *     For PROpoisson the same idea applies with dchi²/ds = 2(1-n/s)
             *     replacing M⁻¹δ — the linearised form is exact (modulo FD
             *     truncation in dδ/dθ).
             *
             * Combined with the FD stencil this gives four configurations:
             *
             *  | Mode                  | δ FD       | M handling        | Pull deriv |
             *  |-----------------------|------------|-------------------|------------|
             *  | GradientCentralFull   | central    | rebuilt per FD    | via FD     |
             *  | GradientOneSidedFull  | one-sided  | rebuilt per FD    | via FD     |
             *  | GradientCentralLin    | central    | frozen at base    | analytic   |
             *  | GradientOneSidedLin   | one-sided  | frozen at base    | analytic   |
             *
             * Boundary handling: any FD step that lands on a parameter bound is
             * downgraded to a one-sided stencil pointing into the interior,
             * regardless of the chosen mode. The "out-of-bounds gradient bounce"
             * (zeroing dchi²/dθ when it pushes further out of the box) is
             * preserved across all modes — LBFGSB depends on it.
             */
            enum GradientMode {
                GradientCentralFull,    ///< Central FD on full chi². Most accurate, slowest (rebuilds covariance + Cholesky per FD step).
                GradientOneSidedFull,   ///< One-sided forward FD on full chi². ~2× faster, O(h) vs O(h²).
                GradientCentralLin,     ///< Default: central FD on δ only, M frozen at base (Gauss-Newton). 5–10× faster; exact at the minimum.
                GradientOneSidedLin,    ///< One-sided FD on δ only, M frozen at base. 10–20× faster.
            };

            std::vector<bool> is_fixed; ///< Per-parameter flags: true if the parameter is held fixed during fitting.
            Eigen::VectorXf  lb;        ///< Lower bounds for all parameters.
            Eigen::VectorXf  ub;        ///< Upper bounds for all parameters.


            /** @brief Replace the internal systematic object pointer with @p new_syst. */
            virtual void override_systs(const PROsyst &new_syst) = 0;
            /**
             * @brief Evaluate the chi-squared and its gradient.
             * @param param     Current parameter vector.
             * @param gradient  Output gradient vector (same size as @p param); filled on return.
             * @return Chi-squared value.
             */
            float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient) {
                return (*this)(param, gradient, true);
            }
            /**
             * @brief Evaluate the chi-squared, optionally skipping gradient computation.
             * @param param       Current parameter vector.
             * @param gradient    Output gradient vector; only filled when @p rungradient is true.
             * @param rungradient If true, compute the gradient; if false, skip it for speed.
             * @return Chi-squared value.
             */
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool rungradient) = 0;
            /** @brief Reset any cached state (e.g. last parameter vector and value). */
            virtual void reset() = 0;
            /** @brief Return a heap-allocated deep copy of this PROmetric. */
            virtual PROmetric *Clone() const = 0;
            /** @brief Return a const reference to the physics model used by this metric. */
            const PROmodel &GetModel() const { return model; }
            /** @brief Return a const reference to the systematic object used by this metric. */
            const PROsyst &GetSysts() const { return *syst; }
            /**
             * @brief Compute the chi-squared contribution from a single channel.
             * @param channel_index  Global channel index.
             * @param cv             Central-value (predicted) spectrum.
             * @param var_index      Variable index.
             * @return Chi-squared for that channel.
             */
            virtual float getSingleChannelChi(size_t channel_index, const PROspec& cv, size_t var_index, const Eigen::MatrixXf &projection = Eigen::MatrixXf()) = 0;
            virtual ~PROmetric() {}
            /**
             * @brief Fix a spline nuisance parameter at a specific value.
             * @param idx   0-based spline index.
             * @param val   Value to fix the spline at.
             */
            void fixSpline(int idx, float val) {
                fixed_index = idx;
                fixed_val = val;
            }
            /**
             * @brief Compute the Gaussian pull penalty for the given nuisance parameter vector.
             * @param systs  Spline nuisance parameter values.
             * @return Scalar pull penalty (chi2 contribution from priors).
             */
            virtual float Pull(const Eigen::VectorXf &systs) {
                Eigen::VectorXf centered = systs - syst->spline_centers;
                for(size_t i = 0; i < syst->spline_prior_types.size(); ++i) {
                    if(syst->spline_prior_types[i] == SplinePriorType::Uniform) centered(i) = 0.0f;
                }
                if(!correlated_systematics) {
                    return (centered.array().square() / syst->spline_priors.array().square()).sum();
                }
                return centered.dot(prior_covariance_inv * centered);
            }
            /**
             * @brief Print a human-readable summary of the metric evaluation at @p param.
             * @param param  Parameter vector to evaluate at.
             */
            virtual void print(const Eigen::VectorXf &param) = 0;
            /**
             * @brief Return the total number of parameters (physics + spline nuisance).
             * @return nparams from the model plus the number of spline systematics.
             */
            size_t nParams() const {return GetModel().nparams + GetSysts().GetNSplines();}

            PROmetric(const PROmetric &other)
                : model_tag(other.model_tag), syst(other.syst), model(other.model), data(other.data),
                  strat(other.strat), shape_only(other.shape_only),
                  physics_param_fixed(other.physics_param_fixed), fixed_index(other.fixed_index),
                  fixed_val(other.fixed_val), last_param(other.last_param), last_value(other.last_value),
                  correlated_systematics(other.correlated_systematics),
                  prior_covariance(other.prior_covariance),
                  prior_covariance_inv(other.prior_covariance_inv), fs_cache(other.fs_cache),
                  call_count(other.call_count.load()), gradient_mode(other.gradient_mode),
                  active_bins(other.active_bins) {}
            PROmetric& operator=(const PROmetric&) { return *this; }


            Eigen::VectorXf LowerBound() const {
                size_t nphys = GetModel().nparams;
                size_t nparams = nParams();
                Eigen::VectorXf lb = Eigen::VectorXf::Constant(nparams, -3.0);
                for (size_t i = 0; i < nphys; ++i) {
                    lb(i) = GetModel().lb(i);
                }
                for(size_t i = nphys; i < nparams; ++i) {
                    lb(i) = GetSysts().spline_lo[i-nphys];
                }
                return lb;
            }

            Eigen::VectorXf UpperBound() const {
                size_t nphys = GetModel().nparams;
                size_t nparams = nParams();
                Eigen::VectorXf ub = Eigen::VectorXf::Constant(nparams, 3.0);
                for (size_t i = 0; i < nphys; ++i) {
                    ub(i) = GetModel().ub(i);
                }
                for(size_t i = nphys; i < nparams; ++i) {
                    ub(i) = GetSysts().spline_hi[i-nphys];
                }
                return ub;
            }

            /** @brief True if a fit-region (active-bin) mask was snapshotted from the config. */
            bool hasActiveBinMask() const { return !active_bins.empty(); }

            /** @brief True if collapsed bin @p i (fitting variable) is in the fit region. Always true without a mask. */
            bool binActive(Eigen::Index i) const {
                return active_bins.empty() || (i >= 0 && (size_t)i < active_bins.size() && active_bins[(size_t)i] != 0);
            }

            /** @brief Return the total number of times operator() has been called since last reset. */
            size_t getCallCount() const { return call_count; }
            /** @brief Reset the call counter to zero. */
            void resetCallCount() { call_count = 0; }


            /**
             * @brief Set parameter bounds and mark zero-range parameters as fixed.
             * @param lbin  Lower bounds vector.
             * @param ubin  Upper bounds vector.
             */
            void setBounds(const Eigen::VectorXf& lbin, const Eigen::VectorXf& ubin) {
                lb = lbin;
                ub = ubin;
                is_fixed.resize(lbin.size());
                for(int i = 0; i < lbin.size(); ++i) {
                    is_fixed[i] = (std::abs(ubin(i) - lbin(i)) < 1e-10);
                }
            }

            /** @brief Clear the is_fixed mask so that all parameters are free to be optimised. */
            void freeParams() {
                is_fixed.clear();
            }

            /** @brief Currently configured gradient mode. */
            GradientMode getGradientMode() const { return gradient_mode; }

            /** @brief Set the gradient mode used by operator() when rungradient=true. */
            void setGradientMode(GradientMode m) { gradient_mode = m; }

            /**
             * @brief Parse a string token into a GradientMode.
             * @details Accepts (case-insensitive) "central", "central-full",
             * "one-sided", "one-sided-full", "central-lin", "central-linearised",
             * "one-sided-lin", "one-sided-linearised". Returns @p fallback on
             * unrecognised input (caller is expected to log a warning).
             */
            static GradientMode parseGradientMode(const std::string &tok,
                                                   GradientMode fallback = GradientCentralFull) {
                std::string s = tok;
                for (auto &c : s) c = (char)std::tolower((unsigned char)c);
                if (s == "central"          || s == "central-full")             return GradientCentralFull;
                if (s == "one-sided"        || s == "one-sided-full"
                                            || s == "onesided"
                                            || s == "onesided-full")            return GradientOneSidedFull;
                if (s == "central-lin"      || s == "central-linearised"
                                            || s == "central-linearized")       return GradientCentralLin;
                if (s == "one-sided-lin"    || s == "one-sided-linearised"
                                            || s == "onesided-lin"
                                            || s == "onesided-linearised"
                                            || s == "one-sided-linearized")     return GradientOneSidedLin;
                return fallback;
            }

            /** @brief Human-readable label for a GradientMode (for diagnostic logging). */
            static const char *gradientModeName(GradientMode m) {
                switch (m) {
                    case GradientCentralFull:  return "central-full";
                    case GradientOneSidedFull: return "one-sided-full";
                    case GradientCentralLin:   return "central-linearised";
                    case GradientOneSidedLin:  return "one-sided-linearised";
                }
                return "unknown";
            }

        protected:
            PROmetric(const std::string &tag, const PROconfig &config, const PROsyst *systin,
                      const PROmodel &modelin, const PROdata &datain, EvalStrategy strategy,
                      bool shape_only_in, const std::vector<float> &fixed_physics)
                : model_tag(tag), syst(systin), model(modelin), data(datain), strat(strategy),
                  shape_only(shape_only_in), physics_param_fixed(fixed_physics), fixed_index(-999),
                  fixed_val(0.0f),
                  last_param(Eigen::VectorXf::Zero(modelin.nparams + systin->GetNSplines())),
                  last_value(0.0f), correlated_systematics(false) {
                snapshotActiveBins(config);
            }

            std::string model_tag; ///< String tag identifying the physics model in use.
            const PROsyst *syst;   ///< Systematic object (non-owning pointer).
            const PROmodel &model; ///< Physics model (non-owning reference).
            const PROdata data;    ///< Observed data spectrum (owned copy).
            EvalStrategy strat;    ///< Evaluation strategy.
            bool shape_only;       ///< If true, evaluate area-normalised spectra.
            std::vector<float> physics_param_fixed; ///< Fixed physics-parameter values (empty = none fixed).
            int fixed_index;       ///< Index fixed during a scan (-1-like sentinel = none).
            float fixed_val;       ///< Value at which the scanned parameter is fixed.

            Eigen::VectorXf last_param; ///< Parameter vector from the most recent evaluation.
            float last_value;           ///< Metric value from the most recent evaluation.

            bool correlated_systematics;          ///< Whether correlated nuisance priors are enabled.
            Eigen::MatrixXf prior_covariance;     ///< Prior covariance for nuisance parameters.
            Eigen::MatrixXf prior_covariance_inv; ///< Cached inverse prior covariance.

            FillSpectraCache fs_cache; ///< Per-metric cache for FillSpectra.

            mutable std::atomic<size_t> call_count{0}; ///< Thread-safe counter of operator() invocations.
            GradientMode gradient_mode = GradientCentralLin; ///< Default: Gauss-Newton linearised gradient (M frozen at base). Use --grad-mode central-full for the legacy full-FD behaviour.

            /** @brief Snapshot of PROconfig's fit-region mask for the fitting variable
             *  (collapsed space); empty = no mask, all bins active. Taken at construction
             *  in every concrete metric — Clone() re-runs the constructor with the same
             *  config, and FC/AFC workers construct fresh metrics from the same config,
             *  so the mask propagates everywhere without per-call-site plumbing.
             *  NOTE: like lb/ub/is_fixed, this does NOT survive a raw copy-construction
             *  (PROmetric's copy ctor is a no-op); always use Clone(). */
            std::vector<char> active_bins;

            /** @brief Fill active_bins from the config's mask for its primary fitting variable.
             *  Call from every concrete metric constructor. */
            void snapshotActiveBins(const PROconfig &c) {
                active_bins.clear();
                if(!c.HasActiveBins(c.i_prime)) return;
                const size_t n = c.m_num_variable_bins_total_collapsed[c.i_prime];
                active_bins.resize(n);
                size_t n_active = 0;
                for(size_t i = 0; i < n; ++i) {
                    active_bins[i] = c.IsBinActive(c.i_prime, i) ? 1 : 0;
                    n_active += active_bins[i];
                }
                log<LOG_INFO>(L"%1% || Metric fit region: %2% of %3% collapsed bins active.") % __func__ % n_active % n;
            }

    };

};

#endif
