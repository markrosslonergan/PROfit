/**
 * @file PROcovariance.h
 * @brief Shared covariance-matrix chi-squared machinery for PROfit metrics.
 * @author PROfit Collaboration
 *
 * @details Implements the shared covariance-matrix statistic:
 *   chi2 = (data - pred)^T M^{-1} (data - pred) + pull_penalty
 * where M is the combined statistical and systematic covariance. Concrete
 * subclasses provide the statistical variance convention used to build M.
 */
#ifndef PROCOVARIANCE_H_
#define PROCOVARIANCE_H_

// STANDARD
#include <string>
#include <vector>
#include <optional>

#include <Eigen/Eigen>

// OUR INCLUDES
#include "PROconfig.h"
#include "PROdata.h"
#include "PROsyst.h"
#include "PROpeller.h"
#include "PROmodel.h"
#include "PROmetric.h"
#include "PROcess.h"

namespace PROfit{

    /**
     * @brief Common engine for covariance-matrix chi-squared metrics.
     * @details Gathers the MC store (PROpeller), systematic object (PROsyst), and oscillation
     * model (PROmodel) into a single callable object whose operator() returns the chi-squared
     * value and gradient for use by PROfitter. PROchi, PROpearson, and PROCNP
     * specialize only the per-bin statistical variance. All heavy objects are
     * stored as references or pointers to objects owned by the caller.
     */
    class PROcovariance : public PROmetric
    {
        protected:
            // NOTE: declaration order is load-bearing. owned_config / owned_peller must be
            // declared (and therefore initialised) BEFORE the config / peller references
            // below, which bind to them when own_config_and_peller is set.
            //
            // CNP historically owns immutable config/propeller snapshots for
            // FC workers. Other covariance metrics leave these empty and bind
            // config/peller directly to the caller-owned inputs.
            std::optional<PROconfig> owned_config;
            std::optional<PROpeller> owned_peller;

            /**
             * @brief Per-bin statistical variance used to build M inside operator().
             * @param collapsed_prediction  Predicted spectrum, already collapsed to the
             *                              fitting variable's bin space.
             * @param comparison            The spectrum the prediction is compared against:
             *                              the observed data, or its area-normalised form in
             *                              shape_only mode.
             * @param param                 Full parameter vector when available, so a metric
             *                              can derive its variance from something other than
             *                              the shifted prediction (PROCNP uses the
             *                              physics-only CV). nullptr when not available.
             * @return Vector of variances in the collapsed bin space. Bins whose variance is
             *         not strictly positive are dropped from the fit, so an implementation
             *         that wants to keep every bin must floor its variance.
             */
            virtual Eigen::VectorXf statisticalVariances(
                const Eigen::VectorXf &collapsed_prediction, const Eigen::VectorXf &comparison,
                const Eigen::VectorXf *param = nullptr) const = 0;

            /**
             * @brief True if statisticalVariances() depends on the prediction.
             * @details When true, the full-FD gradient modes rebuild the statistical part of
             * M at every finite-difference point instead of reusing the base-point one.
             */
            virtual bool statisticalVariancesDependOnPrediction() const { return false; }

            /**
             * @brief Per-bin statistical variance used by getSingleChannelChi().
             * @details Defaults to statisticalVariances() with no parameter vector. PROchi and
             * PROCNP override it because their per-channel diagnostic path has historically
             * used a different convention from their fit path (see those overrides).
             * @param collapsed_cv  Predicted spectrum, already collapsed.
             * @param comparison    Observed data, area-normalised in shape_only mode.
             */
            virtual Eigen::VectorXf singleChannelStatVariances(
                const Eigen::VectorXf &collapsed_cv, const Eigen::VectorXf &comparison) const;

            /**
             * @brief Gradient mode used when GradientAnalytic is requested but unavailable.
             * @details Defaults to PROmetric::GradientFallback (Gauss-Newton linearised FD),
             * matching upstream's choice for PROCNP. PROpearson overrides this with a
             * full-FD mode: its statistical variance is the prediction itself, so the
             * (M⁻¹δ)ᵀ(dM/dθ)(M⁻¹δ) term the linearised mode drops is first-order in the
             * fitted parameters, and dropping it stalls the minimiser.
             */
            virtual GradientMode analyticFallbackMode() const { return GradientFallback; }

            /**
             * @brief Populate the constant-bin-selection cache from a fixed variance vector.
             * @details Opt-in, called from a concrete metric's constructor. Only valid when
             * that metric guarantees its variances (and therefore the set of bins with a
             * positive variance) never change during the fit — PROchi outside shape_only mode
             * is the only such case today. Metrics that do not call it rebuild the reduced
             * statistical covariance on every operator() invocation.
             */
            void buildConstantStatCache(const Eigen::VectorXf &variances);

        public:
            const PROconfig &config;  ///< Analysis configuration (non-owning reference).
            const PROpeller &peller;  ///< MC event store (non-owning reference).

            // Cached non-empty-bin slicing, valid only when the selected bins AND their
            // variances are both independent of the prediction (i.e. PROchi outside
            // shape_only mode). Every other case rebuilds them per call.
            std::vector<Eigen::Index> nec_indices;          ///< Indices of bins with positive variance; empty if cache invalid.
            Eigen::MatrixXf nec_reduced_stat_cov;           ///< Reduced statistical covariance for nec_indices.
            bool nec_valid = false;                         ///< True iff buildConstantStatCache() populated the cache above.

            /*Function: Constructor bringing all objects together*/
            PROcovariance(const std::string tag, const PROconfig &conin, const PROpeller &pin, const PROsyst *systin, const PROmodel &modelin, const PROdata &datain, EvalStrategy strat = EventByEvent, bool shape_only = false, std::vector<float> physics_param_fixed = std::vector<float>(), bool own_config_and_peller = false);

            PROcovariance(const PROcovariance &) = delete;
            PROcovariance &operator=(const PROcovariance &) = delete;

            /*Function: operator() is what is passed to minimizer.*/
            using PROmetric::operator();
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool rungradient);

            /** @brief Reset cached state and clear any fixed-parameter list. */
            virtual void reset();

            /** @brief Replace the internal systematic pointer with @p new_syst. */
            virtual void override_systs(const PROsyst &new_syst) {
                syst = &new_syst;
                fs_cache.invalidate();
            }

            /**
             * @brief Compute the chi-squared contribution from a single analysis channel.
             * @param global_channel_index  Global channel index.
             * @param cv                    Predicted (CV) spectrum.
             * @param var_index             Variable index.
             * @return Chi-squared for that channel.
             */
            float getSingleChannelChi(size_t global_channel_index, const PROspec &cv, size_t var_index, const Eigen::MatrixXf &projection = Eigen::MatrixXf());

            /**
             * @brief Print a breakdown of the chi-squared contributions at @p param.
             * @param param  Full parameter vector (physics + splines).
             */
            void print(const Eigen::VectorXf &param);
    };
}
#endif
