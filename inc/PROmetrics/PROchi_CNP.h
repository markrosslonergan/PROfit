/**
 * @file PROchi_CNP.h
 * @brief Combined Neyman-Pearson covariance metric for PROfit (canonical CLI name: "CNP").
 * @author PROfit Collaboration
 *
 * @details Defines PROCNP, the Combined Neyman-Pearson statistic whose per-bin variance is
 * 3/(1/n + 2/mu) — the harmonic blend of the Neyman (observed n) and Pearson (predicted mu)
 * prescriptions — with mu the full prediction being tested (physics and nuisance parameters),
 * as in X. Ji et al., NIM A 961 (2020) 163677. See PROcovariance for the shared covariance,
 * pull, and gradient machinery.
 */
#ifndef PROCNP_H_
#define PROCNP_H_

#include "PROcovariance.h"

namespace PROfit {
    /**
     * @brief Combined Neyman-Pearson covariance metric.
     * @details The variance is a per-bin function of the collapsed prediction, so the
     * analytic gradient comes from statisticalVarianceSlope() exactly as for PROchi_pearson.
     * @note PROCNP retains owned copies of the configuration and propeller for
     * thread-safe FC pseudo-experiments. The model remains a non-owning
     * reference to avoid slicing derived PROmodel implementations, while data
     * is owned by PROmetric.
     */
    class PROCNP : public PROcovariance {
        protected:
            Eigen::VectorXf statisticalVariances(
                const Eigen::VectorXf &collapsed_prediction, const Eigen::VectorXf &comparison,
                const Eigen::VectorXf *param = nullptr) const override;

            bool statisticalVariancesDependOnPrediction() const override;

            /** @brief d var / d mu per bin: 6n²/(mu+2n)², 1/2 at n = 0, 0 where mu is floored. */
            Eigen::VectorXf statisticalVarianceSlope(const Eigen::VectorXf &collapsed_prediction) const override;

            /** @brief One-sided full FD (EventByEvent only), as for PROchi_pearson. */
            GradientMode analyticFallbackMode() const override;

            Eigen::VectorXf singleChannelStatVariances(
                const Eigen::VectorXf &collapsed_cv, const Eigen::VectorXf &comparison) const override;

        public:
            PROCNP(const std::string tag, const PROconfig &conin, const PROpeller &pin,
                   const PROsyst *systin, const PROmodel &modelin, const PROdata &datain,
                   EvalStrategy strat = EventByEvent, bool shape_only = false,
                   std::vector<float> physics_param_fixed = std::vector<float>());

            PROmetric *Clone() const override;
    };
}

#endif
