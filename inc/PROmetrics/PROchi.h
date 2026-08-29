/**
 * @file PROchi.h
 * @brief Neyman covariance-matrix chi-squared metric for PROfit.
 * @author PROfit Collaboration
 *
 * @details Defines PROchi, the Neyman chi-squared: the per-bin statistical variance is the
 * observed event count, so the statistical covariance is fixed for the whole fit. See
 * PROcovariance for the shared covariance, pull, and gradient machinery.
 */
#ifndef PROCHI_H_
#define PROCHI_H_

#include "PROcovariance.h"

namespace PROfit {
    /** @brief Neyman chi-squared using observed bin counts for the statistical variance. */
    class PROchi : public PROcovariance {
        protected:
            Eigen::VectorXf statisticalVariances(
                const Eigen::VectorXf &collapsed_prediction, const Eigen::VectorXf &comparison,
                const Eigen::VectorXf *param = nullptr) const override;

            Eigen::VectorXf singleChannelStatVariances(
                const Eigen::VectorXf &collapsed_cv, const Eigen::VectorXf &comparison) const override;

        public:
            PROchi(const std::string tag, const PROconfig &conin, const PROpeller &pin,
                   const PROsyst *systin, const PROmodel &modelin, const PROdata &datain,
                   EvalStrategy strat = EventByEvent, bool shape_only = false,
                   std::vector<float> physics_param_fixed = std::vector<float>());

            PROmetric *Clone() const override;
    };
}

#endif
