/**
 * @file PROchi_pearson.h
 * @brief Pearson covariance-matrix chi-squared metric for PROfit (canonical CLI name: "pearson").
 * @author PROfit Collaboration
 * 
 * @details Defines PROchi_pearson, which implements the Pearson chi-squared statistic
 * using the predicted event count as the statistical variance in each bin:
 *   chi2 = (prediction - data)^T M^-1 (prediction - data) + pull_penalty,
 * where M is the sum of the prediction-dependent statistical covariance
 * diag(prediction) and the systematic covariance. This differs from PROchi
 * (Neyman), which uses observed data counts for the statistical variance, and
 * PROCNP, which combines the Neyman and Pearson prescriptions. PROchi_pearson
 * inherits PROcovariance's common covariance, systematic-pull, and gradient
 * machinery while overriding the statistical-variance calculation.
 */
#ifndef PROCHI_PEARSON_H_
#define PROCHI_PEARSON_H_

#include "PROcovariance.h"

namespace PROfit {
    /** Pearson chi-squared using predicted bin counts for statistical variance. */
    class PROchi_pearson : public PROcovariance {
        protected:
            Eigen::VectorXf statisticalVariances(
                const Eigen::VectorXf &collapsed_prediction, const Eigen::VectorXf &comparison,
                const Eigen::VectorXf *param = nullptr) const override;

            bool statisticalVariancesDependOnPrediction() const override;

            Eigen::VectorXf statisticalVarianceSlope(const Eigen::VectorXf &collapsed_prediction) const override;

            GradientMode analyticFallbackMode() const override;

        public:
            PROchi_pearson(const std::string tag, const PROconfig &conin, const PROpeller &pin,
                       const PROsyst *systin, const PROmodel &modelin, const PROdata &datain,
                       EvalStrategy strat = EventByEvent, bool shape_only = false,
                       std::vector<float> physics_param_fixed = std::vector<float>());

            PROmetric *Clone() const override;
    };
}

#endif
