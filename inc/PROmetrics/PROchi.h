/**
 * @file PROchi.h
 * @brief Neyman covariance-matrix chi-squared metric for PROfit.
 */
#ifndef PROCHI_H_
#define PROCHI_H_

#include "PROcovariance.h"

namespace PROfit {
    class PROchi : public PROcovariance {
        protected:
            Eigen::VectorXf statisticalVariances(
                const PROspec &prediction, const Eigen::VectorXf &data,
                const Eigen::VectorXf *param = nullptr) const override;

        public:
            PROchi(const std::string tag, const PROconfig &conin, const PROpeller &pin,
                   const PROsyst *systin, const PROmodel &modelin, const PROdata &datain,
                   EvalStrategy strat = EventByEvent, bool shape_only = false,
                   std::vector<float> physics_param_fixed = std::vector<float>());

            PROmetric *Clone() const override;
    };
}

#endif
