/**
 * @file PROCNP.h
 * @brief Combined Neyman-Pearson covariance metric for PROfit.
 */
#ifndef PROCNP_H_
#define PROCNP_H_

#include "PROcovariance.h"

namespace PROfit {
    /**
     * @brief Combined Neyman-Pearson covariance metric.
     * @note PROCNP retains owned copies of the configuration and propeller for
     * thread-safe FC pseudo-experiments. The model remains a non-owning
     * reference to avoid slicing derived PROmodel implementations, while data
     * is owned by PROmetric.
     */
    class PROCNP : public PROcovariance {
        protected:
            Eigen::VectorXf statisticalVariances(const PROspec &prediction,
                                                 const Eigen::VectorXf &data,
                                                 const Eigen::VectorXf *param = nullptr) const override;
            bool statisticalVariancesDependOnPrediction() const override;

        public:
            PROCNP(const std::string tag, const PROconfig &conin, const PROpeller &pin,
                   const PROsyst *systin, const PROmodel &modelin, const PROdata &datain,
                   EvalStrategy strat = EventByEvent, bool shape_only = false,
                   std::vector<float> physics_param_fixed = std::vector<float>());

            PROmetric *Clone() const override;
    };
}

#endif
