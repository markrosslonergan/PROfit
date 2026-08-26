#include "PROmetrics/PROchi.h"

using namespace PROfit;

PROchi::PROchi(const std::string tag, const PROconfig &conin, const PROpeller &pin,
               const PROsyst *systin, const PROmodel &modelin, const PROdata &datain,
               EvalStrategy strat, bool shape_only,
               std::vector<float> physics_param_fixed)
    : PROcovariance(tag, conin, pin, systin, modelin, datain, strat, shape_only,
                    physics_param_fixed) {
    gradient_mode = GradientOneSidedFull;
}

Eigen::VectorXf PROchi::statisticalVariances(
    [[maybe_unused]] const PROspec &prediction, const Eigen::VectorXf &data,
    [[maybe_unused]] const Eigen::VectorXf *param) const {
    return data;
}

PROmetric *PROchi::Clone() const {
    return new PROchi(model_tag, config, peller, syst, model, data,
                      strat, shape_only, physics_param_fixed);
}
