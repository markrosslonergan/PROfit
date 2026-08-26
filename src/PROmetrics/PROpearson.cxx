#include "PROmetrics/PROpearson.h"
#include "PROcess.h"

using namespace PROfit;

PROpearson::PROpearson(const std::string tag, const PROconfig &conin, const PROpeller &pin,
                       const PROsyst *systin, const PROmodel &modelin, const PROdata &datain,
                       EvalStrategy strat, bool shape_only,
                       std::vector<float> physics_param_fixed)
    : PROcovariance(tag, conin, pin, systin, modelin, datain, strat, shape_only,
             physics_param_fixed) {
    
    // We have to calculate non-empty prediction bins as pull parameters change, 
    // as opposed to Neyman where the non-emtpy data bins remain constant.
    nec_valid = false;
    nec_indices.clear();
    nec_reduced_stat_cov.resize(0, 0);
    gradient_mode = GradientOneSidedFull;
}

bool PROpearson::statisticalVariancesDependOnPrediction() const {
    return true;
}

PROmetric *PROpearson::Clone() const {
    return new PROpearson(model_tag, config, peller, syst, model, data,
                          strat, shape_only, physics_param_fixed);
}

Eigen::VectorXf PROpearson::statisticalVariances(
    const PROspec &prediction, [[maybe_unused]] const Eigen::VectorXf &data,
    [[maybe_unused]] const Eigen::VectorXf *param) const {
    return CollapseMatrix(config, prediction.Spec());
}
