#include "PROmetrics/PROpearson.h"

using namespace PROfit;

namespace {
    /// Floor applied to the predicted count used as the Pearson variance. Keeps M
    /// positive-definite, and keeps the set of usable bins independent of the
    /// parameters: see PROpearson::statisticalVariances.
    constexpr float kMinPearsonMu = 1e-6f;
}

PROpearson::PROpearson(const std::string tag, const PROconfig &conin, const PROpeller &pin,
                       const PROsyst *systin, const PROmodel &modelin, const PROdata &datain,
                       EvalStrategy strat, bool shape_only,
                       std::vector<float> physics_param_fixed)
    : PROcovariance(tag, conin, pin, systin, modelin, datain, strat, shape_only,
             physics_param_fixed) {
    // No buildConstantStatCache() here: the Pearson variance is the prediction, so the
    // reduced statistical covariance has to be rebuilt on every evaluation. The floor in
    // statisticalVariances() does keep the *set* of bins constant.
}

bool PROpearson::statisticalVariancesDependOnPrediction() const {
    return true;
}

Eigen::VectorXf PROpearson::statisticalVariances(
    const Eigen::VectorXf &collapsed_prediction, [[maybe_unused]] const Eigen::VectorXf &comparison,
    [[maybe_unused]] const Eigen::VectorXf *param) const {
    // The floor matters for more than conditioning. PROcovariance drops any bin whose
    // variance is not strictly positive, so an unfloored Pearson variance would let the
    // set of fitted bins move with the parameters: a bin whose prediction reaches zero
    // would silently leave the fit, making the chi2 discontinuous and — worse — costing
    // nothing at all when that bin still holds data. With the floor the bin set is fixed
    // and such a bin instead contributes a huge (but finite) penalty, which is the
    // behaviour Pearson's chi2 -> infinity limit should produce.
    return collapsed_prediction.array().cwiseMax(kMinPearsonMu).matrix();
}

PROmetric *PROpearson::Clone() const {
    return new PROpearson(model_tag, config, peller, syst, model, data,
                          strat, shape_only, physics_param_fixed);
}
