#include "PROmetrics/PROchi_CNP.h"

#include <algorithm>

using namespace PROfit;

namespace {
    /// Floor on the predicted count entering the CNP variance. Without it a bin with
    /// mu <= 0 would zero the statistical diagonal and make M singular wherever the
    /// systematic covariance is also empty; it also keeps the set of fitted bins fixed.
    constexpr float kMinCNPMu = 1e-6f;
}

PROCNP::PROCNP(const std::string tag, const PROconfig &conin, const PROpeller &pin,
               const PROsyst *systin, const PROmodel &modelin, const PROdata &datain,
               EvalStrategy strat, bool shape_only,
               std::vector<float> physics_param_fixed)
    : PROcovariance(tag, conin, pin, systin, modelin, datain, strat, shape_only,
                    physics_param_fixed, true) {
    // No buildConstantStatCache() here: mu, and therefore the CNP variance, is the
    // prediction, so the reduced statistical covariance is rebuilt per evaluation.
}

bool PROCNP::statisticalVariancesDependOnPrediction() const {
    return true;
}

PROmetric::GradientMode PROCNP::analyticFallbackMode() const {
    // Reached only with EventByEvent. As for pearson, the variance moves with the prediction,
    // so the Gauss-Newton modes would drop a first-order term.
    return GradientOneSidedFull;
}

Eigen::VectorXf PROCNP::statisticalVariances(const Eigen::VectorXf &collapsed_prediction,
                                             const Eigen::VectorXf &comparison,
                                             [[maybe_unused]] const Eigen::VectorXf *param) const {
    // mu is the full prediction at the tested point (per-channel rescaled onto the data when
    // shape-only), nuisance pulls included: CNP's cancellation of the Neyman and Pearson
    // biases needs mu to move in every direction the prediction does.
    Eigen::VectorXf variances(comparison.size());
    for(Eigen::Index i = 0; i < comparison.size(); ++i) {
        const float mu = std::max(collapsed_prediction(i), kMinCNPMu);
        variances(i) = comparison(i) == 0.0f
            ? mu / 2.0f
            : (float)(3.0 / (1.0 / comparison(i) + 2.0 / mu));
    }
    return variances;
}

Eigen::VectorXf PROCNP::statisticalVarianceSlope(const Eigen::VectorXf &collapsed_prediction) const {
    // var = 3 n mu / (mu + 2n)  =>  d var / d mu = 6 n^2 / (mu + 2n)^2;  var = mu/2 at n = 0.
    // The engine passes the observed data as the comparison, so n is data.Spec().
    const Eigen::VectorXf &obs = data.Spec();
    Eigen::VectorXf slope(collapsed_prediction.size());
    for(Eigen::Index i = 0; i < collapsed_prediction.size(); ++i) {
        const double mu = collapsed_prediction(i), n = obs(i);
        if(!(mu > kMinCNPMu)) slope(i) = 0.0f;
        else if(n == 0.0)     slope(i) = 0.5f;
        else                  slope(i) = (float)(6.0 * n * n / ((mu + 2.0 * n) * (mu + 2.0 * n)));
    }
    return slope;
}

Eigen::VectorXf PROCNP::singleChannelStatVariances(const Eigen::VectorXf &collapsed_cv,
                                                   const Eigen::VectorXf &comparison) const {
    // Same convention as the fit path: mu is the prediction handed in by the caller.
    // (The comparison is the observed data in every mode.)
    const Eigen::VectorXf &obs = data.Spec();
    Eigen::VectorXf variances(obs.size());
    for(Eigen::Index i = 0; i < obs.size(); ++i) {
        const float mu = std::max(collapsed_cv(i), kMinCNPMu);
        variances(i) = obs(i) == 0.0f
            ? mu / 2.0f
            : (float)(3.0 / (1.0 / comparison(i) + 2.0 / mu));
    }
    return variances;
}

PROmetric *PROCNP::Clone() const {
    return new PROCNP(model_tag, config, peller, syst, model, data,
                      strat, shape_only, physics_param_fixed);
}
