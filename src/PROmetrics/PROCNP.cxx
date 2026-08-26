#include "PROmetrics/PROCNP.h"

#include "PROcess.h"

#include <algorithm>

using namespace PROfit;

PROCNP::PROCNP(const std::string tag, const PROconfig &conin, const PROpeller &pin,
               const PROsyst *systin, const PROmodel &modelin, const PROdata &datain,
               EvalStrategy strat, bool shape_only,
               std::vector<float> physics_param_fixed)
    : PROcovariance(tag, conin, pin, systin, modelin, datain, strat, shape_only,
                    physics_param_fixed, true) {
    gradient_mode = GradientOneSidedFull;
    nec_valid = false;
    nec_indices.clear();
    nec_reduced_stat_cov.resize(0, 0);
}

bool PROCNP::statisticalVariancesDependOnPrediction() const {
    return true;
}

PROmetric *PROCNP::Clone() const {
    return new PROCNP(model_tag, config, peller, syst, model, data,
                      strat, shape_only, physics_param_fixed);
}

Eigen::VectorXf PROCNP::statisticalVariances(const PROspec &prediction,
                                             const Eigen::VectorXf &data,
                                             const Eigen::VectorXf *param) const {
    Eigen::VectorXf cv;
    if(param) {
        Eigen::VectorXf no_shift = Eigen::VectorXf::Zero(param->size());
        no_shift.head(model.nparams) = param->head(model.nparams);
        cv = CollapseMatrix(config, FillSpectra(config, peller, *syst, model, no_shift,
                                                strat != EventByEvent, config.i_prime).Spec());
    } else {
        cv = CollapseMatrix(config, prediction.Spec());
    }

    constexpr float kMinCNPMu = 1e-6f;
    Eigen::VectorXf variances(data.size());
    for(Eigen::Index i = 0; i < data.size(); ++i) {
        const float mu = std::max(cv(i), kMinCNPMu);
        variances(i) = data(i) == 0.0f
            ? mu / 2.0f
            : 3.0f / (1.0f / data(i) + 2.0f / mu);
    }
    return variances;
}
