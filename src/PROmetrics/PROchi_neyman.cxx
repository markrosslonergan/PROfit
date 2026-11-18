#include "PROmetrics/PROchi_neyman.h"

using namespace PROfit;

PROchi::PROchi(const std::string tag, const PROconfig &conin, const PROpeller &pin,
               const PROsyst *systin, const PROmodel &modelin, const PROdata &datain,
               EvalStrategy strat, bool shape_only,
               std::vector<float> physics_param_fixed)
    : PROcovariance(tag, conin, pin, systin, modelin, datain, strat, shape_only,
                    physics_param_fixed) {
    // Neyman variance is the observed data, so both the set of usable bins and their
    // variances are constant across the whole fit and can be built once here. This holds
    // in shape_only mode too: it is the PREDICTION that is rescaled per channel there,
    // never the data.
    // Statistical overlap with another analysis: the shared counts are the off-diagonals.
    buildConstantStatCache(data.Spec(), data.has_overlap ? &data.overlap : nullptr);
}

Eigen::VectorXf PROchi::statisticalVariances(
    [[maybe_unused]] const Eigen::VectorXf &collapsed_prediction, const Eigen::VectorXf &comparison,
    [[maybe_unused]] const Eigen::VectorXf *param) const {
    return comparison;
}

Eigen::VectorXf PROchi::singleChannelStatVariances(
    [[maybe_unused]] const Eigen::VectorXf &collapsed_cv, const Eigen::VectorXf &comparison) const {
    // Unlike the fit path (which drops empty data bins outright), the per-channel
    // diagnostic keeps every bin and floors the variance instead:
    //
    // GP: What do you do if the MC has 0 events in a bin?
    //     Proposed solution (hack?) here. Set the error to 1. This will return
    //     the correct answer if there are no data events in the bin. It is a bit
    //     iffier if there are data events in the bin, we may want to implement some
    //     error handling there.
    //
    // The comparison is the observed data in every mode (shape_only rescales the
    // prediction, not the data), so the one-event floor always applies.
    return comparison.array().cwiseMax(1.0f).matrix();
}

PROmetric *PROchi::Clone() const {
    return new PROchi(model_tag, config, peller, syst, model, data,
                      strat, shape_only, physics_param_fixed);
}
