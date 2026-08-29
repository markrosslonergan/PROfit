#include "PROmetrics/PROCNP.h"

#include "PROcess.h"

#include <algorithm>

using namespace PROfit;

namespace {
    /// Floor on the predicted count entering the CNP variance. Without it a bin with
    /// mu <= 0 would zero the statistical diagonal and make M singular wherever the
    /// systematic covariance is also empty.
    constexpr float kMinCNPMu = 1e-6f;
}

PROCNP::PROCNP(const std::string tag, const PROconfig &conin, const PROpeller &pin,
               const PROsyst *systin, const PROmodel &modelin, const PROdata &datain,
               EvalStrategy strat, bool shape_only,
               std::vector<float> physics_param_fixed)
    : PROcovariance(tag, conin, pin, systin, modelin, datain, strat, shape_only,
                    physics_param_fixed, true) {
    // No buildConstantStatCache() here: mu, and therefore the CNP variance, moves with the
    // physics parameters, so the reduced statistical covariance is rebuilt per evaluation.
}

bool PROCNP::statisticalVariancesDependOnPrediction() const {
    return true;
}

void PROCNP::reset() {
    PROcovariance::reset();
    cnp_cv_cache_valid = false;
}

void PROCNP::override_systs(const PROsyst &new_syst) {
    PROcovariance::override_systs(new_syst);
    cnp_cv_cache_valid = false;
}

const Eigen::VectorXf &PROCNP::cachedNoshiftCollapsedCV(const Eigen::VectorXf &phys,
                                                        Eigen::Index param_size) const {
    if(cnp_cv_cache_valid && cnp_cached_phys.size() == phys.size() && cnp_cached_phys == phys)
        return cnp_cached_collapsed_cv;
    Eigen::VectorXf noshiftvec = Eigen::VectorXf::Zero(param_size);
    noshiftvec.head(model.nparams) = phys;
    // Deliberately the uncached FillSpectra overload: fs_cache is a single-slot cache keyed
    // on the physics/systematic half of the parameter vector, and routing this no-shift
    // call through it would evict the entry operator() is about to reuse.
    cnp_cached_collapsed_cv = CollapseMatrix(config, FillSpectra(config, peller, *syst, model,
                                                                noshiftvec, strat != EventByEvent,
                                                                config.i_prime).Spec());
    cnp_cached_phys = phys;
    cnp_cv_cache_valid = true;
    return cnp_cached_collapsed_cv;
}

Eigen::VectorXf PROCNP::statisticalVariances(const Eigen::VectorXf &collapsed_prediction,
                                             const Eigen::VectorXf &comparison,
                                             const Eigen::VectorXf *param) const {
    // mu comes from the physics-only CV when the parameter vector is available, so the
    // statistical part of M does not respond to the nuisance pulls. Fall back to the
    // shifted prediction when it is not.
    const Eigen::VectorXf &cv = param
        ? cachedNoshiftCollapsedCV(param->head(model.nparams), param->size())
        : collapsed_prediction;

    Eigen::VectorXf variances(comparison.size());
    for(Eigen::Index i = 0; i < comparison.size(); ++i) {
        const float mu = std::max(cv(i), kMinCNPMu);
        // Doubles in the harmonic blend match the historical promotion of the
        // 1.0 / 2.0 / 3.0 literals, so results stay bit-identical to pre-refactor CNP.
        variances(i) = comparison(i) == 0.0f
            ? mu / 2.0f
            : (float)(3.0 / (1.0 / comparison(i) + 2.0 / mu));
    }
    return variances;
}

Eigen::VectorXf PROCNP::singleChannelStatVariances(const Eigen::VectorXf &collapsed_cv,
                                                   const Eigen::VectorXf &comparison) const {
    // This per-channel diagnostic has always differed from the fit path above in three
    // ways, all preserved here: mu is the *shifted* prediction handed in by the caller
    // rather than the physics-only CV, the zero test is against the raw observed data
    // rather than the (possibly area-normalised) comparison spectrum, and mu is not
    // floored. Doubles match the historical promotion of the 1.0/2.0/3.0 literals.
    const Eigen::VectorXf &obs = data.Spec();
    Eigen::VectorXf variances(obs.size());
    for(Eigen::Index i = 0; i < obs.size(); ++i)
        variances(i) = obs(i) == 0.0f
            ? collapsed_cv(i) / 2.0f
            : (float)(3.0 / (1.0 / comparison(i) + 2.0 / collapsed_cv(i)));
    return variances;
}

PROmetric *PROCNP::Clone() const {
    return new PROCNP(model_tag, config, peller, syst, model, data,
                      strat, shape_only, physics_param_fixed);
}
