/**
 * @file PROmodelSimple.cxx
 * @brief Implementation of the non-oscillation models NullModel and PROtemplate.
 * @author PROfit Collaboration
 */
#include "PROmodels/PROmodelSimple.h"

#include <unordered_map>

namespace PROfit {

// ------------------------------------------------------------------
// NullModel
// ------------------------------------------------------------------

NullModel::NullModel(const PROpeller &prop) {
    nparams = 0;
    // Empty ivars: no placeholder physics-grid variable. Each reco variable is binned
    // independently, so events out-of-range in one variable don't get dropped from another.
    ivars = {};
    is_trivial = true;
    model_functions.push_back([](const Eigen::VectorXf &, float){ return 1.0f; });
    prob_types = {0};

    build_hists_and_combined(prop, /*filter_by_model_rule=*/false);
    is_log10.clear();
}

// ------------------------------------------------------------------
// PROtemplate
// ------------------------------------------------------------------

PROtemplate::PROtemplate(const PROconfig &config, const PROpeller &prop) {
    const size_t K = config.m_model_parameter_names.size();
    if(K == 0) {
        log<LOG_ERROR>(L"%1% || template_fit model needs at least one <parameter> naming a subchannel to float. Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }

    // Map each floated subchannel's global index -> column (1..K). Column 0 is the fixed
    // remainder (every non-floated subchannel), permanently at scale 1.
    std::unordered_map<size_t, int> subchan_to_col;
    for(size_t k = 0; k < K; ++k) {
        // GetSubchannelIndex terminates with a clear error if the name is not a known subchannel.
        size_t gsi = config.GetSubchannelIndex(config.m_model_parameter_names[k]);
        subchan_to_col[gsi] = (int)(k + 1);
    }

    // No truth grid: pure per-subchannel normalization (n_phys_bins == 1). Not trivial:
    // we still need get_probs() + the GEMV to apply the per-column scales.
    ivars = {};
    is_trivial = false;

    // K+1 components: column 0 fixed (=1); column k+1 returns the scale of subchannel k.
    model_functions.push_back([](const Eigen::VectorXf &, float){ return 1.0f; });
    prob_types.push_back(0);
    for(size_t k = 0; k < K; ++k) {
        model_functions.push_back([k](const Eigen::VectorXf &v, float){ return v((Eigen::Index)k); });
        prob_types.push_back(k + 1);
    }

    // Route each event to its column by subchannel membership, derived from the reco global
    // bin (subchannels occupy contiguous reco-bin ranges per variable).
    std::function<int(size_t, size_t, int)> block_fn =
        [&config, subchan_to_col](size_t v, size_t, int rbin) -> int {
            size_t gsi = config.GetSubchannelIndexFromVariableGlobalBin((size_t)rbin, v);
            auto it = subchan_to_col.find(gsi);
            return it == subchan_to_col.end() ? 0 : it->second;
        };
    build_hists_and_combined(prop, /*filter_by_model_rule=*/false, block_fn);

    nparams = K;
    param_names.clear();
    pretty_param_names.clear();
    pretty_param_units.clear();
    is_log10.assign(K, false);
    lb          = Eigen::VectorXf(K);
    ub          = Eigen::VectorXf(K);
    default_val = Eigen::VectorXf(K);
    for(size_t k = 0; k < K; ++k) {
        param_names.push_back(config.m_model_parameter_names[k]);
        pretty_param_names.push_back(config.m_model_parameter_names[k]);
        pretty_param_units.push_back("");
        lb(k)          = config.m_model_parameter_min[k];
        ub(k)          = config.m_model_parameter_max[k];
        default_val(k) = 1.0f; // nominal normalization
    }
    build_param_index();

    log<LOG_INFO>(L"%1% || template_fit model: floating %2% subchannel normalization(s).") % __func__ % K;
    for(size_t k = 0; k < K; ++k)
        log<LOG_INFO>(L"%1% || Param %2% = '%3%' scale in [%4%, %5%], default 1.") % __func__ % k % param_names[k].c_str() % lb(k) % ub(k);
}

Eigen::MatrixXf PROtemplate::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &) const {
    Eigen::MatrixXf probs(1, model_functions.size());
    probs(0, 0) = 1.0f;
    for(size_t k = 0; k < nparams; ++k)
        probs(0, (Eigen::Index)(k + 1)) = phys((Eigen::Index)k);
    return probs;
}

std::vector<Eigen::MatrixXf> PROtemplate::get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &) const {
    // get_probs returns probs(0, k+1) = p_k (the k-th scale parameter itself), so
    // ∂probs(0, k+1)/∂p_k = 1 and every other entry is 0. Times the chain factor
    // (ln10 · p_k) if the parameter were log10; the template scales are linear.
    constexpr float LN10 = 2.302585093f;
    std::vector<Eigen::MatrixXf> grads(nparams, Eigen::MatrixXf::Zero(1, model_functions.size()));
    for(size_t k = 0; k < nparams; ++k)
        grads[k](0, (Eigen::Index)(k + 1)) = is_log10[k] ? LN10 * std::pow(10.0f, phys((Eigen::Index)k)) : 1.0f;
    return grads;
}

}
