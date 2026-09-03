/**
 * @file PROmodel.cxx
 * @brief Implementation of the PROmodel base class and the model factory.
 * @author PROfit Collaboration
 */
#include "PROmodel.h"
#include "PROmodels/PROmodelSimple.h"
#include "PROmodels/PROmodel2flav.h"
#include "PROmodels/PROmodel3p1.h"
#include "PROmodels/PROmodel3p1decayvis.h"
#include "PROmodels/PROmodel3p2.h"
#include "PROmodels/PROmodelLBL.h"
#include "PROmodels/PROmodelSine.h"

namespace PROfit {

void PROmodel::build_hists_and_combined(const PROpeller &prop, bool filter_by_model_rule,
                                        const std::function<int(size_t, size_t, int)> &block_fn) {
    // Compute flat physics grid size and per-ivar bin counts
    std::vector<size_t> ivar_sizes(ivars.size());
    n_phys_bins = 1;
    for(size_t k = 0; k < ivars.size(); ++k) {
        ivar_sizes[k] = prop.variable_midbin[ivars[k]].size();
        n_phys_bins *= (long int)ivar_sizes[k];
    }

    size_t nvar = prop.variable_mc_stat_err.size();
    // Decay models compute counts directly in get_counts and leave model_functions empty;
    // for those, the number of components comes from prob_types instead.
    size_t J    = model_functions.empty() ? prob_types.size() : model_functions.size();
    H_combined.resize(nvar);

    // This pre-bins the WHOLE MC event store into the dense binned-eval
    // response matrices for every variable, and is the silent chunk of model
    // construction — announce the size up front so a long stall here is
    // labeled (it scales with subchannel count via the uncollapsed reco bins).
    {
        size_t total_floats = 0;
        for(size_t v = 0; v < nvar; ++v)
            total_floats += prop.variable_mc_stat_err[v].size() * (size_t)n_phys_bins * J;
        log<LOG_INFO>(L"%1% || Pre-binning %2% MC events into response matrices: %3% variables x %4% components x %5% physics-grid bins (~%6% MB). This runs once per model construction.")
            % __func__ % prop.NEvent() % nvar % J % n_phys_bins
            % (total_floats * sizeof(float) / (1024.0 * 1024.0));
    }

    for(size_t v = 0; v < nvar; ++v) {
        size_t n_reco_v = prop.variable_mc_stat_err[v].size();
        // Fill events directly into component m's column block
        // [m*n_phys_bins, (m+1)*n_phys_bins) of H_combined[v] — no per-component
        // staging copy (nothing read the old `hists` member, and the duplicate
        // doubled peak memory plus a full zero+copy pass).
        H_combined[v] = Eigen::MatrixXf::Zero(n_reco_v, n_phys_bins * J);
        for(size_t i = 0; i < prop.NEvent(); ++i) {
            int rbin = prop.VariableBinIndex(v, i);
            if(rbin < 0) continue;

            // Determine the destination column for this event.
            int col;
            if(block_fn) {
                col = block_fn(v, i, rbin);
            } else {
                col = filter_by_model_rule ? prop.model_rule[i] : 0;
            }
            if(col < 0 || col >= (int)J) continue;

            // Compute row-major flat index over ivars
            long int flat_phys = 0;
            bool valid = true;
            for(size_t k = 0; k < ivars.size(); ++k) {
                int tbin = prop.VariableBinIndex(ivars[k], i);
                if(tbin < 0) { valid = false; break; }
                flat_phys = flat_phys * (long int)ivar_sizes[k] + tbin;
            }
            if(!valid) continue;

            H_combined[v](rbin, (long int)col * n_phys_bins + flat_phys) += prop.added_weights[i];
        }
    }

    // Store per-ivar bin counts for downstream use (e.g. decay redistribution).
    phys_grid_sizes.assign(ivar_sizes.begin(), ivar_sizes.end());
}

Eigen::MatrixXf PROmodel::compute_N_truth(const PROpeller &prop, bool filter_by_model_rule) const {
    size_t J = model_functions.empty() ? prob_types.size() : model_functions.size();
    std::vector<size_t> ivar_sizes(ivars.size());
    for(size_t k = 0; k < ivars.size(); ++k)
        ivar_sizes[k] = prop.variable_midbin[ivars[k]].size();

    Eigen::MatrixXf N = Eigen::MatrixXf::Zero(n_phys_bins, J);
    for(size_t i = 0; i < prop.NEvent(); ++i) {
        int j = filter_by_model_rule ? prop.model_rule[i] : 0;
        if(j < 0 || (size_t)j >= J) continue;
        long int flat_phys = 0;
        bool valid = true;
        for(size_t k = 0; k < ivars.size(); ++k) {
            int tbin = prop.VariableBinIndex(ivars[k], i);
            if(tbin < 0) { valid = false; break; }
            flat_phys = flat_phys * (long int)ivar_sizes[k] + tbin;
        }
        if(!valid) continue;
        N(flat_phys, j) += prop.added_weights[i];
    }
    return N;
}

Eigen::MatrixXf PROmodel::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    const auto &le_arr = var_arrs[0];
    Eigen::MatrixXf probs(le_arr.size(), prob_types.size());
    for(size_t i = 0; i < le_arr.size(); ++i) {
        for(size_t j = 0; j < prob_types.size(); ++j) {
            probs(i, j) = model_functions[j](phys, le_arr[i]);
        }
    }
    return probs;
}

std::vector<Eigen::MatrixXf> PROmodel::get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    // Central FD on the probabilities only. Probabilities are O(1), so this is
    // far better conditioned than FD on the full chi²; models with cheap
    // closed-form derivatives should still override for exactness and speed.
    std::vector<Eigen::MatrixXf> grads(nparams);
    const float h = 1e-3f;
    Eigen::VectorXf p = phys;
    for(size_t i = 0; i < nparams; ++i) {
        p(i) = phys(i) + h;
        Eigen::MatrixXf plus = get_probs(p, var_arrs);
        p(i) = phys(i) - h;
        Eigen::MatrixXf minus = get_probs(p, var_arrs);
        p(i) = phys(i);
        grads[i] = (plus - minus) / (2.0f * h);
    }
    return grads;
}

void PROmodel::build_param_index() {
    param_name_to_index.clear();
    for(size_t i = 0; i < param_names.size(); ++i){
        param_name_to_index[param_names[i]] = i;
    }
}

// Canonical model tags follow regime_model_parameterization(_NC); legacy tags
// remain accepted as deprecated aliases and are mapped here (with a one-time
// warning per tag) before any dispatch.
static std::string canonicalize_model_tag(const std::string &name) {
    static const std::map<std::string, std::string> legacy_aliases = {
        {"numudis",              "SBL_2flav_numudis"},
        {"nueapp",               "SBL_2flav_nueapp"},
        {"nuedis",               "SBL_2flav_nuedis"},
        {"NCnumudisapp",         "SBL_2flav_numudis_NC"},
        {"NCdisapp",             "SBL_2flav_nudis_NC"},
        {"3+1",                  "SBL_3+1_Usq"},
        {"3+1_NC",               "SBL_3+1_Usq_NC"},
        {"3+1_angles",           "SBL_3+1_angles"},
        {"3+1_angles_NC",        "SBL_3+1_angles_NC"},
        {"3+1_3A",               "SBL_3+1_sinsq2thee"},
        {"3+1_3A_NC",            "SBL_3+1_sinsq2thee_NC"},
        {"3+1_3B",               "SBL_3+1_sinsq2thmumu"},
        {"3+1_3B_NC",            "SBL_3+1_sinsq2thmumu_NC"},
        {"3+1_3C",               "SBL_3+1_sinsq2thmue"},
        {"3+1_3C_NC",            "SBL_3+1_sinsq2thmue_NC"},
        {"3+1_decay_invis",      "SBL_3+1+decay_invis"},
        {"3+1_decay_vis_model1", "SBL_3+1+decay_vis1"},
        {"3+1_decay_vis_model2", "SBL_3+1+decay_vis2"},
        {"3+2",                  "SBL_3+2_Usq"},
        {"LBL",                  "LBL_3nu-matter_angles"},
        {"nullmodel",            "null"},
        {"template_fit",         "template"},
    };
    auto it = legacy_aliases.find(name);
    if(it == legacy_aliases.end()) return name;
    static std::set<std::string> warned;
    if(warned.insert(name).second) {
        log<LOG_WARNING>(L"%1% || Model tag '%2%' is deprecated; use '%3%' in the XML <model tag=...>. The legacy tag keeps working as an alias.")
            % __func__ % name.c_str() % it->second.c_str();
    }
    return it->second;
}

std::unique_ptr<PROmodel> get_model_from_string(const PROconfig& config, const PROpeller &prop) {
    std::string name = canonicalize_model_tag(config.m_model_tag);

    if(name == "null") {
        return std::unique_ptr<PROmodel>(new NullModel(prop));
    } else if(name == "numudisTEST") {
        return std::unique_ptr<PROmodel>(new PROnumudisTEST(prop,config.m_model_parameter_map));
    } else if(name == "SBL_3+1+decay_invis") {
        return std::unique_ptr<PROmodel>(new PRO3p1_decay_invis(prop,config.m_model_parameter_map));
    } else if(name == "SBL_3+1+decay_vis1") {
        return std::unique_ptr<PROmodel>(new PRO3p1_decay_vis_model1(prop,config.m_model_parameter_map));
    } else if(name == "SBL_3+1+decay_vis2") {
        return std::unique_ptr<PROmodel>(new PRO3p1_decay_vis_model2(prop,config.m_model_parameter_map));
    } else if(name == "SBL_3+2_Usq") {
        return std::unique_ptr<PROmodel>(new PRO3p2(prop, config.m_model_parameter_map));
    } else if(name == "LBL_3nu-matter_angles") {
        return std::unique_ptr<PROmodel>(new PROLBL(prop, config.m_model_parameter_map));
    } else if(name == "template") {
        return std::unique_ptr<PROmodel>(new PROtemplate(config, prop));
    }
    // Sine-kernel family (SBL_2flav_*, SBL_3+1_* and their _NC variants):
    // recipe-driven, all evaluated by PROsineModel. See PROmodelSine.h.
    if(const SineModelRecipe *recipe = find_sine_recipe(name)) {
        return std::unique_ptr<PROmodel>(new PROsineModel(prop, config.m_model_parameter_map, *recipe));
    }
    log<LOG_ERROR>(L"%1% || Unrecognized model name %2%. Valid tags (regime_model_parameterization(_NC); legacy pre-v3.1 names are auto-mapped): null, template, SBL_2flav_(numudis,nueapp,nuedis), SBL_2flav_(numudis,nudis)_NC, SBL_3+1_(Usq,angles,sinsq2thee,sinsq2thmumu,sinsq2thmue) and their _NC versions, SBL_3+1+decay_(invis,vis1,vis2), SBL_3+2_Usq, LBL_3nu-matter_angles. Terminating.") % __func__ % name.c_str();
    exit(EXIT_FAILURE);
}

}
