/**
 * @file PROmodel.cxx
 * @brief Implementation of the PROmodel base class and the model factory.
 * @author PROfit Collaboration
 */
#include "PROmodel.h"
#include "PROmodels/PROmodelSimple.h"
#include "PROmodels/PROmodel2flav.h"
#include "PROmodels/PROmodel3p1.h"
#include "PROmodels/PROmodel3p2.h"
#include "PROmodels/PROmodelLBL.h"

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
    size_t J    = model_functions.size();
    hists.resize(nvar);
    H_combined.resize(nvar);

    for(size_t v = 0; v < nvar; ++v) {
        size_t n_reco_v = prop.variable_mc_stat_err[v].size();
        hists[v].assign(J, Eigen::MatrixXf::Zero(n_reco_v, n_phys_bins));
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

            hists[v][col](rbin, flat_phys) += prop.added_weights[i];
        }
        // Build H_combined[v] = [hists[v][0] | hists[v][1] | ... | hists[v][J-1]]
        // shape: (n_reco_v, n_phys_bins * J)
        H_combined[v].resize(n_reco_v, n_phys_bins * J);
        for(size_t m = 0; m < J; ++m)
            H_combined[v].block(0, m * n_phys_bins, n_reco_v, n_phys_bins) = hists[v][m];
    }
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

std::unique_ptr<PROmodel> get_model_from_string(const PROconfig& config, const PROpeller &prop) {
    std::string name = config.m_model_tag;

    if(name == "nullmodel") {
        return std::unique_ptr<PROmodel>(new NullModel(prop));
    } else if(name == "numudis") {
        return std::unique_ptr<PROmodel>(new PROnumudis(prop,config.m_model_parameter_map));
    } else if(name == "numudisTEST") {
        return std::unique_ptr<PROmodel>(new PROnumudisTEST(prop,config.m_model_parameter_map));
    } else if(name == "nueapp") {
        return std::unique_ptr<PROmodel>(new PROnueapp(prop,config.m_model_parameter_map));
    } else if(name == "nuedis") {
        return std::unique_ptr<PROmodel>(new PROnuedis(prop,config.m_model_parameter_map));
    } else if(name == "3+1") {
        return std::unique_ptr<PROmodel>(new PRO3p1(prop,config.m_model_parameter_map));
    } else if(name == "3+1_angles") {
        return std::unique_ptr<PROmodel>(new PRO3p1_angles(prop,config.m_model_parameter_map));
    } else if(name == "3+1_3A") {
        return std::unique_ptr<PROmodel>(new PRO3p1_3A(prop,config.m_model_parameter_map));
    } else if(name == "3+1_3B") {
        return std::unique_ptr<PROmodel>(new PRO3p1_3B(prop,config.m_model_parameter_map));
    } else if(name == "3+1_3C") {
        return std::unique_ptr<PROmodel>(new PRO3p1_3C(prop,config.m_model_parameter_map));
    } else if(name == "3+1_decay_invis") {
        return std::unique_ptr<PROmodel>(new PRO3p1_decay_invis(prop,config.m_model_parameter_map));
    } else if(name == "3+2") {
        return std::unique_ptr<PROmodel>(new PRO3p2(prop, config.m_model_parameter_map));
    } else if(name == "LBL") {
        return std::unique_ptr<PROmodel>(new PROLBL(prop, config.m_model_parameter_map));
    } else if(name == "template" || name == "template_fit") {
        return std::unique_ptr<PROmodel>(new PROtemplate(config, prop));
    }
    log<LOG_ERROR>(L"%1% || Unrecognized model name %2%. Try numudis, nueapp, nuedis, 3+1, 3+1_angles, 3+1_3(A,B,C), 3+1_decay_invis, 3+2, LBL, template_fit. for now. Terminating.") % __func__ % name.c_str();
    exit(EXIT_FAILURE);
}

}
