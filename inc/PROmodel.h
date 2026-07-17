#ifndef PROMODEL_H
#define PROMODEL_H

#include "PROconfig.h"
#include "PROpeller.h"

#include <Eigen/Eigen>
#include <Eigen/src/Core/Matrix.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace PROfit {

class PROmodel {
public:
    size_t nparams;
    int ivar;
    std::vector<std::string> param_names;
    std::vector<std::string> pretty_param_names;
    std::vector<std::string> pretty_param_units;
    Eigen::VectorXf lb, ub, default_val;
    std::vector<std::function<float(const Eigen::VectorXf&, float)>> model_functions;
    std::function<int(const Eigen::VectorXf&)> model_constraint;
    std::vector<std::vector<Eigen::MatrixXf>> hists;

    std::vector<size_t> prob_types;
    std::vector<bool> is_log10;

    virtual Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys,
                                      const std::vector<float> &le_arr) const {
        Eigen::MatrixXf probs(le_arr.size(), prob_types.size());
        for(size_t i = 0; i < le_arr.size(); ++i) {
            for(size_t j = 0; j < prob_types.size(); ++j) {
                probs(i, j) = model_functions[j](phys, le_arr[i]);
            }
        }
        return probs;
    }

    std::unordered_map<std::string, size_t> param_name_to_index;

    inline void build_param_index() {
        param_name_to_index.clear();
        for(size_t i = 0; i < param_names.size(); ++i) {
            param_name_to_index[param_names[i]] = i;
        }
    }

    inline float maybe_convert_log(const std::string &param_name, float value) const {
        auto it = param_name_to_index.find(param_name);
        if(it == param_name_to_index.end()) {
            log<LOG_ERROR>(L"%1% || Parameter name '%2%' not found in this model. Terminating.")
                % __func__ % param_name.c_str();
            exit(EXIT_FAILURE);
        }
        size_t idx = it->second;
        return is_log10[idx] ? std::pow(10.0f, value) : value;
    }

    virtual ~PROmodel() {}
};

class NullModel : public PROmodel {
public:
    NullModel(const PROpeller &prop) {
        nparams = 0;
        ivar = 1;
        model_functions.push_back([](const Eigen::VectorXf &, float){ return 1.0f; });
        prob_types = {0};

        size_t nvar = prop.variable_mc_stat_err.size();
        hists.resize(nvar);
        for(size_t v = 0; v < nvar; v++) {
            for(size_t m = 0; m < model_functions.size(); ++m) {
                hists.at(v).emplace_back(Eigen::MatrixXf::Constant(
                    prop.variable_hist_storage(ivar, v).rows(),
                    prop.variable_hist_storage(ivar, v).cols(),
                    0.0f));
                Eigen::MatrixXf &h = hists.at(v).back();
                for(size_t i = 0; i < prop.NEvent(); ++i) {
                    int tbin = prop.VariableBinIndex(ivar, i);
                    int rbin = prop.VariableBinIndex(v, i);
                    if(tbin < 0 || rbin < 0) continue;
                    h(tbin, rbin) += prop.added_weights[i];
                }
            }
        }
        is_log10.clear();
    }
};

// ===========================================================================
//  Recipe-driven dynamic oscillation models
//
//  Kept original model tags plus NC-disappearance variants:
//    numudis, nueapp, nuedis, NCnumudisapp, NCdisapp,
//    3+1, 3+1_angles, 3+1_3A, 3+1_3B, 3+1_3C,
//    3+1_3A_NC, 3+1_3B_NC, 3+1_3C_NC,
//    3+1_decay_invis, 3+2.
//
//  The NC variants have all 8 channels. They also add a tau/sterile
//  split fitting parameter:
//    3A_NC and 3B_NC: cosq34 = cos^2(theta_34)
//    3C_NC:           Uta4sq = |U_tau4|^2
// ===========================================================================

enum class ProbForm {
    Null,
    Disappearance,   // P = 1 - A sin^2(Delta)
    Appearance,      // P = A sin^2(Delta)
    Custom           // Use custom_get_probs_function or custom_channel_function
};

struct ParameterSpec {
    std::string name;
    std::string pretty_name;
    std::string unit;
    bool log10;
    float lower;
    float upper;
    float def;
};

struct ChannelSpec {
    size_t index;
    std::string name;
    ProbForm form;
};

class PRODynamicOscModel;

using AmplitudeFunction = std::function<std::vector<float>(
    const Eigen::VectorXf&, const PRODynamicOscModel&)>;

using CustomChannelFunction = std::function<float(
    size_t, const Eigen::VectorXf&, float, const PRODynamicOscModel&)>;

using CustomGetProbsFunction = std::function<Eigen::MatrixXf(
    const Eigen::VectorXf&, const std::vector<float>&, const PRODynamicOscModel&)>;

using ConstraintFunction = std::function<int(
    const Eigen::VectorXf&, const PRODynamicOscModel&)>;

struct ModelRecipe {
    std::string name;
    std::vector<ParameterSpec> params;
    std::vector<ChannelSpec> channels;

    bool uses_simple_sine_kernel = true;
    AmplitudeFunction amplitude_function;
    CustomChannelFunction custom_channel_function;
    CustomGetProbsFunction custom_get_probs_function;
    ConstraintFunction constraint_function;
};

class PROBinnedModelBase : public PROmodel {
protected:
    static float neg_inf() {
        return -std::numeric_limits<float>::infinity();
    }

    static float clamp01(float x) {
        if(x < 0.0f) return 0.0f;
        if(x > 1.0f) return 1.0f;
        return x;
    }

    static float safe_sqrt(float x) {
        return std::sqrt(std::max(0.0f, x));
    }

    static float apply_prob_form(ProbForm form, float amplitude, float sin2) {
        amplitude = clamp01(amplitude);
        switch(form) {
            case ProbForm::Null:
                return 1.0f;
            case ProbForm::Appearance:
                return amplitude * sin2;
            case ProbForm::Disappearance:
                return 1.0f - amplitude * sin2;
            case ProbForm::Custom:
                log<LOG_ERROR>(L"%1% || Custom probability requested in simple kernel. Terminating.")
                    % __func__;
                exit(EXIT_FAILURE);
        }
        return 0.0f;
    }

    void set_parameters_from_specs(const std::vector<ParameterSpec> &params) {
        nparams = params.size();
        param_names.clear();
        pretty_param_names.clear();
        pretty_param_units.clear();
        is_log10.clear();

        lb = Eigen::VectorXf(nparams);
        ub = Eigen::VectorXf(nparams);
        default_val = Eigen::VectorXf(nparams);

        for(size_t i = 0; i < params.size(); ++i) {
            param_names.push_back(params[i].name);
            pretty_param_names.push_back(params[i].pretty_name);
            pretty_param_units.push_back(params[i].unit);
            is_log10.push_back(params[i].log10);
            lb((Eigen::Index)i) = params[i].lower;
            ub((Eigen::Index)i) = params[i].upper;
            default_val((Eigen::Index)i) = params[i].def;
        }
        build_param_index();
    }

    void require_le_variable(const std::map<std::string,int> &parameter_map) {
        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.")
                % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivar = parameter_map.at("L/E");
    }

    void build_model_rule_hists(const PROpeller &prop, size_t nchannels) {
        size_t nvar = prop.variable_mc_stat_err.size();
        hists.resize(nvar);
        for(size_t v = 0; v < nvar; ++v) {
            for(size_t m = 0; m < nchannels; ++m) {
                hists.at(v).emplace_back(Eigen::MatrixXf::Constant(
                    prop.variable_hist_storage(ivar, v).rows(),
                    prop.variable_hist_storage(ivar, v).cols(),
                    0.0f));
                Eigen::MatrixXf &h = hists.at(v).back();
                for(size_t i = 0; i < prop.NEvent(); ++i) {
                    if(prop.model_rule[i] != (int)m) continue;
                    int tbin = prop.VariableBinIndex(ivar, i);
                    int rbin = prop.VariableBinIndex(v, i);
                    if(tbin < 0 || rbin < 0) continue;
                    h(tbin, rbin) += prop.added_weights[i];
                }
            }
        }
    }
};

class PRODynamicOscModel : public PROBinnedModelBase {
public:
    PRODynamicOscModel(const PROpeller &prop,
                       const PROconfig &config,
                       const ModelRecipe &recipe)
        : recipe_(recipe)
    {
        set_parameters_from_specs(recipe_.params);
        setup_channels();
        require_le_variable(config.m_model_parameter_map);
        setup_model_functions();

        if(recipe_.constraint_function) {
            model_constraint = [this](const Eigen::VectorXf &v) {
                return recipe_.constraint_function(v, *this);
            };
        }

        build_model_rule_hists(prop, nchannels_);
    }

    float par(const Eigen::VectorXf &v, const std::string &name) const {
        auto it = param_name_to_index.find(name);
        if(it == param_name_to_index.end()) {
            log<LOG_ERROR>(L"%1% || Parameter name '%2%' not found in this model. Terminating.")
                % __func__ % name.c_str();
            exit(EXIT_FAILURE);
        }
        return maybe_convert_log(name, v((Eigen::Index)it->second));
    }

    bool has_param(const std::string &name) const {
        return param_name_to_index.find(name) != param_name_to_index.end();
    }

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys,
                              const std::vector<float> &le_arr) const override {
        if(!recipe_.uses_simple_sine_kernel) {
            if(!recipe_.custom_get_probs_function) {
                log<LOG_ERROR>(L"%1% || Model '%2%' needs custom_get_probs_function. Terminating.")
                    % __func__ % recipe_.name.c_str();
                exit(EXIT_FAILURE);
            }
            return recipe_.custom_get_probs_function(phys, le_arr, *this);
        }

        const float dmsq = par(phys, "dmsq");
        const float freq = 1.266932679f * dmsq;
        const std::vector<float> A = amplitudes_or_zero(phys);

        Eigen::MatrixXf probs(le_arr.size(), nchannels_);
        for(size_t i = 0; i < le_arr.size(); ++i) {
            float s = std::sin(freq * le_arr[i]);
            float sin2 = s * s;
            for(size_t c = 0; c < nchannels_; ++c) {
                probs(i, c) = apply_prob_form(channel_forms_[c], A[c], sin2);
            }
        }
        return probs;
    }

private:
    ModelRecipe recipe_;
    size_t nchannels_ = 0;
    std::vector<ProbForm> channel_forms_;
    std::vector<std::string> channel_names_;

    void setup_channels() {
        nchannels_ = 0;
        for(const auto &c : recipe_.channels) {
            nchannels_ = std::max(nchannels_, c.index + 1);
        }
        channel_forms_.assign(nchannels_, ProbForm::Custom);
        channel_names_.assign(nchannels_, "");
        prob_types.clear();
        for(const auto &c : recipe_.channels) {
            channel_forms_[c.index] = c.form;
            channel_names_[c.index] = c.name;
        }
        for(size_t c = 0; c < nchannels_; ++c) {
            prob_types.push_back(c);
        }
    }

    void setup_model_functions() {
        model_functions.clear();
        for(size_t c = 0; c < nchannels_; ++c) {
            model_functions.push_back([this, c](const Eigen::VectorXf &v, float le) {
                return this->single_channel_prob(c, v, le);
            });
        }
    }

    std::vector<float> amplitudes_or_zero(const Eigen::VectorXf &phys) const {
        if(recipe_.amplitude_function) {
            std::vector<float> A = recipe_.amplitude_function(phys, *this);
            if(A.size() < nchannels_) A.resize(nchannels_, 0.0f);
            return A;
        }
        return std::vector<float>(nchannels_, 0.0f);
    }

    float single_channel_prob(size_t c, const Eigen::VectorXf &phys, float le) const {
        if(c >= nchannels_) return 0.0f;
        if(channel_forms_[c] == ProbForm::Null) return 1.0f;

        if(!recipe_.uses_simple_sine_kernel) {
            if(recipe_.custom_channel_function) {
                return recipe_.custom_channel_function(c, phys, le, *this);
            }
            if(recipe_.custom_get_probs_function) {
                std::vector<float> one_le = {le};
                Eigen::MatrixXf p = recipe_.custom_get_probs_function(phys, one_le, *this);
                return p(0, (Eigen::Index)c);
            }
        }

        const float dmsq = par(phys, "dmsq");
        float s = std::sin(1.266932679f * dmsq * le);
        float sin2 = s * s;
        std::vector<float> A = amplitudes_or_zero(phys);
        return apply_prob_form(channel_forms_[c], A[c], sin2);
    }
};

static inline float neg_inf() {
    return -std::numeric_limits<float>::infinity();
}

static inline std::vector<ChannelSpec> channels_null_plus_1(const std::string &channel,
                                                             ProbForm form) {
    return {
        {0, "null", ProbForm::Null},
        {1, channel, form}
    };
}

static inline std::vector<ChannelSpec> channels_3p1_standard() {
    return {
        {0, "null", ProbForm::Null},
        {1, "mumu", ProbForm::Disappearance},
        {2, "mue",  ProbForm::Appearance},
        {3, "ee",   ProbForm::Disappearance}
    };
}

static inline std::vector<ChannelSpec> channels_ncdisapp() {
    return {
        {0, "null", ProbForm::Null},
        {1, "mus",  ProbForm::Disappearance},
        {2, "es",   ProbForm::Disappearance}
    };
}

static inline std::vector<ChannelSpec> channels_3p1_nc() {
    return {
        {0, "null",  ProbForm::Null},
        {1, "mumu",  ProbForm::Disappearance},
        {2, "mue",   ProbForm::Appearance},
        {3, "ee",    ProbForm::Disappearance},
        {4, "mus",   ProbForm::Disappearance}, // visible NC deficit: 1 - A_mus sin^2(Delta)
        {5, "es",    ProbForm::Disappearance}, // visible NC deficit: 1 - A_es  sin^2(Delta)
        {6, "mutau", ProbForm::Appearance},
        {7, "etau",  ProbForm::Appearance}
    };
}

static inline ModelRecipe make_sbl2_recipe(const std::string &name,
                                           const std::string &mixing_name,
                                           const std::string &mixing_pretty,
                                           const std::string &channel_name,
                                           ProbForm form,
                                           float mixing_lower,
                                           float dmsq_default = -10.0f,
                                           float mixing_default = -10.0f) {
    ModelRecipe r;
    r.name = name;
    r.params = {
        {"dmsq", "#Deltam^{2}", "eV^{2}", true, -2.0f, 2.0f, dmsq_default},
        {mixing_name, mixing_pretty, "", true, mixing_lower, 0.0f, mixing_default}
    };
    r.channels = channels_null_plus_1(channel_name, form);
    r.amplitude_function = [mixing_name](const Eigen::VectorXf &v, const PRODynamicOscModel &m) {
        std::vector<float> A(2, 0.0f);
        A[1] = m.par(v, mixing_name);
        return A;
    };
    return r;
}

static inline ModelRecipe make_ncdisapp_recipe() {
    ModelRecipe r;
    r.name = "NCdisapp";
    r.params = {
        {"dmsq",       "#Deltam^{2}", "eV^{2}", true, -2.0f, 2.0f, -2.0f},
        {"sinsq2thms", "sin^{2}2#theta_{#mus}", "", true, neg_inf(), 0.0f, -10.0f},
        {"sinsq2thes", "sin^{2}2#theta_{es}",   "", true, neg_inf(), 0.0f, -10.0f}
    };
    r.channels = channels_ncdisapp();
    r.amplitude_function = [](const Eigen::VectorXf &v, const PRODynamicOscModel &m) {
        std::vector<float> A(3, 0.0f);
        A[1] = m.par(v, "sinsq2thms");
        A[2] = m.par(v, "sinsq2thes");
        return A;
    };
    return r;
}

static inline ModelRecipe make_3p1_recipe() {
    ModelRecipe r;
    r.name = "3+1";
    r.params = {
        {"dmsq",  "#Deltam^{2}", "eV^{2}", true, -2.0f, 2.0f, -2.0f},
        {"Ue4^2", "|U_{e4}|^{2}", "", true, neg_inf(), -1e-4f, -8.0f},
        {"Um4^2", "|U_{#mu4}|^{2}", "", true, neg_inf(), -1e-4f, -8.0f}
    };
    r.channels = channels_3p1_standard();
    r.amplitude_function = [](const Eigen::VectorXf &v, const PRODynamicOscModel &m) {
        std::vector<float> A(4, 0.0f);
        const float Ue4sq = m.par(v, "Ue4^2");
        const float Um4sq = m.par(v, "Um4^2");
        A[1] = 4.0f * Um4sq * (1.0f - Um4sq);
        A[2] = 4.0f * Ue4sq * Um4sq;
        A[3] = 4.0f * Ue4sq * (1.0f - Ue4sq);
        return A;
    };
    r.constraint_function = [](const Eigen::VectorXf &v, const PRODynamicOscModel &m) {
        return (m.par(v, "Ue4^2") + m.par(v, "Um4^2") < 1.0f) ? 1 : 0;
    };
    return r;
}

static inline ModelRecipe make_3p1_angles_recipe() {
    ModelRecipe r;
    r.name = "3+1_angles";
    r.params = {
        {"dmsq", "#Deltam^{2}", "eV^{2}", true, -2.0f, 2.0f, -2.0f},
        {"sinsq2th14", "sin^{2}2#theta_{14}", "", true, neg_inf(), -1e-4f, -8.0f},
        {"sinsqth24",  "sin^{2}#theta_{24}",  "", true, neg_inf(), -1e-4f, -8.0f}
    };
    r.channels = channels_3p1_standard();
    r.amplitude_function = [](const Eigen::VectorXf &v, const PRODynamicOscModel &m) {
        std::vector<float> A(4, 0.0f);
        const float s214 = m.par(v, "sinsq2th14");
        const float s24  = m.par(v, "sinsqth24");
        const float c14  = (1.0f + std::sqrt(std::max(0.0f, 1.0f - s214))) / 2.0f;
        A[1] = c14 * s24 * (1.0f - c14 * s24);
        A[2] = s214 * s24;
        A[3] = s214;
        return A;
    };
    r.constraint_function = [](const Eigen::VectorXf &, const PRODynamicOscModel &) { return 1; };
    return r;
}

static inline ModelRecipe make_3p1_nue_disappearance_recipe(bool include_nc = false) {
    ModelRecipe r;
    r.name = include_nc ? "3+1_nue_disapp_NC" : "3+1_nue_disapp";
    r.params = {
        {"dmsq", "#Deltam^{2}", "eV^{2}", true, -2.0f, 2.0f, -2.0f},
        {"sinsq2thee", "sin^{2}2#theta_{ee}", "", true, neg_inf(), -1e-3f, -8.0f},
        {"sinsqth24",  "sin^{2}#theta_{24}",  "", true, neg_inf(), -1e-3f, -8.0f}
    };
    if(include_nc) {
        // NC version fits the sterile/tau split.
        // cosq34 = 1 sends the off-diagonal NC strength to sterile.
        // cosq34 = 0 sends it to tau.
        r.params.push_back({"cosq34", "cos^{2}#theta_{34}", "", false, 0.0f, 1.0f, 1.0f});
    }
    r.channels = include_nc ? channels_3p1_nc() : channels_3p1_standard();
    r.amplitude_function = [include_nc](const Eigen::VectorXf &v, const PRODynamicOscModel &m) {
        std::vector<float> A(include_nc ? 8 : 4, 0.0f);
        const float s2ee = m.par(v, "sinsq2thee");
        const float rA   = m.par(v, "sinsqth24");
        const float q    = std::sqrt(std::max(0.0f, 1.0f - s2ee));
        const float Um4sq = 0.5f * rA * (1.0f + q);

        A[1] = 4.0f * Um4sq * (1.0f - Um4sq);  // numu disappearance amplitude
        A[2] = rA * s2ee;                       // numu -> nue appearance amplitude
        A[3] = s2ee;                            // nue disappearance amplitude

        if(include_nc) {
            const float c34sq  = m.par(v, "cosq34");
            const float onepq2 = (1.0f + q) * (1.0f + q);
            const float mu_off = rA * (1.0f - rA) * onepq2;
            const float e_off  = (1.0f - rA) * s2ee;

            A[4] = mu_off * c34sq;              // numu -> sterile, visible NC deficit
            A[5] = e_off  * c34sq;              // nue  -> sterile, visible NC deficit
            A[6] = mu_off * (1.0f - c34sq);     // numu -> nutau appearance
            A[7] = e_off  * (1.0f - c34sq);     // nue  -> nutau appearance
        }
        return A;
    };
    r.constraint_function = [](const Eigen::VectorXf &, const PRODynamicOscModel &) { return 1; };
    return r;
}

static inline ModelRecipe make_3p1_numu_disappearance_recipe(bool include_nc = false) {
    ModelRecipe r;
    r.name = include_nc ? "3+1_numu_disapp_NC" : "3+1_numu_disapp";
    r.params = {
        {"dmsq", "#Deltam^{2}", "eV^{2}", true, -2.0f, 2.0f, -2.0f},
        {"sinsq2thmumu", "sin^{2}2#theta_{#mu#mu}", "", true, neg_inf(), -1e-3f, -8.0f},
        {"sB", "sB", "", true, neg_inf(), -1e-3f, -8.0f}
    };
    if(include_nc) {
        // NC version fits the sterile/tau split.
        // cosq34 = 1 sends the off-diagonal NC strength to sterile.
        // cosq34 = 0 sends it to tau.
        r.params.push_back({"cosq34", "cos^{2}#theta_{34}", "", false, 0.0f, 1.0f, 1.0f});
    }
    r.channels = include_nc ? channels_3p1_nc() : channels_3p1_standard();
    r.amplitude_function = [include_nc](const Eigen::VectorXf &v, const PRODynamicOscModel &m) {
        std::vector<float> A(include_nc ? 8 : 4, 0.0f);
        const float s2mumu = m.par(v, "sinsq2thmumu");
        const float sB     = m.par(v, "sB");
        const float q      = std::sqrt(std::max(0.0f, 1.0f - s2mumu));
        const float Ue4sq  = 0.5f * sB * (1.0f + q);
        const float s2ee   = 4.0f * (1.0f - Ue4sq) * Ue4sq;

        A[1] = s2mumu;       // numu disappearance amplitude
        A[2] = sB * s2mumu;  // numu -> nue appearance amplitude
        A[3] = s2ee;         // nue disappearance amplitude

        if(include_nc) {
            const float c34sq   = m.par(v, "cosq34");
            const float c14sq   = 1.0f - Ue4sq;
            const float bracket = 2.0f * c14sq - 1.0f + q;
            const float mu_off  = (1.0f - q) * bracket;
            const float e_off   = 2.0f * (1.0f - c14sq) * bracket;

            A[4] = mu_off * c34sq;              // numu -> sterile, visible NC deficit
            A[5] = e_off  * c34sq;              // nue  -> sterile, visible NC deficit
            A[6] = mu_off * (1.0f - c34sq);     // numu -> nutau appearance
            A[7] = e_off  * (1.0f - c34sq);     // nue  -> nutau appearance
        }
        return A;
    };
    r.constraint_function = [](const Eigen::VectorXf &v, const PRODynamicOscModel &m) {
        const float s2mumu = m.par(v, "sinsq2thmumu");
        const float sB     = m.par(v, "sB");
        const float q      = std::sqrt(std::max(0.0f, 1.0f - s2mumu));
        const float Um4sq  = (1.0f - q) / 2.0f;
        const float Ue4sq  = sB * (1.0f - Um4sq);
        return (Um4sq + Ue4sq < 0.999f) ? 1 : 0;
    };
    return r;
}

static inline ModelRecipe make_3p1_numu_to_nue_appearance_recipe(bool include_nc = false) {
    ModelRecipe r;
    r.name = include_nc ? "3+1_numu_to_nue_app_NC" : "3+1_numu_to_nue_app";
    r.params = {
        {"dmsq", "#Deltam^{2}", "eV^{2}", true, -2.0f, 2.0f, -2.0f},
        {"sinsq2thmue", "sin^{2}2#theta_{#mue}", "", true, neg_inf(), -1e-3f, -8.0f},
        {"xi", "#xi", "", false, -10.0f, 10.0f, 0.0f}
    };
    if(include_nc) {
        // NC version fits the tau component directly.
        r.params.push_back({"Uta4sq", "|U_{#tau4}|^{2}", "", true, neg_inf(), -1e-3f, -8.0f});
    }
    r.channels = include_nc ? channels_3p1_nc() : channels_3p1_standard();
    r.amplitude_function = [include_nc](const Eigen::VectorXf &v, const PRODynamicOscModel &m) {
        std::vector<float> A(include_nc ? 8 : 4, 0.0f);
        const float s2mue = m.par(v, "sinsq2thmue");
        const float xi    = m.par(v, "xi");
        const float sq    = std::sqrt(std::max(0.0f, s2mue));
        const float Um4sq = (std::exp(-xi) * sq) / 2.0f;
        const float Ue4sq = (std::exp( xi) * sq) / 2.0f;

        A[1] = 4.0f * Um4sq * (1.0f - Um4sq);  // numu disappearance amplitude
        A[2] = s2mue;                           // numu -> nue appearance amplitude
        A[3] = 4.0f * Ue4sq * (1.0f - Ue4sq);   // nue disappearance amplitude

        if(include_nc) {
            const float Uta4sq = m.par(v, "Uta4sq");
            const float Us4sq  = std::max(0.0f, 1.0f - Ue4sq - Um4sq - Uta4sq);
            A[4] = 4.0f * Um4sq * Us4sq;        // numu -> sterile, visible NC deficit
            A[5] = 4.0f * Ue4sq * Us4sq;        // nue  -> sterile, visible NC deficit
            A[6] = 4.0f * Um4sq * Uta4sq;       // numu -> nutau appearance
            A[7] = 4.0f * Ue4sq * Uta4sq;       // nue  -> nutau appearance
        }
        return A;
    };
    r.constraint_function = [include_nc](const Eigen::VectorXf &v, const PRODynamicOscModel &m) {
        const float s2mue  = m.par(v, "sinsq2thmue");
        const float xi     = m.par(v, "xi");
        const float Uta4sq = include_nc ? m.par(v, "Uta4sq") : 0.0f;
        return (std::sqrt(std::max(0.0f, s2mue)) * std::cosh(xi) + Uta4sq < 0.999f) ? 1 : 0;
    };
    return r;
}

static inline ModelRecipe make_decay_invis_recipe() {
    ModelRecipe r;
    r.name = "3+1_decay_invis";
    r.uses_simple_sine_kernel = false;
    r.params = {
        {"dmsq", "#Deltam^{2}", "eV^{2}", true, -2.0f, 2.0f, -2.0f},
        {"Ue4^2", "|U_{e4}|^{2}", "", true, neg_inf(), -1e-4f, -8.0f},
        {"Um4^2", "|U_{#mu4}|^{2}", "", true, neg_inf(), -1e-4f, -8.0f},
        {"g2", "g^{2}", "", false, 0.0f, 10.0f, 0.0f}
    };
    r.channels = {
        {0, "null", ProbForm::Null},
        {1, "mumu", ProbForm::Custom},
        {2, "mue",  ProbForm::Custom},
        {3, "ee",   ProbForm::Custom}
    };
    r.constraint_function = [](const Eigen::VectorXf &v, const PRODynamicOscModel &m) {
        const float Ue4sq = m.par(v, "Ue4^2");
        const float Um4sq = m.par(v, "Um4^2");
        const float g2    = m.par(v, "g2");
        return ((Ue4sq + Um4sq) < 1.0f && g2 > 0.0f) ? 1 : 0;
    };
    r.custom_get_probs_function = [](const Eigen::VectorXf &phys,
                                     const std::vector<float> &le_arr,
                                     const PRODynamicOscModel &m) {
        const float dmsq   = m.par(phys, "dmsq");
        const float Ue4sq  = m.par(phys, "Ue4^2");
        const float Um4sq  = m.par(phys, "Um4^2");
        const float g2     = m.par(phys, "g2");
        const float freq   = 1.266932679f * dmsq;
        const float pi     = 3.14159f;

        Eigen::MatrixXf probs(le_arr.size(), 4);
        for(size_t i = 0; i < le_arr.size(); ++i) {
            const float delta = freq * le_arr[i];
            const float costerm = std::cos(2.0f * delta);
            const float expterm = std::exp(-g2 * delta / (8.0f * pi));
            const float cos_mult_exp_term = costerm * expterm;
            const float osc_term = 1.0f - 2.0f * cos_mult_exp_term + expterm * expterm;

            probs(i, 0) = 1.0f;
            probs(i, 1) = 1.0f - 2.0f * Um4sq * (1.0f - cos_mult_exp_term) + Um4sq * Um4sq * osc_term;
            probs(i, 2) = Ue4sq * Um4sq * osc_term;
            probs(i, 3) = 1.0f - 2.0f * Ue4sq * (1.0f - cos_mult_exp_term) + Ue4sq * Ue4sq * osc_term;
        }
        return probs;
    };
    return r;
}

static inline ModelRecipe make_3p2_recipe() {
    ModelRecipe r;
    r.name = "3+2";
    r.uses_simple_sine_kernel = false;
    r.params = {
        {"dmsq41", "#Delta m^{2}_{41}", "eV^{2}", true, -2.0f, 2.0f, -1.0f},
        {"dmsq51", "#Delta m^{2}_{51}", "eV^{2}", true, -2.0f, 2.0f,  0.0f},
        {"Ue4sq",  "|U_{e4}|^{2}",      "", true, -5.0f, -0.01f, -4.0f},
        {"Um4sq",  "|U_{#mu4}|^{2}",    "", true, -5.0f, -0.01f, -4.0f},
        {"Ue5sq",  "|U_{e5}|^{2}",      "", true, -5.0f, -0.01f, -4.0f},
        {"Um5sq",  "|U_{#mu5}|^{2}",    "", true, -5.0f, -0.01f, -4.0f},
        {"phi54",  "#phi_{54}",         "rad", false, 0.0f, 6.28318f, 0.0f}
    };
    r.channels = {
        {0, "null", ProbForm::Null},
        {1, "mumu", ProbForm::Custom},
        {2, "mue",  ProbForm::Custom},
        {3, "ee",   ProbForm::Custom}
    };
    r.constraint_function = [](const Eigen::VectorXf &v, const PRODynamicOscModel &m) {
        const float Ue4sq = m.par(v, "Ue4sq");
        const float Um4sq = m.par(v, "Um4sq");
        const float Ue5sq = m.par(v, "Ue5sq");
        const float Um5sq = m.par(v, "Um5sq");

        if(Ue4sq + Ue5sq >= 1.0f) return 0;
        if(Um4sq + Um5sq >= 1.0f) return 0;
        if(4.0f * Um4sq * Ue4sq + 4.0f * Um5sq * Ue5sq
           + 8.0f * std::sqrt(Um4sq * Ue4sq * Um5sq * Ue5sq) >= 1.0f) return 0;
        if(1.0f - 4.0f * (1.0f - Ue4sq - Ue5sq) * (Ue4sq + Ue5sq)
           - 4.0f * Ue4sq * Ue5sq >= 1.0f) return 0;
        if(1.0f - 4.0f * (1.0f - Um4sq - Um5sq) * (Um4sq + Um5sq)
           - 4.0f * Um4sq * Um5sq >= 1.0f) return 0;
        return 1;
    };
    r.custom_get_probs_function = [](const Eigen::VectorXf &phys,
                                     const std::vector<float> &le_arr,
                                     const PRODynamicOscModel &m) {
        const float dm41  = m.par(phys, "dmsq41");
        const float dm51  = m.par(phys, "dmsq51");
        const float Ue4sq = m.par(phys, "Ue4sq");
        const float Um4sq = m.par(phys, "Um4sq");
        const float Ue5sq = m.par(phys, "Ue5sq");
        const float Um5sq = m.par(phys, "Um5sq");
        const float phi54 = m.par(phys, "phi54");
        const float eps = 1e-6f;

        Eigen::MatrixXf probs(le_arr.size(), 4);
        for(size_t i = 0; i < le_arr.size(); ++i) {
            const float le = le_arr[i];
            const float x41 = 1.266932679f * dm41 * le;
            const float x51 = 1.266932679f * dm51 * le;
            const float x54 = 1.266932679f * (dm51 - dm41) * le;

            const float s41 = std::sin(x41);
            const float s51 = std::sin(x51);
            const float s54 = std::sin(x54);

            float Pmue = 4.0f * Um4sq * Ue4sq * s41 * s41
                       + 4.0f * Um5sq * Ue5sq * s51 * s51
                       + 8.0f * std::sqrt(Um4sq * Ue4sq * Um5sq * Ue5sq)
                             * s41 * s51 * std::cos(x54 - phi54);

            float Pmumu = 1.0f
                         - 4.0f * (1.0f - Um4sq - Um5sq) * (Um4sq * s41 * s41 + Um5sq * s51 * s51)
                         - 4.0f * Um4sq * Um5sq * s54 * s54;

            float Pee = 1.0f
                      - 4.0f * (1.0f - Ue4sq - Ue5sq) * (Ue4sq * s41 * s41 + Ue5sq * s51 * s51)
                      - 4.0f * Ue4sq * Ue5sq * s54 * s54;

            if(Pmue < 0.0f && Pmue > -eps) Pmue = 0.0f;
            if(Pmumu < 0.0f && Pmumu > -eps) Pmumu = 0.0f;
            if(Pee < 0.0f && Pee > -eps) Pee = 0.0f;
            if(Pmue > 1.0f && Pmue < 1.0f + eps) Pmue = 1.0f;
            if(Pmumu > 1.0f && Pmumu < 1.0f + eps) Pmumu = 1.0f;
            if(Pee > 1.0f && Pee < 1.0f + eps) Pee = 1.0f;

            probs(i, 0) = 1.0f;
            probs(i, 1) = Pmumu;
            probs(i, 2) = Pmue;
            probs(i, 3) = Pee;
        }
        return probs;
    };
    return r;
}

class ModelRecipeRegistry {
public:
    static ModelRecipe get(const std::string &name) {
        if(name == "numudis") {
            return make_sbl2_recipe("numudis", "sinsq2thmm",
                                    "sin^{2}2#theta_{#mu#mu}", "mumu",
                                    ProbForm::Disappearance, neg_inf());
        }
        if(name == "nueapp") {
            return make_sbl2_recipe("nueapp", "sinsq2thme",
                                    "sin^{2}2#theta_{#mue}", "mue",
                                    ProbForm::Appearance, -10.0f);
        }
        if(name == "nuedis") {
            return make_sbl2_recipe("nuedis", "sinsq2thee",
                                    "sin^{2}2#theta_{ee}", "ee",
                                    ProbForm::Disappearance, neg_inf());
        }
        if(name == "NCnumudisapp") {
            return make_sbl2_recipe("NCnumudisapp", "sinsq2thms",
                                    "sin^{2}2#theta_{#mus}", "mus",
                                    ProbForm::Disappearance, neg_inf());
        }
        if(name == "NCdisapp") return make_ncdisapp_recipe();
        if(name == "3+1") return make_3p1_recipe();
        if(name == "3+1_angles") return make_3p1_angles_recipe();
        if(name == "3+1_3A") return make_3p1_nue_disappearance_recipe(false);
        if(name == "3+1_3B") return make_3p1_numu_disappearance_recipe(false);
        if(name == "3+1_3C") return make_3p1_numu_to_nue_appearance_recipe(false);
        if(name == "3+1_3A_NC") return make_3p1_nue_disappearance_recipe(true);
        if(name == "3+1_3B_NC") return make_3p1_numu_disappearance_recipe(true);
        if(name == "3+1_3C_NC") return make_3p1_numu_to_nue_appearance_recipe(true);
        if(name == "3+1_decay_invis") return make_decay_invis_recipe();
        if(name == "3+2") return make_3p2_recipe();

        log<LOG_ERROR>(L"%1% || Unrecognized model name %2%. Try numudis, nueapp, nuedis, NCnumudisapp, NCdisapp, 3+1, 3+1_angles, 3+1_3(A,B,C), 3+1_3(A,B,C)_NC, 3+1_decay_invis, 3+2. Terminating.")
            % __func__ % name.c_str();
        exit(EXIT_FAILURE);
    }
};

// Main interface to different models
static inline
std::unique_ptr<PROmodel> get_model_from_string(const PROconfig& config, const PROpeller &prop) {
    std::string name = config.m_model_tag;

    ModelRecipe recipe = ModelRecipeRegistry::get(name);
    return std::unique_ptr<PROmodel>(new PRODynamicOscModel(prop, config, recipe));
}

} // namespace PROfit

#endif
