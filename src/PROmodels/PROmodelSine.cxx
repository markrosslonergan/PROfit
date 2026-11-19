/**
 * @file PROmodelSine.cxx
 * @brief Recipe-driven sine-kernel oscillation models: shared kernel + the recipe registry.
 * @author PROfit Collaboration
 *
 * @details The recipes below that replace former concrete classes (numudis, nueapp,
 * nuedis, 3+1, 3+1_angles, 3+1_3A/3B/3C) copy the legacy amplitude/derivative
 * expression blocks VERBATIM — including mixed float/double sub-expressions and the
 * exact association order — so the recipe models reproduce the old get_probs and
 * get_probs_grad output bit-for-bit. Do not "clean up" those expressions; see the
 * GradStyle/SineJacs documentation in PROmodelSine.h.
 */
#include "PROmodels/PROmodelSine.h"

#include <cmath>
#include <limits>

namespace PROfit {

// ------------------------------------------------------------------
// PROsineModel kernel
// ------------------------------------------------------------------

PROsineModel::PROsineModel(const PROpeller &prop, const std::map<std::string,int> &parameter_map,
                           const SineModelRecipe &recipe)
    : recipe_(recipe)
{
    J_ = recipe.channels.size();
    if(J_ == 0 || J_ > kMaxSineChannels) {
        log<LOG_ERROR>(L"%1% || Recipe %2% has %3% channels; must be 1..%4%. Terminating.")
            % __func__ % recipe.tag % J_ % kMaxSineChannels;
        throw std::runtime_error("SineModelRecipe channel count out of range");
    }
    for(size_t c = 0; c < J_; ++c) {
        forms_[c] = recipe.channels[c].form;
        if(recipe.channels[c].form != ProbForm::Null)
            osc_channels_.push_back((uint8_t)c);
        model_functions.push_back([this, c](const Eigen::VectorXf &v, float le) {
            return this->channel_prob(c, v, le);
        });
        prob_types.push_back(c);
    }

    if(parameter_map.find("L/E") == parameter_map.end()) {
        log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
        throw std::runtime_error("Missing parameter: L/E");
    }
    ivars = {parameter_map.at("L/E")};

    if(recipe.constraint)
        model_constraint = [this](const Eigen::VectorXf &v) { return recipe_.constraint(*this, v); };

    build_hists_and_combined(prop);

    nparams = recipe.params.size();
    if(nparams == 0 || nparams > kMaxSineParams) {
        log<LOG_ERROR>(L"%1% || Recipe %2% has %3% params; must be 1..%4%. Terminating.")
            % __func__ % recipe.tag % nparams % kMaxSineParams;
        throw std::runtime_error("SineModelRecipe parameter count out of range");
    }
    lb = Eigen::VectorXf(nparams);
    ub = Eigen::VectorXf(nparams);
    default_val = Eigen::VectorXf(nparams);
    for(size_t i = 0; i < nparams; ++i) {
        const SineParameterSpec &p = recipe.params[i];
        param_names.push_back(p.name);
        pretty_param_names.push_back(p.pretty);
        pretty_param_units.push_back(p.unit);
        is_log10.push_back(p.log10);
        lb((Eigen::Index)i) = p.lb;
        ub((Eigen::Index)i) = p.ub;
        default_val((Eigen::Index)i) = p.def;
        // The pin-by-equal-bounds idiom (FC/AFC/bkg-only seeds) requires the
        // default to lie inside the box.
        if(!(p.def >= p.lb && p.def <= p.ub)) {
            log<LOG_ERROR>(L"%1% || Recipe %2% param %3% default %4% outside [%5%, %6%]. Terminating.")
                % __func__ % recipe.tag % p.name % p.def % p.lb % p.ub;
            throw std::runtime_error("SineModelRecipe default outside bounds");
        }
    }
    build_param_index();

    log<LOG_INFO>(L"%1% || setting up sine-kernel model %2% with %3% params and %4% channels.")
        % __func__ % recipe.tag % nparams % J_;
    for(size_t i = 0; i < nparams; i++) {
        log<LOG_INFO>(L"%1% || Param %2% is %3% with lower bound/upper bound of %4%/%5% and default %6%")
            % __func__ % i % param_names[i].c_str() % lb[i] % ub[i] % default_val[i];
    }
}

float PROsineModel::channel_prob(size_t c, const Eigen::VectorXf &phys, float le) const {
    // Per-event scalar path backing model_functions; used only by the base-class
    // FD fallback and pybind, never by the built binary's hot path (get_probs and
    // get_probs_grad are overridden below).
    SineAmps amp;
    recipe_.amps(*this, phys, amp);
    if(forms_[c] == ProbForm::Null) return 1.0f;
    float sinterm = std::sin(1.266932679f * amp.dmsq * le);
    return forms_[c] == ProbForm::Appearance ? amp.A[c] * sinterm * sinterm
                                             : 1.0f - amp.A[c] * sinterm * sinterm;
}

float PROsineModel::cell_sin2(float freq, float le, float hw, float *ds2_dfreq) {
    // <sin^2 x> = (1 - cos 2x sinc u)/2 over a uniform window of half-width hw in L/E
    // (u = 2 freq hw), in double: x is large at high dmsq.
    const double x2 = 2.0 * (double)freq * le;
    double dS;
    const double S = osc_sinc(2.0 * (double)freq * hw, ds2_dfreq ? &dS : nullptr);
    if(ds2_dfreq) *ds2_dfreq = (float)(le * std::sin(x2) * S - std::cos(x2) * hw * dS);
    return (float)(0.5 * (1.0 - std::cos(x2) * S));
}

Eigen::MatrixXf PROsineModel::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    const auto &le_arr = var_arrs[0];

    // Amplitudes once per call; per grid point only sin + one multiply-add per channel.
    SineAmps amp;
    recipe_.amps(*this, phys, amp);
    const float freq = 1.266932679f * amp.dmsq;

    // Column-major (Eigen default) is REQUIRED: probs.data() is consumed flattened
    // as [col0|col1|...] against H_combined (see FillSpectra, src/PROcess.cxx).
    Eigen::MatrixXf probs(le_arr.size(), J_);

    // Underflow/overflow cells of the binned grid (hw > 0) are averaged over their spread.
    const std::vector<float> *hw = &var_arrs == &phys_grid ? &phys_axes_hw[0] : nullptr;

    for(size_t i = 0; i < le_arr.size(); ++i) {
        if(hw && (*hw)[i] > 0) {
            const float s2 = cell_sin2(freq, le_arr[i], (*hw)[i], nullptr);
            for(size_t c = 0; c < J_; ++c)
                probs(i, c) = forms_[c] == ProbForm::Null ? 1.0f
                            : forms_[c] == ProbForm::Appearance ? amp.A[c] * s2 : 1.0f - amp.A[c] * s2;
            continue;
        }
        const float sinterm = std::sin(freq * le_arr[i]);
        for(size_t c = 0; c < J_; ++c) {
            switch(forms_[c]) {
                case ProbForm::Null:
                    probs(i, c) = 1.0f;
                    break;
                case ProbForm::Appearance:
                    probs(i, c) = amp.A[c] * sinterm * sinterm;
                    break;
                case ProbForm::Disappearance:
                    probs(i, c) = 1.0f - amp.A[c] * sinterm * sinterm;
                    break;
            }
        }
    }
    return probs;
}

std::vector<Eigen::MatrixXf> PROsineModel::get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    const auto &le_arr = var_arrs[0];

    SineJacs jac;
    recipe_.jacs(*this, phys, jac);

    constexpr float k = 1.266932679f;
    const float freq = k * jac.dmsq;
    const bool hoisted = (recipe_.grad_style == GradStyle::HoistedPhase);

    std::vector<Eigen::MatrixXf> grads(nparams, Eigen::MatrixXf::Zero(le_arr.size(), J_));
    const std::vector<float> *hw = &var_arrs == &phys_grid ? &phys_axes_hw[0] : nullptr;
    for(size_t i = 0; i < le_arr.size(); ++i) {
        const float le = le_arr[i];
        if(hw && (*hw)[i] > 0) {
            float ds2;
            const float s2 = cell_sin2(freq, le, (*hw)[i], &ds2);
            for(uint8_t c : osc_channels_)
                grads[0](i, c) = jac.Cdm[c] * ds2 * k * jac.ddm;
            for(const SineGradEntry &e : recipe_.grad_entries)
                grads[e.param](i, e.channel) = jac.pre[e.param][e.channel] * s2 * jac.post[e.param][e.channel];
            continue;
        }
        const float x = freq * le;
        const float sinterm = std::sin(x);
        const float s2 = sinterm * sinterm;

        // Delta m^2 column: two legacy association orders (see GradStyle).
        if(hoisted) {
            const float dsin2_ddm = std::sin(2.0f*x) * k * le * jac.ddm;
            for(uint8_t c : osc_channels_)
                grads[0](i, c) = jac.Cdm[c] * dsin2_ddm;
        } else {
            for(uint8_t c : osc_channels_)
                grads[0](i, c) = jac.Cdm[c] * std::sin(2.0f*x) * 1.266932679f * le * jac.ddm;
        }

        // Mixing columns: only the listed structurally non-zero entries are written,
        // so the untouched entries stay exactly Zero as in the legacy code.
        for(const SineGradEntry &e : recipe_.grad_entries)
            grads[e.param](i, e.channel) = jac.pre[e.param][e.channel] * s2 * jac.post[e.param][e.channel];
    }
    return grads;
}

// ------------------------------------------------------------------
// Recipe makers.
//
// LEGACY-VERBATIM blocks: the amps/jacs bodies of the converted models are copied
// character-for-character from the deleted concrete classes (PROmodel2flav.cxx,
// PROmodel3p1.cxx) — including double-precision sub-expressions like
// (1.0+sqrt(1.0-s214))/2.0f — to keep the output bitwise identical.
// ------------------------------------------------------------------

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float LN10 = 2.302585093f;

/// Shared maker for the single-channel two-flavour-like models
/// (SBL_2flav_numudis / _nueapp / _nuedis / _numudis_NC). sign = -1 for disappearance,
/// +1 for appearance; the clamp/chain-factor blocks are the legacy 2-flavour code.
SineModelRecipe make_twoflav(const char *tag, const char *mixing_name, const char *mixing_pretty,
                             const char *channel_name, ProbForm form, float mixing_lb) {
    SineModelRecipe r;
    r.tag = tag;
    r.params = {
        {"dmsq", "#Deltam^{2}", "eV^{2}", true, -2.0f, 2.0f, -2.0f},
        {mixing_name, mixing_pretty, "", true, mixing_lb, 0.0f, -10.0f},
    };
    r.channels = {{"null", ProbForm::Null}, {channel_name, form}};
    r.grad_entries = {{1, 1}};
    r.grad_style = GradStyle::InlineTwoFlav;
    const float sign = (form == ProbForm::Appearance) ? 1.0f : -1.0f;
    r.amps = [mixing_name](const PROsineModel &m, const Eigen::VectorXf &phys, SineAmps &a) {
        float dmsq = m.maybe_convert_log("dmsq", phys(0));
        float s = m.maybe_convert_log(mixing_name, phys(1));
        if(s > 1) s = 1;
        if(s < 0) s = 0;
        a.dmsq = dmsq;
        a.A[1] = s;
    };
    r.jacs = [mixing_name, sign](const PROsineModel &m, const Eigen::VectorXf &phys, SineJacs &j) {
        float dmsq = m.maybe_convert_log("dmsq", phys(0));
        float s = m.maybe_convert_log(mixing_name, phys(1));
        float ddm = m.is_log10[0] ? LN10 * dmsq : 1.0f;
        float dss = m.is_log10[1] ? LN10 * s : 1.0f;
        // Match get_probs' clamp: the clamped parameter has zero local sensitivity.
        if(s > 1) { s = 1; dss = 0; }
        if(s < 0) { s = 0; dss = 0; }
        j.dmsq = dmsq;
        j.ddm = ddm;
        j.Cdm[1] = sign * s;
        j.pre[1][1] = sign * 1.0f;
        j.post[1][1] = dss;
    };
    return r;
}

/// Full 3+1 (|U|^2 parameterisation) and its NC-disappearance extension.
/// The NC variant stays in the family's parameter language: it adds
/// Uta4sq = |Utau4|^2 (log10), with |Us4|^2 = max(0, 1 - |Ue4|^2 - |Um4|^2 - |Utau4|^2).
SineModelRecipe make_3p1(bool with_nc) {
    SineModelRecipe r;
    r.tag = with_nc ? "SBL_3+1_Usq_NC" : "SBL_3+1_Usq";
    r.params = {
        {"dmsq", "#Deltam^{2}", "eV^{2}", true, -2.0f, 2.0f, -2.0f},
        {"Ue4^2", "|U_{e4}|^{2}", "", true, -kInf, -1e-4f, -8.0f},
        {"Um4^2", "|U_{#mu4}|^{2}", "", true, -kInf, -1e-4f, -8.0f},
    };
    r.channels = {{"null", ProbForm::Null}, {"mumu", ProbForm::Disappearance},
                  {"mue", ProbForm::Appearance}, {"ee", ProbForm::Disappearance}};
    r.grad_entries = {{2, 1}, {1, 2}, {2, 2}, {1, 3}};
    r.grad_style = GradStyle::HoistedPhase;
    if(with_nc) {
        r.params.push_back({"Uta4sq", "|U_{#tau4}|^{2}", "", true, -kInf, -1e-4f, -8.0f});
        r.channels.push_back({"mus", ProbForm::Disappearance});
        r.channels.push_back({"es", ProbForm::Disappearance});
        r.channels.push_back({"mutau", ProbForm::Appearance});
        r.channels.push_back({"etau", ProbForm::Appearance});
        // Exact sparsity: A6 = 4 Um Uta has no Ue dependence, A7 = 4 Ue Uta no Um.
        r.grad_entries.insert(r.grad_entries.end(),
            {{1, 4}, {2, 4}, {3, 4}, {1, 5}, {2, 5}, {3, 5}, {2, 6}, {3, 6}, {1, 7}, {3, 7}});
    }
    r.amps = [with_nc](const PROsineModel &m, const Eigen::VectorXf &phys, SineAmps &a) {
        float dmsq = m.maybe_convert_log("dmsq", phys(0));
        float Ue4sq = m.maybe_convert_log("Ue4^2", phys(1));
        float Um4sq = m.maybe_convert_log("Um4^2", phys(2));
        a.dmsq = dmsq;
        a.A[1] = 4.0f*Um4sq*(1.0f-Um4sq);
        a.A[2] = 4.0f*Ue4sq*Um4sq;
        a.A[3] = 4.0f*Ue4sq*(1.0f-Ue4sq);
        if(with_nc) {
            float Uta4sq = m.maybe_convert_log("Uta4sq", phys(3));
            float Us4sq = std::max(0.0f, 1.0f - Ue4sq - Um4sq - Uta4sq);
            a.A[4] = 4.0f*Um4sq*Us4sq;
            a.A[5] = 4.0f*Ue4sq*Us4sq;
            a.A[6] = 4.0f*Um4sq*Uta4sq;
            a.A[7] = 4.0f*Ue4sq*Uta4sq;
        }
    };
    r.jacs = [with_nc](const PROsineModel &m, const Eigen::VectorXf &phys, SineJacs &j) {
        float dmsq  = m.maybe_convert_log("dmsq",  phys(0));
        float Ue4sq = m.maybe_convert_log("Ue4^2", phys(1));
        float Um4sq = m.maybe_convert_log("Um4^2", phys(2));
        float ddm = m.is_log10[0] ? LN10 * dmsq  : 1.0f;
        float dUe = m.is_log10[1] ? LN10 * Ue4sq : 1.0f;
        float dUm = m.is_log10[2] ? LN10 * Um4sq : 1.0f;
        j.dmsq = dmsq;
        j.ddm = ddm;
        j.Cdm[1] = -4.0f * Um4sq * (1.0f - Um4sq);
        j.Cdm[2] =  4.0f * Ue4sq * Um4sq;
        j.Cdm[3] = -4.0f * Ue4sq * (1.0f - Ue4sq);
        j.pre[2][1] = -4.0f * (1.0f - 2.0f*Um4sq); j.post[2][1] = dUm;
        j.pre[1][2] =  4.0f * Um4sq;               j.post[1][2] = dUe;
        j.pre[2][2] =  4.0f * Ue4sq;               j.post[2][2] = dUm;
        j.pre[1][3] = -4.0f * (1.0f - 2.0f*Ue4sq); j.post[1][3] = dUe;
        if(with_nc) {
            float Uta4sq = m.maybe_convert_log("Uta4sq", phys(3));
            float dUta = m.is_log10[3] ? LN10 * Uta4sq : 1.0f;
            float Usraw = 1.0f - Ue4sq - Um4sq - Uta4sq;
            float Us4sq = std::max(0.0f, Usraw);
            // At the max() boundary |Us4|^2 is locally constant (clamp-zero convention).
            // The dU* tails already carry their chain factors, so post stays 1.
            float dUs_dUe  = (Usraw > 0.0f ? -dUe  : 0.0f);
            float dUs_dUm  = (Usraw > 0.0f ? -dUm  : 0.0f);
            float dUs_dUta = (Usraw > 0.0f ? -dUta : 0.0f);
            j.Cdm[4] = -4.0f * Um4sq * Us4sq;
            j.Cdm[5] = -4.0f * Ue4sq * Us4sq;
            j.Cdm[6] =  4.0f * Um4sq * Uta4sq;
            j.Cdm[7] =  4.0f * Ue4sq * Uta4sq;
            // ch 4 (mus, Dis): A4 = 4 Um Us
            j.pre[1][4] = -4.0f*Um4sq*dUs_dUe;               j.post[1][4] = 1.0f;
            j.pre[2][4] = -4.0f*(dUm*Us4sq + Um4sq*dUs_dUm); j.post[2][4] = 1.0f;
            j.pre[3][4] = -4.0f*Um4sq*dUs_dUta;              j.post[3][4] = 1.0f;
            // ch 5 (es, Dis): A5 = 4 Ue Us
            j.pre[1][5] = -4.0f*(dUe*Us4sq + Ue4sq*dUs_dUe); j.post[1][5] = 1.0f;
            j.pre[2][5] = -4.0f*Ue4sq*dUs_dUm;               j.post[2][5] = 1.0f;
            j.pre[3][5] = -4.0f*Ue4sq*dUs_dUta;              j.post[3][5] = 1.0f;
            // ch 6 (mutau, App): A6 = 4 Um Uta
            j.pre[2][6] = 4.0f*dUm*Uta4sq;                   j.post[2][6] = 1.0f;
            j.pre[3][6] = 4.0f*Um4sq*dUta;                   j.post[3][6] = 1.0f;
            // ch 7 (etau, App): A7 = 4 Ue Uta
            j.pre[1][7] = 4.0f*dUe*Uta4sq;                   j.post[1][7] = 1.0f;
            j.pre[3][7] = 4.0f*Ue4sq*dUta;                   j.post[3][7] = 1.0f;
        }
    };
    if(with_nc) {
        // Unitarity extended by the tau row, same style as the base constraint.
        r.constraint = [](const PROsineModel &m, const Eigen::VectorXf &v) {
            const float Ue4sq = m.maybe_convert_log("Ue4^2", v(1));
            const float Um4sq = m.maybe_convert_log("Um4^2", v(2));
            const float Uta4sq = m.maybe_convert_log("Uta4sq", v(3));
            return   ((Ue4sq+Um4sq+Uta4sq)<1 ? 1 : 0);
        };
    } else {
        r.constraint = [](const PROsineModel &m, const Eigen::VectorXf &v) {
            const float Ue4sq = m.maybe_convert_log("Ue4^2", v(1));
            const float Um4sq = m.maybe_convert_log("Um4^2", v(2));
            return   ((Ue4sq+Um4sq)<1 ? 1 : 0);
        };
    }
    return r;
}

/// 3+1 angle parameterisation and its NC-disappearance extension.
/// The NC variant stays in the family's parameter language: it adds
/// cosq34 = cos^2(theta_34) (linear, [0,1], default 1 = all-sterile), splitting
/// |Us4|^2 + |Utau4|^2 = c14^2 c24^2 = c14*(1 - s24) in the model's internal
/// variables (c14 = cos^2 theta_14, s24 = sin^2 theta_24).
SineModelRecipe make_3p1_angles(bool with_nc) {
    SineModelRecipe r;
    r.tag = with_nc ? "SBL_3+1_angles_NC" : "SBL_3+1_angles";
    r.params = {
        {"dmsq", "#Deltam^{2}", "eV^{2}", true, -2.0f, 2.0f, -2.0f},
        {"sinsq2th14", "sin^{2}2#theta_{14}", "", true, -kInf, -1e-4f, -8.0f},
        {"sinsqth24", "sin^{2}#theta_{24}", "", true, -kInf, -1e-4f, -8.0f},
    };
    r.channels = {{"null", ProbForm::Null}, {"mumu", ProbForm::Disappearance},
                  {"mue", ProbForm::Appearance}, {"ee", ProbForm::Disappearance}};
    r.grad_entries = {{1, 1}, {2, 1}, {1, 2}, {2, 2}, {1, 3}};
    r.grad_style = GradStyle::HoistedPhase;
    if(with_nc) {
        r.params.push_back({"cosq34", "cos^{2}#theta_{34}", "", false, 0.0f, 1.0f, 1.0f});
        r.channels.push_back({"mus", ProbForm::Disappearance});
        r.channels.push_back({"es", ProbForm::Disappearance});
        r.channels.push_back({"mutau", ProbForm::Appearance});
        r.channels.push_back({"etau", ProbForm::Appearance});
        for(uint8_t c = 4; c <= 7; ++c)
            for(uint8_t p = 1; p <= 3; ++p)
                r.grad_entries.push_back({p, c});
    }
    r.amps = [with_nc](const PROsineModel &m, const Eigen::VectorXf &phys, SineAmps &a) {
        float dmsq = m.maybe_convert_log("dmsq", phys(0));
        float s214 = m.maybe_convert_log("sinsq2th14", phys(1));
        float s24 = m.maybe_convert_log("sinsqth24", phys(2));
        float c14 = (1.0+sqrt(1.0-s214))/2.0f;
        a.dmsq = dmsq;
        a.A[1] = 4.0f*c14*s24*(1.0f-c14*s24);
        a.A[2] = s214*s24;
        a.A[3] = s214;
        if(with_nc) {
            float c34 = m.maybe_convert_log("cosq34", phys(3));
            float off = c14*(1.0f-s24);            // |Us4|^2 + |Utau4|^2 = c14^2 c24^2
            float mu_off = 4.0f*(c14*s24)*off;     // 4 |Um4|^2 (|Us4|^2 + |Utau4|^2)
            float e_off  = 4.0f*(1.0f-c14)*off;    // 4 |Ue4|^2 (|Us4|^2 + |Utau4|^2)
            a.A[4] = mu_off*c34;
            a.A[5] = e_off*c34;
            a.A[6] = mu_off*(1.0f-c34);
            a.A[7] = e_off*(1.0f-c34);
        }
    };
    r.jacs = [with_nc](const PROsineModel &m, const Eigen::VectorXf &phys, SineJacs &j) {
        float dmsq = m.maybe_convert_log("dmsq", phys(0));
        float s214 = m.maybe_convert_log("sinsq2th14", phys(1));
        float s24  = m.maybe_convert_log("sinsqth24", phys(2));
        float root = std::sqrt(std::max(0.0f, 1.0f - s214));
        float c14  = (1.0f + root) / 2.0f;
        float ddm   = m.is_log10[0] ? LN10 * dmsq : 1.0f;
        float ds214 = m.is_log10[1] ? LN10 * s214 : 1.0f;
        float ds24  = m.is_log10[2] ? LN10 * s24  : 1.0f;
        // A = c14 * s24 (the |Um4|^2 of Pmumu); dc14/ds214 = -1/(4 root).
        float A = c14 * s24;
        float dA_ds214 = (root > 0.0f ? -s24 / (4.0f * root) : 0.0f) * ds214;
        float dA_ds24  = c14 * ds24;
        j.dmsq = dmsq;
        j.ddm = ddm;
        j.Cdm[1] = -4.0f * A * (1.0f - A);
        j.Cdm[2] = s214 * s24;
        j.Cdm[3] = -s214;
        j.pre[1][1] = -4.0f * (1.0f - 2.0f*A); j.post[1][1] = dA_ds214;
        j.pre[2][1] = -4.0f * (1.0f - 2.0f*A); j.post[2][1] = dA_ds24;
        j.pre[1][2] = s24;                     j.post[1][2] = ds214;
        j.pre[2][2] = s214;                    j.post[2][2] = ds24;
        j.pre[1][3] = -1.0f;                   j.post[1][3] = ds214;
        if(with_nc) {
            float c34 = m.maybe_convert_log("cosq34", phys(3));  // linear, chain factor 1
            float Ue = 1.0f - c14;                               // |Ue4|^2 = sin^2 theta_14
            float off = c14*(1.0f-s24);
            float mu_off = 4.0f*A*off;
            float e_off  = 4.0f*Ue*off;
            // Physical partials via product rule (chain factors ride in post).
            float dc14_phys = (root > 0.0f ? -1.0f/(4.0f*root) : 0.0f);
            float doff_ds214 = (1.0f-s24)*dc14_phys;
            float doff_ds24  = -c14;
            float dmu_off_ds214 = 4.0f*(s24*dc14_phys*off + A*doff_ds214);
            float dmu_off_ds24  = 4.0f*(c14*off + A*doff_ds24);
            float de_off_ds214  = 4.0f*(-dc14_phys*off + Ue*doff_ds214);
            float de_off_ds24   = 4.0f*(Ue*doff_ds24);
            j.Cdm[4] = -mu_off*c34;
            j.Cdm[5] = -e_off*c34;
            j.Cdm[6] = mu_off*(1.0f-c34);
            j.Cdm[7] = e_off*(1.0f-c34);
            // Disappearance channels contribute -dA * s2, appearance +dA * s2.
            j.pre[1][4] = -(dmu_off_ds214*c34);      j.post[1][4] = ds214;
            j.pre[1][5] = -(de_off_ds214*c34);       j.post[1][5] = ds214;
            j.pre[1][6] = dmu_off_ds214*(1.0f-c34);  j.post[1][6] = ds214;
            j.pre[1][7] = de_off_ds214*(1.0f-c34);   j.post[1][7] = ds214;
            j.pre[2][4] = -(dmu_off_ds24*c34);       j.post[2][4] = ds24;
            j.pre[2][5] = -(de_off_ds24*c34);        j.post[2][5] = ds24;
            j.pre[2][6] = dmu_off_ds24*(1.0f-c34);   j.post[2][6] = ds24;
            j.pre[2][7] = de_off_ds24*(1.0f-c34);    j.post[2][7] = ds24;
            // d/d cosq34: dA4 = mu_off, dA6 = -mu_off (likewise e_off) — all four
            // entries come out -mu_off/-e_off after the Dis/App sign.
            j.pre[3][4] = -mu_off; j.post[3][4] = 1.0f;
            j.pre[3][5] = -e_off;  j.post[3][5] = 1.0f;
            j.pre[3][6] = -mu_off; j.post[3][6] = 1.0f;
            j.pre[3][7] = -e_off;  j.post[3][7] = 1.0f;
        }
    };
    r.constraint = [](const PROsineModel &, const Eigen::VectorXf &) { return 1; };
    return r;
}

/// 3A (nue-disappearance parameterisation) and its NC-disappearance extension.
/// with_nc adds cosq34 = cos^2(theta_34) (linear, [0,1], default 1) and the four
/// NC channels; channels 1-3 share the legacy-verbatim 3A expressions either way.
SineModelRecipe make_3p1_3A(bool with_nc) {
    SineModelRecipe r;
    r.tag = with_nc ? "SBL_3+1_sinsq2thee_NC" : "SBL_3+1_sinsq2thee";
    r.params = {
        {"dmsq", "#Deltam^{2}", "eV^{2}", true, -2.0f, 2.0f, -2.0f},
        {"sinsq2thee", "sin^{2}2#theta_{ee}", "", true, -kInf, -1e-3f, -8.0f},
        {"sinsqth24", "sin^{2}#theta_{24}", "", true, -kInf, -1e-3f, -8.0f},
    };
    r.channels = {{"null", ProbForm::Null}, {"mumu", ProbForm::Disappearance},
                  {"mue", ProbForm::Appearance}, {"ee", ProbForm::Disappearance}};
    r.grad_entries = {{1, 1}, {2, 1}, {1, 2}, {2, 2}, {1, 3}};
    r.grad_style = GradStyle::HoistedPhase;
    if(with_nc) {
        // cosq34 = 1 sends all of the off-diagonal strength |Us4|^2+|Utau4|^2 to
        // sterile (maximal NC deficit); cosq34 = 0 sends it all to nu_tau.
        r.params.push_back({"cosq34", "cos^{2}#theta_{34}", "", false, 0.0f, 1.0f, 1.0f});
        r.channels.push_back({"mus", ProbForm::Disappearance});
        r.channels.push_back({"es", ProbForm::Disappearance});
        r.channels.push_back({"mutau", ProbForm::Appearance});
        r.channels.push_back({"etau", ProbForm::Appearance});
        for(uint8_t c = 4; c <= 7; ++c)
            for(uint8_t p = 1; p <= 3; ++p)
                r.grad_entries.push_back({p, c});
    }
    r.amps = [with_nc](const PROsineModel &m, const Eigen::VectorXf &phys, SineAmps &a) {
        float dmsq = m.maybe_convert_log("dmsq", phys(0));
        float sinsq2thee = m.maybe_convert_log("sinsq2thee", phys(1));
        float sinsqth24 = m.maybe_convert_log("sinsqth24", phys(2));
        float Um4sq = sinsqth24/2.0*(1.0+sqrt(1.0f- sinsq2thee));
        float sinsq2thmue = sinsqth24*sinsq2thee;
        a.dmsq = dmsq;
        a.A[1] = 4.0f*Um4sq*(1.0f-Um4sq);
        a.A[2] = sinsq2thmue;
        a.A[3] = sinsq2thee;
        if(with_nc) {
            // Split of the off-diagonal amplitudes 4|Ua4|^2 (|Us4|^2 + |Utau4|^2)
            // = 4|Ua4|^2 c14^2 c24^2 into sterile (cosq34) and tau (1-cosq34) parts.
            float q = std::sqrt(std::max(0.0f, 1.0f - sinsq2thee));
            float u = 1.0f + q;
            float c34 = m.maybe_convert_log("cosq34", phys(3));
            float mu_off = sinsqth24*(1.0f-sinsqth24)*(u*u);
            float e_off = (1.0f-sinsqth24)*sinsq2thee;
            a.A[4] = mu_off*c34;
            a.A[5] = e_off*c34;
            a.A[6] = mu_off*(1.0f-c34);
            a.A[7] = e_off*(1.0f-c34);
        }
    };
    r.jacs = [with_nc](const PROsineModel &m, const Eigen::VectorXf &phys, SineJacs &j) {
        float dmsq = m.maybe_convert_log("dmsq", phys(0));
        float see  = m.maybe_convert_log("sinsq2thee", phys(1));
        float s24  = m.maybe_convert_log("sinsqth24", phys(2));
        float root = std::sqrt(std::max(0.0f, 1.0f - see));
        float Um4sq = s24 / 2.0f * (1.0f + root);
        float smue  = s24 * see;
        float ddm  = m.is_log10[0] ? LN10 * dmsq : 1.0f;
        float dsee = m.is_log10[1] ? LN10 * see  : 1.0f;
        float ds24 = m.is_log10[2] ? LN10 * s24  : 1.0f;
        float dUm_dsee = (root > 0.0f ? -s24 / (4.0f * root) : 0.0f) * dsee;
        float dUm_ds24 = (1.0f + root) / 2.0f * ds24;
        j.dmsq = dmsq;
        j.ddm = ddm;
        j.Cdm[1] = -4.0f * Um4sq * (1.0f - Um4sq);
        j.Cdm[2] = smue;
        j.Cdm[3] = -see;
        j.pre[1][1] = -4.0f * (1.0f - 2.0f*Um4sq); j.post[1][1] = dUm_dsee;
        j.pre[2][1] = -4.0f * (1.0f - 2.0f*Um4sq); j.post[2][1] = dUm_ds24;
        j.pre[1][2] = s24;                         j.post[1][2] = dsee;
        j.pre[2][2] = see;                         j.post[2][2] = ds24;
        j.pre[1][3] = -1.0f;                       j.post[1][3] = dsee;
        if(with_nc) {
            float c34 = m.maybe_convert_log("cosq34", phys(3));  // linear, chain factor 1
            float u = 1.0f + root;
            float mu_off = s24*(1.0f-s24)*(u*u);
            float e_off = (1.0f-s24)*see;
            // Physical partials of the off-diagonal amplitudes (du/dsee = -1/(2 root)).
            float dmu_off_dsee = (root > 0.0f ? -s24*(1.0f-s24)*u/root : 0.0f);
            float dmu_off_ds24 = (1.0f-2.0f*s24)*(u*u);
            float de_off_dsee = (1.0f-s24);
            float de_off_ds24 = -see;
            j.Cdm[4] = -mu_off*c34;
            j.Cdm[5] = -e_off*c34;
            j.Cdm[6] = mu_off*(1.0f-c34);
            j.Cdm[7] = e_off*(1.0f-c34);
            // Disappearance channels contribute -dA * s2, appearance +dA * s2;
            // chain factors ride in post.
            j.pre[1][4] = -(dmu_off_dsee*c34);      j.post[1][4] = dsee;
            j.pre[1][5] = -(de_off_dsee*c34);       j.post[1][5] = dsee;
            j.pre[1][6] = dmu_off_dsee*(1.0f-c34);  j.post[1][6] = dsee;
            j.pre[1][7] = de_off_dsee*(1.0f-c34);   j.post[1][7] = dsee;
            j.pre[2][4] = -(dmu_off_ds24*c34);      j.post[2][4] = ds24;
            j.pre[2][5] = -(de_off_ds24*c34);       j.post[2][5] = ds24;
            j.pre[2][6] = dmu_off_ds24*(1.0f-c34);  j.post[2][6] = ds24;
            j.pre[2][7] = de_off_ds24*(1.0f-c34);   j.post[2][7] = ds24;
            // d/d cosq34: dA4 = mu_off, dA6 = -mu_off (and likewise e_off) — all four
            // entries come out -mu_off/-e_off after the Dis/App sign.
            j.pre[3][4] = -mu_off; j.post[3][4] = 1.0f;
            j.pre[3][5] = -e_off;  j.post[3][5] = 1.0f;
            j.pre[3][6] = -mu_off; j.post[3][6] = 1.0f;
            j.pre[3][7] = -e_off;  j.post[3][7] = 1.0f;
        }
    };
    r.constraint = [](const PROsineModel &, const Eigen::VectorXf &) { return 1; };
    return r;
}

/// 3B (numu-disappearance parameterisation) and its NC-disappearance extension.
SineModelRecipe make_3p1_3B(bool with_nc) {
    SineModelRecipe r;
    r.tag = with_nc ? "SBL_3+1_sinsq2thmumu_NC" : "SBL_3+1_sinsq2thmumu";
    r.params = {
        {"dmsq", "#Deltam^{2}", "eV^{2}", true, -2.0f, 2.0f, -2.0f},
        {"sinsq2thmumu", "sin^{2}2#theta_{#mu#mu}", "", true, -kInf, -1e-3f, -8.0f},
        {"sB", "sB", "", true, -kInf, -1e-3f, -8.0f},
    };
    r.channels = {{"null", ProbForm::Null}, {"mumu", ProbForm::Disappearance},
                  {"mue", ProbForm::Appearance}, {"ee", ProbForm::Disappearance}};
    r.grad_entries = {{1, 1}, {1, 2}, {2, 2}, {1, 3}, {2, 3}};
    r.grad_style = GradStyle::HoistedPhase;
    if(with_nc) {
        r.params.push_back({"cosq34", "cos^{2}#theta_{34}", "", false, 0.0f, 1.0f, 1.0f});
        r.channels.push_back({"mus", ProbForm::Disappearance});
        r.channels.push_back({"es", ProbForm::Disappearance});
        r.channels.push_back({"mutau", ProbForm::Appearance});
        r.channels.push_back({"etau", ProbForm::Appearance});
        for(uint8_t c = 4; c <= 7; ++c)
            for(uint8_t p = 1; p <= 3; ++p)
                r.grad_entries.push_back({p, c});
    }
    r.amps = [with_nc](const PROsineModel &m, const Eigen::VectorXf &phys, SineAmps &a) {
        float dmsq = m.maybe_convert_log("dmsq", phys(0));
        float sinsq2thmumu = m.maybe_convert_log("sinsq2thmumu", phys(1));
        float sB = m.maybe_convert_log("sB", phys(2));
        float Ue4sq = (sB/2.0)*(1.0+sqrt(1.0f-sinsq2thmumu));
        a.dmsq = dmsq;
        a.A[1] = sinsq2thmumu;
        a.A[2] = sB*sinsq2thmumu;
        a.A[3] = 4.0f*(1-Ue4sq)*Ue4sq;
        if(with_nc) {
            // With q = sqrt(1 - s2mumu), Um4sq = (1-q)/2 and
            // 2(1-Ue4sq) - 1 + q = 2(1 - Ue4sq - Um4sq), so
            // mu_off = 4 Um4sq (|Us4|^2 + |Utau4|^2), split by cosq34.
            float q = std::sqrt(std::max(0.0f, 1.0f - sinsq2thmumu));
            float bracket = 2.0f*(1.0f - Ue4sq) - 1.0f + q;
            float mu_off = (1.0f - q)*bracket;
            float e_off = 2.0f*Ue4sq*bracket;
            float c34 = m.maybe_convert_log("cosq34", phys(3));
            a.A[4] = mu_off*c34;
            a.A[5] = e_off*c34;
            a.A[6] = mu_off*(1.0f-c34);
            a.A[7] = e_off*(1.0f-c34);
        }
    };
    r.jacs = [with_nc](const PROsineModel &m, const Eigen::VectorXf &phys, SineJacs &j) {
        float dmsq = m.maybe_convert_log("dmsq", phys(0));
        float smm  = m.maybe_convert_log("sinsq2thmumu", phys(1));
        float sB   = m.maybe_convert_log("sB", phys(2));
        float root = std::sqrt(std::max(0.0f, 1.0f - smm));
        float Ue4sq = (sB / 2.0f) * (1.0f + root);
        float ddm  = m.is_log10[0] ? LN10 * dmsq : 1.0f;
        float dsmm = m.is_log10[1] ? LN10 * smm  : 1.0f;
        float dsB  = m.is_log10[2] ? LN10 * sB   : 1.0f;
        float dUe_dsmm = (root > 0.0f ? -sB / (4.0f * root) : 0.0f) * dsmm;
        float dUe_dsB  = (1.0f + root) / 2.0f * dsB;
        j.dmsq = dmsq;
        j.ddm = ddm;
        j.Cdm[1] = -smm;
        j.Cdm[2] = sB * smm;
        j.Cdm[3] = -4.0f * Ue4sq * (1.0f - Ue4sq);
        j.pre[1][1] = -1.0f;                       j.post[1][1] = dsmm;
        j.pre[1][2] = sB;                          j.post[1][2] = dsmm;
        j.pre[2][2] = smm;                         j.post[2][2] = dsB;
        j.pre[1][3] = -4.0f * (1.0f - 2.0f*Ue4sq); j.post[1][3] = dUe_dsmm;
        j.pre[2][3] = -4.0f * (1.0f - 2.0f*Ue4sq); j.post[2][3] = dUe_dsB;
        if(with_nc) {
            float c34 = m.maybe_convert_log("cosq34", phys(3));  // linear, chain factor 1
            float q = root;
            float bracket = 2.0f*(1.0f - Ue4sq) - 1.0f + q;
            float mu_off = (1.0f - q)*bracket;
            float e_off = 2.0f*Ue4sq*bracket;
            // Physical partials (chain factors applied via post below).
            float dq_dsmm_phys  = (root > 0.0f ? -1.0f/(2.0f*root) : 0.0f);
            float dUe_dsmm_phys = (root > 0.0f ? -sB/(4.0f*root) : 0.0f);
            float dUe_dsB_phys  = (1.0f + root) / 2.0f;
            float dbr_dsmm = -2.0f*dUe_dsmm_phys + dq_dsmm_phys;
            float dbr_dsB  = -2.0f*dUe_dsB_phys;
            float dmu_off_dsmm = -dq_dsmm_phys*bracket + (1.0f-q)*dbr_dsmm;
            float dmu_off_dsB  = (1.0f-q)*dbr_dsB;
            float de_off_dsmm  = 2.0f*(dUe_dsmm_phys*bracket + Ue4sq*dbr_dsmm);
            float de_off_dsB   = 2.0f*(dUe_dsB_phys*bracket + Ue4sq*dbr_dsB);
            j.Cdm[4] = -mu_off*c34;
            j.Cdm[5] = -e_off*c34;
            j.Cdm[6] = mu_off*(1.0f-c34);
            j.Cdm[7] = e_off*(1.0f-c34);
            j.pre[1][4] = -(dmu_off_dsmm*c34);      j.post[1][4] = dsmm;
            j.pre[1][5] = -(de_off_dsmm*c34);       j.post[1][5] = dsmm;
            j.pre[1][6] = dmu_off_dsmm*(1.0f-c34);  j.post[1][6] = dsmm;
            j.pre[1][7] = de_off_dsmm*(1.0f-c34);   j.post[1][7] = dsmm;
            j.pre[2][4] = -(dmu_off_dsB*c34);       j.post[2][4] = dsB;
            j.pre[2][5] = -(de_off_dsB*c34);        j.post[2][5] = dsB;
            j.pre[2][6] = dmu_off_dsB*(1.0f-c34);   j.post[2][6] = dsB;
            j.pre[2][7] = de_off_dsB*(1.0f-c34);    j.post[2][7] = dsB;
            j.pre[3][4] = -mu_off; j.post[3][4] = 1.0f;
            j.pre[3][5] = -e_off;  j.post[3][5] = 1.0f;
            j.pre[3][6] = -mu_off; j.post[3][6] = 1.0f;
            j.pre[3][7] = -e_off;  j.post[3][7] = 1.0f;
        }
    };
    r.constraint = [](const PROsineModel &, const Eigen::VectorXf &v) {
        float sinsq2thmumu = std::pow(10.0f, v(1));  // sin^2 2theta_mumu
        float sB = std::pow(10.0f, v(2));                   // ratio parameter
        float rad = 1.0f - sinsq2thmumu;
        float Um4sq = (1.0f - std::sqrt(rad)) / 2.0f;
        float Ue4sq = sB * (1.0f - Um4sq);     // from definition of sB
        return Um4sq + Ue4sq < 0.999 ? 1 :0;  // allowed
    };
    return r;
}

/// 3C (numu->nue appearance parameterisation) and its NC-disappearance extension.
/// The NC variant fits Uta4sq = |Utau4|^2 directly (log10), with
/// |Us4|^2 = max(0, 1 - |Ue4|^2 - |Um4|^2 - |Utau4|^2).
SineModelRecipe make_3p1_3C(bool with_nc) {
    SineModelRecipe r;
    r.tag = with_nc ? "SBL_3+1_sinsq2thmue_NC" : "SBL_3+1_sinsq2thmue";
    r.params = {
        {"dmsq", "#Deltam^{2}", "eV^{2}", true, -2.0f, 2.0f, -2.0f},
        {"sinsq2thmue", "sin^{2}2#theta_{#mue}", "", true, -kInf, -1e-3f, -8.0f},
        {"xi", "#xi", "", false, -10.0f, 10.0f, 0.0f},
    };
    r.channels = {{"null", ProbForm::Null}, {"mumu", ProbForm::Disappearance},
                  {"mue", ProbForm::Appearance}, {"ee", ProbForm::Disappearance}};
    r.grad_entries = {{1, 1}, {2, 1}, {1, 2}, {1, 3}, {2, 3}};
    r.grad_style = GradStyle::HoistedPhase;
    if(with_nc) {
        r.params.push_back({"Uta4sq", "|U_{#tau4}|^{2}", "", true, -kInf, -1e-3f, -8.0f});
        r.channels.push_back({"mus", ProbForm::Disappearance});
        r.channels.push_back({"es", ProbForm::Disappearance});
        r.channels.push_back({"mutau", ProbForm::Appearance});
        r.channels.push_back({"etau", ProbForm::Appearance});
        for(uint8_t c = 4; c <= 7; ++c)
            for(uint8_t p = 1; p <= 3; ++p)
                r.grad_entries.push_back({p, c});
    }
    r.amps = [with_nc](const PROsineModel &m, const Eigen::VectorXf &phys, SineAmps &a) {
        float dmsq = m.maybe_convert_log("dmsq", phys(0));
        float sinsq2thmue = m.maybe_convert_log("sinsq2thmue", phys(1));
        float xi = m.maybe_convert_log("xi", phys(2));
        float sqrtsin = std::sqrt(sinsq2thmue);
        float Um4sq=(std::exp(-xi) *sqrtsin ) / 2.0;
        float Ue4sq=(std::exp(xi) *sqrtsin) / 2.0;
        a.dmsq = dmsq;
        a.A[1] = 4.0f*(1-Um4sq)*Um4sq;
        a.A[2] = sinsq2thmue;
        a.A[3] = 4.0f*(1-Ue4sq)*Ue4sq;
        if(with_nc) {
            float Uta4sq = m.maybe_convert_log("Uta4sq", phys(3));
            float Us4sq = std::max(0.0f, 1.0f - Ue4sq - Um4sq - Uta4sq);
            a.A[4] = 4.0f*Um4sq*Us4sq;
            a.A[5] = 4.0f*Ue4sq*Us4sq;
            a.A[6] = 4.0f*Um4sq*Uta4sq;
            a.A[7] = 4.0f*Ue4sq*Uta4sq;
        }
    };
    r.jacs = [with_nc](const PROsineModel &m, const Eigen::VectorXf &phys, SineJacs &j) {
        float dmsq = m.maybe_convert_log("dmsq", phys(0));
        float smue = m.maybe_convert_log("sinsq2thmue", phys(1));
        float xi   = m.maybe_convert_log("xi", phys(2));
        float sqrtsin = std::sqrt(std::max(0.0f, smue));
        float Um4sq = std::exp(-xi) * sqrtsin / 2.0f;
        float Ue4sq = std::exp( xi) * sqrtsin / 2.0f;
        float ddm   = m.is_log10[0] ? LN10 * dmsq : 1.0f;
        float dsmue = m.is_log10[1] ? LN10 * smue : 1.0f;
        float dxi   = m.is_log10[2] ? LN10 * xi   : 1.0f;
        // d sqrt(smue) / d(internal smue): for log10 params LN10*sqrt/2 (finite at 0).
        float dsqrt = m.is_log10[1] ? LN10 * sqrtsin / 2.0f : (sqrtsin > 0.0f ? 1.0f / (2.0f * sqrtsin) : 0.0f);
        float dUm_dsmue = std::exp(-xi) * dsqrt / 2.0f, dUm_dxi = -Um4sq * dxi;
        float dUe_dsmue = std::exp( xi) * dsqrt / 2.0f, dUe_dxi =  Ue4sq * dxi;
        j.dmsq = dmsq;
        j.ddm = ddm;
        j.Cdm[1] = -4.0f * Um4sq * (1.0f - Um4sq);
        j.Cdm[2] = smue;
        j.Cdm[3] = -4.0f * Ue4sq * (1.0f - Ue4sq);
        j.pre[1][1] = -4.0f * (1.0f - 2.0f*Um4sq); j.post[1][1] = dUm_dsmue;
        j.pre[2][1] = -4.0f * (1.0f - 2.0f*Um4sq); j.post[2][1] = dUm_dxi;
        j.pre[1][2] = 1.0f;                        j.post[1][2] = dsmue;
        j.pre[1][3] = -4.0f * (1.0f - 2.0f*Ue4sq); j.post[1][3] = dUe_dsmue;
        j.pre[2][3] = -4.0f * (1.0f - 2.0f*Ue4sq); j.post[2][3] = dUe_dxi;
        if(with_nc) {
            float Uta4sq = m.maybe_convert_log("Uta4sq", phys(3));
            float dUta = m.is_log10[3] ? LN10 * Uta4sq : 1.0f;
            float Usraw = 1.0f - Ue4sq - Um4sq - Uta4sq;
            float Us4sq = std::max(0.0f, Usraw);
            // At the max() boundary |Us4|^2 is locally constant (clamp-zero convention).
            // The dU*_d* tails already carry their chain factors, so post stays 1.
            float dUs_dsmue = (Usraw > 0.0f ? -(dUe_dsmue + dUm_dsmue) : 0.0f);
            float dUs_dxi   = (Usraw > 0.0f ? -(dUe_dxi + dUm_dxi) : 0.0f);
            float dUs_dUta  = (Usraw > 0.0f ? -dUta : 0.0f);
            j.Cdm[4] = -4.0f * Um4sq * Us4sq;
            j.Cdm[5] = -4.0f * Ue4sq * Us4sq;
            j.Cdm[6] =  4.0f * Um4sq * Uta4sq;
            j.Cdm[7] =  4.0f * Ue4sq * Uta4sq;
            // ch 4 (mus, Dis): A4 = 4 Um Us
            j.pre[1][4] = -4.0f*(dUm_dsmue*Us4sq + Um4sq*dUs_dsmue); j.post[1][4] = 1.0f;
            j.pre[2][4] = -4.0f*(dUm_dxi*Us4sq + Um4sq*dUs_dxi);     j.post[2][4] = 1.0f;
            j.pre[3][4] = -4.0f*Um4sq*dUs_dUta;                      j.post[3][4] = 1.0f;
            // ch 5 (es, Dis): A5 = 4 Ue Us
            j.pre[1][5] = -4.0f*(dUe_dsmue*Us4sq + Ue4sq*dUs_dsmue); j.post[1][5] = 1.0f;
            j.pre[2][5] = -4.0f*(dUe_dxi*Us4sq + Ue4sq*dUs_dxi);     j.post[2][5] = 1.0f;
            j.pre[3][5] = -4.0f*Ue4sq*dUs_dUta;                      j.post[3][5] = 1.0f;
            // ch 6 (mutau, App): A6 = 4 Um Uta
            j.pre[1][6] = 4.0f*dUm_dsmue*Uta4sq;                     j.post[1][6] = 1.0f;
            j.pre[2][6] = 4.0f*dUm_dxi*Uta4sq;                       j.post[2][6] = 1.0f;
            j.pre[3][6] = 4.0f*Um4sq*dUta;                           j.post[3][6] = 1.0f;
            // ch 7 (etau, App): A7 = 4 Ue Uta
            j.pre[1][7] = 4.0f*dUe_dsmue*Uta4sq;                     j.post[1][7] = 1.0f;
            j.pre[2][7] = 4.0f*dUe_dxi*Uta4sq;                       j.post[2][7] = 1.0f;
            j.pre[3][7] = 4.0f*Ue4sq*dUta;                           j.post[3][7] = 1.0f;
        }
    };
    if(with_nc) {
        // Unitarity extended by the tau row: |Ue4|^2 + |Um4|^2 + |Utau4|^2 < 0.999,
        // with sqrt(s2mue) cosh(xi) = |Ue4|^2 + |Um4|^2.
        r.constraint = [](const PROsineModel &m, const Eigen::VectorXf &v) {
            const float sinsq2thmue = m.maybe_convert_log("sinsq2thmue", v(1));
            const float xi = m.maybe_convert_log("xi", v(2));
            const float Uta4sq = m.maybe_convert_log("Uta4sq", v(3));
            return (std::sqrt(std::max(0.0f, sinsq2thmue))*std::cosh(xi) + Uta4sq < 0.999f ? 1 : 0);
        };
    } else {
        r.constraint = [](const PROsineModel &m, const Eigen::VectorXf &v) {
            const float sinsq2thmue = m.maybe_convert_log("sinsq2thmue", v(1));
            const float xi = m.maybe_convert_log("xi", v(2));
            return   (std::sqrt(sinsq2thmue)*std::cosh(xi)<0.999 ? 1 : 0);
        };
    }
    return r;
}

/// SBL_2flav_nudis_NC (legacy NCdisapp): independent phenomenological NC-deficit
/// amplitudes for nu_mu and nu_e.
/// No joint unitarity constraint links the two amplitudes — intended for
/// limit-setting, not as an exact 3+1 sub-model.
SineModelRecipe make_ncdisapp() {
    SineModelRecipe r;
    r.tag = "SBL_2flav_nudis_NC";
    r.params = {
        {"dmsq", "#Deltam^{2}", "eV^{2}", true, -2.0f, 2.0f, -2.0f},
        {"sinsq2thms", "sin^{2}2#theta_{#mus}", "", true, -kInf, 0.0f, -10.0f},
        {"sinsq2thes", "sin^{2}2#theta_{es}", "", true, -kInf, 0.0f, -10.0f},
    };
    r.channels = {{"null", ProbForm::Null}, {"mus", ProbForm::Disappearance},
                  {"es", ProbForm::Disappearance}};
    r.grad_entries = {{1, 1}, {2, 2}};
    r.grad_style = GradStyle::HoistedPhase;
    r.amps = [](const PROsineModel &m, const Eigen::VectorXf &phys, SineAmps &a) {
        float dmsq = m.maybe_convert_log("dmsq", phys(0));
        float sms = m.maybe_convert_log("sinsq2thms", phys(1));
        float ses = m.maybe_convert_log("sinsq2thes", phys(2));
        if(sms > 1) sms = 1;
        if(sms < 0) sms = 0;
        if(ses > 1) ses = 1;
        if(ses < 0) ses = 0;
        a.dmsq = dmsq;
        a.A[1] = sms;
        a.A[2] = ses;
    };
    r.jacs = [](const PROsineModel &m, const Eigen::VectorXf &phys, SineJacs &j) {
        float dmsq = m.maybe_convert_log("dmsq", phys(0));
        float sms = m.maybe_convert_log("sinsq2thms", phys(1));
        float ses = m.maybe_convert_log("sinsq2thes", phys(2));
        float ddm  = m.is_log10[0] ? LN10 * dmsq : 1.0f;
        float dsms = m.is_log10[1] ? LN10 * sms : 1.0f;
        float dses = m.is_log10[2] ? LN10 * ses : 1.0f;
        if(sms > 1) { sms = 1; dsms = 0; }
        if(sms < 0) { sms = 0; dsms = 0; }
        if(ses > 1) { ses = 1; dses = 0; }
        if(ses < 0) { ses = 0; dses = 0; }
        j.dmsq = dmsq;
        j.ddm = ddm;
        j.Cdm[1] = -sms;
        j.Cdm[2] = -ses;
        j.pre[1][1] = -1.0f; j.post[1][1] = dsms;
        j.pre[2][2] = -1.0f; j.post[2][2] = dses;
    };
    return r;
}

}  // namespace

const SineModelRecipe *find_sine_recipe(const std::string &tag) {
    static const std::map<std::string, SineModelRecipe> registry = [] {
        std::map<std::string, SineModelRecipe> reg;
        auto add = [&reg](SineModelRecipe rec) {
            std::string t = rec.tag;
            reg.emplace(std::move(t), std::move(rec));
        };
        add(make_twoflav("SBL_2flav_numudis", "sinsq2thmm", "sin^{2}2#theta_{#mu#mu}", "mumu",
                         ProbForm::Disappearance, -kInf));
        add(make_twoflav("SBL_2flav_nueapp", "sinsq2thme", "sin^{2}2#theta_{#mue}", "mue",
                         ProbForm::Appearance, -10.0f));
        add(make_twoflav("SBL_2flav_nuedis", "sinsq2thee", "sin^{2}2#theta_{ee}", "ee",
                         ProbForm::Disappearance, -kInf));
        add(make_twoflav("SBL_2flav_numudis_NC", "sinsq2thms", "sin^{2}2#theta_{#mus}", "mus",
                         ProbForm::Disappearance, -kInf));
        add(make_ncdisapp());
        add(make_3p1(false));
        add(make_3p1_angles(false));
        add(make_3p1_3A(false));
        add(make_3p1_3B(false));
        add(make_3p1_3C(false));
        add(make_3p1(true));
        add(make_3p1_angles(true));
        add(make_3p1_3A(true));
        add(make_3p1_3B(true));
        add(make_3p1_3C(true));
        return reg;
    }();
    auto it = registry.find(tag);
    return it == registry.end() ? nullptr : &it->second;
}

}
