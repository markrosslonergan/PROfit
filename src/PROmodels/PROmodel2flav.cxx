/**
 * @file PROmodel2flav.cxx
 * @brief Implementation of the two-flavour-like single-channel 3+1 models.
 * @author PROfit Collaboration
 */
#include "PROmodels/PROmodel2flav.h"

namespace PROfit {

// ------------------------------------------------------------------
// PROnumudis
// ------------------------------------------------------------------

PROnumudis::PROnumudis(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {
    // model_functions is the non-unified version, these are optional
    // these get combined into one get_probs function in the constructor, but we can override this for faster computation
    model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0;});
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),le);});
    prob_types = {0, 1};

    if(parameter_map.find("L/E") == parameter_map.end()) {
        log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'.Make sure its in your model section of XML. ") % __func__ % __LINE__;
        throw std::runtime_error("Missing parameter: L/E");
    }
    ivars = {parameter_map.at("L/E")};

    build_hists_and_combined(prop);
    nparams = 2;
    param_names = {"dmsq", "sinsq2thmm"};
    pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mu#mu}"};
    pretty_param_units = {"eV^{2}", ""};
    is_log10 = {true, true};
    build_param_index();
    lb = Eigen::VectorXf(2);
    ub = Eigen::VectorXf(2);
    default_val = Eigen::VectorXf(2);
    lb << -2, -std::numeric_limits<float>::infinity();
    ub << 2, 0;
    default_val << -2, -10;

}

float PROnumudis::Pmumu(float dmsq, float sinsq2thmumu, float le) const{
    dmsq = maybe_convert_log("dmsq", dmsq);
    sinsq2thmumu = maybe_convert_log("sinsq2thmm", sinsq2thmumu);

    if(sinsq2thmumu > 1) {
        //log<LOG_ERROR>(L"%1% || sinsq2thmumu is %2% which is greater than 1. Setting to 1.")     % __func__ % sinsq2thmumu;
        sinsq2thmumu = 1;
    }
    if(sinsq2thmumu < 0) {
        log<LOG_ERROR>(L"%1% || sinsq2thmumu is %2% which is less than 0. Setting to 0.")
            % __func__ % sinsq2thmumu;
        sinsq2thmumu = 0;
    }

    float sinterm = std::sin(1.266932679f*dmsq*(le));
    float prob    = 1.0f - (sinsq2thmumu*sinterm*sinterm);

    if(prob<0.0 || prob >1.0 ){;//|| std::isnan(prob)){
        log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                       L"dmsq = %3%, sinsq2thmumu = %4%, L/E = %5%")
            % __func__ % prob % dmsq % sinsq2thmumu % le;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }

    return prob;
}

Eigen::MatrixXf PROnumudis::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    const auto &le_arr = var_arrs[0];
    //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;
    // Precompute physics parameters once
    float dmsq = maybe_convert_log("dmsq", phys(0));
    float sinsq2thmumu = maybe_convert_log("sinsq2thmm", phys(1));

    float freq = 1.266932679f * dmsq;

    if(sinsq2thmumu > 1) sinsq2thmumu = 1;
    if(sinsq2thmumu < 0) sinsq2thmumu = 0;

    Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

    for(size_t i = 0; i < le_arr.size(); ++i) {

        // no oscillation
        probs(i, 0) = 1.0f;

        // P_mumu
        float sinterm = std::sin(freq * le_arr[i]);
        probs(i, 1) = 1.0f - (sinsq2thmumu * sinterm * sinterm);
    }

    return probs;
}

std::vector<Eigen::MatrixXf> PROnumudis::get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    // Model:  P_mumu(Δm², s; L/E) = 1 − s · sin²x,   x = k · Δm² · (L/E),  s = sin²2θ_μμ
    // (column 0 is the constant "no oscillation" probability, derivative 0).
    // Derivatives w.r.t. the physical parameters:
    //   ∂P/∂Δm² = −s · d(sin²x)/dx · dx/dΔm² = −s · sin(2x) · k · (L/E)
    //   ∂P/∂s   = −sin²x
    // Returned w.r.t. the internal parameters by multiplying with the chain
    // factors ddm = dΔm²/dθ₀ and dss = ds/dθ₁ (see PROmodel::get_probs_grad).
    const auto &le_arr = var_arrs[0];
    float dmsq = maybe_convert_log("dmsq", phys(0));
    float sinsq2thmumu = maybe_convert_log("sinsq2thmm", phys(1));

    // Chain factors d(linear)/d(internal); log10 params: d(10^x)/dx = ln10 * 10^x.
    constexpr float LN10 = 2.302585093f;
    float ddm = is_log10[0] ? LN10 * dmsq : 1.0f;
    float dss = is_log10[1] ? LN10 * sinsq2thmumu : 1.0f;
    // Match get_probs' clamp: the clamped parameter has zero local sensitivity.
    if(sinsq2thmumu > 1) { sinsq2thmumu = 1; dss = 0; }
    if(sinsq2thmumu < 0) { sinsq2thmumu = 0; dss = 0; }

    float freq = 1.266932679f * dmsq;
    std::vector<Eigen::MatrixXf> grads(2, Eigen::MatrixXf::Zero(le_arr.size(), model_functions.size()));
    for(size_t i = 0; i < le_arr.size(); ++i) {
        float x = freq * le_arr[i];
        float sinterm = std::sin(x);
        grads[0](i, 1) = -sinsq2thmumu * std::sin(2.0f*x) * 1.266932679f * le_arr[i] * ddm;  // ∂P_mumu/∂θ₀ (Δm²)
        grads[1](i, 1) = -sinterm * sinterm * dss;                                          // ∂P_mumu/∂θ₁ (s)
    }
    return grads;
}

// ------------------------------------------------------------------
// PROnumudisTEST
// ------------------------------------------------------------------

PROnumudisTEST::PROnumudisTEST(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {
    prob_types = {0, 1};
    model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0f;});
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),le);});

    if(parameter_map.find("L") == parameter_map.end()) {
        log<LOG_ERROR>(L"%1%, %2% || PROnumudisTEST: Missing expected parameter: 'L'.") % __func__ % __LINE__;
        throw std::runtime_error("Missing parameter: L");
    }
    if(parameter_map.find("E") == parameter_map.end()) {
        log<LOG_ERROR>(L"%1%, %2% || PROnumudisTEST: Missing expected parameter: 'E'.") % __func__ % __LINE__;
        throw std::runtime_error("Missing parameter: E");
    }
    // ivars[0] = L variable index, ivars[1] = E variable index.
    // build_hists_and_combined will make the flat grid L x E.
    ivars = {parameter_map.at("L"), parameter_map.at("E")};

    build_hists_and_combined(prop);

    nparams = 2;
    param_names = {"dmsq", "sinsq2thmm"};
    pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mu#mu}"};
    pretty_param_units = {"eV^{2}", ""};
    is_log10 = {true, true};
    build_param_index();
    lb = Eigen::VectorXf(2);
    ub = Eigen::VectorXf(2);
    default_val = Eigen::VectorXf(2);
    lb << -2, -std::numeric_limits<float>::infinity();
    ub << 2, 0;
    default_val << -2, -10;
}

float PROnumudisTEST::Pmumu(float dmsq, float sinsq2thmumu, float le) const {
    dmsq         = maybe_convert_log("dmsq",       dmsq);
    sinsq2thmumu = maybe_convert_log("sinsq2thmm", sinsq2thmumu);
    if(sinsq2thmumu > 1) sinsq2thmumu = 1;
    if(sinsq2thmumu < 0) sinsq2thmumu = 0;
    float sinterm = std::sin(1.266932679f * dmsq * le);
    float prob    = 1.0f - (sinsq2thmumu * sinterm * sinterm);
    if(prob < 0.0f || prob > 1.0f) {
        log<LOG_ERROR>(L"%1% || Probability %2% outside [0,1]. dmsq=%3%, sinsq2thmumu=%4%, L/E=%5%")
            % __func__ % prob % dmsq % sinsq2thmumu % le;
        exit(EXIT_FAILURE);
    }
    return prob;
}

Eigen::MatrixXf PROnumudisTEST::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    float dmsq         = maybe_convert_log("dmsq",       phys(0));
    float sinsq2thmumu = maybe_convert_log("sinsq2thmm", phys(1));
    if(sinsq2thmumu > 1) sinsq2thmumu = 1;
    if(sinsq2thmumu < 0) sinsq2thmumu = 0;

    float freq = 1.266932679f * dmsq;
    const size_t n_flat = var_arrs[0].size(); // = n_L * n_E
    Eigen::MatrixXf probs(n_flat, 2);

    for(size_t i = 0; i < n_flat; ++i) {
        float L = var_arrs[0][i];
        float E = var_arrs[1][i];
        // Guard against zero energy — same convention as L/E variable (out-of-range events
        // get bin index -1 in PROpeller so they never enter H; but be safe here too).
        float le = (E > 0.0f) ? L / E : 0.0f;
        probs(i, 0) = 1.0f;
        float sinterm = std::sin(freq * le);
        probs(i, 1) = 1.0f - (sinsq2thmumu * sinterm * sinterm);
    }
    return probs;
}

std::vector<Eigen::MatrixXf> PROnumudisTEST::get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    // Same physics and derivatives as PROnumudis::get_probs_grad (P_mumu = 1 − s·sin²x),
    // evaluated on the flat L × E grid with L/E formed per grid point.
    float dmsq         = maybe_convert_log("dmsq",       phys(0));
    float sinsq2thmumu = maybe_convert_log("sinsq2thmm", phys(1));
    constexpr float LN10 = 2.302585093f;
    float ddm = is_log10[0] ? LN10 * dmsq : 1.0f;
    float dss = is_log10[1] ? LN10 * sinsq2thmumu : 1.0f;
    if(sinsq2thmumu > 1) { sinsq2thmumu = 1; dss = 0; }
    if(sinsq2thmumu < 0) { sinsq2thmumu = 0; dss = 0; }

    float freq = 1.266932679f * dmsq;
    const size_t n_flat = var_arrs[0].size();
    std::vector<Eigen::MatrixXf> grads(2, Eigen::MatrixXf::Zero(n_flat, 2));
    for(size_t i = 0; i < n_flat; ++i) {
        float L = var_arrs[0][i];
        float E = var_arrs[1][i];
        float le = (E > 0.0f) ? L / E : 0.0f;
        float x = freq * le;
        float sinterm = std::sin(x);
        grads[0](i, 1) = -sinsq2thmumu * std::sin(2.0f*x) * 1.266932679f * le * ddm;
        grads[1](i, 1) = -sinterm * sinterm * dss;
    }
    return grads;
}

// ------------------------------------------------------------------
// PROnueapp
// ------------------------------------------------------------------

PROnueapp::PROnueapp(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {
    model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0;});
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),le);});
    prob_types = {0, 1};
    if(parameter_map.find("L/E") == parameter_map.end()) {
        log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
        throw std::runtime_error("Missing parameter: L/E");
    }
    ivars = {parameter_map.at("L/E")};

    build_hists_and_combined(prop);
     nparams = 2;
    param_names = {"dmsq", "sinsq2thme"};
    pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mue}"};
    pretty_param_units = {"eV^{2}", ""};
    is_log10 = {true, true};
    build_param_index();
    lb = Eigen::VectorXf(2);
    ub = Eigen::VectorXf(2);
    default_val = Eigen::VectorXf(2);
    lb << -2, -10; //-std::numeric_limits<float>::infinity();
    ub << 2, 0;
    //default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
    default_val << -2, -10; //std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();

    log<LOG_INFO>(L"%1% || setting up a model nueapp, with  %2% params.")     % __func__ % nparams;
    for(size_t i=0; i< nparams;i++){
        log<LOG_INFO>(L"%1% || Param %2% is %3% with lower bound/upper bound of %4%/%5% and default %6%")     % __func__ % i % param_names[i].c_str() % lb[i] % ub[i] % default_val[i];
    }

}

float PROnueapp::Pmue(float dmsq, float sinsq2thmue, float le) const{
    dmsq = maybe_convert_log("dmsq", dmsq);
    sinsq2thmue = maybe_convert_log("sinsq2thme", sinsq2thmue);

    if(sinsq2thmue > 1) {
        //log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is greater than 1. Setting to 1.")  % __func__ % sinsq2thmue;
        sinsq2thmue = 1;
    }
    if(sinsq2thmue < 0) {
        log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is less than 0. Setting to 0.")
            % __func__ % sinsq2thmue;
        sinsq2thmue = 0;
    }

    float sinterm = std::sin(1.266932679f*dmsq*(le));
    float prob    = sinsq2thmue*sinterm*sinterm;

    if(prob<0.0 || prob >1.0){
        log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                       L"dmsq = %3%, sinsq2thmue = %4%, L/E = %5%")
            % __func__ % prob % dmsq % sinsq2thmue % le;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }

    return prob;
}

Eigen::MatrixXf PROnueapp::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    const auto &le_arr = var_arrs[0];
    //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;
    // Precompute physics parameters once
    float dmsq = maybe_convert_log("dmsq", phys(0));
    float sinsq2thmue = maybe_convert_log("sinsq2thme", phys(1));

    float freq = 1.266932679f * dmsq;

    if(sinsq2thmue > 1) sinsq2thmue = 1;
    if(sinsq2thmue < 0) sinsq2thmue = 0;

    Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

    for(size_t i = 0; i < le_arr.size(); ++i) {

        // no oscillation
        probs(i, 0) = 1.0f;

        // P_mumu
        float sinterm = std::sin(freq * le_arr[i]);
        probs(i, 1) = (sinsq2thmue * sinterm * sinterm);
    }

    return probs;
}

std::vector<Eigen::MatrixXf> PROnueapp::get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    // Model:  P_mue(Δm², s; L/E) = s · sin²x,   x = k · Δm² · (L/E),  s = sin²2θ_μe
    //   ∂P/∂Δm² = s · sin(2x) · k · (L/E),   ∂P/∂s = sin²x      (then × chain factors)
    const auto &le_arr = var_arrs[0];
    float dmsq = maybe_convert_log("dmsq", phys(0));
    float sinsq2thmue = maybe_convert_log("sinsq2thme", phys(1));

    constexpr float LN10 = 2.302585093f;
    float ddm = is_log10[0] ? LN10 * dmsq : 1.0f;
    float dss = is_log10[1] ? LN10 * sinsq2thmue : 1.0f;
    if(sinsq2thmue > 1) { sinsq2thmue = 1; dss = 0; }
    if(sinsq2thmue < 0) { sinsq2thmue = 0; dss = 0; }

    float freq = 1.266932679f * dmsq;
    std::vector<Eigen::MatrixXf> grads(2, Eigen::MatrixXf::Zero(le_arr.size(), model_functions.size()));
    for(size_t i = 0; i < le_arr.size(); ++i) {
        float x = freq * le_arr[i];
        float sinterm = std::sin(x);
        grads[0](i, 1) = sinsq2thmue * std::sin(2.0f*x) * 1.266932679f * le_arr[i] * ddm;  // ∂P_mue/∂θ₀ (Δm²)
        grads[1](i, 1) = sinterm * sinterm * dss;                                          // ∂P_mue/∂θ₁ (s)
    }
    return grads;
}

// ------------------------------------------------------------------
// PROnuedis
// ------------------------------------------------------------------

PROnuedis::PROnuedis(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {
    model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0;});
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),le);});
    prob_types = {0, 1};

    if(parameter_map.find("L/E") == parameter_map.end()) {
        log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'.Make sure its in your model section of XML. ") % __func__ % __LINE__;
        throw std::runtime_error("Missing parameter: L/E");
    }
    ivars = {parameter_map.at("L/E")};

    build_hists_and_combined(prop);
    nparams = 2;
    param_names = {"dmsq", "sinsq2thee"};
    pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{ee}"};
    pretty_param_units = {"eV^{2}", ""};
    is_log10 = {true, true};
    build_param_index();
    lb = Eigen::VectorXf(2);
    ub = Eigen::VectorXf(2);
    default_val = Eigen::VectorXf(2);
    lb << -2, -std::numeric_limits<float>::infinity();
    ub << 2, 0;
    default_val << -2, -10;

}

float PROnuedis::Pee(float dmsq, float sinsq2thee, float le) const{
    dmsq = maybe_convert_log("dmsq", dmsq);
    sinsq2thee = maybe_convert_log("sinsq2thee", sinsq2thee);

    if(sinsq2thee > 1) {
        //log<LOG_ERROR>(L"%1% || sinsq2thee is %2% which is greater than 1. Setting to 1.")     % __func__ % sinsq2thee;
        sinsq2thee = 1;
    }
    if(sinsq2thee < 0) {
        log<LOG_ERROR>(L"%1% || sinsq2thee is %2% which is less than 0. Setting to 0.")
            % __func__ % sinsq2thee;
        sinsq2thee = 0;
    }

    float sinterm = std::sin(1.266932679f*dmsq*(le));
    float prob    = 1.0f - (sinsq2thee*sinterm*sinterm);

    if(prob<0.0 || prob >1.0 ){;//|| std::isnan(prob)){
        log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                       L"dmsq = %3%, sinsq2thee = %4%, L/E = %5%")
            % __func__ % prob % dmsq % sinsq2thee % le;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }

    return prob;
}

Eigen::MatrixXf PROnuedis::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    const auto &le_arr = var_arrs[0];
    //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;
    // Precompute physics parameters once
    float dmsq = maybe_convert_log("dmsq", phys(0));
    float sinsq2thee = maybe_convert_log("sinsq2thee", phys(1));

    float freq = 1.266932679f * dmsq;

    if(sinsq2thee > 1) sinsq2thee = 1;
    if(sinsq2thee < 0) sinsq2thee = 0;

    Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

    for(size_t i = 0; i < le_arr.size(); ++i) {

        // no oscillation
        probs(i, 0) = 1.0f;

        // P_mumu
        float sinterm = std::sin(freq * le_arr[i]);
        probs(i, 1) = 1.0f-(sinsq2thee * sinterm * sinterm);
    }

    return probs;
}

std::vector<Eigen::MatrixXf> PROnuedis::get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    // Model:  P_ee(Δm², s; L/E) = 1 − s · sin²x,   x = k · Δm² · (L/E),  s = sin²2θ_ee
    //   ∂P/∂Δm² = −s · sin(2x) · k · (L/E),   ∂P/∂s = −sin²x     (then × chain factors)
    const auto &le_arr = var_arrs[0];
    float dmsq = maybe_convert_log("dmsq", phys(0));
    float sinsq2thee = maybe_convert_log("sinsq2thee", phys(1));

    constexpr float LN10 = 2.302585093f;
    float ddm = is_log10[0] ? LN10 * dmsq : 1.0f;
    float dss = is_log10[1] ? LN10 * sinsq2thee : 1.0f;
    if(sinsq2thee > 1) { sinsq2thee = 1; dss = 0; }
    if(sinsq2thee < 0) { sinsq2thee = 0; dss = 0; }

    float freq = 1.266932679f * dmsq;
    std::vector<Eigen::MatrixXf> grads(2, Eigen::MatrixXf::Zero(le_arr.size(), model_functions.size()));
    for(size_t i = 0; i < le_arr.size(); ++i) {
        float x = freq * le_arr[i];
        float sinterm = std::sin(x);
        grads[0](i, 1) = -sinsq2thee * std::sin(2.0f*x) * 1.266932679f * le_arr[i] * ddm;  // ∂P_ee/∂θ₀ (Δm²)
        grads[1](i, 1) = -sinterm * sinterm * dss;                                         // ∂P_ee/∂θ₁ (s)
    }
    return grads;
}

}
