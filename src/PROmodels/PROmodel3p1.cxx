/**
 * @file PROmodel3p1.cxx
 * @brief Implementation of the 3+1 sterile-neutrino model family.
 * @author PROfit Collaboration
 */
#include "PROmodels/PROmodel3p1.h"

namespace PROfit {

// ------------------------------------------------------------------
// PRO3p1
// ------------------------------------------------------------------

PRO3p1::PRO3p1(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

    model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0; });
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),v(2),le); });
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),v(2),le); });
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),v(2),le); });
    prob_types = {0, 1, 2, 3};
    if(parameter_map.find("L/E") == parameter_map.end()) {
        log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
        throw std::runtime_error("Missing parameter: L/E");
    }
    ivars = {parameter_map.at("L/E")};

    //constraints
    model_constraint = [this](const Eigen::VectorXf &v){return this->UnitarityConstraint(v);};


    build_hists_and_combined(prop);


    nparams = 3;
    param_names = {"dmsq", "Ue4^2", "Um4^2"};
    pretty_param_names = {"#Deltam^{2}", "|U_{e4}|^{2}", "|U_{#mu4}|^{2}"};
    pretty_param_units = {"eV^{2}", "",""};
    is_log10 = {true, true, true};
    build_param_index();
    lb = Eigen::VectorXf(3);
    ub = Eigen::VectorXf(3);
    default_val = Eigen::VectorXf(3);
    lb << -2, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
    ub << 2, -1e-4, -1e-4;
    //default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
    default_val << -2, -8, -8;
}

int PRO3p1::UnitarityConstraint(const Eigen::VectorXf &v){
    const float Ue4sq = maybe_convert_log("Ue4^2", v(param_name_to_index.at("Ue4^2")));
    const float Um4sq = maybe_convert_log("Um4^2", v(param_name_to_index.at("Um4^2")));
    return   ((Ue4sq+Um4sq)<1 ? 1 : 0);
}

float PRO3p1::Pmue(float dmsq, float Ue4sq, float Um4sq, float le) const{
    dmsq =  maybe_convert_log("dmsq", dmsq);
    Ue4sq = maybe_convert_log("Ue4^2", Ue4sq);
    Um4sq = maybe_convert_log("Um4^2", Um4sq);

    if(Ue4sq > 1) {
        log<LOG_ERROR>(L"%1% || Ue4sq is %2% which is greater than 1. Setting to 1.")
            % __func__ % Ue4sq;
        Ue4sq = 1;
    }
    if(Ue4sq < 0) {
        log<LOG_ERROR>(L"%1% || Ue4sq is %2% which is less than 0. Setting to 0.")
            % __func__ % Ue4sq;
        Ue4sq = 0;
    }
    if(Um4sq > 1) {
        log<LOG_ERROR>(L"%1% || Um4sq is %2% which is greater than 1. Setting to 1.")
            % __func__ % Um4sq;
        Um4sq = 1;
    }
    if(Um4sq < 0) {
        log<LOG_ERROR>(L"%1% || Um4sq is %2% which is less than 0. Setting to 0.")
            % __func__ % Um4sq;
        Um4sq = 0;
    }

    float sinterm = std::sin(1.266932679f*dmsq*(le));
    float prob    = 4.0f*Ue4sq*Um4sq*sinterm*sinterm;

    if(prob<0.0 || prob >1.0){
        log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                       L"dmsq = %3%, Ue4sq = %4%, Um4sq = %5%, L/E = %6%")
            % __func__ % prob % dmsq % Ue4sq % Um4sq % le;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }

    return prob;
}

float PRO3p1::Pmumu(float dmsq, [[maybe_unused]]float Ue4sq, float Um4sq, float le) const{
    dmsq = maybe_convert_log("dmsq", dmsq);
    Um4sq = maybe_convert_log("Um4^2", Um4sq);

    if(Um4sq > 1) {
        log<LOG_ERROR>(L"%1% || Um4sq is %2% which is greater than 1. Setting to 1.")
            % __func__ % Um4sq;
        Um4sq = 1;
    }
    if(Um4sq < 0) {
        log<LOG_ERROR>(L"%1% || Um4sq is %2% which is less than 0. Setting to 0.")
            % __func__ % Um4sq;
        Um4sq = 0;
    }

    float sinterm = std::sin(1.266932679f*dmsq*(le));
    float prob    = 1.0f - 4.0f*Um4sq*(1.0f-Um4sq)*sinterm*sinterm;

    if(prob<0.0 || prob >1.0){
        log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Um4sq = %4%, L/E = %5%") % __func__ % prob % dmsq % Um4sq % le;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }

    return prob;
}

float PRO3p1::Pee(float dmsq, float Ue4sq, [[maybe_unused]]float Um4sq, float le) const{
    dmsq = maybe_convert_log("dmsq", dmsq);
    Ue4sq = maybe_convert_log("Ue4^2", Ue4sq);

    if(Ue4sq > 1) {
        log<LOG_ERROR>(L"%1% || Ue4sq is %2% which is greater than 1. Setting to 1.")
            % __func__ % Ue4sq;
        Ue4sq = 1;
    }
    if(Ue4sq < 0) {
        log<LOG_ERROR>(L"%1% || Ue4sq is %2% which is less than 0. Setting to 0.")
            % __func__ % Ue4sq;
        Ue4sq = 0;
    }

    float sinterm = std::sin(1.266932679f*dmsq*(le));
    float prob    = 1.0f - 4.0f*Ue4sq*(1.0f-Ue4sq)*sinterm*sinterm;

    if(prob<0.0 || prob >1.0){
        log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Ue4sq = %4%, L/E = %5%") % __func__ % prob % dmsq % Ue4sq % le;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }

    return prob;
}

Eigen::MatrixXf PRO3p1::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    const auto &le_arr = var_arrs[0];
    //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;

    // Precompute physics parameters once
    float dmsq = maybe_convert_log("dmsq", phys(0));
    float Ue4sq = maybe_convert_log("Ue4^2", phys(1));
    float Um4sq = maybe_convert_log("Um4^2", phys(2));

    Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

    for(size_t i = 0; i < le_arr.size(); ++i) {

        float sinterm = std::sin(1.266932679f*dmsq*(le_arr[i]));

        // no oscillation
        probs(i, 0) = 1.0f;


        // P_mumu
        probs(i, 1) = 1.0f - 4.0f*Um4sq*(1.0f-Um4sq)*sinterm*sinterm;

        // P_mue
        probs(i, 2) = 4.0f*Ue4sq*Um4sq*sinterm*sinterm;

        // P_ee
        probs(i, 3) = 1.0f - 4.0f*Ue4sq*(1.0f-Ue4sq)*sinterm*sinterm;

    }

    return probs;
}

// ------------------------------------------------------------------
// PRO3p1_angles
// ------------------------------------------------------------------

PRO3p1_angles::PRO3p1_angles(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

    model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0; });
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),v(2),le); });
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),v(2),le); });
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),v(2),le); });
    prob_types = {0, 1, 2, 3};
    if(parameter_map.find("L/E") == parameter_map.end()) {
        log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
        throw std::runtime_error("Missing parameter: L/E");
    }
    ivars = {parameter_map.at("L/E")};

    //constraints
    model_constraint = [this](const Eigen::VectorXf &v){return this->UnitarityConstraint(v);};


    build_hists_and_combined(prop);


    nparams = 3;
    param_names = {"dmsq", "sinsq2th14", "sinsqth24"};
    pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{14}", "sin^{2}#theta_{24}"};
    pretty_param_units = {"eV^{2}", "",""};
    is_log10 = {true, true, true};
    build_param_index();
    lb = Eigen::VectorXf(3);
    ub = Eigen::VectorXf(3);
    default_val = Eigen::VectorXf(3);
    lb << -2, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
    ub << 2, -1e-4, -1e-4;
    //default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
    default_val << -2, -8, -8;
}

int PRO3p1_angles::UnitarityConstraint(const Eigen::VectorXf &){
    return   1;
}

float PRO3p1_angles::Pmue(float dmsq, float sinsq2th14, float sinsqth24, float le) const{
    dmsq =  maybe_convert_log("dmsq", dmsq);
    float s214 = maybe_convert_log("sinsq2th14", sinsq2th14);
    float s24 = maybe_convert_log("sinsqth24", sinsqth24);

    float sinterm = std::sin(1.266932679f*dmsq*(le));
    float prob    = s214*s24*sinterm*sinterm;

    if(prob<0.0 || prob >1.0){
        log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                       L"dmsq = %3%, s214 = %4%, s24 = %5%, L/E = %6%")
            % __func__ % prob % dmsq % s214 % s24 % le;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }

    return prob;
}

float PRO3p1_angles::Pmumu(float dmsq, float sinsq2th14, float sinsqth24, float le) const{
    dmsq =  maybe_convert_log("dmsq", dmsq);
    float s214 = maybe_convert_log("sinsq2th14", sinsq2th14);
    float s24 = maybe_convert_log("sinsqth24", sinsqth24);
    float c14 = (1.0+sqrt(1.0-s214))/2.0f;

    float sinterm = std::sin(1.266932679f*dmsq*(le));
    float prob    = 1.0f - 4.0f*(c14*s24*(1.0f-c14*s24))*sinterm*sinterm;


    if(prob<0.0 || prob >1.0){
        log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                       L"dmsq = %3%, s214 = %4%, s24 = %5%, L/E = %6%")
            % __func__ % prob % dmsq % s214 % s24 % le;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }
    return prob;
}

float PRO3p1_angles::Pee(float dmsq, float sinsq2th14, [[maybe_unused]]float sinsqth24, float le) const{
    dmsq = maybe_convert_log("dmsq", dmsq);
    float s214 = maybe_convert_log("sinsq2th14", sinsq2th14);

    float sinterm = std::sin(1.266932679f*dmsq*(le));
    float prob    = 1.0f - s214*sinterm*sinterm;

    if(prob<0.0 || prob >1.0){
        log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                       L"dmsq = %3%, s214 = %4%,  L/E = %5%")
            % __func__ % prob % dmsq % s214   % le;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }
    return prob;
}

Eigen::MatrixXf PRO3p1_angles::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    const auto &le_arr = var_arrs[0];
    //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;

    // Precompute physics parameters once
    float dmsq = maybe_convert_log("dmsq", phys(0));
    float s214 = maybe_convert_log("sinsq2th14", phys(1));
    float s24 = maybe_convert_log("sinsqth24", phys(2));
    float c14 = (1.0+sqrt(1.0-s214))/2.0f;

    Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

    for(size_t i = 0; i < le_arr.size(); ++i) {

        float sinterm = std::sin(1.266932679f*dmsq*(le_arr[i]));

        // no oscillation
        probs(i, 0) = 1.0f;

        // P_mumu
        probs(i, 1) = 1.0f - 4.0f*c14*s24*(1.0f-c14*s24)*sinterm*sinterm;

        // P_mue
        probs(i, 2) = s214*s24*sinterm*sinterm;

        // P_ee
        probs(i, 3) = 1.0f - s214*sinterm*sinterm;

    }

    return probs;
}

// ------------------------------------------------------------------
// PRO3p1_3A
// ------------------------------------------------------------------

PRO3p1_3A::PRO3p1_3A(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

    model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0; });
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),v(2),le); });
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),v(2),le); });
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),v(2),le); });
    prob_types = {0, 1, 2, 3};
    if(parameter_map.find("L/E") == parameter_map.end()) {
        log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
        throw std::runtime_error("Missing parameter: L/E");
    }
    ivars = {parameter_map.at("L/E")};

    //constraints
    model_constraint = [](const Eigen::VectorXf &){return 1;};


     build_hists_and_combined(prop);


    nparams = 3;
    param_names = {"dmsq", "sinsq2thee", "sinsqth24"};
    pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{ee}", "sin^{2}#theta_{24}"};
    pretty_param_units = {"eV^{2}", "",""};
    is_log10 = {true, true, true};
    build_param_index();
    lb = Eigen::VectorXf(3);
    ub = Eigen::VectorXf(3);
    default_val = Eigen::VectorXf(3);
    lb << -2, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
    ub << 2, -1e-3, -1e-3;
    //default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
    default_val << -2, -8, -8;
}

int PRO3p1_3A::UnitarityConstraint(const Eigen::VectorXf &){
    return   1;
}

float PRO3p1_3A::Pmue(float dmsq, float sinsq2thee, [[maybe_unused]]float sinsqth24, float le) const{
    dmsq = maybe_convert_log("dmsq",dmsq);
    float sinsq2thmue = maybe_convert_log("sinsqth24",sinsqth24)*maybe_convert_log("sinsq2thee", sinsq2thee);

    if(sinsq2thmue > 1) {
        log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is greater than 1. Setting to 1.")
            % __func__ % sinsq2thmue;
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

float PRO3p1_3A::Pmumu(float dmsq, float sinsq2thee, [[maybe_unused]]float sinsqth24, float le) const{
    float Um4sq = maybe_convert_log("sinsqth24",sinsqth24)/2.0*(1.0+sqrt(1- maybe_convert_log("sinsq2thee", sinsq2thee)));
    dmsq =maybe_convert_log("dmsq",dmsq);


    float sinterm = std::sin(1.266932679f*dmsq*(le));
    float prob    = 1.0f - 4.0f*Um4sq*(1.0f-Um4sq)*sinterm*sinterm;

    if(prob<0.0 || prob >1.0){
        log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Um4sq = %4%, sinsq2thee = %5%, sinsqth24 = %6%, L/E = %7%") % __func__ % prob % dmsq % Um4sq % sinsq2thee % sinsqth24 % le;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }

    return prob;
}

float PRO3p1_3A::Pee(float dmsq, float sinsq2thee, [[maybe_unused]]float sinsqth24, float le) const{

    dmsq =maybe_convert_log("dmsq",dmsq);
    sinsq2thee =maybe_convert_log("sinsq2thee",sinsq2thee);

    float sinterm = std::sin(1.266932679f*dmsq*(le));
    float prob    = 1.0f - sinsq2thee*sinterm*sinterm;

    if(prob<0.0 || prob >1.0){
        log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%,sinsq2thee = %4%, sinsqth24 = %5%, L/E = %6%") % __func__ % prob % dmsq % sinsq2thee % sinsqth24 % le;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }
    return prob;
}

Eigen::MatrixXf PRO3p1_3A::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    const auto &le_arr = var_arrs[0];
    //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;

    // Precompute physics parameters once
    float dmsq = maybe_convert_log("dmsq", phys(0));
    float sinsq2thee = maybe_convert_log("sinsq2thee", phys(1));
    float sinsqth24 = maybe_convert_log("sinsqth24", phys(2));

    float Um4sq = sinsqth24/2.0*(1.0+sqrt(1.0f- sinsq2thee));
    float sinsq2thmue = sinsqth24*sinsq2thee;

    Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

    for(size_t i = 0; i < le_arr.size(); ++i) {

        float sinterm = std::sin(1.266932679f*dmsq*(le_arr[i]));

        // no oscillation
        probs(i, 0) = 1.0f;

        // P_mumu
        probs(i, 1) =  1.0f - 4.0f*Um4sq*(1.0f-Um4sq)*sinterm*sinterm;

        // P_mue
        probs(i, 2) = sinsq2thmue*sinterm*sinterm;

        // P_ee
        probs(i, 3) = 1.0f - sinsq2thee*sinterm*sinterm;

    }

    return probs;
}

// ------------------------------------------------------------------
// PRO3p1_3B
// ------------------------------------------------------------------

PRO3p1_3B::PRO3p1_3B(const PROpeller &prop,
          const std::map<std::string,int> &parameter_map) {

    // -----------------------------------------
    // 1) Model functions:
    //    0: constant (unosc / NC-like)
    //    1: Pmumu
    //    2: Pmue
    //    3: Pee
    // -----------------------------------------
    model_functions.push_back(
        [this]([[maybe_unused]] const Eigen::VectorXf &v, float) {
            (void)this;
            return 1.0f;
        }
    );
    model_functions.push_back(
        [this](const Eigen::VectorXf &v, float le) {
            return this->Pmumu(v(0), v(1), v(2), le);
        }
    );
    model_functions.push_back(
        [this](const Eigen::VectorXf &v, float le) {
            return this->Pmue(v(0), v(1), v(2), le);
        }
    );
    model_functions.push_back(
        [this](const Eigen::VectorXf &v, float le) {
            return this->Pee(v(0), v(1), v(2), le);
        }
    );
    prob_types = {0, 1, 2, 3};

    // -----------------------------------------
    // 2) L/E variable index
    // -----------------------------------------
    if (parameter_map.find("L/E") == parameter_map.end()) {
        log<LOG_ERROR>(
            L"%1%, %2% || Missing expected parameter: 'L/E'. "
            L"Make sure it's in your model section of the XML."
        ) % __func__ % __LINE__;
        throw std::runtime_error("Missing parameter: L/E");
    }
    ivars = {parameter_map.at("L/E")};

    // -----------------------------------------
    // 3) Build histograms for each model component
    // -----------------------------------------
    build_hists_and_combined(prop);

    // -----------------------------------------
    // 4) Parameters and bounds
    // v(0) = dmsq_log      = log10(Δm²_41)
    // v(1) = s2mumu_log    = log10(sin²2θ_μμ)
    // v(2) = sB  (can be thought of as sinsqth24prime = log10(sin²θ_24′))
    // -----------------------------------------
    nparams = 3;
    param_names        = {"dmsq", "sinsq2thmumu", "sB"};
    pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mu#mu}", "sB"};
    pretty_param_units = {"eV^{2}", "",""};
    is_log10         = {true, true, true};
    build_param_index();

    lb          = Eigen::VectorXf(3);
    ub          = Eigen::VectorXf(3);
    default_val = Eigen::VectorXf(3);
    lb << -2.0f, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
    ub <<  2.0f, -1e-3f, -1e-3 ;
    // Some reasonable defaults
    default_val << -2.0f, -8.0f, -8.0f;

    // -----------------------------------------
    // 5) Unitarity / physicality constraint
    // -----------------------------------------
    model_constraint = [this](const Eigen::VectorXf &v) {
        return this->UnitarityConstraint(v);
    };
}

// ---------------------------------------------
// Unitarity / physicality constraint:
// 0 ≤ sB ≤ 1 and Ue4² + Uμ4² < 1
// ---------------------------------------------
int PRO3p1_3B::UnitarityConstraint(const Eigen::VectorXf &v) {
    float sinsq2thmumu = std::pow(10.0f, v(1));  // sin²2θμμ
    float sB = std::pow(10.0f, v(2));                   // ratio parameter

    float rad = 1.0f - sinsq2thmumu;
    float Um4sq = (1.0f - std::sqrt(rad)) / 2.0f;
    float Ue4sq = sB * (1.0f - Um4sq);     // from definition of sB

    return Um4sq + Ue4sq < 0.999 ? 1 :0;  // allowed
}

// ---------------------------------------------
// νμ → νμ disappearance
// ---------------------------------------------
float PRO3p1_3B::Pmumu(float dmsq, float sinsq2thmumu, [[maybe_unused]] float sinsqth24prime, float le) const {
    dmsq   = std::pow(10.0f, dmsq);
    sinsq2thmumu = std::pow(10.0f, sinsq2thmumu);


    float sinterm = std::sin(1.266932679f * dmsq * le);
    float prob    = 1.0f - sinsq2thmumu * sinterm * sinterm;

    if (prob < 0.0f || prob > 1.0f) {
        log<LOG_ERROR>(
            L"%1% || Pmumu %2% outside [0,1]. "
            L"dmsq = %3% L/E = %5%"
        ) % __func__ % prob % dmsq % le;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }

    return prob;
}

// ---------------------------------------------
// νμ → νe appearance
// ---------------------------------------------
float PRO3p1_3B::Pmue(float dmsq, float sinsq2thmumu, float sB, float le) const {
    dmsq   = std::pow(10.0f, dmsq);
    sinsq2thmumu = std::pow(10.0f, sinsq2thmumu);

    float sinterm = std::sin(1.266932679f * dmsq * le);
    float prob    = sB*sinsq2thmumu * sinterm * sinterm;

    if (prob < 0.0f || prob > 1.0f) {
        log<LOG_ERROR>(
            L"%1% || Pmue %2% outside [0,1]. "
            L"dmsq = %3%, sinsq2thmuu = %4%, sB = %5%, L/E = %6%"
        ) % __func__ % prob % dmsq % sinsq2thmumu % sB % le;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }

    return prob;
}

// ---------------------------------------------
// νe → νe disappearance
// ---------------------------------------------
float PRO3p1_3B::Pee(float dmsq, float sinsq2thmumu, float sB, float le) const {
    dmsq   = std::pow(10.0f, dmsq);
    sinsq2thmumu = std::pow(10.0f, sinsq2thmumu);

    float rad = 1.0f - sinsq2thmumu;
    float Um4sq = (1.0f - std::sqrt(rad)) / 2.0f;
    float Ue4sq = sB * (1.0f - Um4sq);


    float sinterm = std::sin(1.266932679f * dmsq * le);
    float prob    = 1.0f - 4.0f * Ue4sq * (1.0f - Ue4sq) * sinterm * sinterm;

    if (prob < 0.0f || prob > 1.0f) {
        log<LOG_ERROR>(
            L"%1% || Pee %2% outside [0,1]. "
            L"dmsq = %3%, Ue4sq = %4%, L/E = %5%"
        ) % __func__ % prob % dmsq % Ue4sq % le;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }

    return prob;
}

Eigen::MatrixXf PRO3p1_3B::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    const auto &le_arr = var_arrs[0];
    //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;

    // Precompute physics parameters once
    float dmsq = maybe_convert_log("dmsq", phys(0));
    float sinsq2thmumu = maybe_convert_log("sinsq2thmumu", phys(1));
    float sB = maybe_convert_log("sB", phys(2));
    float Ue4sq = (sB/2.0)*(1.0+sqrt(1.0f-sinsq2thmumu));

    Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

    for(size_t i = 0; i < le_arr.size(); ++i) {

        float sinterm = std::sin(1.266932679f*dmsq*(le_arr[i]));

        // no oscillation
        probs(i, 0) = 1.0f;

        // P_mumu
        probs(i, 1) =  1.0f - sinsq2thmumu * sinterm * sinterm;

        // P_mue
        probs(i, 2) =  sB*sinsq2thmumu* sinterm * sinterm;


        // P_ee
        probs(i, 3) = 1.0f-4.0f*(1-Ue4sq)*Ue4sq *sinterm*sinterm;

    }

    return probs;
}

// ------------------------------------------------------------------
// PRO3p1_3C
// ------------------------------------------------------------------

PRO3p1_3C::PRO3p1_3C(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

    model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0; });
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),v(2),le); });
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),v(2),le); });
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),v(2),le); });
    prob_types = {0, 1, 2, 3};
    if(parameter_map.find("L/E") == parameter_map.end()) {
        log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
        throw std::runtime_error("Missing parameter: L/E");
    }
    ivars = {parameter_map.at("L/E")};

    //constraints
    model_constraint = [this](const Eigen::VectorXf &v){return this->UnitarityConstraint(v);};


    build_hists_and_combined(prop);


    nparams = 3;
    param_names = {"dmsq", "sinsq2thmue", "xi"};
    pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mue}", "#xi"};
    pretty_param_units = {"eV^{2}", "", ""};
    is_log10 = {true, true, false};
    build_param_index();
    lb = Eigen::VectorXf(3);
    ub = Eigen::VectorXf(3);
    default_val = Eigen::VectorXf(3);
    lb << -2, -std::numeric_limits<float>::infinity(), -10;
    ub << 2, -1e-3, 10;
    //default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
    default_val << -2, -8, 0;
}

int PRO3p1_3C::UnitarityConstraint(const Eigen::VectorXf &v){
    const float sinsq2thmue = maybe_convert_log("sinsq2thmue", v(param_name_to_index.at("sinsq2thmue")));
    const float xi = maybe_convert_log("xi", v(param_name_to_index.at("xi")));
    return   (std::sqrt(sinsq2thmue)*std::cosh(xi)<0.999 ? 1 : 0);
}

float PRO3p1_3C::Pmue(float dmsq, float sinsq2thmue, [[maybe_unused]]float xi, float le) const{
    dmsq = maybe_convert_log("dmsq", dmsq);
    sinsq2thmue = maybe_convert_log("sinsq2thmue", sinsq2thmue);

    if(sinsq2thmue > 1) {
        log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is greater than 1. Setting to 1.")
            % __func__ % sinsq2thmue;
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

float PRO3p1_3C::Pmumu(float dmsq, float sinsq2thmue, float xi, float le) const{
    dmsq = maybe_convert_log("dmsq", dmsq);
    sinsq2thmue = maybe_convert_log("sinsq2thmue", sinsq2thmue);

    float Um4sq=(std::exp(-xi) * std::sqrt(sinsq2thmue)) / 2.0;

    float sinterm = std::sin(1.266932679f*dmsq*(le));
    float prob    = 1.0f - 4.0f*Um4sq*(1.0f-Um4sq)*sinterm*sinterm;

    if(prob<0.0 || prob >1.0){
        log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Um4sq = %4%, sinsq2thmue = %5%, xi = %6%, L/E = %7%") % __func__ % prob % dmsq % Um4sq % sinsq2thmue % xi % le;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }

    return prob;
}

float PRO3p1_3C::Pee(float dmsq, float sinsq2thmue, float xi, float le) const{
    dmsq = maybe_convert_log("dmsq", dmsq);
    sinsq2thmue = maybe_convert_log("sinsq2thmue", sinsq2thmue);

    float Ue4sq=(std::exp(xi) * std::sqrt(sinsq2thmue)) / 2.0;

    float sinterm = std::sin(1.266932679f*dmsq*(le));
    float prob    = 1.0f - 4.0f*Ue4sq*(1.0f-Ue4sq)*sinterm*sinterm;

    if(prob<0.0 || prob >1.0){
        log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Ue4sq = %4%, sinsq2thmue = %5%, xi = %6%, L/E = %7%") % __func__ % prob % dmsq % Ue4sq % sinsq2thmue % xi % le;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }

    return prob;
}

Eigen::MatrixXf PRO3p1_3C::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    const auto &le_arr = var_arrs[0];
    //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;

    // Precompute physics parameters once
    float dmsq = maybe_convert_log("dmsq", phys(0));
    float sinsq2thmue = maybe_convert_log("sinsq2thmue", phys(1));
    float xi = maybe_convert_log("xi", phys(2));

    float sqrtsin = std::sqrt(sinsq2thmue);
    float Um4sq=(std::exp(-xi) *sqrtsin ) / 2.0;
    float Ue4sq=(std::exp(xi) *sqrtsin) / 2.0;


    Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

    for(size_t i = 0; i < le_arr.size(); ++i) {

        float sinterm = std::sin(1.266932679f*dmsq*(le_arr[i]));

        // no oscillation
        probs(i, 0) = 1.0f;

        // P_mumu
        probs(i, 1) =  1.0f - 4.0f*(1-Um4sq)*Um4sq * sinterm * sinterm;

        // P_mue
        probs(i, 2) =  sinsq2thmue* sinterm * sinterm;


        // P_ee
        probs(i, 3) = 1.0f-4.0f*(1-Ue4sq)*Ue4sq *sinterm*sinterm;

    }

    return probs;
}

// ------------------------------------------------------------------
// PRO3p1_decay_invis
// ------------------------------------------------------------------

PRO3p1_decay_invis::PRO3p1_decay_invis(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {
    // 3+1+decay to invisible particles, example from IceCube: https://arxiv.org/pdf/2204.00612
    // (invisible means no active or sterile-oscillating-to-active neutrinos after the decay)

    // model_functions is the non-unified version, these are optional
    // these get combined into one get_probs function in the constructor, but we can override this for faster computation
    model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0; });
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),v(2),v(3),le); });
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),v(2),v(3),le); });
    model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),v(2),v(3),le); });
    prob_types = {0, 1, 2, 3};
    if(parameter_map.find("L/E") == parameter_map.end()) {
        log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
        throw std::runtime_error("Missing parameter: L/E");
    }
    ivars = {parameter_map.at("L/E")};

    model_constraint = [this](const Eigen::VectorXf &v){return this->UnitarityConstraint(v);};

    build_hists_and_combined(prop);

    nparams = 4;
    param_names = {"dmsq", "Ue4^2", "Um4^2", "g2"};
    pretty_param_names = {"#Deltam^{2}", "|U_{e4}|^{2}", "|U_{#mu4}|^{2}", "g^{2}"};
    pretty_param_units = {"eV^{2}", "", "", ""};
    is_log10 = {true, true, true, false};
    build_param_index();
    lb = Eigen::VectorXf(4);
    ub = Eigen::VectorXf(4);
    default_val = Eigen::VectorXf(4);
    lb << -2, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 0;
    ub << 2, -1e-4, -1e-4, 10;
    default_val << -2, -8, -8, 0;
}

int PRO3p1_decay_invis::UnitarityConstraint(const Eigen::VectorXf &v){
    // ensures positive g2 in addition to the usual unitarity constraints
    const float Ue4sq = maybe_convert_log("Ue4^2", v(param_name_to_index.at("Ue4^2")));
    const float Um4sq = maybe_convert_log("Um4^2", v(param_name_to_index.at("Um4^2")));
    const float g2 = maybe_convert_log("g2", v(param_name_to_index.at("g2")));
    return   ((Ue4sq+Um4sq)<1 && g2>=0 ? 1 : 0);
}

// Equations from Jesse Mendez, slide 5 bottom https://microboone-docdb.fnal.gov/cgi-bin/sso/RetrieveFile?docid=45475&filename=2025-10-31-mendez-sterile-deacy.pdf&version=1
//
// Derivation from references:
// See equation 10 here, written in terms of L_osc and L_dec: https://journals.aps.org/prd/pdf/10.1103/PhysRevD.110.075002
//     This is the only term in equation 9 if we set the visible decay term to zero (P_dec == 0)
// Using Delta = 1/4 1/(hbar c) * (m / 1 eV)^2 * ((L / 1 km) / (E / 1 GeV)) = 1/4 m^2 L/E (natural units) = 1.266932679 m^2 L / E (km and GeV units)
// L_osc = 2 pi E / m^2 (just after equation 10)
// Simplifying this term: pi L / L_osc = pi L / (2 pi E / m^2) = 1/2 m^2 L/E = 2 Delta
// From equation 1 of https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.151801: tau = 16 pi / (g^2 m)
// L_dec = (relativistic gamma factor) * c * tau = E/m * tau = E/m * 16 pi / (g^2 m) = 16 pi E / (g^2 m^2)
// Simplifying this term: L / (2 L_dec) = g^2 m^2 L / (32 pi E) = g^2 / 8 pi * (1/4 m^2 L/E) = g^2 / 8 pi * Delta
// Final formula: P_ab = delta_ab - 2 delta_ab |U_a4 U_b4| [1 - exp(-g^2 / 8 pi * Delta) cos(2 Delta)]
//                       + |U_a4 U_b4|^2 [1 - 2 exp(-g^2 / 8 pi * Delta) cos(2 Delta) + exp(-g^2 / 4 pi * Delta)]
//
// Another reference for 3+1+invisible decay: Equation 15 of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.97.055017
//     P_aa = cos^4(theta) + 1/2 exp_term cos(2 Delta) sin^2(2 theta) + exp_term^2 sin^4(theta)
//          = (1 - sin^2(theta))^2 + 1/2 exp_term cos(2 Delta) (4 cos^2(theta) sin^2(theta)) + exp_term^2 sin^4(theta)
//          = (1 - 2 sin^2(theta) + sin^4(theta)) + 1/2 exp_term cos(2 Delta) (4 sin^2(theta) - 4 sin^4(theta)) + exp_term^2 sin^4(theta)
//          = 1 - 2 sin^2(theta) + sin^4(theta) + 2 exp_term cos(2 Delta) sin^2(theta) - 2 exp_term cos(2 Delta) sin^4(theta) + exp_term^2 sin^4(theta)
//          = 1 - 2 sin^2(theta) [1 - exp_term cos(2 Delta)] + sin^4(theta) [1 - 2 exp_term cos(2 Delta) + exp_term^2]
// Taking our full equation, looking at just the disappearance case, and substituting |U_a4|^2 = sin^2(theta):
//     P_aa = 1 - 2 |U_a4|^2 [1 - exp_term cos(2 Delta)] + |U_a4|^4 [1 - 2 exp_term cos(2 Delta) + exp_term^2]
//          = 1 - 2 sin^2(theta) [1 - exp_term cos(2 Delta)] + sin^4(theta) [1 - 2 exp_term cos(2 Delta) + exp_term^2]
// This exactly matches the equation above, confirming that the references are consistent.

float PRO3p1_decay_invis::Pmue(float dmsq, float Ue4sq, float Um4sq, float g2, float le) const{
    dmsq = maybe_convert_log("dmsq", dmsq);
    Ue4sq = maybe_convert_log("Ue4^2", Ue4sq);
    Um4sq = maybe_convert_log("Um4^2", Um4sq);
    g2 = maybe_convert_log("g2", g2);

    if (Ue4sq > 1 || Ue4sq < 0 || Um4sq > 1 || Um4sq < 0 || g2 < 0) {
        log<LOG_ERROR>(L"%1% || Parameter(s) out of bounds. Setting to limits. Values: Ue4sq=%2%, Um4sq=%3%, g2=%4%, dmsq=%5%, le=%6%")
            % __func__ % Ue4sq % Um4sq % g2 % dmsq % le;
        if (Ue4sq > 1) Ue4sq = 1;
        if (Ue4sq < 0) Ue4sq = 0;
        if (Um4sq > 1) Um4sq = 1;
        if (Um4sq < 0) Um4sq = 0;
        if (g2 < 0) g2 = 0;
        exit(EXIT_FAILURE);
    }

    float delta = 1.266932679f*dmsq*le;
    float costerm = std::cos(2.0f*delta);
    float expterm = std::exp(-g2*delta/(8.0f*3.14159f));
    float prob    = Ue4sq*Um4sq*(1.0f-2.0f*expterm*costerm + expterm*expterm);
    //exit(0);

    // numerical precision issues can cause small negative probabilities
    if(-1e-6f < prob && prob<0.0f){
        prob = 0.0f;
    }

    if(prob<0.0 || prob >1.0 || std::isnan(prob)){
        log<LOG_ERROR>(L"%1% || Your probability %2% is outside the 0-1 range. dmsq = %3%, Ue4sq = %4%, Um4sq = %5%, g2 = %6%, L/E = %7%") % __func__ % prob % dmsq % Ue4sq % Um4sq % g2 % le;
        log<LOG_ERROR>(L"delta = %1%, costerm = %2%, expterm = %3%") % delta % costerm % expterm;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }

    return prob;
}

float PRO3p1_decay_invis::Pmumu(float dmsq, [[maybe_unused]]float Ue4sq, float Um4sq, float g2, float le) const{
    dmsq = maybe_convert_log("dmsq", dmsq);
    Um4sq = maybe_convert_log("Um4^2", Um4sq);
    g2 = maybe_convert_log("g2", g2);

    if (Um4sq > 1 || Um4sq < 0 || g2 < 0) {
        log<LOG_ERROR>(L"%1% || Parameter(s) out of bounds. Setting to limits. Values: Um4sq=%2%, g2=%3%, dmsq=%4%, le=%5%")
            % __func__ % Um4sq % g2 % dmsq % le;
        if (Um4sq > 1) Um4sq = 1;
        if (Um4sq < 0) Um4sq = 0;
        if (g2 < 0) g2 = 0;
        exit(EXIT_FAILURE);
    }

    float delta = 1.266932679f*dmsq*le;
    float costerm = std::cos(2.0f*delta);
    float expterm = std::exp(-g2*delta/(8.0f*3.14159f));
    float prob    = 1.0f - 2.0f*Um4sq*(1.0f-expterm*costerm) + Um4sq*Um4sq*(1.0f-2.0f*expterm*costerm + expterm*expterm);

    // numerical precision issues can cause small negative probabilities
    if(-1e-6f < prob && prob<0.0f){
        prob = 0.0f;
    }

    if(prob<0.0 || prob >1.0 || std::isnan(prob)){
        log<LOG_ERROR>(L"%1% || Your probability %2% is the 0-1 range. dmsq = %3%, Um4sq = %4%, g2 = %5%, L/E = %6%") % __func__ % prob % dmsq % Um4sq % g2 % le;
        log<LOG_ERROR>(L"delta = %1%, costerm = %2%, expterm = %3%") % delta % costerm % expterm;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }

    return prob;
}

float PRO3p1_decay_invis::Pee(float dmsq, float Ue4sq, [[maybe_unused]]float Um4sq, float g2, float le) const{
    dmsq = maybe_convert_log("dmsq", dmsq);
    Ue4sq = maybe_convert_log("Ue4^2", Ue4sq);
    g2 = maybe_convert_log("g2", g2);

    if (Ue4sq > 1 || Ue4sq < 0 || g2 < 0) {
        log<LOG_ERROR>(L"%1% || Parameter(s) out of bounds. Setting to limits. Values: Ue4sq=%2%, g2=%3%, dmsq=%4%, le=%5%")
            % __func__ % Ue4sq % g2 % dmsq % le;
        if (Ue4sq > 1) Ue4sq = 1;
        if (Ue4sq < 0) Ue4sq = 0;
        if (g2 < 0) g2 = 0;
        exit(EXIT_FAILURE);
    }

    float delta = 1.266932679f*dmsq*le;
    float costerm = std::cos(2.0f*delta);
    float expterm = std::exp(-g2*delta/(8.0f*3.14159f));
    float prob    = 1.0f - 2.0f*Ue4sq*(1.0f-expterm*costerm) + Ue4sq*Ue4sq*(1.0f-2.0f*expterm*costerm + expterm*expterm);

    // numerical precision issues can cause small negative probabilities
    if(-1e-6f < prob && prob<0.0f){
        prob = 0.0f;
    }

    if(prob<0.0 || prob >1.0 || std::isnan(prob)){
        log<LOG_ERROR>(L"%1% || Your probability %2% is outside the 0-1 range. dmsq = %3%, Ue4sq = %4%, g2 = %5%, L/E = %6%") % __func__ % prob % dmsq % Ue4sq % g2 % le;
        log<LOG_ERROR>(L"delta = %1%, costerm = %2%, expterm = %3%") % delta % costerm % expterm;
        log<LOG_ERROR>(L"term1 = %1%, term2 = %2%, term3 = %3%") % 1.0f % (-2.0f*Ue4sq*(1.0f-expterm*costerm)) % (Ue4sq*Ue4sq*(1.0f-2.0f*expterm*costerm + expterm*expterm));
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }

    return prob;
}

Eigen::MatrixXf PRO3p1_decay_invis::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    const auto &le_arr = var_arrs[0];
    //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;

    // Precompute physics parameters once
    float dmsq = maybe_convert_log("dmsq", phys(0));
    float Ue4sq = maybe_convert_log("Ue4^2", phys(1));
    float Um4sq = maybe_convert_log("Um4^2", phys(2));
    float g2 = maybe_convert_log("g2", phys(3));

    //log<LOG_ERROR>(L"%1% || dmsq = %2%, Ue4sq = %3%, Um4sq = %4%, g2 = %5%") % __func__ % dmsq % Ue4sq % Um4sq % g2;

    float freq = 1.266932679f * dmsq;

    Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

    for(size_t i = 0; i < le_arr.size(); ++i) {

        // no oscillation
        probs(i, 0) = 1.0f;

        float delta = freq*le_arr[i];
        float costerm = std::cos(2.0f*delta);
        float expterm = std::exp(-g2*delta/(8.0f*3.14159f));
        float cos_mult_exp_term = costerm*expterm;
        float osc_term =(1.0f-2.0f*cos_mult_exp_term + expterm*expterm);

        // P_mumu
        probs(i, 1) = 1.0f - 2.0f*Um4sq*(1.0f-cos_mult_exp_term) + Um4sq*Um4sq*osc_term;

        // P_mue
        probs(i, 2) = Ue4sq*Um4sq*osc_term;

        // P_ee
        probs(i, 3) = 1.0f - 2.0f*Ue4sq*(1.0f-cos_mult_exp_term) + Ue4sq*Ue4sq*osc_term;

    }

    return probs;
}

}
