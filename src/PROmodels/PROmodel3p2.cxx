/**
 * @file PROmodel3p2.cxx
 * @brief Implementation of the PRO3p2 3+2 sterile-neutrino model.
 * @author PROfit Collaboration
 */
#include "PROmodels/PROmodel3p2.h"

namespace PROfit {

PRO3p2::PRO3p2(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

    // model functions: 0 = null, 1 = numu->numu, 2 = numu->nue, 3 = nue->nue, 4 = antinumu->antinue
    model_functions.push_back(
        [this]([[maybe_unused]] const Eigen::VectorXf &v, float) { (void)this; return 1.0f; });
    model_functions.push_back(
        [this](const Eigen::VectorXf &v, float le) { return this->Pmumu(v, le); });
    model_functions.push_back(
        [this](const Eigen::VectorXf &v, float le) { return this->Pmue(v, le); });
    model_functions.push_back(
        [this](const Eigen::VectorXf &v, float le) { return this->Pee(v, le); });
    model_functions.push_back(
        [this](const Eigen::VectorXf &v, float le) { return this->Pmue_anti(v, le); });
    prob_types = {0, 1, 2, 3, 4};

    if(parameter_map.find("L/E") == parameter_map.end()) {
        log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.")
            % __func__ % __LINE__;
        throw std::runtime_error("Missing parameter: L/E");
    }
    ivars = {parameter_map.at("L/E")};

    // Unitarity constraints for e and mu rows
    model_constraint = [this](const Eigen::VectorXf &v){ return this->UnitarityConstraint(v); };

    // build histograms as in other models
    build_hists_and_combined(prop);

    // parameters: dmsq41, dmsq51, log10(Ue4^2), log10(Um4^2), log10(Ue5^2), log10(Um5^2), phi54
    nparams = 7;
    param_names = {"dmsq41", "dmsq51", "Ue4sq", "Um4sq", "Ue5sq", "Um5sq", "phi54"};
    pretty_param_names = {
        "#Delta m^{2}_{41}", "#Delta m^{2}_{51}",
        "|U_{e4}|^{2}",      "|U_{#mu4}|^{2}",
        "|U_{e5}|^{2}",      "|U_{#mu5}|^{2}",
        "#phi_{54}"
    };
    pretty_param_units = {"eV^{2}", "eV^{2}", "", "", "", "", "rad"};
    is_log10 = {true, true, true, true, true, true, false};

    lb = Eigen::VectorXf(nparams);
    ub = Eigen::VectorXf(nparams);
    default_val = Eigen::VectorXf(nparams);

    // log10(Δm^2) between 10^-2 and 10^2, mixings between ~10^-8 and 1, phi in [-pi,pi]
    lb << -2, -2,
          -5, -5,
          -5, -5,
           0.0f;
    ub <<  2,  2,
          -0.01,  -0.01,
          -0.01,  -0.01,
          6.28318f;

    // some reasonable defaults
    default_val << -1, 0, -4, -4, -4, -4, 0.0f;
}

// Enforce |Ue4|^2 + |Ue5|^2 <= 1 and |Um4|^2 + |Um5|^2 <= 1
int PRO3p2::UnitarityConstraint(const Eigen::VectorXf &v) {
    float Ue4sq = std::pow(10.0f, v(2));
    float Um4sq = std::pow(10.0f, v(3));
    float Ue5sq = std::pow(10.0f, v(4));
    float Um5sq = std::pow(10.0f, v(5));

    if(Ue4sq + Ue5sq >= 1.0f) return 0;
    if(Um4sq + Um5sq >= 1.0f) return 0;
    // more careful with unitarity since tau and steriles are not specified and checking the max value of probability
    if(4*Um4sq*Ue4sq + 4*Um5sq*Ue5sq + 8*std::sqrt(Um4sq*Ue4sq*Um5sq*Ue5sq) >= 1.0f) return 0;

    return 1;
}

// Convenience to unpack parameters

// ---------- Appearance: νμ → νe (α=μ, β=e) ----------
float PRO3p2::Pmue(const Eigen::VectorXf &v, float le) const {

// Convert log10 parameters to physical values
float dm41  = std::pow(10.0f, v(0));
float dm51  = std::pow(10.0f, v(1));
float Ue4sq = std::pow(10.0f, v(2));
float Um4sq = std::pow(10.0f, v(3));
float Ue5sq = std::pow(10.0f, v(4));
float Um5sq = std::pow(10.0f, v(5));
float phi54 = v(6);

// Oscillation phases
float x41 = 1.266932679f * dm41 * le;
float x51 = 1.266932679f * dm51 * le;
float x54 = 1.266932679f * (dm51 - dm41) * le;

// Standard 3+2 appearance terms
float term1 = 4.0f * Um4sq * Ue4sq * std::sin(x41) * std::sin(x41);
float term2 = 4.0f * Um5sq * Ue5sq * std::sin(x51) * std::sin(x51);
float term3 = 8.0f * std::sqrt(Um4sq * Ue4sq * Um5sq * Ue5sq)
                    * std::sin(x41) * std::sin(x51)
                    * std::cos(x54 - phi54);

float prob = term1 + term2 + term3;

const float eps = 1e-6f;

if (prob < 0.0f && prob > -eps) prob = 0.0f;

if (prob > 1.0f && prob < 1.0f + eps) prob = 1.0f;

if(prob < 0.0f || prob > 1.0f || std::isnan(prob)) {
    log<LOG_ERROR>(L"%1% || Bad Pmue = %2%  (le=%3%)"
        L"\ndm41=%4% dm51=%5%  Ue4sq=%6% Um4sq=%7%  Ue5sq=%8% Um5sq=%9%  phi54=%10% term1=%11% term2=%12% term3=%13%"
         ) % __func__ % prob % le
         % dm41 % dm51 % Ue4sq % Um4sq % Ue5sq % Um5sq % phi54% term1 % term2 % term3;
        exit(EXIT_FAILURE);
    }
    return prob;
}


// ---------- Appearance: anti-νμ → anti-νe (CP-conjugate, phi54 flipped) ----------
float PRO3p2::Pmue_anti(const Eigen::VectorXf &v, float le) const {

float dm41  = std::pow(10.0f, v(0));
float dm51  = std::pow(10.0f, v(1));
float Ue4sq = std::pow(10.0f, v(2));
float Um4sq = std::pow(10.0f, v(3));
float Ue5sq = std::pow(10.0f, v(4));
float Um5sq = std::pow(10.0f, v(5));
float phi54 = v(6);

float x41 = 1.266932679f * dm41 * le;
float x51 = 1.266932679f * dm51 * le;
float x54 = 1.266932679f * (dm51 - dm41) * le;

// Standard 3+2 appearance terms (anti-nu: +phi54 instead of -phi54)
float term1 = 4.0f * Um4sq * Ue4sq * std::sin(x41) * std::sin(x41);
float term2 = 4.0f * Um5sq * Ue5sq * std::sin(x51) * std::sin(x51);
float term3 = 8.0f * std::sqrt(Um4sq * Ue4sq * Um5sq * Ue5sq)
                    * std::sin(x41) * std::sin(x51)
                    * std::cos(x54 + phi54);

float prob = term1 + term2 + term3;

const float eps = 1e-6f;

if (prob < 0.0f && prob > -eps) prob = 0.0f;

if (prob > 1.0f && prob < 1.0f + eps) prob = 1.0f;

if(prob < 0.0f || prob > 1.0f || std::isnan(prob)) {
    log<LOG_ERROR>(L"%1% || Bad Pmue_anti = %2%  (le=%3%)"
        L"\ndm41=%4% dm51=%5%  Ue4sq=%6% Um4sq=%7%  Ue5sq=%8% Um5sq=%9%  phi54=%10% term1=%11% term2=%12% term3=%13%"
         ) % __func__ % prob % le
         % dm41 % dm51 % Ue4sq % Um4sq % Ue5sq % Um5sq % phi54 % term1 % term2 % term3;
        exit(EXIT_FAILURE);
    }
    return prob;
}

// ---------- Disappearance: νμ → νμ (α=μ) ----------
float PRO3p2::Pmumu(const Eigen::VectorXf &v, float le) const {

float dm41  = std::pow(10.0f, v(0));
float dm51  = std::pow(10.0f, v(1));
float Ue4sq = std::pow(10.0f, v(2));
float Um4sq = std::pow(10.0f, v(3));
float Ue5sq = std::pow(10.0f, v(4));
float Um5sq = std::pow(10.0f, v(5));
float phi54 = v(6);

float x41 = 1.266932679f * dm41 * le;
float x51 = 1.266932679f * dm51 * le;
float x54 = 1.266932679f * (dm51 - dm41) * le;

float one_minus = 1.0f - Um4sq - Um5sq;

float s41 = std::sin(x41);
float s51 = std::sin(x51);
float s54 = std::sin(x54);

// Individual components
float term1 = 1.0f;

float term2 = -4.0f * one_minus *
          (Um4sq * s41 * s41 +
           Um5sq * s51 * s51);

float term3 = -4.0f * Um4sq * Um5sq *
          (s54 * s54);

// Full probability
float prob = term1 + term2 + term3;


const float eps = 1e-6f;

if (prob < 0.0f && prob > -eps) prob = 0.0f;

if (prob > 1.0f && prob < 1.0f + eps) prob = 1.0f;

if(prob < 0.0f || prob > 1.0f || std::isnan(prob)) {
    log<LOG_ERROR>(L"%1% || Bad Pmumu = %2%  (le=%3%)"
        L"\ndm41=%4% dm51=%5%  Ue4sq=%6% Um4sq=%7%  Ue5sq=%8% Um5sq=%9%  phi54=%10% term1=%11% term2=%12% term3=%13%"
    ) % __func__ % prob % le
      % dm41 % dm51 % Ue4sq % Um4sq % Ue5sq % Um5sq % phi54 % term1 % term2 % term3;
    exit(EXIT_FAILURE);
}
return prob;
}


// ---------- Disappearance: νe → νe (α=e) ----------
float PRO3p2::Pee(const Eigen::VectorXf &v, float le) const {

float dm41  = std::pow(10.0f, v(0));
float dm51  = std::pow(10.0f, v(1));
float Ue4sq = std::pow(10.0f, v(2));
float Um4sq = std::pow(10.0f, v(3));
float Ue5sq = std::pow(10.0f, v(4));
float Um5sq = std::pow(10.0f, v(5));
float phi54 = v(6);

float x41 = 1.266932679f * dm41 * le;
float x51 = 1.266932679f * dm51 * le;
float x54 = 1.266932679f * (dm51 - dm41) * le;

float one_minus = 1.0f - Ue4sq - Ue5sq;

float s41 = std::sin(x41);
float s51 = std::sin(x51);
float s54 = std::sin(x54);

float term1 = 1.0f;
float term2 = -4.0f * one_minus * (Ue4sq * s41 * s41 + Ue5sq * s51 * s51);
float term3 = -4.0f * Ue4sq * Ue5sq * s54 * s54;

float prob = term1 + term2 + term3;

const float eps = 1e-6f;

if (prob < 0.0f && prob > -eps) prob = 0.0f;

if (prob > 1.0f && prob < 1.0f + eps) prob = 1.0f;

if(prob < 0.0f || prob > 1.0f || std::isnan(prob)) {
    log<LOG_ERROR>(L"%1% || Bad Pee = %2%  (le=%3%)"
        L"\ndm41=%4% dm51=%5%  Ue4sq=%6% Um4sq=%7%  Ue5sq=%8% Um5sq=%9%  phi54=%10% term1=%11% term2=%12% term3=%13%"
    ) % __func__ % prob % le
      % dm41 % dm51 % Ue4sq % Um4sq % Ue5sq % Um5sq % phi54 % term1 % term2 % term3;
    exit(EXIT_FAILURE);
}
return prob;
}

}
