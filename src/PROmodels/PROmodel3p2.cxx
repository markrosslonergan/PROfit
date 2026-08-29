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

// Closed-form derivatives of the five probability columns (0 = null, 1 = Pmumu,
// 2 = Pmue, 3 = Pee, 4 = Pmue_anti) in the fitter's internal space.
//
// Phases (k = 1.266932679):  x41 = k Δm²41 L/E,  x51 = k Δm²51 L/E,  x54 = x51 − x41,
// so  ∂x41/∂Δm²41 = k L/E,  ∂x51/∂Δm²51 = k L/E,  ∂x54/∂Δm²41 = −k L/E,  ∂x54/∂Δm²51 = +k L/E.
// Mixings Ua4 = |Ua4|², Ua5 = |Ua5|² (a = e, μ); R = √(Ue4 Um4 Ue5 Um5).
//
// Appearance (ν: cos(x54 − φ); ν̄: cos(x54 + φ)), writing cs/sn for cos/sin of that argument:
//   P = 4 Um4 Ue4 sin²x41 + 4 Um5 Ue5 sin²x51 + 8 R sin x41 sin x51 cs
//   ∂P/∂x41 = 4 Um4 Ue4 sin 2x41 + 8 R cos x41 sin x51 cs
//   ∂P/∂x51 = 4 Um5 Ue5 sin 2x51 + 8 R sin x41 cos x51 cs
//   ∂P/∂x54 = −8 R sin x41 sin x51 sn
//   ∂P/∂φ   = −8 R sin x41 sin x51 sn · (∓1)        (sign of φ inside the cosine)
//   ∂P/∂Ue4 = 4 Um4 sin²x41 + 8 sin x41 sin x51 cs · ∂R/∂Ue4,  with ∂R/∂U = R / (2U)
//   (and likewise for Um4, Ue5, Um5). For the log10 mixings the chain factor ln10·U
//   turns R/(2U) into ln10 · R / 2, which is finite as U → 0.
// The Δm² derivatives combine the phase derivatives: ∂P/∂Δm²41 = ∂P/∂x41 · ∂x41/∂Δm²41 + ∂P/∂x54 · ∂x54/∂Δm²41, etc.
//
// Disappearance, row a ∈ {μ, e} (col 1 / col 3), with B = Ua4 sin²x41 + Ua5 sin²x51:
//   P_aa = 1 − 4 (1 − Ua4 − Ua5) B − 4 Ua4 Ua5 sin²x54
//   ∂P/∂x41 = −4 (1 − Ua4 − Ua5) Ua4 sin 2x41,   ∂P/∂x51 = −4 (1 − Ua4 − Ua5) Ua5 sin 2x51,
//   ∂P/∂x54 = −4 Ua4 Ua5 sin 2x54
//   ∂P/∂Ua4 = +4 B − 4 (1 − Ua4 − Ua5) sin²x41 − 4 Ua5 sin²x54   (the +4B from d(1 − Ua4 − Ua5)/dUa4 = −1)
//   ∂P/∂Ua5 = +4 B − 4 (1 − Ua4 − Ua5) sin²x51 − 4 Ua4 sin²x54
std::vector<Eigen::MatrixXf> PRO3p2::get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    const auto &le_arr = var_arrs[0];
    float dm41  = maybe_convert_log("dmsq41", phys(0));
    float dm51  = maybe_convert_log("dmsq51", phys(1));
    float Ue4sq = maybe_convert_log("Ue4sq",  phys(2));
    float Um4sq = maybe_convert_log("Um4sq",  phys(3));
    float Ue5sq = maybe_convert_log("Ue5sq",  phys(4));
    float Um5sq = maybe_convert_log("Um5sq",  phys(5));
    float phi54 = maybe_convert_log("phi54",  phys(6));

    constexpr float LN10 = 2.302585093f;
    const float lin[7] = {dm41, dm51, Ue4sq, Um4sq, Ue5sq, Um5sq, phi54};
    float ch[7];   // d(linear)/d(internal)
    for(int j = 0; j < 7; ++j) ch[j] = is_log10[j] ? LN10 * lin[j] : 1.0f;

    const float R = std::sqrt(Ue4sq * Um4sq * Ue5sq * Um5sq);
    // dR/d(internal U): R/(2U) * chain; the guard only matters for linear params at U=0.
    auto dR = [&](float U, float c){ return U > 0.0f ? R / (2.0f*U) * c : 0.0f; };
    const float dR_dUe4 = dR(Ue4sq, ch[2]), dR_dUm4 = dR(Um4sq, ch[3]);
    const float dR_dUe5 = dR(Ue5sq, ch[4]), dR_dUm5 = dR(Um5sq, ch[5]);
    const float om_mu = 1.0f - Um4sq - Um5sq, om_e = 1.0f - Ue4sq - Ue5sq;

    constexpr float k = 1.266932679f;
    std::vector<Eigen::MatrixXf> grads(7, Eigen::MatrixXf::Zero(le_arr.size(), model_functions.size()));
    for(size_t i = 0; i < le_arr.size(); ++i) {
        const float le = le_arr[i];
        const float x41 = k * dm41 * le, x51 = k * dm51 * le, x54 = k * (dm51 - dm41) * le;
        const float s41 = std::sin(x41), c41 = std::cos(x41);
        const float s51 = std::sin(x51), c51 = std::cos(x51);
        const float s54 = std::sin(x54);
        const float sin2x41 = 2.0f*s41*c41, sin2x51 = 2.0f*s51*c51, sin2x54 = std::sin(2.0f*x54);
        // Phase derivatives wrt the internal mass splittings.
        const float dx41_d0 =  k * le * ch[0];            // dx41/d(dm41)
        const float dx51_d1 =  k * le * ch[1];            // dx51/d(dm51)
        const float dx54_d0 = -k * le * ch[0], dx54_d1 = k * le * ch[1];

        // --- appearance, col 2 (nu, cos(x54 - phi)) and col 4 (antinu, cos(x54 + phi)) ---
        for(int col = 2; col <= 4; col += 2) {
            const float sgn = (col == 2) ? -1.0f : 1.0f;      // sign of phi in the argument
            const float arg = x54 + sgn * phi54;
            const float cs = std::cos(arg), sn = std::sin(arg);
            const float dP_dx41 = 4.0f*Um4sq*Ue4sq*sin2x41 + 8.0f*R*c41*s51*cs;
            const float dP_dx51 = 4.0f*Um5sq*Ue5sq*sin2x51 + 8.0f*R*s41*c51*cs;
            const float dP_dx54 = -8.0f*R*s41*s51*sn;
            grads[0](i, col) = dP_dx41 * dx41_d0 + dP_dx54 * dx54_d0;
            grads[1](i, col) = dP_dx51 * dx51_d1 + dP_dx54 * dx54_d1;
            grads[2](i, col) = 4.0f*Um4sq*s41*s41 * ch[2] + 8.0f*s41*s51*cs * dR_dUe4;
            grads[3](i, col) = 4.0f*Ue4sq*s41*s41 * ch[3] + 8.0f*s41*s51*cs * dR_dUm4;
            grads[4](i, col) = 4.0f*Um5sq*s51*s51 * ch[4] + 8.0f*s41*s51*cs * dR_dUe5;
            grads[5](i, col) = 4.0f*Ue5sq*s51*s51 * ch[5] + 8.0f*s41*s51*cs * dR_dUm5;
            grads[6](i, col) = -8.0f*R*s41*s51*sn * sgn * ch[6];
        }
        // --- disappearance: P_aa = 1 - 4(1-Ua4-Ua5)(Ua4 s41^2 + Ua5 s51^2) - 4 Ua4 Ua5 s54^2 ---
        // col 1: mu row; col 3: e row.
        {
            const float B = Um4sq*s41*s41 + Um5sq*s51*s51;
            grads[0](i, 1) = -4.0f*om_mu*Um4sq*sin2x41 * dx41_d0 - 4.0f*Um4sq*Um5sq*sin2x54 * dx54_d0;
            grads[1](i, 1) = -4.0f*om_mu*Um5sq*sin2x51 * dx51_d1 - 4.0f*Um4sq*Um5sq*sin2x54 * dx54_d1;
            grads[3](i, 1) = (4.0f*B - 4.0f*om_mu*s41*s41 - 4.0f*Um5sq*s54*s54) * ch[3];
            grads[5](i, 1) = (4.0f*B - 4.0f*om_mu*s51*s51 - 4.0f*Um4sq*s54*s54) * ch[5];
        }
        {
            const float B = Ue4sq*s41*s41 + Ue5sq*s51*s51;
            grads[0](i, 3) = -4.0f*om_e*Ue4sq*sin2x41 * dx41_d0 - 4.0f*Ue4sq*Ue5sq*sin2x54 * dx54_d0;
            grads[1](i, 3) = -4.0f*om_e*Ue5sq*sin2x51 * dx51_d1 - 4.0f*Ue4sq*Ue5sq*sin2x54 * dx54_d1;
            grads[2](i, 3) = (4.0f*B - 4.0f*om_e*s41*s41 - 4.0f*Ue5sq*s54*s54) * ch[2];
            grads[4](i, 3) = (4.0f*B - 4.0f*om_e*s51*s51 - 4.0f*Ue4sq*s54*s54) * ch[4];
        }
    }
    return grads;
}

}
