/**
 * @file PROmodel3p2.h
 * @brief 3+2 sterile-neutrino oscillation model.
 * @author PROfit Collaboration
 * @internal PRO3p2 is constructed only via get_model_from_string
 * (src/PROmodel.cxx); prefer the factory over direct construction.
 */
#ifndef PROMODEL3P2_H
#define PROMODEL3P2_H

#include "PROmodel.h"

namespace PROfit {

/**
 * @brief 3+2 sterile-neutrino model with two independent heavy mass eigenstates.
 * @details Provides nu_mu disappearance, nu_mu → nu_e appearance, and nu_e disappearance channels
 * driven by two mass splittings Delta m^2_41 and Delta m^2_51 and four mixing elements
 * |U_e4|^2, |U_mu4|^2, |U_e5|^2, |U_mu5|^2 plus an inter-sterile CP phase phi_54.
 * Seven parameters total: dmsq41, dmsq51 [log10], Ue4sq, Um4sq, Ue5sq, Um5sq [log10], phi54 [linear, rad].
 * Unitarity constraints |U_ea|^2 sum < 1 and |U_mua|^2 sum < 1 are enforced via model_constraint.
 */
class PRO3p2 : public PROmodel {
public:
    /**
     * @brief Construct the PRO3p2 model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PRO3p2(const PROpeller &prop, const std::map<std::string,int> &parameter_map);

    // Enforce |Ue4|^2 + |Ue5|^2 <= 1 and |Um4|^2 + |Um5|^2 <= 1
    int UnitarityConstraint(const Eigen::VectorXf &v);

    // ---------- Appearance: νμ → νe (α=μ, β=e) ----------
    float Pmue(const Eigen::VectorXf &v, float le) const;

    // ---------- Appearance: anti-νμ → anti-νe (CP-conjugate, phi54 flipped) ----------
    float Pmue_anti(const Eigen::VectorXf &v, float le) const;

    // ---------- Disappearance: νμ → νμ (α=μ) ----------
    float Pmumu(const Eigen::VectorXf &v, float le) const;

    // ---------- Disappearance: νe → νe (α=e) ----------
    float Pee(const Eigen::VectorXf &v, float le) const;

    /** @brief Closed-form derivatives of get_probs (see PROmodel::get_probs_grad). */
    std::vector<Eigen::MatrixXf> get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
};

}

#endif
