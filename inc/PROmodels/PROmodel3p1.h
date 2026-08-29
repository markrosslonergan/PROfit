/**
 * @file PROmodel3p1.h
 * @brief 3+1 sterile-neutrino oscillation models in the short-baseline approximation.
 * @author PROfit Collaboration
 * @internal These models are constructed only via
 * get_model_from_string (src/PROmodel.cxx); prefer the factory over direct construction.
 *
 * @details Declares the 3+1 family:
 *   - PRO3p1             — parameterised by (dmsq, |U_e4|^2, |U_mu4|^2).
 *   - PRO3p1_angles      — parameterised by (dmsq, sin^2 2theta_14, sin^2 theta_24).
 *   - PRO3p1_3A          — variant 3A: (dmsq, sin^2 2theta_ee, sin^2 theta_24).
 *   - PRO3p1_3B          — variant 3B: (dmsq, sin^2 2theta_mumu, sB).
 *   - PRO3p1_3C          — variant 3C: (dmsq, sin^2 2theta_mue, xi).
 *   - PRO3p1_decay_invis — 3+1 with invisible decay: (dmsq, |U_e4|^2, |U_mu4|^2, g^2).
 */
#ifndef PROMODEL3P1_H
#define PROMODEL3P1_H

#include "PROmodel.h"

namespace PROfit {

/**
 * @brief Full 3+1 sterile-neutrino model parameterised by |U_e4|^2 and |U_mu4|^2.
 * @details Provides all three oscillation channels in the short-baseline approximation:
 *   - P(nu_mu -> nu_mu) = 1 - 4*|U_mu4|^2*(1 - |U_mu4|^2) * sin^2(1.267 * Dm^2 * L/E)
 *   - P(nu_mu -> nu_e)  = 4*|U_e4|^2*|U_mu4|^2          * sin^2(1.267 * Dm^2 * L/E)
 *   - P(nu_e  -> nu_e)  = 1 - 4*|U_e4|^2*(1 - |U_e4|^2) * sin^2(1.267 * Dm^2 * L/E)
 * A unitarity constraint |U_e4|^2 + |U_mu4|^2 < 1 is enforced via model_constraint.
 * All three parameters (dmsq, Ue4^2, Um4^2) are stored in log10 space.
 */
class PRO3p1 : public PROmodel {
public:
    /**
     * @brief Construct the PRO3p1 model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PRO3p1(const PROpeller &prop, const std::map<std::string,int> &parameter_map);

    /**
     * @brief Enforce the unitarity constraint |U_e4|^2 + |U_mu4|^2 < 1.
     * @param v  Physics parameter vector in log10 space.
     * @return 1 if the point is physically allowed, 0 otherwise.
     */
    int UnitarityConstraint(const Eigen::VectorXf &v);

    /**
     * @brief Compute the 3+1 nu_mu → nu_e appearance probability.
     * @param dmsq   Mass splitting Delta m^2 in eV^2 (log10 space; converted internally).
     * @param Ue4sq  |U_e4|^2 (log10 space; converted internally).
     * @param Um4sq  |U_mu4|^2 (log10 space; converted internally).
     * @param le     L/E ratio in km/GeV.
     * @return Appearance probability in [0, 1].
     */
    float Pmue(float dmsq, float Ue4sq, float Um4sq, float le) const;

    /**
     * @brief Compute the 3+1 nu_mu survival probability.
     * @param dmsq   Mass splitting Delta m^2 in eV^2 (log10 space; converted internally).
     * @param Ue4sq  |U_e4|^2 — unused in this channel.
     * @param Um4sq  |U_mu4|^2 (log10 space; converted internally).
     * @param le     L/E ratio in km/GeV.
     * @return Survival probability in [0, 1].
     */
    float Pmumu(float dmsq, float Ue4sq, float Um4sq, float le) const;

    /**
     * @brief Compute the 3+1 nu_e survival probability.
     * @param dmsq   Mass splitting Delta m^2 in eV^2 (log10 space; converted internally).
     * @param Ue4sq  |U_e4|^2 (log10 space; converted internally).
     * @param Um4sq  |U_mu4|^2 — unused in this channel.
     * @param le     L/E ratio in km/GeV.
     * @return Survival probability in [0, 1].
     */
    float Pee(float dmsq, float Ue4sq, float Um4sq, float le) const;

    /**
     * @brief Compute all oscillation probabilities for the PRO3p1 model at each L/E grid point.
     * @param phys     Physics vector: (log10(dmsq), log10(Ue4sq), log10(Um4sq)).
     * @param var_arrs 1-element vector containing the L/E array [km/GeV] of length n_phys_bins.
     * @return Matrix (n_phys_bins, 4): columns = {1, P_mumu, P_mue, P_ee}.
     */
    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;

    /** @brief Closed-form derivatives of get_probs (see PROmodel::get_probs_grad). */
    std::vector<Eigen::MatrixXf> get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
};

/**
 * @brief 3+1 sterile-neutrino model parameterised by mixing angles sin^2(2*theta_14) and sin^2(theta_24).
 * @details An alternative to PRO3p1 that uses the angle parameterisation instead of |U|^2 elements directly.
 * Provides nu_mu disappearance, nu_mu → nu_e appearance, and nu_e disappearance channels.
 * All parameters (dmsq, sinsq2th14, sinsqth24) are stored in log10 space.
 */
class PRO3p1_angles : public PROmodel {
public:
    /**
     * @brief Construct the PRO3p1_angles model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PRO3p1_angles(const PROpeller &prop, const std::map<std::string,int> &parameter_map);

    int UnitarityConstraint(const Eigen::VectorXf &);

    float Pmue(float dmsq, float sinsq2th14, float sinsqth24, float le) const;

    float Pmumu(float dmsq, float sinsq2th14, float sinsqth24, float le) const;

    float Pee(float dmsq, float sinsq2th14, float sinsqth24, float le) const;

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
    /** @brief Closed-form derivatives of get_probs (see PROmodel::get_probs_grad). */
    std::vector<Eigen::MatrixXf> get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
};

/**
 * @brief 3+1 model variant 3A: parameterised by sin^2(2*theta_ee) and sin^2(theta_24).
 * @details This variant expresses all three channels — nu_mu disappearance, nu_mu → nu_e appearance,
 * and nu_e disappearance — in terms of the nue-sector mixing angle sin^2(2*theta_ee) = 4*|U_e4|^2*(1-|U_e4|^2)
 * and the muon-sector angle sin^2(theta_24).  Parameters (dmsq, sinsq2thee, sinsqth24) are in log10 space.
 */
class PRO3p1_3A : public PROmodel {
public:
    /**
     * @brief Construct the PRO3p1_3A model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PRO3p1_3A(const PROpeller &prop, const std::map<std::string,int> &parameter_map);

    int UnitarityConstraint(const Eigen::VectorXf &);

    float Pmue(float dmsq, float sinsq2thee, float sinsqth24, float le) const;

    float Pmumu(float dmsq, float sinsq2thee, float sinsqth24, float le) const;

    float Pee(float dmsq, float sinsq2thee, float sinsqth24, float le) const;

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
    /** @brief Closed-form derivatives of get_probs (see PROmodel::get_probs_grad). */
    std::vector<Eigen::MatrixXf> get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
};

/**
 * @brief 3+1 model variant 3B: parameterised by sin^2(2*theta_mumu) and an asymmetry ratio sB.
 * @details Variant B uses the nu_mu disappearance amplitude directly (sin^2(2*theta_mumu)) together
 * with a ratio parameter sB that controls how much of the nu_mu mixing leaks into the nu_e sector.
 * The nu_e sector mixing is derived as Ue4^2 = sB * (1 - Um4^2) where
 * Um4^2 = (1 - sqrt(1 - sin^2(2*theta_mumu))) / 2.
 * A unitarity constraint is enforced via model_constraint.
 * Parameters (dmsq, sinsq2thmumu, sB) are all in log10 space.
 */
class PRO3p1_3B : public PROmodel {
public:
    /**
     * @brief Construct the PRO3p1_3B model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PRO3p1_3B(const PROpeller &prop,
              const std::map<std::string,int> &parameter_map);

    // Unitarity / physicality constraint:
    // 0 ≤ sB ≤ 1 and Ue4² + Uμ4² < 1
    int UnitarityConstraint(const Eigen::VectorXf &v);

    // νμ → νμ disappearance
    float Pmumu(float dmsq, float sinsq2thmumu, float sinsqth24prime, float le) const;

    // νμ → νe appearance
    float Pmue(float dmsq, float sinsq2thmumu, float sB, float le) const;

    // νe → νe disappearance
    float Pee(float dmsq, float sinsq2thmumu, float sB, float le) const;

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
    /** @brief Closed-form derivatives of get_probs (see PROmodel::get_probs_grad). */
    std::vector<Eigen::MatrixXf> get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
};

/**
 * @brief 3+1 model variant 3C: parameterised by sin^2(2*theta_mue) and an asymmetry angle xi.
 * @details Variant C uses the appearance amplitude sin^2(2*theta_mue) directly together with a
 * hyperbolic asymmetry parameter xi that sets the relative size of the nu_e and nu_mu mixing elements:
 *   |U_mu4|^2 = exp(-xi) * sqrt(sin^2(2*theta_mue)) / 2
 *   |U_e4|^2  = exp(+xi) * sqrt(sin^2(2*theta_mue)) / 2
 * A unitarity constraint sqrt(sin^2(2*theta_mue)) * cosh(xi) < 1 is enforced via model_constraint.
 * dmsq and sinsq2thmue are in log10 space; xi is linear.
 */
class PRO3p1_3C : public PROmodel {
public:
    /**
     * @brief Construct the PRO3p1_3C model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PRO3p1_3C(const PROpeller &prop, const std::map<std::string,int> &parameter_map);

    int UnitarityConstraint(const Eigen::VectorXf &v);

    float Pmue(float dmsq, float sinsq2thmue, float xi, float le) const;

    float Pmumu(float dmsq, float sinsq2thmue, float xi, float le) const;

    float Pee(float dmsq, float sinsq2thmue, float xi, float le) const;

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
    /** @brief Closed-form derivatives of get_probs (see PROmodel::get_probs_grad). */
    std::vector<Eigen::MatrixXf> get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
};

/**
 * @brief 3+1 sterile-neutrino model with invisible decay of the heavy mass eigenstate.
 * @details Extends the standard 3+1 picture by allowing the fourth mass eigenstate to decay into
 * invisible (non-interacting) particles with coupling strength g^2.  The oscillation probability
 * is modified by an exponential damping factor exp(-g^2 * Delta / (8 pi)) where
 * Delta = 1.267 * dmsq * L/E.  See arxiv:2204.00612 (IceCube) and PRD 110, 075002 for derivation.
 * Parameters: dmsq [log10], |U_e4|^2 [log10], |U_mu4|^2 [log10], g^2 [linear, >= 0].
 * A combined unitarity + positivity constraint is enforced via model_constraint.
 */
class PRO3p1_decay_invis : public PROmodel {
public:
    /**
     * @brief Construct the PRO3p1_decay_invis model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PRO3p1_decay_invis(const PROpeller &prop, const std::map<std::string,int> &parameter_map);

    int UnitarityConstraint(const Eigen::VectorXf &v);

    float Pmue(float dmsq, float Ue4sq, float Um4sq, float g2, float le) const;

    float Pmumu(float dmsq, float Ue4sq, float Um4sq, float g2, float le) const;

    float Pee(float dmsq, float Ue4sq, float Um4sq, float g2, float le) const;

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
    /** @brief Closed-form derivatives of get_probs (see PROmodel::get_probs_grad). */
    std::vector<Eigen::MatrixXf> get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
};

}

#endif
