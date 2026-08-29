/**
 * @file PROmodel2flav.h
 * @brief Two-flavour-like 3+1 single-channel oscillation models (short-baseline approximation).
 * @author PROfit Collaboration
 * @internal These models are constructed only via
 * get_model_from_string (src/PROmodel.cxx); prefer the factory over direct construction.
 *
 * @details Declares the single-channel family:
 *   - PROnumudis     — nu_mu disappearance, parameterised by (dmsq, sin^2 2theta_mumu).
 *   - PROnumudisTEST — two-variable (L, E) version of PROnumudis for validation.
 *   - PROnueapp      — nu_mu → nu_e appearance, parameterised by (dmsq, sin^2 2theta_mue).
 *   - PROnuedis      — nu_e disappearance, parameterised by (dmsq, sin^2 2theta_ee).
 */
#ifndef PROMODEL2FLAV_H
#define PROMODEL2FLAV_H

#include "PROmodel.h"

namespace PROfit {

/**
 * @brief 3+1 sterile-neutrino nu_mu disappearance model in the short-baseline approximation.
 * @details Parameterises the two-flavour-like nu_mu survival probability as:
 *   P(nu_mu -> nu_mu) = 1 - sin^2(2*theta_mumu) * sin^2(1.267 * Delta m^2 * L/E)
 * where Delta m^2 is in eV^2 and L/E is in km/GeV.  Both parameters are stored in log10 space.
 * The model uses a single physics variable (L/E) identified by the "L/E" entry in parameter_map.
 */
class PROnumudis : public PROmodel {
public:
    /**
     * @brief Construct the PROnumudis model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PROnumudis(const PROpeller &prop, const std::map<std::string,int> &parameter_map);

    /**
     * @brief Compute the 3+1 nu_mu survival probability in the short-baseline approximation.
     * @details P(nu_mu -> nu_mu) = 1 - sin^2(2*theta_mumu) * sin^2(1.267 * Delta m^2 * L/E).
     * @param dmsq          Mass splitting Delta m^2 in eV^2 (may be in log10 space; converted internally).
     * @param sinsq2thmumu  sin^2(2 theta_mumu) (may be in log10 space; converted internally).
     * @param le            L/E ratio in km/GeV.
     * @return Survival probability in [0, 1].
     */
    float Pmumu(float dmsq, float sinsq2thmumu, float le) const;

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;

    /** @brief Closed-form derivatives of get_probs (see PROmodel::get_probs_grad). */
    std::vector<Eigen::MatrixXf> get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
};

/**
 * @brief Two-variable test version of PROnumudis operating on separate L and E variables.
 * @details Takes separate "L" and "E" variables from parameter_map and builds H_combined on the
 * 2D (L x E) physics grid.  get_probs() computes L/E internally, so the physics is identical
 * to PROnumudis.  The two models should produce identical spectra and can be used to validate
 * the multi-variable code path against the standard single L/E variable approach.
 */
class PROnumudisTEST : public PROmodel {
public:
    /**
     * @brief Construct the two-variable PROnumudisTEST model.
     * @param prop          MC event store; used to build H_combined on the (L, E) grid.
     * @param parameter_map Map from physics variable name to variable index in PROpeller.
     *                      Must contain both "L" and "E".
     */
    PROnumudisTEST(const PROpeller &prop, const std::map<std::string,int> &parameter_map);

    /**
     * @brief Compute the 3+1 nu_mu survival probability (identical physics to PROnumudis::Pmumu).
     * @param dmsq          Mass splitting Delta m^2 in eV^2 (may be in log10 space; converted internally).
     * @param sinsq2thmumu  sin^2(2 theta_mumu) (may be in log10 space; converted internally).
     * @param le            L/E ratio in km/GeV, computed internally from var_arrs[0]=L / var_arrs[1]=E.
     * @return Survival probability in [0, 1].
     */
    float Pmumu(float dmsq, float sinsq2thmumu, float le) const;

    /**
     * @brief Compute oscillation probabilities on the 2D (L×E) grid.
     * @details var_arrs[0] = L values, var_arrs[1] = E values, each of length n_L * n_E (flat row-major order).
     *          L/E is computed internally for each grid point, so the result is physically identical to PROnumudis::get_probs.
     * @param phys     Physics parameter vector: (log10(dmsq), log10(sinsq2thmm)).
     * @param var_arrs 2-element vector: {L array [km], E array [GeV]}, each of size n_phys_bins.
     * @return Matrix of shape (n_phys_bins, 2): column 0 = 1 (no-osc), column 1 = P(nu_mu -> nu_mu).
     */
    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
    /** @brief Closed-form derivatives of get_probs (see PROmodel::get_probs_grad). */
    std::vector<Eigen::MatrixXf> get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
};

/**
 * @brief 3+1 sterile-neutrino nu_mu → nu_e appearance model in the short-baseline approximation.
 * @details Parameterises the two-flavour-like nu_e appearance probability as:
 *   P(nu_mu -> nu_e) = sin^2(2*theta_mue) * sin^2(1.267 * Delta m^2 * L/E)
 * where Delta m^2 is in eV^2 and L/E is in km/GeV.  Both parameters are stored in log10 space.
 * The model uses a single physics variable (L/E) identified by the "L/E" entry in parameter_map.
 */
class PROnueapp : public PROmodel {
public:
    /**
     * @brief Construct the PROnueapp model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PROnueapp(const PROpeller &prop, const std::map<std::string,int> &parameter_map);

    /**
     * @brief Compute the 3+1 nu_mu → nu_e appearance probability in the short-baseline approximation.
     * @details P(nu_mu -> nu_e) = sin^2(2*theta_mue) * sin^2(1.267 * Delta m^2 * L/E).
     * @param dmsq         Mass splitting Delta m^2 in eV^2 (may be in log10 space; converted internally).
     * @param sinsq2thmue  sin^2(2 theta_mue) (may be in log10 space; converted internally).
     * @param le           L/E ratio in km/GeV.
     * @return Appearance probability in [0, 1].
     */
    float Pmue(float dmsq, float sinsq2thmue, float le) const;

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;

    /** @brief Closed-form derivatives of get_probs (see PROmodel::get_probs_grad). */
    std::vector<Eigen::MatrixXf> get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
};

/**
 * @brief 3+1 sterile-neutrino nu_e disappearance model in the short-baseline approximation.
 * @details Parameterises the two-flavour-like nu_e survival probability as:
 *   P(nu_e -> nu_e) = 1 - sin^2(2*theta_ee) * sin^2(1.267 * Delta m^2 * L/E)
 * where Delta m^2 is in eV^2 and L/E is in km/GeV.  Both parameters are stored in log10 space.
 * The model uses a single physics variable (L/E) identified by the "L/E" entry in parameter_map.
 */
class PROnuedis : public PROmodel {
public:
    /**
     * @brief Construct the PROnuedis model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PROnuedis(const PROpeller &prop, const std::map<std::string,int> &parameter_map);

    /**
     * @brief Compute the 3+1 nu_e survival probability in the short-baseline approximation.
     * @details P(nu_e -> nu_e) = 1 - sin^2(2*theta_ee) * sin^2(1.267 * Delta m^2 * L/E).
     * @param dmsq        Mass splitting Delta m^2 in eV^2 (may be in log10 space; converted internally).
     * @param sinsq2thee  sin^2(2 theta_ee) (may be in log10 space; converted internally).
     * @param le          L/E ratio in km/GeV.
     * @return Survival probability in [0, 1].
     */
    float Pee(float dmsq, float sinsq2thee, float le) const;

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;

    /** @brief Closed-form derivatives of get_probs (see PROmodel::get_probs_grad). */
    std::vector<Eigen::MatrixXf> get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
};

}

#endif
