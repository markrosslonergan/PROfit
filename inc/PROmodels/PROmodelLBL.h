/**
 * @file PROmodelLBL.h
 * @brief Standard three-flavour long-baseline oscillation model (NuFastLBL wrapper).
 * @author PROfit Collaboration
 * @internal PROLBL is constructed only via get_model_from_string
 * (src/PROmodel.cxx); prefer the factory over direct construction.
 */
#ifndef PROMODELLBL_H
#define PROMODELLBL_H

#include "PROmodel.h"

namespace PROfit {

/// Which kinematic inputs an LBL model's config supplied (see PROLBL).
enum class LBLInputMode { PairLE, EOnly, RatioLE };

/**
 * @brief Resolve the kinematic `<parameter>` layout shared by the LBL models (see PROLBL).
 * @param config      Supplies m_model_parameter_map and the baseline= option.
 * @param matter      Matter regime: a bare `L/E` is fatal (the MSW potential needs E).
 * @param ivars       Out: PROpeller variable indices, ordered (L, E), (E) or (L/E).
 * @param baseline_km Out: the fixed baseline [km] in EOnly mode (untouched otherwise).
 * @return The input layout; throws std::runtime_error on an invalid combination.
 */
LBLInputMode resolve_lbl_inputs(const PROconfig &config, bool matter, std::vector<int> &ivars, double &baseline_km);

/**
 * @brief Mass-ordering × θ23-octant mirrors of an LBL physics point (PROmodel::degenerate_variants).
 * @details Ordering: Δm²₃₁ → 2Δm²₂₁ − Δm²₃₁, which flips the sign of Δm²₃₂ and keeps its size.
 * Octant: sin²θ23 → 1 − sin²θ23. Returns the three non-identity combinations.
 */
std::vector<Eigen::VectorXf> lbl_ordering_octant_variants(const Eigen::VectorXf &phys, int i_dm21, int i_dm31, int i_s23);

/**
 * @brief Standard three-flavour long-baseline oscillation model, in matter or in vacuum.
 * @details Wraps the NuFastLBL library (Denton, arXiv:2405.02400) to compute the full 3x3
 * oscillation probability matrix. Two model tags share this class:
 *  - `LBL_3nu-matter_angles` (legacy alias `LBL`): constant-density matter oscillations
 *    via NuFastLBL::Probability_Matter_LBL.
 *  - `LBL_3nu-vacuum_angles`: vacuum oscillations via NuFastLBL::Probability_Vacuum_LBL
 *    (cheaper, and finite for every dmsq_31 including the Dmsqee=0 crossing where the
 *    matter solver is singular).
 *
 * Kinematic inputs (event-by-event `<variable>`s, like every oscillation model);
 * E is always signed, negative = antineutrino (NuFast's convention):
 *  - `<parameter name="L">` [km] + `name="E"` [GeV]: exact per-event kinematics, for a
 *    baseline that varies across the MC. The evaluation grid is n_L x n_E GLOBAL bins,
 *    so keep both truth binnings lean. `baseline=` is a fatal conflict here.
 *  - `name="E"` alone + the `baseline=` attribute [km, REQUIRED in this mode]: fixed
 *    baseline, 1D grid over E — the cheap form most experiments want.
 *  - A single signed `name="L/E"` [km/GeV]: vacuum only, since the matter potential
 *    scales with E alone and the ratio underdetermines it. `baseline=` is fatal here.
 *  - E == 0 (or L/E == 0) is treated as the no-oscillation limit (identity probabilities).
 * NOTE (v3.0.4-dev): before this version the model took one `"L/E"` parameter that was
 * silently passed to NuFast as E [GeV] at a fixed L = 1300 km — older LBL configs must
 * migrate to the inputs above.
 *
 * Matter-only `<model>` attributes (the vacuum tag warns and ignores them; defaults
 * are the legacy hardcoded values):
 *  - `density=` [g/cm^3, >= 0, default 3 — note 3 is the legacy value, not DUNE's 2.848],
 *  - `electron_fraction=` [(0,1], default 0.5],
 *  - `n_newton=` [int 0-10, default 0; NuFast recommends 1 for many-year DUNE/HK precision].
 *
 * Parameters: dmsq_21, dmsq_31 [eV^2, linear], sin^2(theta_12), sin^2(theta_13),
 * sin^2(theta_23) [linear], delta_CP [rad, linear]. None are stored in log10 space.
 * The matter regime sets a model_constraint rejecting the narrow dmsq_31 band around
 * 0 and around Dmsqee = 0 (dmsqee_guard) where NuFast's DMP solver is singular; the
 * vacuum solver has no such band.
 */
class PROLBL : public PROmodel {
public:
    /// Which kinematic inputs the config supplied (see the class doc).
    using InputMode = LBLInputMode;

    // Legacy hardcoded values, kept as the attribute defaults.
    static constexpr double default_density = 3.0;            ///< g/cm^3 (legacy; DUNE uses 2.848).
    static constexpr double default_electron_fraction = 0.5;  ///< Electron fraction Ye.
    static constexpr int default_n_newton = 0;                ///< NuFast Newton iterations.
    /// Half-width [eV^2] of the excluded band around dmsq_31 = 0 and Dmsqee = 0 where
    /// NuFast's matter solver is singular; enforced by model_constraint (matter only).
    static constexpr float dmsqee_guard = 5e-5f;

    /**
     * @brief Construct the PROLBL model.
     * @param config PROconfig; supplies m_model_parameter_map ("L"+"E", or "L/E" for
     *               vacuum) and the optional matter m_model_options.
     * @param prop   MC event store; used to build H_combined.
     * @param matter true for LBL_3nu-matter_angles, false for LBL_3nu-vacuum_angles.
     */
    PROLBL(const PROconfig &config, const PROpeller &prop, bool matter);

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;

    bool wants_preconditioning() const override { return true; }

    std::vector<Eigen::VectorXf> degenerate_variants(const Eigen::VectorXf &phys) const override {
        return lbl_ordering_octant_variants(phys, 0, 1, 4);
    }

    bool matter;              ///< Matter (NuFast Probability_Matter_LBL) vs vacuum regime.
    InputMode input_mode;     ///< Kinematic input layout.
    double baseline_km;       ///< Fixed baseline [km] (EOnly mode; unset otherwise).
    double density;           ///< Matter density [g/cm^3] (matter regime only).
    double electron_fraction; ///< Electron fraction Ye (matter regime only).
    int n_newton;             ///< NuFast Newton-iteration count (matter regime only).

private:
    /**
     * @brief Fill the 3x3 oscillation probability matrix at one kinematic point.
     * @param params Physics vector (6 params).
     * @param L      Baseline [km] (in RatioLE mode, the signed L/E with E = 1).
     * @param E      True energy [GeV], negative = antineutrino. 0 = no-osc limit.
     * @param out    Row-major [from][to] probabilities in flavour order (e, mu, tau).
     */
    void oscillate(const Eigen::VectorXf &params, float L, float E, float (*out)[3][3]) const;
};

}

#endif
