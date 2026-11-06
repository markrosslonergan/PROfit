/**
 * @file PROmodelLBL3p1.h
 * @brief 3+1 (one sterile state) long-baseline oscillation model in vacuum.
 * @author PROfit Collaboration
 * @internal PROLBL3p1 is constructed only via get_model_from_string
 * (src/PROmodel.cxx); prefer the factory over direct construction.
 */
#ifndef PROMODELLBL3P1_H
#define PROMODELLBL3P1_H

#include "PROmodels/PROmodelLBL.h"

namespace PROfit {

/**
 * @brief Exact four-flavour 3+1 oscillations in vacuum for long-baseline configs
 * (tag `LBL_3+1-vacuum_angles`).
 * @details NuFast is three-flavour only, but vacuum 3+1 is closed form. The mixing matrix is
 *
 *     U = R34 R~24 R~14 R23 R~13 R12,
 *
 * with the PDG phase placement on R~ij ((R~ij)_ij = s_ij e^{-i delta}): delta_CP on R~13,
 * delta_14 on R~14, delta_24 on R~24. Hence |U_e4|^2 = s14^2, |U_mu4|^2 = c14^2 s24^2 and
 * |U_tau4|^2 = c14^2 c24^2 s34^2 (the SBL_3+1_angles relations), and U is PDG PMNS at
 * theta_i4 = 0, where columns 1-9 reproduce LBL_3nu-vacuum_angles. The probabilities are
 *
 *     P(a->b) = delta_ab - 4 sum_{i>j} Re W sin^2 X_ij + 2 sum_{i>j} Im W sin 2X_ij,
 *     W = U*_ai U_bi U_aj U*_bj,   X_ij = 1.266932679 dmsq_ij L/E  [eV^2, km, GeV].
 *
 * L/E is signed, so the CP-odd term flips for antineutrinos (E < 0, NuFast's convention).
 *
 * Kinematic inputs: exactly PROLBL's vacuum layouts (`L`+`E`, `E` + baseline=, or one
 * signed `L/E`; see PROLBL and resolve_lbl_inputs). E = 0 is the no-oscillation limit.
 *
 * Probability columns (model_rule): 0 = no oscillation; 1-9 = the active block in PROLBL's
 * layout, 1 + 3*from + to over (e, mu, tau) (1 = Pee, 2 = Pemu, 4 = Pmue, 5 = Pmumu, ...),
 * so a 3nu LBL config switches by tag alone; 10/11/12 = the NC weight 1 - P(a->s) for
 * a = e, mu, tau (NC is flavour-blind: only the sterile fraction is lost). H_combined keeps
 * only the columns some event's model_rule uses (prob_types), so on the binned grid
 * get_probs returns those columns in that order; per-event calls return all n_cols, since
 * their callers index by model_rule.
 *
 * Bin averaging (`<model bin_average=>`, default 1): FillSpectra's binned path evaluates
 * each truth bin at its midpoint, but dmsq_41 ~ 1 eV^2 at a far detector wraps the phase
 * many times per bin, so the midpoint value is aliased noise that can fake a signal. Every
 * oscillating term is low-pass filtered with its bin's phase window, keeping the midpoint
 * phase:
 *
 *     <sin^2 X> = (1 - cos 2X sinc h)/2,   <sin 2X> = sin 2X sinc h,
 *
 * h = half-range of 2X over the bin. h is exact (events uniform in L/E) for an `L/E`
 * variable and first order (midpoint derivative, as in the decay models) for E bins; the
 * `L`+`E` pair multiplies the L-bin and E-bin sincs. Slow oscillations are untouched
 * (sinc -> 1), fast ones average (sinc -> 0). Calls off the binned grid (event-by-event,
 * per-event subsets) are evaluated exactly per value. bin_average="0" samples the
 * midpoint like every other model.
 *
 * Parameters, sterile block first (the harmonic seed scan runs over physics parameter 0):
 * 0 dmsq_41 [eV^2, log10]; 1-3 sinsqt14, sinsqt24, sinsqt34 [log10, -inf = no mixing];
 * 4-5 delta_14, delta_24 [rad]; 6-11 PROLBL's 3nu set with the same names, bounds and
 * defaults (dmsq_21, dmsq_31, sinsqt12, sinsqt13, sinsqt23 linear; delta_CP [rad]).
 * get_probs_grad is closed form. Matter effects (incl. the NC potential on nu_s) are not
 * included.
 */
class PROLBL3p1 : public PROmodel {
public:
    static constexpr size_t n_cols = 13;  ///< Probability columns: null, 3x3 active, 3 NC.

    /**
     * @brief Construct the 3+1 vacuum LBL model.
     * @param config PROconfig; supplies the kinematic <parameter>s, binnings and the
     *               baseline=/bin_average= model options.
     * @param prop   MC event store; used to build H_combined.
     */
    PROLBL3p1(const PROconfig &config, const PROpeller &prop);

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;

    /** @brief Closed-form derivatives of get_probs (see PROmodel::get_probs_grad). */
    std::vector<Eigen::MatrixXf> get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;

    bool has_analytic_gradient() const override { return true; }

    bool wants_preconditioning() const override { return true; }

    std::vector<Eigen::VectorXf> degenerate_variants(const Eigen::VectorXf &phys) const override;

    LBLInputMode input_mode;  ///< Kinematic input layout.
    double baseline_km;       ///< Fixed baseline [km] (EOnly mode; 0 otherwise).
    bool bin_average;         ///< Low-pass filter the binned grid (see the class doc).

private:
    /// One evaluation point: signed L/E [km/GeV] and up to two half-widths of its L/E window.
    struct OscPoint { double le; double hw[2]; };

    std::vector<std::vector<float>> grid_;  ///< FillSpectra's binned midpoint grid, built identically.
    std::vector<OscPoint> grid_points_;     ///< Distinct points of grid_ (subchannel blocks repeat bins).
    std::vector<int> grid_index_;           ///< Flat grid index -> grid_points_ index.

    void build_grid(const PROconfig &config, const PROpeller &prop);
    double signed_le(double v0, double v1) const;
    std::vector<OscPoint> exact_points(const std::vector<std::vector<float>> &var_arrs) const;

    /**
     * @brief Probabilities (and optionally their gradient) at each point.
     * @param P  Out: (n_cols x n_points), one column per point.
     * @param dP If non-null, out: one (n_cols x n_points) matrix per physics parameter.
     */
    void evaluate(const Eigen::VectorXf &phys, const std::vector<OscPoint> &pts,
                  Eigen::MatrixXf &P, std::vector<Eigen::MatrixXf> *dP) const;

    /// Per-point columns -> get_probs' layout: (n_flat x prob_types) on the grid, (n x n_cols) off it.
    Eigen::MatrixXf expand(const Eigen::MatrixXf &P, bool on_grid) const;
};

}

#endif
