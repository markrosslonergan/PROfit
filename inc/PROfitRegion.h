/**
 * @file PROfitRegion.h
 * @brief Fit only a subset of reco channels and evaluate the constrained prediction in the rest.
 * @author PROfit Collaboration
 *
 * @details A "fit region" restricts every chi2 to the collapsed bins of the selected channels
 * (e.g. sidebands), via the PROconfig active-bin mask that every metric snapshots at
 * construction (so FC/AFC workers and their regenerated pseudo-data inherit it too). Unlike
 * PROjector, the data outside the fit region is kept by default: the point is to compare the
 * fit-region-constrained prediction with it.
 *
 * After a fit (or directly at the CV for the plot subcommand), the "evaluation region"
 * (default: every channel outside the fit region) is tested with the Gaussian conditional
 * constraint. Linearising the prediction about the fit point theta_hat,
 *
 *     mu(theta) ~ mu(theta_hat) + J (theta - theta_hat)
 *
 * with Gaussian-prior splines theta ~ N(theta_0, P) and the covariance-type systematics C,
 * the prior-predictive is N(m, Sigma) with m = mu(theta_hat) + J (theta_0 - theta_hat) and
 * V = C + J P J^T (systematic part), Sigma = diag(stat) + V. Conditioning the evaluation
 * bins S on the fit-region data d_B (the usual constraint procedure) gives
 *
 *     m_S'  = m_S + V_SB Sigma_BB^-1 (d_B - m_B)
 *     V_SS' = V_SS - V_SB Sigma_BB^-1 V_BS
 *
 * and chi2 = (d_S - m_S')^T (diag(stat_S) + V_SS')^-1 (d_S - m_S') with ndof = |S| (minus one
 * per channel in shape-only mode), since the evaluation data never entered the fit. For a
 * linear model this equals chi2_min(joint fit) - chi2_min(fit-region fit). No covariance is
 * promoted to splines: the covariance systematics are constrained analytically.
 *
 * Physics parameters are held at the fit point (fixed at CV by default). Splines that were
 * fixed, or have a uniform prior (no Gaussian prior to condition), are held at the fit point
 * and their uncertainty is not propagated (logged).
 */
#ifndef PROFITREGION_H_
#define PROFITREGION_H_

#include <map>
#include <string>
#include <vector>

#include <Eigen/Eigen>

#include "PROconfig.h"
#include "PROdata.h"
#include "PROmodel.h"
#include "PROmetric.h"
#include "PROpeller.h"

class TObject;

namespace PROfit {

    /** @brief Command-line driven configuration for a fit-region run (plus the resolved channels). */
    struct FitRegionRunConfig {
        std::string fit_pattern;    ///< Non-empty => fit-region mode; unanchored regex on subchannel fullnames (whole channels).
        std::string eval_pattern;   ///< Channels to evaluate; empty => every channel outside the fit region.
        bool blind_eval = false;    ///< Zero the data outside the fit region (constrained prediction only, no chi2).
        bool float_physics = false; ///< Float physics parameters in the fit-region fit (default: fixed at CV).

        std::vector<size_t> fit_channels;  ///< Filled by FitRegionSetup: global channel indices in the fit.
        std::vector<size_t> eval_channels; ///< Filled by FitRegionSetup: global channel indices evaluated.

        bool active() const { return !fit_pattern.empty(); }
    };

    /**
     * @brief Resolve a subchannel pattern into whole channels.
     * @details Matches @p pattern against every subchannel fullname
     * ("<mode>_<detector>_<channel>_<subchannel>") as an unanchored regex. The chi2 lives in
     * collapsed (channel) space, so a channel whose subchannels are only partly matched is an
     * error. @p what names the option in log messages.
     * @return True if at least one channel matched and none were partly matched.
     */
    bool SelectChannelsByPattern(const PROconfig &config, const std::string &pattern,
                                 std::vector<size_t> &matched_channels, const std::string &what);

    /** @brief Every global channel index NOT in @p channels, in ascending order. */
    std::vector<size_t> ComplementChannels(const PROconfig &config, const std::vector<size_t> &channels);

    /** @brief "<mode>_<detector>_<channel>" for a global channel index. */
    std::string GlobalChannelName(const PROconfig &config, size_t global_channel_index);

    /** @brief 0/1 mask over the collapsed bins of variable @p var_index, 1 on @p channels. */
    Eigen::VectorXf CollapsedChannelMask(const PROconfig &config, const std::vector<size_t> &channels, size_t var_index);

    /** @brief Install @p channels as the PROconfig active-bin mask for every variable.
     *  Must run before any PROmetric is constructed. */
    void InstallFitRegion(PROconfig &config, const std::vector<size_t> &channels);

    /** @brief Zero every data bin outside @p channels (all variables) and refresh @p data.
     *  @return False if the remaining primary-variable data is empty. */
    bool MaskDataToChannels(const PROconfig &config, const std::vector<size_t> &channels,
                            std::vector<PROdata> &variable_data, PROdata &data, const std::string &what);

    /**
     * @brief One-stop fit-region hook called from PROfit.cxx.
     * @details Same placement rules as PROjectorSetup: after data creation and systematic
     * selection, before CVParams sizing, the bounds/--fix section and metric construction.
     * Resolves the fit/eval channels into @p frconf, installs the mask, optionally blinds
     * the data outside the fit region, and (unless float_physics) appends the physics
     * parameter names to @p fixed_params.
     * @return True on success; false means the caller should terminate.
     */
    bool FitRegionSetup(FitRegionRunConfig &frconf, PROconfig &config,
                        std::vector<PROdata> &variable_data, PROdata &data,
                        std::vector<std::string> &fixed_params, const PROmodel &model);

    /** @brief Per-channel evaluation-region goodness of fit. */
    struct FitRegionChannelGoF {
        size_t channel = 0;      ///< Global channel index.
        std::string name;        ///< GlobalChannelName.
        std::vector<Eigen::Index> rows;      ///< Rows of FitRegionConstraint::eval_bins belonging to this channel.
        std::vector<Eigen::Index> chi2_rows; ///< The subset of rows entering the chi2 (positive stat variance).
        long ndof = 0;
        float chi2_pre = 0, chi2_post = 0;
    };

    /** @brief Result of ComputeFitRegionConstraint. All vectors/matrices live on eval_bins. */
    struct FitRegionConstraint {
        std::vector<Eigen::Index> eval_bins;  ///< Collapsed bins of the eval channels (the chi2 uses those with positive stat variance, see chi2_rows).
        std::vector<Eigen::Index> chi2_rows;  ///< Rows of eval_bins entering the chi2.
        std::vector<Eigen::Index> fit_bins;   ///< Collapsed bins conditioned on (fit region, positive stat variance).
        Eigen::VectorXf data;                 ///< Observed data (zero when blinded).
        Eigen::VectorXf pre_mean, post_mean;  ///< Prediction before / after the fit-region constraint.
        Eigen::MatrixXf pre_syst_cov, post_syst_cov; ///< Systematic covariance (covariance systs + propagated splines).
        Eigen::VectorXf pre_stat, post_stat;  ///< Statistical variances (the metric's convention) at pre/post mean.
        float chi2_pre = 0, chi2_post = 0;
        long ndof = 0;
        float chi2_fit_linear = 0;            ///< Linearised fit-region chi2 (d_B - m_B)^T Sigma_BB^-1 (d_B - m_B); ~ the fit chi2 when the model is near linear.
        size_t n_splines_propagated = 0;      ///< Gaussian-prior splines whose uncertainty enters V.
        size_t n_splines_held = 0;            ///< Fixed or uniform-prior splines held at the fit point.
        bool blind = false;
        bool shape_only = false;
        std::vector<FitRegionChannelGoF> channels;
    };

    /**
     * @brief Compute the fit-region-constrained prediction and goodness of fit in the eval region.
     * @param metric The fit metric (supplies systematics, model, stat-variance convention, priors, bounds).
     * @param data   Collapsed data for the fitting variable (the same object the metric fits).
     * @param params Linearisation point: the fit-region best fit (global) or the CV (plot).
     * @param fixed  Per-parameter fixed flags (physics then splines); may be empty.
     */
    bool ComputeFitRegionConstraint(const PROconfig &config, const PROpeller &prop, const PROmetric &metric,
                                    const PROdata &data, const Eigen::VectorXf &params,
                                    const std::vector<int> &fixed, const FitRegionRunConfig &frconf,
                                    FitRegionConstraint &out);

    /** @brief Log the evaluation-region report. @p fit_chi2 < 0 omits the fit-region line. */
    void PrintFitRegionConstraint(const FitRegionConstraint &res, float fit_chi2, const PROndof *fit_ndof);

    /**
     * @brief Draw each evaluation channel: data, pre- and post-constraint prediction with
     * systematic bands, and a data/prediction ratio panel; multi-page PDF @p filename.
     * @return Named objects (histograms, covariance TH2Ds) for persistence in a ROOT file.
     */
    std::map<std::string, TObject *> PlotFitRegionConstraint(const PROconfig &config, const FitRegionConstraint &res,
                                                             const std::string &filename);

}

#endif
