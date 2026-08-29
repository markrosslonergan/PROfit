/**
 * @file PROplot.h
 * @brief Plotting utilities for PROfit spectra, covariance matrices, error bands, and MCMC posteriors.
 * @author PROfit Collaboration
 *
 * @details Provides functions and helper types for producing publication-quality ROOT-based
 * plots from PROfit analysis outputs.  Key capabilities include:
 *   - Stacked subchannel spectra with pre- and post-fit error bands (plot_channels),
 *   - Detector-ratio comparison plots (plot_detector_ratios),
 *   - Fractional systematic breakdown bar charts,
 *   - MCMC-derived posterior error bands (getMCMCErrorBand),
 *   - Spline graphs and covariance/correlation matrix heatmaps.
 *
 * Also defines PlotBounds (axis range control) and PlotOptions (bitmask flags for plot style).
 */
#ifndef PROPLOT_H
#define PROPLOT_H

// C++ include 
#include <algorithm>
#include <unordered_map>
#include <string>
#include <iomanip>
// PROfit include 
#include "PROlog.h"
#include "PROconfig.h"
#include "PROspec.h"
#include "PROsyst.h"
#include "PROMCMC.h"
#include "PROtocall.h"
#include "PROseed.h"
#include "PROcess.h"
#include "PROversion.h"

// Root includes
#include "TAttLine.h"
#include "TAttMarker.h"
#include "THStack.h"
#include "TStyle.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TGraph.h"
#include "TGraphAsymmErrors.h"
#include "TGraphErrors.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TRatioPlot.h"
#include "TPaveText.h"
#include "TTree.h"
#include "TLine.h"
namespace PROfit{

    /**
     * @brief Axis range control for PROfit plots.
     * @details Default value of -9999 for any field means "let ROOT auto-range that axis".
     * Use Load() to populate from a named key-value map (e.g. from command-line options).
     */
    struct PlotBounds {
        float xmin   = -9999; ///< Minimum x-axis value (-9999 = auto).
        float xmax   = -9999; ///< Maximum x-axis value (-9999 = auto).
        float ymin   = -9999; ///< Minimum y-axis value (-9999 = auto).
        float ymax   = -9999; ///< Maximum y-axis value (-9999 = auto).
        float ratmin = -9999; ///< Minimum ratio-panel y value (-9999 = auto).
        float ratmax = -9999; ///< Maximum ratio-panel y value (-9999 = auto).

        int Load(std::map<std::string, float> bound_list){
            log<LOG_INFO>(L"%1% || Loading Bounds for plot_channels ") % __func__;
            for(const auto &[bound_name, value]: bound_list) {
                log<LOG_INFO>(L"%1% || --on bound %2% val %3% ") % __func__ % bound_name.c_str() % value;
                if(bound_name == "xmin") {
                    xmin=value;
                }else if(bound_name == "xmax") {
                    xmax=value;
                }else if(bound_name == "ymin") {
                    ymin=value;
                }else if(bound_name == "ymax") {
                    ymax=value;
                }else if(bound_name == "ratmin") {
                    ratmin=value;
                }else if(bound_name == "ratmax") {
                    ratmax=value;
                }else{
                    log<LOG_ERROR>(L"%1% || ERROR! you passed a plot-bounds string that is not allowed (%2%). Needs to be xmin,xmax,ymin,ymax,ratmin,ratmax. ") % __func__ % bound_name.c_str();
                    throw std::invalid_argument(std::string("Invalid plot-bounds ")+bound_name );
                }
            }
        return 1;
        };
        bool hasBound(std::string bound_name) const {
                if(bound_name == "xmin") {
                    return xmin!=-9999? true : false;
                }else if(bound_name == "xmax") {
                    return xmax!=-9999? true : false;
                }else if(bound_name == "ymin") {
                    return ymin!=-9999? true : false;
                }else if(bound_name == "ymax") {
                    return ymax!=-9999? true : false;
                }else if(bound_name == "ratmin") {
                    return ratmin!=-9999? true : false;
                }else if(bound_name == "ratmax") {
                    return ratmax!=-9999? true : false;
                }else{
                    log<LOG_ERROR>(L"%1% || ERROR! you passed a plot-bounds string that is not allowed (%2%). Needs to be ymax,ratmin,ratmax. ") % __func__ % bound_name.c_str();
                    throw std::invalid_argument(std::string("Invalid plot-bounds ")+bound_name );
                }
        return false;
        };
        float getBound(std::string bound_name) const {
                if(bound_name == "xmin") {
                    return xmin;
                }else if(bound_name == "xmax") {
                    return xmax;
                }else if(bound_name == "ymin") {
                    return ymin;
                }else if(bound_name == "ymax") {
                    return ymax;
                }else if(bound_name == "ratmin") {
                    return ratmin;
                }else if(bound_name == "ratmax") {
                    return ratmax;
                }else{
                    log<LOG_ERROR>(L"%1% || ERROR! you passed a plot-bounds string that is not allowed (%2%). Needs to be ymax,ratmin,ratmax. ") % __func__ % bound_name.c_str();
                    throw std::invalid_argument(std::string("Invalid plot-bounds ")+bound_name );
                }
        return -999;
        };
    };
    
    /**
     * @brief Bitmask flags controlling the style and content of plot_channels() output.
     * @details Flags can be combined with operator|.  Example:
     * @code
     *   plot_channels(..., PlotOptions::BinWidthScaled | PlotOptions::DataMCRatio);
     * @endcode
     */
    enum class PlotOptions {
        Default          = 0,       ///< Standard stacked histogram with no special options.
        CVasStack        = 1 << 0,  ///< Draw the CV prediction as a stacked (filled) histogram.
        AreaNormalized   = 1 << 1,  ///< Normalise all histograms to unit area before plotting.
        BinWidthScaled   = 1 << 2,  ///< Divide bin contents by bin width (events/unit).
        DataMCRatio      = 1 << 3,  ///< Show a data/MC ratio panel below the main plot.
        DataPostfitRatio = 1 << 4,  ///< Show a data/post-fit ratio panel below the main plot.
    };

    inline PlotOptions operator|(PlotOptions a, PlotOptions b) {
        return static_cast<PlotOptions>(static_cast<int>(a) | static_cast<int>(b));
    }

    inline PlotOptions operator|=(PlotOptions &a, PlotOptions b) {
        return a = a | b;
    }

    inline PlotOptions operator&(PlotOptions a, PlotOptions b) {
        return static_cast<PlotOptions>(static_cast<int>(a) & static_cast<int>(b));
    }

    inline PlotOptions operator&=(PlotOptions &a, PlotOptions b) {
        return a = a & b;
    }

    /** @brief Set a custom colour palette for 2D matrix plots (correlation/covariance). */
    void set_matrix_palette();

    /**
     * @brief Plot the spectrum of the ratio between each pair of detectors, one page per channel.
     * @param c                Canvas printing into an already-open multipage PDF.
     * @param config           Analysis configuration.
     * @param cv_coll          Collapsed CV prediction.
     * @param bf_coll          Optional collapsed best-fit prediction.
     * @param data_coll        Optional collapsed data spectrum.
     * @param errband          Optional pre-fit error band, whose covariance propagates the inter-detector correlation.
     * @param posterrband      Optional post-fit error band, used with bf_coll.
     * @param channel_offsets  Collapsed bin start of each mode/detector/channel, in loop order.
     * @param filename         Output PDF filename.
     * @param other_index      Variable index (default 0).
     */
    void plot_detector_ratio_spectra(TCanvas &c, const PROconfig &config, const Eigen::VectorXf &cv_coll, const std::optional<Eigen::VectorXf> &bf_coll, const std::optional<Eigen::VectorXf> &data_coll, const std::optional<PROerrorbar> &errband, const std::optional<PROerrorbar> &posterrband, const std::vector<size_t> &channel_offsets, const std::string &filename, int other_index = 0);
    void plot_channel_ratio_spectra(TCanvas &c, const PROconfig &config, const Eigen::VectorXf &cv_coll, const std::optional<Eigen::VectorXf> &bf_coll, const std::optional<Eigen::VectorXf> &data_coll, const std::optional<PROerrorbar> &errband, const std::optional<PROerrorbar> &posterrband, const std::vector<size_t> &channel_offsets, const std::string &filename, int other_index = 0, PlotOptions opt = PlotOptions{});

    /**
     * @brief Produce a multi-panel detector ratio comparison plot.
     * @param config       Analysis configuration.
     * @param data_hists   Data histograms, one per channel.
     * @param cv_hists     CV prediction histograms, one per channel.
     * @param errband      Optional pre-fit error band.
     * @param bf_hists     Best-fit prediction histograms, one per channel.
     * @param posterrband  Optional post-fit error band.
     * @param pre_corr     Pre-fit correlation matrix (TH2D) for labelling.
     * @param post_corr    Post-fit correlation matrix (TH2D) for labelling.
     * @param filename     Output PDF filename.
     * @param var_index    Variable index (default 0).
     */
    void plot_detector_ratios(const PROconfig &config, std::vector<TH1D> data_hists,  std::vector<TH1D> cv_hists, std::optional<PROerrorbar> errband, std::vector<TH1D> bf_hists, std::optional<PROerrorbar> posterrband, TH2D &pre_corr, TH2D &post_corr, std::string filename, int var_index = 0);

    /**
     * @brief Produce a stacked spectrum plot for all channels.
     * @param filename     Output filename (ROOT or PDF).
     * @param config       Analysis configuration.
     * @param cv           Optional CV prediction spectrum.
     * @param best_fit     Optional best-fit prediction spectrum.
     * @param data         Optional observed data spectrum.
     * @param errband      Optional pre-fit error band.
     * @param posterrband  Optional post-fit error band.
     * @param texts        Vector of TPaveText annotation boxes to overlay.
     * @param bounds       Axis range settings.
     * @param opt          Bitmask of PlotOptions flags.
     * @param var_index    Variable index (default 0).
     * @param ratio_bool   If true, add a ratio panel.
     * @param plot_channel_ratios   If true, plot the ratios across different channels.
     * @param skip_stack_subchannels  Optional global subchannel indices to omit from the
     *        CV stack and legend (used by --bkg-subtract; their contents are expected to
     *        already be zero in `cv`).
     */
    std::map<std::string, TObject *> plot_channels(const std::string &filename, const PROconfig &config, std::optional<PROspec> cv, std::optional<PROspec> best_fit, std::optional<PROdata> data, std::optional<PROerrorbar> errband, std::optional<PROerrorbar> posterrband, std::vector<TPaveText> &texts, const PlotBounds &bounds, PlotOptions opt = PlotOptions::Default, int var_index = 0, bool ratio_bool = false, bool plot_channel_ratios = false, const std::vector<size_t> *skip_stack_subchannels = nullptr, PROmetric *chi_metric = nullptr, const PROspec *chi_spec = nullptr);

    /**
     * @brief Return global subchannel indices whose `m_fullnames[i]` contains `pattern` as a substring.
     * @details Matches PROsyst's wildcard convention — substring match used by
     * CreateFlatMatrix in src/PROsyst.cxx. Useful for picking out a set of
     * "background" subchannels by name (e.g. "numu_bkg" matches every
     * detector's *_numu_bkg subchannel). Empty pattern returns an empty list.
     * @param config   Analysis configuration (uses config.m_fullnames).
     * @param pattern  Substring to match against each full subchannel name.
     * @return Vector of global subchannel indices matching the pattern; empty if none.
     */
    std::vector<size_t> find_subchannels_by_pattern(const PROconfig &config,
                                                    const std::string &pattern);

    /**
     * @brief Build a full-bin 0/1 indicator vector for the bins owned by
     * `matched_subchannel_indices`.
     * @details The returned vector has size config.m_num_variable_bins_total[var_index],
     * with 1 in every bin belonging to a matched subchannel and 0 elsewhere. Variable
     * index controls bin-start lookup via config.GetGlobalVariableBinStart.
     * @param config                       Analysis configuration.
     * @param matched_subchannel_indices   Global subchannel indices to mark.
     * @param var_index                    Variable index for bin-range lookup.
     * @return Full-bin 0/1 indicator vector.
     */
    Eigen::VectorXf build_subchannel_bin_mask(const PROconfig &config,
                                              const std::vector<size_t> &matched_subchannel_indices,
                                              int var_index);

    /**
     * @brief Build a full-bin vector that copies `spec`'s values only in the bins
     * owned by `matched_subchannel_indices`, zero elsewhere.
     * @details Thin wrapper over build_subchannel_bin_mask: returns
     * mask.cwiseProduct(spec.Spec()). Used by the --bkg-subtract plot path to mask
     * out only the bkg subchannel bins on the full-bin PROspec.
     * @param config                       Analysis configuration.
     * @param spec                         Source full-bin spectrum.
     * @param matched_subchannel_indices   Global subchannel indices to retain.
     * @param var_index                    Variable index for bin-range lookup.
     * @return Full-bin vector with spec's values in the matched subchannels' bins, 0 elsewhere.
     */
    Eigen::VectorXf build_subchannel_mask_spec(const PROconfig &config,
                                               const PROspec &spec,
                                               const std::vector<size_t> &matched_subchannel_indices,
                                               int var_index);

    /**
     * @brief Return a map of subchannel-name to 1D ROOT histogram from a PROspec.
     * @param spec       Input spectrum.
     * @param inconfig   Analysis configuration.
     * @param scale      If true, divide by bin width.
     * @param var_index  Variable index.
     * @return Map from subchannel full name to unique TH1D.
     */
    std::map<std::string, std::unique_ptr<TH1D>> getCV1DHists(const PROspec & spec, const PROconfig& inconfig, bool scale = false, int var_index = 0);

    /**
     * @brief Return a map of subchannel-name to 2D ROOT histogram from a PROspec.
     * @param spec       Input spectrum.
     * @param inconfig   Analysis configuration.
     * @param scale      If true, divide by bin area.
     * @param var_index  Variable index.
     * @return Map from subchannel full name to unique TH2D.
     */
    std::map<std::string, std::unique_ptr<TH2D>> getCV2DHists(const PROspec & spec, const PROconfig& inconfig, bool scale = false, int var_index = 0);

    /**
     * @brief Return per-systematic covariance and correlation matrix TH2D histograms.
     * @param syst    The PROsyst containing all systematics.
     * @param config  Analysis configuration.
     * @param cv      CV spectrum used to convert fractional to absolute covariance.
     * @return Map from systematic name to unique TH2D.
     */
    std::map<std::string, std::unique_ptr<TH2D>> covarianceTH2D(const PROsyst &syst, const PROconfig &config, const PROspec &cv);

    /**
     * @brief Return TGraph pairs (CV ± 1 sigma) for all spline systematics.
     * @param systs   The PROsyst containing all splines.
     * @param config  Analysis configuration.
     * @return Map from spline name to a vector of (down, up) TGraph pairs per bin.
     */
    std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> getSplineGraphs(const PROsyst &systs, const PROconfig &config);

    /**
     * @brief Compute a pre-fit error band from systematic throws.
     * @param config      Analysis configuration.
     * @param prop        MC event store.
     * @param syst        Systematic object.
     * @param model       Physics model.
     * @param cv_spec     CV predicted spectrum.
     * @param cvparams    CV physics parameter vector.
     * @param scale       If true, divide errors by bin width.
     * @param other_index Variable index.
     * @return PROerrorbar with asymmetric per-bin uncertainties.
     */
    PROerrorbar getErrorBand(const PROconfig &config, const PROpeller &prop, const PROsyst &syst, const PROmodel &model, const PROspec &cv_spec, const Eigen::VectorXf &cvparams,bool scale=false, int other_index=0);

    /**
     * @brief Compute an error band analytically from the covariance-type systematics alone.
     * @details Exact replacement for getMCMCErrorBand when the sampling chain has zero
     * free parameters (no spline nuisances, or all of them --fix'd): the chain never
     * moves, so its band reduces to Gaussian throws of the collapsed scaled covariance
     * around a single spectrum, whose 16/84 quantiles are exactly ±sqrt(diag).  Symmetric
     * by construction; `covariance` is the collapsed scaled covariance itself.
     * @param config    Analysis configuration.
     * @param prop      MC event store.
     * @param syst      Systematic object (only covariance-type systs contribute).
     * @param model     Physics model.
     * @param params    Parameter vector at which to evaluate the spectrum (e.g. best fit).
     * @param scale     If true, divide errors by bin width.
     * @param var_index Variable index.
     * @return PROerrorbar with symmetric per-bin uncertainties and the bin covariance.
     */
    /** @brief Analytic error band from the summed covariance (no MCMC). With an empty
     *  @p data_spec this is the prior band sqrt(diag(Sigma)) about the prediction. When a
     *  collapsed data spectrum is given, the band is the data-constrained posterior of the
     *  covariance systematics (Putnam SBN note Eqs. 7-8): center shifted by
     *  Sigma(C+Sigma)^-1 u and covariance Sigma - Sigma(C+Sigma)^-1 Sigma, restricted to
     *  active bins with data>0 (PROchi convention, C = diag(max(data,1))). Exact when the
     *  covariance systs are the only free parameters (the post-fit degenerate-chain path). */
    PROerrorbar getCovarianceOnlyErrorBand(const PROconfig &config, const PROpeller &prop, const PROsyst &syst, const PROmodel &model, const Eigen::VectorXf &params, bool scale=false, int var_index=0, const Eigen::VectorXf &data_spec = Eigen::VectorXf());

    /**
     * @brief Result of getErrorBandBkgSubtracted: a signal-only error band plus the
     * background pieces needed to correct the data points.
     * @details `band` follows getErrorBand's conventions (error_up/down/point bin-width
     * scaled if requested; covariance unscaled). The three bkg vectors are deliberately
     * UNSCALED counts because they feed PROdata, which is width/area scaled later inside
     * plot_channels.
     */
    struct PROsubtractedErrorBand {
        PROerrorbar band;                         ///< Signal-only band; error_point = (CV - bkg_CV), bin-width scaled if scale.
        Eigen::VectorXf bkg_cv_collapsed;         ///< Collapsed bkg CV, unscaled counts.
        Eigen::VectorXf bkg_sigma_collapsed;      ///< Per-bin sqrt(Var) of the bkg systematic throws, unscaled.
        Eigen::VectorXf bkg_mcstat_var_collapsed; ///< Collapsed sum of squared MC-stat errors in bkg bins, unscaled.
        PROsubtractedErrorBand(size_t n) : band(n), bkg_cv_collapsed(Eigen::VectorXf::Zero(n)),
            bkg_sigma_collapsed(Eigen::VectorXf::Zero(n)), bkg_mcstat_var_collapsed(Eigen::VectorXf::Zero(n)) {}
    };

    /**
     * @brief Background-subtracted pre-fit error band, publication convention.
     * @details Each of the 2500 systematic throws is split into signal and background
     * pieces (FillSystRandomThrowSplit) and the throw's OWN background is subtracted, so
     * background systematic variations cancel out of the band: the band shows
     * signal-only systematics around the subtracted CV. The background's per-bin
     * systematic sigma (from the same throws) and MC-stat variance are returned so the
     * caller can inflate the data errors to sqrt(N + sigma_bkg_syst^2 + sigma_bkg_MCstat^2).
     * Note: the signal-background systematic correlation is retained in the band (via the
     * per-throw cancellation) but discarded in the data error — inherent to this convention.
     * @param config          Analysis configuration.
     * @param prop            MC event store.
     * @param syst            Systematic object.
     * @param model           Physics model.
     * @param cv_spec         CV predicted spectrum (full binning, UNsubtracted).
     * @param cvparams        CV physics parameter vector.
     * @param bkg_subchannels Global subchannel indices to subtract (from find_subchannels_by_pattern).
     * @param scale           If true, divide band errors by bin width.
     * @param other_index     Variable index.
     * @return PROsubtractedErrorBand (see struct docs).
     */
    PROsubtractedErrorBand getErrorBandBkgSubtracted(const PROconfig &config, const PROpeller &prop, const PROsyst &syst, const PROmodel &model, const PROspec &cv_spec, const Eigen::VectorXf &cvparams, const std::vector<size_t> &bkg_subchannels, bool scale=false, int other_index=0);

    /**
     * @brief Produce a bar chart showing fractional prior uncertainty per systematic.
     * @param config       Analysis configuration.
     * @param spec         CV spectrum used for fractional normalisation.
     * @param allsplinesyst PROsyst containing all spline systematics.
     * @param filename     Output filename.
     * @param var_index    Variable index (default 0).
     * @return 0 on success.
     */
    int plotPriorFractionalSystematicBreakdown(const PROconfig &config, const PROspec &spec, const PROsyst &allsplinesyst, std::string filename, int var_index = 0);

    /**
     * @brief Produce a ratio plot of prior fractional uncertainties across systematics.
     * @param config       Analysis configuration.
     * @param spec         CV spectrum.
     * @param allsplinesyst PROsyst containing all spline systematics.
     * @param filename     Output filename.
     * @param other_index  Variable index.
     * @return 0 on success.
     */
    int plotPriorFractionalSystematicRatios(const PROconfig &config, const PROspec &spec, const PROsyst &allsplinesyst, std::string filename, int other_index);
    int plotPriorFractionalSystematicChannelRatios(const PROconfig &config, const PROspec &spec, const PROsyst &allsplinesyst, std::string filename, int other_index);

    /**
     * @brief Produce a multi-page diagnostic PDF for every covariance_to_spline systematic.
     * @details For each parent systematic the PDF includes: a summary page, the original vs.
     * reconstructed fractional covariance with residual, the per-bin fractional uncertainty
     * (sqrt of the diagonal) original vs. reconstructed, the eigenvalue scree plot and the
     * cumulative variance, an eigenvector heatmap of the kept modes, per-knob bin-response
     * panels, per-knob CV ± 1σ bands, and an aggregate CV band built from the original
     * covariance vs. summed over the synthesized knobs.  Only writes the file if at least
     * one covariance_to_spline systematic was processed.
     * @param config    Analysis configuration.
     * @param cv        CV spectrum (used for the ±1σ band overlays).
     * @param syst      PROsyst whose cov2spline_debug_info map will be iterated.
     * @param filename  Output PDF path.
     * @param var_index Variable (binning) index.
     * @return 0 on success, 1 if there were no covariance_to_spline systematics.
     */
    int plotCov2SplineChecks(const PROconfig &config, const PROspec &cv, const PROsyst &syst, const std::string &filename, int var_index);

    /**
     * @brief Compute a posterior error band using Markov Chain Monte Carlo sampling.
     * @details Runs the Metropolis algorithm for @p burnin + @p iterations steps.  At each
     * accepted step after burn-in, the corresponding spectrum is computed and accumulated;
     * asymmetric 1-sigma error bars are derived from the 16th and 84th percentiles of the
     * per-bin sample distributions.  Posterior distributions for each spline parameter and
     * the per-bin histogram covariance are also accumulated.
     * @tparam T  Metropolis proposal distribution type.
     * @tparam P  Metropolis target density type.
     * @param met        Metropolis sampler object.
     * @param burnin     Number of burn-in steps to discard.
     * @param iterations Number of post-burn-in steps to keep.
     * @param config     Analysis configuration.
     * @param prop       MC event store.
     * @param metric     The PROmetric providing access to model and systematics.
     * @param best_fit   Best-fit parameter vector (used as the starting point and as reference).
     * @param posteriors Output vector of TH1D histograms for each spline nuisance parameter.
     * @param post_covar Output post-fit parameter covariance matrix (splines only).
     * @param scale      If true, divide error bars by bin width.
     * @param var_index  Variable index (default 0).
     * @param data_spec  Optional collapsed data spectrum. When empty (default), the band
     *                   includes prior throws of the covariance-type systematics. When given,
     *                   each MCMC sample instead receives the data-constrained posterior pull
     *                   of the covariance systematics (G. Putnam, "How to Obtain Pull Terms
     *                   for Systematic Uncertainties Embedded in a Covariance Matrix", SBN
     *                   note, May 2026), shrinking the post-fit band. center_shift then
     *                   reports the ANALYTIC pull Sigma(C+Sigma)^-1 (d - cv) at the best
     *                   fit (exactly 0 for Asimov), not the sample median. Pass this only
     *                   for post-fit bands; the pre-fit/prior band must stay unconstrained.
     * @return PROerrorbar with per-bin asymmetric uncertainties and the histogram covariance.
     */
    template<class T, class P>
        PROerrorbar getMCMCErrorBand(Metropolis<T, P> met, size_t burnin, size_t iterations, const PROconfig &config, const PROpeller &prop, PROmetric &metric, const Eigen::VectorXf &best_fit, std::vector<TH1D> &posteriors, Eigen::MatrixXf &post_covar, Eigen::VectorXf &param_err_lo, Eigen::VectorXf &param_err_hi, bool scale = false, int var_index=0, PROgressBar *pbar = nullptr, const Eigen::VectorXf &data_spec = Eigen::VectorXf()) {
            for(size_t i = 0; i < metric.GetSysts().GetNSplines(); ++i)
                posteriors.emplace_back("", (";"+config.m_mcgen_variation_plotname_map.at(metric.GetSysts().spline_names[i])).c_str(), 60, -3, 3);

            Eigen::VectorXf cv = FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), best_fit, true, var_index).Spec();

            for (int i = 0; i < cv.size(); ++i) {
                if (cv(i) <= 0.0f) {
                    cv(i) = 1e-6f; // Floor zero-count / inactive subchannels
                }
            }

            Eigen::VectorXf cv_coll = CollapseMatrix(config, cv);
            // The data constraint only makes sense in this variable's collapsed
            // space; a mismatched spectrum (e.g. from another variable) would
            // index out of range below. Ignore it loudly rather than crash.
            bool use_data = data_spec.size() != 0;
            if(use_data && data_spec.size() != cv_coll.size()) {
                log<LOG_ERROR>(L"%1% || data_spec has %2% bins but variable %3% has %4% collapsed bins; ignoring the data constraint.") % __func__ % data_spec.size() % var_index % cv_coll.size();
                use_data = false;
            }
            Eigen::MatrixXf L;
            if(metric.GetSysts().GetNCovar() > 0) L = metric.GetSysts().DecomposeFractionalCovariance(config, cv);
            else L = Eigen::MatrixXf::Zero(config.m_num_variable_bins_total_collapsed[var_index], config.m_num_variable_bins_total_collapsed[var_index]);
            std::normal_distribution<float> nd;
            Eigen::VectorXf throws = Eigen::VectorXf::Constant(config.m_num_variable_bins_total_collapsed[var_index], 0);

            int nspline = metric.GetSysts().GetNSplines();
            int nphys = metric.GetModel().nparams;
            Eigen::VectorXf splines_bf = best_fit.segment(nphys, nspline);
            post_covar = Eigen::MatrixXf::Constant(nspline, nspline, 0);
            Eigen::MatrixXf post_hist_covar = Eigen::MatrixXf::Constant(cv_coll.size(), cv_coll.size(), 0);
            Eigen::VectorXf hist_diff_sum = Eigen::VectorXf::Zero(cv_coll.size());
            size_t nsteps = 0;
            std::vector<Eigen::VectorXf> specs;
            std::vector<std::vector<float>> param_samples(nspline);

	    std::function<void(const Eigen::VectorXf&)> action;
	    
            if (!use_data){
	        action = [&](const Eigen::VectorXf &value) {
                nsteps += 1;
		for(size_t i = 0; i < config.m_num_variable_bins_total_collapsed[var_index]; ++i)
                        throws(i) = nd(PROseed::global_rng);
                specs.push_back(CollapseMatrix(config, FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), value, true,var_index).Spec())+L*throws);
                for(int i = 0; i < nspline; ++i) {
                    posteriors[i].Fill(value(i+nphys));
                    param_samples[i].push_back(value(i+nphys));
                }
                Eigen::VectorXf splines = value.segment(nphys, nspline);
                Eigen::VectorXf diff = splines-splines_bf;
                post_covar += diff * diff.transpose();
		Eigen::VectorXf diff_hist = specs.back() - cv_coll;
                hist_diff_sum += diff_hist;
                post_hist_covar += diff_hist * diff_hist.transpose();
                };
            }
	    else{
                action = [&](const Eigen::VectorXf &value) {
                    nsteps += 1;
                    specs.push_back(CollapseMatrix(config, FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), value, true,var_index).Spec()));
                    for(int i = 0; i < nspline; ++i) {
                        posteriors[i].Fill(value(i+nphys));
                        param_samples[i].push_back(value(i+nphys));
                    }
		    Eigen::VectorXf splines = value.segment(nphys, nspline);
                    Eigen::VectorXf diff = splines-splines_bf;
                    post_covar += diff * diff.transpose();
                };
	    }
            met.run(burnin, iterations, action, pbar);

            post_covar /= nsteps;
            // MAP-consistent covariance pull: Sigma(C+Sigma)^-1 (d - cv_bf) at the
            // best fit, matching the analytic getCovarianceOnlyErrorBand (exactly 0
            // for Asimov). The sample median is NOT used for center_shift — it
            // carries the posterior-median-vs-MAP offset plus MCMC noise.
            Eigen::VectorXf analytic_shift = Eigen::VectorXf::Zero(cv_coll.size());
            if (use_data) {
                // Constrained posterior pull for covariance-type systematics,
               // the collapsed systematic covariance is
                // Sigma = L*L^T (L = U*sqrt(S) from DecomposeFractionalCovariance)
                // and the spectrum shift is L*alpha (matchs the note's R = L^T)
            
                // Mark Note: Restrict the constraint to the bins the fit metric actually
                // used, aka the active bins PR from a while back with data > 0 (PROchi drops zero-data bins
                // and its stat term is diag(max(data,1))). PROjector-masked or
                // empty bins must not pull on alpha.
                std::vector<int> contrib;
                for(int i = 0; i < data_spec.size(); ++i)
                    if(config.IsBinActive(var_index, i) && data_spec(i) > 0)
                        contrib.push_back(i);
            
                std::vector<int> modes;
                for(int j = 0; j < L.cols(); ++j)
                    if(L.col(j).squaredNorm() > 0) modes.push_back(j);

                if(!contrib.empty() && !modes.empty()) {
                    const size_t k = modes.size(), nb = contrib.size();
                    Eigen::MatrixXd L_shift(L.rows(), k);
                    for(size_t j = 0; j < k; ++j)
                        L_shift.col(j) = L.col(modes[j]).cast<double>();
                    Eigen::MatrixXd L_red(nb, k);
                    Eigen::VectorXd C_inv_red(nb);
                    for(size_t i = 0; i < nb; ++i) {
                        L_red.row(i) = L_shift.row(contrib[i]);
                        C_inv_red(i) = 1.0 / std::max<double>(data_spec(contrib[i]), 1.0);
                    }
                    Eigen::MatrixXd inner = Eigen::MatrixXd::Identity(k, k)
                                          + L_red.transpose() * C_inv_red.asDiagonal() * L_red;
                    Eigen::LLT<Eigen::MatrixXd> inner_llt(inner);
                    if(inner_llt.info() != Eigen::Success) {
                        // inner is PD by construction. any failures here are float math noise from L. Clamp eigenvalues and retry.
                        log<LOG_WARNING>(L"%1% || LLT of posterior pull matrix failed; clamping eigenvalues.") % __func__;
                        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(inner);
                        inner = es.eigenvectors()
                              * es.eigenvalues().cwiseMax(1e-12).asDiagonal()
                              * es.eigenvectors().transpose();
                        inner_llt.compute(inner);
                    }

                    // Conditional pull at the best fit: L*alpha_min(d - cv) equals
                    // Sigma[:,contrib] (C+Sigma)^-1 (d - cv) by the push-through identity.
                    Eigen::VectorXd u_bf(nb);
                    for(size_t i = 0; i < nb; ++i)
                        u_bf(i) = data_spec(contrib[i]) - cv_coll(contrib[i]);
                    analytic_shift = (L_shift * inner_llt.solve(
                        L_red.transpose() * (C_inv_red.asDiagonal() * u_bf))).cast<float>();

                    Eigen::VectorXd throws_k(k), residual(nb);
                    for(size_t ai = 0; ai < nsteps; ++ai) {
                        for(size_t i = 0; i < k; ++i)
                            throws_k(i) = nd(PROseed::global_rng);
                        for(size_t i = 0; i < nb; ++i)
                            residual(i) = data_spec(contrib[i]) - specs.at(ai)(contrib[i]);
                        Eigen::VectorXd alpha_min =
                            inner_llt.solve(L_red.transpose() * (C_inv_red.asDiagonal() * residual));
                        // delta = U^-1 v with inner = U^T U has cov(delta) = inner^-1.
                        Eigen::VectorXd alpha_hat =
                            alpha_min + inner_llt.matrixU().solve(throws_k);
                        specs.at(ai) += (L_shift * alpha_hat).cast<float>();

                        Eigen::VectorXf diff_hist = specs.at(ai) - cv_coll;
                        hist_diff_sum += diff_hist;
                        post_hist_covar += diff_hist * diff_hist.transpose();
                    }
                } else {
                    // Nothing to constrain , but still accumulate the band covariance
                    // the constraint loop would otherwise have provided.
                    for(size_t ai = 0; ai < nsteps; ++ai) {
                        Eigen::VectorXf diff_hist = specs.at(ai) - cv_coll;
                        hist_diff_sum += diff_hist;
                        post_hist_covar += diff_hist * diff_hist.transpose();
                    }
                }
            }

            post_hist_covar /= nsteps;
            // Make the covariance CENTRAL (about the sample mean, not the
            // best-fit CV): the data-pull displacement lives in center_shift,
            // and projected band widths built from this covariance must not
            // absorb it. Matches the analytic getCovarianceOnlyErrorBand.
            hist_diff_sum /= nsteps;
            post_hist_covar -= hist_diff_sum * hist_diff_sum.transpose();
            param_err_lo = Eigen::VectorXf::Zero(nspline);
            param_err_hi = Eigen::VectorXf::Zero(nspline);
            for(int i = 0; i < nspline; ++i) {
                auto &v = param_samples[i];
                std::sort(v.begin(), v.end());
                float bf_val = best_fit(nphys + i);
                param_err_lo(i) = std::abs(bf_val - v[int(0.160f * v.size())]);
                param_err_hi(i) = std::abs(v[int(0.840f * v.size())] - bf_val);
            }
            log<LOG_INFO>(L"%1% || Acceptance rate %2%") % __func__ % ((float)met.naccept / iterations);

            cv = CollapseMatrix(config, cv);
            PROerrorbar ebar(cv.size());
            for(int i = 0; i < cv.size(); ++i) {
                std::vector<float> binconts(specs.size());
                for(size_t j = 0; j < specs.size(); ++j) {
                    binconts[j] = specs[j](i);
                }
                float scale_factor = scale ? 1.0/config.collapsed_bin_widths.at(var_index)(i) :  1.0;
                if(std::isnan(scale_factor)) scale_factor = 1;
                std::sort(binconts.begin(), binconts.end());
                // Percentile widths about the sample MEDIAN, not cv: for the
                // data-constrained band the sample cloud is pulled toward the
                // data, and |quantile - cv| would fold a thin displaced band
                // into a fat one straddling the best fit. The displacement is
                // reported separately in center_shift; error_point stays the
                // best-fit spectrum (plot code uses it for unit conversion).
                float med = binconts[int(0.500*specs.size())];
                float ehi = (binconts[int(0.840*specs.size())] - med)*scale_factor;
                float elo = (med - binconts[int(0.160*specs.size())])*scale_factor;
                ebar.error_up(i) =  ehi;
                ebar.error_down(i) =  elo;
                ebar.error_point(i) = cv(i)*scale_factor;
                ebar.center_shift(i) = use_data ? analytic_shift(i)*scale_factor
                                                : (med - cv(i))*scale_factor;
                log<LOG_INFO>(L"%1% || ErrorBand bin %2% %3% %4% %5% %6% shift %7%") % __func__ % i % cv(i) % ehi % elo % scale_factor % ebar.center_shift(i);
            }
            ebar.covariance = post_hist_covar;

            // =========================================================================
            //  DIAGNOSTIC SANITY CHECK LOOP
            // =========================================================================
            log<LOG_INFO>(L"%1% || Running pre-return sanity checks on PROerrorbar...") % __func__;

            for (int i = 0; i < ebar.error_point.size(); ++i) {
                float pt  = ebar.error_point(i);
                float eup = ebar.error_up(i);
                float elo = ebar.error_down(i);

                if (std::isnan(pt) || std::isinf(pt)) {
                    log<LOG_ERROR>(L"%1% || CRITICAL ERROR: error_point bin %2% is invalid (value: %3%)") % __func__ % i % pt;
                    throw std::runtime_error("PROerrorbar error_point contains NaN or Inf at bin " + std::to_string(i));
                }
                if (std::isnan(eup) || std::isinf(eup)) {
                    log<LOG_ERROR>(L"%1% || CRITICAL ERROR: error_up bin %2% is invalid (value: %3%)") % __func__ % i % eup;
                    throw std::runtime_error("PROerrorbar error_up contains NaN or Inf at bin " + std::to_string(i));
                }
                if (std::isnan(elo) || std::isinf(elo)) {
                    log<LOG_ERROR>(L"%1% || CRITICAL ERROR: error_down bin %2% is invalid (value: %3%)") % __func__ % i % elo;
                    throw std::runtime_error("PROerrorbar error_down contains NaN or Inf at bin " + std::to_string(i));
                }
                float shf = ebar.center_shift(i);
                if (std::isnan(shf) || std::isinf(shf)) {
                    log<LOG_ERROR>(L"%1% || CRITICAL ERROR: center_shift bin %2% is invalid (value: %3%)") % __func__ % i % shf;
                    throw std::runtime_error("PROerrorbar center_shift contains NaN or Inf at bin " + std::to_string(i));
                }
            }

            // Check covariance matrix dimensions and values
            if (ebar.covariance.rows() != cv.size() || ebar.covariance.cols() != cv.size()) {
                log<LOG_ERROR>(L"%1% || CRITICAL ERROR: Covariance matrix dimension mismatch! Expected (%2%x%3%), got (%4%x%5%)") 
                    % __func__ % cv.size() % cv.size() % ebar.covariance.rows() % ebar.covariance.cols();
                throw std::runtime_error("PROerrorbar covariance matrix dimension mismatch!");
            }

            for (int r = 0; r < ebar.covariance.rows(); ++r) {
                for (int c = 0; c < ebar.covariance.cols(); ++c) {
                    float val = ebar.covariance(r, c);
                    if (std::isnan(val) || std::isinf(val)) {
                        log<LOG_ERROR>(L"%1% || CRITICAL ERROR: Covariance matrix element (%2%, %3%) is invalid (value: %4%)") 
                            % __func__ % r % c % val;
                        throw std::runtime_error("PROerrorbar covariance contains NaN or Inf at (" + std::to_string(r) + ", " + std::to_string(c) + ")");
                    }
                }
            }

            log<LOG_INFO>(L"%1% || Sanity check passed successfully!") % __func__;
            // =========================================================================


            return ebar;
        }

    /**
     * @brief Produce a 1-sigma summary plot from MCMC results only (no profile scan).
     * @details Mirrors the post-MCMC pieces of PROfile::Plot's "_1sigma_detailed.pdf":
     * a gray ±1 prior band, blue post-fit MCMC bars centered on @p best_fit (widths
     * from @p param_err_lo / @p param_err_hi), red squares for the global best-fit, and
     * orange diamonds for any injected truth values. Intended to be called between the
     * MCMC error-band step and the (slow) profile scan.
     * @param filename     Output prefix; final file is @p filename + "_1sigmaMCMC.pdf".
     * @param config       Analysis configuration (used for parameter pretty names).
     * @param systs        PROsyst (provides spline names and count).
     * @param model        Physics model.
     * @param best_fit     Best-fit parameter vector (length nphys + nspline).
     * @param param_err_lo Per-spline lower 1σ from MCMC quantiles (length nspline).
     * @param param_err_hi Per-spline upper 1σ from MCMC quantiles (length nspline).
     * @param with_osc     If true, also plot physics parameters; if false, splines only.
     * @param true_params  Optional injected truth values.
     */
    void plot_mcmc_1sigma(const std::string &filename, const PROconfig &config, const PROsyst &systs, const PROmodel &model, const Eigen::VectorXf &best_fit, const Eigen::VectorXf &param_err_lo, const Eigen::VectorXf &param_err_hi, bool with_osc = false, const Eigen::VectorXf &true_params = Eigen::VectorXf());

};

#endif
