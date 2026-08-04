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
        bool hasBound(std::string bound_name){
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
        float getBound(std::string bound_name){
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
     * @param skip_stack_subchannels  Optional global subchannel indices to omit from the
     *        CV stack and legend (used by --bkg-subtract; their contents are expected to
     *        already be zero in `cv`).
     */
    std::map<std::string, TObject *> plot_channels(const std::string &filename, const PROconfig &config, std::optional<PROspec> cv, std::optional<PROspec> best_fit, std::optional<PROdata> data, std::optional<PROerrorbar> errband, std::optional<PROerrorbar> posterrband, std::vector<TPaveText> &texts, PlotBounds &bounds, PlotOptions opt = PlotOptions::Default, int var_index = 0, bool ratio_bool = false, const std::vector<size_t> *skip_stack_subchannels = nullptr);

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

            /*log<LOG_INFO>(L"%1% || very first cv %2%") % __func__ % cv;
            bool has_zero = (cv.array() <= 0.0f).any();
            bool has_floor = (cv.array() == 1e-06f).any(); // Change 1e-06f to match your exact floor value if different
            bool is_finite = cv.allFinite();
            
            if (!is_finite || has_zero || has_floor) {
                log<LOG_INFO>(L"%1% || Input spectrum flooring warning! "
                              L"allFinite: %2%, has_zero/neg: %3%, contains_exact_floor: %4%")
                    % __func__ 
                    % is_finite 
                    % has_zero 
                    % has_floor;
            
                // Optional: Print the exact indices causing the issue
                for (int i = 0; i < cv.size(); ++i) {
                    if (std::isnan(cv(i)) || std::isinf(cv(i)) || cv(i) <= 0.0f || cv(i) == 1e-06f) {
                        std::cout << "[DEBUG] cv bin " << i << " has problematic value: " << cv(i) << std::endl;
                    }
                }
            }*/
            Eigen::VectorXf cv_coll = CollapseMatrix(config, cv);
            //log<LOG_INFO>(L"%1% || very first cv_coll %2%") % __func__ % cv_coll;
            Eigen::MatrixXf L;
            if(metric.GetSysts().GetNCovar() > 0) L = metric.GetSysts().DecomposeFractionalCovariance(config, cv);
            else L = Eigen::MatrixXf::Zero(config.m_num_variable_bins_total_collapsed[var_index], config.m_num_variable_bins_total_collapsed[var_index]);
            //log<LOG_INFO>(L"%1% || very first L %2%") % __func__ % L;
            std::normal_distribution<float> nd;
            Eigen::VectorXf throws = Eigen::VectorXf::Constant(config.m_num_variable_bins_total_collapsed[var_index], 0);

            int nspline = metric.GetSysts().GetNSplines();
            int nphys = metric.GetModel().nparams;
            Eigen::VectorXf splines_bf = best_fit.segment(nphys, nspline);
            post_covar = Eigen::MatrixXf::Constant(nspline, nspline, 0);
            Eigen::MatrixXf post_hist_covar = Eigen::MatrixXf::Constant(cv_coll.size(), cv_coll.size(), 0);
            size_t nsteps = 0;
            std::vector<Eigen::VectorXf> specs;
            std::vector<std::vector<float>> param_samples(nspline);

            //log<LOG_INFO>(L"%1% || nsteps %2%") % __func__ % nsteps;
	    std::function<void(const Eigen::VectorXf&)> action;
	    
            if (data_spec.rows() == 0){
	        action = [&](const Eigen::VectorXf &value) {
                nsteps += 1;
		for(size_t i = 0; i < config.m_num_variable_bins_total_collapsed[var_index]; ++i)
                        throws(i) = nd(PROseed::global_rng);
                specs.push_back(CollapseMatrix(config, FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), value, true,var_index).Spec())+L*throws);
                //log<LOG_INFO>(L"%1% || first specs.back %2%") % __func__ % specs.back();
                for(int i = 0; i < nspline; ++i) {
                    posteriors[i].Fill(value(i+nphys));
                    param_samples[i].push_back(value(i+nphys));
                }
                Eigen::VectorXf splines = value.segment(nphys, nspline);
                Eigen::VectorXf diff = splines-splines_bf;
                post_covar += diff * diff.transpose();
		Eigen::VectorXf diff_hist = specs.back() - cv_coll;
                post_hist_covar += diff_hist * diff_hist.transpose();
                };
            }
	    else{
                action = [&](const Eigen::VectorXf &value) {
                    nsteps += 1;
                    specs.push_back(CollapseMatrix(config, FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), value, true,var_index).Spec()));
                    //log<LOG_INFO>(L"%1% || second specs.back %2%") % __func__ % specs.back();
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
            log<LOG_INFO>(L"%1% || nsteps again %2%") % __func__ % nsteps;
            if (data_spec.size() != 0) {
                // 1. Pre-compute static matrices ONCE outside the loop
                Eigen::MatrixXf C_stat = cv_coll.asDiagonal();
                log<LOG_INFO>(L"%1% || C_stat %2%") % __func__ % C_stat;
                Eigen::MatrixXf M = C_stat + L.transpose() * L;
                log<LOG_INFO>(L"%1% || M %2%") % __func__ % M;
                auto M_solver = M.ldlt(); // Pre-factorize M once
            
                size_t k = L.cols();
                Eigen::VectorXf C_inv_diag = cv_coll.cwiseInverse();
                log<LOG_INFO>(L"%1% || C_inv_diag %2%") % __func__ % C_inv_diag;
                Eigen::MatrixXf inner_matrix = Eigen::MatrixXf::Identity(k, k) 
                                             + L.transpose() * C_inv_diag.asDiagonal() * L;
                log<LOG_INFO>(L"%1% || inner_matrix %2%") % __func__ % inner_matrix;
                Eigen::MatrixXf inner_inv = inner_matrix.ldlt().solve(Eigen::MatrixXf::Identity(k, k));
		if (inner_matrix.ldlt().info() != Eigen::Success) {
		    log<LOG_ERROR>(L"inner_matrix LDLT factorization failed!");
		}
                log<LOG_INFO>(L"%1% || inner_inv %2%") % __func__ % inner_inv;
                Eigen::LLT<Eigen::MatrixXf> llt(inner_inv);
                Eigen::MatrixXf C_chol = llt.matrixL();
                log<LOG_INFO>(L"%1% || C_chol %2%") % __func__ % C_chol;
            
                // 2. Fast loop over steps
                for(int ai = 0; ai < nsteps; ai++) {
                    log<LOG_INFO>(L"%1% || ai %2%") % __func__ % ai;
                    for(size_t i = 0; i < throws.size(); ++i) {
                        throws(i) = nd(PROseed::global_rng);
                    }
            
                    // Uses pre-factorized solver: fast O(k^2) instead of O(k^3)
                    Eigen::VectorXf residual = data_spec - specs.at(ai);
                    log<LOG_INFO>(L"%1% || initial specs %2%") % __func__ % specs.at(ai);
                    log<LOG_INFO>(L"%1% || data_spec %2%") % __func__ % data_spec;
                    log<LOG_INFO>(L"%1% || residual %2%") % __func__ % residual;
                    log<LOG_INFO>(L"%1% || L %2%") % __func__ % L;
                    Eigen::VectorXf alpha_min = L * M_solver.solve(residual);
            
                    log<LOG_INFO>(L"%1% || alpha_min %2%") % __func__ % alpha_min;
                    log<LOG_INFO>(L"%1% || C_chol %2%") % __func__ % C_chol;
                    log<LOG_INFO>(L"%1% || throws %2%") % __func__ % throws;
                    Eigen::VectorXf alpha_hat = alpha_min + C_chol * throws;
                    log<LOG_INFO>(L"%1% || alpha_hat %2%") % __func__ % alpha_hat;
                    specs.at(ai) += L * alpha_hat;
                    
                    Eigen::VectorXf diff_hist = specs.at(ai) - cv_coll;
		    if (!cv_coll.allFinite()){
                        log<LOG_INFO>(L"%1% || really bad cv_coll %2%") % __func__ % cv_coll;
		    }
		    if (!diff_hist.allFinite()){
                        log<LOG_INFO>(L"%1% || really bad diff_hist %2%") % __func__ % diff_hist;
		    }

                    post_hist_covar += diff_hist * diff_hist.transpose();
		    if (!specs.at(ai).allFinite()){
                        log<LOG_INFO>(L"%1% || really bad specs.at(ai) %2%") % __func__ % specs.at(ai);
		    }
                }
            }

            log<LOG_INFO>(L"%1% || nsteps again again %2%") % __func__ % nsteps;
            post_hist_covar /= nsteps;
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

            log<LOG_INFO>(L"%1% || cv %2% ") % __func__ % cv;
            //std::vector<float> centers;
            //size_t global_channel_index = 0;
            //for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
            //    for(size_t det = 0; det < config.m_num_detectors; ++det) {
            //        for(size_t channel = 0; channel < config.m_num_channels; ++channel) {
            //            std::vector<float> tedges =  config.GetChannelVariableBins(global_channel_index, var_index).Edges();
            //            global_channel_index++;
            //            for(size_t p=0; p<tedges.size(); p++){
            //                if(p<tedges.size()-1){
            //                    centers.push_back((tedges[p+1]+tedges[p])/2.0);
            //                }
            //            }

            //        }
            //    }
            //}

            PROerrorbar ebar(cv.size());
            for(int i = 0; i < cv.size(); ++i) {
                std::vector<float> binconts(specs.size());
                for(size_t j = 0; j < specs.size(); ++j) {
                log<LOG_INFO>(L"%1% || inside binconts") % __func__;
                    binconts[j] = specs[j](i);
                }
                float scale_factor = scale ? 1.0/config.collapsed_bin_widths.at(var_index)(i) :  1.0;
                if(std::isnan(scale_factor)) scale_factor = 1;
                std::sort(binconts.begin(), binconts.end());
                log<LOG_INFO>(L"%1% || specs.size() %2% ") % __func__ % specs.size();
		int testint = int(0.840*specs.size());
                log<LOG_INFO>(L"%1% || testin %2% ") % __func__ % testint;
                log<LOG_INFO>(L"%1% || cvi %2% ") % __func__ % cv(i);
                float inside = (binconts[int(0.840*specs.size())] - cv(i));
                log<LOG_INFO>(L"%1% || inside %2% ") % __func__ % inside;
                float ehi = std::abs((binconts[int(0.840*specs.size())] - cv(i))*scale_factor);
                float elo = std::abs((cv(i) - binconts[int(0.160*specs.size())])*scale_factor);
                log<LOG_INFO>(L"%1% || ehi %2% ") % __func__ % ehi;
                log<LOG_INFO>(L"%1% || elo %2% ") % __func__ % elo;
                ebar.error_up(i) =  ehi;
                ebar.error_down(i) =  elo;
                ebar.error_point(i) = cv(i)*scale_factor;
                log<LOG_INFO>(L"%1% || ErrorBand bin %2% %3% %4% %5% %6% ") % __func__ % i % cv(i) % ehi % elo % scale_factor ;
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
