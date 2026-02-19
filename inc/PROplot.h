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

    struct PlotBounds {
        float xmin = -9999;
        float xmax = -9999;
        float ymin = -9999;
        float ymax = -9999;
        float ratmin = -9999;
        float ratmax = -9999;

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
    
    enum class PlotOptions {
        Default = 0,
        CVasStack = 1 << 0,
        AreaNormalized = 1 << 1,
        BinWidthScaled = 1 << 2,
        DataMCRatio = 1 << 3,
        DataPostfitRatio = 1 << 4,
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


    void plot_detector_ratios(const PROconfig &config, std::vector<TH1D> data_hists,  std::vector<TH1D> cv_hists, std::optional<PROerrorbar> errband, std::vector<TH1D> bf_hists, std::optional<PROerrorbar> posterrband, TH2D &pre_corr, TH2D &post_corr, std::string filename, int var_index = 0);

    void plot_channels(const std::string &filename, const PROconfig &config, std::optional<PROspec> cv, std::optional<PROspec> best_fit, std::optional<PROdata> data, std::optional<PROerrorbar> errband, std::optional<PROerrorbar> posterrband, std::optional<PROsyst> pre_allcovsyst, std::optional<PROsyst> post_allcovsyst, std::vector<TPaveText> &texts, PlotBounds &bounds, PlotOptions opt = PlotOptions::Default, int var_index = 0);

    //some helper functions for PROplot
    std::map<std::string, std::unique_ptr<TH1D>> getCV1DHists(const PROspec & spec, const PROconfig& inconfig, bool scale = false, int var_index = 0);
    std::map<std::string, std::unique_ptr<TH2D>> getCV2DHists(const PROspec & spec, const PROconfig& inconfig, bool scale = false, int var_index = 0);
    std::map<std::string, std::unique_ptr<TH2D>> covarianceTH2D(const PROsyst &syst, const PROconfig &config, const PROspec &cv);
    std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> getSplineGraphs(const PROsyst &systs, const PROconfig &config);
    PROerrorbar getErrorBand(const PROconfig &config, const PROpeller &prop, const PROsyst &syst, const PROmodel &model, const PROspec &cv_spec, const Eigen::VectorXf &cvparams,bool scale=false, int other_index=0);

    int plotPriorFractionalSystematicBreakdown(const PROconfig &config, const PROspec &spec, const PROsyst &allsplinesyst, std::string filename, int var_index = 0, std::map<std::string, double> alphas = {});

    template<class T, class P>
        PROerrorbar getMCMCErrorBand(Metropolis<T, P> met, size_t burnin, size_t iterations, const PROconfig &config, const PROpeller &prop, PROmetric &metric, const Eigen::VectorXf &best_fit, std::vector<TH1D> &posteriors, Eigen::MatrixXf &post_covar,  bool scale = false,int var_index=0) {
            for(size_t i = 0; i < metric.GetSysts().GetNSplines(); ++i)
                posteriors.emplace_back("", (";"+config.m_mcgen_variation_plotname_map.at(metric.GetSysts().spline_names[i])).c_str(), 60, -3, 3);

            Eigen::VectorXf cv = FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), best_fit, true, var_index).Spec();
            Eigen::MatrixXf L; 
            if(metric.GetSysts().GetNCovar() > 0) L = metric.GetSysts().DecomposeFractionalCovariance(config, cv);
            else L = Eigen::MatrixXf::Zero(config.m_num_variable_bins_total_collapsed[var_index], config.m_num_variable_bins_total_collapsed[var_index]);
            std::normal_distribution<float> nd;
            Eigen::VectorXf throws = Eigen::VectorXf::Constant(config.m_num_variable_bins_total_collapsed[var_index], 0);

            int nspline = metric.GetSysts().GetNSplines();
            int nphys = metric.GetModel().nparams;
            Eigen::VectorXf splines_bf = best_fit.segment(nphys, nspline);
            post_covar = Eigen::MatrixXf::Constant(nspline, nspline, 0);
            size_t accepted = 0;
            std::vector<Eigen::VectorXf> specs;
            const auto action = [&](const Eigen::VectorXf &value) {
                accepted += 1;
                for(size_t i = 0; i < config.m_num_variable_bins_total_collapsed[var_index]; ++i)
                    throws(i) = nd(PROseed::global_rng);
                specs.push_back(CollapseMatrix(config, FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), value, true,var_index).Spec())+L*throws);
                for(size_t i = 0; i < metric.GetSysts().GetNSplines(); ++i)
                    posteriors[i].Fill(value(i+nphys));
                Eigen::VectorXf splines = value.segment(nphys, nspline);
                Eigen::VectorXf diff = splines-splines_bf;
                post_covar += diff * diff.transpose();
                bool print_autocorrelation_values = false;
                if(print_autocorrelation_values){
                    log<LOG_INFO>(L"%1% || AUTO  %2% : %3%") % __func__ % accepted % value;
                }
            };
            met.run(burnin, iterations, action);
            post_covar /= accepted;
            log<LOG_INFO>(L"%1% || Acceptance rate %2%") % __func__ % ((float)accepted / iterations);

            cv = CollapseMatrix(config, cv);

            std::vector<float> centers;
            size_t global_channel_index = 0;
            for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
                for(size_t det = 0; det < config.m_num_detectors; ++det) {
                    for(size_t channel = 0; channel < config.m_num_channels; ++channel) {
                        std::vector<float> tedges =  config.GetChannelVariableBins(global_channel_index, var_index).Edges();
                        global_channel_index++;
                        for(size_t p=0; p<tedges.size(); p++){
                            if(p<tedges.size()-1){
                                centers.push_back((tedges[p+1]+tedges[p])/2.0);
                            }
                        }

                    }
                }
            }

            PROerrorbar ebar(cv.size());
            for(int i = 0; i < cv.size(); ++i) {
                std::vector<float> binconts(specs.size());
                for(size_t j = 0; j < specs.size(); ++j) {
                    binconts[j] = specs[j](i);
                }
                float scale_factor = scale ? 1.0/config.collapsed_bin_widths.at(var_index)(i) :  1.0;
                if(std::isnan(scale_factor)) scale_factor = 1;
                std::sort(binconts.begin(), binconts.end());
                float ehi = std::abs((binconts[int(0.840*specs.size())] - cv(i))*scale_factor);
                float elo = std::abs((cv(i) - binconts[int(0.160*specs.size())])*scale_factor);
                ebar.error_up(i) =  ehi;
                ebar.error_down(i) =  elo;
                ebar.error_point(i) = cv(i)*scale_factor;
                log<LOG_DEBUG>(L"%1% || ErrorBand bin %2% %3% %4% %5% %6% ") % __func__ % i % cv(i) % ehi % elo % scale_factor ;
            }
            return ebar;
        }


};

#endif
