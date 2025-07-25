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


    void plot_channels(const std::string &filename, const PROconfig &config, std::optional<PROspec> cv, std::optional<PROspec> best_fit, std::optional<PROdata> data, std::optional<TGraphAsymmErrors*> errband, std::optional<TGraphAsymmErrors*> posterrband, std::vector<TPaveText> &texts, PlotOptions opt = PlotOptions::Default, int other_index = -1);

    //some helper functions for PROplot
    std::map<std::string, std::unique_ptr<TH1D>> getCVHists(const PROspec & spec, const PROconfig& inconfig, bool scale = false, int other_index = -1);
    std::map<std::string, std::unique_ptr<TH2D>> covarianceTH2D(const PROsyst &syst, const PROconfig &config, const PROspec &cv);
    std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> getSplineGraphs(const PROsyst &systs, const PROconfig &config);
    std::unique_ptr<TGraphAsymmErrors> getErrorBand(const PROconfig &config, const PROpeller &prop, const PROsyst &syst, bool scale = false, int other_index = -1);

    int plotPriorFractionalSystematicBreakdown(const PROconfig &config, const PROspec &spec, const PROsyst &allsplinesyst, std::string filename);

    template<class T, class P>
    std::unique_ptr<TGraphAsymmErrors> getMCMCErrorBand(Metropolis<T, P> met, size_t burnin, size_t iterations, const PROconfig &config, const PROpeller &prop, PROmetric &metric, const Eigen::VectorXf &best_fit, std::vector<TH1D> &posteriors, Eigen::MatrixXf &post_covar, bool scale = false) {
            for(size_t i = 0; i < metric.GetSysts().GetNSplines(); ++i)
                posteriors.emplace_back("", (";"+config.m_mcgen_variation_plotname_map.at(metric.GetSysts().spline_names[i])).c_str(), 60, -3, 3);

            Eigen::VectorXf cv = FillRecoSpectra(config, prop, metric.GetSysts(), metric.GetModel(), best_fit, true).Spec();
            Eigen::MatrixXf L; 
            if(metric.GetSysts().GetNCovar() > 0) L = metric.GetSysts().DecomposeFractionalCovariance(config, cv);
            else L = Eigen::MatrixXf::Zero(config.m_num_bins_total_collapsed, config.m_num_bins_total_collapsed);
            std::normal_distribution<float> nd;
            Eigen::VectorXf throws = Eigen::VectorXf::Constant(config.m_num_bins_total_collapsed, 0);

            int nspline = metric.GetSysts().GetNSplines();
            int nphys = metric.GetModel().nparams;
            Eigen::VectorXf splines_bf = best_fit.segment(nphys, nspline);
            post_covar = Eigen::MatrixXf::Constant(nspline, nspline, 0);
            size_t accepted = 0;
            std::vector<Eigen::VectorXf> specs;
            const auto action = [&](const Eigen::VectorXf &value) {
                accepted += 1;
                for(size_t i = 0; i < config.m_num_bins_total_collapsed; ++i)
                    throws(i) = nd(PROseed::global_rng);
                specs.push_back(CollapseMatrix(config, FillRecoSpectra(config, prop, metric.GetSysts(), metric.GetModel(), value, true).Spec())+L*throws);
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

            //TODO: Only works with 1 mode/detector/channel
            cv = CollapseMatrix(config, cv);
            std::vector<float> edges = config.GetChannelBinEdges(0);
            std::vector<float> centers;
            for(size_t i = 0; i < edges.size() - 1; ++i)
                centers.push_back((edges[i+1] + edges[i])/2);
            TH1D tmphist("th", "", cv.size(), edges.data());
            for(int i = 0; i < cv.size(); ++i)
                tmphist.SetBinContent(i+1, cv(i));
            if(scale) tmphist.Scale(1, "width");
            std::unique_ptr<TGraphAsymmErrors> ret = std::make_unique<TGraphAsymmErrors>(&tmphist);
            for(int i = 0; i < cv.size(); ++i) {
                std::vector<float> binconts(specs.size());
                for(size_t j = 0; j < specs.size(); ++j) {
                    binconts[j] = specs[j](i);
                }
                if(!binconts.size()) continue;
                float scale_factor = tmphist.GetBinContent(i+1)/cv(i);
                if(std::isnan(scale_factor)) scale_factor = 1;
                std::sort(binconts.begin(), binconts.end());
                float ehi = std::abs((binconts[0.84*specs.size()] - cv(i))*scale_factor);
                float elo = std::abs((cv(i) - binconts[0.16*specs.size()])*scale_factor);
                ret->SetPointEYhigh(i, ehi);
                ret->SetPointEYlow(i, elo);
                log<LOG_DEBUG>(L"%1% || ErrorBand bin %2% %3% %4% %5% %6% %7%") % __func__ % i % cv(i) % ehi % elo % scale_factor % tmphist.GetBinContent(i+1);
            }
            return ret;
        }


};

#endif
