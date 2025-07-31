#include "PROplot.h"

namespace PROfit{


    std::map<std::string, std::unique_ptr<TH1D>> getCVHists(const PROspec &spec, const PROconfig& inconfig, bool scale, int other_index) {
        std::map<std::string, std::unique_ptr<TH1D>> hists;  

        size_t global_subchannel_index = 0;
        size_t global_channel_index = 0;
        for(size_t im = 0; im < inconfig.m_num_modes; im++){
            for(size_t id =0; id < inconfig.m_num_detectors; id++){
                for(size_t ic = 0; ic < inconfig.m_num_channels; ic++){
                    for(size_t sc = 0; sc < inconfig.m_num_subchannels[ic]; sc++){
                        const std::string& subchannel_name  = inconfig.m_fullnames[global_subchannel_index];
                        const std::string& color = inconfig.m_subchannel_colors[ic][sc];
                        int rcolor = color == "NONE" ? kRed - 7 : inconfig.HexToROOTColor(color);
                        std::unique_ptr<TH1D> htmp = std::make_unique<TH1D>(spec.toTH1D(inconfig, global_subchannel_index, other_index));
                        htmp->SetLineWidth(1);
                        htmp->SetLineColor(kBlack);
                        htmp->SetFillColor(rcolor);
                        if(scale) htmp->Scale(1,"width");
                        hists[subchannel_name] = std::move(htmp);

                        log<LOG_DEBUG>(L"%1% || Printot %2% %3% %4% %5% %6% : Integral %7% ") % __func__ % global_channel_index % global_subchannel_index % subchannel_name.c_str() % sc % ic % hists[subchannel_name]->Integral();
                        ++global_subchannel_index;
                    }//end subchan
                    ++global_channel_index;
                }//end chan
            }//end det
        }//end mode
        return hists;
    }

    std::map<std::string, std::unique_ptr<TH2D>> covarianceTH2D(const PROsyst &syst, const PROconfig &config, const PROspec &cv) {
        std::map<std::string, std::unique_ptr<TH2D>> ret;
        Eigen::MatrixXf fractional_cov = syst.fractional_covariance;
        Eigen::MatrixXf diag = cv.Spec().array().matrix().asDiagonal(); 
        Eigen::MatrixXf full_covariance =  diag*fractional_cov*diag;
        Eigen::MatrixXf collapsed_full_covariance =  CollapseMatrix(config,full_covariance);  
        Eigen::VectorXf collapsed_cv = CollapseMatrix(config, cv.Spec());
        Eigen::MatrixXf collapsed_cv_inv_diag = collapsed_cv.asDiagonal().inverse();
        Eigen::MatrixXf collapsed_frac_cov = collapsed_cv_inv_diag * collapsed_full_covariance * collapsed_cv_inv_diag;

        std::unique_ptr<TH2D> cov_hist = std::make_unique<TH2D>("cov", "Fractional Covariance Matrix;Bin # ;Bin #", config.m_num_bins_total, 0, config.m_num_bins_total, config.m_num_bins_total, 0, config.m_num_bins_total);
        std::unique_ptr<TH2D> collapsed_cov_hist = std::make_unique<TH2D>("ccov", "Collapsed Fractional Covariance Matrix;Bin # ;Bin #", config.m_num_bins_total_collapsed, 0, config.m_num_bins_total_collapsed, config.m_num_bins_total_collapsed, 0, config.m_num_bins_total_collapsed);

        std::unique_ptr<TH2D> cor_hist = std::make_unique<TH2D>("cor", "Correlation Matrix;Bin # ;Bin #", config.m_num_bins_total, 0, config.m_num_bins_total, config.m_num_bins_total, 0, config.m_num_bins_total);
        std::unique_ptr<TH2D> collapsed_cor_hist = std::make_unique<TH2D>("ccor", "Collapsed Correlation Matrix;Bin # ;Bin #", config.m_num_bins_total_collapsed, 0, config.m_num_bins_total_collapsed, config.m_num_bins_total_collapsed, 0, config.m_num_bins_total_collapsed);

        for(size_t i = 0; i < config.m_num_bins_total; ++i)
            for(size_t j = 0; j < config.m_num_bins_total; ++j){
                cov_hist->SetBinContent(i+1,j+1,fractional_cov(i,j));
                cor_hist->SetBinContent(i+1,j+1,fractional_cov(i,j)/(sqrt(fractional_cov(i,i))*sqrt(fractional_cov(j,j))));
            }

        for(size_t i = 0; i < config.m_num_bins_total_collapsed; ++i)
            for(size_t j = 0; j < config.m_num_bins_total_collapsed; ++j){
                collapsed_cov_hist->SetBinContent(i+1,j+1,collapsed_frac_cov(i,j));
                collapsed_cor_hist->SetBinContent(i+1,j+1,collapsed_frac_cov(i,j)/(sqrt(collapsed_frac_cov(i,i))*sqrt(collapsed_frac_cov(j,j))));
            }

        float cov_max_abs = std::max(cov_hist->GetMaximum(), std::abs(cov_hist->GetMinimum()));
        cov_hist->SetMaximum(cov_max_abs);
        cov_hist->SetMinimum(-cov_max_abs);
        float coll_cov_max_abs = std::max(collapsed_cov_hist->GetMaximum(), std::abs(collapsed_cov_hist->GetMinimum()));
        collapsed_cov_hist->SetMaximum(coll_cov_max_abs);
        collapsed_cov_hist->SetMinimum(-coll_cov_max_abs);
        cor_hist->SetMaximum(1);
        cor_hist->SetMinimum(-1);
        collapsed_cor_hist->SetMaximum(1);
        collapsed_cor_hist->SetMinimum(-1);

        ret["total_frac_cov"] = std::move(cov_hist);
        ret["collapsed_total_frac_cov"] = std::move(collapsed_cov_hist);
        ret["total_cor"] = std::move(cor_hist);
        ret["collapsed_total_cor"] = std::move(collapsed_cor_hist);

        for(const auto &name: syst.covar_names) {
            const Eigen::MatrixXf &covar = syst.GrabMatrix(name);
            const Eigen::MatrixXf &corr = syst.GrabCorrMatrix(name);

            std::unique_ptr<TH2D> cov_h = std::make_unique<TH2D>(("cov"+name).c_str(), (name+" Fractional Covariance;Bin # ;Bin #").c_str(), config.m_num_bins_total, 0, config.m_num_bins_total, config.m_num_bins_total, 0, config.m_num_bins_total);
            std::unique_ptr<TH2D> corr_h = std::make_unique<TH2D>(("cor"+name).c_str(), (name+" Correlation;Bin # ;Bin #").c_str(), config.m_num_bins_total, 0, config.m_num_bins_total, config.m_num_bins_total, 0, config.m_num_bins_total);
            for(size_t i = 0; i < config.m_num_bins_total; ++i){
                for(size_t j = 0; j < config.m_num_bins_total; ++j){
                    cov_h->SetBinContent(i+1,j+1,covar(i,j));
                    corr_h->SetBinContent(i+1,j+1,corr(i,j));
                }
            }

            float cov_max_abs = std::max(cov_h->GetMaximum(), std::abs(cov_h->GetMinimum()));
            cov_h->SetMaximum(cov_max_abs);
            cov_h->SetMinimum(-cov_max_abs);
            corr_h->SetMaximum(1);
            corr_h->SetMinimum(-1);

            ret[name+"_cov"] = std::move(cov_h);
            ret[name+"_corr"] = std::move(corr_h);
        }

        return ret;
    }

    std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> 
        getSplineGraphs(const PROsyst &systs, const PROconfig &config) {
            std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> spline_graphs;

            for(size_t i = 0; i < systs.GetNSplines(); ++i) {
                const std::string &name = systs.spline_names[i];
                const PROsyst::Spline &spline = systs.GrabSpline(name);
                //using Spline = std::vector<std::vector<std::pair<float, std::array<float, 4>>>>;
                std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>> bin_graphs;
                size_t nbins = 
                    systs.spline_binnings[i] == -2 ? config.m_num_truebins_total :
                    systs.spline_binnings[i] == -1 ? config.m_num_bins_total
                    : config.m_num_other_bins_total[systs.spline_binnings[i]];
                bin_graphs.reserve(nbins);

                for(size_t j = 0; j < nbins; ++j) {
                    const std::vector<std::pair<float, std::array<float, 4>>> &spline_for_bin = spline[j];
                    std::unique_ptr<TGraph> curve = std::make_unique<TGraph>();
                    std::unique_ptr<TGraph> fixed_pts = std::make_unique<TGraph>();
                    for(size_t k = 0; k < spline_for_bin.size(); ++k) {
                        //const auto &[lo, coeffs] = spline_for_bin[k];
                        float lo = spline_for_bin[k].first;
                        std::array<float, 4> coeffs = spline_for_bin[k].second;
                        float hi = k < spline_for_bin.size() - 1 ? spline_for_bin[k+1].first : systs.spline_hi[i];
                        auto fn = [coeffs](float shift){
                            return coeffs[0] + coeffs[1]*shift + coeffs[2]*shift*shift + coeffs[3]*shift*shift*shift;
                        };
                        fixed_pts->SetPoint(fixed_pts->GetN(), lo, fn(0)); 
                        if(k == spline_for_bin.size() - 1)
                            fixed_pts->SetPoint(fixed_pts->GetN(), hi, fn(hi - lo));
                        float width = (hi - lo) / 20;
                        for(size_t l = 0; l < 20; ++l)
                            curve->SetPoint(curve->GetN(), lo + l * width, fn(l * width));
                    }
                    bin_graphs.push_back(std::make_pair(std::move(fixed_pts), std::move(curve)));
                }
                spline_graphs[name] = std::move(bin_graphs);
            }

            return spline_graphs;
        }

    std::unique_ptr<TGraphAsymmErrors> getErrorBand(const PROconfig &config, const PROpeller &prop, const PROsyst &syst, bool scale, int other_index) {
        //TODO: Only works with 1 mode/detector/channel
        Eigen::VectorXf cv = other_index < 0 ? CollapseMatrix(config, FillCVSpectrum(config, prop, true).Spec()) :
            CollapseMatrix(config, FillOtherCVSpectrum(config, prop, other_index).Spec(), other_index);
        std::vector<float> edges = other_index < 0 ? config.GetChannelBinEdges(0) : config.GetChannelOtherBinEdges(0, other_index);
        log<LOG_DEBUG>(L"%1% || For other var %2% the cv is %3% and the edges are %4%")
            % __func__ % other_index % cv % edges;
        std::vector<float> centers;
        size_t nerrorsample = 5000;
        for(size_t i = 0; i < edges.size() - 1; ++i)
            centers.push_back((edges[i+1] + edges[i])/2);
        std::vector<Eigen::VectorXf> specs;
        std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());
        for(size_t i = 0; i < nerrorsample; ++i)
            specs.push_back(FillSystRandomThrow(config, prop, syst, dseed(PROseed::global_rng), other_index).Spec());
        //specs.push_back(CollapseMatrix(config, FillSystRandomThrow(config, prop, syst).Spec()));
        TH1D tmphist("th", "", cv.size(), edges.data());
        for(int i = 0; i < cv.size(); ++i)
            tmphist.SetBinContent(i+1, cv(i));
        if(scale) tmphist.Scale(1, "width");
        //std::unique_ptr<TGraphAsymmErrors> ret = std::make_unique<TGraphAsymmErrors>(cv.size(), centers.data(), cv.data());
        std::unique_ptr<TGraphAsymmErrors> ret = std::make_unique<TGraphAsymmErrors>(&tmphist);
        for(int i = 0; i < cv.size(); ++i) {
            std::vector<float> binconts(nerrorsample);
            for(size_t j = 0; j < nerrorsample; ++j) {
                binconts[j] = specs[j](i);
            }
            float scale_factor = tmphist.GetBinContent(i+1)/cv(i);
            if(std::isnan(scale_factor)) scale_factor = 1;
            std::sort(binconts.begin(), binconts.end());
            float ehi = std::abs((binconts[5*840] - cv(i))*scale_factor);
            float elo = std::abs((cv(i) - binconts[5*160])*scale_factor);
            ret->SetPointEYhigh(i, ehi);
            ret->SetPointEYlow(i, elo);

            log<LOG_DEBUG>(L"%1% || ErrorBand bin %2% %3% %4% %5% %6% %7%") % __func__ % i % cv(i) % ehi % elo % scale_factor % tmphist.GetBinContent(i+1);
        }
        return ret;
    }


    void plot_channels(const std::string &filename, const PROconfig &config, std::optional<PROspec> cv, std::optional<PROspec> best_fit, std::optional<PROdata> data, std::optional<TGraphAsymmErrors*> errband, std::optional<TGraphAsymmErrors*> posterrband, std::vector<TPaveText> &texts, PlotOptions opt, int other_index) {
        TCanvas c;
        c.Print((filename+"[").c_str());

        std::map<std::string, std::unique_ptr<TH1D>> cvhists;
        if(cv) cvhists = getCVHists(*cv, config, (bool)(opt & PlotOptions::BinWidthScaled), other_index);

        Eigen::VectorXf bf_spec;
        if(best_fit) {
            bf_spec = other_index < 0 ? CollapseMatrix(config, best_fit->Spec()) : CollapseMatrix(config, best_fit->Spec(), other_index);
        }

        std::string ytitle = bool(opt&PlotOptions::AreaNormalized)
            ? "Area Normalized"
            : bool(opt&PlotOptions::BinWidthScaled) 
            ? "Events/GeV" 
            : "Events";

        size_t global_subchannel_index = 0;
        size_t global_channel_index = 0;
        for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
            for(size_t det = 0; det < config.m_num_detectors; ++det) {
                for(size_t channel = 0; channel < config.m_num_channels; ++channel) {
                    size_t channel_nbins = other_index < 0 ? config.m_channel_num_bins[channel] : config.m_channel_num_other_bins[channel][other_index];

                    Color_t bfcol = TColor::GetColor(234, 67, 53);//ncie red
                    Color_t cvcol =  TColor::GetColor(66, 103, 210);//nice blue :)
                    if(!best_fit)cvcol=kBlack;

                    std::vector<float> edges = other_index < 0 ? config.GetChannelBinEdges(0) : config.GetChannelOtherBinEdges(0, other_index);
                    std::string xtitle = other_index < 0 ? config.m_channel_units[channel] : config.m_channel_other_units[channel][other_index];
                    std::string hist_title = config.m_detector_plotnames[det]  + " "+ config.m_channel_plotnames[channel]+";"+xtitle+";"+ytitle;
                    //std::unique_ptr<TLegend> leg = std::make_unique<TLegend>(0.11,0.75,0.89,0.89); 4
                    std::unique_ptr<TLegend> leg = std::make_unique<TLegend>(0.38,0.69,0.89,0.89);
                    leg->SetNColumns(2);
                    leg->SetFillStyle(0);
                    leg->SetLineWidth(0);
                    TH1D cv_hist(std::to_string(global_channel_index).c_str(), hist_title.c_str(), channel_nbins, edges.data());
                    cv_hist.SetLineWidth(2);
                    cv_hist.SetLineColor(cvcol);
                    cv_hist.SetFillStyle(0);
                    for(size_t bin = 0; bin < channel_nbins; ++bin) {
                        cv_hist.SetBinContent(bin+1, 0);
                    }
                    if(bool(opt&PlotOptions::BinWidthScaled))
                        cv_hist.Scale(1, "width");

                    // Set up TPads for ratios, unused if ratio option not chosen
                    TPad p1("p1", "p1", 0, 0.25, 1, 1);
                    p1.SetBottomMargin(0);

                    TPad p2("p2", "p2", 0, 0, 1, 0.25);
                    p2.SetTopMargin(0);
                    p2.SetBottomMargin(0.3);



                    THStack *cvstack = NULL;
                    if(cv) {
                        if(bool(opt&PlotOptions::CVasStack)) cvstack = new THStack(std::to_string(global_channel_index).c_str(), "");
                        std::vector<std::pair<std::string, const char*>> subplots;
                        for(size_t subchannel = 0; subchannel < config.m_num_subchannels[channel]; ++subchannel){
                            const std::string& subchannel_name  = config.m_fullnames[global_subchannel_index];
                            if(bool(opt&PlotOptions::CVasStack)) {
                                cvstack->Add(cvhists[subchannel_name].get());
                                subplots.push_back({subchannel_name, config.m_subchannel_plotnames[channel][subchannel].c_str()});
                            }
                            cv_hist.Add(cvhists[subchannel_name].get());
                            ++global_subchannel_index;
                        }
                        if(bool(opt&PlotOptions::CVasStack)) {
                            for(size_t sc = subplots.size(); sc > 0; --sc)
                                leg->AddEntry(cvhists[subplots[sc-1].first].get(), subplots[sc-1].second ,"f");
                        }
                        if(bool(opt&PlotOptions::AreaNormalized)) {
                            float integral = cv_hist.Integral();
                            cv_hist.Scale(1 / integral);
                            if(bool(opt&PlotOptions::CVasStack)) {
                                TList *stlists = (TList*)cvstack->GetHists();
                                for(const auto&& obj: *stlists){
                                    ((TH1*)obj)->Scale(1/integral);
                                }
                            }
                        }

                        TH1 *leg_hack = (TH1*)cv_hist.Clone();
                        leg_hack->SetFillStyle(3144);
                        leg_hack->SetFillColorAlpha(cvcol, 0.2);
                        //leg_hack->SetFillColorAlpha(kGray+2, 0.2);
                        leg_hack->SetLineColor(cvcol);
                        leg_hack->SetLineWidth(2);

                        if(errband){
                            leg->AddEntry(leg_hack,"CV Prediction #pm 1#sigma" ,"fl"); 
                        }else{
                            leg->AddEntry(&cv_hist, "CV Prediction #pm 1#sigma", "l");
                        }
                    }

                    TGraphAsymmErrors *channel_errband = NULL;
                    if(errband) {
                        channel_errband = new TGraphAsymmErrors(&cv_hist);
                        int channel_start = other_index < 0 ? config.GetCollapsedGlobalBinStart(global_channel_index) : config.GetCollapsedGlobalOtherBinStart(global_channel_index, other_index);

                        for(size_t bin = 0; bin < channel_nbins; ++bin) {
                            float scale = 1.0;
                            if(bool(opt&PlotOptions::AreaNormalized)) {
                                scale = channel_errband->GetPointY(bin) / (*errband)->GetPointY(bin+channel_start);
                            }
                            channel_errband->SetPointEYhigh(bin, scale*(*errband)->GetErrorYhigh(bin+channel_start));
                            channel_errband->SetPointEYlow(bin, scale*(*errband)->GetErrorYlow(bin+channel_start));
                        }
                        channel_errband->SetFillStyle(3144);
                        //channel_errband->SetFillColorAlpha(kGray+2, 0.2);
                        channel_errband->SetFillColorAlpha(cvcol, 0.2);
                        channel_errband->SetLineColor(cvcol);
                        channel_errband->SetLineWidth(1);
                        //leg->AddEntry(channel_errband, "#pm 1#sigma", "f");
                    }

                    TH1D bf_hist(("bf"+std::to_string(global_channel_index)).c_str(), "", channel_nbins, edges.data());
                    if(best_fit) {
                        int channel_start = other_index < 0 ? config.GetCollapsedGlobalBinStart(global_channel_index) : config.GetCollapsedGlobalOtherBinStart(global_channel_index, other_index);
                        for(size_t bin = 0; bin < channel_nbins; ++bin) {
                            bf_hist.SetBinContent(bin+1, bf_spec(bin+channel_start));
                        }
                        //bf_hist.SetLineColor(TColor::GetColor(234, 67, 53)); // pastel red
                        bf_hist.SetLineColor(bfcol); 
                        bf_hist.SetLineStyle(kDashed); 
                        bf_hist.SetLineWidth(2);
                        //leg->AddEntry(&bf_hist, "Best Fit", "l");

                        TH1 *leg_hack = (TH1*)bf_hist.Clone("bf");
                        leg_hack->SetFillStyle(3254);
                        leg_hack->SetFillColor(bfcol);
                        leg_hack->SetLineColor(bfcol);
                        leg_hack->SetLineWidth(2);

                        if(errband){
                            leg->AddEntry(leg_hack,"Best Fit #pm 1#sigma (post-fit)" ,"fl"); 
                        }else{
                            leg->AddEntry(&bf_hist, "Best Fit #pm 1#sigma (post-fit)", "l");
                        }
                        //cv_hist.Draw("hist");

                        if(bool(opt&PlotOptions::BinWidthScaled))
                            bf_hist.Scale(1, "width");
                        if(bool(opt&PlotOptions::AreaNormalized))
                            bf_hist.Scale(1.0/bf_hist.Integral());
                    }

                    TGraphAsymmErrors *post_channel_errband = NULL;
                    if(posterrband) {
                        post_channel_errband = new TGraphAsymmErrors(&bf_hist);
                        int channel_start = other_index < 0 ? config.GetCollapsedGlobalBinStart(global_channel_index) : config.GetCollapsedGlobalOtherBinStart(global_channel_index, other_index);
                        for(size_t bin = 0; bin < channel_nbins; ++bin) {
                            float scale = 1.0;
                            if(bool(opt&PlotOptions::AreaNormalized)) {
                                scale = post_channel_errband->GetPointY(bin) / (*posterrband)->GetPointY(bin+channel_start);
                            }
                            post_channel_errband->SetPointEYhigh(bin, scale*(*posterrband)->GetErrorYhigh(bin+channel_start));
                            post_channel_errband->SetPointEYlow(bin, scale*(*posterrband)->GetErrorYlow(bin+channel_start));
                        }
                        post_channel_errband->SetFillColor(bfcol);
                        post_channel_errband->SetFillStyle(3254);
                        post_channel_errband->SetLineColor(bfcol);
                        post_channel_errband->SetLineWidth(1);
                        //leg->AddEntry(post_channel_errband, "post-fit #pm 1#sigma", "f");
                    }

                    TH1D data_hist;
                    if(data) {

                        data_hist = data->toTH1D(config, global_channel_index, other_index);
                        data_hist.SetLineColor(kBlack);
                        data_hist.SetLineWidth(2);
                        data_hist.SetMarkerStyle(kFullCircle);
                        data_hist.SetMarkerColor(kBlack);
                        data_hist.SetMarkerSize(1);
                        std::string dat_str = "Data: ";
                        std::ostringstream oss;
                        int exponent = static_cast<int>(std::log10(std::abs(config.m_plot_pot)));
                        float mantissa = config.m_plot_pot/ std::pow(10, exponent);
                        oss << std::fixed << std::setprecision(2) << mantissa << "x10^{" << exponent << "} POT";
                        dat_str+= oss.str();
                        leg->AddEntry(&data_hist,dat_str.c_str(), "lp");
                        if(bool(opt&PlotOptions::BinWidthScaled))
                            data_hist.Scale(1, "width");
                        if(bool(opt&PlotOptions::AreaNormalized))
                            data_hist.Scale(1.0/data_hist.Integral());
                    }


                    /*******************/
                    /* Draw everything */
                    /*******************/
                    double top_modifier = 1.35;

                    if(bool(opt&PlotOptions::DataMCRatio) || bool(opt&PlotOptions::DataPostfitRatio))
                        p1.cd();


                    if(cv) {
                        if(bool(opt&PlotOptions::CVasStack)) {
                            cvstack->SetMaximum(std::max(top_modifier*cvstack->GetMaximum(),top_modifier*data_hist.GetMaximum()));
                            cvstack->Draw("hist");
                            cv_hist.Draw("same hist");

                        } else {
                            cv_hist.SetMaximum(top_modifier*cv_hist.GetMaximum());

                            cv_hist.Draw("hist");
                            cv_hist.GetYaxis()->SetTitleSize(0.06);  
                            cv_hist.GetYaxis()->SetTitleOffset(0.75);
                            cv_hist.GetYaxis()->SetLabelSize(0.05);
                            cv_hist.SetMinimum(0.01);
                        }
                    }

                    if(errband) channel_errband->Draw("2 same");

                    if(best_fit) {
                        bf_hist.SetTitle("");
                        if(cv) bf_hist.Draw("hist same");
                        else bf_hist.Draw("hist");
                    }

                    if(posterrband) post_channel_errband->Draw("2 same");

                    if(data) {
                        TGraphErrors *g = new TGraphErrors(data_hist.GetNbinsX());
                        float datmax =-999;
                        for (int i = 1; i <= data_hist.GetNbinsX(); ++i) {
                            double x = data_hist.GetBinCenter(i);
                            double y = data_hist.GetBinContent(i);
                            double ex = 0;
                            double ey = data_hist.GetBinError(i);
                            if(y>datmax){
                                datmax=y;
                            }
                            g->SetPoint(i - 1, x, y);
                            g->SetPointError(i - 1, ex, ey);
                            g->SetLineColor(kBlack);
                            g->SetLineWidth(2);
                            g->SetMarkerStyle(kFullCircle);
                            g->SetMarkerColor(kBlack);
                            g->SetMarkerSize(1);

                        }


                        if(cv || best_fit) {
                            g->Draw("PE1 same");
                            cv_hist.SetMaximum(std::max(cv_hist.GetMaximum(),top_modifier*datmax));

                        } else {
                            g->Draw("PE1");
                        }
                    }

                    if(texts.size()!=0) {
                        TLine *dummy_line = new TLine(0,0,0.1,0);
                        dummy_line->SetLineColor(kWhite);
                        dummy_line->SetLineWidth(0);
                        if(texts.size()==1){
                            //texts.front().Draw("same");
                            TText* text = (TText*)texts.front().GetListOfLines()->First();
                            const char* label = text->GetTitle(); 
                            leg->AddEntry(dummy_line,label,"l"); 
                        }else{
                            //texts[global_channel_index].Draw("same");
                            TText* text = (TText*)texts.at(global_channel_index).GetListOfLines()->First();
                            const char* label = text->GetTitle(); 
                            leg->AddEntry(dummy_line,label,"l"); 
                        }
                    }

                    leg->Draw("same");

                    TH1D *ratio, *one;
                    TGraphAsymmErrors *ratio_err;
                    if(bool(opt&PlotOptions::DataMCRatio) || bool(opt&PlotOptions::DataPostfitRatio)) {
                        p2.cd();

                        std::string y_title = bool(opt&PlotOptions::DataMCRatio) ? "Data/MC" : "Data/Best-Fit";
                        ratio = new TH1D(("rat"+std::to_string(global_channel_index)).c_str(), (";"+xtitle+";"+y_title).c_str(), channel_nbins, edges.data());
                        one = new TH1D(("one"+std::to_string(global_channel_index)).c_str(), (";"+xtitle+";"+y_title).c_str(), channel_nbins, edges.data());
                        ratio_err = new TGraphAsymmErrors(); 
                        *ratio_err = bool(opt&PlotOptions::DataMCRatio)
                            ? *channel_errband
                            : *post_channel_errband;


                       
                        float ymin = 1e9, ymax = -1e9;

                        for(size_t i = 0; i < channel_nbins; ++i) {
                            float numerator = data_hist.GetBinContent(i+1);
                            float denonminator = 
                                bool(opt&PlotOptions::DataMCRatio)
                                ? cv_hist.GetBinContent(i+1)
                                : bf_hist.GetBinContent(i+1);
                            float rat = numerator/denonminator;
                            if(isnan(rat)) rat = 1;
                            ratio->SetBinError(i+1, 1.0 / sqrt(numerator));
                            ratio->SetBinContent(i+1, rat);
                            one->SetBinContent(i+1, 1.0);
                            ratio_err->SetPointEYhigh(i, ratio_err->GetErrorYhigh(i)/ratio_err->GetPointY(i));
                            ratio_err->SetPointEYlow(i, ratio_err->GetErrorYlow(i)/ratio_err->GetPointY(i));
                            ratio_err->SetPointY(i, 1.0);
                            ymin = std::min(ymin, rat);
                            ymax = std::max(ymax, rat);


                        }
                        for (int i = 0; i < ratio_err->GetN(); ++i) {
                            float y, eyh, eyl;
                            y = ratio_err->GetPointY(i);
                            eyh = ratio_err->GetErrorYhigh(i);
                            eyl = ratio_err->GetErrorYlow(i);
                            ymin = std::min(ymin, y - eyl);
                            ymax = std::max(ymax, y + eyh);

                        }
                        float yrange = ymax - ymin;
                        float ylow = ymin - 0.05 * yrange;  // 15% padding below
                        float yhigh = ymax + 0.05 * yrange; // 15% padding above

                        if(!posterrband){
                            //one->SetMinimum(std::max(0.5f,std::min(ylow,0.85f)));
                            //one->SetMaximum(std::min(1.5f,std::max(yhigh,1.148f)));
                            one->SetMinimum(std::min(ylow,0.85f));
                            one->SetMaximum(std::max(yhigh,1.148f));

                        }else{
                            one->SetMinimum(std::min(ylow,0.85f));
                            one->SetMaximum(std::max(yhigh,1.148f));

                        }

                        one->SetLineColor(kBlack);
                        one->SetLineStyle(kDashed);
                        one->Draw("hist");
                        one->SetTitle("");
                        one->GetYaxis()->SetTitleSize(0.15);
                        one->GetYaxis()->SetLabelSize(0.12);
                        one->GetXaxis()->SetTitleSize(0.14);
                        one->GetXaxis()->SetLabelSize(0.14);
                        one->GetYaxis()->SetTitleOffset(0.21);
                        one->GetXaxis()->SetTitleOffset(0.85);
                        ratio->SetLineColor(kBlack);
                        ratio->SetLineWidth(2);
                        ratio->SetMarkerStyle(kFullCircle);
                        ratio->SetMarkerColor(kBlack);
                        ratio->SetMarkerSize(1);

                        //ratio_err->SetFillColor(kRed);
                        //ratio_err->SetFillStyle(3345);
                        ratio_err->Draw("2 same");

                        ratio->Draw("PE1 E0 same");

                        c.cd();
                        p1.Draw();
                        p2.Draw();
                    }

                    c.Print(filename.c_str());

                    ++global_channel_index;
                }
            }
        }
        c.Print((filename+"]").c_str());
    }


    int plotPriorFractionalSystematicBreakdown(const PROconfig &config, const PROspec &spec, const PROsyst &allsplinesyst, std::string filename) {
        //Input PROsyst needs to be the allsplinesyst for now

        std::vector<int> colors = {
            kAzure+1,      // Light blue
            kRed+1,        // Bright red
            kGreen+3,      // Medium green
            kOrange+7,      // Deep orange
            kBlue+2,        // Darker blue
            kViolet+2,      // Purple/violet
            kGray+1,         // Light gray
            kYellow+2,      // Golden yellow
            kTeal+3,        // Teal
            kPink+2,        // Pink
            kMagenta+2,     // Magenta
            kSpring+5      // Blue-green
        };

        std::vector<int> line_styles = {
            1,  // Solid (base style)
            1,  // Dashed
        };



        //some testing
        for (const auto& [syst_name, tags] : config.m_mcgen_variation_tags) {
            std::string tag_list;
            for (const auto& tag : tags) {
                tag_list += tag + ", ";
            }
            if (!tag_list.empty()) {
                tag_list.erase(tag_list.size() - 2);  // Remove trailing ", "
            }
            log<LOG_INFO>(L"Systematic: %1% | Tags: [%2%]") 
                % syst_name.c_str() 
                % tag_list.c_str();
        }

        //This is for prior everthing of course
        std::map<std::string,std::vector<std::string>> used_tags;
        for(const auto &name: allsplinesyst.covar_names){
            log<LOG_INFO>(L"%1% || Systematic %2% ") % __func__ % name.c_str();
            auto it = config.m_mcgen_variation_tags.find(name);
            if (it == config.m_mcgen_variation_tags.end()) {
                log<LOG_WARNING>(L"%1% || Systematic %2% not in tags map") % __func__ % name.c_str();
                continue;
            }
            const vector<std::string>& mtags = it->second;
            log<LOG_INFO>(L"%1% || -- has tags %2%") % __func__ %  mtags;
            for(auto &t: mtags){
                used_tags[t].push_back(name);
            }
        }
        for(const auto &[tag, vec]: used_tags) {
            log<LOG_INFO>(L"%1% || So for tag %2% we include %3%") % __func__ % tag.c_str() % vec;
        }



        int nTags = used_tags.size()+1;
        int gridCols = std::ceil(std::sqrt(nTags));
        int gridRows = std::ceil(nTags / float(gridCols));

        TCanvas c("c", "Systematics Comparison", gridCols*1600, gridRows*1200);  
        c.Print((filename+"[").c_str());
        c.Divide(gridCols, gridRows);

        Eigen::MatrixXf diag = spec.Spec().array().matrix().asDiagonal(); 
        Eigen::MatrixXf collapsed_diag = CollapseMatrix(config, diag);

        size_t global_subchannel_index = 0;
        size_t global_channel_index = 0;
        for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
            for(size_t det = 0; det < config.m_num_detectors; ++det) {
                for(size_t channel = 0; channel < config.m_num_channels; ++channel) {

                    c.Clear();
                    c.Divide(gridCols, gridRows);

                    int padIndex = 1;

                    std::vector<float> bin_edges = config.GetChannelBinEdges(global_channel_index);
                    size_t binstart = config.GetCollapsedGlobalBinStart(global_channel_index);
                    size_t nbins = config.m_channel_num_bins[channel];
                    std::vector<int> channel_bins(nbins);
                    std::iota(channel_bins.begin(), channel_bins.end(), binstart);

                    std::vector<TH1F*> vsums;
                    std::vector<std::string> vnames;
                    for (const auto &[tag, vec] : used_tags) {

                        c.cd(padIndex++);
                        bool first=true;

                        TLegend* leg = new TLegend(0.11, 0.6, 0.89, 0.89);
                        leg->SetNColumns(3);
                        //leg->SetHeader(tag.c_str(), "C");  // Center-aligned header


                        TH1F* hsum = new TH1F( ("Sum_"+tag+"_"+std::to_string(global_channel_index)).c_str(), tag.c_str(), bin_edges.size()-1, bin_edges.data());
                        hsum->Reset();
                        std::vector<TH1F*> hvec;
                        int i =0;
                        for(const auto & systname:vec){

                            Eigen::MatrixXf frac_covariance = allsplinesyst.GrabMatrix(systname);
                            Eigen::MatrixXf full_covariance = diag*(frac_covariance)*diag;
                            Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance);
                            Eigen::MatrixXf collapsed_frac_covariance = collapsed_diag.inverse()*collapsed_full_covariance*collapsed_diag.inverse();


                            //submatix ffractional
                            Eigen::MatrixXf channel_cov = collapsed_frac_covariance(channel_bins, channel_bins);

                            log<LOG_INFO>(L"%1% || Channel: %2%/%3% | Det: %4%/%5% | Mode: %6%/%7% | Tag: %8% | Syst: %9% | Bins: %10% [%11%:%12%]") 
                                % __func__ 
                                % channel % config.m_num_channels
                                % det % config.m_num_detectors
                                % mode % config.m_num_modes
                                % tag.c_str() 
                                % systname.c_str()
                                % nbins
                                % binstart % (binstart + nbins - 1);

                            int color_idx = i % colors.size();
                            int style_idx = (i / 4) % line_styles.size();  
                            i++;

                            TH1F* h = new TH1F((tag+"_Channel_"+std::to_string(global_channel_index)+"_"+std::to_string(i)).c_str(), tag.c_str(), bin_edges.size()-1, bin_edges.data());

                            for (size_t i = 0; i < nbins; ++i) {
                                h->SetBinContent(i+1, sqrt(channel_cov(i,i)));
                                hsum->SetBinContent(i+1, hsum->GetBinContent(i+1)+channel_cov(i,i));
                            }

                            const std::string &plotname = config.m_mcgen_variation_plotname_map.at(systname);
                            leg->AddEntry(h, plotname.c_str(), "l");
                            h->SetLineColor(colors[color_idx]);
                            h->SetLineStyle(line_styles[style_idx]);
                            hvec.push_back(h);

                        }//end syst
                        for (size_t i = 0; i < nbins; ++i) {
                            hsum->SetBinContent(i+1, sqrt(hsum->GetBinContent(i+1)));
                        }
                        leg->AddEntry(hsum,"Sum","l");

                        hsum->SetXTitle(config.m_channel_plotnames[channel].c_str());
                        hsum->SetYTitle("Fractional Uncertainty");
                        hsum->SetLineColor(kBlack);
                        hsum->SetLineWidth(2);
                        hsum->SetLineStyle(1);
                        hsum->SetMinimum(0);
                        hsum->SetStats(0);  
                        hsum->Draw("HIST");
                        hsum->SetMaximum(hsum->GetMaximum()*1.7);
                        gPad->Modified();
                        gPad->Update();


                        vsums.push_back(hsum);
                        vnames.push_back(tag);
                        for(auto &h:hvec) h->Draw("HIST SAME");

                        leg->Draw();
                    }//end tag


                    //and each sum of sums to wrap it off!
                    c.cd(padIndex++);

                    TH1F* hsum = new TH1F( ("USum_"+std::to_string(global_channel_index)).c_str(),"Summary!", bin_edges.size()-1, bin_edges.data());
                    hsum->Reset();
                    TLegend* leg = new TLegend(0.11, 0.6, 0.89, 0.89);
                    leg->SetNColumns(3);
                    std::vector<TH1F*> hvec;
                    for(size_t t=0; t< vsums.size(); t++){
                        int color_idx = t % colors.size();
                        for (size_t i = 0; i < nbins; ++i) {
                            hsum->SetBinContent(i+1, hsum->GetBinContent(i+1)+pow(vsums.at(t)->GetBinContent(i+1),2));
                        }
                        TH1F * h = (TH1F*)vsums.at(t)->Clone((to_string(global_channel_index)+vnames[t]).c_str());
                        leg->AddEntry(h, vnames[t].c_str(), "l");
                        h->SetLineColor(colors[color_idx]);
                        h->SetLineStyle(1);
                        h->SetLineWidth(1);
                        hvec.push_back(h);
                    }

                    for (size_t i = 0; i < nbins; ++i) {
                        hsum->SetBinContent(i+1, sqrt(hsum->GetBinContent(i+1)));
                    }
                    leg->AddEntry(hsum,"Sum","l");
                    hsum->SetXTitle(config.m_channel_plotnames[channel].c_str());
                    hsum->SetTitle("Summary!");
                    hsum->SetYTitle("Fractional Uncertainty");
                    hsum->SetLineColor(kBlack);
                    hsum->SetLineWidth(2);
                    hsum->SetLineStyle(1);
                    hsum->SetMinimum(0);
                    hsum->SetStats(0);  
                    hsum->Draw("HIST");
                    hsum->SetMaximum(hsum->GetMaximum()*1.7);
                    gPad->Modified();
                    gPad->Update();
                    for(auto &h:hvec) h->Draw("HIST SAME");
                    leg->Draw();


                    c.Print(filename.c_str());
                    global_channel_index++;
                }
            }
        }



        c.Print((filename+"]").c_str());
        return 0;
    };
}
