#include "PROplot.h"
#include "TStyle.h"
#include <cmath>
#include <Eigen/SVD>

namespace PROfit{

    std::map<std::string, std::unique_ptr<TH1D>> getCV1DHists(const PROspec &spec, const PROconfig& inconfig, bool scale, int other_index) {
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
                        std::unique_ptr<TH1D> htmp_slc = std::make_unique<TH1D>(spec.toTH1DSlices(inconfig, global_subchannel_index, other_index));
                        if(htmp_slc){
                            hists[subchannel_name+"slc"] = std::move(htmp_slc);
                        }
                        ++global_subchannel_index;
                    }//end subchan
                    ++global_channel_index;
                }//end chan
            }//end det
        }//end mode
        return hists;
    }

    std::map<std::string, std::unique_ptr<TH2D>> getCV2DHists(const PROspec &spec, const PROconfig& inconfig, bool scale, int other_index) {
        std::map<std::string, std::unique_ptr<TH2D>> hists;

        size_t global_subchannel_index = 0;
        size_t global_channel_index = 0;
        for(size_t im = 0; im < inconfig.m_num_modes; im++){
            for(size_t id = 0; id < inconfig.m_num_detectors; id++){
                for(size_t ic = 0; ic < inconfig.m_num_channels; ic++){
                    for(size_t sc = 0; sc < inconfig.m_num_subchannels[ic]; sc++){
                        const std::string& subchannel_name  = inconfig.m_fullnames[global_subchannel_index];
                        const std::string& color = inconfig.m_subchannel_colors[ic][sc];
                        std::unique_ptr<TH2D> htmp = std::make_unique<TH2D>(spec.toTH2D(inconfig, global_subchannel_index, other_index));
                        if(scale) htmp->Scale(1,"width");
                        hists[subchannel_name] = std::move(htmp);
                        ++global_subchannel_index;
                    }//end subchan
                    ++global_channel_index;
                }//end chan
            }//end det
        }//end mode
        return hists;
    }

    std::map<std::string, std::unique_ptr<TH2D>> covarianceTH2D(const PROsyst &syst, const PROconfig &config, const PROspec &cv) {
        log<LOG_DEBUG>(L"%1% || just inside") % __func__;
        std::map<std::string, std::unique_ptr<TH2D>> ret;
        Eigen::MatrixXf fractional_cov = syst.fractional_covariance;
        Eigen::MatrixXf diag = cv.Spec().array().matrix().asDiagonal(); 
        Eigen::MatrixXf full_covariance =  diag*fractional_cov*diag;
        Eigen::MatrixXf collapsed_full_covariance =  CollapseMatrix(config,full_covariance);  
        Eigen::VectorXf collapsed_cv = CollapseMatrix(config, cv.Spec());
        Eigen::MatrixXf collapsed_cv_inv_diag = collapsed_cv.asDiagonal().inverse();
        Eigen::MatrixXf collapsed_frac_cov = collapsed_cv_inv_diag * collapsed_full_covariance * collapsed_cv_inv_diag;

        std::unique_ptr<TH2D> cov_hist = std::make_unique<TH2D>("cov", "Fractional Covariance Matrix;Bin # ;Bin #", config.m_num_variable_bins_total[config.i_prime], 0, config.m_num_variable_bins_total[config.i_prime], config.m_num_variable_bins_total[config.i_prime], 0, config.m_num_variable_bins_total[config.i_prime]);
        std::unique_ptr<TH2D> collapsed_cov_hist = std::make_unique<TH2D>("ccov", "Collapsed Fractional Covariance Matrix;Bin # ;Bin #", config.m_num_variable_bins_total_collapsed[config.i_prime], 0, config.m_num_variable_bins_total_collapsed[config.i_prime], config.m_num_variable_bins_total_collapsed[config.i_prime], 0, config.m_num_variable_bins_total_collapsed[config.i_prime]);

        std::unique_ptr<TH2D> cor_hist = std::make_unique<TH2D>("cor", "Correlation Matrix;Bin # ;Bin #", config.m_num_variable_bins_total[config.i_prime], 0, config.m_num_variable_bins_total[config.i_prime], config.m_num_variable_bins_total[config.i_prime], 0, config.m_num_variable_bins_total[config.i_prime]);
        std::unique_ptr<TH2D> collapsed_cor_hist = std::make_unique<TH2D>("ccor", "Collapsed Correlation Matrix;Bin # ;Bin #", config.m_num_variable_bins_total_collapsed[config.i_prime], 0, config.m_num_variable_bins_total_collapsed[config.i_prime], config.m_num_variable_bins_total_collapsed[config.i_prime], 0, config.m_num_variable_bins_total_collapsed[config.i_prime]);

        for(size_t i = 0; i < config.m_num_variable_bins_total[config.i_prime]; ++i)
            for(size_t j = 0; j < config.m_num_variable_bins_total[config.i_prime]; ++j){
                cov_hist->SetBinContent(i+1,j+1,fractional_cov(i,j));
                cor_hist->SetBinContent(i+1,j+1,fractional_cov(i,j)/(sqrt(fractional_cov(i,i))*sqrt(fractional_cov(j,j))));
            }

        for(size_t i = 0; i < config.m_num_variable_bins_total_collapsed[config.i_prime]; ++i)
            for(size_t j = 0; j < config.m_num_variable_bins_total_collapsed[config.i_prime]; ++j){
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

            std::unique_ptr<TH2D> cov_h = std::make_unique<TH2D>(("cov"+name).c_str(), (name+" Fractional Covariance;Bin # ;Bin #").c_str(), config.m_num_variable_bins_total[config.i_prime], 0, config.m_num_variable_bins_total[config.i_prime], config.m_num_variable_bins_total[config.i_prime], 0, config.m_num_variable_bins_total[config.i_prime]);
            std::unique_ptr<TH2D> corr_h = std::make_unique<TH2D>(("cor"+name).c_str(), (name+" Correlation;Bin # ;Bin #").c_str(), config.m_num_variable_bins_total[config.i_prime], 0, config.m_num_variable_bins_total[config.i_prime], config.m_num_variable_bins_total[config.i_prime], 0, config.m_num_variable_bins_total[config.i_prime]);
            for(size_t i = 0; i < config.m_num_variable_bins_total[config.i_prime]; ++i){
                for(size_t j = 0; j < config.m_num_variable_bins_total[config.i_prime]; ++j){
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

    std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>, std::unique_ptr<TGraph>>>>
        getSplineGraphs(const PROsyst &systs, const PROconfig &config) {
            std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>, std::unique_ptr<TGraph>>>> spline_graphs;

            for (size_t i = 0; i < systs.GetNSplines(); ++i) {
                const std::string &name = systs.spline_names[i];
                const Spline &spline = systs.GrabSpline(name);
                std::vector<std::pair<std::unique_ptr<TGraph>, std::unique_ptr<TGraph>>> bin_graphs;
                size_t nbins = config.m_num_variable_bins_total.at(systs.spline_binnings[i]);
                int nsegs = spline.segments_per_bin;
                bin_graphs.reserve(nbins);

                for (size_t j = 0; j < nbins; ++j) {
                    std::unique_ptr<TGraph> curve = std::make_unique<TGraph>();
                    std::unique_ptr<TGraph> fixed_pts = std::make_unique<TGraph>();

                    // Access the segment range for this bin
                    size_t seg_offset = j * nsegs;

                    for (int k = 0; k < nsegs; ++k) {
                        const SplineSegment &seg = spline.segments[seg_offset + k];
                        float lo = seg.knot;
                        std::array<float, 4> coeffs = seg.coeffs;
                        // Determine hi for this segment
                        float hi;
                        if (k < nsegs - 1) {
                            hi = spline.segments[seg_offset + k + 1].knot;
                        } else {
                            hi = systs.spline_hi[i];
                        }
                        auto fn = [coeffs](float shift) {
                            return coeffs[0] + coeffs[1] * shift + coeffs[2] * shift * shift + coeffs[3] * shift * shift * shift;
                        };
                        fixed_pts->SetPoint(fixed_pts->GetN(), lo, fn(0));
                        if (k == nsegs - 1)
                            fixed_pts->SetPoint(fixed_pts->GetN(), hi, fn(hi - lo));
                        float width = (hi - lo) / 20.0f;
                        for (int l = 0; l < 20; ++l)
                            curve->SetPoint(curve->GetN(), lo + l * width, fn(l * width));
                    }
                    bin_graphs.push_back(std::make_pair(std::move(fixed_pts), std::move(curve)));
                }
                spline_graphs[name] = std::move(bin_graphs);
            }

            return spline_graphs;
        }
    PROerrorbar getErrorBand(const PROconfig &config, const PROpeller &prop, const PROsyst &syst, const PROmodel &model, const PROspec &cv_spec, const Eigen::VectorXf &cvparams,bool scale, int other_index) {

        Eigen::VectorXf cv = CollapseMatrix(config, cv_spec.Spec(), other_index);

        std::vector<float> centers;
        size_t global_channel_index = 0;
        for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
            for(size_t det = 0; det < config.m_num_detectors; ++det) {
                for(size_t channel = 0; channel < config.m_num_channels; ++channel) {
                    std::vector<float> tedges =  config.GetChannelVariableBins(global_channel_index, other_index).Edges();
                    global_channel_index++;
                    for(size_t p=0; p<tedges.size(); p++){
                        if(p<tedges.size()-1){
                            centers.push_back((tedges[p+1]+tedges[p])/2.0);
                        }
                    }

                }
            }
        }

        size_t nerrorsample = 2500;

        std::vector<Eigen::VectorXf> specs;
        std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());

        //Fills already collapsed
        for(size_t i = 0; i < nerrorsample; ++i){
            Eigen::VectorXf var = FillSystRandomThrow(config, prop, syst, model,cv_spec, cvparams, dseed(PROseed::global_rng), other_index).Spec();
            specs.push_back(var);

        }

        PROerrorbar ebar(cv.size());
        for(int i = 0; i < cv.size(); ++i) {
            std::vector<float> binconts(nerrorsample);
            for(size_t j = 0; j < nerrorsample; ++j) {
                binconts[j] = specs[j](i);
            }
            float scale_factor = scale ? 1.0/config.collapsed_bin_widths.at(other_index)(i) :  1.0;
            if(std::isnan(scale_factor)) scale_factor = 1;
            std::sort(binconts.begin(), binconts.end());
            float ehi = std::abs((binconts[2.5*840] - cv(i))*scale_factor);
            float elo = std::abs((cv(i) - binconts[2.5*160])*scale_factor);
            ebar.error_up(i) =  ehi;
            ebar.error_down(i) =  elo;
            ebar.error_point(i) = cv(i)*scale_factor;
        }
        return ebar;
    }

    // sort through generic PROspec and combine bins to get projection
    Eigen::VectorXf make_1d_spec(Eigen::VectorXf input_spec, size_t nbinsx, size_t nbinsy=1, int offset = 0, int dims=1){

        log<LOG_DEBUG>(L"%1% || Making 1d spec nbinsx %2% nbinsy %3% offset %4% dims %5%") % __func__ % nbinsx % nbinsy % offset % dims;
        Eigen::VectorXf output_spec_1d = Eigen::VectorXf::Zero(nbinsx);
        for(size_t bx = 0; bx < nbinsx; bx++){
            if(dims == 2){
                for(size_t by = 0; by < nbinsy; by++){
                    // default is (x1, y1)...(x1,yn), (x2,y1)...
                    size_t b = bx*nbinsy + by;
                    output_spec_1d(bx) = output_spec_1d(bx)+input_spec(b+offset);
                }
            }
            else{
                output_spec_1d(bx) = input_spec(bx+offset);
            }
        }

        log<LOG_DEBUG>(L"%1% || Done with 1d spec") % __func__;
        return output_spec_1d;
    }

    PROerrorbar* make_1d_err(PROerrorbar errband, size_t nbinsx, size_t nbinsy=1, int offset = 0, int dims=1){
        PROerrorbar* errband_1d = new PROerrorbar(nbinsx);
        log<LOG_DEBUG>(L"%1% || input err_band pt: %2%") % __func__ % errband.error_point;
        log<LOG_DEBUG>(L"%1% || input err_band up: %2%") % __func__ % errband.error_up;
        // sum error points, combine errors in quadrature
        for(size_t bx = 0; bx < nbinsx; bx++){
            if(dims == 2){
                double up_err_sq = 0;
                double down_err_sq = 0;
                for(size_t by = 0; by < nbinsy; by++){
                    size_t b = bx*nbinsy + by;
                    errband_1d->error_point(bx) += errband.error_point(b+offset);
                    up_err_sq += std::pow(errband.error_up(b+offset), 2);
                    down_err_sq += std::pow(errband.error_down(b+offset), 2);
                }
                errband_1d->error_up(bx) = std::sqrt(up_err_sq);
                errband_1d->error_down(bx) = std::sqrt(down_err_sq);
            }
            else{
                errband_1d->error_point(bx) = errband.error_point(bx+offset);
                errband_1d->error_up(bx) = errband.error_up(bx+offset);
                errband_1d->error_down(bx) = errband.error_down(bx+offset);
            }
        }
        log<LOG_DEBUG>(L"%1% || err_band pt: %2%") % __func__ % errband_1d->error_point;
        log<LOG_DEBUG>(L"%1% || err_band up: %2%") % __func__ % errband_1d->error_up;
        return errband_1d;
    }

    double ratio_err(double A, double dA, double B, double dB, double corr){
        return (A/B)*sqrt(pow(dA/A,2)+pow(dB/B,2)-2*(corr*dA*dB/(A*B)));
    }

    void plot_detector_ratios(
            const PROconfig &config,
            std::vector<TH1D> data_hists, 
            std::vector<TH1D> cv_hists, 
            std::optional<PROerrorbar> errband, 
            std::vector<TH1D> bf_hists, 
            std::optional<PROerrorbar> posterrband, 
            TH2D &pre_corr, 
            TH2D &post_corr, 
            std::string filename,
            int var_index)
    {
        log<LOG_DEBUG>(L"%1% || Starting plot_detector_ratios") % __func__;

        size_t n_hists = data_hists.size();

        if(n_hists < 2) {
            log<LOG_WARNING>(L"%1% || Need at least 2 channels to compute ratios. Found %2%.") 
                % __func__ % n_hists;
            return;
        }

        log<LOG_INFO>(L"%1% || Creating ratio plots for %2% channels (%3% unique pairs)") 
            % __func__ % n_hists % (n_hists * (n_hists - 1) / 2);

        //make opffsets
        std::vector<int> bin_offsets(n_hists + 1, 0);
        for(size_t i = 0; i < n_hists; ++i) {
            bin_offsets[i + 1] = bin_offsets[i] + cv_hists[i].GetNbinsX();
        }

        // Build channel labels from config (mode_detector_channel naming)
        std::vector<std::string> channel_labels;
        std::vector<std::string> channel_short_labels;  // For filenames
        size_t hist_idx = 0;
        for(size_t mode = 0; mode < config.m_num_modes && hist_idx < n_hists; ++mode) {
            for(size_t det = 0; det < config.m_num_detectors && hist_idx < n_hists; ++det) {
                for(size_t channel = 0; channel < config.m_num_channels && hist_idx < n_hists; ++channel) {
                    std::string label = config.m_mode_plotnames[mode] + " " + 
                        config.m_detector_plotnames[det] + " " + 
                        config.m_channel_plotnames[channel];
                    channel_labels.push_back(label);

                    std::string short_label = config.m_mode_names[mode] + "_" + 
                        config.m_detector_names[det] + "_" + 
                        config.m_channel_names[channel];
                    channel_short_labels.push_back(short_label);
                    ++hist_idx;
                }
            }
        }

        // Fallback labels if indexing doesn't match expected structure
        while(channel_labels.size() < n_hists) {
            size_t idx = channel_labels.size();
            channel_labels.push_back("Channel " + std::to_string(idx));
            channel_short_labels.push_back("ch" + std::to_string(idx));
        }

        bool has_bf = (bf_hists.size() == n_hists);
        if(!has_bf) {
            log<LOG_DEBUG>(L"%1% || Best-fit histograms not available or count mismatch. Skipping BF in ratio plots.") 
                % __func__;
        }

        for(size_t idx_den = 0; idx_den < n_hists; ++idx_den) {           // denominator (bottom of ratio)
            for(size_t idx_num = 0; idx_num < n_hists; ++idx_num) {       // numerator (top of ratio)
                if(idx_num == idx_den) continue;  // Skip self-ratios
                size_t local_channel_num = idx_num % config.m_num_channels;
                size_t local_channel_den = idx_den % config.m_num_channels;
                if(local_channel_num != local_channel_den) continue;


                log<LOG_DEBUG>(L"%1% || Processing ratio: %2% / %3%") 
                    % __func__ % channel_labels[idx_num].c_str() % channel_labels[idx_den].c_str();

                // Get histograms for this pair
                TH1D& data_num = data_hists[idx_num];
                TH1D& data_den = data_hists[idx_den];
                TH1D& cv_num = cv_hists[idx_num];
                TH1D& cv_den = cv_hists[idx_den];

                // Verify bin counts match
                int nbins_num = data_num.GetNbinsX();
                int nbins_den = data_den.GetNbinsX();

                if(nbins_num != nbins_den) {
                    log<LOG_WARNING>(L"%1% || Skipping ratio %2%/%3%: bin count mismatch (%4% vs %5%)") 
                        % __func__ % idx_num % idx_den % nbins_num % nbins_den;
                    continue;
                }

                int channel_nbins = nbins_num;
                int offset_num = bin_offsets[idx_num];
                int offset_den = bin_offsets[idx_den];

                // Create ratio histograms
                TAxis* xAxis = data_num.GetXaxis();
                const Double_t* binEdges = xAxis->GetXbins()->GetArray();

                std::string ratio_label = channel_labels[idx_num] + " / " + channel_labels[idx_den];
                std::string name_suffix = "_" + std::to_string(idx_num) + "_over_" + std::to_string(idx_den);

                TH1D* data_ratio = new TH1D(("Data" + name_suffix).c_str(), 
                        ("Data;" + std::string(xAxis->GetTitle()) + ";" + ratio_label).c_str(), 
                        channel_nbins, binEdges);
                TH1D* cv_ratio = new TH1D(("CV" + name_suffix).c_str(), 
                        ("CV MC;" + std::string(xAxis->GetTitle()) + ";" + ratio_label).c_str(), 
                        channel_nbins, binEdges);
                TH1D* bf_ratio = nullptr;
                TH1D* err_ratio = nullptr;

                if(has_bf) {
                    bf_ratio = new TH1D(("bf" + name_suffix).c_str(), 
                            ("Best-Fit MC;" + std::string(xAxis->GetTitle()) + ";" + ratio_label).c_str(), 
                            channel_nbins, binEdges);
                    err_ratio = new TH1D(("err" + name_suffix).c_str(), 
                            (";" + std::string(xAxis->GetTitle()) + ";Post/Pre Err Ratio").c_str(), 
                            channel_nbins, binEdges);
                }

                Color_t cvcol = TColor::GetColor(66, 103, 210);  // Nice blue

                cv_ratio->SetLineWidth(2);
                cv_ratio->SetLineColor(cvcol);

                data_ratio->SetLineColor(kBlack);
                data_ratio->SetLineWidth(2);
                data_ratio->SetMarkerStyle(kFullCircle);
                data_ratio->SetMarkerColor(kBlack);
                data_ratio->SetMarkerSize(1);

                TLegend* leg = new TLegend(0.15, 0.7, 0.89, 0.89);

                // Error band graphs
                TGraphAsymmErrors* channel_errband = new TGraphAsymmErrors(&cv_num);
                TGraphAsymmErrors* post_channel_errband = nullptr;
                if(has_bf) {
                    post_channel_errband = new TGraphAsymmErrors(&bf_hists[idx_num]);
                }

                // Fill ratio histograms bin by bin
                for(int i = 0; i < channel_nbins; i++) {
                    double cv_num_val = cv_num.GetBinContent(i + 1);
                    double cv_den_val = cv_den.GetBinContent(i + 1);

                    // Bin indices in full collapsed matrix
                    int bin_idx_num = offset_num + i;
                    int bin_idx_den = offset_den + i;

                    double pre_corr_val = pre_corr.GetBinContent(bin_idx_num + 1, bin_idx_den + 1);
                    double post_corr_val = post_corr.GetBinContent(bin_idx_num + 1, bin_idx_den + 1);

                    double cv_rat = (cv_den_val > 0) ? cv_num_val / cv_den_val : 0;
                    cv_ratio->SetBinContent(i + 1, cv_rat);

                    if(errband) {
                        double err_num_up = errband->error_up(bin_idx_num);
                        double err_num_down = errband->error_down(bin_idx_num);
                        double err_den_up = errband->error_up(bin_idx_den);
                        double err_den_down = errband->error_down(bin_idx_den);

                        double pre_err_up = ratio_err(cv_num_val, err_num_up, cv_den_val, err_den_up, pre_corr_val);
                        double pre_err_down = ratio_err(cv_num_val, err_num_down, cv_den_val, err_den_down, pre_corr_val);

                        channel_errband->SetPointY(i, cv_rat);
                        channel_errband->SetPointEYhigh(i, pre_err_up);
                        channel_errband->SetPointEYlow(i, pre_err_down);

                        cv_ratio->SetBinError(i + 1, ratio_err(cv_num_val, cv_num.GetBinError(i + 1), 
                                    cv_den_val, cv_den.GetBinError(i + 1), pre_corr_val));
                    }

                    // Data ratio
                    double data_num_val = data_num.GetBinContent(i + 1);
                    double data_den_val = data_den.GetBinContent(i + 1);
                    double data_rat = (data_den_val > 0) ? data_num_val / data_den_val : 0;
                    data_ratio->SetBinContent(i + 1, data_rat);
                    data_ratio->SetBinError(i + 1, ratio_err(data_num_val, sqrt(data_num_val), 
                                data_den_val, sqrt(data_den_val), 0.0));

                    // Best-fit ratio
                    if(has_bf && bf_ratio) {
                        double bf_num_val = bf_hists[idx_num].GetBinContent(i + 1);
                        double bf_den_val = bf_hists[idx_den].GetBinContent(i + 1);
                        double bf_rat = (bf_den_val > 0) ? bf_num_val / bf_den_val : 0;
                        bf_ratio->SetBinContent(i + 1, bf_rat);

                        if(posterrband && post_channel_errband) {
                            double post_err_num_up = posterrband->error_up(bin_idx_num);
                            double post_err_num_down = posterrband->error_down(bin_idx_num);
                            double post_err_den_up = posterrband->error_up(bin_idx_den);
                            double post_err_den_down = posterrband->error_down(bin_idx_den);

                            double post_err_up = ratio_err(bf_num_val, post_err_num_up, 
                                    bf_den_val, post_err_den_up, post_corr_val);
                            double post_err_down = ratio_err(bf_num_val, post_err_num_down, 
                                    bf_den_val, post_err_den_down, post_corr_val);

                            post_channel_errband->SetPointY(i, bf_rat);
                            post_channel_errband->SetPointEYhigh(i, post_err_up);
                            post_channel_errband->SetPointEYlow(i, post_err_down);

                            bf_ratio->SetBinError(i + 1, ratio_err(bf_num_val, bf_hists[idx_num].GetBinError(i + 1), 
                                        bf_den_val, bf_hists[idx_den].GetBinError(i + 1), 
                                        post_corr_val));

                            // Error improvement ratio
                            if(errband && err_ratio) {
                                double pre_err = ratio_err(cv_num_val, errband->error_up(bin_idx_num), 
                                        cv_den_val, errband->error_up(bin_idx_den), pre_corr_val);
                                if(pre_err > 0) {
                                    err_ratio->SetBinContent(i + 1, post_err_up / pre_err);
                                }
                            }
                        }
                    }
                }

                leg->AddEntry(data_ratio, "Data", "pe");
                leg->AddEntry(cv_ratio, "CV Prediction", "l");
                if(has_bf && bf_ratio) {
                    leg->AddEntry(bf_ratio, "Best-Fit", "l");
                }

                std::string canvas_name = "ratio_" + channel_short_labels[idx_num] + "_over_" + channel_short_labels[idx_den];
                auto ratio_c = new TCanvas(canvas_name.c_str(), ratio_label.c_str());

                TPad p1("p1", "p1", 0, has_bf ? 0.25 : 0.0, 1, 1);
                TPad p2("p2", "p2", 0, 0, 1, 0.25);

                if(has_bf) {
                    p1.SetBottomMargin(0);
                    p2.SetTopMargin(0);
                    p2.SetBottomMargin(0.3);
                }
                p1.cd();

                double max_val = std::max(cv_ratio->GetMaximum(), data_ratio->GetMaximum());
                if(has_bf && bf_ratio) {
                    max_val = std::max(max_val, bf_ratio->GetMaximum());
                }
                cv_ratio->GetYaxis()->SetRangeUser(0.0, 1.33* max_val);
                cv_ratio->SetTitle(ratio_label.c_str());
                cv_ratio->Draw("hist");

                // Draw CV error band
                if(errband) {
                    channel_errband->SetFillStyle(3144);
                    channel_errband->SetFillColorAlpha(cvcol, 0.2);
                    channel_errband->SetLineColor(cvcol);
                    channel_errband->SetLineWidth(1);
                    channel_errband->Draw("2 same");
                }

                // Draw best-fit
                if(has_bf && bf_ratio) {
                    bf_ratio->SetLineColor(kRed);
                    bf_ratio->SetLineWidth(2);
                    bf_ratio->Draw("hist same");

                    if(posterrband && post_channel_errband) {
                        post_channel_errband->SetFillColor(kRed);
                        post_channel_errband->SetFillStyle(3254);
                        post_channel_errband->SetLineColor(kRed);
                        post_channel_errband->SetLineWidth(1);
                        post_channel_errband->Draw("2 same");
                    }
                }

                // Draw data
                data_ratio->Draw("PE1 same");
                leg->SetNColumns(3);
                leg->SetFillStyle(0);
                leg->SetLineWidth(0);
                leg->Draw("same");

                // Draw error ratio panel (bottom)
                if(has_bf && posterrband && err_ratio) {
                    p2.cd();
                    err_ratio->GetYaxis()->SetTitleSize(0.1);
                    err_ratio->GetYaxis()->SetTitleOffset(0.5);
                    err_ratio->GetYaxis()->SetLabelSize(0.1);
                    err_ratio->GetXaxis()->SetTitleSize(0.1);
                    err_ratio->GetXaxis()->SetLabelSize(0.1);
                    err_ratio->Draw("hist");
                }

                // Finalize canvas
                ratio_c->cd(0);
                p1.Draw();
                if(has_bf) {
                    p2.Draw();
                }

                ratio_c->Modified();
                ratio_c->Update();

                // Save with unique filename
                std::string output_filename = "ratio_" + channel_short_labels[idx_num] + "_over_" + 
                    channel_short_labels[idx_den] + "_" + filename;
                ratio_c->Print(output_filename.c_str());

                log<LOG_INFO>(L"%1% || Created ratio plot: %2%") % __func__ % output_filename.c_str();

                //delete ratio_c;
            }
        }

        log<LOG_DEBUG>(L"%1% || Finished plot_detector_ratios") % __func__;
    }                                                                              

    void plot_hist1ds(TCanvas* c, TH1D* cv_hist, TGraphAsymmErrors* errband, THStack* cvstack, std::vector<std::pair<std::string, const char*>>* subplots, TH1D* bf_hist, TGraphAsymmErrors* posterrband, TH1D* data_hist, std::string* dat_str, PlotOptions opt, std::string hist_titles, std::string ratio_titles, const std::string &filename, PlotBounds &bounds, TText* text){

        log<LOG_DEBUG>(L"%1% || Plotting 1D Histogram %2%") % __func__ % hist_titles.c_str();
        std::unique_ptr<TLegend> leg = std::make_unique<TLegend>(0.38,0.74,0.89,0.91);
        leg->SetNColumns(2);
        leg->SetFillStyle(0);
        leg->SetLineWidth(0);



        Color_t cvcol =  TColor::GetColor(66, 103, 210);
        if(!bf_hist)cvcol=kBlack;
        Color_t bfcol = TColor::GetColor(234, 67, 53);

        TPad* p1 = NULL;
        if(bool(opt&PlotOptions::DataMCRatio) || bool(opt&PlotOptions::DataPostfitRatio)){
            p1 = new TPad("p1", "p1", 0, 0.25, 1, 1);
            p1->cd();
            p1->SetBottomMargin(0);
        }
        else{
            p1 = new TPad("p1", "p1", 0, 0, 1, 1);
            p1->cd();
        }

        TPad* p2 = new TPad("p2", "p2", 0, 0, 1, 0.25);
        p2->SetTopMargin(0);
        p2->SetBottomMargin(0.3);

        double top_modifier = 1.35;
        double y_max = 0.0;

        if(cv_hist) {
            y_max = std::max(y_max, cv_hist->GetMaximum());
        }
        if(bf_hist) {
            y_max = std::max(y_max, bf_hist->GetMaximum());
        }
        if(data_hist) {
            y_max = std::max(y_max, data_hist->GetMaximum());
        }
        if(bool(opt&PlotOptions::CVasStack) && cvstack) {
            y_max = std::max(y_max, cvstack->GetMaximum());
        }

        y_max *= top_modifier;
        if(bounds.hasBound("ymax")) {
             y_max = bounds.getBound("ymax");
        }
        
        if(cv_hist) {

            cv_hist->SetLineColor(cvcol);
            cv_hist->SetLineStyle(kDashed);

            if(bool(opt&PlotOptions::CVasStack)) {
                log<LOG_DEBUG>(L"%1% || Using CVStack %2%") % __func__ % hist_titles.c_str();

                cvstack->Draw("hist");
                cvstack->SetTitle(hist_titles.c_str());
                cv_hist->Draw("same hist");

                cvstack->SetMaximum(y_max);

                TList* hist_list = cvstack->GetHists();
                TIter next(hist_list); 
                TObject* obj;
                size_t sc = 0; 
                while ((obj = next())) {

                    TH1D* hist = dynamic_cast<TH1D*>(obj);
                    if (hist) {
                        leg->AddEntry(hist, subplots->at(sc).second, "f");
                    }
                    sc = sc + 1;
                }


            }
            else{
                cv_hist->SetMaximum(y_max);
                cv_hist->SetMinimum(0.01);
                cv_hist->Draw("hist");
            }

            TH1 *leg_hack = (TH1*)cv_hist->Clone((std::string(cv_hist->GetTitle())+"leg_hack").c_str());
            if(errband){
                log<LOG_DEBUG>(L"%1% || Using errband %2%") % __func__ % hist_titles.c_str();
                leg_hack->SetFillStyle(3144);
                leg_hack->SetFillColorAlpha(cvcol, 0.2);
                leg_hack->SetLineColor(cvcol);
                leg_hack->SetLineStyle(kDashed);
                leg_hack->SetLineWidth(2);
                leg->AddEntry(leg_hack,"CV Prediction #pm 1#sigma" ,"fl");
            }else{
                leg->AddEntry(leg_hack, "CV Prediction", "l");
            }
        }

        if(bf_hist) {
            bf_hist->SetLineStyle(kSolid);
            bf_hist->SetLineWidth(2);
            bf_hist->SetLineColor(bfcol);

            if(cv_hist) bf_hist->Draw("hist same");
            else bf_hist->Draw("hist");

            TH1 *leg_hack = (TH1*)bf_hist->Clone((std::string(bf_hist->GetTitle())+"bf").c_str());
            if(posterrband){
                leg_hack->SetFillStyle(3254);
                leg_hack->SetFillColor(bfcol);
                leg_hack->SetLineColor(bfcol);
                leg_hack->SetLineWidth(2);
                leg->AddEntry(leg_hack,"Best Fit #pm 1#sigma (post-fit)" ,"fl");
            }else{
                leg->AddEntry(leg_hack, "Best Fit #pm 1#sigma (post-fit)", "l");
            }
        }

        if(data_hist) {
            log<LOG_DEBUG>(L"%1% || Using data %2%") % __func__ % hist_titles.c_str();
            TGraphErrors *g = new TGraphErrors(data_hist->GetNbinsX());
            for (int i = 1; i <= data_hist->GetNbinsX(); ++i) {
                double x = data_hist->GetBinCenter(i);
                double y = data_hist->GetBinContent(i);
                double ex = 0;
                double ey = data_hist->GetBinError(i);
                // g->SetPoint(i - 1, x, y);
                g->SetPoint(i, x, y);
                // g->SetPointError(i - 1, ex, ey);
                g->SetPointError(i, ex, ey);
                g->SetLineColor(kBlack);
                g->SetLineWidth(2);
                g->SetMarkerStyle(kFullCircle);
                g->SetMarkerColor(kBlack);
                g->SetMarkerSize(1);
            }

            g->Draw("PE1 same");
            TH1 *leg_hack = (TH1*)data_hist->Clone((std::string(data_hist->GetTitle())).c_str());
            leg->AddEntry(leg_hack, dat_str->c_str(), "lp");
        }

        if(errband) {
            errband->SetFillStyle(3144);
            errband->SetFillColorAlpha(cvcol, 0.2);
            errband->SetLineColor(cvcol);
            errband->SetLineWidth(1);
            errband->Draw("2 same");
        }

        if(posterrband) {
            posterrband->SetFillStyle(3254);
            posterrband->SetFillColorAlpha(bfcol, 1.0);
            posterrband->SetLineColor(bfcol);
            posterrband->SetLineWidth(1);
            posterrband->Draw("2 same");
        }

        if(text) {
            TLine *dummy_line = new TLine(0,0,0.1,0);
            dummy_line->SetLineColor(kWhite);
            dummy_line->SetLineWidth(0);
            const char* label = text->GetTitle();
            leg->AddEntry(dummy_line,label,"l");
        }


        TH1D* ratio = (TH1D*)cv_hist->Clone("rat");
        ratio->Reset();
        ratio->SetTitle(ratio_titles.c_str());

        TH1D* one = (TH1D*)cv_hist->Clone("one");
        one->Reset();
        one->SetTitle(ratio_titles.c_str());

        TGraphAsymmErrors *ratio_err;
        if(bool(opt&PlotOptions::DataMCRatio) || bool(opt&PlotOptions::DataPostfitRatio)) {
            log<LOG_DEBUG>(L"%1% || Using ratio %2%") % __func__ % hist_titles.c_str();
            p2->cd();

            ratio_err = new TGraphAsymmErrors(); 

            if (posterrband){
                if(bool(opt&PlotOptions::DataMCRatio)){
                    *ratio_err = *errband;  
                }
                else{
                    *ratio_err = *posterrband;  
                }
            }
            else if(errband){
                *ratio_err = *errband;  
            }

            float ymin = 1e9, ymax = -1e9;

            if(data_hist){
                for(int i = 0; i < data_hist->GetNbinsX(); ++i) {
                    float numerator = data_hist->GetBinContent(i+1);
                    float denominator = 
                        bool(opt&PlotOptions::DataMCRatio)
                        ? cv_hist->GetBinContent(i+1)
                        : bf_hist->GetBinContent(i+1);

                    float rat = numerator/denominator;
                    if(isnan(rat) || isinf(rat) || denominator == 0) rat = 1;
                    float rat_err = (denominator != 0) ? data_hist->GetBinError(i+1)/denominator : 0;
                    if(isnan(rat_err) || isinf(rat_err)) rat_err = 0;
                    ratio->SetBinError(i+1, rat_err);
                    ratio->SetBinContent(i+1, rat);
                    one->SetBinContent(i+1, 1.0);

                    // Avoid division by zero when normalizing error band
                    float point_y = ratio_err->GetPointY(i);
                    if(point_y != 0 && !isnan(point_y) && !isinf(point_y)) {
                        ratio_err->SetPointEYhigh(i, ratio_err->GetErrorYhigh(i)/point_y);
                        ratio_err->SetPointEYlow(i, ratio_err->GetErrorYlow(i)/point_y);
                    } else {
                        ratio_err->SetPointEYhigh(i, 0);
                        ratio_err->SetPointEYlow(i, 0);
                    }
                    ratio_err->SetPointY(i, 1.0);
                    if(!isnan(rat) && !isinf(rat)) {
                        ymin = std::min(ymin, rat);
                        ymax = std::max(ymax, rat);
                    }
                }

                for (int i = 0; i < ratio_err->GetN(); ++i) {
                    float y, eyh, eyl;
                    y = ratio_err->GetPointY(i);
                    eyh = ratio_err->GetErrorYhigh(i);
                    eyl = ratio_err->GetErrorYlow(i);
                    if(!isnan(y) && !isinf(y) && !isnan(eyh) && !isinf(eyh) && !isnan(eyl) && !isinf(eyl)) {
                        ymin = std::min(ymin, y - eyl);
                        ymax = std::max(ymax, y + eyh);
                    }
                }
            }
            
            // Ensure valid axis range - use defaults if no valid data
            if(ymin >= ymax || ymin > 1e8 || ymax < -1e8) {
                ymin = 0.5f;
                ymax = 1.5f;
            }
            
            float yrange = ymax - ymin;
            float ylow = ymin - 0.05 * yrange;  // 5% padding below
            float yhigh = ymax + 0.05 * yrange; // 5% padding above

            float kmin = bounds.hasBound("ratmin") ? bounds.getBound("ratmin") : std::min(ylow,0.95f);
            float kmax = bounds.hasBound("ratmax") ? bounds.getBound("ratmax") : std::max(yhigh,1.05f);

            // Final safety check to ensure kmin != kmax
            if(kmin >= kmax) {
                kmin = 0.5f;
                kmax = 1.5f;
            }

            one->SetMinimum(kmin);
            one->SetMaximum(kmax);

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

            ratio_err->Draw("2 same");
            ratio->Draw("PE1 E0 same");
        }

        c->cd();
        p1->Draw();
        if(bool(opt&PlotOptions::DataMCRatio) || bool(opt&PlotOptions::DataPostfitRatio)) p2->Draw();

        TText *t = new TText();
        t->SetNDC();
        t->SetTextFont(42);
        t->SetTextSize(0.03); 
        t->SetTextAlign(33); 
        std::string pv = "PROfit v"+std::string(PROJECT_VERSION_STR);
        t->DrawText(0.895, 0.955, pv.c_str()); 

        leg->Draw("same");
        c->Print(filename.c_str());
        log<LOG_DEBUG>(L"%1% || Finishing Plotting 1D Histogram %2%") % __func__ % hist_titles.c_str();
    }

    void plot_channels(const std::string &filename, const PROconfig &config, std::optional<PROspec> cv, std::optional<PROspec> best_fit, std::optional<PROdata> data, std::optional<PROerrorbar> errband, std::optional<PROerrorbar> posterrband, std::optional<PROsyst> pre_allcovsyst, std::optional<PROsyst> post_allcovsyst, std::vector<TPaveText> &texts, PlotBounds &bounds, PlotOptions opt, int other_index) {

        log<LOG_DEBUG>(L"%1% || Starting plot_channels") % __func__;
        std::string rat_y_title = bool(opt&PlotOptions::DataMCRatio) ? "Data/MC" : "Data/Best-Fit";

        std::string ytitle = bool(opt&PlotOptions::AreaNormalized)
            ? "Area Normalized"
            : bool(opt&PlotOptions::BinWidthScaled) 
            ? "Events/GeV" 
            : "Events";

        TCanvas c;
        c.Print((filename+"[").c_str());

        std::map<std::string, std::unique_ptr<TH1D>> cv1dhists;
        std::map<std::string, std::unique_ptr<TH2D>> cv2dhists;

        // Keep as we go through loop for ratio plot
        std::vector<TH1D> data_hists;
        std::vector<TH1D> cv_hists;
        std::vector<TH1D> bf_hists;

        size_t global_subchannel_index = 0;
        size_t global_subchannel_index_2d = 0;
        size_t global_channel_index = 0;
        int tot_offset = 0;

        for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
            for(size_t det = 0; det < config.m_num_detectors; ++det) {
                // data string for legend is only det specific
                std::string dat_str;
                if(data){
                    dat_str = "Data: ";
                    std::ostringstream oss;
                    int exponent = static_cast<int>(std::log10(std::abs(config.m_det_pot[det])));
                    float mantissa = config.m_det_pot[det]/ std::pow(10, exponent);
                    oss << std::fixed << std::setprecision(2) << mantissa << "x10^{" << exponent << "} POT";
                    dat_str+= oss.str();
                }
                for(size_t channel = 0; channel < config.m_num_channels; ++channel) {

                    std::string xtitle = config.m_channel_variable_units[channel][other_index];
                    std::string ratio_titles = ";"+xtitle+";"+rat_y_title;

                    size_t channel_nbins_x = config.m_channel_variable_bins[channel][other_index].NBinsAlong(0);

                    // default is 1d, but catch 2d case for ybins
                    size_t channel_nbins_y = 1;
                    if(config.m_channel_variable_dims[channel][other_index] == 2)  channel_nbins_y = config.m_channel_variable_bins[channel][other_index].NBinsAlong(1);

                    int nbins_p_2dchan = channel_nbins_y*channel_nbins_x;

                    if(cv){
                        if(config.m_channel_variable_dims[channel][other_index] == 2){
                            cv2dhists = getCV2DHists(*cv, config, (bool)(opt & PlotOptions::BinWidthScaled), other_index);
                        }
                        cv1dhists = getCV1DHists(*cv, config, (bool)(opt & PlotOptions::BinWidthScaled), other_index);
                    }

                    Eigen::VectorXf bf_spec;
                    Eigen::VectorXf bf_spec_1d = Eigen::VectorXf::Zero(channel_nbins_x);
                    if(best_fit){ 
                        bf_spec = CollapseMatrix(config, best_fit->Spec(), other_index);
                        bf_spec_1d = make_1d_spec(bf_spec, channel_nbins_x, channel_nbins_y, tot_offset, config.m_channel_variable_dims[channel][other_index]);
                    }

                    Eigen::VectorXf data_spec_1d = Eigen::VectorXf::Zero(channel_nbins_x);
                    if(data){
                        data_spec_1d = make_1d_spec(data->Spec(), channel_nbins_x, channel_nbins_y, tot_offset, config.m_channel_variable_dims[channel][other_index]);
                    }

                    PROerrorbar *errband_1d = NULL;
                    if(errband){
                        errband_1d = make_1d_err(*errband, channel_nbins_x, channel_nbins_y, tot_offset, config.m_channel_variable_dims[channel][other_index]);
                        log<LOG_DEBUG>(L"%1% || err") % __func__;
                    }

                    PROerrorbar *posterrband_1d = NULL;
                    if(posterrband){
                        posterrband_1d = make_1d_err(*posterrband, channel_nbins_x, channel_nbins_y, tot_offset, config.m_channel_variable_dims[channel][other_index]);
                        log<LOG_DEBUG>(L"%1% || posterr") % __func__;
                    }

                    if(config.m_channel_variable_dims[channel][other_index] == 2){
                        gStyle->SetPalette(kViridis);
                        std::string joined_title = config.m_channel_variable_units[channel][other_index];
                        string del = ";";
                        auto pos = joined_title.find(del);
                        std::string xtitle2d = joined_title.substr(0, pos);
                        xtitle = joined_title.substr(0, pos);
                        std::string ytitle2d = joined_title.erase(0, pos + del.length());
                        ratio_titles = ";"+xtitle2d+";"+rat_y_title;
                        std::string hist_title = config.m_detector_plotnames[det]  + " "+ config.m_channel_names[channel]+" CV;"+xtitle2d+";"+ytitle2d;

                        std::vector<float> edges_x = config.m_channel_variable_bins[channel][other_index].Edges(0);
                        std::vector<float> edges_y = config.m_channel_variable_bins[channel][other_index].Edges(1);

                        // this needs to be a sum over all subchannels
                        //
                        TH2D* cv_hist = new TH2D(hist_title.c_str(),hist_title.c_str(), channel_nbins_x, edges_x.data(), channel_nbins_y, edges_y.data());
                        for(size_t subchannel = 0; subchannel < config.m_num_subchannels[channel]; ++subchannel){
                            const std::string& subchannel_name  = config.m_fullnames[global_subchannel_index_2d];
                            cv_hist->Add(cv2dhists[subchannel_name].get());
                            ++global_subchannel_index_2d;
                        }

                        TPad p2d("p2d", "p2d", 0, 0, 1, 1);
                        p2d.cd();
                        cv_hist->SetTitle(hist_title.c_str());
                        cv_hist->Draw("colz");
                        c.cd();
                        p2d.Draw();
                        c.Print(filename.c_str());

                        TH2D* bf_hist = NULL;
                        if(best_fit){
                            std::string bf_hist_title = config.m_detector_plotnames[det]  + " "+ config.m_channel_plotnames[channel]+" Best-Fit;"+xtitle2d+";"+ytitle2d;
                            bf_hist = new TH2D(bf_hist_title.c_str(),bf_hist_title.c_str(), channel_nbins_x, edges_x.data(), channel_nbins_y, edges_y.data());

                            Eigen::VectorXf tmp_bf = CollapseMatrix(config, best_fit->Spec(), other_index);
                            for(size_t xbin = 0; xbin < channel_nbins_x; xbin++){
                                for(size_t ybin = 0; ybin < channel_nbins_y; ybin++) {
                                    bf_hist->SetBinContent(xbin+1, ybin+1, tmp_bf(xbin*channel_nbins_y+ybin + tot_offset));
                                }
                            }

                            TPad pbfd("pbfd", "pbfd", 0, 0, 1, 1);
                            pbfd.cd();
                            bf_hist->Draw("colz");
                            c.cd();
                            pbfd.Draw();
                            c.Print(filename.c_str());
                        }

                        TH2D* data_hist = NULL;
                        if(data){
                            std::string data_hist_title = config.m_detector_plotnames[det]  + " "+ config.m_channel_plotnames[channel]+" Data;"+xtitle2d+";"+ytitle2d;
                            data_hist = new TH2D(data_hist_title.c_str(),data_hist_title.c_str(), channel_nbins_x, edges_x.data(), channel_nbins_y, edges_y.data());

                            Eigen::VectorXf tmp_data = data->Spec();
                            for(size_t xbin = 0; xbin < channel_nbins_x; xbin++){
                                for(size_t ybin = 0; ybin < channel_nbins_y; ybin++) {
                                    data_hist->SetBinContent(xbin+1, ybin+1, tmp_data(xbin*channel_nbins_y+ybin+tot_offset));
                                }
                            }
                            TPad pdata("pdata", "pdata", 0, 0, 1, 1);
                            pdata.cd();
                            data_hist->Draw("colz");
                            c.cd();
                            pdata.Draw();
                            c.Print(filename.c_str());
                        }


                        float scale = 1.0;

                        for(size_t ybin = 1; ybin <= channel_nbins_y; ybin++) {
                            TGraphAsymmErrors *post_channel_errband = NULL; 
                            if(posterrband) {
                                post_channel_errband = new TGraphAsymmErrors(bf_hist->ProjectionX("slc", ybin, ybin));
                                for(size_t xbin = 0; xbin < channel_nbins_x; ++xbin) {
                                    post_channel_errband->SetPointEYhigh(xbin, scale*(posterrband->error_up((ybin-1)*channel_nbins_x+xbin+tot_offset)));
                                    post_channel_errband->SetPointEYlow(xbin, scale*(posterrband->error_down((ybin-1)*channel_nbins_x+xbin+tot_offset)));
                                }
                            }

                            TGraphAsymmErrors* channel_errband = NULL;
                            if(errband) {
                                channel_errband = new TGraphAsymmErrors(cv_hist->ProjectionX("slc", ybin, ybin));
                                for(size_t xbin = 0; xbin < channel_nbins_x; ++xbin) {
                                    channel_errband->SetPointEYhigh(xbin, scale*(errband->error_up((ybin-1)*channel_nbins_x+xbin+tot_offset)));
                                    channel_errband->SetPointEYlow(xbin, scale*(errband->error_down((ybin-1)*channel_nbins_x+xbin+tot_offset)));
                                }
                            }

                            std::string ybin_str = "Slice "+std::to_string(ybin)+" "+hist_title;

                            TH1D cv_hist_slice = *(cv_hist->ProjectionX("slc", ybin, ybin));
                            cv_hist_slice.SetTitle(ybin_str.c_str());
                            cv_hist_slice.GetYaxis()->SetTitle(ytitle.c_str());
                            TH1D *bf_hist_slice = NULL;
                            if(best_fit){
                                bf_hist_slice = bf_hist->ProjectionX("bfslc", ybin, ybin);
                                bf_hist_slice->SetTitle(ybin_str.c_str());
                                bf_hist_slice->GetYaxis()->SetTitle(ytitle.c_str());
                            }
                            TH1D* data_hist_slice = NULL;
                            if(data){
                                data_hist_slice = data_hist->ProjectionX("dataslc", ybin, ybin);
                                data_hist_slice->SetTitle(dat_str.c_str());
                                data_hist_slice->GetYaxis()->SetTitle(ytitle.c_str());
                                data_hist_slice->SetLineColor(kBlack);
                                data_hist_slice->SetLineWidth(2);
                                data_hist_slice->SetMarkerStyle(kFullCircle);
                                data_hist_slice->SetMarkerColor(kBlack);
                                data_hist_slice->SetMarkerSize(1);
                            }

                            plot_hist1ds(&c, &cv_hist_slice, channel_errband, {}, {}, bf_hist_slice, post_channel_errband, data_hist_slice, &dat_str, {}, ybin_str, ratio_titles, filename, bounds, {});
                        }
                    }

                    std::vector<float> edges = config.m_channel_variable_bins[channel][other_index].Edges();

                    std::string hist_titles = config.m_mode_plotnames[mode]+" "+config.m_detector_plotnames[det]  + " "+ config.m_channel_names[channel]+";"+xtitle+";"+ytitle;
                    TH1D cv_hist(("cv_hist"+std::to_string(global_channel_index)).c_str(), hist_titles.c_str(), channel_nbins_x, edges.data());
                    cv_hist.SetLineWidth(2);

                    cv_hist.SetFillStyle(0);
                    for(size_t bin = 0; bin < channel_nbins_x; ++bin) {
                        cv_hist.SetBinContent(bin+1, 0);
                    }
                    if(bool(opt&PlotOptions::BinWidthScaled))
                        cv_hist.Scale(1, "width");

                    THStack *cvstack = NULL;
                    std::vector<std::pair<std::string, const char*>> subplots;
                    if(cv) {
                        if(bool(opt&PlotOptions::CVasStack)) cvstack = new THStack(std::to_string(global_channel_index).c_str(), "");

                        for(size_t subchannel = 0; subchannel < config.m_num_subchannels[channel]; ++subchannel){
                            const std::string& subchannel_name  = config.m_fullnames[global_subchannel_index];
                            if(bool(opt&PlotOptions::CVasStack)) {
                                cvstack->Add(cv1dhists[subchannel_name].get());
                                subplots.push_back({subchannel_name, config.m_subchannel_plotnames[channel][subchannel].c_str()});
                            }
                            cv_hist.Add(cv1dhists[subchannel_name].get());
                            ++global_subchannel_index;
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
                    }

                    TGraphAsymmErrors *channel_errband = NULL;
                    if(errband) {
                        channel_errband = new TGraphAsymmErrors(&cv_hist);

                        for(size_t bin = 0; bin < channel_nbins_x; ++bin) {
                            float scale = 1.0;
                            if(bool(opt&PlotOptions::AreaNormalized) || bool(opt&PlotOptions::BinWidthScaled)) {
                                scale = channel_errband->GetPointY(bin) / errband_1d->error_point(bin);
                            }

                            channel_errband->SetPointEYhigh(bin, scale*(errband_1d->error_up(bin)));
                            channel_errband->SetPointEYlow(bin, scale*(errband_1d->error_down(bin)));
                        }
                    }

                    TH1D* bf_hist = NULL;
                    if(best_fit) {
                        bf_hist = new TH1D(("bf"+std::to_string(global_channel_index)).c_str(), "", channel_nbins_x, edges.data());
                        for(size_t bin = 0; bin < channel_nbins_x; ++bin) {
                            bf_hist->SetBinContent(bin+1, bf_spec_1d(bin));
                        }
                        if(bool(opt&PlotOptions::BinWidthScaled))
                            bf_hist->Scale(1, "width");
                        if(bool(opt&PlotOptions::AreaNormalized))
                            bf_hist->Scale(1.0/bf_hist->Integral());
                    }

                    TH1D* data_hist = NULL;
                    if(data) {
                        data_hist = new TH1D(("data"+std::to_string(global_channel_index)).c_str(), "", channel_nbins_x, edges.data());
                        for(size_t bin = 0; bin < channel_nbins_x; ++bin) {
                            data_hist->SetBinContent(bin+1, data_spec_1d(bin));
                        }
                        data_hist->SetLineColor(kBlack);
                        data_hist->SetLineWidth(2);
                        data_hist->SetMarkerStyle(kFullCircle);
                        data_hist->SetMarkerColor(kBlack);
                        data_hist->SetMarkerSize(1);
                        if(bool(opt&PlotOptions::BinWidthScaled))
                            data_hist->Scale(1, "width");
                        if(bool(opt&PlotOptions::AreaNormalized))
                            data_hist->Scale(1.0/data_hist->Integral());
                    }

                    TGraphAsymmErrors *post_channel_errband = NULL;
                    if(posterrband) {
                        post_channel_errband = new TGraphAsymmErrors(bf_hist);
                        for(size_t bin = 0; bin < channel_nbins_x; ++bin) {
                            float scale = 1.0;
                            if(bool(opt&PlotOptions::AreaNormalized) || bool(opt&PlotOptions::BinWidthScaled)) {
                                scale = post_channel_errband->GetPointY(bin) / (posterrband_1d->error_point(bin));
                            }
                            post_channel_errband->SetPointEYhigh(bin, scale*(posterrband_1d->error_up(bin)));
                            post_channel_errband->SetPointEYlow(bin, scale*(posterrband_1d->error_down(bin)));
                        }
                    }

                    TGraphAsymmErrors *ratio_err;

                    TText* text = NULL;
                    if(texts.size()!=0) {
                        if(texts.size()==1){
                            text = (TText*)texts.front().GetListOfLines()->First();
                        }else{
                            text = (TText*)texts.at(global_channel_index).GetListOfLines()->First();
                        }
                    }
                    // should probably be switching this to a more clear boolean...
                    plot_hist1ds(&c, &cv_hist, channel_errband, cvstack, &subplots, bf_hist, post_channel_errband, data_hist, &dat_str, opt, hist_titles, ratio_titles, filename, bounds, text);
                    ++global_channel_index;
                    if(pre_allcovsyst.has_value()){
                        cv_hists.push_back(cv_hist);
                        data_hists.push_back(*data_hist);
                        bf_hists.push_back(*bf_hist);

                    }
                    tot_offset += nbins_p_2dchan;
                }
            }
        }
        c.Print((filename+"]").c_str());

        if(pre_allcovsyst.has_value() && data_hists.size() >= 2){
            Eigen::MatrixXf fractional_cov = pre_allcovsyst->fractional_covariance;
            Eigen::MatrixXf diag = cv->Spec().array().matrix().asDiagonal();

            if(diag.rows() == fractional_cov.rows()){
                std::map<std::string, std::unique_ptr<TH2D>> pre_matrices = covarianceTH2D(pre_allcovsyst.value(), config, *cv);
                std::map<std::string, std::unique_ptr<TH2D>> post_matrices = covarianceTH2D(post_allcovsyst.value(), config, *best_fit);

                plot_detector_ratios(config, data_hists, cv_hists, errband, bf_hists, posterrband, *pre_matrices["collapsed_total_cor"], *post_matrices["collapsed_total_cor"],  filename, other_index);
            }
        }


        log<LOG_DEBUG>(L"%1% || Finishing plot_channels") % __func__;
    }


    int plotFractionalSystematicBreakdown(const PROconfig &config, const PROspec &spec, const PROsyst &allsplinesyst, std::string filename, int other_index, std::map<std::string, double> alphas) {
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

        TCanvas c("c", "Systematics Comparison", 1600, 900);
        c.Print((filename+"[").c_str());
        c.Divide(gridCols, gridRows);

        Eigen::MatrixXf diag = spec.Spec().array().matrix().asDiagonal();
        Eigen::MatrixXf collapsed_diag = CollapseMatrix(config, diag);

        // Diagnostic: Check collapsed_diag for issues
        bool has_nan = collapsed_diag.array().isNaN().any();
        bool has_inf = collapsed_diag.array().isInf().any();
        float min_diag = collapsed_diag.diagonal().minCoeff();
        float max_diag = collapsed_diag.diagonal().maxCoeff();
        log<LOG_INFO>(L"%1% || collapsed_diag diagnostics: size=%2%x%3%, has_nan=%4%, has_inf=%5%, diag_min=%6%, diag_max=%7%")
            % __func__ % collapsed_diag.rows() % collapsed_diag.cols() % has_nan % has_inf % min_diag % max_diag;

        // Check if matrix is actually diagonal (off-diagonal elements should be ~0)
        Eigen::MatrixXf off_diag = collapsed_diag;
        off_diag.diagonal().setZero();
        float max_off_diag = off_diag.array().abs().maxCoeff();
        log<LOG_INFO>(L"%1% || collapsed_diag max off-diagonal element: %2% (should be ~0 if truly diagonal)")
            % __func__ % max_off_diag;

        // Check determinant / condition for invertibility
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(collapsed_diag);
        float cond_number = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
        float min_singular = svd.singularValues().minCoeff();
        log<LOG_INFO>(L"%1% || collapsed_diag condition number: %2%, min singular value: %3%")
            % __func__ % cond_number % min_singular;

        if (min_singular < 1e-10) {
            log<LOG_ERROR>(L"%1% || WARNING: collapsed_diag is nearly singular (min singular value = %2%). Matrix inverse will be unreliable!")
                % __func__ % min_singular;
        }

        // adding 1e-6 to diagonal entries that are zero, to make the matrix invertible when there are zero-uncertainty bins
        // logging a warning in each of these cases.
        for (size_t i = 0; i < collapsed_diag.rows(); ++i) {
            if (collapsed_diag(i,i) == 0) {
                log<LOG_WARNING>(L"%1% || WARNING: collapsed_diag(i,i) is zero for bin %2%! Adding 1e-6 to make the matrix invertible.")
                    % __func__ % i;
                collapsed_diag(i,i) = 1e-6;
            }
        }

        // Compute inverse and check it
        Eigen::MatrixXf collapsed_diag_inv = collapsed_diag.inverse();
        bool inv_has_nan = collapsed_diag_inv.array().isNaN().any();
        bool inv_has_inf = collapsed_diag_inv.array().isInf().any();
        float inv_max = collapsed_diag_inv.array().abs().maxCoeff();
        log<LOG_INFO>(L"%1% || collapsed_diag inverse diagnostics: has_nan=%2%, has_inf=%3%, max_abs=%4%")
            % __func__ % inv_has_nan % inv_has_inf % inv_max;

        if (inv_has_nan || inv_has_inf) {
            log<LOG_ERROR>(L"%1% || FATAL: collapsed_diag.inverse() contains NaN or Inf! This will cause empty plots.") % __func__;
        }


        size_t global_channel_index = 0;
        for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
            for(size_t det = 0; det < config.m_num_detectors; ++det) {
                for(size_t channel = 0; channel < config.m_num_channels; ++channel) {
                    std::string xtitle;
                    if(config.m_channel_variable_dims[channel][other_index] == 2){
                    std::string joined_title = config.m_channel_variable_units[channel][other_index];
                    string del = ";";
                    auto pos = joined_title.find(del);
                    std::string xtitle2d = joined_title.substr(0, pos);
                    xtitle = joined_title.substr(0, pos);}
                            else{
                    xtitle = config.m_channel_variable_units[channel][other_index];
                    }

                    std::string name = config.m_mode_plotnames[mode]+" "+config.m_detector_plotnames[det]+" "+config.m_channel_plotnames[channel]; 
                    c.Clear();
                    c.Divide(gridCols, gridRows);

                    int padIndex = 1;

                    std::vector<float> bin_edges = config.GetChannelVariableBins(global_channel_index,other_index).Edges();
                    size_t binstart = config.GetCollapsedGlobalVariableBinStart(global_channel_index,other_index);
                    size_t nbins = config.m_channel_variable_bins[channel][other_index].NBins();
                    std::vector<int> channel_bins(nbins);
                    std::iota(channel_bins.begin(), channel_bins.end(), binstart);

                    // Diagnostic: Check bin_edges validity
                    log<LOG_INFO>(L"%1% || Channel %2%: bin_edges.size()=%3%, nbins=%4%, binstart=%5%")
                        % __func__ % global_channel_index % bin_edges.size() % nbins % binstart;
                    if (!bin_edges.empty()) {
                        log<LOG_INFO>(L"%1% || Channel %2%: bin_edges range [%3%, %4%]")
                            % __func__ % global_channel_index % bin_edges.front() % bin_edges.back();
                    }
                    if (bin_edges.empty()) {
                        log<LOG_ERROR>(L"%1% || FATAL: bin_edges is empty for channel %2%! Histogram will be invalid.")
                            % __func__ % global_channel_index;
                    }
                    if (bin_edges.size() != nbins + 1) {
                        log<LOG_ERROR>(L"%1% || WARNING: bin_edges.size() (%2%) != nbins+1 (%3%) for channel %4%")
                            % __func__ % bin_edges.size() % (nbins + 1) % global_channel_index;
                    }
                    log<LOG_INFO>(L"%1% || Channel %2%: gridCols=%3%, gridRows=%4%, nTags=%5%")
                        % __func__ % global_channel_index % gridCols % gridRows % nTags;

                    std::vector<TH1F*> vsums;
                    std::vector<std::string> vnames;
                    for (const auto &[tag, vec] : used_tags) {

                        c.cd(padIndex++);
                        if (!gPad) {
                            log<LOG_ERROR>(L"%1% || FATAL: gPad is null after c.cd(%2%)! Canvas subdivision failed.")
                                % __func__ % (padIndex-1);
                        } else {
                            log<LOG_INFO>(L"%1% || Pad %2%: gPad=%3%, name=%4%")
                                % __func__ % (padIndex-1) % (void*)gPad % gPad->GetName();
                        }

                        TLegend* leg = new TLegend(0.11, 0.6, 0.89, 0.89);
                        leg->SetNColumns(3);


                        TH1F* hsum = new TH1F( ("Sum_"+tag+"_"+std::to_string(global_channel_index)).c_str(), tag.c_str(), bin_edges.size()-1, bin_edges.data());
                        hsum->Reset();
                        std::vector<TH1F*> hvec;
                        int i = 0;
                        size_t channel_nbins_y = 1;// start with assumption of 1d
                        size_t channel_nbins_x = config.m_channel_variable_bins[channel][other_index].NBinsAlong(0);

                        for(const auto & systname:vec){

                            Eigen::MatrixXf frac_covariance = allsplinesyst.GrabMatrix(systname);

                            Eigen::MatrixXf full_covariance = diag*(frac_covariance)*diag;
                            Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance);
                            Eigen::MatrixXf collapsed_frac_covariance = collapsed_diag_inv*collapsed_full_covariance*collapsed_diag_inv;
                            Eigen::MatrixXf channel_cov = collapsed_full_covariance(channel_bins, channel_bins);

                            // Diagnostic: Check channel_cov diagonal for bad values
                            bool cov_has_nan = channel_cov.diagonal().array().isNaN().any();
                            bool cov_has_inf = channel_cov.diagonal().array().isInf().any();
                            bool cov_has_neg = (channel_cov.diagonal().array() < 0).any();
                            float cov_diag_min = channel_cov.diagonal().minCoeff();
                            float cov_diag_max = channel_cov.diagonal().maxCoeff();
                            if (cov_has_nan || cov_has_inf || cov_has_neg) {
                                log<LOG_ERROR>(L"%1% || BAD channel_cov for %2%: has_nan=%3%, has_inf=%4%, has_neg=%5%, min=%6%, max=%7%")
                                    % __func__ % systname.c_str() % cov_has_nan % cov_has_inf % cov_has_neg % cov_diag_min % cov_diag_max;
                            }

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

                            if(config.m_channel_variable_dims[channel][other_index] == 2)  channel_nbins_y = config.m_channel_variable_bins[channel][other_index].NBinsAlong(1);

                            Eigen::VectorXf VarVec = Eigen::VectorXf::Zero(channel_nbins_x);
                            Eigen::VectorXf diag1d = Eigen::VectorXf::Zero(channel_nbins_x);
                            Eigen::MatrixXf channel_diag = collapsed_diag(channel_bins, channel_bins);

                            if(allsplinesyst.GetLinearResponse().size() < alphas.size()){
                                log<LOG_ERROR>(L"%1% || Number of linear response fit parameters %2% greater than num available linear response functions %3%!") % __func__ % alphas.size() % allsplinesyst.GetLinearResponse().size();
                                return 1;
                            }

                            if(allsplinesyst.GetLinearResponse().count(systname) >= 1 && alphas.count(systname) >= 1){
                                log<LOG_INFO>(L"%1% || found linear response for %2%") % __func__ % systname.c_str();
                                double alpha = alphas[systname];
                                Eigen::MatrixXf content = allsplinesyst.GetLinearResponse()[systname].transpose() * alpha;
                                
                                for (size_t i = 0; i < channel_nbins_x; ++i) {
                                    double fractional_unc = content(i + binstart);
                                    h->SetBinContent(i + 1, std::abs(fractional_unc));
                                    
                                    hsum->SetBinContent(i + 1, hsum->GetBinContent(i + 1) + fractional_unc * fractional_unc);
                                }
                            }
                            else{
                                log<LOG_INFO>(L"%1% || did not find linear response for %2%") % __func__ % systname.c_str();

                                for(int i = 0; i < channel_nbins_x; i++){
                                    for(int j = channel_nbins_y*i; j < channel_nbins_y*(i+1); j++){
                                        diag1d(i) += channel_diag(j, j);
                                        for(int k = channel_nbins_y*i; k < channel_nbins_y*(i+1); k++){
                                            VarVec(i) += channel_cov(j, k);
                                        }
                                    }
                                }

                                float inv_diag1d;
                                for (size_t i = 0; i < channel_nbins_x; ++i) {
                                    inv_diag1d = 1/diag1d(i);
                                    h->SetBinContent(i+1, sqrt(inv_diag1d*VarVec(i)*inv_diag1d));
                                    hsum->SetBinContent(i+1, hsum->GetBinContent(i+1)+inv_diag1d*VarVec(i)*inv_diag1d);
                                }
                            }

                            const std::string &plotname = config.m_mcgen_variation_plotname_map.at(systname);
                            leg->AddEntry(h, plotname.c_str(), "l");
                            h->SetLineColor(colors[color_idx]);
                            h->SetLineStyle(line_styles[style_idx]);
                            hvec.push_back(h);

                        }//end syst
                        for (size_t i = 0; i < channel_nbins_x; ++i) {
                            hsum->SetBinContent(i+1, sqrt(hsum->GetBinContent(i+1)));
                        }
                        leg->AddEntry(hsum,"Sum","l");
                        // Diagnostic: Check hsum for bad values before drawing
                        bool hsum_has_bad = false;
                        for (int bin = 1; bin <= hsum->GetNbinsX(); ++bin) {
                            double val = hsum->GetBinContent(bin);
                            if (std::isnan(val) || std::isinf(val)) {
                                hsum_has_bad = true;
                                break;
                            }
                        }
                        if (hsum_has_bad) {
                            log<LOG_ERROR>(L"%1% || FATAL: hsum histogram for tag '%2%' contains NaN or Inf values! Plot will be empty.")
                                % __func__ % tag.c_str();
                        }
                        log<LOG_INFO>(L"%1% || hsum for tag '%2%': nbins=%3%, max=%4%, integral=%5%")
                            % __func__ % tag.c_str() % hsum->GetNbinsX() % hsum->GetMaximum() % hsum->Integral();


                        hsum->SetXTitle(xtitle.c_str());
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

                        TText *t = new TText();
                        t->SetNDC();                
                        t->SetTextFont(42);                          
                        t->SetTextSize(0.03);      
                        t->SetTextAlign(33);        
                        std::string pv = "PROfit v"+std::string(PROJECT_VERSION_STR);
                        t->DrawText(0.895, 0.945, pv.c_str()); 

                    }//end tag


                    //and each sum of sums to wrap it off!
                    c.cd(padIndex++);

                    TH1F* hsum = new TH1F( ("USum_"+std::to_string(global_channel_index)).c_str(),("Summary! "+name).c_str(), bin_edges.size()-1, bin_edges.data());
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
                    hsum->SetXTitle(xtitle.c_str());
                    hsum->SetTitle(("Summary: "+name).c_str());
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

                    TText *t = new TText();
                    t->SetNDC();                
                    t->SetTextFont(42);                          
                    t->SetTextSize(0.03);      
                    t->SetTextAlign(33);        
                    std::string pv = "PROfit v"+std::string(PROJECT_VERSION_STR);
                    t->DrawText(0.895, 0.945, pv.c_str()); 

                    c.Update();
                    c.Print(filename.c_str());
                    global_channel_index++;
                }
            }
        }

        c.Print((filename+"]").c_str());
        return 0;
    };
}
