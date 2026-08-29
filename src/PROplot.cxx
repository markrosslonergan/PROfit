#include "PROplot.h"
#include "TStyle.h"
#include "TArrow.h"
#include "TBox.h"
#include "TH1F.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TLegendEntry.h"
#include "TMarker.h"
#include "TText.h"
#include <cmath>
#include <Eigen/SVD>

namespace PROfit{

    void set_matrix_palette() {
        //Covariance colors, move this eslewher
        const Int_t NCont = 255;
        const Int_t NRGBs = 9;  // Reduced control points for smoother white stretch
        Double_t stops[NRGBs] = {
            0.0,    // -1.0 (dark blue)
            0.075,    // -0.6 (transition to white)
            0.3,    // -0.2 (mostly white)
            0.4,   // -0.04 (almost pure white)
            0.5,    //  0.0 (pure white)
            0.6,   // +0.04 (almost pure white)
            0.7,    // +0.2 (transition to red)
            0.925,    // +0.6 (strong red)
            1.0     // +1.0 (dark red)
        };

        Double_t red[NRGBs]   = {0.00,0.259,0.824, 0.949, 1.0, 0.988, 0.980, 0.918,0.839};
        Double_t green[NRGBs] = {0.341,0.404,0.890, 0.961, 1.0, 0.933, 0.824, 0.263,0.125};
        Double_t blue[NRGBs]  = {0.906,0.824,0.989, 0.980, 1.0, 0.929, 0.812, 0.208,0.024};

        TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
        gStyle->SetNumberContours(NCont);
    }

    std::vector<size_t> find_subchannels_by_pattern(const PROconfig &config,
                                                    const std::string &pattern) {
        std::vector<size_t> matched;
        if (pattern.empty()) return matched;
        // Unanchored regex (plain substrings behave as before; see PROconfig.h),
        // matching PROsyst's wildcard convention used in CreateFlatMatrix.
        // m_fullnames is the canonical list of
        // "<mode>_<detector>_<channel>_<subchannel>" names indexed by global
        // subchannel index.
        std::regex re = CompilePattern(pattern, "subchannel pattern");
        for (size_t i = 0; i < config.m_fullnames.size(); ++i) {
            if (PatternMatches(config.m_fullnames[i], re)) {
                matched.push_back(i);
            }
        }
        return matched;
    }

    Eigen::VectorXf build_subchannel_bin_mask(const PROconfig &config,
                                              const std::vector<size_t> &matched_subchannel_indices,
                                              int var_index) {
        Eigen::VectorXf mask = Eigen::VectorXf::Zero(config.m_num_variable_bins_total[var_index]);
        for (size_t isub : matched_subchannel_indices) {
            const size_t ic    = config.GetLocalChannelIndexFromGlobalSubchannelIndex(isub);
            const size_t start = config.GetGlobalVariableBinStart(isub, var_index);
            const size_t nbins = config.m_channel_variable_bins[ic][var_index].NBins();
            for (size_t b = 0; b < nbins; ++b) {
                mask(static_cast<Eigen::Index>(start + b)) = 1.0f;
            }
        }
        return mask;
    }

    Eigen::VectorXf build_subchannel_mask_spec(const PROconfig &config,
                                               const PROspec &spec,
                                               const std::vector<size_t> &matched_subchannel_indices,
                                               int var_index) {
        return build_subchannel_bin_mask(config, matched_subchannel_indices, var_index)
                   .cwiseProduct(spec.Spec());
    }

    std::map<std::string, std::unique_ptr<TH1D>> getCV1DHists(const PROspec &spec, const PROconfig& inconfig, bool scale, int other_index) {
        std::map<std::string, std::unique_ptr<TH1D>> hists;  


        size_t global_subchannel_index = 0;
        for(size_t im = 0; im < inconfig.m_num_modes; im++){
            for(size_t id =0; id < inconfig.m_num_detectors; id++){
                for(size_t ic = 0; ic < inconfig.m_num_channels; ic++){
                    for(size_t sc = 0; sc < inconfig.m_num_subchannels[ic]; sc++){
                        const std::string& subchannel_name  = inconfig.m_fullnames[global_subchannel_index];
                        const std::string& color = inconfig.m_subchannel_colors[ic][sc];
                        int rcolor = color == "NONE" ? kRed - 7 : inconfig.HexToROOTColor(color);
                        std::unique_ptr<TH1D> htmp = std::make_unique<TH1D>(spec.toTH1D(inconfig, global_subchannel_index, other_index));
                        htmp->SetDirectory(nullptr);  // copy ctor re-registers with gDirectory; detach to avoid name-collision warnings on repeated calls.
                        htmp->SetLineWidth(1);
                        htmp->SetLineColor(kBlack);
                        htmp->SetFillColor(rcolor);
                        if(scale) htmp->Scale(1,"width");
                        hists[subchannel_name] = std::move(htmp);
                        std::unique_ptr<TH1D> htmp_slc = std::make_unique<TH1D>(spec.toTH1DSlices(inconfig, global_subchannel_index, other_index));
                        if(htmp_slc){
                            htmp_slc->SetDirectory(nullptr);
                            hists[subchannel_name+"slc"] = std::move(htmp_slc);
                        }
                        ++global_subchannel_index;
                    }//end subchan
                }//end chan
            }//end det
        }//end mode
        return hists;
    }

    std::map<std::string, std::unique_ptr<TH2D>> getCV2DHists(const PROspec &spec, const PROconfig& inconfig, bool scale, int other_index) {
        std::map<std::string, std::unique_ptr<TH2D>> hists;

        size_t global_subchannel_index = 0;
        for(size_t im = 0; im < inconfig.m_num_modes; im++){
            for(size_t id = 0; id < inconfig.m_num_detectors; id++){
                for(size_t ic = 0; ic < inconfig.m_num_channels; ic++){
                    // toTH2D reads Edges(1)/NBinsAlong(1), which is out-of-bounds for
                    // channels whose binning for this variable is 1D.
                    if(inconfig.m_channel_variable_dims[ic][other_index] != 2) {
                        global_subchannel_index += inconfig.m_num_subchannels[ic];
                        continue;
                    }
                    for(size_t sc = 0; sc < inconfig.m_num_subchannels[ic]; sc++){
                        const std::string& subchannel_name  = inconfig.m_fullnames[global_subchannel_index];
                        std::unique_ptr<TH2D> htmp = std::make_unique<TH2D>(spec.toTH2D(inconfig, global_subchannel_index, other_index));
                        htmp->SetDirectory(nullptr);  // copy ctor re-registers with gDirectory; detach to avoid name-collision warnings on repeated calls.
                        if(scale) htmp->Scale(1,"width");
                        hists[subchannel_name] = std::move(htmp);
                        ++global_subchannel_index;
                    }//end subchan
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
                constexpr int samples_per_segment = 20;
                bin_graphs.reserve(nbins);

                for (size_t j = 0; j < nbins; ++j) {
                    std::unique_ptr<TGraph> curve = std::make_unique<TGraph>();
                    std::unique_ptr<TGraph> fixed_pts = std::make_unique<TGraph>();

                    // Access the segment range for this bin
                    size_t seg_offset = j * nsegs;

                    for (int k = 0; k < nsegs; ++k) {
                        float shift = spline.segments[seg_offset + k].knot;
                        fixed_pts->SetPoint(k, shift, systs.GetSplineShift(i, shift, j));
                    }
                    fixed_pts->SetPoint(nsegs, systs.spline_hi[i], systs.GetSplineShift(i, systs.spline_hi[i], j));

                    float lo = systs.spline_has_restrict[i] ? systs.spline_restrict_lo[i] : systs.spline_lo[i];
                    float hi = systs.spline_has_restrict[i] ? systs.spline_restrict_hi[i] : systs.spline_hi[i];
                    int samples = samples_per_segment * nsegs;
                    for (int k = 0; k <= samples; ++k) {
                        float shift = lo + (hi - lo) * k / samples;
                        curve->SetPoint(k, shift, systs.GetSplineShift(i, shift, j));
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

	Eigen::MatrixXf cov = Eigen::MatrixXf::Zero(cv.size(), cv.size());
        Eigen::VectorXf delta;
        //Fills already collapsed
        for(size_t i = 0; i < nerrorsample; ++i){
            Eigen::VectorXf var = FillSystRandomThrow(config, prop, syst, model,cv_spec, cvparams, dseed(PROseed::global_rng), other_index).Spec();
            specs.push_back(var);
            delta = cv - var;
            cov += delta  * delta.transpose();
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
	ebar.covariance = cov/nerrorsample;
        return ebar;
    }

    PROerrorbar getCovarianceOnlyErrorBand(const PROconfig &config, const PROpeller &prop, const PROsyst &syst, const PROmodel &model, const Eigen::VectorXf &params, bool scale, int var_index, const Eigen::VectorXf &data_spec) {
        Eigen::VectorXf cv = FillSpectra(config, prop, syst, model, params, true, var_index).Spec();
        Eigen::VectorXf cv_coll = CollapseMatrix(config, cv, var_index);

        Eigen::MatrixXf cov;
        if(syst.GetNCovar() > 0) {
            Eigen::MatrixXf L = syst.DecomposeFractionalCovariance(config, cv);
            cov = L * L.transpose();
        } else {
            cov = Eigen::MatrixXf::Zero(cv_coll.size(), cv_coll.size());
        }

        // Data-constrained posterior of the covariance systematics (Putnam SBN
        // note Eqs. 7-8): with u = d - cv on the contributing bins (active and
        // data > 0, PROchi convention, C = diag(max(data,1))), the prediction
        // shifts by Sigma(C+Sigma)^-1 u and the band covariance becomes
        // Sigma - Sigma(C+Sigma)^-1 Sigma. Exact here: with no free fit
        // parameters the Gaussian conditional is the whole posterior.
        Eigen::VectorXf shift = Eigen::VectorXf::Zero(cv_coll.size());
        if(data_spec.size() != 0 && data_spec.size() != cv_coll.size()) {
            log<LOG_ERROR>(L"%1% || data_spec has %2% bins but variable %3% has %4% collapsed bins; ignoring the data constraint.") % __func__ % data_spec.size() % var_index % cv_coll.size();
        } else if(data_spec.size() != 0 && syst.GetNCovar() > 0) {
            std::vector<int> contrib;
            for(int i = 0; i < data_spec.size(); ++i)
                if(config.IsBinActive(var_index, i) && data_spec(i) > 0)
                    contrib.push_back(i);
            if(!contrib.empty()) {
                const size_t nb = contrib.size();
                Eigen::MatrixXd Sig_full = cov.cast<double>();
                Eigen::MatrixXd K(cv_coll.size(), nb);      // Sigma[:, contrib]
                Eigen::MatrixXd M(nb, nb);                  // C + Sigma on contrib bins
                Eigen::VectorXd u(nb);
                for(size_t a = 0; a < nb; ++a) {
                    K.col(a) = Sig_full.col(contrib[a]);
                    for(size_t b = 0; b < nb; ++b) M(a, b) = Sig_full(contrib[a], contrib[b]);
                    M(a, a) += std::max<double>(data_spec(contrib[a]), 1.0);
                    u(a) = data_spec(contrib[a]) - cv_coll(contrib[a]);
                }
                Eigen::LDLT<Eigen::MatrixXd> M_ldlt(M);
                shift = (K * M_ldlt.solve(u)).cast<float>();
                cov = (Sig_full - K * M_ldlt.solve(K.transpose())).cast<float>();
            }
        }

        PROerrorbar ebar(cv_coll.size());
        for(int i = 0; i < cv_coll.size(); ++i) {
            float scale_factor = scale ? 1.0/config.collapsed_bin_widths.at(var_index)(i) : 1.0;
            if(std::isnan(scale_factor)) scale_factor = 1;
            float err = std::sqrt(std::max(cov(i,i), 0.0f)) * scale_factor;
            ebar.error_up(i) = err;
            ebar.error_down(i) = err;
            ebar.error_point(i) = cv_coll(i)*scale_factor;
            ebar.center_shift(i) = shift(i)*scale_factor;
        }
        ebar.covariance = cov;
        return ebar;
    }

    PROsubtractedErrorBand getErrorBandBkgSubtracted(const PROconfig &config, const PROpeller &prop, const PROsyst &syst, const PROmodel &model, const PROspec &cv_spec, const Eigen::VectorXf &cvparams, const std::vector<size_t> &bkg_subchannels, bool scale, int other_index) {

        Eigen::VectorXf mask        = build_subchannel_bin_mask(config, bkg_subchannels, other_index);
        Eigen::VectorXf bkg_cv_full = cv_spec.Spec().cwiseProduct(mask);
        Eigen::VectorXf cv_sig      = CollapseMatrix(config, Eigen::VectorXf(cv_spec.Spec() - bkg_cv_full), other_index);
        Eigen::VectorXf bkg_cv_coll = CollapseMatrix(config, bkg_cv_full, other_index);

        size_t nerrorsample = 2500;

        std::vector<Eigen::VectorXf> sig_specs;
        std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());

        Eigen::MatrixXf cov = Eigen::MatrixXf::Zero(cv_sig.size(), cv_sig.size());
        Eigen::VectorXf bkg_var = Eigen::VectorXf::Zero(cv_sig.size());
        Eigen::VectorXf delta;
        for(size_t i = 0; i < nerrorsample; ++i){
            // Subtract each throw's OWN background so bkg systematic variations
            // cancel: the band that comes out is signal-only.
            auto [sig, bkg] = FillSystRandomThrowSplit(config, prop, syst, model, cv_spec, cvparams, dseed(PROseed::global_rng), other_index, mask);
            sig_specs.push_back(sig.Spec());
            delta = cv_sig - sig.Spec();
            cov += delta * delta.transpose();
            bkg_var += (bkg.Spec() - bkg_cv_coll).array().square().matrix();
        }

        PROsubtractedErrorBand result(cv_sig.size());
        for(int i = 0; i < cv_sig.size(); ++i) {
            std::vector<float> binconts(nerrorsample);
            for(size_t j = 0; j < nerrorsample; ++j) {
                binconts[j] = sig_specs[j](i);
            }
            float scale_factor = scale ? 1.0/config.collapsed_bin_widths.at(other_index)(i) :  1.0;
            if(std::isnan(scale_factor)) scale_factor = 1;
            std::sort(binconts.begin(), binconts.end());
            float ehi = std::abs((binconts[2.5*840] - cv_sig(i))*scale_factor);
            float elo = std::abs((cv_sig(i) - binconts[2.5*160])*scale_factor);
            result.band.error_up(i) =  ehi;
            result.band.error_down(i) =  elo;
            result.band.error_point(i) = cv_sig(i)*scale_factor;
        }
        result.band.covariance = cov/nerrorsample;
        result.bkg_cv_collapsed = bkg_cv_coll;
        result.bkg_sigma_collapsed = (bkg_var/nerrorsample).array().sqrt();
        result.bkg_mcstat_var_collapsed = CollapseMatrix(config, Eigen::VectorXf(cv_spec.Error().array().square().matrix().cwiseProduct(mask)), other_index);
        return result;
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
	// note: since this is just for plotting diagonals, won't worry about filling in
	//       covariance for collapsed 2d histogram.
        PROerrorbar* errband_1d = new PROerrorbar(nbinsx);
        log<LOG_DEBUG>(L"%1% || input err_band pt: %2%") % __func__ % errband.error_point;
        log<LOG_DEBUG>(L"%1% || input err_band up: %2%") % __func__ % errband.error_up;
        // sum error points, combine errors in quadrature
        for(size_t bx = 0; bx < nbinsx; bx++){
            if(dims == 2){

                double err_sq = 0;
                for(int by1 = (int)(nbinsy*bx)+offset; by1 < (int)(nbinsy*(bx+1))+offset; by1++){
                    errband_1d->error_point(bx) += errband.error_point(by1);
                    errband_1d->center_shift(bx) += errband.center_shift(by1);
                    for(int by2 = (int)(nbinsy*bx)+offset; by2 < (int)(nbinsy*(bx+1))+offset; by2++){
                        err_sq += errband.covariance(by1, by2);
                    }
                }

                // Clamp: the central covariance can make a fully-constrained
                // block sum slightly negative in float.
                errband_1d->error_up(bx) = std::sqrt(std::max(0.0, err_sq));
                errband_1d->error_down(bx) = std::sqrt(std::max(0.0, err_sq));
            }
            else{
                errband_1d->error_point(bx) = errband.error_point(bx+offset);
                errband_1d->center_shift(bx) = errband.center_shift(bx+offset);
                errband_1d->error_up(bx) = errband.error_up(bx+offset);
                errband_1d->error_down(bx) = errband.error_down(bx+offset);
            }
        }
        log<LOG_DEBUG>(L"%1% || err_band pt: %2%") % __func__ % errband_1d->error_point;
        log<LOG_DEBUG>(L"%1% || err_band up: %2%") % __func__ % errband_1d->error_up;
        log<LOG_DEBUG>(L"%1% || err_band down: %2%") % __func__ % errband_1d->error_down;
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
            [[maybe_unused]] int var_index)
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
                data_ratio->SetLineStyle(kSolid);
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
                    leg->AddEntry(bf_ratio, posterrband ? "Constrained Best-Fit" : "Best-Fit", "l");
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

    void plot_hist1ds(TCanvas* c, TH1D* cv_hist, TGraphAsymmErrors* errband, THStack* cvstack, std::vector<std::pair<std::string, const char*>>* subplots, TH1D* bf_hist, TGraphAsymmErrors* posterrband, TH1D* data_hist, std::string* dat_str, PlotOptions opt, std::string hist_titles, std::string ratio_titles, const std::string &filename, const PlotBounds &bounds, const std::string &text){

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
                cv_hist->SetMinimum(0.000001); // to avoid including a zero label, which can overlap the ratio plot
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
                leg->AddEntry(leg_hack,"Best-Fit #pm 1#sigma (post-fit)" ,"fl");
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
                g->SetLineStyle(kSolid);
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

        if(!text.empty()) {
            TLine *dummy_line = new TLine(0,0,0.1,0);
            dummy_line->SetLineColor(kWhite);
            dummy_line->SetLineWidth(0);
            TLegendEntry *chi_entry = leg->AddEntry(dummy_line,text.c_str(),"l");
            chi_entry->SetTextFont(42);
            chi_entry->SetTextSize(0.03);
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
            ratio->SetLineStyle(kSolid);
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

    // Ratio of the same channel between two detectors, drawn as a spectrum.
    // Uses the error-band covariance rather than TH1::Divide so the correlation
    // between detectors is propagated. For 2D channels the y-axis is summed,
    // matching the projection used for the 1D pages.
    void plot_detector_ratio_spectra(TCanvas &c,
                                     const PROconfig &config,
                                     const Eigen::VectorXf &cv_coll,
                                     const std::optional<Eigen::VectorXf> &bf_coll,
                                     const std::optional<Eigen::VectorXf> &data_coll,
                                     const std::optional<PROerrorbar> &errband,
                                     const std::optional<PROerrorbar> &posterrband,
                                     const std::vector<size_t> &channel_offsets,
                                     const std::string &filename,
                                     int other_index)
    {
        if(config.m_num_detectors < 2) return;

        // Deliberately empty: --plot-bounds ymax etc. refer to event counts, not ratios.
        PlotBounds ratio_bounds;

        auto sum_vec = [](const Eigen::VectorXf &v, size_t off, size_t bx, size_t ny) {
            float s = 0.0f;
            for(size_t by = 0; by < ny; ++by) s += v(off + bx*ny + by);
            return s;
        };
        auto sum_block = [](const Eigen::MatrixXf &m, size_t offr, size_t offc, size_t bx, size_t ny) {
            float s = 0.0f;
            for(size_t i = 0; i < ny; ++i)
                for(size_t j = 0; j < ny; ++j)
                    s += m(offr + bx*ny + i, offc + bx*ny + j);
            return s;
        };

        for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
        for(size_t det1 = 0; det1 < config.m_num_detectors; ++det1) {
        for(size_t det2 = det1 + 1; det2 < config.m_num_detectors; ++det2) {
        for(size_t ch = 0; ch < config.m_num_channels; ++ch) {

            const size_t gidx1 = (mode*config.m_num_detectors + det1)*config.m_num_channels + ch;
            const size_t gidx2 = (mode*config.m_num_detectors + det2)*config.m_num_channels + ch;
            if(gidx1 >= channel_offsets.size() || gidx2 >= channel_offsets.size()) continue;
            const size_t off1 = channel_offsets[gidx1];
            const size_t off2 = channel_offsets[gidx2];

            const size_t nx = config.m_channel_variable_bins[ch][other_index].NBinsAlong(0);
            const size_t ny = config.m_channel_variable_dims[ch][other_index] == 2
                            ? config.m_channel_variable_bins[ch][other_index].NBinsAlong(1) : 1;
            std::vector<float> edges = config.m_channel_variable_bins[ch][other_index].Edges(0);

            const std::string ratname = config.m_detector_plotnames[det1] + " / " + config.m_detector_plotnames[det2];
            const std::string xtitle   = config.GetChannelXAxisTitle(ch, other_index);
            const std::string title    = config.m_mode_plotnames[mode] + " " + config.m_channel_plotnames[ch]
                                       + ";" + xtitle + ";" + ratname;
            const std::string sfx = "_ratspec_" + std::to_string(mode) + "_" + std::to_string(det1)
                                  + "_" + std::to_string(det2) + "_" + std::to_string(ch);

            std::unique_ptr<TH1D> cv_rat = std::make_unique<TH1D>(("cvrat"+sfx).c_str(), title.c_str(), nx, edges.data());
            cv_rat->SetDirectory(nullptr);
            cv_rat->SetLineWidth(2);

            std::unique_ptr<TH1D> bf_rat, data_rat;
            if(bf_coll) {
                bf_rat = std::make_unique<TH1D>(("bfrat"+sfx).c_str(), title.c_str(), nx, edges.data());
                bf_rat->SetDirectory(nullptr);
            }
            if(data_coll) {
                data_rat = std::make_unique<TH1D>(("datrat"+sfx).c_str(), title.c_str(), nx, edges.data());
                data_rat->SetDirectory(nullptr);
                data_rat->SetLineColor(kBlack);
                data_rat->SetLineWidth(2);
                data_rat->SetMarkerStyle(kFullCircle);
                data_rat->SetMarkerColor(kBlack);
                data_rat->SetMarkerSize(1);
            }

            std::vector<float> rel_err(nx, 0.0f), post_rel_err(nx, 0.0f);

            for(size_t bx = 0; bx < nx; ++bx) {
                const float a = sum_vec(cv_coll, off1, bx, ny);
                const float b = sum_vec(cv_coll, off2, bx, ny);
                const float r = (b != 0.0f) ? a/b : 0.0f;
                cv_rat->SetBinContent(bx+1, r);

                if(errband && a != 0.0f && b != 0.0f) {
                    const Eigen::MatrixXf &C = errband->covariance;
                    const float caa = sum_block(C, off1, off1, bx, ny);
                    const float cbb = sum_block(C, off2, off2, bx, ny);
                    const float cab = sum_block(C, off1, off2, bx, ny);
                    // clamp: the cancellation can go slightly negative in float
                    const float relvar = std::max(0.0f, caa/(a*a) + cbb/(b*b) - 2.0f*cab/(a*b));
                    rel_err[bx] = std::sqrt(relvar);
                    cv_rat->SetBinError(bx+1, r*rel_err[bx]);
                }

                if(bf_coll) {
                    const float ba = sum_vec(*bf_coll, off1, bx, ny);
                    const float bb = sum_vec(*bf_coll, off2, bx, ny);
                    const float br = (bb != 0.0f) ? ba/bb : 0.0f;
                    bf_rat->SetBinContent(bx+1, br);
                    if(posterrband && ba != 0.0f && bb != 0.0f) {
                        const Eigen::MatrixXf &C = posterrband->covariance;
                        const float caa = sum_block(C, off1, off1, bx, ny);
                        const float cbb = sum_block(C, off2, off2, bx, ny);
                        const float cab = sum_block(C, off1, off2, bx, ny);
                        const float relvar = std::max(0.0f, caa/(ba*ba) + cbb/(bb*bb) - 2.0f*cab/(ba*bb));
                        post_rel_err[bx] = std::sqrt(relvar);
                        bf_rat->SetBinError(bx+1, br*post_rel_err[bx]);
                    }
                }

                if(data_coll) {
                    const float da = sum_vec(*data_coll, off1, bx, ny);
                    const float db = sum_vec(*data_coll, off2, bx, ny);
                    const float dr = (db != 0.0f) ? da/db : 0.0f;
                    data_rat->SetBinContent(bx+1, dr);
                    // data detectors are statistically independent
                    const float dvar = (da > 0.0f && db > 0.0f) ? (1.0f/da + 1.0f/db) : 0.0f;
                    data_rat->SetBinError(bx+1, dr*std::sqrt(dvar));
                }
            }

            std::unique_ptr<TGraphAsymmErrors> band, post_band;
            if(errband) {
                band = std::make_unique<TGraphAsymmErrors>(cv_rat.get());
                for(size_t bx = 0; bx < nx; ++bx) {
                    const float e = cv_rat->GetBinContent(bx+1)*rel_err[bx];
                    band->SetPointEYhigh(bx, e);
                    band->SetPointEYlow(bx, e);
                }
            }
            if(bf_coll && posterrband) {
                post_band = std::make_unique<TGraphAsymmErrors>(bf_rat.get());
                for(size_t bx = 0; bx < nx; ++bx) {
                    const float e = bf_rat->GetBinContent(bx+1)*post_rel_err[bx];
                    post_band->SetPointEYhigh(bx, e);
                    post_band->SetPointEYlow(bx, e);
                }
            }

            std::string dat_str = "Data";
            plot_hist1ds(&c, cv_rat.get(), band.get(), nullptr, nullptr,
                         bf_rat.get(), post_band.get(), data_rat.get(), &dat_str,
                         PlotOptions{}, title, "", filename, ratio_bounds, std::string());
        }}}}
    }

    // Ratio between two channels in the same detector, drawn as a spectrum.
    // Basically a copy of `plot_detector_ratio_spectra`, with two differences: 
    // in the 2D case, the two slots can have different y binning, so the covariance blocks
    // are rectangular and the y stride is per-slot; and the x binning has to be checked
    // rather than assumed equal (which was more natural for multi-detector ratios).
    void plot_channel_ratio_spectra(TCanvas &c,
                                    const PROconfig &config,
                                    const Eigen::VectorXf &cv_coll,
                                    const std::optional<Eigen::VectorXf> &bf_coll,
                                    const std::optional<Eigen::VectorXf> &data_coll,
                                    const std::optional<PROerrorbar> &errband,
                                    const std::optional<PROerrorbar> &posterrband,
                                    const std::vector<size_t> &channel_offsets,
                                    const std::string &filename,
                                    int other_index,
                                    PlotOptions opt)
    {
        if(config.m_num_channels < 2) return;

        PlotBounds ratio_bounds;

        auto sum_vec = [](const Eigen::VectorXf &v, size_t off, size_t bx, size_t ny) {
            float s = 0.0f;
            for(size_t by = 0; by < ny; ++by) s += v(off + bx*ny + by);
            return s;
        };

        // rows and columns may come from channels with different y binning, so per-slot stride 
        auto sum_block = [](const Eigen::MatrixXf &m, size_t offr, size_t ny_r,
                            size_t offc, size_t ny_c, size_t bx) {
            float s = 0.0f;
            for(size_t i = 0; i < ny_r; ++i)
                for(size_t j = 0; j < ny_c; ++j)
                    s += m(offr + bx*ny_r + i, offc + bx*ny_c + j);
            return s;
        };
        auto same_edges = [](const std::vector<float> &a, const std::vector<float> &b) {
            if(a.size() != b.size()) return false;
            for(size_t i = 0; i < a.size(); ++i) {
                const float sc = std::max(1.0f, std::max(std::fabs(a[i]), std::fabs(b[i])));
                if(std::fabs(a[i] - b[i]) > 1e-4f*sc) return false;
            }
            return true;
        };

        for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
        for(size_t det = 0; det < config.m_num_detectors; ++det) {
        for(size_t ch1 = 0; ch1 < config.m_num_channels; ++ch1) {
        for(size_t ch2 = ch1 + 1; ch2 < config.m_num_channels; ++ch2) {

            const size_t gidx1 = (mode*config.m_num_detectors + det)*config.m_num_channels + ch1;
            const size_t gidx2 = (mode*config.m_num_detectors + det)*config.m_num_channels + ch2;
            if(gidx1 >= channel_offsets.size() || gidx2 >= channel_offsets.size()) continue;
            const size_t off1 = channel_offsets[gidx1];
            const size_t off2 = channel_offsets[gidx2];

            const size_t nx   = config.m_channel_variable_bins[ch1][other_index].NBinsAlong(0);
            const size_t nx_d = config.m_channel_variable_bins[ch2][other_index].NBinsAlong(0);
            std::vector<float> edges   = config.m_channel_variable_bins[ch1][other_index].Edges(0);
            std::vector<float> edges_d = config.m_channel_variable_bins[ch2][other_index].Edges(0);
            if(nx != nx_d || !same_edges(edges, edges_d)) {
                log<LOG_WARNING>(L"%1% || Skipping channel ratio %2%/%3%: x binning differs (%4% vs %5% bins). A ratio requires identical binning in both channels.")
                    % __func__ % config.m_channel_names[ch1].c_str() % config.m_channel_names[ch2].c_str() % nx % nx_d;
                continue;
            }

            const size_t ny1 = config.m_channel_variable_dims[ch1][other_index] == 2
                             ? config.m_channel_variable_bins[ch1][other_index].NBinsAlong(1) : 1;
            const size_t ny2 = config.m_channel_variable_dims[ch2][other_index] == 2
                             ? config.m_channel_variable_bins[ch2][other_index].NBinsAlong(1) : 1;

            const std::string xt1 = config.GetChannelXAxisTitle(ch1, other_index);
            const std::string xt2 = config.GetChannelXAxisTitle(ch2, other_index);
            if(xt1 != xt2) {
                log<LOG_WARNING>(L"%1% || Channel ratio %2%/%3%: x axis titles differ (%4% vs %5%). Binning is compatible so the ratio is drawn, but check that the two channels bin the same observable.")
                    % __func__ % config.m_channel_names[ch1].c_str() % config.m_channel_names[ch2].c_str() % xt1.c_str() % xt2.c_str();
            }

            const std::string ratname = config.m_channel_plotnames[ch1] + " / " + config.m_channel_plotnames[ch2];
            const std::string title   = config.m_mode_plotnames[mode] + " " + config.m_detector_plotnames[det]
                                      + ";" + xt1 + ";" + ratname;
            const std::string sfx = "_chratspec_" + std::to_string(mode) + "_" + std::to_string(det)
                                  + "_" + std::to_string(ch1) + "_" + std::to_string(ch2);

            std::unique_ptr<TH1D> cv_rat = std::make_unique<TH1D>(("cvrat"+sfx).c_str(), title.c_str(), nx, edges.data());
            cv_rat->SetDirectory(nullptr);
            cv_rat->SetLineWidth(2);

            std::unique_ptr<TH1D> bf_rat, data_rat;
            if(bf_coll) {
                bf_rat = std::make_unique<TH1D>(("bfrat"+sfx).c_str(), title.c_str(), nx, edges.data());
                bf_rat->SetDirectory(nullptr);
            }
            if(data_coll) {
                data_rat = std::make_unique<TH1D>(("datrat"+sfx).c_str(), title.c_str(), nx, edges.data());
                data_rat->SetDirectory(nullptr);
                data_rat->SetLineColor(kBlack);
                data_rat->SetLineWidth(2);
                data_rat->SetMarkerStyle(kFullCircle);
                data_rat->SetMarkerColor(kBlack);
                data_rat->SetMarkerSize(1);
            }

            std::vector<float> rel_err(nx, 0.0f), post_rel_err(nx, 0.0f);

            for(size_t bx = 0; bx < nx; ++bx) {
                const float a = sum_vec(cv_coll, off1, bx, ny1);
                const float b = sum_vec(cv_coll, off2, bx, ny2);
                const float r = (b != 0.0f) ? a/b : 0.0f;
                cv_rat->SetBinContent(bx+1, r);

                if(errband && a != 0.0f && b != 0.0f) {
                    const Eigen::MatrixXf &C = errband->covariance;
                    const float caa = sum_block(C, off1, ny1, off1, ny1, bx);
                    const float cbb = sum_block(C, off2, ny2, off2, ny2, bx);
                    const float cab = sum_block(C, off1, ny1, off2, ny2, bx);
                    const float raw = caa/(a*a) + cbb/(b*b) - 2.0f*cab/(a*b);
                    
                    if(raw < 0.0f)
                        log<LOG_WARNING>(L"%1% || Channel ratio %2%/%3% bin %4%: relative variance came out negative (%5%) before clamping. The systematic cancellation between these channels is at or below the error-band throw noise.")
                            % __func__ % config.m_channel_names[ch1].c_str() % config.m_channel_names[ch2].c_str() % bx % raw;
                    rel_err[bx] = std::sqrt(std::max(0.0f, raw));
                    cv_rat->SetBinError(bx+1, r*rel_err[bx]);
                }

                if(bf_coll) {
                    const float ba = sum_vec(*bf_coll, off1, bx, ny1);
                    const float bb = sum_vec(*bf_coll, off2, bx, ny2);
                    const float br = (bb != 0.0f) ? ba/bb : 0.0f;
                    bf_rat->SetBinContent(bx+1, br);
                    if(posterrband && ba != 0.0f && bb != 0.0f) {
                        const Eigen::MatrixXf &C = posterrband->covariance;
                        const float caa = sum_block(C, off1, ny1, off1, ny1, bx);
                        const float cbb = sum_block(C, off2, ny2, off2, ny2, bx);
                        const float cab = sum_block(C, off1, ny1, off2, ny2, bx);
                        const float relvar = std::max(0.0f, caa/(ba*ba) + cbb/(bb*bb) - 2.0f*cab/(ba*bb));
                        post_rel_err[bx] = std::sqrt(relvar);
                        bf_rat->SetBinError(bx+1, br*post_rel_err[bx]);
                    }
                }

                if(data_coll) {
                    const float da = sum_vec(*data_coll, off1, bx, ny1);
                    const float db = sum_vec(*data_coll, off2, bx, ny2);
                    const float dr = (db != 0.0f) ? da/db : 0.0f;
                    data_rat->SetBinContent(bx+1, dr);
                    
                    // assumes the two channels are mutually exclusive selections (no correlation term...)
                    const float dvar = (da > 0.0f && db > 0.0f) ? (1.0f/da + 1.0f/db) : 0.0f;
                    data_rat->SetBinError(bx+1, dr*std::sqrt(dvar));
                }
            }

            std::unique_ptr<TGraphAsymmErrors> band, post_band;
            if(errband) {
                band = std::make_unique<TGraphAsymmErrors>(cv_rat.get());
                for(size_t bx = 0; bx < nx; ++bx) {
                    const float e = cv_rat->GetBinContent(bx+1)*rel_err[bx];
                    band->SetPointEYhigh(bx, e);
                    band->SetPointEYlow(bx, e);
                }
            }
            if(bf_coll && posterrband) {
                post_band = std::make_unique<TGraphAsymmErrors>(bf_rat.get());
                for(size_t bx = 0; bx < nx; ++bx) {
                    const float e = bf_rat->GetBinContent(bx+1)*post_rel_err[bx];
                    post_band->SetPointEYhigh(bx, e);
                    post_band->SetPointEYlow(bx, e);
                }
            }

            PlotOptions ropt = PlotOptions{};
            std::string ratio_titles = "";
            if(data_rat && bf_rat && post_band && bool(opt&PlotOptions::DataPostfitRatio)) {
                ropt = PlotOptions::DataPostfitRatio;
                ratio_titles = ";" + xt1 + ";Data/Best-Fit";
            } else if(data_rat && band) {
                ropt = PlotOptions::DataMCRatio;
                ratio_titles = ";" + xt1 + ";Data/MC";
            }

            std::string dat_str = "Data";
            plot_hist1ds(&c, cv_rat.get(), band.get(), nullptr, nullptr,
                         bf_rat.get(), post_band.get(), data_rat.get(), &dat_str,
                         ropt, title, ratio_titles, filename, ratio_bounds, std::string());
        }}}}
    }
    
    std::map<std::string, TObject *> plot_channels(const std::string &filename, const PROconfig &config, std::optional<PROspec> cv, std::optional<PROspec> best_fit, std::optional<PROdata> data, std::optional<PROerrorbar> errband, std::optional<PROerrorbar> posterrband, std::vector<TPaveText> &texts, const PlotBounds &bounds, PlotOptions opt, int other_index, bool ratio_bool, bool plot_channel_ratios, const std::vector<size_t> *skip_stack_subchannels, PROmetric *chi_metric, const PROspec *chi_spec) {

        log<LOG_DEBUG>(L"%1% || Starting plot_channels") % __func__;
        std::string rat_y_title = bool(opt&PlotOptions::DataMCRatio) ? "Data/MC" : "Data/Best-Fit";
        std::map<std::string, TObject *> objs;

        TCanvas c;
        c.Print((filename+"[").c_str());

        std::map<std::string, std::unique_ptr<TH1D>> cv1dhists;
        std::map<std::string, std::unique_ptr<TH2D>> cv2dhists;

        // Keep as we go through loop for ratio plot
        std::vector<TH1D> data_hists;
        std::vector<TH1D> cv_hists;
        std::vector<TH1D> bf_hists;
        std::vector<size_t> channel_offsets;   // collapsed bin start of each mode/det/channel

        size_t global_subchannel_index = 0;
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

                    log<LOG_DEBUG>(L"%1% || channel %2%") % __func__ % channel;
                    channel_offsets.push_back(tot_offset);
                    std::string mdc = config.m_mode_names[mode]+"_"+config.m_detector_names[det]+"_"+config.m_channel_names[channel];

                    std::string xtitle = config.GetChannelXAxisTitle(channel, other_index);
                    std::string ratio_titles = ";"+xtitle+";"+rat_y_title;

                    std::string chan_unit = config.GetChannelUnit(channel, other_index);
                    std::string ytitle;
                    if(bool(opt&PlotOptions::AreaNormalized)) {
                        ytitle = "Area Normalized";
                    } else if(bool(opt&PlotOptions::BinWidthScaled)) {
                        if(chan_unit.empty()) {
                            ytitle = "Events/unit";
                        } else {
                            ytitle = "Events/" + chan_unit;
                        }
                    } else {
                        ytitle = "Events";
                    }

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
                    Eigen::VectorXf data_err_1d = Eigen::VectorXf::Zero(channel_nbins_x);
                    if(data){
                        data_spec_1d = make_1d_spec(data->Spec(), channel_nbins_x, channel_nbins_y, tot_offset, config.m_channel_variable_dims[channel][other_index]);
                        // Combine per-bin errors in quadrature (summing squared errors is
                        // what make_1d_spec's bin-sum does for dims==2; passthrough for 1D).
                        data_err_1d = make_1d_spec(data->Error().array().square(), channel_nbins_x, channel_nbins_y, tot_offset, config.m_channel_variable_dims[channel][other_index]).array().sqrt();
                    }

                    PROerrorbar *errband_1d = NULL;
                    if(errband){
                        errband_1d = make_1d_err(*errband, channel_nbins_x, channel_nbins_y, tot_offset, config.m_channel_variable_dims[channel][other_index]);
                    }

                    PROerrorbar *posterrband_1d = NULL;
                    if(posterrband){
                        posterrband_1d = make_1d_err(*posterrband, channel_nbins_x, channel_nbins_y, tot_offset, config.m_channel_variable_dims[channel][other_index]);
                    }

                    std::string projected_x_chi_label; // We want to pass this to the 1d plotter outside the 2d plotting
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

                        // Helpers to plot chi^2 for each heatmap/slice/projection
                        const size_t channel_start = config.GetCollapsedGlobalVariableBinStart(global_channel_index, other_index);

                        // In `groups`, each outer vector is one displayed bin; its inner vector lists the flattened
                        // source bins to sum. For a 3x2 Y projection, groups looks like {{0, 2, 4}, {1, 3, 5}}, i.e.,
                        // the first bin contains bins 0, 2, and 4 from the flat bins, and the second bin contains bins
                        // 1, 3, and 5 from the flat bins.
                        auto make_projection = [&](const std::vector<std::vector<size_t>> &groups) {
                            size_t nrows = 0;
                            for(const auto &group : groups) {
                                if(std::any_of(group.begin(), group.end(), [&](size_t bin) {
                                    return config.IsBinActive(other_index, channel_start + bin);
                                })) ++nrows;
                            }

                            // Maps from original bins (nbins_p_2dchan) to active bins in projection (nrows)
                            Eigen::MatrixXf projection = Eigen::MatrixXf::Zero(nrows, nbins_p_2dchan);
                            size_t row = 0;
                            for(const auto &group : groups) {
                                bool active = false;
                                for(size_t bin : group) {
                                    if(config.IsBinActive(other_index, channel_start + bin)) {
                                        projection(row, bin) = 1.0f;
                                        active = true;
                                    }
                                }
                                if (active) ++row; // Only advance matrix row if at least one bin in group is active
                            }
                            return projection;
                        };
                        auto chi_label = [&](const Eigen::MatrixXf &projection) {
                            if(!chi_metric || !chi_spec || projection.rows() == 0) return std::string();
                            const float chi2 = chi_metric->getSingleChannelChi(global_channel_index, *chi_spec, other_index, projection);
                            return std::string("#chi^{2}/nbins = ") + to_string_prec(chi2, 2) + "/" + std::to_string(projection.rows());
                        };
                        auto draw_chi_label = [&](const std::string &label) {
                            if(label.empty()) return;
                            TPaveText text(0.62, 0.91, 0.89, 0.96, "NDC");
                            text.AddText(label.c_str());
                            text.SetFillColor(0);
                            text.SetBorderSize(0);
                            text.SetTextAlign(12);
                            text.SetTextFont(42);
                            text.SetTextSize(0.03);
                            text.DrawClone();
                        };

                        std::vector<std::vector<size_t>> full_groups(nbins_p_2dchan);
                        for(size_t bin = 0; bin < (size_t)nbins_p_2dchan; ++bin) full_groups[bin] = {bin};
                        const std::string full_chi_label = chi_label(make_projection(full_groups));
                        std::vector<std::vector<size_t>> x_projection_groups(channel_nbins_x);
                        for(size_t xbin = 0; xbin < channel_nbins_x; ++xbin) {
                            for(size_t ybin = 0; ybin < channel_nbins_y; ++ybin) {
                                x_projection_groups[xbin].push_back(xbin*channel_nbins_y + ybin);
                            }
                        }
                        projected_x_chi_label = chi_label(make_projection(x_projection_groups));

                        const std::string plot_title_2d = config.m_detector_plotnames[det]+" "+config.m_channel_names[channel]+" CV";
                        auto make_slice_title = [&](size_t slice_number, const std::string &slice_variable, float low_edge, float high_edge, const std::string &horizontal_axis) {
                            std::ostringstream title;
                            title << plot_title_2d << " - Slice " << slice_number << ": "
                                  << slice_variable << " in [" << low_edge << ", " << high_edge << ")"
                                  << ";" << horizontal_axis << ";" << ytitle;
                            return title.str();
                        };

                        // this needs to be a sum over all subchannels
                        TH2D* cv_hist = new TH2D(hist_title.c_str(),hist_title.c_str(), channel_nbins_x, edges_x.data(), channel_nbins_y, edges_y.data());
                        // global_subchannel_index still points at this channel's first
                        // subchannel here; it is only advanced in the stack loop below.
                        for(size_t subchannel = 0; subchannel < config.m_num_subchannels[channel]; ++subchannel){
                            const std::string& subchannel_name  = config.m_fullnames[global_subchannel_index + subchannel];
                            cv_hist->Add(cv2dhists[subchannel_name].get());
                        }

                        TPad p2d("p2d", "p2d", 0, 0, 1, 1);
                        p2d.cd();
                        cv_hist->SetTitle(hist_title.c_str());
                        cv_hist->Draw("colz");
                        draw_chi_label(full_chi_label);
                        objs[mdc+"_cv2d"] = cv_hist->Clone();
                        c.cd();
                        p2d.Draw();
                        c.Print(filename.c_str());

                        TH2D* bf_hist = NULL;
                        if(best_fit){
                            std::string bf_hist_title = config.m_detector_plotnames[det]  + " "+ config.m_channel_plotnames[channel]+(posterrband ? " Constrained Best-Fit;" : " Best-Fit;")+xtitle2d+";"+ytitle2d;
                            bf_hist = new TH2D(bf_hist_title.c_str(),bf_hist_title.c_str(), channel_nbins_x, edges_x.data(), channel_nbins_y, edges_y.data());

                            Eigen::VectorXf tmp_bf = CollapseMatrix(config, best_fit->Spec(), other_index);
                            for(size_t xbin = 0; xbin < channel_nbins_x; xbin++){
                                for(size_t ybin = 0; ybin < channel_nbins_y; ybin++) {
                                    const size_t flat_bin = xbin*channel_nbins_y+ybin + tot_offset;
                                    float val = tmp_bf(flat_bin);
                                    // Fold the posterior covariance pull in: the 2D map,
                                    // and every slice/projection taken from it below,
                                    // shows the CONSTRAINED best fit.
                                    if(posterrband) {
                                        float denom = posterrband->error_point(flat_bin);
                                        float f = (denom != 0 && std::isfinite(denom)) ? val/denom : 1.0f;
                                        if(!std::isfinite(f)) f = 1.0f;
                                        val += f*posterrband->center_shift(flat_bin);
                                    }
                                    bf_hist->SetBinContent(xbin+1, ybin+1, val);
                                }
                            }

                            TPad pbfd("pbfd", "pbfd", 0, 0, 1, 1);
                            pbfd.cd();
                            bf_hist->Draw("colz");
                            draw_chi_label(full_chi_label);
                            objs[mdc+"_bestfit2d"] = bf_hist->Clone();
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
                                    data_hist->SetBinError(xbin+1, ybin+1, data->Error()(xbin*channel_nbins_y+ybin+tot_offset));
                                }
                            }
                            TPad pdata("pdata", "pdata", 0, 0, 1, 1);
                            pdata.cd();
                            data_hist->Draw("colz");
                            draw_chi_label(full_chi_label);
                            objs[mdc+"_data2d"] = data_hist->Clone();
                            c.cd();
                            pdata.Draw();
                            c.Print(filename.c_str());
                        }


                        float scale = 1.0;

                        // Plot the first variable in bins of the second variable
                        for(size_t ybin = 1; ybin <= channel_nbins_y; ybin++) {
                            std::vector<std::vector<size_t>> groups(channel_nbins_x);
                            for(size_t xbin = 0; xbin < channel_nbins_x; ++xbin) {
                                groups[xbin] = {xbin*channel_nbins_y + (ybin-1)};
                            }
                            const std::string slice_chi_label = chi_label(make_projection(groups));
                            TGraphAsymmErrors *post_channel_errband = NULL; 
                            if(posterrband) {
                                post_channel_errband = new TGraphAsymmErrors(bf_hist->ProjectionX("xslc_bf", ybin, ybin));
                                for(size_t xbin = 0; xbin < channel_nbins_x; ++xbin) {
                                    const size_t flat_bin = xbin * channel_nbins_y + (ybin - 1) + tot_offset;
                                    // Anchored via the shifted bf_hist projection above.
                                    post_channel_errband->SetPointEYhigh(xbin, scale*posterrband->error_up(flat_bin));
                                    post_channel_errband->SetPointEYlow(xbin, scale*posterrband->error_down(flat_bin));
                                }
                                objs[mdc+"_posterrband_slice_ybin"+std::to_string(ybin)] = post_channel_errband->Clone();
                            }

                            TGraphAsymmErrors* channel_errband = NULL;
                            if(errband) {
                                channel_errband = new TGraphAsymmErrors(cv_hist->ProjectionX("xslc_cv", ybin, ybin));
                                for(size_t xbin = 0; xbin < channel_nbins_x; ++xbin) {
                                    const size_t flat_bin = xbin * channel_nbins_y + (ybin - 1) + tot_offset;
                                    channel_errband->SetPointEYhigh(xbin, scale*errband->error_up(flat_bin));
                                    channel_errband->SetPointEYlow(xbin, scale*errband->error_down(flat_bin));
                                }
                                objs[mdc+"_preerrband_slice_ybin"+std::to_string(ybin)] = channel_errband->Clone();
                            }

                            std::string ybin_str = make_slice_title(ybin, ytitle2d,
                                                                    edges_y[ybin-1], edges_y[ybin],
                                                                    xtitle2d);

                            TH1D cv_hist_slice = *(cv_hist->ProjectionX("slc", ybin, ybin));
                            cv_hist_slice.SetTitle(ybin_str.c_str());
                            cv_hist_slice.GetYaxis()->SetTitle(ytitle.c_str());
                            objs[mdc+"_cv_slice_ybin"+std::to_string(ybin)] = cv_hist_slice.Clone();
                            TH1D *bf_hist_slice = NULL;
                            if(best_fit){
                                bf_hist_slice = bf_hist->ProjectionX("bfslc", ybin, ybin);
                                bf_hist_slice->SetTitle(ybin_str.c_str());
                                bf_hist_slice->GetYaxis()->SetTitle(ytitle.c_str());
                                objs[mdc+"_bestfit_slice_ybin"+std::to_string(ybin)] = bf_hist_slice->Clone();
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
                                objs[mdc+"_data_slice_ybin"+std::to_string(ybin)] = data_hist_slice->Clone();
                            }

                            plot_hist1ds(&c, &cv_hist_slice, channel_errband, {}, {}, bf_hist_slice, post_channel_errband, data_hist_slice, &dat_str, {}, ybin_str, ratio_titles, filename, bounds, slice_chi_label);
                        }

                        // Plot the second variable in bins of the first variable
                        const std::string ratio_titles_y = ";"+ytitle2d+";"+rat_y_title;
                        for(size_t xbin = 1; xbin <= channel_nbins_x; ++xbin) {
                            std::vector<std::vector<size_t>> groups(channel_nbins_y);
                            for(size_t ybin = 0; ybin < channel_nbins_y; ++ybin) {
                                groups[ybin] = {(xbin-1)*channel_nbins_y + ybin};
                            }
                            const std::string slice_chi_label = chi_label(make_projection(groups));
                            TGraphAsymmErrors *post_channel_errband = NULL;
                            if(posterrband) {
                                post_channel_errband = new TGraphAsymmErrors(bf_hist->ProjectionY("yslc_bf", xbin, xbin));
                                for(size_t ybin = 0; ybin < channel_nbins_y; ++ybin) {
                                    const size_t flat_bin = (xbin-1)*channel_nbins_y+ybin+tot_offset;
                                    // Anchored via the shifted bf_hist projection above.
                                    post_channel_errband->SetPointEYhigh(ybin, scale*posterrband->error_up(flat_bin));
                                    post_channel_errband->SetPointEYlow(ybin, scale*posterrband->error_down(flat_bin));
                                }
                                objs[mdc+"_posterrband_slice_xbin"+std::to_string(xbin)] = post_channel_errband->Clone();
                            }

                            TGraphAsymmErrors *channel_errband = NULL;
                            if(errband) {
                                channel_errband = new TGraphAsymmErrors(cv_hist->ProjectionY("yslc_cv", xbin, xbin));
                                for(size_t ybin = 0; ybin < channel_nbins_y; ++ybin) {
                                    const size_t flat_bin = (xbin-1)*channel_nbins_y+ybin+tot_offset;
                                    channel_errband->SetPointEYhigh(ybin, scale*errband->error_up(flat_bin));
                                    channel_errband->SetPointEYlow(ybin, scale*errband->error_down(flat_bin));
                                }
                                objs[mdc+"_preerrband_slice_xbin"+std::to_string(xbin)] = channel_errband->Clone();
                            }

                            std::string xbin_str = make_slice_title(xbin, xtitle2d,
                                                                    edges_x[xbin-1], edges_x[xbin],
                                                                    ytitle2d);
                            TH1D cv_hist_slice = *(cv_hist->ProjectionY("yslc", xbin, xbin));
                            cv_hist_slice.SetTitle(xbin_str.c_str());
                            cv_hist_slice.GetYaxis()->SetTitle(ytitle.c_str());
                            objs[mdc+"_cv_slice_xbin"+std::to_string(xbin)] = cv_hist_slice.Clone();

                            TH1D *bf_hist_slice = NULL;
                            if(best_fit) {
                                bf_hist_slice = bf_hist->ProjectionY("bfyslc", xbin, xbin);
                                bf_hist_slice->SetTitle(xbin_str.c_str());
                                bf_hist_slice->GetYaxis()->SetTitle(ytitle.c_str());
                                objs[mdc+"_bestfit_slice_xbin"+std::to_string(xbin)] = bf_hist_slice->Clone();
                            }

                            TH1D *data_hist_slice = NULL;
                            if(data) {
                                data_hist_slice = data_hist->ProjectionY("datayslc", xbin, xbin);
                                data_hist_slice->SetTitle(dat_str.c_str());
                                data_hist_slice->GetYaxis()->SetTitle(ytitle.c_str());
                                data_hist_slice->SetLineColor(kBlack);
                                data_hist_slice->SetLineWidth(2);
                                data_hist_slice->SetMarkerStyle(kFullCircle);
                                data_hist_slice->SetMarkerColor(kBlack);
                                data_hist_slice->SetMarkerSize(1);
                                objs[mdc+"_data_slice_xbin"+std::to_string(xbin)] = data_hist_slice->Clone();
                            }

                            plot_hist1ds(&c, &cv_hist_slice, channel_errband, {}, {}, bf_hist_slice, post_channel_errband, data_hist_slice, &dat_str, {}, xbin_str, ratio_titles_y, filename, bounds, slice_chi_label);
                        }

                        const std::string hist_title_y = config.m_mode_plotnames[mode]+" "+config.m_detector_plotnames[det]+" "+config.m_channel_names[channel]+";"+ytitle2d+";"+ytitle;
                        TH1D cv_hist_y(("cv_y"+std::to_string(global_channel_index)).c_str(), hist_title_y.c_str(), channel_nbins_y, edges_y.data());
                        cv_hist_y.SetLineWidth(2);
                        cv_hist_y.SetFillStyle(0);

                        THStack *cvstack_y = NULL;
                        std::vector<std::pair<std::string, const char*>> subplots_y;
                        std::vector<std::unique_ptr<TH1D>> cv_subchannel_hists_y;
                        if(cv) {
                            if(bool(opt&PlotOptions::CVasStack))
                                cvstack_y = new THStack(("y"+std::to_string(global_channel_index)).c_str(), "");

                            // global_subchannel_index still points at this channel's first
                            // subchannel; it is only advanced in the 1D stack loop below.
                            const size_t first_subchannel = global_subchannel_index;
                            for(size_t subchannel = 0; subchannel < config.m_num_subchannels[channel]; ++subchannel) {
                                const size_t global_index = first_subchannel+subchannel;
                                const std::string &subchannel_name = config.m_fullnames[global_index];
                                auto sub_hist = std::make_unique<TH1D>(cv->toTH1D(config, global_index, other_index, 1));
                                sub_hist->SetDirectory(nullptr);
                                sub_hist->SetLineWidth(1);
                                sub_hist->SetLineColor(kBlack);
                                const std::string &color = config.m_subchannel_colors[channel][subchannel];
                                sub_hist->SetFillColor(color == "NONE" ? kRed-7 : config.HexToROOTColor(color));
                                if(bool(opt&PlotOptions::BinWidthScaled)) sub_hist->Scale(1, "width");

                                const bool skip_stack = skip_stack_subchannels &&
                                    std::find(skip_stack_subchannels->begin(), skip_stack_subchannels->end(), global_index) != skip_stack_subchannels->end();
                                if(cvstack_y && !skip_stack) {
                                    cvstack_y->Add(sub_hist.get());
                                    subplots_y.push_back({subchannel_name, config.m_subchannel_plotnames[channel][subchannel].c_str()});
                                }
                                cv_hist_y.Add(sub_hist.get());
                                cv_subchannel_hists_y.push_back(std::move(sub_hist));
                            }
                            if(bool(opt&PlotOptions::AreaNormalized)) {
                                const float integral = cv_hist_y.Integral();
                                cv_hist_y.Scale(1/integral);
                                if(cvstack_y) {
                                    TList *stack_hists = cvstack_y->GetHists();
                                    for(const auto &&obj: *stack_hists) ((TH1*)obj)->Scale(1/integral);
                                }
                            }
                            objs[mdc+"_cv1d_y"] = cv_hist_y.Clone();
                            if(cvstack_y) objs[mdc+"_cvstack_y"] = cvstack_y->Clone();
                        }

                        TH1D *bf_hist_y = best_fit ? bf_hist->ProjectionY("bf_integrated_x") : NULL;
                        if(bf_hist_y) {
                            if(bool(opt&PlotOptions::BinWidthScaled)) bf_hist_y->Scale(1, "width");
                            if(bool(opt&PlotOptions::AreaNormalized)) bf_hist_y->Scale(1/bf_hist_y->Integral());
                            objs[mdc+"_bestfit_y"] = bf_hist_y->Clone();
                        }
                        TH1D *data_hist_y = data ? data_hist->ProjectionY("data_integrated_x") : NULL;
                        if(data_hist_y) {
                            data_hist_y->SetLineColor(kBlack);
                            data_hist_y->SetLineWidth(2);
                            data_hist_y->SetMarkerStyle(kFullCircle);
                            data_hist_y->SetMarkerColor(kBlack);
                            data_hist_y->SetMarkerSize(1);
                            if(bool(opt&PlotOptions::BinWidthScaled)) data_hist_y->Scale(1, "width");
                            if(bool(opt&PlotOptions::AreaNormalized)) data_hist_y->Scale(1/data_hist_y->Integral());
                            objs[mdc+"_data_y"] = data_hist_y->Clone();
                        }

                        auto make_y_errorband = [&](const PROerrorbar &band, TH1D *central) {
                            // The central hist is projected from the (already shifted)
                            // bf_hist for the post band, so the graph is born anchored
                            // at the constrained best fit; only widths are set here.
                            TGraphAsymmErrors *graph = new TGraphAsymmErrors(central);
                            for(size_t ybin = 0; ybin < channel_nbins_y; ++ybin) {
                                double variance = 0.0;
                                for(size_t x1 = 0; x1 < channel_nbins_x; ++x1)
                                    for(size_t x2 = 0; x2 < channel_nbins_x; ++x2)
                                        variance += band.covariance(tot_offset+x1*channel_nbins_y+ybin,
                                                                    tot_offset+x2*channel_nbins_y+ybin);
                                double error = std::sqrt(std::max(0.0, variance));
                                if(bool(opt&PlotOptions::AreaNormalized) || bool(opt&PlotOptions::BinWidthScaled)) {
                                    // central is shifted, so the ebar-units reference is
                                    // error_point + center_shift: keeps the conversion
                                    // factor free of the pull.
                                    double point = 0.0;
                                    for(size_t xbin = 0; xbin < channel_nbins_x; ++xbin)
                                        point += band.error_point(tot_offset+xbin*channel_nbins_y+ybin)
                                               + band.center_shift(tot_offset+xbin*channel_nbins_y+ybin);
                                    if(point != 0.0 && std::isfinite(point)) error *= central->GetBinContent(ybin+1)/point;
                                }
                                graph->SetPointEYhigh(ybin, error);
                                graph->SetPointEYlow(ybin, error);
                            }
                            return graph;
                        };
                        TGraphAsymmErrors *channel_errband_y = errband ? make_y_errorband(*errband, &cv_hist_y) : NULL;
                        TGraphAsymmErrors *post_channel_errband_y = posterrband ? make_y_errorband(*posterrband, bf_hist_y) : NULL;
                        if(channel_errband_y) objs[mdc+"_preerrband_y"] = channel_errband_y->Clone();
                        if(post_channel_errband_y) objs[mdc+"_posterrband_y"] = post_channel_errband_y->Clone();

                        std::vector<std::vector<size_t>> y_projection_groups(channel_nbins_y);
                        for(size_t ybin = 0; ybin < channel_nbins_y; ++ybin) {
                            for(size_t xbin = 0; xbin < channel_nbins_x; ++xbin) {
                                y_projection_groups[ybin].push_back(xbin*channel_nbins_y + ybin);
                            }
                        }
                        const std::string y_chi_label = chi_label(make_projection(y_projection_groups));
                        plot_hist1ds(&c, &cv_hist_y, channel_errband_y, cvstack_y, &subplots_y, bf_hist_y, post_channel_errband_y, data_hist_y, &dat_str, opt, hist_title_y, ratio_titles_y, filename, bounds, y_chi_label);
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
                            // Background-subtracted subchannels are dropped from the
                            // stack and legend (their contents are already zero); the
                            // cv_hist sum still includes them so bookkeeping is unchanged.
                            const bool skip_stack = skip_stack_subchannels &&
                                std::find(skip_stack_subchannels->begin(), skip_stack_subchannels->end(),
                                          global_subchannel_index) != skip_stack_subchannels->end();
                            if(bool(opt&PlotOptions::CVasStack) && !skip_stack) {
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
                        objs[mdc+"_cv1d"] = cv_hist.Clone();
                        if(cvstack) objs[mdc+"_cvstack"] = cvstack->Clone();
                    }

                    TGraphAsymmErrors *channel_errband = NULL;
                    if(errband) {
                        channel_errband = new TGraphAsymmErrors(&cv_hist);

                        for(size_t bin = 0; bin < channel_nbins_x; ++bin) {
                            float scale = 1.0;
                            if(bool(opt&PlotOptions::AreaNormalized) || bool(opt&PlotOptions::BinWidthScaled)) {
                                // Guard 0/0 when the (possibly bkg-subtracted) CV is zero
                                // in a bin; error_point is already bin-width scaled, so
                                // 1.0 is the exact fallback in the BinWidthScaled case.
                                float denom = errband_1d->error_point(bin);
                                scale = (denom != 0 && std::isfinite(denom)) ? channel_errband->GetPointY(bin) / denom : 1.0f;
                                if(!std::isfinite(scale)) scale = 1.0f;
                            }

                            channel_errband->SetPointEYhigh(bin, scale*(errband_1d->error_up(bin)));
                            channel_errband->SetPointEYlow(bin, scale*(errband_1d->error_down(bin)));
                        }
                        objs[mdc+"_preerrband"] = channel_errband->Clone();
                    }

                    TH1D* bf_hist = NULL;
                    // ebar-units -> drawn-units conversion per bin, computed from the
                    // UNSHIFTED curve (the shift below must not contaminate it).
                    std::vector<float> post_conv;
                    if(best_fit) {
                        bf_hist = new TH1D(("bf"+std::to_string(global_channel_index)).c_str(), "", channel_nbins_x, edges.data());
                        for(size_t bin = 0; bin < channel_nbins_x; ++bin) {
                            bf_hist->SetBinContent(bin+1, bf_spec_1d(bin));
                        }
                        if(bool(opt&PlotOptions::BinWidthScaled))
                            bf_hist->Scale(1, "width");
                        if(bool(opt&PlotOptions::AreaNormalized))
                            bf_hist->Scale(1.0/bf_hist->Integral());
                        // Fold the posterior covariance pull into the drawn curve:
                        // this is the CONSTRAINED best fit (spline best fit + data
                        // constraint on the covariance systematics), the same center
                        // the band is measured about. Zero shift for prior bands.
                        if(posterrband) {
                            post_conv.assign(channel_nbins_x, 1.0f);
                            for(size_t bin = 0; bin < channel_nbins_x; ++bin) {
                                if(bool(opt&PlotOptions::AreaNormalized) || bool(opt&PlotOptions::BinWidthScaled)) {
                                    float denom = posterrband_1d->error_point(bin);
                                    float f = (denom != 0 && std::isfinite(denom)) ? bf_hist->GetBinContent(bin+1)/denom : 1.0f;
                                    if(!std::isfinite(f)) f = 1.0f;
                                    post_conv[bin] = f;
                                }
                                bf_hist->SetBinContent(bin+1, bf_hist->GetBinContent(bin+1) + post_conv[bin]*posterrband_1d->center_shift(bin));
                            }
                        }
                        objs[mdc+"_bestfit"] = bf_hist->Clone();
                    }

                    TH1D* data_hist = NULL;
                    if(data) {
                        data_hist = new TH1D(("data"+std::to_string(global_channel_index)).c_str(), "", channel_nbins_x, edges.data());
                        for(size_t bin = 0; bin < channel_nbins_x; ++bin) {
                            data_hist->SetBinContent(bin+1, data_spec_1d(bin));
                            // SetBinError allocates Sumw2, so the Scale calls below
                            // transform the errors correctly too. Without this ROOT
                            // falls back to sqrt(|content|), which is wrong for
                            // bin-width-scaled or background-subtracted data.
                            data_hist->SetBinError(bin+1, data_err_1d(bin));
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
                        objs[mdc+"_data"] = data_hist->Clone();
                    }

                    TGraphAsymmErrors *post_channel_errband = NULL;
                    if(posterrband) {
                        // bf_hist already IS the constrained best fit (shift folded in
                        // above), so the graph is born anchored at the band center;
                        // only the widths need the unit conversion, taken from the
                        // pre-shift factors so the shift does not contaminate them.
                        post_channel_errband = new TGraphAsymmErrors(bf_hist);
                        for(size_t bin = 0; bin < channel_nbins_x; ++bin) {
                            float scale = post_conv.empty() ? 1.0f : post_conv[bin];
                            post_channel_errband->SetPointEYhigh(bin, scale*(posterrband_1d->error_up(bin)));
                            post_channel_errband->SetPointEYlow(bin, scale*(posterrband_1d->error_down(bin)));
                        }
                        objs[mdc+"_posterrband"] = post_channel_errband->Clone();
                    }

                    std::string chi_label_text;
                    if(config.m_channel_variable_dims[channel][other_index] == 2 && !projected_x_chi_label.empty()) {
                        chi_label_text = projected_x_chi_label;
                    } else if(texts.size()!=0) {
                        TPaveText &box = texts.size() == 1 ? texts.front() : texts.at(global_channel_index);
                        if(TText *line = (TText*)box.GetListOfLines()->First()) chi_label_text = line->GetTitle();
                    }
                    // should probably be switching this to a more clear boolean...
                    plot_hist1ds(&c, &cv_hist, channel_errband, cvstack, &subplots, bf_hist, post_channel_errband, data_hist, &dat_str, opt, hist_titles, ratio_titles, filename, bounds, chi_label_text);
                    ++global_channel_index;
                    cv_hists.push_back(cv_hist);
		    if(data){
                        data_hists.push_back(*data_hist);
                        if(posterrband) bf_hists.push_back(*bf_hist);
                    }
                    tot_offset += nbins_p_2dchan;
                }
            }
        }
        
        // plot ratio across detectors and/or across channels
        if(cv && (config.m_num_detectors > 1 || (plot_channel_ratios && config.m_num_channels > 1))) {
            Eigen::VectorXf cv_coll = CollapseMatrix(config, cv->Spec(), other_index);
            std::optional<Eigen::VectorXf> bf_coll, data_coll;
            if(best_fit) bf_coll = CollapseMatrix(config, best_fit->Spec(), other_index);
            if(data)     data_coll = data->Spec();
            if(config.m_num_detectors > 1)
                plot_detector_ratio_spectra(c, config, cv_coll, bf_coll, data_coll,
                                            errband, posterrband, channel_offsets, filename, other_index);
            if(plot_channel_ratios && config.m_num_channels > 1)
                plot_channel_ratio_spectra(c, config, cv_coll, bf_coll, data_coll,
                                           errband, posterrband, channel_offsets, filename, other_index, opt);
        }

        c.Print((filename+"]").c_str());

        log<LOG_DEBUG>(L"%1% || ratio_bool: %2%") % __func__ % ratio_bool;
        if(ratio_bool && data_hists.size() >= 2){

	    //std::map<std::string, std::unique_ptr<TH2D>> pre_matrices = covarianceTH2D(errband->covariance, config, *cv);
	    //std::map<std::string, std::unique_ptr<TH2D>> post_matrices = covarianceTH2D(posterrband->covariance, config, *best_fit);

	    std::unique_ptr<TH2D> corr_coll = std::make_unique<TH2D>("ccor", "Collapsed Correlation Matrix;Bin # ;Bin #", config.m_num_variable_bins_total_collapsed[config.i_prime], 0,
			                                             config.m_num_variable_bins_total_collapsed[config.i_prime], config.m_num_variable_bins_total_collapsed[config.i_prime],
								     0, config.m_num_variable_bins_total_collapsed[config.i_prime]);

	    std::unique_ptr<TH2D> post_corr_coll = std::make_unique<TH2D>("ccor", "Collapsed Correlation Matrix;Bin # ;Bin #", config.m_num_variable_bins_total_collapsed[config.i_prime], 0,
			                                                  config.m_num_variable_bins_total_collapsed[config.i_prime], config.m_num_variable_bins_total_collapsed[config.i_prime],
									  0, config.m_num_variable_bins_total_collapsed[config.i_prime]);

	    Eigen::MatrixXf cov_coll = errband->covariance;
	    Eigen::MatrixXf post_cov_coll = posterrband->covariance;

            log<LOG_DEBUG>(L"%1% || Converting covariance to correlation") % __func__;
	    for(int i = 0; i < cov_coll.rows(); i++){
	        for(int j = 0; j < cov_coll.rows(); j++){
		    corr_coll->SetBinContent(i+1, j+1, cov_coll(i, j)/cov_coll(i, i)/cov_coll(j, j));
		    post_corr_coll->SetBinContent(i+1, j+1, post_cov_coll(i, j)/post_cov_coll(i, i)/post_cov_coll(j, j));
		}
	    }

            plot_detector_ratios(config, data_hists, cv_hists, errband, bf_hists, posterrband, *corr_coll, *post_corr_coll,  filename, other_index);
        }


        log<LOG_DEBUG>(L"%1% || Finishing plot_channels") % __func__;
        return objs;
    }


    int plotPriorFractionalSystematicBreakdown(const PROconfig &config, const PROspec &spec, const PROsyst &allsplinesyst, std::string filename, int other_index) {
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

        TCanvas c("c", "Systematics Comparison", 1600, 1200);  
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
        for (Eigen::Index i = 0; i < collapsed_diag.rows(); ++i) {
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

                            for(int i = 0; i < (int)channel_nbins_x; i++){
                                for(int j = (int)channel_nbins_y*i; j < (int)channel_nbins_y*(i+1); j++){
                                diag1d(i) += channel_diag(j, j);
                                    for(int k = (int)channel_nbins_y*i; k < (int)channel_nbins_y*(i+1); k++){
                                        VarVec(i) += channel_cov(j, k);
                                    }
                                }
                            }

                            float inv_diag1d;
                            for (size_t i = 0; i < channel_nbins_x; ++i) {
                                inv_diag1d = 1/diag1d(i);
                                // clamp to >=0: covariance diagonals can be tiny-negative from float cancellation
                                // in rat_frac_cov = Cov(d1,d1) + Cov(d2,d2) − Cov(d1,d2) − Cov(d2,d1)
                                float var_i = std::max(0.0f, inv_diag1d*VarVec(i)*inv_diag1d);
                                h->SetBinContent(i+1, sqrt(var_i));
                                hsum->SetBinContent(i+1, hsum->GetBinContent(i+1)+var_i);
                            }

                            const std::string &plotname = config.m_mcgen_variation_plotname_map.at(systname);
                            leg->AddEntry(h, plotname.c_str(), "l");
                            h->SetLineColor(colors[color_idx]);
                            h->SetLineStyle(line_styles[style_idx]);
                            hvec.push_back(h);

                        }//end syst
                        for (size_t i = 0; i < channel_nbins_x; ++i) {
                            hsum->SetBinContent(i+1, sqrt(std::max(0.0, hsum->GetBinContent(i+1))));
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

                        hsum->SetXTitle(config.GetChannelAxisTitle(channel, other_index, 0).c_str());
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
                    hsum->SetXTitle(config.GetChannelAxisTitle(channel, other_index, 0).c_str());
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

    int plotPriorFractionalSystematicRatios(const PROconfig &config, const PROspec &spec, const PROsyst &allsplinesyst, std::string filename, int other_index) {
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

        TCanvas c("c", "Systematics Comparison", 1600, 1200);  
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
        for (Eigen::Index i = 0; i < collapsed_diag.rows(); ++i) {
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
                for(size_t det2 = det+1; det2 < config.m_num_detectors; ++det2) {
                    for(size_t channel = 0; channel < config.m_num_channels; ++channel) {
                        size_t global_channel_index2 = global_channel_index + (det2 - det) * config.m_num_channels;

                        std::string name = config.m_mode_plotnames[mode]+" "+config.m_detector_plotnames[det]+"/"+config.m_detector_plotnames[det2]+" "+config.m_channel_plotnames[channel]; 
                        c.Clear();
                        c.Divide(gridCols, gridRows);

                        int padIndex = 1;

                        std::vector<float> bin_edges = config.GetChannelVariableBins(global_channel_index,other_index).Edges();
                        size_t binstart = config.GetCollapsedGlobalVariableBinStart(global_channel_index,other_index);
                        size_t binstart2 = config.GetCollapsedGlobalVariableBinStart(global_channel_index2,other_index);
                        size_t nbins = config.m_channel_variable_bins[channel][other_index].NBins();
                        std::vector<int> channel_bins(nbins);
                        std::iota(channel_bins.begin(), channel_bins.end(), binstart);
                        std::vector<int> channel_bins2(nbins);
                        std::iota(channel_bins2.begin(), channel_bins2.end(), binstart2);

                        // Diagnostic: Check bin_edges validity
                        log<LOG_INFO>(L"%1% || Channel %2%: bin_edges.size()=%3%, nbins=%4%, binstart=%5%")
                            % __func__ % global_channel_index % bin_edges.size() % nbins % binstart;
                        log<LOG_INFO>(L"%1% || Channel %2%: bin_edges.size()=%3%, nbins=%4%, binstart=%5%")
                            % __func__ % global_channel_index2 % bin_edges.size() % nbins % binstart2;
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
                                Eigen::MatrixXf rat_frac_cov = collapsed_frac_covariance(channel_bins, channel_bins) + collapsed_frac_covariance(channel_bins2, channel_bins2) - collapsed_frac_covariance(channel_bins, channel_bins2) - collapsed_frac_covariance(channel_bins2, channel_bins);

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
                                TH1F* h = new TH1F((tag+"_Channel_"+std::to_string(global_channel_index)+"_"+std::to_string(i)).c_str(), (tag + ";" + config.GetChannelXAxisTitle(channel, other_index)).c_str(), bin_edges.size()-1, bin_edges.data());

                                if(config.m_channel_variable_dims[channel][other_index] == 2) {
                                    channel_nbins_y = config.m_channel_variable_bins[channel][other_index].NBinsAlong(1);
                                    Eigen::MatrixXf rat_diag = collapsed_diag(channel_bins, channel_bins) + collapsed_diag(channel_bins2, channel_bins2);
                                    Eigen::MatrixXf rat_full_cov = rat_diag * rat_frac_cov * rat_diag;
                                    Eigen::VectorXf VarVec = Eigen::VectorXf::Zero(channel_nbins_x);
                                    Eigen::VectorXf diag1d = Eigen::VectorXf::Zero(channel_nbins_x);
                                    for(int i = 0; i < (int)channel_nbins_x; i++){
                                        for(int j = (int)channel_nbins_y*i; j < (int)channel_nbins_y*(i+1); j++){
                                            diag1d(i) += rat_diag(j, j);
                                            for(int k = (int)channel_nbins_y*i; k < (int)channel_nbins_y*(i+1); k++){
                                                VarVec(i) += rat_full_cov(j, k);
                                            }
                                        }
                                    }

                                    float inv_diag1d;
                                    for (size_t i = 0; i < channel_nbins_x; ++i) {
                                        inv_diag1d = 1/diag1d(i);
                                        float var_i = std::max(0.0f, inv_diag1d*VarVec(i)*inv_diag1d);
                                        h->SetBinContent(i+1, sqrt(var_i));
                                        hsum->SetBinContent(i+1, hsum->GetBinContent(i+1)+var_i);
                                    }
                                } else {
                                    for(size_t i = 0; i < channel_bins.size(); ++i) {
                                        float var_i = std::max(0.0f, rat_frac_cov(i,i));
                                        h->SetBinContent(i+1, sqrt(var_i));
                                        hsum->SetBinContent(i+1, hsum->GetBinContent(i+1)+var_i);
                                    }
                                }

                                const std::string &plotname = config.m_mcgen_variation_plotname_map.at(systname);
                                leg->AddEntry(h, plotname.c_str(), "l");
                                h->SetLineColor(colors[color_idx]);
                                h->SetLineStyle(line_styles[style_idx]);
                                hvec.push_back(h);

                            }//end syst
                            for (size_t i = 0; i < channel_nbins_x; ++i) {
                                hsum->SetBinContent(i+1, sqrt(std::max(0.0, hsum->GetBinContent(i+1))));
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

                            hsum->SetXTitle((config.m_detector_plotnames[det]+"/"+config.m_detector_plotnames[det2]+" "+config.GetChannelXAxisTitle(channel, other_index)).c_str());
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
                        hsum->SetXTitle((config.m_detector_plotnames[det]+"/"+config.m_detector_plotnames[det2]+" "+config.GetChannelXAxisTitle(channel, other_index)).c_str());
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
                    global_channel_index -= config.m_num_channels;
                }
                global_channel_index += config.m_num_channels;
            }
        }

        c.Print((filename+"]").c_str());
        return 0;
    };

int plotPriorFractionalSystematicChannelRatios(const PROconfig &config, const PROspec &spec, const PROsyst &allsplinesyst, std::string filename, int other_index) {
        //Input PROsyst needs to be the allsplinesyst for now

        if(config.m_num_channels < 2) {
            log<LOG_WARNING>(L"%1% || Need at least 2 channels for channel-ratio systematics. Found %2%.")
                % __func__ % config.m_num_channels;
            return 1;
        }

        std::vector<int> colors = {
            kAzure+1, kRed+1, kGreen+3, kOrange+7, kBlue+2, kViolet+2,
            kGray+1, kYellow+2, kTeal+3, kPink+2, kMagenta+2, kSpring+5
        };
        std::vector<int> line_styles = {1, 1};

        std::map<std::string,std::vector<std::string>> used_tags;
        for(const auto &name: allsplinesyst.covar_names){
            auto it = config.m_mcgen_variation_tags.find(name);
            if (it == config.m_mcgen_variation_tags.end()) {
                log<LOG_WARNING>(L"%1% || Systematic %2% not in tags map") % __func__ % name.c_str();
                continue;
            }
            for(auto &t: it->second) used_tags[t].push_back(name);
        }

        int nTags = used_tags.size()+1;
        int gridCols = std::ceil(std::sqrt(nTags));
        int gridRows = std::ceil(nTags / float(gridCols));

        TCanvas c("c", "Channel Ratio Systematics", 1600, 1200);
        c.Print((filename+"[").c_str());
        c.Divide(gridCols, gridRows);

        Eigen::MatrixXf diag = spec.Spec().array().matrix().asDiagonal();
        // NOTE: the matrix CollapseMatrix overload takes no variable index, so this
        // (like plotPriorFractionalSystematicRatios) is only valid for i_prime.
        Eigen::VectorXf collapsed_cv = CollapseMatrix(config, spec.Spec(), other_index);

        auto sum_vec = [](const Eigen::VectorXf &v, size_t off, size_t bx, size_t ny) {
            float s = 0.0f;
            for(size_t by = 0; by < ny; ++by) s += v(off + bx*ny + by);
            return s;
        };
        auto sum_block = [](const Eigen::MatrixXf &m, size_t offr, size_t ny_r,
                            size_t offc, size_t ny_c, size_t bx) {
            float s = 0.0f;
            for(size_t i = 0; i < ny_r; ++i)
                for(size_t j = 0; j < ny_c; ++j)
                    s += m(offr + bx*ny_r + i, offc + bx*ny_c + j);
            return s;
        };
        auto same_edges = [](const std::vector<float> &a, const std::vector<float> &b) {
            if(a.size() != b.size()) return false;
            for(size_t i = 0; i < a.size(); ++i) {
                const float sc = std::max(1.0f, std::max(std::fabs(a[i]), std::fabs(b[i])));
                if(std::fabs(a[i] - b[i]) > 1e-4f*sc) return false;
            }
            return true;
        };

        for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
        for(size_t det = 0; det < config.m_num_detectors; ++det) {
        for(size_t ch1 = 0; ch1 < config.m_num_channels; ++ch1) {
        for(size_t ch2 = ch1 + 1; ch2 < config.m_num_channels; ++ch2) {

            const size_t gidx1 = (mode*config.m_num_detectors + det)*config.m_num_channels + ch1;
            const size_t gidx2 = (mode*config.m_num_detectors + det)*config.m_num_channels + ch2;

            const size_t nx   = config.m_channel_variable_bins[ch1][other_index].NBinsAlong(0);
            const size_t nx_d = config.m_channel_variable_bins[ch2][other_index].NBinsAlong(0);
            std::vector<float> bin_edges = config.m_channel_variable_bins[ch1][other_index].Edges(0);
            std::vector<float> edges_d   = config.m_channel_variable_bins[ch2][other_index].Edges(0);
            if(nx != nx_d || !same_edges(bin_edges, edges_d)) {
                log<LOG_WARNING>(L"%1% || Skipping channel ratio %2%/%3%: x binning differs (%4% vs %5% bins).")
                    % __func__ % config.m_channel_names[ch1].c_str() % config.m_channel_names[ch2].c_str() % nx % nx_d;
                continue;
            }

            const size_t ny1 = config.m_channel_variable_dims[ch1][other_index] == 2
                             ? config.m_channel_variable_bins[ch1][other_index].NBinsAlong(1) : 1;
            const size_t ny2 = config.m_channel_variable_dims[ch2][other_index] == 2
                             ? config.m_channel_variable_bins[ch2][other_index].NBinsAlong(1) : 1;

            const size_t off1 = config.GetCollapsedGlobalVariableBinStart(gidx1, other_index);
            const size_t off2 = config.GetCollapsedGlobalVariableBinStart(gidx2, other_index);

            const std::string ratname = config.m_channel_plotnames[ch1] + "/" + config.m_channel_plotnames[ch2];
            const std::string name    = config.m_mode_plotnames[mode] + " " + config.m_detector_plotnames[det] + " " + ratname;
            const std::string xtitle  = ratname + " " + config.GetChannelXAxisTitle(ch1, other_index);
            const std::string sfx     = "_" + std::to_string(mode) + "_" + std::to_string(det)
                                      + "_" + std::to_string(ch1) + "_" + std::to_string(ch2);

            c.Clear();
            c.Divide(gridCols, gridRows);
            int padIndex = 1;

            std::vector<TH1F*> vsums;
            std::vector<std::string> vnames;

            for (const auto &[tag, vec] : used_tags) {

                c.cd(padIndex++);
                TLegend* leg = new TLegend(0.11, 0.6, 0.89, 0.89);
                leg->SetNColumns(3);

                TH1F* hsum = new TH1F(("Sum_"+tag+sfx).c_str(), tag.c_str(), bin_edges.size()-1, bin_edges.data());
                hsum->Reset();
                std::vector<TH1F*> hvec;
                int i = 0;

                for(const auto &systname : vec){

                    Eigen::MatrixXf frac_covariance = allsplinesyst.GrabMatrix(systname);
                    Eigen::MatrixXf full_covariance = diag*frac_covariance*diag;
                    Eigen::MatrixXf C = CollapseMatrix(config, full_covariance);

                    int color_idx = i % colors.size();
                    int style_idx = (i / 4) % line_styles.size();
                    i++;
                    TH1F* h = new TH1F((tag+"_ChanRat"+sfx+"_"+std::to_string(i)).c_str(),
                                       (tag + ";" + xtitle).c_str(), bin_edges.size()-1, bin_edges.data());

                    for(size_t bx = 0; bx < nx; ++bx) {
                        const float a = sum_vec(collapsed_cv, off1, bx, ny1);
                        const float b = sum_vec(collapsed_cv, off2, bx, ny2);
                        float var_i = 0.0f;
                        if(a != 0.0f && b != 0.0f) {
                            const float caa = sum_block(C, off1, ny1, off1, ny1, bx);
                            const float cbb = sum_block(C, off2, ny2, off2, ny2, bx);
                            const float cab = sum_block(C, off1, ny1, off2, ny2, bx);
                            var_i = std::max(0.0f, caa/(a*a) + cbb/(b*b) - 2.0f*cab/(a*b));
                        }
                        h->SetBinContent(bx+1, std::sqrt(var_i));
                        hsum->SetBinContent(bx+1, hsum->GetBinContent(bx+1) + var_i);
                    }

                    const std::string &plotname = config.m_mcgen_variation_plotname_map.at(systname);
                    leg->AddEntry(h, plotname.c_str(), "l");
                    h->SetLineColor(colors[color_idx]);
                    h->SetLineStyle(line_styles[style_idx]);
                    hvec.push_back(h);

                }//end syst

                for (size_t bx = 0; bx < nx; ++bx)
                    hsum->SetBinContent(bx+1, std::sqrt(std::max(0.0, hsum->GetBinContent(bx+1))));
                leg->AddEntry(hsum, "Sum", "l");

                hsum->SetXTitle(xtitle.c_str());
                hsum->SetYTitle("Fractional Uncertainty on Ratio");
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
                for(auto &h : hvec) h->Draw("HIST SAME");
                leg->Draw();

                TText *t = new TText();
                t->SetNDC(); t->SetTextFont(42); t->SetTextSize(0.03); t->SetTextAlign(33);
                std::string pv = "PROfit v"+std::string(PROJECT_VERSION_STR);
                t->DrawText(0.895, 0.945, pv.c_str());

            }//end tag

            //and each sum of sums to wrap it off!
            c.cd(padIndex++);

            TH1F* hsum = new TH1F(("USum"+sfx).c_str(), ("Summary! "+name).c_str(), bin_edges.size()-1, bin_edges.data());
            hsum->Reset();
            TLegend* leg = new TLegend(0.11, 0.6, 0.89, 0.89);
            leg->SetNColumns(3);
            std::vector<TH1F*> hvec;
            for(size_t t = 0; t < vsums.size(); t++){
                int color_idx = t % colors.size();
                for (size_t bx = 0; bx < nx; ++bx)
                    hsum->SetBinContent(bx+1, hsum->GetBinContent(bx+1)+pow(vsums.at(t)->GetBinContent(bx+1),2));
                TH1F *h = (TH1F*)vsums.at(t)->Clone((vnames[t]+sfx).c_str());
                leg->AddEntry(h, vnames[t].c_str(), "l");
                h->SetLineColor(colors[color_idx]);
                h->SetLineStyle(1);
                h->SetLineWidth(1);
                hvec.push_back(h);
            }
            for (size_t bx = 0; bx < nx; ++bx)
                hsum->SetBinContent(bx+1, std::sqrt(hsum->GetBinContent(bx+1)));
            leg->AddEntry(hsum, "Sum", "l");

            hsum->SetXTitle(xtitle.c_str());
            hsum->SetTitle(("Summary: "+name).c_str());
            hsum->SetYTitle("Fractional Uncertainty on Ratio");
            hsum->SetLineColor(kBlack);
            hsum->SetLineWidth(2);
            hsum->SetLineStyle(1);
            hsum->SetMinimum(0);
            hsum->SetStats(0);
            hsum->Draw("HIST");
            hsum->SetMaximum(hsum->GetMaximum()*1.7);
            gPad->Modified();
            gPad->Update();
            for(auto &h : hvec) h->Draw("HIST SAME");
            leg->Draw();

            TText *t = new TText();
            t->SetNDC(); t->SetTextFont(42); t->SetTextSize(0.03); t->SetTextAlign(33);
            std::string pv = "PROfit v"+std::string(PROJECT_VERSION_STR);
            t->DrawText(0.895, 0.945, pv.c_str());

            c.Update();
            c.Print(filename.c_str());
        }}}}

        c.Print((filename+"]").c_str());
        return 0;
    };

    void plot_mcmc_1sigma(const std::string &filename, const PROconfig &config, const PROsyst &systs, const PROmodel &model, const Eigen::VectorXf &best_fit, const Eigen::VectorXf &param_err_lo, const Eigen::VectorXf &param_err_hi, bool with_osc, const Eigen::VectorXf &true_params) {
        const int nBins = (int)systs.GetNSplines() + (with_osc ? (int)model.nparams : 0);
        if(nBins == 0) {
            log<LOG_WARNING>(L"%1% || No parameters to plot, skipping _1sigmaMCMC.pdf") % __func__;
            return;
        }

        std::vector<std::string> names;
        if(with_osc) for(const auto &n: model.pretty_param_names) names.push_back(n);
        for(const auto &n: systs.spline_names) names.push_back(n);

        std::vector<float> bfvalues(nBins, 0.0f);
        for(int i = 0; i < nBins; ++i) {
            int vec_idx = with_osc ? i : (i + (int)model.nparams);
            if(vec_idx < (int)best_fit.size()) bfvalues[i] = (float)best_fit(vec_idx);
        }

        // Y range: cover post-fit bars, always show ±1
        float minVal = -1.2f, maxVal = 1.2f;
        for(int i = 0; i < nBins; ++i) {
            int syst_idx = with_osc ? (i - (int)model.nparams) : i;
            if(syst_idx >= 0 && syst_idx < (int)param_err_lo.size() && syst_idx < (int)param_err_hi.size()) {
                minVal = std::min(minVal, bfvalues[i] - param_err_lo(syst_idx));
                maxVal = std::max(maxVal, bfvalues[i] + param_err_hi(syst_idx));
            }
        }
        // Clamp to ±5 so a runaway bar doesn't squash everything
        minVal = std::max(minVal, -5.0f);
        maxVal = std::min(maxVal,  5.0f);
        const float y_axis_min = minVal * 1.15f;
        const float y_axis_max = maxVal * 1.15f;
        const float y_range_size = y_axis_max - y_axis_min;
        const float arrow_margin = y_range_size * 0.07f;
        const float arrow_length = y_range_size * 0.05f;

        const int c_width  = std::max(600, std::min(5000, 50 * nBins));
        const int c_height = 500;
        const float axis_label_size = std::max(0.030f, std::min(0.045f, 1.8f / nBins));
        const float x_label_size    = std::max(0.015f, std::min(0.030f, 1.2f / nBins));
        const float bar_halfwidth   = std::max(0.08f, std::min(0.4f,  4.0f / nBins));
        const float marker_offset   = bar_halfwidth * 0.6f;
        const float marker_size     = std::max(0.5f, std::min(1.4f, 6.0f / std::sqrt((float)nBins)));

        TCanvas *c = new TCanvas((filename+"1sigmaMCMC").c_str(), (filename+"1sigmaMCMC").c_str(), c_width, c_height);
        c->cd();
        c->SetLeftMargin(0.09);
        c->SetBottomMargin(0.30);
        c->SetRightMargin(0.28);
        c->SetTopMargin(0.08);

        TH1F *frame = new TH1F((filename+"_frame_mcmc1s").c_str(), "", nBins, 0, nBins);
        frame->SetMinimum(y_axis_min);
        frame->SetMaximum(y_axis_max);
        frame->SetStats(0);
        frame->GetXaxis()->SetLabelSize(0);
        frame->GetXaxis()->SetTickLength(0);
        frame->GetYaxis()->SetTitle("Parameter value (prior #kern[0.3]{} #sigma = 1)");
        frame->GetYaxis()->SetTitleSize(axis_label_size);
        frame->GetYaxis()->SetLabelSize(axis_label_size);
        frame->GetYaxis()->SetTitleOffset(1.0);
        frame->Draw("AXIS");

        TBox *prior_band = new TBox(0.0f, -1.0f, (float)nBins, 1.0f);
        prior_band->SetFillColor(kGray);
        prior_band->SetFillStyle(1001);
        prior_band->SetLineColor(kGray+1);
        prior_band->SetLineWidth(1);
        prior_band->Draw("same");

        TGraphAsymmErrors *postbars = new TGraphAsymmErrors(nBins);
        for(int i = 0; i < nBins; ++i) {
            int syst_idx = with_osc ? (i - (int)model.nparams) : i;
            float err_lo = 0.0f, err_hi = 0.0f;
            if(syst_idx >= 0 && syst_idx < (int)param_err_lo.size() && syst_idx < (int)param_err_hi.size()) {
                err_lo = param_err_lo(syst_idx);
                err_hi = param_err_hi(syst_idx);
            }
            postbars->SetPoint(i, i + 0.5f, bfvalues[i]);
            postbars->SetPointError(i, bar_halfwidth, bar_halfwidth, err_lo, err_hi);
        }
        postbars->SetFillColor(kBlue-7);
        postbars->SetFillStyle(1001);
        postbars->SetLineColor(kBlue-8);
        postbars->SetLineWidth(1);
        postbars->Draw("2 same");

        TLine *l_zero = new TLine(0, 0, nBins, 0);
        l_zero->SetLineStyle(2); l_zero->SetLineColor(kGray+2); l_zero->SetLineWidth(1);
        l_zero->Draw();
        TLine *l_pm1 = new TLine(0, 1, nBins, 1);
        l_pm1->SetLineStyle(3); l_pm1->SetLineColor(kGray+2); l_pm1->SetLineWidth(1);
        l_pm1->Draw();
        TLine *l_mm1 = new TLine(0, -1, nBins, -1);
        l_mm1->SetLineStyle(3); l_mm1->SetLineColor(kGray+2); l_mm1->SetLineWidth(1);
        l_mm1->Draw();

        auto drawMarkerWithArrow = [&](float x, float y, int color, int marker_style, float msize) {
            bool clamp_below = y < -5.0f;
            bool clamp_above = y >  5.0f;
            if(!clamp_below && !clamp_above && (y < y_axis_min || y > y_axis_max)) return;
            float draw_y = clamp_below ? y_axis_min + arrow_margin : (clamp_above ? y_axis_max - arrow_margin : y);
            TMarker *m = new TMarker(x, draw_y, marker_style);
            m->SetMarkerSize(msize); m->SetMarkerColor(color); m->Draw();
            if(clamp_below) {
                TArrow *a = new TArrow(x, draw_y - arrow_length*0.3f, x, y_axis_min + arrow_length*0.2f, 0.008, "|>");
                a->SetLineColor(color); a->SetFillColor(color); a->SetLineWidth(1); a->Draw();
            } else if(clamp_above) {
                TArrow *a = new TArrow(x, draw_y + arrow_length*0.3f, x, y_axis_max - arrow_length*0.2f, 0.008, "|>");
                a->SetLineColor(color); a->SetFillColor(color); a->SetLineWidth(1); a->Draw();
            }
        };

        for(int i = 0; i < nBins; ++i) {
            int vec_idx = with_osc ? i : (i + (int)model.nparams);
            float x_center = i + 0.5f;
            if(vec_idx < (int)best_fit.size()) {
                drawMarkerWithArrow(x_center - marker_offset, (float)best_fit(vec_idx), kRed, 21, marker_size);
            }
            if(vec_idx < (int)true_params.size()) {
                drawMarkerWithArrow(x_center + marker_offset, (float)true_params(vec_idx), kOrange+7, 33, marker_size * 1.1f);
            }
        }

        const float label_y = y_axis_min - y_range_size * 0.04f;
        for(int i = 0; i < nBins; ++i) {
            std::string label;
            if(with_osc && i < (int)model.nparams) {
                const bool is_log_param = (i < (int)model.is_log10.size()) && model.is_log10[i];

                if(is_log_param) {
                    label = "Log_{10}(" + model.pretty_param_names[i] + ")";
                } else {
                    label = model.pretty_param_names[i];
                }
            } else {
                auto it = config.m_mcgen_variation_plotname_map.find(names[i]);
                label = (it != config.m_mcgen_variation_plotname_map.end()) ? it->second : names[i];
            }
            TLatex *t = new TLatex(i + 0.5f, label_y, label.c_str());
            t->SetTextAlign(13);
            t->SetTextSize(x_label_size);
            t->SetTextAngle(-45);
            t->Draw();
        }

        TLegend *leg = new TLegend(0.73, 0.60, 0.99, 0.92);
        leg->SetFillStyle(1001);
        leg->SetBorderSize(1);
        leg->SetTextSize(std::max(0.022f, std::min(0.030f, axis_label_size * 0.85f)));
        leg->AddEntry(prior_band, "Pre-fit #pm1#sigma (prior)", "f");
        leg->AddEntry(postbars,    "Post-fit #pm1#sigma (MCMC)", "f");
        TGraph *leg_bf = new TGraph(1);
        leg_bf->SetPoint(0, 0, 0);
        leg_bf->SetMarkerStyle(21);
        leg_bf->SetMarkerColor(kRed);
        leg_bf->SetMarkerSize(marker_size);
        leg->AddEntry(leg_bf, "Global best-fit", "p");
        TGraph *leg_tp = nullptr;
        if(true_params.size() > 0) {
            leg_tp = new TGraph(1);
            leg_tp->SetPoint(0, 0, 0);
            leg_tp->SetMarkerStyle(33);
            leg_tp->SetMarkerColor(kOrange+7);
            leg_tp->SetMarkerSize(marker_size * 1.1f);
            leg->AddEntry(leg_tp, "Injected values", "p");
        }
        leg->Draw();

        TText *vt = new TText();
        vt->SetNDC();
        vt->SetTextFont(42);
        vt->SetTextSize(0.028f);
        vt->SetTextAlign(33);
        std::string pv = "PROfit v" + std::string(PROJECT_VERSION_STR);
        vt->DrawText(0.96, 0.97, pv.c_str());

        c->Update();
        c->SaveAs((filename+"_1sigmaMCMC.pdf").c_str(), "pdf");
        delete c;
    }

    namespace {
        struct ChannelSpan {
            std::string label;     // subchannel plot/full name
            size_t start_bin = 0;  // inclusive, in full-bin space
            size_t nbins   = 0;
        };

        // Walk all subchannels and return their [start, nbins) span in the full bin space for var_index.
        std::vector<ChannelSpan> cov2spline_channel_spans(const PROconfig &config, int var_index) {
            std::vector<ChannelSpan> spans;
            size_t isub = 0;
            for(size_t im = 0; im < config.m_num_modes; ++im) {
                for(size_t id = 0; id < config.m_num_detectors; ++id) {
                    for(size_t ic = 0; ic < config.m_num_channels; ++ic) {
                        for(size_t sc = 0; sc < config.m_num_subchannels[ic]; ++sc) {
                            ChannelSpan s;
                            s.label    = config.m_fullnames[isub];
                            s.start_bin = config.GetGlobalVariableBinStart(isub, var_index);
                            s.nbins   = config.m_channel_variable_bins[ic][var_index].NBins();
                            spans.push_back(std::move(s));
                            ++isub;
                        }
                    }
                }
            }
            return spans;
        }

        void draw_channel_dividers_1d(const std::vector<ChannelSpan> &spans, double ymin, double ymax) {
            for(size_t s = 1; s < spans.size(); ++s) {
                const double x = static_cast<double>(spans[s].start_bin);
                auto *ln = new TLine(x, ymin, x, ymax);
                ln->SetLineColor(kGray + 2);
                ln->SetLineStyle(2);
                ln->Draw();
            }
        }

        void draw_channel_dividers_2d(const std::vector<ChannelSpan> &spans, double nbins_total) {
            for(size_t s = 1; s < spans.size(); ++s) {
                const double x = static_cast<double>(spans[s].start_bin);
                auto *lnv = new TLine(x, 0, x, nbins_total);
                lnv->SetLineColor(kGray + 2);
                lnv->SetLineStyle(2);
                lnv->Draw();
                auto *lnh = new TLine(0, x, nbins_total, x);
                lnh->SetLineColor(kGray + 2);
                lnh->SetLineStyle(2);
                lnh->Draw();
            }
        }

        void draw_version_stamp() {
            // Bottom-right of the master canvas so the stamp never sits over a sub-pad's
            // histogram title on multi-panel pages (top-right NDC lies inside the top-right
            // sub-pad and overlaps its title there).
            TText *t = new TText();
            t->SetNDC();
            t->SetTextFont(42);
            t->SetTextSize(0.022f);
            t->SetTextAlign(31); // bottom-right anchor
            std::string pv = "PROfit v" + std::string(PROJECT_VERSION_STR);
            t->DrawText(0.995, 0.005, pv.c_str());
        }

        std::string cov2spline_display_name(const PROconfig &config, const std::string &systname) {
            auto it = config.m_mcgen_variation_plotname_map.find(systname);
            if(it != config.m_mcgen_variation_plotname_map.end() && !it->second.empty()) {
                return it->second;
            }
            return systname;
        }
    }

    int plotCov2SplineChecks(const PROconfig &config, const PROspec &cv, const PROsyst &syst, const std::string &filename, int var_index) {
        if(syst.cov2spline_debug_info.empty()) {
            return 1;
        }

        set_matrix_palette();

        TCanvas c("c_cov2spline", "covariance_to_spline checks", 1600, 1100);
        c.Print((filename + "[").c_str(), "pdf");

        const std::vector<ChannelSpan> spans = cov2spline_channel_spans(config, var_index);

        for(const auto &[systname, dbg]: syst.cov2spline_debug_info) {
            const std::string display = cov2spline_display_name(config, systname);
            const int nbins = static_cast<int>(dbg.original_frac_cov.rows());
            const int K = static_cast<int>(dbg.kept_indices.size());
            const int n_eig = static_cast<int>(dbg.eigenvalues.size());

            // ---- Reconstructed covariance and residual (rank-K approximation) ----
            Eigen::MatrixXf recon = Eigen::MatrixXf::Zero(nbins, nbins);
            for(int k = 0; k < K; ++k) {
                const int idx = dbg.kept_indices[k];
                const float lam = dbg.eigenvalues(idx);
                const Eigen::VectorXf v = dbg.eigenvectors.col(idx);
                recon.noalias() += lam * (v * v.transpose());
            }
            // "difference" = what the rank-K splines alone fail to capture (distinct from the
            // "residual covariance" feature, which folds the un-kept eigenpairs back in).
            const Eigen::MatrixXf difference = dbg.original_frac_cov - recon;
            const float diff_frob = difference.norm();
            const float orig_frob = dbg.original_frac_cov.norm();
            const float diff_max_abs = difference.cwiseAbs().maxCoeff();

            // ---- Page 1: summary text ----
            int n_pos = 0, n_zero = 0, n_neg = 0;
            const float tol = 10.0f * Eigen::NumTraits<float>::dummy_precision();
            for(int i = 0; i < n_eig; ++i) {
                if(dbg.eigenvalues(i) > tol) ++n_pos;
                else if(dbg.eigenvalues(i) < -tol) ++n_neg;
                else ++n_zero;
            }
            {
                c.Clear();
                TPaveText pt(0.05, 0.05, 0.95, 0.95, "NDC");
                pt.SetFillColor(0);
                pt.SetBorderSize(0);
                pt.SetTextAlign(12);
                pt.SetTextFont(42);
                pt.SetTextSize(0.030f);
                TText *title_text = pt.AddText(("covariance_to_spline diagnostics: " + display).c_str());
                if(title_text) title_text->SetTextSize(0.045f);
                pt.AddText("");
                pt.AddText(("systname:  " + systname).c_str());
                pt.AddText(Form("Bins: %d", nbins));
                pt.AddText(Form("Eigenvalues: %d total  (positive %d, near-zero %d, negative %d)", n_eig, n_pos, n_zero, n_neg));
                pt.AddText(Form("Knobs retained: K = %d  of %d positive eigenvalues", K, n_pos));
                if(K > 0) {
                    const float lam_max = dbg.eigenvalues(dbg.kept_indices.front());
                    const float lam_min_kept = dbg.eigenvalues(dbg.kept_indices.back());
                    pt.AddText(Form("Largest eigenvalue: %.4g    Smallest kept eigenvalue: %.4g", lam_max, lam_min_kept));
                }
                pt.AddText(Form("Pre-symmetrization asymmetry (||C - C^T||_F): %.4g", dbg.pre_symm_asymmetry));
                pt.AddText(Form("Difference ||C_orig - C_recon||_F: %.4g   (relative: %.4g)", diff_frob, orig_frob > 0 ? diff_frob/orig_frob : 0.0f));
                pt.AddText(Form("Max |difference element|: %.4g", diff_max_abs));
                pt.AddText("");
                if(dbg.has_residual) {
                    const float resid_cov_frob = dbg.residual_cov.norm();
                    TText *rt = pt.AddText(Form("Residual covariance: ON  -  %d un-kept mode(s) retained as '%s'",
                                                dbg.n_residual_modes, dbg.residual_cov_name.c_str()));
                    if(rt) rt->SetTextColor(kGreen + 3);
                    pt.AddText(Form("   ||C_resid_cov||_F: %.4g   (relative to original: %.4g)",
                                    resid_cov_frob, orig_frob > 0 ? resid_cov_frob/orig_frob : 0.0f));
                    pt.AddText("   The K splines plus this covariance reproduce the full systematic;");
                    pt.AddText("   only negative-eigenvalue numerical noise is discarded.");
                } else if(K < n_pos) {
                    TText *rt = pt.AddText(Form("Residual covariance: OFF  -  %d smaller mode(s) DROPPED (include_resid_cov=false)", n_pos - K));
                    if(rt) rt->SetTextColor(kRed + 1);
                    pt.AddText("   The rank-K model underestimates the systematic by the difference shown above.");
                } else {
                    pt.AddText("Residual covariance: n/a  -  all positive eigenpairs kept as splines.");
                }
                pt.Draw();
                draw_version_stamp();
                c.Print(filename.c_str(), "pdf");
            }

            // ---- Page 2: original, rank-K reconstruction, and their difference ----
            auto fill_th2 = [&](const Eigen::MatrixXf &M, const std::string &name, const std::string &title) {
                auto h = std::make_unique<TH2D>(name.c_str(), title.c_str(), nbins, 0, nbins, nbins, 0, nbins);
                h->SetDirectory(nullptr);
                for(int i = 0; i < nbins; ++i)
                    for(int j = 0; j < nbins; ++j)
                        h->SetBinContent(i+1, j+1, M(i,j));
                const float m = std::max(std::fabs(h->GetMaximum()), std::fabs(h->GetMinimum()));
                h->SetMaximum(m);
                h->SetMinimum(-m);
                return h;
            };
            {
                auto h_orig  = fill_th2(dbg.original_frac_cov, "h_cov_orig_"+systname,  (display+" original frac cov;Bin index;Bin index").c_str());
                auto h_recon = fill_th2(recon,                  "h_cov_recon_"+systname, (display+Form(" reconstructed (rank %d);Bin index;Bin index", K)).c_str());
                auto h_diff  = fill_th2(difference,             "h_cov_diff_"+systname,  (display+" difference = orig - recon;Bin index;Bin index").c_str());
                c.Clear();
                c.Divide(3, 1);
                c.cd(1); gPad->SetRightMargin(0.14); gPad->SetTopMargin(0.13); h_orig->Draw("colz");  draw_channel_dividers_2d(spans, nbins);
                c.cd(2); gPad->SetRightMargin(0.14); gPad->SetTopMargin(0.13); h_recon->Draw("colz"); draw_channel_dividers_2d(spans, nbins);
                c.cd(3); gPad->SetRightMargin(0.14); gPad->SetTopMargin(0.13); h_diff->Draw("colz");  draw_channel_dividers_2d(spans, nbins);
                c.cd(0);
                draw_version_stamp();
                c.Print(filename.c_str(), "pdf");
            }

            // ---- Page 2b: closure check when include_resid_cov is on ----
            // Compare the original against the actual model used in the fit (K splines + residual
            // covariance). Their difference should be ~0 (only negative-eigenvalue noise remains),
            // which is the meaningful check now that the un-kept eigenpairs are retained.
            if(dbg.has_residual) {
                const Eigen::MatrixXf full_model = recon + dbg.residual_cov;
                const Eigen::MatrixXf closure = dbg.original_frac_cov - full_model;
                const float closure_frob = closure.norm();
                auto h_orig  = fill_th2(dbg.original_frac_cov, "h_cov_orig2_"+systname, (display+" original frac cov;Bin index;Bin index").c_str());
                auto h_model = fill_th2(full_model,            "h_cov_model_"+systname, (display+Form(" K=%d splines + residual cov;Bin index;Bin index", K)).c_str());
                auto h_clos  = fill_th2(closure,               "h_cov_clos_"+systname,  (display+Form(" difference (||.||_F=%.2g);Bin index;Bin index", closure_frob)).c_str());
                c.Clear();
                c.Divide(3, 1);
                // Titles read left-to-right as: [original] vs [splines + residual cov] -> [difference ~ 0].
                c.cd(1); gPad->SetRightMargin(0.14); gPad->SetTopMargin(0.13); h_orig->Draw("colz");  draw_channel_dividers_2d(spans, nbins);
                c.cd(2); gPad->SetRightMargin(0.14); gPad->SetTopMargin(0.13); h_model->Draw("colz"); draw_channel_dividers_2d(spans, nbins);
                c.cd(3); gPad->SetRightMargin(0.14); gPad->SetTopMargin(0.13); h_clos->Draw("colz");  draw_channel_dividers_2d(spans, nbins);
                c.cd(0);
                draw_version_stamp();
                c.Print(filename.c_str(), "pdf");
            }

            // ---- Page 3: sqrt(diag) per bin, original vs reconstructed (+ splines+resid when on) ----
            {
                auto h_d_orig  = std::make_unique<TH1D>(("h_diag_orig_"+systname).c_str(), (display+" fractional uncertainty per bin;Bin index;#sqrt{diag}").c_str(), nbins, 0, nbins);
                auto h_d_recon = std::make_unique<TH1D>(("h_diag_recon_"+systname).c_str(), "", nbins, 0, nbins);
                auto h_d_sum   = std::make_unique<TH1D>(("h_diag_sum_"+systname).c_str(), "", nbins, 0, nbins);
                h_d_orig->SetDirectory(nullptr); h_d_recon->SetDirectory(nullptr); h_d_sum->SetDirectory(nullptr);
                for(int b = 0; b < nbins; ++b) {
                    h_d_orig ->SetBinContent(b+1, std::sqrt(std::max(0.0f, dbg.original_frac_cov(b,b))));
                    h_d_recon->SetBinContent(b+1, std::sqrt(std::max(0.0f, recon(b,b))));
                    if(dbg.has_residual)
                        h_d_sum->SetBinContent(b+1, std::sqrt(std::max(0.0f, recon(b,b) + dbg.residual_cov(b,b))));
                }
                h_d_orig->SetLineColor(kBlack);
                h_d_orig->SetLineWidth(2);
                h_d_recon->SetLineColor(kRed);
                h_d_recon->SetLineWidth(2);
                h_d_recon->SetLineStyle(2);
                h_d_sum->SetLineColor(kGreen + 2);
                h_d_sum->SetLineWidth(2);
                h_d_sum->SetLineStyle(3);
                const double ymax = 1.15 * std::max(h_d_orig->GetMaximum(), h_d_recon->GetMaximum());
                h_d_orig->SetMaximum(ymax);
                h_d_orig->SetMinimum(0.0);
                c.Clear();
                gPad->SetTopMargin(0.13);
                h_d_orig->Draw("hist");
                h_d_recon->Draw("hist same");
                if(dbg.has_residual) h_d_sum->Draw("hist same");
                draw_channel_dividers_1d(spans, 0.0, ymax);
                auto *leg = new TLegend(0.62, 0.74, 0.92, 0.9);
                leg->SetBorderSize(0);
                leg->SetFillStyle(0);
                leg->AddEntry(h_d_orig.get(), "Original (multisim)", "l");
                leg->AddEntry(h_d_recon.get(), Form("K=%d splines only", K), "l");
                if(dbg.has_residual) leg->AddEntry(h_d_sum.get(), "Splines + residual cov", "l");
                leg->Draw();
                draw_version_stamp();
                c.Print(filename.c_str(), "pdf");
            }

            // ---- Page 4: scree plot (eigenvalues sorted descending, log y) ----
            std::vector<double> sorted_eigs(n_eig);
            for(int i = 0; i < n_eig; ++i) sorted_eigs[i] = dbg.eigenvalues(n_eig - 1 - i);
            {
                std::vector<double> xs(n_eig), ys(n_eig);
                bool any_positive = false;
                double y_floor = 1e-30;
                for(int i = 0; i < n_eig; ++i) {
                    xs[i] = i;
                    ys[i] = sorted_eigs[i] > 0 ? sorted_eigs[i] : y_floor;
                    if(sorted_eigs[i] > 0) any_positive = true;
                }
                auto *g = new TGraph(n_eig, xs.data(), ys.data());
                g->SetTitle((display + " eigenvalue scree;eigenvalue index (descending);eigenvalue").c_str());
                g->SetMarkerStyle(20);
                g->SetMarkerSize(0.6);
                g->SetMarkerColor(kBlue + 1);
                g->SetLineColor(kBlue + 1);
                c.Clear();
                gPad->SetTopMargin(0.13);
                if(any_positive) gPad->SetLogy(1);
                g->Draw("APL");
                if(K > 0 && K <= n_eig) {
                    // Mark the boundary of the kept region (right edge of last kept index).
                    const double x = K - 0.5;
                    const double y_top = sorted_eigs[0] * 10;
                    const double y_bot = y_floor;
                    auto *ln = new TLine(x, y_bot, x, y_top);
                    ln->SetLineColor(kRed);
                    ln->SetLineStyle(2);
                    ln->SetLineWidth(2);
                    ln->Draw();
                }
                draw_version_stamp();
                c.Print(filename.c_str(), "pdf");
                gPad->SetLogy(0);
            }

            // ---- Page 5: cumulative variance ----
            {
                double total = 0.0;
                for(int i = 0; i < n_eig; ++i) total += std::max(0.0, static_cast<double>(dbg.eigenvalues(i)));
                std::vector<double> xs(n_eig), ys(n_eig);
                double running = 0.0;
                for(int i = 0; i < n_eig; ++i) {
                    running += std::max(0.0, sorted_eigs[i]);
                    xs[i] = i + 1;
                    ys[i] = total > 0 ? running / total : 0.0;
                }
                auto *g = new TGraph(n_eig, xs.data(), ys.data());
                g->SetTitle((display + " cumulative variance captured;Number of modes kept;Fraction of trace").c_str());
                g->SetMarkerStyle(20);
                g->SetMarkerSize(0.6);
                g->SetLineColor(kBlue + 1);
                g->SetMarkerColor(kBlue + 1);
                c.Clear();
                gPad->SetTopMargin(0.13);
                g->Draw("APL");
                g->GetHistogram()->SetMinimum(0.0);
                g->GetHistogram()->SetMaximum(1.05);
                auto draw_h = [&](double y, int col, const char *label){
                    auto *ln = new TLine(0, y, n_eig, y);
                    ln->SetLineColor(col); ln->SetLineStyle(2); ln->Draw();
                    auto *t = new TLatex(0.92 * n_eig, y + 0.01, label);
                    t->SetTextSize(0.025f); t->SetTextColor(col); t->Draw();
                };
                draw_h(0.90, kGray+2, "90%");
                draw_h(0.95, kGray+2, "95%");
                draw_h(0.99, kGray+2, "99%");
                if(K > 0 && K <= n_eig) {
                    auto *lnK = new TLine(K, 0, K, 1.05);
                    lnK->SetLineColor(kRed); lnK->SetLineStyle(2); lnK->SetLineWidth(2);
                    lnK->Draw();
                }
                draw_version_stamp();
                c.Print(filename.c_str(), "pdf");
            }

            // ---- Page 6: eigenvector heatmap (kept modes only), entries = sqrt(λ_k) * v_k[b] ----
            if(K > 0) {
                auto h_ev = std::make_unique<TH2D>(("h_eigvec_"+systname).c_str(), (display + " eigenvectors (rows = kept knobs);Bin index;Knob k").c_str(), nbins, 0, nbins, K, 0, K);
                h_ev->SetDirectory(nullptr);
                for(int k = 0; k < K; ++k) {
                    const int idx = dbg.kept_indices[k];
                    const float s = std::sqrt(std::max(0.0f, dbg.eigenvalues(idx)));
                    const Eigen::VectorXf v = dbg.eigenvectors.col(idx);
                    for(int b = 0; b < nbins; ++b) {
                        h_ev->SetBinContent(b+1, k+1, s * v(b));
                    }
                }
                const float m_ev = std::max(std::fabs(h_ev->GetMaximum()), std::fabs(h_ev->GetMinimum()));
                h_ev->SetMaximum(m_ev);
                h_ev->SetMinimum(-m_ev);
                c.Clear();
                gPad->SetRightMargin(0.13);
                gPad->SetTopMargin(0.13);
                h_ev->Draw("colz");
                // Vertical channel dividers
                for(size_t s = 1; s < spans.size(); ++s) {
                    const double x = static_cast<double>(spans[s].start_bin);
                    auto *ln = new TLine(x, 0, x, K);
                    ln->SetLineColor(kGray+2); ln->SetLineStyle(2); ln->Draw();
                }
                draw_version_stamp();
                c.Print(filename.c_str(), "pdf");
            }

            // ---- Per-knob bin response, 4 per page (sqrt(λ_k) * v_k[b]) ----
            for(int k0 = 0; k0 < K; k0 += 4) {
                c.Clear();
                c.Divide(2, 2);
                for(int kk = 0; kk < 4 && (k0 + kk) < K; ++kk) {
                    const int k = k0 + kk;
                    const int idx = dbg.kept_indices[k];
                    const float s = std::sqrt(std::max(0.0f, dbg.eigenvalues(idx)));
                    const Eigen::VectorXf v = dbg.eigenvectors.col(idx);
                    auto h = std::make_unique<TH1D>(Form("h_knob_resp_%s_%d", systname.c_str(), k), Form("%s knob %d   #lambda = %.3g   (+1 #sigma response);Bin index;#sqrt{#lambda} #upoint v_{k}[b]", display.c_str(), k, dbg.eigenvalues(idx)), nbins, 0, nbins);
                    h->SetDirectory(nullptr);
                    double yabs = 0;
                    for(int b = 0; b < nbins; ++b) {
                        const double y = s * v(b);
                        h->SetBinContent(b+1, y);
                        yabs = std::max(yabs, std::fabs(y));
                    }
                    h->SetLineColor(kBlue + 1);
                    h->SetLineWidth(2);
                    h->SetMaximum(1.2 * (yabs > 0 ? yabs : 1.0));
                    h->SetMinimum(-1.2 * (yabs > 0 ? yabs : 1.0));
                    c.cd(kk + 1);
                    gPad->SetTopMargin(0.13);
                    h->DrawCopy("hist");
                    draw_channel_dividers_1d(spans, h->GetMinimum(), h->GetMaximum());
                    auto *zero = new TLine(0, 0, nbins, 0);
                    zero->SetLineColor(kBlack); zero->Draw();
                }
                c.cd(0);
                draw_version_stamp();
                c.Print(filename.c_str(), "pdf");
            }

            // ---- Per-knob CV ± 1σ band (linear knob: response = 1 ± alpha_b), 4 per page ----
            const Eigen::VectorXf cv_full = cv.Spec();
            const bool cv_size_ok = (cv_full.size() == nbins);
            if(cv_size_ok) {
                for(int k0 = 0; k0 < K; k0 += 4) {
                    c.Clear();
                    c.Divide(2, 2);
                    for(int kk = 0; kk < 4 && (k0 + kk) < K; ++kk) {
                        const int k = k0 + kk;
                        const int idx = dbg.kept_indices[k];
                        const float s = std::sqrt(std::max(0.0f, dbg.eigenvalues(idx)));
                        const Eigen::VectorXf v = dbg.eigenvectors.col(idx);
                        auto h_cv = std::make_unique<TH1D>(Form("h_cv_%s_%d", systname.c_str(), k), Form("%s knob %d   CV #pm 1 #sigma;Bin index;Events", display.c_str(), k), nbins, 0, nbins);
                        auto h_up = std::make_unique<TH1D>(Form("h_up_%s_%d", systname.c_str(), k), "", nbins, 0, nbins);
                        auto h_dn = std::make_unique<TH1D>(Form("h_dn_%s_%d", systname.c_str(), k), "", nbins, 0, nbins);
                        h_cv->SetDirectory(nullptr); h_up->SetDirectory(nullptr); h_dn->SetDirectory(nullptr);
                        double ymax = 0;
                        for(int b = 0; b < nbins; ++b) {
                            const double alpha = s * v(b);
                            const double y0 = cv_full(b);
                            h_cv->SetBinContent(b+1, y0);
                            h_up->SetBinContent(b+1, y0 * (1.0 + alpha));
                            h_dn->SetBinContent(b+1, y0 * (1.0 - alpha));
                            ymax = std::max({ymax, std::fabs(y0 * (1.0 + alpha)), std::fabs(y0 * (1.0 - alpha))});
                        }
                        h_cv->SetLineColor(kBlack);
                        h_cv->SetLineWidth(2);
                        h_up->SetLineColor(kRed + 1);
                        h_dn->SetLineColor(kAzure + 2);
                        h_up->SetLineStyle(2);
                        h_dn->SetLineStyle(2);
                        h_cv->SetMaximum(1.15 * ymax);
                        h_cv->SetMinimum(0.0);
                        c.cd(kk + 1);
                        gPad->SetTopMargin(0.13);
                        h_cv->DrawCopy("hist");
                        h_up->DrawCopy("hist same");
                        h_dn->DrawCopy("hist same");
                        draw_channel_dividers_1d(spans, 0.0, 1.15 * ymax);
                    }
                    c.cd(0);
                    draw_version_stamp();
                    c.Print(filename.c_str(), "pdf");
                }
            } else {
                log<LOG_WARNING>(L"%1% || CV spectrum size %2% != covariance size %3% for systematic %4%; skipping CV-band pages.")
                    % __func__ % static_cast<int>(cv_full.size()) % nbins % systname.c_str();
            }

            // ---- Aggregate band: CV ± from original cov vs ± from rank-K sum-of-knobs ----
            if(cv_size_ok) {
                auto h_cv  = std::make_unique<TH1D>(("h_aggcv_"+systname).c_str(), (display + " aggregate CV #pm 1 #sigma;Bin index;Events").c_str(), nbins, 0, nbins);
                auto h_oup = std::make_unique<TH1D>(("h_aggoup_"+systname).c_str(), "", nbins, 0, nbins);
                auto h_odn = std::make_unique<TH1D>(("h_aggodn_"+systname).c_str(), "", nbins, 0, nbins);
                auto h_kup = std::make_unique<TH1D>(("h_aggkup_"+systname).c_str(), "", nbins, 0, nbins);
                auto h_kdn = std::make_unique<TH1D>(("h_aggkdn_"+systname).c_str(), "", nbins, 0, nbins);
                auto h_sup = std::make_unique<TH1D>(("h_aggsup_"+systname).c_str(), "", nbins, 0, nbins);
                auto h_sdn = std::make_unique<TH1D>(("h_aggsdn_"+systname).c_str(), "", nbins, 0, nbins);
                h_cv->SetDirectory(nullptr); h_oup->SetDirectory(nullptr); h_odn->SetDirectory(nullptr);
                h_kup->SetDirectory(nullptr); h_kdn->SetDirectory(nullptr); h_sup->SetDirectory(nullptr); h_sdn->SetDirectory(nullptr);
                double ymax = 0;
                for(int b = 0; b < nbins; ++b) {
                    const double y0 = cv_full(b);
                    const double sigma_orig  = std::sqrt(std::max(0.0f, dbg.original_frac_cov(b,b))) * y0;
                    const double sigma_recon = std::sqrt(std::max(0.0f, recon(b,b))) * y0;
                    // Combined model = K splines (in quadrature) + residual covariance diagonal.
                    const double var_comb = recon(b,b) + (dbg.has_residual ? dbg.residual_cov(b,b) : 0.0f);
                    const double sigma_comb = std::sqrt(std::max(0.0, var_comb)) * y0;
                    h_cv->SetBinContent(b+1, y0);
                    h_oup->SetBinContent(b+1, y0 + sigma_orig);
                    h_odn->SetBinContent(b+1, y0 - sigma_orig);
                    h_kup->SetBinContent(b+1, y0 + sigma_recon);
                    h_kdn->SetBinContent(b+1, y0 - sigma_recon);
                    h_sup->SetBinContent(b+1, y0 + sigma_comb);
                    h_sdn->SetBinContent(b+1, y0 - sigma_comb);
                    ymax = std::max({ymax, y0 + sigma_orig, y0 + sigma_recon, y0 + sigma_comb});
                }
                h_cv->SetLineColor(kBlack);
                h_cv->SetLineWidth(2);
                h_oup->SetLineColor(kBlack); h_odn->SetLineColor(kBlack);
                h_oup->SetLineStyle(2); h_odn->SetLineStyle(2);
                h_kup->SetLineColor(kRed + 1); h_kdn->SetLineColor(kRed + 1);
                h_kup->SetLineStyle(1); h_kdn->SetLineStyle(1);
                h_sup->SetLineColor(kGreen + 2); h_sdn->SetLineColor(kGreen + 2);
                h_sup->SetLineStyle(1); h_sdn->SetLineStyle(1);
                h_sup->SetLineWidth(2); h_sdn->SetLineWidth(2);
                h_cv->SetMaximum(1.15 * ymax);
                h_cv->SetMinimum(0.0);
                c.Clear();
                gPad->SetTopMargin(0.13);
                h_cv->Draw("hist");
                // Draw rank-K band first, then the combined (splines+resid) band, then the
                // original-multisim dashed lines on top so each stays visible where they overlap.
                h_kup->Draw("hist same");
                h_kdn->Draw("hist same");
                if(dbg.has_residual) { h_sup->Draw("hist same"); h_sdn->Draw("hist same"); }
                h_oup->Draw("hist same");
                h_odn->Draw("hist same");
                draw_channel_dividers_1d(spans, 0.0, 1.15 * ymax);
                auto *leg = new TLegend(0.6, 0.72, 0.92, 0.9);
                leg->SetBorderSize(0); leg->SetFillStyle(0);
                leg->AddEntry(h_cv.get(),  "CV", "l");
                leg->AddEntry(h_oup.get(), "CV #pm 1#sigma (original multisim)", "l");
                leg->AddEntry(h_kup.get(), Form("CV #pm 1#sigma (K=%d splines only)", K), "l");
                if(dbg.has_residual) leg->AddEntry(h_sup.get(), "CV #pm 1#sigma (splines + residual cov)", "l");
                leg->Draw();
                draw_version_stamp();
                c.Print(filename.c_str(), "pdf");
            }
        }

        c.Print((filename + "]").c_str(), "pdf");
        return 0;
    }
}
