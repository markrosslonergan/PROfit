#include "PROfit_common.h"

void run_profile(float &global_fit_chi2, Eigen::VectorXf &global_fit_result, const PROconfig &config, const PROpeller &prop, PROmetric &metric, const PROdata &data, const Eigen::VectorXf &CVParams, const Eigen::VectorXf &fakedataparams, const Eigen::VectorXf &lb, const Eigen::VectorXf &ub, const std::vector<int> &fixed, const PROfitterConfig &fitConfig, PROfitterConfig &scanFitConfig, const PROpt &options, PROseed &myseed) {
    GlobalFitOptions opt = GlobalFitOptions::Default;
    if(options.progress_bar) opt |= GlobalFitOptions::Progress;
    if(options.binwidth_scale) opt |= GlobalFitOptions::BinWidthScaled;
    if(!fixed[0] || !options.systs_only) opt |= GlobalFitOptions::FreqSeedPts;
    opt |= options.MCMC_prefit_errors ? GlobalFitOptions::MCMCPrefitErrorBand : GlobalFitOptions::PrefitErrorBand;
    opt |= GlobalFitOptions::PostFitErrorBand;
    opt |= GlobalFitOptions::Correlations;
    PROspec cv = FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), CVParams , true ,config.i_prime);
    GlobalFitResult fitres = run_global_fit(config, prop, data, metric, ub, lb, fitConfig, CVParams, cv, fixed, opt); 
    global_fit_chi2 = fitres.chi2;
    global_fit_result = fitres.fitter.best_fit;
    log<LOG_INFO>(L"%1% || MCMC acceptance is  %2%. ") % __func__% ((double)fitres.mh->naccept /fitConfig.MCMCiter);

    PlotOptions popt; 
    if(options.data_mc_ratio){
        popt = PlotOptions::DataMCRatio;
    } else {
        popt = PlotOptions::DataPostfitRatio;
    }
    if(options.binwidth_scale) popt |= PlotOptions::BinWidthScaled;
    if(options.area_normalized) popt |= PlotOptions::AreaNormalized;

    std::map<std::string, TObject *> drawn_objs = draw_fit_result(config, prop, metric.GetModel(), metric.GetSysts(), metric, cv, data, fitres, options.final_output_tag+"_PROfile", popt, options.pbounds, options.plot_channel_ratios);

    plot_mcmc_1sigma(options.final_output_tag+"_PROfile", config, metric.GetSysts(), metric.GetModel(), fitres.fitter.best_fit, fitres.post_param_lo, fitres.post_param_hi, !options.systs_only, fakedataparams);

    log<LOG_INFO>(L"%1% ||  Beginning full PROfile ") % __func__;

    if(options.progress_bar) scanFitConfig.progress_bar = true;

    std::vector<Eigen::VectorXf> seeds = fitres.fitter.freq_seed_points;//to be updated to v1.1.5 harmoincs [DONE]
    if(!seeds.size()) seeds.push_back(fitres.fitter.best_fit);
    // Toggle scan-mode timing instrumentation around the PROfile dispatch.
    // The constructor reads PROfit::GetScanTimingEnabled() once; PROfitter::Fit
    // also reads it (cached per call) to gate its sub-timers.
    if (options.profile_timing) PROfit::GetScanTimingEnabled() = true;
    PROfile profile(config, metric.GetSysts(), metric.GetModel(), metric, myseed, scanFitConfig,
            options.final_output_tag+"_PROfile", fitres.chi2, !options.systs_only, options.nthread, seeds,
            fakedataparams, options.use_probe, options.n_probe_chunks,
            buildBkgOnlyFixedSeeds(metric.GetModel(), CVParams));
    if (options.profile_timing) PROfit::GetScanTimingEnabled() = false;
    // The scan can find a lower global minimum than run_global_fit (trapped in a
    // local minimum); PROfile has already re-baselined its curves against
    // it. Mirror the harmonic-seed adoption in run_global_fit: everything
    // downstream (Plot's best-fit markers, global_fit_result, the
    // global_fit_result histogram written below) reads fitres, so update
    // it to the minimum the curves are actually baselined on. The spectra
    // PDFs from draw_fit_result above were drawn at the pre-scan fit.
    if(profile.newglob_param.size() > 0) {
        log<LOG_WARNING>(L"%1% || PROfile found a lower global minimum (chi^2 %2% vs global fit's %3%); adopting it as the global best fit for plots and recorded results.")
            % __func__ % profile.newglob % fitres.chi2;
        log<LOG_WARNING>(L"%1% || -- new best-fit params: %2%") % __func__ % profile.newglob_param;
        fitres.chi2 = profile.newglob;
        fitres.fitter.best_fit = profile.newglob_param;
        global_fit_chi2 = profile.newglob;
        global_fit_result = profile.newglob_param;
    }
    log<LOG_INFO>(L"%1% || fakedataparams for Plot (true_params/red stars): %2%") % __func__ % fakedataparams;
    profile.Plot(config, metric.GetSysts(), metric.GetModel(), metric, myseed,
            options.final_output_tag+"_PROfile", !options.systs_only, fitres.fitter.best_fit,
            fakedataparams, fitres.spline_covariance, fitres.post_param_lo, fitres.post_param_hi);

    TFile fout((options.final_output_tag+"_PROfile.root").c_str(), "RECREATE");
    profile.onesig.Write("one_sigma_errs");
    for(const auto &[name, obj] : drawn_objs)
        obj->Write(name.c_str());

    for(size_t i = 0; i < profile.graphs.size(); ++i) {
        std::string name = i < metric.GetModel().nparams
                         ? metric.GetModel().param_names[i]
                         : metric.GetSysts().spline_names[i - metric.GetModel().nparams];
        profile.graphs[i]->Write((name + "_profile").c_str());
    }

    if(global_fit_result.size() > 0) {
        bool use_phys_gfr = (size_t)global_fit_result.size() == metric.GetModel().nparams + metric.GetSysts().GetNSplines();
        TH1D global_fit_hist("global_fit_result", "Global Best Fit Parameters", global_fit_result.size(), 0, global_fit_result.size());
        for(long i = 0; i < global_fit_result.size(); i++) {
            std::string pname;
            if(use_phys_gfr && i < (long)metric.GetModel().nparams)
                pname = metric.GetModel().param_names[i];
            else {
                long idx = use_phys_gfr ? i - metric.GetModel().nparams : i;
                pname = config.m_mcgen_variation_plotname_map.at(metric.GetSysts().spline_names[idx]);
            }
            global_fit_hist.GetXaxis()->SetBinLabel(i+1, pname.c_str());
            global_fit_hist.SetBinContent(i+1, global_fit_result(i));
        }
        global_fit_hist.Write();
    }
}
