#include "PROfit_common.h"

void run_global(float &global_fit_chi2, Eigen::VectorXf &global_fit_result, const PROconfig &config, const PROpeller &prop, PROmetric &metric, const PROdata &data, const Eigen::VectorXf CVParams, const Eigen::VectorXf &lb, const Eigen::VectorXf &ub, const std::vector<int> fixed, const PROfitterConfig &fitConfig, const PROpt &options) {
    GlobalFitOptions opt = GlobalFitOptions::Default;
    if(options.progress_bar) opt |= GlobalFitOptions::Progress;
    if(options.binwidth_scale) opt |= GlobalFitOptions::BinWidthScaled;
    if(!fixed[0] || !options.systs_only) opt |= GlobalFitOptions::FreqSeedPts;
    opt |= options.MCMC_prefit_errors ? GlobalFitOptions::MCMCPrefitErrorBand : GlobalFitOptions::PrefitErrorBand;
    opt |= GlobalFitOptions::PostFitErrorBand;
    opt |= GlobalFitOptions::Correlations;
    PROspec cv = FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), CVParams , true , config.i_prime);
    GlobalFitResult fitres = run_global_fit(config, prop, data, metric, ub, lb, fitConfig, CVParams, cv, fixed, opt); 
    global_fit_chi2 = fitres.chi2;
    global_fit_result = fitres.fitter.best_fit;

    PlotOptions popt; 
    if(options.data_mc_ratio){
        popt = PlotOptions::DataMCRatio;
    } else {
        popt = PlotOptions::DataPostfitRatio;
    }
    if(options.binwidth_scale) popt |= PlotOptions::BinWidthScaled;
    if(options.area_normalized) popt |= PlotOptions::AreaNormalized;

    std::map<std::string, TObject *> drawn_objs = draw_fit_result(config, prop, metric.GetModel(), metric.GetSysts(), metric, cv, data, fitres, options.final_output_tag+"_PROglobal", popt, options.pbounds, options.plot_channel_ratios);

    TFile fout((options.final_output_tag+"_PROglobal.root").c_str(), "RECREATE");
    for(const auto &[n, o] : drawn_objs)
        o->Write(n.c_str());

    draw_harmonic_scan_pdf(fitres, fitConfig, metric.GetModel(), options.final_output_tag+"_PROglobal_Harmonic_Scan.pdf");

    // PROjector pre-fit: the global fit above WAS the pre-fit (masked to the matched
    // channels, physics fixed at CV unless floated); save its nuisance posterior.
    if(options.projector_config.prefit_mode()) {
        if(!PROjectorSaveConstraint(options.final_output_tag+"_PROjector_constraint.bin", options.projector_config,
                    config, metric, fitres.fitter.best_fit, fitres.chi2, fixed, options.chi2))
            exit(1);
    }
}
