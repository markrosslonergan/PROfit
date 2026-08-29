#include "PROfit_common.h"

void run_fc(const PROconfig &config, const PROpeller &prop, PROmetric &metric, const PROdata &data, const Eigen::VectorXf &CVParams, const Eigen::VectorXf &fakeDataParams, const Eigen::VectorXf &lb, const Eigen::VectorXf &ub, const std::vector<int> &fixed, const PROfitterConfig &fitConfig, const PROfitterConfig &scanFitConfig, const PROpt &options, PROseed &myseed) {
    float global_chi2 = 0, null_chi2 = 0;
    if(options.gof_pvalue || options.pvalue) {
        // Nominal Fit with all parameters
        GlobalFitOptions opt = GlobalFitOptions::Default;
        if(options.progress_bar) opt |= GlobalFitOptions::Progress;
        if(!fixed[0] || !options.systs_only) opt |= GlobalFitOptions::FreqSeedPts;
        PROspec cv = FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), CVParams , true ,config.i_prime);
        GlobalFitResult fitres = run_global_fit(config, prop, data, metric, ub, lb, fitConfig, CVParams, cv, fixed, opt); 
        global_chi2 = fitres.chi2;
    }
    if(options.pvalue) {
        // Fit with fixed osc parameters
        size_t nphys = metric.GetModel().nparams;
        Eigen::VectorXf flb = lb;
        Eigen::VectorXf fub = ub;
        std::vector<int> ffixed = fixed;
        for(size_t i = 0; i < nphys; ++i) {
            flb(i) = metric.GetModel().default_val(i);
            fub(i) = metric.GetModel().default_val(i);
            ffixed[i] = 1;
        }
        
        GlobalFitOptions opt = GlobalFitOptions::Default;
        if(options.progress_bar) opt |= GlobalFitOptions::Progress;
        PROspec cv = FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), CVParams , true ,config.i_prime);
        GlobalFitResult fitres = run_global_fit(config, prop, data, metric, fub, flb, fitConfig, CVParams, cv, ffixed, opt); 
        null_chi2 = fitres.chi2;
    }

    size_t FCthreads = options.nthread > options.nuniv ? options.nuniv : options.nthread;
    Eigen::MatrixXf cv_vec = FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), CVParams , true,config.i_prime).Spec();
    Eigen::MatrixXf L = metric.GetSysts().DecomposeFractionalCovariance(config, cv_vec);

    std::vector<std::vector<float>> dchi2s;
    dchi2s.reserve(FCthreads);
    std::vector<std::vector<fc_out>> outs;
    outs.reserve(FCthreads);
    std::vector<std::thread> threads;
    size_t todo = options.nuniv/FCthreads;
    size_t addone = FCthreads - options.nuniv%FCthreads;
    bool gof_mode = options.gof_pvalue;

    std::vector<std::pair<int, std::string>> fc_PB_configs;
    for (size_t i = 0; i < FCthreads; ++i) {
            fc_PB_configs.push_back({int(options.nuniv/FCthreads), "Thread " + std::to_string(i)});
    }
    MultiPROgressBar fc_progress(fc_PB_configs);
    fc_progress.initialize_display();
    fc_progress.start_display_thread(); 


    for(size_t i = 0; i < FCthreads; i++) {
        dchi2s.emplace_back();
        outs.emplace_back();
        fc_args args{todo + (i >= addone), &dchi2s.back(), &outs.back(), config, prop, metric.GetSysts(), options.chi2, fakeDataParams, L, scanFitConfig,(*myseed.getThreadSeeds())[i], (int)i, !options.eventbyevent, gof_mode};


        threads.emplace_back([args, &fc_progress]() {
                    PROfit::fc_worker(args, std::ref(fc_progress));
                    });
    }
    for(auto&& t: threads) {
        t.join();
    }
    fc_progress.finish_all();

    std::vector<float> flattened_dchi2s;
    for(const auto& v: dchi2s) for(const auto& dchi2: v) flattened_dchi2s.push_back(dchi2);
    std::sort(flattened_dchi2s.begin(), flattened_dchi2s.end());
    log<LOG_INFO>(L"%1% || 90%% Feldman-Cousins delta chi2 after throwing %2% universes is %3%") 
        % __func__ % options.nuniv % flattened_dchi2s[0.9*flattened_dchi2s.size()];
    if(options.gof_pvalue) {
        std::vector<float> flattened_syst_chi2;
        for(const auto &out : outs) for(const auto &fco : out) flattened_syst_chi2.push_back(fco.chi2_syst);
        std::sort(flattened_syst_chi2.begin(), flattened_syst_chi2.end());
        log<LOG_ERROR>(L"%1% || All: %2% ") % __func__ % flattened_syst_chi2;
        log<LOG_ERROR>(L"%1% || chi: %2% ") % __func__ % global_chi2;
        auto it = std::lower_bound(flattened_syst_chi2.begin(), flattened_syst_chi2.end(), global_chi2);
        size_t index =  std::distance(flattened_syst_chi2.begin(),it);
        size_t count_above = flattened_syst_chi2.size()-index;
        float pval = (float)count_above/(float)options.nuniv;
        log<LOG_ERROR>(L"%1% || Finished throws. %2% %3%") % __func__ % index % count_above;
        log<LOG_ERROR>(L"%1% || GOF pval after throwing %2% universes is %3%") % __func__ % options.nuniv % pval ;
    }
    if(options.pvalue) {
        log<LOG_ERROR>(L"%1% || All Delta Chis: %2% ") % __func__ % flattened_dchi2s;
        log<LOG_ERROR>(L"%1% || Delta chi bkg-min: %2% ") % __func__ % float(null_chi2 - global_chi2);
        auto itFC = std::lower_bound(flattened_dchi2s.begin(), flattened_dchi2s.end(), float(null_chi2-global_chi2));
        size_t indexFC =  std::distance(flattened_dchi2s.begin(),itFC);
        size_t count_aboveFC = flattened_dchi2s.size()-indexFC;
        float pvalFC = (float)count_aboveFC/(float)options.nuniv;

        log<LOG_ERROR>(L"%1% || Finished throws. %2% %3%") % __func__ % indexFC % count_aboveFC;
        log<LOG_ERROR>(L"%1% || FC Corrected pval after throwing %2% universes is %3%") % __func__ % options.nuniv % pvalFC ;
    }

    {
        TFile fout((options.final_output_tag+"_FC.root").c_str(), "RECREATE");
        fout.cd();
        float chi2_osc, chi2_syst;
        // One float per physics parameter — plain branches named "best_<param_name>".
        // Vector kept alive for the full lifetime of the TTree.
        std::vector<float> best_phys_vals(metric.GetModel().nparams, 0.0f);
        std::map<std::string, float> best_systs_osc, best_systs, syst_throw;
        TTree tree("tree", "tree");
        tree.Branch("chi2_osc",  &chi2_osc);
        tree.Branch("chi2_syst", &chi2_syst);
        for(size_t i = 0; i < metric.GetModel().nparams; ++i)
            tree.Branch(("best_" + metric.GetModel().param_names[i]).c_str(), &best_phys_vals[i]);
        tree.Branch("best_systs_osc", &best_systs_osc);
        tree.Branch("best_systs",     &best_systs);
        tree.Branch("syst_throw",     &syst_throw);

        for(const auto &out: outs) {
            for(const auto &fco: out) {
                chi2_osc  = fco.chi2_osc;
                chi2_syst = fco.chi2_syst;
                for(size_t i = 0; i < metric.GetModel().nparams; ++i) {
                    float raw = fco.best_phys_osc.size() > (Eigen::Index)i ? fco.best_phys_osc(i) : 0.0f;
                    best_phys_vals[i] = metric.GetModel().is_log10[i] ? std::pow(10.0f, raw) : raw;
                }
                for(size_t i = 0; i < metric.GetSysts().GetNSplines(); ++i) {
                    if(!options.gof_pvalue) best_systs_osc[metric.GetSysts().spline_names[i]] = fco.best_fit_osc(i);
                    best_systs[metric.GetSysts().spline_names[i]] = fco.best_fit_syst(i);
                    syst_throw[metric.GetSysts().spline_names[i]] = fco.syst_throw(i);
                }
                tree.Fill();
            }
        }

        tree.Write();
    }
    {
        ofstream fcout(options.final_output_tag+"_FC.csv");
        fcout << "chi2_osc,chi2_syst";
        for(const std::string &name: metric.GetModel().param_names)
            fcout << ",best_" << name;
        for(const std::string &name: metric.GetSysts().spline_names)
            fcout << ",best_" << name << "_osc,best_" << name << "," << name << "_throw";
        fcout << "\r\n";

        for(const auto &out: outs) {
            for(const auto &fco: out) {
                fcout << fco.chi2_osc << "," << fco.chi2_syst;
                for(size_t i = 0; i < metric.GetModel().nparams; ++i) {
                    float raw = fco.best_phys_osc.size() > (Eigen::Index)i ? fco.best_phys_osc(i) : 0.0f;
                    float val = metric.GetModel().is_log10[i] ? std::pow(10.0f, raw) : raw;
                    fcout << "," << val;
                }
                for(size_t i = 0; i < metric.GetSysts().GetNSplines(); ++i)
                    fcout << "," << (options.gof_pvalue ? 0 : fco.best_fit_osc(i)) << "," << fco.best_fit_syst(i) << "," << fco.syst_throw(i);
                fcout << "\r\n";
            }
        }
    }
}
