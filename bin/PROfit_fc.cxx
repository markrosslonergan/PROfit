#include "PROfit_common.h"

#include "TBranch.h"
#include "TFile.h"
#include "TTree.h"

// Read a previously written <tag>_<out>_FC.root back into fc_out records so the
// p-value calculation can re-run without re-throwing universes (--reuse).
// Returns an empty vector (after logging why) if the file is unreadable or its
// contents cannot support the requested mode (e.g. --pval from a --gof file).
static std::vector<fc_out> readAnalysisFromFile(const std::string& filename,
                                                const PROmodel &model,
                                                const PROsyst &systs,
                                                bool gof_mode_requested) {
    std::vector<fc_out> entries;

    TFile* inFile = TFile::Open(filename.c_str(), "READ");
    if (!inFile || inFile->IsZombie()) {
        log<LOG_ERROR>(L"%1% || Error: Could not open file %2%") % __func__ % filename.c_str();
        return entries;
    }

    TTree* tree = nullptr;
    inFile->GetObject("tree", tree);
    if (!tree) {
        log<LOG_ERROR>(L"%1% || Error: Could not find TTree 'tree' in %2%") % __func__ % filename.c_str();
        inFile->Close();
        delete inFile;
        return entries;
    }

    TBranch* b_osc  = tree->GetBranch("chi2_osc");
    TBranch* b_sosc = tree->GetBranch("best_systs_osc");

    if ((!b_osc || !b_sosc) && !gof_mode_requested) {
        log<LOG_ERROR>(L"%1% || ERROR: The reused file '%2%' is missing oscillation branches ('chi2_osc' / 'best_systs_osc').")
            % __func__ % filename.c_str();
        log<LOG_ERROR>(L"%1% || Cannot compute standard Feldman-Cousins p-values (--pval) without oscillation fit data.") % __func__;
        log<LOG_ERROR>(L"%1% || Please re-run fresh throws without --gof to generate a complete distribution.") % __func__;
        inFile->Close();
        delete inFile;
        return entries;
    }

    float chi2_osc = 0;
    float chi2_syst = 0;
    std::map<std::string, float>* p_best_systs_osc = nullptr;
    std::map<std::string, float>* p_best_systs = nullptr;
    std::map<std::string, float>* p_syst_throw = nullptr;

    // One float per physics parameter, matching the "best_<param_name>" write branches.
    std::vector<float> phys_param_vals(model.nparams, 0.0f);

    if (b_osc) tree->SetBranchAddress("chi2_osc", &chi2_osc);
    tree->SetBranchAddress("chi2_syst", &chi2_syst);
    if (b_sosc) tree->SetBranchAddress("best_systs_osc", &p_best_systs_osc);
    tree->SetBranchAddress("best_systs", &p_best_systs);
    tree->SetBranchAddress("syst_throw", &p_syst_throw);

    for (size_t i = 0; i < model.nparams; ++i) {
        std::string branch_name = "best_" + model.param_names[i];
        if (tree->GetBranch(branch_name.c_str())) {
            tree->SetBranchAddress(branch_name.c_str(), &phys_param_vals[i]);
        } else {
            log<LOG_WARNING>(L"%1% || Warning: Branch %2% not found in file.") % __func__ % branch_name.c_str();
            phys_param_vals[i] = 0.0f;
        }
    }

    size_t n_splines = systs.GetNSplines();

    Long64_t nEntries = tree->GetEntries();
    entries.reserve(nEntries);

    bool non_zero_osc_found = false;

    for (Long64_t entry = 0; entry < nEntries; ++entry) {
        // Reset buffers to sentinel value before reading entry
        chi2_osc = -999.0f;
        chi2_syst = -999.0f;

        tree->GetEntry(entry);

        // Only count oscillation fits as valid if the branch existed AND yielded a physical chi2
        if (b_osc && chi2_osc >= 0.0f) {
            non_zero_osc_found = true;
        }

        fc_out fco;
        fco.chi2_osc = chi2_osc;
        fco.chi2_syst = chi2_syst;

        // Branches store physics parameters in linear space; undo the pow10 the
        // writer applied for log10-space model parameters.
        fco.best_phys_osc.resize(model.nparams);
        for (size_t i = 0; i < model.nparams; ++i) {
            float val = phys_param_vals[i];
            if (model.is_log10[i]) {
                fco.best_phys_osc(i) = (val > 0.0f) ? std::log10(val) : 0.0f;
            } else {
                fco.best_phys_osc(i) = val;
            }
        }

        fco.best_fit_osc.setZero(n_splines);
        fco.best_fit_syst.setZero(n_splines);
        fco.syst_throw.setZero(n_splines);

        for (size_t i = 0; i < n_splines; ++i) {
            const std::string& spline_name = systs.spline_names[i];

            if (p_best_systs_osc && p_best_systs_osc->count(spline_name)) {
                fco.best_fit_osc(i) = (*p_best_systs_osc)[spline_name];
            }
            if (p_best_systs && p_best_systs->count(spline_name)) {
                fco.best_fit_syst(i) = (*p_best_systs)[spline_name];
            }
            if (p_syst_throw && p_syst_throw->count(spline_name)) {
                fco.syst_throw(i) = (*p_syst_throw)[spline_name];
            }
        }

        entries.push_back(fco);
    }

    inFile->Close();
    delete inFile;

    if (!gof_mode_requested && !non_zero_osc_found) {
        log<LOG_ERROR>(L"%1% || ERROR: The reused file '%2%' contains no valid oscillation fits (chi2_osc < 0 everywhere).")
            % __func__ % filename.c_str();
        log<LOG_ERROR>(L"%1% || This occurs when reusing a file generated with --gof.") % __func__;
        log<LOG_ERROR>(L"%1% || Cannot compute standard Feldman-Cousins p-values (--pval) from a GOF-only distribution.") % __func__;
        log<LOG_ERROR>(L"%1% || Please re-run fresh throws without --gof to generate a full distribution.") % __func__;
        entries.clear();
        return entries;
    }

    if (gof_mode_requested && non_zero_osc_found) {
        log<LOG_INFO>(L"%1% || Reusing full Feldman-Cousins distribution for GOF evaluation (extracting chi2_syst).") % __func__;
    }

    return entries;
}

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

    const bool throw_systematics = !options.fc_stat_only_throws;
    const bool throw_poisson = !options.fc_syst_only_throws;

    const char *throw_mode = options.fc_stat_only_throws ? "stat-only"
                            : options.fc_syst_only_throws ? "syst-only"
                            : "stat+syst";

    log<LOG_INFO>(
        L"%1% || FC pseudo-experiment mode: %2% "
        L"(systematic throws=%3%, Poisson throws=%4%).")
        % __func__ % throw_mode % throw_systematics % throw_poisson;

    // --reuse: read a previous <tag>_<out>_FC.root back instead of throwing new
    // universes; fall back to fresh generation if the file isn't there.
    size_t nuniv = options.nuniv;
    const std::string FC_file = options.final_output_tag + "_FC.root";
    bool gen_null_dist = true;
    if(options.reuse_dist) {
        TFile *probe = TFile::Open(FC_file.c_str(), "READ");
        if(probe && !probe->IsZombie()) {
            gen_null_dist = false;
        } else {
            log<LOG_WARNING>(L"%1% || --reuse requested but %2% is not readable; generating fresh throws.")
                % __func__ % FC_file.c_str();
        }
        if(probe) { probe->Close(); delete probe; }
    }

    std::vector<std::pair<int, std::string>> fc_PB_configs;
    for (size_t i = 0; i < FCthreads; ++i) {
            fc_PB_configs.push_back({int(options.nuniv/FCthreads), "Thread " + std::to_string(i)});
    }
    MultiPROgressBar fc_progress(fc_PB_configs);

    if(gen_null_dist) {
        fc_progress.initialize_display();
        fc_progress.start_display_thread();


        for(size_t i = 0; i < FCthreads; i++) {
            dchi2s.emplace_back();
            outs.emplace_back();
            fc_args args{todo + (i >= addone), &dchi2s.back(), &outs.back(), config, prop, metric.GetSysts(), options.chi2, fakeDataParams, L, scanFitConfig,(*myseed.getThreadSeeds())[i], (int)i, !options.eventbyevent, gof_mode, options.shapeonly, throw_systematics, throw_poisson};


            threads.emplace_back([args, &fc_progress]() {
                        PROfit::fc_worker(args, std::ref(fc_progress));
                        });
        }
        for(auto&& t: threads) {
            t.join();
        }
        fc_progress.finish_all();
    } else {
        std::vector<fc_out> flat_outs = readAnalysisFromFile(FC_file, metric.GetModel(), metric.GetSysts(), options.gof_pvalue);
        if(flat_outs.empty()) {
            log<LOG_ERROR>(L"%1% || Aborting execution due to invalid or unreadable Feldman-Cousins distribution.") % __func__;
            exit(EXIT_FAILURE);
        }
        nuniv = flat_outs.size();
        outs.push_back(std::move(flat_outs));
    }

    // Unified flattening across fresh generation and reused distributions.
    // fc_worker fills dchi2s with exactly this quantity, so the fresh path is unchanged.
    std::vector<float> flattened_dchi2s;
    std::vector<float> flattened_syst_chi2;
    for(const auto &out: outs) {
        for(const auto &fco: out) {
            flattened_dchi2s.push_back(options.gof_pvalue ? fco.chi2_syst : fco.chi2_syst - fco.chi2_osc);
            flattened_syst_chi2.push_back(fco.chi2_syst);
        }
    }
    std::sort(flattened_dchi2s.begin(), flattened_dchi2s.end());
    std::sort(flattened_syst_chi2.begin(), flattened_syst_chi2.end());
    log<LOG_INFO>(L"%1% || 90%% Feldman-Cousins delta chi2 after throwing %2% universes is %3%")
        % __func__ % nuniv % flattened_dchi2s[0.9*flattened_dchi2s.size()];
    if(options.gof_pvalue) {
        log<LOG_ERROR>(L"%1% || All: %2% ") % __func__ % flattened_syst_chi2;
        log<LOG_ERROR>(L"%1% || chi: %2% ") % __func__ % global_chi2;
        auto it = std::lower_bound(flattened_syst_chi2.begin(), flattened_syst_chi2.end(), global_chi2);
        size_t index =  std::distance(flattened_syst_chi2.begin(),it);
        size_t count_above = flattened_syst_chi2.size()-index;
        float pval = (float)count_above/(float)nuniv;
        log<LOG_ERROR>(L"%1% || Finished throws. %2% %3%") % __func__ % index % count_above;
        log<LOG_ERROR>(L"%1% || GOF pval after throwing %2% universes is %3%") % __func__ % nuniv % pval ;
    }
    if(options.pvalue) {
        log<LOG_ERROR>(L"%1% || All Delta Chis: %2% ") % __func__ % flattened_dchi2s;
        log<LOG_ERROR>(L"%1% || Delta chi bkg-min: %2% ") % __func__ % float(null_chi2 - global_chi2);
        auto itFC = std::lower_bound(flattened_dchi2s.begin(), flattened_dchi2s.end(), float(null_chi2-global_chi2));
        size_t indexFC =  std::distance(flattened_dchi2s.begin(),itFC);
        size_t count_aboveFC = flattened_dchi2s.size()-indexFC;
        float pvalFC = (float)count_aboveFC/(float)nuniv;

        log<LOG_ERROR>(L"%1% || Finished throws. %2% %3%") % __func__ % indexFC % count_aboveFC;
        log<LOG_ERROR>(L"%1% || FC Corrected pval after throwing %2% universes is %3%") % __func__ % nuniv % pvalFC ;
    }

    // Under --reuse the distribution came verbatim from <tag>_FC.root: rewriting
    // it (and the CSV) can only lose information — a crash mid-RECREATE destroys
    // the archive, and --gof --reuse would drop the stored best_systs_osc. Only
    // freshly generated throws are persisted.
    if(gen_null_dist) {
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
    if(gen_null_dist) {
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
