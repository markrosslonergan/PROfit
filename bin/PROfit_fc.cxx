#include "PROfit_common.h"

#include <set>

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

// Provenance of one FC run: the tested point and everything that shapes the
// toy distribution. Written as the "fc_meta" tree in FC.root plus a key: value
// sidecar, and checked by --reuse so a file is never silently reused for a
// different point, metric or config.
struct FCMeta {
    std::vector<std::pair<std::string, float>> nums; // test_<p>, fixed_<p>, run settings
    std::vector<std::pair<std::string, std::string>> strs;
};

static FCMeta makeFCMeta(const PROconfig &config, const PROpeller &prop, const PROmodel &model, const Eigen::VectorXf &test_point, const Eigen::VectorXf &phys_lb, const Eigen::VectorXf &phys_ub, const PROfitterConfig &scanFitConfig, const PROpt &options, const char *throw_mode) {
    FCMeta m;
    m.nums.push_back({"meta_version", 1});
    for(size_t i = 0; i < model.nparams; ++i) {
        const float v = test_point(i);
        m.nums.push_back({"test_" + model.param_names[i], model.is_log10[i] ? std::pow(10.0f, v) : v});
    }
    for(size_t i = 0; i < model.nparams; ++i)
        m.nums.push_back({"fixed_" + model.param_names[i], phys_lb(i) == phys_ub(i) ? 1.0f : 0.0f});
    m.nums.push_back({"nthread", (float)options.nthread});
    m.nums.push_back({"nuniv", (float)options.nuniv});
    m.nums.push_back({"shape_only", (float)options.shapeonly});
    m.nums.push_back({"gof", (float)options.gof_pvalue});
    m.strs.push_back({"config_hash", std::to_string(config.hash)});
    m.strs.push_back({"seed", std::to_string(options.global_seed)});
    m.strs.push_back({"metric", options.chi2});
    m.strs.push_back({"model", config.m_model_tag});
    m.strs.push_back({"throw_mode", throw_mode});
    m.strs.push_back({"preset", options.fit_preset.back()});
    m.strs.push_back({"grad_mode", PROmetric::gradientModeName(scanFitConfig.gradient_mode)});
    m.strs.push_back({"version", PROJECT_VERSION_STR});
    // Toys thrown with the MC statistical overlap (PROoverlap.h) are a different distribution.
    if(prop.overlap) m.strs.push_back({"stat_overlap", std::to_string(OverlapFingerprint(*prop.overlap))});
    // ln det M / a frozen M change the statistic of every toy and of the data.
    const PROconfig::CovarianceOptions &co = config.m_covariance_options;
    if(co.lndet) m.nums.push_back({"cov_lndet", 1.0f});
    if(co.frozen) m.nums.push_back({"cov_frozen", 1.0f});
    if((co.lndet || co.frozen) && co.reference.size() >= model.nparams) {
        for(size_t i = 0; i < model.nparams; ++i)
            m.nums.push_back({"covref_" + model.param_names[i], model.is_log10[i] ? std::pow(10.0f, co.reference[i]) : co.reference[i]});
        for(size_t i = model.nparams; i < co.reference.size(); ++i)
            if(co.reference[i] != 0.0f) m.nums.push_back({"covref_s" + std::to_string(i - model.nparams), co.reference[i]});
    }
    // Physics priors shape every toy (re-observed centres) and the data statistic.
    for(const auto &p : model.physics_priors) {
        const std::string &n = p.name;
        m.nums.push_back({"prior_center_" + n, p.center});
        m.nums.push_back({"prior_sigma_" + n, p.sigma});
        m.nums.push_back({"prior_abs_" + n, (float)p.abs});
    }
    return m;
}

static void writeFCMeta(const FCMeta &m, TFile &fout, const std::string &txtfile) {
    fout.cd();
    TTree tree("fc_meta", "FC provenance");
    std::vector<float> vals(m.nums.size());
    for(size_t i = 0; i < m.nums.size(); ++i) {
        vals[i] = m.nums[i].second;
        tree.Branch(m.nums[i].first.c_str(), &vals[i]);
    }
    std::vector<std::string> svals(m.strs.size());
    for(size_t i = 0; i < m.strs.size(); ++i) {
        svals[i] = m.strs[i].second;
        tree.Branch(m.strs[i].first.c_str(), &svals[i]);
    }
    tree.Fill();
    tree.Write();

    std::ofstream txt(txtfile);
    for(const auto &[k, v]: m.nums) txt << k << ": " << v << "\n";
    for(const auto &[k, v]: m.strs) txt << k << ": " << v << "\n";
}

// --reuse guard: the stored provenance must match this run's tested point,
// fixed and profiled parameters, toy truth, metric, model and config. Every such
// key found in either the file or this run is compared, so a profile-FC file is
// not reused for a plain FC run or vice versa. Old files without fc_meta are
// accepted with a warning. Returns false on a mismatch.
static bool checkFCMeta(const std::string &filename, const FCMeta &now) {
    TFile *f = TFile::Open(filename.c_str(), "READ");
    if(!f || f->IsZombie()) { delete f; return true; }
    TTree *tree = nullptr;
    f->GetObject("fc_meta", tree);
    if(!tree || tree->GetEntries() < 1) {
        log<LOG_WARNING>(L"%1% || --reuse: %2% predates FC provenance (no fc_meta); cannot verify it was thrown at this point, metric and config.")
            % __func__ % filename.c_str();
        f->Close(); delete f;
        return true;
    }
    std::vector<std::string> fnames, snames;
    TIter next(tree->GetListOfBranches());
    while(TBranch *br = static_cast<TBranch*>(next()))
        (std::string(br->GetClassName()) == "string" ? snames : fnames).push_back(br->GetName());
    std::vector<float> fvals(fnames.size(), 0.0f);
    std::vector<std::string*> svals(snames.size(), nullptr);
    for(size_t i = 0; i < fnames.size(); ++i) tree->SetBranchAddress(fnames[i].c_str(), &fvals[i]);
    for(size_t i = 0; i < snames.size(); ++i) tree->SetBranchAddress(snames[i].c_str(), &svals[i]);
    tree->GetEntry(0);

    // Run settings (seed, threads, preset...) may differ between the throwing run and the reuse.
    auto checked_num = [](const std::string &k) {
        for(const char *pre: {"test_", "fixed_", "profiled_", "truth_", "prior_", "cov_", "covref_"})
            if(k.rfind(pre, 0) == 0) return true;
        return k == "shape_only";
    };
    auto checked_str = [](const std::string &k) {
        return k == "config_hash" || k == "metric" || k == "model" || k == "profile_truth" || k == "stat_overlap";
    };
    std::map<std::string, float> stored_num, now_num;
    std::map<std::string, std::string> stored_str, now_str;
    for(size_t i = 0; i < fnames.size(); ++i) if(checked_num(fnames[i])) stored_num[fnames[i]] = fvals[i];
    for(size_t i = 0; i < snames.size(); ++i) if(checked_str(snames[i])) stored_str[snames[i]] = svals[i] ? *svals[i] : "";
    for(const auto &[k, v]: now.nums) if(checked_num(k)) now_num[k] = v;
    for(const auto &[k, v]: now.strs) if(checked_str(k)) now_str[k] = v;

    bool ok = true;
    std::set<std::string> keys;
    for(const auto &[k, v]: stored_num) keys.insert(k);
    for(const auto &[k, v]: now_num) keys.insert(k);
    for(const auto &k: keys) {
        const bool in_file = stored_num.count(k), in_run = now_num.count(k);
        if(in_file && in_run && std::abs(stored_num[k] - now_num[k]) <= 1e-5f * std::max(1.0f, std::abs(now_num[k]))) continue;
        log<LOG_ERROR>(L"%1% || --reuse: %2% in %3% is %4%, this run has %5%.") % __func__ % k.c_str() % filename.c_str()
            % (in_file ? std::to_string(stored_num[k]) : std::string("<absent>")).c_str()
            % (in_run ? std::to_string(now_num[k]) : std::string("<absent>")).c_str();
        ok = false;
    }
    keys.clear();
    for(const auto &[k, v]: stored_str) keys.insert(k);
    for(const auto &[k, v]: now_str) keys.insert(k);
    for(const auto &k: keys) {
        const std::string a = stored_str.count(k) ? stored_str[k] : "<absent>";
        const std::string b = now_str.count(k) ? now_str[k] : "<absent>";
        if(a == b) continue;
        log<LOG_ERROR>(L"%1% || --reuse: %2% in %3% is '%4%', this run has '%5%'.") % __func__ % k.c_str() % filename.c_str() % a.c_str() % b.c_str();
        ok = false;
    }
    f->Close(); delete f;
    return ok;
}

void run_fc(const PROconfig &config, const PROpeller &prop, PROmetric &metric, const PROdata &data, const Eigen::VectorXf &CVParams, const Eigen::VectorXf &fakeDataParams, const Eigen::VectorXf &lb, const Eigen::VectorXf &ub, const std::vector<int> &fixed, const PROfitterConfig &fitConfig, const PROfitterConfig &scanFitConfig, const PROpt &options, PROseed &myseed) {
    const PROmodel &model = metric.GetModel();
    const size_t nphys = model.nparams;
    const Eigen::VectorXf phys_lb = lb.head(nphys), phys_ub = ub.head(nphys);

    // The tested point is the -i point unless --fc-point names another one.
    Eigen::VectorXf test_point = fakeDataParams.head(nphys);
    for(const auto &[name, value]: options.fc_point) {
        const auto idx = ResolveParamIndex(name, model, metric.GetSysts(), config);
        if(!idx || *idx >= nphys) {
            log<LOG_ERROR>(L"%1% || --fc-point: '%2%' is not a physics parameter of %3%.") % __func__ % name.c_str() % config.m_model_tag.c_str();
            exit(EXIT_FAILURE);
        }
        test_point(*idx) = model.is_log10[*idx] ? std::log10(value) : value;
    }

    // Profile FC: these physics parameters float in the constrained fit.
    std::vector<uint8_t> free_mask;
    if(!options.fc_profile_params.empty()) {
        free_mask.assign(nphys, 0);
        std::set<size_t> listed;
        for(const auto &name: options.fc_profile_params) {
            const auto idx = ResolveParamIndex(name, model, metric.GetSysts(), config);
            if(!idx || *idx >= nphys) {
                log<LOG_ERROR>(L"%1% || --profile-params: '%2%' is not a physics parameter of %3%.") % __func__ % name.c_str() % config.m_model_tag.c_str();
                exit(EXIT_FAILURE);
            }
            listed.insert(*idx);
            if(phys_lb(*idx) < phys_ub(*idx)) free_mask[*idx] = 1;
            else log<LOG_WARNING>(L"%1% || --profile-params: %2% is --fix'd, so it stays pinned.") % __func__ % name.c_str();
        }
        if(listed.size() >= nphys) {
            log<LOG_ERROR>(L"%1% || --profile-params lists every physics parameter: nothing is left to test.") % __func__;
            exit(EXIT_FAILURE);
        }
        if(std::none_of(free_mask.begin(), free_mask.end(), [](uint8_t f) { return f; })) {
            log<LOG_WARNING>(L"%1% || no profiled parameter is free; running the unprofiled FC.") % __func__;
            free_mask.clear();
        }
    }
    const bool profiled = !free_mask.empty();

    for(size_t i = 0; i < nphys; ++i) {
        if(phys_lb(i) == phys_ub(i) && test_point(i) != phys_lb(i))
            log<LOG_WARNING>(L"%1% || %2% is fixed at %3% in the fits but the toys are thrown at %4% (-i): the fit cannot reach the truth.")
                % __func__ % model.param_names[i].c_str() % phys_lb(i) % test_point(i);
    }
    if(options.systs_only && !options.gof_pvalue)
        log<LOG_WARNING>(L"%1% || --syst-only pins every physics parameter, so the free fit equals the constrained one and every delta chi^2 is 0. Use --gof for a goodness-of-fit distribution.") % __func__;
    if(nphys > 2 && !profiled)
        log<LOG_INFO>(L"%1% || %2% physics parameters are all pinned in the constrained fit: this tests the full %2%D point (--profile-params profiles some of them).") % __func__ % nphys;

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

    // Fit of the dataset at the tested point: unprofiled physics pinned there,
    // profiled physics free (or pinned at its default when at_default). Returns
    // the chi^2 and the physics best fit.
    auto conditional_fit = [&](bool at_default) -> std::pair<float, Eigen::VectorXf> {
        Eigen::VectorXf flb = lb, fub = ub, seed = CVParams;
        std::vector<int> ffixed = fixed;
        for(size_t i = 0; i < nphys; ++i) {
            const bool prof_i = profiled && free_mask[i];
            seed(i) = prof_i ? model.default_val(i) : test_point(i);
            if(prof_i && !at_default) continue;
            flb(i) = seed(i);
            fub(i) = seed(i);
            ffixed[i] = 1;
        }
        GlobalFitOptions opt = GlobalFitOptions::Default;
        if(options.progress_bar) opt |= GlobalFitOptions::Progress;
        PROspec cv = FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), CVParams , true ,config.i_prime);
        GlobalFitResult fitres = run_global_fit(config, prop, data, metric, fub, flb, fitConfig, seed, cv, ffixed, opt);
        return {fitres.chi2, Eigen::VectorXf(fitres.fitter.best_fit.head(nphys))};
    };

    // Toy truth of the physics: the tested point, with profiled parameters at
    // the -i value (inject) or at their conditional best fit to the data
    // (profile construction).
    Eigen::VectorXf truth = test_point;
    Eigen::VectorXf zhat_data;
    bool snapped = false;
    float cond_chi2 = 0;
    const bool data_truth = profiled && options.fc_profile_truth == "data";
    if(data_truth || (profiled && options.pvalue)) {
        const auto [chi2_free, phys_free] = conditional_fit(false);
        const auto [chi2_def, phys_def] = conditional_fit(true);
        // A profiled parameter the data cannot see would sit wherever the minimiser
        // drifted along a flat direction: tie-break to its default instead.
        snapped = chi2_def - chi2_free < 1e-2f;
        zhat_data = snapped ? phys_def : phys_free;
        cond_chi2 = std::min(chi2_free, chi2_def);
        for(size_t i = 0; i < nphys; ++i) {
            if(!free_mask[i]) continue;
            log<LOG_INFO>(L"%1% || conditional best fit of %2% at the tested point: %3%%4% (chi^2 %5% free, %6% at its default %7%).")
                % __func__ % model.param_names[i].c_str() % zhat_data(i) % (snapped ? " (snapped to default)" : "")
                % chi2_free % chi2_def % model.default_val(i);
            if(data_truth) truth(i) = zhat_data(i);
        }
    }
    if(options.pvalue) {
        null_chi2 = profiled ? cond_chi2 : conditional_fit(false).first;
        if(null_chi2 < global_chi2)
            log<LOG_WARNING>(L"%1% || the fit at the tested point (chi^2 %2%) beat the global fit (%3%): the global minimiser missed a minimum.")
                % __func__ % null_chi2 % global_chi2;
    }
    if(profiled) {
        for(size_t i = 0; i < nphys; ++i)
            log<LOG_INFO>(L"%1% || profile FC toy truth %2% = %3% (%4%).") % __func__ % model.param_names[i].c_str() % truth(i)
                % (free_mask[i] ? (data_truth ? "profiled, data truth" : "profiled, injected truth") : "tested");
    }
    Eigen::VectorXf toy_truth = fakeDataParams;
    toy_truth.head(nphys) = truth;

    size_t FCthreads = options.nthread > options.nuniv ? options.nuniv : options.nthread;
    Eigen::MatrixXf R = CovarianceThrowRoot(config, metric.GetSysts());

    std::vector<std::vector<float>> dchi2s;
    dchi2s.reserve(FCthreads);
    std::vector<std::vector<fc_out>> outs;
    outs.reserve(FCthreads);
    std::vector<std::thread> threads;
    std::vector<FitTiming> timings(FCthreads);
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
    FCMeta meta = makeFCMeta(config, prop, model, test_point, phys_lb, phys_ub, scanFitConfig, options, throw_mode);
    if(profiled) {
        auto lin = [&](size_t i, float v) { return model.is_log10[i] ? std::pow(10.0f, v) : v; };
        for(size_t i = 0; i < nphys; ++i) meta.nums.push_back({"profiled_" + model.param_names[i], (float)free_mask[i]});
        for(size_t i = 0; i < nphys; ++i) meta.nums.push_back({"truth_" + model.param_names[i], lin(i, truth(i))});
        if(zhat_data.size()) {
            for(size_t i = 0; i < nphys; ++i)
                if(free_mask[i]) meta.nums.push_back({"zhat_data_" + model.param_names[i], lin(i, zhat_data(i))});
            meta.nums.push_back({"snapped", (float)snapped});
        }
        if(options.pvalue) meta.nums.push_back({"t_obs", null_chi2 - global_chi2});
        meta.strs.push_back({"profile_truth", options.fc_profile_truth});
    }
    if(options.reuse_dist) {
        TFile *probe = TFile::Open(FC_file.c_str(), "READ");
        if(probe && !probe->IsZombie()) {
            gen_null_dist = false;
            if(!checkFCMeta(FC_file, meta)) {
                if(!options.force) {
                    log<LOG_ERROR>(L"%1% || --reuse: %2% was not thrown for this setup (see above). Pass --force to reuse it anyway.") % __func__ % FC_file.c_str();
                    exit(EXIT_FAILURE);
                }
                log<LOG_WARNING>(L"%1% || --reuse: provenance mismatch overridden by --force.") % __func__;
            }
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
            fc_args args{todo + (i >= addone), &dchi2s.back(), &outs.back(), config, prop, metric.GetSysts(), options.chi2, toy_truth, R, scanFitConfig,(*myseed.getThreadSeeds())[i], (int)i, !options.eventbyevent, gof_mode, options.shapeonly, throw_systematics, throw_poisson, phys_lb, phys_ub, &timings[i], free_mask};


            threads.emplace_back([args, &fc_progress]() {
                        PROfit::fc_worker(args, std::ref(fc_progress));
                        });
        }
        for(auto&& t: threads) {
            t.join();
        }
        fc_progress.finish_all();
        FitTiming total;
        for(const auto &t: timings) total += t;
        total.log_summary("fc");
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
    log<LOG_INFO>(L"%1% || delta chi2 quantiles: 68.3%% %2%, 90%% %3%, 95.4%% %4%")
        % __func__ % flattened_dchi2s[0.683*flattened_dchi2s.size()] % flattened_dchi2s[0.9*flattened_dchi2s.size()]
        % flattened_dchi2s[0.954*flattened_dchi2s.size()];
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
        // Profile FC: where the constrained fit put each profiled parameter.
        std::vector<float> cbest_vals(nphys, 0.0f);
        for(size_t i = 0; i < nphys && profiled; ++i)
            if(free_mask[i]) tree.Branch(("cbest_" + model.param_names[i]).c_str(), &cbest_vals[i]);

        for(const auto &out: outs) {
            for(const auto &fco: out) {
                chi2_osc  = fco.chi2_osc;
                chi2_syst = fco.chi2_syst;
                for(size_t i = 0; i < metric.GetModel().nparams; ++i) {
                    float raw = fco.best_phys_osc.size() > (Eigen::Index)i ? fco.best_phys_osc(i) : 0.0f;
                    best_phys_vals[i] = metric.GetModel().is_log10[i] ? std::pow(10.0f, raw) : raw;
                }
                for(size_t i = 0; i < nphys && profiled; ++i)
                    cbest_vals[i] = model.is_log10[i] ? std::pow(10.0f, fco.best_phys_syst(i)) : fco.best_phys_syst(i);
                for(size_t i = 0; i < metric.GetSysts().GetNSplines(); ++i) {
                    if(!options.gof_pvalue) best_systs_osc[metric.GetSysts().spline_names[i]] = fco.best_fit_osc(i);
                    best_systs[metric.GetSysts().spline_names[i]] = fco.best_fit_syst(i);
                    syst_throw[metric.GetSysts().spline_names[i]] = fco.syst_throw(i);
                }
                tree.Fill();
            }
        }

        tree.Write();
        writeFCMeta(meta, fout, options.final_output_tag + "_FC_meta.txt");
    }
    if(gen_null_dist) {
        ofstream fcout(options.final_output_tag+"_FC.csv");
        fcout << "chi2_osc,chi2_syst";
        for(const std::string &name: metric.GetModel().param_names)
            fcout << ",best_" << name;
        for(const std::string &name: metric.GetSysts().spline_names)
            fcout << ",best_" << name << "_osc,best_" << name << "," << name << "_throw";
        for(size_t i = 0; i < nphys && profiled; ++i)
            if(free_mask[i]) fcout << ",cbest_" << model.param_names[i];
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
                for(size_t i = 0; i < nphys && profiled; ++i)
                    if(free_mask[i]) fcout << "," << (model.is_log10[i] ? std::pow(10.0f, fco.best_phys_syst(i)) : fco.best_phys_syst(i));
                fcout << "\r\n";
            }
        }
    }
}
