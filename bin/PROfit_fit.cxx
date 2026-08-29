#include "PROfit_common.h"

// Background-only fixed seed: physics pinned at the model defaults (the
// no-oscillation point for the SBN sterile models), nuisances started from CV.
// PROfitter refines it with the physics HELD fixed (a nuisances-only fit of the
// null hypothesis) and records the result as a candidate; a fit that restricts
// any pinned physics parameter to a different value (--fix, profile scans of
// that parameter) skips the seed inside PROfitter::Fit. Empty for physics-free
// models.
std::vector<FixedSeed> buildBkgOnlyFixedSeeds(const PROmodel &model, const Eigen::VectorXf &CVParams) {
    std::vector<FixedSeed> out;
    if(model.nparams == 0) return out;
    FixedSeed bkg;
    bkg.point = CVParams;
    bkg.fixed.assign(CVParams.size(), 0);
    for(size_t i = 0; i < model.nparams; ++i) {
        bkg.point((int)i) = model.default_val((int)i);
        bkg.fixed[i] = 1;
    }
    out.push_back(std::move(bkg));
    return out;
}

// Walks the collapsed reco bins and logs any with prediction < threshold,
// printing the channel, bin index/edges, prediction, and data count side-by-side.
static void logLowPredictionBins(const PROconfig &config, const Eigen::VectorXf &pred_collapsed, const Eigen::VectorXf &data_collapsed, float threshold = 1.0f, size_t var_index = 0) {
    log<LOG_INFO>(L"%1% || ----- Low-prediction reco bins (pred < %2%) -----") % __func__ % threshold;
    log<LOG_INFO>(L"%1% || %2$-30s  %3$-12s  %4$-18s  %5$-12s  %6$-12s")
        % __func__ % "channel" % "bin_in_chan" % "edges [MeV]" % "pred" % "data";
    size_t flat = 0;
    size_t n_low = 0;
    size_t global_channel_index = 0;
    for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
        for(size_t det = 0; det < config.m_num_detectors; ++det) {
            for(size_t channel = 0; channel < config.m_num_channels; ++channel) {
                std::vector<float> edges = config.GetChannelVariableBins(global_channel_index, var_index).Edges();
                const std::string &cname = config.m_channel_names[channel];
                for(size_t b = 0; b + 1 < edges.size(); ++b, ++flat) {
                    if(flat >= (size_t)pred_collapsed.size()) break;
                    float p = pred_collapsed(flat);
                    float d = data_collapsed(flat);
                    if(p < threshold) {
                        ++n_low;
                        char edge_str[32];
                        std::snprintf(edge_str, sizeof(edge_str), "[%.0f, %.0f)", edges[b], edges[b+1]);
                        log<LOG_INFO>(L"%1% || %2$-30s  %3$-12zu  %4$-18s  %5$-12.4f  %6$-12.2f")
                            % __func__ % cname.c_str() % b % edge_str % p % d;
                    }
                }
                ++global_channel_index;
            }
        }
    }
    log<LOG_INFO>(L"%1% || %2% / %3% reco bins have prediction < %4%.")
        % __func__ % n_low % (size_t)pred_collapsed.size() % threshold;
}

GlobalFitResult run_global_fit(const PROconfig &config, const PROpeller &prop, const PROdata &data, PROmetric &metric, const Eigen::VectorXf &ub, const Eigen::VectorXf &lb, const PROfitterConfig &fit_config, const Eigen::VectorXf &CVParams, const PROspec &cv, const std::vector<int> &global_fixed, GlobalFitOptions opt) {
    GlobalFitResult res(ub, lb, fit_config);
    metric.setBounds(lb, ub);

    log<LOG_INFO>(L"%1% || ########### Starting Global Best Fit Minimizing ############") % __func__;

    std::vector<std::pair<int, std::string>> progress_configs;
    progress_configs.push_back({fit_config.n_latin_points, "(1) LatinHyperCube"});
    progress_configs.push_back({fit_config.n_swarm_iterations, "(2) ParticleSwarm"});
    progress_configs.push_back({fit_config.n_localfit, "(3) BestLBFGSB"});
    progress_configs.push_back({fit_config.harmonic_num_test_points, "(4) HarmonicScan"});
    progress_configs.push_back({100, "(5) HarmonicLBFGSB"});
    MultiPROgressBar progress(progress_configs);

    bool progress_bar = (opt & GlobalFitOptions::Progress) != GlobalFitOptions::Default;
    if(progress_bar){
        progress.initialize_display();
        progress.start_display_thread(); 
        res.fitter.setProgressBar(&progress);
    }

    // The CV point seeds the full-float fit; the background-only fixed seed
    // additionally guarantees a nuisances-only fit of the null hypothesis is a
    // candidate (skipped automatically if it conflicts with --fix'd physics).
    std::vector<FixedSeed> bkg_fixed_seeds = buildBkgOnlyFixedSeeds(metric.GetModel(), CVParams);
    float best_chi2 = res.fitter.Fit(metric, std::vector<Eigen::VectorXf>{CVParams}, bkg_fixed_seeds);
    Eigen::VectorXf best_fit = res.fitter.best_fit;
    if((opt & GlobalFitOptions::FreqSeedPts) != GlobalFitOptions::Default) res.fitter.calcFreqSeedPoints(metric);

    long best_harmonic_idx = -1;
    for(size_t i=0; i< res.fitter.freq_seed_points.size(); i++){
        float chi_freq = res.fitter.freq_seed_values.at(i);
        if( chi_freq < best_chi2){
            log<LOG_INFO>(L"%1% || One of the harmonics of first pass best fit, is a lower chi :  %2% ") % __func__ % res.fitter.freq_seed_values.at(i);
            log<LOG_INFO>(L"%1% || -- at params:  %2% ") % __func__ % res.fitter.freq_seed_points.at(i);
            best_chi2 = chi_freq;
            best_fit = res.fitter.freq_seed_points.at(i);
            best_harmonic_idx = (long)i;
        }
    }
    if(best_harmonic_idx >= 0){
        // Everything downstream (global_fit_result, draw_fit_result,
        // PROfile::Plot) reads res.fitter.best_fit; keep it in sync with
        // res.chi2 so the reported best-fit point is the one this chi2
        // belongs to.
        res.fitter.best_fit = best_fit;
        log<LOG_WARNING>(L"%1% || A harmonic seed (#%2%) beat the first-pass global fit; adopting it as the global best fit with chi^2 %3%.") % __func__ % best_harmonic_idx % best_chi2;
        log<LOG_WARNING>(L"%1% || -- best harmonic seed params: %2%") % __func__ % best_fit;
    }
    res.chi2 = best_chi2;
    if(progress_bar) progress.finish_all();

    if (res.fitter.exception_string_map.empty()) {
        log<LOG_INFO>(L"%1% || No exceptions were caught from LBFGSB [ --INFO-- ]") % __func__;
    } else {
        log<LOG_INFO>(L"%1% || Some exceptions were caught in LBFGSB [ --INFO-- ]") % __func__;
        for (const auto &[msg, count] : res.fitter.exception_string_map) {
            log<LOG_INFO>(L"%1% ||  -- Exception \"%2%\" occurred %3% time(s)") % __func__ % msg.c_str() % count;
        }
    }

    log<LOG_INFO>(L"%1% || ################################################") % __func__;
    log<LOG_INFO>(L"%1% || ########### Global Best Fit Results ############") % __func__;
    log<LOG_INFO>(L"%1% || ################################################") % __func__;
    log<LOG_INFO>(L"%1% || Global Best Fit chi^2: %2%") %__func__ % best_chi2;
    log<LOG_INFO>(L"%1% || at paramters: ") % __func__;

    size_t N_phys_params = metric.GetModel().nparams;
    size_t N_nuisance = metric.GetSysts().GetNSplines();
    size_t N_params = N_phys_params + N_nuisance;

    for(size_t i = 0; i< N_params; i++){

        if(i<N_phys_params){
            log<LOG_INFO>(L"%1% || %2%  : %3% (log) %4% (nonlog) ") % __func__ % metric.GetModel().pretty_param_names[i].c_str() % best_fit(i) % pow(10,best_fit(i));
        }else{
            log<LOG_INFO>(L"%1% || %2%  :  %3% ") % __func__ % config.m_mcgen_variation_plotname_map.at(metric.GetSysts().spline_names[i-N_phys_params]).c_str() % best_fit(i);
        }
    }
    log<LOG_INFO>(L"%1% || ################################################") % __func__;

    {
        Eigen::VectorXf bf_spec_full = FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), best_fit, true, config.i_prime).Spec();
        Eigen::VectorXf bf_spec_coll = CollapseMatrix(config, bf_spec_full);
        logLowPredictionBins(config, bf_spec_coll, data.Spec(), 1.0f, config.i_prime);
    }

    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());
    if((opt & GlobalFitOptions::Correlations) != GlobalFitOptions::Default) {
        log<LOG_INFO>(L"%1% || Starting a metropolis hastings chain to estimate the covariance matrix aroud the above best fit. Run and Burn is (%2%,%3%);") % __func__%fit_config.MCMCiter % fit_config.MCMCburn;

        std::vector<int> fixed;
        for(size_t i = 0; i< global_fixed.size();i++){
            if(global_fixed.at(i) == 1)
                fixed.push_back(i);
        }
        res.mh.emplace(simple_target{metric}, adaptive_proposal(metric, dseed(PROseed::global_rng), fixed), best_fit, dseed(PROseed::global_rng));

        res.covmat = Eigen::MatrixXf::Constant(N_params, N_params, 0);
        size_t count = 0;
        const auto action = [&](const Eigen::VectorXf &value) {
            res.covmat += (value-best_fit) * (value-best_fit).transpose();
            count += 1;
        };
        std::optional<PROgressBar> mh_pbar;
        if((opt & GlobalFitOptions::Progress) != GlobalFitOptions::Default) 
            mh_pbar.emplace(int(fit_config.MCMCburn + fit_config.MCMCiter), 30, "MCMC postfit");
        res.mh->run(fit_config.MCMCburn,fit_config.MCMCiter, action, mh_pbar ? &*mh_pbar : nullptr);

        res.covmat /= count;
        Eigen::VectorXf inv_best_fit = best_fit.array().abs().max(1e-10f).inverse();
        res.fraccovmat = inv_best_fit.asDiagonal() * res.covmat * inv_best_fit.asDiagonal();

        Eigen::VectorXf inv_sqrt_diag = res.fraccovmat.diagonal().array().abs().max(1e-10f).sqrt().inverse();
        res.corrmat = inv_sqrt_diag.asDiagonal() * res.fraccovmat * inv_sqrt_diag.asDiagonal();

        log<LOG_INFO>(L"%1% || Finished the metropolis hastings chain ") % __func__;
    }

    bool preerr = (opt & GlobalFitOptions::PrefitErrorBand) != GlobalFitOptions::Default;
    bool mcmcpre = (opt & GlobalFitOptions::MCMCPrefitErrorBand) != GlobalFitOptions::Default;
    bool binwidth_scale = (opt & GlobalFitOptions::BinWidthScaled) != GlobalFitOptions::Default;

    // The error-band chains fix all physics params plus any --fix'd splines; if
    // that leaves zero free parameters the chain never moves and its band is
    // exactly Gaussian throws of the covariance around the best-fit spectrum —
    // computed analytically instead (getCovarianceOnlyErrorBand), skipping
    // MCMCburn+MCMCiter identical spectrum evaluations.
    std::vector<int> errband_fixed_pars;
    for(size_t i = 0; i < N_phys_params; ++i) errband_fixed_pars.push_back(i);
    for(size_t i = N_phys_params; i< global_fixed.size();i++){
        if(global_fixed.at(i)==1)errband_fixed_pars.push_back(i);
    }
    const bool errband_chain_degenerate = errband_fixed_pars.size() >= N_params;
    // Delta-function stand-ins for the chain's parameter-posterior outputs.
    const auto degenerate_mcmc_params = [&](std::vector<TH1D> &posts, Eigen::MatrixXf &covar, Eigen::VectorXf &lo, Eigen::VectorXf &hi) {
        for(size_t i = 0; i < N_nuisance; ++i) {
            posts.emplace_back("", (";"+config.m_mcgen_variation_plotname_map.at(metric.GetSysts().spline_names[i])).c_str(), 60, -3, 3);
            posts.back().Fill(best_fit(N_phys_params + i));
        }
        covar = Eigen::MatrixXf::Zero(N_nuisance, N_nuisance);
        lo = Eigen::VectorXf::Zero(N_nuisance);
        hi = Eigen::VectorXf::Zero(N_nuisance);
    };

    if(preerr || mcmcpre) {
        log<LOG_INFO>(L"%1% || Starting global getErrorBand() ") % __func__;
        if(mcmcpre && errband_chain_degenerate) {
            log<LOG_INFO>(L"%1% || No free nuisance parameters; computing the pre-fit error band analytically from the covariance instead of MCMC.") % __func__;
            res.err_band = getCovarianceOnlyErrorBand(config, prop, metric.GetSysts(), metric.GetModel(), best_fit, binwidth_scale, config.i_prime);
            degenerate_mcmc_params(res.priors, res.prior_covariance, res.prior_param_lo, res.prior_param_hi);
        } else if(mcmcpre) {
            Metropolis mh_pre(prior_only_target{metric}, adaptive_proposal(metric, dseed(PROseed::global_rng), errband_fixed_pars), best_fit, dseed(PROseed::global_rng));
            std::optional<PROgressBar> errband_pre_pbar;
            if(progress_bar) errband_pre_pbar.emplace(int(fit_config.MCMCburn + fit_config.MCMCiter), 30, "MCMC prefit");
            res.err_band = getMCMCErrorBand(mh_pre, fit_config.MCMCburn, fit_config.MCMCiter, config, prop, metric, best_fit, res.priors, res.prior_covariance, res.prior_param_lo, res.prior_param_hi, binwidth_scale, config.i_prime, errband_pre_pbar ? &*errband_pre_pbar : nullptr);
        } else {
            // Keep the global RNG stream where existing seeded runs expect it:
            // this branch used to construct an unused prefit Metropolis chain,
            // consuming two seed draws before getErrorBand's own throws.
            dseed(PROseed::global_rng);
            dseed(PROseed::global_rng);
            res.err_band = getErrorBand(config, prop, metric.GetSysts(), metric.GetModel(), cv ,CVParams, binwidth_scale, config.i_prime);
        }
    }

    if((opt & GlobalFitOptions::PostFitErrorBand) != GlobalFitOptions::Default) {
        log<LOG_INFO>(L"%1% || Starting global getPostFitErrorBand() ") % __func__;
        if(errband_chain_degenerate) {
            log<LOG_INFO>(L"%1% || No free nuisance parameters; computing the post-fit error band analytically from the data-constrained covariance instead of MCMC.") % __func__;
            res.post_err_band = getCovarianceOnlyErrorBand(config, prop, metric.GetSysts(), metric.GetModel(), best_fit, binwidth_scale, config.i_prime, data.Spec());
            degenerate_mcmc_params(res.posteriors, res.spline_covariance, res.post_param_lo, res.post_param_hi);
        } else {
            Metropolis mh_post(simple_target{metric}, adaptive_proposal(metric, dseed(PROseed::global_rng), errband_fixed_pars), best_fit, dseed(PROseed::global_rng));
            std::optional<PROgressBar> errband_post_pbar;
            if(progress_bar) errband_post_pbar.emplace(int(fit_config.MCMCburn + fit_config.MCMCiter), 30, "MCMC postfit band");
            // data.Spec() enables the data-constrained posterior pull of the
            // covariance-type systematics (post-fit band only; the pre-fit call
            // above stays unconstrained). Lost in merge 7078697, restored.
            res.post_err_band = getMCMCErrorBand(mh_post, fit_config.MCMCburn, fit_config.MCMCiter, config, prop, metric, best_fit, res.posteriors, res.spline_covariance, res.post_param_lo, res.post_param_hi, binwidth_scale,config.i_prime, errband_post_pbar ? &*errband_post_pbar : nullptr, data.Spec());
        }
    }
    
    return res;
}

// One-page summary of the harmonic seed scan: the Delta chi2(freq) curve, a
// dotted vertical line per surviving seed, the adopted global BF frequency,
// and a side panel with the scan configuration and per-seed values.
void draw_harmonic_scan_pdf(const GlobalFitResult &fitres, const PROfitterConfig &fit_config, const PROmodel &model, const std::string &filename) {
    const std::vector<float> &pos = fitres.fitter.harmonic_scan_pos;
    const std::vector<float> &chi = fitres.fitter.harmonic_scan_chi;
    if(pos.size() < 2) {
        log<LOG_INFO>(L"%1% || No harmonic scan curve available (scan skipped or empty); not drawing %2%.") % __func__ % filename.c_str();
        return;
    }

    // Delta chi2 reference: the lowest chi2 seen anywhere (scan curve, refit
    // seeds, adopted global BF) so the plotted curve never dips below zero.
    float ref = *std::min_element(chi.begin(), chi.end());
    for(float v : fitres.fitter.freq_seed_values) ref = std::min(ref, v);
    if(std::isfinite(fitres.chi2)) ref = std::min(ref, fitres.chi2);

    std::vector<float> dchi(chi.size());
    float dmax = 0;
    for(size_t i = 0; i < chi.size(); ++i) {
        dchi[i] = chi[i] - ref;
        dmax = std::max(dmax, dchi[i]);
    }
    float seed_dmax = 0;
    for(float v : fitres.fitter.freq_seed_values) seed_dmax = std::max(seed_dmax, v - ref);

    // Clip the y range to where seeds live; the far tail (often hundreds of
    // chi2 units up) flattens the structure that matters. The full curve is
    // saved as the harmonic_scan TGraph in the ROOT output.
    float ymax = std::min(dmax * 1.05f, std::max(2.0f * seed_dmax + 5.0f, fit_config.harmonic_refit_window * 1.2f));
    if(!(ymax > 0)) ymax = 1.0f;
    const bool clipped = ymax < dmax;

    const std::string fname = model.nparams ? model.pretty_param_names[0] : std::string("frequency");

    TCanvas c("harmonic_scan_c", "Harmonic Scan", 1400, 800);
    TPad plot_pad("hs_plot", "", 0.0, 0.0, 0.70, 1.0);
    TPad text_pad("hs_text", "", 0.70, 0.0, 1.0, 1.0);
    plot_pad.Draw();
    text_pad.Draw();

    plot_pad.cd();
    plot_pad.SetGridy();
    plot_pad.SetLeftMargin(0.11);
    TGraph curve(pos.size(), pos.data(), dchi.data());
    curve.SetTitle(("Harmonic seed scan;" + fname + " (scan space);#Delta#chi^{2}").c_str());
    curve.SetLineWidth(2);
    curve.SetLineColor(kAzure + 2);
    curve.SetMinimum(0);
    curve.SetMaximum(ymax);
    curve.Draw("AL");

    // Dotted vertical line + star marker at each surviving seed.
    std::vector<std::unique_ptr<TLine>> seed_lines;
    std::vector<float> seed_x, seed_y;
    for(size_t i = 0; i < fitres.fitter.freq_seed_points.size(); ++i) {
        const float x = fitres.fitter.freq_seed_points[i](0);
        const float y = fitres.fitter.freq_seed_values[i] - ref;
        seed_x.push_back(x);
        seed_y.push_back(std::min(y, ymax));
        auto l = std::make_unique<TLine>(x, 0.0, x, ymax);
        l->SetLineColor(kRed + 1);
        l->SetLineStyle(3);
        l->SetLineWidth(2);
        l->Draw("same");
        seed_lines.push_back(std::move(l));
    }
    TGraph seeds(seed_x.size(), seed_x.data(), seed_y.data());
    seeds.SetMarkerStyle(29);
    seeds.SetMarkerSize(2.0);
    seeds.SetMarkerColor(kRed + 1);
    if(!seed_x.empty()) seeds.Draw("P same");

    // The adopted global best-fit frequency.
    TLine bf_line;
    const bool have_bf = fitres.fitter.best_fit.size() > 0;
    if(have_bf) {
        bf_line.SetLineColor(kGreen + 2);
        bf_line.SetLineStyle(7);
        bf_line.SetLineWidth(2);
        bf_line.DrawLine(fitres.fitter.best_fit(0), 0.0, fitres.fitter.best_fit(0), ymax);
    }

    TLegend leg(0.55, 0.72, 0.89, 0.89);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.AddEntry(&curve, "Harmonic scan #Delta#chi^{2}", "l");
    if(!seed_x.empty()) leg.AddEntry(&seeds, "Kept seed points", "p");
    if(have_bf) leg.AddEntry(&bf_line, "Global best fit", "l");
    leg.Draw();

    // Side panel: scan configuration and per-seed values.
    text_pad.cd();
    TPaveText info(0.02, 0.02, 0.98, 0.98, "NDC");
    info.SetBorderSize(0);
    info.SetFillColor(kWhite);
    info.SetTextAlign(12);
    info.SetTextFont(42);
    info.SetTextSize(0.028);
    char buf[256];
    info.AddText("#font[62]{Harmonic seed scan summary}");
    snprintf(buf, sizeof buf, "scan mode: %d (%s)", fit_config.harmonic_scan_mode,
            fit_config.harmonic_scan_mode == 2 ? "full profile" : fit_config.harmonic_scan_mode == 1 ? "physics fit" : "eval at BF");
    info.AddText(buf);
    snprintf(buf, sizeof buf, "points scanned: %zu on [%.3g, %.3g]", pos.size(), pos.front(), pos.back());
    info.AddText(buf);
    snprintf(buf, sizeof buf, "dense window: [%.3g, %.3g], %zu pts", fit_config.harmonic_dense_lo, fit_config.harmonic_dense_hi, fit_config.harmonic_num_test_points);
    info.AddText(buf);
    snprintf(buf, sizeof buf, "refine: %zu round(s), d#chi^{2} > %.3g", fit_config.harmonic_refine_rounds, fit_config.harmonic_refine_dchi);
    info.AddText(buf);
    snprintf(buf, sizeof buf, "persistence: rel %.3g, floor %.3g, cap %.3g", fit_config.harmonic_persistence_rel, fit_config.harmonic_persistence_floor, fit_config.harmonic_prominence_threshold);
    info.AddText(buf);
    snprintf(buf, sizeof buf, "phys ladder: %zu pts/dim", fit_config.harmonic_phys_ladder);
    info.AddText(buf);
    snprintf(buf, sizeof buf, "min spacing: %.3g, seeds: %zu-%zu", fit_config.harmonic_min_spacing_log, fit_config.harmonic_min_num_seeds, fit_config.harmonic_max_num_seeds);
    info.AddText(buf);
    snprintf(buf, sizeof buf, "refit window: %.3g", fit_config.harmonic_refit_window);
    info.AddText(buf);
    snprintf(buf, sizeof buf, "global BF #chi^{2}: %.4g", fitres.chi2);
    info.AddText(buf);
    if(have_bf) {
        snprintf(buf, sizeof buf, "global BF freq: %.4g", fitres.fitter.best_fit(0));
        info.AddText(buf);
    }
    if(clipped) {
        snprintf(buf, sizeof buf, "y clipped at %.3g (curve max %.3g)", ymax, dmax);
        info.AddText(buf);
    }
    info.AddText("#font[62]{Kept seeds (freq, #Delta#chi^{2}):}");
    const size_t n_list = std::min<size_t>(fitres.fitter.freq_seed_points.size(), 10);
    for(size_t i = 0; i < n_list; ++i) {
        snprintf(buf, sizeof buf, "  seed %zu: %.4g, %.4g", i, fitres.fitter.freq_seed_points[i](0), fitres.fitter.freq_seed_values[i] - ref);
        info.AddText(buf);
    }
    if(fitres.fitter.freq_seed_points.size() > n_list) {
        snprintf(buf, sizeof buf, "  (+%zu more)", fitres.fitter.freq_seed_points.size() - n_list);
        info.AddText(buf);
    }
    info.Draw();

    // Match the PDF page to the landscape canvas (default paper is portrait,
    // which top-aligns the canvas and leaves the lower half blank).
    float paper_w, paper_h;
    gStyle->GetPaperSize(paper_w, paper_h);
    gStyle->SetPaperSize(28.0f, 16.0f);
    c.Print(filename.c_str());
    gStyle->SetPaperSize(paper_w, paper_h);
    log<LOG_INFO>(L"%1% || Wrote harmonic scan summary to %2%") % __func__ % filename.c_str();
}

std::map<std::string, TObject *> draw_fit_result(const PROconfig &config, const PROpeller &prop, const PROmodel &model, const PROsyst &syst, PROmetric &metric, const PROspec &cv, const PROdata &data, const GlobalFitResult &fitres, const std::string &prefix, PlotOptions popt, PlotBounds pbounds, bool plot_channel_ratios) {
    std::map<std::string, TObject *> drawn_objs;

    // Harmonic-scan diagnostics: the scan curve chi2(freq) and the refit seed
    // points, so a missed basin is visible by inspection (overlay a PROfile
    // physics curve on these -- every profile basin should contain a seed).
    if(!fitres.fitter.harmonic_scan_pos.empty()) {
        TGraph *scan_g = new TGraph(fitres.fitter.harmonic_scan_pos.size(),
                fitres.fitter.harmonic_scan_pos.data(), fitres.fitter.harmonic_scan_chi.data());
        scan_g->SetName("harmonic_scan");
        scan_g->SetTitle("Harmonic frequency scan;log_{10}(#Deltam^{2});#chi^{2}");
        drawn_objs["harmonic_scan"] = scan_g;

        std::vector<float> seed_x, seed_y;
        for(size_t i = 0; i < fitres.fitter.freq_seed_points.size(); ++i) {
            seed_x.push_back(fitres.fitter.freq_seed_points[i](0));
            seed_y.push_back(fitres.fitter.freq_seed_values[i]);
        }
        TGraph *seed_g = new TGraph(seed_x.size(), seed_x.data(), seed_y.data());
        seed_g->SetName("harmonic_seeds");
        seed_g->SetTitle("Harmonic seed points;log_{10}(#Deltam^{2});#chi^{2}");
        seed_g->SetMarkerStyle(29);
        seed_g->SetMarkerSize(1.5);
        drawn_objs["harmonic_seeds"] = seed_g;
    }

    size_t N_params = model.nparams + syst.GetNSplines();
    size_t N_phys_params = model.nparams;
    std::vector<std::string> param_names;
    if(fitres.corrmat.size()) {
        TH2D corrhist("crh", "", N_params, 0, N_params, N_params, 0, N_params);
        TH2D fraccovhist("fch", "", N_params, 0, N_params, N_params, 0, N_params);
        TH2D covhist("ch", "", N_params, 0, N_params, N_params, 0, N_params);
        for(size_t i = 0; i < N_params; ++i) {
            std::string label = i < N_phys_params 
                ? model.pretty_param_names[i]
                : config.m_mcgen_variation_plotname_map.at(syst.spline_names[i-N_phys_params]).c_str();
            param_names.push_back(label);
            covhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            covhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            fraccovhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            fraccovhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            corrhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            corrhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            for(size_t j = 0; j < N_params; ++j) {
                covhist.SetBinContent(i+1, j+1, fitres.covmat(i,j));
                fraccovhist.SetBinContent(i+1, j+1, fitres.fraccovmat(i,j));
                corrhist.SetBinContent(i+1, j+1, fitres.corrmat(i,j));
            }
        }
        TCanvas c1;
        corrhist.SetMaximum(1);
        corrhist.SetMinimum(-1);
        covhist.SetMaximum(1);
        covhist.SetMinimum(-1);
        fraccovhist.SetMaximum(100);
        fraccovhist.SetMinimum(-100);
        //covhist.Draw("colz");
        //c1.Print((prefix+"_postfit_cov.pdf").c_str());
        //fraccovhist.Draw("colz");
        //c1.Print((prefix+"_postfit_fraccov.pdf").c_str());
        c1.SetLeftMargin(0.18);   
        corrhist.SetTitle("Post-Fit Correlation Matrix");
        corrhist.Draw("colz");
        gPad->Update();

        TLine line;
        line.SetLineColor(kBlack);
        line.SetLineWidth(2);
        line.DrawLine(N_phys_params, 0, N_phys_params, N_params);
        line.DrawLine(0, N_phys_params, N_params, N_phys_params);
        c1.Print((prefix+"_postfit_correlation_matrix.pdf").c_str());
        drawn_objs["post_fit_param_correlations"] = corrhist.Clone();
    }

    if(fitres.mh) {
        fitres.mh->plot_autocorrelation(prefix+"_corrmat_mcmc_autocorrelation.pdf", param_names, &drawn_objs);
        std::string name = prefix+"_mcmc_chain";
        drawn_objs[name] = new TTree(name.c_str(), name.c_str());
        TTree *tree = (TTree*)drawn_objs[name];
        Eigen::VectorXf v = Eigen::VectorXf::Zero(N_params);
        for(size_t i = 0; i < model.nparams; ++i) {
            tree->Branch(model.param_names[i].c_str(), &v(i));
        }
        for(size_t i = model.nparams; i < N_params; ++i) {
            const std::string &sname = syst.spline_names[i-model.nparams];
            std::string::size_type l = sname.find(':');
            // TODO: This only handles names with a single colon in them.
            // I don't think we ever have more than that, it's really just meant for the 'flat' and 'norm' systs.
            if(l != std::string::npos) {
                std::string bname = sname;
                bname[l] = '_';
                tree->Branch(bname.c_str(), &v(i));
            } else {
                tree->Branch(sname.c_str(), &v(i));
            }
        }
        for(const auto &p : fitres.mh->chain) {
            v = p;
            tree->Fill();
        }
    }

    if(fitres.fitter.best_fit.size()) {
        // NActiveBins == total bins unless a fit-region mask (e.g. PROjector) is installed.
        std::string hname = "#chi^{2}/nbins = " + to_string(fitres.chi2) + "/" + to_string(config.NActiveBins(config.i_prime));
        PROspec bf = FillSpectra(config, prop, syst, model, fitres.fitter.best_fit, true, config.i_prime);
        // Concatenated bins across all channels share no common x-axis, so use bin-index axis.
        TH1D post_hist("ph", hname.c_str(), config.m_num_variable_bins_total_collapsed[config.i_prime], 0, config.m_num_variable_bins_total_collapsed[config.i_prime]);
        TH1D pre_hist("prh", hname.c_str(), config.m_num_variable_bins_total_collapsed[config.i_prime], 0, config.m_num_variable_bins_total_collapsed[config.i_prime]);
        for(size_t i = 0; i < config.m_num_variable_bins_total_collapsed[config.i_prime]; ++i) {
            post_hist.SetBinContent(i+1, bf.Spec()(i));
            pre_hist.SetBinContent(i+1, cv.Spec()(i));
        }

        std::vector<TPaveText> texts;
        TPaveText chi2text(0.55, 0.50, 0.85, 0.58, "NDC");
        chi2text.AddText(hname.c_str());
        chi2text.SetFillColor(0);
        chi2text.SetBorderSize(0);
        chi2text.SetTextAlign(12);
        texts.push_back(chi2text);

        std::map<std::string, TObject *> tmp_objs = plot_channels((prefix+"_hists.pdf"), config, cv, bf, data, fitres.err_band, fitres.post_err_band, texts, pbounds, popt, config.i_prime, false, plot_channel_ratios, nullptr, &metric, &bf);
        for(const auto &[name, obj] : tmp_objs)
            drawn_objs[name] = obj;
    }

    TCanvas c;
    if(fitres.posteriors.size()) {
        c.Print((prefix+"_postfit_posteriors.pdf[").c_str());
        for(auto &h: fitres.posteriors) {
            TH1 *h1 = (TH1*)h.Clone();
            h1->Draw("hist");
            drawn_objs[h1->GetXaxis()->GetTitle()+std::string("_posterior")] = h1;
            c.Print((prefix+"_postfit_posteriors.pdf").c_str());
        }
        c.Print((prefix+"_postfit_posteriors.pdf]").c_str());
    }

    if(fitres.spline_covariance.size()) {
        Eigen::VectorXf inv_sqrt_diag_nuis = fitres.spline_covariance.diagonal().array().abs().max(1e-10f).sqrt().inverse();
        Eigen::MatrixXf corrmat_nuis = inv_sqrt_diag_nuis.asDiagonal() * fitres.spline_covariance * inv_sqrt_diag_nuis.asDiagonal();

        TH2F spline_cov("postfit_corr_nuisance_only", "", corrmat_nuis.cols(), 0, corrmat_nuis.cols(), corrmat_nuis.rows(), 0, corrmat_nuis.rows());
        TH2F spline_cov_cov("postfit_cov_nuisance_only", "", fitres.spline_covariance.cols(), 0, fitres.spline_covariance.cols(), fitres.spline_covariance.rows(), 0, fitres.spline_covariance.rows());
        for(int i = 0; i < corrmat_nuis.cols(); ++i) {
            spline_cov.GetXaxis()->SetBinLabel(i+1, config.m_mcgen_variation_plotname_map.at(syst.spline_names[i]).c_str());
            spline_cov.GetYaxis()->SetBinLabel(i+1, config.m_mcgen_variation_plotname_map.at(syst.spline_names[i]).c_str());
            spline_cov_cov.GetXaxis()->SetBinLabel(i+1, config.m_mcgen_variation_plotname_map.at(syst.spline_names[i]).c_str());
            spline_cov_cov.GetYaxis()->SetBinLabel(i+1, config.m_mcgen_variation_plotname_map.at(syst.spline_names[i]).c_str());
            for(int j = 0; j < corrmat_nuis.rows(); ++j) {
                spline_cov.SetBinContent(i+1, j+1, corrmat_nuis(i,j));
                spline_cov_cov.SetBinContent(i+1, j+1, fitres.spline_covariance(i,j));
            }
        }
        drawn_objs["postfit_corr_nuis"] = spline_cov.Clone();
        spline_cov.Draw("colz");
        spline_cov.SetMaximum(1);
        spline_cov.SetMinimum(-1);

        c.Print((prefix+"_postfit_correlation_matrix_nuisance_only.pdf").c_str());
    }

    return drawn_objs;
}
