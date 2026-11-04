#include "PROfit_common.h"
#include "TLatex.h"

void run_plot(const PROconfig &config, const PROpeller &prop, const PROmetric &metric, const PROmodel &model, const std::vector<PROsyst> &variable_systs, const Eigen::VectorXf &CVParams, const Eigen::VectorXf &fakeDataParams, const Eigen::VectorXf &fake_data_osc_param_vector, const std::vector<PROdata> &variable_data, const PROpt &options) {
    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());
    // allcovsyst (every spline converted to a covariance) is only needed for the
    // fractional-systematics PDFs, the Covar.pdf/ROOT covariance plots, and the
    // chi2/ndf labels on the error-band plots. Skip the conversion when none of
    // those are requested — but ALWAYS consume the seed draw so the global RNG
    // stream (and hence every downstream error-band throw) is flag-independent.
    const bool need_allcov = options.with_covar || !options.no_frac_syst || options.with_subcovar;
    const uint32_t allcov_seed = dseed(PROseed::global_rng);
    PROsyst allcovsyst;
    if(need_allcov) {
        log<LOG_INFO>(L"%1% || Making a PROsyst thats full covariance for future error bar creation (might be slow) ")% __func__ ;
        allcovsyst = variable_systs[config.i_prime].allsplines2cov(config, prop, model, CVParams, allcov_seed);
    } else {
        log<LOG_INFO>(L"%1% || --no-frac-syst without --with-covar: skipping the spline->covariance conversion; no chi2/ndf labels on error-band plots.")% __func__ ;
    }

    // --bkg-subtract: resolve the wildcard once. The same matched subchannel
    // list is used for every variable in this block; bkg_full / bkg_collapsed
    // are rebuilt per variable just before each plot_channels call. Empty
    // bkg_subchannels short-circuits all subtraction below.
    std::vector<size_t> bkg_subchannels;
    if (!options.bkg_subtract_pattern.empty()) {
        bkg_subchannels = find_subchannels_by_pattern(config, options.bkg_subtract_pattern);
        if (bkg_subchannels.empty()) {
            log<LOG_WARNING>(L"%1% || --bkg-subtract pattern '%2%' matched no subchannels; ignoring.")
                % __func__ % options.bkg_subtract_pattern.c_str();
        } else {
            log<LOG_INFO>(L"%1% || --bkg-subtract '%2%' matched %3% subchannel(s).")
                % __func__ % options.bkg_subtract_pattern.c_str() % bkg_subchannels.size();
        }
    }
    const bool do_bkg_subtract = !bkg_subchannels.empty();
    if (do_bkg_subtract && options.area_normalized)
        log<LOG_WARNING>(L"%1% || --bkg-subtract combined with area normalization: normalization uses the subtracted integral; interpret error bands with care.") % __func__;

    PlotOptions opt = PlotOptions::CVasStack;
    std::vector<TPaveText> notext;
    if(options.binwidth_scale) opt |= PlotOptions::BinWidthScaled;
    if(options.area_normalized) opt |= PlotOptions::AreaNormalized;
    if(options.shapeonly) opt |= PlotOptions::ShapeOnly;
    if(options.legend_counts) opt |= PlotOptions::LegendCounts;
    std::vector<PROspec> variable_cvs;
    std::vector<std::map<std::string, TObject *>> cv_objs;
    for(size_t io = 0; io < config.m_num_variables; ++io) {

        variable_cvs.push_back(FillSpectra(config, prop, variable_systs[config.i_prime], model, CVParams, !options.eventbyevent, io));

        // Local subtracted copy for the CV plot. variable_cvs is preserved
        // intact so downstream consumers (fractional-systematics breakdown,
        // error-band plot at L2156) see the unsubtracted CV unless they
        // also subtract.
        PROspec cv_plot = variable_cvs.back();
        if (do_bkg_subtract) {
            Eigen::VectorXf bkg_full = build_subchannel_mask_spec(
                config, cv_plot, bkg_subchannels, io);
            cv_plot.Spec() -= bkg_full;
        }
        auto objs = plot_channels(options.final_output_tag+"_PROplot_Variable_"+std::to_string(io)+"_CV.pdf", config, cv_plot, {}, {}, {}, {}, notext, options.pbounds, opt, io,
                false, options.plot_channel_ratios, do_bkg_subtract ? &bkg_subchannels : nullptr);
        cv_objs.push_back(objs);
    }

    // Shape-only: the fractional breakdowns and covariance plots show the per-channel shape
    // part of every systematic (MiniBooNE M_shape) about the CV, as the fit sees them. The
    // chi^2 labels (allcov_metric) are unaffected: the metric re-projects at evaluation.
    if(options.shapeonly && need_allcov)
        allcovsyst.ProjectCovariancesOntoShape(config, variable_cvs[config.i_prime].Spec());

    if(!options.no_frac_syst) {
        std::string filename = options.final_output_tag+"_fractional_systematics.pdf";
        plotPriorFractionalSystematicBreakdown(config, variable_cvs[config.i_prime], allcovsyst, filename,config.i_prime);
        std::string rfilename = options.final_output_tag+"_ratio_fractional_systematics.pdf";
        if(config.m_num_detectors > 1)
            plotPriorFractionalSystematicRatios(config, variable_cvs[config.i_prime], allcovsyst, rfilename,config.i_prime);

        std::string crfilename = options.final_output_tag+"_channel_ratio_fractional_systematics.pdf";
        if(options.plot_channel_ratios && config.m_num_channels > 1)
            plotPriorFractionalSystematicChannelRatios(config, variable_cvs[config.i_prime], allcovsyst, crfilename, config.i_prime);
    }

    std::vector<std::map<std::string, std::unique_ptr<TH1D>>> other_hists;
    for(size_t io = 0; io < config.m_num_variables; ++io) {
        other_hists.push_back(getCV1DHists(variable_cvs[io], config, options.binwidth_scale, io));
    }
    
    // DetVar plotting (uses combined DetVar propellers binary)
    if(config.m_has_detvar_section) {
        log<LOG_INFO>(L"%1% || Plotting detector variations...") % __func__;

        std::string dvAllPropsBin = options.analysis_tag + "_detvar_props.bin";
        std::map<std::string, PROpeller> plot_dvprops;
        std::vector<PROspec> detvar_specs;
        std::vector<std::string> detvar_names;
        std::vector<int> detvar_binning;
        // Matched pairs for _DetVarOverlapping PDF (var file index -> matched cv+var specs)
        struct MatchedPair { PROspec cv; std::map<double, PROspec> vars; };
        std::map<size_t, MatchedPair> matched_pairs;

        if(!std::filesystem::exists(dvAllPropsBin)) {
            log<LOG_ERROR>(L"%1% || DetVar combined binary not found: %2%. Run 'process' first.") % __func__ % dvAllPropsBin.c_str();
        } else {
            uint32_t loaded_detvar_hash = loadDetVarProps(plot_dvprops, dvAllPropsBin);
            if(config.detvar_hash != loaded_detvar_hash) {
                log<LOG_WARNING>(L"%1% || WARNING config detvar_hash (%2%) and binary detvar_hash (%3%) not compatible. DetVar plots may be stale.") % __func__ % config.detvar_hash % loaded_detvar_hash;
            }

            // Precompute section CV index by section for matching
            std::map<size_t, size_t> plot_cv_idx_by_section; // section_idx -> detvar file index
            for(size_t idv = 0; idv < config.GetNumDetVarFiles(); ++idv) {
                if(config.m_detvar_files[idv].is_cv)
                    plot_cv_idx_by_section[config.m_detvar_files[idv].section_index] = idv;
            }

            // One spectrum per DetVar file, index-aligned with m_detvar_files (the pages rely on it).
            for(size_t idv = 0; idv < config.GetNumDetVarFiles(); ++idv) {
                const std::string& name = config.m_detvar_files[idv].name;
                const std::string key = DetVarKey(config, idv);
                if(plot_dvprops.count(key) == 0) {
                    log<LOG_ERROR>(L"%1% || DetVar entry '%2%' not found in combined binary. Run 'process' first.") % __func__ % name.c_str();
                    break;
                }

                int binningIndex = config.m_mcgen_variation_binning_map.count(name) ? config.m_mcgen_variation_binning_map.at(name) : config.i_prime;
                if(binningIndex < 0 || binningIndex >= (int)config.m_num_variables)
                    binningIndex = config.i_prime;

                PROconfig dvconfig = config.BuildDetVarConfig(idv);
                PROpeller& dvprop = plot_dvprops.at(key);
                std::unique_ptr<PROmodel> dv_model = std::make_unique<NullModel>(dvprop);
                PROsyst dvsysts;
                Eigen::VectorXf dvparams = Eigen::VectorXf::Constant(dv_model->nparams, 0);

                // Always use full spec for _DetVarFull PDF
                detvar_specs.push_back(FillSpectra(dvconfig, dvprop, dvsysts, *dv_model, dvparams, !options.eventbyevent, binningIndex));
                detvar_names.push_back(name);
                detvar_binning.push_back(binningIndex);
            }

            // Matched pairs for the _DetVarOverlapping PDF: one per (section, name), keyed by its first file.
            std::set<std::pair<size_t, std::string>> paired;
            for(size_t idv = 0; idv < detvar_specs.size(); ++idv) {
                const PROconfig::DetVarFile &dvf = config.m_detvar_files[idv];
                if(dvf.is_cv || !paired.insert({dvf.section_index, dvf.name}).second) continue;
                const std::string &name = dvf.name;
                const size_t sec = dvf.section_index;
                const int binningIndex = detvar_binning[idv];

                std::map<double, size_t> syst_files;
                std::map<double, const PROpeller*> props;
                MatchedPair mp;
                for(size_t i = idv; i < detvar_specs.size(); ++i) {
                    const PROconfig::DetVarFile &f = config.m_detvar_files[i];
                    if(f.is_cv || f.section_index != sec || f.name != name) continue;
                    syst_files[f.knobval] = i;
                    props[f.knobval] = &plot_dvprops.at(DetVarKey(config, i));
                    mp.vars[f.knobval] = detvar_specs[i];
                }

                {
                    auto cv_it = plot_cv_idx_by_section.find(sec);
                    if(cv_it != plot_cv_idx_by_section.end()) {
                        PROpeller& cvprop_plot = plot_dvprops.at(DetVarKey(config, cv_it->second));
                        PROconfig cvconfig = config.BuildDetVarConfig(cv_it->second);
                        std::unique_ptr<PROmodel> cv_model = std::make_unique<NullModel>(cvprop_plot);
                        Eigen::VectorXf cvparams = Eigen::VectorXf::Constant(cv_model->nparams, 0);
                        mp.cv = FillSpectra(cvconfig, cvprop_plot, PROsyst(), *cv_model, cvparams, !options.eventbyevent, binningIndex);
                        if(BuildDetVarMatchedSpecs(cvprop_plot, props, binningIndex,
                                                   (int)config.m_num_variable_bins_total[binningIndex],
                                                   mp.cv, mp.vars, true)) {
                            // Undo POT scaling from both CV and variation matched spectra so
                            // the overlapping plot shows raw event-weight units (no POT scaling)
                            const double det_pot_ov = config.m_det_pot[0];
                            const double cv_pot_ov = config.m_detvar_files[cv_it->second].pot;
                            if(det_pot_ov > 0.0 && cv_pot_ov > 0.0) {
                                const float cv_unscale_ov = (float)(cv_pot_ov / det_pot_ov);
                                mp.cv.Spec() *= cv_unscale_ov;
                                mp.cv.Error() *= cv_unscale_ov;
                                for(auto &[kv_plot, var_file_idx_plot] : syst_files) {
                                    const double var_pot_ov = config.m_detvar_files[var_file_idx_plot].pot;
                                    if(var_pot_ov > 0.0) {
                                        const float var_unscale_ov = (float)(var_pot_ov / det_pot_ov);
                                        mp.vars[kv_plot].Spec() *= var_unscale_ov;
                                        mp.vars[kv_plot].Error() *= var_unscale_ov;
                                    }
                                }
                            }
                            matched_pairs[idv] = std::move(mp);
                            log<LOG_INFO>(L"%1% || DetVar plot '%2%' (section %3%): matched pair built for Overlapping PDF") % __func__ % name.c_str() % sec;
                        }
                    }
                }
            }
        }

        if(detvar_specs.size() == config.GetNumDetVarFiles()) {
            // Build per-section CV hist maps (each DetVarSection has its own CV)
            std::map<size_t, std::map<std::string, std::unique_ptr<TH1D>>> cv_hists_by_section;
            bool any_cv_missing = false;
            for(size_t i = 0; i < config.m_detvar_files.size(); ++i) {
                if(config.m_detvar_files[i].is_cv) {
                    size_t sec = config.m_detvar_files[i].section_index;
                    cv_hists_by_section[sec] = getCV1DHists(detvar_specs[i], config, options.binwidth_scale, detvar_binning[i]);
                }
            }
            if(cv_hists_by_section.empty()) {
                log<LOG_ERROR>(L"%1% || ERROR: No CV files found in DetVar files!") % __func__;
                any_cv_missing = true;
            }
            if(!any_cv_missing) {
                TCanvas detvar_canvas;
                std::string detvar_pdf = options.final_output_tag + "_PROplot_DetVarFull.pdf";
                detvar_canvas.Print((detvar_pdf + "[").c_str(), "pdf");

                size_t global_subchannel_index = 0;
                    for(size_t im = 0; im < config.m_num_modes; im++){
                        for(size_t id = 0; id < config.m_num_detectors; id++){
                            for(size_t ic = 0; ic < config.m_num_channels; ic++){
                                // Direct DetVar plot: show detvar_cv and detvar_variation spectra directly
                                int colors[] = {kRed, kBlue, kGreen+2, kMagenta, kCyan+1, kOrange+1, kViolet+1, kTeal+1};
                                int ncolors = sizeof(colors)/sizeof(colors[0]);

                                // Per-section pages: draw only variations belonging to the same section as the CV
                                for(const auto& [sec_idx, sec_cv_hists] : cv_hists_by_section) {
                                    TH1D* cv_total = nullptr;
                                    for(size_t sc = 0; sc < config.m_num_subchannels[ic]; sc++) {
                                        const std::string& subchannel_name = config.m_fullnames[global_subchannel_index + sc];
                                        auto cv_it = sec_cv_hists.find(subchannel_name);
                                        if(cv_it != sec_cv_hists.end()) {
                                            if(!cv_total)
                                                cv_total = (TH1D*)cv_it->second->Clone(("cv_total_sec" + std::to_string(sec_idx)).c_str());
                                            else
                                                cv_total->Add(&*(cv_it->second));
                                        }
                                    }
                                    if(!cv_total) continue;

                                    cv_total->SetLineColor(kBlack);
                                    cv_total->SetLineWidth(3);
                                    cv_total->SetFillColor(kWhite);
                                    cv_total->SetFillStyle(0);
                                    cv_total->SetTitle((config.m_mode_names[im] + " " + config.m_detector_names[id] + " " + config.m_channel_names[ic] + " DetVar (sec " + std::to_string(sec_idx) + ")").c_str());
                                    {
                                        std::string chan_unit = config.GetChannelUnit(ic, config.i_prime);
                                        std::string ytitle;
                                        if(!options.binwidth_scale) {
                                            ytitle = "Events";
                                        } else if(chan_unit.empty()) {
                                            ytitle = "Events/unit";
                                        } else {
                                            ytitle = "Events/" + chan_unit;
                                        }
                                        cv_total->GetYaxis()->SetTitle(ytitle.c_str());
                                    }

                                    float ymax = cv_total->GetMaximum();

                                    // Build variation totals for this section only
                                    std::vector<TH1D*> var_totals;
                                    std::vector<std::string> var_labels;
                                    int color_idx = 0;

                                    for(size_t idv = 0; idv < detvar_specs.size(); ++idv) {
                                        if(config.m_detvar_files[idv].is_cv) continue;
                                        if(config.m_detvar_files[idv].section_index != sec_idx) continue;

                                        std::map<std::string, std::unique_ptr<TH1D>> var_hists = getCV1DHists(detvar_specs[idv], config, options.binwidth_scale, detvar_binning[idv]);

                                        TH1D* var_total = nullptr;
                                        for(size_t sc = 0; sc < config.m_num_subchannels[ic]; sc++) {
                                            const std::string& subchannel_name = config.m_fullnames[global_subchannel_index + sc];
                                            auto var_it = var_hists.find(subchannel_name);
                                            if(var_it != var_hists.end()) {
                                                if(!var_total)
                                                    var_total = (TH1D*)var_it->second->Clone(("var_" + detvar_names[idv]).c_str());
                                                else
                                                    var_total->Add(&*(var_it->second));
                                            }
                                        }

                                        if(var_total) {
                                            var_total->SetLineColor(colors[color_idx % ncolors]);
                                            var_total->SetLineWidth(2);
                                            var_total->SetFillColor(kWhite);
                                            var_total->SetFillStyle(0);
                                            if(var_total->GetMaximum() > ymax) ymax = var_total->GetMaximum();
                                            var_totals.push_back(var_total);
                                            var_labels.push_back(detvar_names[idv] + " " + FormatKnobVal(config.m_detvar_files[idv].knobval));
                                        }
                                        color_idx++;
                                    }

                                    cv_total->SetMaximum(ymax * 1.15);
                                    cv_total->Draw("hist");

                                    std::unique_ptr<TLegend> leg = std::make_unique<TLegend>(0.55, 0.65, 0.89, 0.89);
                                    leg->SetFillStyle(0);
                                    leg->SetLineWidth(0);
                                    leg->AddEntry(cv_total, ("DetVar CV (sec " + std::to_string(sec_idx) + ")").c_str(), "l");

                                    for(size_t vi = 0; vi < var_totals.size(); ++vi) {
                                        var_totals[vi]->Draw("hist same");
                                        leg->AddEntry(var_totals[vi], var_labels[vi].c_str(), "l");
                                    }
                                    leg->Draw("same");

                                    drawVersionWatermark(&detvar_canvas);
                                    detvar_canvas.Print(detvar_pdf.c_str(), "pdf");

                                    delete cv_total;
                                    for(auto* h : var_totals) delete h;
                                }

                                global_subchannel_index += config.m_num_subchannels[ic];
                            }
                        }
                    }
                detvar_canvas.Print((detvar_pdf + "]").c_str(), "pdf");
                log<LOG_INFO>(L"%1% || DetVar full plots saved to %2%") % __func__ % detvar_pdf.c_str();

                // _DetVarOverlapping PDF: one page per channel × variation, matched CV vs matched var
                if(!matched_pairs.empty()) {
                    TCanvas ov_canvas;
                    std::string ov_pdf = options.final_output_tag + "_PROplot_DetVarOverlapping.pdf";
                    ov_canvas.Print((ov_pdf + "[").c_str(), "pdf");

                    size_t ov_global_subchannel_index = 0;
                    for(size_t im = 0; im < config.m_num_modes; im++){
                        for(size_t id = 0; id < config.m_num_detectors; id++){
                            for(size_t ic = 0; ic < config.m_num_channels; ic++){
                                for(size_t idv = 0; idv < detvar_specs.size(); ++idv) {
                                    if(config.m_detvar_files[idv].is_cv) continue;
                                    auto mp_it = matched_pairs.find(idv);
                                    if(mp_it == matched_pairs.end()) continue;

                                    const MatchedPair& mp = mp_it->second;
                                    std::map<std::string, std::unique_ptr<TH1D>> cv_hists_ov = getCV1DHists(mp.cv, config, options.binwidth_scale, detvar_binning[idv]);
                                    std::map<double, std::map<std::string, std::unique_ptr<TH1D>>> var_hists_ov;
                                    for(auto &[kv, vspec] : mp.vars)
                                        var_hists_ov[kv] = getCV1DHists(vspec, config, options.binwidth_scale, detvar_binning[idv]);

                                    TH1D* cv_total_ov = nullptr;
                                    std::map<double, TH1D*> var_total_ov;
                                    for(size_t sc = 0; sc < config.m_num_subchannels[ic]; sc++) {
                                        const std::string& subchannel_name = config.m_fullnames[ov_global_subchannel_index + sc];
                                        auto cv_hit = cv_hists_ov.find(subchannel_name);
                                        if(cv_hit != cv_hists_ov.end()) {
                                            if(!cv_total_ov) cv_total_ov = (TH1D*)cv_hit->second->Clone("cv_matched_ov_total");
                                            else cv_total_ov->Add(&*(cv_hit->second));
                                        }
                                        for(auto &[kv, specs] : var_hists_ov) {
                                            auto var_hit = specs.find(subchannel_name);
                                            if(var_hit != specs.end()) {
                                                if(var_total_ov.count(kv) == 0) 
                                                    var_total_ov[kv] 
                                                        = (TH1D*)var_hit->second->Clone("var_matched_ov_total");
                                                else 
                                                    var_total_ov[kv]->Add(&*(var_hit->second));
                                            }
                                        }
                                    }

                                    if(cv_total_ov && var_total_ov.size()) {
                                        cv_total_ov->SetLineColor(kBlack);
                                        cv_total_ov->SetLineWidth(3);
                                        cv_total_ov->SetFillColor(kWhite);
                                        cv_total_ov->SetFillStyle(0);
                                        for(auto &[kv, h] : var_total_ov) {
                                            h->SetLineWidth(2);
                                            h->SetFillColor(kWhite);
                                            h->SetFillStyle(0);
                                        }

                                        // Leave room for the error bars.
                                        float ymax_ov = cv_total_ov->GetMaximum() + cv_total_ov->GetBinError(cv_total_ov->GetMaximumBin());
                                        for(auto &[kv, h] : var_total_ov)
                                            ymax_ov = std::max(ymax_ov, (float)(h->GetMaximum() + h->GetBinError(h->GetMaximumBin())));
                                        cv_total_ov->SetMaximum(ymax_ov * 1.15);
                                        std::string ov_title = config.m_mode_names[im] + " " + config.m_detector_names[id] + " " + config.m_channel_names[ic] + " " + detvar_names[idv] + " (Matched, sec " + std::to_string(config.m_detvar_files[idv].section_index) + ")";
                                        cv_total_ov->SetTitle(ov_title.c_str());
                                        {
                                            std::string chan_unit = config.GetChannelUnit(ic, config.i_prime);
                                            std::string ytitle;
                                            if(!options.binwidth_scale) {
                                                ytitle = "Events";
                                            } else if(chan_unit.empty()) {
                                                ytitle = "Events/unit";
                                            } else {
                                                ytitle = "Events/" + chan_unit;
                                            }
                                            cv_total_ov->GetYaxis()->SetTitle(ytitle.c_str());
                                        }

                                        std::unique_ptr<TLegend> ov_leg = std::make_unique<TLegend>(0.55, 0.75, 0.89, 0.89);
                                        ov_leg->SetFillStyle(0);
                                        ov_leg->SetLineWidth(0);

                                        int ov_var_colors[] = {kRed, kBlue, kGreen+2, kMagenta, kCyan+1, kOrange+1, kViolet+1, kTeal+1};
                                        int n_ov_var_colors = sizeof(ov_var_colors)/sizeof(ov_var_colors[0]);
                                        int ov_color_idx = 0;
                                        // Error bars are the weighted MC stat errors, sqrt(sum w^2), of each matched sample.
                                        cv_total_ov->SetMarkerColor(kBlack);
                                        cv_total_ov->Draw("hist");
                                        cv_total_ov->Draw("E1 same");
                                        ov_leg->AddEntry(cv_total_ov, "Matched CV", "le");
                                        for(auto &[kv, h] : var_total_ov) {
                                            h->SetLineColor(ov_var_colors[ov_color_idx % n_ov_var_colors]);
                                            h->SetMarkerColor(ov_var_colors[ov_color_idx % n_ov_var_colors]);
                                            ++ov_color_idx;
                                            h->Draw("hist same");
                                            h->Draw("E1 same");
                                            ov_leg->AddEntry(h, (detvar_names[idv]+" "+FormatKnobVal(kv)).c_str(), "le");
                                        }

                                        ov_leg->Draw("same");

                                        drawVersionWatermark(&ov_canvas);
                                        ov_canvas.Print(ov_pdf.c_str(), "pdf");

                                        // For 2D channels also draw every slice, per subchannel: the DetVar spline is
                                        // built bin-by-bin in exactly these bins, so low matched statistics show up here
                                        // rather than being averaged away in the projection above.
                                        const int ov_binning = detvar_binning[idv];
                                        if(config.m_channel_variable_dims[ic][ov_binning] == 2) {
                                            const auto &bins2d = config.m_channel_variable_bins[ic][ov_binning];
                                            const size_t nx = bins2d.NBinsAlong(0), ny = bins2d.NBinsAlong(1);
                                            const std::vector<float> edges_x = bins2d.Edges(0), edges_y = bins2d.Edges(1);
                                            const std::string title_x = config.GetChannelAxisTitle(ic, ov_binning, 0);
                                            const std::string title_y = config.GetChannelAxisTitle(ic, ov_binning, 1);

                                            for(size_t sc = 0; sc < config.m_num_subchannels[ic]; sc++) {
                                                const size_t gsc = ov_global_subchannel_index + sc;
                                                const int start = config.GetGlobalVariableBinStart(gsc, ov_binning);
                                                // fixed_dim is the variable held fixed within each pad (1: slices in y, 0: slices in x).
                                                for(int fixed_dim : {1, 0}) {
                                                    const size_t nslices = fixed_dim == 1 ? ny : nx;
                                                    const size_t nalong  = fixed_dim == 1 ? nx : ny;
                                                    const std::vector<float> &along_edges = fixed_dim == 1 ? edges_x : edges_y;
                                                    const std::vector<float> &slice_edges = fixed_dim == 1 ? edges_y : edges_x;
                                                    const std::string &along_title = fixed_dim == 1 ? title_x : title_y;
                                                    const std::string &slice_title = fixed_dim == 1 ? title_y : title_x;
                                                    auto flat = [&](size_t islice, size_t ialong) {
                                                        return start + (fixed_dim == 1 ? ialong*ny + islice : islice*ny + ialong);
                                                    };

                                                    // Hists/legend/lines drawn on this page; freed after the page is printed and cleared.
                                                    std::vector<std::unique_ptr<TObject>> keep;
                                                    ov_canvas.Clear();
                                                    ov_canvas.cd();
                                                    TPad *head = new TPad("dv_ov_head", "", 0, 0.95, 1, 1);
                                                    TPad *body = new TPad("dv_ov_body", "", 0, 0, 1, 0.95);
                                                    // Pads are owned by their parent pad (kCanDelete), freed by ov_canvas.Clear().
                                                    head->SetBit(kCanDelete); body->SetBit(kCanDelete);
                                                    head->Draw(); body->Draw();
                                                    head->cd();
                                                    const std::string page_title = config.m_mode_names[im] + " " + config.m_detector_names[id] + " " + config.m_channel_names[ic]
                                                        + " | " + config.m_fullnames[gsc] + " | " + detvar_names[idv] + " (Matched), slices in " + slice_title;
                                                    TLatex *lat = new TLatex(0.01, 0.5, page_title.c_str());
                                                    keep.emplace_back(lat);
                                                    lat->SetNDC(); lat->SetTextAlign(12); lat->SetTextSize(0.35);
                                                    lat->Draw();

                                                    const int ncols = (int)std::ceil(std::sqrt((double)nslices));
                                                    const int nrows = (int)std::ceil(nslices / (double)ncols);
                                                    body->Divide(ncols, nrows);
                                                    for(size_t is = 0; is < nslices; ++is) {
                                                        TVirtualPad *cell = body->cd(is + 1);
                                                        const std::string sfx = "_" + std::to_string(gsc) + "_" + std::to_string(fixed_dim) + "_" + std::to_string(is);
                                                        TPad *top = new TPad(("dv_ov_top"+sfx).c_str(), "", 0, 0.32, 1, 1);
                                                        TPad *bot = new TPad(("dv_ov_bot"+sfx).c_str(), "", 0, 0, 1, 0.32);
                                                        top->SetBit(kCanDelete); bot->SetBit(kCanDelete);
                                                        top->SetBottomMargin(0.02); top->SetLeftMargin(0.14);
                                                        bot->SetTopMargin(0.02); bot->SetBottomMargin(0.32); bot->SetLeftMargin(0.14);
                                                        cell->cd(); top->Draw(); bot->Draw();

                                                        std::ostringstream st;
                                                        st << slice_title << " in [" << slice_edges[is] << ", " << slice_edges[is+1] << ")";
                                                        TH1D *hcv = new TH1D(("dv_ov_cv"+sfx).c_str(), (st.str()+";;Events").c_str(), nalong, along_edges.data());
                                                        hcv->SetDirectory(nullptr);
                                                        keep.emplace_back(hcv);
                                                        for(size_t ia = 0; ia < nalong; ++ia) {
                                                            hcv->SetBinContent(ia+1, mp.cv.Spec()(flat(is, ia)));
                                                            hcv->SetBinError(ia+1, mp.cv.Error()(flat(is, ia)));
                                                        }
                                                        std::vector<TH1D*> hvars, hrats;
                                                        double ymax = hcv->GetMaximum() + hcv->GetBinError(hcv->GetMaximumBin());
                                                        double rmin = 1.0, rmax = 1.0;
                                                        int icol = 0;
                                                        for(const auto &[kv, vspec] : mp.vars) {
                                                            const std::string vsfx = sfx + "_" + FormatKnobVal(kv);
                                                            TH1D *hv = new TH1D(("dv_ov_var"+vsfx).c_str(), "", nalong, along_edges.data());
                                                            TH1D *hr = new TH1D(("dv_ov_rat"+vsfx).c_str(), ";"+TString(along_title.c_str())+";Var/CV", nalong, along_edges.data());
                                                            hv->SetDirectory(nullptr); hr->SetDirectory(nullptr);
                                                            keep.emplace_back(hv); keep.emplace_back(hr);
                                                            for(size_t ia = 0; ia < nalong; ++ia) {
                                                                const float v = vspec.Spec()(flat(is, ia)), c = mp.cv.Spec()(flat(is, ia));
                                                                hv->SetBinContent(ia+1, v);
                                                                hv->SetBinError(ia+1, vspec.Error()(flat(is, ia)));
                                                                if(c != 0) {
                                                                    hr->SetBinContent(ia+1, v/c);
                                                                    rmin = std::min(rmin, (double)v/c);
                                                                    rmax = std::max(rmax, (double)v/c);
                                                                }
                                                            }
                                                            ymax = std::max(ymax, hv->GetMaximum() + hv->GetBinError(hv->GetMaximumBin()));
                                                            const int col = ov_var_colors[icol++ % n_ov_var_colors];
                                                            hv->SetLineColor(col); hv->SetMarkerColor(col); hv->SetLineWidth(2);
                                                            hr->SetLineColor(col); hr->SetLineWidth(2);
                                                            hvars.push_back(hv); hrats.push_back(hr);
                                                        }

                                                        top->cd();
                                                        hcv->SetStats(0);
                                                        hcv->SetLineColor(kBlack); hcv->SetMarkerColor(kBlack); hcv->SetLineWidth(2);
                                                        hcv->SetMinimum(0);
                                                        hcv->SetMaximum(ymax > 0 ? 1.2*ymax : 1.0);
                                                        hcv->GetXaxis()->SetLabelSize(0);
                                                        hcv->GetYaxis()->SetLabelSize(0.06); hcv->GetYaxis()->SetTitleSize(0.06);
                                                        hcv->Draw("E1");
                                                        for(TH1D *hv : hvars) hv->Draw("E1 same");
                                                        if(is == 0) {
                                                            TLegend *leg = new TLegend(0.6, 0.72, 0.89, 0.88);
                                                            keep.emplace_back(leg);
                                                            leg->SetFillStyle(0); leg->SetLineWidth(0);
                                                            leg->AddEntry(hcv, "Matched CV", "le");
                                                            int il = 0;
                                                            for(const auto &[kv, vspec] : mp.vars)
                                                                leg->AddEntry(hvars[il++], (detvar_names[idv]+" "+FormatKnobVal(kv)).c_str(), "le");
                                                            leg->Draw();
                                                        }

                                                        bot->cd();
                                                        const double pad_r = std::max(0.1, 0.1*(rmax - rmin));
                                                        for(size_t ir = 0; ir < hrats.size(); ++ir) {
                                                            TH1D *hr = hrats[ir];
                                                            if(ir == 0) {
                                                                hr->SetStats(0);
                                                                hr->SetMinimum(std::max(0.0, rmin - pad_r));
                                                                hr->SetMaximum(rmax + pad_r);
                                                                hr->GetXaxis()->SetLabelSize(0.12); hr->GetXaxis()->SetTitleSize(0.12);
                                                                hr->GetYaxis()->SetLabelSize(0.10); hr->GetYaxis()->SetTitleSize(0.11);
                                                                hr->GetYaxis()->SetTitleOffset(0.55); hr->GetYaxis()->SetNdivisions(505);
                                                                hr->Draw("hist");
                                                            } else {
                                                                hr->Draw("hist same");
                                                            }
                                                        }
                                                        TLine *one = new TLine(along_edges.front(), 1.0, along_edges.back(), 1.0);
                                                        keep.emplace_back(one);
                                                        one->SetLineStyle(2);
                                                        one->Draw();
                                                    }
                                                    drawVersionWatermark(&ov_canvas);
                                                    ov_canvas.Print(ov_pdf.c_str(), "pdf");
                                                    ov_canvas.Clear();
                                                }
                                            }
                                            ov_canvas.cd();
                                        }

                                        delete cv_total_ov;
                                        //delete var_total_ov;
                                    } else {
                                        if(cv_total_ov) delete cv_total_ov;
                                        //if(var_total_ov) delete var_total_ov;
                                    }
                                }
                                ov_global_subchannel_index += config.m_num_subchannels[ic];
                            }
                        }
                    }
                    ov_canvas.Print((ov_pdf + "]").c_str(), "pdf");
                    log<LOG_INFO>(L"%1% || DetVar overlapping plots saved to %2%") % __func__ % ov_pdf.c_str();
                    set_matrix_palette();
                }
            }
        }
    }

    TCanvas c;
    if(options.fake_data_osc_params.size()) {

        c.Print((options.final_output_tag +"_PROplot_Osc.pdf"+ "[").c_str(), "pdf");

        PROspec osc_spec = FillSpectra(config, prop, variable_systs[config.i_prime], model, fakeDataParams, !options.eventbyevent,config.i_prime );
        std::map<std::string, std::unique_ptr<TH1D>> osc_hists = getCV1DHists(osc_spec, config, options.binwidth_scale);
        size_t global_subchannel_index = 0;
        for(size_t im = 0; im < config.m_num_modes; im++){
            for(size_t id =0; id < config.m_num_detectors; id++){
                for(size_t ic = 0; ic < config.m_num_channels; ic++){
                    TH1D* osc_hist = NULL;
                    TH1D* cv_hist = NULL;
                    for(size_t sc = 0; sc < config.m_num_subchannels[ic]; sc++){
                        const std::string& subchannel_name  = config.m_fullnames[global_subchannel_index];
                        const auto &h = other_hists[config.i_prime][subchannel_name];
                        const auto &o = osc_hists[subchannel_name];
                        if(sc == 0) {
                            // Clone() keeps the subchannel hist's name and re-registers it with
                            // gDirectory; these clones are page-local, so detach them or the
                            // later ErrorBand pages' same-named hists warn "Replacing existing TH1".
                            cv_hist = (TH1D*)h->Clone();
                            cv_hist->SetDirectory(nullptr);
                            osc_hist = (TH1D*)o->Clone();
                            osc_hist->SetDirectory(nullptr);
                        } else {
                            cv_hist->Add(&*h);
                            osc_hist->Add(&*o);
                        }
                        ++global_subchannel_index;
                    }
                    {
                        std::string chan_unit = config.GetChannelUnit(ic, config.i_prime);
                        std::string ytitle;
                        if(!options.binwidth_scale) {
                            ytitle = "Events";
                        } else if(chan_unit.empty()) {
                            ytitle = "Events/unit";
                        } else {
                            ytitle = "Events/" + chan_unit;
                        }
                        cv_hist->GetYaxis()->SetTitle(ytitle.c_str());
                    }
                    if(options.area_normalized) {
                        cv_hist->GetYaxis()->SetTitle("Area Normalized");
                        // The sub-hists are already width-scaled when --scale-by-width:
                        // normalise by the RAW integral (sum of counts), not the sum of
                        // densities. Guard empty histograms.
                        const char *iopt = options.binwidth_scale ? "width" : "";
                        const double cv_int = cv_hist->Integral(iopt), osc_int = osc_hist->Integral(iopt);
                        if(cv_int > 0) cv_hist->Scale(1.0 / cv_int);
                        if(osc_int > 0) osc_hist->Scale(1.0 / osc_int);
                    }
                    cv_hist->SetTitle((config.m_mode_names[im]  +" "+ config.m_detector_names[id]+" "+ config.m_channel_names[ic]).c_str());
                    cv_hist->GetXaxis()->SetTitle("");
                    cv_hist->SetLineColor(kBlack);
                    cv_hist->SetFillColor(kWhite);
                    cv_hist->SetFillStyle(0);
                    osc_hist->SetLineColor(kBlue);
                    osc_hist->SetFillColor(kWhite);
                    osc_hist->SetFillStyle(0);
                    cv_hist->SetLineWidth(3);
                    osc_hist->SetLineWidth(3);
                    TH1D *rat = (TH1D*)osc_hist->Clone();
                    rat->SetDirectory(nullptr);
                    rat->Divide(cv_hist);
                    rat->SetTitle("");
                    rat->GetYaxis()->SetTitle("Ratio");
                    TH1D *one = (TH1D*)rat->Clone();
                    one->SetDirectory(nullptr);
                    one->Divide(one);
                    one->SetLineColor(kBlack);
                    one->GetYaxis()->SetTitle("Ratio");

                    // A real box: the old zero-area (0.59,0.89,0.59,0.89) made ROOT warn
                    // "Legend too large to be automatically placed" and pick its own spot.
                    std::unique_ptr<TLegend> leg = std::make_unique<TLegend>(0.35,0.75,0.89,0.89);
                    leg->SetFillStyle(0);
                    leg->SetLineWidth(0);
                    leg->AddEntry(cv_hist, "No Oscillations", "l");
                    std::string oscstr = "";//"#splitline{Oscilations:}{";
                    for(size_t j=0;j<model.nparams;j++){
                        float val_maybe_log = model.is_log10[j] ? std::pow(10.0f, fake_data_osc_param_vector(j)) : fake_data_osc_param_vector(j);
                        oscstr+=model.pretty_param_names[j]+ " : "+ to_string_prec(val_maybe_log,3) +" "+model.pretty_param_units[j] + (j==0 ? ", " : "" );
                    }
                    //oscstr+="}";

                    leg->AddEntry(osc_hist, oscstr.c_str(), "l");

                    TPad p1("p1", "p1", 0, 0.25, 1, 1);
                    p1.SetBottomMargin(0);
                    p1.cd();
                    cv_hist->Draw("hist");
                    hideFirstYLabel(cv_hist->GetYaxis()); // ratio pad below covers the bottom y label
                    osc_hist->Draw("hist same");
                    cv_hist->SetMaximum(std::max(cv_hist->GetMaximum(),osc_hist->GetMaximum())*1.1);
                    leg->Draw("same");

                    TPad p2("p2", "p2", 0, 0, 1, 0.25);
                    p2.SetTopMargin(0);
                    p2.SetBottomMargin(0.3);
                    p2.cd();
                    one->GetYaxis()->SetTitleSize(0.1);
                    one->GetYaxis()->SetLabelSize(0.1);
                    one->GetXaxis()->SetTitleSize(0.1);
                    one->GetXaxis()->SetLabelSize(0.1);
                    one->GetYaxis()->SetTitleOffset(0.5);
                    one->GetYaxis()->SetNdivisions(505); // tiny ratio ranges otherwise smear ~8 overlapping labels
                    hideLastYLabel(one->GetYaxis()); // top label clips at the pad's zero top margin
                    one->Draw("hist");
                    one->SetMaximum(rat->GetMaximum()*1.2);
                    one->SetMinimum(rat->GetMinimum()*0.8);
                    rat->Draw("hist same");

                    c.cd();
                    p1.Draw();
                    p2.Draw();

                    drawVersionWatermark(&c);
                    c.Print((options.final_output_tag+"_PROplot_Osc.pdf").c_str(), "pdf");

                    delete cv_hist;
                    delete osc_hist;
                }
            }
        }
        c.Print((options.final_output_tag+"_PROplot_Osc.pdf" + "]").c_str(), "pdf");

    }

    std::map<std::string, std::unique_ptr<TH2D>> matrices;
    if(options.with_covar || options.with_subcovar) {
        matrices = covarianceTH2D(allcovsyst, config, variable_cvs[config.i_prime]);
        const std::string covar_pdf = options.final_output_tag + "_PROplot_Covar.pdf";
        log<LOG_INFO>(L"%1% || Writing covariance plots to %2% (%3% matrices available)") % __func__ % covar_pdf.c_str() % matrices.size();
        c.Print((covar_pdf + "[").c_str(), "pdf");

        std::vector<std::string> plot_filters;
        std::vector<std::string> first_plots;
        // Printed first, in this order
        if(options.with_subcovar){
            std::vector<std::string> first_plots = {"collapsed_total_cor", "collapsed_total_frac_cov", "total_cor", "total_frac_cov"};
            plot_filters = {};
        }else{
            std::vector<std::string> first_plots = {"collapsed_total_cor", "collapsed_total_frac_cov"};
            // Any other matrix is printed only if its name contains one of these
            // (the full-resolution matrices make the PDF huge). Candidate for a command-line option.
            plot_filters = {"collapsed"};
        }


        auto passes_filter = [&](const std::string &name) {
            for (const auto &f : plot_filters)
                if (name.find(f) != std::string::npos) return true;
            return false;
        };

        int n_printed = 0, n_skipped = 0;

        for(const auto &name: first_plots){
            auto it = matrices.find(name);
            if (it == matrices.end()) {
                log<LOG_WARNING>(L"%1% || First plot %2% not found in covariance matrices, skipping") % __func__ % name.c_str();
                continue;
            }
            auto &mat = it->second;
            log<LOG_INFO>(L"%1% || Printing first plot %2% (%3% x %4% bins)") % __func__ % name.c_str() % mat->GetNbinsX() % mat->GetNbinsY();
            mat->Draw("colz");
            if(options.shapeonly) drawShapeOnlyNote(&c);
            drawVersionWatermark(&c);
            c.Print(covar_pdf.c_str(), "pdf");
            ++n_printed;
        }

        for(const auto &[name, mat]: matrices) {
            if (std::find(first_plots.begin(), first_plots.end(), name) != first_plots.end()) {
                log<LOG_DEBUG>(L"%1% || Skipping %2%: already printed as a first plot") % __func__ % name.c_str();
                continue;
            }
            if (!passes_filter(name)) {
                log<LOG_DEBUG>(L"%1% || Skipping %2%: matches no plot filter (%3% x %4% bins)") % __func__ % name.c_str() % mat->GetNbinsX() % mat->GetNbinsY();
                ++n_skipped;
                continue;
            }
            log<LOG_INFO>(L"%1% || Printing %2% (%3% x %4% bins)") % __func__ % name.c_str() % mat->GetNbinsX() % mat->GetNbinsY();
            mat->Draw("colz");
            if(options.shapeonly) drawShapeOnlyNote(&c);
            drawVersionWatermark(&c);
            c.Print(covar_pdf.c_str(), "pdf");
            ++n_printed;
        }
        c.Print((covar_pdf + "]").c_str(), "pdf");
        log<LOG_INFO>(L"%1% || Covariance PDF done: %2% pages printed, %3% matrices filtered out") % __func__ % n_printed % n_skipped;
    }

    //errorband
    // The chi2/ndf labels need the all-covariance metric; without allcovsyst
    // (--no-frac-syst and no --with-covar) they are skipped entirely.
    std::unique_ptr<PROmetric> allcov_metric;
    if(need_allcov) {
        allcov_metric.reset(metric.Clone());
        allcov_metric->override_systs(allcovsyst);
    }
    std::vector<std::vector<TPaveText>> other_channel_chitexts;

    for(size_t io = 0; io < config.m_num_variables; ++io) {

        log<LOG_INFO>(L"%1% || On Variable number %2%:") % __func__ % io ;
        int global_channel_index = 0;
        std::vector<TPaveText> channel_chitexts;

        //Currently metrics are for i_prime only. TO BE DONE

        for(size_t im = 0; im < config.m_num_modes; im++){
            for(size_t id =0; id < config.m_num_detectors; id++){
                for(size_t ic = 0; ic < config.m_num_channels; ic++){
                    TPaveText chi2text(0.59, 0.50, 0.89, 0.59, "NDC");
                    if(io==config.i_prime && need_allcov){
                        log<LOG_INFO>(L"%1% || On channel %2%:") % __func__ % global_channel_index ;
                        // Intentionally uses the UNsubtracted CV and data even under
                        // --bkg-subtract: the chi2 is numerically invariant under the
                        // subtraction, and the fit machinery stays untouched.
                        double chival = allcov_metric->getSingleChannelChi(global_channel_index, variable_cvs[io], io);
                        int ndf = config.m_channel_variable_bins[ic][io].NBinsAlong(0) - bool(opt&PlotOptions::AreaNormalized);
                        log<LOG_INFO>(L"%1% || -- the datamc chi^2/ndof is %2%/%3% .") % __func__ % chival % ndf;
                        chi2text.AddText(("#chi^{2}/ndf = "+chi2LabelValue(chival)+"/"+std::to_string(ndf)).c_str());
                        chi2text.SetFillColor(0);
                        chi2text.SetBorderSize(0);
                        chi2text.SetTextAlign(12);
                        channel_chitexts.push_back(chi2text);
                    }else{
                        chi2text.AddText("");
                    }
                    // For now don't add chi2text to non-prime variables
                    // We just use an empty string anyway and there's a weird
                    // bug that shows up in the ErrorBand plots with this.
                    // (They show "A line segment" in the space where the chi2 would be.)
                    //chi2text.SetFillColor(0);
                    //chi2text.SetBorderSize(0);
                    //chi2text.SetTextAlign(12);
                    //channel_chitexts.push_back(chi2text);
                    global_channel_index++;
                }
            }
        }
        other_channel_chitexts.push_back(channel_chitexts);
    }


    std::vector<PROerrorbar> other_err_bands;
    std::vector<std::map<std::string, TObject*>> errband_objs;
    for(size_t io = 0; io < config.m_num_variables; ++io) {
        if(!config.m_channel_variable_plot_bool.at(io)) { errband_objs.push_back({}); continue; } // For now skip the L/E 250 bin.
        PROspec cv_plot   = variable_cvs[io];
        PROdata data_plot = variable_data[io];
        if (do_bkg_subtract) {
            // Publication convention: the band shows signal-only systematics
            // (each throw's own bkg is subtracted inside
            // getErrorBandBkgSubtracted, so bkg variations cancel); the bkg's
            // systematic and MC-stat uncertainty moves onto the data points.
            // Everything here is in unscaled counts; width/area scaling of the
            // data happens once inside plot_channels.
            PROsubtractedErrorBand sub = getErrorBandBkgSubtracted(config, prop, variable_systs[io],
                    model, variable_cvs[io], CVParams, bkg_subchannels, io, (size_t)options.band_throws);
            other_err_bands.push_back(sub.band);
            cv_plot.Spec() -= build_subchannel_mask_spec(config, cv_plot, bkg_subchannels, io);
            Eigen::VectorXf new_err = (data_plot.Error().array().square()
                                       + sub.bkg_sigma_collapsed.array().square()
                                       + sub.bkg_mcstat_var_collapsed.array()).sqrt();
            data_plot = PROdata(Eigen::VectorXf(data_plot.Spec() - sub.bkg_cv_collapsed), new_err);
        } else {
            other_err_bands.push_back(getErrorBand(config, prop, variable_systs[io], model, variable_cvs[io], CVParams, io, (size_t)options.band_throws, options.area_normalized));
        }
        auto objs = plot_channels(options.final_output_tag+"_PROplot_Variable_"+std::to_string(io)+"_ErrorBand.pdf", config, cv_plot, {}, data_plot,
                other_err_bands.back(), {}, other_channel_chitexts[io], options.pbounds, opt | PlotOptions::DataMCRatio, io,
                false, options.plot_channel_ratios, do_bkg_subtract ? &bkg_subchannels : nullptr,
                io == config.i_prime ? allcov_metric.get() : nullptr,
                io == config.i_prime ? &variable_cvs[io] : nullptr);
        errband_objs.push_back(objs);
    }



    if(options.with_splines) {

        c.Print((options.final_output_tag+"_PROplot_Spline.pdf" + "[").c_str(), "pdf");

        std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> spline_graphs = getSplineGraphs(variable_systs[config.i_prime], config);
        c.Clear();
        c.Divide(4,4);
        int chan=0, snum = 0;
        for(const auto &[syst_name, syst_bins]: spline_graphs) {
            int bin = 0;
            int binning = variable_systs[config.i_prime].spline_binnings[snum++];
            auto plotname = config.m_mcgen_variation_plotname_map.find(syst_name);
            std::string spline_label = plotname == config.m_mcgen_variation_plotname_map.end() ? syst_name : plotname->second;
            bool unprinted = true;
            chan++;
            int col = chan%2==0 ? kRed: kBlue;

            for(const auto &[fixed_pts, curve]: syst_bins) {

                unprinted = true;
                c.cd(bin%16+1);
                size_t sbi = config.GetSubchannelIndexFromVariableGlobalBin(bin,binning);
                std::string nsubchannel = config.GetSubchannelName(sbi);
                size_t local_channel_index = config.GetLocalChannelIndexFromGlobalSubchannelIndex(sbi);
                std::string chan_units = config.GetChannelXAxisTitle(local_channel_index, binning);
                const auto &bins = config.GetChannelVariableBins(local_channel_index, binning);
                size_t local_bin = bin - config.GetGlobalVariableBinStart(sbi, binning);
                std::string bin_detail;
                if(bins.NDim() == 2) {
                    size_t ix = bins.ProjectIndex(local_bin, 0);
                    size_t iy = bins.ProjectIndex(local_bin, 1);
                    const auto ex = bins.Edges(0);
                    const auto ey = bins.Edges(1);
                    size_t sep = chan_units.find(';');
                    std::string xlabel = sep == std::string::npos ? "x" : chan_units.substr(0, sep);
                    std::string ylabel = sep == std::string::npos ? "y" : chan_units.substr(sep + 1);
                    bin_detail = xlabel+" ["+to_string_prec(ex[ix],2)+"->"+to_string_prec(ex[ix+1],2)+"], "+
                                 ylabel+" ["+to_string_prec(ey[iy],2)+"->"+to_string_prec(ey[iy+1],2)+"]";
                } else {
                    size_t ibin = bins.ProjectIndex(local_bin, 0);
                    const auto edges = bins.Edges(0);
                    bin_detail = chan_units+" ["+to_string_prec(edges[ibin],2)+"->"+to_string_prec(edges[ibin+1],2)+"]";
                }

                fixed_pts->SetMarkerColor(col);
                fixed_pts->SetMarkerStyle(kFullCircle);
                fixed_pts->GetXaxis()->SetTitle("#sigma");
                fixed_pts->GetYaxis()->SetTitle("Weight");
                fixed_pts->SetTitle(("#splitline{"+spline_label+"}{#splitline{"+nsubchannel+" bin "+std::to_string(local_bin)+"}{"+bin_detail+"}}").c_str());
                double xlo = curve->GetN() > 0 ? curve->GetX()[0] : -3.0;
                double xhi = curve->GetN() > 0 ? curve->GetX()[curve->GetN()-1] : 3.0;
                double xpad = 0.05 * (xhi - xlo);
                fixed_pts->GetXaxis()->SetLimits(xlo - xpad, xhi + xpad);
                fixed_pts->Draw("PA");
                double max_y = 1.0, min_y = 1.0;
                if(fixed_pts->GetN() > 0 && curve->GetN() > 0) {
                    max_y = std::max(TMath::MaxElement(fixed_pts->GetN(), fixed_pts->GetY()), TMath::MaxElement(curve->GetN(), curve->GetY()));
                    min_y = std::min(TMath::MinElement(fixed_pts->GetN(), fixed_pts->GetY()), TMath::MinElement(curve->GetN(), curve->GetY()));
                }
                double range = max_y - min_y;
                // Flat splines (e.g. norm systs on non-matching subchannels) give
                // max_y == min_y; a zero-height axis makes ROOT emit 'nan' PDF
                // coordinates and whole pages render blank. Pad the range so a flat
                // response draws as a visible horizontal line at its weight.
                if(!std::isfinite(range) || range <= 0) {
                    const double center = std::isfinite(max_y) ? max_y : 1.0;
                    max_y = center + 0.1;
                    min_y = center - 0.1;
                    range = max_y - min_y;
                }
                fixed_pts->SetMaximum(max_y + 0.2 * range);
                fixed_pts->SetMinimum(min_y);

                curve->Draw("C same");
                ++bin;
                if(bin % 16 == 0) {
                    drawVersionWatermark(&c, WatermarkPos::RightEdge);
                    c.Print((options.final_output_tag+"_PROplot_Spline.pdf").c_str(), "pdf");
                    c.Clear();
                    c.Divide(4,4);
                    unprinted = false;
                }
            }
            if(unprinted) {
                drawVersionWatermark(&c, WatermarkPos::RightEdge);
                c.Print((options.final_output_tag+"_PROplot_Spline.pdf").c_str(), "pdf");
            }
        }

        c.Print((options.final_output_tag+"_PROplot_Spline.pdf" + "]").c_str(), "pdf");
        c.Clear();

    }

    // Debug PDF for covariance_to_spline systematics — only emitted if at least one is present.
    log<LOG_INFO>(L"%1% || cov2spline debug info %2%") % __func__ % !variable_systs[config.i_prime].cov2spline_debug_info.empty();
    if(!variable_systs[config.i_prime].cov2spline_debug_info.empty()) {
        const std::string cov2spline_pdf = options.final_output_tag + "_covariance_to_spline_checks.pdf";
        plotCov2SplineChecks(config, variable_cvs[config.i_prime], variable_systs[config.i_prime], cov2spline_pdf, config.i_prime);
        log<LOG_INFO>(L"%1% || covariance_to_spline diagnostics written to %2%") % __func__ % cov2spline_pdf.c_str();
    }

    //now onto root files
    TFile fout((options.final_output_tag+"_PROplot.root").c_str(), "RECREATE");

    int io = 0;
    for(const auto &other: other_hists) {
        for(const auto &[name, hist]: other) {
            hist->Write(("Variable_"+std::to_string(io)+name).c_str());
        }
        io++;
    }

    if((options.fake_data_osc_params.size())) {
        PROspec osc_spec = FillSpectra(config, prop, variable_systs[config.i_prime], model, fakeDataParams, !options.eventbyevent, config.i_prime);
        std::map<std::string, std::unique_ptr<TH1D>> osc_hists = getCV1DHists(osc_spec, config, options.binwidth_scale);
        fout.mkdir("Osc_hists");
        fout.cd("Osc_hists");
        for(const auto &[name, hist]: osc_hists) {
            hist->Write(name.c_str());
        }
    }

    if(options.with_covar) {
        fout.mkdir("Covariance");
        fout.cd("Covariance");
        for(const auto &[name, mat]: matrices)
            mat->Write(name.c_str());
    }

    fout.mkdir("ErrorBand");
    fout.cd("ErrorBand");
    for(size_t i = 0; i < errband_objs.size(); ++i) {
        fout.mkdir(("ErrorBand/Var"+std::to_string(i)).c_str());
        fout.cd(("ErrorBand/Var"+std::to_string(i)).c_str());
        for(const auto &[n, o] : errband_objs[i])
            o->Write(n.c_str());
    }

    if((options.with_splines)) {
        std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> spline_graphs = getSplineGraphs(variable_systs[config.i_prime], config);
        fout.mkdir("Splines");
        fout.cd("Splines");
        for(const auto &[name, syst_splines]: spline_graphs) {
            size_t bin = 0;
            for(const auto &[fixed_pts, curve]: syst_splines) {
                fixed_pts->Write((name+"_fixedpts_"+std::to_string(bin)).c_str());
                curve->Write((name+"_curve_"+std::to_string(bin)).c_str());
                bin++;
            }
        }
    }

    fout.Close();
}
