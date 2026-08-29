#include "PROfit_common.h"

void run_surface(float &global_fit_chi2, Eigen::VectorXf &global_fit_result, const PROconfig &config, const PROpeller &prop, PROmetric &metric, const PROdata &data, const Eigen::VectorXf &CVParams, const Eigen::VectorXf &fakeDataParams, const Eigen::VectorXf &lb, const Eigen::VectorXf &ub, const std::vector<int> &fixed, PROfitterConfig &fitConfig, PROfitterConfig &scanFitConfig, PROpt &options, PROseed &myseed) {

    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());

    if(global_fit_chi2 < 0) {
        GlobalFitOptions opt = GlobalFitOptions::Default;
        if(options.progress_bar) opt |= GlobalFitOptions::Progress;
        opt |= GlobalFitOptions::FreqSeedPts;
        PROspec cv = FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), CVParams , true ,config.i_prime);
        // Should we pass in global fixed here? This mostly gets used with syst_only which would not make sense for a surface.
        GlobalFitResult fitres = run_global_fit(config, prop, data, metric, ub, lb, fitConfig, CVParams, cv, fixed, opt); 
        global_fit_chi2 = fitres.chi2;
        global_fit_result = fitres.fitter.best_fit;
    }

    if (options.grid_size.empty()) {
        options.grid_size = {40, 40};
    }
    if (options.grid_size.size() == 1) {
        options.grid_size.push_back(options.grid_size[0]); //make it square
    }

    if(*options.xlim_opt) {
        options.xlo = options.xlims[0];
        options.xhi = options.xlims[1];
    }
    if(*options.ylim_opt) {
        options.ylo = options.ylims[0];
        options.yhi = options.ylims[1];
    }

    //Define grid and Surface
    size_t xaxis_idx = 1, yaxis_idx = 0;
    if(const auto loc = std::find(metric.GetModel().param_names.begin(), metric.GetModel().param_names.end(), options.xvar); loc != metric.GetModel().param_names.end()) {
        xaxis_idx = std::distance(metric.GetModel().param_names.begin(), loc);

    } else if(const auto loc = std::find(metric.GetSysts().spline_names.begin(), metric.GetSysts().spline_names.end(), options.xvar); loc != metric.GetSysts().spline_names.end()) {
        xaxis_idx = std::distance(metric.GetSysts().spline_names.begin(), loc);
    } else {
        for(const auto &[xml_name, plot_name]: config.m_mcgen_variation_plotname_map) {
            if(options.xvar == plot_name) {
                const auto loc = std::find(metric.GetSysts().spline_names.begin(), metric.GetSysts().spline_names.end(), xml_name);
                if(loc != metric.GetSysts().spline_names.end()) {
                    xaxis_idx = std::distance(metric.GetSysts().spline_names.begin(), loc);
                }
                break;
            }
        }
    }
    if(const auto loc = std::find(metric.GetModel().param_names.begin(), metric.GetModel().param_names.end(), options.yvar); loc != metric.GetModel().param_names.end()) {
        yaxis_idx = std::distance(metric.GetModel().param_names.begin(), loc);
    } else if(const auto loc = std::find(metric.GetSysts().spline_names.begin(),metric.GetSysts().spline_names.end(), options.yvar); loc != metric.GetSysts().spline_names.end()) {
        yaxis_idx = std::distance(metric.GetSysts().spline_names.begin(), loc);
    } else {
        for(const auto &[xml_name, plot_name]: config.m_mcgen_variation_plotname_map) {
            if(options.yvar == plot_name) {
                const auto loc = std::find(metric.GetSysts().spline_names.begin(), metric.GetSysts().spline_names.end(), xml_name);
                if(loc != metric.GetSysts().spline_names.end()) {
                    yaxis_idx = std::distance(metric.GetSysts().spline_names.begin(), loc);
                }
                break;
            }
        }

    }
    size_t nbinsx = options.grid_size[0], nbinsy = options.grid_size[1];
    PROsurf surface(metric, xaxis_idx, yaxis_idx, nbinsx, options.logx ? PROsurf::LogAxis : PROsurf::LinAxis, options.xlo, options.xhi,
            nbinsy, options.logy ? PROsurf::LogAxis : PROsurf::LinAxis, options.ylo, options.yhi);

    if(options.procurve_points.size()!=0){

        size_t mid = options.procurve_points.size() / 2;
        std::vector<float> A(options.procurve_points.begin(), options.procurve_points.begin() + mid);
        std::vector<float> B(options.procurve_points.begin() + mid, options.procurve_points.end());
        size_t Ncurvep = options.grid_size.front();
        log<LOG_INFO>(L"%1% || Running a PROcurve from %2% to point %3% with %4% points") % __func__ % A% B %Ncurvep;
        
        if(options.progress_bar) fitConfig.progress_bar = true;
        std::vector<surfOut> cpoints = surface.FillCurve(fitConfig, myseed, global_fit_chi2, global_fit_result, options.nthread, A, B, Ncurvep);
        surface.PlotCurve(config, metric.GetModel(), metric.GetSysts(), cpoints,options.final_output_tag,options.logx,options.logy,xaxis_idx,yaxis_idx,A, B, Ncurvep); 
        exit(0);
    }


    if(options.progress_bar) scanFitConfig.progress_bar = true;
    if(!options.only_brazil) {
        if(options.statonly) {
            surface.FillSurfaceStat(config, scanFitConfig, options.final_output_tag+"_statonly_surface.txt", CVParams, dseed(myseed.global_rng));
        } else if (options.use_surface_amr) {
            // Adaptive-mesh-refinement path (PROmesh::run_amr) — concentrates evaluations
            // near the target contour. Reuses surface.surface() for plot-compat via the
            // bilinear-reconstructed dense matrix.
            PROmesh::AMROptions opts;
            opts.initial_nx     = options.amr_initial;
            opts.initial_ny     = options.amr_initial;
            opts.max_levels     = options.amr_levels;
            opts.delta_widen    = options.amr_delta;
            opts.dense_nx       = (int)surface.nbinsx;
            opts.dense_ny       = (int)surface.nbinsy;
            opts.produce_dense  = true;
            if (!options.amr_contour_levels.empty()) opts.contour_levels = options.amr_contour_levels;
            // Use the global-fit best fit (from project-SBN-dev's run_global_fit
            // pre-pass) as a warm-start seed for AMR's initial level-0 grid.
            // Subsequent level fits still get cell-corner best_fits from
            // PROmesh::run_amr.
            std::vector<Eigen::VectorXf> caller_seeds;
            if (global_fit_result.size() > 0)
                caller_seeds.push_back(global_fit_result);
            PROmesh::AMRResult amr_result = surface.FillSurfaceAMR(
                scanFitConfig,
                options.final_output_tag+"_surface_amr.txt",
                myseed, options.nthread,
                caller_seeds,
                opts);
            // Mesh visualisation: cells coloured by refinement level + the
            // contour polylines overlaid in red. Saved next to the heatmap.
            surface.PlotAMRMesh(amr_result, metric.GetModel(), options.final_output_tag,
                                options.logx, options.logy, xaxis_idx, yaxis_idx);
        } else {
            surface.FillSurface(scanFitConfig, options.final_output_tag+"_surface.txt",myseed, global_fit_chi2, global_fit_result, options.nthread);
        }
    }

    std::vector<float> binedges_x, binedges_y;
    // Edges are stored in model's native space (log if is_log10, linear otherwise)
    // Convert to linear for ROOT histogram bin edges
    for(size_t i = 0; i < surface.nbinsx+1; i++)
        binedges_x.push_back(metric.GetModel().is_log10[xaxis_idx] ? std::pow(10, surface.edges_x(i)) : surface.edges_x(i));
    for(size_t i = 0; i < surface.nbinsy+1; i++)
        binedges_y.push_back(metric.GetModel().is_log10[yaxis_idx] ? std::pow(10, surface.edges_y(i)) : surface.edges_y(i));

    if(options.xlabel == "") 
        options.xlabel = xaxis_idx < metric.GetModel().nparams ? metric.GetModel().pretty_param_names[xaxis_idx] : 
            config.m_mcgen_variation_plotname_map.at(metric.GetSysts().spline_names[xaxis_idx]);
    if(options.ylabel == "") 
        options.ylabel = yaxis_idx < metric.GetModel().nparams ? metric.GetModel().pretty_param_names[yaxis_idx] : 
            config.m_mcgen_variation_plotname_map.at(metric.GetSysts().spline_names[yaxis_idx]);
    TH2D surf("surf", (";"+options.xlabel+";"+options.ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());

    for(size_t i = 0; i < surface.nbinsx; i++) {
        for(size_t j = 0; j < surface.nbinsy; j++) {
            surf.SetBinContent(i+1, j+1, surface.surface(i, j));
        }
    }

    log<LOG_INFO>(L"%1% || Saving surface to %2% as TH2D named \"surf.\"") % __func__ % options.final_output_tag.c_str();
    TFile fout((options.final_output_tag+"_surf.root").c_str(), "RECREATE");
    if(!options.only_brazil) {
        surf.Write();
        float chisq;
        int xbin, ybin;
        std::map<std::string, float> best_fit;
        TTree tree("tree", "BestFitTree");
        tree.Branch("chi2", &chisq); 
        tree.Branch("xbin", &xbin); 
        tree.Branch("ybin", &ybin); 
        tree.Branch("best_fit", &best_fit); 

        for(const auto &res: surface.results) {
            chisq = res.chi2;
            xbin = res.binx;
            ybin = res.biny;
            // If all fit points fail
            if(!res.best_fit.size()) { tree.Fill(); continue; }
            for(size_t i = 0; i < metric.GetModel().nparams; ++i) {
                best_fit[metric.GetModel().param_names[i]] = res.best_fit(i);
            }
            for(size_t i = 0; i < metric.GetSysts().GetNSplines(); ++i) {
                best_fit[metric.GetSysts().spline_names[i]] = res.best_fit(i + metric.GetModel().nparams);
            }
            tree.Fill();
        }
        // TODO: Should we save the spectra as TH1s?

        tree.Write();

        TCanvas c;
        if(options.logy)
            c.SetLogy();
        if(options.logx)
            c.SetLogx();
        c.SetLogz();
        gStyle->SetPalette(kViridis);
        surf.Draw("colz");
        c.Print((options.final_output_tag+"_surface.pdf").c_str());
    }

    std::vector<PROsurf> brazil_band_surfaces;
    if(options.run_brazil && options.brazil_throws.size() == 0) {
        std::normal_distribution<float> d;
        PROspec cv = FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), CVParams , true,config.i_prime);

        PROspec collapsed_cv = PROspec(CollapseMatrix(config, cv.Spec()), CollapseMatrix(config, cv.Error()));
        Eigen::MatrixXf L = metric.GetSysts().DecomposeFractionalCovariance(config, cv.Spec());
        for(size_t i = 0; i < (size_t)options.n_brazil_throws; ++i) {
            Eigen::VectorXf throwp = fakeDataParams;
            Eigen::VectorXf throwC = Eigen::VectorXf::Constant(config.m_num_variable_bins_total_collapsed[config.i_prime], 0);
            // Shared truncated-Gaussian helper: samples each spline's actual prior
            // N(center, sigma) within its restrict bounds (the raw d(rng) here
            // ignored both, which was wrong for XML priors and for PROjector's
            // constrained posterior).
            for(size_t i = 0; i < metric.GetSysts().GetNSplines(); i++)
                throwp(i+metric.GetModel().nparams) = ThrowRestrictedSplinePull(metric.GetSysts(), i, PROseed::global_rng, d);
            for(size_t i = 0; i < config.m_num_variable_bins_total_collapsed[config.i_prime]; i++)
                throwC(i) = d(PROseed::global_rng);
            bool binned = (options.eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2) != 0;
            // Fill the fitting variable explicitly: the CollapseMatrix below uses the
            // i_prime collapsing matrix, so letting var_index default to 0 would mix
            // variables whenever i_prime != 0 (same fix as src/PROfc.cxx).
            PROspec shifted = FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), throwp, binned, config.i_prime);
            PROspec newSpec = options.statonly_brazil ? PROspec::PoissonVariation(collapsed_cv, dseed(myseed.global_rng)) :
                PROspec::PoissonVariation(PROspec(CollapseMatrix(config, shifted.Spec()) + L * throwC, CollapseMatrix(config, shifted.Error())), dseed(myseed.global_rng));
            PROdata data(newSpec.Spec(), newSpec.Error());
            PROmetric *brazil_metric;
            if(options.chi2 == "PROchi") {
                brazil_metric = new PROchi("", config, prop, &metric.GetSysts(), metric.GetModel(), data, options.eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
            } else if(options.chi2 == "PROpearson") {
                brazil_metric = new PROpearson("", config, prop, &metric.GetSysts(), metric.GetModel(), data, options.eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
            } else if(options.chi2 == "PROCNP") {
                brazil_metric = new PROCNP("", config, prop, &metric.GetSysts(), metric.GetModel(), data, options.eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
            } else if(options.chi2 == "Poisson") {
                brazil_metric = new PROpoisson("", config, prop, &metric.GetSysts(), metric.GetModel(), data, options.eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
            } else {
                log<LOG_ERROR>(L"%1% || Unrecognized chi2 function %2%") % __func__ % options.chi2.c_str();
                abort();
            }
            GlobalFitOptions opt = GlobalFitOptions::Default;
            if(options.progress_bar) opt |= GlobalFitOptions::Progress;
            opt |= GlobalFitOptions::FreqSeedPts;
            PROspec cv = FillSpectra(config, prop, brazil_metric->GetSysts(), brazil_metric->GetModel(), CVParams , true ,config.i_prime);
            GlobalFitResult fitres = run_global_fit(config, prop, data, *brazil_metric, ub, lb, fitConfig, CVParams, cv, fixed, opt); 

            brazil_band_surfaces.emplace_back(*brazil_metric, xaxis_idx, yaxis_idx, nbinsx, options.logx ? PROsurf::LogAxis : PROsurf::LinAxis, options.xlo, options.xhi,
                    nbinsy, options.logy ? PROsurf::LogAxis : PROsurf::LinAxis, options.ylo, options.yhi);

            if(options.statonly)
                brazil_band_surfaces.back().FillSurfaceStat(config, scanFitConfig, "", CVParams, dseed(myseed.global_rng));
            else
                brazil_band_surfaces.back().FillSurface(scanFitConfig, "", myseed, fitres.chi2, fitres.fitter.best_fit, options.nthread);

            TH2D surf("surf", (";"+options.xlabel+";"+options.ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());

            for(size_t i = 0; i < surface.nbinsx; i++) {
                for(size_t j = 0; j < surface.nbinsy; j++) {
                    surf.SetBinContent(i+1, j+1, brazil_band_surfaces.back().surface(i, j));
                }
            }
            surf.Write(("brazil_throw_surf_"+std::to_string(i)).c_str());

            // WARNING: Metric reference stored in surface. DO NOT USE IT AFTER THIS POINT.
            delete brazil_metric;
            if(options.single_brazil) break;
        }
    } else if(options.run_brazil) { // if brazil_thows.size() > 0
        for(const std::string &in: options.brazil_throws) {
            brazil_band_surfaces.emplace_back(metric, xaxis_idx, yaxis_idx, nbinsx, options.logx ? PROsurf::LogAxis : PROsurf::LinAxis, options.xlo, options.xhi,
                    nbinsy, options.logy ? PROsurf::LogAxis : PROsurf::LinAxis, options.ylo, options.yhi);

            TFile fin(in.c_str());
            // TODO: Check that axes and labels are the same
            TH2D *surf = fin.Get<TH2D>("brazil_throw_surf_0");
            if(!surf) {
                log<LOG_ERROR>(L"%1% || Could not find a TH2D called 'surf' in the file %2%. Skipping this file.")
                    % __func__ % in.c_str();
                continue;
                //return EXIT_FAILURE;
            }
            for(size_t i = 0; i < surface.nbinsx; ++i) {
                for(size_t j = 0; j < surface.nbinsy; ++j) {
                    brazil_band_surfaces.back().surface(i,j) = surf->GetBinContent(i+1,j+1);
                }
            }
        }
    }

    if(options.run_brazil && !options.single_brazil) {
        TH2D surf16("surf16", (";"+options.xlabel+";"+options.ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());
        TH2D surf84("surf84", (";"+options.xlabel+";"+options.ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());
        TH2D surf98("surf98", (";"+options.xlabel+";"+options.ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());
        TH2D surf02("surf02", (";"+options.xlabel+";"+options.ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());
        TH2D surf50("surf50", (";"+options.xlabel+";"+options.ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());

        for(size_t i = 0; i < surface.nbinsx; ++i) {
            for(size_t j = 0; j < surface.nbinsy; ++j) {
                std::vector<float> values;
                for(const auto &bbsurf: brazil_band_surfaces)
                    values.push_back(bbsurf.surface(i,j));
                std::sort(values.begin(), values.end());
                surf02.SetBinContent(i+1, j+1, values[(size_t)(0.023 * values.size())]);
                surf16.SetBinContent(i+1, j+1, values[(size_t)(0.159 * values.size())]);
                surf50.SetBinContent(i+1, j+1, values[(size_t)(0.500 * values.size())]);
                surf84.SetBinContent(i+1, j+1, values[(size_t)(0.841 * values.size())]);
                surf98.SetBinContent(i+1, j+1, values[(size_t)(0.977 * values.size())]);
            }
        }

        fout.cd();
        surf02.Write();
        surf16.Write();
        surf50.Write();
        surf84.Write();
        surf98.Write();
        if(options.brazil_throws.size() != 0) {
            int i = 0;
            for(const auto &bbsurf: brazil_band_surfaces) {
                TH2D *surf = new TH2D(("brazil_throw_surf_"+std::to_string(i)).c_str(),(";"+options.xlabel+";"+options.ylabel).c_str(),surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data())     ;
                for(size_t j = 0; j < surface.nbinsx; ++j) {
                    for(size_t k = 0; k < surface.nbinsy; ++k) {
                        surf->SetBinContent(j+1,k+1, bbsurf.surface(j,k));
                    }
                }
                surf->Write();
                ++i;
            }
        }
    }
}
