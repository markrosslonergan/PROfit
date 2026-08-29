#include "PROfit_common.h"

void run_afc(const PROconfig &config, const PROpeller &prop, PROmetric &metric, const PROdata &data, const Eigen::VectorXf &fakeDataParams, const PROfitterConfig &scanFitConfig, const PROpt &options, PROseed &myseed) {
    PROfit::AdaptiveFCConfig acfg;
    if      (options.afc_mode_str == "build-mesh") acfg.mode = PROfit::AdaptiveFCMode::BuildMesh;
    else if (options.afc_mode_str == "init-bank")  acfg.mode = PROfit::AdaptiveFCMode::InitBank;
    else if (options.afc_mode_str == "print-bank") acfg.mode = PROfit::AdaptiveFCMode::PrintBank;
    else if (options.afc_mode_str == "asimov")     acfg.mode = PROfit::AdaptiveFCMode::Asimov;
    else if (options.afc_mode_str == "brazil")     acfg.mode = PROfit::AdaptiveFCMode::Brazil;
    else if (options.afc_mode_str == "classify")   acfg.mode = PROfit::AdaptiveFCMode::Classify;
    else if (options.afc_mode_str == "merge-mesh") acfg.mode = PROfit::AdaptiveFCMode::MergeMesh;
    else if (options.afc_mode_str == "merge-bank") acfg.mode = PROfit::AdaptiveFCMode::MergeBank;
    else if (options.afc_mode_str == "merge-brazil") acfg.mode = PROfit::AdaptiveFCMode::MergeBrazil;
    else if (options.afc_mode_str == "brazil-cleanup") acfg.mode = PROfit::AdaptiveFCMode::BrazilCleanup;
    else if (options.afc_mode_str == "print-mesh") acfg.mode = PROfit::AdaptiveFCMode::PrintMesh;
    else {
        log<LOG_WARNING>(L"%1% || fc-adaptive: unknown --mode '%2%', defaulting to build-mesh.")
            % __func__ % options.afc_mode_str.c_str();
        acfg.mode = PROfit::AdaptiveFCMode::BuildMesh;
    }
    acfg.n_throws = options.afc_n_throws;
    if (options.afc_prepass_initial.size() == 1) {
        acfg.prepass_amr_initial_x = options.afc_prepass_initial[0];
        acfg.prepass_amr_initial_y = options.afc_prepass_initial[0];
    } else if (options.afc_prepass_initial.size() >= 2) {
        acfg.prepass_amr_initial_x = options.afc_prepass_initial[0];
        acfg.prepass_amr_initial_y = options.afc_prepass_initial[1];
    }
    acfg.prepass_amr_levels    = options.afc_prepass_levels;
    acfg.prepass_delta_widen   = options.afc_prepass_delta;
    acfg.prepass_contour_levels = options.afc_prepass_contour_levels;
    acfg.p_thresh        = options.afc_p_thresh;
    acfg.baseline_level  = options.afc_baseline_level;
    acfg.stat_only_throws = options.afc_stat_only_throws;
    acfg.xvar = options.afc_xvar;
    acfg.yvar = options.afc_yvar;
    acfg.x_lo = options.afc_xlo; acfg.x_hi = options.afc_xhi;
    acfg.y_lo = options.afc_ylo; acfg.y_hi = options.afc_yhi;
    acfg.logx = options.afc_logx; acfg.logy = options.afc_logy;
    acfg.output_tag = options.final_output_tag;
    acfg.chi2 = options.chi2;
    acfg.binned = !options.eventbyevent;
    acfg.cl_targets = options.afc_cl_targets;
    acfg.wilson_eps = options.afc_wilson_eps;
    acfg.n_pe_min = options.afc_n_pe_min;
    acfg.n_pe_max = options.afc_n_pe_max;
    acfg.update_layer = options.afc_update_layer;
    acfg.only_layer = options.afc_only_layer;
    acfg.n_brazil_throws = options.afc_n_brazil_throws;
    acfg.band_flag = options.afc_flag;
    if (!options.afc_flag.empty() && acfg.mode != PROfit::AdaptiveFCMode::Brazil
        && acfg.mode != PROfit::AdaptiveFCMode::MergeBrazil) {
        log<LOG_WARNING>(L"%1% || fc-adaptive: --flag %2% only styles the brazil / merge-brazil band PDF; ignored for --mode %3%.")
            % __func__ % options.afc_flag.c_str() % options.afc_mode_str.c_str();
    }
    acfg.roi_band = options.afc_roi_band;
    acfg.cleanup_quantiles = options.afc_cleanup_quantiles;
    acfg.cleanup_halo      = options.afc_cleanup_halo;

    // Expand --merge-input entries: each may be a literal filename or a
    // glob pattern (quoted through the shell, e.g. 'run*_mesh.bin').
    // Unmatched patterns are kept verbatim so the merge modes report a
    // clean per-file load error instead of silently shrinking the list.
    for (const auto &pattern : options.afc_merge_inputs) {
        glob_t g;
        const int rc = glob(pattern.c_str(), GLOB_TILDE, nullptr, &g);
        if (rc == 0 && g.gl_pathc > 0) {
            for (size_t p = 0; p < g.gl_pathc; ++p)
                acfg.merge_inputs.push_back(g.gl_pathv[p]);
        } else {
            acfg.merge_inputs.push_back(pattern);
        }
        globfree(&g);
    }
    // Dedupe while preserving order (a glob plus an explicit filename can
    // both match the same artifact).
    {
        std::set<std::string> seen_inputs;
        std::vector<std::string> uniq;
        for (auto &f : acfg.merge_inputs)
            if (seen_inputs.insert(f).second) uniq.push_back(f);
        acfg.merge_inputs = std::move(uniq);
    }
    if (!acfg.merge_inputs.empty()) {
        log<LOG_INFO>(L"%1% || fc-adaptive: %2% merge input(s) after glob expansion.")
            % __func__ % (int)acfg.merge_inputs.size();
    }

    // Outer "AFC throws" bar is only meaningful for build-mesh (the prepass
    // throws each tick this bar via generate_throws). Other modes (init-bank,
    // print-bank, asimov, brazil) create their own dedicated bars sized for
    // the work they actually do — they don't use this one. Starting its
    // display in those modes leaves "AFC throws X/N_throws" frozen above
    // the real bar and looks like overcounting. So: skip display for
    // non-BuildMesh and skip the round-up at finish.
    const bool needs_outer_throws_bar =
        (acfg.mode == PROfit::AdaptiveFCMode::BuildMesh);
    std::vector<std::pair<int, std::string>> afc_PB_configs;
    afc_PB_configs.push_back({
        needs_outer_throws_bar ? acfg.n_throws : 1,
        "AFC throws"});
    MultiPROgressBar afc_progress(afc_PB_configs);
    if (needs_outer_throws_bar) {
        afc_progress.initialize_display();
        afc_progress.start_display_thread();
    }

    PROfit::AdaptiveFCResult ares = PROfit::run_adaptive_fc(
        config, prop, metric.GetSysts(), scanFitConfig,
        myseed, fakeDataParams, data, acfg, options.nthread, afc_progress);

    if (needs_outer_throws_bar) afc_progress.finish_all();
    else                         afc_progress.finish_all(false);
    log<LOG_INFO>(L"%1% || fc-adaptive done: throws=%2%, meta_cells=%3% (baseline=%4%, refined=%5%), "
                  L"diag=%6%, bank=%7% (pes=%8%, mean/cell=%9%, topped_up=%10%, capped=%11%).")
        % __func__ % ares.n_throws_done % ares.n_meta_cells
        % ares.n_baseline_cells % ares.n_refined_cells % ares.diag_root_path.c_str()
        % (ares.bank_path.empty() ? "<not written>" : ares.bank_path.c_str())
        % (int64_t)ares.total_pes_generated % ares.mean_pes_per_cell
        % ares.cells_topped_up % ares.cells_hit_n_pe_max;

}
