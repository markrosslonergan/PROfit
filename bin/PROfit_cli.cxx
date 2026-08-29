#include "PROfit_common.h"

PROpt::PROpt(int argc, char **argv) {
        //Global Arguments for all PROfit enables subcommands.
        app.add_option("-x,--xml", xmlname, "Input PROfit XML configuration file.")->required();
        app.add_option("-t,--tag", analysis_tag, "Analysis Tag used for output identification.")->default_str("PROfit");
        app.add_option("-v,--verbosity", GLOBAL_LEVEL, "Verbosity Level [1-4]->[Error,Warning,Info,Debug].")->default_val(GLOBAL_LEVEL);
        app.add_option("-l,--log", log_file, "File to save log to. Warning: Will overwrite this file.");
        app.add_option("-w,--file-verbosity", FILE_LEVEL, "File (log) Verbosity Level [1-4]->[Error,Warning,Info,Debug].")->default_val(static_cast<log_level_t>(-1));
        app.add_flag("-b,--progress", progress_bar, "Use a progress bar when applicable.");
        app.add_option("-o,--output", output_tag,"Additional output filename quantifier")->default_str("v1");
        app.add_option("-n, --nthread", nthread, "Number of threads to parallelize over.")->default_val(1);
        app.add_option("-m,--max", maxevents, "Max number of events to run over.");
        app.add_option("-c, --chi2", chi2, "Which chi2 function to use. Options are PROchi, PROpearson, PROCNP, or Poisson")->default_str("PROchi");
        app.add_option("-d, --data", data_xml, "Load from a seperate data xml/data file instead of signal injection. Only used with plot subcommand.")->default_str("");
        app.add_option("-i, --inject", fake_data_osc_params, "Physics parameters to inject as fake-data true signal. Example: dmsq 3 sinsq2thmm 0.25")->expected(-1);
        app.add_option("--inject-cv", cv_osc_params, "Physics parameters to inject as CV. Example: dmsq 3 sinsq2thmm 0.25")->expected(-1);
        app.add_option("--fix", fixed_params, "Fix Certain Physics or Systematics parameters. Fixed to CV.");
        app.add_option("-s, --seed", global_seed, "A global seed for PROseed rng. Default to -1 for hardware rng seed.")->default_val(-1);
        app.add_option("-p,--preset", fit_preset, "Preset fitting params. Available `fast`, `good`, `overkill`, `sensitivity`, plus the analytic-gradient presets `grad-fast`, `grad-good`, `grad-deep`, `grad-overkill`. Takes up to a vector of 2, first for global. 2nd for scan. Defaults to `grad-good` `grad-fast`.");
        app.add_option("--fit-options", global_fit_options, "Parameters for single, detailed global best fit LBFGSB. See PROfitter.h or run --fit-help for available settings.");
        app.add_option("--scan-fit-options", scan_fit_options, "Parameters for simpier, multiple best fits in PROfile/surface LBFGSB.");
        app.add_flag("--fit-help", show_fit_help, "Show detailed help for all fitting parameters (L-BFGS-B, PSO, MCMC, etc.)");
        app.add_option("--grad-mode", gradient_mode_str,
                       "Gradient evaluation strategy passed to the metric. One of: "
                       "central-full (central FD on full chi^2; most accurate FD, slowest), "
                       "one-sided-full (forward FD on full chi^2; ~2x faster, O(h)), "
                       "central-lin (central FD on delta only, M frozen at base; Gauss-Newton, ~5-10x), "
                       "one-sided-lin (forward FD on delta only, M frozen at base; ~10-20x), "
                       "analytic (DEFAULT; exact closed-form gradient incl. the dM/dtheta term; no FD truncation, "
                       "no extra spectrum fills; PROchi binned strategies only — PROCNP, PROpoisson and the "
                       "event-by-event strategy fall back to central-lin with a warning). "
                       "Applies to every fit (global, scan, FC).")
            ->default_str("");

        app.add_option("--inject-systs", injected_systs, "Systematic shifts to inject. Map of name and shift value in sigmas. Only spline systs are supported right now.");
        app.add_option("--inject-systs-cv", cv_injected_systs, "Systematic shifts to inject.  as CV Map of name and shift value in sigmas. Only spline systs are supported right now.");
        app.add_option("--syst-list", syst_list, "Override list of systematics to use (note: all systs must be in the xml).");
        app.add_option("--exclude-systs", systs_excluded, "List of systematics to exclude.")->excludes("--syst-list"); 

        app.add_flag("--use-fake-data", use_fake_data, "Ignore any data XML or embedded <data> section and use fake (MC) data instead.");
        app.add_flag("--poisson-throw", poisson_throw, "Do a Poisson stats throw of fake data.");
        app.add_flag("--pseudo-experiment", pseudo_experiment,
            "Generate a true FC-style pseudo-experiment as fake data: spline Gaussian pulls (rejection-sampled "
            "within each spline's restrict bounds) + covariance-systematic bin shifts via Cholesky factor of the "
            "total covariance + Poisson stats variation. Combines with --inject (the injection sets the underlying "
            "truth signal). Applied only to the i_prime variable. Mutually informative with --poisson-throw, but "
            "the pseudo-experiment already includes its own Poisson step — passing both is redundant.");
        app.add_flag("--scale-by-width", binwidth_scale, "Scale histgrams by 1/(bin width).");
        app.add_flag("--data-mc-ratio", data_mc_ratio, "For ratio plots, use data/pre-fit mc instead of data/best-fit mc.");
        app.add_option("--scale", scale_arg, "Scale detector POT by a given value: <pattern> <factor> pairs, pattern matched against subchannel fullnames as an unanchored regex (plain substrings work).");
        app.add_option("--plot-bounds", bound_list, "Plot bounds, set by  string float pairs. Available strings are ymax,ratmin,ratmax."); 
        app.add_flag("--plot-ratios", plot_channel_ratios,
            "Also draw channel-to-channel ratio spectra within each detector (plot, "
            "global and profile). The two channels must share the same binning for the "
            "ratio to be defined; pairs with different binning are skipped and logged "
            "as a warning.");

        app.add_flag("--event-by-event", eventbyevent, "Do you want to weight event-by-event?");
        app.add_flag("--statonly", statonly, "Run a stats only surface instead of fitting systematics");
        app.add_flag("--force", force,"Force loading binary data even if hash is incorrect (Be Careful!)");
        app.add_flag("--no-xrootd", noxrootd,"Do not use XRootD, which is enabled by default");
        app.add_flag("--syst-only", systs_only, "Force fitting over nuisance parameters only, currently just --fix's them");
        app.add_flag("--area-norm", area_normalized, "Make area normalized histograms.");

        // PROjector: two-stage pre-fit / projected fit (see inc/PROjector.h for the scheme).
        CLI::Option *projector_prefit_opt = app.add_option("--projector-prefit", projector_config.prefit_pattern,
            "PROjector stage 1: wildcard (unanchored regex; plain substrings work, quote it in the shell) matching the subchannels to PRE-FIT (whole "
            "channels only, e.g. a detector name). Covariance systematics are promoted to "
            "eigenmode splines, the data is masked to the matched channels, physics parameters "
            "are fixed at CV, and running the 'global' subcommand writes the nuisance posterior "
            "to <tag>_<output>_PROjector_constraint.bin.");
        app.add_option("--projector", projector_config.constraint_file,
            "PROjector stage 2: path to a constraint file from --projector-prefit. The pre-fit "
            "channels are masked OUT and the saved nuisance posterior is used as a correlated "
            "prior; any subcommand (global, profile, surface, plot, ...) then runs the projected "
            "fit. Requires the same XML/binaries and systematic selection as the pre-fit.")
            ->excludes(projector_prefit_opt);
        app.add_option("--projector-knobs", projector_config.num_decomp_knobs,
            "PROjector: number of covariance eigenmodes promoted to spline knobs in the pre-fit "
            "(-1 = all positive modes, exact, no residual). Stored in the constraint file and "
            "reused automatically in stage 2.")->default_val(-1);
        app.add_option("--projector-keep-cov", projector_config.keep_covariance,
            "PROjector: covariance systematics to NOT promote (they stay as unconstrained "
            "covariance in both stages, e.g. detector-local systematics with no near/far correlation).");
        app.add_flag("--projector-float-physics", projector_config.float_physics,
            "PROjector pre-fit: float the physics parameters instead of fixing them at CV; the "
            "saved posterior is then the physics-marginalized nuisance covariance.");

        auto* shape_flag = app.add_flag("--shapeonly", shapeonly, "Run a shape only analysis");
        auto* rate_flag = app.add_flag("--rateonly", rateonly, "Run a rate only analysis");
        app.add_option("--fit-variable", fit_variable,
                "Index of the variable to fit, overriding the XML's fit=\"true\" binning. Variables are numbered from 0 within a channel, <bins2D> first then <bins>. No re-`process` is needed: all variables are already in the cached binaries.");
        shape_flag->excludes(rate_flag);

        //PROcess, into binary data [Do this once first!]
        process_command = app.add_subcommand("process", "PROcess the MC and systematics in root files into binary data for future rapid loading.");

        //PROsurf, make a 2D surface scan of physics parameters
        surface_command = app.add_subcommand("surface", "Make a 2D surface scan of two physics parameters, profiling over all others.");
        surface_command->add_option("-g, --grid", grid_size, "Set grid size. If one dimension passed, grid assumed to be square, else rectangular")->expected(0, 2)->default_val(40);
        surface_command->add_option("--xvar", xvar, "Name of variable to put on x-axis")->default_val("sinsq2thmm");
        surface_command->add_option("--yvar", yvar, "Name of variable to put on x-axis")->default_val("dmsq");
        xlim_opt = surface_command->add_option("--xlims", xlims, "Limits for x-axis");
        ylim_opt = surface_command->add_option("--ylims", ylims, "Limits for y-axis");
        surface_command->add_option("--xlo", xlo, "Lower limit for x-axis")->excludes(xlim_opt)->default_val(1e-4);
        surface_command->add_option("--xhi", xhi, "Upper limit for x-axis")->excludes(xlim_opt)->default_val(1);
        surface_command->add_option("--ylo", ylo, "Lower limit for y-axis")->excludes(ylim_opt)->default_val(1e-2);
        surface_command->add_option("--yhi", yhi, "Upper limit for y-axis")->excludes(ylim_opt)->default_val(1e2);
        surface_command->add_option("--xlabel", xlabel, "X-axis label");
        surface_command->add_option("--ylabel", ylabel, "Y-axis label");
        surface_command->add_flag("--logx,!--linx", logx, "Specify if x-axis is logarithmic or linear (default log)");
        surface_command->add_flag("--logy,!--liny", logy, "Specify if y-axis is logarithmic or linear (default log)");
        surface_command->add_flag("--brazil-band", run_brazil, "Run throws of stats+systs and draw 1 sigma and 2 sigma Brazil bands");
        surface_command->add_option("--n-brazil-throws", n_brazil_throws, "Number of throws for the Brazil band")->needs("--brazil-band")->default_val(1000);
        surface_command->add_flag("--stat-throws", statonly_brazil, "Only do stat throws for the Brazil band")->needs("--brazil-band");
        surface_command->add_flag("--single-throw", single_brazil, "Only run a single iteration of the Brazil band")->needs("--brazil-band");
        surface_command->add_flag("--only-throw", only_brazil, "Only run Brazil band throws and not the nominal surface")->needs("--brazil-band");
        surface_command->add_option("--from-many", brazil_throws, "Make Brazil band from many provided throws")->needs("--brazil-band");
        surface_command->add_option("--curve-mode", procurve_points , "Make a PROcurve plot from param A to param B.");
        surface_command->add_flag("--surface-amr", use_surface_amr, "Use adaptive-mesh-refinement (PROmesh) instead of the fixed dense grid. Concentrates fits near the target chi^2 contour for ~6-8x wall-time win on equivalent contour quality.");
        surface_command->add_option("--amr-initial", amr_initial, "AMR coarsest grid size (NxN). Default 10.")->default_val(10);
        surface_command->add_option("--amr-levels", amr_levels, "AMR refinement depth. Effective resolution along the contour is amr_initial * 2^amr_levels. Default 3.")->default_val(3);
        surface_command->add_option("--amr-delta", amr_delta, "AMR straddle-band widening (chi^2 units). Refines a cell if any corner is within delta of any contour level. Default 0.5.")->default_val(0.5f);
        surface_command->add_option("--amr-levels-chi2", amr_contour_levels, "Vector of Delta-chi^2 target levels for AMR contour finding. Default {5.99} = 95% CL at 2 dof. Pass e.g. --amr-levels-chi2 2.30 5.99 11.83 for 1/2/3 sigma in one pass.");

        //PROfile, make N profile'd chi^2 for each physics and nuisence parameters
        profile_command = app.add_subcommand("profile", "Make a 1D profiled chi2 for each physics and nuisence parameter.");
        profile_command->add_flag("--mcmc-prefit", MCMC_prefit_errors, "Use MCMC to sample the systematic priors for the pre-fit error band.");
        profile_command->add_flag("--probe", use_probe, "Use PRObe adaptive importance sampling instead of the legacy 18-uniform scan.");
        profile_command->add_option("--probe-chunks", n_probe_chunks, "When --probe is set, split each physics parameter scan into N parallel chunks. Default 1 (no chunking). Useful only when physics scans are the wall-time bottleneck and you have spare threads beyond nuisance work. Hard-capped at nthreads.")->default_val(1);
        profile_command->add_flag("--profile-timing", profile_timing, "Emit a scan-timing summary at end of PROfile (per-fit cost, parallel efficiency, latin/PSO/LBFGS breakdown). Diagnostic only.");

        //PROplot, plot things
        proplot_command = app.add_subcommand("plot", "Make plots of CV, or injected point with error bars and covariance.");
        proplot_command->add_flag("--with-splines", with_splines, "Include graphs of splines in output.");
        proplot_command->add_option("--bkg-subtract", bkg_subtract_pattern,
            "Wildcard (unanchored regex; plain substrings work, quote it in the shell) matching one or more subchannel names; that "
            "background's central-value prediction is subtracted from data and CV "
            "at plot time (publication convention). The error band shows "
            "signal-only systematics: each systematic throw's own background is "
            "subtracted, so background variations cancel out of the band. The "
            "background's uncertainty moves onto the data points instead, which "
            "become N - bkg_CV with errors sqrt(N + sigma_bkg_syst^2 + "
            "sigma_bkg_MCstat^2). Note the signal-background systematic "
            "correlation is retained in the band but not in the data errors. "
            "Example: --bkg-subtract numu_bkg matches every <detector>_numu_bkg "
            "subchannel.");
            
        //PROfc, Feldmand-Cousins
        profc_command = app.add_subcommand("fc", "Run Feldman-Cousins for this injected signal");
        profc_command->add_option("-u,--universes", nuniv, "Number of Feldman Cousins universes to throw")->default_val(1000);
        profc_command->add_flag("--gof", gof_pvalue, "Get GOF pvalue");
        profc_command->add_flag("--pval", pvalue, "Get FC pvalue")->excludes("--gof");

        // PROAdaptiveFC, adaptive FC pipeline. Slice 1: Wilks prepass + meta-mesh + diagnostics.
        afc_command = app.add_subcommand("fc-adaptive",
            "Adaptive Feldman-Cousins. Sub-modes (--mode): build-mesh, init-bank, "
            "print-bank, asimov. Each mode reads/writes <output_tag>-prefixed artifacts. "
            "Typical workflow: build-mesh -> init-bank -> print-bank / asimov.");
        afc_command->add_option("--mode", afc_mode_str,
            "Pipeline mode: build-mesh, init-bank, print-bank, asimov, brazil, merge-mesh, merge-bank. "
            "build-mesh: Wilks prepass -> <tag>_mesh.bin + diagnostic PDFs. "
            "init-bank: requires <tag>_mesh.bin, generates <tag>_bank.bin. "
            "print-bank: load <tag>_bank.bin and write summary PDFs. "
            "asimov: load <tag>_bank.bin and write FC contour + verdict PDFs. "
            "merge-mesh: union-merge >=2 --merge-input mesh binaries into <tag>_mesh.bin. "
            "merge-bank: harvest PEs from >=1 --merge-input bank binaries onto <tag>_mesh.bin. "
            "merge-brazil: union throws from >=1 --merge-input brazil archives into <tag>_brazil.bin, "
            "re-classify against <tag>_bank.bin and emit the band PDF + ROOT (no fits; "
            "bitwise-duplicate throws from same---seed runs are dropped). "
            "brazil-cleanup: mesh densified at the Brazil +-2sigma contours -> <tag>_cleanup_mesh.bin. "
            "print-mesh: plot <tag>_mesh.bin (or --merge-input mesh files) as PDFs.")
            ->default_str("build-mesh");
        afc_command->add_option("--merge-input", afc_merge_inputs,
            "Input artifact filenames for merge-mesh / merge-bank / merge-brazil "
            "(repeatable; glob patterns like 'run*_mesh.bin' are expanded). Output "
            "goes to the normal <output_tag>-prefixed artifacts.")->expected(-1);
        afc_command->add_option("--cleanup-quantiles", afc_cleanup_quantiles,
            "brazil-cleanup: inclusion-fraction quantile levels whose contour "
            "crossings get finest refinement (default 0.025 0.975 = the Brazil "
            "+-2sigma band edges).")->expected(-1);
        afc_command->add_option("--cleanup-halo", afc_cleanup_halo,
            "brazil-cleanup: dilate the flagged contour path by this many finest "
            "bins so the mesh brackets the curve on both sides.")->default_val(1);
        afc_command->add_option("--throws", afc_n_throws,
            "Number of Wilks pre-pass throws (each produces one AMR mesh).")->default_val(200);
        afc_command->add_option("--prepass-amr-initial", afc_prepass_initial,
            "AMR coarsest grid size (one or two ints; default 10 10).")->expected(0, 2);
        afc_command->add_option("--prepass-amr-levels", afc_prepass_levels,
            "AMR refinement depth for the Wilks pre-pass.")->default_val(3);
        afc_command->add_option("--prepass-delta-widen", afc_prepass_delta,
            "AMR straddle-band widening (chi^2 units). Default 0.05: with per-throw global-fit "
            "warm-starts, only cells whose corner range strictly brackets the contour need refinement; "
            "the small non-zero default just absorbs floating-point edge cases. Bump if you want a "
            "visual halo of refined cells around the contour polyline.")->default_val(0.05f);
        afc_command->add_option("--prepass-contour-levels", afc_prepass_contour_levels,
            "Wilks Delta-chi^2 targets per CL (default 2.30 5.99 for 1sigma, 2sigma at 2 dof).");
        afc_command->add_option("--p-thresh", afc_p_thresh,
            "Refine cell in meta-mesh if fraction of throws refining it >= p_thresh.")->default_val(0.05f);
        afc_command->add_option("--baseline-level", afc_baseline_level,
            "Levels strictly below baseline-level are always kept in the meta-mesh.")->default_val(2);
        afc_command->add_flag("--stat-only-throws", afc_stat_only_throws,
            "Use only statistical throws (no systematic throws).");
        afc_command->add_option("--xvar", afc_xvar, "Name of x-axis variable.")->default_str("sinsq2thmm");
        afc_command->add_option("--yvar", afc_yvar, "Name of y-axis variable.")->default_str("dmsq");
        afc_command->add_option("--xlo", afc_xlo, "Lower x-axis limit.")->default_val(1e-4f);
        afc_command->add_option("--xhi", afc_xhi, "Upper x-axis limit.")->default_val(1.0f);
        afc_command->add_option("--ylo", afc_ylo, "Lower y-axis limit.")->default_val(1e-2f);
        afc_command->add_option("--yhi", afc_yhi, "Upper y-axis limit.")->default_val(1e2f);
        afc_command->add_flag("--logx,!--linx", afc_logx, "x-axis log/linear (default log).");
        afc_command->add_flag("--logy,!--liny", afc_logy, "y-axis log/linear (default log).");
        // PE-bank generation knobs (used by --mode init-bank; consumed by asimov/classify too).
        afc_command->add_option("--cl", afc_cl_targets, "Target CLs (one or more).");
        afc_command->add_option("--n-pe-min", afc_n_pe_min,
            "PEs ADDED to each cell on this run, at level == update-layer. Doubles per deeper level: "
            "--n-pe-min 50 adds 50/100/200/400 to L=0/1/2/3 cells. Re-running adds another batch on top.");
        afc_command->add_option("--n-pe-max", afc_n_pe_max,
            "Hard total-per-cell cap. No cell ever exceeds this PE count, even across repeated init-bank runs.");
        afc_command->add_option("--update-layer", afc_update_layer,
            "Only add to cells at AMR level >= update-layer (default 0 = all). Layer L gets n-pe-min PEs added, "
            "deeper layers double. Cells below update-layer keep whatever PEs they already have, untouched.");
        afc_command->add_option("--update-only-layer", afc_only_layer,
            "Only add to cells at AMR level == update-only-layer (no doubling, no other layers). "
            "Default -1 = disabled. Overrides --update-layer when >= 0. "
            "Example: --update-only-layer 2 --n-pe-min 100 adds exactly 100 PEs to L=2 cells, nothing to L=0/1/3.");
        afc_command->add_option("--wilson-eps", afc_wilson_eps,
            "Wilson half-width target. Unused for init-bank now (doubling rule); reserved for slice 2c classification.");
        afc_command->add_option("--n-brazil-throws", afc_n_brazil_throws,
            "Number of pseudo-experiment throws for --mode brazil. Each throw is one FC-style realisation "
            "(syst+stat) classified against the bank. Aggregated into per-cell inclusion fractions and "
            "median +/- 1sigma / +/- 2sigma Brazil-band contours.");
        afc_command->add_option("--flag", afc_flag,
            "Draw the --mode brazil band PDF styled after a national flag (same <tag>_brazil.bin archive). "
            "america: +-1sigma blue with white stars, +-2sigma red/white horizontal stripes. "
            "ireland: alternating green/off-white/orange/off-white vertical stripes (+-2sigma paler).")
            ->check(CLI::IsMember({"america", "ireland"}));
        afc_command->add_option("--roi-band", afc_roi_band, "ROI Delta-chi^2 band (slice 2c).");

        //PROglobal
        global_command = app.add_subcommand("global", "Just do a single global fit.");

        mcmc_command = app.add_subcommand("mcmc", "Get bayesian posteriors using MCMC");
        mcmc_command->add_option("--vars", mcmc_vars, "Variables to find posteriors of.");
        mcmc_command->add_option("--nchains", mcmc_chains, "Number of chains to run with MCMC.")->default_val(1);
        mcmc_command->add_flag("--hmc", hmc, "Run Hamiltonian MC instead of Metropolis");

        //PROtest, test things
        protest_command = app.add_subcommand("protest", "Testing ground for rapid quick tests.");

        //PRObench, scaling/timing benchmarks. Loud greppable LOG output via [SCALETEST] tag.
        // Uses the live PROmetric built by the main chain (PROchi/PROCNP/PROpoisson)
        // — no separate metric-class flag needed here.
        bench_command = app.add_subcommand("scale-test", "Run timing benchmarks for FillSpectra / metric / fit hot paths and emit greppable [SCALETEST] LOG lines.");
        bench_command->add_option("-N,--n", bench_N, "Base call count: FillSpectra=N, metric=N/10, fit=N/100.")->default_val(1000);
        bench_command->add_option("--tests", bench_tests_str, "Comma-separated subset of {a..p} or {fillspectra,metric,metricgrad,fit,pseudo,collapse,mcmc,all,gradcheck,gradmodes,grad}. Default 'all'. gradcheck (o) validates every gradient mode vs central-full FD; gradmodes (p) runs N/100 full fits per (fit preset x gradient mode) with matched seeds and emits [GRADBENCH] lines. Neither is in 'all'.")->default_val("all");
        bench_command->add_option("--grad-presets", bench_grad_presets, "Comma-separated preset names restricting the gradmodes (p) benchmark grid, e.g. 'grad-fast,grad-good'. Empty = all presets. Universes/seeds depend only on the fixed rng seed, so a filtered run's CSV rows can be concatenated with an earlier full run's.")->default_val("");
        bench_command->add_flag("--throw-systs", bench_throw_systs, "gradmodes (p) benchmark: generate universes as FC-style pseudo-experiments (thrown spline pulls + covariance shift + Poisson) instead of Poisson-only fluctuations. CSV name gains a _syst suffix.");
        bench_command->add_flag("--throw-phys", bench_throw_phys, "gradmodes (p) benchmark: draw each universe's truth physics point uniformly within the fit bounds (per-universe truth columns added to the CSV). CSV name gains a _phys/_systphys suffix.");

        //PROletariat, stage+tar+submit grid jobs (replaces grid/maketar_submit_v2.4.sh)
        proletariat_command = app.add_subcommand("proletariat",
            "Stage the PROfit binary, XML and analysis artifacts into grid_dir.tar and submit N grid jobs running a worker script via jobsub_submit. Replaces grid/maketar_submit_v2.4.sh.");
        proletariat_command->add_option("--script", grid_opts.script,
            "Worker script executed on each grid node (e.g. grid/runFC_v2.4_v4_AL9.sh, which handles both AL9/Spack and SL7/UPS).")->required();
        proletariat_command->add_option("-N,--n-jobs", grid_opts.njobs, "Number of grid jobs.")->default_val(2);
        proletariat_command->add_option("--lifetime", grid_opts.lifetime, "Expected job lifetime (3d is the FermiGrid ceiling).")->default_str("2d");
        proletariat_command->add_option("--memory", grid_opts.memory_mb, "Requested memory in MB.")->default_val(4000);
        proletariat_command->add_option("--disk", grid_opts.disk_mb, "Requested scratch disk in MB.")->default_val(10000);
        proletariat_command->add_option("--input", grid_opts.extra_inputs,
            "Extra file(s) to bundle into the tarball (repeatable). A missing file is a hard error. The XML and any <tag>_prop.bin/_syst.bin and <tag>_<output>_mesh.bin/_bank.bin in the current directory are bundled automatically.");
        proletariat_command->add_flag("--dry-run", grid_opts.dry_run,
            "Stage and build the tarball, print the exact jobsub_submit command, do not submit.");
        proletariat_command->add_option("--backend", grid_backend_str, "Scheduler backend: jobsub or slurm (slurm not yet implemented).")->default_str("jobsub");
        proletariat_command->add_option("--group", grid_opts.group, "jobsub experiment group (-G).")->default_str("sbnd");
        proletariat_command->add_option("--role", grid_opts.role, "jobsub --role.")->default_str("Analysis");
        CLI::Option *grid_image_opt = proletariat_command->add_option("--singularity-image", grid_opts.singularity_image,
            "Apptainer/Singularity image path (default: the AL9 image " + std::string(PROletariatOptions::kImageAL9) + "; see also --sl7).");
        proletariat_command->add_flag("--sl7", grid_sl7,
            "Submit with the legacy SL7 container image instead of the default AL9 (el9) one. Worker scripts detect the OS at runtime and use UPS (SL7) or Spack (AL9) setup.")->excludes(grid_image_opt);
        proletariat_command->add_option("--resource-provides", grid_opts.resource_provides, "jobsub --resource-provides usage model.");
        proletariat_command->add_option("--lines", grid_opts.condor_lines,
            "Condor classad --lines entries. REPLACES the FERMIHTC defaults when given; to append instead, use --jobsub-arg.");
        proletariat_command->add_option("--jobsub-arg", grid_opts.extra_jobsub_args, "Extra raw argument passed through to jobsub_submit verbatim (repeatable).");
        proletariat_command->add_option("--profit-bin", grid_opts.profit_bin, "Override the PROfit binary to ship (default: this executable, via /proc/self/exe).");

        app.set_config("--config");
        surface_command->configurable(true);
        process_command->configurable(true);
        profile_command->configurable(true);
        protest_command->configurable(true);
        global_command->configurable(true);
        profc_command->configurable(true);
        afc_command->configurable(true);
        proplot_command->configurable(true);
        mcmc_command->configurable(true);
        bench_command->configurable(true);
        proletariat_command->configurable(true);

        // This is an ugly hack to deal with CLI11_PARSE using return for --help
        // We could always just run CLI11_PARSE in the main function to avoid this.
        try {
            int r = [this, argc, argv](){
                //Parse inputs. 
                CLI11_PARSE(app, argc, argv);
                throw std::runtime_error("");
            }();
            exit(r);
        } catch(std::runtime_error &e) {
            // ignore, we parsed params correctly
        }

        if(show_fit_help) {
            PROfit::PROfitterConfig::PrintHelp();
            exit(0);
        }

        if(log_file != "") {

            if(FILE_LEVEL == static_cast<log_level_t>(-1)) {
                FILE_LEVEL = GLOBAL_LEVEL;
            }

            log_impl::EnableFileLogging(log_file, FILE_LEVEL);
        }

        if(shapeonly) area_normalized = true;

        pbounds.Load(bound_list);
        final_output_tag = analysis_tag +"_"+ output_tag;
}
