#include "PROfit_common.h"

log_level_t GLOBAL_LEVEL = LOG_INFO;
log_level_t FILE_LEVEL = LOG_INFO;

std::wostream *OSTREAM = &wcout;

std::wofstream LOG_FILE_STREAM;
bool LOGGING_TO_FILE = false;

int main(int argc, char* argv[])
{
    auto start_time = std::chrono::high_resolution_clock::now();

    gStyle->SetOptStat(0);

    // Define options
    PROpt options(argc, argv);

    log<LOG_WARNING>(L" %1% ") % getIcon().c_str();

    // Grid submission dispatches here, BEFORE PROconfig/CAF/syst setup: it
    // only needs the XML path and tags, not the loaded analysis.
    if(*options.proletariat_command) return run_proletariat(options);

    log<LOG_WARNING>(L"%1% || ##################################################################") % __func__  ;
    log<LOG_WARNING>(L"%1% || ####################### PROfit version v%2% ######################") % __func__ % PROJECT_VERSION_STR ;
    log<LOG_WARNING>(L"%1% || ##################################################################") % __func__  ;
    log<LOG_WARNING>(L"%1% || PROfit commandline input arguments. xml: %2%, tag: %3%, output %4%, nthread: %5% ") % __func__ % options.xmlname.c_str() % options.analysis_tag.c_str() % options.output_tag.c_str() % options.nthread ;

    //Initilize configuration from the XML;
    // fit_variable (-1 unless --fit-variable was given) overrides the XML's fit="true" binning.
    PROconfig config(options.xmlname, options.rateonly, options.fit_variable);

    // Process input files, save in prop and systsstructs
    PROpeller prop;
    std::vector<std::vector<SystStruct>> systsstructs;
    run_process(prop, systsstructs, config, options);

    //Before building, if we fixed a nuisence parameter to a value, lets shift to that value.
    for(auto &sys: options.fixed_params){
           float def = config.m_mcgen_variation_prior_centers.count(sys) ? config.m_mcgen_variation_prior_centers[sys] : 0.0;
           config.m_mcgen_variation_prior_centers[sys]= options.cv_injected_systs.count(sys) ? options.cv_injected_systs.at(sys) : def ;
    }

    // Build internal objects: PROseed, PROmodel, and PROsyst
    PROseed myseed(options.nthread, options.global_seed);
    std::unique_ptr<PROmodel> model = get_model_from_string(config, prop);
    std::vector<PROsyst> variable_systs;
    for(size_t i = 0; i < config.m_num_variables; ++i){
        if(config.m_channel_variable_plot_bool.at(i) || i == config.i_prime){ 
            variable_systs.emplace_back(prop, config, systsstructs.at(i), options.shapeonly, i, model.get(), nullptr);
        }else{
            variable_systs.emplace_back();
        }
    }

    // Setup parameter vectors pre-exclusion/subset of PROsyst (needed to make fake data)
    Eigen::VectorXf fake_data_osc_param_vector = model->default_val;
    Eigen::VectorXf fakedataparams = make_fakedata_params(fake_data_osc_param_vector, config, options, *model, variable_systs[config.i_prime]);

    // variable_data is data for all variable, data is data for i_prime
    std::vector<PROdata> variable_data;
    bool use_real_data = (!options.data_xml.empty() || config.m_has_data_section) && !options.use_fake_data;
    PROdata data = construct_data(variable_data, use_real_data, config, prop, *model, variable_systs, fakedataparams, options);

    // Leave this after creating fake data so we can make fake data using systs that aren't
    // included in the fit.
    include_or_exclude_systs(variable_systs, config, options);

    //***********************************************************************
    //******************** PROjector pre-fit / projected fit ****************
    //***********************************************************************
    // Everything PROjector changes about the inputs (covariance->spline promotion, data
    // masking, prior installation, physics fixing) happens inside this one helper so it
    // is in place before CVParams sizing, the bounds/--fix section, and the metric
    // construction below. See inc/PROjector.h for the scheme.
    options.projector_config.force = options.force;
    if(options.projector_config.active()) {
        if(!PROjectorSetup(options.projector_config, config, variable_systs[config.i_prime],
                    variable_data, data, fakedataparams, options.fixed_params, *model, options.chi2))
            exit(1);
    }

    empty_bin_check(config, options, prop, *model, variable_systs[config.i_prime], data, use_real_data);

    //Pysics parameter input
    //Spline CV  injection studies [NEED TO GO AFTER the remove exclude systs]
    Eigen::VectorXf fakeDataParams = Eigen::VectorXf::Constant(model->nparams + variable_systs[config.i_prime].GetNSplines(), 0);
    Eigen::VectorXf CVParams = Eigen::VectorXf::Constant(model->nparams + variable_systs[config.i_prime].GetNSplines(), 0);
    make_param_vectors(fakeDataParams, CVParams, config, options, *model, variable_systs[config.i_prime], fake_data_osc_param_vector);


    log<LOG_INFO>(L"%1% || Starting from fit preset :  %2%.")% __func__ % options.fit_preset;
    for(auto &fit_pre: options.fit_preset){
        if (options.allowed_preset.find(fit_pre) == options.allowed_preset.end()) {
            log<LOG_ERROR>(L"%1% || ERROR allowed fit_presets are good, fast, sensitivity, overkill, grad-fast, grad-good, grad-deep or grad-overkill. You entred : %2%.")% __func__ % fit_pre.c_str();
            exit(1);
        }
    }
    //Some global minimizer params
    // This runs for the single best gobal fit
    PROfitterConfig fitConfig(options.global_fit_options, options.fit_preset.front(), false);



    //Some Scan minimizer params.
    // This runs lots during PROfile and surface.
    PROfitterConfig scanFitConfig(options.scan_fit_options, options.fit_preset.back(), true);

    // Apply --grad-mode to BOTH fit configurations. PROfitter::Fit calls
    // metric.setGradientMode(...) at the start of every fit, so the same flag
    // controls global fits, profile fits, surface fits, and FC fits uniformly.
    // An empty string (the default) keeps the PROfitterConfig default (analytic).
    // The double-parse here detects an unrecognised token: the parser returns
    // the fallback for unknown input, so calling it with two different
    // sentinels and comparing flags any input that wasn't matched against
    // either of them.
    if (!options.gradient_mode_str.empty()) {
        const PROmetric::GradientMode gmode_a =
            PROmetric::parseGradientMode(options.gradient_mode_str, PROmetric::GradientAnalytic);
        const PROmetric::GradientMode gmode_b =
            PROmetric::parseGradientMode(options.gradient_mode_str, PROmetric::GradientOneSidedFull);
        if (gmode_a != gmode_b) {
            log<LOG_WARNING>(L"%1% || Unknown --grad-mode '%2%'; using the default (analytic).")
                % __func__ % options.gradient_mode_str.c_str();
        }
        fitConfig.gradient_mode     = gmode_a;
        scanFitConfig.gradient_mode = gmode_a;
    }
    log<LOG_INFO>(L"%1% || Gradient mode: %2% (global) / %3% (scan)") % __func__
        % PROmetric::gradientModeName(fitConfig.gradient_mode)
        % PROmetric::gradientModeName(scanFitConfig.gradient_mode);

    size_t N_params = model->nparams+variable_systs[config.i_prime].GetNSplines();
    Eigen::VectorXf global_lb = Eigen::VectorXf::Constant(N_params, -3.0);
    Eigen::VectorXf global_ub = Eigen::VectorXf::Constant(N_params, 3.0);
    std::vector<int> global_fixed(N_params, 0); 
    set_global_bounds(global_lb, global_ub, global_fixed, config, options, *model, variable_systs[config.i_prime], CVParams);

    //Metric Time
    //Metrics are for i_prime only for now
    PROmetric *metric;
    if(options.chi2 == "PROchi") {
        metric = new PROchi("", config, prop, &(variable_systs[config.i_prime]), *model, data, options.eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2, options.shapeonly);
    } else if(options.chi2 == "PROpearson") {
        metric = new PROpearson("", config, prop, &(variable_systs[config.i_prime]), *model, data, options.eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2, options.shapeonly);
    } else if(options.chi2 == "PROCNP") {
        metric = new PROCNP("", config, prop, &(variable_systs[config.i_prime]), *model, data, options.eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2, options.shapeonly);
    } else if(options.chi2 == "Poisson") {
        metric = new PROpoisson("", config, prop, &(variable_systs[config.i_prime]), *model, data, options.eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2,options.shapeonly);
    } else {
        log<LOG_ERROR>(L"%1% || Unrecognized chi2 function %2%") % __func__ % options.chi2.c_str();
        abort();
    }

    // Set color palette for covar and correlation matrices
    set_matrix_palette();

    //***********************************************************************
    //***********************************************************************
    //********************     Run Subcommands    ***************************
    //***********************************************************************
    //***********************************************************************
    
    Eigen::VectorXf global_fit_result;
    float global_fit_chi2 = -1;

    if(*options.proplot_command){
        run_plot(config, prop, *metric, *model, variable_systs, CVParams, fakeDataParams, fake_data_osc_param_vector, variable_data, options);
    }
    if(*options.global_command){
        run_global(global_fit_chi2, global_fit_result, config, prop, *metric, data, CVParams, global_lb, global_ub, global_fixed, fitConfig, options);
    }
    if(*options.profile_command){
        run_profile(global_fit_chi2, global_fit_result, config, prop, *metric, data, CVParams, fakedataparams, global_lb, global_ub, global_fixed, fitConfig, scanFitConfig, options, myseed);
    }
    if(*options.surface_command ){
        run_surface(global_fit_chi2, global_fit_result, config, prop, *metric, data, CVParams, fakeDataParams, global_lb, global_ub, global_fixed, fitConfig, scanFitConfig, options, myseed);
    }
    if(*options.profc_command) {
        run_fc(config, prop, *metric, data, CVParams, fakeDataParams, global_lb, global_ub, global_fixed, fitConfig, scanFitConfig, options, myseed);
    }
    if(*options.afc_command) {
        run_afc(config, prop, *metric, data, fakeDataParams, scanFitConfig, options, myseed);
    }
    if(*options.mcmc_command) {
        run_mcmc(config, *metric, global_lb, global_ub, fitConfig, options, myseed);
    }
    if(*options.bench_command) {
        run_bench(config, prop, *metric, data, fakeDataParams, fitConfig, options);
    }
    if(*options.protest_command){
        run_test(config, prop, *metric, data, CVParams);
    }

    print_global_fit_results(global_fit_chi2, global_fit_result, config, options, *metric);

    delete metric;
    auto stop_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop_time - start_time);
    log<LOG_INFO>(L"%1% || Total run time: %2% seconds") % __func__ % duration.count();

    return 0;
}
