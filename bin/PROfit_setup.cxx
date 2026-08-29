#include "PROfit_common.h"

PROdata construct_data(std::vector<PROdata> &variable_data, bool use_real_data, const PROconfig &config, const PROpeller &prop, const PROmodel &model, const std::vector<PROsyst> &variable_systs, const Eigen::VectorXf &fakedataparams, const PROpt &options) {
    PROdata data;
    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());
    if(use_real_data){
        PROconfig dataconfig;
        if(!options.data_xml.empty()){
            // Explicit --data flag takes precedence. Force the data config onto the same
            // fitting variable as the MC config — the two must agree bin-for-bin, and a
            // hand-written data XML will not carry a fit="true" of its own.
            dataconfig = PROconfig(options.data_xml, options.rateonly, (int)config.i_prime);
        } else {
            // Use embedded <data> section from the unified XML
            log<LOG_INFO>(L"%1% || Using embedded <data> section from XML for data config") % __func__;
            dataconfig = config.BuildDataConfig();
        }
        std::string dataBinName = options.analysis_tag+"_data.bin";
        for(size_t i = 0; i < dataconfig.m_num_channels; ++i) {
            size_t nsubch = dataconfig.m_num_subchannels[i];
            if(nsubch != 1) {
                log<LOG_ERROR>(L"%1% || Data xml required to have exactly 1 subchannel per channel. Found %2% for channel %3%")
                    % __func__ % nsubch % i;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }
            std::string &subchname = dataconfig.m_subchannel_names[i][0];
            if(subchname != "data") {
                log<LOG_ERROR>(L"%1% || Data subchannel required to be called \"data.\" Found name %2% for channel %3%")
                    % __func__ % subchname.c_str() % i;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }
        }
        if(!PROconfig::SameChannels(config, dataconfig)) {
            log<LOG_ERROR>(L"%1% || Require data and MC to have same channels. A difference was found, check messages above.")
                % __func__;
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }

        if((*options.process_command) || (!std::filesystem::exists(dataBinName))  ){
            log<LOG_INFO>(L"%1% || Processing Data Spectrum and saving to binary output also: %2%") % __func__ % dataBinName.c_str();

            //Process the CAF files to grab and fill spectrum directly
            std::vector<PROdata> alldata = CreatePROdata(dataconfig);
            PROdata::saveVector(dataconfig, alldata, dataBinName);
            data = alldata[config.i_prime];
            //data.save(dataconfig,dataBinName);
            
            for(size_t io = 0; io < dataconfig.m_num_variables; ++io)
                variable_data.push_back(alldata[io]);

            log<LOG_INFO>(L"%1% || Done processing Data from XML defined root files, and saving to binary output also: %2%") % __func__ % dataBinName.c_str();
        }else{
            log<LOG_INFO>(L"%1% || Loading Data from precalc binary input: %2%") % __func__ % dataBinName.c_str();
            //data.load(dataBinName);
            std::vector<PROdata> alldata;
            PROdata::loadVector(alldata, dataBinName);
            data = alldata[config.i_prime];
            //data.save(dataconfig,dataBinName);

            for(size_t io = 0; io < dataconfig.m_num_variables; ++io)
                variable_data.push_back(alldata[io]);

            log<LOG_INFO>(L"%1% || Done loading. Config hash (%2%) and binary loaded Data (%3%) hash are here. ") % __func__ %  dataconfig.hash % data.hash;
            if(dataconfig.hash!=data.hash){
                if(options.force){
                    log<LOG_WARNING>(L"%1% || WARNING config hash (%2%) and binary loaded data (%3%) hash not compatable! ") % __func__ %  dataconfig.hash % data.hash ;
                    log<LOG_WARNING>(L"%1% || WARNING But we are forcing ahead, be SUPER clear and happy you understand what your doing.  ") % __func__;
                }else{
                    log<LOG_ERROR>(L"%1% || ERROR config hash (%2%) and binary loaded data (%3%) hash not compatable! ") % __func__ %  dataconfig.hash % data.hash ;
                    return 1;
                }
            }
        }

    }//if no data, use injected or fake data;
    else{
        log<LOG_INFO>(L"%1% || Going to get fake data set up for each variable.") % __func__ ;
        for(size_t io = 0; io < config.m_num_variables; ++io) {
            // plot="false" only suppresses drawing; the fitting variable always needs real
            // fake-data, so i_prime bypasses the flag here and in the Asimov branch below
            // (matching the variable_systs build, which also special-cases i_prime).
            if (options.pseudo_experiment && io == config.i_prime) {
                // True FC-style pseudo-experiment for the i_prime variable.
                // Pattern lifted verbatim from src/PROfc.cxx::fc_worker's per-PE body:
                //   1. CV spectrum + Cholesky of the bin-bin covariance once.
                //   2. Throw spline pulls Gaussian, rejection-sampled within restrict bounds.
                //   3. Throw covariance bin shifts (Gaussian in standardised units).
                //   4. Build shifted spectrum, add L*throwC to the collapsed spec, Poisson-variate.
                std::normal_distribution<float> d;
                const size_t nphys   = model.nparams;
                const size_t nspline = variable_systs[io].GetNSplines();

                PROspec cv_for_L = FillSpectra(config, prop, variable_systs[io], model,
                                               fakedataparams, !options.eventbyevent, io);
                Eigen::MatrixXf L_chol = variable_systs[io].DecomposeFractionalCovariance(config, cv_for_L.Spec());

                Eigen::VectorXf throws = fakedataparams;
                // Shared truncated-Gaussian helper: samples each spline's actual prior
                // N(center, sigma) within its restrict bounds, OOB-safe (never spins
                // forever on unreachable bounds; clamps to the in-range value nearest
                // the prior center and warns).
                for (size_t i = 0; i < nspline; ++i) {
                    throws((int)(i + nphys)) = ThrowRestrictedSplinePull(variable_systs[io], i, PROseed::global_rng, d);
                }

                const int nbins_coll = config.m_num_variable_bins_total_collapsed[io];
                Eigen::VectorXf throwC(nbins_coll);
                for (int i = 0; i < nbins_coll; ++i) throwC(i) = d(PROseed::global_rng);

                PROspec shifted = FillSpectra(config, prop, variable_systs[io], model,
                                              throws, !options.eventbyevent, io);
                PROspec newSpec = PROspec::PoissonVariation(
                    PROspec(CollapseMatrix(config, shifted.Spec(), io) + L_chol * throwC,
                            CollapseMatrix(config, shifted.Error(), io)),
                    dseed(PROseed::global_rng));

                log<LOG_INFO>(L"%1% || Generated FC-style pseudo-experiment for i_prime variable %2% "
                              L"(splines thrown=%3%, cov bins thrown=%4%).")
                    % __func__ % io % (int)nspline % nbins_coll;

                // List the thrown spline pulls (in sigma) that produced this pseudo-experiment.
                std::string thrown_str;
                for (size_t i = 0; i < nspline; ++i) {
                    const std::string sn = i < variable_systs[io].spline_names.size()
                        ? variable_systs[io].spline_names[i] : ("spline#" + std::to_string(i));
                    thrown_str += sn + "=" + std::to_string(throws((int)(i + nphys)));
                    if (i + 1 < nspline) thrown_str += ", ";
                }
                log<LOG_INFO>(L"%1% || Pseudo-experiment thrown spline pulls (sigma): %2%")
                    % __func__ % thrown_str.c_str();

                variable_data.push_back(PROdata(newSpec.Spec(), newSpec.Error()));
                continue;
            }

            const bool fill_this_variable = config.m_channel_variable_plot_bool.at(io) || io == config.i_prime;
            PROspec data_spec = fill_this_variable ?  FillSpectra(config, prop, variable_systs[io], model, fakedataparams, !options.eventbyevent, io) : PROspec(config.m_num_variable_bins_total[io]) ;
            if(options.poisson_throw) data_spec = PROspec::PoissonVariation(data_spec, dseed(PROseed::global_rng));
            Eigen::VectorXf data_vec = CollapseMatrix(config, data_spec.Spec(), io);
            variable_data.push_back(PROdata(data_vec, data_vec.array().sqrt()));
        }
    }

    data = variable_data[config.i_prime];
    return data;
}

Eigen::VectorXf make_fakedata_params(Eigen::VectorXf &fake_data_osc_param_vector, const PROconfig &config, const PROpt &options, const PROmodel &model, const PROsyst &systs) {
    //loop over input fake data physics params and check/set
    for(const auto &[name, value]: options.fake_data_osc_params) {
        const auto it = std::find(model.param_names.begin(), model.param_names.end(), name);
        if(it == std::end(model.param_names)) {
            log<LOG_ERROR>(L"%1% || Unrecognized model parameter name %2%.\n"
                    L"Valid names for model %3% are %4%") %
                __func__% name.c_str()% config.m_model_tag.c_str()%
                model.param_names;
            exit(1);
        }
        int loc = std::distance(model.param_names.begin(), it);
        fake_data_osc_param_vector(loc) = model.is_log10[loc] ? std::log10(value) : value;
        log<LOG_INFO>(L"%1% Set fake data injected parameter %2% to value %3%, internally %4%") % __func__ % name.c_str() % value % fake_data_osc_param_vector(loc);
    }

    //Spline fake data injection studies
    Eigen::VectorXf fakedataparams = Eigen::VectorXf::Constant(model.nparams + systs.GetNSplines(), 0);
    for(size_t i = 0; i < model.nparams; ++i) fakedataparams(i) = fake_data_osc_param_vector(i);
    log<LOG_INFO>(L"%1% || model.default_val: %2%") % __func__ % model.default_val;
    log<LOG_INFO>(L"%1% || fake_data_osc_param_vector: %2%") % __func__ % fake_data_osc_param_vector;
    if(fakedataparams.size() >= 2) {
        log<LOG_INFO>(L"%1% || fakedataparams (physics portion): %2% %3%") % __func__ % fakedataparams(0) % fakedataparams(1);
    } else if(fakedataparams.size() == 1) {
        log<LOG_INFO>(L"%1% || fakedataparams (physics portion): %2%") % __func__ % fakedataparams(0);
    } else {
        log<LOG_INFO>(L"%1% || fakedataparams (physics portion): empty") % __func__;
    }
    for(const auto& [name, shift]: options.injected_systs) {
        log<LOG_INFO>(L"%1% || Injected syst: %2% shifted by %3%") % __func__ % name.c_str() % shift;

        auto it = std::find(systs.spline_names.begin(), systs.spline_names.end(), name);
        if(it == systs.spline_names.end()) {
            for(const auto &[xml_name, plot_name]: config.m_mcgen_variation_plotname_map) {
                if(name == plot_name) {
                    it = std::find(systs.spline_names.begin(), systs.spline_names.end(), xml_name);
                    break;
                }
            }
            if(it == systs.spline_names.end()) {
                log<LOG_ERROR>(L"%1% || Error: Unrecognized spline %2%. Ignoring this injected shift.") % __func__ % name.c_str();
                continue;
            }
        }
        int idx = std::distance(systs.spline_names.begin(), it);
        fakedataparams(idx+model.nparams) = shift;
    }
    return fakedataparams;
}

void make_param_vectors(Eigen::VectorXf &fakeDataParams, Eigen::VectorXf &CVParams, const PROconfig &config, const PROpt &options, const PROmodel &model, const PROsyst &systs, const Eigen::VectorXf &fake_data_osc_param_vector) {
    for(const auto& [name, shift]: options.cv_injected_systs) {
        log<LOG_INFO>(L"%1% || Injected syst: %2% shifted by %3%") % __func__ % name.c_str() % shift;

        auto it = std::find(systs.spline_names.begin(), systs.spline_names.end(), name);
        if(it == systs.spline_names.end()) {
            for(const auto &[xml_name, plot_name]: config.m_mcgen_variation_plotname_map) {
                if(name == plot_name) {
                    it = std::find(systs.spline_names.begin(), systs.spline_names.end(), xml_name);
                    break;
                }
            }
            if(it == systs.spline_names.end()) {
                log<LOG_ERROR>(L"%1% || Error: Unrecognized spline %2%. Ignoring this injected shift.") % __func__ % name.c_str();
                continue;
            }

        }
        int idx = std::distance(systs.spline_names.begin(), it);
        CVParams(idx+model.nparams) = shift;
    }

    for(long i = 0; i < fake_data_osc_param_vector.size(); ++i) {
        fakeDataParams(i) = fake_data_osc_param_vector(i);
        CVParams(i) = model.default_val(i); // Set default here, cv-injected below
    }
    //loop over input CV physics params and check/set
    for(const auto &[name, value]: options.cv_osc_params) {
        const auto it = std::find(model.param_names.begin(), model.param_names.end(), name);
        if(it == std::end(model.param_names)) {
            log<LOG_ERROR>(L"%1% || Unrecognized model parameter name %2%.\n"
                    L"Valid names for model %3% are %4%") %
                __func__% name.c_str()% config.m_model_tag.c_str()%
                model.param_names;
            exit(1);
        }
        int loc = std::distance(model.param_names.begin(), it);
        CVParams(loc) = model.is_log10[loc] ? std::log10(value) : value;
        log<LOG_INFO>(L"%1% Set CV injected parameter %2% to value %3%, internally %4%") % __func__ % name.c_str() % value % fake_data_osc_param_vector(loc);
    }
}

void include_or_exclude_systs(std::vector<PROsyst> &variable_systs, const PROconfig &config, const PROpt &options) {
    if(options.syst_list.size()) {

        std::vector<std::string> systs_to_include;
        for(const auto &s: options.syst_list) {
            bool istag = false;
            for(const auto &[syst, tags]: config.m_mcgen_variation_tags) {
                if(std::find(tags.begin(), tags.end(), s) != std::end(tags)) {
                    istag = true;
                    systs_to_include.push_back(syst);
                }
            }
            if(!istag) systs_to_include.push_back(s);
        }
        for(std::string &name: systs_to_include) {
            for(const auto &[xml_name, plot_name]: config.m_mcgen_variation_plotname_map) {
                if(name == plot_name) {
                    name = xml_name;
                }
            }
        }
        int io=0;
        for(PROsyst &syst: variable_systs){
            if(config.m_channel_variable_plot_bool.at(io)){
                syst = syst.subset(systs_to_include);
            }
            io++;
        }
    } else if(options.systs_excluded.size()) {

        std::vector<std::string> systs_to_exclude;
        for(const auto &s: options.systs_excluded) {
            log<LOG_INFO>(L"%1% || Excluding systematic %2% by command line argument.") % __func__ % s.c_str();
            bool istag = false;
            for(const auto &[syst, tags]: config.m_mcgen_variation_tags) {
                if(std::find(tags.begin(), tags.end(), s) != std::end(tags)) {
                    istag = true;
                    systs_to_exclude.push_back(syst);
                }
            }
            if(!istag) systs_to_exclude.push_back(s);
        }
        for(std::string &name: systs_to_exclude) {
            for(const auto &[xml_name, plot_name]: config.m_mcgen_variation_plotname_map) {
                if(name == plot_name) {
                    name = xml_name;
                }
            }
        }
        int io=0;
        for(PROsyst &syst: variable_systs){
            if(config.m_channel_variable_plot_bool.at(io)){
                syst = syst.excluding(systs_to_exclude);
            }
            io++;
        }
    }
}

void empty_bin_check(const PROconfig &config, const PROpt &options, const PROpeller &prop, const PROmodel &model, const PROsyst &systs, const PROdata data, bool use_real_data) {
    Eigen::VectorXf cv_check_params =
        Eigen::VectorXf::Zero(model.nparams + systs.GetNSplines());
    for(size_t i = 0; i < model.nparams; ++i)
        cv_check_params(i) = model.default_val(i);

    PROspec cv_check_spec = FillSpectra(config, prop, systs, model,
                                        cv_check_params, !options.eventbyevent, config.i_prime);
    Eigen::VectorXf collapsed_cv = CollapseMatrix(config, cv_check_spec.Spec(), config.i_prime);

    int n_zero = 0, n_tiny = 0;
    for(Eigen::Index b = 0; b < collapsed_cv.size(); ++b) {
        // Bins outside the fit region (PROjector / future bin-off masks) never enter
        // a chi2, so an empty CV there is not a problem.
        if(!config.IsBinActive(config.i_prime, (size_t)b)) continue;
        if(collapsed_cv(b) <= 0.0f) {
            log<LOG_ERROR>(L"%1% || Default-CV collapsed bin %2% has %3% expected events (<=0). Empty-bin would make CNP/stat covariance singular.") % __func__ % (long)b % collapsed_cv(b);
            ++n_zero;
        } else if(collapsed_cv(b) < 1.0f) {
            log<LOG_WARNING>(L"%1% || Default-CV collapsed bin %2% has %3% expected events (<1). Low-stat region; results in this bin may be unreliable.") % __func__ % (long)b % collapsed_cv(b);
            ++n_tiny;
        }
    }
    if(n_zero > 0) {
        if(!options.force) {
            log<LOG_ERROR>(L"%1% || %2% collapsed CV bins are empty. Aborting. Re-run with --force to override (NOT recommended).") % __func__ % n_zero;
            exit(1);
        }
        log<LOG_WARNING>(L"%1% || %2% collapsed CV bins are empty but --force was set; proceeding at user's risk.") % __func__ % n_zero;
    }
    if(n_tiny > 0)
        log<LOG_WARNING>(L"%1% || %2% collapsed CV bins have <1 expected event.") % __func__ % n_tiny;

    if(use_real_data) {
        int n_nan = 0, n_neg = 0;
        const Eigen::VectorXf &dvec = data.Spec();
        for(Eigen::Index b = 0; b < dvec.size(); ++b) {
            if(std::isnan(dvec(b)) || std::isinf(dvec(b))) {
                log<LOG_ERROR>(L"%1% || Data bin %2% is NaN/inf.") % __func__ % (long)b;
                ++n_nan;
            } else if(dvec(b) < 0.0f) {
                log<LOG_WARNING>(L"%1% || Data bin %2% is negative (%3%).") % __func__ % (long)b % dvec(b);
                ++n_neg;
            }
        }
        if(n_nan > 0 && !options.force) {
            log<LOG_ERROR>(L"%1% || %2% data bins are NaN/inf. Aborting. Re-run with --force to override.") % __func__ % n_nan;
            exit(1);
        }
        (void)n_neg; // Get rid of unused variable warning
    }
}

void set_global_bounds(Eigen::VectorXf &lb, Eigen::VectorXf &ub, std::vector<int> &fixed, const PROconfig &config, const PROpt &options, PROmodel &model, PROsyst &systs, const Eigen::VectorXf &CVParams) {
    log<LOG_INFO>(L"%1% || We are hoping to FIX : %2% ") % __func__ % options.fixed_params;
    for(size_t i = 0; i < model.nparams; ++i) {
                std::string name = model.param_names[i];
                std::string pname = model.pretty_param_names[i];
                if( options.systs_only || std::find(options.fixed_params.begin(), options.fixed_params.end(), pname) != options.fixed_params.end() || std::find(options.fixed_params.begin(), options.fixed_params.end(), name) != options.fixed_params.end()){
                    log<LOG_INFO>(L"%1% || We are FIXING physics parameter %2% (%3%) at value %4% ") % __func__ % i % name.c_str() % CVParams(i);  
                    model.lb(i) = CVParams(i);
                    model.ub(i) = CVParams(i);
                    fixed.at(i)=1;
                }
                lb(i) = model.lb(i);
                ub(i) = model.ub(i);
    }
    for(size_t i = model.nparams; i < model.nparams + systs.GetNSplines(); ++i) {
                std::string name = systs.spline_names[i-model.nparams];
                std::string pname =config.m_mcgen_variation_plotname_map.at(name); 

                if( std::find(options.fixed_params.begin(), options.fixed_params.end(), name) != options.fixed_params.end() || std::find(options.fixed_params.begin(), options.fixed_params.end(), pname) != options.fixed_params.end()){
                    log<LOG_INFO>(L"%1% || We are FIXING syst parameter %2% (%3%) at value %4% ") % __func__ % i % name.c_str() % CVParams(i);  
                    systs.spline_hi[i-model.nparams] = CVParams(i);
                    systs.spline_lo[i-model.nparams] = CVParams(i);
                    fixed.at(i)=1;
                }
                lb(i) = systs.spline_lo[i-model.nparams];
                ub(i) = systs.spline_hi[i-model.nparams];

    }
    if( (options.fixed_params.size()!=std::accumulate(fixed.begin(), fixed.end(), (size_t)0)) && !options.systs_only ){
            log<LOG_ERROR>(L"%1% || ERROR. The fixed parameters you passed, check they exist? the number of fixed params is not the same as input params.") % __func__;
            log<LOG_ERROR>(L"%1% || ERROR. fixed_params %2% ") % __func__ % options.fixed_params;
            log<LOG_ERROR>(L"%1% || ERROR. global_fixed %2% : sum %3% ") % __func__ % fixed % ((int)std::accumulate(fixed.begin(), fixed.end(), 0)) ;
            exit(EXIT_FAILURE);
    }
}

void print_global_fit_results(float global_fit_chi2, const Eigen::VectorXf &global_fit_result, const PROconfig &config, const PROpt &options, const PROmetric &metric) {
    std::ofstream global_fit_out;
    if(global_fit_result.size() > 0) {
        global_fit_out.open(options.final_output_tag+"_global_fit.txt");
        log<LOG_INFO>(L"%1% || ################################################") % __func__;
        log<LOG_INFO>(L"%1% || ########### Global Best Fit Results ############") % __func__;
        log<LOG_INFO>(L"%1% || ################################################") % __func__;
        log<LOG_INFO>(L"%1% || Global Best Fit chi^2: %2%") %__func__ % global_fit_chi2;
        log<LOG_INFO>(L"%1% || at paramters: ") % __func__;

        global_fit_out << "Global best fit:\n";

        bool use_phys = (size_t)global_fit_result.size() == metric.GetModel().nparams + metric.GetSysts().GetNSplines();
        for(long i = 0; i < global_fit_result.size(); i++){

            if(use_phys && i < (long)metric.GetModel().nparams){
                log<LOG_INFO>(L"%1% || %2%  : %3% (log) %4% (nonlog) ") % __func__ % metric.GetModel().pretty_param_names[i].c_str() % global_fit_result(i) % pow(10,global_fit_result(i));
                global_fit_out << metric.GetModel().param_names[i] << " : " << global_fit_result(i) << "\n";
            }else{
                long idx = use_phys ? i - metric.GetModel().nparams : i;
                log<LOG_INFO>(L"%1% || %2%  :  %3% ") % __func__ % config.m_mcgen_variation_plotname_map.at(metric.GetSysts().spline_names[idx]).c_str() % global_fit_result(i);

                global_fit_out <<  config.m_mcgen_variation_plotname_map.at(metric.GetSysts().spline_names[idx])
                    << " : " << global_fit_result(i) << "\n";
            }
        }
        log<LOG_INFO>(L"%1% || ################################################") % __func__;
    }
    if(global_fit_out.is_open()) global_fit_out.close();
}
