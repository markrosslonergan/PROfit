#include "PROconfig.h"
#include "PROdata.h"
#include "PROlog.h"
#include "PROmetric.h"
#include "PROspec.h"
#include "PROsyst.h"
#include "PROcreate.h"
#include "PROpeller.h"
#include "PROchi.h"
#include "PROCNP.h"
#include "PROpoisson.h"
#include "PROcess.h"
#include "PROsurf.h"
#include "PROfc.h"
#include "PROfitter.h"
#include "PROmodel.h"
#include "PROMCMC.h"
#include "PROtocall.h"
#include "PROseed.h"
#include "PROversion.h"
#include "PROplot.h"

#include "CLI11.h"
#include "LBFGSB.h"

#include <Eigen/Eigen>

#include <Eigen/src/Core/Matrix.h>
#include <LBFGSpp/Param.h>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "TMath.h"

using namespace PROfit;

log_level_t GLOBAL_LEVEL = LOG_INFO;
log_level_t FILE_LEVEL = LOG_INFO;  

std::wostream *OSTREAM = &wcout;

std::wofstream LOG_FILE_STREAM;
bool LOGGING_TO_FILE = false;


int main(int argc, char* argv[])
{
    gStyle->SetOptStat(0);
    CLI::App app{"PROfit: a PROfessional, PROductive fitting and oscillation framework. Together let's minimize PROfit!"}; 

    // Define options
    std::string xmlname = "NULL.xml"; 
    std::string data_xml = "";
    std::string analysis_tag = "PROfit";
    std::string output_tag = "v1";
    std::string chi2 = "PROchi";
    bool show_fit_help = false;
    bool eventbyevent=false;
    bool shapeonly = false;
    bool rateonly = false;
    bool force = false;
    bool noxrootd = false;
    bool poisson_throw = false;
    bool progress_bar = false;
    std::vector<std::string> scale_arg;
    std::map<std::string, float> scale_map;
    size_t nthread = 1;
    std::map<std::string, float> scan_fit_options;
    std::map<std::string, float> global_fit_options;
    size_t maxevents;
    int global_seed = -1;
    std::string log_file = "";
    std::string fit_preset = "good";
    static const std::unordered_set<std::string> allowed_preset = {"good","fast","overkill"};
    bool with_splines = false, binwidth_scale = false, area_normalized = false;
    std::map<std::string, float> fake_data_osc_params;
    std::map<std::string, float> cv_osc_params;
    std::map<std::string, float> injected_systs;
    std::vector<std::string> syst_list, systs_excluded;
    bool MCMC_prefit_errors = false;
    bool systs_only_profile = false;

    float xlo, xhi, ylo, yhi;
    std::array<float, 2> xlims, ylims;
    std::vector<int> grid_size;
    bool statonly = false, logx=true, logy=true;
    std::string xlabel, ylabel;
    std::string xvar = "sinsq2thmm", yvar = "dmsq";
    bool run_brazil = false;
    bool statonly_brazil = false;
    bool single_brazil = false;
    bool only_brazil = false;
    std::vector<std::string> brazil_throws;
    std::vector<float> procurve_points;

    std::string reweights_file;
    std::vector<std::string> mockreweights;
    std::vector<TH2D*> weighthists;

    std::map<std::string, float> bound_list;
    PlotBounds pbounds; 
    size_t nuniv;


    //Global Arguments for all PROfit enables subcommands.
    app.add_option("-x,--xml", xmlname, "Input PROfit XML configuration file.")->required();
    app.add_option("-t,--tag", analysis_tag, "Analysis Tag used for output identification.")->default_str("PROfit");
    app.add_option("-v,--verbosity", GLOBAL_LEVEL, "Verbosity Level [1-4]->[Error,Warning,Info,Debug].")->default_val(GLOBAL_LEVEL);
    app.add_option("-l,--log", log_file, "File to save log to. Warning: Will overwrite this file.");
    app.add_option("-w,--file-verbosity", FILE_LEVEL, "File (log) Verbosity Level [1-4]->[Error,Warning,Info,Debug].")->default_val(static_cast<log_level_t>(-1));
    app.add_flag("-b,--progress", progress_bar, "Use a progress bar when applicable.");
    app.add_option("-o,--output",output_tag,"Additional output filename quantifier")->default_str("v1");
    app.add_option("-n, --nthread",   nthread, "Number of threads to parallelize over.")->default_val(1);
    app.add_option("-m,--max", maxevents, "Max number of events to run over.");
    app.add_option("-c, --chi2", chi2, "Which chi2 function to use. Options are PROchi or PROCNP")->default_str("PROchi");
    app.add_option("-d, --data", data_xml, "Load from a seperate data xml/data file instead of signal injection. Only used with plot subcommand.")->default_str("");
    app.add_option("-i, --inject", fake_data_osc_params, "Physics parameters to inject as fake-data true signal. Example: dmsq 3 sinsq2thmm 0.25")->expected(-1);
    app.add_option("--inject-cv", cv_osc_params, "Physics parameters to inject as CV. Example: dmsq 3 sinsq2thmm 0.25")->expected(-1);
    app.add_option("-s, --seed", global_seed, "A global seed for PROseed rng. Default to -1 for hardware rng seed.")->default_val(-1);
    app.add_option("-p,--preset", fit_preset, "Preset fitting params. Available `fast`, `good` and `overkill` .");
    app.add_option("--fit-options", global_fit_options, "Parameters for single, detailed global best fit LBFGSB. See PROfitter.h or run --fit-help for available settings.");
    app.add_option("--scan-fit-options", scan_fit_options, "Parameters for simpier, multiple best fits in PROfile/surface LBFGSB.");
    app.add_flag("--fit-help", show_fit_help, "Show detailed help for all fitting parameters (L-BFGS-B, PSO, MCMC, etc.)");

    app.add_option("--inject-systs", injected_systs, "Systematic shifts to inject. Map of name and shift value in sigmas. Only spline systs are supported right now.");
    app.add_option("--syst-list", syst_list, "Override list of systematics to use (note: all systs must be in the xml).");
    app.add_option("--exclude-systs", systs_excluded, "List of systematics to exclude.")->excludes("--syst-list"); 

    app.add_flag("--poisson-throw", poisson_throw, "Do a Poisson stats throw of fake data.");
    app.add_flag("--scale-by-width", binwidth_scale, "Scale histgrams by 1/(bin width).");
    app.add_option("--scale", scale_arg, "Scale detector POT by a given value.");
    app.add_option("--plot-bounds", bound_list, "Plot bounds, set by  string float pairs. Available strings are ymax,ratmin,ratmax."); 

    //app.add_option("-f, --rwfile", reweights_file, "File containing histograms for reweighting");//deprociated, add back in later
    //app.add_option("-r, --mockrw",   mockreweights, "Vector of reweights to use for mock data");
    app.add_flag("--event-by-event", eventbyevent, "Do you want to weight event-by-event?");
    app.add_flag("--statonly", statonly, "Run a stats only surface instead of fitting systematics");
    app.add_flag("--force",force,"Force loading binary data even if hash is incorrect (Be Careful!)");
    app.add_flag("--no-xrootd",noxrootd,"Do not use XRootD, which is enabled by default");
    auto* shape_flag = app.add_flag("--shapeonly", shapeonly, "Run a shape only analysis");
    auto* rate_flag = app.add_flag("--rateonly", rateonly, "Run a rate only analysis");
    shape_flag->excludes(rate_flag);   

    //PROcess, into binary data [Do this once first!]
    CLI::App *process_command = app.add_subcommand("process", "PROcess the MC and systematics in root files into binary data for future rapid loading.");

    //PROsurf, make a 2D surface scan of physics parameters
    CLI::App *surface_command = app.add_subcommand("surface", "Make a 2D surface scan of two physics parameters, profiling over all others.");
    surface_command->add_option("-g, --grid", grid_size, "Set grid size. If one dimension passed, grid assumed to be square, else rectangular")->expected(0, 2)->default_val(40);
    surface_command->add_option("--xvar", xvar, "Name of variable to put on x-axis")->default_val("sinsq2thmm");
    surface_command->add_option("--yvar", yvar, "Name of variable to put on x-axis")->default_val("dmsq");
    CLI::Option *xlim_opt = surface_command->add_option("--xlims", xlims, "Limits for x-axis");
    CLI::Option *ylim_opt = surface_command->add_option("--ylims", ylims, "Limits for y-axis");
    surface_command->add_option("--xlo", xlo, "Lower limit for x-axis")->excludes(xlim_opt)->default_val(1e-4);
    surface_command->add_option("--xhi", xhi, "Upper limit for x-axis")->excludes(xlim_opt)->default_val(1);
    surface_command->add_option("--ylo", ylo, "Lower limit for y-axis")->excludes(ylim_opt)->default_val(1e-2);
    surface_command->add_option("--yhi", yhi, "Upper limit for y-axis")->excludes(ylim_opt)->default_val(1e2);
    surface_command->add_option("--xlabel", xlabel, "X-axis label");
    surface_command->add_option("--ylabel", ylabel, "Y-axis label");
    surface_command->add_flag("--logx,!--linx", logx, "Specify if x-axis is logarithmic or linear (default log)");
    surface_command->add_flag("--logy,!--liny", logy, "Specify if y-axis is logarithmic or linear (default log)");
    surface_command->add_flag("--brazil-band", run_brazil, "Run 1000 throws of stats+systs and draw 1 sigma and 2 sigma Brazil bands");
    surface_command->add_flag("--stat-throws", statonly_brazil, "Only do stat throws for the Brazil band")->needs("--brazil-band");
    surface_command->add_flag("--single-throw", single_brazil, "Only run a single iteration of the Brazil band")->needs("--brazil-band");
    surface_command->add_flag("--only-throw", only_brazil, "Only run Brazil band throws and not the nominal surface")->needs("--brazil-band");
    surface_command->add_option("--from-many", brazil_throws, "Make Brazil band from many provided throws")->needs("--brazil-band");
    surface_command->add_option("--curve-mode", procurve_points , "Make a PROcurve plot from param A to param B.");

    //PROfile, make N profile'd chi^2 for each physics and nuisence parameters
    CLI::App *profile_command = app.add_subcommand("profile", "Make a 1D profiled chi2 for each physics and nuisence parameter.");
    profile_command->add_flag("--syst-only", systs_only_profile, "Profile over nuisance parameters only");
    profile_command->add_flag("--mcmc-prefit", MCMC_prefit_errors, "Use MCMC to sample the systematic priors for the pre-fit error band.");

    //PROplot, plot things
    CLI::App *proplot_command = app.add_subcommand("plot", "Make plots of CV, or injected point with error bars and covariance.");
    proplot_command->add_flag("--with-splines", with_splines, "Include graphs of splines in output.");
    proplot_command->add_flag("--area-norm", area_normalized, "Make area normalized histograms.");

    //PROfc, Feldmand-Cousins
    CLI::App *profc_command = app.add_subcommand("fc", "Run Feldman-Cousins for this injected signal");
    profc_command->add_option("-u,--universes", nuniv, "Number of Feldman Cousins universes to throw")->default_val(1000);

    //PROglobal
    CLI::App *proglobal_command = app.add_subcommand("global", "Just do a single global fit.");

    //PROtest, test things
    CLI::App *protest_command = app.add_subcommand("protest", "Testing ground for rapid quick tests.");

    app.set_config("--config");
    surface_command->configurable(true);
    process_command->configurable(true);
    profile_command->configurable(true);
    protest_command->configurable(true);
    proglobal_command->configurable(true);
    profc_command->configurable(true);
    proplot_command->configurable(true);

    //Parse inputs. 
    CLI11_PARSE(app, argc, argv);

    if(show_fit_help) {
        PROfit::PROfitterConfig::PrintHelp();
        return 0;
    }

    if(log_file != "") {

        if(FILE_LEVEL == static_cast<log_level_t>(-1)) {
            FILE_LEVEL = GLOBAL_LEVEL;
        }

        log_impl::EnableFileLogging(log_file,FILE_LEVEL);
    }

    pbounds.Load(bound_list);

    log<LOG_WARNING>(L" %1% ") % getIcon().c_str()  ;
    std::string final_output_tag =analysis_tag +"_"+output_tag;

    log<LOG_WARNING>(L"%1% || ##################################################################") % __func__  ;
    log<LOG_WARNING>(L"%1% || ####################### PROfit version v%2% ######################") % __func__ % PROJECT_VERSION_STR ;
    log<LOG_WARNING>(L"%1% || ##################################################################") % __func__  ;
    log<LOG_WARNING>(L"%1% || PROfit commandline input arguments. xml: %2%, tag: %3%, output %4%, nthread: %5% ") % __func__ % xmlname.c_str() % analysis_tag.c_str() % output_tag.c_str() % nthread ;

    //Initilize configuration from the XML;
    PROconfig config(xmlname, rateonly);


    //Inititilize PROpeller to keep MC
    PROpeller prop;

    //Initilize objects for systematics storage
    std::vector<std::vector<SystStruct>> systsstructs;

    //input/output logic
    std::string propBinName = analysis_tag+"_prop.bin";
    std::string systBinName = analysis_tag+"_syst.bin";

    if((*process_command) || (!std::filesystem::exists(systBinName) || !std::filesystem::exists(propBinName))  ){
        log<LOG_INFO>(L"%1% || Processing PROpeller and PROsysts from XML defined root files, and saving to binary output also: %2%") % __func__ % propBinName.c_str();
        //Process the CAF files to grab and fill all SystStructs and PROpeller
        PROcess_CAFAna(config, systsstructs, prop,noxrootd);
        prop.save(propBinName);    
        saveSystStructVector(systsstructs,systBinName);
        log<LOG_INFO>(L"%1% || Done processing PROpeller and PROsysts from XML defined root files, and saving to binary output also: %2%") % __func__ % propBinName.c_str();

    }else{
        log<LOG_INFO>(L"%1% || Loading PROpeller and PROsysts from precalc binary input: %2%") % __func__ % propBinName.c_str();
        prop.load(propBinName);
        loadSystStructVector(systsstructs, systBinName);

        //is hash right for PROpeller first?
        log<LOG_INFO>(L"%1% || Done loading. Config hash (%2%) and binary loaded PROpeller (%3%) are here. ") % __func__ %  config.hash % prop.hash ;
        if(config.hash!=prop.hash){
            if(force){
                log<LOG_WARNING>(L"%1% || WARNING config hash (%2%) and binary loaded PROpeller (%3%)  not compatable! ") % __func__ %  config.hash % prop.hash ;
                log<LOG_WARNING>(L"%1% || WARNING But we are forcing ahead, be SUPER clear and happy you understand what your doing.  ") % __func__;
            }else{
                log<LOG_ERROR>(L"%1% || ERROR config hash (%2%) and binary loaded PROpeller (%3%)  not compatable! ") % __func__ %  config.hash % prop.hash ;
                return 1;
            }
        }
        //Now check syststructs, if there is any!
        if(systsstructs.front().size()>0){
            log<LOG_INFO>(L"%1% || Done loading. Config hash (%2%) and binary loaded PROsyst hash(%3%) are here. ") % __func__ %  config.hash % systsstructs[0][0].hash;
            if( config.hash!=systsstructs.front().front().hash){
                if(force){
                    log<LOG_WARNING>(L"%1% || WARNING config hash (%2%) and binary loaded PROsyst hash(%3%) not compatable! ") % __func__ %  config.hash %  systsstructs.front().front().hash;
                    log<LOG_WARNING>(L"%1% || WARNING But we are forcing ahead, be SUPER clear and happy you understand what your doing.  ") % __func__;
                }else{
                    log<LOG_ERROR>(L"%1% || ERROR config hash (%2%) and binary loaded PROsyst hash(%3%) not compatable! ") % __func__ %  config.hash %  systsstructs.front().front().hash;
                    return 1;
                }
            }
        }

    }

    //Scale events by some percentage of total detector POT
    if(scale_arg.size()) {
        if (scale_arg.size() % 2 != 0) {
            log<LOG_ERROR>(L"%1% || Expected pairs of detector and scaling values (e.g., ICARUS 0.5)") % __func__;
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < scale_arg.size(); i += 2) {
            scale_map[scale_arg[i]] = std::stof(scale_arg[i + 1]);
        }
        prop.scale(config, scale_map);
    }

    //Build a PROsyst to sort and analyze all systematics
    //PROsyst systs(prop, config, systsstructs.front(), shapeonly);
    std::vector<PROsyst> variable_systs;
    for(size_t i = 0; i < config.m_num_variables; ++i){

        if(config.m_channel_variable_plot_bool.at(i)){ 
            variable_systs.emplace_back(prop, config, systsstructs.at(i), shapeonly, i);
        }else{
            variable_systs.emplace_back();
        }
        //variable_systs.back().PrintSplines();
        //return 0;
    }



    std::unique_ptr<PROmodel> model = get_model_from_string(config, prop);
    std::unique_ptr<PROmodel> null_model = std::make_unique<NullModel>(prop);

    Eigen::VectorXf fake_data_osc_param_vector = model->default_val;
    for(const auto &[name, value]: fake_data_osc_params) {
        const auto it = std::find(model->param_names.begin(), model->param_names.end(), name);
        if(it == std::end(model->param_names)) {
            log<LOG_ERROR>(L"%1% || Unrecognized model parameter name %2%.\n"
                    L"Valid names for model %3% are %4%") %
                __func__% name.c_str()% config.m_model_tag.c_str()%
                model->param_names;
            return 1;
        }
        int loc = std::distance(model->param_names.begin(), it);
        fake_data_osc_param_vector(loc) = std::log10(value);
    }

    Eigen::VectorXf cv_osc_param_vector = model->default_val;
    for(const auto &[name, value]: cv_osc_params) {
        const auto it = std::find(model->param_names.begin(), model->param_names.end(), name);
        if(it == std::end(model->param_names)) {
            log<LOG_ERROR>(L"%1% || Unrecognized model parameter name %2%.\n"
                    L"Valid names for model %3% are %4%") %
                __func__% name.c_str()% config.m_model_tag.c_str()%
                model->param_names;
            return 1;
        }
        int loc = std::distance(model->param_names.begin(), it);
        cv_osc_param_vector(loc) = std::log10(value);
    }


    //Seed time
    PROseed myseed(nthread, global_seed);
    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());


    //Spline injection studies
    Eigen::VectorXf allparams = Eigen::VectorXf::Constant(model->nparams + variable_systs[config.i_prime].GetNSplines(), 0);
    Eigen::VectorXf systparams = Eigen::VectorXf::Constant(variable_systs[config.i_prime].GetNSplines(), 0);
    for(size_t i = 0; i < model->nparams; ++i) allparams(i) = fake_data_osc_param_vector(i);
    for(const auto& [name, shift]: injected_systs) {
        log<LOG_INFO>(L"%1% || Injected syst: %2% shifted by %3%") % __func__ % name.c_str() % shift;

        auto it = std::find(variable_systs[config.i_prime].spline_names.begin(), variable_systs[config.i_prime].spline_names.end(), name);
        if(it == variable_systs[config.i_prime].spline_names.end()) {
            for(const auto &[xml_name, plot_name]: config.m_mcgen_variation_plotname_map) {
                if(name == plot_name) {
                    it = std::find(variable_systs[config.i_prime].spline_names.begin(), variable_systs[config.i_prime].spline_names.end(), xml_name);
                    break;
                }
            }
            if(it == variable_systs[config.i_prime].spline_names.end()) {
                log<LOG_ERROR>(L"%1% || Error: Unrecognized spline %2%. Ignoring this injected shift.") % __func__ % name.c_str();
                continue;
            }

        }
        int idx = std::distance(variable_systs[config.i_prime].spline_names.begin(), it);
        allparams(idx+model->nparams) = shift;
        systparams(idx) = shift;
    }

    //Some logic for EITHER injecting fake/mock data of oscillated signal/syst shifts OR using real data
    //Main data for the i_prime fitting.
    PROdata data;

    //We will load all other variables too, but many are truth level so data won't be as common.
    std::vector<PROdata> variable_data;
    if(!data_xml.empty()){
        PROconfig dataconfig(data_xml);
        std::string dataBinName = analysis_tag+"_data.bin";
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

        if((*process_command) || (!std::filesystem::exists(dataBinName))  ){
            log<LOG_INFO>(L"%1% || Processing Data Spectrum and saving to binary output also: %2%") % __func__ % dataBinName.c_str();

            //Process the CAF files to grab and fill spectrum directly
            std::vector<PROdata> alldata = CreatePROdata(dataconfig);
            PROdata::saveVector(dataconfig, alldata, dataBinName);
            data = alldata[0];
            //data.save(dataconfig,dataBinName);
            for(size_t io = 0; io < dataconfig.m_num_variables; ++io)
                variable_data.push_back(alldata[io+1]);

            log<LOG_INFO>(L"%1% || Done processing Data from XML defined root files, and saving to binary output also: %2%") % __func__ % dataBinName.c_str();
        }else{
            log<LOG_INFO>(L"%1% || Loading Data from precalc binary input: %2%") % __func__ % dataBinName.c_str();
            //data.load(dataBinName);
            std::vector<PROdata> alldata;
            PROdata::loadVector(alldata, dataBinName);
            data = alldata[0];
            //data.save(dataconfig,dataBinName);
            for(size_t io = 0; io < dataconfig.m_num_variables; ++io)
                variable_data.push_back(alldata[io+1]);

            log<LOG_INFO>(L"%1% || Done loading. Config hash (%2%) and binary loaded Data (%3%) hash are here. ") % __func__ %  dataconfig.hash % data.hash;
            if(dataconfig.hash!=data.hash){
                if(force){
                    log<LOG_WARNING>(L"%1% || WARNING config hash (%2%) and binary loaded data (%3%) hash not compatable! ") % __func__ %  dataconfig.hash % data.hash ;
                    log<LOG_WARNING>(L"%1% || WARNING But we are forcing ahead, be SUPER clear and happy you understand what your doing.  ") % __func__;
                }else{
                    log<LOG_ERROR>(L"%1% || ERROR config hash (%2%) and binary loaded data (%3%) hash not compatable! ") % __func__ %  dataconfig.hash % data.hash ;
                    return 1;
                }
            }
        }

        if(*profile_command || *surface_command || *protest_command){
            log<LOG_ERROR>(L"%1% || ERROR --data can only be used with plot subcommand! ") % __func__  ;
            return 1;
        }


    }//if no data, use injected or fake data;
    else{
        log<LOG_INFO>(L"%1% || Going to get fake data set up for each variable.") % __func__ ;
        for(size_t io = 0; io < config.m_num_variables; ++io) {
            PROspec data_spec = config.m_channel_variable_plot_bool.at(io) ?  FillSpectra(config, prop, variable_systs[io], *model, allparams, !eventbyevent, io) : PROspec(config.m_num_variable_bins_total[io]) ;
            if(poisson_throw) data_spec = PROspec::PoissonVariation(data_spec, dseed(myseed.global_rng));
            Eigen::VectorXf data_vec = CollapseMatrix(config, data_spec.Spec(), io);
            variable_data.push_back(PROdata(data_vec, data_vec.array().sqrt()));
        }
    }

    data = variable_data[config.i_prime];

    // Leave this after creating fake data so we can make fake data using systs that aren't
    // included in the fit.

    //for(auto & systs : variable_systs){
    {
        if(syst_list.size()) {

            std::vector<std::string> systs_to_include;
            for(const auto &s: syst_list) {
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
            //systs = systs.subset(systs_to_include);
            int io=0;
            for(PROsyst &syst: variable_systs){
                if(config.m_channel_variable_plot_bool.at(io)){
                    syst = syst.subset(systs_to_include);
                }
                io++;
            }
        } else if(systs_excluded.size()) {

            std::vector<std::string> systs_to_exclude;
            for(const auto &s: systs_excluded) {
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
            //systs = systs.excluding(systs_to_exclude);
            int io=0;
            for(PROsyst &syst: variable_systs){
                if(config.m_channel_variable_plot_bool.at(io)){
                    syst = syst.excluding(systs_to_exclude);
                }
                io++;
            }
        }
    }

    //Pysics parameter input
    Eigen::VectorXf pparams = Eigen::VectorXf::Constant(model->nparams + variable_systs[config.i_prime].GetNSplines(), 0);
    Eigen::VectorXf CVpparams = Eigen::VectorXf::Constant(model->nparams + variable_systs[config.i_prime].GetNSplines(), 0);
    for(long i = 0; i < fake_data_osc_param_vector.size(); ++i) {
        pparams(i) = fake_data_osc_param_vector(i);
        CVpparams(i) = model->default_val(i);
    }
    for(long i = 0; i < cv_osc_param_vector.size(); ++i) {
        CVpparams(i) = cv_osc_param_vector(i);
    }


    log<LOG_INFO>(L"%1% || Starting from fit preset :  %2%.")% __func__ % fit_preset.c_str();
    if (allowed_preset.find(fit_preset) == allowed_preset.end()) {
        log<LOG_ERROR>(L"%1% || ERROR allowed fit_presets are good, fast or overkill. You entred : %2%.")% __func__ % fit_preset.c_str();
        return 1;
    }
    //Some global minimizer params
    // This runs for the single best gobal fit
    PROfitterConfig fitConfig(global_fit_options, fit_preset, false);



    //Some Scan minimizer params.
    // This runs lots during PROfile and surface. 
    PROfitterConfig scanFitConfig(scan_fit_options, "fast", true);



    //Metric Time
    //Metrics are for i_prime only for now
    PROmetric *metric, *null_metric;
    if(chi2 == "PROchi") {
        metric = new PROchi("", config, prop, &(variable_systs[config.i_prime]), *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2,shapeonly);
        null_metric = new PROchi("", config, prop, &(variable_systs[config.i_prime]), *null_model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2,shapeonly);
    } else if(chi2 == "PROCNP") {
        metric = new PROCNP("", config, prop, &(variable_systs[config.i_prime]), *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2,shapeonly);
        null_metric = new PROCNP("", config, prop, &(variable_systs[config.i_prime]), *null_model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2,shapeonly);
    } else if(chi2 == "Poisson") {
        metric = new PROpoisson("", config, prop, &(variable_systs[config.i_prime]), *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2,shapeonly);
        null_metric = new PROpoisson("", config, prop, &(variable_systs[config.i_prime]), *null_model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2,shapeonly);
    } else {
        log<LOG_ERROR>(L"%1% || Unrecognized chi2 function %2%") % __func__ % chi2.c_str();
        abort();
    }

    //Covariance colors, move this eslewher
    const Int_t NCont = 255;
    const Int_t NRGBs = 9;  // Reduced control points for smoother white stretch
    Double_t stops[NRGBs] = {
        0.0,    // -1.0 (dark blue)
        0.075,    // -0.6 (transition to white)
        0.3,    // -0.2 (mostly white)
        0.4,   // -0.04 (almost pure white)
        0.5,    //  0.0 (pure white)
        0.6,   // +0.04 (almost pure white)
        0.7,    // +0.2 (transition to red)
        0.925,    // +0.6 (strong red)
        1.0     // +1.0 (dark red)
    };

    Double_t red[NRGBs]   = {0.00,0.259,0.824, 0.949, 1.0, 0.988, 0.980, 0.918,0.839};
    Double_t green[NRGBs] = {0.341,0.404,0.890, 0.961, 1.0, 0.933, 0.824, 0.263,0.125};
    Double_t blue[NRGBs]  = {0.906,0.824,0.989, 0.980, 1.0, 0.929, 0.812, 0.208,0.024};

    TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    gStyle->SetNumberContours(NCont);


    // Need a second one for case where we do syst_only profile and surface in same command
    Eigen::VectorXf global_fit_result, global_fit_result_surf;
    float global_fit_chi2 = -1, global_fit_chi2_surf = -1;

    //***********************************************************************
    //***********************************************************************
    //******************** PROfile PROfile PROfile **************************
    //***********************************************************************
    //***********************************************************************

    if(*profile_command){

        PROmetric *metric_to_use = systs_only_profile ? null_metric : metric;
        size_t nparams = metric_to_use->nParams();
        size_t nphys = metric_to_use->GetModel().nparams;
        PROfitter fitter(metric_to_use->UpperBound(), metric_to_use->LowerBound(), fitConfig);
        metric_to_use->setBounds(metric_to_use->UpperBound(), metric_to_use->LowerBound());

        log<LOG_INFO>(L"%1% || ########### Starting Global Best Fit Minimizing ############") % __func__;


        std::vector<std::pair<int, std::string>> global_PB_configs;
        global_PB_configs.push_back({fitConfig.n_latin_points, "(1) LatinHyperCube"});
        global_PB_configs.push_back({fitConfig.n_swarm_iterations, "(2) ParticleSwarm"});
        global_PB_configs.push_back({fitConfig.n_localfit, "(3) BestLBFGSB"});
        global_PB_configs.push_back({fitConfig.harmonic_num_test_points, "(4) HarmonicScan"});
        global_PB_configs.push_back({100, "(5) HarmonicLBFGSB"});
        MultiPROgressBar global_progress(global_PB_configs);

        if(progress_bar){
            global_progress.initialize_display();
            global_progress.start_display_thread(); 
            fitter.setProgressBar(&global_progress);
        }

        float best_chi2 = fitter.Fit(*metric_to_use,CVpparams); 
        Eigen::VectorXf best_fit = fitter.best_fit;
        Eigen::MatrixXf post_covar = fitter.Covariance();
        if(!systs_only_profile) fitter.calcFreqSeedPoints(*metric_to_use);

        for(size_t i=0; i< fitter.freq_seed_points.size(); i++){
            float chi_freq = fitter.freq_seed_values.at(i);
            if( chi_freq< best_chi2){
                log<LOG_INFO>(L"%1% || One of the harmonics of first pass best fit, is a lower chi :  %2% ") % __func__ % chi_freq;
                log<LOG_INFO>(L"%1% || -- at params:  %2% ") % __func__ % fitter.freq_seed_points.at(i);
                best_chi2 = chi_freq;
                best_fit = fitter.freq_seed_points.at(i);
            }
        }
        if(progress_bar)global_progress.finish_all();
        global_fit_chi2 = best_chi2;
        global_fit_result = best_fit;

        if (fitter.exception_string_map.empty()) {
            log<LOG_INFO>(L"%1% || No exceptions were caught from LBFGSB [ --INFO-- ]") % __func__;
        } else {
            log<LOG_INFO>(L"%1% || Some exceptions were caught in LBFGSB [ --INFO-- ]") % __func__;
            for (const auto &[msg, count] : fitter.exception_string_map) {
                log<LOG_INFO>(L"%1% ||  -- Exception \"%2%\" occurred %3% time(s)") % __func__ % msg.c_str() % count;
            }
        }


        log<LOG_INFO>(L"%1% || ################################################") % __func__;
        log<LOG_INFO>(L"%1% || ########### Global Best Fit Results ############") % __func__;
        log<LOG_INFO>(L"%1% || ################################################") % __func__;
        log<LOG_INFO>(L"%1% || Global Best Fit chi^2: %2%") %__func__ % best_chi2;
        log<LOG_INFO>(L"%1% || at paramters: ") % __func__;

        for(size_t i = 0; i< nparams; i++){

            if(i<nphys){
                log<LOG_INFO>(L"%1% || %2%  : %3% (log) %4% (nonlog) ") % __func__ % metric_to_use->GetModel().pretty_param_names[i].c_str() % best_fit(i) % pow(10,best_fit(i));
            }else{
                log<LOG_INFO>(L"%1% || %2%  :  %3% ") % __func__ % metric_to_use->GetSysts().spline_names[i-nphys].c_str() % best_fit(i) ;
            }
        }
        log<LOG_INFO>(L"%1% || ################################################") % __func__;

        // TODO: Not sure I understand this covariance matrix
        log<LOG_INFO>(L"%1% || Starting a metropolis hastings chain to estimate the covariace matrix aroud the above best fit. Run and Burn is (%2%,%3%);") % __func__%fitConfig.MCMCiter % fitConfig.MCMCburn;
        //Metropolis mh(simple_target{*metric_to_use}, simple_proposal(*metric_to_use, dseed(PROseed::global_rng)), best_fit, dseed(PROseed::global_rng));
        Metropolis mh(simple_target{*metric_to_use}, adaptive_proposal(*metric_to_use, dseed(PROseed::global_rng)), best_fit, dseed(PROseed::global_rng));

        Eigen::MatrixXf covmat = Eigen::MatrixXf::Constant(nparams, nparams, 0);
        size_t count = 0;
        const auto action = [&](const Eigen::VectorXf &value) {
            covmat += (value-best_fit) * (value-best_fit).transpose();
            count += 1; 
        };
        mh.run(fitConfig.MCMCburn,fitConfig.MCMCiter, action);

        covmat /= count;
        Eigen::VectorXf inv_best_fit = best_fit.array().abs().max(1e-10f).inverse();
        Eigen::MatrixXf fraccovmat = inv_best_fit.asDiagonal() * covmat * inv_best_fit.asDiagonal();

        Eigen::VectorXf inv_sqrt_diag = fraccovmat.diagonal().array().abs().max(1e-10f).sqrt().inverse();
        Eigen::MatrixXf corrmat = inv_sqrt_diag.asDiagonal() * fraccovmat * inv_sqrt_diag.asDiagonal();


        TH2D corrhist("crh", "", nparams, 0, nparams, nparams, 0, nparams);
        TH2D fraccovhist("fch", "", nparams, 0, nparams, nparams, 0, nparams);
        TH2D covhist("ch", "", nparams, 0, nparams, nparams, 0, nparams);
        TH2D physhist;
        if(nphys > 0) physhist = TH2D("ph","", nparams, 0, nparams, nphys, 0, nphys);
        for(size_t i = 0; i < nparams; ++i) {
            std::string label = i < metric_to_use->GetModel().nparams 
                ? metric_to_use->GetModel().pretty_param_names[i]
                : config.m_mcgen_variation_plotname_map[metric_to_use->GetSysts().spline_names[i-metric_to_use->GetModel().nparams]].c_str();
            covhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            covhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            fraccovhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            fraccovhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            corrhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            corrhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            if(nphys > 0) physhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            if(i < metric_to_use->GetModel().nparams) physhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            for(size_t j = 0; j < nparams; ++j) {
                covhist.SetBinContent(i+1, j+1, covmat(i,j));
                fraccovhist.SetBinContent(i+1, j+1, fraccovmat(i,j));
                corrhist.SetBinContent(i+1, j+1, corrmat(i,j));
                if(j < metric_to_use->GetModel().nparams)
                    physhist.SetBinContent(i+1, j+1, covmat(i,j));
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
        //c1.Print((final_output_tag+"_postfit_cov.pdf").c_str());
        //fraccovhist.Draw("colz");
        //c1.Print((final_output_tag+"_postfit_fraccov.pdf").c_str());
        c1.SetLeftMargin(0.18);   
        corrhist.SetTitle("Post-Fit Correlation Matrix");
        corrhist.Draw("colz");
        gPad->Update();

        TLine line;
        line.SetLineColor(kBlack);
        line.SetLineWidth(2);
        line.DrawLine(nphys, 0, nphys, nparams);
        line.DrawLine(0, nphys, nparams, nphys);
        c1.Print((final_output_tag+"_postfit_correlation_matrix.pdf").c_str());
        if(nphys > 0) {
            physhist.Draw("colz");
            c1.Print("phys_cov.pdf");
        }
        log<LOG_INFO>(L"%1% || MCMC acceptance is  %2%. ") % __func__% ((double)count /fitConfig.MCMCiter);

        std::string hname = "#chi^{2}/ndf = " + to_string(best_chi2) + "/" + to_string(config.m_num_variable_bins_total_collapsed[config.i_prime]);
        PROspec cv = FillSpectra(config, prop, metric_to_use->GetSysts(), metric_to_use->GetModel(), CVpparams , true,config.i_prime);
        PROspec bf = FillSpectra(config, prop, metric_to_use->GetSysts(), metric_to_use->GetModel(), best_fit, true,config.i_prime);
        TH1D post_hist("ph", hname.c_str(), config.m_num_variable_bins_total_collapsed[config.i_prime], config.m_channel_variable_bins[config.i_prime][0].Edges().data());
        TH1D pre_hist("prh", hname.c_str(), config.m_num_variable_bins_total_collapsed[config.i_prime], config.m_channel_variable_bins[config.i_prime][0].Edges().data());
        for(size_t i = 0; i < config.m_num_variable_bins_total_collapsed[config.i_prime]; ++i) {
            post_hist.SetBinContent(i+1, bf.Spec()(i));
            pre_hist.SetBinContent(i+1, cv.Spec()(i));
        }

        log<LOG_INFO>(L"%1% || Finished the metropolis hastings chain ") % __func__;

        std::vector<TH1D> priors, posteriors;
        Eigen::MatrixXf prior_covariance, spline_covariance;
        // Fix physics parameters
        std::vector<int> fixed_pars;
        for(size_t i = 0; i < metric_to_use->GetModel().nparams; ++i) fixed_pars.push_back(i);

        log<LOG_INFO>(L"%1% || Starting global getErrorBand() ") % __func__;
        Metropolis mh_pre(prior_only_target{*metric_to_use}, adaptive_proposal(*metric_to_use, dseed(PROseed::global_rng), fixed_pars), best_fit, dseed(PROseed::global_rng));
        //log<LOG_INFO>(L"%1% ||ARSOut %2% ") % __func__ % cv.Spec();
        //log<LOG_INFO>(L"%1% || address %2%") % __func__ % &cv;

        PROerrorbar  err_band = 
            MCMC_prefit_errors
            ? getMCMCErrorBand(mh_pre, fitConfig.MCMCburn, fitConfig.MCMCiter, config, prop, *metric_to_use, best_fit, priors, prior_covariance, binwidth_scale,config.i_prime)
            : getErrorBand(config, prop, variable_systs[config.i_prime], *model, cv,CVpparams, binwidth_scale,config.i_prime);

        //Metropolis mh_post(simple_target{*metric_to_use}, simple_proposal(*metric_to_use, dseed(PROseed::global_rng), 0.2, fixed_pars), best_fit, dseed(PROseed::global_rng));
        Metropolis mh_post(simple_target{*metric_to_use}, adaptive_proposal(*metric_to_use, dseed(PROseed::global_rng), fixed_pars), best_fit, dseed(PROseed::global_rng));
        log<LOG_INFO>(L"%1% || Starting global getPostFitErrorBand() ") % __func__;
        PROerrorbar post_err_band = getMCMCErrorBand(mh_post, fitConfig.MCMCburn, fitConfig.MCMCiter, config, prop, *metric_to_use, best_fit, posteriors, spline_covariance, binwidth_scale,config.i_prime);

        std::vector<TPaveText> texts;
        TPaveText chi2text(0.55, 0.50, 0.85, 0.58, "NDC");
        chi2text.AddText(hname.c_str());
        chi2text.SetFillColor(0);
        chi2text.SetBorderSize(0);
        chi2text.SetTextAlign(12);
        //chi2text.SetTextSize(0.035); 
        texts.push_back(chi2text);

        PlotOptions opt = PlotOptions::DataPostfitRatio;
        if(binwidth_scale) opt |= PlotOptions::BinWidthScaled;
        if(area_normalized) opt |= PlotOptions::AreaNormalized;
        plot_channels((final_output_tag+"_PROfile_hists.pdf"), config, cv, bf, data, err_band, post_err_band, texts, pbounds, opt);

        TCanvas c;
        c.Print((final_output_tag+"_postfit_posteriors.pdf[").c_str());
        for(auto &h: posteriors) {
            h.Draw("hist");
            c.Print((final_output_tag+"_postfit_posteriors.pdf").c_str());
        }
        c.Print((final_output_tag+"_postfit_posteriors.pdf]").c_str());

        Eigen::VectorXf inv_sqrt_diag_nuis = spline_covariance.diagonal().array().abs().max(1e-10f).sqrt().inverse();
        Eigen::MatrixXf corrmat_nuis = inv_sqrt_diag_nuis.asDiagonal() * spline_covariance * inv_sqrt_diag_nuis.asDiagonal();

        TH2F spline_cov("pc", "", corrmat_nuis.cols(), 0, corrmat_nuis.cols(), corrmat_nuis.rows(), 0, corrmat_nuis.rows());
        for(int i = 0; i < corrmat_nuis.cols(); ++i) {
            spline_cov.GetXaxis()->SetBinLabel(i+1, config.m_mcgen_variation_plotname_map[metric_to_use->GetSysts().spline_names[i]].c_str());
            spline_cov.GetYaxis()->SetBinLabel(i+1, config.m_mcgen_variation_plotname_map[metric_to_use->GetSysts().spline_names[i]].c_str());
            for(int j = 0; j < corrmat_nuis.rows(); ++j) {
                spline_cov.SetBinContent(i+1, j+1, corrmat_nuis(i,j));
            }
        }
        spline_cov.Draw("colz");
        spline_cov.SetMaximum(1);
        spline_cov.SetMinimum(-1);

        c.Print((final_output_tag+"_postfit_correlation_matrix_nuisance_only.pdf").c_str());
        log<LOG_INFO>(L"%1% ||  Beginning full PROfile ") % __func__;

        if(progress_bar)scanFitConfig.progress_bar = true;

        std::vector<Eigen::VectorXf> seeds = fitter.freq_seed_points;//to be updated to v1.1.5 harmoincs [DONE]
        if(!seeds.size()) seeds.push_back(best_fit);
        PROfile profile(config, metric_to_use->GetSysts(), metric_to_use->GetModel(), *metric_to_use, myseed, scanFitConfig, 
                final_output_tag+"_PROfile", best_chi2, !systs_only_profile, nthread, seeds,
                systs_only_profile ? systparams : allparams);
        profile.Plot(config, metric_to_use->GetSysts(), metric_to_use->GetModel(), *metric_to_use, myseed,
                final_output_tag+"_PROfile", !systs_only_profile, best_fit,
                systs_only_profile ? systparams : allparams);
        TFile fout((final_output_tag+"_PROfile.root").c_str(), "RECREATE");
        profile.onesig.Write("one_sigma_errs");
        pre_hist.Write("cv");
        //err_band->Write("prefit_errband");
        //post_err_band->Write("postfit_errband");
        post_hist.Write("best_fit");

        //***********************************************************************
        //***********************************************************************
        //******************** PROsurf PROsurf PROsurf **************************
        //***********************************************************************
        //***********************************************************************
    }
    if(*surface_command ){

        size_t nparams = metric->GetModel().nparams + metric->GetSysts().GetNSplines();
        if(global_fit_result.size() == 0 || global_fit_result.size() != (int)nparams) {
            size_t nphys = metric->GetModel().nparams;
            Eigen::VectorXf lb = Eigen::VectorXf::Constant(nparams, -3.0);
            Eigen::VectorXf ub = Eigen::VectorXf::Constant(nparams, 3.0);
            for(size_t i = 0; i < nphys; ++i) {
                lb(i) = metric->GetModel().lb(i);
                ub(i) = metric->GetModel().ub(i);
            }
            for(size_t i = nphys; i < nparams; ++i) {
                lb(i) = metric->GetSysts().spline_lo[i-nphys];
                ub(i) = metric->GetSysts().spline_hi[i-nphys];


            }
            PROfitter fitter(ub, lb, fitConfig);
            metric->setBounds(lb, ub);

            log<LOG_INFO>(L"%1% || ########### Starting Global Best Fit Minimizing ############") % __func__;


            float fit_chi2 = fitter.Fit(*metric, CVpparams); 
            global_fit_chi2_surf = fit_chi2;
            Eigen::VectorXf best_fit = fitter.best_fit;
            if(global_fit_result.size() == 0) global_fit_result = best_fit;
            else if(global_fit_result.size() != best_fit.size()) global_fit_result_surf = best_fit;

            log<LOG_INFO>(L"%1% || ################################################") % __func__;
            log<LOG_INFO>(L"%1% || ########### Global Best Fit Results ############") % __func__;
            log<LOG_INFO>(L"%1% || ################################################") % __func__;
            log<LOG_INFO>(L"%1% || Global Best Fit chi^2: %2%") %__func__ % fit_chi2;
            log<LOG_INFO>(L"%1% || at paramters: ") % __func__;

            for(size_t i = 0; i< nparams; i++){

                if(i<nphys){
                    log<LOG_INFO>(L"%1% || %2%  :  %3% (non-log %4%)") % __func__ % metric->GetModel().pretty_param_names[i].c_str() % best_fit(i) % pow(10,best_fit(i));
                }else{
                    log<LOG_INFO>(L"%1% || %2%  :  %3% ") % __func__ % config.m_mcgen_variation_plotname_map.at(metric->GetSysts().spline_names[i-nphys]).c_str() % best_fit(i);
                }
            }
            log<LOG_INFO>(L"%1% || ################################################") % __func__;
        }
        if (grid_size.empty()) {
            grid_size = {40, 40};
        }
        if (grid_size.size() == 1) {
            grid_size.push_back(grid_size[0]); //make it square
        }

        if(*xlim_opt) {
            xlo = xlims[0];
            xhi = xlims[1];
        }
        if(*ylim_opt) {
            ylo = ylims[0];
            yhi = ylims[1];
        }

        //Define grid and Surface
        size_t xaxis_idx = 1, yaxis_idx = 0;
        if(const auto loc = std::find(model->param_names.begin(), model->param_names.end(), xvar); loc != model->param_names.end()) {
            xaxis_idx = std::distance(model->param_names.begin(), loc);

        } else if(const auto loc = std::find(variable_systs[config.i_prime].spline_names.begin(), variable_systs[config.i_prime].spline_names.end(), xvar); loc != variable_systs[config.i_prime].spline_names.end()) {
            xaxis_idx = std::distance(variable_systs[config.i_prime].spline_names.begin(), loc);
        } else {
            for(const auto &[xml_name, plot_name]: config.m_mcgen_variation_plotname_map) {
                if(xvar == plot_name) {
                    const auto loc = std::find(variable_systs[config.i_prime].spline_names.begin(), variable_systs[config.i_prime].spline_names.end(), xml_name);
                    if(loc != variable_systs[config.i_prime].spline_names.end()) {
                        xaxis_idx = std::distance(variable_systs[config.i_prime].spline_names.begin(), loc);
                    }
                    break;
                }
            }
        }
        if(const auto loc = std::find(model->param_names.begin(), model->param_names.end(), yvar); loc != model->param_names.end()) {
            yaxis_idx = std::distance(model->param_names.begin(), loc);
        } else if(const auto loc = std::find(variable_systs[config.i_prime].spline_names.begin(),variable_systs[config.i_prime].spline_names.end(), yvar); loc != variable_systs[config.i_prime].spline_names.end()) {
            yaxis_idx = std::distance(variable_systs[config.i_prime].spline_names.begin(), loc);
        } else {
            for(const auto &[xml_name, plot_name]: config.m_mcgen_variation_plotname_map) {
                if(yvar == plot_name) {
                    const auto loc = std::find(variable_systs[config.i_prime].spline_names.begin(), variable_systs[config.i_prime].spline_names.end(), xml_name);
                    if(loc != variable_systs[config.i_prime].spline_names.end()) {
                        yaxis_idx = std::distance(variable_systs[config.i_prime].spline_names.begin(), loc);
                    }
                    break;
                }
            }

        }
        size_t nbinsx = grid_size[0], nbinsy = grid_size[1];
        PROsurf surface(*metric, xaxis_idx, yaxis_idx, nbinsx, logx ? PROsurf::LogAxis : PROsurf::LinAxis, xlo, xhi,
                nbinsy, logy ? PROsurf::LogAxis : PROsurf::LinAxis, ylo, yhi);

        if(procurve_points.size()!=0){


            size_t mid = procurve_points.size() / 2;
            std::vector<float> A(procurve_points.begin(), procurve_points.begin() + mid);
            std::vector<float> B(procurve_points.begin() + mid, procurve_points.end());
            size_t Ncurvep = 20;
            log<LOG_INFO>(L"%1% || Running a PROcurve from %2% to point %3% with %4% points") % __func__ % A% B %Ncurvep;
            
            std::vector<surfOut> cpoints = surface.FillCurve(fitConfig, myseed, nthread, A, B, Ncurvep);
            surface.PlotCurve(config,*model,variable_systs[config.i_prime],cpoints,final_output_tag,logx,logy,xaxis_idx,yaxis_idx,A, B, Ncurvep); 
            return 0;
        }


        if(!only_brazil) {
            if(statonly)
                surface.FillSurfaceStat(config, scanFitConfig, final_output_tag+"_statonly_surface.txt");
            else
                surface.FillSurface(scanFitConfig, final_output_tag+"_surface.txt",myseed,nthread);
        }

        std::vector<float> binedges_x, binedges_y;
        for(size_t i = 0; i < surface.nbinsx+1; i++)
            binedges_x.push_back(logx ? std::pow(10, surface.edges_x(i)) : surface.edges_x(i));
        for(size_t i = 0; i < surface.nbinsy+1; i++)
            binedges_y.push_back(logy ? std::pow(10, surface.edges_y(i)) : surface.edges_y(i));

        if(xlabel == "") 
            xlabel = xaxis_idx < model->nparams ? model->pretty_param_names[xaxis_idx] : 
                config.m_mcgen_variation_plotname_map[variable_systs[config.i_prime].spline_names[xaxis_idx]];
        if(ylabel == "") 
            ylabel = yaxis_idx < model->nparams ? model->pretty_param_names[yaxis_idx] : 
                config.m_mcgen_variation_plotname_map[variable_systs[config.i_prime].spline_names[yaxis_idx]];
        TH2D surf("surf", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());

        for(size_t i = 0; i < surface.nbinsx; i++) {
            for(size_t j = 0; j < surface.nbinsy; j++) {
                surf.SetBinContent(i+1, j+1, surface.surface(i, j));
            }
        }

        log<LOG_INFO>(L"%1% || Saving surface to %2% as TH2D named \"surf.\"") % __func__ % final_output_tag.c_str();
        TFile fout((final_output_tag+"_surf.root").c_str(), "RECREATE");
        if(!only_brazil) {
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
                for(size_t i = 0; i < model->nparams; ++i) {
                    best_fit[model->param_names[i]] = res.best_fit(i);
                }
                for(size_t i = 0; i < variable_systs[config.i_prime].GetNSplines(); ++i) {
                    best_fit[variable_systs[config.i_prime].spline_names[i]] = res.best_fit(i + model->nparams);
                }
                tree.Fill();
            }
            // TODO: Should we save the spectra as TH1s?

            tree.Write();

            TCanvas c;
            if(logy)
                c.SetLogy();
            if(logx)
                c.SetLogx();
            c.SetLogz();
            surf.Draw("colz");
            c.Print((final_output_tag+"_surface.pdf").c_str());
        }

        std::vector<PROsurf> brazil_band_surfaces;
        if(run_brazil && brazil_throws.size() == 0) {
            std::normal_distribution<float> d;
            size_t nphys = metric->GetModel().nparams;
            PROspec cv = FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), CVpparams , true,config.i_prime);

            PROspec collapsed_cv = PROspec(CollapseMatrix(config, cv.Spec()), CollapseMatrix(config, cv.Error()));
            Eigen::MatrixXf L = metric->GetSysts().DecomposeFractionalCovariance(config, cv.Spec());
            for(size_t i = 0; i < 1000; ++i) {
                Eigen::VectorXf throwp = pparams;
                Eigen::VectorXf throwC = Eigen::VectorXf::Constant(config.m_num_variable_bins_total_collapsed[config.i_prime], 0);
                for(size_t i = 0; i < metric->GetSysts().GetNSplines(); i++)
                    throwp(i+nphys) = d(PROseed::global_rng);
                for(size_t i = 0; i < config.m_num_variable_bins_total[config.i_prime]; i++)
                    throwC(i) = d(PROseed::global_rng);
                PROspec shifted = FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), throwp, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
                PROspec newSpec = statonly_brazil ? PROspec::PoissonVariation(collapsed_cv, dseed(myseed.global_rng)) :
                    PROspec::PoissonVariation(PROspec(CollapseMatrix(config, shifted.Spec()) + L * throwC, CollapseMatrix(config, shifted.Error())), dseed(myseed.global_rng));
                PROdata data(newSpec.Spec(), newSpec.Error());
                PROmetric *metric;
                if(chi2 == "PROchi") {
                    metric = new PROchi("", config, prop, &variable_systs[config.i_prime], *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
                } else if(chi2 == "PROCNP") {
                    metric = new PROCNP("", config, prop, &variable_systs[config.i_prime], *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
                } else if(chi2 == "Poisson") {
                    metric = new PROpoisson("", config, prop, &variable_systs[config.i_prime], *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
                } else {
                    log<LOG_ERROR>(L"%1% || Unrecognized chi2 function %2%") % __func__ % chi2.c_str();
                    abort();
                }

                brazil_band_surfaces.emplace_back(*metric, xaxis_idx, yaxis_idx, nbinsx, logx ? PROsurf::LogAxis : PROsurf::LinAxis, xlo, xhi,
                        nbinsy, logy ? PROsurf::LogAxis : PROsurf::LinAxis, ylo, yhi);

                if(statonly)
                    brazil_band_surfaces.back().FillSurfaceStat(config, scanFitConfig, "");
                else
                    brazil_band_surfaces.back().FillSurface(scanFitConfig, "", myseed, nthread);

                TH2D surf("surf", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());

                for(size_t i = 0; i < surface.nbinsx; i++) {
                    for(size_t j = 0; j < surface.nbinsy; j++) {
                        surf.SetBinContent(i+1, j+1, brazil_band_surfaces.back().surface(i, j));
                    }
                }
                surf.Write(("brazil_throw_surf_"+std::to_string(i)).c_str());

                // WARNING: Metric reference stored in surface. DO NOT USE IT AFTER THIS POINT.
                delete metric;
                if(single_brazil) break;
            }
        } else if(run_brazil) { // if brazil_thows.size() > 0
            for(const std::string &in: brazil_throws) {
                brazil_band_surfaces.emplace_back(*metric, xaxis_idx, yaxis_idx, nbinsx, logx ? PROsurf::LogAxis : PROsurf::LinAxis, xlo, xhi,
                        nbinsy, logy ? PROsurf::LogAxis : PROsurf::LinAxis, ylo, yhi);

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

        if(run_brazil && !single_brazil) {
            TH2D surf16("surf16", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());
            TH2D surf84("surf84", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());
            TH2D surf98("surf98", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());
            TH2D surf02("surf02", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());
            TH2D surf50("surf50", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());

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
            if(brazil_throws.size() != 0) {
                int i = 0;
                for(const auto &bbsurf: brazil_band_surfaces) {
                    TH2D *surf = new TH2D(("brazil_throw_surf_"+std::to_string(i)).c_str(),(";"+xlabel+";"+ylabel).c_str(),surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data())     ;
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

        //***********************************************************************
        //***********************************************************************
        //******************** PROplot PROplot PROplot **************************
        //***********************************************************************
        //***********************************************************************
    }
    if(*proplot_command){

        log<LOG_INFO>(L"%1% || Making a PROsyst thats full covarinace for future error bar creation (might be slow) ")% __func__ ;
        PROsyst allcovsyst = variable_systs[config.i_prime].allsplines2cov(config, prop, *model, CVpparams, dseed(PROseed::global_rng));

        PlotOptions opt = PlotOptions::CVasStack;
        std::vector<TPaveText> notext;
        if(binwidth_scale) opt |= PlotOptions::BinWidthScaled;
        if(area_normalized) opt |= PlotOptions::AreaNormalized;
        std::vector<PROspec> variable_cvs;
        for(size_t io = 0; io < config.m_num_variables; ++io) {
            variable_cvs.push_back(FillSpectra(config, prop, variable_systs[config.i_prime],*model,CVpparams, !eventbyevent, io));
            plot_channels(final_output_tag+"_PROplot_Variable_"+std::to_string(io)+"_CV.pdf", config, variable_cvs.back(), {}, {}, {}, {}, notext, pbounds, opt, io);
        }

        std::string filename = final_output_tag+"_fractional_systematics.pdf";
        plotPriorFractionalSystematicBreakdown(config, variable_cvs[config.i_prime], allcovsyst, filename,config.i_prime);

        std::vector<std::map<std::string, std::unique_ptr<TH1D>>> other_hists;
        for(size_t io = 0; io < config.m_num_variables; ++io) {
            other_hists.push_back(getCVHists(variable_cvs[io], config, binwidth_scale, io));
        }

        TCanvas c;
        if(fake_data_osc_params.size()) {

            c.Print((final_output_tag +"_PROplot_Osc.pdf"+ "[").c_str(), "pdf");

            PROspec osc_spec = FillSpectra(config, prop, variable_systs[config.i_prime], *model, pparams, !eventbyevent,config.i_prime );
            std::map<std::string, std::unique_ptr<TH1D>> osc_hists = getCVHists(osc_spec, config, binwidth_scale);
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
                                cv_hist = (TH1D*)h->Clone();
                                osc_hist = (TH1D*)o->Clone();
                            } else {
                                cv_hist->Add(&*h);
                                osc_hist->Add(&*o);
                            }
                            ++global_subchannel_index;
                        }
                        if(binwidth_scale )
                            cv_hist->GetYaxis()->SetTitle("Events/GeV");
                        else
                            cv_hist->GetYaxis()->SetTitle("Events");
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
                        rat->Divide(cv_hist);
                        rat->SetTitle("");
                        rat->GetYaxis()->SetTitle("Ratio");
                        TH1D *one = (TH1D*)rat->Clone();
                        one->Divide(one);
                        one->SetLineColor(kBlack);
                        one->GetYaxis()->SetTitle("Ratio");

                        std::unique_ptr<TLegend> leg = std::make_unique<TLegend>(0.59,0.89,0.59,0.89);
                        leg->SetFillStyle(0);
                        leg->SetLineWidth(0);
                        leg->AddEntry(cv_hist, "No Oscillations", "l");
                        std::string oscstr = "";//"#splitline{Oscilations:}{";
                        for(size_t j=0;j<model->nparams;j++){
                            oscstr+=model->pretty_param_names[j]+ " : "+ to_string_prec(pow(10,fake_data_osc_param_vector(j)),3) +" "+model->pretty_param_units[j] + (j==0 ? ", " : "" );
                        }
                        //oscstr+="}";

                        leg->AddEntry(osc_hist, oscstr.c_str(), "l");

                        TPad p1("p1", "p1", 0, 0.25, 1, 1);
                        p1.SetBottomMargin(0);
                        p1.cd();
                        cv_hist->Draw("hist");
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
                        one->Draw("hist");
                        one->SetMaximum(rat->GetMaximum()*1.2);
                        one->SetMinimum(rat->GetMinimum()*0.8);
                        rat->Draw("hist same");

                        c.cd();
                        p1.Draw();
                        p2.Draw();

                        c.Print((final_output_tag+"_PROplot_Osc.pdf").c_str(), "pdf");

                        delete cv_hist;
                        delete osc_hist;
                    }
                }
            }
            c.Print((final_output_tag+"_PROplot_Osc.pdf" + "]").c_str(), "pdf");

        }

        //Now some covariances
        std::map<std::string, std::unique_ptr<TH2D>> matrices = covarianceTH2D(allcovsyst, config, variable_cvs[config.i_prime]);
        c.Print((final_output_tag+"_PROplot_Covar.pdf" + "[").c_str(), "pdf");

        std::vector<std::string> first_plots = {"collapsed_total_cor","collapsed_total_frac_cov","total_cor","total_frac_cov"};

        for(const auto &name: first_plots){
            auto &mat = matrices.at(name);
            mat->Draw("colz");
            TText *t = new TText();
            t->SetNDC();                
            t->SetTextFont(42);                          
            t->SetTextSize(0.03);      
            t->SetTextAlign(33);        
            std::string pv = "PROfit v"+std::string(PROJECT_VERSION_STR);
            t->DrawText(0.895, 0.955, pv.c_str()); 
            c.Print((final_output_tag+"_PROplot_Covar.pdf").c_str(), "pdf");
        }


        for(const auto &[name, mat]: matrices) {
            if (std::find(first_plots.begin(), first_plots.end(), name) != first_plots.end())continue;
            mat->Draw("colz");
            TText *t = new TText();
            t->SetNDC();                
            t->SetTextFont(42);                          
            t->SetTextSize(0.03);      
            t->SetTextAlign(33);        
            std::string pv = "PROfit v"+std::string(PROJECT_VERSION_STR);
            t->DrawText(0.895, 0.955, pv.c_str()); 
            c.Print((final_output_tag+"_PROplot_Covar.pdf").c_str(), "pdf");
        }
        c.Print((final_output_tag+"_PROplot_Covar.pdf" + "]").c_str(), "pdf");

        //errorband
        std::unique_ptr<PROmetric> allcov_metric(metric->Clone());
        allcov_metric->override_systs(allcovsyst);
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
                        if(io==config.i_prime){
                            log<LOG_INFO>(L"%1% || On channel %2%:") % __func__ % global_channel_index ;
                            double chival = allcov_metric->getSingleChannelChi(global_channel_index, variable_cvs[io], io);
                            int ndf = config.m_channel_variable_bins[ic][io].NBinsAlong(0) - bool(opt&PlotOptions::AreaNormalized);
                            log<LOG_INFO>(L"%1% || -- the datamc chi^2/ndof is %2%/%3% .") % __func__ % chival % ndf;
                            chi2text.AddText(("#chi^{2}/ndf = "+to_string_prec(chival,2)+"/"+std::to_string(ndf)).c_str());
                        }else{
                            chi2text.AddText("");
                        }
                        chi2text.SetFillColor(0);
                        chi2text.SetBorderSize(0);
                        chi2text.SetTextAlign(12);
                        channel_chitexts.push_back(chi2text);
                        global_channel_index++;
                    }
                }
            }
            other_channel_chitexts.push_back(channel_chitexts);
        }


        std::vector<PROerrorbar> other_err_bands;
        for(size_t io = 0; io < config.m_num_variables; ++io) {
            if(!config.m_channel_variable_plot_bool.at(io))continue;// For now skip the L/E 250 bin. 
            other_err_bands.push_back(getErrorBand(config, prop, variable_systs[io], *model, variable_cvs[io], CVpparams, binwidth_scale, io));
            plot_channels(final_output_tag+"_PROplot_Variable_"+std::to_string(io)+"_ErrorBand.pdf", config, variable_cvs[io], {}, variable_data[io], 
                    other_err_bands.back(), {}, other_channel_chitexts[io], pbounds, opt | PlotOptions::DataMCRatio, io);
        }



        if(with_splines) {
            c.Print((final_output_tag+"_PROplot_Spline.pdf" + "[").c_str(), "pdf");

            std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> spline_graphs = getSplineGraphs(variable_systs[config.i_prime], config);
            c.Clear();
            c.Divide(4,4);
            int chan=0;
            for(const auto &[syst_name, syst_bins]: spline_graphs) {
                int bin = 0;
                bool unprinted = true;
                chan++;
                int col = chan%2==0 ? kRed: kBlue;
                int lastsbindex = 0;
                for(const auto &[fixed_pts, curve]: syst_bins) {
                    unprinted = true;
                    c.cd(bin%16+1);
                    size_t sbi = config.GetSubchannelIndexFromVariableGlobalBin(bin,config.i_prime);
                    std::string nsubchannel = config.GetSubchannelName(sbi);
                    std::string chan_units =   config.m_channel_variable_units[config.i_prime][config.GetLocalChannelIndexFromGlobalSubchannelIndex(sbi)];
                    int edges_vec_sz = (int)config.m_variable_bin_to_edges[config.i_prime].size();
                    size_t safe_bin = (edges_vec_sz>0) ? std::min((size_t)bin, (size_t)edges_vec_sz-1) : 0;
                    std::pair<float,float> edg = config.m_variable_bin_to_edges[config.i_prime][safe_bin];

                    fixed_pts->SetMarkerColor(col);
                    fixed_pts->SetMarkerStyle(kFullCircle);
                    fixed_pts->GetXaxis()->SetTitle("#sigma");
                    fixed_pts->GetYaxis()->SetTitle("Weight");
                    fixed_pts->SetTitle(("#splitline{"+syst_name+"}{#splitline{"+nsubchannel+" "+std::to_string(bin)+"}{"+chan_units+" ["+to_string_prec(edg.first,2)+"->"+to_string_prec(edg.second,2)+ "]}}").c_str());
                    fixed_pts->Draw("PA");
                    double max_y = TMath::MaxElement(fixed_pts->GetN(), fixed_pts->GetY());
                    double min_y = TMath::MinElement(fixed_pts->GetN(), fixed_pts->GetY());
                    double range = max_y - min_y;
                    fixed_pts->SetMaximum(max_y + 0.2 * range);
                    fixed_pts->SetMinimum(min_y);

                    curve->Draw("C same");
                    ++bin;
                    if(bin % 16 == 0) {
                        c.Print((final_output_tag+"_PROplot_Spline.pdf").c_str(), "pdf");
                        c.Clear();
                        c.Divide(4,4);
                        unprinted = false;
                    }
                }
                if(unprinted)
                    c.Print((final_output_tag+"_PROplot_Spline.pdf").c_str(), "pdf");
            }

            c.Print((final_output_tag+"_PROplot_Spline.pdf" + "]").c_str(), "pdf");
        }

        //now onto root files
        TFile fout((final_output_tag+"_PROplot.root").c_str(), "RECREATE");

        int io = 0;
        for(const auto &other: other_hists) {
            for(const auto &[name, hist]: other) {
                hist->Write(("Variable_"+std::to_string(io)+name).c_str());
            }
            io++;
        }

        if((fake_data_osc_params.size())) {
            PROspec osc_spec = FillSpectra(config, prop, variable_systs[config.i_prime], *model, pparams, !eventbyevent,config.i_prime);
            std::map<std::string, std::unique_ptr<TH1D>> osc_hists = getCVHists(osc_spec, config, binwidth_scale);
            fout.mkdir("Osc_hists");
            fout.cd("Osc_hists");
            for(const auto &[name, hist]: osc_hists) {
                hist->Write(name.c_str());
            }
        }

        fout.mkdir("Covariance");
        fout.cd("Covariance");
        for(const auto &[name, mat]: matrices)
            mat->Write(name.c_str());

        //fout.mkdir("ErrorBand");
        //fout.cd("ErrorBand");
        //err_band->Write("err_band");
        //io = 0;
        //for(const auto &band: other_err_bands)
        //band->Write(("other_"+std::to_string(io++)+"_err_band").c_str());


        if((with_splines)) {
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

    //***********************************************************************
    //***********************************************************************
    //********************     Feldman-Cousins    ***************************
    //***********************************************************************
    //***********************************************************************

    if(*profc_command) {
        size_t FCthreads = nthread > nuniv ? nuniv : nthread;
        Eigen::MatrixXf cv_vec = FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), CVpparams , true,config.i_prime).Spec();

        Eigen::MatrixXf L = variable_systs[config.i_prime].DecomposeFractionalCovariance(config, cv_vec);

        std::vector<std::vector<float>> dchi2s;
        dchi2s.reserve(FCthreads);
        std::vector<std::vector<fc_out>> outs;
        outs.reserve(FCthreads);
        std::vector<std::thread> threads;
        size_t todo = nuniv/FCthreads;
        size_t addone = FCthreads - nuniv%FCthreads;
        for(size_t i = 0; i < nthread; i++) {
            dchi2s.emplace_back();
            outs.emplace_back();
            fc_args args{todo + (i >= addone), &dchi2s.back(), &outs.back(), config, prop, variable_systs[config.i_prime], chi2, pparams, L, scanFitConfig,(*myseed.getThreadSeeds())[i], (int)i, !eventbyevent};

            threads.emplace_back([args]() {
                    PROfit::fc_worker(args);
                    });
        }
        for(auto&& t: threads) {
            t.join();
        }

        {
            TFile fout((final_output_tag+"_FC.root").c_str(), "RECREATE");
            fout.cd();
            float chi2_osc, chi2_syst, best_dmsq, best_sinsq2t;
            std::map<std::string, float> best_systs_osc, best_systs, syst_throw;
            TTree tree("tree", "tree");
            tree.Branch("chi2_osc", &chi2_osc); 
            tree.Branch("chi2_syst", &chi2_syst); 
            tree.Branch("best_dmsq", &best_dmsq); 
            tree.Branch("best_sinsq2t", &best_sinsq2t); 
            tree.Branch("best_systs_osc", &best_systs_osc); 
            tree.Branch("best_systs", &best_systs); 
            tree.Branch("syst_throw", &syst_throw);

            for(const auto &out: outs) {
                for(const auto &fco: out) {
                    chi2_osc = fco.chi2_osc;
                    chi2_syst = fco.chi2_syst;
                    best_dmsq = fco.dmsq;
                    best_sinsq2t = fco.sinsq2tmm;
                    for(size_t i = 0; i < variable_systs[config.i_prime].GetNSplines(); ++i) {
                        best_systs_osc[variable_systs[config.i_prime].spline_names[i]] = fco.best_fit_osc(i);
                        best_systs[variable_systs[config.i_prime].spline_names[i]] = fco.best_fit_syst(i);
                        syst_throw[variable_systs[config.i_prime].spline_names[i]] = fco.syst_throw(i);
                    }
                    tree.Fill();
                }
            }

            tree.Write();
        }
        {
            ofstream fcout(final_output_tag+"_FC.csv");
            fcout << "chi2_osc,chi2_syst,best_dmsq,best_sinsq2t";
            for(const std::string &name: variable_systs[config.i_prime].spline_names) {
                fcout << ",best_" << name << "_osc,best_" << name << "," << name << "_throw";
            }
            fcout << "\r\n";

            for(const auto &out: outs) {
                for(const auto &fco: out) {
                    fcout << fco.chi2_osc << "," << fco.chi2_syst << "," << fco.dmsq << "," << fco.sinsq2tmm;
                    for(size_t i = 0; i < variable_systs[config.i_prime].GetNSplines(); ++i) {
                        fcout << fco.best_fit_osc(i) << "," << fco.best_fit_syst(i) << "," << fco.syst_throw(i);
                    }
                    fcout << "\r\n";
                }
            }
        }
        std::vector<float> flattened_dchi2s;
        for(const auto& v: dchi2s) for(const auto& dchi2: v) flattened_dchi2s.push_back(dchi2);
        std::sort(flattened_dchi2s.begin(), flattened_dchi2s.end());
        log<LOG_INFO>(L"%1% || 90%% Feldman-Cousins delta chi2 after throwing %2% universes is %3%") 
            % __func__ % nuniv % flattened_dchi2s[0.9*flattened_dchi2s.size()];
    }


    //***********************************************************************
    //***********************************************************************
    //******************** global       **************************
    //***********************************************************************
    //***********************************************************************
    if(*proglobal_command){


        PROmetric *metric_to_use = systs_only_profile ? null_metric : metric;
        size_t nparams = metric_to_use->nParams();
        size_t nphys = metric_to_use->GetModel().nparams;
        PROfitter fitter(metric_to_use->UpperBound(), metric_to_use->LowerBound(), fitConfig);
        metric_to_use->setBounds(metric_to_use->UpperBound(), metric_to_use->LowerBound());

        log<LOG_INFO>(L"%1% || ########### Print of inputs ############") % __func__;
        //metric_to_use->print(allparams); //fix
        log<LOG_INFO>(L"%1% || ########### Starting Global Best Fit Minimizing ############") % __func__;

        std::vector<std::pair<int, std::string>> global_PB_configs;
        global_PB_configs.push_back({fitConfig.n_latin_points, "(1) LatinHyperCube"});
        global_PB_configs.push_back({fitConfig.n_swarm_iterations, "(2) ParticleSwarm"});
        global_PB_configs.push_back({fitConfig.n_localfit, "(3) BestLBFGSB"});
        global_PB_configs.push_back({fitConfig.harmonic_num_test_points, "(4) HarmonicScan"});
        global_PB_configs.push_back({100, "(5) HarmonicLBFGSB"});
        MultiPROgressBar global_progress(global_PB_configs);

        if(progress_bar){
            global_progress.initialize_display();
            global_progress.start_display_thread(); 
            fitter.setProgressBar(&global_progress);
        }

        float best_chi2 = fitter.Fit(*metric_to_use,CVpparams); 
        Eigen::VectorXf best_fit = fitter.best_fit;
        Eigen::MatrixXf post_covar = fitter.Covariance();
        fitter.calcFreqSeedPoints(*metric_to_use);

        for(size_t i=0; i< fitter.freq_seed_points.size(); i++){
            float chi_freq = fitter.freq_seed_values.at(i);
            if( chi_freq < best_chi2){
                log<LOG_INFO>(L"%1% || One of the harmonics of first pass best fit, is a lower chi :  %2% ") % __func__ % fitter.freq_seed_values.at(i);
                log<LOG_INFO>(L"%1% || -- at params:  %2% ") % __func__ % fitter.freq_seed_points.at(i);
                best_chi2 = chi_freq;
                best_fit = fitter.freq_seed_points.at(i);
            }
        }
        if(progress_bar)global_progress.finish_all();
        global_fit_chi2 = best_chi2;
        if(global_fit_result.size() == 0) global_fit_result = best_fit;

        if (fitter.exception_string_map.empty()) {
            log<LOG_INFO>(L"%1% || No exceptions were caught from LBFGSB [ --INFO-- ]") % __func__;
        } else {
            log<LOG_INFO>(L"%1% || Some exceptions were caught in LBFGSB [ --INFO-- ]") % __func__;
            for (const auto &[msg, count] : fitter.exception_string_map) {
                log<LOG_INFO>(L"%1% ||  -- Exception \"%2%\" occurred %3% time(s)") % __func__ % msg.c_str() % count;
            }
        }


        log<LOG_INFO>(L"%1% || ################################################") % __func__;
        log<LOG_INFO>(L"%1% || ########### Global Best Fit Results ############") % __func__;
        log<LOG_INFO>(L"%1% || ################################################") % __func__;
        log<LOG_INFO>(L"%1% || Global Best Fit chi^2: %2%") %__func__ % best_chi2;
        log<LOG_INFO>(L"%1% || at paramters: ") % __func__;

        for(size_t i = 0; i< nparams; i++){

            if(i<nphys){
                log<LOG_INFO>(L"%1% || %2%  : %3% (log) %4% (nonlog) ") % __func__ % metric_to_use->GetModel().pretty_param_names[i].c_str() % best_fit(i) % pow(10,best_fit(i));
            }else{
                log<LOG_INFO>(L"%1% || %2%  :  %3% ") % __func__ % config.m_mcgen_variation_plotname_map.at(metric->GetSysts().spline_names[i-nphys]).c_str() % best_fit(i);
            }
        }
        log<LOG_INFO>(L"%1% || ################################################") % __func__;

        log<LOG_INFO>(L"%1% || Starting a metropolis hastings chain to estimate the covariace matrix aroud the above best fit. Run and Burn is (%2%,%3%);") % __func__%fitConfig.MCMCiter % fitConfig.MCMCburn;
        Metropolis mh(simple_target{*metric_to_use}, adaptive_proposal(*metric_to_use, dseed(PROseed::global_rng)), best_fit, dseed(PROseed::global_rng));

        Eigen::MatrixXf covmat = Eigen::MatrixXf::Constant(nparams, nparams, 0);
        size_t count = 0;
        const auto action = [&](const Eigen::VectorXf &value) {
            covmat += (value-best_fit) * (value-best_fit).transpose();
            count += 1; 
        };
        mh.run(fitConfig.MCMCburn,fitConfig.MCMCiter, action);

        covmat /= count;
        Eigen::VectorXf inv_best_fit = best_fit.array().abs().max(1e-10f).inverse();
        Eigen::MatrixXf fraccovmat = inv_best_fit.asDiagonal() * covmat * inv_best_fit.asDiagonal();

        Eigen::VectorXf inv_sqrt_diag = fraccovmat.diagonal().array().abs().max(1e-10f).sqrt().inverse();
        Eigen::MatrixXf corrmat = inv_sqrt_diag.asDiagonal() * fraccovmat * inv_sqrt_diag.asDiagonal();


        TH2D corrhist("crh", "", nparams, 0, nparams, nparams, 0, nparams);
        TH2D fraccovhist("fch", "", nparams, 0, nparams, nparams, 0, nparams);
        TH2D covhist("ch", "", nparams, 0, nparams, nparams, 0, nparams);
        TH2D physhist;
        if(nphys > 0) physhist = TH2D("ph","", nparams, 0, nparams, nphys, 0, nphys);
        for(size_t i = 0; i < nparams; ++i) {
            std::string label = i < metric_to_use->GetModel().nparams 
                ? metric_to_use->GetModel().pretty_param_names[i]
                : config.m_mcgen_variation_plotname_map[metric_to_use->GetSysts().spline_names[i-metric_to_use->GetModel().nparams]].c_str();
            covhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            covhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            fraccovhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            fraccovhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            corrhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            corrhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            if(nphys > 0) physhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            if(i < metric_to_use->GetModel().nparams) physhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            for(size_t j = 0; j < nparams; ++j) {
                covhist.SetBinContent(i+1, j+1, covmat(i,j));
                fraccovhist.SetBinContent(i+1, j+1, fraccovmat(i,j));
                corrhist.SetBinContent(i+1, j+1, corrmat(i,j));
                if(j < metric_to_use->GetModel().nparams)
                    physhist.SetBinContent(i+1, j+1, covmat(i,j));
            }
        }
        TCanvas c1;
        corrhist.SetMaximum(1);
        corrhist.SetMinimum(-1);
        covhist.SetMaximum(1);
        covhist.SetMinimum(-1);
        fraccovhist.SetMaximum(100);
        fraccovhist.SetMinimum(-100);
        c1.SetLeftMargin(0.18);   
        corrhist.SetTitle("Post-Fit Correlation Matrix");
        corrhist.Draw("colz");
        gPad->Update();

        TLine line;
        line.SetLineColor(kBlack);
        line.SetLineWidth(2);
        line.DrawLine(nphys, 0, nphys, nparams);
        line.DrawLine(0, nphys, nparams, nphys);
        c1.Print((final_output_tag+"_postfit_correlation_matrix.pdf").c_str());
        if(nphys > 0) {
            physhist.Draw("colz");
            c1.Print("phys_cov.pdf");
        }
        log<LOG_INFO>(L"%1% || MCMC acceptance is  %2%. ") % __func__% ((double)count /fitConfig.MCMCiter);

        std::string hname = "#chi^{2}/ndf = " + to_string(best_chi2) + "/" + to_string(config.m_num_variable_bins_total_collapsed[config.i_prime]);
        PROspec cv = FillSpectra(config, prop, metric_to_use->GetSysts(), metric_to_use->GetModel(), CVpparams , true,config.i_prime);
        PROspec bf = FillSpectra(config, prop, metric_to_use->GetSysts(), metric_to_use->GetModel(), best_fit, true,config.i_prime);

        TH1D post_hist("ph", hname.c_str(), config.m_num_variable_bins_total_collapsed[config.i_prime], config.m_channel_variable_bins[config.i_prime][0].Edges().data());
        TH1D pre_hist("prh", hname.c_str(), config.m_num_variable_bins_total_collapsed[config.i_prime], config.m_channel_variable_bins[config.i_prime][0].Edges().data());
        for(size_t i = 0; i < config.m_num_variable_bins_total_collapsed[config.i_prime]; ++i) {
            post_hist.SetBinContent(i+1, bf.Spec()(i));
            pre_hist.SetBinContent(i+1, cv.Spec()(i));
        }

        log<LOG_INFO>(L"%1% || Finished the metropolis hastings chain ") % __func__;

        std::vector<TH1D> priors, posteriors;
        Eigen::MatrixXf prior_covariance, spline_covariance;
        // Fix physics parameters
        std::vector<int> fixed_pars;
        for(size_t i = 0; i < metric_to_use->GetModel().nparams; ++i) fixed_pars.push_back(i);

        log<LOG_INFO>(L"%1% || Starting global getErrorBand() ") % __func__;
        Metropolis mh_pre(prior_only_target{*metric_to_use}, adaptive_proposal(*metric_to_use, dseed(PROseed::global_rng), fixed_pars), best_fit, dseed(PROseed::global_rng));

        PROerrorbar  err_band = 
            MCMC_prefit_errors
            ? getMCMCErrorBand(mh_pre, fitConfig.MCMCburn, fitConfig.MCMCiter, config, prop, *metric_to_use, best_fit, priors, prior_covariance, binwidth_scale,config.i_prime)
            : getErrorBand(config, prop, variable_systs[config.i_prime], *model, cv,CVpparams, binwidth_scale,config.i_prime);

        Metropolis mh_post(simple_target{*metric_to_use}, adaptive_proposal(*metric_to_use, dseed(PROseed::global_rng), fixed_pars), best_fit, dseed(PROseed::global_rng));
        log<LOG_INFO>(L"%1% || Starting global getPostFitErrorBand() ") % __func__;
        PROerrorbar post_err_band = getMCMCErrorBand(mh_post, fitConfig.MCMCburn, fitConfig.MCMCiter, config, prop, *metric_to_use, best_fit, posteriors, spline_covariance, binwidth_scale,config.i_prime);

        std::vector<TPaveText> texts;
        TPaveText chi2text(0.55, 0.50, 0.85, 0.58, "NDC");
        chi2text.AddText(hname.c_str());
        chi2text.SetFillColor(0);
        chi2text.SetBorderSize(0);
        chi2text.SetTextAlign(12);
        //chi2text.SetTextSize(0.035); 
        texts.push_back(chi2text);

        PlotOptions opt = PlotOptions::DataPostfitRatio;
        if(binwidth_scale) opt |= PlotOptions::BinWidthScaled;
        if(area_normalized) opt |= PlotOptions::AreaNormalized;
        plot_channels((final_output_tag+"_PROglobal_hists.pdf"), config, cv, bf, data, err_band, post_err_band, texts, pbounds,opt);


    }

    //***********************************************************************
    //***********************************************************************
    //******************** TEST AREA TEST AREA     **************************
    //***********************************************************************
    //***********************************************************************
    if(*protest_command){
        log<LOG_INFO>(L"%1% || PROtest. Place anything here, a playground for testing things .") % __func__;
        PrintVariableInfo(config);
        //***************************** END *********************************
    }

    std::ofstream global_fit_out;
    if(global_fit_result.size() > 0) {
        global_fit_out.open(final_output_tag+"_global_fit.txt");
        float chi2 = global_fit_chi2 >= 0 ? global_fit_chi2 : global_fit_chi2_surf;
        log<LOG_INFO>(L"%1% || ################################################") % __func__;
        log<LOG_INFO>(L"%1% || ########### Global Best Fit Results ############") % __func__;
        log<LOG_INFO>(L"%1% || ################################################") % __func__;
        log<LOG_INFO>(L"%1% || Global Best Fit chi^2: %2%") %__func__ % chi2;
        log<LOG_INFO>(L"%1% || at paramters: ") % __func__;

        global_fit_out << "Global best fit:\n";

        bool use_phys = (size_t)global_fit_result.size() == metric->GetModel().nparams + metric->GetSysts().GetNSplines();
        for(long i = 0; i < global_fit_result.size(); i++){

            if(use_phys && i < (long)metric->GetModel().nparams){
                log<LOG_INFO>(L"%1% || %2%  : %3% (log) %4% (nonlog) ") % __func__ % metric->GetModel().pretty_param_names[i].c_str() % global_fit_result(i) % pow(10,global_fit_result(i));
                global_fit_out << metric->GetModel().param_names[i] << " : " << global_fit_result(i) << "\n";
            }else{
                long idx = use_phys ? i - metric->GetModel().nparams : i;
                log<LOG_INFO>(L"%1% || %2%  :  %3% ") % __func__ % config.m_mcgen_variation_plotname_map.at(metric->GetSysts().spline_names[idx]).c_str() % global_fit_result(i);

                global_fit_out <<  config.m_mcgen_variation_plotname_map.at(metric->GetSysts().spline_names[idx])
                    << " : " << global_fit_result(i) << "\n";
            }
        }
        log<LOG_INFO>(L"%1% || ################################################") % __func__;
    }
    if(global_fit_result_surf.size() > 0) {
        if(!global_fit_out.is_open()) {
            global_fit_out.open(final_output_tag+"_global_fit.txt");
            global_fit_out << "Global best fit:\n";
        } else {
            global_fit_out << "\nSurface global best fit:\n";
        }
        log<LOG_INFO>(L"%1% || ########################################################") % __func__;
        log<LOG_INFO>(L"%1% || ########### Surface Global Best Fit Results ############") % __func__;
        log<LOG_INFO>(L"%1% || ########################################################") % __func__;
        log<LOG_INFO>(L"%1% || Global Best Fit chi^2: %2%") %__func__ % global_fit_chi2_surf;
        log<LOG_INFO>(L"%1% || at paramters: ") % __func__;

        for(long i = 0; i < global_fit_result.size(); i++){
            if(i < (long)metric->GetModel().nparams){
                log<LOG_INFO>(L"%1% || %2%  : %3% (log) %4% (nonlog) ") % __func__ % metric->GetModel().pretty_param_names[i].c_str() % global_fit_result(i) % pow(10,global_fit_result(i));
                global_fit_out << metric->GetModel().param_names[i] << " : " << global_fit_result(i) << "\n";
            }else{
                log<LOG_INFO>(L"%1% || %2%  :  %3% ") % __func__ % metric->GetSysts().spline_names[i - metric->GetModel().nparams].c_str() % global_fit_result(i);
                global_fit_out << metric->GetSysts().spline_names[i - metric->GetModel().nparams]
                    << " : " << global_fit_result(i) << "\n";
            }
        }
        log<LOG_INFO>(L"%1% || ########################################################") % __func__;
    }
    if(global_fit_out.is_open()) global_fit_out.close();

    delete metric;

    return 0;
}



