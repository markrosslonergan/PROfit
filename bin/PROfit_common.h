#ifndef PROFIT_BIN_PROFIT_COMMON_H
#define PROFIT_BIN_PROFIT_COMMON_H

// Shared declarations for the PROfit executable's translation units
// (bin/PROfit_*.cxx). Private to bin/ -- not part of the PROfitLib API.

#include "PROconfig.h"
#include "PROdata.h"
#include "PROlog.h"
#include "PROmetric.h"
#include "PROspec.h"
#include "PROsyst.h"
#include "PROcreate.h"
#include "PROpeller.h"
#include "PROmetrics/PROchi.h"
#include "PROmetrics/PROpearson.h"
#include "PROmetrics/PROCNP.h"
#include "PROmetrics/PROpoisson.h"
#include "PROcess.h"
#include "PROsurf.h"
#include "PROfc.h"
#include "PROAdaptiveFC.h"
#include "PROfitter.h"
#include "PROmodel.h"
#include "PROmodels/PROmodelSimple.h"
#include "PROMCMC.h"
#include "PROtocall.h"
#include "PROseed.h"
#include "PROversion.h"
#include "PROplot.h"
#include "PROjector.h"
#include "PRObench.h"
#include "PROletariat.h"

#include "CLI11.h"
#include "LBFGSB.h"

#include <Eigen/Eigen>

#include <Eigen/src/Core/Matrix.h>
#include <LBFGSpp/Param.h>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <future>
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
#include <set>

#include <glob.h>
#include <vector>
#include <chrono>
#include "TMath.h"
#include "TLegend.h"

using namespace PROfit;

struct GlobalFitResult {
    PROfitter fitter;
    std::optional<Metropolis<simple_target, adaptive_proposal>> mh;

    std::vector<TH1D> priors, posteriors;
    Eigen::VectorXf prior_param_lo, prior_param_hi, post_param_lo, post_param_hi;
    Eigen::MatrixXf covmat, fraccovmat, corrmat, prior_covariance, spline_covariance;

    std::optional<PROerrorbar> err_band, post_err_band;

    float chi2;

    GlobalFitResult(const Eigen::VectorXf &ub, const Eigen::VectorXf &lb, const PROfitterConfig &config)
        : fitter(ub, lb, config) {}
};
enum struct GlobalFitOptions {
    Default             = 0,
    Progress            = 1 << 0,
    FreqSeedPts         = 1 << 1,
    Correlations        = 1 << 2,
    PrefitErrorBand     = 1 << 3,
    MCMCPrefitErrorBand = 1 << 4,
    PostFitErrorBand    = 1 << 5,
    BinWidthScaled      = 1 << 6,
};
inline GlobalFitOptions operator|(GlobalFitOptions lhs, GlobalFitOptions rhs) { 
    return static_cast<GlobalFitOptions>(static_cast<int>(lhs) | static_cast<int>(rhs)); 
}
inline GlobalFitOptions operator|=(GlobalFitOptions &lhs, GlobalFitOptions rhs) {
    lhs = lhs | rhs;
    return lhs;
}
inline GlobalFitOptions operator&(GlobalFitOptions lhs, GlobalFitOptions rhs) { 
    return static_cast<GlobalFitOptions>(static_cast<int>(lhs) & static_cast<int>(rhs)); 
}
inline GlobalFitOptions operator&=(GlobalFitOptions &lhs, GlobalFitOptions rhs) {
    lhs = lhs & rhs;
    return lhs;
}

GlobalFitResult run_global_fit(const PROconfig &config, const PROpeller &prop, const PROdata &data, PROmetric &metric, const Eigen::VectorXf &ub, const Eigen::VectorXf &lb, const PROfitterConfig &fit_config, const Eigen::VectorXf &CVParams, const PROspec &cv, const std::vector<int> &global_fixed, GlobalFitOptions opt);
std::map<std::string, TObject *> draw_fit_result(const PROconfig &config, const PROpeller &prop, const PROmodel &model, const PROsyst &syst, PROmetric &metric, const PROspec &cv, const PROdata &data, const GlobalFitResult &fitres, const std::string &prefix, PlotOptions popt, PlotBounds pbounds, bool plot_channel_ratios = false);
void draw_harmonic_scan_pdf(const GlobalFitResult &fitres, const PROfitterConfig &fit_config, const PROmodel &model, const std::string &filename);
std::vector<FixedSeed> buildBkgOnlyFixedSeeds(const PROmodel &model, const Eigen::VectorXf &CVParams);

// DetVar helpers (defined in PROfit_process.cxx, also used by run_plot).
std::string DetVarKey(const PROconfig& config, size_t file_index);
std::vector<int> DetVarMatchingKey(const PROpeller& prop, size_t i_event);
bool BuildDetVarMatchedSpecs(
        const PROpeller& cvprop, const std::map<int, const PROpeller*> &varprop,
        int var_idx, int spec_size,
        PROspec& out_cv, std::map<int, PROspec> &out_var);

struct PROpt {
    std::string xmlname = "NULL.xml"; 
    std::string data_xml = "";
    std::string analysis_tag = "PROfit";
    std::string output_tag = "v1";
    std::string chi2 = "PROchi";
    bool show_fit_help = false;
    bool eventbyevent=false;
    bool shapeonly = false;
    bool rateonly = false;
    int fit_variable = -1; // -1 => take the fitting variable from the XML (fit="true", else var0)
    bool force = false;
    bool noxrootd = false;
    bool poisson_throw = false;
    bool pseudo_experiment = false;
    bool progress_bar = false;
    std::vector<std::string> scale_arg;
    std::map<std::string, float> scale_map;
    size_t nthread = 1;
    std::map<std::string, float> scan_fit_options;
    std::map<std::string, float> global_fit_options;
    size_t maxevents;
    int global_seed = -1;
    std::string log_file = "";
    std::vector<std::string> fit_preset = {"grad-good","grad-fast"};
    inline static const std::unordered_set<std::string> allowed_preset = {"good","fast","overkill","sensitivity","grad-fast","grad-good","grad-deep","grad-overkill"};
    bool with_splines = false, binwidth_scale = false, area_normalized = false, data_mc_ratio = false;
    std::map<std::string, float> fake_data_osc_params;
    std::map<std::string, float> cv_osc_params;
    std::map<std::string, float> injected_systs;
    std::map<std::string, float> cv_injected_systs;
    std::vector<std::string> fixed_params;
    std::vector<std::string> syst_list, systs_excluded;
    bool MCMC_prefit_errors = false;
    bool systs_only = false;
    PROjectorRunConfig projector_config; // Two-stage pre-fit / projected fit (PROjector).
    bool use_fake_data = false;
    bool use_probe = false;
    int n_probe_chunks = 1; // 1 = no chunking by default. Opt in via --probe-chunks N when physics is the wall-time bottleneck.
    bool profile_timing = false; // Toggles PROfile/PROfitter scan-mode timing instrumentation (latin/PSO/LBFGS phase breakdown + parallel efficiency).
    bool use_surface_amr = false; // Adaptive-mesh-refinement surface scan (PROsurf::FillSurfaceAMR / PROmesh).
    int amr_initial = 10;
    int amr_levels  = 3;
    float amr_delta = 0.5f;
    std::vector<float> amr_contour_levels;

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
    int n_brazil_throws = 1000;
    std::vector<std::string> brazil_throws;
    std::vector<float> procurve_points;

    std::string reweights_file;
    std::vector<std::string> mockreweights;
    std::vector<TH2D*> weighthists;

    std::vector<std::string> mcmc_vars;
    size_t mcmc_chains;

    std::map<std::string, float> bound_list;
    PlotBounds pbounds; 
    size_t nuniv;
    bool gof_pvalue = false;
    bool pvalue = false;

    // fc-adaptive (slice 1: Wilks prepass + meta-mesh + diagnostics).
    std::string afc_mode_str = "init-bank";
    int   afc_n_throws = 200;
    std::vector<int> afc_prepass_initial = {10, 10};
    int   afc_prepass_levels = 3;
    float afc_prepass_delta  = 0.05f;
    std::vector<float> afc_prepass_contour_levels = {2.30f, 5.99f};
    float afc_p_thresh = 0.05f;
    int   afc_baseline_level = 2;
    bool  afc_stat_only_throws = false;
    std::string afc_xvar = "sinsq2thmm", afc_yvar = "dmsq";
    float afc_xlo = 1e-4f, afc_xhi = 1.0f, afc_ylo = 1e-2f, afc_yhi = 1e2f;
    bool  afc_logx = true, afc_logy = true;
    std::vector<float> afc_cl_targets = {0.683f, 0.90f, 0.954f};
    float afc_wilson_eps = 0.05f;
    int   afc_n_pe_min = 50;
    int   afc_n_pe_max = 5000;
    int   afc_update_layer = 0;
    int   afc_only_layer   = -1;
    int   afc_n_brazil_throws = 100;
    std::string afc_flag;
    float afc_roi_band = 8.0f;
    std::vector<std::string> afc_merge_inputs;
    std::vector<float> afc_cleanup_quantiles = {0.025f, 0.975f};
    int afc_cleanup_halo = 1;
    bool hmc = false;

    std::string gradient_mode_str = "";
    bool plot_channel_ratios = false;
    std::string bkg_subtract_pattern = "";

    int    bench_N           = 1000;
    std::string bench_tests_str = "all";
    std::string bench_grad_presets = "";
    bool bench_throw_systs = false, bench_throw_phys = false;

    PROletariatOptions grid_opts;
    std::string grid_backend_str = "jobsub";

    bool grid_sl7 = false;

    std::string final_output_tag;

    CLI::App app{"PROfit: a PROfessional, PROductive fitting and oscillation framework. Together let's minimize PROfit!"}; 
    CLI::App *process_command, *surface_command, *proplot_command, *global_command, *profile_command, *profc_command, 
             *mcmc_command, *protest_command, *bench_command, *proletariat_command, *afc_command;
    CLI::Option *xlim_opt, *ylim_opt;


    PROpt(int argc, char **argv);
};

// Setup-chain helpers (defined in PROfit_setup.cxx, called from main).
PROdata construct_data(std::vector<PROdata> &variable_data, bool use_real_data, const PROconfig &config, const PROpeller &prop, const PROmodel &model, const std::vector<PROsyst> &variable_systs, const Eigen::VectorXf &fakedataparams, const PROpt &options);
Eigen::VectorXf make_fakedata_params(Eigen::VectorXf &fake_data_osc_param_vector, const PROconfig &config, const PROpt &options, const PROmodel &model, const PROsyst &systs);
void make_param_vectors(Eigen::VectorXf &fakeDataParams, Eigen::VectorXf &CVParams, const PROconfig &config, const PROpt &options, const PROmodel &model, const PROsyst &systs, const Eigen::VectorXf &fake_data_osc_param_vector);
void include_or_exclude_systs(std::vector<PROsyst> &variable_systs, const PROconfig &config, const PROpt &options);
void empty_bin_check(const PROconfig &config, const PROpt &options, const PROpeller &prop, const PROmodel &model, const PROsyst &systs, const PROdata data, bool use_real_data);
void set_global_bounds(Eigen::VectorXf &lb, Eigen::VectorXf &ub, std::vector<int> &fixed, const PROconfig &config, const PROpt &options, PROmodel &model, PROsyst &systs, const Eigen::VectorXf &CVParams);
void print_global_fit_results(float global_fit_chi2, const Eigen::VectorXf &global_fit_result, const PROconfig &config, const PROpt &options, const PROmetric &metric);

// Per-subcommand entry points (one PROfit_<name>.cxx each).
void run_process(PROpeller &prop, std::vector<std::vector<SystStruct>> &systsstructs, const PROconfig &config, PROpt &options);
void run_plot(const PROconfig &config, const PROpeller &prop, const PROmetric &metric, const PROmodel &model, const std::vector<PROsyst> &variable_systs, const Eigen::VectorXf &CVParams, const Eigen::VectorXf &fakeDataParams, const Eigen::VectorXf &fake_data_osc_param_vector, const std::vector<PROdata> &variable_data, const PROpt &options);
void run_global(float &global_fit_chi2, Eigen::VectorXf &global_fit_result, const PROconfig &config, const PROpeller &prop, PROmetric &metric, const PROdata &data, const Eigen::VectorXf CVParams, const Eigen::VectorXf &lb, const Eigen::VectorXf &ub, const std::vector<int> fixed, const PROfitterConfig &fitConfig, const PROpt &options);
void run_profile(float &global_fit_chi2, Eigen::VectorXf &global_fit_result, const PROconfig &config, const PROpeller &prop, PROmetric &metric, const PROdata &data, const Eigen::VectorXf &CVParams, const Eigen::VectorXf &fakedataparams, const Eigen::VectorXf &lb, const Eigen::VectorXf &ub, const std::vector<int> &fixed, const PROfitterConfig &fitConfig, PROfitterConfig &scanFitConfig, const PROpt &options, PROseed &myseed);
void run_surface(float &global_fit_chi2, Eigen::VectorXf &global_fit_result, const PROconfig &config, const PROpeller &prop, PROmetric &metric, const PROdata &data, const Eigen::VectorXf &CVParams, const Eigen::VectorXf &fakeDataParams, const Eigen::VectorXf &lb, const Eigen::VectorXf &ub, const std::vector<int> &fixed, PROfitterConfig &fitConfig, PROfitterConfig &scanFitConfig, PROpt &options, PROseed &myseed);
void run_fc(const PROconfig &config, const PROpeller &prop, PROmetric &metric, const PROdata &data, const Eigen::VectorXf &CVParams, const Eigen::VectorXf &fakeDataParams, const Eigen::VectorXf &lb, const Eigen::VectorXf &ub, const std::vector<int> &fixed, const PROfitterConfig &fitConfig, const PROfitterConfig &scanFitConfig, const PROpt &options, PROseed &myseed);
void run_afc(const PROconfig &config, const PROpeller &prop, PROmetric &metric, const PROdata &data, const Eigen::VectorXf &fakeDataParams, const PROfitterConfig &scanFitConfig, const PROpt &options, PROseed &myseed);
void run_mcmc(const PROconfig &config, PROmetric &metric, const Eigen::VectorXf &lb, const Eigen::VectorXf &ub, const PROfitterConfig fitConfig, const PROpt &options, PROseed &myseed);
void run_bench(const PROconfig &config, const PROpeller &prop, PROmetric &metric, const PROdata &data, const Eigen::VectorXf &fakeDataParams, const PROfitterConfig &fitConfig, const PROpt &options);
void run_test(const PROconfig &config, const PROpeller &prop, PROmetric &metric, const PROdata &data, const Eigen::VectorXf &CVParams);
int run_proletariat(PROpt &options);

#endif // PROFIT_BIN_PROFIT_COMMON_H
