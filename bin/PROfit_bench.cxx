#include "PROfit_common.h"

void run_bench(const PROconfig &config, const PROpeller &prop, PROmetric &metric, const PROdata &data, const Eigen::VectorXf &fakeDataParams, const PROfitterConfig &fitConfig, const PROpt &options) {
    // Parse the comma-separated --tests selector. "all"/"fillspectra"/
    // "metric"/"fit" expand to the corresponding group; individual letters
    // a–i map to single tests.
    unsigned bench_mask = 0u;
    auto match_token = [&bench_mask](const std::string &tok) {
        using namespace PROfit::PRObench;
        if      (tok == "all")         bench_mask |= Bench_All;
        else if (tok == "fillspectra") bench_mask |= Bench_FillSpectra_Group;
        else if (tok == "metric")      bench_mask |= Bench_Metric_Group;
        else if (tok == "metricgrad")  bench_mask |= Bench_MetricGrad_Group;
        else if (tok == "fit")         bench_mask |= Bench_Fit_Group;
        else if (tok == "pseudo")      bench_mask |= Bench_PseudoUniverse;
        else if (tok == "collapse")    bench_mask |= Bench_Collapse;
        else if (tok == "mcmc")        bench_mask |= Bench_MCMC_Group;
        else if (tok == "a")           bench_mask |= Bench_FillSpectra_All;
        else if (tok == "b")           bench_mask |= Bench_FillSpectra_Phys;
        else if (tok == "c")           bench_mask |= Bench_FillSpectra_Nuis;
        else if (tok == "d")           bench_mask |= Bench_Metric_All;
        else if (tok == "e")           bench_mask |= Bench_Metric_Phys;
        else if (tok == "f")           bench_mask |= Bench_Metric_Nuis;
        else if (tok == "g")           bench_mask |= Bench_Fit;
        else if (tok == "h")           bench_mask |= Bench_PseudoUniverse;
        else if (tok == "i")           bench_mask |= Bench_Collapse;
        else if (tok == "j")           bench_mask |= Bench_MetricGrad_All;
        else if (tok == "k")           bench_mask |= Bench_MetricGrad_Phys;
        else if (tok == "l")           bench_mask |= Bench_MetricGrad_Nuis;
        else if (tok == "m")           bench_mask |= Bench_MCMC_Burnin;
        else if (tok == "n")           bench_mask |= Bench_MCMC_Post;
        else if (tok == "grad")        bench_mask |= Bench_Grad_Group;
        else if (tok == "gradcheck" || tok == "o") bench_mask |= Bench_GradCheck;
        else if (tok == "gradmodes" || tok == "gradfit" || tok == "p") bench_mask |= Bench_GradModeFit;
        else log<LOG_WARNING>(L"%1% || scale-test: unknown --tests token '%2%' (ignored)") % __func__ % tok.c_str();
    };
    const std::string &s = options.bench_tests_str;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        std::string tok = s.substr(pos, (comma == std::string::npos ? s.size() : comma) - pos);
        while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.front()))) tok.erase(tok.begin());
        while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.back())))  tok.pop_back();
        if (!tok.empty()) match_token(tok);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    if (bench_mask == 0u) bench_mask = PROfit::PRObench::Bench_All;

    PROfit::PRObench::BenchOptions bopts;
    bopts.N        = options.bench_N;
    bopts.tests    = bench_mask;
    bopts.binned   = !options.eventbyevent;
    bopts.nthreads = (int)options.nthread;
    bopts.truth_params = fakeDataParams;
    std::string bench_suffix;
    if (options.bench_throw_phys)       bench_suffix = options.bench_throw_systs ? "_systphys" : "_phys";
    else if (options.bench_throw_systs) bench_suffix = "_syst";
    bopts.grad_csv = options.analysis_tag + "_gradbench_fits" + bench_suffix + ".csv";
    bopts.grad_presets = options.bench_grad_presets;
    bopts.throw_systs = options.bench_throw_systs;
    bopts.throw_phys  = options.bench_throw_phys;

    PROfit::PRObench::run_scale_test(config, prop, metric, data, fitConfig, bopts);
}
