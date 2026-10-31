#include "PROfc.h"

namespace PROfit {

void fc_worker(fc_args args, MultiPROgressBar &progress) {
    log<LOG_INFO>(L"%1% || FC for point %2%") % __func__ % args.phy_params;
    std::mt19937 rng{args.seed};
    std::unique_ptr<PROmodel> model = get_model_from_string(args.config, args.prop);
    // The fresh model carries the XML bounds; restore the --fix / --syst-only
    // pins, since the harmonic scan reads its physics ranges from the model.
    model->lb = args.phys_lb;
    model->ub = args.phys_ub;

    Eigen::VectorXf throws = Eigen::VectorXf::Constant(model->nparams + args.systs.GetNSplines(), 0);
    for(size_t i = 0; i < model->nparams; ++i) throws(i) = args.phy_params(i);
    size_t nparams = model->nparams + args.systs.GetNSplines();
    Eigen::VectorXf lb_osc = Eigen::VectorXf::Constant(nparams, -3.0);
    Eigen::VectorXf ub_osc = Eigen::VectorXf::Constant(nparams, 3.0);
    Eigen::VectorXf lb = Eigen::VectorXf::Constant(nparams, -3.0);
    Eigen::VectorXf ub = Eigen::VectorXf::Constant(nparams, 3.0);
    size_t nphys = model->nparams;
    const bool profiled = !args.free_mask.empty();
    //set physics to correct values: pinned at the tested point, except profiled
    //parameters, which float over their full range in the constrained fit too
    for(size_t j=0; j<nphys; j++){
        ub_osc(j) = args.phys_ub(j);
        lb_osc(j) = args.phys_lb(j);
        const bool free_j = profiled && args.free_mask[j];
        ub(j) = free_j ? args.phys_ub(j) : args.phy_params(j);
        lb(j) = free_j ? args.phys_lb(j) : args.phy_params(j);
    }
    //upper lower bounds for splines
    for(size_t j = nphys; j < nparams; ++j) {
        size_t si = j - nphys;
        float lo = args.systs.spline_has_restrict[si] ? args.systs.spline_restrict_lo[si] : args.systs.spline_lo[si];
        float hi = args.systs.spline_has_restrict[si] ? args.systs.spline_restrict_hi[si] : args.systs.spline_hi[si];
        lb_osc(j) = lo;
        ub_osc(j) = hi;
        lb(j) = lo;
        ub(j) = hi;
    }
    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());
    Eigen::VectorXf cached_seed_syst = Eigen::VectorXf::Zero(nparams);
    Eigen::VectorXf cached_seed_osc = Eigen::VectorXf::Zero(nparams);

    // Profile mode: the constrained fit has free physics, so seed it where the
    // data fit would look (truth physics, splines at their prior centres), in
    // every basin of each profiled range (ProfileLadderSeeds, per toy), and
    // always at the profiled parameters' default (e.g. no mixing in that
    // channel), whose log10 floor a Latin hypercube never reaches.
    Eigen::VectorXf truth_seed = Eigen::VectorXf::Zero(nparams);
    std::vector<FixedSeed> profile_fixed_seeds;
    if(profiled) {
        for(size_t j = 0; j < nphys; ++j) truth_seed(j) = args.phy_params(j);
        for(size_t i = 0; i < args.systs.GetNSplines() && i < (size_t)args.systs.spline_centers.size(); ++i)
            truth_seed(nphys + i) = args.systs.spline_centers(i);
        FixedSeed at_default{truth_seed, std::vector<int>(nparams, 0)};
        for(size_t j = 0; j < nphys; ++j) {
            if(args.free_mask[j]) at_default.point(j) = model->default_val(j);
            at_default.fixed[j] = 1;
        }
        profile_fixed_seeds.push_back(std::move(at_default));
    }

    for(size_t u = 0; u < args.todo; ++u) {
        log<LOG_INFO>(L"%1% | Thread #%2% Throw #%3%") % __func__ % args.thread % u;
        std::normal_distribution<float> d;
        // Bounded, OOB-safe truncated-Gaussian throws (shared helper; the old
        // do/while here could spin forever on unreachable restrict bounds).
        // Unthrown splines sit at their prior centre (XML center= / PROjector θ̂), not 0.
        for(size_t i = 0; i < args.systs.GetNSplines(); i++) {
            throws(i+nphys) = args.throw_systematics
                ? ThrowRestrictedSplinePull(args.systs, i, rng, d)
                : (i < (size_t)args.systs.spline_centers.size() ? args.systs.spline_centers(i) : 0.0f);
        }
        // Fill the i_prime variable explicitly: the previous call passed an
        // EvalStrategy enum where FillSpectra takes `bool binned` and let
        // var_index default to 0, while the CollapseMatrix below collapses
        // with the i_prime matrix — wrong-variable physics when i_prime != 0.
        PROspec shifted = FillSpectra(args.config, args.prop, args.systs, *model, throws, args.binned, args.config.i_prime);
        log<LOG_DEBUG>(L"%1% || Shifted spectrum %2%\nfor throw %3%")
            % __func__ % shifted.Spec() % throws;
        // Covariance fluctuation scaled by this throw's own prediction, so the toy
        // follows the covariance the metric assumes at the truth.
        Eigen::VectorXf collapsed = CollapseMatrix(args.config, shifted.Spec());
        if(args.throw_systematics)
            collapsed += ThrowCovarianceShift(args.config, args.R, shifted.Spec(), rng, d, args.config.i_prime);
        PROspec variedSpec(collapsed, CollapseMatrix(args.config, shifted.Error()));

        // Without Poisson, clamp negative bins to 0 as PoissonVariation does.
        PROspec newSpec = args.throw_poisson
            ? PROspec::PoissonVariation(variedSpec, dseed(rng))
            : PROspec(variedSpec.Spec().cwiseMax(0.0f), variedSpec.Error());
        PROdata data(newSpec.Spec(), newSpec.Error());
        // The toy re-measures any physics prior about its own truth (systematic-like: held
        // at the truth when systematics are not thrown).
        data.physics_prior_centers = ToyPhysicsPriorCenters(*model, args.phy_params, args.throw_systematics, rng);
        if(!data.physics_prior_centers.empty())
            log<LOG_DEBUG>(L"%1% || Throw #%2% physics-prior centres: %3%") % __func__ % u % data.physics_prior_centers;
        //Metric Time
        // Same construction point as the data fit (carries shape_only etc.).
        PROmetric *metric = MakeMetric(args.chi2, args.config, args.prop, &args.systs, *model, data,
                                       !args.binned ? PROmetric::EventByEvent : PROmetric::BinnedChi2,
                                       args.shape_only).release();
        if(!metric) abort();



        // No oscillations, fix the values at the test point, and fit nuisennce only
        PROfitter fitter(ub, lb, args.fitconfig, dseed(rng));
        metric->setBounds(lb,ub);
        auto t0 = std::chrono::steady_clock::now();
        // Profile mode: this toy's own basins of the profiled parameters.
        std::vector<Eigen::VectorXf> cseeds;
        if(profiled) {
            if(u > 0) cseeds.push_back(cached_seed_syst);
            cseeds.push_back(truth_seed);
            for(auto &x : ProfileLadderSeeds(*metric, truth_seed, lb, ub, args.free_mask)) cseeds.push_back(std::move(x));
        }
        float chi2_syst = profiled
            ? fitter.Fit(*metric, cseeds, profile_fixed_seeds)
            : fitter.Fit(*metric,cached_seed_syst);
        args.timing->constrained_us += FitTiming::since_us(t0);
        metric->freeParams();
        cached_seed_syst = fitter.best_fit;

        // With oscillations, aka global best fit over nuisence and osc param
        PROfitter fitter_osc(ub_osc, lb_osc, args.fitconfig, dseed(rng));
        float chi2_osc = -999;
        if(!args.gof_mode){
            std::vector<Eigen::VectorXf> seed_points = {cached_seed_syst, cached_seed_osc};
            metric->setBounds(lb_osc,ub_osc);
            t0 = std::chrono::steady_clock::now();
            chi2_osc = fitter_osc.Fit(*metric, seed_points); 
            args.timing->free_us += FitTiming::since_us(t0);
            t0 = std::chrono::steady_clock::now();
            fitter_osc.calcFreqSeedPoints(*metric);
            args.timing->harmonic_us += FitTiming::since_us(t0);
            for(size_t i=0; i< fitter_osc.freq_seed_points.size(); i++){
                float chi_freq = fitter_osc.freq_seed_values.at(i);
                if(chi_freq<chi2_osc){
                    chi2_osc = chi_freq;
                    fitter_osc.best_fit = fitter_osc.freq_seed_points.at(i);
                }
            }
        }

        if(chi2_syst < chi2_osc && !args.gof_mode) {
            log<LOG_DEBUG>(L"%1% || Negative delta chi2 detected! chi2_syst=%2% < chi2_osc=%3%") % __func__ % chi2_syst % chi2_osc;
            log<LOG_DEBUG>(L"%1% || Attempting enhanced global search") % __func__;

            PROfitterConfig enhanced_config = args.fitconfig;
            enhanced_config.n_latin_points *= 2;
            enhanced_config.n_swarm_particles = std::max(10, enhanced_config.n_swarm_particles * 2);
            enhanced_config.n_localfit = std::max(5, enhanced_config.n_localfit + 2);

            PROfitter fitter_osc_enhanced(ub_osc, lb_osc, enhanced_config, dseed(rng));
            t0 = std::chrono::steady_clock::now();
            float chi2_osc_enhanced = fitter_osc_enhanced.Fit(*metric, fitter.best_fit);
            args.timing->rescue_us += FitTiming::since_us(t0);
            args.timing->n_rescue++;
            
            if(chi2_osc_enhanced < chi2_osc) {
                args.timing->n_rescue_improved++;
                log<LOG_DEBUG>(L"%1% || Enhanced search improved chi2_osc from %2% to %3%") 
                    % __func__ % chi2_osc % chi2_osc_enhanced;
                chi2_osc = chi2_osc_enhanced;
                fitter_osc.best_fit = fitter_osc_enhanced.best_fit;
            }


    } // end of cross-checks
    cached_seed_osc = fitter_osc.best_fit;    

    // Splines only, indexed like spline_names (the writer and the --reuse reader both assume it).
    Eigen::VectorXf t = throws.tail(nparams - nphys);

    // Evaluate all Eigen expressions to concrete VectorXf before constructing fc_out.
    Eigen::VectorXf best_phys  = args.gof_mode ? Eigen::VectorXf::Zero(nphys)
                                                : Eigen::VectorXf(fitter_osc.best_fit.head(nphys));
    Eigen::VectorXf best_syst  = Eigen::VectorXf(fitter.best_fit.segment(nphys, nparams - nphys));
    Eigen::VectorXf best_osc   = args.gof_mode ? Eigen::VectorXf()
                                                : Eigen::VectorXf(fitter_osc.best_fit.segment(nphys, nparams - nphys));
    Eigen::VectorXf best_phys_syst = Eigen::VectorXf(fitter.best_fit.head(nphys));
    args.out->push_back({chi2_syst, chi2_osc, best_phys, best_syst, best_osc, t, best_phys_syst});

    args.timing->n++;
    progress.increment_bar(args.thread);
    args.dchi2s->push_back( args.gof_mode ? chi2_syst : chi2_syst - chi2_osc);
    delete metric;
}
};

}
