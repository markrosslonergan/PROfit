#include "PROfc.h"

namespace PROfit {


    void fc_worker(fc_args args, MultiPROgressBar &progress) {
        log<LOG_INFO>(L"%1% || FC for point %2%") % __func__ % args.phy_params;
        std::mt19937 rng{args.seed};
        std::unique_ptr<PROmodel> model = get_model_from_string(args.config.m_model_tag, args.prop);
        std::unique_ptr<PROmodel> null_model = std::make_unique<NullModel>(args.prop);

        PROchi::EvalStrategy strat = args.binned ? PROchi::BinnedChi2 : PROchi::EventByEvent;
        Eigen::VectorXf throws = Eigen::VectorXf::Constant(model->nparams + args.systs.GetNSplines(), 0);
        for(size_t i = 0; i < model->nparams; ++i) throws(i) = args.phy_params(i);
        size_t nparams = model->nparams + args.systs.GetNSplines();
        Eigen::VectorXf lb_osc = Eigen::VectorXf::Constant(nparams, -3.0);
        Eigen::VectorXf ub_osc = Eigen::VectorXf::Constant(nparams, 3.0);
        Eigen::VectorXf lb = Eigen::VectorXf::Constant(nparams, -3.0);
        Eigen::VectorXf ub = Eigen::VectorXf::Constant(nparams, 3.0);
        size_t nphys = model->nparams;
        //set physics to correct values
        for(size_t j=0; j<nphys; j++){
            ub_osc(j) = model->ub(j);
            lb_osc(j) = model->lb(j); 
            ub(j) = args.phy_params(j);
            lb(j) = args.phy_params(j); 
        }
        //upper lower bounds for splines
        for(size_t j = nphys; j < nparams; ++j) {
            lb_osc(j) = args.systs.spline_lo[j-nphys];
            ub_osc(j) = args.systs.spline_hi[j-nphys];
            lb(j) = args.systs.spline_lo[j-nphys];
            ub(j) = args.systs.spline_hi[j-nphys];
        }
        std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());
        Eigen::VectorXf cached_seed_syst = Eigen::VectorXf::Zero(nparams);
        Eigen::VectorXf cached_seed_osc = Eigen::VectorXf::Zero(nparams);

        for(size_t u = 0; u < args.todo; ++u) {
            log<LOG_INFO>(L"%1% | Thread #%2% Throw #%3%") % __func__ % args.thread % u;
            std::normal_distribution<float> d;
            Eigen::VectorXf throwC = Eigen::VectorXf::Constant(args.config.m_num_bins_total_collapsed, 0);
            for(size_t i = 0; i < args.systs.GetNSplines(); i++) {
                do {
                    throws(i+nphys) = d(rng);
                } while(throws(i+nphys) < args.systs.spline_lo[i]
                        || throws(i+nphys) > args.systs.spline_hi[i]);
            }
            for(size_t i = 0; i < args.config.m_num_bins_total_collapsed; i++)
                throwC(i) = d(rng);
            PROspec shifted = FillRecoSpectra(args.config, args.prop, args.systs, *model, throws, strat);
            log<LOG_DEBUG>(L"%1% || Shifted spectrum %2%\nfor throw %3%")
                % __func__ % shifted.Spec() % throws;
            PROspec newSpec = PROspec::PoissonVariation(PROspec(CollapseMatrix(args.config, shifted.Spec()) + args.L * throwC, CollapseMatrix(args.config, shifted.Error())), dseed(rng));
            PROdata data(newSpec.Spec(), newSpec.Error());
            //Metric Time
            PROmetric *metric;
            if(args.chi2 == "PROchi") {
                metric = new PROchi("", args.config, args.prop, &args.systs, *model, data, !args.binned ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
            } else if(args.chi2 == "PROCNP") {
                metric = new PROCNP("", args.config, args.prop, &args.systs, *model, data, !args.binned ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
            } else if(args.chi2 == "Poisson") {
                metric = new PROpoisson("", args.config, args.prop, &args.systs, *model, data, !args.binned ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
            } else {
                log<LOG_ERROR>(L"%1% || Unrecognized chi2 function %2%") % __func__ % args.chi2.c_str();
                abort();
            }

    

            // No oscillations, fix the values at the test point, and fit nuisennce only
            PROfitter fitter(ub, lb, args.fitconfig, dseed(rng));
            metric->setBounds(lb,ub);
            float chi2_syst = fitter.Fit(*metric,cached_seed_syst);
            metric->freeParams();
            cached_seed_syst = fitter.best_fit;

            // With oscillations, aka global best fit over nuisence and osc param
            PROfitter fitter_osc(ub_osc, lb_osc, args.fitconfig, dseed(rng));
            float chi2_osc = -999;
            if(!args.gof_mode){
                std::vector<Eigen::VectorXf> seed_points = {cached_seed_syst, cached_seed_osc};
                metric->setBounds(lb_osc,ub_osc);
                chi2_osc = fitter_osc.Fit(*metric, seed_points); 
                int nminima = fitter_osc.calcFreqSeedPoints(*metric);
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
                enhanced_config.n_multistart *= 2;
                enhanced_config.n_swarm_particles = std::max(10, enhanced_config.n_swarm_particles * 2);
                enhanced_config.n_localfit = std::max(5, enhanced_config.n_localfit + 2);

                PROfitter fitter_osc_enhanced(ub_osc, lb_osc, enhanced_config, dseed(rng));
                float chi2_osc_enhanced = fitter_osc_enhanced.Fit(*metric, fitter.best_fit);
                
                if(chi2_osc_enhanced < chi2_osc) {
                    log<LOG_DEBUG>(L"%1% || Enhanced search improved chi2_osc from %2% to %3%") 
                        % __func__ % chi2_osc % chi2_osc_enhanced;
                    chi2_osc = chi2_osc_enhanced;
                    fitter_osc.best_fit = fitter_osc_enhanced.best_fit;
                }


        } // end of cross-checks
        cached_seed_osc = fitter_osc.best_fit;    

        Eigen::VectorXf t = Eigen::VectorXf::Map(throws.data(), throws.size());

        args.out->push_back({
                chi2_syst, chi2_osc, 
                args.gof_mode ? 0 : std::pow(10.0f, fitter_osc.best_fit(0)), 
                args.gof_mode ? 0 : std::pow(10.0f, fitter_osc.best_fit(1)), 
                fitter.best_fit.segment(2, nparams - 2) , 
                args.gof_mode ? Eigen::VectorXf() : fitter_osc.best_fit.segment(2, nparams-2) , t
                });

        progress.increment_bar(args.thread);
        args.dchi2s->push_back( args.gof_mode ? chi2_syst : chi2_syst - chi2_osc);
        delete metric;
    }
};

}
