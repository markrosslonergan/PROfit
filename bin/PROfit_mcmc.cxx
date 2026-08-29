#include "PROfit_common.h"

void mcmc_worker(std::vector<std::unique_ptr<Metropolis<unilin_prior_target, adaptive_proposal>>> &mets, Eigen::VectorXf initial, PROmetric *metric, uint32_t seed, size_t nchains, size_t burnin, size_t steps);
//void mcmc_worker(std::vector<std::unique_ptr<Metropolis<simple_target, adaptive_proposal>>> &mets, Eigen::VectorXf initial, PROmetric *metric, uint32_t seed, size_t nchains, size_t burnin, size_t steps);
void nuts_worker(std::vector<std::vector<Eigen::VectorXf>> &chains, Eigen::VectorXf initial, PROmetric *metric, uint32_t seed, size_t nchains, size_t burnin, size_t steps, const Eigen::VectorXf &lb, const Eigen::VectorXf &ub);

void run_mcmc(const PROconfig &config, PROmetric &metric, const Eigen::VectorXf &lb, const Eigen::VectorXf &ub, const PROfitterConfig fitConfig, const PROpt &options, PROseed &myseed) {
    metric.setBounds(ub, lb);
    size_t nparams = metric.GetModel().nparams + metric.GetSysts().GetNSplines();
    std::uniform_real_distribution<float> latin_distribution(-2, 2);
    std::vector<std::vector<float>> samples = latin_hypercube_sampling(options.mcmc_chains, nparams, latin_distribution, myseed.global_rng);
    recenter_latin_samples(samples, ub, lb);
    std::vector<Eigen::VectorXf> samples_eigen; 
    for(size_t i = 0; i < samples.size(); ++i) {
        samples_eigen.push_back(Eigen::VectorXf::Map(samples[i].data(), samples[i].size()));
        //samples_eigen.back()(0) = std::pow(10, samples_eigen.back()(0));
        //samples_eigen.back()(1) = std::pow(10, samples_eigen.back()(1));
    }
    size_t mcmc_threads = options.mcmc_chains >= options.nthread ? options.nthread : options.mcmc_chains;
    std::vector<std::vector<std::unique_ptr<Metropolis<unilin_prior_target, adaptive_proposal>>>> mets;
    std::vector<std::vector<std::vector<Eigen::VectorXf>>> chains;
    //std::vector<std::vector<std::unique_ptr<Metropolis<simple_target, adaptive_proposal>>>> mets;
    mets.reserve(mcmc_threads);
    std::vector<std::thread> threads;
    size_t chains_per_thread = options.mcmc_chains / mcmc_threads;
    size_t addone = mcmc_threads - options.mcmc_chains%mcmc_threads;
    for(size_t i = 0; i < mcmc_threads; ++i) {
        if(options.hmc) {
            chains.emplace_back();
            threads.emplace_back(
                    [&, i](){
                        nuts_worker(chains[i], samples_eigen[i], metric.Clone(), myseed.getThreadSeeds()->at(i), chains_per_thread + (i >= addone), fitConfig.MCMCburn, fitConfig.MCMCiter, lb, ub);
                    });
        } else {
            mets.emplace_back();
            threads.emplace_back(
                    [&, i](){
                        mcmc_worker(mets[i], samples_eigen[i], metric.Clone(), myseed.getThreadSeeds()->at(i), chains_per_thread + (i >= addone), fitConfig.MCMCburn, fitConfig.MCMCiter);
                    });
        }
    }
    for(auto&& t : threads) {
        t.join();
    }

    //std::vector<TH2D> twod;
    //std::vector<TH1D> oned;
    // TODO: This is hardcoded for numu disappearance right now
    // TODO: How do we input binnings for these plots in a nice way?
    // TODO: Is it better to write out the chain to a cvs/root file and plot externally?
    //twod.push_back(TH2D("two", ";sin^{2}2#theta_{#mu#mu};#Deltam^{2}_{41} [eV^{2}];MCMC Points",
    //                     200, -3, 0, 200, -2, 2));
    //oned.push_back(TH1D("one1", ";sin^{2}2#theta_{#mu#mu};Posterior PDF", 200, -3, 0));
    //oned.push_back(TH1D("one2", ";#Deltam^{2}_{41} [eV^{2}];Posterior PDF", 200, -2, 2));
    TFile fout((options.final_output_tag+"_PROMCMC_chains.root").c_str(), "RECREATE");
    size_t chain_counter = 0;
    if(options.hmc) {
        for(const auto &tchain : chains) {
            for(const auto &chain : tchain) {
                chain_counter++;
                std::string name = "chain"+std::to_string(chain_counter);
                TTree tree(name.c_str(), name.c_str());
                Eigen::VectorXf v = Eigen::VectorXf::Zero(nparams);
                std::vector<std::string> param_names;
                for(size_t i = 0; i < metric.GetModel().nparams; ++i) {
                    tree.Branch(metric.GetModel().param_names[i].c_str(), &v(i));
                    param_names.push_back(metric.GetModel().pretty_param_names[i]);
                }
                for(size_t i = metric.GetModel().nparams; i < nparams; ++i) {
                    const std::string &sname = metric.GetSysts().spline_names[i-metric.GetModel().nparams];
                    std::string::size_type l = sname.find(':');
                    // TODO: This only handles names with a single colon in them. I don't think we ever have more than that, it's really just meant for the 'flat' and 'norm' systs.
                    if(l != std::string::npos) {
                        std::string bname = sname;
                        bname[l] = '_';
                        tree.Branch(bname.c_str(), &v(i));
                    } else {
                        tree.Branch(sname.c_str(), &v(i));
                    }
                    param_names.push_back(config.m_mcgen_variation_plotname_map.at(metric.GetSysts().spline_names[i-metric.GetModel().nparams]));
                }
                for(const auto &p : chain) {
                //    twod[0].Fill(p(1), p(0));
                //    oned[0].Fill(p(1));
                //    oned[1].Fill(p(0));
                    v = p;
                    tree.Fill();
                }
                tree.Write();
                //met->plot_autocorrelation((final_output_tag+"_PROMCMC_autocorrelation_chain"+std::to_string(chain_counter)+".pdf").c_str(), param_names);
            }
        }

    } else {
        for(const auto &tmets : mets) {
            for(const auto &met : tmets) {
                chain_counter++;
                std::string name = "chain"+std::to_string(chain_counter);
                TTree tree(name.c_str(), name.c_str());
                Eigen::VectorXf v = Eigen::VectorXf::Zero(nparams);
                std::vector<std::string> param_names;
                for(size_t i = 0; i < metric.GetModel().nparams; ++i) {
                    tree.Branch(metric.GetModel().param_names[i].c_str(), &v(i));
                    param_names.push_back(metric.GetModel().pretty_param_names[i]);
                }
                for(size_t i = metric.GetModel().nparams; i < nparams; ++i) {
                    const std::string &sname = metric.GetSysts().spline_names[i-metric.GetModel().nparams];
                    std::string::size_type l = sname.find(':');
                    // TODO: This only handles names with a single colon in them. I don't think we ever have more than that, it's really just meant for the 'flat' and 'norm' systs.
                    if(l != std::string::npos) {
                        std::string bname = sname;
                        bname[l] = '_';
                        tree.Branch(bname.c_str(), &v(i));
                    } else {
                        tree.Branch(sname.c_str(), &v(i));
                    }
                    param_names.push_back(config.m_mcgen_variation_plotname_map.at(metric.GetSysts().spline_names[i-metric.GetModel().nparams]));
                }
                for(const auto &p : met->chain) {
                //    twod[0].Fill(p(1), p(0));
                //    oned[0].Fill(p(1));
                //    oned[1].Fill(p(0));
                    v = p;
                    tree.Fill();
                }
                tree.Write();
                met->plot_autocorrelation((options.final_output_tag+"_PROMCMC_autocorrelation_chain"+std::to_string(chain_counter)+".pdf").c_str(), param_names, {});
            }
        }
    }
    //TCanvas c;
    //c.Divide(2,2);
    //c.cd(1);
    //oned[0].Scale(1.0/oned[0].Integral());
    //oned[0].Draw("hist");
    //c.cd(3);
    //twod[0].Draw("colz");
    //c.cd(4);
    //oned[1].Scale(1.0/oned[1].Integral());
    //oned[1].Draw("hist");
    //c.Print("mcmc_corner_plot.pdf");
}

void nuts_worker(std::vector<std::vector<Eigen::VectorXf>> &chains, Eigen::VectorXf initial, PROmetric *metric, uint32_t seed, size_t nchains, size_t burnin, size_t steps, const Eigen::VectorXf &lb, const Eigen::VectorXf &ub) {
    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());
    std::mt19937 rng(seed);
    metric->setBounds(lb, ub);
    for(size_t i = 0; i < nchains; ++i) {
        NUTS nuts;
        nuts.M = burnin+steps;
        nuts.Madapt = burnin;
        nuts(initial, *metric, dseed(rng));
        chains.emplace_back(std::move(nuts.chain));
    }
}

void mcmc_worker(std::vector<std::unique_ptr<Metropolis<unilin_prior_target, adaptive_proposal>>> &mets, Eigen::VectorXf initial, PROmetric *metric, uint32_t seed, size_t nchains, size_t burnin, size_t steps) {
//void mcmc_worker(std::vector<std::unique_ptr<Metropolis<simple_target, adaptive_proposal>>> &mets, Eigen::VectorXf initial, PROmetric *metric, uint32_t seed, size_t nchains, size_t burnin, size_t steps) {
    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());
    std::mt19937 rng(seed);
    mets.reserve(nchains);
    for(size_t i = 0; i < nchains; ++i) {
        //simple_target target{*metric};
        unilin_prior_target target{*metric};
        adaptive_proposal proposal(*metric, dseed(rng));
        auto met = std::make_unique<Metropolis<unilin_prior_target, adaptive_proposal>>(target, proposal, initial, dseed(rng));
        met->run(burnin, steps);
        mets.emplace_back(std::move(met));
    }
}
