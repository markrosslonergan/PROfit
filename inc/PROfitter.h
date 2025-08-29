#ifndef PROFITTER_H
#define PROFITTER_H

#include "PROmetric.h"

#include <Eigen/Eigen>
#include "LBFGSB.h"

namespace PROfit {

    struct PROfitterConfig {
        LBFGSpp::LBFGSBParam<float> param;
        int n_multistart = 1500, n_swarm_particles = 1, n_swarm_iterations=1, n_localfit=3;
        size_t n_max_local_retries = 3;
        size_t MCMCiter = 20'000;
        size_t MCMCburn = 25'000;

        PROfitterConfig(){};
        PROfitterConfig(std::map<std::string, float> input_fit_options, std::string fit_preset, bool isScan){

            if(!isScan){
                //Global Big presets
                if(fit_preset == "good"){
                    param.epsilon = 1e-6;
                    param.max_iterations = 10'000;
                    param.max_linesearch = 400;
                    param.delta = 1e-6;
                    n_multistart = 3000;
                    n_swarm_particles = 45;
                    n_swarm_iterations = 250;
                    n_localfit=3;
                    n_max_local_retries = 4;
                    param.wolfe = 0.99;
                    param.ftol = 1e-8;
                }else if (fit_preset == "fast"){
                    param.epsilon = 1e-6;
                    param.max_iterations = 100;
                    param.max_linesearch = 250;
                    param.delta = 1e-6;
                    n_multistart = 1250;
                    n_swarm_particles = 25;
                    n_swarm_iterations = 100;
                    n_localfit=2;
                    n_max_local_retries = 1;
                }else if(fit_preset == "overkill"){
                    param.epsilon = 1e-6;
                    param.max_iterations = 100'000;
                    param.max_linesearch = 1000;
                    param.delta = 1e-6;
                    n_multistart = 3000;
                    n_swarm_particles = 100;
                    n_swarm_iterations = 250;
                    n_localfit=4;
                    n_max_local_retries = 8;
                    param.wolfe = 0.99;
                    param.ftol = 1e-8;
                }

            }else{

                // the lesser Scan version
                if(fit_preset == "good"){
                    param.epsilon = 1e-6;
                    param.max_iterations = 10'000;
                    param.max_linesearch = 250;
                    param.delta = 1e-6;
                    n_multistart = 1500;
                    n_swarm_particles = 5;
                    n_swarm_iterations = 100;
                    n_localfit=2;
                    n_max_local_retries = 3;
                    param.wolfe = 0.99;
                    param.ftol = 1e-8;
                }else if (fit_preset == "fast"){
                    param.epsilon = 1e-6;
                    param.max_iterations = 100;
                    param.max_linesearch = 200;
                    param.delta = 1e-6;
                    n_multistart = 1000;
                    n_swarm_particles = 1;
                    n_swarm_iterations = 100;
                    n_localfit=2;
                    n_max_local_retries = 1;
                }else if(fit_preset == "overkill"){
                    param.epsilon = 1e-6;
                    param.max_iterations = 100'000;
                    param.max_linesearch = 500;
                    param.delta = 1e-6;
                    n_multistart = 2500;
                    n_swarm_particles = 30;
                    n_swarm_iterations = 250;
                    n_localfit=4;
                    n_max_local_retries = 7;
                    param.wolfe = 0.99;
                    param.ftol = 1e-8;
                }


            }


            std::string whichFit = ( isScan? "Simplier Scan" : "Detailed Global");
            log<LOG_INFO>(L"%1% ||Fit and  L-BFGS-B parameters for the %2% minimia finder.  ") % __func__ % whichFit.c_str();
            for(const auto &[param_name, value]: input_fit_options) {
                log<LOG_WARNING>(L"%1% || L-BFGS-B %2% set to %3% ") % __func__% param_name.c_str() % value ;
                if(param_name == "epsilon") {
                    param.epsilon = value;
                } else if(param_name == "delta") {
                    param.delta = value;
                } else if(param_name == "m") {
                    param.m = value;
                    if(value < 3) {
                        log<LOG_WARNING>(L"%1% || Number of corrections to approximate inverse Hessian in"
                                L" L-BFGS-B is recommended to be at least 3, provided value is %2%."
                                L" Note: this is controlled via --fit-options m.")
                            % __func__ % value;
                    }
                } else if(param_name == "epsilon_rel") {
                    param.epsilon_rel = value;
                } else if(param_name == "past") {
                    param.past = value;
                    if(value == 0) {
                        log<LOG_WARNING>(L"%1% || L-BFGS-B 'past' parameter set to 0. This will disable delta convergence test")
                            % __func__;
                    }
                } else if(param_name == "max_iterations") {
                    param.max_iterations = value;
                } else if(param_name == "max_submin") {
                    param.max_submin = value;
                } else if(param_name == "max_linesearch") {
                    param.max_linesearch = value;
                } else if(param_name == "min_step") {
                    param.min_step = value;
                    log<LOG_WARNING>(L"%1% || Modifying the minimum step size in the line search to be %2%."
                            L" This is not usually needed according to the LBFGSpp documentation.")
                        % __func__ % value;
                } else if(param_name == "max_step") {
                    param.max_step = value;
                    log<LOG_WARNING>(L"%1% || Modifying the maximum step size in the line search to be %2%."
                            L" This is not usually needed according to the LBFGSpp documentation.")
                        % __func__ % value;
                } else if(param_name == "ftol") {
                    param.ftol = value;
                } else if(param_name == "wolfe") {
                    param.wolfe = value;
                } else if(param_name == "n_multistart") {
                    n_multistart = value;
                    if(n_multistart < 1) {
                        log<LOG_ERROR>(L"%1% || Expected to run at least 1 multistart point. Provided value is %2%.")
                            % __func__ % value;
                        exit(EXIT_FAILURE);
                    }
                } else if(param_name == "n_localfit") {
                    n_localfit = value;
                    if(n_localfit < 1) {
                        log<LOG_ERROR>(L"%1% || Expected to run at least 1 local fit point. Provided value is %2%.")
                            % __func__ % value;
                        exit(EXIT_FAILURE);
                    }
                }else if(param_name == "n_swarm_particles") {
                    n_swarm_particles = value;
                    if(n_swarm_particles < 1) {
                        log<LOG_ERROR>(L"%1% || Expected to run at least 1 PSO swarm particle point. Provided value is %2%.")
                            % __func__ % value;
                        exit(EXIT_FAILURE);
                    }
                }else if(param_name == "n_swarm_iterations") {
                    n_swarm_iterations = value;
                    if(n_swarm_iterations < 1) {
                        log<LOG_ERROR>(L"%1% || Expected to run at least 1 swarm_iterations point. Provided value is %2%.")
                            % __func__ % value;
                        exit(EXIT_FAILURE);
                    }
                }else if(param_name == "MCMC-Burnin") {
                    MCMCburn = value;
                    if(MCMCburn < 1) {
                        log<LOG_WARNING>(L"%1% || Warning: Running without any burnin for MCMC.");
                    }
                }else if(param_name == "MCMC-Iterations") {
                    MCMCiter = value;
                    if(MCMCiter < 1) {
                        log<LOG_ERROR>(L"%1% || Requested to run MCMC with no iterations.");
                    }
                } else {
                    log<LOG_WARNING>(L"%1% || Unrecognized LBFGSB parameter %2%. Will ignore.") 
                        % __func__ % param_name.c_str();
                }
            }
            try {
                print();
            } catch(std::invalid_argument &except) {
                log<LOG_ERROR>(L"%1% || Invalid L-BFGS-B parameters: %2%") % __func__ % except.what();
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }

        }

        void print(){
            log<LOG_INFO>(L"%1% || Printing PROfitterConifg Values.") % __func__;
            log<LOG_INFO>(L"%1% || ------------ PROfitter specific -------------- ") % __func__ ;
            log<LOG_INFO>(L"%1% || n_multistart: %2% (default 1500) ") % __func__ % n_multistart;
            log<LOG_INFO>(L"%1% || n_swarm_particles: %2% (default 1) ") % __func__ % n_swarm_particles;
            log<LOG_INFO>(L"%1% || n_swarm_iterations: %2% (default 1) ") % __func__ % n_swarm_iterations;
            log<LOG_INFO>(L"%1% || n_localfit: %2% (default 3) ") % __func__ % n_localfit;
            log<LOG_INFO>(L"%1% || n_max_local_retries: %2% (default 3) ") % __func__ % n_max_local_retries;
            log<LOG_INFO>(L"%1% || ------------ LBFGSBParam -------------- ") % __func__ ;
            log<LOG_INFO>(L"%1% || m: %2%  (default %3%) ") % __func__ % param.m % 6;
            log<LOG_INFO>(L"%1% || epsilon: %2%  (default %3%) ") % __func__ % param.epsilon % 1e-5;
            log<LOG_INFO>(L"%1% || epsilon_rel: %2%  (default %3%) ") % __func__ % param.epsilon_rel % 1e-5;
            log<LOG_INFO>(L"%1% || past: %2%  (default %3%) ") % __func__ % param.past % 1;
            log<LOG_INFO>(L"%1% || delta: %2%  (default %3%) ") % __func__ % param.delta % 1e-10;
            log<LOG_INFO>(L"%1% || max_iterations: %2%  (default %3%) ") % __func__ % param.max_iterations % 0;
            log<LOG_INFO>(L"%1% || max_submin: %2%  (default %3%) ") % __func__ % param.max_submin % 20;
            log<LOG_INFO>(L"%1% || max_linesearch: %2%  (default %3%) ") % __func__ % param.max_linesearch % 20;
            log<LOG_INFO>(L"%1% || min_step: %2%  (default %3%) ") % __func__ % param.min_step % 1e-20;
            log<LOG_INFO>(L"%1% || max_step: %2%  (default %3%) ") % __func__ % param.max_step % 1e20;
            log<LOG_INFO>(L"%1% || ftol: %2%  (default %3%) ") % __func__ % param.ftol % 1e-4;
            log<LOG_INFO>(L"%1% || wolfe: %2%  (default %3%) ") % __func__ % param.wolfe % 0.9;
            log<LOG_INFO>(L"%1% || ------------ Check Param LBFGSBParam Below -------------- ") % __func__ ;
            param.check_param();

        }
    };

    class PROfitter {
        public:
            Eigen::VectorXf ub, lb, best_fit;
            PROfitterConfig fitconfig;
            LBFGSpp::LBFGSBSolver<float> solver;
            uint32_t seed;

            PROfitter(const Eigen::VectorXf ub, const Eigen::VectorXf lb, PROfitterConfig fitconfig_ = {}, uint32_t inseed = 0)
                : ub(ub), lb(lb), fitconfig(fitconfig_), solver(fitconfig.param), seed(inseed) {}

            float Fit(PROmetric &metric, const Eigen::VectorXf &seed_pt = Eigen::VectorXf());

            Eigen::VectorXf FinalGradient() const {return solver.final_grad();}
            float FinalGradientNorm() const {return solver.final_grad_norm();}
            Eigen::MatrixXf Hessian() const {return solver.final_approx_hessian();}
            Eigen::MatrixXf InverseHessian() const {return solver.final_approx_inverse_hessian();}
            Eigen::MatrixXf Covariance() const {return InverseHessian();}
            Eigen::VectorXf BestFit() const {return best_fit;}

            // If you don't belive the uncertainties on the parameters, you can use the final fit value to estimate the variance
            Eigen::MatrixXf ScaledCovariance(float chi2, int n_datapoint) const {return Covariance()*chi2/float(n_datapoint-best_fit.size());}

    };

}

#endif

