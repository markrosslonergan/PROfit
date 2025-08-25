#include "PROfitter.h"
#include "PROlog.h"
#include "PROmetric.h"
#include "PROswarm.h"
#include <Eigen/Eigen>

#include <random>

using namespace PROfit;

static inline


std::vector<std::vector<float>> latin_hypercube_sampling(size_t num_samples, size_t dimensions, std::uniform_real_distribution<float>&dis, std::mt19937 &gen) {
    std::vector<std::vector<float>> samples(num_samples, std::vector<float>(dimensions));

    for (size_t d = 0; d < dimensions; ++d) {

        std::vector<float> perm(num_samples);
        for (size_t i = 0; i < num_samples; ++i) {
            perm[i] = (i + dis(gen)) / num_samples;  
        }
        std::shuffle(perm.begin(), perm.end(), gen);  
        for (size_t i = 0; i < num_samples; ++i) {
            samples[i][d] = perm[i]; 
        }
    }

    return samples;
}

static inline
std::vector<int> sorted_indices(const std::vector<float>& vec) {
    std::vector<int> indices(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&vec](int i1, int i2) { return vec[i1] < vec[i2]; });
    return indices;
}

float PROfitter::Fit(PROmetric &metric, const Eigen::VectorXf &seed_pt ) {
    const std::vector<Eigen::VectorXf> seed_points = seed_pt.size() > 0 ? std::vector<Eigen::VectorXf>{seed_pt} : std::vector<Eigen::VectorXf>{};
    return Fit(metric, seed_points);

}

float PROfitter::Fit(PROmetric &metric, const std::vector<Eigen::VectorXf> &seed_points ) {
    std::mt19937 rng;
    rng.seed(seed);
    std::normal_distribution<float> d;
    std::uniform_real_distribution<float> d_uni(-2.0, 2.0);

    //n_multistart is how many initial latin cube points
    std::vector<std::vector<float>> latin_samples = latin_hypercube_sampling(fitconfig.n_multistart, ub.size(), d_uni,rng);
    //Rescale the latin hypercube now at -2 to 2, scale to real bounds.
    for(std::vector<float> &pt: latin_samples) {
        for(size_t i = 0; i < pt.size(); ++i) {
            if(ub(i) != 3 || lb(i) != -3) {
                float width = std::isinf(ub(i)) || std::isinf(lb(i)) ? 4 : ub(i) - lb(i);
                float center = std::isinf(ub(i)) ? lb(i) + width/2.0 :
                    std::isinf(lb(i)) ? ub(i) - width/2.0 :
                    (ub(i) + lb(i)) / 2.0;
                float randpt = pt[i] / 4.0;
                pt[i] = center + randpt * width;
            }
        }
    }
    if(seed_points.size()>0 && seed_points.front().size()>0){
        log<LOG_INFO>(L"%1% || Seed point passed in. Being included.") % __func__  ;
        for(auto & pt: seed_points){
            std::vector<float> std_vec(pt.data(), pt.data() + pt.size());
            latin_samples.push_back(std_vec);
        }
    }else{
        log<LOG_INFO>(L"%1% || No seed point passed in. ") % __func__  ;
    }



    std::vector<float> chi2s_multistart;
    chi2s_multistart.reserve(fitconfig.n_multistart);

    log<LOG_INFO>(L"%1% || Starting MultiGlobal runs (i.e latin hypercube runs, pure chi^2 no grad) : %2%") % __func__ % fitconfig.n_multistart ;
    for(int s = 0; s < fitconfig.n_multistart; s++){
        Eigen::VectorXf x = Eigen::Map<Eigen::VectorXf>(latin_samples[s].data(), latin_samples[s].size());
        Eigen::VectorXf grad = Eigen::VectorXf::Constant(x.size(), 0);
        float fx =  metric(x, grad, false);
        chi2s_multistart.push_back(fx);
    }
    //Sort so we can take the best N_localfits for further zoning with a PSO
    std::vector<int> best_multistart = sorted_indices(chi2s_multistart);    

    log<LOG_INFO>(L"%1% || Best Point after latin hypercube has chi^2 %2% with pts  : %3% ") % __func__ % chi2s_multistart[best_multistart[0]] % latin_samples[best_multistart[0]];


    std::string swarm_string = "";
    std::vector<std::vector<float>> swarm_start_points;
    int niter=0;
    float fx;
    if(fitconfig.n_swarm_particles < 1){
        fitconfig.n_swarm_particles = 1;
    }

    for(int s = 0; s < fitconfig.n_swarm_particles; s++){
        swarm_string += " " + std::to_string(chi2s_multistart[best_multistart[s]]);
        swarm_start_points.push_back(latin_samples[best_multistart[s]]);
    }
    log<LOG_INFO>(L"%1% || Will swarm with %2% swarm points chis of %3% ") % __func__ % fitconfig.n_swarm_particles % swarm_string.c_str();

    PROswarm PSO(metric, rng, swarm_start_points, lb, ub , fitconfig.n_swarm_iterations);
    PSO.runSwarm(metric, rng);

    Eigen::VectorXf x;  

    float chimin = 9999999;
    std::vector<float> chi2s_localfits;
    niter=0;

    bool success = false;


    log<LOG_INFO>(L"%1% || Starting local fit of best swarm point. ") % __func__ ;

    for (size_t attempt = 1; attempt <= fitconfig.n_max_local_retries; ++attempt) {
        try {
            x = PSO.getGlobalBestPosition();
            log<LOG_INFO>(L"%1% || Starting local minimization attempt %2%/%3%") % __func__ % attempt % fitconfig.n_max_local_retries;



            //NEW NEW
            try {
                niter = solver.minimize(metric, x, fx, lb, ub);
            } catch(const std::runtime_error &e) {
                std::string error_msg = e.what();

                if(error_msg.find("line search step became smaller") != std::string::npos) {
                    log<LOG_ERROR>(L"%1% || Line search failure detected! Debugging info:") % __func__;

                    // Evaluate function and gradient at current point
                    Eigen::VectorXf grad(x.size());
                    float f_current = metric(x, grad, true);

                    log<LOG_ERROR>(L"%1% || Current position x: %2%") % __func__ % x;
                    log<LOG_ERROR>(L"%1% || Current f(x): %2%") % __func__ % f_current;
                    log<LOG_ERROR>(L"%1% || Gradient norm: %2%") % __func__ % grad.norm();
                    log<LOG_ERROR>(L"%1% || Gradient: %2%") % __func__ % grad;

                    // Check for NaN or Inf
                    if(!std::isfinite(f_current)) {
                        log<LOG_ERROR>(L"%1% || ERROR: Function value is NaN or Inf!") % __func__;
                    }
                    if(grad.hasNaN()) {
                        log<LOG_ERROR>(L"%1% || ERROR: Gradient contains NaN!") % __func__;
                    }
                    if((grad.array().abs() > 1e10).any()) {
                        log<LOG_ERROR>(L"%1% || WARNING: Gradient has extremely large values!") % __func__;
                    }

                    // Check proximity to bounds
                    log<LOG_ERROR>(L"%1% || Boundary proximity check:") % __func__;
                    for(int i = 0; i < x.size(); ++i) {
                        float dist_to_lower = x(i) - lb(i);
                        float dist_to_upper = ub(i) - x(i);

                        if(dist_to_lower < 1e-6 || dist_to_upper < 1e-6) {
                            log<LOG_ERROR>(L"%1% ||   Param[%2%] near boundary: x=%3%, bounds=[%4%, %5%], distances=[%6%, %7%]")
                                % __func__ % i % x(i) % lb(i) % ub(i) % dist_to_lower % dist_to_upper;
                        }
                    }

                                        // Check condition number (gradient vs parameter scales)
                    float max_grad = grad.cwiseAbs().maxCoeff();
                    float min_grad = grad.cwiseAbs().minCoeff();
                    if(min_grad > 0 && max_grad / min_grad > 1e6) {
                        log<LOG_ERROR>(L"%1% || WARNING: Poorly scaled problem! Gradient range: [%2%, %3%], ratio: %4%")
                            % __func__ % min_grad % max_grad % (max_grad/min_grad);
                    }

                    // Log solver parameters
                    log<LOG_ERROR>(L"%1% || Solver parameters:") % __func__;
                    log<LOG_ERROR>(L"%1% ||   ftol=%2%, wolfe=%3%, min_step=%4%, max_linesearch=%5%")
                        % __func__ % fitconfig.param.ftol % fitconfig.param.wolfe % fitconfig.param.min_step % fitconfig.param.max_linesearch;

                    // Try to diagnose the specific issue
                    if(grad.norm() < 1e-10) {
                        log<LOG_ERROR>(L"%1% || DIAGNOSIS: Gradient near zero - might be at a stationary point") % __func__;
                    } else if(max_grad / min_grad > 1e6) {
                        log<LOG_ERROR>(L"%1% || DIAGNOSIS: Poorly scaled parameters - consider rescaling") % __func__;
                    } else if(f_current < -1e10 || f_current > 1e10) {
                        log<LOG_ERROR>(L"%1% || DIAGNOSIS: Function value out of reasonable range") % __func__;
                    } else {
                        log<LOG_ERROR>(L"%1% || DIAGNOSIS: Likely too strict tolerances (ftol=%2%) or gradient calculation issues") 
                            % __func__ % fitconfig.param.ftol;
                    }

                    // Re-throw the exception after logging
                    throw;

                } else {
                    // Different error, just re-throw
                    throw;
                }
            }

            chi2s_localfits.push_back(fx);

            if (fx < chimin) {
                best_fit = x;
                chimin = fx;
            }

            log<LOG_INFO>(L"%1% || Minimization successful, chi %2% after %3% iterations") % __func__ % fx % niter;

            std::string spec_string = "";
            for (auto &f : x) spec_string += " " + std::to_string(f);
            log<LOG_DEBUG>(L"%1% || Best Point after minimization: %2%") % __func__ % spec_string.c_str();

            success = true;
            break;

        } catch (const std::runtime_error &except) {
            log<LOG_WARNING>(L"%1% || Minimization attempt %2%/%3% failed: %4%") % __func__ % attempt % fitconfig.n_max_local_retries % except.what();
            std::string msg = except.what();
            exception_string_map[msg]++;
        }
    }

    if (!success) {
        log<LOG_WARNING>(L"%1% || All minimization attempts failed to converge, using best internal minimum or best PSO value found.") % __func__;

        //log<LOG_WARNING>(L"%1% || PSO chi %2%  and local: %3% ") % __func__ % PSO.getGlobalBestScore() % fx;
        if (fx < chimin) {
            best_fit = x;
            chimin = fx;
        }
        if(PSO.getGlobalBestScore()< chimin){
            best_fit = PSO.getGlobalBestPosition();
            chimin = PSO.getGlobalBestScore();
        }
    }



    int fudge = 0;
    if(seed_points.size()>0 && seed_points.front().size()>0){
        fudge = seed_points.size();
        log<LOG_INFO>(L"%1% || Starting local fit of seed point. ") % __func__ ;

        for(size_t s = 0; s < seed_points.size();s++){
            for (size_t attempt = 1; attempt <= fitconfig.n_max_local_retries; ++attempt) {
                try {
                    x = seed_points.at(s);
                    log<LOG_INFO>(L"%1% || Starting local minimization attempt %2%/%3%") % __func__ % attempt % fitconfig.n_max_local_retries;
                    niter = solver.minimize(metric, x, fx, lb, ub);
                    chi2s_localfits.push_back(fx);

                    if (fx < chimin) {
                        best_fit = x;
                        chimin = fx;
                    }

                    log<LOG_INFO>(L"%1% || Minimization successful, chi %2% after %3% iterations") % __func__ % fx % niter;

                    std::string spec_string = "";
                    for (auto &f : x) spec_string += " " + std::to_string(f);
                    log<LOG_DEBUG>(L"%1% || Best Point after minimization: %2%") % __func__ % spec_string.c_str();

                    success = true;
                    break;

                } catch (const std::runtime_error &except) {
                    log<LOG_WARNING>(L"%1% || Minimization attempt %2%/%3% failed: %4%") % __func__ % attempt % fitconfig.n_max_local_retries % except.what();
                }
            }
        }
        if (!success) {
            log<LOG_WARNING>(L"%1% || All minimization attempts failed, falling back to PSO best") % __func__;
            if (fx < chimin) {
                best_fit = x;
                chimin = fx;
            }


        }
    }


    for(int i=0; i< fitconfig.n_localfit-1-fudge; i++){
        success = false;

        //After the best best fit, do you want to do more of the latin ones?
        x = Eigen::Map<Eigen::VectorXf>(latin_samples[best_multistart[i+1]].data(), latin_samples[best_multistart[i+1]].size());
        log<LOG_INFO>(L"%1% || Starting n_localfit local fit number %2%/%3% ") % __func__ % i  % fitconfig.n_localfit;


        for (size_t attempt = 1; attempt <= fitconfig.n_max_local_retries; ++attempt) {
            try {
                log<LOG_INFO>(L"%1% || Starting local minimization attempt %2%/%3%") % __func__ % attempt % fitconfig.n_max_local_retries;
                niter = solver.minimize(metric, x, fx, lb, ub);
                chi2s_localfits.push_back(fx);

                if (fx < chimin) {
                    best_fit = x;
                    chimin = fx;
                }

                log<LOG_INFO>(L"%1% || Minimization successful, chi %2% after %3% iterations") % __func__ % fx % niter;

                std::string spec_string = "";
                for (auto &f : x) spec_string += " " + std::to_string(f);
                log<LOG_DEBUG>(L"%1% || Best Point after minimization: %2%") % __func__ % spec_string.c_str();

                success = true;
                break;

            } catch (const std::runtime_error &except) {
                log<LOG_WARNING>(L"%1% || Minimization attempt %2%/%3% failed: %4%") % __func__ % attempt % fitconfig.n_max_local_retries % except.what();
                std::string msg = except.what();
                exception_string_map[msg]++;

            }
        }

        if (!success) {
            log<LOG_WARNING>(L"%1% || LBFGSB minimization attempts failed to fully converge. Using best found minimum or best PSO minimum.") % __func__;
            if (fx < chimin) {
                best_fit = x;
                chimin = fx;
            }


        }
    }

    log<LOG_INFO>(L"%1% || FINAL has a chi %2%") % __func__ %  chimin;
    std::string spec_string = "";
    for(auto &f : best_fit) spec_string+=" "+std::to_string(f); 
    log<LOG_INFO>(L"%1% || FINAL is  : %2% ") % __func__ % spec_string.c_str();

    return chimin;
}

