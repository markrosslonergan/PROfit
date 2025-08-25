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

            try {
                niter = solver.minimize(metric, x, fx, lb, ub);
            } catch(const std::runtime_error &e) {
                std::string error_msg = e.what();
            }

            chi2s_localfits.push_back(fx);

            if (fx < chimin) {
                best_fit = x;
                chimin = fx;
            }

            log<LOG_INFO>(L"%1% || Minimization successful, chi %2% after %3% iterations") % __func__ % fx % niter;
            log<LOG_WARNING>(L"%1% || -- chi %2% at ||||| %3% ") % __func__ % fx %x;

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
                    log<LOG_INFO>(L"%1% || --On Seed: %2%") % __func__ % x;
                    niter = solver.minimize(metric, x, fx, lb, ub);
                    chi2s_localfits.push_back(fx);


                    log<LOG_WARNING>(L"%1% || Min worked (seed)") % __func__;
                    log<LOG_WARNING>(L"%1% || -- chi %2% at ||||| %3% ") % __func__ % fx % x;
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
            log<LOG_WARNING>(L"%1% || All minimization attempts failed, The best internal was") % __func__;
            log<LOG_WARNING>(L"%1% || -- chi %2% at ||||| %3% ") % __func__ % fx % x;
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
        log<LOG_INFO>(L"%1% || #########################  ") % __func__ ;
        log<LOG_INFO>(L"%1% || Starting n_localfit local fit number %2%/%3% ") % __func__ % i  % fitconfig.n_localfit;

        for (size_t attempt = 1; attempt <= fitconfig.n_max_local_retries; ++attempt) {
            try {
                log<LOG_INFO>(L"%1% || Starting local minimization attempt %2%/%3%") % __func__ % attempt % fitconfig.n_max_local_retries;
                log<LOG_INFO>(L"%1% || Seed is %2% ") % __func__ %  x;

                niter = solver.minimize(metric, x, fx, lb, ub);
                chi2s_localfits.push_back(fx);

                log<LOG_WARNING>(L"%1% || Min worked (latin)") % __func__;
                log<LOG_WARNING>(L"%1% || -- chi %2% at ||||| %3% ") % __func__ % fx % x;

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

            log<LOG_WARNING>(L"%1% || All minimization attempts failed (latin), The best internal was") % __func__;
            log<LOG_WARNING>(L"%1% || -- chi %2% at ||||| %3% ") % __func__ % fx % x;

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


int PROfitter::calcFreqSeedPoints(PROmetric &metric) {
    freq_seed_points.clear();
    freq_seed_values.clear();

    if(best_fit.size()==0){
        return 0;
    }

    size_t nparams = metric.GetModel().nparams + metric.GetSysts().GetNSplines();
    size_t nphys = metric.GetModel().nparams;

    Eigen::VectorXf lb = Eigen::VectorXf::Constant(nparams, -3.0);
    Eigen::VectorXf ub = Eigen::VectorXf::Constant(nparams, 3.0);

    std::vector<float> chivalues;
    std::vector<float> chipos;

    //STEP 1, fix all syst values at best fit, and vary only physics
    Eigen::VectorXf test_point = best_fit;
    Eigen::VectorXf grad = Eigen::VectorXf::Constant(best_fit.size(),0);
    lb = best_fit;
    ub = best_fit;

    size_t osc_par = 0;
    for(size_t i=0; i<nphys;i++){
        if(i==osc_par)continue;
        lb(i)=metric.GetModel().lb(i);
        ub(i)=metric.GetModel().ub(i);
    }

    for(float k=lb(osc_par); k<=ub(osc_par);k+=0.01){
        test_point(osc_par)=k;
        lb(osc_par)=k;
        ub(osc_par)=k;
        metric.setBounds(lb,ub);

        //simple chi at best fit
        float chi_simple = metric(test_point,grad,false);

        float fx = -9;
        LBFGSpp::LBFGSBSolver<float> solver(fitconfig.param);
        try{
            int niter = solver.minimize(metric, test_point, fx, lb, ub);
        } catch (const std::runtime_error &except) {
            std::string msg = except.what();
            exception_string_map[msg]++;
        }
        chivalues.push_back(fx);
        chipos.push_back(k);
    }

    //#STEP 2 very simple minima finder in 1D
    std::vector<float> minima_dm;
    std::vector<float> minima_sin;
    for (int i = 2; i < chivalues.size()-2; ++i) {
        if (chivalues.at(i) < chivalues.at(i-1) && chivalues.at(i) < chivalues.at(i-2) && chivalues.at(i) < chivalues.at(i+1) && chivalues.at(i)< chivalues.at(i+2)) {
            log<LOG_INFO>(L"%1% || Local Minima found at position %2% with chi value of %3% ") %__func__% chipos.at(i) %  chivalues.at(i);
            minima_dm.push_back(chipos.at(i));
            minima_sin.push_back(chivalues.at(i));
        }
    }

    //STEP 3, loop over all mimima and do twofold minimzation. 
    //First with minima fixed to get BF of other values, then fully free.
    for(int p=0;p<minima_dm.size();p++){
        log<LOG_INFO>(L"%1% || ##################  ") %__func__;

        for(size_t i = 0; i < nphys; ++i) {
            lb(i) = metric.GetModel().lb(i);
            ub(i) = metric.GetModel().ub(i);
        }
        for(size_t i = nphys; i < nparams; ++i) {
            lb(i) = metric.GetSysts().spline_lo[i-nphys];
            ub(i) = metric.GetSysts().spline_hi[i-nphys];
        }

        //fix dm at minima
        lb(osc_par)=minima_dm.at(p);
        ub(osc_par)=minima_dm.at(p);

        metric.setBounds(lb,ub);

        //Best fit, with minima points
        Eigen::VectorXf test_minima = best_fit;
        test_minima(osc_par) = minima_dm.at(p);
        test_minima(1) = minima_sin.at(p);

        float fx;
        LBFGSpp::LBFGSBSolver<float> solver(fitconfig.param);
        try{
            int niter = solver.minimize(metric, test_minima, fx, lb, ub);
        } catch (const std::runtime_error &except) {
            std::string msg = except.what();
            exception_string_map[msg]++;
        }

        log<LOG_INFO>(L"%1% || FIXED freq MINIMA number %2% (@ %3%) has chi %4% ") %__func__% p % minima_dm.at(p) % fx;
        log<LOG_INFO>(L"%1% || -- at bf pt %2%  ") %__func__%  test_minima;

        //free the freq 
        lb(osc_par)=metric.GetModel().lb(osc_par);
        ub(osc_par)=metric.GetModel().ub(osc_par);
        metric.setBounds(lb,ub);

        try{
            int niter = solver.minimize(metric, test_minima, fx, lb, ub);
        } catch (const std::runtime_error &except) {
            std::string msg = except.what();
            exception_string_map[msg]++;
        }

        log<LOG_INFO>(L"%1% || FLOAT freq MINIMA number %2% (@ %3%) has chi %4% ") %__func__% p % minima_dm.at(p) % fx;
        log<LOG_INFO>(L"%1% || -- at bf pt %2%  ") %__func__%  test_minima;
        log<LOG_INFO>(L"%1% || ##################  ") %__func__;

        freq_seed_points.push_back(test_minima);
        freq_seed_values.push_back(fx);
    }

    log<LOG_INFO>(L"%1% || We have calculated %2% frequency seed points and saved for future use! ") %__func__% freq_seed_values.size();

    return freq_seed_values.size();
}

