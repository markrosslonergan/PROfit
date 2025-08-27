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


            if (fx < chimin) {
                best_fit = x;
                chimin = fx;
            }

            log<LOG_INFO>(L"%1% || Minimization successful, chi %2% after %3% iterations") % __func__ % fx % niter;
            log<LOG_INFO>(L"%1% || -- chi %2% at ||||| %3% ") % __func__ % fx %x;

            //std::string spec_string = "";
            //for (auto &f : x) spec_string += " " + std::to_string(f);
            //log<LOG_DEBUG>(L"%1% || Best Point after minimization: %2%") % __func__ % spec_string.c_str();

            success = true;
            break;

        } catch (const std::runtime_error &except) {
            log<LOG_INFO>(L"%1% || Minimization attempt %2%/%3% failed: %4%") % __func__ % attempt % fitconfig.n_max_local_retries % except.what();
            std::string msg = except.what();
            exception_string_map[msg]++;
        }
    }

    if (!success) {
        log<LOG_INFO>(L"%1% || All minimization attempts failed to converge, using best internal minimum or best PSO value found.") % __func__;

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

                    //first fix freq and whaver is supposed to be fixed  
                    Eigen::VectorXf tmp_lb = lb;
                    Eigen::VectorXf tmp_ub = ub;
                    if(lb(0)!=ub(0)){//already fixed the mass
                        tmp_lb(0) = x(0);
                        tmp_ub(0) = x(0);
                    }
                    metric.setBounds(tmp_lb,tmp_ub);
                    try{niter = solver.minimize(metric, x, fx, tmp_lb, tmp_ub);
                    }catch (const std::runtime_error &except) {}

                    if (fx < chimin) {
                        best_fit = x;
                        chimin = fx;
                    }
                    metric.freeParams();

                    //now release and fit with past bf from fixed above as seed
                    metric.setBounds(lb,ub);
                    try{
                        niter=solver.minimize(metric, x, fx, lb, ub);
                    }catch (const std::runtime_error &except) {}

                    if (fx < chimin) {
                        best_fit = x;
                        chimin = fx;
                    }

                    log<LOG_INFO>(L"%1% || Minimization successful, chi %2% after %3% iterations") % __func__ % fx % niter;
                    std::string spec_string = "";
                    for (auto &f : x) spec_string += " " + std::to_string(f);
                    log<LOG_DEBUG>(L"%1% || Best Point after minimization: %2%") % __func__ % spec_string.c_str();

                    metric.freeParams();
                    success = true;
                    break;

                } catch (const std::runtime_error &except) {
                    log<LOG_INFO>(L"%1% || Minimization attempt %2%/%3% failed: %4%") % __func__ % attempt % fitconfig.n_max_local_retries % except.what();
                    metric.freeParams();
                }
            }
        }//end seed
        if (!success) {
            log<LOG_INFO>(L"%1% || All minimization attempts failed, The best internal was") % __func__;
            log<LOG_INFO>(L"%1% || -- chi %2% at ||||| %3% ") % __func__ % fx % x;
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
                log<LOG_INFO>(L"%1% || Minimization attempt %2%/%3% failed: %4%") % __func__ % attempt % fitconfig.n_max_local_retries % except.what();
                std::string msg = except.what();
                exception_string_map[msg]++;

            }
        }

        if (!success) {
            log<LOG_INFO>(L"%1% || LBFGSB minimization attempts failed to fully converge. Using best found minimum or best PSO minimum.") % __func__;

            log<LOG_INFO>(L"%1% || All minimization attempts failed (latin), The best internal was") % __func__;
            log<LOG_INFO>(L"%1% || -- chi %2% at ||||| %3% ") % __func__ % fx % x;

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
        log<LOG_WARNING>(L"%1% || WARNING need to run a global fit using this PROfitter before asking to calculate Frequencey Seed Points.  ") %__func__;
        return 0;
    }

    size_t nparams = metric.GetModel().nparams + metric.GetSysts().GetNSplines();
    size_t nphys = metric.GetModel().nparams;

    Eigen::VectorXf lb = Eigen::VectorXf::Constant(nparams, -3.0);
    Eigen::VectorXf ub = Eigen::VectorXf::Constant(nparams, 3.0);

    std::vector<float> chivalues;
    std::vector<float> chipos;

    //STEP 1, fix all syst values at best fit, and vary only physics
    Eigen::VectorXf local_candidate ;
    Eigen::VectorXf grad = Eigen::VectorXf::Constant(best_fit.size(),0);

    //TODO hardcoded 1 mass splittin for now
    size_t osc_par = 0;
    for(size_t i=0; i<nphys;i++){
        lb(i)=metric.GetModel().lb(i);
        ub(i)=metric.GetModel().ub(i);
    }

    //staggered test points, focus on active refion for efficiency
    float s1=0.0, s2=1.5;
    float w1=0.1, w2=0.01, w3 =0.05;
    std::vector<float> test_p;
    for (float val = lb(osc_par); val <= s1; val += w1) test_p.push_back(val);
    for (float val = s1 + w2; val <= s2; val += w2) test_p.push_back(val);
    for (float val = s2 + w3; val <= ub(osc_par); val += w3) test_p.push_back(val);

    for(float &k : test_p){
        local_candidate = best_fit;
        local_candidate(osc_par) =k;
        Eigen::VectorXf temp_lb = local_candidate;
        Eigen::VectorXf temp_ub = local_candidate;

        //vary ONLY dm manually, and fit other_physics. Nuisence at BF
        for(size_t i=0; i<nphys;i++){
            if(i==osc_par) continue;
            temp_lb(i)=metric.GetModel().lb(i);
            temp_ub(i)=metric.GetModel().ub(i);
        }
        metric.setBounds(temp_lb,temp_ub);

        float fx = -9;
        try{
            int niter = solver.minimize(metric, local_candidate, fx, temp_lb, temp_ub);
        } catch (const std::runtime_error &except) {
            std::string msg = except.what();
            exception_string_map[msg]++;
        }
        chivalues.push_back(fx);
        chipos.push_back(k);

        //log<LOG_INFO>(L"%1% || PARG  %2% %3% %4% ") %__func__% k %  local_candidate % fx;
    }

    //#STEP 2 From the scan above, find local minima in freq
    float chi2_drop_param = 0.5;
    float min_dist_minima_param = 0.025;
    std::vector<std::pair<float,float>> minima = findSignificantMinima(chipos, chivalues,  chi2_drop_param, min_dist_minima_param, true);


    //STEP 3, loop over all mimima and do twofold minimzation. 
    //First with DM minima fixed to get BF of pull terms, then fully free to optimize the mass splitting to high precisin
    for(int p=0;p<minima.size();p++){
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
        lb(osc_par)=minima.at(p).first;
        ub(osc_par)=minima.at(p).first;
        metric.setBounds(lb,ub);

        //Best fit, with minima points
        Eigen::VectorXf test_minima = best_fit;
        test_minima(osc_par) = minima.at(p).first;
        test_minima(1) = minima.at(p).second;

        float fx;
        LBFGSpp::LBFGSBSolver<float> solver(fitconfig.param);
        try{
            int niter = solver.minimize(metric, test_minima, fx, lb, ub);
        } catch (const std::runtime_error &except) {
            std::string msg = except.what();
            exception_string_map[msg]++;
        }

        log<LOG_INFO>(L"%1% || Local min num (%2%) with FIXED frequency (%3%) has chi %4% ") %__func__% p % minima.at(p).first % fx;
        log<LOG_INFO>(L"%1% || -- at bf pt %2%  ") %__func__%  test_minima;

        //free the freq and repeat 
        lb(osc_par)=metric.GetModel().lb(osc_par);
        ub(osc_par)=metric.GetModel().ub(osc_par);
        metric.setBounds(lb,ub);

        try{
            int niter = solver.minimize(metric, test_minima, fx, lb, ub);
        } catch (const std::runtime_error &except) {
            std::string msg = except.what();
            exception_string_map[msg]++;
        }

        log<LOG_INFO>(L"%1% || Local min num (%2%) with floating frequency (%3%) has chi %4% (but bf freq was %5%) ") %__func__% p % minima.at(p).first % fx % test_minima(osc_par);
        log<LOG_INFO>(L"%1% || -- at bf pt %2%  ") %__func__%  test_minima;

        freq_seed_points.push_back(test_minima);
        freq_seed_values.push_back(fx);
    }
    log<LOG_INFO>(L"%1% || ##################  ") %__func__;
    log<LOG_INFO>(L"%1% || We have calculated %2% frequency seed points and saved for future use! ") %__func__% freq_seed_values.size();

    return freq_seed_values.size();
}

std::vector<std::pair<float, float>> PROfitter::findSignificantMinima(  const std::vector<float>& x_values,const std::vector<float>& y_values,     
        float in_prominence_threshold ,  float in_min_spacing_log ,     bool use_log_spacing ){


    std::vector<std::pair<float, float>> minima;  
    float prominence_threshold = in_prominence_threshold;
    float min_spacing_log = in_min_spacing_log;
    bool first = true;

    while(first || minima.size()>20){

        if(x_values.size() != y_values.size() || x_values.size() < 3) {
            return minima;
        }

        for(size_t i = 1; i < y_values.size() - 1; ++i) {
            // Check if local minimum
            if(y_values[i] >= y_values[i-1] || y_values[i] >= y_values[i+1]) {
                continue;
            }

            float left_peak = y_values[i];
            float right_peak = y_values[i];

            // Search left for peak
            // Stop if sufficient
            for(int j = i - 1; j >= 0; --j) {
                left_peak = std::max(left_peak, y_values[j]);
                if(y_values[j] > y_values[i] + prominence_threshold * 2) {
                    break;
                }
            }
            //same othr
            for(size_t j = i + 1; j < y_values.size(); ++j) {
                right_peak = std::max(right_peak, y_values[j]);
                if(y_values[j] > y_values[i] + prominence_threshold * 2) {
                    break;
                }
            }

            float prominence = std::min(left_peak - y_values[i], right_peak - y_values[i]);

            // Check prominence threshold
            if(prominence < prominence_threshold) {
                continue;
            }

            // Check spacing from last accepted minimum
            bool too_close = false;
            if(!minima.empty()) {
                float spacing;
                if(use_log_spacing && x_values[i] > 0 && minima.back().first > 0) {
                    spacing = std::abs(std::log10(x_values[i]) - std::log10(minima.back().first));
                } else {
                    spacing = std::abs(x_values[i] - minima.back().first);
                }

                if(spacing < min_spacing_log) {
                    if(y_values[i] < minima.back().second) {
                        minima.back() = {x_values[i], y_values[i]};
                    }
                    too_close = true;
                }
            }

            if(!too_close) {
                minima.push_back({x_values[i], y_values[i]});
            }
        }

        first = false;
        prominence_threshold = prominence_threshold*0.9;
        min_spacing_log = min_spacing_log*1.1;

    }
    return minima;
}
