#include "PROfitter.h"
#include "PROlog.h"
#include "PROmetric.h"
#include "PROswarm.h"
#include <Eigen/Eigen>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

using namespace PROfit;

namespace PROfit {
    // Definitions of the global scan-timing machinery declared in PROfitter.h.
    // Function-local statics so the order of construction across translation
    // units doesn't bite.
    ScanTimingStats& GetScanTimingStats() {
        static ScanTimingStats s;
        return s;
    }
    bool& GetScanTimingEnabled() {
        static bool b = false;
        return b;
    }
}

std::vector<std::vector<float>> PROfit::latin_hypercube_sampling(size_t num_samples, size_t dimensions, std::uniform_real_distribution<float>&dis, std::mt19937 &gen) {
    // Stratified LHS on the unit cube: dimension-wise, one sample per stratum
    // [i/n, (i+1)/n), independently shuffled. The jitter draw is normalised
    // onto [0,1) from whatever uniform range the caller's distribution has, so
    // strata never overlap and the output always spans [0,1).
    std::vector<std::vector<float>> samples(num_samples, std::vector<float>(dimensions));

    const float dis_lo = dis.min(), dis_span = dis.max() - dis.min();
    for (size_t d = 0; d < dimensions; ++d) {

        std::vector<float> perm(num_samples);
        for (size_t i = 0; i < num_samples; ++i) {
            const float u01 = dis_span > 0 ? (dis(gen) - dis_lo) / dis_span : 0.5f;
            perm[i] = (i + u01) / num_samples;
        }
        std::shuffle(perm.begin(), perm.end(), gen);
        for (size_t i = 0; i < num_samples; ++i) {
            samples[i][d] = perm[i];
        }
    }

    return samples;
}

void PROfit::recenter_latin_samples(std::vector<std::vector<float>> &samples, const Eigen::VectorXf &ub, const Eigen::VectorXf &lb) {
    // Map unit-cube LHS samples onto the full fit box [lb, ub] in every
    // dimension. An infinite bound falls back to a 4-unit window against the
    // finite side, or [-2, 2] if both sides are unbounded.
    for(std::vector<float> &pt: samples) {
        for(size_t i = 0; i < pt.size(); ++i) {
            const bool lo_inf = std::isinf(lb(i)), hi_inf = std::isinf(ub(i));
            float lo, width;
            if(!lo_inf && !hi_inf) { lo = lb(i);     width = ub(i) - lb(i); }
            else if(!lo_inf)       { lo = lb(i);     width = 4; }
            else if(!hi_inf)       { lo = ub(i) - 4; width = 4; }
            else                   { lo = -2;        width = 4; }
            pt[i] = lo + pt[i] * width;
        }
    }
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



static inline
std::vector<int> select_diverse_best(const std::vector<float>& chi2s, const std::vector<std::vector<float>>& points,size_t n_select,
                                     const Eigen::VectorXf& ub,const Eigen::VectorXf& lb,float diversity_factor = 0.3f) {
    
    if(n_select >= chi2s.size()) {
        return sorted_indices(chi2s);
    }
    
    std::vector<int> sorted_idx = sorted_indices(chi2s);
    
    std::vector<int> selected;
    selected.reserve(n_select);
    
    // always include the best point
    selected.push_back(sorted_idx[0]);
    
    // Use diversity_factor * average normalized dimension range
    size_t ndim = points[0].size();
    float threshold = diversity_factor / std::sqrt(static_cast<float>(ndim));
    
    for(size_t i = 1; i < sorted_idx.size() && selected.size() < n_select; ++i) {
        int candidate_idx = sorted_idx[i];
        bool is_diverse = true;
        
        for(int sel_idx : selected) {
            float normalized_dist_sq = 0.0;
            
            // Calculate normalized Euclidean distance
            for(size_t d = 0; d < ndim; ++d) {
                float range = (ub(d) - lb(d));
                if(std::isinf(range) || range <= 0) {
                    range = 1.0;  // Default normalization for unbounded dimensions
                }
                float diff = (points[candidate_idx][d] - points[sel_idx][d]) / range;
                normalized_dist_sq += diff * diff;
            }
            
            float normalized_dist = std::sqrt(normalized_dist_sq);
            
            if(normalized_dist < threshold) {
                is_diverse = false;
                break;
            }
        }
        
        if(is_diverse) {
            selected.push_back(candidate_idx);
        }
    }
    
    // i not enough, fill with best remaining
    if(selected.size() < n_select) {
        log<LOG_DEBUG>(L"%1% || Only found %2% diverse points out of requested %3% with diversity_factor=%4%")%  __func__ %  selected.size() % n_select % diversity_factor;
        
        for(size_t i = 0; i < sorted_idx.size() && selected.size() < n_select; ++i) {
            int idx = sorted_idx[i];
            if(std::find(selected.begin(), selected.end(), idx) == selected.end()) {
                selected.push_back(idx);
            }
        }
    }
    
    return selected;
}


float PROfitter::Fit(PROmetric &metric, const Eigen::VectorXf &seed_pt ) {
    const std::vector<Eigen::VectorXf> seed_points = seed_pt.size() > 0 ? std::vector<Eigen::VectorXf>{seed_pt} : std::vector<Eigen::VectorXf>{};
    return Fit(metric, seed_points);

}

float PROfitter::Fit(PROmetric &metric, const std::vector<Eigen::VectorXf> &seed_points, const std::vector<FixedSeed> &fixed_seeds ) {

    metric.setGradientMode(fitconfig.gradient_mode);
    total_lbfgs_iterations = 0;

    const bool tim_on = PROfit::GetScanTimingEnabled();
    auto fit_t0 = tim_on ? std::chrono::steady_clock::now()
                         : std::chrono::steady_clock::time_point{};

    std::mt19937 rng;
    rng.seed(seed);
    std::normal_distribution<float> d;
    std::uniform_real_distribution<float> d_uni(0.0, 1.0);

    //n_latin_points is how many initial latin cube points, sampled on the unit
    //cube and then mapped onto the full [lb, ub] fit box.
    std::vector<std::vector<float>> latin_samples = latin_hypercube_sampling(fitconfig.n_latin_points, ub.size(), d_uni,rng);
    recenter_latin_samples(latin_samples, ub, lb);
    const Eigen::Index n_phys_dims = (Eigen::Index)metric.GetModel().nparams;
    if(fitconfig.latin_phys_only){
        // Physics-only LHS: nuisance parameters at nominal (0, clamped into the box)
        // so the LHS chi2 ranks oscillation basins rather than random nuisance pulls.
        for(auto &pt : latin_samples)
            for(Eigen::Index i = n_phys_dims; i < (Eigen::Index)pt.size(); ++i)
                pt[i] = std::min(std::max(0.0f, lb(i)), ub(i));
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
    chi2s_multistart.reserve(fitconfig.n_latin_points);

    log<LOG_INFO>(L"%1% || Starting MultiGlobal runs (i.e latin hypercube runs, pure chi^2 no grad) : %2%") % __func__ % fitconfig.n_latin_points ;
    auto latin_t0 = tim_on ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point{};
    for(int s = 0; s < fitconfig.n_latin_points; s++){
        Eigen::VectorXf x = Eigen::Map<Eigen::VectorXf>(latin_samples[s].data(), latin_samples[s].size());
        Eigen::VectorXf grad = Eigen::VectorXf::Constant(x.size(), 0);
        float fx =  metric(x, grad, false);
        chi2s_multistart.push_back(fx);
        if(run_progress){progress->increment_bar(0);}

    }
    if (tim_on) {
        const auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - latin_t0).count();
        PROfit::GetScanTimingStats().latin_us.fetch_add((uint64_t)dt, std::memory_order_relaxed);
    }
    //Sort so we can take the best N_localfits for further zoning with a PSO
    //std::vector<int> best_multistart = sorted_indices(chi2s_multistart);    
    std::vector<int> best_multistart = select_diverse_best(chi2s_multistart, latin_samples, fitconfig.n_swarm_particles, ub, lb, fitconfig.latin_diversity_factor); 

    log<LOG_INFO>(L"%1% || Best Point after latin hypercube has chi^2 %2% with pts  : %3% ") % __func__ % chi2s_multistart[best_multistart[0]] % latin_samples[best_multistart[0]];

    std::string swarm_string = "";
    std::vector<std::vector<float>> swarm_start_points;
    int niter=0;
    float fx = std::numeric_limits<float>::infinity();
    if(fitconfig.n_swarm_particles < 1){
        fitconfig.n_swarm_particles = 1;
    }

    const size_t n_swarm_avail = std::min((size_t)fitconfig.n_swarm_particles, best_multistart.size());
    for(size_t s = 0; s < n_swarm_avail; s++){
        swarm_string += " " + std::to_string(chi2s_multistart[best_multistart[s]]);
        swarm_start_points.push_back(latin_samples[best_multistart[s]]);
    }
    log<LOG_INFO>(L"%1% || Will swarm with %2% swarm points chis of %3% ") % __func__ % fitconfig.n_swarm_particles % swarm_string.c_str();

    auto pso_t0 = tim_on ? std::chrono::steady_clock::now()
                         : std::chrono::steady_clock::time_point{};
    PROswarm PSO(metric, rng, swarm_start_points, lb, ub , fitconfig.n_swarm_iterations);
    if(run_progress){
        PSO.runSwarm(metric,rng,fitconfig, progress);
    }else{
        PSO.runSwarm(metric,rng,fitconfig);
    }
    if (tim_on) {
        const auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - pso_t0).count();
        PROfit::GetScanTimingStats().pso_us.fetch_add((uint64_t)dt, std::memory_order_relaxed);
    }

    auto lbfgs_t0 = tim_on ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point{};

    Eigen::VectorXf x;

    float chimin = 9999999;
    std::vector<float> chi2s_localfits;
    niter=0;

    // Per-Fit tally of local-refinement failures. Individual throws are
    // routine (near-optimal starts break the More-Thuente line search, see the
    // seed-candidate note below) so per-attempt detail goes to LOG_DEBUG and a
    // single LOG_WARNING summary is emitted at the end of the fit.
    std::map<std::string, size_t> fit_exception_counts;
    size_t n_fail_pso = 0, n_fail_seed = 0, n_fail_latin = 0, n_latin_starts = 0;

    bool success = false;
    best_fit = PSO.getGlobalBestPosition();
    chimin = PSO.getGlobalBestScore();


    log<LOG_INFO>(L"%1% || Starting local fit of best swarm point. ") % __func__ ;

    // NOTE on exception handling in the local-fit blocks below: the solver is
    // called directly inside the attempt loop's try. A throwing attempt leaves
    // fx/x in an unspecified state, so failed attempts record the exception,
    // retry, and never feed fx/x into the best-fit bookkeeping. (Previously an
    // inner catch swallowed every exception: retries could never happen and an
    // uninitialized fx could be recorded as the best chi².)
    for (size_t attempt = 1; attempt <= fitconfig.n_max_local_retries; ++attempt) {
        if(run_progress)progress->increment_bar(2);

        try {
            x = PSO.getGlobalBestPosition();
            log<LOG_INFO>(L"%1% || Starting local minimization attempt %2%/%3%") % __func__ % attempt % fitconfig.n_max_local_retries;

            niter = solver.minimize(metric, x, fx, lb, ub);
            total_lbfgs_iterations += (size_t)std::max(niter, 0);

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

        } catch (const std::exception &except) {
            exception_string_map[std::string(except.what())]++;
            fit_exception_counts[std::string(except.what())]++;
            log<LOG_DEBUG>(L"%1% || Minimization attempt %2%/%3% failed: %4%") % __func__ % attempt % fitconfig.n_max_local_retries % except.what();
        }
    }

    if (!success) {
        // best_fit/chimin already hold the PSO best (set above); nothing valid
        // came out of the local attempts.
        ++n_fail_pso;
        log<LOG_DEBUG>(L"%1% || All minimization attempts failed, falling back to PSO best with chi %2%") % __func__ % chimin;
    }



    int fudge = 0;
    if(seed_points.size()>0 && seed_points.front().size()>0){
        fudge = seed_points.size();
        log<LOG_INFO>(L"%1% || Starting local fit of seed point. ") % __func__ ;

        if(run_progress)progress->increment_bar(2);
        for(size_t s = 0; s < seed_points.size();s++){
            // Candidate guarantee: the (bound-clamped) seed is itself a valid
            // point whose chi2 costs one evaluation. Record it BEFORE the LBFGS
            // refinement: a seed that is already at/near a minimum routinely
            // makes the More-Thuente line search throw ("step became smaller
            // than the minimum value allowed" — float-level chi2 changes cannot
            // satisfy the Wolfe conditions), and previously that discarded the
            // known-good seed entirely, leaving the much cruder LHS/PSO value
            // as the result. This is what produced spiky PROfile curves: scan
            // points whose every seed refinement threw sat several chi2 units
            // above their neighbours. With the seed recorded first, refinement
            // failure degrades to "keep the seed's own chi2".
            {
                Eigen::VectorXf x0 = seed_points.at(s).cwiseMax(lb).cwiseMin(ub);
                Eigen::VectorXf g0 = Eigen::VectorXf::Zero(x0.size());
                const float f0 = metric(x0, g0, false);
                if (std::isfinite(f0) && f0 < chimin) {
                    best_fit = x0;
                    chimin = f0;
                }
            }
            bool seed_success = false;
            for (size_t attempt = 1; attempt <= fitconfig.n_max_local_retries; ++attempt) {
                try {
                    x = seed_points.at(s);
                    log<LOG_INFO>(L"%1% || Starting local minimization attempt %2%/%3%") % __func__ % attempt % fitconfig.n_max_local_retries;

                    niter = solver.minimize(metric, x, fx, lb, ub);
                    total_lbfgs_iterations += (size_t)std::max(niter, 0);

                    chi2s_localfits.push_back(fx);

                    if (fx < chimin) {
                        best_fit = x;
                        chimin = fx;
                    }

                    log<LOG_INFO>(L"%1% || Minimization successful, chi %2% after %3% iterations") % __func__ % fx % niter;

                    std::string spec_string = "";
                    for (auto &f : x) spec_string += " " + std::to_string(f);
                    log<LOG_DEBUG>(L"%1% || Best Point after minimization: %2%") % __func__ % spec_string.c_str();

                    seed_success = true;
                    break;

                } catch (const std::exception &except) {
                    exception_string_map[std::string(except.what())]++;
                    fit_exception_counts[std::string(except.what())]++;
                    log<LOG_DEBUG>(L"%1% || Minimization attempt %2%/%3% failed: %4%") % __func__ % attempt % fitconfig.n_max_local_retries % except.what();
                }
            }

            if (!seed_success) {
                ++n_fail_seed;
                log<LOG_DEBUG>(L"%1% || Seed-point refinement failed; keeping best-so-far (chi %2%, includes the seed's own chi2 as a candidate)") % __func__ % chimin;
            }
        }
    }

    // Fixed seeds: start points whose flagged parameters are HELD at their seed
    // values during refinement (zero-width solver bounds for this seed only) —
    // e.g. the background-only seed with physics pinned at the model defaults,
    // making a nuisances-only fit of the null hypothesis a guaranteed candidate.
    // A seed whose pinned value the fit does not allow (a --fix'd parameter, or
    // a profile/surface scan pinning the same axis at a different value) is
    // skipped: its constrained minimum would sit outside this fit's space.
    // Gated by fitconfig.use_bkg_seed (--fit-options / --scan-fit-options).
    if(!fitconfig.use_bkg_seed && !fixed_seeds.empty()) {
        log<LOG_INFO>(L"%1% || use_bkg_seed=0: skipping %2% fixed seed(s).") % __func__ % fixed_seeds.size();
    }
    for(size_t s = 0; fitconfig.use_bkg_seed && s < fixed_seeds.size(); s++){
        const FixedSeed &fs = fixed_seeds[s];
        if(fs.point.size() != lb.size() || (long)fs.fixed.size() != fs.point.size()){
            log<LOG_WARNING>(L"%1% || Fixed seed %2% is mis-sized (point %3%, flags %4%, fit has %5% params); skipping.")
                % __func__ % s % fs.point.size() % fs.fixed.size() % lb.size();
            continue;
        }
        bool conflict = false;
        int npinned = 0;
        for(long i = 0; i < fs.point.size(); ++i){
            if(!fs.fixed[i]) continue;
            ++npinned;
            if(fs.point(i) < lb(i) || fs.point(i) > ub(i)){
                log<LOG_INFO>(L"%1% || Fixed seed %2% pins param %3% at %4% but this fit restricts it to [%5%, %6%]; seed skipped.")
                    % __func__ % s % i % fs.point(i) % lb(i) % ub(i);
                conflict = true;
                break;
            }
        }
        if(conflict) continue;
        ++fudge; // fixed seeds consume local-fit budget like regular seeds

        log<LOG_INFO>(L"%1% || Starting local fit of fixed seed %2% (%3% pinned params).") % __func__ % s % npinned;
        if(run_progress)progress->increment_bar(2);

        // Candidate guarantee, same as regular seeds: record the seed's own
        // chi2 before refinement (pinned entries are in-range by the check
        // above, free entries are clamped, so this is a valid fit-space point).
        Eigen::VectorXf x0 = fs.point.cwiseMax(lb).cwiseMin(ub);
        {
            Eigen::VectorXf g0 = Eigen::VectorXf::Zero(x0.size());
            const float f0 = metric(x0, g0, false);
            if (std::isfinite(f0) && f0 < chimin) {
                best_fit = x0;
                chimin = f0;
            }
        }

        Eigen::VectorXf flb = lb, fub = ub;
        for(long i = 0; i < fs.point.size(); ++i){
            if(fs.fixed[i]){ flb(i) = x0(i); fub(i) = x0(i); }
        }

        bool seed_success = false;
        for (size_t attempt = 1; attempt <= fitconfig.n_max_local_retries; ++attempt) {
            try {
                x = x0;
                log<LOG_INFO>(L"%1% || Starting fixed-seed local minimization attempt %2%/%3%") % __func__ % attempt % fitconfig.n_max_local_retries;

                niter = solver.minimize(metric, x, fx, flb, fub);
                total_lbfgs_iterations += (size_t)std::max(niter, 0);

                chi2s_localfits.push_back(fx);

                if (fx < chimin) {
                    best_fit = x;
                    chimin = fx;
                }

                log<LOG_INFO>(L"%1% || Minimization successful, chi %2% after %3% iterations") % __func__ % fx % niter;

                std::string spec_string = "";
                for (auto &f : x) spec_string += " " + std::to_string(f);
                log<LOG_DEBUG>(L"%1% || Best Point after minimization: %2%") % __func__ % spec_string.c_str();

                seed_success = true;
                break;

            } catch (const std::exception &except) {
                exception_string_map[std::string(except.what())]++;
                fit_exception_counts[std::string(except.what())]++;
                log<LOG_DEBUG>(L"%1% || Minimization attempt %2%/%3% failed: %4%") % __func__ % attempt % fitconfig.n_max_local_retries % except.what();
            }
        }

        if (!seed_success) {
            ++n_fail_seed;
            log<LOG_DEBUG>(L"%1% || Fixed-seed refinement failed; keeping best-so-far (chi %2%, includes the seed's own chi2 as a candidate)") % __func__ % chimin;
        }
    }

    for(int i=0; i< fitconfig.n_localfit-1-fudge && (size_t)(i+1) < best_multistart.size(); i++){
        ++n_latin_starts;
        success = false;

        //After the best best fit, do you want to do more of the latin ones?
        x = Eigen::Map<Eigen::VectorXf>(latin_samples[best_multistart[i+1]].data(), latin_samples[best_multistart[i+1]].size());
        if(fitconfig.localfit_warm_nuisance && std::isfinite(chimin) && best_fit.size() == x.size() && x.size() > n_phys_dims){
            // Physics from the latin point, nuisances from the best fit so far.
            x.tail(x.size() - n_phys_dims) = best_fit.tail(x.size() - n_phys_dims);
        }
        log<LOG_INFO>(L"%1% || Starting n_localfit local fit number %2%/%3% ") % __func__ % i  % fitconfig.n_localfit;

        if(run_progress)progress->increment_bar(2);

        for (size_t attempt = 1; attempt <= fitconfig.n_max_local_retries; ++attempt) {
            try {
                log<LOG_INFO>(L"%1% || Starting local minimization attempt %2%/%3%") % __func__ % attempt % fitconfig.n_max_local_retries;

                niter = solver.minimize(metric, x, fx, lb, ub);
                total_lbfgs_iterations += (size_t)std::max(niter, 0);

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

            } catch (const std::exception &except) {
                exception_string_map[std::string(except.what())]++;
                fit_exception_counts[std::string(except.what())]++;
                log<LOG_DEBUG>(L"%1% || Minimization attempt %2%/%3% failed: %4%") % __func__ % attempt % fitconfig.n_max_local_retries % except.what();
            }
        }

        if (!success) {
            ++n_fail_latin;
            log<LOG_DEBUG>(L"%1% || All minimization attempts failed for this start point, keeping current best (chi %2%).") % __func__ % chimin;
        }
    }

    const size_t n_fail_total = n_fail_pso + n_fail_seed + n_fail_latin;
    if (n_fail_total > 0) {
        std::string exc_breakdown;
        for (const auto &[msg, count] : fit_exception_counts)
            exc_breakdown += " " + std::to_string(count) + "x \"" + msg + "\";";
        log<LOG_INFO>(L"%1% || Fit summary: refinement threw on %2%/%3% start points (PSO-best %4%/1, seeds %5%/%6%, latin %7%/%8%) -- benign, pre-refinement candidate chi2s retained. Exceptions:%9%")
            % __func__ % n_fail_total % (1 + fudge + n_latin_starts)
            % n_fail_pso % n_fail_seed % fudge % n_fail_latin % n_latin_starts
            % exc_breakdown.c_str();
    }

    log<LOG_INFO>(L"%1% || FINAL has a chi %2%") % __func__ %  chimin;
    std::string spec_string = "";
    for(auto &f : best_fit) spec_string+=" "+std::to_string(f);
    log<LOG_INFO>(L"%1% || FINAL is  : %2% ") % __func__ % spec_string.c_str();

    if (tim_on) {
        const auto now = std::chrono::steady_clock::now();
        const auto dt_lbfgs = std::chrono::duration_cast<std::chrono::microseconds>(
            now - lbfgs_t0).count();
        const auto dt_total = std::chrono::duration_cast<std::chrono::microseconds>(
            now - fit_t0).count();
        auto& s = PROfit::GetScanTimingStats();
        s.lbfgs_us.fetch_add((uint64_t)dt_lbfgs, std::memory_order_relaxed);
        s.total_fit_us.fetch_add((uint64_t)dt_total, std::memory_order_relaxed);
        s.n_fits.fetch_add(1, std::memory_order_relaxed);
    }

    return chimin;
}


int PROfitter::calcFreqSeedPoints(PROmetric &metric) {
    freq_seed_points.clear();
    freq_seed_values.clear();
    harmonic_scan_pos.clear();
    harmonic_scan_chi.clear();

    if(best_fit.size()==0){
        log<LOG_WARNING>(L"%1% || WARNING need to run a global fit using this PROfitter before asking to calculate Frequencey Seed Points.  ") %__func__;
        return 0;
    }

    size_t nparams = metric.GetModel().nparams + metric.GetSysts().GetNSplines();
    size_t nphys = metric.GetModel().nparams;

    if(nphys == 0){
        log<LOG_INFO>(L"%1% || No physics parameters in model, skipping frequency seed point calculation.") % __func__;
        return 0;
    }

    // The scan repeatedly rebinds the metric's bounds and (for fit modes) its
    // gradient mode; callers keep using the metric afterwards, so both are
    // restored before returning.
    const Eigen::VectorXf saved_lb = metric.lb;
    const Eigen::VectorXf saved_ub = metric.ub;
    const PROmetric::GradientMode saved_gmode = metric.getGradientMode();

    //TODO hardcoded 1 mass splittin for now
    const size_t osc_par = 0;
    const int scan_mode = std::min(std::max(fitconfig.harmonic_scan_mode, 0), 2);

    const float model_lo = std::isfinite(metric.GetModel().lb(osc_par)) ? metric.GetModel().lb(osc_par) : -3.0f;
    const float model_hi = std::isfinite(metric.GetModel().ub(osc_par)) ? metric.GetModel().ub(osc_par) : 3.0f;

    // Densely sampled window, clamped into the model range so no test point
    // can land outside the fit bounds.
    float dense_lo = std::min(std::max(fitconfig.harmonic_dense_lo, model_lo), model_hi);
    float dense_hi = std::min(std::max(fitconfig.harmonic_dense_hi, model_lo), model_hi);
    if(dense_hi < dense_lo) std::swap(dense_lo, dense_hi);

    const int n_dense = std::max((int)fitconfig.harmonic_num_test_points, 2);
    const float w2 = (dense_hi > dense_lo) ? (dense_hi - dense_lo) / (float)n_dense : 0.0f;

    std::vector<float> test_p;
    // Region 1: coarse log-spaced ladder below the dense window.
    if(w2 > 0 && dense_lo > model_lo) {
        const float w1 = 10.0f * w2;
        const int n1 = (int)std::floor((dense_lo - model_lo) / w1);
        for(int i = 0; i <= n1; ++i) test_p.push_back(model_lo + (float)i * w1);
    }
    // Region 2: dense log-spaced window.
    if(w2 > 0) {
        for(int i = 0; i <= n_dense; ++i) test_p.push_back(dense_lo + (float)i * w2);
    } else {
        test_p.push_back(dense_lo);
    }
    // Region 3: above the dense window the chi2 minima are equally spaced in
    // LINEAR dm2, so a log-spaced ladder aliases where harmonics crowd
    // hardest. Continue with the linear step the log grid had at dense_hi
    // (continuous sampling density), capped at 2x the dense budget.
    if(model_hi > dense_hi && w2 > 0) {
        const float dm_lo = std::pow(10.0f, dense_hi);
        const float dm_hi = std::pow(10.0f, model_hi);
        const float dm_step = dm_lo * std::log(10.0f) * w2;
        int n3 = (int)std::ceil((dm_hi - dm_lo) / dm_step);
        n3 = std::min(std::max(n3, 1), 2 * n_dense);
        for(int i = 1; i <= n3; ++i)
            test_p.push_back(std::log10(dm_lo + (dm_hi - dm_lo) * (float)i / (float)n3));
    }
    for(float &v : test_p) v = std::min(std::max(v, model_lo), model_hi);
    std::sort(test_p.begin(), test_p.end());
    test_p.erase(std::unique(test_p.begin(), test_p.end(),
                [](float a, float b){ return std::fabs(a - b) < 1e-5f; }), test_p.end());

    log<LOG_INFO>(L"%1% || Harmonic scan: mode %2%, %3% test points on [%4%, %5%] (dense window [%6%, %7%]).")
        % __func__ % scan_mode % test_p.size() % model_lo % model_hi % dense_lo % dense_hi;

    // Scan-fit bounds for modes 1/2: the frequency is pinned per point.
    // Mode 1 additionally pins every spline at its global-BF value; mode 2
    // frees them over restrict-else-lo/hi (a true per-point profile).
    Eigen::VectorXf fit_lb(nparams), fit_ub(nparams);
    if(scan_mode > 0) {
        for(size_t i = 0; i < nphys; ++i) {
            fit_lb(i) = metric.GetModel().lb(i);
            fit_ub(i) = metric.GetModel().ub(i);
        }
        for(size_t i = nphys; i < nparams; ++i) {
            const size_t si = i - nphys;
            if(scan_mode == 1) {
                fit_lb(i) = fit_ub(i) = best_fit(i);
            } else {
                fit_lb(i) = metric.GetSysts().spline_has_restrict[si] ? metric.GetSysts().spline_restrict_lo[si] : metric.GetSysts().spline_lo[si];
                fit_ub(i) = metric.GetSysts().spline_has_restrict[si] ? metric.GetSysts().spline_restrict_hi[si] : metric.GetSysts().spline_hi[si];
            }
        }
        // Scan fits use the configured gradient mode (default analytic — exact
        // and the cheapest mode; the FD fallback applies where it is not implemented).
        metric.setGradientMode(fitconfig.gradient_mode);
    }

    Eigen::VectorXf grad = Eigen::VectorXf::Constant(best_fit.size(), 0);

    // Trial ladder for each non-frequency physics parameter (typically the
    // oscillation amplitude). The plain BF slice is blind to basins whose
    // depth only appears at a different amplitude: with the BF amplitude
    // near zero, chi2 barely depends on the frequency even when a
    // (freq, amplitude) pair several chi2 units deeper exists. Taking the
    // minimum over a coarse per-dimension ladder at every scan point
    // recovers that depth with pure evals and hands every refit a start
    // inside the right basin.
    std::vector<std::pair<size_t, std::vector<float>>> phys_ladders;
    if(fitconfig.harmonic_phys_ladder > 1) {
        for(size_t j = 0; j < nphys; ++j) {
            if(j == osc_par) continue;
            const float jlo = std::isfinite(metric.GetModel().lb(j)) ? metric.GetModel().lb(j) : -3.0f;
            const float jhi = std::isfinite(metric.GetModel().ub(j)) ? metric.GetModel().ub(j) : 3.0f;
            if(!(jhi > jlo)) continue;
            std::vector<float> vals;
            for(size_t t = 0; t < fitconfig.harmonic_phys_ladder; ++t)
                vals.push_back(jlo + (jhi - jlo) * (float)t / (float)(fitconfig.harmonic_phys_ladder - 1));
            phys_ladders.emplace_back(j, std::move(vals));
        }
    }

    // Best slice at frequency k: the BF slice plus single-dimension ladder
    // variations of each non-frequency physics parameter.
    auto slice_candidate = [&](float k) -> std::pair<float, Eigen::VectorXf> {
        Eigen::VectorXf x0 = best_fit;
        x0(osc_par) = k;
        float best = metric(x0, grad, false);
        if(!std::isfinite(best)) best = std::numeric_limits<float>::infinity();
        Eigen::VectorXf bx = x0;
        for(const auto &[j, vals] : phys_ladders) {
            for(float v : vals) {
                Eigen::VectorXf x = x0;
                x(j) = v;
                const float fx = metric(x, grad, false);
                if(std::isfinite(fx) && fx < best) { best = fx; bx = x; }
            }
        }
        return {best, bx};
    };

    // Evaluate/fit a single scan point. Fit modes record the warm start's own
    // chi2 first, so a solver throw degrades to that value instead of leaving
    // a hole (a +inf hole fabricates sign changes for the minima finder).
    auto scan_point = [&](float k, const Eigen::VectorXf &warm) -> std::pair<float, Eigen::VectorXf> {
        if(scan_mode == 0) {
            return slice_candidate(k);
        }
        fit_lb(osc_par) = k;
        fit_ub(osc_par) = k;
        metric.setBounds(fit_lb, fit_ub);
        Eigen::VectorXf g0 = Eigen::VectorXf::Zero(best_fit.size());

        // Slice candidate (BF + ladder): floors every point, so a stalled or
        // throwing continuation fit can never record worse than the slice.
        // Without it a stretch of solver throws leaves the curve evaluated
        // on a distant neighbour's physics.
        auto [f_slice, x_slice_raw] = slice_candidate(k);
        Eigen::VectorXf x_slice = x_slice_raw.cwiseMax(fit_lb).cwiseMin(fit_ub);
        x_slice(osc_par) = k;
        float best = std::isfinite(f_slice) ? f_slice : std::numeric_limits<float>::infinity();
        Eigen::VectorXf bx = x_slice;

        Eigen::VectorXf x_warm = warm.cwiseMax(fit_lb).cwiseMin(fit_ub);
        x_warm(osc_par) = k;
        const bool warm_differs = (x_warm - x_slice).squaredNorm() > 0;
        if(warm_differs) {
            const float f_warm = metric(x_warm, g0, false);
            if(std::isfinite(f_warm) && f_warm < best) { best = f_warm; bx = x_warm; }
        }

        float fit_fx = std::numeric_limits<float>::infinity();
        try {
            Eigen::VectorXf xf = x_warm;
            solver.minimize(metric, xf, fit_fx, fit_lb, fit_ub);
            if(std::isfinite(fit_fx) && fit_fx < best) { best = fit_fx; bx = xf; }
        } catch (const std::exception &except) {
            exception_string_map[std::string(except.what())]++;
            fit_fx = std::numeric_limits<float>::infinity();
        }
        // Cold-start retry from the slice when the warm-started fit threw or
        // ended above the slice candidate (continuation left the basin).
        if(warm_differs && !(fit_fx <= f_slice + 1e-3f)) {
            try {
                Eigen::VectorXf xf = x_slice;
                float fx = std::numeric_limits<float>::infinity();
                solver.minimize(metric, xf, fx, fit_lb, fit_ub);
                if(std::isfinite(fx) && fx < best) { best = fx; bx = xf; }
            } catch (const std::exception &except) {
                exception_string_map[std::string(except.what())]++;
            }
        }
        return {best, bx};
    };

    //STEP 1: scan the frequency axis. Two center-out half-sweeps from the
    //global BF so continuation warm starts stay on their own side of the
    //best-fit basin (mode 0 ignores the warm start beyond the BF slice).
    const int npts = (int)test_p.size();
    std::vector<float> chivalues(npts, std::numeric_limits<float>::infinity());
    std::vector<Eigen::VectorXf> scan_x(npts);

    int center = 0;
    for(int i = 1; i < npts; ++i)
        if(std::fabs(test_p[i] - best_fit(osc_par)) < std::fabs(test_p[center] - best_fit(osc_par))) center = i;

    Eigen::VectorXf warm = best_fit;
    for(int i = center; i < npts; ++i) {
        auto [fx, x] = scan_point(test_p[i], warm);
        chivalues[i] = fx;
        scan_x[i] = x;
        if(scan_mode > 0 && std::isfinite(fx)) warm = x;
        if(run_progress) progress->increment_bar(3);
    }
    warm = best_fit;
    for(int i = center - 1; i >= 0; --i) {
        auto [fx, x] = scan_point(test_p[i], warm);
        chivalues[i] = fx;
        scan_x[i] = x;
        if(scan_mode > 0 && std::isfinite(fx)) warm = x;
        if(run_progress) progress->increment_bar(3);
    }

    // Adaptive refinement: a basin narrower than the ladder shows up as a
    // large jump between neighbours long before it is resolved; insert
    // midpoints there and re-evaluate.
    const float min_gap = std::max(w2 * 0.25f, 1e-4f);
    for(size_t round = 0; round < fitconfig.harmonic_refine_rounds; ++round) {
        std::vector<float> new_pos;
        for(size_t i = 0; i + 1 < test_p.size(); ++i) {
            if(!std::isfinite(chivalues[i]) || !std::isfinite(chivalues[i+1])) continue;
            if(std::fabs(chivalues[i+1] - chivalues[i]) > fitconfig.harmonic_refine_dchi
                    && (test_p[i+1] - test_p[i]) > 2.0f * min_gap)
                new_pos.push_back(0.5f * (test_p[i] + test_p[i+1]));
            if((int)new_pos.size() >= n_dense) break;
        }
        if(new_pos.empty()) break;
        for(float k : new_pos) {
            auto it = std::lower_bound(test_p.begin(), test_p.end(), k);
            const size_t ins = it - test_p.begin();
            // Warm-start from the nearest already-evaluated neighbour.
            const size_t nb = (ins > 0 && (ins >= test_p.size()
                        || std::fabs(test_p[ins-1] - k) < std::fabs(test_p[ins] - k))) ? ins - 1 : ins;
            const Eigen::VectorXf &w0 = (scan_mode > 0 && scan_x[nb].size()) ? scan_x[nb] : best_fit;
            auto [fx, x] = scan_point(k, w0);
            test_p.insert(test_p.begin() + ins, k);
            chivalues.insert(chivalues.begin() + ins, fx);
            scan_x.insert(scan_x.begin() + ins, x);
        }
        log<LOG_INFO>(L"%1% || Harmonic refinement round %2%: added %3% midpoints (dchi threshold %4%).")
            % __func__ % (round + 1) % new_pos.size() % fitconfig.harmonic_refine_dchi;
    }

    // Keep only finite points; store the curve for diagnostics (persisted as
    // a TGraph by draw_fit_result).
    std::vector<float> chipos, chivals;
    std::vector<size_t> scan_idx;
    for(size_t i = 0; i < test_p.size(); ++i) {
        if(!std::isfinite(chivalues[i])) continue;
        chipos.push_back(test_p[i]);
        chivals.push_back(chivalues[i]);
        scan_idx.push_back(i);
    }
    harmonic_scan_pos = chipos;
    harmonic_scan_chi = chivals;
    log<LOG_DEBUG>(L"%1% || Harmonic scan positions: %2%") % __func__ % harmonic_scan_pos;
    log<LOG_DEBUG>(L"%1% || Harmonic scan chi2s: %2%") % __func__ % harmonic_scan_chi;

    //#STEP 2 From the scan above, find local minima in freq
    std::vector<std::pair<float,float>> minima = findSignificantMinima(chipos, chivals, true);

    // Refit window: a scan minimum this far above the best scan point cannot
    // become a useful seed; skip its two LBFGS refits.
    if(!chivals.empty()) {
        const float best_scan_chi = *std::min_element(chivals.begin(), chivals.end());
        const size_t n_all = minima.size();
        minima.erase(std::remove_if(minima.begin(), minima.end(),
                    [&](const std::pair<float,float> &m){ return m.second > best_scan_chi + fitconfig.harmonic_refit_window; }),
                minima.end());
        if(minima.size() < n_all)
            log<LOG_INFO>(L"%1% || Dropped %2% scan minima more than %3% chi2 above the best scan point (%4%).")
                % __func__ % (n_all - minima.size()) % fitconfig.harmonic_refit_window % best_scan_chi;
    }


    //STEP 3, loop over all mimima and do twofold minimzation.
    //First with DM minima fixed to get BF of pull terms, then fully free to optimize the mass splitting to high precisin

    Eigen::VectorXf full_lb(nparams), full_ub(nparams);
    for(size_t i = 0; i < nphys; ++i) {
        full_lb(i) = metric.GetModel().lb(i);
        full_ub(i) = metric.GetModel().ub(i);
    }
    for(size_t i = nphys; i < nparams; ++i) {
        size_t si = i - nphys;
        full_lb(i) = metric.GetSysts().spline_has_restrict[si] ? metric.GetSysts().spline_restrict_lo[si] : metric.GetSysts().spline_lo[si];
        full_ub(i) = metric.GetSysts().spline_has_restrict[si] ? metric.GetSysts().spline_restrict_hi[si] : metric.GetSysts().spline_hi[si];
    }

    for(size_t p=0;p<minima.size();p++){
        log<LOG_INFO>(L"%1% || ##################  ") %__func__;

        const float mpos = minima.at(p).first;

        // Start from the scan's own best point at this frequency (the fitted
        // point in modes 1/2; the ladder-best slice in mode 0 -- crucially
        // NOT the BF slice, whose amplitude can sit in the wrong basin). The
        // scan chi2 is the guaranteed candidate: a refit throw degrades to
        // it instead of discarding the minimum (LBFGS mutates its start in
        // place, so each stage gets a fresh copy).
        Eigen::VectorXf start = best_fit;
        {
            auto it = std::lower_bound(chipos.begin(), chipos.end(), mpos);
            if(it != chipos.end() && std::fabs(*it - mpos) < 1e-6f) {
                const size_t src = scan_idx[it - chipos.begin()];
                if(scan_x[src].size()) start = scan_x[src];
            }
        }
        start(osc_par) = mpos;

        float best_chi = minima.at(p).second;
        Eigen::VectorXf best_x = start.cwiseMax(full_lb).cwiseMin(full_ub);
        best_x(osc_par) = mpos;

        // Stage a: frequency pinned, everything else free. Mode 2's scan
        // already profiled every parameter at exactly this frequency, so it
        // skips straight to the floating-frequency polish.
        if(scan_mode != 2) {
            full_lb(osc_par) = mpos;
            full_ub(osc_par) = mpos;
            metric.setBounds(full_lb, full_ub);
            try{
                Eigen::VectorXf xf = best_x;
                float fx = std::numeric_limits<float>::infinity();
                solver.minimize(metric, xf, fx, full_lb, full_ub);
                if(std::isfinite(fx) && fx < best_chi) { best_chi = fx; best_x = xf; }
            } catch (const std::exception &except) {
                exception_string_map[std::string(except.what())]++;
            }

            log<LOG_INFO>(L"%1% || Local min num (%2%) with FIXED frequency (%3%) has chi %4% ") %__func__% p % mpos % best_chi;
            log<LOG_INFO>(L"%1% || -- at bf pt %2%  ") %__func__%  best_x;
        }

        //free the freq and repeat
        full_lb(osc_par)=metric.GetModel().lb(osc_par);
        full_ub(osc_par)=metric.GetModel().ub(osc_par);
        metric.setBounds(full_lb,full_ub);

        try{
            Eigen::VectorXf xf = best_x;
            float fx = std::numeric_limits<float>::infinity();
            solver.minimize(metric, xf, fx, full_lb, full_ub);
            if(std::isfinite(fx) && fx < best_chi) { best_chi = fx; best_x = xf; }
        } catch (const std::exception &except) {
            exception_string_map[std::string(except.what())]++;
        }

        log<LOG_INFO>(L"%1% || Local min num (%2%) with floating frequency (%3%) has chi %4% (but bf freq was %5%) ") %__func__% p % mpos % best_chi % best_x(osc_par);
        log<LOG_INFO>(L"%1% || -- at bf pt %2%  ") %__func__%  best_x;

        if(std::isfinite(best_chi)) {
            freq_seed_points.push_back(best_x);
            freq_seed_values.push_back(best_chi);
        }
        if(run_progress){
            for(int jj=0;jj<std::ceil(100/minima.size())+1; jj++)progress->increment_bar(4);
        }
    }

    //ensure best fit is in the seeds, most likely mreoved in next step
    freq_seed_points.push_back(best_fit);
    float chibf = metric(best_fit, grad, false);
    freq_seed_values.push_back(chibf);

    log<LOG_INFO>(L"%1% || ##################  ") %__func__;
    log<LOG_DEBUG>(L"%1% || We have calculated %2% frequency seed points in total, plus 1 for global BF. Checking norm for differences! ") %__func__% freq_seed_values.size();

    std::vector<bool> keep(freq_seed_values.size(), true);
    float norm_tol = fitconfig.harmonic_seed_norm_tolerance; 
    float chi_tol = fitconfig.harmonic_seed_chi_tolerence;    

    for(size_t p = 0; p < freq_seed_values.size(); p++){
        if(!keep[p]) continue;  // needa to skip if already marked for del

        for(size_t q = p + 1; q < freq_seed_values.size(); q++){  
            if(!keep[q]) continue;

            float chip = freq_seed_values.at(p);
            float chiq = freq_seed_values.at(q);
            float normpq = (freq_seed_points.at(p) - freq_seed_points.at(q)).norm();
            float freq_sep = std::fabs(freq_seed_points.at(p)(osc_par) - freq_seed_points.at(q)(osc_par));

            log<LOG_DEBUG>(L"%1% || NORM freq seed (%2%,%3%) is : %4%, chi_diff: %5%, freq_sep: %6%") % __func__ % p % q % normpq % std::abs(chip - chiq) % freq_sep;

            // Two refits inside the same frequency basin are one seed: each
            // survivor costs one LBFGS pass in every downstream scan fit, so
            // dedup at basin level (freq separation), not just exact match.
            if((freq_sep < fitconfig.harmonic_min_spacing_log) ||
               (normpq < norm_tol && std::abs(chip - chiq) < chi_tol)) {
                if(chip <= chiq) {
                    keep[q] = false;
                    log<LOG_DEBUG>(L"%1% || Removing duplicate %2% (keeping %3%)") % __func__ % q % p;
                } else {
                    keep[p] = false;
                    log<LOG_DEBUG>(L"%1% || Removing duplicate %2% (keeping %3%)") % __func__ % p % q;
                }
            }
        }
    }

    std::vector<Eigen::VectorXf> unique_points;
    std::vector<float> unique_values;
    for(size_t i = 0; i < keep.size(); i++) {
        if(keep[i]) {
            unique_points.push_back(freq_seed_points[i]);
            unique_values.push_back(freq_seed_values[i]);
        }
    }

    // Sort
    std::vector<size_t> sort_indices(unique_values.size());
    std::iota(sort_indices.begin(), sort_indices.end(), 0);
    std::sort(sort_indices.begin(), sort_indices.end(),[&](size_t i, size_t j) { return unique_values[i] < unique_values[j]; });

    std::vector<Eigen::VectorXf> sorted_points;
    std::vector<float> sorted_values;
    for(size_t idx : sort_indices) {
        sorted_points.push_back(unique_points[idx]);
        sorted_values.push_back(unique_values[idx]);
    }

    // Replace originals
    freq_seed_points = sorted_points;
    freq_seed_values = sorted_values;

    log<LOG_INFO>(L"%1% || Reduced from %2% to %3% unique seed points that are kept for future use!")   % __func__ % keep.size() % unique_points.size();

    metric.setBounds(saved_lb, saved_ub);
    metric.setGradientMode(saved_gmode);

    return freq_seed_values.size();
}

std::vector<std::pair<float, float>> PROfitter::findSignificantMinima(const std::vector<float>& x_values, const std::vector<float>& y_values, bool use_log_spacing){
    if (x_values.size() != y_values.size() || x_values.size() < 3)
        return {};

    // Non-finite scan values fabricate sign changes on both sides and pass
    // any prominence cut; drop them before differencing.
    std::vector<float> xs, ys;
    xs.reserve(x_values.size());
    ys.reserve(y_values.size());
    for (size_t i = 0; i < y_values.size(); ++i) {
        if (std::isfinite(x_values[i]) && std::isfinite(y_values[i])) {
            xs.push_back(x_values[i]);
            ys.push_back(y_values[i]);
        }
    }
    if (xs.size() < 3)
        return {};

    const size_t n = ys.size();
    // Scan positions are already log10(dm2); spacing is a plain x difference.
    (void)use_log_spacing;

    // ---- 0-dimensional topological persistence over the 1D curve ----
    // Activate points from lowest y upward, union-finding adjacent active
    // runs. When two runs merge at a saddle, the run with the shallower
    // minimum "dies" there: persistence = saddle height - its minimum. Every
    // basin gets exactly ONE representative minimum (no same-basin
    // duplicates) and an unambiguous significance, edge basins included --
    // with no walk cutoffs, threshold relaxation, or spacing decay to tune.
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&ys](size_t a, size_t b) {
        return ys[a] != ys[b] ? ys[a] < ys[b] : a < b;
    });

    std::vector<int> parent(n, -1);   // -1 = not yet activated
    std::vector<size_t> comp_min(n);  // root -> index of the component's lowest point
    auto find_root = [&parent](int a) {
        while (parent[a] != a) {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        return a;
    };

    std::vector<std::pair<size_t, float>> basins; // (min index, persistence)
    for (size_t idx : order) {
        parent[idx] = (int)idx;
        comp_min[idx] = idx;
        for (int nb : {(int)idx - 1, (int)idx + 1}) {
            if (nb < 0 || nb >= (int)n || parent[nb] == -1) continue;
            int r1 = find_root((int)idx), r2 = find_root(nb);
            if (r1 == r2) continue;
            const int keep = ys[comp_min[r1]] <= ys[comp_min[r2]] ? r1 : r2;
            const int die = (keep == r1) ? r2 : r1;
            const float pers = ys[idx] - ys[comp_min[die]];
            // Zero-persistence deaths are non-minima (or plateau ties), not basins.
            if (pers > 0) basins.emplace_back(comp_min[die], pers);
            parent[die] = keep;
        }
    }
    const float y_min = ys[order.front()];
    const float y_range = ys[order.back()] - y_min;
    // The surviving component is the global minimum's basin; give it top rank.
    basins.emplace_back(comp_min[find_root((int)order.front())], std::numeric_limits<float>::infinity());

    // Significance threshold: relative to the curve's own dynamic range, so
    // shallow-but-clear structure on nearly-flat curves (e.g. projected fits
    // whose slice barely depends on the frequency) is kept; floored against
    // float noise; capped by the absolute chi2 threshold so strongly
    // structured curves keep their shallow real harmonics.
    const float threshold = std::min(fitconfig.harmonic_prominence_threshold,
            std::max(fitconfig.harmonic_persistence_rel * y_range, fitconfig.harmonic_persistence_floor));

    const float min_spacing = fitconfig.harmonic_min_spacing_log;
    auto far_from = [&](const std::vector<std::pair<float,float>> &sel, float xv) {
        for (const auto &m : sel)
            if (std::abs(m.first - xv) < min_spacing) return false;
        return true;
    };

    // Select significant basins, deepest first, respecting min spacing.
    std::vector<std::pair<size_t, float>> significant, insignificant;
    for (const auto &b : basins)
        (b.second >= threshold ? significant : insignificant).push_back(b);
    std::sort(significant.begin(), significant.end(),
            [&ys](const auto &a, const auto &b) { return ys[a.first] < ys[b.first]; });

    std::vector<std::pair<float, float>> minima;
    for (const auto &[idx, pers] : significant) {
        if (minima.size() >= fitconfig.harmonic_max_num_seeds) break;
        if (far_from(minima, xs[idx])) minima.emplace_back(xs[idx], ys[idx]);
    }
    const size_t n_significant = minima.size();
    if (significant.size() > n_significant)
        log<LOG_INFO>(L"%1% || %2% significant basins beyond the max-seed/spacing cuts were dropped.")
            % __func__ % (significant.size() - n_significant);

    // Top-up to the requested minimum count from the MOST PERSISTENT
    // remaining basins (not the lowest raw points, which just shadow the
    // global minimum's own basin).
    if (minima.size() < fitconfig.harmonic_min_num_seeds) {
        std::sort(insignificant.begin(), insignificant.end(),
                [](const auto &a, const auto &b) { return a.second > b.second; });
        for (const auto &[idx, pers] : insignificant) {
            if (minima.size() >= fitconfig.harmonic_min_num_seeds) break;
            if (far_from(minima, xs[idx])) minima.emplace_back(xs[idx], ys[idx]);
        }
        if (minima.size() > n_significant)
            log<LOG_INFO>(L"%1% || Topped up minima from %2% to %3% with the most persistent sub-threshold basins (requested minimum %4%).")
                % __func__ % n_significant % minima.size() % fitconfig.harmonic_min_num_seeds;
    }

    std::sort(minima.begin(), minima.end(), [](const auto& a, const auto& b) { return a.second < b.second; });

    log<LOG_INFO>(L"%1% || Persistence minima finder: %2% basins, %3% significant at threshold %4% (cap %5%, rel %6% x range %7%, floor %8%); %9% selected.")
        % __func__ % basins.size() % n_significant % threshold
        % fitconfig.harmonic_prominence_threshold % fitconfig.harmonic_persistence_rel
        % y_range % fitconfig.harmonic_persistence_floor % minima.size();
    for(const auto &[mx, my] : minima)
        log<LOG_INFO>(L"%1% ||   minimum at %2%, chi2 %3% (dchi2 %4%)") % __func__ % mx % my % (my - y_min);

    return minima;
}

