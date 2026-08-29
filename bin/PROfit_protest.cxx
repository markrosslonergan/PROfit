#include "PROfit_common.h"

void run_test(const PROconfig &config, const PROpeller &prop, PROmetric &metric, const PROdata &data, const Eigen::VectorXf &CVParams) {
    log<LOG_INFO>(L"%1% || PROtest: Testing FillSpectra with fixed seed random spline throws") % __func__;
    
    size_t n_splines = metric.GetSysts().GetNSplines();
    size_t n_tests = 3;
    
    // Fixed seed for reproducibility across code versions
    std::mt19937 test_rng(12345);
    std::normal_distribution<float> d(0.0f, 1.0f);
    
    for(size_t test = 0; test < n_tests; ++test) {
        log<LOG_INFO>(L"%1% || ===== TEST %2% =====") % __func__ % (test + 1);
        
        // Create params: physics from CVParams, random splines
        Eigen::VectorXf testParams = Eigen::VectorXf::Zero(metric.GetModel().nparams + n_splines);
        for(size_t i = 0; i < metric.GetModel().nparams; ++i) {
            testParams(i) = CVParams(i);
        }
        for(size_t i = 0; i < n_splines; ++i) {
            testParams(metric.GetModel().nparams + i) = d(test_rng);
        }

        log<LOG_INFO>(L"%1% || Test %2% Parameters:") % __func__ % (test + 1);
        for(Eigen::Index i = 0; i < testParams.size(); ++i) {
            log<LOG_INFO>(L"%1% || Test %2% Parameter %3%: %4%") % __func__ % (test + 1) % i % testParams(i);
        }
        
        // Fill spectrum
        PROspec spec = FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), testParams, true, config.i_prime);
        
        // Print all bin values
        log<LOG_INFO>(L"%1% || Test %2% Spectrum (%3% bins):") % __func__ % (test + 1) % spec.Spec().size();
        for(long b = 0; b < spec.Spec().size(); ++b) {
            log<LOG_INFO>(L"%1% || Test %2% bin %3%: %4%") % __func__ % (test + 1) % b % spec.Spec()(b);
        }
        log<LOG_INFO>(L"%1% || Test %2% Total: %3%") % __func__ % (test + 1) % spec.Spec().sum();
    }

    // ---- chi2 algorithm comparison: 4 paths exercising covariance-build and inverse-vs-solve ----
    log<LOG_INFO>(L"%1% || ===== chi2 algorithm comparison =====") % __func__;
    const PROsyst &cmp_syst = metric.GetSysts();
    const Eigen::MatrixXf &cmp_fcov = cmp_syst.fractional_covariance;
    const Eigen::VectorXf cmp_data = data.Spec();

    std::vector<Eigen::Index> cmp_nonempty;
    for(Eigen::Index i = 0; i < cmp_data.size(); ++i)
        if(cmp_data(i) > 0) cmp_nonempty.push_back(i);
    const size_t cmp_red = cmp_nonempty.size();
    if(cmp_red == 0) {
        log<LOG_ERROR>(L"%1% || All data bins empty, skipping chi2 comparison.") % __func__;
    } else {
        const size_t cmp_N = 200;
        std::vector<Eigen::VectorXf> cmp_params;
        cmp_params.reserve(cmp_N);
        std::mt19937 cmp_rng(67890);
        std::normal_distribution<float> cmp_d(0.0f, 1.0f);
        for(size_t k = 0; k < cmp_N; ++k) {
            Eigen::VectorXf p = Eigen::VectorXf::Zero(metric.GetModel().nparams + n_splines);
            for(size_t i = 0; i < metric.GetModel().nparams; ++i) p(i) = CVParams(i);
            for(size_t i = 0; i < n_splines; ++i) p(metric.GetModel().nparams + i) = cmp_d(cmp_rng);
            cmp_params.push_back(p);
        }

        std::vector<float> chi2_old(cmp_N), chi2_mid(cmp_N), chi2_new(cmp_N), chi2_alt(cmp_N);
        Eigen::MatrixXf cmp_red_stat(cmp_red, cmp_red);

        // OLD: dense diag*F*diag + .inverse()
        auto cmp_t0 = std::chrono::high_resolution_clock::now();
        for(size_t k = 0; k < cmp_N; ++k) {
            PROspec r = FillSpectra(config, prop, cmp_syst, metric.GetModel(), cmp_params[k], true, config.i_prime);
            Eigen::MatrixXf diag = r.Spec().array().matrix().asDiagonal();
            Eigen::MatrixXf full_cov = diag * cmp_fcov * diag;
            Eigen::MatrixXf coll = CollapseMatrix(config, full_cov);
            Eigen::MatrixXf stat = cmp_data.matrix().asDiagonal();
            Eigen::MatrixXf red_full(cmp_red, cmp_red);
            for(size_t i = 0; i < cmp_red; ++i)
                for(size_t j = 0; j < cmp_red; ++j) {
                    red_full(i,j) = coll(cmp_nonempty[i], cmp_nonempty[j]);
                    cmp_red_stat(i,j) = stat(cmp_nonempty[i], cmp_nonempty[j]);
                }
            Eigen::MatrixXf inv = (cmp_red_stat + red_full).inverse();
            Eigen::VectorXf coll_mc = CollapseMatrix(config, r.Spec());
            Eigen::VectorXf delta(cmp_red);
            for(size_t i = 0; i < cmp_red; ++i)
                delta(i) = coll_mc(cmp_nonempty[i]) - cmp_data(cmp_nonempty[i]);
            chi2_old[k] = (delta.transpose() * inv * delta)(0,0);
        }
        auto cmp_t1 = std::chrono::high_resolution_clock::now();

        // INTERMEDIATE: cwiseProduct (fast covariance build) + .inverse() (slow inversion)
        for(size_t k = 0; k < cmp_N; ++k) {
            PROspec r = FillSpectra(config, prop, cmp_syst, metric.GetModel(), cmp_params[k], true, config.i_prime);
            const Eigen::VectorXf &s = r.Spec();
            Eigen::MatrixXf full_cov = (s * s.transpose()).cwiseProduct(cmp_fcov);
            Eigen::MatrixXf coll = CollapseMatrix(config, full_cov);
            Eigen::MatrixXf red_full(cmp_red, cmp_red);
            for(size_t i = 0; i < cmp_red; ++i)
                for(size_t j = 0; j < cmp_red; ++j) {
                    red_full(i,j) = coll(cmp_nonempty[i], cmp_nonempty[j]);
                    cmp_red_stat(i,j) = (i == j) ? cmp_data(cmp_nonempty[i]) : 0.0f;
                }
            Eigen::MatrixXf inv = (cmp_red_stat + red_full).inverse();
            Eigen::VectorXf coll_mc = CollapseMatrix(config, r.Spec());
            Eigen::VectorXf delta(cmp_red);
            for(size_t i = 0; i < cmp_red; ++i)
                delta(i) = coll_mc(cmp_nonempty[i]) - cmp_data(cmp_nonempty[i]);
            chi2_mid[k] = (delta.transpose() * inv * delta)(0,0);
        }
        auto cmp_t2 = std::chrono::high_resolution_clock::now();

        // NEW: cwiseProduct + .llt().solve()
        for(size_t k = 0; k < cmp_N; ++k) {
            PROspec r = FillSpectra(config, prop, cmp_syst, metric.GetModel(), cmp_params[k], true, config.i_prime);
            const Eigen::VectorXf &s = r.Spec();
            Eigen::MatrixXf full_cov = (s * s.transpose()).cwiseProduct(cmp_fcov);
            Eigen::MatrixXf coll = CollapseMatrix(config, full_cov);
            Eigen::MatrixXf red_full(cmp_red, cmp_red);
            for(size_t i = 0; i < cmp_red; ++i)
                for(size_t j = 0; j < cmp_red; ++j) {
                    red_full(i,j) = coll(cmp_nonempty[i], cmp_nonempty[j]);
                    cmp_red_stat(i,j) = (i == j) ? cmp_data(cmp_nonempty[i]) : 0.0f;
                }
            Eigen::MatrixXf M = cmp_red_stat + red_full;
            Eigen::VectorXf coll_mc = CollapseMatrix(config, r.Spec());
            Eigen::VectorXf delta(cmp_red);
            for(size_t i = 0; i < cmp_red; ++i)
                delta(i) = coll_mc(cmp_nonempty[i]) - cmp_data(cmp_nonempty[i]);
            chi2_new[k] = delta.dot(M.llt().solve(delta));
        }
        auto cmp_t3 = std::chrono::high_resolution_clock::now();

        // ALT: inline asDiagonal()*F*asDiagonal() (no materialization) + .llt().solve() (matches upstream's covariance build)
        for(size_t k = 0; k < cmp_N; ++k) {
            PROspec r = FillSpectra(config, prop, cmp_syst, metric.GetModel(), cmp_params[k], true, config.i_prime);
            Eigen::MatrixXf full_cov = r.Spec().asDiagonal() * cmp_fcov * r.Spec().asDiagonal();
            Eigen::MatrixXf coll = CollapseMatrix(config, full_cov);
            Eigen::MatrixXf red_full(cmp_red, cmp_red);
            for(size_t i = 0; i < cmp_red; ++i)
                for(size_t j = 0; j < cmp_red; ++j) {
                    red_full(i,j) = coll(cmp_nonempty[i], cmp_nonempty[j]);
                    cmp_red_stat(i,j) = (i == j) ? cmp_data(cmp_nonempty[i]) : 0.0f;
                }
            Eigen::MatrixXf M = cmp_red_stat + red_full;
            Eigen::VectorXf coll_mc = CollapseMatrix(config, r.Spec());
            Eigen::VectorXf delta(cmp_red);
            for(size_t i = 0; i < cmp_red; ++i)
                delta(i) = coll_mc(cmp_nonempty[i]) - cmp_data(cmp_nonempty[i]);
            chi2_alt[k] = delta.dot(M.llt().solve(delta));
        }
        auto cmp_t4 = std::chrono::high_resolution_clock::now();

        double cmp_old_s = std::chrono::duration<double>(cmp_t1 - cmp_t0).count();
        double cmp_mid_s = std::chrono::duration<double>(cmp_t2 - cmp_t1).count();
        double cmp_new_s = std::chrono::duration<double>(cmp_t3 - cmp_t2).count();
        double cmp_alt_s = std::chrono::duration<double>(cmp_t4 - cmp_t3).count();

        auto worst_diff = [&](const std::vector<float> &ref, const std::vector<float> &test) {
            double max_abs = 0, max_rel = 0;
            size_t max_rel_k = 0;
            for(size_t k = 0; k < cmp_N; ++k) {
                double a = std::abs(double(test[k]) - double(ref[k]));
                double denom = std::max(std::abs(double(ref[k])), 1e-30);
                double r = a / denom;
                if(a > max_abs) max_abs = a;
                if(r > max_rel) { max_rel = r; max_rel_k = k; }
            }
            return std::make_tuple(max_abs, max_rel, max_rel_k);
        };
        auto [mid_abs, mid_rel, mid_k] = worst_diff(chi2_old, chi2_mid);
        auto [new_abs, new_rel, new_k] = worst_diff(chi2_old, chi2_new);
        auto [alt_abs, alt_rel, alt_k] = worst_diff(chi2_old, chi2_alt);

        log<LOG_INFO>(L"%1% || ----- chi2 algorithm comparison: 4 paths, identical parameter sets -----") % __func__;
        log<LOG_INFO>(L"%1% || N = %2% chi2 evaluations per path; reduced bin count = %3% (after dropping empty-data bins)") % __func__ % cmp_N % cmp_red;
        log<LOG_INFO>(L"%1% ||") % __func__;
        log<LOG_INFO>(L"%1% || Path 1 (OLD)         : full_cov = diag(spec) * F * diag(spec)        [dense diag materialized -> O(N^3) matmuls]") % __func__;
        log<LOG_INFO>(L"%1% ||                        chi2     = delta^T * (M).inverse() * delta    [O(N^3) inverse + 2 matvec]") % __func__;
        log<LOG_INFO>(L"%1% || Path 2 (INTERMEDIATE): full_cov = (spec * spec^T).cwiseProduct(F)    [O(N^2): outer product + Hadamard]") % __func__;
        log<LOG_INFO>(L"%1% ||                        chi2     = delta^T * (M).inverse() * delta    [same inverse step as OLD]") % __func__;
        log<LOG_INFO>(L"%1% || Path 3 (NEW)         : full_cov = (spec * spec^T).cwiseProduct(F)    [same as INTERMEDIATE]") % __func__;
        log<LOG_INFO>(L"%1% ||                        chi2     = delta . M.llt().solve(delta)       [Cholesky solve, no explicit inverse]") % __func__;
        log<LOG_INFO>(L"%1% || Path 4 (ALT)         : full_cov = spec.asDiagonal()*F*spec.asDiagonal() [Eigen DiagonalWrapper, matches upstream build]") % __func__;
        log<LOG_INFO>(L"%1% ||                        chi2     = delta . M.llt().solve(delta)       [same solve as NEW]") % __func__;
        log<LOG_INFO>(L"%1% ||") % __func__;
        log<LOG_INFO>(L"%1% || Path 1 (OLD)         : %2% s total, %3% ms/call (baseline)")
            % __func__ % cmp_old_s % (cmp_old_s / cmp_N * 1e3);
        log<LOG_INFO>(L"%1% || Path 2 (INTERMEDIATE): %2% s total, %3% ms/call (%4%x vs OLD)  -> isolates the covariance-build speedup")
            % __func__ % cmp_mid_s % (cmp_mid_s / cmp_N * 1e3) % (cmp_old_s / std::max(cmp_mid_s, 1e-12));
        log<LOG_INFO>(L"%1% || Path 3 (NEW)         : %2% s total, %3% ms/call (%4%x vs OLD)  -> cwiseProduct build + Cholesky solve")
            % __func__ % cmp_new_s % (cmp_new_s / cmp_N * 1e3) % (cmp_old_s / std::max(cmp_new_s, 1e-12));
        log<LOG_INFO>(L"%1% || Path 4 (ALT)         : %2% s total, %3% ms/call (%4%x vs OLD)  -> asDiagonal build + Cholesky solve (= current code)")
            % __func__ % cmp_alt_s % (cmp_alt_s / cmp_N * 1e3) % (cmp_old_s / std::max(cmp_alt_s, 1e-12));
        log<LOG_INFO>(L"%1% ||") % __func__;
        log<LOG_INFO>(L"%1% || NEW vs ALT covariance-build comparison: %2%x (>1 means ALT/asDiagonal is faster)")
            % __func__ % (cmp_new_s / std::max(cmp_alt_s, 1e-12));
        log<LOG_INFO>(L"%1% ||") % __func__;
        log<LOG_INFO>(L"%1% || Numerical agreement (chi2 differences from OLD path):") % __func__;
        log<LOG_INFO>(L"%1% || INTERMEDIATE vs OLD: worst abs diff %2%, worst rel diff %3% (at k=%4%, old=%5%, mid=%6%)")
            % __func__ % mid_abs % mid_rel % mid_k % chi2_old[mid_k] % chi2_mid[mid_k];
        log<LOG_INFO>(L"%1% || NEW          vs OLD: worst abs diff %2%, worst rel diff %3% (at k=%4%, old=%5%, new=%6%)")
            % __func__ % new_abs % new_rel % new_k % chi2_old[new_k] % chi2_new[new_k];
        log<LOG_INFO>(L"%1% || ALT          vs OLD: worst abs diff %2%, worst rel diff %3% (at k=%4%, old=%5%, alt=%6%)")
            % __func__ % alt_abs % alt_rel % alt_k % chi2_old[alt_k] % chi2_alt[alt_k];
    }
}
