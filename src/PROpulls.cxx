#include "PROpulls.h"
#include "PROcess.h"
#include "PROlog.h"
#include "PROtocall.h"
#include "PROwatermark.h"

#include "TCanvas.h"
#include "TColor.h"
#include "TF1.h"
#include "TH1D.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TLine.h"
#include "TMath.h"
#include "TPad.h"
#include "TStyle.h"
#include "TText.h"
#include "TTree.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace PROfit {

namespace {

    constexpr double kInvSqrt2Pi = 0.3989422804014327;

    double normPdf(double x) { return kInvSqrt2Pi * std::exp(-0.5 * x * x); }
    double normCdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }
    double normQuantile(double p) { return TMath::NormQuantile(std::clamp(p, 1e-15, 1.0 - 1e-15)); }

    // phi(a)/Phi(a), with the asymptotic Mills ratio where Phi underflows.
    double millsLower(double a) {
        if(a < -8.0) {
            const double x = -a;
            return x / (1.0 - 1.0 / (x * x) + 3.0 / (x * x * x * x));
        }
        return normPdf(a) / normCdf(a);
    }

    double logNormCdf(double x) {
        if(x < -30.0) return -0.5 * x * x - std::log(-x) - 0.5 * std::log(2.0 * M_PI);
        return std::log(normCdf(x));
    }

    // CDF of theta_hat when the truth is uniform on [lo, hi] and the estimate scatters by
    // sigma about it: (sigma/W) [G((t-lo)/sigma) - G((t-hi)/sigma)], G(x) = x Phi(x) + phi(x).
    double uniformSmearedCdf(double t, double lo, double hi, double sigma) {
        auto G = [](double x) { return x * normCdf(x) + normPdf(x); };
        return sigma / (hi - lo) * (G((t - lo) / sigma) - G((t - hi) / sigma));
    }

    struct CategoryInfo {
        const char *key;
        const char *title;
        const char *xtitle;
        const char *fill;   ///< Pastel fill, one per page.
        const char *line;   ///< Darker shade of the fill for the outline.
    };

    const std::array<CategoryInfo, 4> kCategories = {{
        {"symmetric_splines",
         "(1) Symmetric splines: Gaussian prior, fit range #supseteq #theta_{0} #pm 3#sigma",
         "z = (#hat{#theta} #minus #theta_{0}) / #sqrt{#sigma_{prior}^{2} #minus #sigma_{post}^{2}}",
         "#9CC3EC", "#5E8FC7"},
        {"covariance_modes",
         "(2) Covariance systematics: data-constrained eigenmode pulls",
         "z = #hat{#alpha} / #sqrt{1 #minus #sigma_{post}^{2}}",
         "#A8DCB0", "#5FA86B"},
        {"bounded_splines",
         "(3) Bounded / one-sided splines: a fit bound within 3#sigma of #theta_{0}",
         "z = (#hat{#theta} #minus #theta_{0}) / #sqrt{#sigma_{prior}^{2} #minus #sigma_{post}^{2}}",
         "#F7C99B", "#D18A4A"},
        {"uniform_splines",
         "(4) Other: uniform-prior splines, plus parameters on no page",
         "z = #Phi^{-1}(F(#hat{#theta}))",
         "#CDB4E6", "#8D69B8"},
    }};

    // Unbinned ML fit of N(mu, sigma) to z. Censored entries (cens = -1 / +1) only bound
    // the latent value from above / below; EM brings them in through truncated-normal moments.
    PullGaussFit fitGaussian(const std::vector<double> &z, const std::vector<int> &cens) {
        PullGaussFit f;
        f.n = (int)z.size();
        std::vector<double> zu;
        for(size_t i = 0; i < z.size(); ++i) {
            if(cens[i]) ++f.n_censored;
            else zu.push_back(z[i]);
        }
        if(zu.size() < 2) return f;

        const double n = (double)z.size();
        double mu = std::accumulate(zu.begin(), zu.end(), 0.0) / zu.size();
        double s2 = 0;
        for(double v : zu) s2 += (v - mu) * (v - mu);
        double s = std::sqrt(s2 / zu.size());
        f.fitted = true;
        if(s < 1e-9) {
            f.mu = mu;
            return f;
        }

        if(f.n_censored) {
            for(int it = 0; it < 2000; ++it) {
                double m1 = 0, m2 = 0;
                for(size_t i = 0; i < z.size(); ++i) {
                    double e1 = z[i], var = 0;
                    if(cens[i]) {
                        const double a = (z[i] - mu) / s;
                        if(cens[i] < 0) {
                            const double lam = millsLower(a);
                            e1 = mu - s * lam;
                            var = s * s * std::max(0.0, 1.0 - a * lam - lam * lam);
                        } else {
                            const double lam = millsLower(-a);
                            e1 = mu + s * lam;
                            var = s * s * std::max(0.0, 1.0 + a * lam - lam * lam);
                        }
                    }
                    m1 += e1;
                    m2 += var + e1 * e1;
                }
                const double mu_new = m1 / n;
                const double s_new = std::sqrt(std::max(m2 / n - mu_new * mu_new, 1e-18));
                const bool done = std::abs(mu_new - mu) < 1e-10 && std::abs(s_new - s) < 1e-10;
                mu = mu_new;
                s = s_new;
                if(done) break;
            }
        }
        f.mu = mu;
        f.sigma = s;
        f.mu_err = s / std::sqrt(n);
        f.sigma_err = s / std::sqrt(2.0 * n);
        if(!f.n_censored) return f;

        // Errors from the observed information of the censored likelihood.
        auto ll = [&](double m, double sg) {
            double v = 0;
            for(size_t i = 0; i < z.size(); ++i) {
                const double a = (z[i] - m) / sg;
                if(cens[i] < 0) v += logNormCdf(a);
                else if(cens[i] > 0) v += logNormCdf(-a);
                else v += -0.5 * a * a - std::log(sg);
            }
            return v;
        };
        const double hm = 1e-3 * s, hs = 1e-3 * s;
        const double f0 = ll(mu, s);
        const double hmm = (ll(mu + hm, s) - 2 * f0 + ll(mu - hm, s)) / (hm * hm);
        const double hss = (ll(mu, s + hs) - 2 * f0 + ll(mu, s - hs)) / (hs * hs);
        const double hms = (ll(mu + hm, s + hs) - ll(mu + hm, s - hs) - ll(mu - hm, s + hs) + ll(mu - hm, s - hs)) / (4 * hm * hs);
        const double det = hmm * hss - hms * hms;
        if(hmm < 0 && det > 0) {
            f.mu_err = std::sqrt(-hss / det);
            f.sigma_err = std::sqrt(-hmm / det);
        }
        return f;
    }

    // Inverse of a symmetric positive-definite matrix; eigenvalues clamped if LLT fails.
    Eigen::MatrixXd spdInverse(const Eigen::MatrixXd &A) {
        Eigen::LLT<Eigen::MatrixXd> llt(A);
        if(llt.info() == Eigen::Success)
            return llt.solve(Eigen::MatrixXd::Identity(A.rows(), A.cols()));
        log<LOG_WARNING>(L"%1% || LLT failed on the linearised posterior precision; clamping eigenvalues.") % __func__;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
        const double floor = 1e-12 * std::max(1.0, es.eigenvalues().cwiseAbs().maxCoeff());
        return es.eigenvectors() * es.eigenvalues().cwiseMax(floor).cwiseInverse().asDiagonal() * es.eigenvectors().transpose();
    }

    std::string fmt(const char *f, double a) {
        char buf[128];
        std::snprintf(buf, sizeof buf, f, a);
        return buf;
    }

    // Replace (deleting) an entry of the drawn-object sink.
    void putObject(std::map<std::string, TObject*> *sink, const std::string &name, TObject *obj) {
        auto it = sink->find(name);
        if(it != sink->end()) delete it->second;
        (*sink)[name] = obj;
    }

}

    PullSummary computePostfitPulls(const PROconfig &config, const PROpeller &prop, const PROmetric &metric,
            const Eigen::VectorXf &best_fit, const Eigen::VectorXf &lb, const Eigen::VectorXf &ub,
            const Eigen::VectorXf &data_spec, double min_var_reduction) {
        PullSummary out;
        out.min_var_reduction = min_var_reduction;
        out.shape_only = metric.ShapeOnly();

        const int var = (int)config.i_prime;
        const PROsyst &syst = metric.GetSysts();
        const PROmodel &model = metric.GetModel();
        const size_t nphys = model.nparams, nspl = syst.GetNSplines(), ntot = nphys + nspl;
        const size_t nbins = config.m_num_variable_bins_total_collapsed[var];
        if((size_t)best_fit.size() != ntot || (size_t)lb.size() != ntot || (size_t)ub.size() != ntot || (size_t)data_spec.size() != nbins) {
            log<LOG_ERROR>(L"%1% || Size mismatch (best fit %2%, bounds %3%/%4%, data %5%; expected %6% params, %7% bins); no pull distributions.")
                % __func__ % best_fit.size() % lb.size() % ub.size() % data_spec.size() % ntot % nbins;
            return out;
        }

        // Prediction exactly as the metric compares it: per-channel rescaled onto the data
        // under shape-only. Zero bins floored as in the covariance band/pull code.
        auto predictFull = [&](const Eigen::VectorXf &p) {
            Eigen::VectorXf full = FillSpectra(config, prop, syst, model, p, true, var).Spec();
            if(out.shape_only) full = ShapeRescaleToData(config, full, data_spec, var);
            for(Eigen::Index i = 0; i < full.size(); ++i)
                if(full(i) <= 0.0f) full(i) = 1e-6f;
            return full;
        };
        const Eigen::VectorXf full_bf = predictFull(best_fit);
        const Eigen::VectorXf pred = CollapseMatrix(config, full_bf, var);
        const Eigen::VectorXf stat = metric.GetStatVariances(pred, data_spec, best_fit);

        std::vector<int> bins;
        for(size_t i = 0; i < nbins; ++i)
            if(config.IsBinActive(var, i) && stat(i) > 0) bins.push_back((int)i);
        if(bins.empty()) {
            log<LOG_WARNING>(L"%1% || No bins enter the fit; no pull distributions.") % __func__;
            return out;
        }
        const Eigen::Index nb = (Eigen::Index)bins.size();

        Eigen::MatrixXf L = syst.DecomposeFractionalCovariance(config, full_bf);
        if(out.shape_only) L = Eigen::MatrixXf(ShapeProjectorCollapsed(config, pred, var) * L);
        std::vector<int> modes;
        double total_var = 0;
        for(Eigen::Index j = 0; j < L.cols(); ++j) {
            const double s = L.col(j).squaredNorm();
            total_var += s;
            if(s > 0) modes.push_back((int)j);
        }

        const auto fixedParam = [&](size_t i) { return std::abs(ub((Eigen::Index)i) - lb((Eigen::Index)i)) < 1e-10f; };
        const auto isUniform = [&](size_t s) {
            return s < syst.spline_prior_types.size() && syst.spline_prior_types[s] == SplinePriorType::Uniform;
        };
        const auto plotname = [&](size_t s) {
            auto it = config.m_mcgen_variation_plotname_map.find(syst.spline_names[s]);
            return it != config.m_mcgen_variation_plotname_map.end() ? it->second : syst.spline_names[s];
        };

        std::vector<size_t> free_phys, free_spl, gauss_free, gauss_fixed;
        for(size_t i = 0; i < nphys; ++i)
            if(!fixedParam(i)) free_phys.push_back(i);
        for(size_t s = 0; s < nspl; ++s) {
            if(fixedParam(nphys + s)) {
                out.fixed_names.push_back(plotname(s));
                if(!isUniform(s)) gauss_fixed.push_back(s);
            } else {
                free_spl.push_back(s);
                if(!isUniform(s)) gauss_free.push_back(s);
            }
        }

        // System of free parameters: [physics | splines | covariance modes].
        const Eigen::Index nf = (Eigen::Index)free_phys.size(), ns = (Eigen::Index)free_spl.size();
        const Eigen::Index k = (Eigen::Index)modes.size(), m = nf + ns + k;
        if(ns + k == 0) {
            log<LOG_INFO>(L"%1% || No free nuisance parameters or covariance modes; no pull distributions.") % __func__;
            return out;
        }

        // Jacobian of the collapsed prediction by finite differences: central, one-sided
        // inward at a bound (spline responses are only defined inside the knot range).
        Eigen::MatrixXd J(nb, m);
        auto fdColumn = [&](size_t ip) {
            const float lo = lb((Eigen::Index)ip), hi = ub((Eigen::Index)ip), x0 = best_fit((Eigen::Index)ip);
            const float h = std::max(1e-4f, 1e-3f * (hi - lo));
            const bool can_up = x0 + h <= hi, can_dn = x0 - h >= lo;
            Eigen::VectorXf pu = best_fit, pd = best_fit;
            float span;
            if(can_up && can_dn) { pu((Eigen::Index)ip) = x0 + h; pd((Eigen::Index)ip) = x0 - h; span = 2 * h; }
            else if(can_up)      { pu((Eigen::Index)ip) = x0 + h; span = h; }
            else                 { pd((Eigen::Index)ip) = x0 - h; span = h; }
            const Eigen::VectorXf diff = CollapseMatrix(config, predictFull(pu), var) - CollapseMatrix(config, predictFull(pd), var);
            Eigen::VectorXd col(nb);
            for(Eigen::Index r = 0; r < nb; ++r) col(r) = (double)diff(bins[r]) / span;
            return col;
        };
        for(Eigen::Index a = 0; a < nf; ++a) J.col(a) = fdColumn(free_phys[a]);
        for(Eigen::Index a = 0; a < ns; ++a) J.col(nf + a) = fdColumn(nphys + free_spl[a]);
        for(Eigen::Index j = 0; j < k; ++j)
            for(Eigen::Index r = 0; r < nb; ++r) J(r, nf + ns + j) = L(bins[r], modes[j]);

        Eigen::VectorXd cinv(nb), u(nb);
        for(Eigen::Index r = 0; r < nb; ++r) {
            cinv(r) = 1.0 / stat(bins[r]);
            u(r) = (double)data_spec(bins[r]) - pred(bins[r]);
        }

        // Linearised posterior precision. Physics and uniform-prior splines have flat priors on
        // their box; they enter with the box's own variance W^2/12, so a direction the data
        // barely resolves cannot absorb the rest (exactly flat it would, at any tiny slope).
        // A <model><prior> adds its Gauss-Newton curvature g g^T / sigma^2 (g = dq/dphys).
        Eigen::MatrixXd H = J.transpose() * cinv.asDiagonal() * J;
        for(Eigen::Index a = 0; a < nf; ++a) {
            const double W = ub(free_phys[a]) - lb(free_phys[a]);
            H(a, a) += 12.0 / (W * W);
        }
        for(const auto &p : metric.GetPhysicsPriors()) {
            Eigen::VectorXf dq = Eigen::VectorXf::Zero((Eigen::Index)nphys);
            model.physics_prior_quantity(p, best_fit.head((Eigen::Index)nphys), &dq);
            for(Eigen::Index a = 0; a < nf; ++a)
                for(Eigen::Index b = 0; b < nf; ++b)
                    H(a, b) += (double)dq((Eigen::Index)free_phys[a]) * dq((Eigen::Index)free_phys[b]) / ((double)p.sigma * p.sigma);
        }
        const Eigen::MatrixXd Q = metric.SplinePriorPrecision().cast<double>();
        for(Eigen::Index a = 0; a < ns; ++a) {
            for(Eigen::Index b = 0; b < ns; ++b) H(nf + a, nf + b) += Q(free_spl[a], free_spl[b]);
            if(isUniform(free_spl[a])) {
                const double W = ub(nphys + free_spl[a]) - lb(nphys + free_spl[a]);
                H(nf + a, nf + a) += 12.0 / (W * W);
            }
        }
        for(Eigen::Index j = 0; j < k; ++j) H(nf + ns + j, nf + ns + j) += 1.0;
        const Eigen::MatrixXd Spost = spdInverse(H);

        // Prior of the free Gaussian splines given the fixed ones (only differs from the plain
        // prior when priors are correlated AND some correlated partner is --fix'd).
        const Eigen::Index ng = (Eigen::Index)gauss_free.size(), nx = (Eigen::Index)gauss_fixed.size();
        Eigen::MatrixXd Pcond(ng, ng);
        Eigen::VectorXd ccond(ng);
        {
            Eigen::MatrixXd Qgg(ng, ng), Qgx(ng, nx);
            Eigen::VectorXd dx(nx);
            for(Eigen::Index a = 0; a < ng; ++a) {
                ccond(a) = syst.spline_centers(gauss_free[a]);
                for(Eigen::Index b = 0; b < ng; ++b) Qgg(a, b) = Q(gauss_free[a], gauss_free[b]);
                for(Eigen::Index b = 0; b < nx; ++b) Qgx(a, b) = Q(gauss_free[a], gauss_fixed[b]);
            }
            for(Eigen::Index b = 0; b < nx; ++b)
                dx(b) = best_fit(nphys + gauss_fixed[b]) - syst.spline_centers(gauss_fixed[b]);
            if(ng) {
                Pcond = spdInverse(Qgg);
                if(nx) ccond -= Pcond * (Qgx * dx);
            }
        }

        // Prior covariance over the system (Gaussian splines and modes); V = P - Sigma_post is
        // the covariance of the estimates under the model.
        Eigen::MatrixXd Psys = Eigen::MatrixXd::Zero(m, m);
        std::vector<Eigen::Index> sys_of_gauss(ng);
        for(Eigen::Index a = 0, g = 0; a < ns; ++a)
            if(!isUniform(free_spl[a])) sys_of_gauss[g++] = nf + a;
        for(Eigen::Index a = 0; a < ng; ++a)
            for(Eigen::Index b = 0; b < ng; ++b) Psys(sys_of_gauss[a], sys_of_gauss[b]) = Pcond(a, b);
        for(Eigen::Index j = 0; j < k; ++j) Psys(nf + ns + j, nf + ns + j) = 1.0;
        const Eigen::MatrixXd V = Psys - Spost;

        // Covariance mode pulls: conditional mean at the best fit (Putnam), identical to
        // plotCovariancePosteriorPulls and PROerrorbar::center_shift.
        Eigen::VectorXd alpha_hat = Eigen::VectorXd::Zero(k);
        if(k) {
            const Eigen::MatrixXd Lr = J.rightCols(k);
            const Eigen::MatrixXd inner = Eigen::MatrixXd::Identity(k, k) + Lr.transpose() * cinv.asDiagonal() * Lr;
            alpha_hat = spdInverse(inner) * (Lr.transpose() * cinv.cwiseProduct(u));
        }

        std::vector<Eigen::Index> sys_index;   // per entry; -1 for uniform
        const auto addEntry = [&](PullEntry e, Eigen::Index si) {
            out.entries.push_back(std::move(e));
            sys_index.push_back(si);
        };
        const double rmin = min_var_reduction;

        for(Eigen::Index a = 0, g = 0; a < ns; ++a) {
            const size_t s = free_spl[a];
            const Eigen::Index ip = (Eigen::Index)(nphys + s), si = nf + a;
            const double x = best_fit(ip), lo = lb(ip), hi = ub(ip);
            PullEntry e;
            e.name = plotname(s);
            e.value = (float)x;
            const double tol = 1e-4 * (hi - lo);
            e.censored = x <= lo + tol ? -1 : (x >= hi - tol ? 1 : 0);
            if(isUniform(s)) {
                const double W = hi - lo, box_var = W * W / 12.0;
                const double data_prec = 1.0 / Spost(si, si) - 12.0 / (W * W);
                const double sd = data_prec > 0 ? 1.0 / std::sqrt(data_prec) : INFINITY;
                e.category = PullCategory::UniformSpline;
                e.center = (float)(0.5 * (lo + hi));
                e.sigma_prior = (float)std::sqrt(box_var);
                e.sigma_post = (float)sd;
                e.var_reduction = (float)(1.0 - sd * sd / box_var);
                e.raw_pull = (float)((x - e.center) / e.sigma_prior);
                e.used = std::isfinite(sd) && e.var_reduction >= rmin;
                if(std::isfinite(sd)) e.pull = (float)normQuantile(uniformSmearedCdf(x, lo, hi, sd));
                addEntry(e, -1);
                continue;
            }
            const double c = ccond(g), P = Pcond(g, g), sp = std::sqrt(P);
            ++g;
            const double Vaa = V(si, si);
            const double stol = 1e-3 * sp;
            const bool symmetric = lo <= c - 3 * sp + stol && hi >= c + 3 * sp - stol;
            e.category = symmetric ? PullCategory::SymmetricSpline : PullCategory::BoundedSpline;
            e.center = (float)c;
            e.sigma_prior = (float)sp;
            e.sigma_post = (float)std::sqrt(std::max(0.0, Spost(si, si)));
            e.var_reduction = (float)(Vaa / P);
            e.raw_pull = (float)((x - c) / sp);
            e.used = e.var_reduction >= rmin;
            if(Vaa > 0) e.pull = (float)((x - c) / std::sqrt(Vaa));
            addEntry(e, si);
        }

        for(Eigen::Index j = 0; j < k; ++j) {
            const Eigen::Index si = nf + ns + j;
            PullEntry e;
            e.name = "cov mode " + std::to_string(modes[j]) + fmt(" (%.2g%%)", 100.0 * L.col(modes[j]).squaredNorm() / total_var);
            e.category = PullCategory::CovarianceMode;
            e.value = (float)alpha_hat(j);
            e.sigma_prior = 1.0f;
            e.sigma_post = (float)std::sqrt(std::max(0.0, Spost(si, si)));
            e.var_reduction = (float)V(si, si);
            e.raw_pull = e.value;
            e.used = e.var_reduction >= rmin;
            if(V(si, si) > 0) e.pull = (float)(alpha_hat(j) / std::sqrt(V(si, si)));
            addEntry(e, si);
        }

        // Per category: Gaussian fit, the model's spread of the mean, whitened chi^2.
        for(int cat = 0; cat < 4; ++cat) {
            std::vector<double> z;
            std::vector<int> cens;
            std::vector<Eigen::Index> ids;
            int excluded = 0;
            for(size_t i = 0; i < out.entries.size(); ++i) {
                const PullEntry &e = out.entries[i];
                if((int)e.category != cat) continue;
                if(!e.used) { ++excluded; continue; }
                z.push_back(e.pull);
                cens.push_back(e.censored);
                ids.push_back(sys_index[i]);
            }
            PullGaussFit f = fitGaussian(z, cens);
            f.n_excluded = excluded;
            const Eigen::Index nz = (Eigen::Index)z.size();
            if(nz && cat != (int)PullCategory::UniformSpline) {
                Eigen::MatrixXd rho(nz, nz);
                for(Eigen::Index a = 0; a < nz; ++a)
                    for(Eigen::Index b = 0; b < nz; ++b)
                        rho(a, b) = V(ids[a], ids[b]) / std::sqrt(V(ids[a], ids[a]) * V(ids[b], ids[b]));
                f.null_mu_err = std::sqrt(std::max(0.0, rho.sum())) / nz;
                if(!f.n_censored) {
                    // Directions the model pins to under 1% of a single pull's variance are
                    // dropped: there, fit tolerance rather than data sets the residual.
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(rho);
                    const Eigen::VectorXd y = Eigen::Map<const Eigen::VectorXd>(z.data(), nz);
                    const Eigen::VectorXd proj = es.eigenvectors().transpose() * y;
                    for(Eigen::Index q = 0; q < nz; ++q) {
                        if(es.eigenvalues()(q) < 1e-2) continue;
                        f.whitened_chi2 += proj(q) * proj(q) / es.eigenvalues()(q);
                        ++f.whitened_ndf;
                    }
                }
            } else if(nz) {
                f.null_mu_err = 1.0 / std::sqrt((double)nz);
            }
            // E[(1/n) sum (z - zbar)^2] = 1 - Var(zbar): correlated pulls fill a narrower histogram.
            if(nz) f.null_sigma = std::sqrt(std::max(0.0, 1.0 - f.null_mu_err * f.null_mu_err));
            out.fits[cat] = f;
        }

        // Log: per-category summary and the largest pulls; full table at DEBUG.
        static const char *catname[4] = {"symmetric splines", "covariance modes", "bounded splines", "uniform splines"};
        log<LOG_INFO>(L"%1% || Post-fit pull distributions (z ~ N(0,1) if the model is right; %2% fixed splines, weakly constrained = prior variance reduced < %3%%%):")
            % __func__ % out.fixed_names.size() % (100.0 * rmin);
        for(int cat = 0; cat < 4; ++cat) {
            const PullGaussFit &f = out.fits[cat];
            if(f.n + f.n_excluded == 0) continue;
            if(f.fitted)
                log<LOG_INFO>(L"%1% ||   %2%: n=%3% (censored %4%, weakly constrained %5%), fit mu = %6% +- %7%, sigma = %8% +- %9%; model: mu = 0 +- %10%, sigma ~ %11%%12%")
                    % __func__ % catname[cat] % f.n % f.n_censored % f.n_excluded % f.mu % f.mu_err % f.sigma % f.sigma_err % f.null_mu_err % f.null_sigma
                    % (f.whitened_ndf ? fmt(", whitened chi2/ndf = %.3g", f.whitened_chi2) + "/" + std::to_string(f.whitened_ndf)
                                        + fmt(" (p = %.3g)", TMath::Prob(f.whitened_chi2, f.whitened_ndf)) : std::string()).c_str();
            else
                log<LOG_INFO>(L"%1% ||   %2%: n=%3% (censored %4%, weakly constrained %5%), too few to fit")
                    % __func__ % catname[cat] % f.n % f.n_censored % f.n_excluded;
            std::vector<const PullEntry*> used;
            for(const PullEntry &e : out.entries)
                if((int)e.category == cat && e.used) used.push_back(&e);
            std::sort(used.begin(), used.end(), [](const PullEntry *a, const PullEntry *b) { return std::abs(a->pull) > std::abs(b->pull); });
            for(size_t i = 0; i < std::min<size_t>(3, used.size()); ++i)
                log<LOG_INFO>(L"%1% ||      largest |z|: %2% z = %3% (raw pull %4%)%5%") % __func__ % used[i]->name.c_str() % used[i]->pull % used[i]->raw_pull
                    % (used[i]->censored ? " [at fit bound]" : "");
        }
        log<LOG_DEBUG>(L"%1% || %2$-40s %3$4s %4$10s %5$10s %6$10s %7$10s %8$8s %9$9s %10$9s %11$5s %12$4s")
            % __func__ % "name" % "cat" % "value" % "center" % "sig_prior" % "sig_post" % "var_red" % "raw_pull" % "z" % "cens" % "used";
        for(const PullEntry &e : out.entries)
            log<LOG_DEBUG>(L"%1% || %2$-40s %3$4d %4$10.4g %5$10.4g %6$10.4g %7$10.4g %8$8.3f %9$9.3f %10$9.3f %11$5d %12$4d")
                % __func__ % e.name.c_str() % ((int)e.category + 1) % e.value % e.center % e.sigma_prior % e.sigma_post
                % e.var_reduction % e.raw_pull % e.pull % e.censored % (int)e.used;
        return out;
    }

    int plotPostfitPullDistributions(const PullSummary &summary, const std::string &filename,
            std::map<std::string, TObject*> *drawn_objs) {
        if(summary.entries.empty() && summary.fixed_names.empty()) {
            log<LOG_INFO>(L"%1% || No pulls to draw; skipping %2%.") % __func__ % filename.c_str();
            return 1;
        }
        static int call = 0;
        const std::string uid = std::to_string(call++);

        const int nbx = 20;
        const double xr = 5.0, bw = 2 * xr / nbx;

        float paper_w, paper_h;
        gStyle->GetPaperSize(paper_w, paper_h);
        gStyle->SetPaperSize(27.5f, 16.0f);

        TCanvas c(("pulldist_c" + uid).c_str(), "Pull distributions", 1100, 640);
        c.Print((filename + "[").c_str());

        for(int cat = 0; cat < 4; ++cat) {
            const CategoryInfo &info = kCategories[cat];
            const PullGaussFit &f = summary.fits[cat];
            // Clear() deletes the previous page's pads (heap) and everything they own.
            c.Clear();
            c.cd();
            TPad *plot_pad = new TPad(("pd_plot" + uid).c_str(), "", 0.0, 0.0, 0.67, 1.0);
            TPad *text_pad = new TPad(("pd_text" + uid).c_str(), "", 0.67, 0.0, 1.0, 1.0);
            plot_pad->SetLeftMargin(0.11);
            plot_pad->SetRightMargin(0.03);
            plot_pad->SetTopMargin(0.09);
            plot_pad->SetBottomMargin(0.15);
            plot_pad->Draw();
            text_pad->Draw();

            // Total (hatched) under the uncensored entries (solid): the hatch shows the censored excess.
            TH1D h_all(("pd_all_" + std::string(info.key) + uid).c_str(), "", nbx, -xr, xr);
            TH1D h_unc(("pd_unc_" + std::string(info.key) + uid).c_str(), "", nbx, -xr, xr);
            h_all.SetDirectory(nullptr);
            h_unc.SetDirectory(nullptr);
            int n_outside = 0;
            std::vector<const PullEntry*> used;
            for(const PullEntry &e : summary.entries) {
                if((int)e.category != cat || !e.used) continue;
                used.push_back(&e);
                if(std::abs(e.pull) > xr) ++n_outside;
                // A censored entry goes on the side of its bound where the latent z lies.
                const double x = std::clamp((double)e.pull + 1e-6 * e.censored, -xr + 1e-6, xr - 1e-6);
                h_all.Fill(x);
                if(!e.censored) h_unc.Fill(x);
            }
            const double N = (double)used.size();

            plot_pad->cd();
            const double peak_ref = N * bw * kInvSqrt2Pi;
            const double peak_fit = (f.fitted && f.sigma > 1e-4) ? N * bw * kInvSqrt2Pi / f.sigma : 0.0;
            const double ymax = std::max({h_all.GetMaximum(), peak_ref, std::min(peak_fit, 3.0 * std::max(1.0, h_all.GetMaximum())), 1.0}) * 1.35;

            h_all.SetStats(0);
            h_all.SetMinimum(0);
            h_all.SetMaximum(ymax);
            h_all.GetXaxis()->SetTitle(info.xtitle);
            h_all.GetYaxis()->SetTitle(fmt("Parameters / %.1f", bw).c_str());
            h_all.GetXaxis()->SetTitleSize(0.045);
            h_all.GetXaxis()->SetLabelSize(0.040);
            h_all.GetXaxis()->SetTitleOffset(1.25);
            h_all.GetYaxis()->SetTitleSize(0.045);
            h_all.GetYaxis()->SetLabelSize(0.040);
            h_all.GetYaxis()->SetTitleOffset(1.0);
            h_all.SetLineColor(kGray + 2);
            h_all.SetFillColor(kGray + 2);
            h_all.SetFillStyle(3354);
            h_all.Draw("hist");
            h_unc.SetLineColor(TColor::GetColor(info.line));
            h_unc.SetFillColor(TColor::GetColor(info.fill));
            h_unc.SetFillStyle(1001);
            h_unc.Draw("hist same");

            TF1 ref(("pd_ref" + uid).c_str(), [N, bw](double *x, double *) { return N * bw * normPdf(x[0]); }, -xr, xr, 0);
            ref.SetNpx(400);
            ref.SetLineColor(kBlack);
            ref.SetLineStyle(2);
            ref.SetLineWidth(2);
            if(N > 0) ref.Draw("same");

            const double mu = f.mu, sg = f.sigma;
            TF1 fit(("pd_fit" + uid).c_str(), [N, bw, mu, sg](double *x, double *) { return N * bw * normPdf((x[0] - mu) / sg) / sg; }, -xr, xr, 0);
            fit.SetNpx(400);
            fit.SetLineColor(kRed + 1);
            fit.SetLineWidth(2);
            const bool draw_fit = f.fitted && sg > 1e-4;
            if(draw_fit) fit.Draw("same");
            h_all.Draw("axis same");

            TLegend leg(0.13, 0.70, 0.47, 0.89);
            leg.SetBorderSize(0);
            leg.SetFillStyle(0);
            leg.SetTextSize(0.035);
            if(N > 0) {
                leg.AddEntry(&h_unc, "Post-fit pulls", "f");
                if(h_all.Integral() > h_unc.Integral()) leg.AddEntry(&h_all, "At a fit bound (censored)", "f");
                leg.AddEntry(&ref, "N(0,1) expectation", "l");
                if(draw_fit) leg.AddEntry(&fit, "Gaussian fit (unbinned ML)", "l");
                leg.Draw();
            } else {
                TLatex empty;
                empty.SetNDC();
                empty.SetTextAlign(22);
                empty.SetTextSize(0.045);
                empty.SetTextColor(kGray + 2);
                empty.DrawLatex(0.55, 0.55, f.n_excluded ? "All parameters here are weakly constrained" : "No parameters in this category");
            }

            TLatex tl;
            tl.SetNDC();
            tl.SetTextFont(42);
            tl.SetTextSize(0.040);
            tl.SetTextAlign(13);
            tl.DrawLatex(0.11, 0.975, info.title);
            if(summary.shape_only) drawShapeOnlyNote(plot_pad);

            // Side panel: the fit, what the model expects, the largest pulls, what is left out.
            text_pad->cd();
            TLatex tx;
            TText tt;
            tx.SetNDC();
            tx.SetTextFont(42);
            tx.SetTextAlign(13);
            tt.SetNDC();
            tt.SetTextFont(42);
            tt.SetTextAlign(13);
            double y = 0.90;
            const double dy = 0.047;
            auto head = [&](const std::string &s) { tx.SetTextSize(0.050); tx.DrawLatex(0.04, y, ("#font[62]{" + s + "}").c_str()); y -= dy * 1.05; };
            auto line = [&](const std::string &s) { tx.SetTextSize(0.046); tx.DrawLatex(0.07, y, s.c_str()); y -= dy; };
            // Plotnames are drawn as plain text (no TLatex: '_' would subscript), cut to fit the panel.
            auto name = [&](std::string s) {
                if(s.size() > 34) s = s.substr(0, 31) + "...";
                tt.SetTextSize(0.046);
                tt.DrawText(0.07, y, s.c_str());
                y -= dy;
            };

            head("Gaussian fit (unbinned ML)");
            line("N = " + std::to_string(f.n) + (f.n_censored ? "  (" + std::to_string(f.n_censored) + " at a fit bound)" : std::string()));
            if(f.fitted) {
                line(fmt("#mu = %.3f", f.mu) + fmt(" #pm %.3f", f.mu_err));
                line(fmt("#sigma = %.3f", f.sigma) + fmt(" #pm %.3f", f.sigma_err));
            } else {
                line("too few entries to fit");
            }
            y -= dy * 0.4;
            head("If the model is right");
            if(f.n >= 2) {
                line(fmt("#mu = 0 #pm %.3f", f.null_mu_err) + fmt(",  #sigma #approx %.3f", f.null_sigma));
                line("(pulls correlated, so narrower)");
            } else if(f.n == 1) {
                line("z ~ N(0,1)");
            }
            if(f.whitened_ndf)
                line(fmt("whitened #chi^{2}/ndf = %.1f", f.whitened_chi2) + "/" + std::to_string(f.whitened_ndf)
                     + fmt("  (p = %.2g)", TMath::Prob(f.whitened_chi2, f.whitened_ndf)));
            if(n_outside) line(std::to_string(n_outside) + " beyond #pm5 (drawn in the edge bins)");

            if(!used.empty()) {
                y -= dy * 0.4;
                head("Largest |z|");
                std::vector<const PullEntry*> sorted = used;
                std::sort(sorted.begin(), sorted.end(), [](const PullEntry *a, const PullEntry *b) { return std::abs(a->pull) > std::abs(b->pull); });
                for(size_t i = 0; i < std::min<size_t>(5, sorted.size()); ++i) {
                    std::string label = sorted[i]->name;
                    if(label.size() > 24) label = label.substr(0, 21) + "...";
                    name(label + fmt(":  %+.2f", sorted[i]->pull) + (sorted[i]->censored ? " (bound)" : ""));
                }
            }

            y -= dy * 0.4;
            head("Not shown");
            const std::string rpct = fmt("%.0f%%", 100.0 * summary.min_var_reduction);
            if(cat != (int)PullCategory::UniformSpline) {
                line(std::to_string(f.n_excluded) + " weakly constrained");
                line("(prior variance reduced < " + rpct + ")");
            } else {
                // Everything that reached no page, gathered here.
                line("Weakly constrained (prior var. reduced < " + rpct + "):");
                static const char *short_name[4] = {"(1) symmetric", "(2) cov. modes", "(3) bounded", "(4) uniform"};
                for(int o = 0; o < 4; ++o)
                    line(std::string("   ") + short_name[o] + ": " + std::to_string(summary.fits[o].n_excluded));
                line(std::to_string(summary.fixed_names.size()) + " fixed (zero-width bounds)" + (summary.fixed_names.empty() ? "" : ":"));
                for(size_t i = 0; i < std::min<size_t>(4, summary.fixed_names.size()); ++i) name("   " + summary.fixed_names[i]);
                if(summary.fixed_names.size() > 4) line("   (+" + std::to_string(summary.fixed_names.size() - 4) + " more)");
                line("Physics parameters: no prior, no pull");
            }

            c.cd();
            drawVersionWatermark(&c, WatermarkPos::BottomRight);
            c.Print(filename.c_str());

            if(drawn_objs) {
                TH1D *h = (TH1D*)h_all.Clone(("postfit_pulldist_" + std::string(info.key)).c_str());
                h->SetDirectory(nullptr);
                putObject(drawn_objs, h->GetName(), h);
            }
        }
        c.Print((filename + "]").c_str());
        gStyle->SetPaperSize(paper_w, paper_h);
        log<LOG_INFO>(L"%1% || Wrote post-fit pull distributions to %2%") % __func__ % filename.c_str();

        if(drawn_objs) {
            TTree *t = new TTree("postfit_pulls", "Post-fit pulls (category 1-4 = page)");
            t->SetDirectory(nullptr);
            std::string bname;
            int bcat = 0, bcens = 0, bused = 0;
            float bvalue = 0, bcenter = 0, bsprior = 0, bspost = 0, bvr = 0, braw = 0, bz = 0;
            t->Branch("name", &bname);
            t->Branch("category", &bcat);
            t->Branch("value", &bvalue);
            t->Branch("center", &bcenter);
            t->Branch("sigma_prior", &bsprior);
            t->Branch("sigma_post", &bspost);
            t->Branch("var_reduction", &bvr);
            t->Branch("raw_pull", &braw);
            t->Branch("pull", &bz);
            t->Branch("censored", &bcens);
            t->Branch("used", &bused);
            for(const PullEntry &e : summary.entries) {
                bname = e.name; bcat = (int)e.category + 1; bvalue = e.value; bcenter = e.center;
                bsprior = e.sigma_prior; bspost = e.sigma_post; bvr = e.var_reduction;
                braw = e.raw_pull; bz = e.pull; bcens = e.censored; bused = e.used;
                t->Fill();
            }
            t->ResetBranchAddresses();
            putObject(drawn_objs, "postfit_pulls", t);

            TTree *tf = new TTree("postfit_pull_fits", "Gaussian fit per pull category (1-4 = page)");
            tf->SetDirectory(nullptr);
            int fcat = 0, fn = 0, fcens = 0, fexcl = 0, ffit = 0, fndf = 0;
            double fmu = 0, fmue = 0, fsg = 0, fsge = 0, fnull = 0, fnsg = 0, fchi = 0;
            tf->Branch("category", &fcat);
            tf->Branch("n", &fn);
            tf->Branch("n_censored", &fcens);
            tf->Branch("n_excluded", &fexcl);
            tf->Branch("fitted", &ffit);
            tf->Branch("mu", &fmu);
            tf->Branch("mu_err", &fmue);
            tf->Branch("sigma", &fsg);
            tf->Branch("sigma_err", &fsge);
            tf->Branch("null_mu_err", &fnull);
            tf->Branch("null_sigma", &fnsg);
            tf->Branch("whitened_chi2", &fchi);
            tf->Branch("whitened_ndf", &fndf);
            for(int cat = 0; cat < 4; ++cat) {
                const PullGaussFit &f = summary.fits[cat];
                fcat = cat + 1; fn = f.n; fcens = f.n_censored; fexcl = f.n_excluded; ffit = f.fitted;
                fmu = f.mu; fmue = f.mu_err; fsg = f.sigma; fsge = f.sigma_err; fnull = f.null_mu_err; fnsg = f.null_sigma;
                fchi = f.whitened_chi2; fndf = f.whitened_ndf;
                tf->Fill();
            }
            tf->ResetBranchAddresses();
            putObject(drawn_objs, "postfit_pull_fits", tf);
        }
        return 0;
    }

}
