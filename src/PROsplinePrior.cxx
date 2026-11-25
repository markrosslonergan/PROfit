#include "PROsplinePrior.h"
#include "PROcess.h"
#include "PROlog.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace PROfit {

    namespace {
        bool isUniform(const PROsyst &systs, size_t i) {
            return i < systs.spline_prior_types.size() && systs.spline_prior_types[i] == SplinePriorType::Uniform;
        }

        /// Uniform-prior splines carry no Gaussian pull: identity rows/columns decouple them.
        void decoupleUniform(const PROsyst &systs, Eigen::MatrixXf &cov) {
            for(size_t i = 0; i < systs.spline_prior_types.size() && (Eigen::Index)i < cov.rows(); ++i) {
                if(systs.spline_prior_types[i] != SplinePriorType::Uniform) continue;
                cov.row(i).setZero();
                cov.col(i).setZero();
                cov(i, i) = 1.0f;
            }
        }

        /// FNV-1a over the member names: a fixed pilot seed per block, independent of any run seed.
        uint32_t nameHash(const std::vector<std::string> &names) {
            uint32_t h = 2166136261u;
            for(const auto &s : names) {
                for(unsigned char c : s) { h ^= c; h *= 16777619u; }
                h ^= 0xffu; h *= 16777619u;
            }
            return h;
        }

        std::string joinNames(const std::vector<std::string> &names, size_t max_listed = 12) {
            std::ostringstream os;
            for(size_t k = 0; k < names.size() && k < max_listed; ++k) os << (k ? ", " : "") << names[k];
            if(names.size() > max_listed) os << ", ... (" << names.size() << " in total)";
            return os.str();
        }
    }

    Eigen::MatrixXf SplinePriorCovariance(const PROconfig &config, const PROsyst &systs, bool &correlated) {
        correlated = false;
        const size_t n = systs.GetNSplines();
        if(systs.has_external_prior_cov) {
            if(config.m_mcgen_correlations.size())
                log<LOG_WARNING>(L"%1% || Both an external prior covariance and XML prior correlations are set; using the external one.") % __func__;
            correlated = true;
            Eigen::MatrixXf cov = systs.external_prior_cov;
            decoupleUniform(systs, cov);
            return cov;
        }
        if(config.m_mcgen_correlations.empty() || n == 0) return Eigen::MatrixXf();

        Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(n, n);
        size_t matched = 0;
        for(const auto &[a, b, rho] : config.m_mcgen_correlations) {
            auto itA = std::find(systs.spline_names.begin(), systs.spline_names.end(), a);
            auto itB = std::find(systs.spline_names.begin(), systs.spline_names.end(), b);
            if(itA == systs.spline_names.end() || itB == systs.spline_names.end()) {
                log<LOG_DEBUG>(L"%1% || Prior correlation %2% %3% names a spline this systematics object does not carry; skipped.")
                    % __func__ % a.c_str() % b.c_str();
                continue;
            }
            const Eigen::Index iA = std::distance(systs.spline_names.begin(), itA);
            const Eigen::Index iB = std::distance(systs.spline_names.begin(), itB);
            cov(iA, iB) = rho;
            cov(iB, iA) = rho;
            ++matched;
        }
        if(matched == 0) return Eigen::MatrixXf();
        correlated = true;
        cov = systs.spline_priors.asDiagonal() * cov * systs.spline_priors.asDiagonal();
        decoupleUniform(systs, cov);
        return cov;
    }

    Eigen::MatrixXf SplinePriorPrecisionFromCovariance(const Eigen::MatrixXf &cov) {
        if(cov.size() == 0) return Eigen::MatrixXf();
        Eigen::LLT<Eigen::MatrixXd> llt(cov.cast<double>());
        if(llt.info() != Eigen::Success)
            throw std::runtime_error("SplinePriorPrecisionFromCovariance: the spline prior covariance is not positive definite");
        return llt.solve(Eigen::MatrixXd::Identity(cov.rows(), cov.cols())).cast<float>();
    }

    SplinePriorSampler::SplinePriorSampler(const PROconfig &config, const PROsyst &systs,
                                           const std::vector<char> &held, const Eigen::VectorXf &held_values,
                                           bool verbose) {
        const size_t n = systs.GetNSplines();
        kind.assign(n, Kind::Single);
        lo.assign(n, 0.0f); hi.assign(n, 0.0f); mu.assign(n, 0.0f); sig.assign(n, 1.0f); value.assign(n, 0.0f);
        names.resize(n);
        block_of.assign(n, -1);

        // Box, prior and pinning exactly as ThrowRestrictedSplinePull reads them.
        for(size_t i = 0; i < n; ++i) {
            const bool has_r = i < systs.spline_has_restrict.size() && systs.spline_has_restrict[i];
            float tlo = has_r ? systs.spline_restrict_lo[i] : systs.spline_lo[i];
            float thi = has_r ? systs.spline_restrict_hi[i] : systs.spline_hi[i];
            if(tlo > thi) std::swap(tlo, thi);
            lo[i] = tlo;
            hi[i] = thi;
            mu[i] = i < (size_t)systs.spline_centers.size() ? systs.spline_centers((Eigen::Index)i) : 0.0f;
            sig[i] = i < (size_t)systs.spline_priors.size() ? std::abs(systs.spline_priors((Eigen::Index)i)) : 1.0f;
            names[i] = i < systs.spline_names.size() ? systs.spline_names[i] : ("spline#" + std::to_string(i));
            if(i < held.size() && held[i]) {
                kind[i] = Kind::Pinned;
                value[i] = (Eigen::Index)i < held_values.size() ? held_values((Eigen::Index)i) : mu[i];
            } else if(tlo == thi) {
                kind[i] = Kind::Pinned;
                value[i] = tlo;
            } else if(isUniform(systs, i)) {
                kind[i] = Kind::Uniform;
            }
        }

        bool correlated = false;
        const Eigen::MatrixXf Sf = SplinePriorCovariance(config, systs, correlated);
        if(!correlated) return;

        // Independent blocks are the connected components of the prior's non-zero pattern over
        // ALL Gaussian splines, pinned ones included: conditioning on a pinned spline couples its
        // partners (A-P-B with S_AB = 0 has Cov(A, B | P) != 0), so they must be drawn together.
        std::vector<size_t> parent(n);
        std::iota(parent.begin(), parent.end(), 0);
        std::function<size_t(size_t)> root = [&](size_t i) { return parent[i] == i ? i : parent[i] = root(parent[i]); };
        const auto gaussian = [&](size_t i) { return !isUniform(systs, i); };
        for(size_t i = 0; i < n; ++i) {
            if(!gaussian(i)) continue;
            for(size_t j = i + 1; j < n; ++j)
                if(gaussian(j) && Sf((Eigen::Index)i, (Eigen::Index)j) != 0.0f) parent[root(i)] = root(j);
        }
        std::vector<std::vector<size_t>> components(n);
        for(size_t i = 0; i < n; ++i) if(gaussian(i)) components[root(i)].push_back(i);

        for(const auto &comp : components) {
            if(comp.size() < 2) continue;   // no partner: the historic per-spline draw
            std::vector<size_t> F, P;
            for(size_t i : comp) (kind[i] == Kind::Pinned ? P : F).push_back(i);
            if(F.empty()) continue;

            const Eigen::Index nf = F.size(), np = P.size();
            Eigen::MatrixXd S_FF(nf, nf), S_FP(nf, np), S_PP(np, np);
            Eigen::VectorXd c_F(nf), dx_P(np);
            for(Eigen::Index a = 0; a < nf; ++a) {
                c_F(a) = mu[F[a]];
                for(Eigen::Index b = 0; b < nf; ++b) S_FF(a, b) = Sf(F[a], F[b]);
                for(Eigen::Index b = 0; b < np; ++b) S_FP(a, b) = Sf(F[a], P[b]);
            }
            for(Eigen::Index a = 0; a < np; ++a) {
                dx_P(a) = (double)value[P[a]] - mu[P[a]];
                for(Eigen::Index b = 0; b < np; ++b) S_PP(a, b) = Sf(P[a], P[b]);
            }

            JointBlock blk;
            blk.members = F;
            blk.mean = c_F;
            Eigen::MatrixXd cond = S_FF;
            if(np > 0) {
                // Gaussian conditional on the pinned partners: the distribution the fit's Pull
                // implies for the free splines once the pinned ones are held at their values.
                Eigen::LLT<Eigen::MatrixXd> llt_P(S_PP);
                if(llt_P.info() != Eigen::Success)
                    throw std::runtime_error("SplinePriorSampler: prior covariance of the pinned splines is not positive definite");
                const Eigen::MatrixXd K = llt_P.solve(S_FP.transpose()).transpose();
                blk.mean += K * dx_P;
                cond -= K * S_FP.transpose();
            }
            Eigen::LLT<Eigen::MatrixXd> llt(0.5 * (cond + cond.transpose()));
            if(llt.info() != Eigen::Success)
                throw std::runtime_error("SplinePriorSampler: conditional prior covariance of a correlated block is not positive definite");
            blk.L = llt.matrixL();
            blk.lo.resize(nf);
            blk.hi.resize(nf);
            for(Eigen::Index a = 0; a < nf; ++a) { blk.lo(a) = lo[F[a]]; blk.hi(a) = hi[F[a]]; }

            std::vector<std::string> member_names;
            for(size_t i : F) member_names.push_back(names[i]);

            // Pilot acceptance with a private generator, so the caller's stream is untouched.
            const int npilot = 20000;
            std::mt19937 prng(nameHash(member_names));
            std::normal_distribution<float> pd;
            Eigen::VectorXf scratch;
            int nacc = 0;
            for(int t = 0; t < npilot; ++t) if(drawBlock(blk, prng, pd, scratch)) ++nacc;
            blk.acceptance = (double)nacc / npilot;
            if(nacc < 20) {
                std::vector<std::string> tight;
                for(Eigen::Index a = 0; a < nf; ++a) {
                    const double s = std::sqrt(std::max(0.0, cond(a, a)));
                    if(blk.lo(a) > blk.mean(a) - s || blk.hi(a) < blk.mean(a) + s) tight.push_back(names[F[a]]);
                }
                std::ostringstream os;
                os << "A correlated block of " << nf << " spline prior(s) (" << joinNames(member_names)
                   << ") keeps only " << nacc << " of " << npilot << " joint draws inside its fit box "
                   << "(acceptance " << blk.acceptance << " < 1e-3), too little for exact rejection sampling. "
                   << "Splines whose box lies within 1 sigma of the conditional mean: "
                   << (tight.empty() ? std::string("none") : joinNames(tight))
                   << ". Widen their restrict= range, or drop/loosen the correlations that tie them together.";
                log<LOG_ERROR>(L"%1% || %2%") % __func__ % os.str().c_str();
                throw std::runtime_error(os.str());
            }
            blk.max_attempts = (long)std::ceil(1000.0 / blk.acceptance);

            const int idx = (int)blocks.size();
            for(size_t i : F) { kind[i] = Kind::Block; block_of[i] = idx; }
            if(verbose) {
                log<LOG_INFO>(L"%1% || Correlated spline prior: %2% spline(s) thrown jointly (%3%), joint box acceptance %4%.")
                    % __func__ % nf % joinNames(member_names).c_str() % blk.acceptance;
                if(np > 0) {
                    std::vector<std::string> pinned_names;
                    for(size_t i : P) pinned_names.push_back(names[i]);
                    log<LOG_INFO>(L"%1% || ... conditioned on pinned/held partner(s) %2%; conditional means shift by up to %3% sigma.")
                        % __func__ % joinNames(pinned_names).c_str()
                        % ((blk.mean - c_F).cwiseQuotient(S_FF.diagonal().cwiseSqrt()).cwiseAbs().maxCoeff());
                }
            }
            blocks.push_back(std::move(blk));
        }
    }

    bool SplinePriorSampler::drawBlock(const JointBlock &b, std::mt19937 &rng, std::normal_distribution<float> &d,
                                       Eigen::VectorXf &out) const {
        const Eigen::Index nb = b.mean.size();
        Eigen::VectorXd z(nb);
        for(Eigen::Index k = 0; k < nb; ++k) z(k) = d(rng);
        out = (b.mean + b.L * z).cast<float>();
        for(Eigen::Index k = 0; k < nb; ++k)
            if(out(k) < b.lo(k) || out(k) > b.hi(k)) return false;
        return true;
    }

    void SplinePriorSampler::Throw(Eigen::Ref<Eigen::VectorXf> splines, std::mt19937 &rng,
                                   std::normal_distribution<float> &d) const {
        if((size_t)splines.size() != kind.size())
            throw std::runtime_error("SplinePriorSampler::Throw: output size does not match the number of splines");
        Eigen::VectorXf x;
        for(size_t i = 0; i < kind.size(); ++i) {
            switch(kind[i]) {
                case Kind::Pinned:
                    splines(i) = value[i];
                    break;
                case Kind::Uniform:
                    splines(i) = std::uniform_real_distribution<float>(lo[i], hi[i])(rng);
                    break;
                case Kind::Single:
                    splines(i) = ThrowTruncatedNormal(mu[i], sig[i], lo[i], hi[i], rng, d, names[i], i);
                    break;
                case Kind::Block: {
                    const JointBlock &b = blocks[block_of[i]];
                    if(i != b.members.front()) break;   // drawn with its block's first member
                    long attempts = 1;
                    while(!drawBlock(b, rng, d, x)) {
                        if(++attempts > b.max_attempts)
                            throw std::runtime_error("SplinePriorSampler::Throw: correlated block starting at spline '" + names[i]
                                                     + "' exceeded its rejection budget");
                    }
                    for(size_t k = 0; k < b.members.size(); ++k) splines(b.members[k]) = x(k);
                } break;
            }
        }
    }

}
