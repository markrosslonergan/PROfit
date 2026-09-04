#ifndef PROMCMC_H
#define PROMCMC_H

#include "PROmetric.h"
#include "PROgress.h"
#include "PROwatermark.h"
#include <Eigen/Eigen>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <optional>

#include "TStyle.h"
#include "TVirtualFFT.h"

namespace PROfit {

    // In PROplot.h, but I get a weird error when I include here because of recursive inclusion
    // Not sure why, that should be taken care of by include guards.
    void set_matrix_palette();
        
    template<class Target_FN, class Proposal_FN>
        class Metropolis {
            private:
                std::mt19937 rng;
                std::uniform_real_distribution<float> uniform;
                uint32_t seed;
                bool save_chain;
                // Cached target(current). 'current' only changes on an accepted step, so
                // the log-density from that acceptance is reused instead of re-evaluating
                // the (expensive) target on every proposal. Anyone assigning to the public
                // 'current' directly must set current_logp_valid = false.
                float current_logp = 0.0f;
                bool current_logp_valid = false;

            public:
                Target_FN target;
                Proposal_FN proposal;
                Eigen::VectorXf current;
                std::vector<Eigen::VectorXf> chain;
                size_t naccept;

                Metropolis(Target_FN target, Proposal_FN proposal, const Eigen::VectorXf &initial, uint32_t seed, bool save_chain = true) 
                    : seed(seed), target(target), proposal(proposal), current(initial), save_chain(save_chain) {
                        rng.seed(seed);
                        naccept = 0;
                    }

                bool step() {
                    Eigen::VectorXf p = proposal(current);
                    float acceptance = 0;
                    float proposed_logp = 0;
                    bool in_bound = proposal.within_bound(p);
                    if(in_bound) {
                        if(!current_logp_valid) {
                            current_logp = target(current);
                            current_logp_valid = true;
                        }
                        proposed_logp = target(p);
                        // Targets return log-density (e.g. -0.5*chi^2); subtract before exp to avoid float32 underflow when chi^2 is large.
                        acceptance = std::min(1.0f, std::exp(proposed_logp - current_logp) * proposal.P(current, p)/proposal.P(p, current));
                    }
                    // Drawn unconditionally (even out-of-bounds) to keep the RNG stream identical to the pre-cache implementation.
                    float u = uniform(rng);

                    if(u <= acceptance) {
                        //                      log<LOG_DEBUG>(L"%1% || APPROVED acc %2%, rng %3% and proposal: %4%  ") % __func__ % acceptance % u %p;
                        current = p;
                        current_logp = proposed_logp;
                        naccept += 1;
                        return true;
                    }else{
                        //                      log<LOG_DEBUG>(L"%1% || REJECTED acc %2%, rng %3% and proposal: %4%  ") % __func__ % acceptance % u %p;
                    }
                    return false;
                }

                void run(size_t burnin, size_t steps, std::optional<std::function<void(const Eigen::VectorXf&)>> action = {}, PROgressBar *pbar = nullptr) {
                    const size_t pbar_stride = std::max<size_t>(1, (burnin + steps) / 1000);
                    for(size_t i = 0; i < burnin; i++) {
                        if constexpr(Proposal_FN::has_tune) {
                            proposal.tune(step());
                        } else {
                            step();
                        }
                        if(pbar && (i + 1) % pbar_stride == 0) pbar->set_progress(i + 1);
                    }
                    proposal.tune_mode = false;
                    for(size_t i = 0; i < steps; i++) {
                        step();
                        if(save_chain) chain.push_back(current);
                        if(action) (*action)(current);
                        if(pbar && (i + 1) % pbar_stride == 0) pbar->set_progress(burnin + i + 1);
                    }
                    if(pbar) {
                        pbar->finish();
                        std::cerr << std::endl;
                    }
                    if(naccept == 0)
                        log<LOG_WARNING>(L"%1% || Metropolis chain accepted 0 of %2% proposals — the chain is frozen at its start"
                                         L" point (check bounds vs start values and proposal widths); results from this chain are meaningless.")
                            % __func__ % (burnin + steps);
                }

                void plot_autocorrelation(const std::string &filename, const std::vector<std::string> &param_names, std::optional<std::map<std::string, TObject*>*> drawn_objs, size_t max_lag = 1000) const {
                    if(chain.size() == 0) {
                        log<LOG_ERROR>(L"%1% || Error: cannot calculate autocorrelation without a saved chain."
                                       L" Did you forget to run the Metropolis object, or tell the Metropolis"
                                       L" object to not save the chain?")
                            % __func__;
                        log<LOG_ERROR>(L"%1% || Not saving autocorrelations as a result.");
                        return;
                    }
                    long nparam = chain[0].size();
                    if(param_names.size() < (size_t)nparam) {
                        log<LOG_ERROR>(L"%1% || Passed in parameter names is not the same size (%2%) "
                                       L"as the number of parameters in each step of the chain (%3%).")
                            % __func__ % param_names.size() % nparam;
                        log<LOG_ERROR>(L"%1% || Not saving autocorrelations as a result.");
                        return;
                    }
                    std::vector<std::pair<TH1D*,TH1D*>> hs;
                    TH1D zero("z","",max_lag, 0, max_lag);
                    for(size_t i = 1; i <= max_lag; ++i) zero.SetBinContent(i, 0);
                    zero.SetLineStyle(kDashed);
                    TCanvas c;
                    c.Divide(2);
                    c.Print((filename + "[").c_str());
                    for(long i = 0; i < nparam; ++i) {
                        hs.emplace_back(new TH1D(("hautoc"+std::to_string(i)).c_str(), (param_names[i]+";lag;abs(autocorrelation)").c_str(), max_lag, 0, max_lag), new TH1D(("h2autoc"+std::to_string(i)).c_str(), (param_names[i]+";lag;autocorrelation").c_str(), max_lag, 0, max_lag));
                        int n = chain.size();
                        std::vector<double> values;
                        values.reserve(n);
                        double mean = 0;
                        for(const auto &step : chain) {
                            values.push_back(step(i));
                            mean += step(i);
                        }
                        mean /= n;
                        for(double &v : values) v -= mean;
                        TVirtualFFT *fft = TVirtualFFT::FFT(1, &n, "R2C");
                        fft->SetPoints(values.data());
                        fft->Transform();
                        std::vector<double> fft_pts;
                        std::vector<double> ims(n, 0); // Dummy vector to hold imag value of 0 for each point.
                        for(int k = 0; k < n; ++k) {
                            double re, im;
                            fft->GetPointComplex(k, re, im);
                            fft_pts.push_back(re*re+im*im);
                        }
                        TVirtualFFT *ifft = TVirtualFFT::FFT(1, &n, "C2R");
                        ifft->SetPointsComplex(fft_pts.data(), ims.data());
                        ifft->Transform();
                        double lag0 = ifft->GetPointReal(0), klag = 0;
                        // A constant chain column gives lag0 == 0; klag/lag0 would put literal
                        // 'nan's into the PDF stream and corrupt the page. Plot zeros instead.
                        if(!std::isfinite(lag0) || lag0 <= 0) {
                            log<LOG_WARNING>(L"%1% || Chain for parameter %2% is constant/degenerate (lag0 = %3%);"
                                             L" autocorrelation undefined, plotting zeros.")
                                % __func__ % param_names[i].c_str() % lag0;
                        } else {
                            for(int k = 0; k < n && k < max_lag; ++k) {
                                klag = ifft->GetPointReal(k);
                                hs.back().first->SetBinContent(k+1, std::abs(klag/lag0));
                                hs.back().second->SetBinContent(k+1, klag/lag0);
                            }
                            log<LOG_INFO>(L"%1% || Lag %2% autocorrelation for parameter %3% is %4%.")
                                % __func__ % std::min(max_lag, (size_t)n) % param_names[i].c_str() % (klag/lag0);
                        }
                        c.cd(1);
                        gPad->SetLogy(1);
                        hs.back().first->Draw("l");
                        c.cd(2);
                        gPad->SetLogy(0);
                        hs.back().second->Draw("l");
                        zero.Draw("l same");
                        drawVersionWatermark(&c, WatermarkPos::BottomRight);
                        c.Print(filename.c_str());
                        if(drawn_objs) {
                            (*drawn_objs)->insert({param_names[i]+"_autocorr", hs.back().second->Clone()});
                        }
                    }
                    c.Clear();
                    gStyle->SetPalette(kCool);
                    TLegend leg(0.3, 0.6, 0.89, 0.89);
                    leg.SetNColumns(2);
                    leg.SetFillStyle(0);
                    leg.SetLineWidth(0);
                    int i = 0;
                    for(auto [h, _] : hs) {
                        gPad->SetLogy(1);
                        h->SetTitle(";lag;abs(autocorrelation)");
                        h->SetMinimum(1e-5);
                        h->Draw("plclsame");
                        leg.AddEntry(h, param_names[i++].c_str(), "l");
                    }
                    leg.Draw("same");
                    drawVersionWatermark(&c);
                    c.Print(filename.c_str());
                    gPad->SetLogy(0);
                    i = 0;
                    TLegend leg2(0.3, 0.6, 0.89, 0.89);
                    leg2.SetNColumns(2);
                    leg2.SetFillStyle(0);
                    leg2.SetLineWidth(0);
                    zero.SetTitle(";lag;autocorrelation");
                    zero.SetMinimum(-0.07);
                    zero.SetMaximum(1.05);
                    zero.Draw("l");
                    for(auto [_, h] : hs) {
                        gPad->SetLogy(0);
                        h->SetTitle(";lag;autocorrelation");
                        h->Draw("plclsame");
                        leg2.AddEntry(h, param_names[i++].c_str(), "l");
                    }
                    leg2.Draw("same");
                    drawVersionWatermark(&c);
                    c.Print(filename.c_str());
                    c.Print((filename + "]").c_str());
                    set_matrix_palette();
                }

        };

    struct simple_target {
        PROmetric &metric;

        // Returns log-target (-0.5*chi^2). Metropolis::step does exp(target(p) - target(current))
        // so the exp argument stays in safe float32 range even when chi^2 is large.
        float operator()(Eigen::VectorXf &value) {
            Eigen::VectorXf empty = value;
            return -0.5f*metric(value, empty, false);
        }
    };

    // This target distribution uses a prior which is uniform in linear space for models
    // where the parameters are in log space.
    struct unilin_prior_target {
        PROmetric &metric;

        // Returns log-target (-0.5*chi^2). Metropolis::step does exp(target(p) - target(current))
        // so the exp argument stays in safe float32 range even when chi^2 is large.
        float operator()(Eigen::VectorXf &value) {
            Eigen::VectorXf empty = value;
            Eigen::VectorXf corr = value;
            //corr.segment(0,2) = corr.segment(0,2).array().log10();
            //return -0.5f*metric(value, empty, false)+value.segment(0,2).array().sum() - std::log(100-0.01);
            return -0.5f*metric(corr, empty, false)+corr.segment(0,2).array().sum();
            //return -0.5f*metric(corr, empty, false)-corr(0)-corr(1);
        }
    };

    struct prior_only_target {
        PROmetric &metric;

        float operator()(Eigen::VectorXf &value) {
            Eigen::VectorXf nuisance = value.segment(metric.GetModel().nparams, metric.GetSysts().GetNSplines());
            return -0.5f*metric.Pull(nuisance);
        }
    };

    struct simple_proposal {
        PROmetric &metric;
        uint32_t seed;
        float width;
        std::vector<int> fixed;
        std::mt19937 rng;
        static constexpr bool has_tune = true;
        std::vector<bool> accepted_list;
        size_t tune_calls = 0;
        float last_acceptance = -1;
        float last_shift;

        simple_proposal(PROmetric &metric, uint32_t seed, float width = 0.2, std::vector<int> fixed = {}) 
            : metric(metric), seed(seed), width(width), fixed(fixed), rng(seed), last_shift(width) {
                accepted_list = std::vector(1000, false);
            }

        Eigen::VectorXf operator()(Eigen::VectorXf &current) {
            Eigen::VectorXf ret = current;
            long int nparams = metric.GetModel().nparams;
            for(long int i = 0; i < ret.size(); ++i) {
                if(std::find(fixed.begin(), fixed.end(), i) != std::end(fixed)) continue;
                if(i < nparams) {
                    float lo = ret(i) - width;
                    float hi = ret(i) + width;
                    std::uniform_real_distribution<float> ud(lo, hi);
                    ret(i) = ud(rng);
                } else if(metric.GetSysts().spline_lo[i-nparams] == 0) {
                    // Currently there's some weird behavior with the 0-1 systematics
                    // which using a uniform distribution seems to fix
                    // TODO: How to use width with a uniform distribution
                    //float lo = metric.GetSysts().spline_lo[i-nparams];
                    //float hi = metric.GetSysts().spline_hi[i-nparams];
                    float lo = ret(i) - width;
                    float hi = ret(i) + width;
                    std::uniform_real_distribution<float> ud(lo, hi);
                    ret(i) = ud(rng);
                } else {
                    std::normal_distribution<float> nd(current(i), width);
                    float proposed_value = nd(rng);
                    ret(i) = proposed_value;
                    //ret(i) = std::clamp(proposed_value, metric.GetSysts().spline_lo[i-nparams], metric.GetSysts().spline_hi[i-nparams]);
                }
            }
            return ret;
        }

        float P(const Eigen::VectorXf &value, const Eigen::VectorXf &given) {
            float prob = 1.0;
            long int nparams = metric.GetModel().nparams;
            for(long int i = 0; i < value.size(); ++i) {
                if(std::find(fixed.begin(), fixed.end(), i) != std::end(fixed)) continue;
                if(i < nparams) {
                    //float diff = metric.GetModel().ub(i) - metric.GetModel().lb(i);
                    //if(std::isinf(diff)) diff = 5;
                    //prob *= 1.0f / diff;
                    prob *= 1.0f / (2 * width);
                }else if(metric.GetSysts().spline_lo[i-nparams] == 0) {
                    //float lo = metric.GetSysts().spline_lo[i-nparams];
                    //float hi = metric.GetSysts().spline_hi[i-nparams];
                    //prob *= 1.0f / (hi - lo);
                    prob *= 1.0f / (2 * width);
                } else {
                    //if(value(i) <= metric.GetSysts().spline_lo[i-nparams] || value(i) >= metric.GetSysts().spline_hi[i-nparams] || 
                    //   given(i) <= metric.GetSysts().spline_lo[i-nparams] || given(i) >= metric.GetSysts().spline_hi[i-nparams]) {
                    //    // Due to bounds, use CDF to get total probability value is <= bound
                    //    // Symmetry makes this work for upper bound as well
                    //    float v = std::clamp(value(i), metric.GetSysts().spline_lo[i-nparams], metric.GetSysts().spline_hi[i-nparams]);
                    //    float g = std::clamp(given(i), metric.GetSysts().spline_lo[i-nparams], metric.GetSysts().spline_hi[i-nparams]);
                    //    prob *= 0.5f * (1.0f + std::erff((v - g)/(std::sqrt(2.0f)*width)));
                    //    //prob = 0;
                    //} else {
                    prob *= (1.0f / std::sqrt(2 * M_PI * width * width))
                        * std::exp(-(value(i) - given(i))*(value(i) - given(i))/(2 * width * width));
                    //}
                }
            }
            return prob;
        }

        bool within_bound(const Eigen::VectorXf &value) {
            long int nparams = metric.GetModel().nparams;
            for(long int i = 0; i < value.size(); ++i) {
                if(std::find(fixed.begin(), fixed.end(), i) != std::end(fixed)) continue;
                if(i < nparams) {
                    // A finite model bound is authoritative; the -5 floor only stops runaway
                    // exploration of log-space params whose model lb is -inf.
                    float lo = std::isfinite(metric.GetModel().lb(i)) ? metric.GetModel().lb(i) : -5.0f;
                    if(value(i) > metric.GetModel().ub(i) || value(i) < lo)
                        return false;
                } else {
                    size_t si = i - nparams;
                    float lo, hi;
                    if(metric.GetSysts().spline_has_restrict[si]) {
                        lo = metric.GetSysts().spline_restrict_lo[si];
                        hi = metric.GetSysts().spline_restrict_hi[si];
                    } else if(metric.GetSysts().spline_hi[si] == 1.0f) {
                        lo = -1.0f; hi = 1.0f;
                    } else {
                        lo = metric.GetSysts().spline_lo[si];
                        hi = metric.GetSysts().spline_hi[si];
                    }
                    if(value(i) < lo || value(i) > hi) return false;
                }
            }
            return true;
        }

        void tune(bool accepted) {
            accepted_list[tune_calls % 1000] = accepted;
            if(++tune_calls % 1000 == 0) {
                float acceptance = std::count(accepted_list.begin(), accepted_list.end(), true) / 1000.0f;
                if(acceptance < 0.20 || acceptance > 0.30) {
                    if(std::abs(acceptance - 0.234) < std::abs(last_acceptance - 0.234)) {
                        if(last_acceptance < 0) {
                            width *= 1.25; // Default first step
                            last_shift = 0.25 * width;
                        } else {
                            width += last_shift;
                        }
                    } else { // Moved too far
                        width += -0.5 * last_shift;
                        last_shift *= -0.5;
                    }
                }
                for(size_t i = 0; i < 1000; ++i) accepted_list[i] = false;
                last_acceptance = acceptance;
            }
        }
    };

    struct adaptive_proposal {
        PROmetric &metric;
        uint32_t seed;
        Eigen::MatrixXf width;
        std::vector<int> fixed;//fixed and active are opposite. usually active is 
        std::vector<int> active;
        std::mt19937 rng;
        static constexpr bool has_tune = true;
        //std::vector<Eigen::VectorXf> proposed;
        Eigen::VectorXf last_proposed;
        Eigen::VectorXf last_accepted;
        Eigen::VectorXf mean;
        Eigen::MatrixXf cov;
        size_t tune_calls = 0;
        float scale = 5.66;
        float beta = 1.0;
        
        float diag_scale = 0.01;
        Eigen::MatrixXf diagL;
        Eigen::MatrixXf sub_diagL;

        // Adaptive scaling state
        std::vector<bool> accept_history;
        size_t adapt_window = 1000;  // window size for adaptation
        float target_accept = 0.234;
        float adapt_factor = 1.1;


        Eigen::MatrixXf sub_L;
        bool tune_mode;

        adaptive_proposal(PROmetric &metric, uint32_t seed, std::vector<int> fixed = {}) 
            : metric(metric), seed(seed), fixed(fixed), rng(seed) {
                int nparams = metric.GetModel().nparams + metric.GetSysts().GetNSplines();
                scale /= nparams - fixed.size();
                diag_scale /= nparams - fixed.size();
                width = Eigen::MatrixXf::Identity(nparams, nparams);
                mean = Eigen::VectorXf::Constant(nparams, 0);
                cov = Eigen::MatrixXf::Identity(nparams, nparams);
                Eigen::MatrixXf diag = Eigen::MatrixXf::Identity(nparams, nparams);
                Eigen::LLT<Eigen::MatrixXf> llt(diag_scale * diag);
                diagL = llt.matrixL();

                tune_mode = true;
                for (int i = 0; i < nparams; ++i) {
                    if (std::find(fixed.begin(), fixed.end(), i) == fixed.end()) {
                        active.push_back(i);
                    }
                }
                //grab the bits that correspond to the active only.
                sub_diagL = Eigen::MatrixXf::Identity(active.size(), active.size());
                sub_diagL = diagL(active, active);  
                last_accepted = Eigen::VectorXf::Zero(nparams);

            }

        Eigen::VectorXf operator()(Eigen::VectorXf &current) {

            Eigen::VectorXf sub_throw1(active.size());
            Eigen::VectorXf sub_throw2(active.size());
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (size_t i = 0; i < active.size(); ++i) {
                sub_throw1(i) = nd(rng);
                sub_throw2(i) = nd(rng);
            }
            
            Eigen::MatrixXf inp = scale*width;//fulldim

            if(tune_mode)sub_L = ComputeSquareRootCovariance(inp(active, active)); //magic eigen indexing

            last_proposed = current;//fulldim
            last_proposed(active) += (1.0f - beta) * sub_L * sub_throw1 + beta * sub_diagL * sub_throw2; //More index nonsesne?!

            return last_proposed;
        }

        float P(const Eigen::VectorXf &, const Eigen::VectorXf &) {
            return 1;
        }


        bool within_bound(const Eigen::VectorXf &value) {
            long int nparams = metric.GetModel().nparams;
            for(long int i = 0; i < value.size(); ++i) {
                if(std::find(fixed.begin(), fixed.end(), i) != std::end(fixed)) continue;
                if(i < nparams) {
                    // A finite model bound is authoritative; the -5 floor only stops runaway
                    // exploration of log-space params whose model lb is -inf. The old
                    // max(lb, -5) froze the whole chain when the best fit sat below -5
                    // (e.g. Asimov no-signal fits driving a mixing angle to its bound).
                    float lo = std::isfinite(metric.GetModel().lb(i)) ? metric.GetModel().lb(i) : -5.0f;
                    if(value(i) > metric.GetModel().ub(i) || value(i) < lo)
                        return false;
                } else {
                    size_t si = i - nparams;
                    float lo, hi;
                    if(metric.GetSysts().spline_has_restrict[si]) {
                        lo = metric.GetSysts().spline_restrict_lo[si];
                        hi = metric.GetSysts().spline_restrict_hi[si];
                    } else if(metric.GetSysts().spline_hi[si] == 1.0f) {
                        lo = -1.0f; hi = 1.0f;
                    } else {
                        lo = metric.GetSysts().spline_lo[si];
                        hi = metric.GetSysts().spline_hi[si];
                    }
                    if(value(i) < lo || value(i) > hi) return false;
                }
            }
            return true;
        }


        void tune(bool accepted) {

            //if (accepted) {
                ++tune_calls;
                if(accepted) last_accepted = last_proposed;
                Eigen::VectorXf delta = last_accepted - mean;
                mean += delta / tune_calls;
                if (tune_calls > 1) {
                    cov += (delta * (last_accepted - mean).transpose() - cov) / tune_calls;
                }
                for (int idx : fixed) {
                    cov.row(idx).setZero();
                    cov.col(idx).setZero();
                }

            //}
            accept_history.push_back(accepted);
            if (accept_history.size() == adapt_window) {
                float acc_rate = std::count(accept_history.begin(), accept_history.end(), true) / float(adapt_window);

                if (acc_rate > target_accept) {
                    scale *= adapt_factor;
                } else {
                    scale /= adapt_factor;
                }
                accept_history.clear();
                log<LOG_DEBUG>(L"%1% || Adaptive scale updated to %2%, acceptance rate = %3%") % __func__ % scale % acc_rate;
            }

            if(tune_calls > (size_t)(4*last_proposed.size())) {
                Eigen::MatrixXf cov_pd = cov;
                for(long int i = 0; i < cov_pd.rows(); ++i)
                    cov_pd(i, i) += 1e-8;
                width = cov_pd;
                beta = tune_calls > (size_t)(100*last_proposed.size()) ? 0.05 : 0.5;
            }


        }

    };

class HMC {
    public:
        float epsilon, L;
        std::vector<Eigen::VectorXf> chain;

        void leapfrog(Eigen::VectorXf &theta, Eigen::VectorXf &r, PROmetric &metric) {
            Eigen::VectorXf g = theta;
            metric(theta, g, true);
            r += epsilon/2 * g;
            theta += epsilon * r;
            metric(theta, g, true);
            r += epsilon/2 * g;
        }

        void operator()(Eigen::VectorXf theta0, PROmetric &metric, size_t M, uint32_t seed) {
            std::mt19937 rng(seed);
            std::normal_distribution<float> normal(0, 1);
            std::uniform_real_distribution<float> uniform(0, 1);
            Eigen::VectorXf empty;
            chain.push_back(theta0);
            Eigen::VectorXf r0 = theta0, r = r0; // Just set the size, we'll set the values with rng in the loop
            for(size_t m = 0; m < M; ++m) {
                Eigen::VectorXf theta = chain.back();
                for(long i = 0; i < r.size(); ++i) r(i) = normal(rng);
                r0 = r;
                for(size_t i = 0; i < L; ++i) leapfrog(theta, r, metric);
                float alpha = std::min(1.0f, expf(metric(theta, empty, false) - 0.5 * r.dot(r) - metric(chain.back(), empty, false) + 0.5 * r0.dot(r0)));
                if(alpha < uniform(rng)) {
                    chain.push_back(theta);
                } else {
                    chain.push_back(theta0);
                }
            }
        }
};

class NUTS {
        std::mt19937 rng;
    public:
        float M, Madapt;
        // These are the values recommended in the NUTS papaer, but they are tunable
        float delta = 0.6, DeltaMax = 1000;
        std::vector<Eigen::VectorXf> chain;

        void normalize_theta(Eigen::VectorXf &theta) {
            while(theta(0) > 2 || theta(0) < -2) {
                if(theta(0) > 2) {
                    theta(0) -= 2;
                    theta(0) *= -1;
                    theta(0) += 2;
                }
                if(theta(0) < -2) {
                    theta(0) += 2;
                    theta(0) *= -1;
                    theta(0) -= 2;
                }
            }
            while(theta(1) > 0 || theta(1) < -3) {
                if(theta(1) > 0) {
                    theta(1) *= -1;
                }
                if(theta(1) < -3) {
                    theta(1) += 3;
                    theta(1) *= -1;
                    theta(1) -= 3;
                }
            }
        }

        float eval(PROmetric &metric, const Eigen::VectorXf &theta, Eigen::VectorXf &grad, bool run_grad) {
            float v = -0.5 * metric(theta, grad, run_grad) + theta(0) + theta(1);
            if(run_grad) {
                grad(0) += 1;
                grad(1) += 1;
            }
            return v;
        }

        void leapfrog(Eigen::VectorXf &theta, Eigen::VectorXf &r, float epsilon, PROmetric &metric) {
            Eigen::VectorXf g = theta;
            eval(metric, theta, g, true);
            r -= epsilon/4 * g;
            theta += epsilon * r;
            normalize_theta(theta);
            eval(metric, theta, g, true);
            r -= epsilon/4 * g;
        }

        float find_reasonable_epsilon(const Eigen::VectorXf &theta, PROmetric &metric) {
            float epsilon = 1;
            Eigen::VectorXf empty;
            Eigen::VectorXf r = theta;
            std::normal_distribution<float> normal(0, 1);
            for(long i = 0; i < r.size(); ++i) r(i) = normal(rng);
            Eigen::VectorXf thetap = theta;
            Eigen::VectorXf rp = r;
            leapfrog(thetap, rp, epsilon, metric);
            float a = 2*(expf(eval(metric, thetap, empty, false) - eval(metric, theta, empty, false) + 0.5 * (r.dot(r) - rp.dot(rp)) ) > 0.5) - 1;
            while(powf(expf(eval(metric, thetap, empty, false) -  eval(metric, theta, empty, false) + 0.5 * (r.dot(r) - rp.dot(rp))), a) > powf(2, -a)) {
                epsilon *= powf(2, a);
                thetap = theta;
                rp = r;
                leapfrog(thetap, rp, epsilon, metric);
            }
            return epsilon;
        }

        struct BTR {
            Eigen::VectorXf theta_minus, theta_plus, theta_prime,
                            r_minus, r_plus;
            float n, s, alpha, nalpha;
        };

        BTR BuildTree(const Eigen::VectorXf &theta, const Eigen::VectorXf &r, float u, int v, 
                      int j, float epsilon, const Eigen::VectorXf &theta0, const Eigen::VectorXf &r0, PROmetric &metric) {
            BTR ret;
            if(j == 0) {
                ret.theta_minus = theta;
                ret.r_minus = r;
                leapfrog(ret.theta_minus, ret.r_minus, v*epsilon, metric);
                ret.theta_plus = ret.theta_minus;
                ret.theta_prime = ret.theta_minus;
                ret.r_plus = ret.r_minus;
                Eigen::VectorXf empty;
                ret.n = (u <= std::exp(eval(metric, ret.theta_prime, empty, false) - 0.5 * ret.r_minus.dot(ret.r_minus)));
                ret.s = (u < std::exp(DeltaMax + eval(metric, ret.theta_prime, empty, false) - 0.5 * ret.r_minus.dot(ret.r_minus)));
                ret.alpha = std::min(1.0f, std::exp(eval(metric, ret.theta_prime, empty, false) - 0.5f * ret.r_minus.dot(ret.r_minus)
                                                  - eval(metric, theta0, empty, false) + 0.5f * r0.dot(r0)));
                ret.nalpha = 1;
                return ret;
            }
            ret = BuildTree(theta, r, u, v, j-1, epsilon, theta0, r0, metric);
            std::uniform_real_distribution<float> uniform(0, 1);
            if(ret.s == 1) {
                Eigen::VectorXf t,r, tp = ret.theta_prime;
                float np = ret.n, alphap = ret.alpha, nalphap = ret.nalpha;
                if(v == -1) {
                    t = ret.theta_plus;
                    r = ret.r_plus;
                    ret = BuildTree(ret.theta_minus, ret.r_minus, u, v, j-1, epsilon, theta0, r0, metric);
                    ret.theta_plus = t;
                    ret.r_plus = r;
                } else {
                    t = ret.theta_minus;
                    r = ret.r_minus;
                    ret = BuildTree(ret.theta_plus, ret.r_plus, u, v, j-1, epsilon, theta0, r0, metric);
                    ret.theta_minus = t;
                    ret.r_minus = r;
                }
                if(uniform(rng) > std::min((ret.n/(ret.n+np)),1.0f))
                    ret.theta_prime = tp;
                ret.alpha += alphap;
                ret.nalpha += nalphap;
                ret.s = ret.s * ((ret.theta_plus - ret.theta_minus).dot(ret.r_minus) >= 0) 
                              * ((ret.theta_plus - ret.theta_minus).dot(ret.r_plus) >= 0);
                ret.n += np;
            }
            return ret;
        }

        void operator()(const Eigen::VectorXf &theta0, PROmetric &metric, uint32_t seed) {
            rng.seed(seed);
            std::normal_distribution<float> normal(0, 1);
            std::uniform_real_distribution<float> uniform(0, 1);
            Eigen::VectorXf theta = theta0, empty;
            chain.push_back(theta0);
            float epsilon = find_reasonable_epsilon(theta, metric),
                  mu = std::log(10 * epsilon),
                  epsilonbar = 1,
                  Hbar = 0,
                  gamma = 0.05,
                  t0 = 10,
                  kappa = 0.75;
            Eigen::VectorXf r0 = theta;
            for(size_t m = 0; m < M; ++m) {
                for(long i = 0; i < r0.size(); ++i) r0(i) = normal(rng);
                float u = std::uniform_real_distribution<float>(0, std::exp(eval(metric, theta, empty, false) - 0.5 * r0.dot(r0)))(rng);
                BTR t{theta, theta, theta, r0, r0, 1, 1, 0, 0};
                int j = 0;
                while(t.s == 1) {
                    int v = 2 * (uniform(rng) < 0.5) - 1;
                    BTR tp;
                    if(v == -1) {
                        tp = BuildTree(t.theta_minus, t.r_minus, u, v, j, epsilon, chain.back(), r0, metric);
                        t.theta_minus = tp.theta_minus;
                        t.r_minus = tp.r_minus;
                        t.alpha = tp.alpha;
                        t.nalpha = tp.nalpha;
                    } else {
                        tp = BuildTree(t.theta_plus, t.r_plus, u, v, j, epsilon, chain.back(), r0, metric);
                        t.theta_plus = tp.theta_plus;
                        t.r_plus = tp.r_plus;
                        t.alpha = tp.alpha;
                        t.nalpha = tp.nalpha;
                    }
                    if(tp.s == 1) {
                        if(uniform(rng) < std::min(1.0f, tp.n/t.n))
                            theta = tp.theta_prime;
                    }
                    t.n += tp.n;
                    t.s = tp.s * ((t.theta_plus - t.theta_minus).dot(t.r_minus) >= 0) * ((t.theta_plus - t.theta_minus).dot(t.r_plus) >= 0);
                    j++;
                    //log<LOG_ERROR>(L"%1% || Iteration %2%: %3%") % __func__ % j % theta;
                    if(j >= 10) break;
                }
                if(m < Madapt) {
                    Hbar = (1 - 1/(m + 1 + t0)) * Hbar + 1/(m + 1 + t0) * (delta - t.alpha/t.nalpha);
                    epsilon = std::exp(mu - sqrt(m+1)/gamma * Hbar);
                    epsilonbar = std::exp(std::pow(m+1, -kappa) * std::log(epsilon) + (1-std::pow(m+1, -kappa))*std::log(epsilonbar));
                } else {
                    epsilon = epsilonbar;
                }
                log<LOG_ERROR>(L"%1% || Adding %2% to chain of length %3%.") % __func__ % theta % chain.size();
                chain.push_back(theta);
            }
        }
};


};

#endif

