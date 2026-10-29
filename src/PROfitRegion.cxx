#include "PROfitRegion.h"
#include "PROcess.h"
#include "PROlog.h"
#include "PROtocall.h"

#include "TCanvas.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TLegend.h"
#include "TLine.h"
#include "TMath.h"
#include "TPad.h"
#include "TPaveText.h"

#include <algorithm>
#include <cmath>

namespace PROfit {

    bool SelectChannelsByPattern(const PROconfig &config, const std::string &pattern,
                                 std::vector<size_t> &matched_channels, const std::string &what) {
        matched_channels.clear();
        if(pattern.empty()) {
            log<LOG_ERROR>(L"%1% || Empty %2% pattern.") % __func__ % what.c_str();
            return false;
        }

        // Walk the canonical mode -> detector -> channel -> subchannel nesting so the
        // running counter matches the ordering of config.m_fullnames, and demand that a
        // channel's subchannels are either all matched or all unmatched: the chi2 lives
        // in collapsed (channel) space, so a partially-selected channel is ill-defined.
        size_t global_subchannel_index = 0;
        size_t global_channel_index = 0;
        bool ok = true;
        // Unanchored regex (plain substrings work); see PROconfig.h.
        std::regex re = CompilePattern(pattern, what);
        for(size_t im = 0; im < config.m_num_modes; ++im) {
            for(size_t id = 0; id < config.m_num_detectors; ++id) {
                for(size_t ic = 0; ic < config.m_num_channels; ++ic, ++global_channel_index) {
                    size_t n_match = 0;
                    const size_t nsub = config.m_num_subchannels[ic];
                    for(size_t sc = 0; sc < nsub; ++sc, ++global_subchannel_index) {
                        const std::string &fullname = config.m_fullnames[global_subchannel_index];
                        if(PatternMatches(fullname, re)) ++n_match;
                    }
                    if(n_match == nsub && nsub > 0) {
                        matched_channels.push_back(global_channel_index);
                    } else if(n_match != 0) {
                        log<LOG_ERROR>(L"%1% || %2% '%3%' matches %4%/%5% subchannels of channel %6% (mode %7%, detector %8%). "
                                L"A channel must be selected as a whole; refine the pattern.")
                            % __func__ % what.c_str() % pattern.c_str() % n_match % nsub
                            % config.m_channel_names[ic].c_str()
                            % config.m_mode_names[im].c_str() % config.m_detector_names[id].c_str();
                        ok = false;
                    }
                }
            }
        }
        if(!ok) return false;
        if(matched_channels.empty()) {
            log<LOG_ERROR>(L"%1% || %2% '%3%' matched no channels.") % __func__ % what.c_str() % pattern.c_str();
            return false;
        }
        return true;
    }

    std::vector<size_t> ComplementChannels(const PROconfig &config, const std::vector<size_t> &channels) {
        const size_t nch = config.m_num_modes * config.m_num_detectors * config.m_num_channels;
        std::vector<size_t> out;
        for(size_t gc = 0; gc < nch; ++gc)
            if(std::find(channels.begin(), channels.end(), gc) == channels.end()) out.push_back(gc);
        return out;
    }

    std::string GlobalChannelName(const PROconfig &config, size_t gc) {
        const size_t ic = gc % config.m_num_channels;
        const size_t id = (gc / config.m_num_channels) % config.m_num_detectors;
        const size_t im = gc / (config.m_num_channels * config.m_num_detectors);
        return config.m_mode_names[im] + "_" + config.m_detector_names[id] + "_" + config.m_channel_names[ic];
    }

    Eigen::VectorXf CollapsedChannelMask(const PROconfig &config, const std::vector<size_t> &channels, size_t var_index) {
        Eigen::VectorXf mask = Eigen::VectorXf::Zero(config.m_num_variable_bins_total_collapsed[var_index]);
        for(size_t gc : channels) {
            const size_t start = config.GetCollapsedGlobalVariableBinStart(gc, var_index);
            const size_t nbins = config.GetChannelVariableBins(gc, var_index).NBins();
            mask.segment((Eigen::Index)start, (Eigen::Index)nbins).setOnes();
        }
        return mask;
    }

    void InstallFitRegion(PROconfig &config, const std::vector<size_t> &channels) {
        for(size_t io = 0; io < config.m_num_variables; ++io) {
            const Eigen::VectorXf maskv = CollapsedChannelMask(config, channels, io);
            std::vector<char> mask(maskv.size());
            for(Eigen::Index i = 0; i < maskv.size(); ++i)
                mask[(size_t)i] = maskv(i) > 0.5f ? 1 : 0;
            config.SetActiveBins(io, mask);
        }
    }

    bool MaskDataToChannels(const PROconfig &config, const std::vector<size_t> &channels,
                            std::vector<PROdata> &variable_data, PROdata &data, const std::string &what) {
        const size_t nvar = std::min(variable_data.size(), config.m_num_variables);
        for(size_t io = 0; io < nvar; ++io) {
            if(variable_data[io].GetNbins() != config.m_num_variable_bins_total_collapsed[io]) {
                log<LOG_WARNING>(L"%1% || %2%: variable %3% data has %4% bins, expected %5%; leaving it unmasked.")
                    % __func__ % what.c_str() % io % variable_data[io].GetNbins()
                    % config.m_num_variable_bins_total_collapsed[io];
                continue;
            }
            const Eigen::VectorXf mask = CollapsedChannelMask(config, channels, io);
            PROdata masked(mask.cwiseProduct(variable_data[io].Spec()),
                           mask.cwiseProduct(variable_data[io].Error()));
            masked.hash = variable_data[io].hash;
            variable_data[io] = std::move(masked);
        }
        data = variable_data[config.i_prime];
        if(data.Spec().sum() <= 0) {
            log<LOG_ERROR>(L"%1% || %2%: the data bins are all empty after masking. Nothing to fit.") % __func__ % what.c_str();
            return false;
        }
        return true;
    }

    bool FitRegionSetup(FitRegionRunConfig &frconf, PROconfig &config,
                        std::vector<PROdata> &variable_data, PROdata &data,
                        std::vector<std::string> &fixed_params, const PROmodel &model) {
        if(!frconf.active()) return true;
        log<LOG_WARNING>(L"%1% || ############### Fit-region mode ###############") % __func__;

        if(!SelectChannelsByPattern(config, frconf.fit_pattern, frconf.fit_channels, "--fit-region")) return false;
        const std::vector<size_t> outside = ComplementChannels(config, frconf.fit_channels);
        if(outside.empty()) {
            log<LOG_ERROR>(L"%1% || --fit-region '%2%' matched EVERY channel; nothing left to evaluate.")
                % __func__ % frconf.fit_pattern.c_str();
            return false;
        }
        if(frconf.eval_pattern.empty()) {
            frconf.eval_channels = outside;
        } else {
            if(!SelectChannelsByPattern(config, frconf.eval_pattern, frconf.eval_channels, "--eval-region")) return false;
            for(size_t gc : frconf.eval_channels) {
                if(std::find(frconf.fit_channels.begin(), frconf.fit_channels.end(), gc) != frconf.fit_channels.end()) {
                    log<LOG_ERROR>(L"%1% || Channel %2% is in both --fit-region and --eval-region; the evaluation must use data the fit never saw.")
                        % __func__ % GlobalChannelName(config, gc).c_str();
                    return false;
                }
            }
        }
        for(size_t gc : frconf.fit_channels)
            log<LOG_INFO>(L"%1% || Fit region  : %2%") % __func__ % GlobalChannelName(config, gc).c_str();
        for(size_t gc : frconf.eval_channels)
            log<LOG_INFO>(L"%1% || Eval region : %2%") % __func__ % GlobalChannelName(config, gc).c_str();

        InstallFitRegion(config, frconf.fit_channels);
        if(frconf.blind_eval) {
            log<LOG_WARNING>(L"%1% || --blind-eval: data outside the fit region is zeroed; only the constrained prediction is reported.") % __func__;
            if(!MaskDataToChannels(config, frconf.fit_channels, variable_data, data, "--fit-region")) return false;
        }

        if(!frconf.float_physics) {
            for(size_t i = 0; i < model.nparams; ++i) {
                const std::string &name = model.param_names[i];
                const std::string &pname = model.pretty_param_names[i];
                if(std::find(fixed_params.begin(), fixed_params.end(), name) != fixed_params.end()) continue;
                if(std::find(fixed_params.begin(), fixed_params.end(), pname) != fixed_params.end()) continue;
                fixed_params.push_back(name);
                log<LOG_INFO>(L"%1% || Fit region: fixing physics parameter %2% at CV (pass --fit-region-float-physics to float it).")
                    % __func__ % name.c_str();
            }
        }
        return true;
    }

    namespace {

        using IndexMap = Eigen::Map<const Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>>;

        // Collapsed prediction as the metric compares it (shape-only: rescaled onto the data per channel).
        Eigen::VectorXf collapsedPrediction(const PROconfig &config, const PROpeller &prop, const PROmetric &metric,
                                            const Eigen::VectorXf &params, const Eigen::VectorXf &data_spec,
                                            Eigen::VectorXf *full_out = nullptr) {
            Eigen::VectorXf full = FillSpectra(config, prop, metric.GetSysts(), metric.GetModel(), params, true, config.i_prime).Spec();
            if(metric.ShapeOnly()) full = ShapeRescaleToData(config, full, data_spec, config.i_prime);
            if(full_out) *full_out = full;
            return CollapseMatrix(config, full, config.i_prime);
        }

        float gaussChi2(const Eigen::VectorXd &delta, const Eigen::MatrixXd &M) {
            if(delta.size() == 0) return 0.0f;
            return (float)delta.dot(M.ldlt().solve(delta));
        }

    }

    bool ComputeFitRegionConstraint(const PROconfig &config, const PROpeller &prop, const PROmetric &metric,
                                    const PROdata &data, const Eigen::VectorXf &params,
                                    const std::vector<int> &fixed, const FitRegionRunConfig &frconf,
                                    FitRegionConstraint &out) {
        out = FitRegionConstraint();
        out.blind = frconf.blind_eval;
        out.shape_only = metric.ShapeOnly();
        const PROsyst &systs = metric.GetSysts();
        const size_t nphys = metric.GetModel().nparams;
        const size_t nspl = systs.GetNSplines();
        const Eigen::VectorXf &d = data.Spec();
        if((size_t)params.size() != nphys + nspl) {
            log<LOG_ERROR>(L"%1% || Parameter vector has %2% entries, expected %3%.") % __func__ % params.size() % (nphys + nspl);
            return false;
        }

        Eigen::VectorXf full_hat;
        const Eigen::VectorXf mu_hat = collapsedPrediction(config, prop, metric, params, d, &full_hat);
        const Eigen::Index nbins = mu_hat.size();

        // Covariance-type systematics at the linearisation point, as the metric builds them.
        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(nbins, nbins);
        if(systs.GetNCovar() > 0) {
            Eigen::MatrixXf Cf = CollapsedScaledCovariance(config, systs.fractional_covariance, full_hat);
            if(metric.ShapeOnly()) {
                const Eigen::SparseMatrix<float> R = ShapeProjectorCollapsed(config, mu_hat, config.i_prime);
                Cf = Eigen::MatrixXf(R * Cf * R.transpose());
            }
            C = Cf.cast<double>();
        }

        // Linear response of the prediction to every free Gaussian-prior spline. Fixed and
        // uniform-prior splines stay at the fit point (a uniform prior has nothing to condition).
        const Eigen::VectorXf lb = metric.LowerBound();
        const Eigen::VectorXf ub = metric.UpperBound();
        std::vector<size_t> prop_splines;
        for(size_t s = 0; s < nspl; ++s) {
            const size_t i = nphys + s;
            const bool is_fixed = (fixed.size() > i && fixed[i]) || std::abs(ub(i) - lb(i)) < 1e-10f;
            const bool uniform = s < systs.spline_prior_types.size() && systs.spline_prior_types[s] == SplinePriorType::Uniform;
            if(is_fixed || uniform) {
                ++out.n_splines_held;
                if(uniform && !is_fixed)
                    log<LOG_WARNING>(L"%1% || Spline %2% has a uniform prior: held at its fit value %3%, its uncertainty is NOT propagated to the evaluation region.")
                        % __func__ % systs.spline_names[s].c_str() % params(i);
                continue;
            }
            prop_splines.push_back(s);
        }
        out.n_splines_propagated = prop_splines.size();

        const Eigen::Index np = (Eigen::Index)prop_splines.size();
        Eigen::MatrixXd J(nbins, np);
        Eigen::VectorXd dtheta(np); // theta_0 - theta_hat
        Eigen::MatrixXd P(np, np);
        const Eigen::MatrixXf prior_full = metric.GetPriorCovariance();
        for(Eigen::Index k = 0; k < np; ++k) {
            const size_t s = prop_splines[k];
            const size_t i = nphys + s;
            const float width = (s < (size_t)systs.spline_priors.size() && systs.spline_priors(s) > 0) ? systs.spline_priors(s) : 1.0f;
            const float h = 0.1f * width;
            const float lo = std::max(params(i) - h, lb(i));
            const float hi = std::min(params(i) + h, ub(i));
            Eigen::VectorXf p = params;
            p(i) = hi;
            const Eigen::VectorXf mu_hi = collapsedPrediction(config, prop, metric, p, d);
            p(i) = lo;
            const Eigen::VectorXf mu_lo = collapsedPrediction(config, prop, metric, p, d);
            J.col(k).setZero();
            if(hi - lo > 1e-6f) J.col(k) = ((mu_hi - mu_lo) / (hi - lo)).cast<double>();
            dtheta(k) = (double)systs.spline_centers(s) - (double)params(i);
            for(Eigen::Index l = 0; l < np; ++l)
                P(k, l) = prior_full(s, prop_splines[l]);
        }

        // Prior-predictive about the linearisation point.
        const Eigen::MatrixXd V = C + J * P * J.transpose();
        const Eigen::VectorXd m = mu_hat.cast<double>() + J * dtheta;

        const Eigen::VectorXf stat_hat = metric.GetStatVariances(mu_hat, d, params);
        for(Eigen::Index b = 0; b < nbins; ++b)
            if(config.IsBinActive(config.i_prime, (size_t)b) && stat_hat(b) > 0) out.fit_bins.push_back(b);
        const Eigen::VectorXf eval_mask = CollapsedChannelMask(config, frconf.eval_channels, config.i_prime);
        // Every eval bin gets a constrained prediction; only those with a positive statistical
        // variance enter the chi2, as in the metric (e.g. zero-data bins under neyman).
        size_t n_dropped = 0;
        for(Eigen::Index b = 0; b < nbins; ++b) {
            if(eval_mask(b) < 0.5f) continue;
            if(stat_hat(b) > 0) out.chi2_rows.push_back((Eigen::Index)out.eval_bins.size());
            else ++n_dropped;
            out.eval_bins.push_back(b);
        }
        if(n_dropped && !out.blind)
            log<LOG_WARNING>(L"%1% || %2% evaluation bin(s) have zero statistical variance (e.g. zero data under neyman) and are left out of the chi2, as the metric would.")
                % __func__ % n_dropped;
        if(out.fit_bins.empty() || out.eval_bins.empty() || (out.chi2_rows.empty() && !out.blind)) {
            log<LOG_ERROR>(L"%1% || No usable %2% bins.") % __func__ % (out.fit_bins.empty() ? "fit-region" : "evaluation");
            return false;
        }
        const IndexMap B(out.fit_bins.data(), (Eigen::Index)out.fit_bins.size());
        const IndexMap S(out.eval_bins.data(), (Eigen::Index)out.eval_bins.size());

        Eigen::MatrixXd Sigma_BB = V(B, B);
        Sigma_BB.diagonal() += stat_hat(B).cast<double>();
        const Eigen::LDLT<Eigen::MatrixXd> ldlt(Sigma_BB);
        const Eigen::VectorXd u = d(B).cast<double>() - m(B);
        out.chi2_fit_linear = (float)u.dot(ldlt.solve(u));

        const Eigen::MatrixXd V_SB = V(S, B);
        const Eigen::VectorXd post_mean = m(S) + V_SB * ldlt.solve(u);
        const Eigen::MatrixXd post_V = V(S, S) - V_SB * ldlt.solve(V_SB.transpose());

        // Pre-constraint: the actual (non-linearised) prediction at the prior centers.
        Eigen::VectorXf params_pre = params;
        for(size_t s = 0; s < nspl; ++s)
            if(std::find(prop_splines.begin(), prop_splines.end(), s) != prop_splines.end())
                params_pre(nphys + s) = systs.spline_centers(s);
        const Eigen::VectorXf mu_pre = collapsedPrediction(config, prop, metric, params_pre, d);

        out.data = d(S);
        out.pre_mean = mu_pre(S);
        out.post_mean = post_mean.cast<float>();
        out.pre_syst_cov = V(S, S).cast<float>();
        out.post_syst_cov = post_V.cast<float>();

        // Statistical variances in the metric's convention, evaluated about the mean under test.
        auto statAt = [&](const Eigen::VectorXf &mean_S) {
            Eigen::VectorXf pred = mu_hat;
            pred(S) = mean_S;
            const Eigen::VectorXf st = metric.GetStatVariances(pred, d, params);
            return Eigen::VectorXf(st(S));
        };
        out.pre_stat = statAt(out.pre_mean);
        out.post_stat = statAt(out.post_mean);

        // Per-channel rows, then the chi2s.
        for(size_t gc : frconf.eval_channels) {
            FitRegionChannelGoF ch;
            ch.channel = gc;
            ch.name = GlobalChannelName(config, gc);
            const Eigen::Index start = (Eigen::Index)config.GetCollapsedGlobalVariableBinStart(gc, config.i_prime);
            const Eigen::Index nb = (Eigen::Index)config.GetChannelVariableBins(gc, config.i_prime).NBins();
            for(size_t r = 0; r < out.eval_bins.size(); ++r) {
                if(out.eval_bins[r] < start || out.eval_bins[r] >= start + nb) continue;
                ch.rows.push_back((Eigen::Index)r);
                if(std::find(out.chi2_rows.begin(), out.chi2_rows.end(), (Eigen::Index)r) != out.chi2_rows.end())
                    ch.chi2_rows.push_back((Eigen::Index)r);
            }
            ch.ndof = (long)ch.chi2_rows.size() - ((out.shape_only && !ch.chi2_rows.empty()) ? 1 : 0);
            if(!ch.rows.empty()) out.channels.push_back(ch);
        }
        out.ndof = 0;
        for(const auto &ch : out.channels) out.ndof += ch.ndof;

        if(!out.blind) {
            const IndexMap X(out.chi2_rows.data(), (Eigen::Index)out.chi2_rows.size());
            const Eigen::VectorXd dS = out.data.cast<double>();
            Eigen::MatrixXd M_pre = out.pre_syst_cov.cast<double>();
            M_pre.diagonal() += out.pre_stat.cast<double>();
            Eigen::MatrixXd M_post = post_V;
            M_post.diagonal() += out.post_stat.cast<double>();
            const Eigen::VectorXd r_pre = dS - out.pre_mean.cast<double>();
            const Eigen::VectorXd r_post = dS - post_mean;
            out.chi2_pre = gaussChi2(r_pre(X), M_pre(X, X));
            out.chi2_post = gaussChi2(r_post(X), M_post(X, X));
            for(auto &ch : out.channels) {
                const IndexMap R(ch.chi2_rows.data(), (Eigen::Index)ch.chi2_rows.size());
                ch.chi2_pre = gaussChi2(r_pre(R), M_pre(R, R));
                ch.chi2_post = gaussChi2(r_post(R), M_post(R, R));
            }
        }
        return true;
    }

    void PrintFitRegionConstraint(const FitRegionConstraint &res, float fit_chi2, const PROndof *fit_ndof) {
        auto pval = [](float chi2, long ndof) { return ndof > 0 ? TMath::Prob(chi2, (int)ndof) : 1.0; };
        log<LOG_INFO>(L"%1% || ==================== Fit-region constraint ====================") % __func__;
        log<LOG_INFO>(L"%1% || Conditioned on %2% fit-region bins; %3% Gaussian-prior spline(s) propagated, %4% held at the fit point.")
            % __func__ % res.fit_bins.size() % res.n_splines_propagated % res.n_splines_held;
        if(fit_chi2 >= 0 && fit_ndof)
            log<LOG_INFO>(L"%1% || Fit region, post-fit       : chi2 = %2% (%3%, p = %4%)")
                % __func__ % fit_chi2 % fit_ndof->describe().c_str() % pval(fit_chi2, fit_ndof->value());
        log<LOG_INFO>(L"%1% || Fit region, linearised     : chi2 = %2% (matches the fit chi2 when the model is ~linear about the fit point)")
            % __func__ % res.chi2_fit_linear;
        if(res.blind) {
            log<LOG_INFO>(L"%1% || Evaluation region is BLINDED (--blind-eval): constrained prediction only.") % __func__;
            for(const auto &ch : res.channels) {
                double pre = 0, post = 0, epre = 0, epost = 0;
                for(Eigen::Index r : ch.rows) {
                    pre += res.pre_mean(r); post += res.post_mean(r);
                }
                const Eigen::Map<const Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>> R(ch.rows.data(), (Eigen::Index)ch.rows.size());
                epre = std::sqrt(std::max(0.0f, res.pre_syst_cov(R, R).sum()));
                epost = std::sqrt(std::max(0.0f, res.post_syst_cov(R, R).sum()));
                log<LOG_INFO>(L"%1% ||   %2% : total events pre %3% +/- %4% (syst) -> post %5% +/- %6% (syst)")
                    % __func__ % ch.name.c_str() % pre % epre % post % epost;
            }
            return;
        }
        const long n = (long)res.chi2_rows.size();
        log<LOG_INFO>(L"%1% || Eval region, pre-constraint : chi2 = %2% / %3% (p = %4%)")
            % __func__ % res.chi2_pre % res.ndof % pval(res.chi2_pre, res.ndof);
        log<LOG_INFO>(L"%1% || Eval region, post-constraint: chi2 = %2% / %3% (p = %4%)%5%")
            % __func__ % res.chi2_post % res.ndof % pval(res.chi2_post, res.ndof)
            % (res.shape_only ? std::string(" [shape-only: one dof removed per channel, ") + std::to_string(n) + " bins]" : std::string("")).c_str();
        for(const auto &ch : res.channels) {
            log<LOG_INFO>(L"%1% ||   %2% : pre %3% / %4% (p = %5%)   post %6% / %4% (p = %7%)")
                % __func__ % ch.name.c_str() % ch.chi2_pre % ch.ndof % pval(ch.chi2_pre, ch.ndof)
                % ch.chi2_post % pval(ch.chi2_post, ch.ndof);
        }
        log<LOG_INFO>(L"%1% || ===============================================================") % __func__;
    }

    std::map<std::string, TObject *> PlotFitRegionConstraint(const PROconfig &config, const FitRegionConstraint &res,
                                                             const std::string &filename) {
        std::map<std::string, TObject *> objs;
        TCanvas c("c_fit_region", "c_fit_region", 1200, 900);
        auto chi2Label = [&](const std::string &what, float chi2, long ndof) {
            if(res.blind) return what;
            char buf[64];
            snprintf(buf, sizeof buf, " (#chi^{2}/ndf = %.1f/%ld)", chi2, ndof);
            return what + buf;
        };
        const std::string pdf = filename + ".pdf";
        c.Print((pdf + "[").c_str());

        for(const auto &ch : res.channels) {
            const auto &bins = config.GetChannelVariableBins(ch.channel, config.i_prime);
            const Eigen::Index start = (Eigen::Index)config.GetCollapsedGlobalVariableBinStart(ch.channel, config.i_prime);
            const int nb = (int)bins.NBins();
            // 1D channels get their real edges; multi-dimensional ones are drawn by bin index.
            std::vector<double> edges;
            if(bins.NDim() == 1) {
                for(float e : bins.bin_edges[0]) edges.push_back(e);
            } else {
                for(int b = 0; b <= nb; ++b) edges.push_back(b);
            }
            const std::string xtitle = bins.NDim() == 1 ? config.GetChannelXAxisTitle(ch.channel % config.m_num_channels, config.i_prime) : "Collapsed bin";

            auto mk = [&](const std::string &tag) {
                TH1D *h = new TH1D(("fit_region_" + ch.name + "_" + tag).c_str(), (";" + xtitle + ";Events").c_str(), nb, edges.data());
                h->SetDirectory(nullptr);
                return h;
            };
            TH1D *h_data = mk("data"), *h_pre = mk("pre"), *h_post = mk("post");
            for(Eigen::Index r : ch.rows) {
                const int b = (int)(res.eval_bins[(size_t)r] - start) + 1;
                h_pre->SetBinContent(b, res.pre_mean(r));
                h_pre->SetBinError(b, std::sqrt(std::max(0.0f, res.pre_syst_cov(r, r))));
                h_post->SetBinContent(b, res.post_mean(r));
                h_post->SetBinError(b, std::sqrt(std::max(0.0f, res.post_syst_cov(r, r))));
                h_data->SetBinContent(b, res.data(r));
                h_data->SetBinError(b, std::sqrt(std::max(0.0f, res.data(r))));
            }

            c.cd();
            c.Clear();
            TPad top("top", "top", 0, 0.3, 1, 1);
            TPad bot("bot", "bot", 0, 0, 1, 0.3);
            top.SetBottomMargin(0.02);
            top.SetLeftMargin(0.11);
            top.SetRightMargin(0.04);
            bot.SetTopMargin(0.02);
            bot.SetBottomMargin(0.3);
            bot.SetLeftMargin(0.11);
            bot.SetRightMargin(0.04);
            top.Draw();
            bot.Draw();

            top.cd();
            TH1D pre_band(*h_pre), post_band(*h_post);
            pre_band.SetFillColorAlpha(kGray + 1, 0.5);
            pre_band.SetFillStyle(3354);
            pre_band.SetMarkerSize(0);
            post_band.SetFillColorAlpha(kAzure + 1, 0.5);
            post_band.SetMarkerSize(0);
            TH1D pre_line(*h_pre), post_line(*h_post);
            pre_line.SetLineColor(kGray + 2);
            pre_line.SetLineStyle(2);
            pre_line.SetLineWidth(2);
            post_line.SetLineColor(kAzure + 2);
            post_line.SetLineWidth(2);
            double ymax = 0;
            for(int b = 1; b <= nb; ++b)
                ymax = std::max({ymax, h_pre->GetBinContent(b) + h_pre->GetBinError(b),
                                 h_post->GetBinContent(b) + h_post->GetBinError(b),
                                 res.blind ? 0.0 : h_data->GetBinContent(b) + h_data->GetBinError(b)});
            pre_band.SetMaximum(1.3 * ymax);
            pre_band.SetMinimum(0);
            pre_band.SetTitle(ch.name.c_str());
            pre_band.GetXaxis()->SetLabelSize(0);
            pre_band.Draw("E2");
            post_band.Draw("E2 same");
            pre_line.Draw("hist same");
            post_line.Draw("hist same");
            h_data->SetMarkerStyle(20);
            h_data->SetLineColor(kBlack);
            if(!res.blind) h_data->Draw("E1 P same");

            TLegend leg(0.45, 0.68, 0.95, 0.9);
            leg.SetBorderSize(0);
            leg.SetFillStyle(0);
            if(!res.blind) leg.AddEntry(h_data, "Data (stat. error)", "lep");
            leg.AddEntry(&pre_band, chi2Label("Pre-constraint", ch.chi2_pre, ch.ndof).c_str(), "fl");
            leg.AddEntry(&post_band, chi2Label("Fit-region constrained", ch.chi2_post, ch.ndof).c_str(), "fl");
            leg.Draw();

            bot.cd();
            // Ratios to the constrained prediction: its band, the pre-constraint curve, and the data.
            TH1D r_post(*h_post), r_pre(*h_pre), r_data(*h_data);
            for(int b = 1; b <= nb; ++b) {
                const double p = h_post->GetBinContent(b);
                const double inv = p > 0 ? 1.0 / p : 0.0;
                r_post.SetBinContent(b, p > 0 ? 1.0 : 0.0);
                r_post.SetBinError(b, h_post->GetBinError(b) * inv);
                r_pre.SetBinContent(b, h_pre->GetBinContent(b) * inv);
                r_pre.SetBinError(b, 0);
                r_data.SetBinContent(b, h_data->GetBinContent(b) * inv);
                r_data.SetBinError(b, h_data->GetBinError(b) * inv);
            }
            r_post.SetFillColorAlpha(kAzure + 1, 0.5);
            r_post.SetMarkerSize(0);
            r_post.SetTitle("");
            // Fit the band and the data points (a signal in the eval region can sit well off 1).
            double rlo = 0.8, rhi = 1.2;
            for(int b = 1; b <= nb; ++b) {
                if(h_post->GetBinContent(b) <= 0) continue;
                rlo = std::min(rlo, 1.0 - r_post.GetBinError(b));
                rhi = std::max(rhi, 1.0 + r_post.GetBinError(b));
                if(!res.blind && h_data->GetBinContent(b) > 0) {
                    rlo = std::min(rlo, r_data.GetBinContent(b) - r_data.GetBinError(b));
                    rhi = std::max(rhi, r_data.GetBinContent(b) + r_data.GetBinError(b));
                }
            }
            const double pad = 0.1 * (rhi - rlo);
            r_post.SetMinimum(std::max(0.0, rlo - pad));
            r_post.SetMaximum(std::min(3.0, rhi + pad));
            r_post.GetYaxis()->SetTitle("Ratio to constrained");
            r_post.GetYaxis()->SetNdivisions(505);
            r_post.GetYaxis()->SetTitleSize(0.09);
            r_post.GetYaxis()->SetTitleOffset(0.5);
            r_post.GetYaxis()->SetLabelSize(0.09);
            r_post.GetXaxis()->SetTitleSize(0.11);
            r_post.GetXaxis()->SetLabelSize(0.1);
            r_post.Draw("E2");
            TLine one(edges.front(), 1, edges.back(), 1);
            one.SetLineStyle(2);
            one.Draw();
            r_pre.SetLineColor(kGray + 2);
            r_pre.SetLineStyle(2);
            r_pre.SetLineWidth(2);
            r_pre.Draw("hist same");
            r_data.SetMarkerStyle(20);
            r_data.SetLineColor(kBlack);
            if(!res.blind) r_data.Draw("E1 P same");

            c.Print(pdf.c_str());

            objs["fit_region_" + ch.name + "_data"] = h_data;
            objs["fit_region_" + ch.name + "_pre_constraint"] = h_pre;
            objs["fit_region_" + ch.name + "_post_constraint"] = h_post;
        }

        // Pre/post systematic covariance over all evaluated bins.
        const int n = (int)res.eval_bins.size();
        TH2D *cpre = new TH2D("fit_region_pre_syst_cov", "Pre-constraint systematic covariance;Eval bin;Eval bin", n, 0, n, n, 0, n);
        TH2D *cpost = new TH2D("fit_region_post_syst_cov", "Constrained systematic covariance;Eval bin;Eval bin", n, 0, n, n, 0, n);
        cpre->SetDirectory(nullptr);
        cpost->SetDirectory(nullptr);
        for(int i = 0; i < n; ++i)
            for(int j = 0; j < n; ++j) {
                cpre->SetBinContent(i + 1, j + 1, res.pre_syst_cov(i, j));
                cpost->SetBinContent(i + 1, j + 1, res.post_syst_cov(i, j));
            }
        objs["fit_region_pre_syst_cov"] = cpre;
        objs["fit_region_post_syst_cov"] = cpost;
        c.cd();
        c.Clear();
        // Side by side on a shared z range so the shrinkage is visible.
        c.Divide(2, 1);
        const double zmin = std::min(cpre->GetMinimum(), cpost->GetMinimum());
        const double zmax = std::max(cpre->GetMaximum(), cpost->GetMaximum());
        for(int p = 1; p <= 2; ++p) {
            c.cd(p);
            gPad->SetLeftMargin(0.12);
            gPad->SetRightMargin(0.16);
            TH2D *h = p == 1 ? cpre : cpost;
            h->SetMinimum(zmin);
            h->SetMaximum(zmax);
            h->Draw("colz");
        }
        c.Print(pdf.c_str());

        c.Print((pdf + "]").c_str());
        log<LOG_INFO>(L"%1% || Wrote fit-region constraint plots to %2%") % __func__ % pdf.c_str();
        return objs;
    }

}
