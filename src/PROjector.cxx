#include "PROjector.h"
#include "PROlog.h"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace PROfit {

    void PROjectorConstraint::save(const std::string &filename) const {
        std::ofstream ofs(filename, std::ios::binary);
        boost::archive::binary_oarchive oa(ofs);
        oa << *this;
        log<LOG_INFO>(L"%1% || Saved PROjector constraint (%2% nuisance parameters) to %3%")
            % __func__ % nuisance_names.size() % filename.c_str();
    }

    bool PROjectorConstraint::load(const std::string &filename) {
        try {
            std::ifstream ifs(filename, std::ios::binary);
            if(!ifs.good()) {
                log<LOG_ERROR>(L"%1% || Could not open PROjector constraint file %2%") % __func__ % filename.c_str();
                return false;
            }
            boost::archive::binary_iarchive ia(ifs);
            ia >> *this;
        } catch(const std::exception &e) {
            log<LOG_ERROR>(L"%1% || Failed to read PROjector constraint file %2%: %3%")
                % __func__ % filename.c_str() % e.what();
            return false;
        }
        log<LOG_INFO>(L"%1% || Loaded PROjector constraint (%2% nuisance parameters, pattern '%3%') from %4%")
            % __func__ % nuisance_names.size() % prefit_pattern.c_str() % filename.c_str();
        return true;
    }

    bool PROjectorSelectChannels(const PROconfig &config, const std::string &pattern,
                                 std::vector<size_t> &matched_channels) {
        matched_channels.clear();
        if(pattern.empty()) {
            log<LOG_ERROR>(L"%1% || Empty PROjector subchannel pattern.") % __func__;
            return false;
        }

        // Walk the canonical mode -> detector -> channel -> subchannel nesting so the
        // running counter matches the ordering of config.m_fullnames, and demand that a
        // channel's subchannels are either all matched or all unmatched: the chi2 lives
        // in collapsed (channel) space, so a partially-selected channel is ill-defined.
        size_t global_subchannel_index = 0;
        size_t global_channel_index = 0;
        size_t n_unmatched_channels = 0;
        bool ok = true;
        // Unanchored regex (plain substrings behave as before); see PROconfig.h.
        std::regex re = CompilePattern(pattern, "PROjector channel pattern");
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
                    } else if(n_match == 0) {
                        ++n_unmatched_channels;
                    } else {
                        log<LOG_ERROR>(L"%1% || PROjector pattern '%2%' matches %3%/%4% subchannels of channel %5% (mode %6%, detector %7%). "
                                L"A channel must be selected as a whole; refine the pattern.")
                            % __func__ % pattern.c_str() % n_match % nsub
                            % config.m_channel_names[ic].c_str()
                            % config.m_mode_names[im].c_str() % config.m_detector_names[id].c_str();
                        ok = false;
                    }
                }
            }
        }
        if(!ok) return false;
        if(matched_channels.empty()) {
            log<LOG_ERROR>(L"%1% || PROjector pattern '%2%' matched no channels.") % __func__ % pattern.c_str();
            return false;
        }
        if(n_unmatched_channels == 0) {
            log<LOG_ERROR>(L"%1% || PROjector pattern '%2%' matched EVERY channel; nothing left to project onto.")
                % __func__ % pattern.c_str();
            return false;
        }
        log<LOG_INFO>(L"%1% || PROjector pattern '%2%' selected %3% pre-fit channel(s), leaving %4% projected channel(s).")
            % __func__ % pattern.c_str() % matched_channels.size() % n_unmatched_channels;
        return true;
    }

    Eigen::VectorXf PROjectorCollapsedMask(const PROconfig &config,
                                           const std::vector<size_t> &matched_channels,
                                           size_t var_index, bool complement) {
        Eigen::VectorXf mask = Eigen::VectorXf::Constant(
                config.m_num_variable_bins_total_collapsed[var_index], complement ? 1.0f : 0.0f);
        const float fill = complement ? 0.0f : 1.0f;
        for(size_t gc : matched_channels) {
            const size_t start = config.GetCollapsedGlobalVariableBinStart(gc, var_index);
            const size_t nbins = config.GetChannelVariableBins(gc, var_index).NBins();
            for(size_t b = 0; b < nbins; ++b)
                mask(static_cast<Eigen::Index>(start + b)) = fill;
        }
        return mask;
    }

    size_t PROjectorPromoteCovariance(PROconfig &config, PROsyst &systs, size_t var_index,
                                      int num_decomp_knobs,
                                      const std::vector<std::string> &keep_covariance) {
        // Promoted set: every named covariance systematic not explicitly kept. Note the
        // "mcstat" covariance is not in covar_names (it is registered only in the internal
        // syst_map) and so is never promoted — correct, since it is diagonal and carries no
        // cross-detector correlation for a pre-fit to constrain.
        std::vector<std::string> promote_names;
        for(const std::string &name : systs.covar_names) {
            if(name == config.m_mcstat_systname) continue;
            if(std::find(keep_covariance.begin(), keep_covariance.end(), name) != keep_covariance.end()) {
                log<LOG_INFO>(L"%1% || PROjector keeping systematic %2% as unconstrained covariance.") % __func__ % name.c_str();
                continue;
            }
            promote_names.push_back(name);
        }
        if(promote_names.empty()) {
            log<LOG_INFO>(L"%1% || PROjector: no covariance systematics to promote; splines only.") % __func__;
            return 0;
        }
        for(const std::string &name : promote_names)
            log<LOG_INFO>(L"%1% || PROjector promoting covariance systematic %2% to eigenmode splines.") % __func__ % name.c_str();

        const Eigen::MatrixXf promoted_frac = systs.SumMatrices(promote_names);
        const Eigen::MatrixXf old_total = systs.fractional_covariance;

        static const std::string kPromotedName = "PROjector_cov";
        SystStruct ss(kPromotedName, 0);
        ss.num_decomp_knobs = num_decomp_knobs;
        ss.include_resid_cov = true;
        ss.binning = static_cast<int>(var_index);

        const size_t n_before = systs.GetNSplines();
        systs.FillSplinesFromCovarianceMatrix(promoted_frac, ss);
        const size_t n_added = systs.GetNSplines() - n_before;
        if(n_added == 0) {
            log<LOG_WARNING>(L"%1% || PROjector promotion produced no eigenmode splines (no positive modes?); covariance left as-is.")
                % __func__;
            return 0;
        }

        // FillSplinesFromCovarianceMatrix registers the splines (and possibly a residual
        // covariance) but does not touch the pre-summed total. Replace the promoted part
        // by the residual explicitly; fractional_covariance is the single matrix PROchi
        // and the throw machinery consume, so the stale per-systematic covmat entries the
        // promotion leaves behind are inert for fitting (they still feed plot commands).
        Eigen::MatrixXf new_total = old_total - promoted_frac;
        auto dbg_it = systs.cov2spline_debug_info.find(kPromotedName);
        size_t n_residual = 0;
        if(dbg_it != systs.cov2spline_debug_info.end() && dbg_it->second.has_residual) {
            new_total += dbg_it->second.residual_cov;
            n_residual = dbg_it->second.n_residual_modes;
        }
        systs.fractional_covariance = new_total;

        // The PROsyst constructor sizes spline_priors/spline_centers after all splines are
        // built; a post-hoc promotion has to extend them by hand. Unit width, zero center:
        // the sqrt(lambda) response baked into each knob makes that reproduce the promoted
        // covariance exactly.
        Eigen::VectorXf priors = Eigen::VectorXf::Constant(systs.GetNSplines(), 1.0f);
        Eigen::VectorXf centers = Eigen::VectorXf::Constant(systs.GetNSplines(), 0.0f);
        priors.head(n_before) = systs.spline_priors;
        centers.head(n_before) = systs.spline_centers;
        systs.spline_priors = priors;
        systs.spline_centers = centers;

        // Downstream code (bounds section, plotting, logging) looks every spline name up in
        // the plotname map with .at(); register the synthesized knobs.
        for(size_t i = n_before; i < systs.GetNSplines(); ++i) {
            const std::string &knob = systs.spline_names[i];
            if(!config.m_mcgen_variation_plotname_map.count(knob))
                config.m_mcgen_variation_plotname_map[knob] = knob;
        }

        log<LOG_INFO>(L"%1% || PROjector promoted %2% covariance systematic(s) into %3% eigenmode spline(s) (%4% residual mode(s) kept as covariance).")
            % __func__ % promote_names.size() % n_added % n_residual;
        return n_added;
    }

    namespace {

        // Install the fit region as a PROconfig active-bin mask for every variable. This is
        // the authoritative exclusion: every metric constructed afterwards (including the
        // fresh ones FC/AFC workers build around thrown pseudo-data) snapshots it, so
        // masked bins can never re-enter a chi2 through regenerated data.
        void installFitRegion(PROconfig &config, const std::vector<size_t> &matched_channels,
                              bool complement) {
            for(size_t io = 0; io < config.m_num_variables; ++io) {
                Eigen::VectorXf maskv = PROjectorCollapsedMask(config, matched_channels, io, complement);
                std::vector<char> mask(maskv.size());
                for(Eigen::Index i = 0; i < maskv.size(); ++i)
                    mask[(size_t)i] = maskv(i) > 0.5f ? 1 : 0;
                config.SetActiveBins(io, mask);
            }
        }

        // Apply a collapsed-space mask to every PROdata and refresh the primary data object.
        // With the config active-bin mask installed this is statistically redundant for the
        // fit, but it keeps plots/error bands consistent and the excluded data blind.
        bool maskAllData(const PROconfig &config, const std::vector<size_t> &matched_channels,
                         bool complement, std::vector<PROdata> &variable_data, PROdata &data) {
            const size_t nvar = std::min(variable_data.size(), config.m_num_variables);
            for(size_t io = 0; io < nvar; ++io) {
                if(variable_data[io].GetNbins() != config.m_num_variable_bins_total_collapsed[io]) {
                    log<LOG_WARNING>(L"%1% || PROjector: variable %2% data has %3% bins, expected %4%; leaving it unmasked.")
                        % __func__ % io % variable_data[io].GetNbins()
                        % config.m_num_variable_bins_total_collapsed[io];
                    continue;
                }
                Eigen::VectorXf mask = PROjectorCollapsedMask(config, matched_channels, io, complement);
                PROdata masked(mask.cwiseProduct(variable_data[io].Spec()),
                               mask.cwiseProduct(variable_data[io].Error()));
                masked.hash = variable_data[io].hash;
                variable_data[io] = std::move(masked);
            }
            data = variable_data[config.i_prime];
            if(data.Spec().sum() <= 0) {
                log<LOG_ERROR>(L"%1% || PROjector: the %2% data bins are all empty after masking. Nothing to fit.")
                    % __func__ % (complement ? "projected" : "pre-fit");
                return false;
            }
            return true;
        }

    }

    bool PROjectorSetup(const PROjectorRunConfig &pjconf, PROconfig &config, PROsyst &systs,
                        std::vector<PROdata> &variable_data, PROdata &data,
                        Eigen::VectorXf &fakedataparams,
                        std::vector<std::string> &fixed_params,
                        const PROmodel &model, const std::string &chi2_name) {
        if(!pjconf.active()) return true;
        if(pjconf.prefit_mode() && pjconf.projector_mode()) {
            log<LOG_ERROR>(L"%1% || --projector-prefit and --projector are mutually exclusive.") % __func__;
            return false;
        }
        // All three metrics honor the config fit-region (active-bin) mask, so any of them
        // is legal here. Poisson gets a reminder that it ignores covariance-type
        // systematics: anything left unpromoted (residual modes, --projector-keep-cov,
        // mcstat) silently drops out of a Poisson fit.
        if(chi2_name == "Poisson") {
            log<LOG_WARNING>(L"%1% || PROjector with the Poisson metric: covariance-type systematics are ignored by "
                    L"PROpoisson, so any unpromoted covariance (residual eigenmodes, --projector-keep-cov, mcstat) "
                    L"will not enter the fit. Prefer --projector-knobs -1 and no kept covariances.") % __func__;
        }

        log<LOG_WARNING>(L"%1% || ############### PROjector %2% mode ###############")
            % __func__ % (pjconf.prefit_mode() ? "PRE-FIT" : "PROJECTED");

        if(pjconf.prefit_mode()) {
            std::vector<size_t> matched_channels;
            if(!PROjectorSelectChannels(config, pjconf.prefit_pattern, matched_channels)) return false;

            PROjectorPromoteCovariance(config, systs, config.i_prime,
                                       pjconf.num_decomp_knobs, pjconf.keep_covariance);

            installFitRegion(config, matched_channels, /*complement=*/false);
            if(!maskAllData(config, matched_channels, /*complement=*/false, variable_data, data)) return false;

            if(!pjconf.float_physics) {
                for(size_t i = 0; i < model.nparams; ++i) {
                    const std::string &name = model.param_names[i];
                    const std::string &pname = model.pretty_param_names[i];
                    if(std::find(fixed_params.begin(), fixed_params.end(), name) != fixed_params.end()) continue;
                    if(std::find(fixed_params.begin(), fixed_params.end(), pname) != fixed_params.end()) continue;
                    fixed_params.push_back(name);
                    log<LOG_INFO>(L"%1% || PROjector pre-fit: fixing physics parameter %2% at CV (pass --projector-float-physics to float it).")
                        % __func__ % name.c_str();
                }
            } else {
                log<LOG_WARNING>(L"%1% || PROjector pre-fit: physics parameters are FLOATING; the saved nuisance posterior will be "
                        L"the marginal over physics. Make sure that is what you want for the projected fit.") % __func__;
            }
            log<LOG_INFO>(L"%1% || PROjector pre-fit ready. Run the 'global' subcommand to fit and write the constraint file.")
                % __func__;
        } else {
            PROjectorConstraint c;
            if(!c.load(pjconf.constraint_file)) return false;

            if(c.config_hash != config.hash) {
                if(pjconf.force) {
                    log<LOG_WARNING>(L"%1% || PROjector constraint config hash (%2%) does not match current config (%3%), but --force is set.")
                        % __func__ % c.config_hash % config.hash;
                } else {
                    log<LOG_ERROR>(L"%1% || PROjector constraint was made with config hash %2% but current config hash is %3%. "
                            L"Use the same XML/binaries as the pre-fit, or --force if you are sure.")
                        % __func__ % c.config_hash % config.hash;
                    return false;
                }
            }
            if(c.metric_name != chi2_name) {
                log<LOG_ERROR>(L"%1% || PROjector constraint was produced with metric %2% but this run uses %3%.")
                    % __func__ % c.metric_name.c_str() % chi2_name.c_str();
                return false;
            }

            std::vector<size_t> matched_channels;
            if(!PROjectorSelectChannels(config, c.prefit_pattern, matched_channels)) return false;

            // Re-derive the identical eigenmode promotion: same summed covariance, same K,
            // same kept list. The eigendecomposition is deterministic, so the knob names
            // and responses reproduce the pre-fit's; the name check below is the proof.
            PROjectorPromoteCovariance(config, systs, config.i_prime,
                                       c.num_decomp_knobs, c.keep_covariance);

            if(systs.spline_names.size() != c.nuisance_names.size()) {
                log<LOG_ERROR>(L"%1% || PROjector constraint has %2% nuisance parameters but this run built %3%. "
                        L"The systematic selection (--syst-list/--exclude-systs) must match the pre-fit exactly.")
                    % __func__ % c.nuisance_names.size() % systs.spline_names.size();
                return false;
            }
            for(size_t i = 0; i < c.nuisance_names.size(); ++i) {
                if(systs.spline_names[i] != c.nuisance_names[i]) {
                    log<LOG_ERROR>(L"%1% || PROjector nuisance parameter %2% is '%3%' here but '%4%' in the constraint file. "
                            L"The systematic selection must match the pre-fit exactly.")
                        % __func__ % i % systs.spline_names[i].c_str() % c.nuisance_names[i].c_str();
                    return false;
                }
            }

            // Install the pre-fit posterior: centers become the prior centers, the full
            // covariance becomes a correlated external prior consumed by PROchi::Pull.
            // spline_priors gets the marginal widths so any code that only looks at the
            // diagonal (throws, pre-fit error bands, logging) stays consistent.
            systs.spline_centers = c.centers;
            Eigen::VectorXf widths = c.covariance.diagonal().array().max(1e-12f).sqrt();
            systs.spline_priors = widths;
            systs.has_external_prior_cov = true;
            systs.external_prior_cov = c.covariance;

            // Fits (Pull) use the FULL correlated posterior; pseudo-experiment throws
            // (FC, Brazil bands, adaptive FC) go through ThrowRestrictedSplinePull, which
            // samples each nuisance's marginal N(center, sigma) only. Correlated throwing
            // is a known limitation — flag it so FC coverage studies are read accordingly.
            log<LOG_WARNING>(L"%1% || PROjector: nuisance THROWS (fc, brazil bands, pseudo-experiments) sample the "
                    L"constrained posterior's marginal widths only; posterior correlations enter the chi2 pull "
                    L"but are not sampled in throws.") % __func__;

            for(size_t i = 0; i < c.nuisance_names.size(); ++i) {
                log<LOG_INFO>(L"%1% || PROjector prior %2% : center %3% width %4%")
                    % __func__ % c.nuisance_names[i].c_str() % c.centers(i) % widths(i);
            }

            installFitRegion(config, matched_channels, /*complement=*/true);
            if(!maskAllData(config, matched_channels, /*complement=*/true, variable_data, data)) return false;

            log<LOG_INFO>(L"%1% || PROjector projected fit ready: pre-fit channels masked out, %2% constrained nuisance parameters.")
                % __func__ % c.nuisance_names.size();
        }

        // The parameter count may have grown (promotion); zero-extend the injected-truth
        // vector every downstream consumer (PROfile red stars, seeds, ...) indexes by it.
        const size_t new_nparams = model.nparams + systs.GetNSplines();
        if(static_cast<size_t>(fakedataparams.size()) < new_nparams) {
            Eigen::VectorXf extended = Eigen::VectorXf::Zero(new_nparams);
            extended.head(fakedataparams.size()) = fakedataparams;
            fakedataparams = extended;
        }
        return true;
    }

    namespace {

        // chi2-only evaluation helper for the finite-difference Hessian.
        inline float evalChi2(PROmetric &metric, const Eigen::VectorXf &x, Eigen::VectorXf &grad_dummy) {
            return metric(x, grad_dummy, false);
        }

    }

    bool PROjectorSaveConstraint(const std::string &filename, const PROjectorRunConfig &pjconf,
                                 const PROconfig &config, PROmetric &metric,
                                 const Eigen::VectorXf &best_fit, float chi2,
                                 const std::vector<int> &global_fixed,
                                 const std::string &chi2_name) {
        const size_t nphys = metric.GetModel().nparams;
        const size_t nsplines = metric.GetSysts().GetNSplines();
        const size_t nparams = nphys + nsplines;
        if(static_cast<size_t>(best_fit.size()) != nparams || global_fixed.size() != nparams) {
            log<LOG_ERROR>(L"%1% || PROjector: best fit size (%2%) or fixed mask size (%3%) does not match nparams (%4%).")
                % __func__ % best_fit.size() % global_fixed.size() % nparams;
            return false;
        }

        // Deliberately NOT metric.lb/ub: PROfitter::calcFreqSeedPoints leaves the metric's
        // bounds pinned (lb==ub) at scan candidates when its minima loop does not run.
        // LowerBound()/UpperBound() rebuild the true box from the model + spline bounds.
        const Eigen::VectorXf lb = metric.LowerBound();
        const Eigen::VectorXf ub = metric.UpperBound();

        std::vector<size_t> free_idx;
        for(size_t i = 0; i < nparams; ++i)
            if(!global_fixed[i]) free_idx.push_back(i);
        if(free_idx.empty()) {
            log<LOG_ERROR>(L"%1% || PROjector: every parameter is fixed; no posterior to save.") % __func__;
            return false;
        }
        const size_t nfree = free_idx.size();

        // Per-parameter finite-difference step: nuisance parameters live in sigma units so
        // a fixed 0.05 sigma works; physics steps scale with the bound range. Shrink any
        // step that would leave the box (the best fit can sit near a bound).
        Eigen::VectorXf h(nfree);
        for(size_t k = 0; k < nfree; ++k) {
            const size_t i = free_idx[k];
            float step = i < nphys ? std::max(1e-3f * (ub(i) - lb(i)), 1e-4f) : 0.05f;
            step = std::min({step, ub(i) - best_fit(i), best_fit(i) - lb(i)});
            if(step < 1e-6f) {
                log<LOG_WARNING>(L"%1% || PROjector Hessian: parameter %2% is pinned against its bounds; using minimal step.")
                    % __func__ % i;
                step = 1e-6f;
            }
            h(k) = step;
        }

        log<LOG_INFO>(L"%1% || PROjector: computing finite-difference Hessian over %2% free parameter(s) (~%3% chi2 evaluations).")
            % __func__ % nfree % (2 * nfree * nfree);

        Eigen::VectorXf grad_dummy = Eigen::VectorXf::Zero(nparams);
        const float f0 = evalChi2(metric, best_fit, grad_dummy);
        if(std::abs(f0 - chi2) > std::max(1e-2f * std::abs(chi2), 0.5f)) {
            log<LOG_WARNING>(L"%1% || PROjector: chi2 at best fit re-evaluates to %2% vs fit result %3%; Hessian is taken at the re-evaluated point.")
                % __func__ % f0 % chi2;
        }

        // Central second differences. chi2 = chi2_min + d^T (Sigma^-1) d for a Gaussian
        // posterior, so H = Hessian(chi2) = 2 Sigma^-1 and Sigma = 2 H^-1.
        Eigen::MatrixXf H = Eigen::MatrixXf::Zero(nfree, nfree);
        std::vector<float> fplus(nfree), fminus(nfree);
        for(size_t k = 0; k < nfree; ++k) {
            Eigen::VectorXf xp = best_fit, xm = best_fit;
            xp(free_idx[k]) += h(k);
            xm(free_idx[k]) -= h(k);
            fplus[k] = evalChi2(metric, xp, grad_dummy);
            fminus[k] = evalChi2(metric, xm, grad_dummy);
            H(k, k) = (fplus[k] - 2.0f * f0 + fminus[k]) / (h(k) * h(k));
        }
        for(size_t k = 0; k < nfree; ++k) {
            for(size_t l = k + 1; l < nfree; ++l) {
                Eigen::VectorXf xpp = best_fit, xpm = best_fit, xmp = best_fit, xmm = best_fit;
                xpp(free_idx[k]) += h(k); xpp(free_idx[l]) += h(l);
                xpm(free_idx[k]) += h(k); xpm(free_idx[l]) -= h(l);
                xmp(free_idx[k]) -= h(k); xmp(free_idx[l]) += h(l);
                xmm(free_idx[k]) -= h(k); xmm(free_idx[l]) -= h(l);
                const float mixed = (evalChi2(metric, xpp, grad_dummy) - evalChi2(metric, xpm, grad_dummy)
                                   - evalChi2(metric, xmp, grad_dummy) + evalChi2(metric, xmm, grad_dummy))
                                   / (4.0f * h(k) * h(l));
                H(k, l) = mixed;
                H(l, k) = mixed;
            }
            if(nfree > 20 && (k % 10 == 0))
                log<LOG_INFO>(L"%1% || PROjector Hessian progress: row %2% / %3%") % __func__ % k % nfree;
        }

        // PSD-safe inversion: clamp eigenvalues from below before inverting so a slightly
        // indefinite FD Hessian (noise, non-quadratic tails) still yields a usable
        // covariance instead of negative variances.
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(0.5f * (H + H.transpose()));
        if(es.info() != Eigen::Success) {
            log<LOG_ERROR>(L"%1% || PROjector: eigendecomposition of the Hessian failed.") % __func__;
            return false;
        }
        const float max_eig = es.eigenvalues().maxCoeff();
        if(max_eig <= 0) {
            log<LOG_ERROR>(L"%1% || PROjector: Hessian has no positive curvature (max eigenvalue %2%); is this really a minimum?")
                % __func__ % max_eig;
            return false;
        }
        const float floor_eig = 1e-6f * max_eig;
        int n_clamped = 0;
        Eigen::VectorXf inv_eigs(nfree);
        for(size_t k = 0; k < nfree; ++k) {
            float ev = es.eigenvalues()(k);
            if(ev < floor_eig) { ev = floor_eig; ++n_clamped; }
            inv_eigs(k) = 2.0f / ev;  // Sigma = 2 H^-1
        }
        if(n_clamped)
            log<LOG_WARNING>(L"%1% || PROjector: clamped %2% non-positive/tiny Hessian eigenvalue(s); those directions carry the floor width.")
                % __func__ % n_clamped;
        const Eigen::MatrixXf cov_free =
            es.eigenvectors() * inv_eigs.asDiagonal() * es.eigenvectors().transpose();

        // Assemble the full nuisance-block covariance. Free nuisances take the (physics-
        // marginalized, since cov_free is the full inverse) FD result; nuisances that were
        // fixed during the pre-fit saw no data and keep their original prior width,
        // uncorrelated.
        const PROsyst &systs = metric.GetSysts();
        Eigen::MatrixXf Sigma = Eigen::MatrixXf::Zero(nsplines, nsplines);
        for(size_t i = 0; i < nsplines; ++i)
            Sigma(i, i) = systs.spline_priors(i) * systs.spline_priors(i);
        std::vector<int> free_pos(nparams, -1);
        for(size_t k = 0; k < nfree; ++k) free_pos[free_idx[k]] = static_cast<int>(k);
        for(size_t i = 0; i < nsplines; ++i) {
            const int ki = free_pos[nphys + i];
            if(ki < 0) {
                log<LOG_INFO>(L"%1% || PROjector: nuisance %2% was fixed in the pre-fit; keeping its prior width %3%.")
                    % __func__ % systs.spline_names[i].c_str() % systs.spline_priors(i);
                continue;
            }
            for(size_t j = 0; j < nsplines; ++j) {
                const int kj = free_pos[nphys + j];
                if(kj < 0) continue;
                Sigma(i, j) = cov_free(ki, kj);
            }
        }

        PROjectorConstraint c;
        c.config_hash = config.hash;
        c.metric_name = chi2_name;
        c.prefit_pattern = pjconf.prefit_pattern;
        c.num_decomp_knobs = pjconf.num_decomp_knobs;
        c.keep_covariance = pjconf.keep_covariance;
        c.nuisance_names = systs.spline_names;
        c.centers = best_fit.segment(nphys, nsplines);
        c.covariance = Sigma;
        c.physics_names = metric.GetModel().param_names;
        c.physics_best_fit = best_fit.head(nphys);
        c.physics_were_fixed = !pjconf.float_physics;
        c.prefit_chi2 = chi2;

        log<LOG_INFO>(L"%1% || ############ PROjector pre-fit posterior summary ############") % __func__;
        for(size_t i = 0; i < nsplines; ++i) {
            log<LOG_INFO>(L"%1% || %2% : center %3% +/- %4% (prior width was %5%)")
                % __func__ % systs.spline_names[i].c_str()
                % c.centers(i) % std::sqrt(Sigma(i, i)) % systs.spline_priors(i);
        }

        c.save(filename);
        return true;
    }

}
