#include "PROcovariance.h"
#include "PROcess.h"
#include "PROdata.h"
#include "PROlog.h"
#include "PROtocall.h"
#include <Eigen/Eigen>
#include <cmath>
using namespace PROfit;

PROcovariance::PROcovariance(const std::string tag, const PROconfig &conin, const PROpeller &pin, const PROsyst *systin, const PROmodel &modelin, const PROdata &datain, EvalStrategy strat, bool shape_only, std::vector<float> physics_param_fixed, bool own_config_and_peller)
    : PROmetric(tag, conin, systin, modelin, datain, strat, shape_only, physics_param_fixed),
      owned_config(own_config_and_peller ? std::optional<PROconfig>(conin) : std::nullopt),
      owned_peller(own_config_and_peller ? std::optional<PROpeller>(pin) : std::nullopt),
      config(owned_config ? *owned_config : conin),
      peller(owned_peller ? *owned_peller : pin) {
    // An externally supplied posterior (PROjector projected mode) takes precedence over
    // any XML-configured prior correlations: it already IS the full prior covariance.
    if (systin->has_external_prior_cov) {
        if (conin.m_mcgen_correlations.size())
            log<LOG_WARNING>(L"%1% || Both an external prior covariance and XML prior correlations are set; using the external one.") % __func__;
        correlated_systematics = true;
        prior_covariance = systin->external_prior_cov;
        for(size_t i = 0; i < systin->spline_prior_types.size(); ++i) {
            if (systin->spline_prior_types[i] == SplinePriorType::Uniform) {
                prior_covariance.row(i).setZero();
                prior_covariance.col(i).setZero();
                prior_covariance(i, i) = 1.0f;
            }
        }
        prior_covariance_inv = prior_covariance.inverse();
    }
    // Build the correlation matrix between priors if configured to
    else if (conin.m_mcgen_correlations.size()) {
        correlated_systematics = true;
        prior_covariance = Eigen::MatrixXf::Identity(syst->GetNSplines(), syst->GetNSplines());
        for (auto const &t: conin.m_mcgen_correlations) {
          auto itA = std::find(systin->spline_names.begin(), systin->spline_names.end(), std::get<0>(t));
          if (itA == systin->spline_names.end()) {
            log<LOG_WARNING>(L"%1% || Systematic correlation %2% not in list. Skipping.") % __func__ % std::get<0>(t).c_str();
            continue;
          }

          auto itB = std::find(systin->spline_names.begin(), systin->spline_names.end(), std::get<1>(t));
          if (itB == systin->spline_names.end()) {
            log<LOG_WARNING>(L"%1% || Systematic correlation %2% not in list. Skipping.") % __func__ % std::get<1>(t).c_str();
            continue;
          }
         
          int iA = std::distance(systin->spline_names.begin(), itA);
          int iB = std::distance(systin->spline_names.begin(), itB);

          // set correlations
          prior_covariance(iA, iB) = std::get<2>(t);
          prior_covariance(iB, iA) = std::get<2>(t);
        }
        prior_covariance = systin->spline_priors.asDiagonal() * prior_covariance * systin->spline_priors.asDiagonal();
        for(size_t i = 0; i < systin->spline_prior_types.size(); ++i) {
            if (systin->spline_prior_types[i] == SplinePriorType::Uniform) {
                prior_covariance.row(i).setZero();
                prior_covariance.col(i).setZero();
                prior_covariance(i, i) = 1.0f;
            }
        }
        prior_covariance_inv = prior_covariance.inverse();
    }

    // GP: What do you do if the MC has 0 events in a bin?
    //     Proposed solution (hack?) here. Set the error to 1. This will return
    //     the correct answer if there are no data events in the bin. It is a bit
    //     iffier if there are data events in the bin, we may want to implement some
    //     error handling there.
    collapsed_stat_covariance = data.Spec().array().cwiseMax(1).matrix().asDiagonal();

    // Default-mode cache: in non-shape_only mode normdata == data.Spec() is constant
    // across all operator() invocations, so non_empty_indices and the reduced stat
    // covariance are constant. Build them once here and reuse them.
    if(!shape_only) {
        const Eigen::VectorXf &nd = data.Spec();
        for(Eigen::Index i = 0; i < nd.size(); ++i)
            if(nd(i) > 0 && binActive(i)) nec_indices.push_back(i);
        if(!nec_indices.empty()) {
            Eigen::VectorXf reduced_diag(nec_indices.size());
            for(size_t k = 0; k < nec_indices.size(); ++k)
                reduced_diag(k) = nd(nec_indices[k]);
            nec_reduced_stat_cov = Eigen::MatrixXf(reduced_diag.asDiagonal());
            nec_valid = true;
        }
        // If empty, leave nec_valid=false; operator() will throw on first call.
    }
}

float PROcovariance::operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool rungradient){
    call_count++;
    size_t nparams = nParams();
    size_t nsyst = syst->GetNSplines();
    //log<LOG_DEBUG>(L"%1% || nparams is %2%, nsyst is %3% ") % __func__ % nparams % nsyst;    

    Eigen::VectorXf subvector1 = param.segment(0, model.nparams);
    //log<LOG_DEBUG>(L"%1% || Created physics subvector with size %2%") % __func__ % subvector1.size();
    if(model.model_constraint){
        if(!model.model_constraint(subvector1)){
            // Keep value and gradient consistent for the minimizer: a huge
            // flat plateau with a stale gradient sends LBFGSB in a random
            // direction.
            if(rungradient) gradient.setZero();
            return 1e10;
        }
    }

    // Get Spectra from FillSpectra
    Eigen::VectorXf subvector2 = param.segment(nparams - nsyst, nsyst);

    // strat != EventByEvent (not == BinnedChi2): the FD gradient closures below
    // use the same condition, so for BinnedGrad the value and the gradient are
    // computed from the SAME (binned) spectrum model, and the fill cache stays
    // valid instead of being invalidated by an unbinned base call every
    // iteration.
    PROspec result = FillSpectra(config, peller, *syst, model, param, fs_cache, strat != EventByEvent, config.i_prime);

    Eigen::VectorXf normdata = shape_only
        ? data.Normalize(config,result)
        : data.Spec();
    const Eigen::VectorXf stat_variances = statisticalVariances(result, normdata, &param);

    // Collapsed systematic covariance computed as S^T F S with S = diag(spec)*T
    // kept sparse — the full-binning dense diag(s)*F*diag(s) is never
    // materialized (this runs on every evaluation). Note the
    // collapsed_stat_covariance member is deliberately NOT overwritten here:
    // it keeps the ctor's cwiseMax(1)-guarded form for getSingleChannelChi
    // (the per-call overwrite dropped that zero-bin guard and was only read
    // by the NaN diagnostics below).
    Eigen::MatrixXf collapsed_full_covariance = CollapsedScaledCovariance(config, syst->fractional_covariance, result.Spec());

    // non_empty_indices and reduced_collapsed_stat_covariance are constant in default
    // (non-shape_only) mode and were precomputed in the ctor; in shape_only mode
    // normdata depends on `result` so we must rebuild them per call.
    std::vector<Eigen::Index> nei_local;
    Eigen::MatrixXf rstat_local;
    if(!nec_valid) {
        for(Eigen::Index i = 0; i < normdata.size(); ++i)
            if(stat_variances(i) > 0 && binActive(i)) nei_local.push_back(i);
        if(nei_local.empty()) {
            log<LOG_ERROR>(L"%1% || ERROR: All (active) data bins are empty!") % __func__;
            throw std::runtime_error("All data bins are empty in PROchi.");
        }
        const Eigen::Map<const Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>>
            idx_local(nei_local.data(), (Eigen::Index)nei_local.size());
        rstat_local = Eigen::MatrixXf(stat_variances(idx_local).asDiagonal());
    }
    const std::vector<Eigen::Index> &non_empty_indices = nec_valid ? nec_indices : nei_local;
    const Eigen::MatrixXf &reduced_collapsed_stat_covariance = nec_valid ? nec_reduced_stat_cov : rstat_local;
    const size_t reduced_size = non_empty_indices.size();

    // Eigen 3.4 indexing handle for all the per-call reductions below (and in the
    // gradient blocks further down). Lightweight reference, no allocation.
    const Eigen::Map<const Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>>
        idx(non_empty_indices.data(), (Eigen::Index)reduced_size);

    // Per-call: reduced_collapsed_full_covariance depends on `result` via collapsed_full_covariance.
    Eigen::MatrixXf reduced_collapsed_full_covariance = collapsed_full_covariance(idx, idx);

    Eigen::MatrixXf M = reduced_collapsed_stat_covariance + reduced_collapsed_full_covariance;

    // Create reduced delta vector
    Eigen::VectorXf collapsed_mc_spec = CollapseMatrix(config, result.Spec());
    Eigen::VectorXf delta = collapsed_mc_spec(idx) - normdata(idx);

    float pull = Pull(subvector2);
    // delta^T M^-1 delta via Cholesky solve: faster + more stable than forming the inverse.
    float covar_portion = delta.dot(M.llt().solve(delta));
    float value = covar_portion + pull;

    if(std::isnan(value) || value!=value) {
        log<LOG_ERROR>(L"%1% || ERROR: PROcovariance chi2 is NaN (%2%). This is very bad.\n"
                L"covar_portion: %3%\npull: %4%\ndelta: %5%\n"
                L"mc spec: %6%\ndata spec: %7%")
            % __func__ % value % covar_portion % pull % delta % CollapseMatrix(config, result.Spec())
            % data.Spec();
        // stat covariance diagonal (normdata) for diagnostics
        for (Eigen::Index i = 0; i < normdata.size(); ++i) {
            log<LOG_ERROR>(L"%1% || ERROR: stat covariance diagonal (%2%) = %3%") % __func__ % i % normdata(i);
        }
        throw std::runtime_error("NANs in Chi().");

    }

    if(rungradient){
        // ----- Gradient mode dispatch -----
        // Four configurations driven by PROmetric::gradient_mode:
        //   GradientCentralFull  (default): central FD on full chi² (each FD
        //                          rebuilds covariance + Cholesky). Most
        //                          accurate, slowest.
        //   GradientOneSidedFull: forward FD on full chi² using base value.
        //                         ~2× faster, O(h) vs O(h²).
        //   GradientCentralLin:   central FD on δ only. M held at base; the
        //                         Gauss-Newton chain rule
        //                            d chi²/dθ_i ≈ 2 (M⁻¹δ_b)^T (dδ/dθ_i) + dP/dθ_i
        //                         is used. Drops the second-order
        //                         (M⁻¹δ)^T (dM/dθ) (M⁻¹δ) term — exact at the
        //                         minimum, very small far from it.
        //   GradientOneSidedLin:  forward FD on δ only + Gauss-Newton chain.
        //                         Fastest mode.
        //
        // Boundary handling is preserved across all modes: any FD step that
        // would land on a parameter bound is downgraded to a one-sided
        // stencil pointing into the interior, and the "out-of-bounds gradient
        // bounce" is applied (zero the gradient when it would push further
        // out of the box).
        const GradientMode mode = gradient_mode;
        const bool linearised = (mode == GradientCentralLin) || (mode == GradientOneSidedLin);
        const bool one_sided  = (mode == GradientOneSidedFull) || (mode == GradientOneSidedLin);

        // ----- Linearised pre-loop: solve M⁻¹ δ_b once, build analytic dP/dθ -----
        // For linearised modes we factorise the BASE M and reuse Minv_delta_b
        // across every FD step. The Pull derivative is computed analytically
        // from spline_centers / spline_priors (uncorrelated) or the precomputed
        // prior_covariance_inv (correlated) — no FD on the pull.
        Eigen::VectorXf Minv_delta_b;
        Eigen::VectorXf pull_grad_nuis; // size = nsyst, dP/dθ_n
        if (linearised) {
            Minv_delta_b = M.llt().solve(delta);
            const Eigen::VectorXf centered = subvector2 - syst->spline_centers;
            if (!correlated_systematics) {
                // Pull = sum_j (centered_j / sigma_j)^2  → dP/dθ_n_j = 2 centered_j / sigma_j^2
                pull_grad_nuis = 2.0f * centered.array() /
                                 (syst->spline_priors.array() * syst->spline_priors.array());
            } else {
                pull_grad_nuis = 2.0f * (prior_covariance_inv * centered);
            }
        }

        // ----- Helpers (closures over the outer scope) -----
        // compute_delta_at: build the reduced delta vector at an arbitrary
        // param. Uses the BASE call's normdata(idx) — matches the existing
        // semantics that the FD loop holds normdata fixed (relevant only for
        // shape_only mode, where data.Normalize depends on result; the
        // existing FD already holds it constant).
        auto compute_delta_at = [&](const Eigen::VectorXf &param_at,
                                    Eigen::VectorXf &delta_out) -> bool {
            if(model.model_constraint &&
               !model.model_constraint(param_at.segment(0, model.nparams))) return false;
            PROspec rl = FillSpectra(config, peller, *syst, model, param_at, fs_cache,
                                     strat != EventByEvent, config.i_prime);
            Eigen::VectorXf cmcl = CollapseMatrix(config, rl.Spec());
            delta_out = cmcl(idx) - normdata(idx);
            return true;
        };

        // compute_chi2_at: full chi² at an arbitrary param. Each call rebuilds
        // covariance + collapse + Cholesky from scratch. Used by the Full modes.
        auto compute_chi2_at = [&](const Eigen::VectorXf &param_at,
                                   float &chi2_out) -> bool {
            if(model.model_constraint &&
               !model.model_constraint(param_at.segment(0, model.nparams))) return false;
            PROspec rl = FillSpectra(config, peller, *syst, model, param_at, fs_cache,
                                     strat != EventByEvent, config.i_prime);
            Eigen::MatrixXf cfcl  = CollapsedScaledCovariance(config, syst->fractional_covariance, rl.Spec());
            Eigen::MatrixXf gM_lo;
            if(statisticalVariancesDependOnPrediction()) {
                const Eigen::VectorXf varied_stat = statisticalVariances(rl, normdata, &param_at);
                gM_lo = Eigen::MatrixXf(varied_stat(idx).asDiagonal()) + cfcl(idx, idx);
            } else {
                gM_lo = reduced_collapsed_stat_covariance + cfcl(idx, idx);
            }
            Eigen::VectorXf cmcl  = CollapseMatrix(config, rl.Spec());
            Eigen::VectorXf dl    = cmcl(idx) - normdata(idx);
            Eigen::VectorXf nuis  = param_at.segment(model.nparams, syst->GetNSplines());
            chi2_out = dl.dot(gM_lo.llt().solve(dl)) + Pull(nuis);
            return true;
        };

        // One reusable work vector: perturb component i in place and restore,
        // instead of two full parameter-vector copies per FD parameter
        // (O(nparams) copying per gradient instead of O(nparams^2)).
        Eigen::VectorXf param_work = param;

        for (size_t i = 0; i < model.nparams + nsyst; i++) {

            if(is_fixed.size() > 0 && is_fixed.at(i)) {
                gradient(i) = 0.0f;
                continue;
            }

            float h = (i < model.nparams) ? 1e-3f : 1e-4f;

            const float boundary_tol = 2.0f * std::numeric_limits<float>::epsilon();
            const bool at_lower = std::fabs(param(i) - lb(i)) < boundary_tol;
            const bool at_upper = std::fabs(ub(i) - param(i)) < boundary_tol;

            if (at_lower && at_upper) {
                gradient(i) = 0.0f;
                continue;
            }

            // Effective stencil: at boundary or in any one-sided mode → one-sided
            // forward (with sign +1 inland from lower bound, sign -1 inland from
            // upper bound, +1 in interior). Otherwise central.
            const bool boundary_step = (at_lower || at_upper);
            const int  sign          = boundary_step ? (at_lower ? 1 : -1) : 1;
            const bool use_central   = !boundary_step && !one_sided;

            float grad_i = 0.0f;

            if (linearised) {
                // ----- Linearised: FD on δ, M frozen at base, analytic pull deriv -----
                Eigen::VectorXf delta_plus, delta_minus;
                param_work(i) = param(i) + sign * h;
                bool ok_plus  = compute_delta_at(param_work,  delta_plus);
                bool ok_minus = true;
                if (use_central) {
                    param_work(i) = param(i) - sign * h;
                    ok_minus = compute_delta_at(param_work, delta_minus);
                }
                param_work(i) = param(i);

                Eigen::VectorXf ddelta_dtheta;
                if (use_central) {
                    if (!ok_plus && !ok_minus) {
                        gradient(i) = 0.0f; // both sides infeasible
                        if (!std::isfinite(gradient(i))) gradient(i) = 0.0f;
                        continue;
                    }
                    if (!ok_plus) {
                        // Push away from infeasible side.
                        gradient(i) = +1e10f;
                        continue;
                    }
                    if (!ok_minus) {
                        gradient(i) = -1e10f;
                        continue;
                    }
                    ddelta_dtheta = (delta_plus - delta_minus) / (2.0f * h);
                } else {
                    if (!ok_plus) {
                        gradient(i) = sign * 1e10f;
                        if (boundary_step && sign * gradient(i) > 0) gradient(i) = 0.0f;
                        continue;
                    }
                    // ∂δ/∂θ ≈ sign * (δ_+ - δ_b) / h  (forward at lower / interior, backward at upper)
                    ddelta_dtheta = (sign * (delta_plus - delta)) / h;
                }

                // Linearised chain rule: d(δ^T M⁻¹ δ)/dθ = 2 (M⁻¹δ_b)^T dδ/dθ
                grad_i = 2.0f * Minv_delta_b.dot(ddelta_dtheta);
                // Analytic pull contribution (zero for physics indices).
                if (i >= model.nparams) {
                    grad_i += pull_grad_nuis(i - model.nparams);
                }
                gradient(i) = grad_i;
            } else {
                // ----- Full FD path -----
                if (use_central) {
                    float chi2_plus = 0.0f, chi2_minus = 0.0f;
                    param_work(i) = param(i) + sign * h;
                    const bool okp = compute_chi2_at(param_work, chi2_plus);
                    param_work(i) = param(i) - sign * h;
                    const bool okm = compute_chi2_at(param_work, chi2_minus);
                    param_work(i) = param(i);
                    if (!okp) chi2_plus  = 1e10f;
                    if (!okm) chi2_minus = 1e10f;
                    grad_i = (chi2_plus - chi2_minus) / (2.0f * h);
                    gradient(i) = grad_i;
                } else {
                    // One-sided: gradient ≈ sign * (chi²(θ+sign*h) - value) / h
                    float chi2_one = 0.0f;
                    param_work(i) = param(i) + sign * h;
                    const bool ok_one = compute_chi2_at(param_work, chi2_one);
                    param_work(i) = param(i);
                    if (!ok_one) {
                        gradient(i) = sign * 1e10f;
                        // Don't apply bounce check here — we *want* a huge gradient
                        // pushing back into the feasible region.
                        if (!std::isfinite(gradient(i))) gradient(i) = 0.0f;
                        continue;
                    }
                    grad_i = sign * (chi2_one - value) / h;
                    gradient(i) = grad_i;
                }
            }

            // Boundary "bounce": if the gradient would push further out of the
            // box, zero it. LBFGSB's projected step needs this to stay in [lb, ub].
            if (boundary_step && sign * gradient(i) > 0) {
                gradient(i) = 0.0f;
            }

            if (!std::isfinite(gradient(i))) gradient(i) = 0.0f;
        }
    }

    //Update last param
    last_param = param;
    last_value = value;

    return value;
}

float PROcovariance::getSingleChannelChi(size_t global_channel_index, const PROspec &cv, size_t var_index, const Eigen::MatrixXf &projection) {

    size_t nbin = config.m_channel_variable_bins[config.GetLocalChannelIndexFromGlobalChannelIndex(global_channel_index)][var_index].NBins();
    size_t startBin = config.GetCollapsedGlobalVariableBinStart(global_channel_index, var_index);

    const Eigen::VectorXf comparison = shape_only ? data.Normalize(config, cv) : data.Spec();
    const Eigen::VectorXf stat_variances = statisticalVariances(cv, comparison);

    // Restrict to this channel's active bins. Only meaningful for the fitting variable
    // (the mask snapshot is for i_prime); other variables see every bin active.
    const bool masked = (var_index == (size_t)config.i_prime) && hasActiveBinMask();
    std::vector<Eigen::Index> local_idx;
    for(size_t b = 0; b < nbin; ++b)
        if((!masked || binActive((Eigen::Index)(startBin + b))) &&
           stat_variances((Eigen::Index)(startBin + b)) > 0.0f)
            local_idx.push_back((Eigen::Index)(startBin + b));
    if(local_idx.empty()) return 0.0f;
    const Eigen::Map<const Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>>
        idx(local_idx.data(), (Eigen::Index)local_idx.size());

    Eigen::MatrixXf M = Eigen::MatrixXf(stat_variances(idx).asDiagonal());
    if(syst->GetNCovar()){
        Eigen::MatrixXf collapsed_full_covariance = CollapsedScaledCovariance(config, syst->fractional_covariance, cv.Spec());
        M += collapsed_full_covariance(idx, idx);
    }

    Eigen::VectorXf delta = (CollapseMatrix(config, cv.Spec()) - comparison)(idx);
    if(projection.size()) {
        Eigen::MatrixXf active_projection(projection.rows(), idx.size());
        for(Eigen::Index col = 0; col < idx.size(); ++col) {
            active_projection.col(col) = projection.col(idx(col) - (Eigen::Index)startBin);
        }
        M = active_projection * M * active_projection.transpose();
        delta = active_projection * delta;
    }
    float covar_portion = delta.dot(M.llt().solve(delta));
    float value = covar_portion;//pull;

    return value;
}

void PROcovariance::print([[maybe_unused]] const Eigen::VectorXf &param){


return;
}
