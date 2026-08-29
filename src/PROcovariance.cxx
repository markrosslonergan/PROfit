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

}

void PROcovariance::buildConstantStatCache(const Eigen::VectorXf &variances) {
    nec_indices.clear();
    nec_reduced_stat_cov.resize(0, 0);
    nec_valid = false;
    for(Eigen::Index i = 0; i < variances.size(); ++i)
        if(variances(i) > 0 && binActive(i)) nec_indices.push_back(i);
    // If nothing survives, leave the cache invalid: operator() rebuilds per call and
    // throws there, where the error can name the offending evaluation.
    if(nec_indices.empty()) return;
    const Eigen::Map<const Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>>
        idx(nec_indices.data(), (Eigen::Index)nec_indices.size());
    nec_reduced_stat_cov = Eigen::MatrixXf(variances(idx).asDiagonal());
    nec_valid = true;
}

void PROcovariance::reset() {
    physics_param_fixed.clear();
    last_value = 0;
    last_param = Eigen::VectorXf::Constant(last_param.size(), 0);
    fs_cache.invalidate();
}

Eigen::VectorXf PROcovariance::singleChannelStatVariances(
        const Eigen::VectorXf &collapsed_cv, const Eigen::VectorXf &comparison) const {
    return statisticalVariances(collapsed_cv, comparison, nullptr);
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

    // Collapse once and reuse for the variance hook, the delta, and the diagnostics
    // below. PROpearson's variance IS the collapsed prediction, so handing the hook an
    // already-collapsed vector removes a second CollapseMatrix from every evaluation.
    const Eigen::VectorXf collapsed_mc_spec = CollapseMatrix(config, result.Spec());
    const Eigen::VectorXf stat_variances = statisticalVariances(collapsed_mc_spec, normdata, &param);

    // Collapsed systematic covariance computed as S^T F S with S = diag(spec)*T
    // kept sparse — the full-binning dense diag(s)*F*diag(s) is never
    // materialized (this runs on every evaluation).
    Eigen::MatrixXf collapsed_full_covariance = CollapsedScaledCovariance(config, syst->fractional_covariance, result.Spec());

    // non_empty_indices and reduced_collapsed_stat_covariance are precomputed once
    // (PROcovariance::buildConstantStatCache) only when the concrete metric guarantees
    // that neither the selected bins nor their variances move with the prediction.
    // Otherwise they are rebuilt here on every call.
    std::vector<Eigen::Index> nei_local;
    Eigen::MatrixXf rstat_local;
    if(!nec_valid) {
        for(Eigen::Index i = 0; i < normdata.size(); ++i)
            if(stat_variances(i) > 0 && binActive(i)) nei_local.push_back(i);
        if(nei_local.empty()) {
            log<LOG_ERROR>(L"%1% || ERROR: No active bin has a positive statistical variance!") % __func__;
            throw std::runtime_error("No usable bins in PROcovariance.");
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
    Eigen::VectorXf delta = collapsed_mc_spec(idx) - normdata(idx);

    float pull = Pull(subvector2);
    // delta^T M^-1 delta via Cholesky solve: faster + more stable than forming the inverse.
    float covar_portion = delta.dot(M.llt().solve(delta));
    float value = covar_portion + pull;

    if(std::isnan(value) || value!=value) {
        log<LOG_ERROR>(L"%1% || ERROR: PROcovariance chi2 is NaN (%2%). This is very bad.\n"
                L"covar_portion: %3%\npull: %4%\ndelta: %5%\n"
                L"mc spec: %6%\ndata spec: %7%")
            % __func__ % value % covar_portion % pull % delta % collapsed_mc_spec
            % data.Spec();
        // stat covariance diagonal (normdata) for diagnostics
        for (Eigen::Index i = 0; i < normdata.size(); ++i) {
            log<LOG_ERROR>(L"%1% || ERROR: stat covariance diagonal (%2%) = %3%") % __func__ % i % normdata(i);
        }
        throw std::runtime_error("NANs in Chi().");

    }

    if(rungradient){
        // ----- Gradient mode dispatch -----
        // Five configurations driven by PROmetric::gradient_mode:
        //   GradientAnalytic     (default): fully analytic — dδ/dθ from
        //                          FillSpectraGradient plus the closed-form
        //                          uᵀ(dM/dθ)u term. See the analytic block
        //                          below for the derivation and its two
        //                          validity requirements.
        //   GradientCentralFull:   central FD on full chi² (each FD
        //                          rebuilds covariance + Cholesky). Most
        //                          accurate FD mode, slowest.
        //   GradientOneSidedFull: forward FD on full chi² using base value.
        //                         ~2× faster, O(h) vs O(h²).
        //   GradientCentralLin:   central FD on δ only. M held at base; the
        //                         Gauss-Newton chain rule
        //                            d chi²/dθ_i ≈ 2 (M⁻¹δ_b)^T (dδ/dθ_i) + dP/dθ_i
        //                         is used. Drops the second-order
        //                         (M⁻¹δ)^T (dM/dθ) (M⁻¹δ) term — exact at the
        //                         minimum, very small far from it.
        //   GradientOneSidedLin:  forward FD on δ only + Gauss-Newton chain.
        //                         Fastest FD mode.
        //
        // Boundary handling is preserved across all modes: any FD step that
        // would land on a parameter bound is downgraded to a one-sided
        // stencil pointing into the interior, and the "out-of-bounds gradient
        // bounce" is applied (zero the gradient when it would push further
        // out of the box).
        GradientMode mode = gradient_mode;
        // The analytic gradient requires (a) the binned factorised spectrum
        // (spec = systw .* H*probs) for FillSpectraGradient — the event-by-event
        // fill mixes physics and systematics per event — and (b) a statistical
        // covariance that does not move with the parameters, since the closed-form
        // dM/dθ term below only differentiates the systematic part of M. Metrics
        // whose statisticalVariances depend on the prediction (PROpearson, PROCNP)
        // would need an extra diag(dvar/dθ) term that is not wired up yet, so they
        // fall back to the Gauss-Newton FD mode (exact at the minimum).
        if (mode == GradientAnalytic &&
            (strat == EventByEvent || statisticalVariancesDependOnPrediction())) {
            static std::atomic<bool> warned_fallback{false};
            if(!warned_fallback.exchange(true))
                log<LOG_WARNING>(L"%1% || Analytic gradient not available for this metric/strategy (EventByEvent, or prediction-dependent statistical variance); falling back to %2%.") % __func__ % gradientModeName(GradientFallback);
            mode = GradientFallback;
        }
        const bool analytic   = (mode == GradientAnalytic);
        const bool linearised = (mode == GradientCentralLin) || (mode == GradientOneSidedLin);
        const bool one_sided  = (mode == GradientOneSidedFull) || (mode == GradientOneSidedLin);

        // ----- Linearised pre-loop: solve M⁻¹ δ_b once, build analytic dP/dθ -----
        // For linearised modes we factorise the BASE M and reuse Minv_delta_b
        // across every FD step. The Pull derivative is computed analytically
        // from spline_centers / spline_priors (uncorrelated) or the precomputed
        // prior_covariance_inv (correlated) — no FD on the pull.
        Eigen::VectorXf Minv_delta_b;
        Eigen::VectorXf pull_grad_nuis; // size = nsyst, dP/dθ_n
        if (linearised || analytic) {
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

        if (analytic) {
            // ----- Fully analytic gradient -----
            // Notation (θ = all parameters, physics + nuisance):
            //   s(θ)  full-binning MC spectrum (result.Spec(), N bins)
            //   T     collapsing matrix (N × n_c, sparse): collapsed spectrum c = Tᵀ s
            //   d     comparison spectrum (collapsed), idx = usable active bins.
            //         In shape_only mode d is the area-normalised data held FIXED at
            //         the base point — the same frozen-normdata convention every FD
            //         mode below uses (see compute_delta_at) — so d is θ-independent
            //         here in all modes.
            //   δ(θ)  = c(idx) − d(idx)                                  (reduced residual)
            //   F     fractional systematic covariance (N × N, symmetric)
            //   M(θ)  = diag(var(idx)) + [Tᵀ diag(s) F diag(s) T](idx,idx) (stat + syst)
            //         with var θ-independent (guaranteed by the dispatch gate above:
            //         prediction-dependent variances fall back to FD).
            //   P(θ)  nuisance pull penalty
            //   χ²(θ) = δᵀ M⁻¹ δ + P
            //
            // Step 1 — differentiate χ² = δᵀ M⁻¹ δ + P term by term (product rule;
            // both δ and M depend on θ). Since M is symmetric the two residual
            // terms are equal, and d(M⁻¹)/dθ = −M⁻¹ (dM/dθ) M⁻¹:
            //   dχ²/dθ = (dδ/dθ)ᵀ M⁻¹ δ + δᵀ M⁻¹ (dδ/dθ) + δᵀ [d(M⁻¹)/dθ] δ + dP/dθ
            //          = 2 δᵀ M⁻¹ (dδ/dθ) − δᵀ M⁻¹ (dM/dθ) M⁻¹ δ + dP/dθ.
            // Writing u = M⁻¹ δ (one Cholesky solve, shared by both terms):
            //   dχ²/dθ = 2 uᵀ (dδ/dθ) − uᵀ (dM/dθ) u + dP/dθ.
            //
            // Step 2 — residual term. δ depends on θ only through s, so with the
            // spectrum Jacobian G = ds/dθ (N × nparams, from FillSpectraGradient):
            //   dδ/dθ = (Tᵀ G)(idx, :)     ⇒     2 uᵀ (dδ/dθ) = 2 (TᵀG)(idx,:)ᵀ u.
            //
            // Step 3 — covariance term uᵀ (dM/dθ) u (this is the term the linearised
            // FD modes drop). Work one parameter θ_j at a time, and write ṡ = ds/dθ_j
            // (= G(:,j), a full-binning vector). The stat part diag(var) does not depend
            // on θ, so only M_sys = [Tᵀ diag(s) F diag(s) T](idx,idx) contributes;
            // the product rule on diag(s) F diag(s) gives
            //   dM_sys/dθ_j = [Tᵀ ( diag(ṡ) F diag(s) + diag(s) F diag(ṡ) ) T](idx,idx).
            //
            // 3a. Undo the (idx,idx) restriction. u lives on the idx bins only; define
            //     u_c ∈ R^{n_c} as u placed at its idx positions and 0 elsewhere. Then
            //     for any n_c × n_c matrix A,  uᵀ A(idx,idx) u = u_cᵀ A u_c  (the zeros
            //     kill every row/column outside idx), so
            //       uᵀ (dM/dθ_j) u = u_cᵀ Tᵀ ( diag(ṡ) F diag(s) + diag(s) F diag(ṡ) ) T u_c.
            //
            // 3b. Undo the collapse. Define w = T u_c ∈ R^N: since T maps each full
            //     bin to exactly one collapsed bin, w is just u_c broadcast — every
            //     full bin carries the u value of the collapsed bin it belongs to. So
            //       uᵀ (dM/dθ_j) u = wᵀ diag(ṡ) F diag(s) w + wᵀ diag(s) F diag(ṡ) w.
            //
            // 3c. The two summands are transposes of each other (F is symmetric), and
            //     each is a scalar, so they are equal:
            //       uᵀ (dM/dθ_j) u = 2 wᵀ diag(ṡ) F diag(s) w.
            //
            // 3d. Read this as elementwise products of N-vectors (∘ = Hadamard):
            //     diag(s) w = w ∘ s, and diag(ṡ) w = w ∘ ṡ, hence
            //       2 wᵀ diag(ṡ) F diag(s) w = 2 (w ∘ ṡ) · [F (w ∘ s)] = 2 ṡ · ( w ∘ g ),
            //     with g = F (w ∘ s) ∈ R^N (one matrix-vector product with F).
            //     Note g depends on s, u and F but NOT on which parameter θ_j we are
            //     differentiating with respect to — so it is computed once per gradient
            //     call, and the per-parameter cost is one dot product G(:,j)·(w ∘ g).
            //
            // Putting it together, for each parameter j:
            //   dχ²/dθ_j = 2 (TᵀG)(idx,j)·u − 2 G(:,j)·(w ∘ g) + dP/dθ_j,
            // i.e. grad = 2 (TᵀG)(idx,:)ᵀ u − 2 Gᵀ (w∘g) + [0; dP/dθ_nuis]. No finite
            // differences, no extra spectrum fills, no extra Cholesky factorisation.
            const Eigen::SparseMatrix<float> &T = config.GetCollapsingMatrixSparse();
            Eigen::VectorXf u_c = Eigen::VectorXf::Zero(T.cols());
            for(size_t k = 0; k < reduced_size; ++k)
                u_c(non_empty_indices[k]) = Minv_delta_b(k);
            const Eigen::VectorXf w_full = T * u_c;
            const Eigen::VectorXf g_full = syst->fractional_covariance * w_full.cwiseProduct(result.Spec());
            const Eigen::VectorXf wg = w_full.cwiseProduct(g_full);

            Eigen::MatrixXf G = FillSpectraGradient(config, peller, *syst, model, param, fs_cache, config.i_prime);
            Eigen::MatrixXf TtG = T.transpose() * G; // collapsed-space Jacobian (ncollapsed × nparams)
            Eigen::VectorXf grad_vec = 2.0f * (TtG(idx, Eigen::all).transpose() * Minv_delta_b)
                                     - 2.0f * (G.transpose() * wg);
            grad_vec.segment(model.nparams, nsyst) += pull_grad_nuis;

            for (size_t i = 0; i < model.nparams + nsyst; i++) {
                if(is_fixed.size() > 0 && is_fixed.at(i)) { gradient(i) = 0.0f; continue; }
                gradient(i) = grad_vec(i);
                // Same boundary "bounce" as the FD paths: zero the gradient when
                // it would push further out of the box.
                const float boundary_tol = 2.0f * std::numeric_limits<float>::epsilon();
                const bool at_lower = std::fabs(param(i) - lb(i)) < boundary_tol;
                const bool at_upper = std::fabs(ub(i) - param(i)) < boundary_tol;
                if ((at_lower && gradient(i) > 0) || (at_upper && gradient(i) < 0))
                    gradient(i) = 0.0f;
                if (!std::isfinite(gradient(i))) gradient(i) = 0.0f;
            }

            last_param = param;
            last_value = value;
            return value;
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
            Eigen::VectorXf cmcl  = CollapseMatrix(config, rl.Spec());
            Eigen::MatrixXf gM_lo;
            if(statisticalVariancesDependOnPrediction()) {
                const Eigen::VectorXf varied_stat = statisticalVariances(cmcl, normdata, &param_at);
                gM_lo = Eigen::MatrixXf(varied_stat(idx).asDiagonal()) + cfcl(idx, idx);
            } else {
                gM_lo = reduced_collapsed_stat_covariance + cfcl(idx, idx);
            }
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

    const Eigen::VectorXf collapsed_cv = CollapseMatrix(config, cv.Spec());
    const Eigen::VectorXf comparison = shape_only ? data.Normalize(config, cv) : data.Spec();
    const Eigen::VectorXf stat_variances = singleChannelStatVariances(collapsed_cv, comparison);

    // Restrict to this channel's active bins. Only meaningful for the fitting variable
    // (the mask snapshot is for i_prime); other variables see every bin active. Bins are
    // NOT additionally filtered on a positive variance here: this per-channel diagnostic
    // has always reported every masked-in bin, and each metric's
    // singleChannelStatVariances is responsible for keeping its own variances usable.
    const bool masked = (var_index == (size_t)config.i_prime) && hasActiveBinMask();
    std::vector<Eigen::Index> local_idx;
    for(size_t b = 0; b < nbin; ++b)
        if(!masked || binActive((Eigen::Index)(startBin + b)))
            local_idx.push_back((Eigen::Index)(startBin + b));
    if(local_idx.empty()) return 0.0f;
    const Eigen::Map<const Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>>
        idx(local_idx.data(), (Eigen::Index)local_idx.size());

    Eigen::MatrixXf M = Eigen::MatrixXf(stat_variances(idx).asDiagonal());
    if(syst->GetNCovar()){
        Eigen::MatrixXf collapsed_full_covariance = CollapsedScaledCovariance(config, syst->fractional_covariance, cv.Spec());
        M += collapsed_full_covariance(idx, idx);
    }

    Eigen::VectorXf delta = (collapsed_cv - comparison)(idx);
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
