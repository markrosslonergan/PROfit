#include "PROmetrics/PROpoisson.h"
#include "PROcess.h"
#include "PROdata.h"
#include "PROlog.h"
#include "PROtocall.h"

#include <cmath>
using namespace PROfit;

namespace {
    // Baker-Cousins Poisson chi²: 2 Σ_b [ s_b - n_b + n_b ln(n_b/s_b) ].
    // The n_b → 0 limit is handled exactly (the bin contributes 2 s_b); a
    // vectorised 0*log(0) would be NaN. A non-positive prediction with data in
    // the bin is infinitely disfavored in principle; it gets a large finite
    // penalty so the minimizer can still move away from it.
    // When a fit-region mask is given, bin b is skipped unless active[b + offset]
    // is nonzero (offset supports channel-local segments with a global mask).
    float BakerCousinsChi2(const Eigen::VectorXf &vmc, const Eigen::VectorXf &vdata,
                           const std::vector<char> *active = nullptr, Eigen::Index offset = 0) {
        float sum = 0.0f;
        for(Eigen::Index b = 0; b < vmc.size(); ++b) {
            if(active && !active->empty() && !(*active)[(size_t)(b + offset)]) continue;
            const float s = vmc(b), n = vdata(b);
            if(n <= 0.0f) {
                if(s > 0.0f) sum += s;
            } else if(s > 0.0f) {
                sum += s - n + n * std::log(n / s);
            } else {
                sum += 1e8f;
            }
        }
        return 2.0f * sum;
    }
}


PROpoisson::PROpoisson(const std::string tag, const PROconfig &conin, const PROpeller &pin, const PROsyst *systin, const PROmodel &modelin, const PROdata &datain, EvalStrategy strat, bool shape_only, std::vector<float> physics_param_fixed) : PROmetric(tag, conin, systin, modelin, datain, strat, shape_only, physics_param_fixed), config(conin), peller(pin) {

    // PROmetric's constructor has already snapshotted the config's fit-region mask (if
    // any); masked bins are skipped in the Baker-Cousins sum and contribute zero gradient.

    if(syst->GetNCovar()) {
        log<LOG_WARNING>(L"%1% || Warning: Using a systematics object with covariance systematics with"
                          " Poisson log likelihood ratio. This is not supported and the covariance"
                          " systematics will be ignored.") % __func__;
    }

    // Build the correlation matrix between priors if configured to
    if (conin.m_mcgen_correlations.size()) {
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

void PROpoisson::reset() {
    physics_param_fixed.clear();
    last_value = 0;
    last_param = Eigen::VectorXf::Constant(last_param.size(), 0);
    fs_cache.invalidate();
}

PROmetric *PROpoisson::Clone() const {
    return new PROpoisson(model_tag, config, peller, syst, model, data, strat, shape_only, physics_param_fixed);
}

float PROpoisson::operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool rungradient){
    call_count++;
    size_t nparams = model.nparams+syst->GetNSplines();
    size_t nsyst = syst->GetNSplines();
    log<LOG_DEBUG>(L"%1% || nparams is %2%, nsyst is %3% ") % __func__ % nparams % nsyst;    

    // Get Spectra from FillSpectra
    Eigen::VectorXf subvector1 = param.segment(0, nparams - nsyst);
    if(model.model_constraint){
        if(!model.model_constraint(subvector1)){
            // Keep value and gradient consistent for the minimizer.
            if(rungradient) gradient.setZero();
            return 1e10;
        }
    }
    Eigen::VectorXf subvector2 = param.segment(nparams - nsyst, nsyst);

    // strat != EventByEvent (not == BinnedChi2): matches the FD gradient
    // helpers so BinnedGrad uses one consistent spectrum model and keeps the
    // fill cache valid.
    PROspec result = FillSpectra(config, peller, *syst, model, param, fs_cache, strat != EventByEvent, config.i_prime);

    const Eigen::VectorXf vdata = shape_only
        ? data.Normalize(config,result)
        : data.Spec();
    const Eigen::VectorXf vmc = CollapseMatrix(config, result.Spec());
    float poisson = BakerCousinsChi2(vmc, vdata, &active_bins);
    float pull = Pull(subvector2);
    float value = poisson + pull;

    if(rungradient){
        // ----- Gradient mode dispatch (see PROmetric::GradientMode) -----
        //
        // Poisson chi² has the closed form
        //     chi²_Pois = 2 Σ_b [ s_b - n_b + n_b · ln(n_b / s_b) ] + Pull
        // with analytic per-bin sensitivity
        //     d chi²_Pois / d s_b = 2 (1 - n_b / s_b)
        //
        // Linearised modes here compose this analytic ∂chi²/∂s with an FD on s:
        //     d chi²/dθ_i ≈ Σ_b 2(1 - n_b/s_b^base) · (ds_b/dθ_i)_FD + dP/dθ_i
        // For shape_only mode we hold n_b at its base value (same convention as
        // the existing FD code held vdata fixed in the FD branches via the
        // outer-scope `vdata` — see lines 117-119 of the previous version).
        //
        // Note: previous default for PROpoisson was a sign-heuristic 1-sided FD
        // that picked direction based on the LBFGS step. The new default
        // (GradientCentralFull) matches PROchi/PROCNP and gives a more
        // accurate gradient. To preserve the legacy heuristic, set
        // GradientOneSidedFull — but with a fixed forward stencil rather than
        // the LBFGS-direction-tracking heuristic, which was always somewhat
        // approximate.
        GradientMode mode = gradient_mode;
        // Analytic gradient is implemented in PROchi only so far. For Poisson the
        // linearised chain rule is already exact modulo FD truncation in dδ/dθ.
        if (mode == GradientAnalytic) {
            static std::atomic<bool> warned_analytic{false};
            if(!warned_analytic.exchange(true))
                log<LOG_WARNING>(L"%1% || Analytic gradient not implemented for PROpoisson; falling back to %2%.") % __func__ % gradientModeName(GradientFallback);
            mode = GradientFallback;
        }
        const bool linearised = (mode == GradientCentralLin) || (mode == GradientOneSidedLin);
        const bool one_sided  = (mode == GradientOneSidedFull) || (mode == GradientOneSidedLin);

        // Base spectrum + per-bin sensitivity for linearised modes.
        // vdata and vmc were already computed for the base value.
        Eigen::VectorXf w_base;     // 2 (1 - n/s) at base, used by linearised
        Eigen::VectorXf pull_grad_nuis;
        Eigen::VectorXf vmc_base = vmc;  // base collapsed MC (preserve before perturbed FillSpectra calls)
        Eigen::VectorXf vdata_base = vdata;
        if (linearised) {
            // Guard against zero/negative s entries.
            w_base.resize(vmc_base.size());
            for (long b = 0; b < vmc_base.size(); ++b) {
                // Masked-out bins get zero sensitivity so they contribute nothing to
                // the linearised chain rule w_base . ds/dtheta.
                if (!binActive(b)) { w_base(b) = 0.0f; continue; }
                const float s = vmc_base(b);
                w_base(b) = (s > 0.0f) ? 2.0f * (1.0f - vdata_base(b) / s) : 0.0f;
            }
            const Eigen::VectorXf centered = subvector2 - syst->spline_centers;
            if (!correlated_systematics) {
                pull_grad_nuis = 2.0f * centered.array() /
                                 (syst->spline_priors.array() * syst->spline_priors.array());
            } else {
                pull_grad_nuis = 2.0f * (prior_covariance_inv * centered);
            }
        }

        // Helper: collapsed MC spec at arbitrary param. Holds vdata at base
        // (matches the existing FD semantics in shape_only mode).
        auto compute_vmc_at = [&](const Eigen::VectorXf &param_at,
                                  Eigen::VectorXf &vmc_out) -> bool {
            Eigen::VectorXf phys = param_at.segment(0, nparams - nsyst);
            if(model.model_constraint && !model.model_constraint(phys)) return false;
            PROspec rl = FillSpectra(config, peller, *syst, model, param_at, fs_cache,
                                     strat != EventByEvent, config.i_prime);
            vmc_out = CollapseMatrix(config, rl.Spec());
            return true;
        };

        // Helper: full chi² Poisson + Pull at arbitrary param.
        auto compute_chi2_at = [&](const Eigen::VectorXf &param_at,
                                   float &chi2_out) -> bool {
            Eigen::VectorXf phys = param_at.segment(0, nparams - nsyst);
            if(model.model_constraint && !model.model_constraint(phys)) return false;
            PROspec rl = FillSpectra(config, peller, *syst, model, param_at, fs_cache,
                                     strat != EventByEvent, config.i_prime);
            // vdata may depend on result in shape_only mode; preserve that
            // (shape_only re-normalises to perturbed result like the original).
            const Eigen::VectorXf vdata_l = shape_only ? data.Normalize(config, rl) : data.Spec();
            const Eigen::VectorXf vmc_l   = CollapseMatrix(config, rl.Spec());
            float pois = BakerCousinsChi2(vmc_l, vdata_l, &active_bins);
            Eigen::VectorXf nuis = param_at.segment(nparams - nsyst, nsyst);
            chi2_out = pois + Pull(nuis);
            return true;
        };

        // One reusable work vector: perturb component i in place and restore,
        // instead of two full parameter-vector copies per FD parameter.
        Eigen::VectorXf param_work = param;

        for (size_t i = 0; i < nparams; i++) {
            if (is_fixed.size() > 0 && is_fixed.at(i)) {
                gradient(i) = 0.0f;
                continue;
            }

            const float h = (i < nparams - nsyst) ? 1e-3f : 1e-4f;

            // Boundary detection: only meaningful if bounds were set on the
            // metric (setBounds). Otherwise treat as interior.
            const bool have_bounds = (lb.size() > (Eigen::Index)i && ub.size() > (Eigen::Index)i);
            const float boundary_tol = 2.0f * std::numeric_limits<float>::epsilon();
            const bool at_lower = have_bounds && std::fabs(param(i) - lb(i)) < boundary_tol;
            const bool at_upper = have_bounds && std::fabs(ub(i) - param(i)) < boundary_tol;

            if (at_lower && at_upper) {
                gradient(i) = 0.0f;
                continue;
            }

            const bool boundary_step = (at_lower || at_upper);
            const int  sign          = boundary_step ? (at_lower ? 1 : -1) : 1;
            const bool use_central   = !boundary_step && !one_sided;

            if (linearised) {
                Eigen::VectorXf vmc_plus, vmc_minus;
                param_work(i) = param(i) + sign * h;
                bool ok_plus  = compute_vmc_at(param_work,  vmc_plus);
                bool ok_minus = true;
                if (use_central) {
                    param_work(i) = param(i) - sign * h;
                    ok_minus = compute_vmc_at(param_work, vmc_minus);
                }
                param_work(i) = param(i);

                Eigen::VectorXf ds_dtheta;
                if (use_central) {
                    if (!ok_plus && !ok_minus) { gradient(i) = 0.0f; continue; }
                    if (!ok_plus)  { gradient(i) = +1e10f; continue; }
                    if (!ok_minus) { gradient(i) = -1e10f; continue; }
                    ds_dtheta = (vmc_plus - vmc_minus) / (2.0f * h);
                } else {
                    if (!ok_plus) {
                        gradient(i) = sign * 1e10f;
                        if (boundary_step && sign * gradient(i) > 0) gradient(i) = 0.0f;
                        continue;
                    }
                    ds_dtheta = (sign * (vmc_plus - vmc_base)) / h;
                }

                float grad_i = w_base.dot(ds_dtheta);
                if (i >= (size_t)(nparams - nsyst)) {
                    grad_i += pull_grad_nuis(i - (nparams - nsyst));
                }
                gradient(i) = grad_i;
            } else {
                if (use_central) {
                    float chi2_plus = 1e10f, chi2_minus = 1e10f;
                    param_work(i) = param(i) + sign * h;
                    compute_chi2_at(param_work,  chi2_plus);
                    param_work(i) = param(i) - sign * h;
                    compute_chi2_at(param_work, chi2_minus);
                    param_work(i) = param(i);
                    gradient(i) = (chi2_plus - chi2_minus) / (2.0f * h);
                } else {
                    float chi2_one = 0.0f;
                    param_work(i) = param(i) + sign * h;
                    const bool ok_one = compute_chi2_at(param_work, chi2_one);
                    param_work(i) = param(i);
                    if (!ok_one) {
                        gradient(i) = sign * 1e10f;
                        if (!std::isfinite(gradient(i))) gradient(i) = 0.0f;
                        continue;
                    }
                    gradient(i) = sign * (chi2_one - value) / h;
                }
            }

            if (boundary_step && sign * gradient(i) > 0) {
                gradient(i) = 0.0f;
            }
            if (!std::isfinite(gradient(i))) gradient(i) = 0.0f;
        }
    }
    //std::cout<<"Grad: "<<gradient<<std::endl;

    //log<LOG_DEBUG>(L"%1% || value %2%, last_value %3%, pull") % __func__ % value  % last_value % pull;
    log<LOG_DEBUG>(L"%1% || FINISHED ITERATION got vals: %2% %3%") % __func__ % value % last_value ;

    //Update last param
    last_param = param;
    last_value = value;

    return value;
}

float PROpoisson::getSingleChannelChi(size_t global_channel_index, const PROspec &cv, size_t var_index, const Eigen::MatrixXf &projection) {

    // m_channel_variable_bins is indexed by LOCAL channel index (PROchi and
    // PROCNP convert the same way); using the global index breaks any config
    // with more than one mode x detector.
    size_t nbin = config.m_channel_variable_bins[config.GetLocalChannelIndexFromGlobalChannelIndex(global_channel_index)][var_index].NBins();
    size_t startBin = config.GetCollapsedGlobalVariableBinStart(global_channel_index,var_index);


    // const Eigen::VectorXf &vdata = data.Spec().segment(startBin, nbin);
    Eigen::VectorXf vdata = (shape_only
        ? data.Normalize(config,cv)
        : data.Spec()).segment(startBin, nbin);
    Eigen::VectorXf vmc = CollapseMatrix(config, cv.Spec()).segment(startBin, nbin);
    // Mask applies only to the fitting variable (mask snapshot is for i_prime);
    // startBin offsets the channel-local segment into the global mask.
    const bool masked = (var_index == (size_t)config.i_prime) && hasActiveBinMask();
    if(projection.size()) {
        vmc = projection * vmc;
        vdata = projection * vdata;
    }
    float poisson = BakerCousinsChi2(vmc, vdata,
        projection.size() ? nullptr : (masked ? &active_bins : nullptr),
        projection.size() ? 0 : (Eigen::Index)startBin);
    //float pull = Pull(subvector2);
    float value = poisson; //+ pull

    return value;
}

void PROpoisson::print([[maybe_unused]] const Eigen::VectorXf &param){
    return;
}
