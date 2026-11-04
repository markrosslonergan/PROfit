#include "PROcess.h"
#include "PROlog.h"
#include "PROspec.h"
#include "PROsyst.h"
#include "PROtocall.h"
#include "TH2D.h"

#include <Eigen/Eigen>

#include <algorithm>
#include <random>


namespace PROfit {

    PROspec FillSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const std::map<std::string, float> &inparams, bool binned, size_t var_index) {
        // default parameters
        Eigen::VectorXf params = Eigen::VectorXf::Zero(inmodel.nparams + insyst.GetNSplines());

        // default pulls are all 0. Set the default model parameters
        for (size_t ind = 0; ind < inmodel.nparams; ind++) params[ind] = inmodel.default_val[ind];

        // set parameters configured by user
        for (auto const &pair: inparams) {
            auto it1 = std::find(insyst.spline_names.begin(), insyst.spline_names.end(), pair.first);
            auto it2 = std::find(inmodel.param_names.begin(), inmodel.param_names.end(), pair.first);
            if (it1 != insyst.spline_names.end()) {
                int ind = std::distance(insyst.spline_names.begin(), it1);
                params[ind + inmodel.nparams] = pair.second;
            }
            else if (it2 != inmodel.param_names.end()) {
                int ind = std::distance(inmodel.param_names.begin(), it2);
                params[ind] = pair.second;
            }
            else {
                log<LOG_WARNING>(L"%1% | unable to find parameters %2% . Skipping.") % __func__ % pair.first.c_str();
            }
        }


        return FillSpectra(inconfig, inprop, insyst, inmodel, params, binned, var_index);
    }


    PROspec FillSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, FillSpectraCache &cache, bool binned, size_t var_index) {
        // Unbinned path can't be cleanly split phys/syst (per-event Fill mixes them).
        // Fall back, invalidate cache.
        if(!binned) {
            cache.invalidate();
            return FillSpectra(inconfig, inprop, insyst, inmodel, params, binned, var_index);
        }

        // Visible-decay models with pre-migration (flux) splines also can't be split
        // phys/syst: the flux weight enters the physics half at the parent truth-E,
        // BEFORE the decay migration, so systw is not a per-reco-bin factor.  Fall back
        // to the uncached path, which pre-scales N_truth (see below).
        if(inmodel.uses_get_counts()
           && insyst.spline_is_pre_migration.size() == insyst.GetNSplines()
           && std::find(insyst.spline_is_pre_migration.begin(),
                        insyst.spline_is_pre_migration.end(), true)
                  != insyst.spline_is_pre_migration.end()) {
            cache.invalidate();
            return FillSpectra(inconfig, inprop, insyst, inmodel, params, binned, var_index);
        }

        Eigen::VectorXf phys = params.segment(0, inmodel.nparams);
        Eigen::VectorXf shifts = params.segment(inmodel.nparams, params.size() - inmodel.nparams);

        const bool ctx_changed = (cache.last_var_index != (int)var_index ||
                                  cache.last_syst_ptr != &insyst ||
                                  cache.last_model_ptr != &inmodel);
        if(ctx_changed) {
            cache.phys_grid_valid = false;
            cache.unweighted_sums.clear();
        }

        // ---- Systematic-weights half (depends only on shifts) ----
        // Three branches for the systw vector:
        //   * full_systw_hit    : shifts identical to cached -> reuse cache.last_systw.
        //   * try_incremental   : shifts differs in EXACTLY one spline-range element
        //                          -> Tier 1.3: divide out old factor, multiply in new
        //                          factor for that one spline. Cache stays pinned to
        //                          its "central" state (no write-back).
        //   * full recompute    : everything else (size mismatch, ctx change, multi-diff,
        //                          or incremental gave a divide-by-near-zero on some bin).
        //                          Re-runs the full loop AND populates central_factors.
        const size_t       nbins_var      = inconfig.m_num_variable_bins_total[var_index];
        const Eigen::Index nsplines_e     = (Eigen::Index)insyst.GetNSplines();
        const bool         size_match     = (cache.last_shifts.size() == shifts.size());
        const bool         factors_valid  = (cache.central_factors.cols() == nsplines_e &&
                                             cache.central_factors.rows() == (Eigen::Index)nbins_var);
        const bool         diff_check_ok  = !ctx_changed && size_match && factors_valid;

        int diff_count = 0;
        int diff_idx   = -1;
        if(diff_check_ok) {
            for(Eigen::Index i = 0; i < shifts.size(); ++i) {
                if(cache.last_shifts(i) != shifts(i)) {
                    diff_idx = (int)i;
                    if(++diff_count > 1) break;
                }
            }
        }
        const bool full_systw_hit  = diff_check_ok && (diff_count == 0);
        const bool try_incremental = diff_check_ok && (diff_count == 1)
                                                   && (diff_idx >= 0)
                                                   && (diff_idx < (int)insyst.GetNSplines());

        Eigen::VectorXf       systw_local;            // populated only on the incremental path
        const Eigen::VectorXf *systw_to_use = nullptr; // points to whichever vector we'll combine

        if(full_systw_hit) {
            systw_to_use = &cache.last_systw;
        } else if(try_incremental) {
            const size_t j       = (size_t)diff_idx;
            const int    fc      = insyst.FactorColumn((int)j);   // j itself, or its spline_cross_quad group's column
            const size_t binning = insyst.spline_binnings[j];

            Eigen::VectorXf new_factor_j(nbins_var);
            if(binning == var_index) {
                for(size_t k = 0; k < nbins_var; ++k)
                    new_factor_j(k) = insyst.GetSplineFactor(fc, shifts, (int)k);
            } else {
                const size_t nbins_binning = inconfig.m_num_variable_bins_total[binning];
                Eigen::VectorXf spline_shifts_one(nbins_binning);
                for(size_t b = 0; b < nbins_binning; ++b)
                    spline_shifts_one(b) = insyst.GetSplineFactor(fc, shifts, (int)b);
                // Transpose-free migration GEMV; the (constant) column sums are
                // cached per binning instead of recomputed every call.
                Eigen::VectorXf weighted_sum = inprop.variable_hist_storage.WeightedColSum(binning, var_index, spline_shifts_one);
                auto it_us = cache.unweighted_sums.find(binning);
                if(it_us == cache.unweighted_sums.end())
                    it_us = cache.unweighted_sums.emplace(binning, inprop.variable_hist_storage.UnweightedColSum(binning, var_index)).first;
                const Eigen::VectorXf &unweighted_sum = it_us->second;
                for(size_t k = 0; k < nbins_var; ++k)
                    new_factor_j(k) = (unweighted_sum(k) > 0) ? weighted_sum(k) / unweighted_sum(k)
                                                              : 1.0f;
            }

            // Apply division+multiplication into a fresh local systw without touching cache.
            // If any cached factor for spline j is too small, fall back to full recompute.
            constexpr float kTiny = 1e-30f;
            systw_local.resize(nbins_var);
            bool ok = true;
            for(size_t k = 0; k < nbins_var; ++k) {
                const float old_f = cache.central_factors(k, (Eigen::Index)fc);
                if(std::abs(old_f) < kTiny) { ok = false; break; }
                systw_local(k) = cache.last_systw(k) / old_f * new_factor_j(k);
            }

            if(ok) {
                systw_to_use = &systw_local;
                // CRITICAL: do NOT update cache.last_shifts, cache.last_systw, or
                // cache.central_factors. Cache stays pinned to the central point so
                // subsequent single-shift gradient calls also hit the incremental path.
            }
            // else fall through to full recompute below
        }

        if(systw_to_use == nullptr) {
            // Full recompute: rebuild systw AND populate per-spline factors. The
            // existing math is preserved bit-for-bit; we just stash each spline's
            // contribution as we go.
            Eigen::VectorXf systw = Eigen::VectorXf::Constant(nbins_var, 1);
            Eigen::MatrixXf factors(nbins_var, insyst.GetNSplines());
            for(int i = 0; i < (int)insyst.GetNSplines(); ++i) {
                size_t binning = insyst.spline_binnings[i];
                if(binning == var_index) {
                    for(size_t k = 0; k < nbins_var; ++k) {
                        const float f = insyst.GetSplineFactor(i, shifts, (int)k);
                        factors(k, i) = f;
                        systw(k) *= f;
                    }
                } else {
                    const size_t nbins_binning = inconfig.m_num_variable_bins_total[binning];
                    Eigen::VectorXf spline_shifts_loc(nbins_binning);
                    for(size_t b = 0; b < nbins_binning; ++b)
                        spline_shifts_loc(b) = insyst.GetSplineFactor(i, shifts, (int)b);
                    Eigen::VectorXf weighted_sum = inprop.variable_hist_storage.WeightedColSum(binning, var_index, spline_shifts_loc);
                    auto it_us = cache.unweighted_sums.find(binning);
                    if(it_us == cache.unweighted_sums.end())
                        it_us = cache.unweighted_sums.emplace(binning, inprop.variable_hist_storage.UnweightedColSum(binning, var_index)).first;
                    const Eigen::VectorXf &unweighted_sum = it_us->second;
                    for(size_t k = 0; k < nbins_var; ++k) {
                        const float f = (unweighted_sum(k) > 0) ? weighted_sum(k) / unweighted_sum(k)
                                                                : 1.0f;
                        factors(k, i) = f;
                        systw(k) *= f;
                    }
                }
            }
            cache.last_systw      = std::move(systw);
            cache.last_shifts     = shifts;
            cache.central_factors = std::move(factors);
            systw_to_use          = &cache.last_systw;
        }

        // ---- Physics-result half (depends only on phys) ----
        const bool phys_changed = ctx_changed ||
                                  cache.last_phys.size() != phys.size() ||
                                  cache.last_phys != phys;
        if(phys_changed) {
            Eigen::VectorXf result;
            if(inmodel.is_trivial) {
                result = inmodel.H_combined[var_index].col(0);
            } else {
                // Flat physics grid: constant across the whole fit (depends only
                // on the model's ivars and the propagator's midbins) — build it
                // once and reuse from the cache.
                if(!cache.phys_grid_valid) {
                    const size_t N_ivars = inmodel.ivars.size();
                    std::vector<size_t> ivar_sizes(N_ivars);
                    for(size_t k = 0; k < N_ivars; ++k)
                        ivar_sizes[k] = inprop.variable_midbin[inmodel.ivars[k]].size();

                    cache.phys_grid.assign(N_ivars, std::vector<float>(inmodel.n_phys_bins));
                    for(long int flat = 0; flat < inmodel.n_phys_bins; ++flat) {
                        long int rem = flat;
                        for(int k = (int)N_ivars - 1; k >= 0; --k) {
                            cache.phys_grid[k][flat] = inprop.variable_midbin[inmodel.ivars[k]][rem % ivar_sizes[k]];
                            rem /= (long int)ivar_sizes[k];
                        }
                    }
                    cache.phys_grid_valid = true;
                }

                auto probs = inmodel.get_probs(phys, cache.phys_grid);
                Eigen::Map<const Eigen::VectorXf> probs_flat(probs.data(), probs.size());
                result = inmodel.H_combined[var_index] * probs_flat;
            }
            cache.last_result = std::move(result);
            cache.last_phys = phys;
        }

        cache.last_var_index = (int)var_index;
        cache.last_syst_ptr = &insyst;
        cache.last_model_ptr = &inmodel;

        // systw_to_use points to either cache.last_systw (full hit / full recompute) or
        // to the local perturbed vector built by the Tier 1.3 incremental path.
        // The chi² metrics — the only users of this cached overload — never read
        // the error vector, so skip the per-call abs+sqrt and return zero errors.
        Eigen::VectorXf final_spec = systw_to_use->cwiseProduct(cache.last_result);
        return PROspec(final_spec, Eigen::VectorXf::Zero(final_spec.size()));
    }


    Eigen::MatrixXf FillSpectraGradient(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, FillSpectraCache &cache, size_t var_index) {
        const size_t nbins_var = inconfig.m_num_variable_bins_total[var_index];
        const size_t nsplines  = insyst.GetNSplines();
        const size_t nphys     = inmodel.nparams;
        Eigen::VectorXf phys   = params.segment(0, nphys);
        Eigen::VectorXf shifts = params.segment(nphys, params.size() - nphys);

        // ---- Per-spline factors f_i(k) and derivatives f'_i(k), same math as FillSpectra ----
        Eigen::MatrixXf factors  = Eigen::MatrixXf::Ones(nbins_var, nsplines);
        Eigen::MatrixXf dfactors = Eigen::MatrixXf::Zero(nbins_var, nsplines);
        Eigen::VectorXf systw    = Eigen::VectorXf::Constant(nbins_var, 1);
        for(int i = 0; i < (int)nsplines; ++i) {
            size_t binning = insyst.spline_binnings[i];
            if(binning == var_index) {
                for(size_t k = 0; k < nbins_var; ++k) {
                    factors(k, i)  = insyst.GetSplineFactor(i, shifts, (int)k);
                    dfactors(k, i) = insyst.GetSplineFactorDeriv(i, shifts, (int)k);
                }
            } else {
                const size_t nbins_binning = inconfig.m_num_variable_bins_total[binning];
                Eigen::VectorXf spline_vals(nbins_binning), spline_derivs(nbins_binning);
                for(size_t b = 0; b < nbins_binning; ++b) {
                    spline_vals(b)   = insyst.GetSplineFactor(i, shifts, (int)b);
                    spline_derivs(b) = insyst.GetSplineFactorDeriv(i, shifts, (int)b);
                }
                // The migration factor is LINEAR in the spline values, so its derivative
                // is the same GEMV applied to the per-bin spline derivatives.
                Eigen::VectorXf weighted  = inprop.variable_hist_storage.WeightedColSum(binning, var_index, spline_vals);
                Eigen::VectorXf dweighted = inprop.variable_hist_storage.WeightedColSum(binning, var_index, spline_derivs);
                auto it_us = cache.unweighted_sums.find(binning);
                if(it_us == cache.unweighted_sums.end())
                    it_us = cache.unweighted_sums.emplace(binning, inprop.variable_hist_storage.UnweightedColSum(binning, var_index)).first;
                const Eigen::VectorXf &unweighted = it_us->second;
                for(size_t k = 0; k < nbins_var; ++k) {
                    if(unweighted(k) > 0) {
                        factors(k, i)  = weighted(k)  / unweighted(k);
                        dfactors(k, i) = dweighted(k) / unweighted(k);
                    }
                }
            }
            systw.array() *= factors.col(i).array();
        }

        // ---- Physics result (H * probs) and per-parameter probability derivatives ----
        Eigen::VectorXf result;
        std::vector<Eigen::MatrixXf> pgrads;
        if(inmodel.is_trivial) {
            result = inmodel.H_combined[var_index].col(0);
        } else {
            // Reuse (or build) the constant flat physics grid exactly as FillSpectra does.
            if(!cache.phys_grid_valid) {
                const size_t N_ivars = inmodel.ivars.size();
                std::vector<size_t> ivar_sizes(N_ivars);
                for(size_t k = 0; k < N_ivars; ++k)
                    ivar_sizes[k] = inprop.variable_midbin[inmodel.ivars[k]].size();
                cache.phys_grid.assign(N_ivars, std::vector<float>(inmodel.n_phys_bins));
                for(long int flat = 0; flat < inmodel.n_phys_bins; ++flat) {
                    long int rem = flat;
                    for(int k = (int)N_ivars - 1; k >= 0; --k) {
                        cache.phys_grid[k][flat] = inprop.variable_midbin[inmodel.ivars[k]][rem % ivar_sizes[k]];
                        rem /= (long int)ivar_sizes[k];
                    }
                }
                cache.phys_grid_valid = true;
            }
            auto probs = inmodel.get_probs(phys, cache.phys_grid);
            Eigen::Map<const Eigen::VectorXf> probs_flat(probs.data(), probs.size());
            result = inmodel.H_combined[var_index] * probs_flat;
            pgrads = inmodel.get_probs_grad(phys, cache.phys_grid);
        }

        // ---- Assemble d(spec)/d(param) columns; spec = systw .* result ----
        Eigen::MatrixXf G = Eigen::MatrixXf::Zero(nbins_var, nphys + nsplines);
        if(!inmodel.is_trivial) {
            for(size_t p = 0; p < nphys; ++p) {
                Eigen::Map<const Eigen::VectorXf> dprobs_flat(pgrads[p].data(), pgrads[p].size());
                G.col(p) = systw.cwiseProduct(inmodel.H_combined[var_index] * dprobs_flat);
            }
        }
        constexpr float kTiny = 1e-30f;
        for(size_t i = 0; i < nsplines; ++i) {
            // d(systw)/d(shift_i) = (product of every column but the one shift_i lives in) * d(that column);
            // for a spline_cross_quad member that column is the group's, not its own.
            const size_t fc = (size_t)insyst.FactorColumn((int)i);
            for(size_t k = 0; k < nbins_var; ++k) {
                const float f = factors(k, fc);
                float excl; // product of all OTHER columns' factors at bin k
                if(std::abs(f) > kTiny) {
                    excl = systw(k) / f;
                } else {
                    excl = 1.0f;
                    for(size_t j = 0; j < nsplines; ++j)
                        if(j != fc) excl *= factors(k, j);
                }
                G(k, nphys + i) = excl * dfactors(k, i) * result(k);
            }
        }
        return G;
    }

    PROspec FillSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, bool binned, size_t var_index){
        PROspec myspectrum(inconfig.m_num_variable_bins_total[var_index]);
        Eigen::VectorXf phys   = params.segment(0, inmodel.nparams);
        Eigen::VectorXf shifts = params.segment(inmodel.nparams, params.size() - inmodel.nparams);

        if(binned) {
            //log<LOG_INFO>(L"%1% || Starting systw calculation %2%") % __func__ % var_index;
            //auto start_systw = std::chrono::high_resolution_clock::now();

            const size_t nbins_var = inconfig.m_num_variable_bins_total[var_index];

            // Whether spline_is_pre_migration has been populated (requires the full-syst constructor).
            // If not populated, treat all splines as post-migration (backward-compatible behaviour).
            const bool has_pre_mig_info = (insyst.spline_is_pre_migration.size() == insyst.GetNSplines());

            // Only visible-decay models (which override get_counts) have truth-E migration.
            // For non-decay models, all splines are applied at reco level as before.
            const bool model_has_migration = inmodel.uses_get_counts();

            // pre_mig_weight: per flat physics grid point, flux reweighting applied BEFORE the
            // decay energy migration (used to pre-scale N_truth for get_counts).  Length = n_phys_bins.
            // n_E (last phys_grid_sizes entry) = number of truth-E bins; E is the last (fastest-
            // varying) ivar, so E_idx = flat % n_E.
            Eigen::VectorXf pre_mig_weight = Eigen::VectorXf::Constant(inmodel.n_phys_bins, 1.0f);

            // post_mig_systw: per reco bin, cross-section/detector systematic weight applied
            // AFTER the migration.  Replaces the old monolithic systw.
            Eigen::VectorXf post_mig_systw = Eigen::VectorXf::Constant(nbins_var, 1.0f);

            // Iterate up to insyst.GetNSplines(), not shifts.size(): params may be
            // over-sized when shared across variables with different spline counts.
            for(int i = 0; i < (int)insyst.GetNSplines(); ++i) {
                size_t binning = insyst.spline_binnings[i];

                bool is_pre_migration = has_pre_mig_info && model_has_migration
                                        && insyst.spline_is_pre_migration[i];

                if(is_pre_migration) {
                    // Pre-migration flux spline: apply to the flat (L, E) physics grid.
                    // The spline must be binned in the truth-E variable (the last ivar);
                    // n_spline_bins must equal n_E (the number of truth-E bins).
                    const size_t n_E           = inmodel.phys_grid_sizes.back();
                    const size_t n_spline_bins = inconfig.m_num_variable_bins_total[binning];
                    if(n_spline_bins != n_E) {
                        log<LOG_ERROR>(L"%1% || Pre-migration flux spline '%2%' has %3% bins but "
                            L"the model has %4% truth-E bins. The spline must be binned in the "
                            L"truth-E variable (the last model ivar).") % __func__
                            % insyst.spline_names[i].c_str() % n_spline_bins % n_E;
                        exit(EXIT_FAILURE);
                    }
                    // Each flat bin inherits the flux weight from its truth-E index.
                    for(long int flat = 0; flat < inmodel.n_phys_bins; ++flat) {
                        size_t e_idx = flat % n_E;
                        pre_mig_weight(flat) *= insyst.GetSplineFactor(i, shifts, e_idx);
                    }
                } else if(binning == var_index) {
                    // Post-migration, same binning — direct multiplication on reco bins.
                    for(size_t k = 0; k < nbins_var; ++k) {
                        post_mig_systw(k) *= insyst.GetSplineFactor(i, shifts, k);
                    }
                } else {
                    // Post-migration, different binning — use matrix-vector multiplication.
                    const size_t nbins_binning = inconfig.m_num_variable_bins_total[binning];

                    Eigen::VectorXf spline_shifts(nbins_binning);
                    for(size_t j = 0; j < nbins_binning; ++j) {
                        spline_shifts(j) = insyst.GetSplineFactor(i, shifts, j);
                    }

                    // Compute weighted and unweighted sums transpose-free:
                    // weighted_sum[k] = sum_j(spline_shifts[j] * hist(j, k))
                    // unweighted_sum[k] = sum_j(hist(j, k))
                    Eigen::VectorXf weighted_sum = inprop.variable_hist_storage.WeightedColSum(binning, var_index, spline_shifts);
                    Eigen::VectorXf unweighted_sum = inprop.variable_hist_storage.UnweightedColSum(binning, var_index);

                    // Apply the ratio where unweighted > 0
                    for(size_t k = 0; k < nbins_var; ++k) {
                        if(unweighted_sum(k) > 0) {
                            post_mig_systw(k) *= weighted_sum(k) / unweighted_sum(k);
                        }
                    }
                }
            }

            //auto end_systw = std::chrono::high_resolution_clock::now();
            //std::chrono::duration<double> duration_systw = end_systw - start_systw;
            //log<LOG_INFO>(L"%1% || systw calculation took %2% ms") % __func__ % (duration_systw.count() * 1000.);

            //log<LOG_INFO>(L"%1% || Starting le_arr building %2%") % __func__ % var_index;
            //auto start_le = std::chrono::high_resolution_clock::now();

            Eigen::VectorXf result;
            if(inmodel.is_trivial) {
                // Trivial model: probs ≡ 1. H_combined[var_index] has shape (n_reco, 1) and column 0
                // already contains the per-reco-bin event-weight sum, with no physics-grid coupling.
                result = inmodel.H_combined[var_index].col(0);
            } else {
                // Build var_arrs: one entry per ivar, each of length n_phys_bins (flat grid).
                // For 1-var: var_arrs[0] = midbin values of that var (n_phys_bins = n_ivar_bins).
                // For N-var: row-major product grid — var_arrs[k][flat] = midbin of ivar[k] at flat index.
                const size_t N_ivars = inmodel.ivars.size();
                std::vector<size_t> ivar_sizes(N_ivars);
                for(size_t k = 0; k < N_ivars; ++k)
                    ivar_sizes[k] = inprop.variable_midbin[inmodel.ivars[k]].size();

                std::vector<std::vector<float>> var_arrs(N_ivars, std::vector<float>(inmodel.n_phys_bins));
                for(long int flat = 0; flat < inmodel.n_phys_bins; ++flat) {
                    long int rem = flat;
                    for(int k = (int)N_ivars - 1; k >= 0; --k) {
                        var_arrs[k][flat] = inprop.variable_midbin[inmodel.ivars[k]][rem % ivar_sizes[k]];
                        rem /= (long int)ivar_sizes[k];
                    }
                }

                // Effective per-physics-bin oscillation probabilities.  Two paths:
                //   - Visible-decay models (uses_get_counts): pre-scale N_truth rows by
                //     pre_mig_weight so the flux reweight is applied at the PARENT neutrino's
                //     truth energy, then call get_counts and divide by the ORIGINAL N_truth.
                //     Pre-scaling N_truth weights the parent in both the oscillation term (A)
                //     at flat_dst and the decay-migration term (B) at each flat_src, since
                //     get_counts reads the passed N_truth at exactly those bins.
                //   - All other models: the simple get_probs path.  pre_mig_weight is always
                //     all-ones here (model_has_migration == false), so no flux reweight is lost.
                Eigen::MatrixXf probs;
                if(inmodel.uses_get_counts()) {
                    const Eigen::MatrixXf& N_truth = inmodel.get_N_truth();
                    Eigen::MatrixXf N_weighted = N_truth.array().colwise() * pre_mig_weight.array();
                    auto counts = inmodel.get_counts(phys, var_arrs, N_weighted);
                    auto N_arr  = N_truth.array();
                    probs = (N_arr > 0.0f).select(counts.array() / N_arr, 0.0f);
                } else {
                    probs = inmodel.get_probs(phys, var_arrs);
                }

                // Single GEMV: H_combined[var_index] has shape (n_reco, n_phys*J).
                // probs is (n_phys, J) in column-major, so probs.data() = [col0 | col1 | ...] = probs_flat.
                Eigen::Map<const Eigen::VectorXf> probs_flat(probs.data(), probs.size());
                result = inmodel.H_combined[var_index] * probs_flat;
            }

            // Apply post-migration systematic weights (cross-section, detector) at reco level.
            Eigen::VectorXf final_spec = post_mig_systw.cwiseProduct(result);
            Eigen::VectorXf final_error = final_spec.array().abs().sqrt();
            myspectrum = PROspec(final_spec, final_error);

        } else {

            // Unbinned path: per-event var values, one vector per ivar.
            // For trivial models (empty ivars), skip var_arrs / get_probs entirely and use oscw = 1.
            Eigen::MatrixXf probs;
            if(!inmodel.is_trivial) {
                std::vector<std::vector<float>> var_arrs(inmodel.ivars.size(), std::vector<float>(inprop.NEvent()));
                for(size_t k = 0; k < inmodel.ivars.size(); ++k)
                    for(size_t i = 0; i < inprop.NEvent(); ++i)
                        var_arrs[k][i] = inprop.VariableValue(inmodel.ivars[k], i);

                probs = inmodel.get_probs(phys, var_arrs);
            }

            for(size_t i = 0; i < inprop.NEvent(); ++i) {
                // Out-of-range events have bin index -1; PROspec::Fill does no
                // bounds checking, so filling would corrupt memory.
                const int reco_bin = inprop.VariableBinIndex(var_index, i);
                if(reco_bin < 0) continue;

                float oscw = inmodel.is_trivial ? 1.0f : probs(i, inprop.model_rule[i]);
                float add_w = inprop.added_weights[i];

                float systw = 1;
                // Iterate up to insyst.GetNSplines(), not shifts.size(): params
                // may be over-sized when shared across variables with different
                // spline counts (the binned path above already does this).
                for(int j = 0; j < (int)insyst.GetNSplines(); ++j) {
                    int binning = insyst.spline_binnings[j];
                    const int spline_bin = inprop.VariableBinIndex(binning, i);
                    if(spline_bin < 0) continue; // outside this spline's binning: no shift
                    systw *= insyst.GetSplineFactor(j, shifts, spline_bin);
                }
                float finalw = oscw * systw * add_w;
                myspectrum.Fill(reco_bin, finalw);
            }
        }
        return myspectrum;
    }

    PROspec FillSystRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &model, const PROspec &cvspec, const Eigen::VectorXf &cvparams, uint32_t seed, int var_index) {
        int nbins = inconfig.m_num_variable_bins_total[var_index],
        nbins_collapsed = inconfig.m_num_variable_bins_total_collapsed[var_index];
        Eigen::VectorXf spec = Eigen::VectorXf::Constant(nbins, 0);

        Eigen::VectorXf params = cvparams;

        // Local generator seeded per call: a function-local static ignored the
        // seed argument after the first-ever call and raced across threads.
        // Callers are responsible for passing distinct seeds per throw.
        std::mt19937 rng{seed};
        std::vector<std::normal_distribution<float>> d_spline;
        for(size_t i = 0; i < insyst.GetNSplines(); ++i)
            d_spline.emplace_back(insyst.spline_centers(i), insyst.spline_priors(i));
        std::normal_distribution<float> d_cov;
        std::vector<float> throws;
        //Eigen::VectorXf throwC = Eigen::VectorXf::Constant(inconfig.m_num_variable_bins_total[inconfig.i_prime], 0);
        Eigen::VectorXf throwC = Eigen::VectorXf::Constant(nbins_collapsed, 0);
        for(size_t i = 0; i < insyst.GetNSplines(); i++) {
            throws.push_back(d_spline[i](rng));
        }
        for(int i = 0; i < nbins_collapsed; i++)
            throwC(i) = d_cov(rng);

        //get actual splines 
        for(size_t i=0;i<throws.size();i++){
            params(i+model.nparams) = throws.at(i);
        }

        bool binned = true;//dont want to faf around with event by event here lets be honst
        if (binned){
          spec = FillSpectra(inconfig, inprop, insyst, model, params, binned, var_index).Spec();

        }else{//currently never run
            const Eigen::Map<const Eigen::VectorXf> throw_vec(throws.data(), throws.size());
            for(size_t i = 0; i<inprop.NEvent(); ++i){
                float add_w = inprop.added_weights[i]; 
                float systw = 1;
                for(size_t j = 0; j < throws.size(); ++j) {
                    int binning = insyst.spline_binnings[j];
                    const int spline_bin = inprop.VariableBinIndex(binning, i);
                    systw *= insyst.GetSplineFactor(j, throw_vec, spline_bin);
                }
                if(inprop.VariableBinIndex(var_index, i) >= 0) {
                    float finalw = systw * add_w;
                    spec(inprop.VariableBinIndex(var_index, i)) += finalw;
                }
            }
        }

        if(insyst.GetNCovar() == 0) {
            Eigen::VectorXf final_spec = CollapseMatrix(inconfig, spec, var_index);
            return PROspec(final_spec, final_spec.array().sqrt());
        }

        Eigen::MatrixXf decomp_cov = insyst.DecomposeFractionalCovariance(inconfig, cvspec.Spec());
        Eigen::VectorXf collapsed_spec = CollapseMatrix(inconfig, spec, var_index);
        Eigen::VectorXf final_spec = collapsed_spec + decomp_cov * throwC;

        //std::vector<float> stdVec(final_spec.data(), final_spec.data() + final_spec.size());
        //log<LOG_INFO>(L"%1% | final_spec is %2% ") % __func__ % stdVec;


        return PROspec(final_spec, final_spec.array().sqrt());
    }

    std::pair<PROspec, PROspec> FillSystRandomThrowSplit(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &model, const PROspec &cvspec, const Eigen::VectorXf &cvparams, uint32_t seed, int var_index, const Eigen::VectorXf &bkg_bin_mask) {
        int nbins = inconfig.m_num_variable_bins_total[var_index];

        Eigen::VectorXf params = cvparams;

        // Local generator seeded per call, as in FillSystRandomThrow.
        std::mt19937 rng{seed};
        std::vector<std::normal_distribution<float>> d_spline;
        for(size_t i = 0; i < insyst.GetNSplines(); ++i)
            d_spline.emplace_back(insyst.spline_centers(i), insyst.spline_priors(i));
        std::normal_distribution<float> d_cov;
        std::vector<float> throws;
        for(size_t i = 0; i < insyst.GetNSplines(); i++) {
            throws.push_back(d_spline[i](rng));
        }
        // Covariance throw in FULL bin space (nbins normals rather than
        // nbins_collapsed) so the bkg subchannels' variation is separable.
        Eigen::VectorXf throwF = Eigen::VectorXf::Constant(nbins, 0);
        for(int i = 0; i < nbins; i++)
            throwF(i) = d_cov(rng);

        for(size_t i = 0; i < throws.size(); i++) {
            params(i + model.nparams) = throws.at(i);
        }

        Eigen::VectorXf full = FillSpectra(inconfig, inprop, insyst, model, params, true, var_index).Spec();
        if(insyst.GetNCovar() != 0)
            full += insyst.DecomposeFractionalCovarianceFull(inconfig, cvspec.Spec()) * throwF;

        Eigen::VectorXf bkg_full = full.cwiseProduct(bkg_bin_mask);
        Eigen::VectorXf sig_full = full - bkg_full;

        Eigen::VectorXf sig_collapsed = CollapseMatrix(inconfig, sig_full, var_index);
        Eigen::VectorXf bkg_collapsed = CollapseMatrix(inconfig, bkg_full, var_index);

        // abs() before sqrt: covariance throws can drive bins negative.
        return {PROspec(sig_collapsed, sig_collapsed.array().abs().sqrt()),
                PROspec(bkg_collapsed, bkg_collapsed.array().abs().sqrt())};
    }

    PROspec FillSplineRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst,  const PROmodel &model,  const Eigen::VectorXf &cvparams, int spline, uint32_t seed, int other_index) {
        int nbins =  inconfig.m_num_variable_bins_total[other_index];
        Eigen::VectorXf spec = Eigen::VectorXf::Constant(nbins, 0);

        // Local generator seeded per call: a function-local static ignored the
        // seed argument after the first-ever call and raced across threads.
        // Callers are responsible for passing distinct seeds per throw.
        std::mt19937 rng{seed};
        std::normal_distribution<float> d(insyst.spline_centers(spline), insyst.spline_priors(spline));
        float spline_throw = d(rng);
        Eigen::VectorXf params = cvparams;
        params(spline+model.nparams) = spline_throw;

        bool binned = true;//dont want to faf around with event by event here lets be honst
        if (binned){
            spec = FillSpectra(inconfig, inprop, insyst, model, params, binned, other_index).Spec();
        }

        return PROspec(spec, spec.array().sqrt());
    }

    std::optional<size_t> ResolveParamIndex(const std::string &name, const PROmodel &model, const PROsyst &systs, const PROconfig &config) {
        if(auto loc = std::find(model.param_names.begin(), model.param_names.end(), name); loc != model.param_names.end())
            return std::distance(model.param_names.begin(), loc);
        if(auto loc = std::find(systs.spline_names.begin(), systs.spline_names.end(), name); loc != systs.spline_names.end())
            return model.nparams + std::distance(systs.spline_names.begin(), loc);
        for(const auto &[xml_name, plot_name] : config.m_mcgen_variation_plotname_map) {
            if(name != plot_name) continue;
            if(auto loc = std::find(systs.spline_names.begin(), systs.spline_names.end(), xml_name); loc != systs.spline_names.end())
                return model.nparams + std::distance(systs.spline_names.begin(), loc);
        }
        return std::nullopt;
    }

    float ThrowRestrictedSplinePull(const PROsyst &insyst, size_t i, std::mt19937 &rng, std::normal_distribution<float> &d) {
        const bool has_r = i < insyst.spline_has_restrict.size() && insyst.spline_has_restrict[i];
        float tlo = has_r ? insyst.spline_restrict_lo[i] : insyst.spline_lo[i];
        float thi = has_r ? insyst.spline_restrict_hi[i] : insyst.spline_hi[i];
        if (tlo > thi) { const float t = tlo; tlo = thi; thi = t; }
        // A --fix'd spline has a zero-width range: it throws at its pinned value.
        if (tlo == thi) return tlo;

        // prior_type="uniform" has no Gaussian prior: throw flat over its allowed range.
        if (i < insyst.spline_prior_types.size() && insyst.spline_prior_types[i] == SplinePriorType::Uniform)
            return std::uniform_real_distribution<float>(tlo, thi)(rng);

        // Sample the spline's ACTUAL prior N(center, sigma), not a hardcoded N(0,1):
        // center/sigma default to 0/1 (identical behaviour for legacy configs) but honor
        // XML-configured priors and PROjector's constrained posterior. Note only the
        // MARGINAL width is sampled here — prior correlations (XML correlations or a
        // PROjector external prior covariance) are not reflected in per-spline throws.
        const float mu  = i < (size_t)insyst.spline_centers.size() ? insyst.spline_centers((Eigen::Index)i) : 0.0f;
        const float sig = i < (size_t)insyst.spline_priors.size() ? std::abs(insyst.spline_priors((Eigen::Index)i)) : 1.0f;

        const int max_attempts = 10000;
        int attempts = 0;
        float x = mu + sig * d(rng);
        while ((x < tlo || x > thi) && ++attempts < max_attempts)
            x = mu + sig * d(rng);
        if (x < tlo || x > thi) {
            x = (mu < tlo) ? tlo : (mu > thi ? thi : mu);
            const std::string sname = i < insyst.spline_names.size()
                ? insyst.spline_names[i] : ("spline#" + std::to_string(i));
            log<LOG_WARNING>(L"%1% || spline '%2%' (index %3%) has throw bounds [%4%, %5%] unreachable "
                             L"by its N(%6%, %7%) prior after %8% draws; clamping to %9%. Check its knobvals / restrict attribute.")
                % __func__ % sname.c_str() % (int)i % tlo % thi % mu % sig % max_attempts % x;
        }
        return x;
    }

    Eigen::MatrixXf CovarianceThrowRoot(const PROconfig &inconfig, const PROsyst &insyst) {
        const Eigen::Index n = insyst.fractional_covariance.rows();
        if(insyst.GetNCovar() == 0 || n == 0) return Eigen::MatrixXf(n, 0);
        // sqrt(diag(1) F diag(1)) = sqrt(F). Kept modes fill the leading columns.
        const Eigen::MatrixXf R = insyst.DecomposeFractionalCovarianceFull(inconfig, Eigen::VectorXf::Ones(n));
        Eigen::Index kept = 0;
        while(kept < R.cols() && R.col(kept).squaredNorm() > 0.0f) ++kept;
        return R.leftCols(kept);
    }

    Eigen::VectorXf ThrowCovarianceShift(const PROconfig &inconfig, const Eigen::MatrixXf &frac_root, const Eigen::VectorXf &spec_full, std::mt19937 &rng, std::normal_distribution<float> &d, int var_index) {
        if(frac_root.cols() == 0)
            return Eigen::VectorXf::Zero(inconfig.m_num_variable_bins_total_collapsed[var_index]);
        Eigen::VectorXf alpha(frac_root.cols());
        for(Eigen::Index i = 0; i < alpha.size(); ++i) alpha(i) = d(rng);
        return CollapseMatrix(inconfig, Eigen::VectorXf(spec_full.cwiseProduct(frac_root * alpha)), var_index);
    }
};
