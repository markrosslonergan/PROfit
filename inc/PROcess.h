/**
 * @file PROcess.h
 * @brief Spectrum-filling functions combining MC events, systematics, and physics models.
 * @author PROfit Collaboration
 *
 * @details Declares the family of FillSpectra functions that are the primary entry points
 * for computing a predicted event spectrum from Monte Carlo events stored in a PROpeller,
 * applying oscillation weights from a PROmodel, and applying systematic spline weights
 * from a PROsyst.  Both event-by-event and pre-binned modes are supported.
 */
#ifndef PROCESS_H
#define PROCESS_H

#include <Eigen/Eigen>
#include <cstdint>
#include <map>
#include <random>

// PROfit include 
#include "PROconfig.h"
#include "PROmodel.h"
#include "PROpeller.h"
#include "PROspec.h"
#include "PROsyst.h"

#include "TH2D.h"
#include <vector>

namespace PROfit{

    /**
     * @brief Single-slot cache for the binned FillSpectra path, splitting the work into
     *        independently keyed physics and systematic halves.
     * @details The binned FillSpectra factors as `final_spec = systw .* result` where
     *   - `result` (physics half) depends only on @p phys, @p inmodel, @p inprop, and @p var_index;
     *   - `systw`  (syst half)    depends only on @p shifts, @p insyst, @p inprop, and @p var_index.
     *
     * On each call the cached overload compares the new phys/shifts subvectors against the cached
     * keys and recomputes only the half that changed. The cache is invalidated when the model or
     * syst pointer or var_index changes (via `invalidate()` or context check). The cache is NOT
     * thread-safe; each thread should hold its own (PROfile/PROsurf already Clone() metrics per
     * thread, so per-metric caches are naturally per-thread).
     *
     * @note Only the binned FillSpectra path is cached. The unbinned (event-by-event) path
     *       invalidates the cache and falls back to the non-cached overload.
     */
    struct FillSpectraCache {
        Eigen::VectorXf last_phys;            ///< Cached physics subvector for last_result.
        Eigen::VectorXf last_shifts;          ///< Cached spline subvector for last_systw.
        Eigen::VectorXf last_result;          ///< Cached H_combined * probs (or trivial-model result).
        Eigen::VectorXf last_systw;           ///< Cached per-bin systematic-weight vector.

        /// Per-spline factors at the cached "central" point, shape (nbins_var, nsplines).
        /// central_factors(k, i) = spline i's multiplicative contribution to systw at bin k.
        /// systw_central(k) = product over i of central_factors(k, i). Used by the Tier 1.3
        /// incremental update path (single-spline-shift gradient calls) so we can divide
        /// out the old contribution and multiply in the new one without rerunning the full
        /// spline loop. Empty/zero-sized when the cache has not yet been populated by a
        /// full recompute.
        Eigen::MatrixXf central_factors;

        int last_var_index = -1;              ///< var_index for which the cache is valid.
        const PROsyst *last_syst_ptr = nullptr;
        const PROmodel *last_model_ptr = nullptr;

        /// Flat physics grid passed to get_probs — depends only on the model's
        /// ivars and the propagator's midbin vectors, i.e. constant across an
        /// entire fit. Built once on first use and reused (it used to be
        /// reallocated and refilled on every physics-changed call).
        std::vector<std::vector<float>> phys_grid;
        bool phys_grid_valid = false;

        /// Per-cross-binning column sums of the migration histograms
        /// (constant per (binning, var_index)); keyed by binning index.
        std::map<size_t, Eigen::VectorXf> unweighted_sums;

        /// Mark cache contents stale; next call recomputes both halves.
        void invalidate() {
            last_var_index = -1;
            last_syst_ptr = nullptr;
            last_model_ptr = nullptr;
            phys_grid_valid = false;
            unweighted_sums.clear();
        }
    };

    /**
     * @brief Cached overload of the binned FillSpectra. See FillSpectraCache.
     * @param cache  Single-slot cache; reuses physics or systematic half whose subvector matches.
     * @return Same value as the non-cached FillSpectra; cache is updated as a side effect.
     */
    PROspec FillSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, FillSpectraCache &cache, bool binned, size_t var_index);

    /**
     * @brief Analytic Jacobian of the BINNED FillSpectra spectrum with respect to all parameters.
     * @details Column j holds d(spec)/d(param_j) in full (uncollapsed) bin space, evaluated
     * at @p params, for j = 0..nphys-1 (physics, via PROmodel::get_probs_grad) then
     * j = nphys..nphys+nsplines-1 (spline nuisances, via PROsyst::GetSplineShiftDeriv).
     * Self-contained: recomputes the per-spline factors and physics result it needs rather
     * than trusting the cache's pinned state (the Tier 1.3 incremental path deliberately
     * leaves the cache at an older central point). Only the constant cache members
     * (unweighted_sums, phys_grid) are reused/populated.
     * @return Matrix (nbins_var_full, nphys + nsplines).
     */
    Eigen::MatrixXf FillSpectraGradient(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, FillSpectraCache &cache, size_t var_index = 0);

    /**
     * @brief Master spectrum-filling function combining oscillation weights and systematic spline weights.
     * @details Iterates over MC events (or uses pre-binned histograms when @p binned is true),
     * computes oscillation probabilities via @p inmodel, applies spline shifts from @p insyst,
     * and accumulates a predicted spectrum for the analysis variable indexed by @p var_index.
     * @param inconfig   Configuration object describing binning, channels, and subchannels.
     * @param inprop     MC event store (PROpeller).
     * @param insyst     Systematic object holding spline and covariance information.
     * @param inmodel    Physics model providing oscillation probability weights.
     * @param params     Combined parameter vector: physics parameters followed by spline nuisance parameters.
     * @param binned     If true (default), use pre-binned mode (fast); if false, iterate event-by-event.
     * @param var_index  Index of the analysis variable to fill (default 0, i.e. the primary reco variable).
     * @return A PROspec containing the predicted event spectrum.
     */
    PROspec FillSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, bool binned, size_t var_index);

    /**
     * @brief Overload of FillSpectra accepting systematic pulls as a name-to-value map.
     * @details Converts the named pull map to an ordered parameter vector and delegates to the
     * primary FillSpectra overload.
     * @param inconfig   Configuration object.
     * @param inprop     MC event store.
     * @param insyst     Systematic object.
     * @param inmodel    Physics model.
     * @param pulls      Map from spline systematic name to pull value (in units of sigma).
     * @param binned     If true (default), use pre-binned mode.
     * @param var_index  Index of the analysis variable to fill (default 0).
     * @return A PROspec containing the predicted event spectrum.
     */
    PROspec FillSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const std::map<std::string, float> &pulls, bool binned, size_t var_index);


    //Below are depreciated, slightly

    /**
     * @brief Generate a spectrum with a single random systematic throw applied.
     * @details Draws a random Gaussian universe for all systematics and fills a spectrum for
     * use in covariance-matrix estimation or toy-MC studies.
     * @param inconfig   Configuration object.
     * @param inprop     MC event store.
     * @param insyst     Systematic object.
     * @param model      Physics model.
     * @param cvspec     Central-value spectrum (used as denominator for fractional shifts).
     * @param cvparams   Central-value physics parameter vector.
     * @param seed       Random seed.
     * @param var_index  Variable index to fill (default 0).
     * @return A PROspec with one random systematic throw applied.
     */
    PROspec FillSystRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &model, const PROspec &cvspec, const Eigen::VectorXf &cvparams, uint32_t seed, int var_index);

    /**
     * @brief One systematic throw, split into signal and background pieces.
     * @details Splines are thrown exactly as in FillSystRandomThrow; the covariance
     * systematic is thrown in FULL (uncollapsed) bin space so the background
     * subchannels' own variation can be separated per throw, then masked and
     * collapsed. After collapse this is distributed identically to
     * FillSystRandomThrow's collapsed-space throw, but the RNG stream differs —
     * this function is only used by the --bkg-subtract plotting path, so the
     * default (no-subtraction) throws are untouched.
     * @param inconfig     Configuration object.
     * @param inprop       MC event store.
     * @param insyst       Systematic object.
     * @param model        Physics model.
     * @param cvspec       Central-value spectrum (full binning; denominator for fractional shifts).
     * @param cvparams     Central-value physics parameter vector.
     * @param seed         Random seed.
     * @param var_index    Variable index to fill.
     * @param bkg_bin_mask Full-bin 0/1 vector marking background-subchannel bins.
     * @return {signal, background} PROspecs, both collapsed.
     */
    std::pair<PROspec, PROspec> FillSystRandomThrowSplit(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &model, const PROspec &cvspec, const Eigen::VectorXf &cvparams, uint32_t seed, int var_index, const Eigen::VectorXf &bkg_bin_mask);

    /**
     * @brief Generate a spectrum with a single named spline systematic randomly shifted.
     * @details Throws that spline's shift from a Gaussian distribution and fills the spectrum.
     * @param inconfig    Configuration object.
     * @param inprop      MC event store.
     * @param insyst      Systematic object.
     * @param model       Physics model.
     * @param cvparams    Central-value physics parameter vector.
     * @param spline      0-based index of the spline systematic to vary.
     * @param seed        Random seed.
     * @param other_index Variable index (default 0).
     * @return A PROspec with the specified spline thrown.
     */
    PROspec FillSplineRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst,  const PROmodel &model,  const Eigen::VectorXf &cvparams, int spline, uint32_t seed, int other_index);

    /**
     * @brief Throw one nuisance pull from N(0,1) truncated to spline @p i's allowed range.
     * @details Never loops forever: OOB-safe spline_has_restrict lookup (covariance_to_spline
     * knobs may not populate it), inverted-bounds tolerance, and bounded rejection attempts
     * with a clamp-to-nearest-in-range fallback plus warning (pattern from commit 000b3d0).
     * Shared by the FC pseudo-experiment generators (PROfc, PROAdaptiveFC) and the
     * pseudo-experiment CLI path.
     * @param insyst  Systematic object holding the spline bounds.
     * @param i       0-based spline index.
     * @param rng     Generator to draw from (caller owns seeding/threading).
     * @param d       N(0,1) distribution to draw with.
     * @return The truncated Gaussian pull.
     */
    float ThrowRestrictedSplinePull(const PROsyst &insyst, size_t i, std::mt19937 &rng, std::normal_distribution<float> &d);

};

#endif
