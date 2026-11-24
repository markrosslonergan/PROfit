/**
 * @file PROAdaptiveFCinternal.h
 * @brief Internal cross-TU declarations for the adaptive Feldman-Cousins pipeline.
 * @author PROfit Collaboration
 *
 * @internal Not installed and not part of the public API — that is
 * inc/PROAdaptiveFC.h. The adaptive-FC implementation is split across
 * src/PROAdaptiveFC.cxx (mode dispatcher), src/PROAdaptiveFCmesh.cxx
 * (throws, Wilks prepass, meta-mesh), src/PROAdaptiveFCbank.cxx (PE bank
 * serialisation, PE worker/scheduler, asimov observables, classification)
 * and src/PROAdaptiveFCplot.cxx (all ROOT PDF/artifact output). Everything
 * declared here lives in namespace PROfit::afc so formerly file-local names
 * stay out of the library-wide PROfit namespace.
 */
#ifndef PROADAPTIVEFCINTERNAL_H
#define PROADAPTIVEFCINTERNAL_H

#include "PROAdaptiveFC.h"
#include "PROmodel.h"
#include "PROsplinePrior.h"

#include <Eigen/Eigen>

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace PROfit {
namespace afc {

// --------------------------------------------------------------------
//  Throw generation + meta-mesh (src/PROAdaptiveFCmesh.cxx)
// --------------------------------------------------------------------

std::vector<PROmesh::AMRResult> generate_throws(
    const PROconfig &config,
    const PROpeller &prop,
    const PROsyst   &systs,
    const PROmodel  &model,
    const PROfitterConfig &fitconfig,
    PROseed &proseed,
    const Eigen::VectorXf &fakeDataParams,
    const AdaptiveFCConfig &acfg,
    int nthreads,
    size_t xaxis_idx, size_t yaxis_idx,
    MultiPROgressBar &progress);

MetaMesh build_meta_mesh(const std::vector<PROmesh::AMRResult> &throws,
                         float p_thresh,
                         int baseline_level);

// Union-merge N meta-meshes (identical finest grid + bounds, caller-validated)
// into the coarsest tiling refining every input. Returns empty cells on a
// tiling-invariant violation. baseline_level only affects log counters.
MetaMesh merge_meta_meshes(const std::vector<MetaMesh> &inputs,
                           int baseline_level);

// Meta-mesh from a finest-grid flag map ([i * H + j]): finest cells over
// flagged bins, coarsest tiling elsewhere. Union-merges with any sibling
// mesh of the same finest grid + bounds (brazil-cleanup).
MetaMesh build_mesh_from_flags(int finest_nx, int finest_ny, int max_levels,
                               float x_lo, float x_hi, float y_lo, float y_hi,
                               const std::vector<uint8_t> &flags,
                               int baseline_level);

void compute_cell_centers(const MetaMesh &mm,
                          bool xlog, bool ylog,
                          std::vector<float> &cx_out,
                          std::vector<float> &cy_out);

PROdata generate_pseudo_experiment_data(
    const PROconfig &config,
    const PROpeller &prop,
    const PROsyst   &systs,
    const PROmodel  &model,
    const Eigen::VectorXf &fakeDataParams,
    bool binned,
    const Eigen::MatrixXf &frac_root,
    const SplinePriorSampler &prior_sampler,
    PROseed &proseed);

// --------------------------------------------------------------------
//  PE bank, asimov observables, classification (src/PROAdaptiveFCbank.cxx)
// --------------------------------------------------------------------

// Treatment of the physics parameters in every per-cell constrained fit: the
// axes are pinned at the cell, profiled parameters float, every other
// parameter is pinned at pin_val.
struct NuisanceSpec {
    std::vector<uint8_t> free_mask; ///< 1 = profiled: floats in the constrained fit.
    Eigen::VectorXf pin_val;        ///< Pinned value of each physics parameter (--fix value, else default).
    bool active() const { return std::find(free_mask.begin(), free_mask.end(), (uint8_t)1) != free_mask.end(); }
};
NuisanceSpec make_nuisance_spec(const PROmodel &model, const std::vector<int> &profile_idx);

// Footprint of a mesh cell as one key; equal footprints are the same cell
// centre in any mesh of the same finest grid.
inline uint64_t cell_footprint_key(int i_bl, int j_bl, int step) {
    return ((uint64_t)(uint32_t)i_bl << 42) | ((uint64_t)(uint32_t)j_bl << 21) | (uint64_t)(uint32_t)step;
}

// Toy truth (physics, model space) of every cell: the spec's pins with the two
// axes at the cell centre. Profile banks replace the profiled entries with
// their conditional best fit.
std::vector<std::vector<float>> pinned_cell_truths(const NuisanceSpec &spec,
                                                   size_t xaxis_idx, size_t yaxis_idx,
                                                   const std::vector<float> &cell_x_model,
                                                   const std::vector<float> &cell_y_model);

void schedule_pes(const AdaptiveFCConfig &acfg,
                  const PROconfig &config,
                  const PROpeller &prop,
                  const PROsyst   &systs,
                  const PROmodel  &model,
                  const PROfitterConfig &fitconfig,
                  PROseed &proseed,
                  const Eigen::MatrixXf &frac_root,
                  const NuisanceSpec &spec,
                  const std::vector<std::vector<float>> &cell_truth,
                  PEBank &bank_out,
                  int nthreads,
                  MultiPROgressBar &progress,
                  int progress_bar_idx,
                  AdaptiveFCResult &result_out);

// Observed Δχ² on a dataset at every meta-mesh cell.
//
// The global (syst+osc) fit is computed once — physics floats freely, so the
// result is μ-independent. Per-cell fits then re-pin the two scanned axes at
// each cell center and run the constrained fit: splines float, profiled
// physics floats, every other physics param is pinned (NuisanceSpec).
struct AsimovObs {
    float chi2_osc_global = 0.0f;        ///< Single global best fit.
    std::vector<float> chi2_syst;        ///< Per-cell constrained chi^2.
    std::vector<float> dchi2_obs;        ///< chi2_syst[c] - chi2_osc_global per cell.
    /// Profile mode: physics best fit of each cell's constrained fit (model space).
    std::vector<std::vector<float>> cell_phys_hat;
};

// Profile mode (spec.active()) needs cell_keys: each cell's fitter seed is
// hashed from its footprint, so a cell's result does not depend on the thread
// count or the mesh it sits in. cell_seed_phys (optional) seeds each cell's fit,
// e.g. with the bank truths. snap_to_default: a profiled parameter the data
// cannot see (pinning it at its default costs < 0.01 in chi^2) reports its
// default rather than wherever the minimiser drifted; costs one more fit per cell.
AsimovObs compute_asimov_obs(
    const PROconfig &config,
    const PROpeller &prop,
    const PROsyst   &systs,
    const PROmodel  &model,
    const PROfitterConfig &fitconfig,
    const PROdata   &asimov_data,
    const std::string &chi2_kind,
    bool binned,
    bool shape_only,
    size_t xaxis_idx, size_t yaxis_idx,
    const std::vector<float> &cell_x_model,
    const std::vector<float> &cell_y_model,
    PROseed &proseed,
    int nthreads,
    MultiPROgressBar &progress,
    int bar_idx,
    const NuisanceSpec &spec,
    const std::vector<uint64_t> *cell_keys = nullptr,
    const std::vector<std::vector<float>> *cell_seed_phys = nullptr,
    bool snap_to_default = false);

// Per-cell, per-CL verdict map produced by classify_against_bank.
struct CellVerdict {
    float crit_dchi2 = 0.0f;  ///< Empirical critical Δχ² at the bank's α-quantile.
    bool  included  = false;  ///< dchi2_obs ≤ crit_dchi2 at this CL.
    bool  decidable = false;  ///< Bank had enough PEs to give a stable quantile.
};

// Bank-only precompute of crit_dchi2/decidable per [cl_idx][cell_idx]
// (`included` left false). One sort per cell; reuse across many throws.
std::vector<std::vector<CellVerdict>> compute_bank_crits(
    const PEBank &bank,
    const std::vector<float> &cl_targets,
    int min_pes_for_decision);

std::vector<std::vector<CellVerdict>> classify_against_bank(
    const PEBank &bank,
    const AsimovObs &obs,
    const std::vector<float> &cl_targets,
    int min_pes_for_decision);

// --------------------------------------------------------------------
//  ROOT PDF / artifact output (src/PROAdaptiveFCplot.cxx)
// --------------------------------------------------------------------

void write_slice1_diagnostics(
    const std::vector<PROmesh::AMRResult> &throws,
    const MetaMesh &mm,
    const PROmodel &model,
    const PROsyst  &systs,
    const AdaptiveFCConfig &acfg,
    size_t xaxis_idx, size_t yaxis_idx,
    AdaptiveFCResult &result_out);

// Meta-mesh visualisation (level-coloured cells + info panel). Pass
// n_throws <= 0 for meshes loaded from disk / derived meshes (merge,
// brazil-cleanup): throw-tally alpha modulation and info lines are skipped.
void plot_metamesh_pdf(const MetaMesh &mm,
                       const PROmodel &model,
                       const std::string &filename,
                       int n_throws,
                       float p_thresh,
                       int baseline_level,
                       bool logx, bool logy,
                       size_t xaxis_idx, size_t yaxis_idx);

void plot_pebank_summary_pdf(const PEBank &bank,
                             const std::string &filename,
                             const std::string &bank_path,
                             const std::string &xlabel,
                             const std::string &ylabel,
                             bool logx, bool logy,
                             bool xlog_axis, bool ylog_axis);

void plot_pebank_pes_multipage_pdf(const PEBank &bank,
                                   const std::string &filename,
                                   const std::string &xlabel,
                                   const std::string &ylabel,
                                   bool xlog_axis, bool ylog_axis);

void plot_asimov_verdict_pdf(
    const PEBank &bank,
    const AsimovObs &obs,
    const std::vector<float> &cl_targets,
    const std::vector<std::vector<CellVerdict>> &verdicts,
    const std::string &filename,
    const std::string &bank_path,
    const std::string &xlabel,
    const std::string &ylabel,
    bool logx, bool logy,
    bool xlog_axis, bool ylog_axis);

void plot_asimov_contour_pdf(
    const PEBank &bank,
    const AsimovObs &obs,
    const std::vector<float> &cl_targets,
    const std::vector<std::vector<CellVerdict>> &verdicts,
    const std::string &filename,
    const std::string &xlabel,
    const std::string &ylabel,
    bool logx, bool logy,
    bool xlog_axis, bool ylog_axis,
    bool draw_truth_marker,
    float truth_x_phys,
    float truth_y_phys);

// profiled: (name, physics index, is_log10) of each profiled parameter; adds a
// zhat_<name> column (physical units) to the cells tree. Empty = legacy layout.
void save_asimov_root(
    const PEBank &bank,
    const AsimovObs &obs,
    const std::vector<float> &cl_targets,
    const std::vector<std::vector<CellVerdict>> &verdicts,
    const std::string &filename,
    bool xlog_axis, bool ylog_axis,
    const std::vector<std::tuple<std::string, int, bool>> &profiled = {});

// One value per bank cell painted over its footprint (colz + cell borders),
// e.g. the per-cell toy truth or conditional best fit of a profiled parameter.
void plot_cell_value_map_pdf(const PEBank &bank,
                             const std::vector<float> &cell_values,
                             const std::string &title,
                             const std::string &filename,
                             const std::string &xlabel,
                             const std::string &ylabel,
                             const std::string &zlabel,
                             bool logx, bool logy,
                             bool xlog_axis, bool ylog_axis);

// Flag finest-grid bins ([i * H + j]) traversed by the SAVED Brazil quantile
// contour TGraphs in <tag>_brazil.root — the exact curve objects the band
// PDF drew; nothing is recomputed. quantiles must be among the five saved
// levels (0.025 0.16 0.5 0.84 0.975); cl_targets empty = all CLs in the
// file. halo dilates by that many bins.
std::vector<uint8_t> flag_bins_from_saved_brazil_contours(
    const PEBank &bank,
    const std::string &brazil_root_path,
    const std::vector<float> &cl_targets,
    const std::vector<float> &quantiles,
    bool xlog_axis, bool ylog_axis,
    int halo,
    int &n_curves_used);

void plot_brazil_band_pdf(
    const PEBank &bank,
    const std::vector<std::vector<float>> &inclusion_frac, // [cl_idx][cell_idx]
    const std::vector<float> &cl_targets,
    const std::string &filename,
    const std::string &bank_path,
    const std::string &xlabel,
    const std::string &ylabel,
    bool logx, bool logy,
    bool xlog_axis, bool ylog_axis,
    bool draw_truth_marker,
    float truth_x_phys,
    float truth_y_phys,
    const std::vector<int> &n_throws_kept,     // [cl] throws entering the band
    const std::vector<int> &n_throws_dropped,  // [cl] closed-contour throws removed
    const std::string &band_flag = "");        // flag-styled bands: "america", "ireland" ("" = standard)

void save_brazil_root(
    const PEBank &bank,
    const std::vector<std::vector<std::vector<uint8_t>>> &per_throw_verdicts, // [t][cl][cell]
    const std::vector<std::vector<float>> &per_throw_dchi2,                   // [t][cell]
    const std::vector<float> &per_throw_global_chi2,                          // [t]
    const std::vector<std::vector<float>> &inclusion_frac,                    // [cl][cell]
    const std::vector<std::vector<uint8_t>> &throw_kept,                      // [cl][t]
    const std::vector<float> &cl_targets,
    const std::string &filename,
    bool xlog_axis, bool ylog_axis);

}  // namespace afc
}  // namespace PROfit

#endif
