/**
 * @file PROAdaptiveFC.h
 * @brief Adaptive Feldman-Cousins pipeline — public surface.
 *
 * The pipeline covers: a Wilks pre-pass over N independent throws aggregated
 * into a meta-mesh (build-mesh), PE bank generation with a deterministic
 * per-level doubling rule (init-bank), bank inspection (print-bank), and
 * classification of a dataset against the bank (asimov / brazil), each with
 * diagnostic PDF and ROOT artifacts.
 *
 * The implementation is split across src/PROAdaptiveFC.cxx (mode dispatcher),
 * src/PROAdaptiveFCmesh.cxx (throws, prepass, meta-mesh),
 * src/PROAdaptiveFCbank.cxx (serialisation, PE worker/scheduler, asimov
 * observables, classification) and src/PROAdaptiveFCplot.cxx (ROOT output),
 * with internals declared in inc/PROAdaptiveFCinternal.h. The AMR per-point
 * fit body and the mesh drawing are shared with PROsurf via inc/PROmeshEval.h
 * and inc/PROmeshPlot.h. The per-PE worker (run_one_pe) intentionally
 * parallels src/PROfc.cxx::fc_worker (the brute-force FC) and remains
 * duplicated by design.
 */
#ifndef PRO_ADAPTIVE_FC_H
#define PRO_ADAPTIVE_FC_H

#include "PROconfig.h"
#include "PROpeller.h"
#include "PROsyst.h"
#include "PROseed.h"
#include "PROfitter.h"
#include "PROgress.h"
#include "PROmesh.h"
#include "PROdata.h"

#include <Eigen/Eigen>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace PROfit {

    /// Pipeline mode, selected by the fc-adaptive `--mode` flag.
    enum class AdaptiveFCMode {
        BuildMesh, ///< Wilks prepass + meta-mesh build + diagnostics, then save <tag>_mesh.bin and exit.
        InitBank,  ///< Load <tag>_mesh.bin, generate PE bank, save <tag>_bank.bin. Errors if mesh missing.
        PrintBank, ///< Load <tag>_bank.bin and write summary PDF(s). No fitting.
        Asimov,    ///< Load <tag>_bank.bin, classify the asimov dataset, produce contour PDF + ROOT.
        Brazil,    ///< Brazil-band throw loop: classify N pseudo-experiments against the bank.
        Classify,  ///< (not yet implemented) Classify real data against the bank.
        MergeMesh, ///< Union-merge ≥2 mesh binaries (--merge-input) into <tag>_mesh.bin.
        MergeBank, ///< Harvest PEs from ≥1 bank binaries (--merge-input) onto <tag>_mesh.bin, save <tag>_bank.bin.
        MergeBrazil, ///< Union throws from ≥1 brazil archives (--merge-input) into <tag>_brazil.bin, re-classify against <tag>_bank.bin, emit band PDF + ROOT. No fits.
        BrazilCleanup, ///< From <tag>_bank.bin + <tag>_brazil.bin, build <tag>_cleanup_mesh.bin densified at the Brazil ±2σ contours. No fits.
        PrintMesh, ///< Plot <tag>_mesh.bin (or any mesh binaries given via --merge-input) as PDF(s). No fitting.
    };

    /**
     * @brief CLI-driven configuration for the adaptive FC pipeline.
     *
     * Populated by the fc-adaptive subcommand in bin/PROfit.cxx; which fields
     * are consumed depends on the selected AdaptiveFCMode.
     */
    struct AdaptiveFCConfig {
        AdaptiveFCMode mode = AdaptiveFCMode::InitBank;

        // Wilks pre-pass.
        int   n_throws            = 200;
        int   prepass_amr_initial_x = 10;
        int   prepass_amr_initial_y = 10;
        int   prepass_amr_levels    = 3;
        float prepass_delta_widen   = 0.05f; ///< Tiny by design: per-throw global-fit warm-starts make AMR fitter scatter sub-χ², so the "halo" of cells just inside/outside the contour is wasted work for meta-mesh aggregation. (PROsurf surface-amr keeps its 0.5 default — different use case.)
        std::vector<float> prepass_contour_levels = {2.30f, 5.99f}; ///< Wilks Δχ² target levels (1σ, 2σ at 2 dof by default).
        bool  stat_only_throws    = false;

        // Scan axes (mirror the `surface` subcommand semantics).
        std::string xvar = "sinsq2thmm";
        std::string yvar = "dmsq";
        bool xvar_explicit = false, yvar_explicit = false; ///< Given on the CLI: an unknown name is then fatal instead of falling back.
        float x_lo = 1e-4f, x_hi = 1.0f;
        float y_lo = 1e-2f, y_hi = 1e2f;
        bool  logx = true, logy = true;

        // Meta-mesh build.
        float p_thresh       = 0.05f;   ///< Refine cell if fraction of throws refining it ≥ p_thresh.
        int   baseline_level = 2;       ///< Levels < baseline_level are kept regardless of p_thresh.

        // Output naming.
        std::string output_tag = "afc_slice1";

        // chi^2 metric name and binned/eventbyevent flags (passed straight through
        // to the inner per-throw fits — mirrors fc_args.chi2 / fc_args.binned).
        std::string chi2 = "neyman";
        bool        binned = true;
        bool        shape_only = false; ///< --shapeonly: every inner metric is built shape-only (must match the data fit).

        // Physics bounds from the setup chain, carrying --fix / --syst-only pins.
        // The AFC model is rebuilt from the XML, so without these the pins are lost.
        Eigen::VectorXf phys_lb, phys_ub;
        /// Pre-v3.1 behaviour for non-axis physics: bank and asimov pin them at
        /// the model default while the Wilks prepass profiles them. Otherwise
        /// every non-axis physics parameter must be --fix'd or profiled.
        bool legacy_nuisance_pin = false;
        /// Non-axis physics parameters to profile (NOvA profile construction):
        /// they float in every constrained fit, and the bank's toys put them at
        /// their conditional best fit to the analysed dataset in each cell.
        std::vector<std::string> profile_params;
        bool force = false; ///< --force: classify a dataset other than the one a profile bank was built for.
        /// Fit configuration for the profile-mode fits to the analysed dataset (toy
        /// truths, asimov observables): the global-fit settings, as fc's conditional
        /// fit uses. Toy fits keep the scan settings.
        PROfitterConfig data_fitconfig;

        // ---- PE-bank generation knobs ----
        // Per-cell PE generation follows an additive doubling rule:
        //   to_add = n_pe_min * 2^max(0, cell.level - update_layer)
        // Each run ADDS this many PEs to the cell on top of whatever's already
        // there (existing PEs always preserved). Cells with level < update_layer
        // are not touched. n_pe_max is a total-per-cell safety cap — no cell
        // ever exceeds it, even across many runs.
        std::vector<float> cl_targets = {0.683f, 0.90f, 0.954f};
        int   n_pe_min = 50;     ///< PEs added per run at level == update_layer; doubles for each deeper level.
        int   n_pe_max = 5000;   ///< Hard total cap per cell (cumulative across runs).
        int   update_layer = 0;  ///< Cells below this AMR level are skipped entirely.
        int   only_layer   = -1; ///< If >= 0, update ONLY cells at exactly this level (overrides update_layer).
        float wilson_eps = 0.05f; ///< Unused for bank generation now; reserved for the future classify mode.
        float roi_band = 8.0f;   ///< Reserved for the future classify mode; unused for now.
        int   n_brazil_throws = 100; ///< Number of pseudo-experiment throws for --mode brazil.
        std::string band_flag = "";    ///< --flag: brazil band PDF styled after a national flag ("america", "ireland"); empty = standard green/yellow.

        // ---- merge-mesh / merge-bank / merge-brazil inputs ----
        // Concrete artifact filenames (glob patterns already expanded by the
        // CLI layer). merge-mesh: ≥2 *_mesh.bin; merge-bank: ≥1 *_bank.bin
        // harvested onto this tag's already-merged <output_tag>_mesh.bin;
        // merge-brazil: ≥1 *_brazil.bin unioned onto this tag's
        // <output_tag>_bank.bin (whole-archive footprint match required).
        std::vector<std::string> merge_inputs;

        // ---- brazil-cleanup ----
        // Inclusion-fraction quantile levels whose plotted contours (traced
        // on the same IDW-smoothed surface as the band PDF) get finest
        // refinement. Default = the Brazil ±2σ band edges.
        std::vector<float> cleanup_quantiles = {0.025f, 0.975f};
        // Dilate the flagged contour path by this many finest bins so the
        // mesh brackets the curve on both sides.
        int cleanup_halo = 1;
    };

    /**
     * @brief One aggregated meta-mesh cell.
     *
     * Coordinates are in the finest-integer system shared by every per-throw
     * AMRResult (see PROmesh::MeshCell). `per_level_refine_count[L]` is the
     * number of throws that refined this cell to depth ≥ L.
     */
    struct MetaCell {
        int i_bl   = 0;
        int j_bl   = 0;
        int step   = 0;
        int level  = 0;
        std::vector<int> per_level_refine_count;
    };

    /**
     * @brief Aggregated meta-mesh from N per-throw Wilks AMR meshes.
     *
     * `finest_nx`, `finest_ny` and the (x_lo, x_hi, y_lo, y_hi) box must be
     * identical across every per-throw AMRResult that contributed.
     */
    struct MetaMesh {
        static constexpr uint32_t MAGIC = 0x41464D45;  ///< 'AFME' (Adaptive FC MEsh).
        static constexpr uint16_t VERSION = 1;

        std::vector<MetaCell> cells;
        int   finest_nx = 0;
        int   finest_ny = 0;
        int   max_levels = 0;
        float x_lo = 0.0f, x_hi = 0.0f;
        float y_lo = 0.0f, y_hi = 0.0f;

        // Cell counters for quick logging.
        int n_baseline_cells = 0;
        int n_refined_cells  = 0;
    };

    /**
     * @brief Persist a MetaMesh to disk via Boost binary archive.
     *
     * Sibling artifact to the PEBank — meant as a dev-time cache so the
     * prepass+meta-mesh build doesn't have to re-run when iterating on PE-bank
     * parameters. Same magic+version layout as save_bank.
     */
    bool save_mesh(const MetaMesh &mm, const std::string &path);

    /// Inverse of save_mesh. Returns false (and leaves `mm_out` indeterminate)
    /// on missing file, bad magic, or version mismatch.
    bool load_mesh(MetaMesh &mm_out, const std::string &path);

    /**
     * @brief Empirical-quantile helper for PE-bank classification. Pure utility — no PROfit deps.
     *
     * Inlined so the helper has no link dependency on the .cxx (useful for
     * unit-testing).
     */
    struct SequentialFCTest {
        // NOTE: the Wilson-interval sequential stopping rule that used to live
        // here was never wired into schedule_pes (the additive-doubling rule
        // is the real PE budget policy) and has been removed as dead code.
        // Only the empirical-quantile helper below is used.

        /**
         * @brief Empirical Δχ² critical value at a given confidence level.
         *
         * Returns the `cl`-quantile of the supplied (sorted) sample. Used by
         * the classify path to convert a per-cell PE distribution into a
         * critical Δχ²_c against which the observed Δχ² is compared.
         *
         * @param sorted_samples  Δχ² values in ascending order.
         * @param cl              Target confidence level in (0, 1).
         */
        static float crit_dchi2_at_cl(const std::vector<float> &sorted_samples, float cl) {
            if (sorted_samples.empty()) return 0.0f;
            const float p = std::min(1.0f, std::max(0.0f, cl));
            // Linear-interpolated quantile (type-7 in R's parlance).
            const float pos = p * ((float)sorted_samples.size() - 1.0f);
            const int   lo  = (int)std::floor(pos);
            const int   hi  = (int)std::ceil(pos);
            if (lo == hi) return sorted_samples[lo];
            const float frac = pos - (float)lo;
            return (1.0f - frac) * sorted_samples[lo] + frac * sorted_samples[hi];
        }
    };

    /**
     * @brief One pseudo-experiment record in the PE bank.
     *
     * Mirrors fc_out (inc/PROfc.h:34) but keeps only the fields needed
     * to drive classification (the syst-only and syst+osc chi^2s and their
     * difference). Best-fit vectors are dropped to keep the bank small; they
     * can be re-derived if ever needed by re-running the throw with the same seed.
     */
    struct PEBankRecord {
        float    chi2_syst = 0.0f; ///< Best-fit chi^2 from the syst-only minimisation.
        float    chi2_osc  = 0.0f; ///< Best-fit chi^2 from the full (syst + osc) minimisation.
        float    dchi2     = 0.0f; ///< chi2_syst - chi2_osc.
        uint32_t seed      = 0;    ///< RNG seed used to generate this PE (for reproducibility).
        std::vector<float> phys_hat; ///< Profile banks: constrained-fit best fit of each profiled parameter (manifest order, model space).
    };

    /**
     * @brief Where a bank's toys put the physics parameters that are not axes.
     * @details Values 1 (fixed injected truth) and 3 (one bank per fixed truth
     * z_k, the "z-sliced" bank) are reserved for the planned extensions.
     */
    enum class TruthSource : uint8_t {
        LegacyPin      = 0, ///< Pinned (--fix value, else model default) in truth and fits. Every v2 bank.
        ConditionalFit = 2, ///< NOvA profile construction: profiled physics floats in the constrained fit and
                            ///< the truth is its conditional best fit, per cell, to the bank's dataset.
    };

    /**
     * @brief What a profile bank was built for. Banks, archives and the
     * classification run must agree on all of it.
     */
    struct ProfileManifest {
        uint8_t  truth_source = (uint8_t)TruthSource::LegacyPin;
        std::vector<int> profile_idx; ///< Physics indices floated in the constrained fits.
        uint64_t data_fp = 0;         ///< Fingerprint of the dataset the truths were fitted to (DataFingerprint).
        uint32_t config_hash = 0;
        std::string model_tag;
        std::string chi2;
        uint8_t  shape_only = 0;
        uint64_t overlap_fp = 0;      ///< OverlapFingerprint of the MC statistical overlap the toys were thrown with (0 = none).

        bool profiled() const { return truth_source != (uint8_t)TruthSource::LegacyPin; }
        /// Written as a tail: profile banks, and any bank thrown with statistical overlap.
        bool has_tail() const { return profiled() || overlap_fp != 0; }
        bool operator==(const ProfileManifest &o) const {
            return truth_source == o.truth_source && profile_idx == o.profile_idx && data_fp == o.data_fp
                && config_hash == o.config_hash && model_tag == o.model_tag && chi2 == o.chi2
                && shape_only == o.shape_only && overlap_fp == o.overlap_fp;
        }
        bool operator!=(const ProfileManifest &o) const { return !(*this == o); }
    };

    /// 64-bit fingerprint of a dataset's collapsed spectrum (profile banks are data-dependent).
    uint64_t DataFingerprint(const PROdata &data);

    /**
     * @brief Pseudo-experiment bank attached to a meta-mesh.
     *
     * One slot per MetaCell (Option A: PE bank at each cell center). Cell index
     * is the position of the MetaCell in `MetaMesh::cells`. Each slot holds the
     * raw Δχ² outcomes from `n_pes_at_cell[c]` PEs thrown at that cell.
     *
     * The bank is the pre-unblinding artifact: classification at any threshold
     * is a sort+search over the per-cell vector, so the *same* bank serves the
     * Asimov sensitivity, Brazil-band, and real-data analyses without
     * regenerating PEs. That holds for LegacyPin banks only: a profile bank
     * (TruthSource::ConditionalFit) throws its toys at the conditional best
     * fit to ONE dataset, so the real data needs its own bank, and a Brazil
     * band classified against the Asimov's bank is the approximate (NOvA-style)
     * construction.
     */
    struct PEBank {
        static constexpr uint32_t MAGIC = 0x41464342;  ///< 'AFCB' (Adaptive FC Bank).
        static constexpr uint16_t VERSION = 3;
        static constexpr uint16_t VERSION_OVERLAP = 4;
        ///< v2: added cell footprint arrays (i_bl, j_bl, step, level) so the
        ///<     bank is self-describing without needing the MetaMesh sidecar.
        ///< v3: profile banks (manifest, per-cell truths, per-PE phys_hat) as a
        ///<     tail after the unchanged v2 payload. LegacyPin banks are still
        ///<     written as v2, byte for byte.
        ///< v4: v3 plus the manifest's overlap_fp, for banks thrown with
        ///<     statistical overlap (either truth source).

        // Mesh footprint the bank was generated against. Used as a provenance
        // check at load time so we can refuse a (bank, mesh) mismatch.
        int   finest_nx = 0;
        int   finest_ny = 0;
        int   max_levels = 0;
        float x_lo = 0.0f, x_hi = 0.0f;
        float y_lo = 0.0f, y_hi = 0.0f;
        int   n_cells = 0;

        // Per-cell PE storage. Outer index = cell index into MetaMesh::cells;
        // inner vector = PEs thrown at that cell in generation order.
        std::vector<std::vector<PEBankRecord>> cell_pes;

        // Per-cell coordinates in *model space* (log10 of physical for
        // log-axis params; linear otherwise). Classification/plotting code
        // applies pow(10) when mapping back to physical for display.
        std::vector<float> cell_center_x;
        std::vector<float> cell_center_y;

        // Per-cell footprint copied from MetaMesh::cells. Lets the bank be
        // plotted / inspected without re-loading the mesh sidecar.
        std::vector<int> cell_i_bl;
        std::vector<int> cell_j_bl;
        std::vector<int> cell_step;
        std::vector<int> cell_level;

        // v3 (profile banks) only.
        ProfileManifest manifest;
        /// Toy truth of every physics parameter per cell (model space). Empty
        /// for a cell whose truth is not computed yet (e.g. fresh after merge-bank).
        std::vector<std::vector<float>> cell_truth_phys;
    };

    /**
     * @brief Persist a PEBank to disk via Boost binary archive.
     *
     * On-disk layout: magic + version + the serialised PEBank. Mirrors the
     * pattern in src/PROcreate.cxx:25-69.
     * @return true on success, false on any I/O or serialisation error.
     */
    bool save_bank(const PEBank &bank, const std::string &path);

    /**
     * @brief Inverse of save_bank. Verifies magic + version on the way in.
     * @return true on success and `bank_out` populated; false otherwise (and
     *         `bank_out` left in an indeterminate state).
     */
    bool load_bank(PEBank &bank_out, const std::string &path);

    /**
     * @brief Per-throw observables persisted from --mode brazil runs.
     *
     * Stores only the *physics observables* per throw (the global chi^2 and
     * the per-cell dchi2_obs). Verdicts (included / excluded / undecidable)
     * are NOT stored — they're recomputed at use time from these observables
     * against the current bank's empirical alpha-quantile and the current
     * `cl_targets`. That makes the archive bank-agnostic and CL-agnostic:
     * grow the bank or change CLs between brazil runs and the existing
     * throws are reused without regenerating.
     *
     * On-disk: <tag>_brazil.bin, magic + version + serialised struct.
     * Re-running --mode brazil --n-brazil-throws N appends N more throws
     * to whatever's already there (mirrors PEBank additive top-up).
     * v2 (profile banks) appends the manifest and per-throw phys_hat after the
     * unchanged v1 payload; LegacyPin archives are still written as v1.
     */
    struct BrazilArchive {
        static constexpr uint32_t MAGIC = 0x41464252;  ///< 'AFBR'
        static constexpr uint16_t VERSION = 2;
        static constexpr uint16_t VERSION_OVERLAP = 3; ///< v2 plus the manifest's overlap_fp.
        // Footprint sanity check vs. the current bank at load time.
        int finest_nx = 0;
        int finest_ny = 0;
        int n_cells   = 0;
        // Per-throw data. Outer index = throw, inner = cell.
        std::vector<float> per_throw_global_chi2;        ///< chi2_osc for each throw's global fit.
        std::vector<std::vector<float>> per_throw_dchi2; ///< dchi2_obs per cell, per throw.
        // v2 (profile banks) only.
        ProfileManifest manifest;
        std::vector<std::vector<float>> per_throw_phys_hat; ///< [throw][cell * n_profiled + k]: conditional best fit per cell.
    };

    bool save_brazil_archive(const BrazilArchive &arc, const std::string &path);
    bool load_brazil_archive(BrazilArchive &arc_out, const std::string &path);

    /**
     * @brief Output bundle from run_adaptive_fc.
     */
    struct AdaptiveFCResult {
        bool ok = true; ///< false when a stage could not load its inputs or refused to run.

        // Build-mesh fields.
        int  n_throws_done   = 0;
        int  n_meta_cells    = 0;
        int  n_baseline_cells = 0;
        int  n_refined_cells = 0;
        std::vector<int> leaves_per_throw;
        std::string diag_root_path;

        // Bank fields (populated by --mode init-bank).
        std::string bank_path;        ///< Path of the saved PEBank artifact, if any.
        int64_t     total_pes_generated = 0;
        int         cells_hit_n_pe_max  = 0; ///< Number of cells skipped because they already hold n_pe_max PEs.
        int         cells_topped_up     = 0; ///< Number of cells that received new PEs this run.
        float       mean_pes_per_cell   = 0.0f;
    };

    /**
     * @brief Adaptive FC entry point.
     *
     * @param data  The data spectrum PROfit assembled (real data OR injected fake OR
     *              `--poisson-throw`'d OR `--pseudo-experiment`'d). Asimov / classify
     *              modes operate on this; build-mesh / init-bank don't read it.
     *              Without this we'd silently ignore the user's `--use-fake-data`
     *              choices in asimov mode.
     */
    AdaptiveFCResult run_adaptive_fc(
        const PROconfig &config,
        const PROpeller &prop,
        const PROsyst   &systs,
        const PROfitterConfig &fitconfig,
        PROseed         &proseed,
        const Eigen::VectorXf &fakeDataParams,
        const PROdata   &data,
        const AdaptiveFCConfig &acfg,
        int nthreads,
        MultiPROgressBar &progress);

} // namespace PROfit

#endif
