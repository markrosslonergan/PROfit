/**
 * @file PROsurf.h
 * @brief Chi-squared surface, profile likelihood, and sensitivity curve computation.
 * @author PROfit Collaboration
 *
 * @details Defines PROsurf (2D chi-squared surface scanning) and PROfile (1D profile
 * likelihood scanning) for producing exclusion contours and confidence regions.
 * Both classes take a PROmetric and scan physics parameter combinations, running the
 * full PROfitter pipeline at each grid point in parallel threads.
 *
 * Also defines the output structs surfOut and profOut that carry per-grid-point results.
 */
#ifndef PROSURF_H
#define PROSURF_H

#include "PROfitter.h"
#include "PROconfig.h"
#include "PROsyst.h"
#include "PROseed.h"
#include "PROmetric.h"
#include "PROgress.h"
#include "PROversion.h"
#include "PROmesh.h"
#include "PRObe.h"

#include <Eigen/Eigen>

#include <atomic>
#include <mutex>

#include "TGraphAsymmErrors.h"
#include "TMarker.h"
#include "TMultiGraph.h"
#include "TArrow.h"

namespace PROfit {

    /**
     * @brief Output record for a single grid point in a 2D chi-squared surface scan.
     */
    struct surfOut{
        std::vector<int> grid_index;   ///< Indices [ix, iy] of this grid point in the surface matrix.
        std::vector<float> grid_val;   ///< Parameter values [x, y] at this grid point.
        Eigen::VectorXf best_fit;      ///< Best-fit full parameter vector (physics + splines) at this point.
        float chi;                     ///< Minimum chi-squared found at this physics point.
    };

    /**
     * @brief One unit of profile-scan work for the dynamic dispatcher.
     * @details A "task" is "scan parameter `param_idx` over the sub-range
     * [sub_lb, sub_ub]". Most parameters produce a single task spanning the full
     * range. Physics parameters can be split into multiple chunked tasks via
     * --probe-chunks so that several threads can work on the same physics
     * parameter in parallel; chunked task results are merged by param_idx.
     */
    struct ScanTask {
        int   param_idx;   ///< Index in the full (model + splines) parameter vector.
        float sub_lb;      ///< Lower edge of this task's scan range for the scanned parameter.
        float sub_ub;      ///< Upper edge of this task's scan range for the scanned parameter.
    };

    /**
     * @brief Output record for a 1D profile likelihood scan.
     */
    struct profOut{
        int param_idx = -1; ///< Index of the scanned parameter in the full (model + splines) vector. Set by the worker; used by the dispatcher to merge results.
        std::vector<float> knob_vals; ///< Parameter values at each scan point.
        std::vector<float> knob_chis; ///< Profile chi-squared at each scan point.
        std::vector<Eigen::VectorXf> knob_bfs; ///< Best-fit parameter vectors at each scan point.
        float chi; ///< Global minimum chi-squared found during the profile scan.

        void sort(){
            std::vector<size_t> indices(knob_vals.size());
            std::iota(indices.begin(), indices.end(), 0);

            // Sort indices based on knob_vals
            std::sort(indices.begin(), indices.end(),
                    [this](size_t i, size_t j) { return knob_vals[i] < knob_vals[j]; });

            std::vector<float> sorted_vals(knob_vals.size());
            std::vector<float> sorted_chis(knob_chis.size());
            std::vector<Eigen::VectorXf> sorted_bfs(knob_bfs.size());

            for(size_t i = 0; i < indices.size(); ++i) {
                sorted_vals[i] = knob_vals[indices[i]];
                sorted_chis[i] = knob_chis[indices[i]];
                sorted_bfs[i] = knob_bfs[indices[i]];
            }

            knob_vals = std::move(sorted_vals);
            knob_chis = std::move(sorted_chis);
            knob_bfs = std::move(sorted_bfs);
        }
    };

    /**
     * @brief 1D profile likelihood scanner producing exclusion bands for individual parameters.
     * @details Scans one physics parameter at a time while minimising over all others using
     * PROfitter.  Produces 1-sigma error bands and stores the result in ROOT TGraph objects.
     * Supports optional oscillation parameter inclusion and multi-threaded point evaluation.
     */
    class PROfile {

        public:
            PROmetric &metric;                          ///< Reference to the chi-squared metric being scanned.
            TGraphAsymmErrors onesig;                   ///< ROOT graph of the 1-sigma asymmetric error band.
            std::vector<std::unique_ptr<TGraph>> graphs; ///< Per-systematic profile chi-squared graphs.
            std::vector<float> bfvalues;   ///< Best-fit parameter values at each scan point.
            std::vector<float> barvalues;  ///< Bar (central) values at each scan point.
            std::vector<float> values1_up;   ///< Upper 1-sigma boundary values.
            std::vector<float> values1_down; ///< Lower 1-sigma boundary values.

            /** @brief If the scan found a lower global minimum than the caller's
             *  `minchi` (the global fit was trapped in a local minimum): the new
             *  minimum chi-squared. 0 when the global fit stood. All profile
             *  curves, bands, and the points file are already re-baselined
             *  against it, so plots never dip below Δχ²=0. Callers (the profile
             *  subcommand) should adopt it as the global best fit for markers /
             *  recorded results — see the profile block in bin/PROfit.cxx. */
            float newglob = 0;
            Eigen::VectorXf newglob_param;  ///< Full parameter vector at that lower minimum (empty when the global fit stood).

            /** @brief Live cross-thread, cross-parameter tracker of the lowest
             *  chi² seen by any scan fit (see ScanGlobalMin). Initialised with
             *  (minchi, seed_points.front()) before workers dispatch; once a
             *  scan fit beats it, every subsequent fit of EVERY parameter is
             *  additionally seeded from the improved point, so the other
             *  parameters' curves find the deeper basin too instead of being
             *  re-baselined against a minimum they never sampled. */
            ScanGlobalMin global_min_tracker;

            /** @brief Shared warm-start bank for the profile scans: for each scanned
             *  parameter, every completed (scanned value, best-fit vector) pair from ALL
             *  threads, tasks, and chunks. Each new scan-point fit is seeded from the
             *  entry of the SAME parameter closest in scanned value -- robust to the
             *  center-out walk order, to --probe-chunks splitting one parameter across
             *  threads, and to dynamic task dispatch. Guarded by seed_bank_mutex
             *  (contention is negligible: bank ops are microseconds vs ~0.1-1 s fits).
             *  Note: a spline's full scan is a single task, so spline seeding is
             *  deterministic regardless of thread count; chunked physics parameters may
             *  see completion-order-dependent (but still valid) seeds. */
            std::vector<std::vector<ScanPoint>> seed_bank;
            std::mutex seed_bank_mutex;     ///< Guards seed_bank.

            /** @brief Fixed seeds (see PROfitter FixedSeed) forwarded to every scan-point
             *  fit, e.g. the background-only seed with physics pinned at the model
             *  defaults. A seed whose pins conflict with a scan point's bounds (most
             *  commonly: profiling the very parameter the seed pins, at a different
             *  value) is skipped inside PROfitter::Fit for that point. Copied from the
             *  constructor argument before workers dispatch; read-only afterwards. */
            std::vector<FixedSeed> fixed_seed_points;

            PROfile(const PROconfig &config, const PROsyst &systs, const PROmodel &model, PROmetric &metric, PROseed &proseed, const PROfitterConfig &fitconfig, std::string filename, float minchi = 0, bool with_osc = false, int nThreads = 1, const std::vector<Eigen::VectorXf> &seed_points = {}, const Eigen::VectorXf& true_params = Eigen::VectorXf(), bool use_probe = false, int n_physics_chunks = 1, const std::vector<FixedSeed> &fixed_seeds = {} ) ;

            void Plot(const PROconfig &config, const PROsyst &systs, const PROmodel &model, PROmetric &metric, PROseed &proseed, std::string filename, bool with_osc = false, const Eigen::VectorXf& init_seed = Eigen::VectorXf(), const Eigen::VectorXf& true_params = Eigen::VectorXf(), const Eigen::MatrixXf& spline_covariance = Eigen::MatrixXf{}, const Eigen::VectorXf& param_err_lo = Eigen::VectorXf{}, const Eigen::VectorXf& param_err_hi = Eigen::VectorXf{}, bool mask_osc = false) ;

            std::vector<profOut> PROfilePointHelper(const PROsyst *systs, const PROfitterConfig &fitconfig, std::atomic<int> *task_counter, const std::vector<ScanTask> *tasks, float minchi, bool with_osc, MultiPROgressBar& progressbar, const std::vector<Eigen::VectorXf> &seed_points = {}, uint32_t seed=0, bool use_probe = false, std::atomic<int>* tasks_remaining = nullptr, int bar_index_offset = 0, std::atomic<uint64_t>* max_thread_wall_us = nullptr);
    };

    /**
     * @brief 2D chi-squared surface scanner for two-parameter exclusion contours.
     * @details Evaluates the profile chi-squared on a 2D grid of (x, y) physics parameter
     * values, minimising over all other parameters at each grid point via PROfitter.
     * Results are stored in the surface matrix for subsequent contour plotting.
     * Supports linear and logarithmic axis spacing.
     */
    class PROsurf {
        public:
            PROmetric &metric;     ///< Reference to the chi-squared metric being evaluated.
            size_t x_idx;          ///< Index of the x-axis physics parameter.
            size_t y_idx;          ///< Index of the y-axis physics parameter.
            size_t nbinsx;         ///< Number of grid points along the x axis.
            size_t nbinsy;         ///< Number of grid points along the y axis.
            Eigen::VectorXf edges_x; ///< Grid edges along the x axis (length nbinsx+1).
            Eigen::VectorXf edges_y; ///< Grid edges along the y axis (length nbinsy+1).
            Eigen::MatrixXf surface; ///< Profile chi-squared matrix (nbinsx × nbinsy).

            /**
             * @brief Per-grid-point result record.
             */
            struct SurfPointResult {
                int binx;              ///< x grid index.
                int biny;              ///< y grid index.
                Eigen::VectorXf best_fit; ///< Full best-fit parameter vector at this point.
                float chi2;            ///< Profile chi-squared at this point.
            };

            std::vector<SurfPointResult> results; ///< All grid-point results (filled by FillSurface).

            /**
             * @brief Axis spacing mode for surface grid construction.
             */
            enum LogLin {
                LinAxis, ///< Uniform linear spacing.
                LogAxis, ///< Uniform logarithmic spacing.
            };

            PROsurf(PROmetric &metric,  size_t x_idx, size_t y_idx, size_t nbinsx, const Eigen::VectorXf &edges_x, size_t nbinsy, const Eigen::VectorXf &edges_y) : metric(metric), x_idx(x_idx), y_idx(y_idx), nbinsx(nbinsx), nbinsy(nbinsy), edges_x(edges_x), edges_y(edges_y), surface(nbinsx, nbinsy) { }

            PROsurf(PROmetric &metric, size_t x_idx, size_t y_idx, size_t nbinsx, LogLin llx, float x_lo, float x_hi, size_t nbinsy, LogLin lly, float y_lo, float y_hi);

            /// @param seed_pts Warm-start seeds tried at EVERY grid fit (in addition to the
            /// nearest-already-fitted point) — typically the global best fit plus the harmonic
            /// freq_seed_points, mirroring PROfile's per-scan-fit seeding. Each seed costs one
            /// LBFGSB pass per grid point.
            std::vector<surfOut> PointHelper(const PROfitterConfig &fitconfig, std::vector<surfOut> multi_physics_params, std::atomic<int> *point_counter, uint32_t seed, const std::vector<Eigen::VectorXf> &seed_pts, MultiPROgressBar* progressbar = nullptr);

            void FillSurfaceStat(const PROconfig &config, const PROfitterConfig &fitconfig, std::string filename, const Eigen::VectorXf &cv_params, uint32_t seed);
            void FillSurface(const PROfitterConfig &fitconfig, std::string filename, PROseed & proseed, float min_chi, const std::vector<Eigen::VectorXf> &seed_pts, int nthreads = 1);

            /**
             * @brief Adaptive-mesh-refinement surface scan.
             * @details Replaces the fixed 60×60-style grid scan with `PROmesh::run_amr`. Each
             * AMR grid point is evaluated by a per-thread `PROfitter::Fit` call (via a
             * thread-local metric clone) using the AMR-supplied warm-start seeds; the
             * per-point fit body is the shared `PROmesh::pinned_scan_eval`
             * (inc/PROmeshEval.h), also used by the adaptive-FC Wilks prepass. After AMR
             * converges, the sparse evaluated map is written to a text file (one
             * (xphys, yphys, χ²) row per evaluated point), polyline contours are returned for
             * each level in `opts.contour_levels`, and the optional bilinear-reconstructed
             * dense matrix is copied into `surface(nbinsx, nbinsy)` for plot-compat.
             */
            PROmesh::AMRResult FillSurfaceAMR(
                const PROfitterConfig &fitconfig,
                std::string filename,
                PROseed &proseed,
                int nthreads,
                const std::vector<Eigen::VectorXf> &caller_seeds = {},
                const PROmesh::AMROptions &opts = {});

            /**
             * @brief Render the AMR mesh as a "boxes shrinking around the contour" plot.
             * @details Thin wrapper: delegates to the shared
             * `PROmesh::draw_amr_mesh_on_canvas` (inc/PROmeshPlot.h) — level-coloured
             * translucent boxes with opaque outlines (a fill+outline TBox pair per leaf),
             * contour polylines, and an info box with total and per-level fit counts —
             * then prints to `<filename>_amr_mesh.pdf`.
             */
            void PlotAMRMesh(const PROmesh::AMRResult &amr,
                             const PROmodel &model,
                             std::string filename,
                             bool logx, bool logy,
                             size_t xaxis_idx, size_t yaxis_idx);

            std::vector<surfOut> FillCurve(const PROfitterConfig &fitconfig, PROseed &proseed, float min_chi, const std::vector<Eigen::VectorXf> &seed_pts, int nThreads, std::vector<float> &A, std::vector<float> &B, size_t n_points);
            void PlotCurve(const PROconfig &config, const PROmodel &model, const PROsyst &syst, const std::vector<surfOut> & cpoints, std::string final_output_tag, bool logx, bool logy,size_t xaxis_idx,size_t yaxis_idx,std::vector<float> &A, std::vector<float> &B, size_t n_points);

    };

}

#endif

