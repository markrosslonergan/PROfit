/**
 * @file PROsyst.h
 * @brief Systematic uncertainty management for the PROfit framework.
 * @author PROfit Collaboration
 *
 * @details Defines PROsyst, which aggregates all systematic uncertainties (spline-based
 * and covariance-matrix-based) and provides methods to evaluate their effects on predicted
 * spectra.  Spline systematics are stored as per-bin piecewise-cubic splines whose knots
 * and coefficients are built from multi-universe Monte Carlo throws.  Covariance systematics
 * are stored as fractional covariance matrices.
 *
 * Also defines helper types SplineSegment and Spline used internally for the spline evaluation.
 */
#ifndef PROSYST_H_
#define PROSYST_H_

//C++ include 
#include <Eigen/Eigen>
#include <string>
#include <vector>
#include <map>
#include <cmath>

//annoying root headers for file input
#include "TMatrixD.h"
#include "TFile.h"

// Our include
#include "PROconfig.h"
#include "PROcreate.h"
#include "PROlog.h"
#include "PROmodel.h"

namespace PROfit {

    /**
     * @brief A single knot and its four cubic-spline coefficients for one bin segment.
     */
    struct SplineSegment {
        float knot;                  ///< Knob value at which this segment begins.
        std::array<float, 4> coeffs; ///< Cubic polynomial coefficients [c0, c1, c2, c3] for the segment.
    };

    /**
     * @brief Per-systematic piecewise-cubic spline used to evaluate bin-weight shifts.
     * @details Stores segments_per_bin cubic segments per analysis bin.  To evaluate the
     * weight shift for bin @p b at nuisance shift @p x, locate the correct segment and
     * evaluate the polynomial.  Access: segments[bin * segments_per_bin + seg].
     */
    class Spline {
        public:
            int bins;             ///< Number of analysis bins covered by this spline.
            int segments_per_bin; ///< Number of cubic segments per bin (equal to number of knot intervals).
            std::vector<SplineSegment> segments; ///< Flat list of spline segments, ordered bin-major.
            float knot_hi = 0.0f; ///< Upper end of the last segment. Kept apart from spline_hi, which --fix overwrites with the fit bound.
    };

    /**
     * @brief Splines whose joint response is a quadratic form rather than a product.
     * @details Built from a "spline_cross_quad" systematic, and valid ONLY for a joint response of
     * degree <= 2 in its members -- a quadratic form is all this type can represent. 
     * FillSplineCrossQuad checks the premise against the surplus on-axis
     * knots and warns if it fails. The members' own splines supply the
     * on-axis response s_i(eta_i); cross(bin, p) is the eta_i eta_j coefficient of the p-th
     * pair (i<j over members, itertools.combinations order). The group's factor in the
     * systematic-weight product is 1 + sum_i (s_i - 1) + sum_{i<j} cross_ij eta_i eta_j,
     * exact for any per-bin response that is a polynomial of degree <= 2 in the members.
     */
    struct QuadraticSplineGroup {
        std::vector<int> members;   ///< Spline indices, in the order the pairs are enumerated.
        Eigen::MatrixXf cross;      ///< (bins, npairs) cross coefficients e_ij per bin.
    };

    /**
     * @brief Diagnostic info captured when a "covariance_to_spline" systematic is processed.
     * @details Populated by FillSplinesFromCovariance; consumed by plotCov2SplineChecks to
     * make a covariance_to_spline_checks.pdf debug document.
     */
    struct Cov2SplineDebugInfo {
        Eigen::MatrixXf original_frac_cov;  ///< Fractional covariance built from multisim throws (pre-symmetrized residual saved separately).
        float pre_symm_asymmetry = 0.0f;     ///< ||C - C^T||_F before symmetrization (sanity number).
        Eigen::VectorXf eigenvalues;         ///< All eigenvalues, ascending (Eigen convention).
        Eigen::MatrixXf eigenvectors;        ///< Eigenvectors as columns (Eigen convention).
        std::vector<int> kept_indices;       ///< Indices into eigenvalues/eigenvectors for retained modes, descending eigenvalue.
        std::vector<std::string> knob_names; ///< Names of synthesized spline knobs, same order as kept_indices.
        int binning = -1;                    ///< Binning index the covariance lives on.
        bool has_residual = false;           ///< True if the un-kept eigenpairs were retained as a residual covariance.
        int n_residual_modes = 0;            ///< Number of positive eigenpairs folded into the residual covariance.
        Eigen::MatrixXf residual_cov;        ///< Residual fractional covariance from un-kept positive eigenpairs (empty if has_residual is false).
        std::string residual_cov_name;       ///< Name under which the residual covariance was registered ("<systname>_resid_cov").
    };

    /**
     * @brief Aggregator for all systematic uncertainties acting on a PROfit analysis.
     * @details PROsyst groups both spline-based and covariance-matrix-based systematics.
     * On construction it processes the input SystStruct vector and:
     *   - builds piecewise-cubic splines for spline-type systematics,
     *   - generates or loads fractional covariance matrices for covariance-type systematics,
     *   - accumulates a total fractional_covariance from all covariance-type systematics.
     *
     * During fitting, GetSplineShift() evaluates spline weights and
     * DecomposeFractionalCovariance() provides the Cholesky decomposition of the total
     * covariance for correlated throws.
     */
    class PROsyst {
        public:

            /**
             * @brief Enumeration of supported systematic types.
             */
            enum class SystType {
                Spline,     ///< Piecewise-cubic spline systematic built from multi-universe MC.
                Covariance, ///< Fractional covariance matrix (external or generated from multi-universe MC).
                MFA         ///< Multi-variate Frequentist Analysis covariance type.
            };

            /** @brief Default constructor — creates an empty PROsyst. */
            PROsyst(){}

            /**
             * @brief Primary constructor that builds all systematics from a list of SystStructs.
             * @param prop        MC event store (used for spline building).
             * @param config      Analysis configuration.
             * @param systs       Vector of SystStruct objects; one per systematic variation.
             * @param shapeonly   Shape-only mode. The systematics stay PHYSICAL: shape-only is done
             *                    by the metric (per-channel rescale onto data + ShapeProjectorCollapsed
             *                    about the current prediction, i.e. MiniBooNE's M_shape) and by the
             *                    band builders. Here it only projects covariances onto per-channel
             *                    shape, about the physical null prediction, before they are
             *                    decomposed into spline knobs (FillSplinesFromCovarianceMatrix).
             * @param other_index Variable index for which to build systematics (-1 = primary).
             * @param model       Physics model (used when converting splines to covariance).
             * @param params      Physics parameter vector for CV spectrum evaluation.
             */
            PROsyst(const PROpeller &prop, const PROconfig &config,
                const std::vector<SystStruct>& systs,
                bool shapeonly=false, int other_index = -1,
                const PROmodel* model = nullptr, const Eigen::VectorXf* params = nullptr);

            /**
             * @brief Return a new PROsyst containing only the named systematics.
             * @details The mcstat covariance is included only if its name is in @p systs.
             * Every name must be registered (see HasSyst()); an unknown one throws.
             * @param systs  List of systematic names to include.
             * @return Subset PROsyst.
             */
            PROsyst subset(const std::vector<std::string> &systs) const;

            /**
             * @brief Return a new PROsyst with the named systematics removed.
             * @details The mcstat covariance (registered only internally, not in covar_names)
             * is kept unless its name is in @p systs. Unknown names are ignored — callers
             * validate with HasSyst().
             * @param systs  List of systematic names to exclude.
             * @return Complement PROsyst.
             */
            PROsyst excluding(const std::vector<std::string> &systs) const;

            /** @brief True if @p name is a registered spline or covariance (incl. mcstat). */
            bool HasSyst(const std::string &name) const { return syst_map.count(name) > 0; }

            /**
             * @brief Convert all spline systematics to covariance matrices and return the result.
             * @param config  Analysis configuration.
             * @param prop    MC event store.
             * @param model   Physics model used for spectrum filling.
             * @param params  Physics parameter vector for CV evaluation.
             * @param seed    Random seed for Gaussian throws used in the conversion.
             * @return A new PROsyst with all splines replaced by equivalent covariance matrices.
             */
            PROsyst allsplines2cov(const PROconfig &config, const PROpeller &prop,const PROmodel &model, const Eigen::VectorXf &params,  uint32_t seed) const;

            /**
             * @brief Convert a single spline systematic to a covariance matrix via random throws.
             * @param spline  0-based index of the spline to convert.
             * @param config  Analysis configuration.
             * @param prop    MC event store.
             * @param model   Physics model.
             * @param params  Physics parameter vector.
             * @param seed    Random seed.
             * @return Fractional covariance matrix for the specified spline.
             */
            Eigen::MatrixXf spline2cov(int spline, const PROconfig &config, const PROpeller &prop, const PROmodel &model, const Eigen::VectorXf &params, uint32_t seed) const ;

            /// Number of random throws spline2cov draws per spline (--syst-throws; default 500).
            inline static size_t spline2cov_throws = 500;
            /// Worker threads spline2cov spreads its throws over (-n; result is thread-count independent).
            inline static size_t spline2cov_nthreads = 1;

            /* Function: given the systematic name, return corresponding fractional covariance matrix */
            Eigen::MatrixXf GrabMatrix(const std::string& sys) const;
            Eigen::MatrixXf GrabCorrMatrix(const std::string& sys) const;

            /* Function: given the systematic name, return corresponding Spline */
            Spline GrabSpline(const std::string& sys) const;

            /* Function: given systematic name, return type of systematic */
            SystType GetSystType(const std::string& syst) const;

            /** @brief Return the number of spline systematics in this PROsyst. */
            size_t GetNSplines() const { return splines.size(); }

            /** @brief Return the number of spline systematics (non-const overload). */
            size_t GetNSplines() { return splines.size(); }

            /** @brief Return the number of covariance-matrix systematics in this PROsyst. */
            size_t GetNCovar() const { return n_covar; }

            /** @brief True when built in shape-only mode (per-channel shape projection applied). */
            bool ShapeOnly() const { return shape_only; }

            /**
             * @brief Contiguous UNcollapsed-bin block (start, length) of every collapsed channel.
             * @details One entry per (mode, detector, channel) in collapsed-channel order; the block
             * spans all of that channel's subchannels for binning index @p binning. This is the
             * domain over which shape-only normalisation is defined: it matches the per-channel
             * data/prediction rescale done by the metrics (ChannelNormFactors in PROtocall).
             */
            static std::vector<std::pair<size_t,size_t>> ChannelBlocks(const PROconfig &config, int binning);

            /**
             * @brief In place: rescale each channel block of @p var so its integral equals the CV's.
             * @details f_a = sum_a(cv)/sum_a(var) per block a; a block with sum_a(var)==0 is left untouched.
             * A shape-only variation built this way has zero net change per channel exactly.
             */
            static void ShapeRescaleUniverse(const std::vector<std::pair<size_t,size_t>> &blocks, const Eigen::VectorXf &cv, Eigen::VectorXf &var);

            /**
             * @brief Fractional-space shape projector R for a covariance in binning with nominal @p cv.
             * @details R = I - B with B_ij = N_j / N_a for i, j in the same channel block a (N = cv,
             * N_a = block sum), zero across blocks. For a fractional covariance F the shape-only
             * matrix is F_shape = R F R^T, which equals the classic
             * M_shape = M - M_mixed - M_norm on the absolute matrix diag(N) F diag(N), per channel.
             * R annihilates a pure per-channel normalisation (F = s^2 1 1^T on a block) exactly and
             * is idempotent on a covariance whose universes were already ShapeRescaleUniverse'd.
             */
            static Eigen::MatrixXf ShapeProjector(const std::vector<std::pair<size_t,size_t>> &blocks, const Eigen::VectorXf &cv);

            /**
             * @brief In place: project every covariance matrix onto per-channel shape about @p cv.
             * @details Applies ShapeProjector (blocks of this PROsyst's binning) to each covmat,
             * refreshes the correlation matrices and fractional_covariance. For shape-only
             * DISPLAYS (fractional breakdowns, covariance plots) of a physical PROsyst; the
             * chi^2 never needs it (the metric projects about the current prediction).
             * @param config Analysis configuration.
             * @param cv     Uncollapsed spectrum (binning other_index) to project about.
             */
            void ProjectCovariancesOntoShape(const PROconfig &config, const Eigen::VectorXf &cv);

            // ----- apply_to_subchannel scoping (post-build, type-agnostic) -----
            //
            // A systematic's XML apply_to_subchannel= pattern is honoured in two layers:
            // PROcreate fills non-matching events at the CV weight (so the weight branch is
            // never read where it is not needed), and — independently of how the
            // systematic was built (weight universes, DetVar, HistVar, flat, norm, external
            // matrix, mcstat) — the ctor finishes with ApplySubchannelScopes(), which
            // flattens every spline and zeroes every covariance row/column outside the
            // matched subchannels. The second layer is idempotent on the first, so the
            // guarantee "non-matching subchannels get exactly no response" holds for every
            // systematic type, including ones whose spectra never pass through PROcreate.

            /** @brief Per-uncollapsed-bin scope mask (1 = bin belongs to a subchannel matching
             *  @p pattern, 0 otherwise) for binning index @p binning. All 1 when the pattern is empty. */
            static std::vector<char> SubchannelScopeMask(const PROconfig &config, int binning, const std::string &pattern);

            /** @brief Zero every row and column of the (fractional or absolute) matrix @p M whose
             *  bin lies outside the pattern's scope in @p binning. No-op for an empty pattern. */
            static void ScopeMatrixToSubchannels(const PROconfig &config, int binning, const std::string &pattern, Eigen::MatrixXf &M);

            /** @brief Normalise a SystStruct's own spectra to its apply_to_subchannel: every
             *  universe/knob spectrum gets the CV value on out-of-scope bins (ratio exactly 1,
             *  deviation exactly 0). Idempotent; a no-op without a pattern or without spectra.
             *  The spectra are shared (shared_ptr) so the caller's struct is updated too.
             *  @p spectra_binning is the binning the struct's spectra actually live in: the
             *  systematic's own binning for spline-type modes, but the PROsyst's variable
             *  index for mode "covariance" (each variable's copy is filled in its own bins). */
            static void ScopeSystStructSpectra(const PROconfig &config, const SystStruct &syst, int spectra_binning);

            /** @brief Post-build stage: for every spline and covariance whose systematic (or,
             *  for `<name>_decomp_knob_i` / `<name>_resid_cov` children, whose parent) carries an
             *  apply_to_subchannel pattern — from @p systs or from the config map (mcstat,
             *  DetVar) — flatten the spline / zero the covariance outside the matched
             *  subchannels and rebuild the correlation matrix. Called at the end of the ctor. */
            void ApplySubchannelScopes(const PROconfig &config, const std::vector<SystStruct> &systs);

            //----- Spline and Covariance matrix related ---
            //----- Spline and Covariance matrix related ---

            Eigen::MatrixXf SumMatrices() const;
            Eigen::MatrixXf SumMatrices(const std::vector<std::string>& sysnames) const;

            /* Function: given a SystStruct with cv and variation spectra, build full covariance matrix for the systematics, and return it
             * Note: it assumes the SystStruct is filled 
             */
            static Eigen::MatrixXf GenerateFullCovarMatrix(const SystStruct& sys_obj, const std::vector<std::pair<size_t,size_t>> *shape_blocks = nullptr);

            /* Function: Given a SystStruct, generate fractinal covariance matrix, and correlation matrix, and add matrices to covmat_map and corrtmat_map
             * Note: this function is lazy. It wouldn't do anything if it found covariance matrix with the same name already in the map.
             */
            void CreateMatrix(const SystStruct& syst);

            /* Function: Given a SystStruct, generate a FLAT norm fractinal covariance matrix, and correlation matrix, and add matrices to covmat_map and corrtmat_map
             * Note: this function is lazy. It wouldn't do anything if it found covariance matrix with the same name already in the map.
             */
            void CreateFlatMatrix(const PROconfig& config, const SystStruct& syst);

            /* Function: Build a fully correlated fractional covariance over the selected bins. */
            void CreateNormMatrix(const PROconfig& config, const SystStruct& syst);

            /* Function: Given a SystStruct, load an external fractinal covariance matrix, and calculate correlation matrix, and add matrices to covmat_map and corrtmat_map
             */
            void LoadExternalCovarianceMatrix(const PROconfig& config, const SystStruct& syst);

            /* Function: Open syst.external_filename and read the named TMatrixD into an Eigen fractional
             * covariance matrix (sized to syst.binning), zeroing non-finite entries and warning if it is
             * not positive semi-definite. Shared by the "external_covariance" and
             * "external_covariance_to_spline" modes. */
            Eigen::MatrixXf LoadExternalFractionalCovariance(const PROconfig& config, const SystStruct& syst);

            /* Function: given a syst struct with cv and variation spectra, build fractional covariance matrix for the systematics, as well as correlation matrix 
             * Return: {fractional covariance matrix, correlation covariance matrix}
             */
            static std::pair<Eigen::MatrixXf, Eigen::MatrixXf> GenerateCovarMatrices(const SystStruct& sys_obj, const std::vector<std::pair<size_t,size_t>> *shape_blocks = nullptr);

            /* Function: given a SystStruct with cv and variation spectra, build fractional covariance matrix for the systematics, and return it
             * Note: it assumes the SystStruct is filled 
             */
            static Eigen::MatrixXf GenerateFracCovarMatrix(const SystStruct& sys_obj, const std::vector<std::pair<size_t,size_t>> *shape_blocks = nullptr);

            /* Given fractional covariance matrix, calculate the correlation matrix */
            static Eigen::MatrixXf GenerateCorrMatrix(const Eigen::MatrixXf& frac_matrix);

            /* Function: check if matrix has nan, or infinite value */
            static bool isFiniteMatrix(const Eigen::MatrixXf& in_matrix);

            /* Function: if matrix has nan/inf values, change to 0. 
             * Note: this modifies the matrix !! 
             */
            static void toFiniteMatrix(Eigen::MatrixXf& in_matrix);

            /* Function: check if given matrix is positive semi-definite with tolerance. UST THIS ONE!!*/
            static bool isPositiveSemiDefinite_WithTolerance(const Eigen::MatrixXf& in_matrix, float tolerance=1.0e-16);

            /* Function: check if given matrix is positive semi-definite, no tolerance at all (besides precision error from Eigen) */
            static bool isPositiveSemiDefinite(const Eigen::MatrixXf& in_matrix);

            /* Function: Fill splines assuming p_cv and p_multi_spec have been filled in the SystStruct*/
            void FillSpline(const SystStruct& syst, bool unmirrored);

            /* Function: For a "binned_unconstrained" systematic, synthesize one exactly-linear spline
             * per local bin of syst.binning: parameter theta = scale-1 (CV at 0), weight 1+theta in the
             * owned bins of every subchannel listed in syst.norm_bins and exactly 1 elsewhere. Each
             * spline is named "<syst.systname>_bin<j>", carries a Uniform prior (no pull) and is bounded
             * by [restrict_lo, restrict_hi]. No universes are needed: nothing is read from the MC. */
            void FillBinnedUnconstrainedSplines(const PROconfig& config, const SystStruct& syst);

            /* Function: For a "covariance_to_spline" systematic, build the fractional covariance from the
             * multi-universe spectra, eigendecompose it, and synthesize one linear spline per retained
             * eigenpair. Knobs are named "<syst.systname>_decomp_knob_<i>" where i = 0 corresponds to the
             * largest eigenvalue. If syst.num_decomp_knobs > 0, only the top N eigenpairs are kept. */
            void FillSplinesFromCovariance(const SystStruct& syst);

            /* Function: Eigendecompose an already-built fractional covariance matrix and synthesize one
             * linear spline per retained eigenpair (the shared core of FillSplinesFromCovariance). Used
             * by both "covariance_to_spline" (matrix from MC universes) and "external_covariance_to_spline"
             * (matrix loaded from an external TMatrixD). */
            void FillSplinesFromCovarianceMatrix(Eigen::MatrixXf frac_cov, const SystStruct& syst,
                                                 SplinePriorType prior = SplinePriorType::Gaussian,
                                                 float knob_lo = -3.0f, float knob_hi = 3.0f);

            /* Function: after a parent systematic has been expanded into derived splines/covariances
             * (indices [n_spl_before, n_splines) and [n_cov_before, covar_names.size())), copy the
             * parent's tag list and plotname (with the derived suffix) onto every derived name and
             * record them as the parent's children, so --fix/--syst-list/--exclude-systs, PROplot and
             * PROsurf can all resolve the names. Writes into the (nominally const) PROconfig. */
            void PropagateDerivedNames(const PROconfig& config, const std::string& parent, size_t n_spl_before, size_t n_cov_before);

            /* Function: Get weight for bin for a given shift using spline */
            float GetSplineShift(int syst_num, float shift, int bin) const;
            float GetSplineShift(std::string name, float shift, int bin) const;

            /* Function: Analytic derivative d(weight)/d(shift) of GetSplineShift for the
             * same (spline, shift, bin). Evaluates the derivative of the cubic segment
             * GetSplineShift would use, so the two are consistent everywhere including
             * beyond the outermost knots. Returns 0 for an out-of-range bin. */
            float GetSplineShiftDeriv(int syst_num, float shift, int bin) const;

            /* Function: For a "spline_cross_quad" systematic, resolve its member splines and turn its
             * universes (entry 0 = CV, entry 1+p = joint +1 sigma shift of pair p) into the per-bin
             * cross coefficients of a QuadraticSplineGroup. Members must already be built. */
            void FillSplineCrossQuad(const SystStruct& syst, const PROconfig& config);

            /* Function: The factor spline i contributes to the multiplicative systematic weight at
             * (all shifts, bin). This is the generic accessor every spline goes through, grouped or
             * not: an ungrouped spline returns GetSplineShift(i, shifts(i), bin) unchanged. For a
             * QuadraticSplineGroup the first member carries the whole group response and the others
             * return 1, so systw = prod_i GetSplineFactor(i) stays the combination rule everywhere. */
            float GetSplineFactor(int i, const Eigen::VectorXf& shifts, int bin) const;

            /* Function: d(group or spline factor)/d(shift_i), consistent with GetSplineFactor. For a
             * group member this is the derivative of the group response, which lives in column
             * FactorColumn(i) of the factor product. */
            float GetSplineFactorDeriv(int i, const Eigen::VectorXf& shifts, int bin) const;

            /* Function: Index of the factor-product column that depends on shift_i: i itself, or the
             * first member of i's QuadraticSplineGroup. */
            int FactorColumn(int i) const;

            /* Function: Get cv spectrum shifted using spline */
            PROspec GetSplineShiftedSpectrum(const PROconfig& config, const PROpeller& prop, std::string name, float shift) const;
            PROspec GetSplineShiftedSpectrum(const PROconfig& config, const PROpeller& prop, int syst_num, float shift) const;
            PROspec GetSplineShiftedSpectrum(const PROconfig& config, const PROpeller& prop, std::vector<std::string> names, std::vector<float> shifts) const;
            PROspec GetSplineShiftedSpectrum(const PROconfig& config, const PROpeller& prop, std::vector<int> syst_nums, std::vector<float> shifts) const;
            PROspec GetSplineShiftedSpectrum(const PROconfig& config, const PROpeller& prop, std::vector<float> shifts) const;

            Eigen::MatrixXf DecomposeFractionalCovariance(const PROconfig &config, const Eigen::VectorXf &cv_vec) const;

            /** @brief Full-bin (uncollapsed) analogue of DecomposeFractionalCovariance.
             *
             *  Returns a sampler matrix A (nbins_full x nbins_full, zero-padded columns)
             *  with A*A^T ~= diag(cv_vec) * fractional_covariance * diag(cv_vec), i.e. the
             *  absolute covariance in full (subchannel) bin space, NOT collapsed. Throwing
             *  A*z with z ~ N(0,1)^nbins and collapsing afterwards is distributed
             *  identically to the collapsed-space throw from DecomposeFractionalCovariance;
             *  the full-space version exists so a throw can be split into subchannel
             *  pieces (e.g. per-throw background subtraction in plotting) before collapse.
             *  Cached like the collapsed variant. */
            Eigen::MatrixXf DecomposeFractionalCovarianceFull(const PROconfig &config, const Eigen::VectorXf &cv_vec) const;

            void PrintSplines();

            /** @brief Total fractional covariance matrix summed over all covariance-type systematics. */
            Eigen::MatrixXf fractional_covariance;

            std::vector<std::string> spline_names;   ///< Names of all spline systematics in order.
            std::vector<std::string> covar_names;    ///< Names of all covariance systematics in order.
            std::vector<float> spline_lo;            ///< Lower nuisance-parameter bound for each spline.
            std::vector<float> spline_hi;            ///< Upper nuisance-parameter bound for each spline.
            std::vector<bool> spline_has_restrict;   ///< Whether each spline has an explicit restrict range.
            std::vector<float> spline_restrict_lo;   ///< Lower clamp bound for each spline (used only when spline_has_restrict is true).
            std::vector<float> spline_restrict_hi;   ///< Upper clamp bound for each spline (used only when spline_has_restrict is true).
            std::vector<int> spline_binnings;        ///< Binning-scheme index for each spline.
            Eigen::VectorXf spline_priors;           ///< Prior width (sigma) for each spline nuisance parameter.
            Eigen::VectorXf spline_centers;          ///< Prior centre for each spline nuisance parameter.
            /// True for each spline that acts on the parent neutrino's truth-E (beam flux systematics).
            /// These are applied before the decay energy migration for 3+1+visible decay in FillSpectra.
            std::vector<bool> spline_is_pre_migration;

            std::vector<SplinePriorType> spline_prior_types; ///< Prior model for each spline nuisance parameter.
            std::vector<QuadraticSplineGroup> quadratic_groups;   ///< Cross-term groups built from "spline_cross_quad" systematics.
            std::vector<int> quadratic_group_of;         ///< Per spline: index into quadratic_groups, or -1.
            bool has_external_prior_cov = false;     ///< If true, metrics use external_prior_cov as a fully correlated Gaussian prior (PROjector).
            Eigen::MatrixXf external_prior_cov;      ///< Absolute prior covariance over the spline nuisance parameters (used with spline_centers).
            std::map<std::string, Cov2SplineDebugInfo> cov2spline_debug_info; ///< Debug info per "covariance_to_spline" systematic, keyed by parent systname.
        private:
            /* Carry quadratic_groups into a subset/excluding copy; new_index[old spline] = new index or -1.
             * A group with a removed member is dropped (its survivors revert to the plain product). */
            void CopyQuadraticGroups(PROsyst& ret, const std::vector<int>& new_index) const;
            std::map<std::string, std::pair<size_t, SystType>> syst_map; ///< Map from systematic name to (index, type).
            std::vector<Spline> splines;             ///< Ordered list of spline objects.
            size_t n_splines = 0;                    ///< Number of spline systematics.
            size_t n_covar = 0;                      ///< Number of covariance-matrix systematics.
            std::vector<Eigen::MatrixXf> covmat;     ///< Fractional covariance matrices, one per covariance systematic.
            std::vector<Eigen::MatrixXf> corrmat;    ///< Correlation matrices, one per covariance systematic.
            int other_index;                         ///< Variable index for which systematics were built.
            bool shape_only = false;                 ///< Shape-only mode (per instance; copied by subset/excluding/allsplines2cov).
            std::map<int, std::vector<std::pair<size_t,size_t>>> shape_blocks;  ///< binning index -> ChannelBlocks (built in the ctor when shape_only).
            std::map<int, Eigen::VectorXf> shape_nominal;                        ///< binning index -> nominal spectrum defining the projector (ctor, shape_only).
            /// Blocks / nominal spectrum for a SystStruct's binning (binning<0 means other_index).
            const std::vector<std::pair<size_t,size_t>>& shapeBlocksFor(const SystStruct &syst) const { return shape_blocks.at(syst.binning < 0 ? other_index : syst.binning); }
            const Eigen::VectorXf& shapeNominalFor(const SystStruct &syst) const { return shape_nominal.at(syst.binning < 0 ? other_index : syst.binning); }
            mutable Eigen::VectorXf last_decomp_spec; ///< Cached CV spectrum from last DecomposeFractionalCovariance call.
            mutable Eigen::MatrixXf last_decomp_mat;  ///< Cached Cholesky factor from last DecomposeFractionalCovariance call.
            mutable Eigen::VectorXf last_decomp_full_spec; ///< Cached CV spectrum from last DecomposeFractionalCovarianceFull call.
            mutable Eigen::MatrixXf last_decomp_full_mat;  ///< Cached sampler from last DecomposeFractionalCovarianceFull call.
    };

};

#endif
