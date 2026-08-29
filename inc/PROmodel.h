/**
 * @file PROmodel.h
 * @brief Abstract physics oscillation model interface for PROfit.
 * @author PROfit Collaboration
 *
 * @details Defines the abstract PROmodel base class and declares the
 * get_model_from_string factory (defined in src/PROmodel.cxx). Each model
 * exposes a get_probs() virtual method that returns oscillation probabilities
 * on the physics grid, and pre-builds H_combined matrices for fast GEMV-based
 * spectrum filling in FillSpectra.
 *
 * Concrete models live in family files; only the factory constructs them
 * (exception: NullModel, used directly by bin/PROfit.cxx):
 *   - inc/PROmodels/PROmodelSimple.h — NullModel, PROtemplate (non-oscillation).
 *   - inc/PROmodels/PROmodel2flav.h  — PROnumudis, PROnumudisTEST, PROnueapp, PROnuedis.
 *   - inc/PROmodels/PROmodel3p1.h    — PRO3p1 and variants (_angles, _3A, _3B, _3C, _decay_invis).
 *   - inc/PROmodels/PROmodel3p2.h    — PRO3p2.
 *   - inc/PROmodels/PROmodelLBL.h    — PROLBL (NuFastLBL three-flavour matter oscillations).
 */
#ifndef PROMODEL_H
#define PROMODEL_H

#include "PROconfig.h"
#include "PROpeller.h"

#include <Eigen/Eigen>

#include <cstdlib>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace PROfit {

/**
 * @brief Abstract base class representing a physics model for neutrino oscillation probability.
 * @details A PROmodel encapsulates:
 *   - the set of physics parameters (names, bounds, defaults),
 *   - the mapping from analysis variables to a flat physics grid,
 *   - per-event histogram matrices H_combined used for fast spectrum filling, and
 *   - the virtual get_probs() interface for computing oscillation probabilities.
 *
 * Derived classes implement specific oscillation hypotheses.  After construction,
 * build_hists_and_combined() must be called to pre-build the internal matrices from
 * the MC event store (PROpeller).
 */
class PROmodel {
public:
    size_t nparams; ///< Number of physics parameters for this model.
    /// Indices of the physics (grid) variables used by this model within the PROpeller variable list.
    /// For a 1-variable model this is e.g. {L/E_index}; for 2-variable {L_index, E_index}.
    /// n_phys_bins = product of their bin counts.
    std::vector<int> ivars;
    long int n_phys_bins = 0; ///< Total number of flat physics-grid points (product of ivar bin counts).
    std::vector<std::string> param_names;        ///< Internal parameter names used in the fitter.
    std::vector<std::string> pretty_param_names; ///< LaTeX-formatted parameter names for plots.
    std::vector<std::string> pretty_param_units; ///< Unit strings for plots (e.g., "eV^{2}").
    Eigen::VectorXf lb;          ///< Lower bounds for physics parameters in the fitter's internal space.
    Eigen::VectorXf ub;          ///< Upper bounds for physics parameters in the fitter's internal space.
    Eigen::VectorXf default_val; ///< Default (starting) values for physics parameters.
    /// Per-probability-type functions: model_functions[m](phys, x) returns the oscillation weight
    /// for physics parameters @p phys and kinematic variable value @p x.
    std::vector<std::function<float(const Eigen::VectorXf&, float)>> model_functions;
    std::function<int(const Eigen::VectorXf&)> model_constraint; ///< Optional parameter constraint function.
    /// Pre-binned histograms: hists[v][m] has shape (n_reco_v, n_phys_bins).
    /// v = reco variable index, m = probability-type index.  Transposed for cache efficiency.
    std::vector<std::vector<Eigen::MatrixXf>> hists;
    /// Combined histograms H_combined[v] = horizontal concatenation of hists[v][0..J-1],
    /// shape (n_reco_v, n_phys_bins * J).  Enables a single GEMV in FillSpectra.
    std::vector<Eigen::MatrixXf> H_combined;

    std::vector<size_t> prob_types; ///< Probability-type indices, matching model_functions indices.

    std::vector<bool> is_log10; ///< True for each parameter stored in log10 space; false for linear.

    /// Trivial models (e.g., NullModel) have no physics dependence: probabilities are identically 1.
    /// When true, FillSpectra skips var_arrs / get_probs / GEMV and reads the spectrum directly from
    /// H_combined. Also implies an empty `ivars` so events are binned per reco variable independently
    /// (no cross-variable validity coupling through a placeholder physics-grid variable).
    bool is_trivial = false;

    /**
     * @brief Build hists and H_combined from PROpeller event data.
     * @details Must be called after ivars and model_functions are set.  Iterates over all
     * events in @p prop, distributes them onto the flat physics grid, and constructs the
     * concatenated H_combined matrices used by FillSpectra.
     * @param prop                 The MC event store.
     * @param filter_by_model_rule If true (default), each event is placed in the histogram
     *                             matrix corresponding to its model_rule; if false, all events
     *                             go into component 0 (appropriate for NullModel).
     * @param block_fn             Optional override for column (probability-type / component)
     *                             routing. When set, the column for event @p i in reco variable
     *                             @p v (whose reco bin is @p rbin) is block_fn(v, i, rbin); a
     *                             negative return drops the event. Used by normalization models
     *                             (e.g. template_fit) that route by subchannel rather than by
     *                             model_rule. When empty, the model_rule logic above is used.
     *
     * @note Single pass over events per reco variable: each event lands in exactly one column,
     *       so this is equivalent to (and J times cheaper than) a per-column scan.
     */
    void build_hists_and_combined(const PROpeller &prop, bool filter_by_model_rule = true,
                                  const std::function<int(size_t, size_t, int)> &block_fn = {});

    /**
     * @brief Compute oscillation probabilities for all physics-grid points and all probability types.
     * @details Default implementation evaluates each model_function independently for every
     * grid point.  Derived classes may override for a vectorised, faster computation.
     * @param phys      Physics parameter vector in the fitter's internal space (log10 where applicable).
     * @param var_arrs  var_arrs[k] contains the value of ivars[k] for each flat grid point
     *                  (length = n_phys_bins).
     * @return Matrix of shape (n_phys_bins, n_prob_types) where each element is the oscillation
     *         weight for that grid point and probability type.
     */
    virtual Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const;

    /**
     * @brief Derivatives of get_probs with respect to each physics parameter.
     * @details grads[p] has the same shape as get_probs' return (n_phys_bins x
     * n_prob_types) and holds d(prob)/d(phys_p) in the fitter's INTERNAL space
     * (i.e. the log10 chain factor is included for is_log10 parameters). The
     * base-class default computes a central finite difference on get_probs —
     * well-conditioned since probabilities are O(1) — so every model supports
     * the analytic-gradient chi² mode. All sterile-family models (2-flavour,
     * 3+1 and its reparametrisations, 3+1 invisible decay, 3+2) and the
     * template model override with closed-form derivatives; PROLBL (external
     * matter-effect solver) keeps the finite-difference default.
     *
     * Conventions shared by all closed-form overrides:
     *  - Parameters arrive in the fitter's internal space θ. For a log10
     *    parameter the physical value is p = 10^θ, so d p/dθ = ln10 · p; for a
     *    linear parameter d p/dθ = 1. Each override computes d(prob)/d(physical
     *    p) and multiplies by this "chain factor" (named ddm, dss, dUe, ... in
     *    the code) to return d(prob)/dθ.
     *  - The oscillation phase is x = k · Δm² · (L/E) with k = 1.266932679
     *    (eV², km, GeV). Every probability is built from sin²x, and
     *    d(sin²x)/dx = 2 sin x cos x = sin 2x, so
     *    d(sin²x)/dΔm² = sin(2x) · k · (L/E); times the chain factor for Δm².
     *  - Where get_probs clamps a parameter (e.g. sin²2θ to [0,1]) the clamped
     *    value is locally constant, so its derivative is set to zero there.
     * @param phys      Physics parameter vector in the fitter's internal space.
     * @param var_arrs  Same flat-grid layout as get_probs.
     * @return One matrix per physics parameter.
     */
    virtual std::vector<Eigen::MatrixXf> get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const;

    std::unordered_map<std::string, size_t> param_name_to_index; ///< Fast lookup: parameter name -> index in param_names.

    /**
     * @brief Populate param_name_to_index from the current param_names vector.
     * @details Must be called after param_names is finalised in the derived constructor.
     */
    void build_param_index();

    /**
     * @brief Convert a named parameter from log10 to linear space if required.
     * @param param_name  The parameter name as listed in param_names.
     * @param value       The parameter value in the fitter's internal space.
     * @return The value in linear space: 10^value if is_log10[i] is true, otherwise value unchanged.
     * @details The const char* overload is the one string literals bind to; it does an
     * allocation-free linear scan over the (2-7 entry) name list — this runs inside
     * get_probs / model_constraint on every metric evaluation, where the previous
     * std::string temporary + hash lookup per call was measurable.
     */
    inline float maybe_convert_log(const char *param_name, float value) const {
        for(size_t i = 0; i < param_names.size(); ++i) {
            if(param_names[i] == param_name)
                return is_log10[i] ? std::pow(10.0f, value) : value;
        }
        log<LOG_ERROR>(L"%1% || Parameter name '%2%' not found in this model. Terminating.") % __func__ % param_name;
        exit(EXIT_FAILURE);
    }

    inline float maybe_convert_log(const std::string &param_name, float value) const {
        return maybe_convert_log(param_name.c_str(), value);
    }

    virtual ~PROmodel(){}

};

/**
 * @brief Factory function: construct a PROmodel subclass by name.
 * @details Reads `config.m_model_tag` to select the appropriate model and passes
 * `config.m_model_parameter_map` for variable-index lookup.
 * Supported names: "nullmodel", "numudis", "numudisTEST", "nueapp", "nuedis",
 * "3+1", "3+1_angles", "3+1_3A", "3+1_3B", "3+1_3C", "3+1_decay_invis", "3+2", "LBL".
 * Terminates with LOG_ERROR if the name is unrecognised.
 * @param config  Parsed configuration; provides the model tag and parameter map.
 * @param prop    MC event store used to build H_combined histograms.
 * @return        Owning pointer to the constructed PROmodel.
 */
std::unique_ptr<PROmodel> get_model_from_string(const PROconfig& config, const PROpeller &prop);

}

#endif

