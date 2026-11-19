/**
 * @file PROmodelSine.h
 * @brief Recipe-driven sine-kernel oscillation models (single Delta m^2, SBL approximation).
 * @author PROfit Collaboration
 * @internal These models are constructed only via
 * get_model_from_string (src/PROmodel.cxx); prefer the factory over direct construction.
 *
 * @details Every model whose channel probabilities have the shared form
 *
 *     P_c(theta; L/E) = 1                              (ProbForm::Null)
 *                     = A_c(theta) * sin^2 x           (ProbForm::Appearance)
 *                     = 1 - A_c(theta) * sin^2 x       (ProbForm::Disappearance)
 *
 * with x = 1.266932679 * Delta m^2 * (L/E), is expressed as a SineModelRecipe and
 * evaluated by the single PROsineModel class instead of a dedicated concrete class.
 * A recipe supplies the parameter table, the channel table, and two small lambdas
 * that compute the per-channel amplitudes (for get_probs) and the closed-form
 * amplitude derivatives (for get_probs_grad) once per call.
 *
 * Registered recipe tags (see find_sine_recipe / the factory in src/PROmodel.cxx).
 * Tags follow the regime_model_parameterization(_NC) convention; legacy pre-v3.1
 * names in [brackets] are still accepted as deprecated aliases (mapped, with a
 * one-time warning, by canonicalize_model_tag in src/PROmodel.cxx):
 *   - SBL_2flav_numudis [numudis], SBL_2flav_nueapp [nueapp], SBL_2flav_nuedis [nuedis]
 *                                     — single-channel two-flavour-like models.
 *   - SBL_3+1_Usq [3+1]              — four-channel {1, P_mumu, P_mue, P_ee} 3+1 in
 *                                       |U_e4|^2, |U_mu4|^2.
 *   - SBL_3+1_angles [3+1_angles]    — the same in sin^2(2theta_14), sin^2(theta_24).
 *   - SBL_3+1_sinsq2thee [3+1_3A], SBL_3+1_sinsq2thmumu [3+1_3B],
 *     SBL_3+1_sinsq2thmue [3+1_3C]   — 3+1 reparameterised around one headline
 *                                       amplitude (ee, mumu, mue respectively) plus a
 *                                       second shape parameter (sinsqth24, sB, xi).
 *   - SBL_2flav_numudis_NC [NCnumudisapp]
 *                                     — single-channel nu_mu -> nu_s NC disappearance,
 *                                       P = 1 - sin^2(2theta_mus) sin^2 x.
 *   - SBL_2flav_nudis_NC [NCdisapp]  — {1, P_mus, P_es} with independent phenomenological
 *                                       amplitudes sin^2(2theta_mus), sin^2(2theta_es)
 *                                       (no joint unitarity constraint between them).
 *   - SBL_3+1_Usq_NC, SBL_3+1_angles_NC, SBL_3+1_sinsq2thee_NC,
 *     SBL_3+1_sinsq2thmumu_NC, SBL_3+1_sinsq2thmue_NC
 *     [3+1_NC, 3+1_angles_NC, 3+1_3A_NC, 3+1_3B_NC, 3+1_3C_NC]
 *                                     — eight-channel NC-disappearance extensions of the
 *                                       five SBL_3+1 variants (see channel table below).
 *
 * Eight-channel NC models: the XML model_rule value routes each subchannel to a
 * probability column,
 *   0 = null (no oscillation)         4 = mus   (nu_mu -> nu_s, visible NC deficit)
 *   1 = mumu (nu_mu disappearance)    5 = es    (nu_e  -> nu_s, visible NC deficit)
 *   2 = mue  (nu_mu -> nu_e app.)     6 = mutau (nu_mu -> nu_tau appearance)
 *   3 = ee   (nu_e disappearance)     7 = etau  (nu_e  -> nu_tau appearance)
 * NC event rates are flavour-blind, so only the sterile fraction of the disappeared
 * flux is a real NC deficit (channels 4/5, A = 4 |U_a4|^2 |U_s4|^2) while the tau
 * fraction still interacts (channels 6/7, A = 4 |U_a4|^2 |U_tau4|^2). The sterile/tau
 * split of |U_s4|^2 + |U_tau4|^2 = c14^2 c24^2 = 1 - |U_e4|^2 - |U_mu4|^2 is fit in
 * each family's own parameter language: SBL_3+1_angles_NC, SBL_3+1_sinsq2thee_NC and
 * SBL_3+1_sinsq2thmumu_NC use cosq34 = cos^2 theta_34 (linear, [0,1], default 1 =
 * all-sterile), while SBL_3+1_Usq_NC and SBL_3+1_sinsq2thmue_NC fit
 * Uta4sq = |U_tau4|^2 directly (log10) with
 * |U_s4|^2 = max(0, 1 - sum |U_a4|^2).
 */
#ifndef PROMODELSINE_H
#define PROMODELSINE_H

#include "PROmodel.h"

#include <array>
#include <cstdint>

namespace PROfit {

/** @brief Functional form of one probability column in the shared sine kernel. */
enum class ProbForm : uint8_t {
    Null,           ///< P = 1 (no oscillation / CV column).
    Disappearance,  ///< P = 1 - A * sin^2 x.
    Appearance      ///< P = A * sin^2 x.
};

/**
 * @brief Evaluation order of the Delta m^2 gradient column.
 * @details The legacy concrete models used two different (bitwise-inequivalent)
 * floating-point association orders for d(sin^2 x)/d(theta_dmsq); recipes converted
 * from them must pick the matching style to reproduce the old bytes exactly.
 */
enum class GradStyle : uint8_t {
    HoistedPhase,   ///< 3+1 family: dsin2_ddm = sin(2x)*k*le*ddm hoisted, entry = Cdm[c]*dsin2_ddm.
    InlineTwoFlav   ///< 2-flavour family: entry = Cdm[c]*sin(2x)*k*le*ddm, left-associated.
};

/** @brief One physics parameter of a sine-kernel recipe. */
struct SineParameterSpec {
    const char *name;    ///< Internal name (param_names entry; exact-match for --fix etc.).
    const char *pretty;  ///< ROOT-LaTeX display name.
    const char *unit;    ///< Display unit string ("" if dimensionless).
    bool log10;          ///< True if the internal parameter is log10 of the physical value.
    float lb, ub, def;   ///< Bounds and default in INTERNAL space; def must lie in [lb, ub].
};

/** @brief One probability column of a sine-kernel recipe; column index = position in the table. */
struct SineChannelSpec {
    const char *name;  ///< Short channel label (documentation only).
    ProbForm form;     ///< Functional form of this column.
};

/**
 * @brief One structurally non-zero (parameter, channel) entry of the mixing-parameter
 * gradient columns (parameter index >= 1; the Delta m^2 column 0 is handled via Cdm).
 * @details Listing the exact sparsity of the legacy per-model gradient code keeps the
 * untouched entries at exactly Zero, matching the old structural-zero pattern.
 */
struct SineGradEntry {
    uint8_t param;    ///< Internal parameter index (1..nparams-1).
    uint8_t channel;  ///< Probability column index.
};

inline constexpr size_t kMaxSineChannels = 8;  ///< Largest channel count (the NC models).
inline constexpr size_t kMaxSineParams   = 4;  ///< Largest parameter count (dmsq + 3 mixings).

/**
 * @brief Per-call amplitude block filled by a recipe's amps lambda.
 * @details Filled ONCE per get_probs call (never per grid point / per event).
 * dmsq is the LINEAR-space mass splitting (the recipe applies maybe_convert_log);
 * A[c] is channel c's amplitude in physical space with the legacy clamps applied
 * verbatim. Entries for Null channels are ignored.
 */
struct SineAmps {
    float dmsq = 0.0f;
    std::array<float, kMaxSineChannels> A{};
};

/**
 * @brief Per-call gradient block filled by a recipe's jacs lambda.
 * @details Filled ONCE per get_probs_grad call. The kernel evaluates, per grid point i
 * (x = k*dmsq*le, s2 = sin^2 x, k = 1.266932679):
 *   - Delta m^2 column:  grads[0](i,c) = Cdm[c] * d(sin^2 x)/d(theta_0)  per GradStyle,
 *     where Cdm[c] is the full signed coefficient of sin^2 x in channel c
 *     (e.g. -4*Um4sq*(1-Um4sq) for a disappearance channel) and ddm is the
 *     Delta m^2 chain factor (LN10*dmsq for log10, else 1).
 *   - Mixing columns:    grads[e.param](i,e.channel) = (pre[e.param][e.channel] * s2)
 *                                                      * post[e.param][e.channel]
 *     for each listed SineGradEntry.
 * The RECIPE owns all chain factors (d(linear)/d(internal) = ln10 * value for log10
 * parameters) and all clamp-zeroing (a clamped parameter's factors are set to 0),
 * exactly as the legacy per-model gradient code did; the kernel applies no chain
 * factors of its own. For converted models the pre/post split mirrors the legacy
 * expression order bit-for-bit; new models may fold everything into pre (post = 1).
 */
struct SineJacs {
    float dmsq = 0.0f;  ///< Linear-space Delta m^2.
    float ddm  = 0.0f;  ///< Chain factor d(dmsq)/d(theta_0).
    std::array<float, kMaxSineChannels> Cdm{};
    std::array<std::array<float, kMaxSineChannels>, kMaxSineParams> pre{};
    std::array<std::array<float, kMaxSineChannels>, kMaxSineParams> post{};
};

class PROsineModel;

/**
 * @brief Full definition of one sine-kernel model.
 * @details Recipes live in static storage inside find_sine_recipe's registry;
 * PROsineModel holds a reference to its recipe for the model's lifetime.
 */
struct SineModelRecipe {
    const char *tag;                          ///< Factory name (config.m_model_tag value).
    std::vector<SineParameterSpec> params;    ///< Parameter table; index = internal parameter index.
    std::vector<SineChannelSpec>   channels;  ///< Channel table; index = probability column.
    std::vector<SineGradEntry>     grad_entries;  ///< Sparsity of the mixing gradient columns.
    GradStyle grad_style = GradStyle::HoistedPhase;
    /// Amplitude evaluator; called once per get_probs call. Use the const char*
    /// maybe_convert_log overload (string literals) — never build std::string here.
    std::function<void(const PROsineModel&, const Eigen::VectorXf&, SineAmps&)> amps;
    /// Closed-form amplitude-derivative evaluator; called once per get_probs_grad call.
    std::function<void(const PROsineModel&, const Eigen::VectorXf&, SineJacs&)> jacs;
    /// Optional parameter constraint (empty = none); metrics return their invalid
    /// sentinel when it evaluates to 0.
    std::function<int(const PROsineModel&, const Eigen::VectorXf&)> constraint;
};

/**
 * @brief Look up a registered sine-kernel recipe by factory tag.
 * @return Pointer to the recipe (static storage), or nullptr if the tag is not a
 *         sine-kernel model.
 */
const SineModelRecipe *find_sine_recipe(const std::string &tag);

/**
 * @brief The single dynamic model class evaluating any SineModelRecipe.
 * @details Overrides get_probs and get_probs_grad with the shared sine kernel; the
 * recipe lambdas run once per call, so the hot path has no per-grid-point indirect
 * calls or allocations. model_functions/prob_types are still populated (sized to the
 * channel count) for the base-class fallbacks and pybind.
 */
class PROsineModel : public PROmodel {
public:
    /**
     * @brief Construct a sine-kernel model from a recipe.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name to variable index in PROpeller.
     *                      Must contain the key "L/E".
     * @param recipe        The model definition; must outlive this object
     *                      (registry recipes have static storage).
     */
    PROsineModel(const PROpeller &prop, const std::map<std::string,int> &parameter_map,
                 const SineModelRecipe &recipe);

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;

    /** @brief Closed-form derivatives of get_probs (see PROmodel::get_probs_grad and SineJacs). */
    std::vector<Eigen::MatrixXf> get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;

    /** @brief Closed-form derivatives above, so the analytic gradient is exact here (see PROmodel::has_analytic_gradient). */
    bool has_analytic_gradient() const override { return true; }

private:
    const SineModelRecipe &recipe_;
    size_t J_ = 0;                                        ///< Number of probability columns.
    std::array<ProbForm, kMaxSineChannels> forms_{};      ///< Per-column form, from the channel table.
    std::vector<uint8_t> osc_channels_;                   ///< Non-Null column indices.

    /** @brief Scalar per-event probability backing model_functions (base FD / pybind only). */
    float channel_prob(size_t c, const Eigen::VectorXf &phys, float le) const;

    /**
     * @brief sin^2(freq * L/E) averaged over an underflow/overflow cell of the binned grid
     * (centre @p le, half-width @p hw, see PROmodel::phys_axes_hw); truth bins keep the
     * midpoint sin^2 of the legacy expressions.
     * @param ds2_dfreq If non-null, out: the derivative with respect to freq.
     */
    static float cell_sin2(float freq, float le, float hw, float *ds2_dfreq);
};

}

#endif
