/**
 * @file PROjector.h
 * @brief Two-stage pre-fit / projected-fit ("PROjector") machinery for PROfit.
 * @author PROfit Collaboration
 *
 * @details PROjector implements a two-stage alternative to a joint near+far detector fit:
 *
 *   Stage 1 (--projector-prefit "<pattern>"): fit ONLY the subchannels whose fullname
 *   ("<mode>_<detector>_<channel>_<subchannel>") matches the given pattern — an
 *   unanchored regex, so plain substrings work as-is (same wildcard convention as
 *   --bkg-subtract and PROsyst::CreateFlatMatrix; see PROconfig.h). Before
 *   the fit, all covariance-type systematics are promoted to eigenmode spline nuisance
 *   parameters (via PROsyst::FillSplinesFromCovarianceMatrix) so that the ENTIRE
 *   systematic constraint — splines and covariance modes, including their data-induced
 *   cross-correlations — is captured in a single Gaussian posterior (theta_hat, Sigma)
 *   computed from a finite-difference Hessian at the best fit. The posterior is saved
 *   to a "<tag>_PROjector_constraint.bin" file.
 *
 *   Stage 2 (--projector <file>): re-derive the identical eigenmode promotion (the
 *   decomposition is deterministic given the same _syst.bin), mask the data to the
 *   COMPLEMENT of the stored pattern, and install the stage-1 posterior as a fully
 *   correlated Gaussian prior on the nuisance parameters (PROsyst::external_prior_cov,
 *   consumed by PROchi::Pull). Any subcommand (global, profile, surface, plot, ...)
 *   then runs as the constrained "projected" fit.
 *
 * Bin exclusion is enforced by the PROconfig fit-region (active-bin) mask, which every
 * metric (PROchi/PROCNP/PROpoisson) snapshots at construction — including the fresh
 * metrics FC and adaptive-FC workers build around thrown pseudo-data, so regenerated
 * data can never leak masked bins back into a chi2. The excluded data is additionally
 * zeroed for display/blindness. The pattern must select WHOLE channels in collapsed
 * space (the chi2 lives in collapsed bins).
 */
#ifndef PROJECTOR_H_
#define PROJECTOR_H_

#include <string>
#include <vector>

#include <Eigen/Eigen>

#include "PROconfig.h"
#include "PROdata.h"
#include "PROmodel.h"
#include "PROmetric.h"
#include "PROserial.h"
#include "PROsyst.h"

namespace PROfit {

    /**
     * @brief Command-line driven configuration for a PROjector run.
     * @details prefit_pattern and constraint_file are mutually exclusive; exactly one
     * being non-empty selects pre-fit or projected mode respectively.
     */
    struct PROjectorRunConfig {
        std::string prefit_pattern;   ///< Non-empty => stage-1 pre-fit mode; unanchored regex matched against subchannel fullnames (plain substrings work).
        std::string constraint_file;  ///< Non-empty => stage-2 projected mode; path to a saved constraint file.
        int num_decomp_knobs = -1;    ///< Eigenmodes kept when promoting covariance to splines (-1 = all positive modes, no residual).
        std::vector<std::string> keep_covariance; ///< Covariance systematics NOT promoted (stay as unconstrained covariance).
        bool float_physics = false;   ///< Pre-fit: float physics parameters and save the marginal nuisance covariance (default: fix at CV).
        bool force = false;           ///< Skip config-hash consistency check when loading a constraint.

        bool prefit_mode() const { return !prefit_pattern.empty(); }
        bool projector_mode() const { return !constraint_file.empty(); }
        bool active() const { return prefit_mode() || projector_mode(); }
    };

    /**
     * @brief Serializable stage-1 result: the nuisance-parameter posterior and the
     * bookkeeping needed to validate and reproduce it in stage 2.
     */
    struct PROjectorConstraint {
        uint32_t config_hash = 0;         ///< PROconfig hash of the pre-fit run; must match in stage 2 (unless forced).
        std::string metric_name;          ///< chi2 metric used in the pre-fit ("PROchi").
        std::string prefit_pattern;       ///< Subchannel wildcard used to select the pre-fit bins.
        int num_decomp_knobs = -1;        ///< K used for the covariance->spline promotion.
        std::vector<std::string> keep_covariance;   ///< Covariance systematics that were not promoted.
        std::vector<std::string> nuisance_names;    ///< Post-promotion spline names, in parameter-vector order.
        Eigen::VectorXf centers;          ///< Posterior centers theta_hat (pre-fit best-fit nuisance values).
        Eigen::MatrixXf covariance;       ///< Posterior covariance Sigma over the nuisance parameters.
        std::vector<std::string> physics_names;     ///< Physics parameter names of the pre-fit model.
        Eigen::VectorXf physics_best_fit; ///< Pre-fit physics point (fixed CV values or floated best fit).
        bool physics_were_fixed = true;   ///< True if physics parameters were held fixed during the pre-fit.
        float prefit_chi2 = 0;            ///< Pre-fit best-fit chi2 (record keeping).

        template<class Archive>
        void serialize(Archive &ar, [[maybe_unused]] const unsigned int version) {
            ar & config_hash;
            ar & metric_name;
            ar & prefit_pattern;
            ar & num_decomp_knobs;
            ar & keep_covariance;
            ar & nuisance_names;
            ar & centers;
            ar & covariance;
            ar & physics_names;
            ar & physics_best_fit;
            ar & physics_were_fixed;
            ar & prefit_chi2;
        }

        void save(const std::string &filename) const;
        bool load(const std::string &filename);
    };

    /**
     * @brief Resolve a subchannel wildcard into complete channels and validate it.
     * @details Matches @p pattern against every subchannel fullname as an unanchored
     * regex — plain substrings work as-is (the convention of
     * find_subchannels_by_pattern). Because the chi2 lives in collapsed
     * (channel-level) space, a channel is only selectable as a whole: if a pattern
     * matches SOME but not ALL subchannels of a channel this function fails loudly.
     * @param config           Analysis configuration.
     * @param pattern          Pattern (unanchored regex; plain substrings work).
     * @param matched_channels Output: global channel indices whose subchannels all matched.
     * @return True if the selection is valid (no partially-matched channel, at least one
     *         matched and one unmatched channel).
     */
    bool PROjectorSelectChannels(const PROconfig &config, const std::string &pattern,
                                 std::vector<size_t> &matched_channels);

    /**
     * @brief Build a 0/1 mask over the collapsed bins of variable @p var_index.
     * @param complement If false, masked-in bins are those of @p matched_channels;
     *                   if true, everything else.
     */
    Eigen::VectorXf PROjectorCollapsedMask(const PROconfig &config,
                                           const std::vector<size_t> &matched_channels,
                                           size_t var_index, bool complement);

    /**
     * @brief Promote covariance-type systematics of @p systs to eigenmode splines.
     * @details Sums the fractional covariances of every named covariance systematic not
     * listed in @p keep_covariance, eigendecomposes the sum via
     * PROsyst::FillSplinesFromCovarianceMatrix (top-@p num_decomp_knobs modes become
     * linear splines with unit Gaussian priors, remainder kept as residual covariance),
     * and fixes up fractional_covariance, spline_priors/centers, and the config
     * plotname map for the synthesized knobs. NOTE: the "mcstat" covariance is never
     * promoted (it is diagonal with no cross-detector correlations to propagate).
     * @param var_index The variable/binning index this PROsyst was built for (config.i_prime).
     * @return Number of spline knobs added.
     */
    size_t PROjectorPromoteCovariance(PROconfig &config, PROsyst &systs, size_t var_index,
                                      int num_decomp_knobs,
                                      const std::vector<std::string> &keep_covariance);

    /**
     * @brief One-stop PROjector hook called from PROfit.cxx.
     * @details Must run AFTER systematic selection (--syst-list/--exclude-systs) and data
     * creation, but BEFORE CVParams sizing, the bounds/--fix section, and metric
     * construction. Depending on the mode it:
     *   - pre-fit: validates the pattern, promotes covariance, masks data to the matched
     *     channels, and (unless float_physics) appends the physics parameter names to
     *     @p fixed_params so the existing bounds section pins them at CV;
     *   - projected: loads the constraint, re-runs the identical promotion, verifies the
     *     nuisance names match, installs the posterior as spline centers/priors and the
     *     correlated external prior covariance, and masks data to the complement.
     * In both modes @p fakedataparams is zero-extended to the enlarged parameter count.
     * @return True on success; false means the caller should terminate.
     */
    bool PROjectorSetup(const PROjectorRunConfig &pjconf, PROconfig &config, PROsyst &systs,
                        std::vector<PROdata> &variable_data, PROdata &data,
                        Eigen::VectorXf &fakedataparams,
                        std::vector<std::string> &fixed_params,
                        const PROmodel &model, const std::string &chi2_name);

    /**
     * @brief Compute the nuisance-parameter posterior covariance from a finite-difference
     * Hessian of the chi2 at @p best_fit and save the stage-1 constraint file.
     * @details The Hessian is computed with central second differences over all FREE
     * parameters (physics entries of @p global_fixed are typically fixed in pre-fit
     * mode), inverted PSD-safely (eigenvalue clamping), and the nuisance block is stored.
     * Nuisance parameters that were fixed during the pre-fit keep their original prior
     * width on the diagonal with zero cross-correlation.
     * @return True on success (file written).
     */
    bool PROjectorSaveConstraint(const std::string &filename, const PROjectorRunConfig &pjconf,
                                 const PROconfig &config, PROmetric &metric,
                                 const Eigen::VectorXf &best_fit, float chi2,
                                 const std::vector<int> &global_fixed,
                                 const std::string &chi2_name);

}

#endif
