/**
 * @file PROmodelSimple.h
 * @brief Non-oscillation physics models: NullModel and PROtemplate.
 * @author PROfit Collaboration
 *
 * @details NullModel is also constructed directly (e.g. in bin/PROfit.cxx).
 *   - NullModel   — no oscillation; all events receive probability 1.
 *   - PROtemplate — template fit; floats subchannel normalizations as physics parameters.
 */
#ifndef PROMODELSIMPLE_H
#define PROMODELSIMPLE_H

#include "PROmodel.h"

namespace PROfit {

/**
 * @brief Trivial "no oscillation" model — all events receive oscillation probability 1.
 * @details Used as a central-value baseline and for systematic-only fits where oscillation
 * is not being tested.  All events are placed into a single histogram component regardless
 * of their model_rule.
 */
class NullModel : public PROmodel {
public:
    /**
     * @brief Construct the NullModel from an MC event store.
     * @param prop  The PROpeller containing MC events; used only to build hists.
     */
    NullModel(const PROpeller &prop);
};

/**
 * @brief Template-fit model: floats the overall normalization of one or more subchannels.
 * @details A non-oscillation physics model. Each floated subchannel (named by a <parameter> in
 * the XML <model> section) becomes a free physics parameter equal to the multiplicative scale
 * applied to that subchannel's events — default 1 (nominal), bounded to [min, max] from the
 * parameter's "min"/"max" attributes. All non-floated subchannels are held fixed at scale 1.
 *
 * Construction mirrors NullModel (no truth/kinematic grid: `ivars` empty, `n_phys_bins = 1`)
 * but is NOT trivial: the K+1 columns of H_combined separate the fixed remainder (column 0)
 * from each floated subchannel (columns 1..K). get_probs() returns the per-column scale factors
 * so the single GEMV in FillSpectra yields
 *     spec = hist_fixed + sum_k scale_k * hist_subchannel_k.
 * A template fit therefore behaves exactly like an oscillation model to the fitter/surface/MCMC
 * code and inherits the FillSpectra phys/syst split-cache for free.
 *
 * @note Binned FillSpectra only. The unbinned (event-by-event) path routes columns by model_rule,
 *       which is not meaningful for subchannel normalization; all fitting/scanning is binned.
 */
class PROtemplate : public PROmodel {
public:
    /**
     * @brief Construct the template-fit model.
     * @param config  Parsed configuration; supplies the floated subchannel names and scale bounds
     *                (m_model_parameter_names / _min / _max) and the subchannel<->reco-bin mapping.
     * @param prop    MC event store; used to build H_combined.
     */
    PROtemplate(const PROconfig &config, const PROpeller &prop);

    /**
     * @brief Return the per-column scale factors for the template fit.
     * @details Pure normalization: the scales do not depend on any kinematic grid, so @p var_arrs
     * is ignored. Returns a (1, K+1) row [1, phys(0), ..., phys(K-1)] consumed by the single GEMV
     * in FillSpectra.
     */
    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &) const override;
    /** @brief Closed-form derivatives of get_probs (see PROmodel::get_probs_grad). */
    std::vector<Eigen::MatrixXf> get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &) const override;
};

}

#endif
