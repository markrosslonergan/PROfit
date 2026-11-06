/**
 * @file PROpulls.h
 * @brief Post-fit pull distributions: every data-constrained nuisance pull, normalised so a
 * well-modelled fit gives a unit normal, histogrammed and fit with a Gaussian.
 *
 * @details A raw pull (theta_hat - theta_0)/sigma_prior is NOT unit-normal even when the
 * model is right: the best fit only moves a nuisance by the part of its prior spread the
 * data resolves, so Var(theta_hat) = sigma_prior^2 - sigma_post^2 (Demortier & Lyons,
 * CDF/ANAL/PUBLIC/5776). The normalised pull
 *
 *     z = (theta_hat - theta_0) / sqrt(sigma_prior^2 - sigma_post^2)
 *
 * is N(0,1) under the fitted model, so the histogram over all nuisances should be a
 * standard Gaussian. sigma_post comes from one linearised (Gauss-Newton) system at the best
 * fit over every free parameter at once: physics, spline nuisances, and the eigenmodes alpha
 * of the summed covariance (Sigma = L L^T, prior N(0,1), the same modes as
 * plotCovariancePosteriorPulls), with the spline Jacobian from finite differences of the
 * collapsed prediction and C the metric's own statistical variance (its full matrix for
 * overlapping analyses, PROmetric::GetStatCovariance). Each sigma_post is
 * therefore marginal over everything else that floats, which is what the variance identity
 * needs. Shape-only fits linearise the per-channel rescaled prediction and project L,
 * exactly as the metric does.
 *
 * Four categories, one page each:
 *   1. Symmetric splines: Gaussian prior, fit range covering theta_0 +- 3 sigma.
 *   2. Covariance systematics: the posterior mode pulls alpha_hat (Putnam conditional).
 *   3. Bounded / one-sided splines: Gaussian prior with a fit bound inside 3 sigma (0..1
 *      morph knobs, norm splines with f > 1/3, narrow restrict= ranges). A parameter that
 *      stopped at a bound only tells us its latent z lies beyond that bound; it is drawn
 *      hatched at the bound and the Gaussian fit treats it as censored.
 *   4. Uniform-prior splines (prior_type="uniform"): no pull exists, so the estimate is
 *      mapped through the CDF of its own null distribution (a uniform truth on the box,
 *      smeared by sigma_post) and then through Phi^-1; also unit-normal under the model.
 *
 * Parameters the data barely constrains (prior variance reduction below
 * min_var_reduction) have z = 0/0 and are left out of the histograms and counted; --fix'd
 * splines are listed as fixed. The selection uses only sigma_post, which does not depend
 * on the data, so it cannot bias the distribution.
 *
 * Pulls are correlated, so each page also reports what the model predicts for this
 * histogram given those correlations: the spread of its mean, and its expected width
 * sqrt(1 - Var(mean)), below 1 when the pulls move together. It also reports
 * (pages 1 and 2, and 3 when nothing is censored)
 * the whitened quadratic form q = d^T (P - Sigma_post)^+ d, chi^2-distributed under the
 * model, as the correlation-aware goodness-of-fit of the pulls.
 */
#ifndef PROPULLS_H_
#define PROPULLS_H_

#include <array>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Eigen>

#include "TObject.h"

#include "PROconfig.h"
#include "PROmetric.h"
#include "PROpeller.h"

namespace PROfit {

    /** @brief Which pull-distribution page a parameter belongs to (value = page index). */
    enum class PullCategory : int {
        SymmetricSpline = 0, ///< Gaussian prior, fit range covers theta_0 +- 3 sigma.
        CovarianceMode  = 1, ///< Eigenmode of the summed covariance, posterior pull.
        BoundedSpline   = 2, ///< Gaussian prior, a fit bound within 3 sigma of theta_0.
        UniformSpline   = 3  ///< prior_type="uniform": probability-integral-transform score.
    };

    /** @brief One parameter's pull. */
    struct PullEntry {
        std::string name;          ///< Spline plotname, or "cov mode j (x%)".
        PullCategory category = PullCategory::SymmetricSpline;
        float value = 0;           ///< Best-fit value.
        float center = 0;          ///< Prior centre (uniform: box centre).
        float sigma_prior = 0;     ///< Prior sigma (uniform: box width / sqrt(12)).
        float sigma_post = 0;      ///< Linearised marginal posterior sigma (uniform: from data alone).
        float var_reduction = 0;   ///< 1 - sigma_post^2 / sigma_prior^2.
        float raw_pull = 0;        ///< (value - center) / sigma_prior.
        float pull = 0;            ///< Normalised pull z.
        int censored = 0;          ///< -1 / +1: stopped at the lower / upper fit bound (z is then a bound).
        bool used = false;         ///< Enters the histogram and fit (false: weakly constrained).
    };

    /** @brief Gaussian fit and model expectations for one category. */
    struct PullGaussFit {
        int n = 0;                 ///< Entries in the histogram.
        int n_censored = 0;        ///< Of which stopped at a fit bound.
        int n_excluded = 0;        ///< Weakly constrained, left out.
        bool fitted = false;       ///< False when fewer than two uncensored entries.
        double mu = 0, mu_err = 0;
        double sigma = 0, sigma_err = 0;
        double null_mu_err = 0;    ///< Spread of the sample mean expected under the model (pull correlations included).
        double null_sigma = 0;     ///< Expected sample width under the model, sqrt(1 - null_mu_err^2): below 1 when pulls are correlated.
        double whitened_chi2 = 0;  ///< q = d^T (P - Sigma_post)^+ d over the category's entries.
        int whitened_ndf = 0;      ///< 0 = not computed (censored entries, or uniform category).
    };

    /** @brief All pulls of one fit, ready to plot. */
    struct PullSummary {
        std::vector<PullEntry> entries;
        std::array<PullGaussFit, 4> fits;
        std::vector<std::string> fixed_names; ///< Splines pinned by the fit bounds (--fix).
        double min_var_reduction = 0.05;
        bool shape_only = false;
    };

    /**
     * @brief Compute the normalised pulls of every nuisance at a best fit.
     * @param config            Analysis configuration (fitting variable = i_prime).
     * @param prop              MC event store.
     * @param metric            The fit metric (systs, model, stat variance, prior precision, shape-only flag).
     * @param best_fit          Full best-fit vector (physics then splines).
     * @param lb                Fit lower bounds (zero-width = fixed), e.g. PROfitter::lb.
     * @param ub                Fit upper bounds.
     * @param data_spec         Collapsed data the fit used.
     * @param min_var_reduction Minimum 1 - sigma_post^2/sigma_prior^2 for a pull to be shown.
     * @return Per-parameter pulls plus a Gaussian fit per category (empty on bad input).
     */
    PullSummary computePostfitPulls(const PROconfig &config, const PROpeller &prop, const PROmetric &metric,
            const Eigen::VectorXf &best_fit, const Eigen::VectorXf &lb, const Eigen::VectorXf &ub,
            const Eigen::VectorXf &data_spec, double min_var_reduction = 0.05);

    /**
     * @brief Draw the four pull-distribution pages and log the fits.
     * @details One page per PullCategory, always four pages: histogram of z (censored entries
     * hatched), the N(0,1) expectation, the fitted Gaussian, and a box with the fit, the
     * model expectation for the mean and width, the whitened chi^2 and the largest |z|. The last page
     * also lists fixed and weakly constrained parameters.
     * @param summary    Output of computePostfitPulls.
     * @param filename   Output PDF (multi-page).
     * @param drawn_objs Optional sink for the histograms and the per-parameter / per-fit
     *                   TTrees ("postfit_pulls", "postfit_pull_fits"); entries already in the
     *                   map under the same names are deleted and replaced.
     * @return 0 if drawn, 1 if there was nothing to draw.
     */
    int plotPostfitPullDistributions(const PullSummary &summary, const std::string &filename,
            std::map<std::string, TObject*> *drawn_objs = nullptr);

}

#endif
