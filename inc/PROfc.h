/**
 * @file PROfc.h
 * @brief Feldman-Cousins (FC) unified confidence interval infrastructure for PROfit.
 * @author PROfit Collaboration
 *
 * @details Defines the data structures and worker function used to perform
 * Feldman-Cousins frequentist coverage tests.  The fc_worker() function runs many
 * pseudo-experiment fits in parallel threads, filling the dchi2 distribution at a
 * given physics point.  The fc_args struct bundles all the read-only inputs so that
 * worker threads receive them by value and can run concurrently without locking.
 */
#ifndef PRO_FC_H
#define PRO_FC_H

#include "PROfitter.h"
#include "PROconfig.h"
#include "PROsyst.h"
#include "PROseed.h"
#include "PROmetric.h"
#include "PROcess.h"
#include "PROtocall.h"
#include "PROmetrics/PROchi.h"
#include "PROmetrics/PROpearson.h"
#include "PROmetrics/PROCNP.h"
#include "PROmetrics/PROpoisson.h"

#include <Eigen/Eigen>


namespace PROfit {

    /**
     * @brief Output record for a single Feldman-Cousins pseudo-experiment.
     */
    struct fc_out{
        float chi2_syst; ///< Best-fit chi-squared at the fixed physics point (systematics-only minimisation).
        float chi2_osc;  ///< Best-fit chi-squared from the free oscillation fit.
        /// Best-fit physics parameters from the oscillation fit, in the fitter's internal space
        /// (log10 where model->is_log10[i] is true).  Converted using model->param_names at output time.
        Eigen::VectorXf best_phys_osc;
        Eigen::VectorXf best_fit_syst; ///< Full best-fit parameter vector from the syst-only fit.
        Eigen::VectorXf best_fit_osc;  ///< Full best-fit parameter vector from the oscillation fit.
        Eigen::VectorXf syst_throw;    ///< Random systematic parameter throw used for this pseudo-experiment.
    };

    /**
     * @brief Bundle of inputs passed by value to each fc_worker thread.
     * @details All members are copied so worker threads can run concurrently.  The
     * output pointers dchi2s and out point to pre-allocated per-thread storage.
     */
    struct fc_args {
        const size_t todo;               ///< Number of pseudo-experiments this thread should process.
        std::vector<float>* dchi2s;      ///< Output: delta-chi2 = chi2_syst - chi2_osc for each pseudo-experiment.
        std::vector<fc_out>* out;        ///< Output: full fc_out records for each pseudo-experiment.
        const PROconfig config;          ///< Analysis configuration (copied per thread).
        const PROpeller prop;            ///< MC event store (copied per thread).
        const PROsyst systs;             ///< Systematic object (copied per thread).
        std::string chi2;                ///< Name of the chi-squared type to use.
        const Eigen::VectorXf phy_params;///< True physics parameter point at which the FC test is evaluated.
        const Eigen::MatrixXf L;         ///< Cholesky factor of the total covariance for correlated systematic throws.
        PROfitterConfig fitconfig;       ///< Fitter configuration.
        uint32_t seed;                   ///< Random seed for this thread's pseudo-experiments.
        const int thread;                ///< Thread index (used to differentiate seeds).
        const bool binned;               ///< If true, use binned spectrum-filling mode.
        const bool gof_mode;             ///< If true, run as a goodness-of-fit test (both fits free).
    };

    /**
     * @brief Worker function that runs @p arg.todo Feldman-Cousins pseudo-experiments.
     * @details For each pseudo-experiment: throws a systematic parameter vector from a
     * Gaussian distribution, generates a fake data spectrum, fits both a syst-only and
     * a full oscillation hypothesis, and records the delta-chi2.
     * @param arg         Bundle of inputs (passed by value for thread safety).
     * @param progressbar Reference to the shared progress bar for updating the UI.
     */
    void fc_worker(fc_args arg, MultiPROgressBar& progressbar);
}
#endif
