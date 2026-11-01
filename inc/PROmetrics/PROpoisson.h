/**
 * @file PROpoisson.h
 * @brief Poisson log-likelihood ratio chi-squared metric for PROfit.
 * @author PROfit Collaboration
 *
 * @details Defines PROpoisson, which implements the Poisson log-likelihood ratio
 * (Baker-Cousins) chi-squared:
 *   chi2 = 2 * sum_i [ mu_i - n_i + n_i * ln(n_i / mu_i) ] + pull_penalty
 * where mu_i is the predicted count and n_i is the observed count in bin i.
 * This is the statistically optimal test statistic for Poisson-distributed data.
 * Inherits from PROmetric and is compatible with PROfitter.
 */
#ifndef PROPOISSON_H_
#define PROPOISSON_H_

// STANDARD
#include <string>
#include <vector>

#include <Eigen/Eigen>

// OUR INCLUDES
#include "PROconfig.h"
#include "PROdata.h"
#include "PROsyst.h"
#include "PROpeller.h"
#include "PROmodel.h"
#include "PROmetric.h"
#include "PROcess.h"

namespace PROfit{

    /**
     * @brief Poisson log-likelihood ratio chi-squared metric (Baker-Cousins).
     * @details Gathers the MC store, systematic object, and oscillation model into a callable
     * whose operator() returns the Poisson log-likelihood ratio test statistic and gradient.
     * All heavy objects are stored as (const) references or pointers owned by the calling
     * executable.
     * @todo Add capability to define the chi-squared function externally.
     */
    class PROpoisson : public PROmetric
    {
        private:
            /// Full-binning spectrum Jacobian ds/dθ with the shape-only rescale chain rule
            /// (same as PROcovariance::spectrumJacobian, kept separate on purpose).
            Eigen::MatrixXf spectrumJacobian(const Eigen::VectorXf &param, const PROspec &result,
                                             const Eigen::VectorXf &shape_r);

            /// Pull() accumulated in double. Next to the double Baker-Cousins sum, a float sum
            /// of ~10² nuisance terms would be the largest rounding left in the value.
            double pullDouble(const Eigen::VectorXf &systs) const;

        public:
            const PROconfig &config; ///< Analysis configuration (non-owning reference).
            const PROpeller &peller; ///< MC event store (non-owning reference).
            /*Function: Constructor bringing all objects together*/
            PROpoisson(const std::string tag, const PROconfig &conin, const PROpeller &pin, const PROsyst *systin, const PROmodel &modelin, const PROdata &datain, EvalStrategy strat = EventByEvent, bool shape_only = false, std::vector<float> physics_param_fixed = std::vector<float>());


            /*Function: operator() is what is passed to minimizer.*/
            using PROmetric::operator();
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool rungradient);

            /** @brief Reset cached state and clear any fixed-parameter list. */
            virtual void reset();

            /** @brief The Gaussian spline pull, summed in double (see pullDouble). */
            float Pull(const Eigen::VectorXf &systs) override { return (float)pullDouble(systs); }

            /**
             * @brief chi2, gradient and Gauss-Newton Hessian in double (see PROmetric).
             * @details H = Jᵀ diag(2n/s²) J + prior curvature, J = d(collapsed prediction)/dθ
             * over the active bins with s > 0 (n = 0 bins are linear in s: no curvature). For
             * splines whose response is linear (eigenmode decomposition knobs) this is the exact
             * Hessian of the nuisance block. Binned strategies only.
             */
            bool GaussNewtonSystem(const Eigen::VectorXf &param, double &value,
                                   Eigen::VectorXd &grad, Eigen::MatrixXd &hess) override;

            double ValueDouble(const Eigen::VectorXf &param) override;

            /** @brief Return a heap-allocated copy of this PROpoisson. */
            virtual PROmetric *Clone() const;

            /** @brief Replace the internal systematic pointer with @p new_syst. */
            virtual void override_systs(const PROsyst &new_syst) {
                syst = &new_syst;
                fs_cache.invalidate();
            }

            /** @brief The analysis configuration (also public as `config`). */
            const PROconfig &GetConfig() const override { return config; }

            /**
             * @brief Gaussian-equivalent per-bin variance: the prediction.
             * @details Poisson has no Gaussian term; its Fisher information per bin is 1/mu, so
             * linearised consumers (post-fit pull normalisation) see the right weight. Empty
             * bins (mu = 0) carry no information and drop out.
             */
            Eigen::VectorXf GetStatVariances(const Eigen::VectorXf &collapsed_prediction,
                                             [[maybe_unused]] const Eigen::VectorXf &comparison,
                                             [[maybe_unused]] const Eigen::VectorXf &param) const override {
                return collapsed_prediction;
            }

            /**
             * @brief Compute the Poisson log-likelihood ratio for a single analysis channel.
             * @param channel_index  Global channel index.
             * @param cv             Predicted spectrum.
             * @param var_index      Variable index.
             * @return Poisson chi-squared for that channel.
             */
            float getSingleChannelChi(size_t channel_index, const PROspec &cv, size_t var_index, const Eigen::MatrixXf &projection = Eigen::MatrixXf());

            /**
             * @brief Print a breakdown of the Poisson chi-squared contributions at @p param.
             * @param param  Full parameter vector (physics + splines).
             */
            void print(const Eigen::VectorXf &param);
    };
}
#endif
