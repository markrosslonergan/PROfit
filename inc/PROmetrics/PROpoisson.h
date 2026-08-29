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
     * @todo Improve analytic gradient calculation.
     */
    class PROpoisson : public PROmetric
    {
        private:

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

            /** @brief Return a heap-allocated copy of this PROpoisson. */
            virtual PROmetric *Clone() const;

            /** @brief Replace the internal systematic pointer with @p new_syst. */
            virtual void override_systs(const PROsyst &new_syst) {
                syst = &new_syst;
                fs_cache.invalidate();
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
