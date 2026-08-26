/**
 * @file PROcovariance.h
 * @brief Shared covariance-matrix chi-squared machinery for PROfit metrics.
 * @author PROfit Collaboration
 *
 * @details Implements the shared covariance-matrix statistic:
 *   chi2 = (data - pred)^T M^{-1} (data - pred) + pull_penalty
 * where M is the combined statistical and systematic covariance. Concrete
 * subclasses provide the statistical variance convention used to build M.
 */
#ifndef PROCOVARIANCE_H_
#define PROCOVARIANCE_H_

// STANDARD
#include <string>
#include <vector>
#include <optional>

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
     * @brief Common engine for covariance-matrix chi-squared metrics.
     * @details Gathers the MC store (PROpeller), systematic object (PROsyst), and oscillation
     * model (PROmodel) into a single callable object whose operator() returns the chi-squared
     * value and gradient for use by PROfitter. PROchi, PROpearson, and PROCNP
     * specialize only the per-bin statistical variance. All heavy objects are
     * stored as references or pointers to objects owned by the caller.
     */
    class PROcovariance : public PROmetric
    {
        protected:
            // CNP historically owns immutable config/propeller snapshots for
            // FC workers. Other covariance metrics leave these empty and bind
            // config/peller directly to the caller-owned inputs.
            std::optional<PROconfig> owned_config;
            std::optional<PROpeller> owned_peller;

            virtual Eigen::VectorXf statisticalVariances(
                const PROspec &prediction, const Eigen::VectorXf &data,
                const Eigen::VectorXf *param = nullptr) const = 0;
            virtual bool statisticalVariancesDependOnPrediction() const { return false; }

        public:
            const PROconfig &config;  ///< Analysis configuration (non-owning reference).
            const PROpeller &peller;  ///< MC event store (non-owning reference).
            Eigen::MatrixXf collapsed_stat_covariance; ///< Statistical covariance in the collapsed bin space.

            // Cached non-empty-bin slicing for default mode. In shape_only mode
            // normdata depends on `result` so these are rebuilt per call instead.
            std::vector<Eigen::Index> nec_indices;          ///< Indices of bins with positive data; empty if cache invalid.
            Eigen::MatrixXf nec_reduced_stat_cov;           ///< Reduced collapsed_stat_covariance for nec_indices (default mode only).
            bool nec_valid = false;                         ///< True iff !shape_only and cache populated in the ctor.

            /*Function: Constructor bringing all objects together*/
            PROcovariance(const std::string tag, const PROconfig &conin, const PROpeller &pin, const PROsyst *systin, const PROmodel &modelin, const PROdata &datain, EvalStrategy strat = EventByEvent, bool shape_only = false, std::vector<float> physics_param_fixed = std::vector<float>(), bool own_config_and_peller = false);

            PROcovariance(const PROcovariance &) = delete;
            PROcovariance &operator=(const PROcovariance &) = delete;


            /*Function: operator() is what is passed to minimizer.*/
            using PROmetric::operator();
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool rungradient);

            /** @brief Reset cached state and clear any fixed-parameter list. */
            virtual void reset() {
                physics_param_fixed.clear();
                last_value = 0;
                last_param = Eigen::VectorXf::Constant(last_param.size(), 0);
                fs_cache.invalidate();
            }

            /** @brief Replace the internal systematic pointer with @p new_syst. */
            virtual void override_systs(const PROsyst &new_syst) {
                syst = &new_syst;
                fs_cache.invalidate();
            }

            /**
             * @brief Compute the chi-squared contribution from a single analysis channel.
             * @param global_channel_index  Global channel index.
             * @param cv                    Predicted (CV) spectrum.
             * @param var_index             Variable index.
             * @return Chi-squared for that channel.
             */
            float getSingleChannelChi(size_t global_channel_index, const PROspec &cv, size_t var_index, const Eigen::MatrixXf &projection = Eigen::MatrixXf());

            /**
             * @brief Print a breakdown of the chi-squared contributions at @p param.
             * @param param  Full parameter vector (physics + splines).
             */
            void print(const Eigen::VectorXf &param);
    };
}
#endif
