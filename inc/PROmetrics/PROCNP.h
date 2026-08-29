/**
 * @file PROCNP.h
 * @brief Combined Neyman-Pearson covariance metric for PROfit.
 * @author PROfit Collaboration
 *
 * @details Defines PROCNP, the Combined Neyman-Pearson statistic whose per-bin variance is
 * 3/(1/n + 2/mu) — the harmonic blend of the Neyman (observed n) and Pearson (predicted mu)
 * prescriptions — with mu taken from the physics-only central value. See PROcovariance for
 * the shared covariance, pull, and gradient machinery.
 */
#ifndef PROCNP_H_
#define PROCNP_H_

#include "PROcovariance.h"

namespace PROfit {
    /**
     * @brief Combined Neyman-Pearson covariance metric.
     * @note PROCNP retains owned copies of the configuration and propeller for
     * thread-safe FC pseudo-experiments. The model remains a non-owning
     * reference to avoid slicing derived PROmodel implementations, while data
     * is owned by PROmetric.
     */
    class PROCNP : public PROcovariance {
        protected:
            Eigen::VectorXf statisticalVariances(
                const Eigen::VectorXf &collapsed_prediction, const Eigen::VectorXf &comparison,
                const Eigen::VectorXf *param = nullptr) const override;

            bool statisticalVariancesDependOnPrediction() const override;

            Eigen::VectorXf singleChannelStatVariances(
                const Eigen::VectorXf &collapsed_cv, const Eigen::VectorXf &comparison) const override;

        private:
            // Single-slot cache for CollapseMatrix(FillSpectra(noshiftvec)). The CNP mu is
            // taken from the CV with every spline pull zeroed, so it depends on the physics
            // subvector alone. That makes the cache a large win in the finite-difference
            // gradient loop, where every nuisance-parameter step leaves the physics — and
            // therefore mu — untouched. Mutable so the const variance hook can memoise;
            // metrics are per-thread (Clone()), so no locking is needed.
            mutable Eigen::VectorXf cnp_cached_phys;         ///< Physics subvector the cache was filled at.
            mutable Eigen::VectorXf cnp_cached_collapsed_cv; ///< Cached collapsed no-shift CV spectrum.
            mutable bool cnp_cv_cache_valid = false;         ///< True iff the cache matches cnp_cached_phys.

            /**
             * @brief CollapseMatrix(FillSpectra(noshiftvec built from @p phys)), memoised.
             * @param phys        Physics subvector.
             * @param param_size  Length of the full parameter vector (physics + splines).
             * @return Reference to the cached collapsed CV; invalidated by reset() and
             *         override_systs(), and overwritten whenever @p phys changes.
             */
            const Eigen::VectorXf &cachedNoshiftCollapsedCV(const Eigen::VectorXf &phys,
                                                            Eigen::Index param_size) const;

        public:
            PROCNP(const std::string tag, const PROconfig &conin, const PROpeller &pin,
                   const PROsyst *systin, const PROmodel &modelin, const PROdata &datain,
                   EvalStrategy strat = EventByEvent, bool shape_only = false,
                   std::vector<float> physics_param_fixed = std::vector<float>());

            PROmetric *Clone() const override;

            /** @brief Reset cached state, including the no-shift CV cache. */
            void reset() override;

            /** @brief Replace the internal systematic pointer and drop the no-shift CV cache. */
            void override_systs(const PROsyst &new_syst) override;
    };
}

#endif
