/**
 * @file PROsplinePrior.h
 * @brief The Gaussian prior on the spline nuisance parameters, as one definition shared by
 *        the chi^2 metrics (Pull) and every throw of that prior.
 *
 * @details The prior is N(spline_centers, S) truncated to each spline's fit box, with
 * prior_type="uniform" splines flat and independent. S is diagonal (spline_priors^2) unless
 * the XML carries <correlation> entries or PROjector installed its pre-fit posterior
 * (PROsyst::external_prior_cov); see SplinePriorCovariance.
 *
 * SplinePriorSampler throws exactly that prior. Splines pinned by a zero-width box (--fix) or
 * held by the caller (--inject-systs) keep their value, and their correlated partners are drawn
 * from the Gaussian conditional on it, which is what the fit's Pull implies when a partner is
 * fixed. Correlated blocks are drawn by exact joint rejection from the truncated multivariate
 * normal; a block whose box leaves too little prior mass for that is refused when the sampler
 * is built. Uncorrelated splines are drawn by the historic per-spline routine
 * (ThrowRestrictedSplinePull), so configs without correlations reproduce bit for bit.
 */
#ifndef PROSPLINEPRIOR_H
#define PROSPLINEPRIOR_H

#include <Eigen/Eigen>
#include <random>
#include <string>
#include <vector>

#include "PROconfig.h"
#include "PROsyst.h"

namespace PROfit {

    /**
     * @brief Prior covariance of the spline nuisances exactly as the metrics' Pull applies it.
     * @details PROjector's external covariance when installed, else the XML <correlation>
     * pairs (matched by name against the current spline_names) scaled to diag(sigma) R
     * diag(sigma). Rows and columns of uniform-prior splines are replaced by the identity, so
     * they decouple from the Gaussian block. Pairs naming a spline not in this PROsyst (e.g.
     * one the user excluded) are skipped: marginalising a Gaussian drops its rows exactly.
     * @param correlated  Set to false (and an empty matrix returned) when the priors are
     *                    uncorrelated; the prior is then diag(spline_priors^2).
     */
    Eigen::MatrixXf SplinePriorCovariance(const PROconfig &config, const PROsyst &systs, bool &correlated);

    /**
     * @brief Inverse of a SplinePriorCovariance result, factorised in double (LLT).
     * @throws std::runtime_error if the covariance is not positive definite.
     */
    Eigen::MatrixXf SplinePriorPrecisionFromCovariance(const Eigen::MatrixXf &cov);

    /**
     * @brief Throws the spline prior (see file comment). Build once per worker; Throw() is
     *        const and thread-safe, the caller owns the generator.
     */
    class SplinePriorSampler {
        public:
            SplinePriorSampler() = default;

            /**
             * @param held         Optional per-spline flags: held splines keep held_values(i)
             *                     and draw nothing; correlated partners are conditioned on them.
             * @param held_values  Values of the held splines (indexed like the splines).
             * @param verbose      Log the correlated blocks and their acceptance at INFO.
             * @throws std::runtime_error when a correlated block's box leaves too little prior
             *         mass for rejection sampling (estimated acceptance below 1e-3).
             */
            SplinePriorSampler(const PROconfig &config, const PROsyst &systs,
                               const std::vector<char> &held = {},
                               const Eigen::VectorXf &held_values = Eigen::VectorXf(),
                               bool verbose = false);

            /**
             * @brief Fill one draw of every spline into @p splines (size NSplines()).
             * @details Walks the splines in index order. Uncorrelated Gaussian splines use the
             * historic per-spline routine with @p d; a correlated block is drawn whole at its
             * lowest index from @p d and @p rng. Pinned/held splines draw nothing.
             */
            void Throw(Eigen::Ref<Eigen::VectorXf> splines, std::mt19937 &rng,
                       std::normal_distribution<float> &d) const;

            size_t NSplines() const { return kind.size(); }
            /// True when at least one spline is drawn jointly with a partner or conditioned on one.
            bool HasJointBlocks() const { return !blocks.empty(); }

        private:
            enum class Kind : char { Pinned, Uniform, Single, Block };
            struct JointBlock {
                std::vector<size_t> members;   ///< Ascending spline indices.
                Eigen::VectorXd mean;          ///< Conditional mean.
                Eigen::MatrixXd L;             ///< Cholesky factor of the conditional covariance.
                Eigen::VectorXf lo, hi;        ///< Box of each member.
                double acceptance = 1.0;       ///< Pilot estimate.
                long max_attempts = 0;
            };

            /// One rejection draw of block @p b; false when the box rejects it.
            bool drawBlock(const JointBlock &b, std::mt19937 &rng, std::normal_distribution<float> &d,
                           Eigen::VectorXf &out) const;

            std::vector<Kind> kind;
            std::vector<float> lo, hi, mu, sig, value;
            std::vector<std::string> names;
            std::vector<int> block_of;          ///< Block index of each Block-kind spline, else -1.
            std::vector<JointBlock> blocks;
    };

}

#endif
