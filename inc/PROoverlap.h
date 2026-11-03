/**
 * @file PROoverlap.h
 * @brief Statistical overlap between analyses that select the same events.
 * @author PROfit Collaboration
 *
 * @details When two analyses combined in one XML both select some of the same interactions,
 * their observed spectra are not independent. Every interaction is one Poisson count in one
 * *cell*: the set S of collapsed bins it fills (one bin for an event only one analysis
 * selects, two or more for a shared one). With N_S the count in cell S,
 *   n_i = Σ_{S∋i} N_S,   Cov(n_i, n_j) = Σ_{S∋i,j} λ_S,
 * a sum of PSD rank-one terms, so the statistical covariance stays PSD for expected or
 * observed counts alike.
 *
 * The feature is strictly opt-in, through `overlap_matching_vars="run,subrun,evt,slc"` on
 * an `<MCFile>` (comma-separated TTreeFormula expressions identifying one interaction):
 *   - on DATA MCFiles it records the observed shared cells, which the Neyman metric adds
 *     to its statistical covariance (V = diag(n) + off-diagonal shared counts);
 *   - on MC MCFiles it records which MC entries are the same interaction, from which every
 *     toy thrower (fc, fc-adaptive, brazil, --pseudo-experiment, --poisson-throw) and the
 *     Asimov fake data take the EXPECTED cells at the toy truth.
 * Toys never use the data's overlap: their distribution must depend on the hypothesis only.
 *
 * The two analyses' MC need not cover the same files or exposure. `overlap_common="<cut>"`
 * on an MC MCFile marks the entries both analyses processed; the expected overlap is built
 * from FRACTIONS measured on that common sample, so each analysis's normalisation cancels.
 * Only the neyman metric supports overlap (a product of Poissons cannot carry it at all).
 */
#ifndef PROOVERLAP_H_
#define PROOVERLAP_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Eigen>

#include <boost/serialization/vector.hpp>

class TTree;
class TTreeFormula;

namespace PROfit {

    class PROconfig;
    class PROpeller;
    class PROmodel;
    class PROdata;
    class PROspec;

    /**
     * @brief Shared-event cells in one variable's collapsed bin space.
     * @details One entry per distinct set of bins that shared interactions fill. Exclusive
     * events are not listed: they are whatever of n_i the cells do not account for.
     */
    struct OverlapCells {
        std::vector<std::vector<int>> bins; ///< Per cell: sorted collapsed bin indices (two or more).
        std::vector<float> count;           ///< Per cell: observed, thrown or expected count.

        size_t size() const { return count.size(); }

        template<class Archive>
            void serialize(Archive &ar, [[maybe_unused]] const unsigned int version) {
                ar & bins;
                ar & count;
            }
    };

    /**
     * @brief Neyman statistical covariance of one dataset, restricted to the fitted bins.
     * @details diag(n) plus, for every cell, its count on each off-diagonal pair of its bins
     * that lie in @p idx (a cell keeps only its fitted bins; one left behaves as exclusive).
     * A bin pair holding only shared events (n_a = n_b = n_ab) makes the block singular: the
     * observed exclusive counts are zero, so Neyman gives them zero variance. Only then, and
     * only within the singular group of cell-linked bins, are the zero-exclusive bins floored
     * to half an event of exclusive variance.
     * @param n         Observed collapsed counts (the Neyman variance).
     * @param cells     The dataset's own shared cells.
     * @param idx       Collapsed bins entering the fit, in order.
     * @param n_floored Optional output: how many bins received the floor.
     */
    Eigen::MatrixXf OverlapStatCovariance(const Eigen::VectorXf &n, const OverlapCells &cells,
                                          const std::vector<Eigen::Index> &idx, int *n_floored = nullptr);

    /**
     * @brief Inverse of an OverlapStatCovariance matrix, by Cholesky in double precision.
     * @details Positive definite by construction (the singular-block floor), so a failed
     * factorisation is logged and throws rather than being papered over.
     */
    Eigen::MatrixXd OverlapStatPrecision(const Eigen::MatrixXd &C);

    /**
     * @brief Which MC entries can be the same interaction, built once at process time.
     * @details Stored in `<tag>_overlap_mc.bin` (not in `_prop.bin`, which stays untouched).
     */
    struct PROoverlapMC {
        uint32_t hash = 0;                         ///< Config hash of the PROpeller these entries index.
        uint64_t n_event = 0;                      ///< PROpeller::NEvent() when built.
        std::vector<uint8_t> entry_class;          ///< Per entry: 0 file not flagged, 1 flagged, 2 flagged and common.
        std::vector<std::vector<uint32_t>> groups; ///< Entries one shared interaction produced (two or more, distinct channels, all common).

        template<class Archive>
            void serialize(Archive &ar, [[maybe_unused]] const unsigned int version) {
                ar & hash;
                ar & n_event;
                ar & entry_class;
                ar & groups;
            }

        void save(const std::string &filename) const;
        void load(const std::string &filename);
    };

    /**
     * @brief Evaluates an `overlap_matching_vars` (or `overlap_common`) list on a TChain.
     * @details One TTreeFormula per expression, refreshed when the chain moves to a new file.
     * Values are kept as doubles, exact for integer identifiers up to 2^53.
     */
    class OverlapKeyReader {
        public:
            OverlapKeyReader(const std::string &name, const std::vector<std::string> &exprs, TTree *chain);
            ~OverlapKeyReader();
            size_t Arity() const { return formulas.size(); }
            /// Evaluate every expression at the chain's current entry (GetEntry already called).
            void Eval(double *out);
        private:
            TTree *chain;
            int tree_number = -1;
            std::vector<std::unique_ptr<TTreeFormula>> formulas;
    };

    /**
     * @brief Keys recorded while processing one sample with overlap_matching_vars.
     * @details One record per filled entry: its key, the global channel it fills and a
     * payload index (the PROpeller entry for MC, the record number for data). Keys only
     * match within one (mode, detector) block: one interaction never reaches two
     * detectors, and productions for different detectors may reuse run/event numbers.
     */
    struct OverlapRecords {
        size_t arity = 0;
        size_t channels_per_block = 1; ///< PROconfig::m_num_channels: global channel / this = (mode, detector) block.
        std::vector<double> keys;      ///< arity values per record.
        std::vector<int> channel;      ///< Global (mode, detector, channel) index per record.
        std::vector<uint32_t> payload; ///< PROpeller entry (MC) or data record index.
        std::vector<uint8_t> common;   ///< MC only: the entry is in the common sample.

        void add(const double *key, int chan, uint32_t pay, bool is_common = true);
        size_t size() const { return channel.size(); }

        /**
         * @brief Records grouped by identical key (two or more per group, record indices).
         * @details Fatal if one key fills the same channel twice: the key then does not
         * identify a single interaction (add a slice/interaction ID). @p what names the
         * sample in messages.
         */
        std::vector<std::vector<size_t>> Join(const std::string &what) const;
    };

    /**
     * @brief Build the MC overlap structure from the records of a processing pass.
     * @details Groups are formed from common records only. A key shared between a common
     * and a non-common entry means the two analyses' overlap_common cuts disagree: those
     * entries are left out and counted in a warning. Fatal if no group is found at all
     * (the opt-in is then almost certainly misconfigured).
     */
    PROoverlapMC BuildOverlapMC(const OverlapRecords &rec, size_t n_event, uint32_t hash);

    /** @brief Save / load the real data's per-variable cells (`<tag>_overlap_data.bin`). */
    void SaveDataOverlap(const std::string &filename, uint32_t hash, const std::vector<OverlapCells> &cells);
    std::vector<OverlapCells> LoadDataOverlap(const std::string &filename, uint32_t &hash);

    /** @brief Global (mode, detector, channel) index of a global subchannel index. */
    int OverlapChannelOfSubchannel(const PROconfig &config, size_t subchannel_index);

    /**
     * @brief Expected-overlap fractions at one physics point, for one variable.
     * @details frac[s][k] = F_S^(i) for cell s and its k-th bin i: the share of bin i's
     * expected count that sits in S,
     *   F_S^(i) = (shared_S,i / common_i) · (flagged_i / total_i),
     * every term a weight × oscillation-probability sum over MC entries in bin i (shared: the
     * cell's entries in bin i; common, flagged, total: entries of that class or any). The
     * fractions are physics-only (CV weights): systematics enter through the marginals the
     * cells are scaled to, as PROfit's per-bin spline and covariance model implies.
     */
    struct OverlapFractions {
        size_t nbins = 0;                       ///< Collapsed bins of the variable.
        std::vector<std::vector<int>> bins;     ///< Per cell: sorted collapsed bins.
        std::vector<std::vector<double>> frac;  ///< Per cell, per bin of the cell: F_S^(i).
    };

    /**
     * @brief Measure the expected-overlap fractions from the MC overlap structure.
     * @param config  Analysis configuration.
     * @param prop    MC store; must carry `overlap` (fatal otherwise).
     * @param model   Physics model (fatal for visible-decay models: per-entry P is undefined).
     * @param phys    Physics parameters of the truth.
     * @param var     Variable index.
     */
    OverlapFractions ComputeOverlapFractions(const PROconfig &config, const PROpeller &prop, const PROmodel &model,
                                             const Eigen::VectorXf &phys, size_t var);

    /**
     * @brief Expected cells at collapsed marginals @p mu.
     * @details λ_S = geomean_{i∈S}(F_S^(i) μ_i) (= ρ√(μ_a μ_b) for a pair), then clipped so
     * no bin's shared part exceeds μ_i. Bins with μ_i <= 0 carry no shared events.
     */
    OverlapCells ExpectedOverlapCells(const OverlapFractions &f, const Eigen::VectorXf &mu);

    /**
     * @brief The toy-data step every thrower shares.
     * @details Without @p fractions this is exactly the historical PoissonVariation (or the
     * negative-bin clamp when @p poisson is false), so toys are bitwise unchanged. With them
     * each cell and each bin's exclusive remainder is thrown Poisson (expected values when
     * @p poisson is false) and the returned data carries its own cells, which its Neyman
     * fit then uses.
     * @param collapsed  Collapsed toy mean (spectrum and error).
     * @param fractions  Expected-overlap fractions at the toy truth, or nullptr.
     * @param poisson    Poisson-fluctuate (false: the mean itself, negative bins clamped).
     * @param seed       RNG seed for the fluctuation (ignored when @p poisson is false).
     */
    PROdata MakeToyData(const PROspec &collapsed, const OverlapFractions *fractions, bool poisson, uint32_t seed);

    /** @brief One-line description of a dataset's overlap (cells, shared events, max ρ̂) for logs. */
    std::string DescribeOverlap(const PROdata &data);

    /** @brief 64-bit fingerprint of an MC overlap structure (provenance of toy artifacts). */
    uint64_t OverlapFingerprint(const PROoverlapMC &ov);

}

#endif
