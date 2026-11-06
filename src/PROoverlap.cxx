#include "PROoverlap.h"
#include "PROconfig.h"
#include "PROdata.h"
#include "PROlog.h"
#include "PROmodel.h"
#include "PROpeller.h"
#include "PROspec.h"
#include "MurmurHash3.h"

#include "TTree.h"
#include "TTreeFormula.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace PROfit {

namespace {
    /// Exclusive variance (events) given to a zero-exclusive bin of a singular stat block.
    constexpr double kOverlapExclusiveFloor = 0.5;
}

Eigen::MatrixXf OverlapStatCovariance(const Eigen::VectorXf &n, const OverlapCells &cells,
                                      const std::vector<Eigen::Index> &idx, int *n_floored) {
    const Eigen::Index nb = (Eigen::Index)idx.size();
    std::vector<Eigen::Index> pos(n.size(), -1);
    for(Eigen::Index k = 0; k < nb; ++k) pos[idx[k]] = k;

    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(nb, nb);
    for(Eigen::Index k = 0; k < nb; ++k) V(k, k) = n(idx[k]);

    // Bins tied together by cells form components (union-find); singularity is judged per component.
    std::vector<Eigen::Index> root(nb);
    std::iota(root.begin(), root.end(), 0);
    auto find = [&](Eigen::Index k) {
        while(root[k] != k) k = root[k] = root[root[k]];
        return k;
    };
    std::vector<double> shared(nb, 0.0);
    std::vector<char> in_cell(nb, 0);
    std::vector<Eigen::Index> p;
    for(size_t c = 0; c < cells.size(); ++c) {
        p.clear();
        for(int b : cells.bins[c])
            if(b >= 0 && b < n.size() && pos[b] >= 0) p.push_back(pos[b]);
        if(p.size() < 2) continue;
        const double cnt = cells.count[c];
        for(size_t i = 0; i < p.size(); ++i) {
            shared[p[i]] += cnt;
            in_cell[p[i]] = 1;
            root[find(p[i])] = find(p[0]);
            for(size_t j = i + 1; j < p.size(); ++j) {
                V(p[i], p[j]) += cnt;
                V(p[j], p[i]) += cnt;
            }
        }
    }

    std::map<Eigen::Index, std::vector<Eigen::Index>> components;
    for(Eigen::Index k = 0; k < nb; ++k) if(in_cell[k]) components[find(k)].push_back(k);
    int floored = 0;
    for(const auto &[r, sub] : components) {
        // Each pivot over its own diagonal is the share of that bin's variance the earlier
        // pivots do not explain: scale-free, so ND and sparse FD bins are judged alike.
        const Eigen::MatrixXd Vs = V(sub, sub);
        Eigen::LDLT<Eigen::MatrixXd> ldlt(Vs);
        const Eigen::VectorXd D = ldlt.vectorD();
        const Eigen::VectorXd diag = ldlt.transpositionsP() * Vs.diagonal();
        bool singular = ldlt.info() != Eigen::Success;
        for(Eigen::Index k = 0; k < D.size() && !singular; ++k)
            singular = !(D(k) > 1e-6 * diag(k));
        if(!singular) continue;
        for(Eigen::Index k : sub) {
            const double excl = std::max(V(k, k) - shared[k], 0.0);
            if(excl < kOverlapExclusiveFloor) {
                V(k, k) += kOverlapExclusiveFloor - excl;
                ++floored;
            }
        }
    }
    if(n_floored) *n_floored = floored;
    return V.cast<float>();
}

Eigen::MatrixXd OverlapStatPrecision(const Eigen::MatrixXd &C) {
    const Eigen::LLT<Eigen::MatrixXd> llt(C);
    if(llt.info() != Eigen::Success) {
        log<LOG_ERROR>(L"%1% || ERROR: the %2%-bin statistical overlap covariance is not positive definite.") % __func__ % C.rows();
        throw std::runtime_error("statistical overlap covariance not positive definite");
    }
    return llt.solve(Eigen::MatrixXd::Identity(C.rows(), C.cols()));
}

void PROoverlapMC::save(const std::string &filename) const {
    std::ofstream ofs(filename, std::ios::binary);
    boost::archive::binary_oarchive oa(ofs);
    oa << *this;
    log<LOG_INFO>(L"%1% || Saved MC overlap structure (%2% shared interactions) to %3%") % __func__ % groups.size() % filename.c_str();
}

void PROoverlapMC::load(const std::string &filename) {
    std::ifstream ifs(filename, std::ios::binary);
    boost::archive::binary_iarchive ia(ifs);
    ia >> *this;
    log<LOG_INFO>(L"%1% || Loaded MC overlap structure (%2% shared interactions) from %3%") % __func__ % groups.size() % filename.c_str();
}

OverlapKeyReader::OverlapKeyReader(const std::string &name, const std::vector<std::string> &exprs, TTree *t) : chain(t) {
    for(size_t k = 0; k < exprs.size(); ++k) {
        formulas.push_back(std::make_unique<TTreeFormula>((name + "_" + std::to_string(k)).c_str(), exprs[k].c_str(), t));
        if(formulas.back()->GetNdim() == 0) {
            log<LOG_ERROR>(L"%1% || ERROR: overlap expression '%2%' does not compile against the tree.") % __func__ % exprs[k].c_str();
            throw std::invalid_argument("bad overlap expression: " + exprs[k]);
        }
    }
}

OverlapKeyReader::~OverlapKeyReader() = default;

void OverlapKeyReader::Eval(double *out) {
    const int tn = chain->GetTreeNumber();
    if(tn != tree_number) {
        tree_number = tn;
        for(auto &f : formulas) {
            f->UpdateFormulaLeaves();
            f->GetNdata();
        }
    }
    for(size_t k = 0; k < formulas.size(); ++k) out[k] = formulas[k]->EvalInstance();
}

void OverlapRecords::add(const double *key, int chan, uint32_t pay, bool is_common) {
    keys.insert(keys.end(), key, key + arity);
    channel.push_back(chan);
    payload.push_back(pay);
    common.push_back(is_common ? 1 : 0);
}

std::vector<std::vector<size_t>> OverlapRecords::Join(const std::string &what) const {
    const size_t nrec = size();
    std::vector<size_t> order(nrec);
    std::iota(order.begin(), order.end(), 0);
    auto key_of = [&](size_t r) { return keys.data() + r * arity; };
    auto block_of = [&](size_t r) { return (size_t)channel[r] / channels_per_block; };
    auto same = [&](size_t a, size_t b) {
        return block_of(a) == block_of(b) && std::equal(key_of(a), key_of(a) + arity, key_of(b));
    };
    // Stable, so records of one key keep their fill order: groups are deterministic.
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if(block_of(a) != block_of(b)) return block_of(a) < block_of(b);
        return std::lexicographical_compare(key_of(a), key_of(a) + arity, key_of(b), key_of(b) + arity);
    });

    std::vector<std::vector<size_t>> groups;
    for(size_t i = 0; i < nrec;) {
        size_t j = i + 1;
        while(j < nrec && same(order[i], order[j])) ++j;
        if(j - i >= 2) {
            std::vector<size_t> g(order.begin() + i, order.begin() + j);
            for(size_t a = 0; a < g.size(); ++a)
                for(size_t b = a + 1; b < g.size(); ++b)
                    if(channel[g[a]] == channel[g[b]]) {
                        std::ostringstream ks;
                        for(size_t k = 0; k < arity; ++k) ks << (k ? "," : "") << std::setprecision(17) << key_of(g[a])[k];
                        log<LOG_ERROR>(L"%1% || ERROR (%2%): overlap key (%3%) fills channel %4% more than once. overlap_matching_vars must identify a single interaction; add a slice/interaction ID.")
                            % __func__ % what.c_str() % ks.str().c_str() % channel[g[a]];
                        throw std::runtime_error("overlap key is not unique per interaction");
                    }
            groups.push_back(std::move(g));
        }
        i = j;
    }
    return groups;
}

PROoverlapMC BuildOverlapMC(const OverlapRecords &rec, size_t n_event, uint32_t hash) {
    PROoverlapMC ov;
    ov.hash = hash;
    ov.n_event = n_event;
    ov.entry_class.assign(n_event, 0);
    size_t n_common = 0;
    for(size_t r = 0; r < rec.size(); ++r) {
        ov.entry_class[rec.payload[r]] = rec.common[r] ? 2 : 1;
        n_common += rec.common[r];
    }

    size_t n_mixed = 0;
    for(const auto &g : rec.Join("MC")) {
        std::vector<uint32_t> entries;
        for(size_t r : g)
            if(rec.common[r]) entries.push_back(rec.payload[r]);
        if(!entries.empty() && entries.size() != g.size()) ++n_mixed;
        if(entries.size() < 2) continue;
        std::sort(entries.begin(), entries.end());
        ov.groups.push_back(std::move(entries));
    }

    log<LOG_INFO>(L"%1% || MC overlap: %2% flagged entries (%3% common), %4% shared interactions.")
        % __func__ % rec.size() % n_common % ov.groups.size();
    if(n_mixed)
        log<LOG_WARNING>(L"%1% || MC overlap: %2% keys pair a common entry with a non-common one; the overlap_common cuts of the two analyses disagree. Those entries are not counted as shared.")
            % __func__ % n_mixed;
    if(ov.groups.empty()) {
        log<LOG_ERROR>(L"%1% || ERROR: overlap_matching_vars is set on MC MCFiles but no MC interaction is shared between channels. Check the matching expressions and overlap_common cuts.") % __func__;
        throw std::runtime_error("MC overlap enabled but nothing matched");
    }
    return ov;
}

void SaveDataOverlap(const std::string &filename, uint32_t hash, const std::vector<OverlapCells> &cells) {
    std::ofstream ofs(filename, std::ios::binary);
    boost::archive::binary_oarchive oa(ofs);
    oa << hash;
    oa << cells;
}

std::vector<OverlapCells> LoadDataOverlap(const std::string &filename, uint32_t &hash) {
    std::vector<OverlapCells> cells;
    std::ifstream ifs(filename, std::ios::binary);
    boost::archive::binary_iarchive ia(ifs);
    ia >> hash;
    ia >> cells;
    return cells;
}

int OverlapChannelOfSubchannel(const PROconfig &config, size_t subchannel_index) {
    const size_t per_block = std::accumulate(config.m_num_subchannels.begin(), config.m_num_subchannels.end(), (size_t)0);
    const size_t block = subchannel_index / per_block;
    return (int)(block * config.m_num_channels + config.GetLocalChannelIndexFromGlobalSubchannelIndex(subchannel_index));
}

OverlapFractions ComputeOverlapFractions(const PROconfig &config, const PROpeller &prop, const PROmodel &model,
                                         const Eigen::VectorXf &phys, size_t var) {
    if(!prop.overlap) {
        log<LOG_ERROR>(L"%1% || ERROR: no MC overlap structure loaded (overlap_matching_vars on the MC MCFiles).") % __func__;
        throw std::runtime_error("ComputeOverlapFractions without MC overlap");
    }
    if(model.uses_get_counts()) {
        log<LOG_ERROR>(L"%1% || ERROR: statistical overlap is not supported for visible-decay models (no per-entry probability).") % __func__;
        throw std::runtime_error("overlap with a visible-decay model");
    }
    const PROoverlapMC &ov = *prop.overlap;
    const size_t N = prop.NEvent();
    if(ov.n_event != N || ov.entry_class.size() != N) {
        log<LOG_ERROR>(L"%1% || ERROR: MC overlap structure indexes %2% entries but the PROpeller has %3%; re-run process.") % __func__ % ov.n_event % N;
        throw std::runtime_error("MC overlap / PROpeller mismatch");
    }

    const Eigen::SparseMatrix<float> &T = config.GetCollapsingMatrixSparse((int)var);
    std::vector<int> to_coll(T.rows(), -1);
    for(Eigen::Index c = 0; c < T.outerSize(); ++c)
        for(Eigen::SparseMatrix<float>::InnerIterator it(T, c); it; ++it) to_coll[it.row()] = (int)c;
    auto coll_bin = [&](size_t e) {
        const int b = prop.VariableBinIndex(var, e);
        return b < 0 ? -1 : to_coll[b];
    };

    OverlapFractions f;
    f.nbins = (size_t)T.cols();

    // Cells: the distinct bin sets the shared interactions fill in this variable.
    std::map<std::vector<int>, int> cell_index;
    std::vector<int> group_cell(ov.groups.size(), -1);
    for(size_t g = 0; g < ov.groups.size(); ++g) {
        std::vector<int> bins;
        for(uint32_t e : ov.groups[g]) {
            const int b = coll_bin(e);
            if(b >= 0) bins.push_back(b);
        }
        std::sort(bins.begin(), bins.end());
        bins.erase(std::unique(bins.begin(), bins.end()), bins.end());
        if(bins.size() < 2) continue;
        auto [it, inserted] = cell_index.emplace(bins, (int)f.bins.size());
        if(inserted) f.bins.push_back(bins);
        group_cell[g] = it->second;
    }
    f.frac.resize(f.bins.size());
    for(size_t s = 0; s < f.bins.size(); ++s) f.frac[s].assign(f.bins[s].size(), 0.0);
    if(f.bins.empty()) return f;

    std::vector<char> needed_bin(f.nbins, 0);
    for(const auto &bins : f.bins) for(int b : bins) needed_bin[b] = 1;
    std::vector<uint32_t> needed;
    for(size_t e = 0; e < N; ++e) {
        const int b = coll_bin(e);
        if(b >= 0 && needed_bin[b]) needed.push_back((uint32_t)e);
    }

    // Per-entry oscillation probability, as FillSpectra's event-by-event path evaluates it.
    std::vector<float> P(needed.size(), 1.0f);
    if(!model.is_trivial) {
        const size_t chunk = 1 << 16;
        for(size_t start = 0; start < needed.size(); start += chunk) {
            const size_t len = std::min(chunk, needed.size() - start);
            std::vector<std::vector<float>> var_arrs(model.ivars.size(), std::vector<float>(len));
            for(size_t k = 0; k < model.ivars.size(); ++k)
                for(size_t j = 0; j < len; ++j)
                    var_arrs[k][j] = prop.VariableValue(model.ivars[k], needed[start + j]);
            const Eigen::MatrixXf probs = model.get_probs(phys, var_arrs);
            for(size_t j = 0; j < len; ++j)
                P[start + j] = probs((Eigen::Index)j, prop.model_rule[needed[start + j]]);
        }
    }

    std::vector<double> total(f.nbins, 0.0), flagged(f.nbins, 0.0), common(f.nbins, 0.0);
    for(size_t j = 0; j < needed.size(); ++j) {
        const uint32_t e = needed[j];
        const double w = (double)prop.added_weights[e] * P[j];
        const int b = coll_bin(e);
        total[b] += w;
        if(ov.entry_class[e] >= 1) flagged[b] += w;
        if(ov.entry_class[e] == 2) common[b] += w;
    }

    std::vector<std::vector<double>> shared(f.bins.size());
    for(size_t s = 0; s < f.bins.size(); ++s) shared[s].assign(f.bins[s].size(), 0.0);
    for(size_t g = 0; g < ov.groups.size(); ++g) {
        const int s = group_cell[g];
        if(s < 0) continue;
        for(uint32_t e : ov.groups[g]) {
            const int b = coll_bin(e);
            if(b < 0) continue;
            const size_t k = std::lower_bound(f.bins[s].begin(), f.bins[s].end(), b) - f.bins[s].begin();
            const size_t j = std::lower_bound(needed.begin(), needed.end(), e) - needed.begin();
            shared[s][k] += (double)prop.added_weights[e] * P[j];
        }
    }

    for(size_t s = 0; s < f.bins.size(); ++s)
        for(size_t k = 0; k < f.bins[s].size(); ++k) {
            const int b = f.bins[s][k];
            if(common[b] > 0 && total[b] > 0)
                f.frac[s][k] = std::clamp(shared[s][k] / common[b] * (flagged[b] / total[b]), 0.0, 1.0);
        }
    return f;
}

OverlapCells ExpectedOverlapCells(const OverlapFractions &f, const Eigen::VectorXf &mu) {
    const size_t ncell = f.bins.size();
    std::vector<double> lam(ncell, 0.0);
    for(size_t s = 0; s < ncell; ++s) {
        double logsum = 0.0;
        bool ok = true;
        for(size_t k = 0; k < f.bins[s].size() && ok; ++k) {
            const double v = f.frac[s][k] * (double)mu(f.bins[s][k]);
            if(v > 0) logsum += std::log(v);
            else ok = false;
        }
        if(ok) lam[s] = std::exp(logsum / (double)f.bins[s].size());
    }

    // Clip so no bin's shared part exceeds its mean. Shrinking a cell only lowers the other
    // bins' sums, so one pass in bin order leaves every bin within its mean.
    std::vector<std::vector<size_t>> cells_of_bin(f.nbins);
    std::vector<double> sum(f.nbins, 0.0);
    for(size_t s = 0; s < ncell; ++s)
        for(int b : f.bins[s]) {
            cells_of_bin[b].push_back(s);
            sum[b] += lam[s];
        }
    for(size_t b = 0; b < f.nbins; ++b) {
        const double m = std::max((double)mu(b), 0.0);
        if(sum[b] <= m || sum[b] <= 0) continue;
        const double r = m / sum[b];
        for(size_t s : cells_of_bin[b]) {
            const double drop = lam[s] * (1.0 - r);
            for(int b2 : f.bins[s]) sum[b2] -= drop;
            lam[s] *= r;
        }
    }

    OverlapCells out;
    for(size_t s = 0; s < ncell; ++s)
        if(lam[s] > 0) {
            out.bins.push_back(f.bins[s]);
            out.count.push_back((float)lam[s]);
        }
    return out;
}

PROdata MakeToyData(const PROspec &collapsed, const OverlapFractions *fractions, bool poisson, uint32_t seed) {
    if(!fractions) {
        const PROspec s = poisson ? PROspec::PoissonVariation(collapsed, seed)
                                  : PROspec(collapsed.Spec().cwiseMax(0.0f), collapsed.Error());
        return PROdata(s.Spec(), s.Error());
    }
    if((size_t)collapsed.Spec().size() != fractions->nbins) {
        log<LOG_ERROR>(L"%1% || ERROR: toy spectrum has %2% bins but the overlap fractions cover %3%.") % __func__ % collapsed.Spec().size() % fractions->nbins;
        throw std::runtime_error("MakeToyData size mismatch");
    }

    const Eigen::VectorXf mu = collapsed.Spec().cwiseMax(0.0f);
    OverlapCells expected = ExpectedOverlapCells(*fractions, mu);
    if(!poisson) {
        PROdata d(mu, collapsed.Error());
        d.has_overlap = true;
        d.overlap = std::move(expected);
        return d;
    }

    std::mt19937 gen(seed);
    Eigen::VectorXf n = Eigen::VectorXf::Zero(mu.size());
    std::vector<double> shared(mu.size(), 0.0);
    OverlapCells thrown;
    for(size_t s = 0; s < expected.size(); ++s) {
        for(int b : expected.bins[s]) shared[b] += expected.count[s];
        std::poisson_distribution<> d((double)expected.count[s]);
        const int k = d(gen);
        if(k <= 0) continue;
        for(int b : expected.bins[s]) n(b) += (float)k;
        thrown.bins.push_back(expected.bins[s]);
        thrown.count.push_back((float)k);
    }
    for(Eigen::Index b = 0; b < mu.size(); ++b) {
        const double excl = (double)mu(b) - shared[b];
        if(excl <= 0) continue;
        std::poisson_distribution<> d(excl);
        n(b) += (float)d(gen);
    }

    PROdata d(n, n.array().sqrt());
    d.has_overlap = true;
    d.overlap = std::move(thrown);
    return d;
}

std::string DescribeOverlap(const PROdata &data) {
    if(!data.has_overlap) return "no overlap information";
    double shared = 0.0, max_rho = 0.0;
    const Eigen::VectorXf &n = data.Spec();
    for(size_t c = 0; c < data.overlap.size(); ++c) {
        const auto &bins = data.overlap.bins[c];
        const double cnt = data.overlap.count[c];
        shared += cnt;
        for(size_t i = 0; i < bins.size(); ++i)
            for(size_t j = i + 1; j < bins.size(); ++j) {
                const double den = std::sqrt((double)n(bins[i]) * (double)n(bins[j]));
                if(den > 0) max_rho = std::max(max_rho, cnt / den);
            }
    }
    std::ostringstream ss;
    ss << data.overlap.size() << " shared cells holding " << shared << " shared events (of " << n.sum()
       << " in all bins), max correlation " << max_rho;
    return ss.str();
}

uint64_t OverlapFingerprint(const PROoverlapMC &ov) {
    std::vector<uint32_t> flat = {ov.hash, (uint32_t)ov.n_event, (uint32_t)ov.groups.size()};
    for(const auto &g : ov.groups) {
        flat.push_back((uint32_t)g.size());
        flat.insert(flat.end(), g.begin(), g.end());
    }
    uint64_t h[2] = {0, 0}, hc[2] = {0, 0};
    MurmurHash3_x64_128(flat.data(), (int)(flat.size() * sizeof(uint32_t)), 404, h);
    MurmurHash3_x64_128(ov.entry_class.data(), (int)ov.entry_class.size(), 404, hc);
    return h[0] ^ h[1] ^ hc[0] ^ (hc[1] << 1);
}

}
