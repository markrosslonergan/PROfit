/**
 * @file PROAdaptiveFCbank.cxx
 * @brief Adaptive Feldman-Cousins: PE bank / mesh / brazil-archive
 *        serialisation, per-cell PE worker and scheduler, asimov
 *        observables, and bank classification.
 * @author PROfit Collaboration
 *
 * @details Part of the adaptive FC pipeline (see inc/PROAdaptiveFC.h and
 * src/PROAdaptiveFCinternal.h for the file layout). All boost archive I/O
 * for the adaptive-FC types lives in this translation unit.
 */
#include "PROAdaptiveFCinternal.h"

#include "PROlog.h"
#include "PROmetrics/PROchi_neyman.h"
#include "PROmetrics/PROchi_pearson.h"
#include "PROmetrics/PROchi_CNP.h"
#include "PROmetrics/PROpoisson.h"
#include "PROmetrics/PROmetricFactory.h"
#include "PROmetric.h"
#include "PROspec.h"
#include "PROcess.h"
#include "PROtocall.h"
#include "PROserial.h"
#include "MurmurHash3.h"

#include <Eigen/Eigen>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

// ====================================================================
//  PEBank / MetaMesh / BrazilArchive boost serialisation.
//
//  Layout note: boost::serialization overloads must live in the boost
//  namespace, so they are declared before namespace PROfit opens. All
//  archive I/O for these types stays in this translation unit.
// ====================================================================

namespace boost { namespace serialization {

template <class Archive>
void serialize(Archive &ar, PROfit::PEBankRecord &r, [[maybe_unused]] const unsigned int v) {
    ar & r.chi2_syst;
    ar & r.chi2_osc;
    ar & r.dchi2;
    ar & r.seed;
}

template <class Archive>
void serialize(Archive &ar, PROfit::PEBank &b, [[maybe_unused]] const unsigned int v) {
    ar & b.finest_nx;
    ar & b.finest_ny;
    ar & b.max_levels;
    ar & b.x_lo & b.x_hi & b.y_lo & b.y_hi;
    ar & b.n_cells;
    ar & b.cell_pes;
    ar & b.cell_center_x;
    ar & b.cell_center_y;
    ar & b.cell_i_bl;
    ar & b.cell_j_bl;
    ar & b.cell_step;
    ar & b.cell_level;
}

template <class Archive>
void serialize(Archive &ar, PROfit::MetaCell &c, [[maybe_unused]] const unsigned int v) {
    ar & c.i_bl;
    ar & c.j_bl;
    ar & c.step;
    ar & c.level;
    ar & c.per_level_refine_count;
}

template <class Archive>
void serialize(Archive &ar, PROfit::BrazilArchive &a, [[maybe_unused]] const unsigned int v) {
    ar & a.finest_nx;
    ar & a.finest_ny;
    ar & a.n_cells;
    ar & a.per_throw_global_chi2;
    ar & a.per_throw_dchi2;
}

template <class Archive>
void serialize(Archive &ar, PROfit::ProfileManifest &m, [[maybe_unused]] const unsigned int v) {
    ar & m.truth_source;
    ar & m.profile_idx;
    ar & m.data_fp;
    ar & m.config_hash;
    ar & m.model_tag;
    ar & m.chi2;
    ar & m.shape_only;
}

template <class Archive>
void serialize(Archive &ar, PROfit::MetaMesh &m, [[maybe_unused]] const unsigned int v) {
    ar & m.cells;
    ar & m.finest_nx;
    ar & m.finest_ny;
    ar & m.max_levels;
    ar & m.x_lo & m.x_hi & m.y_lo & m.y_hi;
    ar & m.n_baseline_cells;
    ar & m.n_refined_cells;
}

}} // namespace boost::serialization

namespace PROfit {

bool save_bank(const PEBank &bank, const std::string &path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        log<LOG_ERROR>(L"%1% || save_bank: could not open %2% for writing.") % __func__ % path.c_str();
        return false;
    }
    try {
        boost::archive::binary_oarchive oa(ofs);
        // Legacy banks keep the v2 layout byte for byte; profile banks append a tail.
        const bool profiled = bank.manifest.profiled();
        uint32_t magic = PEBank::MAGIC;
        uint16_t version = profiled ? PEBank::VERSION : 2;
        oa & magic;
        oa & version;
        oa & const_cast<PEBank &>(bank); // boost serialize() takes non-const ref
        if (profiled) {
            oa & const_cast<ProfileManifest &>(bank.manifest);
            oa & const_cast<std::vector<std::vector<float>> &>(bank.cell_truth_phys);
            // phys_hat flattened per cell, in PE order.
            std::vector<std::vector<float>> phat(bank.cell_pes.size());
            for (size_t c = 0; c < bank.cell_pes.size(); ++c)
                for (const auto &r : bank.cell_pes[c])
                    phat[c].insert(phat[c].end(), r.phys_hat.begin(), r.phys_hat.end());
            oa & phat;
        }
    } catch (const std::exception &e) {
        log<LOG_ERROR>(L"%1% || save_bank: serialisation error: %2%") % __func__ % e.what();
        return false;
    }
    int64_t total_pes = 0;
    for (const auto &v : bank.cell_pes) total_pes += (int64_t)v.size();
    log<LOG_INFO>(L"%1% || save_bank: wrote %2% (cells=%3%, total_pes=%4%).")
        % __func__ % path.c_str() % bank.n_cells % total_pes;
    return true;
}

bool load_bank(PEBank &bank_out, const std::string &path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        log<LOG_ERROR>(L"%1% || load_bank: could not open %2% for reading.") % __func__ % path.c_str();
        return false;
    }
    try {
        boost::archive::binary_iarchive ia(ifs);
        uint32_t magic = 0;
        uint16_t version = 0;
        ia & magic;
        ia & version;
        if (magic != PEBank::MAGIC) {
            log<LOG_ERROR>(L"%1% || load_bank: bad magic in %2% (got 0x%3$08x, expected 0x%4$08x).")
                % __func__ % path.c_str() % magic % (uint32_t)PEBank::MAGIC;
            return false;
        }
        if (version != 2 && version != PEBank::VERSION) {
            log<LOG_ERROR>(L"%1% || load_bank: version mismatch in %2% (got %3%, expected 2 or %4%).")
                % __func__ % path.c_str() % (int)version % (int)PEBank::VERSION;
            return false;
        }
        ia & bank_out;
        bank_out.manifest = ProfileManifest{};
        bank_out.cell_truth_phys.clear();
        if (version >= 3) {
            ia & bank_out.manifest;
            ia & bank_out.cell_truth_phys;
            std::vector<std::vector<float>> phat;
            ia & phat;
            const size_t nprof = bank_out.manifest.profile_idx.size();
            for (size_t c = 0; c < bank_out.cell_pes.size() && c < phat.size(); ++c) {
                auto &pes = bank_out.cell_pes[c];
                if (phat[c].size() != pes.size() * nprof) {
                    log<LOG_ERROR>(L"%1% || load_bank: %2% cell %3% has %4% phys_hat values for %5% PEs; corrupt bank.")
                        % __func__ % path.c_str() % (int)c % (int)phat[c].size() % (int)pes.size();
                    return false;
                }
                for (size_t k = 0; k < pes.size(); ++k)
                    pes[k].phys_hat.assign(phat[c].begin() + k * nprof, phat[c].begin() + (k + 1) * nprof);
            }
        }
    } catch (const std::exception &e) {
        log<LOG_ERROR>(L"%1% || load_bank: deserialisation error: %2%") % __func__ % e.what();
        return false;
    }
    int64_t total_pes = 0;
    for (const auto &v : bank_out.cell_pes) total_pes += (int64_t)v.size();
    log<LOG_INFO>(L"%1% || load_bank: read %2% (cells=%3%, total_pes=%4%).")
        % __func__ % path.c_str() % bank_out.n_cells % total_pes;
    return true;
}

bool save_mesh(const MetaMesh &mm, const std::string &path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        log<LOG_ERROR>(L"%1% || save_mesh: could not open %2% for writing.") % __func__ % path.c_str();
        return false;
    }
    try {
        boost::archive::binary_oarchive oa(ofs);
        uint32_t magic = MetaMesh::MAGIC;
        uint16_t version = MetaMesh::VERSION;
        oa & magic;
        oa & version;
        oa & const_cast<MetaMesh &>(mm);
    } catch (const std::exception &e) {
        log<LOG_ERROR>(L"%1% || save_mesh: serialisation error: %2%") % __func__ % e.what();
        return false;
    }
    log<LOG_INFO>(L"%1% || save_mesh: wrote %2% (cells=%3%, finest=%4%x%5%).")
        % __func__ % path.c_str() % (int)mm.cells.size() % mm.finest_nx % mm.finest_ny;
    return true;
}

bool load_mesh(MetaMesh &mm_out, const std::string &path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false; // Silent: caller decides whether the absence is an error.
    try {
        boost::archive::binary_iarchive ia(ifs);
        uint32_t magic = 0;
        uint16_t version = 0;
        ia & magic;
        ia & version;
        if (magic != MetaMesh::MAGIC) {
            log<LOG_ERROR>(L"%1% || load_mesh: bad magic in %2% (got 0x%3$08x, expected 0x%4$08x).")
                % __func__ % path.c_str() % magic % (uint32_t)MetaMesh::MAGIC;
            return false;
        }
        if (version != MetaMesh::VERSION) {
            log<LOG_ERROR>(L"%1% || load_mesh: version mismatch in %2% (got %3%, expected %4%).")
                % __func__ % path.c_str() % (int)version % (int)MetaMesh::VERSION;
            return false;
        }
        ia & mm_out;
    } catch (const std::exception &e) {
        log<LOG_ERROR>(L"%1% || load_mesh: deserialisation error: %2%") % __func__ % e.what();
        return false;
    }
    log<LOG_INFO>(L"%1% || load_mesh: read %2% (cells=%3%, finest=%4%x%5%).")
        % __func__ % path.c_str() % (int)mm_out.cells.size() % mm_out.finest_nx % mm_out.finest_ny;
    return true;
}

// --------------------------------------------------------------------
//  BrazilArchive boost-serialisation. Same magic+version+payload layout
//  as save_bank / save_mesh.
// --------------------------------------------------------------------
bool save_brazil_archive(const BrazilArchive &arc, const std::string &path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        log<LOG_ERROR>(L"%1% || save_brazil_archive: could not open %2% for writing.") % __func__ % path.c_str();
        return false;
    }
    try {
        boost::archive::binary_oarchive oa(ofs);
        // Legacy archives keep the v1 layout byte for byte; profile archives append a tail.
        const bool profiled = arc.manifest.profiled();
        uint32_t magic = BrazilArchive::MAGIC;
        uint16_t version = profiled ? BrazilArchive::VERSION : 1;
        oa & magic;
        oa & version;
        oa & const_cast<BrazilArchive &>(arc);
        if (profiled) {
            oa & const_cast<ProfileManifest &>(arc.manifest);
            oa & const_cast<std::vector<std::vector<float>> &>(arc.per_throw_phys_hat);
        }
    } catch (const std::exception &e) {
        log<LOG_ERROR>(L"%1% || save_brazil_archive: serialisation error: %2%") % __func__ % e.what();
        return false;
    }
    log<LOG_INFO>(L"%1% || save_brazil_archive: wrote %2% (throws=%3%, n_cells=%4%).")
        % __func__ % path.c_str() % (int)arc.per_throw_dchi2.size() % arc.n_cells;
    return true;
}

bool load_brazil_archive(BrazilArchive &arc_out, const std::string &path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        log<LOG_ERROR>(L"%1% || load_brazil_archive: could not open %2% for reading.") % __func__ % path.c_str();
        return false;
    }
    try {
        boost::archive::binary_iarchive ia(ifs);
        uint32_t magic = 0;
        uint16_t version = 0;
        ia & magic;
        ia & version;
        if (magic != BrazilArchive::MAGIC) {
            log<LOG_ERROR>(L"%1% || load_brazil_archive: bad magic in %2% (got 0x%3$08x, expected 0x%4$08x).")
                % __func__ % path.c_str() % magic % (uint32_t)BrazilArchive::MAGIC;
            return false;
        }
        if (version != 1 && version != BrazilArchive::VERSION) {
            log<LOG_ERROR>(L"%1% || load_brazil_archive: version mismatch in %2% (got %3%, expected 1 or %4%).")
                % __func__ % path.c_str() % (int)version % (int)BrazilArchive::VERSION;
            return false;
        }
        ia & arc_out;
        arc_out.manifest = ProfileManifest{};
        arc_out.per_throw_phys_hat.clear();
        if (version >= 2) {
            ia & arc_out.manifest;
            ia & arc_out.per_throw_phys_hat;
        }
    } catch (const std::exception &e) {
        log<LOG_ERROR>(L"%1% || load_brazil_archive: deserialisation error: %2%") % __func__ % e.what();
        return false;
    }
    log<LOG_INFO>(L"%1% || load_brazil_archive: read %2% (throws=%3%, n_cells=%4%).")
        % __func__ % path.c_str() % (int)arc_out.per_throw_dchi2.size() % arc_out.n_cells;
    return true;
}

uint64_t DataFingerprint(const PROdata &data) {
    const Eigen::VectorXf spec = data.Spec();
    uint64_t h[2] = {0, 0};
    MurmurHash3_x64_128(spec.data(), (int)(spec.size() * sizeof(float)), (uint32_t)spec.size(), h);
    return h[0] ^ h[1];
}

namespace afc {

NuisanceSpec make_nuisance_spec(const PROmodel &model, const std::vector<int> &profile_idx) {
    NuisanceSpec spec;
    spec.free_mask.assign(model.nparams, 0);
    spec.pin_val = Eigen::VectorXf(model.nparams);
    for (size_t i = 0; i < model.nparams; ++i)
        spec.pin_val(i) = model.lb(i) == model.ub(i) ? model.lb(i) : model.default_val(i);
    for (int i : profile_idx) spec.free_mask[(size_t)i] = 1;
    return spec;
}

std::vector<std::vector<float>> pinned_cell_truths(const NuisanceSpec &spec,
                                                   size_t xaxis_idx, size_t yaxis_idx,
                                                   const std::vector<float> &cell_x_model,
                                                   const std::vector<float> &cell_y_model)
{
    std::vector<std::vector<float>> truths(cell_x_model.size());
    for (size_t c = 0; c < truths.size(); ++c) {
        truths[c].assign(spec.pin_val.data(), spec.pin_val.data() + spec.pin_val.size());
        if (xaxis_idx < truths[c].size()) truths[c][xaxis_idx] = cell_x_model[c];
        if (yaxis_idx < truths[c].size()) truths[c][yaxis_idx] = cell_y_model[c];
    }
    return truths;
}

// Truncated-Gaussian nuisance throws use the shared, bounded, OOB-safe
// ThrowRestrictedSplinePull from PROcess.h (pattern from commit 000b3d0).
static float throw_restricted_spline(const PROsyst &systs, size_t i,
                                     std::mt19937 &rng,
                                     std::normal_distribution<float> &d)
{
    return ThrowRestrictedSplinePull(systs, i, rng, d);
}

// --------------------------------------------------------------------
//  Per-cell PE worker — intentionally kept parallel to
//  src/PROfc.cxx::fc_worker (the brute-force FC); deduplicating the two
//  is deliberately out of scope for now.
//
//  Differences from fc_worker:
//    • Throws *one* PE per call (not args.todo) — outer loop lives in
//      the scheduler so a per-cell stop flag can interrupt mid-batch.
//    • Uses std::unique_ptr<PROmetric> instead of raw new/delete so an
//      early break doesn't leak.
//    • Toy truth per cell (truth_phys, model space): the axes at the cell
//      centre, other physics at its --fix value or default, except profiled
//      parameters (NuisanceSpec), which sit at the bank's per-cell truth and
//      float in the constrained fit.
// --------------------------------------------------------------------

namespace {

// Bundle of inputs to one adaptive PE call.
struct AdaptivePEArgs {
    const PROconfig *config;
    const PROpeller *prop;
    const PROsyst   *systs;
    const PROmodel  *model;
    const Eigen::MatrixXf *R;    ///< Fractional-covariance root (CovarianceThrowRoot), scaled per PE by its own prediction.
    const PROfitterConfig *fitconfig;
    std::string chi2_kind;       ///< "neyman" | "pearson" | "CNP" | "poisson" (legacy aliases accepted)
    bool   binned;
    bool   shape_only = false;   ///< --shapeonly (must match the data fit's metric).
    const float *truth_phys = nullptr;      ///< Toy truth of every physics param (model space); axes at the cell centre.
    const NuisanceSpec *spec = nullptr;     ///< Which physics params float in the constrained fit.
    uint32_t seed;
    FitTiming *timing = nullptr;  ///< Worker-local fit timings (log only).
};

// Run a single PE at the pinned cell coords. Returns the PEBankRecord with
// chi2_syst, chi2_osc, dchi2, and the seed used.
//
// Adapted body of fc_worker's per-PE loop (PROfc.cxx:40-131). Each call here
// is one iteration of that loop, with phy_params pinned to the cell.
static PEBankRecord run_one_pe(const AdaptivePEArgs &args)
{
    PEBankRecord rec;
    rec.seed = args.seed;
    std::mt19937 rng{args.seed};
    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());
    std::normal_distribution<float> d;

    const PROmodel &model = *args.model;
    const PROsyst  &systs = *args.systs;
    const PROconfig &config = *args.config;
    const PROpeller &prop = *args.prop;

    const size_t nphys   = model.nparams;
    const size_t nspline = systs.GetNSplines();
    const size_t nparams = nphys + nspline;

    const NuisanceSpec &spec = *args.spec;
    const bool profiled = spec.active();

    Eigen::VectorXf throws = Eigen::VectorXf::Zero((int)nparams);
    for (size_t i = 0; i < nphys; ++i) throws((int)i) = args.truth_phys[i];

    // Bounds for the constrained fit: physics pinned at the truth, except
    // profiled parameters; splines vary.
    Eigen::VectorXf lb_syst = Eigen::VectorXf::Constant((int)nparams, -3.0f);
    Eigen::VectorXf ub_syst = Eigen::VectorXf::Constant((int)nparams,  3.0f);
    for (size_t j = 0; j < nphys; ++j) {
        lb_syst((int)j) = spec.free_mask[j] ? model.lb(j) : throws((int)j);
        ub_syst((int)j) = spec.free_mask[j] ? model.ub(j) : throws((int)j);
    }
    Eigen::VectorXf lb_osc(lb_syst), ub_osc(ub_syst);
    for (size_t j = 0; j < nphys; ++j) {
        lb_osc((int)j) = model.lb(j);
        ub_osc((int)j) = model.ub(j);
    }
    for (size_t j = nphys; j < nparams; ++j) {
        const size_t si = j - nphys;
        const float lo = systs.spline_has_restrict[si] ? systs.spline_restrict_lo[si] : systs.spline_lo[si];
        const float hi = systs.spline_has_restrict[si] ? systs.spline_restrict_hi[si] : systs.spline_hi[si];
        lb_syst((int)j) = lo; ub_syst((int)j) = hi;
        lb_osc((int)j)  = lo; ub_osc((int)j)  = hi;
    }

    // Throw splines (Gaussian, respecting restrict ranges; bounded + OOB-safe).
    for (size_t i = 0; i < nspline; ++i) {
        throws((int)(i + nphys)) = throw_restricted_spline(systs, i, rng, d);
    }

    // Build fake-data spectrum. Fill the i_prime variable explicitly: the
    // previous call passed an EvalStrategy enum where FillSpectra takes
    // `bool binned` and let var_index default to 0, while the CollapseMatrix
    // below collapses with the i_prime matrix — wrong-variable physics when
    // i_prime != 0.
    PROspec shifted = FillSpectra(config, prop, systs, model, throws, args.binned, config.i_prime);
    // Covariance fluctuation scaled by this PE's own prediction (cell physics +
    // thrown splines): the covariance the metric assumes at this cell.
    const Eigen::VectorXf sys_shift = ThrowCovarianceShift(config, *args.R, shifted.Spec(), rng, d, config.i_prime);
    PROspec newSpec = PROspec::PoissonVariation(
        PROspec(CollapseMatrix(config, shifted.Spec()) + sys_shift,
                CollapseMatrix(config, shifted.Error())),
        dseed(rng));
    PROdata data(newSpec.Spec(), newSpec.Error());
    data.physics_prior_centers = ToyPhysicsPriorCenters(model, throws.head((Eigen::Index)nphys), true, rng);

    // Build metric (unique_ptr — early-stop safe).
    PROmetric::EvalStrategy mstrat = args.binned ? PROmetric::BinnedChi2 : PROmetric::EventByEvent;
    std::unique_ptr<PROmetric> metric = MakeMetric(args.chi2_kind, config, prop, &systs, model, data, mstrat, args.shape_only);
    if (!metric) {
        log<LOG_ERROR>(L"%1% || run_one_pe: unknown chi2 kind '%2%'.") % __func__ % args.chi2_kind.c_str();
        return rec;
    }

    FitTiming local_timing;
    FitTiming &timing = args.timing ? *args.timing : local_timing;

    // Constrained fit at the cell. Profiled parameters float: seed them where a
    // data fit would look (truth physics, splines at their prior centres), in
    // every basin of their range (ProfileLadderSeeds), and always at their
    // default, whose log10 floor a Latin hypercube never reaches.
    PROfitter fitter_syst(ub_syst, lb_syst, *args.fitconfig, dseed(rng));
    metric->setBounds(lb_syst, ub_syst);
    auto t0 = std::chrono::steady_clock::now();
    if (profiled) {
        Eigen::VectorXf truth_seed = throws;
        for (size_t i = 0; i < nspline; ++i)
            truth_seed((int)(nphys + i)) = i < (size_t)systs.spline_centers.size() ? systs.spline_centers(i) : 0.0f;
        FixedSeed at_default{truth_seed, std::vector<int>(nparams, 0)};
        for (size_t j = 0; j < nphys; ++j) {
            if (spec.free_mask[j]) at_default.point((int)j) = model.default_val(j);
            at_default.fixed[j] = 1;
        }
        std::vector<Eigen::VectorXf> seeds{truth_seed};
        for (auto &x : ProfileLadderSeeds(*metric, truth_seed, lb_syst, ub_syst, spec.free_mask)) seeds.push_back(std::move(x));
        rec.chi2_syst = fitter_syst.Fit(*metric, seeds, std::vector<FixedSeed>{at_default});
        for (size_t j = 0; j < nphys; ++j)
            if (spec.free_mask[j]) rec.phys_hat.push_back(fitter_syst.best_fit((int)j));
    } else {
        rec.chi2_syst = fitter_syst.Fit(*metric);
    }
    timing.constrained_us += FitTiming::since_us(t0);
    metric->freeParams();

    // Full (syst + osc) fit.
    PROfitter fitter_osc(ub_osc, lb_osc, *args.fitconfig, dseed(rng));
    metric->setBounds(lb_osc, ub_osc);
    std::vector<Eigen::VectorXf> seed_pts = {fitter_syst.best_fit};
    t0 = std::chrono::steady_clock::now();
    rec.chi2_osc = fitter_osc.Fit(*metric, seed_pts);
    timing.free_us += FitTiming::since_us(t0);
    t0 = std::chrono::steady_clock::now();
    fitter_osc.calcFreqSeedPoints(*metric);
    timing.harmonic_us += FitTiming::since_us(t0);
    for (size_t i = 0; i < fitter_osc.freq_seed_points.size(); ++i) {
        const float c = fitter_osc.freq_seed_values.at(i);
        if (c < rec.chi2_osc) rec.chi2_osc = c;
    }

    // Negative-Δχ² rescue, same as fc_worker: the osc fit missed a minimum
    // the syst-only fit (a subset of its space) found, so retry it wider.
    if (rec.chi2_syst < rec.chi2_osc) {
        log<LOG_DEBUG>(L"%1% || Negative delta chi2 detected! chi2_syst=%2% < chi2_osc=%3%") % __func__ % rec.chi2_syst % rec.chi2_osc;
        log<LOG_DEBUG>(L"%1% || Attempting enhanced global search") % __func__;

        PROfitterConfig enhanced_config = *args.fitconfig;
        enhanced_config.n_latin_points *= 2;
        enhanced_config.n_swarm_particles = std::max(10, enhanced_config.n_swarm_particles * 2);
        enhanced_config.n_localfit = std::max(5, enhanced_config.n_localfit + 2);

        PROfitter fitter_osc_enhanced(ub_osc, lb_osc, enhanced_config, dseed(rng));
        t0 = std::chrono::steady_clock::now();
        const float chi2_osc_enhanced = fitter_osc_enhanced.Fit(*metric, fitter_syst.best_fit);
        timing.rescue_us += FitTiming::since_us(t0);
        timing.n_rescue++;

        if (chi2_osc_enhanced < rec.chi2_osc) {
            timing.n_rescue_improved++;
            log<LOG_DEBUG>(L"%1% || Enhanced search improved chi2_osc from %2% to %3%")
                % __func__ % rec.chi2_osc % chi2_osc_enhanced;
            rec.chi2_osc = chi2_osc_enhanced;
        }
    }

    rec.dchi2 = rec.chi2_syst - rec.chi2_osc;
    timing.n++;
    return rec;
}

} // anonymous

// --------------------------------------------------------------------
//  schedule_pes — owns a threadpool and hands out cell-jobs to workers.
//  The PE budget policy is the additive doubling rule below (the Wilson
//  sequential stopping rule this comment used to mention was never wired
//  in and has been removed as dead code).
// --------------------------------------------------------------------

// Drive PE generation for every MetaCell using a deterministic doubling rule.
//
//   to_add_for_cell = n_pe_min * 2^max(0, level - update_layer)
//
// This many PEs are *added* to the cell on each run — independent of what's
// already there. So running init-bank twice with --n-pe-min 50 on a fresh
// L=0 cell gives 50, then 100 PEs. On an L=2 cell: 200, then 400. Pure
// additive top-up; the bank grows monotonically.
//
// Cells with level < update_layer are left UNTOUCHED. This lets the user grow
// only the deeper layers: `--update-layer 2 --n-pe-min 100` adds 100 to L=2,
// 200 to L=3, etc., while L=0,1 keep whatever PEs they already have.
//
// n_pe_max is a *total-per-cell* safety cap — even with repeated runs, no
// cell ever exceeds this PE count. If existing >= n_pe_max, the cell is
// skipped on this run.
void schedule_pes(const AdaptiveFCConfig &acfg,
                         const PROconfig &config,
                         const PROpeller &prop,
                         const PROsyst   &systs,
                         const PROmodel  &model,
                         const PROfitterConfig &fitconfig,
                         PROseed &proseed,
                         const Eigen::MatrixXf &frac_root,
                         const NuisanceSpec &spec,
                         const std::vector<std::vector<float>> &cell_truth,
                         PEBank &bank_out,
                         int nthreads,
                         MultiPROgressBar &progress,
                         int progress_bar_idx,
                         AdaptiveFCResult &result_out)
{
    const int n_cells = (int)cell_truth.size();
    if (n_cells == 0) return;

    // Preserve any pre-existing PEs (top-up support).
    if ((int)bank_out.cell_pes.size() != n_cells) {
        bank_out.cell_pes.assign(n_cells, {});
    }

    // Per-cell number of PEs to ADD on this run. When acfg.only_layer >= 0,
    // only cells at exactly that level get topped up by n_pe_min (no doubling,
    // all other layers untouched). Otherwise the standard doubling rule:
    // level < update_layer skipped; level >= update_layer gets n_pe_min × 2^(level-update_layer).
    auto compute_to_add = [&](int level) -> int {
        if (acfg.only_layer >= 0) {
            return (level == acfg.only_layer) ? acfg.n_pe_min : -1;
        }
        if (level < acfg.update_layer) return -1; // skip cell entirely
        const int delta = std::max(0, level - acfg.update_layer);
        if (delta > 20) return acfg.n_pe_max; // safety against integer overflow
        return (int)((int64_t)acfg.n_pe_min << delta);
    };

    if (acfg.only_layer >= 0) {
        log<LOG_INFO>(L"%1% || schedule_pes: --update-only-layer=%2%, n_pe_min=%3% added per matching cell, "
                      L"n_pe_max=%4% (total cap).")
            % __func__ % acfg.only_layer % acfg.n_pe_min % acfg.n_pe_max;
    } else {
        log<LOG_INFO>(L"%1% || schedule_pes: additive doubling, n_pe_min=%2% (added per run), "
                      L"n_pe_max=%3% (total cap), update_layer=%4%.")
            % __func__ % acfg.n_pe_min % acfg.n_pe_max % acfg.update_layer;
    }

    // Deterministic per-PE seeding: seeds are derived from (base, cell, PE
    // index) via MurmurHash3, NOT drawn from a per-thread RNG stream. With the
    // work-stealing scheduler below, which thread claims which cell depends on
    // OS scheduling — per-thread streams made the bank irreproducible even
    // with a fixed --global-seed.
    const uint32_t pe_seed_base = (*proseed.getThreadSeeds())[0];
    auto pe_seed_for = [pe_seed_base](int cell, int pe_index) -> uint32_t {
        const uint32_t key[2] = {(uint32_t)cell, (uint32_t)pe_index};
        uint32_t out = 0;
        MurmurHash3_x86_32(key, sizeof(key), pe_seed_base, &out);
        return out;
    };

    std::atomic<int> next_cell{0};
    std::atomic<int64_t> total_pes{0};
    std::atomic<int> cells_skipped_layer{0};  ///< level < update_layer
    std::atomic<int> cells_at_cap{0};         ///< existing already >= n_pe_max
    std::atomic<int> cells_topped_up{0};      ///< added PEs this run
    FitTiming total_timing;
    std::mutex timing_mu;

    auto worker = [&](int) {
        FitTiming timing;
        while (true) {
            int c = next_cell.fetch_add(1);
            if (c >= n_cells) break;

            std::vector<PEBankRecord> local_pes = bank_out.cell_pes[(size_t)c];
            const int n_existing = (int)local_pes.size();
            const int level = (c < (int)bank_out.cell_level.size()) ? bank_out.cell_level[(size_t)c] : 0;
            const int to_add_raw = compute_to_add(level);

            if (to_add_raw < 0) {
                // Below update_layer — preserve existing, do nothing.
                // Bar is sized for *PEs added*, so skipped cells don't tick.
                cells_skipped_layer.fetch_add(1);
                total_pes.fetch_add((int64_t)n_existing);
                continue;
            }
            if (n_existing >= acfg.n_pe_max) {
                // Hard total cap reached — skip.
                cells_at_cap.fetch_add(1);
                total_pes.fetch_add((int64_t)n_existing);
                continue;
            }

            // Clamp to_add so we don't exceed total cap.
            const int final_total = std::min(n_existing + to_add_raw, acfg.n_pe_max);

            cells_topped_up.fetch_add(1);
            for (int i = n_existing; i < final_total; ++i) {
                AdaptivePEArgs args{};
                args.config = &config;
                args.prop   = &prop;
                args.systs  = &systs;
                args.model  = &model;
                args.R      = &frac_root;
                args.fitconfig = &fitconfig;
                args.chi2_kind = acfg.chi2;
                args.binned    = acfg.binned;
                args.shape_only = acfg.shape_only;
                args.truth_phys = cell_truth[(size_t)c].data();
                args.spec = &spec;
                args.seed = pe_seed_for(c, i);
                args.timing = &timing;

                PEBankRecord rec = run_one_pe(args);
                local_pes.push_back(rec);
                progress.increment_bar(progress_bar_idx);  // tick per PE produced
            }

            total_pes.fetch_add((int64_t)local_pes.size());
            bank_out.cell_pes[(size_t)c] = std::move(local_pes);
        }
        std::lock_guard<std::mutex> lk(timing_mu);
        total_timing += timing;
    };

    std::vector<std::thread> tpool;
    tpool.reserve((size_t)std::max(1, nthreads));
    for (int t = 0; t < std::max(1, nthreads); ++t) tpool.emplace_back(worker, t);
    for (auto &th : tpool) th.join();

    result_out.total_pes_generated = total_pes.load();
    result_out.cells_hit_n_pe_max  = cells_at_cap.load();
    result_out.cells_topped_up     = cells_topped_up.load();
    result_out.mean_pes_per_cell   = n_cells > 0 ? (float)total_pes.load() / (float)n_cells : 0.0f;

    total_timing.log_summary("init-bank");
    log<LOG_INFO>(L"%1% || schedule_pes done: cells=%2%, total_pes=%3%, mean=%4%, "
                  L"topped_up=%5%, at_cap=%6%, skipped_below_update_layer=%7%.")
        % __func__ % n_cells % (int64_t)total_pes.load()
        % result_out.mean_pes_per_cell
        % (int)cells_topped_up.load()
        % (int)cells_at_cap.load()
        % (int)cells_skipped_layer.load();
}

// --------------------------------------------------------------------
//  Asimov-mode helpers.
// --------------------------------------------------------------------

AsimovObs compute_asimov_obs(
    const PROconfig &config,
    const PROpeller &prop,
    const PROsyst   &systs,
    const PROmodel  &model,
    const PROfitterConfig &fitconfig,
    const PROdata   &asimov_data,
    const std::string &chi2_kind,
    bool binned,
    bool shape_only,
    size_t xaxis_idx, size_t yaxis_idx,
    const std::vector<float> &cell_x_model,
    const std::vector<float> &cell_y_model,
    PROseed &proseed,
    int nthreads,
    MultiPROgressBar &progress,
    int bar_idx,
    const NuisanceSpec &spec,
    const std::vector<uint64_t> *cell_keys,
    const std::vector<std::vector<float>> *cell_seed_phys,
    bool snap_to_default)
{
    AsimovObs obs;
    const int n_cells = (int)cell_x_model.size();
    obs.chi2_syst.assign(n_cells, 0.0f);
    obs.dchi2_obs.assign(n_cells, 0.0f);
    if (n_cells == 0) return obs;
    const bool profiled = spec.active();
    if (profiled) obs.cell_phys_hat.assign(n_cells, {});

    const size_t nphys   = model.nparams;
    const size_t nspline = systs.GetNSplines();
    const size_t nparams = nphys + nspline;

    auto make_metric = [&](const PROdata &d) -> std::unique_ptr<PROmetric> {
        PROmetric::EvalStrategy mstrat = binned ? PROmetric::BinnedChi2 : PROmetric::EventByEvent;
        auto m = MakeMetric(chi2_kind, config, prop, &systs, model, d, mstrat, shape_only);
        if (!m) log<LOG_ERROR>(L"%1% || compute_asimov_obs: unknown chi2 kind '%2%'.") % __func__ % chi2_kind.c_str();
        return m;
    };

    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());

    // ---- 1. Global (syst + physics) fit, once. -------------------------------
    {
        auto metric = make_metric(asimov_data);
        if (!metric) return obs;
        Eigen::VectorXf lb_osc((int)nparams), ub_osc((int)nparams);
        for (size_t j = 0; j < nphys; ++j) {
            lb_osc((int)j) = model.lb(j);
            ub_osc((int)j) = model.ub(j);
        }
        for (size_t j = nphys; j < nparams; ++j) {
            const size_t si = j - nphys;
            const float lo = systs.spline_has_restrict[si] ? systs.spline_restrict_lo[si] : systs.spline_lo[si];
            const float hi = systs.spline_has_restrict[si] ? systs.spline_restrict_hi[si] : systs.spline_hi[si];
            lb_osc((int)j) = lo; ub_osc((int)j) = hi;
        }
        metric->setBounds(lb_osc, ub_osc);
        std::mt19937 main_rng((*proseed.getThreadSeeds())[0]);
        PROfitter fitter_osc(ub_osc, lb_osc, fitconfig, dseed(main_rng));
        obs.chi2_osc_global = fitter_osc.Fit(*metric);
        // Try the freq-seed alternatives — same trick as fc_worker.
        fitter_osc.calcFreqSeedPoints(*metric);
        for (size_t i = 0; i < fitter_osc.freq_seed_points.size(); ++i) {
            const float c = fitter_osc.freq_seed_values.at(i);
            if (c < obs.chi2_osc_global) obs.chi2_osc_global = c;
        }
        log<LOG_INFO>(L"%1% || asimov global fit: chi2_osc=%2%.") % __func__ % obs.chi2_osc_global;
    }

    // ---- 2. Per-cell constrained fits, parallel. ------------------------------
    std::atomic<int> next_cell{0};
    // Profile mode: a cell's fitter seed depends only on its footprint, not on
    // --seed, so grid jobs with distinct seeds fit identical toy truths and
    // their banks merge.
    const uint32_t cell_seed_base = 0x5EEDu;

    FitTiming total_timing;
    std::mutex timing_mu;

    auto worker = [&](int thread_idx) {
        std::mt19937 thread_rng((*proseed.getThreadSeeds())[thread_idx]);
        auto metric = make_metric(asimov_data);
        if (!metric) return;
        FitTiming timing;

        // Pre-build the spline bounds once (same for every cell).
        Eigen::VectorXf lb((int)nparams), ub((int)nparams);
        for (size_t j = 0; j < nphys; ++j) {
            lb((int)j) = spec.free_mask[j] ? model.lb(j) : spec.pin_val(j);
            ub((int)j) = spec.free_mask[j] ? model.ub(j) : spec.pin_val(j);
        }
        for (size_t j = nphys; j < nparams; ++j) {
            const size_t si = j - nphys;
            const float lo = systs.spline_has_restrict[si] ? systs.spline_restrict_lo[si] : systs.spline_lo[si];
            const float hi = systs.spline_has_restrict[si] ? systs.spline_restrict_hi[si] : systs.spline_hi[si];
            lb((int)j) = lo; ub((int)j) = hi;
        }

        while (true) {
            int c = next_cell.fetch_add(1);
            if (c >= n_cells) break;

            // Pin the two scanned axes at this cell's center.
            lb((int)xaxis_idx) = cell_x_model[(size_t)c];
            ub((int)xaxis_idx) = cell_x_model[(size_t)c];
            lb((int)yaxis_idx) = cell_y_model[(size_t)c];
            ub((int)yaxis_idx) = cell_y_model[(size_t)c];

            metric->setBounds(lb, ub);
            const auto t0 = std::chrono::steady_clock::now();
            if (!profiled) {
                PROfitter fitter(ub, lb, fitconfig, dseed(thread_rng));
                const float chi2_s = fitter.Fit(*metric);
                obs.chi2_syst[(size_t)c] = chi2_s;
                obs.dchi2_obs[(size_t)c] = chi2_s - obs.chi2_osc_global;
            } else {
                const uint64_t key = cell_keys ? (*cell_keys)[(size_t)c] : (uint64_t)c;
                uint32_t fit_seed = 0;
                MurmurHash3_x86_32(&key, sizeof(key), cell_seed_base, &fit_seed);

                // Seed at the given per-cell truth (else the pins), splines at their prior centres.
                Eigen::VectorXf seed = Eigen::VectorXf::Zero((int)nparams);
                const bool have_seed = cell_seed_phys && (*cell_seed_phys)[(size_t)c].size() == nphys;
                for (size_t j = 0; j < nphys; ++j)
                    seed((int)j) = have_seed ? (*cell_seed_phys)[(size_t)c][j] : spec.pin_val(j);
                seed((int)xaxis_idx) = cell_x_model[(size_t)c];
                seed((int)yaxis_idx) = cell_y_model[(size_t)c];
                for (size_t i = 0; i < nspline && i < (size_t)systs.spline_centers.size(); ++i)
                    seed((int)(nphys + i)) = systs.spline_centers(i);
                FixedSeed at_default{seed, std::vector<int>(nparams, 0)};
                for (size_t j = 0; j < nphys; ++j) {
                    if (spec.free_mask[j]) at_default.point((int)j) = model.default_val(j);
                    at_default.fixed[j] = 1;
                }

                std::vector<Eigen::VectorXf> seeds{seed};
                for (auto &x : ProfileLadderSeeds(*metric, seed, lb, ub, spec.free_mask)) seeds.push_back(std::move(x));
                PROfitter fitter(ub, lb, fitconfig, fit_seed);
                float chi2_c = fitter.Fit(*metric, seeds, std::vector<FixedSeed>{at_default});
                Eigen::VectorXf best = fitter.best_fit;
                if (snap_to_default) {
                    // A profiled parameter the data cannot see would sit wherever the
                    // minimiser drifted along a flat direction: report its default.
                    Eigen::VectorXf lb_d = lb, ub_d = ub;
                    for (size_t j = 0; j < nphys; ++j)
                        if (spec.free_mask[j]) lb_d((int)j) = ub_d((int)j) = model.default_val(j);
                    metric->setBounds(lb_d, ub_d);
                    PROfitter fitter_d(ub_d, lb_d, fitconfig, fit_seed ^ 0x9E3779B9u);
                    const float chi2_d = fitter_d.Fit(*metric, std::vector<Eigen::VectorXf>{at_default.point});
                    if (chi2_d - chi2_c < 1e-2f) {
                        best = fitter_d.best_fit;
                        chi2_c = std::min(chi2_c, chi2_d);
                    }
                }
                obs.chi2_syst[(size_t)c] = chi2_c;
                obs.cell_phys_hat[(size_t)c].assign(best.data(), best.data() + nphys);
            }
            timing.constrained_us += FitTiming::since_us(t0);
            timing.n++;
            progress.increment_bar(bar_idx);
        }
        std::lock_guard<std::mutex> lk(timing_mu);
        total_timing += timing;
    };

    std::vector<std::thread> tpool;
    tpool.reserve((size_t)std::max(1, nthreads));
    for (int t = 0; t < std::max(1, nthreads); ++t) tpool.emplace_back(worker, t);
    for (auto &th : tpool) th.join();
    total_timing.log_summary("asimov per-cell");

    if (profiled) {
        // A conditional fit can beat a global fit that missed its basin; the
        // global minimum is then the best cell, so no observed dchi2 is negative.
        const float best_cell = *std::min_element(obs.chi2_syst.begin(), obs.chi2_syst.end());
        if (best_cell < obs.chi2_osc_global) {
            log<LOG_WARNING>(L"%1% || a cell's constrained fit (chi2 %2%) beat the global fit (%3%); using it as the global minimum.")
                % __func__ % best_cell % obs.chi2_osc_global;
            obs.chi2_osc_global = best_cell;
        }
        for (int c = 0; c < n_cells; ++c)
            obs.dchi2_obs[(size_t)c] = obs.chi2_syst[(size_t)c] - obs.chi2_osc_global;
    }

    return obs;
}

// Precompute per-(CL, cell) critical Δχ² thresholds from the bank alone.
// crit_dchi2 and decidable depend only on the bank's PE lists and the CL
// targets — not on any throw's observables — so callers classifying MANY
// throws against one bank (brazil mode) compute this once (one sort per
// cell) and reduce each throw to a comparison. `included` is left false.
std::vector<std::vector<CellVerdict>> compute_bank_crits(
    const PEBank &bank,
    const std::vector<float> &cl_targets,
    int min_pes_for_decision)
{
    const int n_cells = bank.n_cells;
    std::vector<std::vector<CellVerdict>> crits(cl_targets.size());
    for (auto &v : crits) v.assign(n_cells, CellVerdict{});

    for (int c = 0; c < n_cells; ++c) {
        const auto &pes = bank.cell_pes[(size_t)c];
        if ((int)pes.size() < min_pes_for_decision) continue; // leave decidable = false

        std::vector<float> sorted;
        sorted.reserve(pes.size());
        for (const auto &r : pes) sorted.push_back(r.dchi2);
        std::sort(sorted.begin(), sorted.end());

        for (size_t k = 0; k < cl_targets.size(); ++k) {
            CellVerdict v;
            v.crit_dchi2 = SequentialFCTest::crit_dchi2_at_cl(sorted, cl_targets[k]);
            v.decidable  = true;
            crits[k][(size_t)c] = v;
        }
    }
    return crits;
}

// Classify every cell at every requested CL, given asimov observations and a
// bank. Returns verdicts indexed by [cl_idx][cell_idx]. A cell with fewer than
// `min_pes_for_decision` PEs in the bank is marked `decidable = false`.
std::vector<std::vector<CellVerdict>> classify_against_bank(
    const PEBank &bank,
    const AsimovObs &obs,
    const std::vector<float> &cl_targets,
    int min_pes_for_decision)
{
    auto verdicts = compute_bank_crits(bank, cl_targets, min_pes_for_decision);
    for (size_t k = 0; k < verdicts.size(); ++k) {
        for (size_t c = 0; c < verdicts[k].size(); ++c) {
            CellVerdict &v = verdicts[k][c];
            v.included = v.decidable && (obs.dchi2_obs[c] <= v.crit_dchi2);
        }
    }
    return verdicts;
}

} // namespace afc
} // namespace PROfit
