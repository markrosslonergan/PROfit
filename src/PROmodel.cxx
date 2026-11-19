/**
 * @file PROmodel.cxx
 * @brief Implementation of the PROmodel base class and the model factory.
 * @author PROfit Collaboration
 */
#include "PROmodel.h"
#include "PROmodels/PROmodelSimple.h"
#include "PROmodels/PROmodel2flav.h"
#include "PROmodels/PROmodel3p1.h"
#include "PROmodels/PROmodel3p1decayvis.h"
#include "PROmodels/PROmodel3p2.h"
#include "PROmodels/PROmodelLBL.h"
#include "PROmodels/PROmodelLBL3p1.h"
#include "PROmodels/PROmodelSine.h"
#include <stdexcept>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <map>

namespace PROfit {

std::vector<long int> PROmodel::build_phys_grid(const PROpeller &prop) {
    const size_t N = ivars.size();
    const size_t nev = prop.NEvent();
    const bool layout = !prop.block_layout.empty();
    const bool cells = truth_overflow_cells && layout;

    // Each event's block and every block's |weight|, for the cells and for the log: a tail can
    // be a negligible share of the whole MC yet a large share of one subchannel.
    const size_t nblocks = layout ? prop.block_names.size() : 0;
    std::vector<int> block(N > 0 && layout ? nev : 0, -1);
    std::vector<double> w_block(nblocks, 0.0);
    double w_all = 0;
    for(size_t i = 0; i < nev; ++i) {
        const double w = std::fabs(prop.added_weights[i]);
        w_all += w;
        if(block.empty()) continue;
        block[i] = prop.EventBlock(i);
        if(block[i] >= 0) w_block[block[i]] += w;
    }

    // A cell: (block, 0 = underflow / 1 = overflow, 1 = negative value). The sign keeps signed
    // inputs (antineutrino E or L/E < 0) from sharing a mean with positive ones.
    using CellKey = std::array<int, 3>;
    auto cell_key = [&](size_t k, size_t i, CellKey &key) {
        const float x = prop.VariableValue(ivars[k], i);
        if(!std::isfinite(x) || block[i] < 0) return false;
        const PROpeller::BlockLayout &b = prop.block_layout[ivars[k]];
        if(!(b.lo[block[i]] <= b.hi[block[i]])) return false;  // no 1D binning in this block (NaN)
        key = {block[i], x < b.lo[block[i]] ? 0 : 1, x < 0 ? 1 : 0};
        return true;
    };

    // Largest share of one subchannel's weight, in percent, and its name (whole MC without layout).
    auto worst = [&](const std::vector<double> &w_tail, double w_tail_all) {
        std::pair<double, std::string> res{w_all > 0 ? 100.0 * w_tail_all / w_all : 0.0, "the whole MC"};
        if(!layout) return res;
        res.first = 0;
        for(size_t s = 0; s < nblocks; ++s)
            if(w_block[s] > 0 && 100.0 * w_tail[s] / w_block[s] > res.first)
                res = {100.0 * w_tail[s] / w_block[s], prop.block_names[s]};
        return res;
    };
    const char *drop_reason = !truth_overflow_cells ? "this model needs them inside its truth binning"
                            : !layout ? "the PROpeller has no block layout" : "non-finite value or no binning in the block";

    phys_axes.assign(N, {});
    phys_axes_hw.assign(N, {});
    std::vector<std::map<CellKey, int>> cell_index(N);
    for(size_t k = 0; k < N; ++k) {
        const Eigen::VectorXf &mid = prop.variable_midbin[ivars[k]];
        phys_axes[k].assign(mid.data(), mid.data() + mid.size());
        phys_axes_hw[k].assign(mid.size(), 0.0f);

        struct Moments { double w = 0, wx = 0, wxx = 0, x = 0, xx = 0; long n = 0; };
        std::map<CellKey, Moments> moments;
        long n_out = 0, n_dropped = 0;
        double w_out = 0, w_dropped = 0;
        std::vector<double> w_out_block(nblocks, 0.0), w_dropped_block(nblocks, 0.0);
        for(size_t i = 0; i < nev; ++i) {
            if(prop.VariableBinIndex(ivars[k], i) >= 0) continue;
            const double w = std::fabs(prop.added_weights[i]);
            CellKey key;
            if(!cells || !cell_key(k, i, key)) {
                ++n_dropped; w_dropped += w;
                if(!block.empty() && block[i] >= 0) w_dropped_block[block[i]] += w;
                continue;
            }
            const double x = prop.VariableValue(ivars[k], i);
            Moments &m = moments[key];
            m.w += w; m.wx += w * x; m.wxx += w * x * x;
            m.x += x; m.xx += x * x; ++m.n;
            ++n_out; w_out += w; w_out_block[block[i]] += w;
        }
        for(const auto &[key, m] : moments) {
            // Unweighted moments if every weight in the cell is zero.
            const double sw = m.w > 0 ? m.w : (double)m.n;
            const double mean = (m.w > 0 ? m.wx : m.x) / sw;
            const double var = std::max(0.0, (m.w > 0 ? m.wxx : m.xx) / sw - mean * mean);
            cell_index[k][key] = (int)phys_axes[k].size();
            phys_axes[k].push_back((float)mean);
            phys_axes_hw[k].push_back((float)std::sqrt(3.0 * var));
        }

        const auto [pct_out, where_out] = worst(w_out_block, w_out);
        const auto [pct_dropped, where_dropped] = worst(w_dropped_block, w_dropped);
        if(n_out > 0)
            log<LOG_INFO>(L"%1% || Kinematic variable %2%: %3% events lie outside its binning (up to %4%%% of the MC weight of %5%); placed in %6% underflow/overflow cells of the physics grid.")
                % __func__ % ivars[k] % n_out % pct_out % where_out.c_str() % moments.size();
        if(n_dropped > 0)
            log<LOG_INFO>(L"%1% || Kinematic variable %2%: %3% events outside its binning (up to %4%%% of the MC weight of %5%) are left out of the binned prediction: %6%.")
                % __func__ % ivars[k] % n_dropped % pct_dropped % where_dropped.c_str() % drop_reason;
        // Model construction repeats per FC/AFC worker: warn once per process.
        static std::atomic<bool> warned_out{false}, warned_dropped{false};
        if(pct_out > 1.0 && !warned_out.exchange(true))
            log<LOG_WARNING>(L"%1% || %2%%% of the MC weight of %3% lies outside the binning of kinematic variable %4%. The binned prediction oscillates those events at their cell's mean value (averaged over the cell's spread where the model bin-averages); widen that variable's <bins> to treat them exactly.")
                % __func__ % pct_out % where_out.c_str() % ivars[k];
        if(pct_dropped > 0.1 && !warned_dropped.exchange(true))
            log<LOG_WARNING>(L"%1% || %2%%% of the MC weight of %3% is outside the binning of kinematic variable %4% and missing from every binned prediction (%5%). Widen that variable's <bins> to cover every event.")
                % __func__ % pct_dropped % where_dropped.c_str() % ivars[k] % drop_reason;
    }

    phys_grid_sizes.assign(N, 0);
    n_phys_bins = 1;
    for(size_t k = 0; k < N; ++k) {
        phys_grid_sizes[k] = phys_axes[k].size();
        n_phys_bins *= (long int)phys_grid_sizes[k];
    }

    // Row-major over the axes, last ivar fastest.
    phys_grid.assign(N, std::vector<float>(n_phys_bins));
    for(long int flat = 0; flat < n_phys_bins; ++flat) {
        long int rem = flat;
        for(int k = (int)N - 1; k >= 0; --k) {
            phys_grid[k][flat] = phys_axes[k][rem % (long int)phys_grid_sizes[k]];
            rem /= (long int)phys_grid_sizes[k];
        }
    }

    std::vector<long int> flat(nev, 0);
    for(size_t i = 0; i < nev; ++i) {
        long int f = 0;
        for(size_t k = 0; k < N; ++k) {
            int idx = prop.VariableBinIndex(ivars[k], i);
            if(idx < 0) {
                CellKey key;
                auto it = cells && cell_key(k, i, key) ? cell_index[k].find(key) : cell_index[k].end();
                if(it == cell_index[k].end()) { f = -1; break; }
                idx = it->second;
            }
            f = f * (long int)phys_grid_sizes[k] + idx;
        }
        flat[i] = f;
    }
    return flat;
}

void PROmodel::build_hists_and_combined(const PROpeller &prop, bool filter_by_model_rule,
                                        const std::function<int(size_t, size_t, int)> &block_fn) {
    const std::vector<long int> event_flat = build_phys_grid(prop);

    size_t nvar = prop.variable_mc_stat_err.size();
    // Decay models compute counts directly in get_counts and leave model_functions empty;
    // for those, the number of components comes from prob_types instead.
    size_t J    = model_functions.empty() ? prob_types.size() : model_functions.size();
    H_combined.assign(nvar, HMatrix());

    // This pre-bins the WHOLE MC event store into the sparse binned-eval
    // response matrices for every variable, and is the silent chunk of model
    // construction — announce it up front so a long stall here is labeled.
    size_t nbinned = 0;
    for(size_t v = 0; v < nvar; ++v)
        if(prop.IsBinnedVariable(v)) ++nbinned;
    log<LOG_INFO>(L"%1% || Pre-binning %2% MC events into sparse response matrices: %3% of %4% variables x %5% components x %6% physics-grid bins. This runs once per model construction.")
        % __func__ % prop.NEvent() % nbinned % nvar % J % n_phys_bins;

    if((long long)n_phys_bins * (long long)J > (long long)std::numeric_limits<HMatrix::StorageIndex>::max()) {
        log<LOG_ERROR>(L"%1% || Physics grid too large for sparse column index: %2% bins x %3% components.") % __func__ % n_phys_bins % J;
        throw std::runtime_error("H_combined column count exceeds sparse StorageIndex range");
    }

    std::vector<Eigen::Triplet<float>> trips;
    trips.reserve(prop.NEvent());
    for(size_t v = 0; v < nvar; ++v) {
        if(!prop.IsBinnedVariable(v)) continue;
        size_t n_reco_v = prop.variable_mc_stat_err[v].size();
        // Collect (row, col, weight) per event into component m's column block
        // [m*n_phys_bins, (m+1)*n_phys_bins) of H_combined[v]. Element-wise
        // insertion into a compressed sparse matrix is O(nnz) per insert, so
        // build from triplets once; setFromTriplets sums duplicate (row, col)
        // entries, reproducing the dense `+=` accumulation.
        trips.clear();
        for(size_t i = 0; i < prop.NEvent(); ++i) {
            int rbin = prop.VariableBinIndex(v, i);
            if(rbin < 0 || event_flat[i] < 0) continue;

            // Determine the destination column for this event.
            int col;
            if(block_fn) {
                col = block_fn(v, i, rbin);
            } else {
                col = filter_by_model_rule ? prop.model_rule[i] : 0;
            }
            if(col < 0 || col >= (int)J) continue;

            trips.emplace_back(rbin, (long int)col * n_phys_bins + event_flat[i], prop.added_weights[i]);
        }
        H_combined[v] = HMatrix(n_reco_v, n_phys_bins * J);
        H_combined[v].setFromTriplets(trips.begin(), trips.end());
        H_combined[v].makeCompressed();

        const double total = (double)n_reco_v * (double)(n_phys_bins * J);
        log<LOG_INFO>(L"%1% || Variable %2%: response matrix %3% x %4%, %5% non-zeros (%6%%% filled, ~%7% MB).")
            % __func__ % v % n_reco_v % (n_phys_bins * J) % H_combined[v].nonZeros()
            % (total > 0 ? 100.0 * H_combined[v].nonZeros() / total : 0.0)
            % (H_combined[v].nonZeros() * (sizeof(float) + sizeof(HMatrix::StorageIndex)) / (1024.0 * 1024.0));
    }
}

Eigen::MatrixXf PROmodel::compute_N_truth(const PROpeller &prop, bool filter_by_model_rule) const {
    size_t J = model_functions.empty() ? prob_types.size() : model_functions.size();
    std::vector<size_t> ivar_sizes(ivars.size());
    for(size_t k = 0; k < ivars.size(); ++k) {
        ivar_sizes[k] = prop.variable_midbin[ivars[k]].size();
        // Truth-bin indexing only: the grid must carry no underflow/overflow cells.
        if(ivar_sizes[k] != phys_grid_sizes.at(k))
            throw std::logic_error("compute_N_truth on a physics grid with underflow/overflow cells; set truth_overflow_cells = false");
    }

    Eigen::MatrixXf N = Eigen::MatrixXf::Zero(n_phys_bins, J);
    for(size_t i = 0; i < prop.NEvent(); ++i) {
        int j = filter_by_model_rule ? prop.model_rule[i] : 0;
        if(j < 0 || (size_t)j >= J) continue;
        long int flat_phys = 0;
        bool valid = true;
        for(size_t k = 0; k < ivars.size(); ++k) {
            int tbin = prop.VariableBinIndex(ivars[k], i);
            if(tbin < 0) { valid = false; break; }
            flat_phys = flat_phys * (long int)ivar_sizes[k] + tbin;
        }
        if(!valid) continue;
        N(flat_phys, j) += prop.added_weights[i];
    }
    return N;
}

Eigen::MatrixXf PROmodel::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    const auto &le_arr = var_arrs[0];
    Eigen::MatrixXf probs(le_arr.size(), prob_types.size());
    for(size_t i = 0; i < le_arr.size(); ++i) {
        for(size_t j = 0; j < prob_types.size(); ++j) {
            probs(i, j) = model_functions[j](phys, le_arr[i]);
        }
    }
    return probs;
}

std::vector<Eigen::MatrixXf> PROmodel::get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    // Central FD on the probabilities only. Probabilities are O(1), so this is
    // far better conditioned than FD on the full chi²; models with cheap
    // closed-form derivatives should still override for exactness and speed.
    std::vector<Eigen::MatrixXf> grads(nparams);
    const float h = 1e-3f;
    Eigen::VectorXf p = phys;
    for(size_t i = 0; i < nparams; ++i) {
        p(i) = phys(i) + h;
        Eigen::MatrixXf plus = get_probs(p, var_arrs);
        p(i) = phys(i) - h;
        Eigen::MatrixXf minus = get_probs(p, var_arrs);
        p(i) = phys(i);
        grads[i] = (plus - minus) / (2.0f * h);
    }
    return grads;
}

float PROmodel::physics_prior_quantity(const PhysicsPrior &p, const Eigen::VectorXf &phys, Eigen::VectorXf *dqdx) const {
    float q = 0.0f;
    for(const auto &[i, c] : p.terms) {
        const float x = phys((Eigen::Index)i);
        q += c * (is_log10[i] ? std::pow(10.0f, x) : x);
    }
    const float sign = !p.abs ? 1.0f : (q > 0 ? 1.0f : (q < 0 ? -1.0f : 0.0f));
    if(dqdx)
        for(const auto &[i, c] : p.terms) {
            const float x = phys((Eigen::Index)i);
            (*dqdx)((Eigen::Index)i) += sign * c * (is_log10[i] ? (float)M_LN10 * std::pow(10.0f, x) : 1.0f);
        }
    return p.abs ? std::fabs(q) : q;
}

std::vector<float> ToyPhysicsPriorCenters(const PROmodel &model, const Eigen::VectorXf &truth_phys,
                                          bool fluctuate, std::mt19937 &rng) {
    std::vector<float> centers;
    centers.reserve(model.physics_priors.size());
    for(const auto &p : model.physics_priors) {
        const float q = model.physics_prior_quantity(p, truth_phys);
        centers.push_back(fluctuate ? std::normal_distribution<float>(q, p.sigma)(rng) : q);
    }
    return centers;
}

void PROmodel::build_param_index() {
    param_name_to_index.clear();
    for(size_t i = 0; i < param_names.size(); ++i){
        param_name_to_index[param_names[i]] = i;
    }
}

// Canonical model tags follow regime_model_parameterization(_NC); legacy tags
// remain accepted as deprecated aliases and are mapped here (with a one-time
// warning per tag) before any dispatch.
static std::string canonicalize_model_tag(const std::string &name) {
    static const std::map<std::string, std::string> legacy_aliases = {
        {"numudis",              "SBL_2flav_numudis"},
        {"nueapp",               "SBL_2flav_nueapp"},
        {"nuedis",               "SBL_2flav_nuedis"},
        {"NCnumudisapp",         "SBL_2flav_numudis_NC"},
        {"NCdisapp",             "SBL_2flav_nudis_NC"},
        {"3+1",                  "SBL_3+1_Usq"},
        {"3+1_NC",               "SBL_3+1_Usq_NC"},
        {"3+1_angles",           "SBL_3+1_angles"},
        {"3+1_angles_NC",        "SBL_3+1_angles_NC"},
        {"3+1_3A",               "SBL_3+1_sinsq2thee"},
        {"3+1_3A_NC",            "SBL_3+1_sinsq2thee_NC"},
        {"3+1_3B",               "SBL_3+1_sinsq2thmumu"},
        {"3+1_3B_NC",            "SBL_3+1_sinsq2thmumu_NC"},
        {"3+1_3C",               "SBL_3+1_sinsq2thmue"},
        {"3+1_3C_NC",            "SBL_3+1_sinsq2thmue_NC"},
        {"3+1_decay_invis",      "SBL_3+1+decay_invis"},
        {"3+1_decay_vis_model1", "SBL_3+1+decay_vis1"},
        {"3+1_decay_vis_model2", "SBL_3+1+decay_vis2"},
        {"3+2",                  "SBL_3+2_Usq"},
        {"LBL",                  "LBL_3nu-matter_angles"},
        {"nullmodel",            "null"},
        {"template_fit",         "template"},
    };
    auto it = legacy_aliases.find(name);
    if(it == legacy_aliases.end()) return name;
    static std::set<std::string> warned;
    if(warned.insert(name).second) {
        log<LOG_WARNING>(L"%1% || Model tag '%2%' is deprecated; use '%3%' in the XML <model tag=...>. The legacy tag keeps working as an alias.")
            % __func__ % name.c_str() % it->second.c_str();
    }
    return it->second;
}

static std::unique_ptr<PROmodel> construct_model(const std::string &name, const PROconfig &config, const PROpeller &prop) {
    if(name == "null") {
        return std::unique_ptr<PROmodel>(new NullModel(prop));
    } else if(name == "numudisTEST") {
        return std::unique_ptr<PROmodel>(new PROnumudisTEST(prop,config.m_model_parameter_map));
    } else if(name == "SBL_3+1+decay_invis") {
        return std::unique_ptr<PROmodel>(new PRO3p1_decay_invis(prop,config.m_model_parameter_map));
    } else if(name == "SBL_3+1+decay_vis1") {
        return std::unique_ptr<PROmodel>(new PRO3p1_decay_vis_model1(prop,config.m_model_parameter_map));
    } else if(name == "SBL_3+1+decay_vis2") {
        return std::unique_ptr<PROmodel>(new PRO3p1_decay_vis_model2(prop,config.m_model_parameter_map));
    } else if(name == "SBL_3+2_Usq") {
        return std::unique_ptr<PROmodel>(new PRO3p2(prop, config.m_model_parameter_map));
    } else if(name == "LBL_3nu-matter_angles" || name == "LBL_3nu-vacuum_angles") {
        return std::unique_ptr<PROmodel>(new PROLBL(config, prop, name == "LBL_3nu-matter_angles"));
    } else if(name == "LBL_3+1-vacuum_angles") {
        return std::unique_ptr<PROmodel>(new PROLBL3p1(config, prop));
    } else if(name == "template") {
        return std::unique_ptr<PROmodel>(new PROtemplate(config, prop));
    }
    // Sine-kernel family (SBL_2flav_*, SBL_3+1_* and their _NC variants):
    // recipe-driven, all evaluated by PROsineModel. See PROmodelSine.h.
    if(const SineModelRecipe *recipe = find_sine_recipe(name)) {
        return std::unique_ptr<PROmodel>(new PROsineModel(prop, config.m_model_parameter_map, *recipe));
    }
    log<LOG_ERROR>(L"%1% || Unrecognized model name %2%. Valid tags (regime_model_parameterization(_NC); legacy pre-v3.1 names are auto-mapped): null, template, SBL_2flav_(numudis,nueapp,nuedis), SBL_2flav_(numudis,nudis)_NC, SBL_3+1_(Usq,angles,sinsq2thee,sinsq2thmumu,sinsq2thmue) and their _NC versions, SBL_3+1+decay_(invis,vis1,vis2), SBL_3+2_Usq, LBL_3nu-matter_angles, LBL_3nu-vacuum_angles, LBL_3+1-vacuum_angles. Terminating.") % __func__ % name.c_str();
    exit(EXIT_FAILURE);
}

std::unique_ptr<PROmodel> get_model_from_string(const PROconfig& config, const PROpeller &prop) {
    std::string name = canonicalize_model_tag(config.m_model_tag);

    if(!config.m_model_options.empty() && name.rfind("LBL_", 0) != 0) {
        std::string keys;
        for(const auto &kv : config.m_model_options) keys += (keys.empty() ? "" : ", ") + kv.first;
        log<LOG_ERROR>(L"%1% || <model> options [%2%] are only supported by the LBL_* models, but the model tag is '%3%'. Terminating.")
            % __func__ % keys.c_str() % name.c_str();
        exit(EXIT_FAILURE);
    }

    if(name != "template") {
        for(size_t k = 0; k < config.m_model_parameter_names.size(); ++k) {
            if(config.m_model_parameter_subchannels[k].empty() && !config.m_model_parameter_default[k]) continue;
            log<LOG_ERROR>(L"%1% || <parameter name=\"%2%\"> sets subchannels= or default=, which only the template model supports, but the model tag is '%3%'. Terminating.")
                % __func__ % config.m_model_parameter_names[k].c_str() % name.c_str();
            exit(EXIT_FAILURE);
        }
    }

    std::unique_ptr<PROmodel> model = construct_model(name, config, prop);

    for(const PhysicsPriorSpec &spec : config.m_model_physics_priors) {
        PROmodel::PhysicsPrior p;
        p.name = spec.name;
        auto it = std::find(model->param_names.begin(), model->param_names.end(), spec.name);
        if(it != model->param_names.end()) {
            p.terms = {{(size_t)std::distance(model->param_names.begin(), it), 1.0f}};
        } else if(model->derived_quantities.count(spec.name)) {
            p.terms = model->derived_quantities.at(spec.name);
        } else {
            std::vector<std::string> derived;
            for(const auto &[dn, terms] : model->derived_quantities) derived.push_back(dn);
            log<LOG_ERROR>(L"%1% || <model><prior name=\"%2%\"> is neither a physics parameter %3% nor a derived quantity %4% of model '%5%'. Terminating.")
                % __func__ % spec.name.c_str() % model->param_names % derived % name.c_str();
            exit(EXIT_FAILURE);
        }
        p.center = (float)spec.center;
        p.sigma  = (float)spec.sigma;
        p.abs    = spec.abs;
        model->physics_priors.push_back(p);
    }
    // One-time: FC/AFC build a fresh model per worker.
    static std::atomic<bool> logged_priors{false};
    if(!model->physics_priors.empty() && !logged_priors.exchange(true)) {
        for(const auto &p : model->physics_priors) {
            std::string q;
            for(const auto &[i, c] : p.terms) {
                const std::string v = model->is_log10[i] ? "10^" + model->param_names[i] : model->param_names[i];
                q += (q.empty() ? (c < 0 ? "-" : "") : (c < 0 ? " - " : " + ")) + (std::fabs(c) == 1.0f ? v : std::to_string(std::fabs(c)) + "*" + v);
            }
            log<LOG_INFO>(L"%1% || Physics prior on %2%: ((%3%%4%%5% - %6%) / %7%)^2, in physical units.")
                % __func__ % p.name.c_str() % (p.abs ? "|" : "") % q.c_str() % (p.abs ? "|" : "") % p.center % p.sigma;
        }
    }
    return model;
}


}
