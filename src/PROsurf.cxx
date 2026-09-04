#include "PROsurf.h"
#include "PROfitter.h"
#include "PROlog.h"
#include "PROmeshEval.h"
#include "PROmeshPlot.h"
#include "PRObe.h"
#include "PROwatermark.h"

#include <Eigen/Eigen>

#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <algorithm>
#include <functional>
#include <fstream>
#include <iomanip>

#include "TGraph.h"
#include "TLatex.h"
#include "TLine.h"
#include "TMarker.h"
#include "TPaveText.h"
#include "TH2D.h"
#include "TH1F.h"
#include "TBox.h"

using namespace PROfit;

std::vector<float> linspace(float start, float end, int N, bool endpoint = true) {
    std::vector<float> result;
    result.reserve(N);
    if (N == 0) return result;
    if (N == 1) {
        result.push_back(start);
        return result;
    }

    float step = (end - start) / (endpoint ? (N - 1) : N);
    for (int i = 0; i < N; ++i) {
        result.push_back(start + i * step);
    }
    return result;
}
std::vector<float> combined_sparse_dense(float Amin, float Amax, float CV, int Nsparse, int Ndense, float dense_width) {
    std::vector<float> sparse = linspace(Amin, Amax, Nsparse);  // global scan
    std::vector<float> dense = linspace(CV - dense_width / 2.0, CV + dense_width / 2.0, Ndense-1);  // local dense
    dense.push_back(CV);

    sparse.insert(sparse.end(), dense.begin(), dense.end());
    std::sort(sparse.begin(), sparse.end());

    sparse.erase(std::unique(sparse.begin(), sparse.end(),[](float a, float b) { return std::fabs(a - b) < 1e-8; }), sparse.end());

    return sparse;
}
std::vector<float> combined_sparse_seed(float Amin, float Amax, std::vector<float> seeds, int Nsparse, int Nocal) {
    std::vector<float> sparse = linspace(Amin, Amax, Nsparse);  // global scan

    std::vector<float> local_points;
    for (float seed : seeds) {
        local_points.push_back(seed);

        std::vector<float> log_offsets = linspace(-1.5f, -0.5f, Nocal);

        for (float log_offset : log_offsets) {
            float offset = std::pow(10.0f, log_offset);

            // Add both positive and negative offsets around seed
            float pos_point = seed + offset;
            float neg_point = seed - offset;

            if (pos_point >= Amin && pos_point <= Amax) {
                local_points.push_back(pos_point);
            }
            if (neg_point >= Amin && neg_point <= Amax) {
                local_points.push_back(neg_point);
            }
        }
    }

    sparse.insert(sparse.end(), local_points.begin(), local_points.end());

    // Remove duplicates with tolerance and sort
    std::sort(sparse.begin(), sparse.end());
    sparse.erase(std::unique(sparse.begin(), sparse.end(), 
                [](float a, float b) { return std::abs(a - b) < 0.015f; }), 
            sparse.end());

    return sparse;
}



PROsurf::PROsurf(PROmetric &metric,  size_t x_idx, size_t y_idx, size_t nbinsx, LogLin llx, float x_lo, float x_hi, size_t nbinsy, LogLin lly, float y_lo, float y_hi) : metric(metric), x_idx(x_idx), y_idx(y_idx), nbinsx(nbinsx), nbinsy(nbinsy), edges_x(Eigen::VectorXf::Constant(nbinsx + 1, 0)), edges_y(Eigen::VectorXf::Constant(nbinsy + 1, 0)), surface(nbinsx, nbinsy) {
          
    // If it's a log axis, we always convert to log space in order to define the grid
    if(llx == LogAxis) {
        x_lo = std::log10(x_lo);
        x_hi = std::log10(x_hi);
    }
    if(lly == LogAxis) {
        y_lo = std::log10(y_lo);
        y_hi = std::log10(y_hi);
    }
    
    for(size_t i = 0; i < nbinsx + 1; i++) {
        float edge_val = x_lo + i * (x_hi - x_lo) / nbinsx;
        if (llx == LogAxis) edge_val = std::pow(10, edge_val); // Now bin edges are back to linear space
        if (metric.GetModel().is_log10[x_idx])
            edges_x(i) = std::log10(edge_val);
        else
            edges_x(i) = edge_val;
    }
    for(size_t i = 0; i < nbinsy + 1; i++) {
        float edge_val = y_lo + i * (y_hi - y_lo) / nbinsy;
        if (lly == LogAxis) edge_val = std::pow(10, edge_val); // Now bin edges are back to linear space
        if (metric.GetModel().is_log10[y_idx])
            edges_y(i) = std::log10(edge_val);
        else
            edges_y(i) = edge_val;
    }

    /*
    for(size_t i = 0; i < nbinsx + 1; i++)
        log<LOG_ERROR>(L"%1% || xbin edge %2%: %3%") % __func__ % i % edges_x(i);
    for(size_t i = 0; i < nbinsy + 1; i++)
        log<LOG_ERROR>(L"%1% || ybin edge %2%: %3%") % __func__ % i % edges_y(i);
    */

}

void PROsurf::FillSurfaceStat(const PROconfig &config, const PROfitterConfig &fitconfig, std::string filename, const Eigen::VectorXf &cv_params, uint32_t seed) {
    std::ofstream chi_file;
    if(!filename.empty()){
        chi_file.open(filename);
        chi_file << "Dimensions: " << nbinsx << " " << nbinsy << "\n";
        chi_file << "Fixed indices: " << x_idx << " " << y_idx << "\n";
        chi_file << "Parameters:\n";
        for(const auto &name: metric.GetModel().param_names) chi_file << name << "\n";

        chi_file << "xval yval chi2";
        // TODO: Not saving this info for stats only right now (but we should)
        //for(size_t i = 0; i < metric.GetModel().nparams; ++i)
        //    chi_file << " p" << i;
        chi_file << "\n";
    }

    PROmetric *local_metric = metric.Clone();
    PROsyst newsyst = local_metric->GetSysts();
    newsyst.fractional_covariance = Eigen::MatrixXf::Constant(config.m_num_variable_bins_total[config.i_prime], config.m_num_variable_bins_total[config.i_prime], 0);
    local_metric->override_systs(newsyst);
    // When doing actual fits here we need to set the bounds of the metric
    // so the bounds of the systs are the cv values
    float min_chi = 1e9;
    Eigen::VectorXf dummy_grad = cv_params;
    Eigen::VectorXf params = cv_params;
    Eigen::VectorXf last;

    // TODO: If x_idx or y_idx refers to a systematic I think this may or may not be what we want
    // TODO: If one of the model parameters is fixed (using --fix) this will not fix that parameter (Will the regular surface fit fix that parameter? Or fix a systematic we wanted to fix?)

    if(local_metric->GetModel().nparams - 2 == 0) {
        for(size_t i = 0; i < nbinsx; i++) {
            for(size_t j = 0; j < nbinsy; j++) {
                params(y_idx) = (float)edges_y(j);
                params(x_idx) = (float)edges_x(i);
                float fx = (*local_metric)(params, dummy_grad, false);
                if(fx < min_chi) min_chi = fx;
                surface(i, j) = fx;
            }
        }
    } else {
        Eigen::VectorXf lb = params;
        for(size_t i = 0; i < local_metric->GetModel().nparams; ++i)
            lb(i) = local_metric->GetModel().lb(i);
        Eigen::VectorXf ub = params;
        for(size_t i = 0; i < local_metric->GetModel().nparams; ++i)
            ub(i) = local_metric->GetModel().ub(i);

        for(size_t i = 0; i < nbinsx; i++) {
            for(size_t j = 0; j < nbinsy; j++) {
                lb(y_idx) = (float)edges_y(j);
                ub(y_idx) = (float)edges_y(j);
                lb(x_idx) = (float)edges_x(i);
                ub(x_idx) = (float)edges_x(i);
                local_metric->setBounds(lb,ub);
                PROfitter fitter(ub, lb, fitconfig, seed+i+nbinsx*j);
                float fx;
                if(i != 0 || j != 0){
                    fx = fitter.Fit(*local_metric, last);
                }else{
                    fx = fitter.Fit(*local_metric);
                }
                last = fitter.best_fit;
                if(fx < min_chi) min_chi = fx;
                surface(i, j) = fx;
            }
        }
    }
    for(size_t i = 0; i < nbinsx; ++i) {
        for(size_t j = 0; j < nbinsy; ++j) {
            float fx = surface(i,j); 
            fx -= min_chi;
            surface(i,j) = fx;
            if(!filename.empty()){
                chi_file<<"\n"<<edges_x(i)<<" "<<edges_y(j)<<" "<<fx<<std::flush;
            }
        }
    }
    delete local_metric;
}

namespace {

// PROfile's cross-thread seed_bank exposed through the shared ScanPointStore
// interface, bound to one scanned parameter. All access serialized by the
// bank mutex; store ops are microseconds against ~0.1-1 s fits, so a single
// mutex is plenty.
struct ProfileBankView final : public ScanPointStore {
    PROfile &prof;
    int param_idx;
    ProfileBankView(PROfile &p, int idx) : prof(p), param_idx(idx) {}

    std::vector<ScanPoint> snapshot() const override {
        std::lock_guard<std::mutex> lock(prof.seed_bank_mutex);
        if(param_idx < 0 || param_idx >= (int)prof.seed_bank.size()) return {};
        return prof.seed_bank[param_idx];
    }
    Eigen::VectorXf nearest_bf(float value) const override {
        std::lock_guard<std::mutex> lock(prof.seed_bank_mutex);
        if(param_idx < 0 || param_idx >= (int)prof.seed_bank.size()) return {};
        const auto &entries = prof.seed_bank[param_idx];
        if(entries.empty()) return {};
        size_t nearest = 0;
        float best_d = std::fabs(entries[0].value - value);
        for(size_t k = 1; k < entries.size(); ++k) {
            const float d = std::fabs(entries[k].value - value);
            if(d < best_d) { best_d = d; nearest = k; }
        }
        return entries[nearest].best_fit;
    }
    bool has_close(float value, float tol) const override {
        std::lock_guard<std::mutex> lock(prof.seed_bank_mutex);
        if(param_idx < 0 || param_idx >= (int)prof.seed_bank.size()) return false;
        for(const auto &e : prof.seed_bank[param_idx])
            if(std::fabs(e.value - value) < tol) return true;
        return false;
    }
    void add(const ScanPoint &pt) override {
        if(pt.best_fit.size() == 0) return;
        std::lock_guard<std::mutex> lock(prof.seed_bank_mutex);
        if(param_idx < 0 || param_idx >= (int)prof.seed_bank.size()) return;
        prof.seed_bank[param_idx].push_back(pt);
    }
};

// Scan-space bounds for the profile fits: physics from the model (with_osc)
// or pinned (syst-only), splines from their restrict/lo-hi ranges. Single
// home of logic previously duplicated between the osc / non-osc branches.
// Syst-only pinning uses the physics values of the caller's FIRST seed (the
// global best fit — the same point minchi was computed at); model default_val
// is only the fallback when no seeds exist. Pinning at default_val while the
// CV/injected physics differed biased every knob_chi of a syst-only profile.
void buildScanBounds(const PROmodel &model, const PROsyst &systs, bool with_osc,
                     const Eigen::VectorXf *seed_physics,
                     Eigen::VectorXf &lb, Eigen::VectorXf &ub) {
    const int nparams = (int)model.nparams + (int)systs.GetNSplines();
    lb = Eigen::VectorXf::Constant(nparams, -3.0);
    ub = Eigen::VectorXf::Constant(nparams, 3.0);
    const size_t nphys = model.nparams;
    for(size_t j = 0; j < nphys; ++j) {
        if(with_osc) {
            lb(j) = model.lb(j);
            ub(j) = model.ub(j);
        } else {
            const float pin = (seed_physics && (long)seed_physics->size() > (long)j)
                              ? (*seed_physics)((int)j) : model.default_val(j);
            lb(j) = pin;
            ub(j) = pin;
        }
    }
    for(int j = (int)nphys; j < nparams; ++j) {
        const size_t si = j - nphys;
        lb(j) = systs.spline_has_restrict[si] ? systs.spline_restrict_lo[si] : systs.spline_lo[si];
        ub(j) = systs.spline_has_restrict[si] ? systs.spline_restrict_hi[si] : systs.spline_hi[si];
    }
}

} // anonymous namespace

std::vector<profOut> PROfile::PROfilePointHelper(const PROsyst *systs, const PROfitterConfig &fitconfig, std::atomic<int> *task_counter, const std::vector<ScanTask> *tasks, float minchi, bool with_osc, MultiPROgressBar& progressbar, const std::vector<Eigen::VectorXf> &seed_points, uint32_t seed, bool use_probe, std::atomic<int>* tasks_remaining, int bar_index_offset, std::atomic<uint64_t>* max_thread_wall_us) {

    // Per-thread wall start. When max_thread_wall_us is non-null, we atomically
    // max-merge our wall into it at function exit, giving the dispatcher the
    // worst-case thread wall for parallelism-efficiency reporting.
    const auto thread_t0 = std::chrono::steady_clock::now();

    std::vector<profOut> outs;
    // Thread-local metric clone; unique_ptr so an escaping exception cannot leak it.
    std::unique_ptr<PROmetric> local_metric(metric.Clone());
    const int nparams = local_metric->GetModel().nparams + systs->GetNSplines();
    constexpr int nstep = 18; ///< Legacy uniform scan: nstep+1 fixed-value fits per parameter.

    Eigen::VectorXf ub, lb, tub, tlb;
    buildScanBounds(local_metric->GetModel(), *systs, with_osc,
                    seed_points.empty() ? nullptr : &seed_points.front(), lb, ub);

    // Dynamic dispatch: pull the next available task from the shared atomic counter.
    // Each task represents "scan parameter task.param_idx over [task.sub_lb, task.sub_ub]".
    // Most parameters generate a single task spanning their full range; physics
    // parameters may be split into multiple chunked tasks via --probe-chunks. Each task
    // runs atomically (PRObe phases, or the legacy 18-uniform scan) on whichever
    // thread grabs it.
    while (true) {
        int t_idx = task_counter->fetch_add(1);
        if (t_idx >= (int)tasks->size()) break;
        const ScanTask &task = (*tasks)[t_idx];
        int i = task.param_idx;

        tlb = lb;
        tub = ub;
        // Override the scanned parameter's range with this task's sub-range. Other
        // parameters retain their full lb/ub (so the fit's free axes are unchanged).
        tlb(i) = task.sub_lb;
        tub(i) = task.sub_ub;

        local_metric->reset();

        size_t which_spline = i;
        bool isphys = which_spline < local_metric->GetModel().nparams;
        profOut output;
        output.param_idx = i;

        log<LOG_INFO>(L"%1% || Worker picked task #%2%: param %3% on [%4%, %5%]")
            % __func__ % t_idx % i % task.sub_lb % task.sub_ub;

        // Shared task epilogue (previously duplicated byte-for-byte in both
        // branches): sort this task's points, hand them to the dispatcher, and
        // complete the parameter's progress bar when its last task finishes.
        auto finish_task = [&]() {
            output.sort();
            outs.push_back(std::move(output));
            if (tasks_remaining) {
                const int bar_idx = (int)which_spline - bar_index_offset;
                const int prev = tasks_remaining[bar_idx].fetch_sub(1);
                if (prev <= 1 && fitconfig.progress_bar) {
                    progressbar.complete_bar(bar_idx);
                }
            }
        };

        // Task-level isolation: one pathological parameter/point must not abort
        // the whole profile (it previously unwound through std::async, leaked
        // the metric clone and stranded the progress display thread). Per-fit
        // failures are already absorbed inside ScanFitContext::fitAt; this
        // catches anything escaping the scan logic itself.
        try {

        // The shared per-parameter store and fit executor: every fit of this
        // task (either branch) seeds from globals + nearest banked point,
        // deposits into the bank, and advances one RNG stream. The base seed
        // is spaced by task so no two tasks share a per-fit seed sequence
        // (101 > any per-task fit count).
        ProfileBankView store(*this, i);
        std::function<void()> on_fit_cb;
        if (fitconfig.progress_bar) {
            const int bar_idx_for_cb = (int)which_spline - bar_index_offset;
            on_fit_cb = [&progressbar, bar_idx_for_cb]() {
                progressbar.increment_bar(bar_idx_for_cb);
            };
        }
        ScanFitContext ctx{*local_metric, tlb, tub, which_spline, fitconfig,
                           seed + (uint32_t)t_idx * 101u, seed_points, store, on_fit_cb,
                           &fixed_seed_points, &global_min_tracker};

        // Fixed parameter (--fix / lb==ub): one fit at the pinned value, same
        // for both scan strategies (PRObe previously returned an EMPTY scan
        // here, breaking Plot's 1-point special case under --probe).
        if (tlb(i) == tub(i)) {
            auto out = ctx.fitAt(tlb(i));
            if (out.ok) {
                output.knob_vals.push_back(out.pt.value);
                output.knob_chis.push_back(out.pt.chi2 - minchi);
                output.knob_bfs.push_back(out.pt.best_fit);
            }
            finish_task();
            continue;
        }

        // ---- PRObe path: adaptive importance sampling for the dchi2=1 band ----
        // Default off; enabled per-call via use_probe. Produces the same profOut
        // shape as the legacy 18-uniform scan so all downstream plotting works.
        if(use_probe){
            PRObe::CrossingOpts opts;
            // Detect whether this task represents a chunk (sub-range < full range
            // for the scanned param). Spike detection stays ON for all physics
            // tasks (chunked or not) — narrow basins can hide anywhere — but in
            // chunked mode we lower per-chunk anchor and coarse counts since
            // multiple chunks cumulatively cover the param. Per-chunk density is
            // still higher than the unchunked equivalent because each chunk
            // covers only 1/N of the range.
            // NOTE: GetModel().lb/ub are physics-only (size = nphys); guard the
            // access so we don't read out of range when which_spline indexes a
            // nuisance. Splines are never chunked so task_is_chunk = false for them.
            bool task_is_chunk = false;
            if (isphys) {
                const float full_lb_cmp = finite_or(local_metric->GetModel().lb(which_spline), -3.0f);
                const float full_ub_cmp = finite_or(local_metric->GetModel().ub(which_spline),  3.0f);
                task_is_chunk = (task.sub_lb > full_lb_cmp + 1e-4f) ||
                                (task.sub_ub < full_ub_cmp - 1e-4f);
            }
            opts.may_have_spikes = isphys;
            opts.target_dchi2    = 1.0f;          // 1 sigma band
            if (task_is_chunk) {
                // Reduce per-chunk fit count: 5 anchors + 5 coarse instead of 9+9.
                // With chunks=N, total points across the param ~ N x 10 vs unchunked 18,
                // so coverage is comparable while per-chunk wall time drops by ~50%.
                opts.anchor_sigmas = {0.0f, -0.6f, 0.6f, -1.3f, 1.3f};
                opts.coarse_n      = 5;
            }
            // sigma_init: nuisances are already in sigma-units; for physics, derive
            // from this task's sub-range (chunk width / 6 ~ a 3 sigma half-width within
            // the chunk). For unchunked tasks this is the full-range / 6 as before.
            float scale = 1.0f;
            if(isphys){
                const float plb_m = finite_or(task.sub_lb, -3.0f);
                const float pub_m = finite_or(task.sub_ub,  3.0f);
                const float width = pub_m - plb_m;
                scale = (std::isfinite(width) && width > 0.0f) ? width / 6.0f : 1.0f;
            }
            opts.sigma_init = scale;

            // Anchor centre: prefer the global BF if it lies in this task's chunk;
            // otherwise use the chunk midpoint so anchors don't all clamp to a
            // boundary. For unchunked tasks the chunk == full range, so this
            // reduces to the global BF (existing behaviour).
            const float global_bf = !seed_points.empty()
                                    ? seed_points.front()((int)which_spline)
                                    : (isphys ? local_metric->GetModel().default_val((int)which_spline) : 0.0f);
            float bf_value = (global_bf >= task.sub_lb && global_bf <= task.sub_ub)
                             ? global_bf
                             : 0.5f * (task.sub_lb + task.sub_ub);

            // All seeding (globals + bank-nearest per fit), bank deposits,
            // dedup against other chunks' points, and live progress run inside
            // ctx / the shared store now.
            PRObe::CrossingResult r = PRObe::chi2_crossing_1d(ctx, bf_value, opts);

            for(size_t k = 0; k < r.theta.size(); ++k){
                output.knob_vals.push_back(r.theta[k]);
                output.knob_chis.push_back(r.chi2[k] - minchi);
                output.knob_bfs.push_back(r.best_fits[k]);
            }
            log<LOG_INFO>(L"%1% || PRObe done for spline #%2%: %3% fits, surrogate=%4%")
                % __func__ % which_spline % r.n_fits % (int)r.used_surrogate;

            finish_task();
            continue; // skip the legacy 18-uniform loop below
        }

        //first get what values to sample
        std::vector<float> test_values;

        if(!isphys){
            //if lower bound is 0 or both pos/both negative (aka not sym around zero)
            if(lb(which_spline)==0 || (lb(which_spline)*ub(which_spline) >0) ){
                for (int j = 0; j <= nstep; ++j) {
                    float which_value = finite_or(lb(which_spline), -3.0f) + (ub(which_spline) - finite_or(lb(which_spline), -3.0f)) * j / (float)nstep;
                    test_values.push_back(which_value);
                }
            }else{
                // Center-out walk: mid -> top, then jump back to just below mid
                // -> bottom. The seed bank's nearest-by-value lookup makes the
                // jump seamless (the old previous-point chaining did not).
                for (int j = 0; j <= nstep; ++j) {
                    int k;
                    if (j <= nstep - nstep / 2) {
                        k = nstep / 2 + j;  // Forward direction
                    } else {
                        k = nstep - j;  // Backward direction
                    }
                    float which_value = finite_or(lb(which_spline), -3.0f) + (ub(which_spline) - finite_or(lb(which_spline), -3.0f)) * k / (float)nstep;
                    test_values.push_back(which_value);
                }
            }
        }else{
            //if its physics, grab seed points, need to include those
            std::vector<float> seed_values(seed_points.size());
            std::transform(seed_points.begin(), seed_points.end(), seed_values.begin(), [which_spline](const auto& vec) { return vec[which_spline]; });
            float mod = 0.5;
            while(test_values.size()<nstep*1.5){
                test_values = combined_sparse_seed(finite_or(lb(which_spline), -3.0f), ub(which_spline), seed_values, nstep*mod, 2);
                mod=mod*1.2;
            }
            log<LOG_INFO>(L"%1% || PROfileing over physics parameter number %2% has %3% uniform points, and %4% local ones for a total of %5% points. ") % __func__ %  which_spline % int(nstep*0.5) %int((2*2+1)*seed_values.size()) % test_values.size();
        }

        // Legacy fixed-grid scan, one shared-executor fit per value. Failed
        // fits (exception / non-finite chi2) are skipped rather than recorded,
        // so the knob arrays stay consistent.
        for(auto &which_value: test_values){
            auto out = ctx.fitAt(which_value);
            if(!out.ok) continue;
            output.knob_vals.push_back(out.pt.value);
            output.knob_chis.push_back(out.pt.chi2 - minchi);
            output.knob_bfs.push_back(out.pt.best_fit);

            std::string spec_string = "";
            for(auto &f : out.pt.best_fit) spec_string+=" "+std::to_string(f);
            log<LOG_INFO>(L"%1% || Fixed value of spline # %2% is value %3%, has a chi post of : %4% (i %5% nstep %6% ") % __func__ % which_spline % which_value % out.pt.chi2 % i % nstep;
            log<LOG_INFO>(L"%1% || at a BF param value of @ %2%") % __func__ %  spec_string.c_str();
        }    //end step loop
        if(output.knob_vals.empty())
            log<LOG_ERROR>(L"%1% || Every scan-point fit of parameter %2% failed; its profile will be empty.") % __func__ % which_spline;
        finish_task();

        } catch(const std::exception &e) {
            log<LOG_ERROR>(L"%1% || Task #%2% (param %3%) aborted with '%4%'; emitting %5% completed point(s) and continuing.")
                % __func__ % t_idx % i % e.what() % output.knob_vals.size();
            finish_task();
        }
    }//end thread

    // Max-merge our wall time into the dispatcher's atomic. CAS loop because
    // std::atomic<uint64_t>::fetch_max isn't standard until C++26.
    if (max_thread_wall_us) {
        const uint64_t my_wall_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - thread_t0).count();
        uint64_t cur = max_thread_wall_us->load(std::memory_order_relaxed);
        while (my_wall_us > cur &&
               !max_thread_wall_us->compare_exchange_weak(
                   cur, my_wall_us,
                   std::memory_order_relaxed, std::memory_order_relaxed)) {
            // cur is updated by compare_exchange_weak on failure; retry
        }
    }

    return outs;
}



std::vector<surfOut> PROsurf::PointHelper(const PROfitterConfig &fitconfig, std::vector<surfOut> multi_physics_params, std::atomic<int> *point_counter, uint32_t seed, const std::vector<Eigen::VectorXf> &seed_pts, MultiPROgressBar* progressbar){

    std::vector<surfOut> outs;
    const int total = (int)multi_physics_params.size();

    // Make a local copy for this thread
    PROmetric *local_metric = metric.Clone();

    // Dynamic dispatch: pull the next available grid point from the shared atomic
    // counter. The previous static contiguous-block dispatch left fast threads idle
    // at the tail of an uneven surface.
    while (true) {
        int i = point_counter->fetch_add(1);
        if (i >= total) break;
        local_metric->reset();

        surfOut output;
        std::vector<float> physics_params = multi_physics_params[i].grid_val;
        output.grid_val = physics_params;
        output.grid_index = multi_physics_params[i].grid_index;

        int nparams = local_metric->GetModel().nparams + local_metric->GetSysts().GetNSplines() - 2;

        if(nparams == 0) {
            // Zero free parameters: one metric evaluation, no PROfitter.
            // Map grid values by AXIS INDEX — grid_val is stored [y, x] while
            // the parameter vector is model-ordered, so the old positional
            // Map() was transposed whenever (x_idx, y_idx) != (1, 0)
            // (e.g. swapped --xvar/--yvar).
            Eigen::VectorXf empty_vec, params(2);
            params((int)x_idx) = physics_params[1];
            params((int)y_idx) = physics_params[0];
            output.chi = (*local_metric)(params, empty_vec, false);
            output.best_fit = params;
            outs.push_back(output);
            if (progressbar) progressbar->increment_bar(0);
            continue;
        }

        Eigen::VectorXf lb(nparams+2);
        lb << local_metric->GetModel().lb, Eigen::VectorXf::Map(local_metric->GetSysts().spline_lo.data(), local_metric->GetSysts().spline_lo.size());
        Eigen::VectorXf ub(nparams+2);
        ub << local_metric->GetModel().ub, Eigen::VectorXf::Map(local_metric->GetSysts().spline_hi.data(), local_metric->GetSysts().spline_hi.size());
        for(size_t si = 0; si < local_metric->GetSysts().GetNSplines(); ++si) {
            if(!local_metric->GetSysts().spline_has_restrict[si]) continue;
            lb(local_metric->GetModel().nparams + si) = local_metric->GetSysts().spline_restrict_lo[si];
            ub(local_metric->GetModel().nparams + si) = local_metric->GetSysts().spline_restrict_hi[si];
        }

        lb(x_idx) = multi_physics_params[i].grid_val[1];
        ub(x_idx) = multi_physics_params[i].grid_val[1];
        lb(y_idx) = multi_physics_params[i].grid_val[0];
        ub(y_idx) = multi_physics_params[i].grid_val[0];

        local_metric->setBounds(lb,ub);

        PROfitter fitter(ub, lb, fitconfig, seed+(uint32_t)i);
        // Warm-start: every fit gets ALL caller-supplied seeds (global best fit +
        // harmonic freq_seed_points, mirroring PROfile's per-scan-fit seeding)
        // plus the CLOSEST point in (x, y) grid space that this thread has
        // already fitted — not the most recently pulled one: with dynamic
        // dispatch the previous pull can be far across the surface, while the
        // nearest explored point is the best predictor of the local minimum.
        // Distance is Euclidean in the grid's own coordinates (log10 values on
        // log axes), i.e. the fit's parameter space.
        std::vector<Eigen::VectorXf> seeds = seed_pts;
        if(!outs.empty()){
            size_t nearest = 0;
            float best_d2 = std::numeric_limits<float>::max();
            for(size_t k = 0; k < outs.size(); ++k){
                const float dx = outs[k].grid_val[1] - physics_params[1];
                const float dy = outs[k].grid_val[0] - physics_params[0];
                const float d2 = dx*dx + dy*dy;
                if(d2 < best_d2){ best_d2 = d2; nearest = k; }
            }
            seeds.push_back(outs[nearest].best_fit);
        }
        output.chi = seeds.empty() ? fitter.Fit(*local_metric) : fitter.Fit(*local_metric, seeds);
        output.best_fit = fitter.best_fit;
        outs.push_back(output);
        if (progressbar) progressbar->increment_bar(0);
    }

    delete local_metric;

    return outs;
}


void PROsurf::FillSurface(const PROfitterConfig &fitconfig, std::string filename, PROseed &proseed, float min_chi, const std::vector<Eigen::VectorXf> &seed_pts, int nThreads) {
    std::ofstream chi_file;
    if(!filename.empty()){
        chi_file.open(filename);
    }

    std::vector<surfOut> grid;
    for(size_t i = 0; i < nbinsx; i++) {
        for(size_t j = 0; j < nbinsy; j++) {
            std::vector<int> grid_pts = {(int)i,(int)j};
            std::vector<float> physics_params = {(float)edges_y(j), (float)edges_x(i)};  //deltam^2, sin^22thetamumu
            surfOut pt; pt.grid_val = physics_params; pt.grid_index = grid_pts;
            grid.push_back(pt);
        }
    }

    int loopSize = grid.size();

    std::vector<std::future<std::vector<surfOut>>> futures;

    log<LOG_INFO>(L"%1% || Starting THREADS  : %2% , Loops %3% (dynamic dispatch)") % __func__ % nThreads % loopSize;

    // Single-bar progress meter for the whole surface. Each grid point increments;
    // finish_all() rounds it up to 100% at the end so we land cleanly even if a
    // grid point fails or is skipped.
    std::vector<std::pair<int, std::string>> surf_PB_configs;
    surf_PB_configs.push_back({std::max(1, loopSize), "Surface scan"});
    MultiPROgressBar surf_progress(surf_PB_configs);
    if (fitconfig.progress_bar) {
        surf_progress.initialize_display();
        surf_progress.start_display_thread();
    }
    MultiPROgressBar* surf_pb_ptr = fitconfig.progress_bar ? &surf_progress : nullptr;

    // Dynamic dispatch: workers race on this counter; each one grabs the next
    // un-claimed grid point until the surface is fully covered. Avoids tail
    // imbalance on uneven grids.
    std::atomic<int> point_counter{0};

    for (int t = 0; t < nThreads; ++t) {
        futures.emplace_back(std::async(std::launch::async, [&, t]() {
                    return this->PointHelper(fitconfig, grid, &point_counter, proseed.getThreadSeeds()->at(t), seed_pts, surf_pb_ptr);
                    }));
    }

    std::vector<surfOut> combinedResults;
    for (auto& fut : futures) {
        std::vector<surfOut> result = fut.get();
        combinedResults.insert(combinedResults.end(), result.begin(), result.end());
    }
    if (fitconfig.progress_bar) surf_progress.finish_all();

    if(filename != "") {
        chi_file << "Dimensions: " << nbinsx << " " << nbinsy << "\n";
        chi_file << "Fixed indices: " << x_idx << " " << y_idx << "\n";
        chi_file << "Parameters:\n";
        for(const auto &name: metric.GetModel().param_names) chi_file << name << "\n";
        for(const auto &name: metric.GetSysts().spline_names) chi_file << name << "\n";

        chi_file << "\nxval yval chi2";
        for(size_t i = 0; i < metric.GetModel().nparams + metric.GetSysts().GetNSplines(); ++i)
            chi_file << " p" << i;
    }
    float orig_chi = min_chi;
    for(const auto &item: combinedResults) {
        if(item.chi < orig_chi) {
            log<LOG_WARNING>(L"%1% || Found a point in the surface, index (%2%, %3%) with value (%4%, %5%), with chi^2 %6% which is lower than the minimum chi^2 passed into the function %7%. We will use this new value or the lowest other value in the surface instead of the min_chi passed in.") 
                % __func__ % item.grid_index[0] % item.grid_index[1] % item.grid_val[0] % item.grid_val[1] % item.chi % orig_chi;
        }
        if(item.chi < min_chi) {
            min_chi = item.chi;
        }
    }
    for (const auto& item : combinedResults) {
        log<LOG_INFO>(L"%1% || Finished  : %2% %3% %4%") % __func__ % item.grid_val[1] % item.grid_val[0] % (item.chi - min_chi);
        surface(item.grid_index[0], item.grid_index[1]) = item.chi - min_chi;
        results.push_back({item.grid_index[0], item.grid_index[1], item.best_fit, (item.chi-min_chi)});
        if(filename != "") {
            chi_file<<"\n"<<item.grid_val[1]<<" "<<item.grid_val[0]<<" "<<(item.chi-min_chi);
            for(float val: item.best_fit)
                chi_file << " " << val;
        }
    }
}



std::vector<surfOut> PROsurf::FillCurve(const PROfitterConfig &fitconfig, PROseed &proseed, float min_chi, const std::vector<Eigen::VectorXf> &seed_pts, int nThreads, std::vector<float> &A, std::vector<float> &B, size_t n_points) {


    std::vector<surfOut> grid;
    for(size_t i = 0; i < n_points; i++) {
        float t = (n_points == 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(n_points - 1);

        std::vector<float> physics_params(A.size());
        for(size_t k = 0; k < A.size(); k++) {
            physics_params[k] = (1.0f - t) * A[k] + t * B[k];
        }

        std::vector<int> grid_pts = {(int)i, 0};
        surfOut pt;
        pt.grid_val = physics_params;
        pt.grid_index = grid_pts;
        grid.push_back(pt);
    }

    int loopSize = grid.size();

    std::vector<std::future<std::vector<surfOut>>> futures;

    log<LOG_INFO>(L"%1% || Starting THREADS  : %2% , Loops %3% (dynamic dispatch)") % __func__ % nThreads % loopSize;

    std::vector<std::pair<int, std::string>> curve_PB_configs;
    curve_PB_configs.push_back({std::max(1, loopSize), "Curve scan"});
    MultiPROgressBar curve_progress(curve_PB_configs);
    if (fitconfig.progress_bar) {
        curve_progress.initialize_display();
        curve_progress.start_display_thread();
    }
    MultiPROgressBar* curve_pb_ptr = fitconfig.progress_bar ? &curve_progress : nullptr;

    std::atomic<int> point_counter{0};

    for (int t = 0; t < nThreads; ++t) {
        futures.emplace_back(std::async(std::launch::async, [&, t]() {
                    return this->PointHelper(fitconfig, grid, &point_counter, proseed.getThreadSeeds()->at(t), seed_pts, curve_pb_ptr);
                    }));
    }

    std::vector<surfOut> combinedResults;
    for (auto& fut : futures) {
        std::vector<surfOut> result = fut.get();
        combinedResults.insert(combinedResults.end(), result.begin(), result.end());
    }
    if (fitconfig.progress_bar) curve_progress.finish_all();

    float orig_chi = min_chi;
    for(const auto &item: combinedResults) {
        if(item.chi < orig_chi) {
            log<LOG_WARNING>(L"%1% || Found a point in the surface, index (%2%, %3%) with value (%4%, %5%), with chi^2 %6% which is lower than the minimum chi^2 passed into the function %7%. We will use this new value or the lowest other value in the surface instead of the min_chi passed in.") 
                % __func__ % item.grid_index[0] % item.grid_index[1] % item.grid_val[0] % item.grid_val[1] % item.chi % orig_chi;
        }
        if(item.chi < min_chi) {
            min_chi = item.chi;
        }
    }

    for (auto& item : combinedResults) {
        log<LOG_INFO>(L"%1% || Finished  : %2% %3% %4%") % __func__ % item.grid_val[1] % item.grid_val[0] % (item.chi - min_chi);
        item.chi -= min_chi;
    }

    return combinedResults;
}


PROmesh::AMRResult PROsurf::FillSurfaceAMR(
    const PROfitterConfig &fitconfig,
    std::string filename,
    [[maybe_unused]] PROseed &proseed,
    int nthreads,
    const std::vector<Eigen::VectorXf> &caller_seeds,
    const PROmesh::AMROptions &opts_in)
{
    // Copy options so we can fill in defaults that depend on PROsurf state.
    PROmesh::AMROptions opts = opts_in;
    opts.nthreads = nthreads;
    if (opts.dense_nx == 60 && nbinsx > 0) opts.dense_nx = (int)nbinsx;
    if (opts.dense_ny == 60 && nbinsy > 0) opts.dense_ny = (int)nbinsy;
    if (opts.initial_seed_points.empty() && !caller_seeds.empty()) {
        opts.initial_seed_points = caller_seeds;
    }

    // Bounds in transformed (log/lin) space — match the convention used by
    // edges_x / edges_y, which already store transformed values.
    const float x_lo = edges_x.minCoeff();
    const float x_hi = edges_x.maxCoeff();
    const float y_lo = edges_y.minCoeff();
    const float y_hi = edges_y.maxCoeff();

    // Single-point evaluation. Runs concurrently on `nthreads` AMR workers.
    // Per-thread state (a PROmetric clone) lives in thread_local storage so the
    // first call on each thread allocates and subsequent calls reuse it.
    PROmetric *proto = &this->metric;
    const size_t loc_x_idx = this->x_idx;
    const size_t loc_y_idx = this->y_idx;
    auto eval_fn = [proto, loc_x_idx, loc_y_idx, &fitconfig](const PROmesh::EvalRequest &req)
        -> PROmesh::EvalResult
    {
        thread_local std::unique_ptr<PROmetric> tls_metric;
        if (!tls_metric) tls_metric.reset(proto->Clone());
        return PROmesh::pinned_scan_eval(*tls_metric, fitconfig, loc_x_idx, loc_y_idx, req);
    };

    log<LOG_INFO>(L"%1% || PROsurf::FillSurfaceAMR starting AMR on [%2%, %3%] × [%4%, %5%], initial=%6%×%7%, levels=%8%, nthreads=%9%.")
        % __func__ % x_lo % x_hi % y_lo % y_hi
        % opts.initial_nx % opts.initial_ny % opts.max_levels % opts.nthreads;

    PROmesh::AMRResult amr = PROmesh::run_amr(eval_fn, x_lo, x_hi, y_lo, y_hi, opts);

    // Copy reconstructed dense matrix into PROsurf::surface for plot-compat.
    if (opts.produce_dense && amr.reconstructed_dense.size() > 0) {
        // PROsurf::surface is (nbinsx, nbinsy) — surface(i, j) where i is x-bin, j is y-bin.
        // amr.reconstructed_dense is (dense_ny rows × dense_nx cols) with row=y, col=x.
        const int rx = std::min((int)nbinsx, (int)amr.reconstructed_dense.cols());
        const int ry = std::min((int)nbinsy, (int)amr.reconstructed_dense.rows());
        for (int ix = 0; ix < rx; ++ix) {
            for (int iy = 0; iy < ry; ++iy) {
                surface(ix, iy) = amr.reconstructed_dense(iy, ix);
            }
        }
    }

    // Populate results vector for the TTree dump. One row per AMR-evaluated
    // grid point. (binx, biny) are quantised to the dense (nbinsx, nbinsy) grid.
    results.clear();
    const int finest_step = 1 << std::max(0, std::min(opts.max_levels, 7));
    const int finest_nx   = opts.initial_nx * finest_step;
    const int finest_ny   = opts.initial_ny * finest_step;
    for (const auto &kv : amr.chi2_map) {
        const int i = (int)((kv.first >> 32) & 0xffffffffu);
        const int j = (int)( kv.first        & 0xffffffffu);
        const int binx = std::min((int)nbinsx - 1, std::max(0, (int)((long)i * (long)nbinsx / std::max(1, finest_nx))));
        const int biny = std::min((int)nbinsy - 1, std::max(0, (int)((long)j * (long)nbinsy / std::max(1, finest_ny))));
        Eigen::VectorXf bf;
        auto bf_it = amr.bestfit_map.find(kv.first);
        if (bf_it != amr.bestfit_map.end()) bf = bf_it->second;
        results.push_back({binx, biny, bf, kv.second - amr.min_chi2});
    }

    // Sparse evaluation dump.
    if (!filename.empty()) {
        std::ofstream f(filename);
        f << "# AMR sparse evaluation map\n";
        f << "# x_phys y_phys delta_chi2 chi2_abs\n";
        f << "# total_fits=" << amr.total_fits
          << " min_chi2=" << amr.min_chi2 << "\n";
        for (const auto &kv : amr.chi2_map) {
            const int i = (int)((kv.first >> 32) & 0xffffffffu);
            const int j = (int)( kv.first        & 0xffffffffu);
            const float xp = x_lo + (float)i / (float)finest_nx * (x_hi - x_lo);
            const float yp = y_lo + (float)j / (float)finest_ny * (y_hi - y_lo);
            f << xp << " " << yp << " " << (kv.second - amr.min_chi2) << " " << kv.second << "\n";
        }
    }

    log<LOG_ERROR>(L"%1% || AMR done: %2% fits across %3% levels, min_chi2=%4%, %5% contour levels.")
        % __func__ % amr.total_fits % opts.max_levels % amr.min_chi2 % amr.polylines.size();

    return amr;
}


void PROsurf::PlotAMRMesh(const PROmesh::AMRResult &amr,
                          const PROmodel &model,
                          std::string filename,
                          bool logx, bool logy,
                          size_t xaxis_idx, size_t yaxis_idx)
{
    if (amr.leaves.empty() || amr.finest_nx <= 0 || amr.finest_ny <= 0) {
        log<LOG_WARNING>(L"%1% || PlotAMRMesh: no leaves or invalid grid; skipping.") % __func__;
        return;
    }

    // Find the deepest refinement level for the log message.
    int max_lvl = 0;
    for (const auto &leaf : amr.leaves) max_lvl = std::max(max_lvl, leaf.level);

    // Delegate the rendering to the shared drawer (inc/PROmeshPlot.h).
    TCanvas c("amr_mesh", "AMR Mesh", 800, 700);
    PROmesh::draw_amr_mesh_on_canvas(c, amr, model, logx, logy, xaxis_idx, yaxis_idx);
    drawVersionWatermark(&c);
    c.Print((filename + "_amr_mesh.pdf").c_str());

    log<LOG_INFO>(L"%1% || AMR mesh plot written to %2%_amr_mesh.pdf (%3% leaves, max level %4%).")
        % __func__ % filename.c_str() % (int)amr.leaves.size() % max_lvl;
}


void PROsurf::PlotCurve(const PROconfig &config, const PROmodel &model, const PROsyst &syst, const std::vector<surfOut> & cpoints, std::string final_output_tag, bool logx, bool logy, size_t xaxis_idx,size_t yaxis_idx, std::vector<float> &A, std::vector<float> &B, [[maybe_unused]] size_t n_points){

    std::vector<float> binedges_x, binedges_y;
    // Edges are stored in model's native space (log if is_log10, linear otherwise)
    // Convert to linear for ROOT histogram bin edges
    for(size_t i = 0; i < this->nbinsx+1; i++)
        binedges_x.push_back(model.is_log10[xaxis_idx] ? std::pow(10, this->edges_x(i)) : this->edges_x(i));
    for(size_t i = 0; i < this->nbinsy+1; i++)
        binedges_y.push_back(model.is_log10[yaxis_idx] ? std::pow(10, this->edges_y(i)) : this->edges_y(i));

     std::string xlabel = xaxis_idx < model.nparams ? model.pretty_param_names.at(xaxis_idx) : 
            config.m_mcgen_variation_plotname_map.at(syst.spline_names.at(xaxis_idx));
     std::string ylabel = yaxis_idx < model.nparams ? model.pretty_param_names.at(yaxis_idx) : 
            config.m_mcgen_variation_plotname_map.at(syst.spline_names.at(yaxis_idx));
    TH2D surf("surf", (";"+xlabel+";"+ylabel).c_str(), this->nbinsx, binedges_x.data(), this->nbinsy, binedges_y.data());

    TCanvas c1("c1", "PROcurve Analysis", 1600, 800);
    TPad *p1 = new TPad("p1", "left", 0.0, 0.0, 0.3, 1.0);  // 30% width
    TPad *p2 = new TPad("p2", "right", 0.3, 0.0, 1.0, 1.0); // 70% width
    p1->Draw();
    p2->Draw();


    p1->cd();
    p1->SetLeftMargin(0.12);
    p1->SetRightMargin(0.15);
    if(logx) p1->SetLogx();
    if(logy) p1->SetLogy();
    surf.Draw("COLZ");
    // Points are stored in the model's native space (log10 if is_log10, linear otherwise);
    // convert to the linear plot space only for axes that are actually log10.
    auto to_plot_x = [&](float val){ return model.is_log10[xaxis_idx] ? std::pow(10.0f, val) : val; };
    auto to_plot_y = [&](float val){ return model.is_log10[yaxis_idx] ? std::pow(10.0f, val) : val; };
    TMarker *markerA = new TMarker(to_plot_x(A[xaxis_idx]), to_plot_y(A[yaxis_idx]), 29);
    markerA->SetMarkerColor(kBlack);
    markerA->SetMarkerSize(3);
    markerA->Draw();
    TMarker *markerB = new TMarker(to_plot_x(B[xaxis_idx]), to_plot_y(B[yaxis_idx]), 29);
    markerB->SetMarkerColor(kBlack);
    markerB->SetMarkerSize(3);
    markerB->Draw();


    TArrow *arrow = new TArrow(to_plot_x(A[xaxis_idx]), to_plot_y(A[yaxis_idx]), to_plot_x(B[xaxis_idx]), to_plot_y(B[yaxis_idx]),0.01, "|>");
    arrow->SetLineStyle(2);  // Dashed
    arrow->SetLineWidth(2);
    arrow->SetLineColor(kBlack);
    arrow->SetFillColor(kBlack);  
    arrow->Draw();

    TPaveText *textbox = new TPaveText(0.18, 0.75, 0.55, 0.92, "NDC"); 
    textbox->SetFillColor(kWhite);
    textbox->SetBorderSize(1);
    textbox->SetTextAlign(12);  // Left align
    textbox->SetTextSize(0.04);

    textbox->AddText(("Start: " + xlabel + " = " + to_string_prec(to_plot_x(A[xaxis_idx]),3)).c_str());
    textbox->AddText(("       " + ylabel + " = " + to_string_prec(to_plot_y(A[yaxis_idx]),3)).c_str());
    textbox->AddText("");  // Blank line
    textbox->AddText(("End:   " + xlabel + " = " + to_string_prec(to_plot_x(B[xaxis_idx]),3)).c_str());
    textbox->AddText(("       " + ylabel + " = " + to_string_prec(to_plot_y(B[yaxis_idx]),3)).c_str());
    textbox->Draw();

    p1->RedrawAxis();


    p2->cd();
    p2->SetLeftMargin(0.12);
    p2->SetRightMargin(0.05);

    size_t nparams =  syst.GetNSplines();
    //Eigen::VectorXf subvector2 = param.segment();

    TMultiGraph *mg = new TMultiGraph();
    TLegend *leg = new TLegend(0.18,0.69,0.89,0.89);
    int colors[] = {kBlack, kRed, kBlue, kGreen+2, kMagenta, kOrange, kCyan+2, kViolet, kYellow+2, kPink};

    double ymin = 1e10, ymax = -1e10;

    for(size_t iparam = 0; iparam < nparams; iparam++) {
        std::vector<double> x_points, y_points;

        log<LOG_INFO>(L"%1% || Curve plot  %2% ") % __func__ % iparam;

        for(size_t i = 0; i < cpoints.size(); i++) {
            x_points.push_back(i);
            float val = cpoints[i].best_fit(iparam+model.nparams);
            y_points.push_back(val);
            if(val < ymin) ymin = val;
            if(val > ymax) ymax = val;

        }
        log<LOG_INFO>(L"Parameter %1%: min=%2%, max=%3%") % iparam % *std::min_element(y_points.begin(), y_points.end()) % *std::max_element(y_points.begin(), y_points.end());

        TGraph *gr = new TGraph(x_points.size(), x_points.data(), y_points.data());
        gr->SetLineColor(colors[iparam % 10]);
        gr->SetLineWidth(2);
        gr->SetMarkerColor(colors[iparam % 10]);
        gr->SetMarkerStyle(20);
        gr->SetMarkerSize(0.8);

        mg->Add(gr, "LP");  

        std::string param_name = config.m_mcgen_variation_plotname_map.at(syst.spline_names[iparam]);
        leg->AddEntry(gr, param_name.c_str(), "lp");
    }

    mg->Draw("A");
    double margin = (ymax - ymin) * 0.05;
    mg->GetYaxis()->SetRangeUser(ymin - margin, ymax + 5*margin);

    mg->GetXaxis()->SetTitle("Curve Point Index");
    mg->GetYaxis()->SetTitle("Parameter Value");
    mg->SetTitle("Parameter Evolution Along Curve");

    leg->SetNColumns(4);
    leg->SetFillStyle(0);
    leg->SetLineWidth(0);
    leg->Draw();

    drawVersionWatermark(&c1, WatermarkPos::BottomRight);
    c1.SaveAs((final_output_tag+"_PROcurve.pdf").c_str(), "pdf");

}



std::vector<float> findMinAndBounds(TGraph *g, float val, float lo, float hi) {
    // Walk resolution for the crossing search. This directly floors the
    // precision of the reported sigma boundaries AND its cost is (hi-lo)/step
    // TGraph::Eval calls per side, so scale with the range instead of walking
    // a wide axis at fixed 1e-3.
    constexpr float kMinWalkStep = 0.001f;
    const float step = std::max(kMinWalkStep, (hi - lo) / 10000.0f);
    int n = g->GetN();
    float minY = 1e9, minX = 0;
    for (int i = 0; i < n; ++i) {
        double x, y;
        g->GetPoint(i, x,y);
        if (y < minY) {
            minY = y;
            minX = x;
        }
    }
    //..ok so minX is the min and Currentl minY is the chi^2. Want this to be delta chi^2

    float leftX = minX, rightX = minX;

    // Search to the left of the minimum
    for (float x = minX; x >= lo; x -= step) {
        float y = g->Eval(x) - minY; //DeltaChi^2
        if (y >= val) {
            leftX = x;
            break;
        } else if(x - step < lo) {
            // If at end of loop and haven't found left side
            leftX = lo;
        }
    }

    // Search to the right of the minimum
    for (float x = minX; x <= hi; x += step) {
        float y = g->Eval(x)-minY;
        if (y >= val) {
            rightX = x;
            break;
        } else if(x + step > hi) {
            // If at end of loop and haven't found right side
            rightX = hi;
        }
    }

    return {minX,leftX,rightX};
}


PROfile::PROfile(const PROconfig &config, const PROsyst &systs, const PROmodel &model, PROmetric &metric, PROseed &proseed, const PROfitterConfig &fitconfig, [[maybe_unused]] std::string filename, float minchi, bool with_osc, int nThreads, const std::vector<Eigen::VectorXf> &seed_points, [[maybe_unused]] const Eigen::VectorXf & true_params, bool use_probe, int n_physics_chunks, const std::vector<FixedSeed> &fixed_seeds) : metric(metric) {
    // Copy into the member before any worker thread launches; workers read it
    // through ScanFitContext::fixed_seeds.
    fixed_seed_points = fixed_seeds;
    LBFGSpp::LBFGSBSolver<float> solver(fitconfig.param);
    int nparams = systs.GetNSplines() + model.nparams*with_osc;
    std::vector<float> physics_params; 

    //hack
    std::vector<float> priorX;
    std::vector<float> priorY;

    for(int i=0; i<=30;i++){
        float which_value = -3.0+0.2*i;
        priorX.push_back(which_value);
        priorY.push_back(which_value*which_value);

    }
    std::unique_ptr<TGraph> gprior = std::make_unique<TGraph>(priorX.size(), priorX.data(), priorY.data());

    std::vector<std::string> names;
    if(with_osc) for(const auto& name: model.pretty_param_names) names.push_back(name);
    for(const auto &name: systs.spline_names) names.push_back(name);

    int loopSize = nparams;
    if(loopSize == 0){
        // No scanned parameters at all (e.g. syst-only with zero splines):
        // nothing to profile; bail before the thread setup below divides by /
        // clamps to a zero thread count.
        log<LOG_WARNING>(L"%1% || No parameters to profile (nparams == 0); returning empty PROfile.") % __func__;
        return;
    }
    if(nThreads>loopSize){
        nThreads = loopSize;
        log<LOG_INFO>(L"%1% || nThreads is < loopSize (nparams) : %2% <  %3%. Setting equal ") % __func__ % nThreads % loopSize ;
    }

    std::vector<std::future<std::vector<profOut>>> futures;

    // Cross-thread warm-start bank: one slot per full-vector parameter, filled by
    // every completed scan-point fit and queried for the nearest-in-value seed.
    // Must be sized before any worker thread launches.
    seed_bank.assign(model.nparams + systs.GetNSplines(), {});

    // Live global-minimum tracker: starts at the caller's global fit; a scan fit
    // that beats it re-seeds all subsequent fits (of every parameter) from the
    // improved point. Must be initialised before any worker thread launches.
    global_min_tracker.init(minchi, seed_points.empty() ? Eigen::VectorXf{} : seed_points.front());

    // Resolve dispatcher state up-front so the progress-bar config knows per-bar totals.
    const int nparams_helper = (int)model.nparams + (int)systs.GetNSplines();
    const int loop_start_helper = with_osc ? 0 : (int)model.nparams;
    const int n_scanned = nparams_helper - loop_start_helper;

    // Resolve n_physics_chunks:
    //   - sentinel ≤ 0    → 1 (no chunking; opt-in only).
    //   - explicit  ≥ 1   → respected.
    // Then hard-cap at nthreads (chunks > nthreads adds total CPU work without
    // any parallel speedup).
    //
    // Default is 1 because chunking only wins wall time when (a) physics is the
    // bottleneck and (b) there are spare threads beyond what nuisances need.
    // For typical fits (nuisances dominate) chunking adds CPU work for little
    // benefit. Users can set --probe-chunks N when they know physics is slow.
    const int n_phys = (int)model.nparams;
    int chunks_resolved;
    if (n_physics_chunks <= 0) {
        chunks_resolved = 1;
    } else {
        chunks_resolved = n_physics_chunks;
    }
    const int chunk_cap = std::max(1, nThreads);
    if (chunks_resolved > chunk_cap) {
        log<LOG_WARNING>(L"%1% || --probe-chunks=%2% exceeds nthreads=%3%; clamping to %3% (extra chunks would add CPU work with no parallel benefit).")
            % __func__ % chunks_resolved % chunk_cap;
        chunks_resolved = chunk_cap;
    }
    const int chunks = (use_probe && chunks_resolved > 1) ? chunks_resolved : 1;
    log<LOG_INFO>(L"%1% || PRObe chunking: requested=%2%, resolved=%3% (n_phys=%4%, nthreads=%5%)")
        % __func__ % n_physics_chunks % chunks % n_phys % nThreads;

    // Build per-bar configs. Bars are indexed [0, n_scanned); bar k corresponds
    // to helper-space param index (k + loop_start_helper). In syst-only mode we
    // skip the physics indices entirely (no bars for parameters that aren't scanned).
    // Per-bar total = expected total fits across all chunks for that param:
    //   - legacy 18-uniform: 18 fits per scan, 1 chunk → 18
    //   - PRObe smooth (nuisance): ~11 fits per scan, 1 chunk → 11
    //   - PRObe spiky (physics): up to opts.max_fits=25 fits per scan, × n_chunks
    std::vector<std::pair<int, std::string>> prof_PB_configs;
    prof_PB_configs.reserve(n_scanned);
    for (int b = 0; b < n_scanned; ++b) {
        const int helper_idx = b + loop_start_helper;
        const bool isphys = helper_idx < (int)model.nparams;
        std::string nam;
        if (isphys) {
            nam = metric.GetModel().pretty_param_names[helper_idx];
        } else {
            const int sidx = helper_idx - (int)model.nparams;
            const std::string raw = metric.GetSysts().spline_names[sidx];
            auto it = config.m_mcgen_variation_plotname_map.find(raw);
            nam = (it != config.m_mcgen_variation_plotname_map.end()) ? it->second : raw;
        }
        int per_chunk_fits;
        if (use_probe) {
            per_chunk_fits = isphys ? 25 : 13; // smooth ≈ 11, give a hair of headroom; spiky cap = max_fits
        } else {
            per_chunk_fits = 19;               // legacy 18-uniform + a slot for the fixed-bound case
        }
        const int n_chunks_for_param = (use_probe && isphys) ? chunks : 1;
        const int total_for_bar = std::max(1, per_chunk_fits * n_chunks_for_param);
        prof_PB_configs.push_back({total_for_bar, nam});
    }

    MultiPROgressBar prof_progress(prof_PB_configs);
    if(fitconfig.progress_bar){
        prof_progress.initialize_display();
        prof_progress.start_display_thread();
    }

    std::vector<ScanTask> tasks;
    for (int i = loop_start_helper; i < nparams_helper; ++i) {
        const bool isphys = i < (int)model.nparams;
        float pl, pu;
        if (isphys) {
            pl = model.lb(i);
            pu = model.ub(i);
        } else {
            // Match buildScanBounds: a restrict range, when present, IS the
            // spline's scan range (previously tasks used the unrestricted
            // lo/hi, silently overriding the restrict on the scanned axis).
            int s = i - (int)model.nparams;
            pl = systs.spline_has_restrict[s] ? systs.spline_restrict_lo[s] : systs.spline_lo[s];
            pu = systs.spline_has_restrict[s] ? systs.spline_restrict_hi[s] : systs.spline_hi[s];
        }
        if (use_probe && isphys && chunks > 1) {
            // Substitute finite ±3 for ±inf so the split arithmetic is sane.
            const float plf = finite_or(pl, -3.0f);
            const float puf = finite_or(pu,  3.0f);
            for (int k = 0; k < chunks; ++k) {
                ScanTask t;
                t.param_idx = i;
                t.sub_lb = plf + (puf - plf) * float(k)     / float(chunks);
                t.sub_ub = plf + (puf - plf) * float(k + 1) / float(chunks);
                tasks.push_back(t);
            }
        } else {
            tasks.push_back({i, pl, pu});
        }
    }

    log<LOG_INFO>(L"%1% || Starting THREADS  : %2% , Loops %3%, Tasks %4% (probe=%5%, chunks=%6%)")
        % __func__ % nThreads % loopSize % (int)tasks.size() % (int)use_probe % chunks;

    // Per-bar task tracker: count how many tasks share each bar (=1 for non-physics
    // and unchunked physics, =chunks for chunked physics). The helper decrements
    // its bar's counter when a task scan completes; the thread that takes it to
    // zero calls complete_bar() so the bar lands on 100% as that param finishes.
    // Heap-allocated array because std::atomic isn't movable (vectors won't hold them).
    std::unique_ptr<std::atomic<int>[]> tasks_remaining(new std::atomic<int>[std::max(1, n_scanned)]);
    for (int i = 0; i < n_scanned; ++i) tasks_remaining[i].store(0);
    for (const auto& tk : tasks) {
        const int b = tk.param_idx - loop_start_helper;
        if (b >= 0 && b < n_scanned) tasks_remaining[b].fetch_add(1);
    }

    // Dynamic dispatch: workers pull the next available task from this shared
    // atomic counter. Each task runs a single PRObe / 18-uniform scan to
    // completion on one thread (so the seed pool stays local), but the *choice*
    // of which task is next is made when the thread is free.
    std::atomic<int> task_counter{0};

    // Optional scan-mode timing diagnostics. When enabled (via --profile-timing
    // -> SetScanTimingEnabled(true)), PROfitter records per-phase microseconds
    // and we track per-thread wall time here. The summary is logged at the end
    // of the constructor.
    const bool tim_on = PROfit::GetScanTimingEnabled();
    if (tim_on) {
        PROfit::GetScanTimingStats().reset();
    }
    std::atomic<uint64_t> max_thread_wall_us{0};
    std::atomic<uint64_t>* max_wall_ptr = tim_on ? &max_thread_wall_us : nullptr;
    const auto dispatch_t0 = std::chrono::steady_clock::now();

    for (int t = 0; t < nThreads; ++t) {
        std::atomic<int>* tasks_remaining_raw = tasks_remaining.get();
        futures.emplace_back(std::async(std::launch::async, [&, t, tasks_remaining_raw, max_wall_ptr]() {
                    return this->PROfilePointHelper(&systs, fitconfig, &task_counter, &tasks, minchi, with_osc, std::ref(prof_progress), seed_points, proseed.getThreadSeeds()->at(t), use_probe, tasks_remaining_raw, loop_start_helper, max_wall_ptr);
                    }));

    }

    std::vector<profOut> combinedResults(nparams);
    for (auto& fut : futures) {
        std::vector<profOut> result = fut.get();
        for (auto& p : result) {
            // Map helper-space index (full model+spline vector) → constructor-space slot.
            // In with_osc, slot == param_idx. In syst-only, the constructor's vector is
            // sized = nsplines, so spline index k in helper space lands at slot k - nphys.
            int dst_idx = with_osc ? p.param_idx : p.param_idx - (int)model.nparams;
            if (dst_idx < 0 || dst_idx >= (int)nparams) continue;
            // Concatenate: chunked physics scans may produce multiple profOuts
            // sharing the same param_idx — merge their points into one record.
            auto& dst = combinedResults[dst_idx];
            if (dst.knob_vals.empty()) {
                dst = std::move(p);
            } else {
                dst.knob_vals.insert(dst.knob_vals.end(), p.knob_vals.begin(), p.knob_vals.end());
                dst.knob_chis.insert(dst.knob_chis.end(), p.knob_chis.begin(), p.knob_chis.end());
                for (auto& v : p.knob_bfs) dst.knob_bfs.push_back(std::move(v));
            }
        }
    }
    // After concatenation each profOut is unsorted by knob_val; sort now so the
    // downstream TGraph / findMinAndBounds see a monotonic curve.
    for (auto& po : combinedResults) {
        if (!po.knob_vals.empty()) po.sort();
    }
    if(fitconfig.progress_bar) prof_progress.finish_all();

    // ------------------------------------------------------------------
    // Re-baseline the curves to the true minimum found by the scan.
    // knob_chis were collected as (chi2 - minchi) against the CALLER's global
    // fit; if any scan fit found a lower minimum (the global fit was trapped
    // in a local one — the live global_min_tracker will already have propagated
    // it into later fits' seeds), those deltas are negative and every
    // downstream consumer breaks: the Delta-chi2 plots clip at zero with
    // curves diving below the axis, and the points file's deltas reference the
    // wrong minimum. Shift every curve so Delta-chi2 is measured from the
    // actual minimum, and publish a genuinely-lower minimum through
    // newglob/newglob_param for the caller to adopt.
    float updated_minchi = minchi;
    {
        float min_delta = 0.0f;
        const Eigen::VectorXf *min_bf = nullptr;
        for(const auto &out : combinedResults)
            for(size_t u = 0; u < out.knob_chis.size(); ++u)
                if(out.knob_chis[u] < min_delta && out.knob_bfs[u].size() > 0) {
                    min_delta = out.knob_chis[u];
                    min_bf = &out.knob_bfs[u];
                }
        if(min_bf) {
            const float chidiff = -min_delta;
            // Classify how different the new minimum really is: same point up
            // to float noise, degenerate phase space, or a genuinely lower
            // minimum. Only the last is published as a new global best fit;
            // the curves are re-baselined in all three cases so the plotted
            // Delta-chi2 never dips below zero.
            float bfnorm_without_phys = seed_points.size() ? seed_points.front().tail(seed_points.front().size() - model.nparams).norm() : 1;
            Eigen::VectorXf param_diff = seed_points.size() ? *min_bf - seed_points.front() : *min_bf;
            float norm_without_phys = param_diff.tail(param_diff.size() - model.nparams).norm();
            const float relative_tol1 = 1e-3f;
            if( (norm_without_phys < relative_tol1*bfnorm_without_phys)  &&  (chidiff < relative_tol1*minchi) ){
                log<LOG_WARNING>(L"%1% || Warning. A lower global best fit was found during PROfile, but less than relatively 1e-3f from global best fit (%2%), and pull parameters norm less than 1e-3 (%3%) from best_fit nuisence values. AKA same point.") % __func__ % float(chidiff/minchi) % float(norm_without_phys/bfnorm_without_phys);
            } else if(chidiff < relative_tol1*minchi) {
                log<LOG_WARNING>(L"%1% || Warning. A lower global best fit was found during PROfile, but less than relatively 1e-3f from global best fit (%2%), although pull parameters norm more than 1e-3 (%3%) from best_fit nuisence values. Not uncommon in degenerate phase space.") % __func__ % float(chidiff/minchi) % float(norm_without_phys/bfnorm_without_phys);
            } else {
                log<LOG_WARNING>(L"%1% || Warning. A lower global best fit was found during PROfile. Difference greater than relative 1e-3f from global best fit (%2%). Rel Difference pull parameters norm (%3%). Adopting it as the global best fit (newglob/newglob_param).") % __func__ % float(chidiff/minchi) % float(norm_without_phys/bfnorm_without_phys);
                log<LOG_WARNING>(L"%1% || -- minchi %2% and chidiff %3% ") % __func__ % minchi % chidiff;
                newglob = minchi + min_delta;
                newglob_param = *min_bf;
            }
            updated_minchi = minchi + min_delta;
            for(auto &out : combinedResults)
                for(auto &c : out.knob_chis) c -= min_delta;
            log<LOG_WARNING>(L"%1% || Re-baselined all profile curves to the scan minimum chi2 %2% (caller's global fit gave %3%); Delta-chi2 = 0 now sits at the true minimum.") % __func__ % updated_minchi % minchi;
        }
    }

    // Scan-mode timing summary. Logs at LOG_ERROR level so it shows up at any
    // verbosity. Emits parallelism efficiency = total CPU fit-time / (wall × nthreads):
    // close to 1.0 means all threads stayed busy; far below 1.0 means tail imbalance.
    if (tim_on) {
        const uint64_t dispatch_wall_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - dispatch_t0).count();
        const auto& s = PROfit::GetScanTimingStats();
        const uint64_t n_fits        = s.n_fits.load();
        const uint64_t total_fit_us  = s.total_fit_us.load();
        const uint64_t latin_us      = s.latin_us.load();
        const uint64_t pso_us        = s.pso_us.load();
        const uint64_t lbfgs_us      = s.lbfgs_us.load();
        const uint64_t max_wall_us   = max_thread_wall_us.load();

        const double per_fit_ms      = n_fits ? (double)total_fit_us / (double)n_fits / 1000.0 : 0.0;
        const double total_fit_s     = (double)total_fit_us / 1e6;
        const double dispatch_wall_s = (double)dispatch_wall_us / 1e6;
        const double max_wall_s      = (double)max_wall_us / 1e6;
        const double parallel_eff    = (max_wall_us > 0)
            ? (double)total_fit_us / ((double)max_wall_us * (double)nThreads)
            : 0.0;
        const double latin_frac      = total_fit_us ? (double)latin_us / (double)total_fit_us : 0.0;
        const double pso_frac        = total_fit_us ? (double)pso_us   / (double)total_fit_us : 0.0;
        const double lbfgs_frac      = total_fit_us ? (double)lbfgs_us / (double)total_fit_us : 0.0;
        const char* mode             = use_probe ? (chunks > 1 ? "PRObe-chunked" : "PRObe") : "legacy-18uniform";

        log<LOG_ERROR>(L"PROfile timing summary [%1%]:") % mode;
        log<LOG_ERROR>(L"  total fits:           %1%")   % (uint64_t)n_fits;
        log<LOG_ERROR>(L"  total tasks:          %1%")   % (int)tasks.size();
        log<LOG_ERROR>(L"  total fit-time:       %1% s") % total_fit_s;
        log<LOG_ERROR>(L"  per-fit avg:          %1% ms")% per_fit_ms;
        log<LOG_ERROR>(L"  dispatch wall:        %1% s") % dispatch_wall_s;
        log<LOG_ERROR>(L"  max thread wall:      %1% s") % max_wall_s;
        log<LOG_ERROR>(L"  nthreads:             %1%")   % nThreads;
        log<LOG_ERROR>(L"  parallel efficiency:  %1%   (1.0 = perfect, <0.7 = tail imbalance)") % parallel_eff;
        log<LOG_ERROR>(L"  latin frac of fit:    %1%")   % latin_frac;
        log<LOG_ERROR>(L"  pso frac of fit:      %1%")   % pso_frac;
        log<LOG_ERROR>(L"  lbfgs frac of fit:    %1%")   % lbfgs_frac;
    }

    {
    std::string outname = filename.empty()
        ? "PROfile_points.txt"
        : filename + "_points.txt";

    std::ofstream prof_file(outname);

    prof_file << "# PROfile scan points written by PROfit\n";
    prof_file << "# This file contains the profile curves shown in PROfile.pdf\n";
    prof_file << "# columns: param_index param_type param_name fixed_value delta_chi2 chi_post\n";
    prof_file << "# delta_chi2 baseline (global minimum chi2): " << std::setprecision(12) << updated_minchi;
    if(updated_minchi != minchi)
        prof_file << "  (scan found a lower minimum than the caller's global fit chi2 = " << minchi << ")";
    prof_file << "\n";

    size_t n_model_params = with_osc ? model.nparams : 0;

    for(size_t iparam = 0; iparam < combinedResults.size(); ++iparam) {
        std::string param_name = names.at(iparam);

        std::string param_type =
            (iparam < n_model_params) ? "model" : "spline";

        for(size_t j = 0; j < combinedResults[iparam].knob_vals.size(); ++j) {
            float fixed_value = combinedResults[iparam].knob_vals.at(j);
            float delta_chi2  = combinedResults[iparam].knob_chis.at(j);
            float chi_post    = delta_chi2 + updated_minchi;

            prof_file << iparam << " "
                      << param_type << " "
                      << param_name << " "
                      << std::setprecision(12) << fixed_value << " "
                      << std::setprecision(12) << delta_chi2 << " "
                      << std::setprecision(12) << chi_post
                      << "\n";
        }

        prof_file << "\n";
    }

    prof_file.close();

    log<LOG_INFO>(L"%1% || Wrote PROfile scan points to %2%")
        % __func__ % outname.c_str();
}

    //create all graphs, used directly in first setion. knob_chis are already
    //re-baselined to the scan-wide minimum above, so no curve dips below zero.
    for(auto & out: combinedResults){
        log<LOG_INFO>(L"%1% || Knob Values: %2%") % __func__ %  out.knob_vals;
        log<LOG_INFO>(L"%1% || Knob Chis: %2%") % __func__ %  out.knob_chis;
        std::unique_ptr<TGraph> g = std::make_unique<TGraph>(out.knob_vals.size(), out.knob_vals.data(), out.knob_chis.data());
        graphs.push_back(std::move(g));
    }

    //Analyze them, used in later section
    //plot 2sigma also? default no, as its messier
    bool twosig = false;

    std::vector<float> values1_errup;
    std::vector<float> values1_errdown;

    std::vector<float> barvalues_err;

    std::vector<float> values2_up;
    std::vector<float> values2_down;


    log<LOG_INFO>(L"%1% || Getting BF, +/- one sigma ranges. Is Two sigma turned on? : %2% ") % __func__ % twosig;

    size_t count = 0;
    for(auto &g:graphs){
        //if(metric->GetModel().nparams)continue;
        float lo, hi;
        if (with_osc && count < metric.GetModel().nparams) {
            lo = metric.GetModel().lb(count);
            hi = metric.GetModel().ub(count);
        } else {
            size_t spline_idx = with_osc ? count - metric.GetModel().nparams : count;
            lo = metric.GetSysts().spline_has_restrict[spline_idx] ? metric.GetSysts().spline_restrict_lo[spline_idx] : metric.GetSysts().spline_lo[spline_idx];
            hi = metric.GetSysts().spline_has_restrict[spline_idx] ? metric.GetSysts().spline_restrict_hi[spline_idx] : metric.GetSysts().spline_hi[spline_idx];
        }
        if(std::isinf(lo)) lo = lo < 0 ? -5 : 5;
        if(std::isinf(hi)) hi = hi < 0 ? -5 : 5;
        std::vector<float> tmp = findMinAndBounds(g.get(),1.0, lo, hi);
        barvalues.push_back(float(count)+0.5);
        barvalues_err.push_back(0.3);
        bfvalues.push_back(tmp[0]);
        values1_down.push_back(tmp[1]);
        values1_up.push_back(tmp[2]);
        values1_errdown.push_back(abs(tmp[1]-tmp[0]));
        values1_errup.push_back(abs(tmp[2]-tmp[0]));
        log<LOG_DEBUG>(L"%1% || Results of findMinAndBounds : %2% %3% %4% ") % __func__ % tmp[0] % tmp[1] % tmp[2];
        log<LOG_DEBUG>(L"%1% || Barvalues : %2% %3% %4% %5%") % __func__ % count % barvalues[count] % barvalues_err[count] % barvalues_err[count];
        log<LOG_DEBUG>(L"%1% || Bfvalues : %2% %3% ") % __func__ % count % bfvalues[count];
        log<LOG_DEBUG>(L"%1% || RangeValues : %2% %3% %4% ") % __func__ % count % values1_down[count] % values1_up[count];
        log<LOG_DEBUG>(L"%1% || ErrValues : %2% %3% %4% ") % __func__ % count % values1_errdown[count] % values1_errup[count];

        // Greppable per-parameter best-fit and profiled 1-sigma interval (raw names
        // and values, in scan space -- physics params are log10), so the contents of
        // _1sigma_detailed.pdf can be reproduced outside PROfit. Tag: [PROfile1sig]
        std::string pname;
        const char *ptype;
        if (with_osc && count < metric.GetModel().nparams) {
            pname = metric.GetModel().param_names[count];
            ptype = "phys";
        } else {
            pname = metric.GetSysts().spline_names[with_osc ? count - metric.GetModel().nparams : count];
            ptype = "syst";
        }
        log<LOG_INFO>(L"%1% || [PROfile1sig] type=%2% param=%3% BF=%4% lo=%5% hi=%6% errlo=%7% errhi=%8%")
            % __func__ % ptype % pname.c_str() % tmp[0] % tmp[1] % tmp[2]
            % std::abs(tmp[0] - tmp[1]) % std::abs(tmp[2] - tmp[0]);
        if(twosig){
            std::vector<float> tmp2 = findMinAndBounds(g.get(),4.0,lo, hi);
            values2_down.push_back(abs(tmp2[1]-tmp[0]));
            values2_up.push_back(abs(tmp2[2]-tmp[0]));
        }
        count++;
    }
    onesig = TGraphAsymmErrors(barvalues.size(),barvalues.data(), bfvalues.data(), barvalues_err.data(), barvalues_err.data(), values1_errdown.data(), values1_errup.data());

}

void PROfile::Plot(const PROconfig &config, const PROsyst &systs, const PROmodel &model, [[maybe_unused]] PROmetric &metric, [[maybe_unused]] PROseed &proseed, std::string filename, bool with_osc, const Eigen::VectorXf& init_seed, const Eigen::VectorXf & true_params, const Eigen::MatrixXf& spline_covariance, const Eigen::VectorXf& param_err_lo, const Eigen::VectorXf& param_err_hi, bool mask_osc) {

    int nparams = systs.GetNSplines() + model.nparams*with_osc;
    int nBins = nparams;
    std::vector<std::string> names;
    if(with_osc) for(const auto& name: model.pretty_param_names) names.push_back(name);
    for(const auto &name: systs.spline_names) names.push_back(name);

    std::vector<float> priorX;
    std::vector<float> priorY;

    for(int i=0; i<=30;i++){
        float which_value = -3.0+0.2*i;
        priorX.push_back(which_value);
        priorY.push_back(which_value*which_value);

    }
    std::unique_ptr<TGraph> gprior = std::make_unique<TGraph>(priorX.size(), priorX.data(), priorY.data());

    //First plot - use multi-page PDF with 4 columns x 3 rows per page
    const int nCols = 4;
    const int nRows = 3;
    const int plotsPerPage = nCols * nRows;

    TCanvas *c = new TCanvas(filename.c_str(), filename.c_str(), 350*nCols, 350*nRows);
    c->Divide(nCols, nRows);

    // Open multi-page PDF
    c->Print((filename+".pdf[").c_str());

    // First, collect all plots to draw (main plots + zoomed physics plots)
    std::vector<std::function<void()>> plotFunctions;

    for(size_t w = 0; w < graphs.size(); w++) {
        if(mask_osc && w < model.nparams) continue;

        // Capture w by value for the lambda
        size_t idx = w;
        plotFunctions.push_back([&, idx]() {
            // Check if this is a physics param (only possible when with_osc=true)
            bool is_physics = with_osc && idx < model.nparams;
            std::string xval;
            if(is_physics) {
                const bool is_log_param = (idx < model.is_log10.size()) && model.is_log10[idx];
                if(is_log_param) {
                    xval = "Log_{10}(" + model.pretty_param_names[idx] + ")";
                } else {
                    xval = model.pretty_param_names[idx];
                }
            } else {
                xval = "#sigma Shift";
            }
            std::string tit = (is_physics ? names[idx] : config.m_mcgen_variation_plotname_map.at(names[idx])) + ";" + xval + "; #Delta#Chi^{2}";
            graphs[idx]->SetTitle(tit.c_str());
            // Small filled black circles at each sampled / anchored point so the
            // adaptive sampling pattern is visible alongside the interpolated curve.
            graphs[idx]->SetMarkerStyle(20);
            graphs[idx]->SetMarkerSize(0.4);
            graphs[idx]->SetMarkerColor(kBlack);
            graphs[idx]->Draw("ALP");
            graphs[idx]->SetLineWidth(1);
            graphs[idx]->GetYaxis()->SetTitleSize(0.05);
            graphs[idx]->GetYaxis()->SetLabelSize(0.04);
            graphs[idx]->GetXaxis()->SetTitleSize(0.05);
            graphs[idx]->GetXaxis()->SetLabelSize(0.04);
            graphs[idx]->GetYaxis()->SetRangeUser(0, graphs[idx]->GetHistogram()->GetMaximum());

            TLine* line = new TLine(graphs[idx]->GetXaxis()->GetXmin(), 1, graphs[idx]->GetXaxis()->GetXmax(), 1);
            line->SetLineStyle(3);
            line->SetLineWidth(1);
            line->SetLineColor(kBlack);
            line->Draw();

            // Open black circles marking the crossing-calc results: the in-pool
            // minimum (at its actual χ² value) and the ±1σ band edges (at Δχ²=1).
            // Slightly larger than sample markers so the computed band is obvious.
            const double minx_d = bfvalues[idx];
            const double min_y  = graphs[idx]->Eval(minx_d);
            TMarker* m_min = new TMarker(minx_d, min_y, 24);
            m_min->SetMarkerColor(kBlack);
            m_min->SetMarkerSize(1.0);
            m_min->Draw();
            TMarker* m_lo = new TMarker(values1_down[idx], 1.0, 24);
            m_lo->SetMarkerColor(kBlack);
            m_lo->SetMarkerSize(1.0);
            m_lo->Draw();
            TMarker* m_hi = new TMarker(values1_up[idx], 1.0, 24);
            m_hi->SetMarkerColor(kBlack);
            m_hi->SetMarkerSize(1.0);
            m_hi->Draw();

            if(is_physics) graphs[idx]->SetLineColor(kBlue-7);

            if(graphs[idx]->GetN() == 1) {
                float x_val = graphs[idx]->GetPointX(0);
                TLine* linet = new TLine(x_val, 0, x_val, 1);
                linet->SetLineStyle(1);
                linet->SetLineWidth(1);
                linet->SetLineColor(kGreen+2);
                linet->Draw();
            }
            else if(!is_physics) {
                gprior->Draw("L same");
                gprior->SetLineStyle(2);
                gprior->SetLineWidth(1);
                gprior->SetLineColor(kRed-7);
                graphs[idx]->GetYaxis()->SetRangeUser(0, std::min(graphs[idx]->GetHistogram()->GetMaximum(), 10.0));
            }
        });

        // Add zoomed plots for physics parameters after the last physics parameter (only when physics params are included)
        if(with_osc && w == model.nparams - 1) {
            for(int zs = model.nparams - 1; zs >= 0; zs--) {
                size_t zoomIdx = w - zs;
                plotFunctions.push_back([&, zoomIdx]() {
                    TGraph* graphClone = new TGraph(*graphs[zoomIdx]);
                    // Same sample-point + crossing markers as the main plot.
                    graphClone->SetMarkerStyle(20);
                    graphClone->SetMarkerSize(0.4);
                    graphClone->SetMarkerColor(kBlack);
                    graphClone->Draw("ALP");
                    std::string newTitle = std::string(graphClone->GetTitle()) + " Zoomed 1#sigma";
                    graphClone->SetTitle(newTitle.c_str());
                    graphClone->SetLineColor(kViolet);
                    graphClone->SetLineWidth(1);
                    float vd = std::min(values1_down[zoomIdx], values1_up[zoomIdx]);
                    float vu = std::max(values1_down[zoomIdx], values1_up[zoomIdx]);
                    float pd = (vd > 0 ? vd * 0.9 : vd * 1.1);
                    float pu = (vu > 0 ? vu * 1.1 : vu * 0.9);
                    graphClone->GetXaxis()->SetLimits(pd, pu);
                    graphClone->GetYaxis()->SetRangeUser(0, std::max(graphClone->Eval(pu), graphClone->Eval(pd)) * 1.1);
                    graphClone->GetYaxis()->SetTitleSize(0.05);
                    graphClone->GetYaxis()->SetLabelSize(0.04);
                    graphClone->GetXaxis()->SetTitleSize(0.05);
                    graphClone->GetXaxis()->SetLabelSize(0.04);

                    // Open black circles for the crossing-calc results.
                    const double minx_d = bfvalues[zoomIdx];
                    const double min_y  = graphClone->Eval(minx_d);
                    TMarker* zm_min = new TMarker(minx_d, min_y, 24);
                    zm_min->SetMarkerColor(kBlack);
                    zm_min->SetMarkerSize(1.0);
                    zm_min->Draw();
                    TMarker* zm_lo = new TMarker(values1_down[zoomIdx], 1.0, 24);
                    zm_lo->SetMarkerColor(kBlack);
                    zm_lo->SetMarkerSize(1.0);
                    zm_lo->Draw();
                    TMarker* zm_hi = new TMarker(values1_up[zoomIdx], 1.0, 24);
                    zm_hi->SetMarkerColor(kBlack);
                    zm_hi->SetMarkerSize(1.0);
                    zm_hi->Draw();

                    if(graphClone->GetN() == 1) {
                        float x_val = graphClone->GetPointX(0);
                        TLine* linet = new TLine(x_val, 0, x_val, 1);
                        linet->SetLineStyle(1);
                        linet->SetLineWidth(1);
                        linet->SetLineColor(kGreen+2);
                        linet->Draw();
                    }

                    log<LOG_INFO>(L"%1% || Zoom boundaries X %2% %3% Y %4% %5%  ") % __func__ % pd % pu % 0.0 % (std::max(graphClone->Eval(pu), graphClone->Eval(pd)) * 1.1);

                    TLine* line1 = new TLine(pd, 1, pu, 1);
                    line1->SetLineStyle(3);
                    line1->SetLineWidth(1);
                    line1->SetLineColor(kBlack);
                    line1->Draw();

                    TLine* line2 = new TLine(vd, graphClone->Eval(vd), vd, 0);
                    line2->SetLineStyle(3);
                    line2->SetLineWidth(1);
                    line2->SetLineColor(kBlack);
                    line2->Draw();

                    TLine* line3 = new TLine(vu, graphClone->Eval(vu), vu, 0);
                    line3->SetLineStyle(3);
                    line3->SetLineWidth(1);
                    line3->SetLineColor(kBlack);
                    line3->Draw();
                });
            }
        }
    }

    // Now draw plots page by page
    for(size_t i = 0; i < plotFunctions.size(); i++) {
        int padIdx = (i % plotsPerPage) + 1;

        // Start of a new page (except for the first plot)
        if(i > 0 && padIdx == 1) {
            drawVersionWatermark(c, WatermarkPos::RightEdge);
            c->Print((filename+".pdf").c_str());
            c->Clear();
            c->Divide(nCols, nRows);
        }

        c->cd(padIdx);
        plotFunctions[i]();
    }

    // Print the last page and close the PDF
    drawVersionWatermark(c, WatermarkPos::RightEdge);
    c->Print((filename+".pdf").c_str());
    c->Print((filename+".pdf]").c_str());

    delete c;

    // Summary bar chart - scales from 1 to dozens of parameters
    log<LOG_DEBUG>(L"%1% || Are all lines the same : %2% %3% %4% %5% %6%") % __func__ % nBins % barvalues.size() % bfvalues.size() % values1_down.size() % values1_up.size() ;

    // Y-axis range: include everything within ±5σ (band + markers), nothing beyond.
    // Clamp band extremes to ±5σ so the axis never extends past that.
    float minVal = std::max(*std::min_element(values1_down.begin(), values1_down.end()), -5.0f);
    float maxVal = std::min(*std::max_element(values1_up.begin(),   values1_up.end()),    5.0f);
    for(int i = 0; i < nBins; ++i) {
        if(mask_osc && i < (int)model.nparams) continue;
        int vec_idx = with_osc ? i : (i + model.nparams);
        // Collect all marker values for this bin
        std::vector<float> markers;
        if(i        < (int)bfvalues.size())    markers.push_back(bfvalues[i]);
        if(vec_idx  < (int)init_seed.size())   markers.push_back((float)init_seed[vec_idx]);
        if(vec_idx  < (int)true_params.size()) markers.push_back((float)true_params[vec_idx]);
        // Expand range to include each marker, but only if it's within ±5σ
        for(float v : markers) {
            if(v >= -5.0f && v <= 5.0f) {
                minVal = std::min(minVal, v);
                maxVal = std::max(maxVal, v);
            }
        }
    }

    { // _1sigma_detailed.pdf scope
    // 50px/bin keeps the aspect ratio reasonable for large parameter counts;
    // height is fixed so NDC font/line sizes stay consistent.
    int c2_width  = std::max(600, std::min(5000, 50 * nBins));
    int c2_height = 500;
    float axis_label_size = std::max(0.030f, std::min(0.045f, 1.8f / nBins));
    float x_label_size    = std::max(0.015f, std::min(0.030f, 1.2f / nBins));
    int   profile_lw      = (nBins > 20) ? 1 : 2;
    float bar_halfwidth   = std::max(0.08f, std::min(0.4f, 4.0f / nBins));
    float marker_offset   = bar_halfwidth * 0.6f;

    TCanvas *c2 = new TCanvas((filename+"1sigma_detailed").c_str(), (filename+"1sigma_detailed").c_str(), c2_width, c2_height);
    c2->cd();
    c2->SetLeftMargin(0.09);
    c2->SetBottomMargin(0.30);
    c2->SetRightMargin(0.28);
    c2->SetTopMargin(0.08);

    // Derived error vectors for profile_pts
    std::vector<float> values1_errdown, values1_errup;
    for(int i = 0; i < nBins; ++i) {
        values1_errdown.push_back(std::abs(values1_down[i] - bfvalues[i]));
        values1_errup.push_back(std::abs(values1_up[i]   - bfvalues[i]));
    }

    // Y-axis range: cover the post-fit band; always show ±1 (prior reference)
    float minVal_det = minVal;
    float maxVal_det = maxVal;
    minVal_det = std::min(minVal_det, -1.2f);
    maxVal_det = std::max(maxVal_det, 1.2f);
    float y_axis_min = minVal_det * 1.15f;
    float y_axis_max = maxVal_det * 1.15f;

    // TH1F frame to establish axes (persists for the lifetime of the canvas)
    TH1F *frame = new TH1F((filename+"_frame").c_str(), "", nBins, 0, nBins);
    frame->SetMinimum(y_axis_min);
    frame->SetMaximum(y_axis_max);
    frame->SetStats(0);
    frame->GetXaxis()->SetLabelSize(0);
    frame->GetXaxis()->SetTickLength(0);
    frame->GetYaxis()->SetTitle("Parameter value (prior #kern[0.3]{} #sigma = 1)");
    frame->GetYaxis()->SetTitleSize(axis_label_size);
    frame->GetYaxis()->SetLabelSize(axis_label_size);
    frame->GetYaxis()->SetTitleOffset(1.0);
    frame->Draw("AXIS");

    // Full-width gray band for pre-fit ±1σ (prior spans entire plot)
    TBox *prior_band = new TBox(0.0f, -1.0f, (float)nBins, 1.0f);
    prior_band->SetFillColor(kGray);
    prior_band->SetFillStyle(1001);
    prior_band->SetLineColor(kGray+1);
    prior_band->SetLineWidth(1);
    prior_band->Draw("same");

    // Post-fit ±1σ bars (blue): centered on the global best-fit, width from MCMC
    // 16th/84th percentile quantiles.  For oscillation physics parameters (no entry
    // in param_err_lo/hi), fall back to the profile interval.
    // CONTRACT with the caller (bin/PROfit.cxx profile block): init_seed is the
    // FULL parameter vector while param_err_lo/hi are indexed over the SPLINE
    // block only (post_param_lo/hi from getMCMCErrorBand). The vec_idx /
    // syst_idx arithmetic below is the only thing tying the two conventions
    // together — changing either side's indexing breaks the other silently.
    TGraphAsymmErrors todraw = onesig;
    for(int i = 0; i < nBins; ++i) {
        int vec_idx = with_osc ? i : (i + (int)model.nparams);
        float center = (vec_idx < (int)init_seed.size()) ? (float)init_seed[vec_idx] : bfvalues[i];
        int syst_idx = with_osc ? (i - (int)model.nparams) : i;
        bool is_syst = !with_osc || (i >= (int)model.nparams);
        float err_lo, err_hi;
        if(is_syst && syst_idx >= 0 && syst_idx < param_err_lo.size() && syst_idx < param_err_hi.size()) {
            err_lo = param_err_lo(syst_idx);
            err_hi = param_err_hi(syst_idx);
        } else {
            err_lo = todraw.GetErrorYlow(i);
            err_hi = todraw.GetErrorYhigh(i);
        }
        todraw.SetPoint(i, todraw.GetPointX(i), center);
        todraw.SetPointError(i, bar_halfwidth, bar_halfwidth, err_lo, err_hi);
    }
    if(mask_osc) {
        for(size_t i = 0; i < model.nparams; ++i) {
            todraw.SetPoint(i, 0, 0);
            todraw.SetPointError(i, 0, 0, 0, 0);
        }
    }
    todraw.SetFillColor(kBlue-7);
    todraw.SetFillStyle(1001);
    todraw.SetLineColor(kBlue-8);
    todraw.SetLineWidth(1);
    todraw.Draw("2 same");

    // Horizontal reference lines
    TLine l_zero(0, 0, nBins, 0);
    l_zero.SetLineStyle(2);
    l_zero.SetLineColor(kGray+2);
    l_zero.SetLineWidth(1);
    l_zero.Draw();
    TLine l_pm1(0, 1, nBins, 1);
    l_pm1.SetLineStyle(3);
    l_pm1.SetLineColor(kGray+2);
    l_pm1.SetLineWidth(1);
    l_pm1.Draw();
    TLine l_mm1(0, -1, nBins, -1);
    l_mm1.SetLineStyle(3);
    l_mm1.SetLineColor(kGray+2);
    l_mm1.SetLineWidth(1);
    l_mm1.Draw();

    // Profile scan best-fit ±1σ: black circles with error bars
    std::vector<float> zeroes_err(nBins, 0.0f);
    TGraphAsymmErrors profile_pts(nBins, barvalues.data(), bfvalues.data(),
                                   zeroes_err.data(), zeroes_err.data(),
                                   values1_errdown.data(), values1_errup.data());
    if(mask_osc) {
        for(size_t i = 0; i < model.nparams; ++i) {
            profile_pts.SetPoint(i, barvalues[i], y_axis_min - 999.0f);
            profile_pts.SetPointError(i, 0, 0, 0, 0);
        }
    }
    float marker_size = std::max(0.5f, std::min(1.4f, 6.0f / std::sqrt((float)nBins)));
    profile_pts.SetMarkerStyle(20);
    profile_pts.SetMarkerSize(marker_size);
    profile_pts.SetMarkerColor(kBlack);
    profile_pts.SetLineColor(kBlack);
    profile_pts.SetLineWidth(profile_lw);
    profile_pts.Draw("P same");

    // Arrow/marker helper for out-of-range values
    float y_range_size = y_axis_max - y_axis_min;
    float arrow_margin = y_range_size * 0.07f;
    float arrow_length = y_range_size * 0.05f;
    log<LOG_INFO>(L"%1% || _1sigma plot y-axis range: min=%2%, max=%3%") % __func__ % y_axis_min % y_axis_max;

    auto drawMarkerWithArrow2 = [&](float x, float y, int color, int marker_style, float msize) {
        bool clamp_below = y < -5.0f;
        bool clamp_above = y > 5.0f;
        if(!clamp_below && !clamp_above && (y < y_axis_min || y > y_axis_max)) return;

        float draw_y = clamp_below ? y_axis_min + arrow_margin : (clamp_above ? y_axis_max - arrow_margin : y);

        TMarker* marker = new TMarker(x, draw_y, marker_style);
        marker->SetMarkerSize(msize);
        marker->SetMarkerColor(color);
        marker->Draw();

        if(clamp_below) {
            TArrow* arr = new TArrow(x, draw_y - arrow_length * 0.3f, x, y_axis_min + arrow_length * 0.2f, 0.008, "|>");
            arr->SetLineColor(color); arr->SetFillColor(color); arr->SetLineWidth(1);
            arr->Draw();
        } else if(clamp_above) {
            TArrow* arr = new TArrow(x, draw_y + arrow_length * 0.3f, x, y_axis_max - arrow_length * 0.2f, 0.008, "|>");
            arr->SetLineColor(color); arr->SetFillColor(color); arr->SetLineWidth(1);
            arr->Draw();
        }
    };

    for(int i = 0; i < nBins; ++i) {
        if(mask_osc && i < (int)model.nparams) continue;
        int vec_idx = with_osc ? i : (i + model.nparams);
        float x_center = i + 0.5f;

        // Red square: global best-fit (init_seed)
        if(vec_idx < init_seed.size()) {
            drawMarkerWithArrow2(x_center - marker_offset, (float)init_seed[vec_idx], kRed, 21, marker_size);
        }

        // Orange diamond: injected true values (true_params)
        if(vec_idx < (int)true_params.size()) {
            log<LOG_INFO>(L"%1% || _1sigma marker i=%2%: true_params[%3%]=%4%") % __func__ % i % vec_idx % true_params[vec_idx];
            drawMarkerWithArrow2(x_center + marker_offset, (float)true_params[vec_idx], kOrange+7, 33, marker_size * 1.1f);
        }
    }

    // Parameter labels (rotated -45 degrees). Precompute labels so we can also shrink the
    // font for long names: a -45 deg label of L chars drops ~0.42*size*L in NDC height; with
    // the 0.30 bottom margin that gives size ~0.6/L. Take the smaller of that and the
    // nBins-based x_label_size so long names are not clipped off the bottom of the page.
    std::vector<std::string> labels(barvalues.size());
    size_t max_label_len = 1;
    for(size_t i = 0; i < barvalues.size(); ++i) {
        if(mask_osc && with_osc && i < model.nparams) continue;
        labels[i] = (with_osc && i < model.nparams)
            ? ("Log_{10}(" + model.pretty_param_names[i] + ")")
            : config.m_mcgen_variation_plotname_map.at(names[i]);
        max_label_len = std::max(max_label_len, labels[i].size());
    }
    float len_based_size = std::max(0.012f, std::min(0.030f, 0.6f / (float)max_label_len));
    float text_size = std::min(x_label_size, len_based_size);
    float label_y = y_axis_min - y_range_size * 0.04f;
    for(size_t i = 0; i < barvalues.size(); ++i) {
        if(mask_osc && with_osc && i < model.nparams) continue;
        std::string label;
        if(with_osc && i < model.nparams) {
            const bool is_log_param = (i < model.is_log10.size()) && model.is_log10[i];
            if(is_log_param) {
                label = "Log_{10}(" + model.pretty_param_names[i] + ")";
            } else {
                label = model.pretty_param_names[i];
            }
        } else {
            label = config.m_mcgen_variation_plotname_map.at(names[i]);
        }
        TLatex* text = new TLatex(barvalues[i], label_y, label.c_str());
        text->SetTextAlign(13);
        text->SetTextSize(text_size);
        text->SetTextAngle(-45);
        text->Draw();
    }

    // Legend (NDC coordinates, top-right)
    TLegend *leg = new TLegend(0.73, 0.55, 0.99, 0.92);
    leg->SetFillStyle(1001);
    leg->SetBorderSize(1);
    leg->SetTextSize(std::max(0.022f, std::min(0.030f, axis_label_size * 0.85f)));
    leg->AddEntry(prior_band,   "Pre-fit #pm1#sigma (prior)", "f");
    leg->AddEntry(&todraw,      "Post-fit #pm1#sigma (MCMC)", "f");
    leg->AddEntry(&profile_pts, "Profile scan #pm1#sigma", "pe");
    TGraph *leg_global = new TGraph(1);
    leg_global->SetPoint(0, 0, 0);
    leg_global->SetMarkerStyle(21);
    leg_global->SetMarkerColor(kRed);
    leg_global->SetMarkerSize(marker_size);
    leg->AddEntry(leg_global, "Global best-fit", "p");
    TGraph *leg_inj = new TGraph(1);
    leg_inj->SetPoint(0, 0, 0);
    leg_inj->SetMarkerStyle(33);
    leg_inj->SetMarkerColor(kOrange+7);
    leg_inj->SetMarkerSize(marker_size * 1.1f);
    leg->AddEntry(leg_inj, "Injected values", "p");
    leg->Draw();

    c2->Update();
    drawVersionWatermark(c2);
    c2->SaveAs((filename+"_1sigma_detailed.pdf").c_str(),"pdf");
    delete c2;
    } // end _1sigma_detailed.pdf scope

    //Next version
    TCanvas *c2 =  new TCanvas((filename+"1sigma").c_str(), (filename+"1sigma").c_str() , 20*nparams, 400);
    c2->cd();
    c2->SetBottomMargin(0.25);
    c2->SetRightMargin(0.05);

    onesig.SetFillColor(kBlue-7);
    onesig.SetStats(0);
    //onesig.SetMinimum(min(-1.2,minVal*1.2));
    onesig.SetMinimum(minVal*1.1);
    onesig.SetMaximum(maxVal*1.1);

    onesig.GetXaxis()->SetNdivisions(barvalues.size());  // Set number of tick marks
    onesig.GetXaxis()->SetLabelSize(0);  // Hide default numerical labels

    onesig.SetTitle("");
    TGraphAsymmErrors todraw = onesig;
    if(mask_osc) {
        for(size_t i = 0; i < model.nparams; ++i) {
            todraw.SetPoint(i, 0,0);
            todraw.SetPointError(i, 0, 0, 0, 0);
        }
    }
    todraw.Draw("A2");
    //onesig.Draw("A2");
    //onesig.GetYaxis()->SetTitle("#sigma Shift");
    todraw.GetYaxis()->SetTitle("Posterior 1#sigma Error");
    todraw.GetYaxis()->SetTitleOffset(0.8);

    float y_min = todraw.GetMinimum();
    // Precompute labels and shrink the font so the longest -45 deg label fits in the bottom
    // margin (0.25 of the canvas). A rotated label of L chars drops ~0.42*size*L in NDC height;
    // with margin ~0.25 that gives size ~0.5/L, capped at the previous fixed 0.03.
    std::vector<std::string> labels(barvalues.size());
    size_t max_label_len = 1;
    for (size_t i = 0; i < barvalues.size(); ++i) {
        labels[i] = (with_osc && i < model.nparams)
            ? ("Log_{10}(" + model.pretty_param_names[i] + ")")
            : config.m_mcgen_variation_plotname_map.at(names[i]);
        max_label_len = std::max(max_label_len, labels[i].size());
    }
    float label_text_size = std::max(0.012f, std::min(0.03f, 0.5f / (float)max_label_len));
    for (size_t i = 0; i < barvalues.size(); ++i) {
        // In syst-only mode (with_osc=false), all entries are splines
        // In with_osc mode, first model.nparams entries are physics, rest are splines
        std::string label;
        if (with_osc && i < model.nparams) {
            const bool is_log_param = (i < model.is_log10.size()) && model.is_log10[i];
            if(is_log_param) {
                label = "Log_{10}(" + model.pretty_param_names[i] + ")";
            } else {
                label = model.pretty_param_names[i];
            }
        } else {
            label = config.m_mcgen_variation_plotname_map.at(names[i]);
        }
        TLatex* text = new TLatex(barvalues[i], y_min - 0.05, label.c_str());  // Position text below axis
        text->SetTextAlign(13);
        text->SetTextSize(label_text_size);
        text->SetTextAngle(-45);
        text->Draw();
    }
    c2->Update();

    //if (twosig) {
    //    TGraphAsymmErrors *h2 = new TGraphAsymmErrors(barvalues.size(),barvalues.data(), bfvalues.data(), barvalues_err.data(), barvalues_err.data(), values2_down.data(), values2_up.data());
    //    h2->SetFillColor(38);
    //    h2->Se


    TLine l(0,0,nBins+0.5,0);
    l.SetLineStyle(2);
    l.SetLineColor(kBlack);
    l.SetLineWidth(1);
    l.Draw();
    TLine l2(0,-1,nBins+0.5,-1);
    l2.SetLineStyle(3);
    l2.SetLineColor(kBlack);
    l2.SetLineWidth(1);
    l2.Draw();
    TLine l3(0,1,nBins+0.5,1);
    l3.SetLineStyle(3);
    l3.SetLineColor(kBlack);
    l3.SetLineWidth(1);
    l3.Draw();



    // Get y-axis range for out-of-bounds handling
    float y_axis_min = minVal * 1.1;
    float y_axis_max = maxVal * 1.1;
    float y_range = y_axis_max - y_axis_min;
    log<LOG_INFO>(L"%1% || _1sigma plot y-axis range: min=%2%, max=%3%") % __func__ % y_axis_min % y_axis_max;
    float arrow_margin = y_range * 0.08;  // margin from edge for out-of-range markers
    float arrow_length = y_range * 0.06;  // length of the arrow

    // Horizontal offsets to prevent marker overlap
    float offset_blue = -0.12;   // init_seed (blue) - left
    float offset_red = 0.0;      // true_params (red) - center
    float offset_black = 0.12;   // best fit (black) - right

    // Helper lambda to draw a marker with out-of-range arrow handling
    auto drawMarkerWithArrow = [&](float x, float y, int color, float marker_size) {
        bool clamp_below = y < -5.0f;
        bool clamp_above = y > 5.0f;
        if(!clamp_below && !clamp_above && (y < y_axis_min || y > y_axis_max)) return;

        float draw_y = clamp_below ? y_axis_min + arrow_margin : (clamp_above ? y_axis_max - arrow_margin : y);

        TMarker* marker = new TMarker(x, draw_y, 29);
        marker->SetMarkerSize(marker_size);
        marker->SetMarkerColor(color);
        marker->Draw();

        if(clamp_below) {
            TArrow* arr = new TArrow(x, draw_y - arrow_length * 0.3, x, y_axis_min + arrow_length * 0.2, 0.008, "|>");
            arr->SetLineColor(color);
            arr->SetFillColor(color);
            arr->SetLineWidth(1);
            arr->Draw();
        } else if(clamp_above) {
            TArrow* arr = new TArrow(x, draw_y + arrow_length * 0.3, x, y_axis_max - arrow_length * 0.2, 0.008, "|>");
            arr->SetLineColor(color);
            arr->SetFillColor(color);
            arr->SetLineWidth(1);
            arr->Draw();
        }
    };

    for (int i = 0; i < nBins; ++i) {
        if(mask_osc && i < (int)model.nparams) continue;
        // In syst-only mode, init_seed/true_params are full-size but plot indices are spline-only
        int vec_idx = with_osc ? i : (i + model.nparams);
        float x_center = i + 0.5;

        // Blue star: init_seed (best fit seed)
        drawMarkerWithArrow(x_center + offset_blue, init_seed[vec_idx], kBlue, 0.6);

        // Red star: true_params (injected truth)
        if (vec_idx < true_params.size()) {
            if (i < 3) {  // Log first few for debugging
                log<LOG_INFO>(L"%1% || _1sigma marker i=%2%: true_params[%3%]=%4%, below_range=%5%")
                    % __func__ % i % vec_idx % true_params[vec_idx] % (true_params[vec_idx] < y_axis_min);
            }
            drawMarkerWithArrow(x_center + offset_red, true_params[vec_idx], kRed, 0.5);
        }

        // Black star: best fit value
        drawMarkerWithArrow(x_center + offset_black, bfvalues[i], kBlack, 0.5);
    }



    drawVersionWatermark(c2);
    c2->SaveAs((filename+"_1sigma.pdf").c_str(),"pdf");
    delete c2;

    return;
}
