/**
 * @file PROtimer.h
 * @brief Lightweight wall-clock instrumentation for PROfit.
 *
 * Two tools:
 *  - PROtimer::Stage   (RAII) logs "[TIMER] <name> : <time>" when it goes out of scope and
 *                      records the time. Use for one-off phases (config, process, fits...).
 *  - PROtimer::Accum   (RAII) silently accumulates time and call counts per name. Use in hot
 *                      paths (metric evaluations, FillSpectra, per-event loops). Per-thread
 *                      storage, so it is cheap and safe inside the FC / thread pools.
 *  - PROtimer::Add(name, seconds) records an externally measured interval.
 *
 * PROtimer::PrintSummary() prints one table (sorted by total time) with calls, total,
 * mean and max per name. Accumulated totals are summed over threads, so for code run in
 * parallel they can exceed the wall time.
 *
 * Set PROFIT_TIMING=0 in the environment to disable all of it.
 */
#ifndef PROTIMER_H_
#define PROTIMER_H_

#include "PROlog.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace PROtimer {

using clock_t_ = std::chrono::steady_clock;

struct Stat {
    double total = 0.0;
    double max = 0.0;
    long long calls = 0;
};

inline bool Enabled() {
    static const bool on = [] {
        const char* v = std::getenv("PROFIT_TIMING");
        return !(v && std::string(v) == "0");
    }();
    return on;
}

inline double Since(clock_t_::time_point t0) {
    return std::chrono::duration<double>(clock_t_::now() - t0).count();
}

/// Human readable duration: "812 us", "35.2 ms", "12.345 s", "3m 07.2s", "1h 02m 03s".
inline std::string Fmt(double s) {
    char b[64];
    if (s < 1e-3)        std::snprintf(b, sizeof(b), "%.0f us", s * 1e6);
    else if (s < 1.0)    std::snprintf(b, sizeof(b), "%.1f ms", s * 1e3);
    else if (s < 60.0)   std::snprintf(b, sizeof(b), "%.3f s", s);
    else if (s < 3600.0) std::snprintf(b, sizeof(b), "%dm %04.1fs", int(s / 60), s - 60 * int(s / 60));
    else                 std::snprintf(b, sizeof(b), "%dh %02dm %02ds", int(s / 3600), int(s / 60) % 60, int(s) % 60);
    return b;
}

namespace detail {
    struct ThreadStats { std::unordered_map<std::string, Stat> m; };
    inline std::mutex& Mutex() { static std::mutex m; return m; }
    inline std::vector<std::shared_ptr<ThreadStats>>& All() {
        static std::vector<std::shared_ptr<ThreadStats>> v;
        return v;
    }
    /// Per-thread table, registered once so it survives the thread (merged in the summary).
    inline ThreadStats& Local() {
        thread_local std::shared_ptr<ThreadStats> mine = [] {
            auto p = std::make_shared<ThreadStats>();
            std::lock_guard<std::mutex> lock(Mutex());
            All().push_back(p);
            return p;
        }();
        return *mine;
    }
    inline clock_t_::time_point& ProgramStart() {
        static clock_t_::time_point t0 = clock_t_::now();
        return t0;
    }
}  // namespace detail

/// Record an interval measured elsewhere.
inline void Add(const std::string& name, double seconds) {
    if (!Enabled()) return;
    Stat& s = detail::Local().m[name];
    s.total += seconds;
    s.calls++;
    s.max = std::max(s.max, seconds);
}

/// Silent accumulating timer for hot paths.
class Accum {
public:
    explicit Accum(const char* name) : name_(name), t0_(clock_t_::now()) {}
    ~Accum() { if (Enabled()) Add(name_, Since(t0_)); }
    Accum(const Accum&) = delete;
    Accum& operator=(const Accum&) = delete;
private:
    const char* name_;
    clock_t_::time_point t0_;
};

/// Like Accum, for names built at run time (e.g. per systematic mode).
class AccumS {
public:
    explicit AccumS(std::string name) : name_(std::move(name)), t0_(clock_t_::now()) {}
    ~AccumS() { if (Enabled()) Add(name_, Since(t0_)); }
    AccumS(const AccumS&) = delete;
    AccumS& operator=(const AccumS&) = delete;
private:
    std::string name_;
    clock_t_::time_point t0_;
};

/// Logged stage timer: prints at start (LOG_DEBUG) and end (LOG_INFO), and records.
class Stage {
public:
    explicit Stage(std::string name, log_level_t level = LOG_INFO)
        : name_(std::move(name)), level_(level), t0_(clock_t_::now()) {
        detail::ProgramStart();
        if (Enabled()) log<LOG_DEBUG>(L"%1% || [TIMER] start : %2%") % "PROtimer" % name_.c_str();
    }
    ~Stage() { Stop(); }
    /// Stop early (idempotent); returns the elapsed seconds.
    double Stop() {
        if (stopped_) return elapsed_;
        stopped_ = true;
        elapsed_ = Since(t0_);
        if (Enabled()) {
            Add(name_, elapsed_);
            const double wall = Since(detail::ProgramStart());
            if (level_ <= LOG_WARNING)
                log<LOG_WARNING>(L"%1% || [TIMER] %2% : %3%   (at +%4%)") % "PROtimer" % name_.c_str() % Fmt(elapsed_).c_str() % Fmt(wall).c_str();
            else if (level_ == LOG_INFO)
                log<LOG_INFO>(L"%1% || [TIMER] %2% : %3%   (at +%4%)") % "PROtimer" % name_.c_str() % Fmt(elapsed_).c_str() % Fmt(wall).c_str();
            else
                log<LOG_DEBUG>(L"%1% || [TIMER] %2% : %3%   (at +%4%)") % "PROtimer" % name_.c_str() % Fmt(elapsed_).c_str() % Fmt(wall).c_str();
        }
        return elapsed_;
    }
    Stage(const Stage&) = delete;
    Stage& operator=(const Stage&) = delete;
private:
    std::string name_;
    log_level_t level_;
    clock_t_::time_point t0_;
    bool stopped_ = false;
    double elapsed_ = 0.0;
};

/// Mark the program start (call first thing in main so "+wall" offsets are meaningful).
inline void MarkProgramStart() { detail::ProgramStart() = clock_t_::now(); }

/// One table of everything recorded so far, sorted by total time.
inline void PrintSummary() {
    if (!Enabled()) return;
    std::map<std::string, Stat> merged;
    {
        std::lock_guard<std::mutex> lock(detail::Mutex());
        for (auto& t : detail::All())
            for (auto& [k, v] : t->m) {
                Stat& s = merged[k];
                s.total += v.total;
                s.calls += v.calls;
                s.max = std::max(s.max, v.max);
            }
    }
    std::vector<std::pair<std::string, Stat>> rows(merged.begin(), merged.end());
    std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.second.total > b.second.total; });
    const double wall = Since(detail::ProgramStart());
    log<LOG_WARNING>(L"%1% || [TIMER] ===================== timing summary (wall %2%) =====================") % "PROtimer" % Fmt(wall).c_str();
    log<LOG_WARNING>(L"%1% || [TIMER] %2%") % "PROtimer" % "      total     %wall      calls        mean         max   name";
    for (auto& [name, s] : rows) {
        char line[512];
        std::snprintf(line, sizeof(line), "%11s  %7.1f%%  %9lld  %10s  %10s   %s",
                      Fmt(s.total).c_str(), wall > 0 ? 100.0 * s.total / wall : 0.0, s.calls,
                      Fmt(s.calls ? s.total / s.calls : 0.0).c_str(), Fmt(s.max).c_str(), name.c_str());
        log<LOG_WARNING>(L"%1% || [TIMER] %2%") % "PROtimer" % line;
    }
    log<LOG_WARNING>(L"%1% || [TIMER] (totals of code run in parallel are summed over threads and can exceed 100%%)") % "PROtimer";
}

}  // namespace PROtimer

#endif  // PROTIMER_H_
