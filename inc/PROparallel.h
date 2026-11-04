#ifndef PROPARALLEL_H_
#define PROPARALLEL_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace PROfit {

/// Run job(i) for every i in [0, n) on up to nthreads std::threads (the calling thread is one
/// of them). Indices are handed out dynamically; each job writes only its own result slot, so
/// the outcome does not depend on the thread count.
template <class F>
void parallel_for(size_t n, size_t nthreads, F &&job) {
    std::atomic<size_t> next{0};
    auto worker = [&]() { for(size_t i = next++; i < n; i = next++) job(i); };
    nthreads = std::max<size_t>(1, std::min(nthreads, n));
    std::vector<std::thread> pool;
    for(size_t t = 1; t < nthreads; ++t) pool.emplace_back(worker);
    worker();
    for(auto &th : pool) th.join();
}

}  // namespace PROfit

#endif
