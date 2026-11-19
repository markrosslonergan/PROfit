// Header-only response-product regression: no ROOT files or PROfit build needed.
#include "CSRDoubleProduct.h"
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>
#include <type_traits>
#include <vector>

namespace {
using Matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>;
int failures = 0;

void check(bool passed, const char* message) {
    std::cout << (passed ? "PASS " : "FAIL ") << message << '\n';
    if (!passed) ++failures;
}

bool same_bits(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    return a.size() == b.size() &&
        (a.size() == 0 || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}
}

int main() {
    using PROfit::CSRDoubleProduct::multiply;
    {
        // A derivative column can have large opposite-sign terms. Float
        // accumulation loses the unit residual; double accumulation retains it.
        Matrix h(1, 3);
        std::vector<Eigen::Triplet<float>> entries = {{0,0,1.f}, {0,1,1.f}, {0,2,1.f}};
        h.setFromTriplets(entries.begin(), entries.end());
        const float storage[] = {100000000.f, 1.f, -100000000.f};
        Eigen::Map<const Eigen::VectorXf> values(storage, 3);
        static_assert(std::is_same<decltype(multiply(h,values)), Eigen::VectorXf>::value,
                      "Response product must retain the native float-vector API");
        check(multiply(h,values)(0) == 1.f, "mapped input retains a cancellation residual");
    }
    {
        // Exercise structural zeros, several component blocks and many inputs.
        // Binary fractions make the independently evaluated dense64 reference
        // exactly representable, so precision or indexing regressions fail hard.
        const int rows = 31, cols = 96;
        Matrix h(rows,cols);
        std::vector<Eigen::Triplet<float>> entries;
        for (int r=0; r<rows; ++r)
            for (int k=0; k<7; ++k)
                entries.emplace_back(r,(r*13+k*11)%cols, float((r+k)%9+1)/8.f);
        h.setFromTriplets(entries.begin(),entries.end());
        h.makeCompressed();
        const Eigen::MatrixXd dense = Eigen::MatrixXf(h).cast<double>();
        const Matrix original = h;
        bool exact = true;
        for (int sample=0; sample<41; ++sample) {
            std::vector<float> storage(cols);
            for (int c=0; c<cols; ++c)
                storage[c] = float(((c+3)*(sample+5))%47-23)/16.f;
            Eigen::Map<const Eigen::VectorXf> values(storage.data(),cols);
            const Eigen::VectorXd reference = dense * values.cast<double>();
            const Eigen::VectorXf expected = reference.cast<float>();
            exact = exact && same_bits(multiply(h,values),expected);
        }
        check(exact, "CSR products match a dense64 reference at 41 points");
        check(h.nonZeros() == original.nonZeros() &&
              std::memcmp(h.valuePtr(),original.valuePtr(),h.nonZeros()*sizeof(float)) == 0,
              "products leave float response coefficients unchanged");

        // Profile workers share the model but evaluate different parameter
        // vectors. Temporaries must remain local to each call.
        std::atomic<bool> parallel_ok(true);
        std::vector<std::thread> workers;
        for (int worker=0; worker<8; ++worker) {
            workers.emplace_back([&,worker]() {
                Eigen::VectorXf values(cols);
                for (int c=0; c<cols; ++c) values(c)=float((c+worker)%17-8)/8.f;
                const Eigen::VectorXd reference = dense * values.cast<double>();
                const Eigen::VectorXf expected = reference.cast<float>();
                for (int i=0; i<100; ++i)
                    if (!same_bits(multiply(h,values),expected)) parallel_ok.store(false);
            });
        }
        for (auto& worker: workers) worker.join();
        check(parallel_ok.load(), "concurrent products use independent temporaries");
    }
    {
        Matrix h(4,5);
        const Eigen::VectorXf values = Eigen::VectorXf::Ones(5);
        check(multiply(h,values).isZero(), "empty response rows produce zero counts");
        Matrix zero_columns(3,0);
        check(multiply(zero_columns,Eigen::VectorXf(0)).isZero(),
              "zero-column response produces a correctly sized zero vector");
        check(multiply(zero_columns,Eigen::VectorXf(0)).size()==3,
              "zero-column response retains its reco dimension");
    }
    return failures ? 1 : 0;
}
