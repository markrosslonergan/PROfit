// Multiply float CSR response matrices using double accumulation.
// Keep the native float result API; do not materialize a second response matrix.
#pragma once
#include <Eigen/Eigen>
#include <Eigen/Sparse>
namespace PROfit { namespace CSRDoubleProduct {
template<class Index,class Derived>
inline Eigen::VectorXf multiply(const Eigen::SparseMatrix<float,Eigen::RowMajor,Index>& h,
                                const Eigen::MatrixBase<Derived>& values) {
    const Eigen::VectorXd promoted=values.template cast<double>();
    const Eigen::VectorXd result=h.template cast<double>()*promoted;
    return result.cast<float>();
}
}}
