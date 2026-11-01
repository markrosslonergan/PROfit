/**
 * @file PROmodelLBL3p1.cxx
 * @brief Implementation of the PROLBL3p1 3+1 vacuum long-baseline model.
 * @author PROfit Collaboration
 */
#include "PROmodels/PROmodelLBL3p1.h"
#include "PROconfig.h"

#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <stdexcept>

namespace PROfit {

namespace {

enum Param : int { kDm41, kS14, kS24, kS34, kD14, kD24, kDm21, kDm31, kS12, kS13, kS23, kDCP, kNParams };

constexpr double kPhase = 1.266932679;  // X = kPhase * dmsq [eV^2] * L [km] / E [GeV], as the SBL models
constexpr double kLn10  = 2.302585092994046;

// Mass-eigenstate pairs (i > j); masses are offsets {0, dmsq_21, dmsq_31, dmsq_41}.
constexpr int kNPair = 6;
constexpr int kPairI[kNPair] = {1, 2, 3, 2, 3, 3};
constexpr int kPairJ[kNPair] = {0, 0, 0, 1, 1, 2};

// Gradient bookkeeping: splitting parameters (kMassParam[k] moves mass k+1) and mixing parameters.
constexpr std::array<int, 3> kMassParam = {kDm21, kDm31, kDm41};
constexpr std::array<int, 9> kMixParam = {kS14, kS24, kS34, kD14, kD24, kS12, kS13, kS23, kDCP};

// Coefficient rows: 0-8 = P(a->b), a, b in (e, mu, tau), stored at column 1 + row;
// 9-11 = NC weight 1 - P(a->s). P_row = kDelta[row] + C.row(row) * [<sin^2 X_q> ; <sin 2X_q>].
constexpr int kNRow = 12;
constexpr std::array<double, kNRow> kDelta = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};

using cplx  = std::complex<double>;
using Mat4c = Eigen::Matrix<cplx, 4, 4>;
using Coef  = Eigen::Matrix<double, kNRow, 2 * kNPair>;
using Terms = Eigen::Matrix<double, 2 * kNPair, 1>;
using Rows  = Eigen::Matrix<double, kNRow, 1>;

// The six rotations of U = R34 R~24 R~14 R23 R~13 R12, in product order.
struct Rotation { int i, j, theta, delta; };  // plane, sin^2 parameter, phase parameter (-1: none)
constexpr std::array<Rotation, 6> kRotations = {{
    {2, 3, kS34, -1}, {1, 3, kS24, kD24}, {0, 3, kS14, kD14},
    {1, 2, kS23, -1}, {0, 2, kS13, kDCP}, {0, 1, kS12, -1},
}};

enum class Deriv { None, Theta, Delta };

// R_ij(theta, delta) with the PDG phase placement, or its derivative in theta or delta.
Mat4c rotation(const Rotation &r, double s, double c, double delta, Deriv d) {
    Mat4c R = Mat4c::Zero();
    if(d == Deriv::None) R.setIdentity();
    const cplx e = std::polar(1.0, -delta);
    const cplx mi(0, -1);
    switch(d) {
        case Deriv::None:  R(r.i, r.i) = c;  R(r.j, r.j) = c;  R(r.i, r.j) = s * e;      R(r.j, r.i) = -s * std::conj(e); break;
        case Deriv::Theta: R(r.i, r.i) = -s; R(r.j, r.j) = -s; R(r.i, r.j) = c * e;      R(r.j, r.i) = -c * std::conj(e); break;
        case Deriv::Delta:                                     R(r.i, r.j) = mi * s * e; R(r.j, r.i) = mi * s * std::conj(e); break;
    }
    return R;
}

// C(row, q) = -4 sign Re W, C(row, 6+q) = 2 sign Im W, with W = z_i conj(z_j), z_i = conj(U_ai) U_bi
// and sign = -1 on the NC rows. Given dU, fills the derivative of C instead.
void fill_coefs(const Mat4c &U, const Mat4c *dU, Coef &C) {
    for(int row = 0; row < kNRow; ++row) {
        const int a = row < 9 ? row / 3 : row - 9;
        const int b = row < 9 ? row % 3 : 3;
        const double sign = row < 9 ? 1.0 : -1.0;
        std::array<cplx, 4> z, dz;
        for(int i = 0; i < 4; ++i) {
            z[i] = std::conj(U(a, i)) * U(b, i);
            if(dU) dz[i] = std::conj((*dU)(a, i)) * U(b, i) + std::conj(U(a, i)) * (*dU)(b, i);
        }
        for(int q = 0; q < kNPair; ++q) {
            const int i = kPairI[q], j = kPairJ[q];
            const cplx W = dU ? dz[i] * std::conj(z[j]) + z[i] * std::conj(dz[j]) : z[i] * std::conj(z[j]);
            C(row, q) = -4.0 * sign * W.real();
            C(row, kNPair + q) = 2.0 * sign * W.imag();
        }
    }
}

// sinc(h) = sin(h)/h and, if wanted, its derivative; series below |h| = 1e-4.
inline double sinc(double h, double *dS) {
    if(std::fabs(h) < 1e-4) {
        if(dS) *dS = -h / 3;
        return 1 - h * h / 6;
    }
    const double S = std::sin(h) / h;
    if(dS) *dS = (std::cos(h) - S) / h;
    return S;
}

}

PROLBL3p1::PROLBL3p1(const PROconfig &config, const PROpeller &prop) : baseline_km(0) {

    input_mode = resolve_lbl_inputs(config, false, ivars, baseline_km);

    const std::map<std::string, double> &opts = config.m_model_options;
    bin_average = !opts.count("bin_average") || opts.at("bin_average") != 0;

    // One-time: FC/AFC construct fresh models per universe.
    std::string ignored;
    for(const char *key : {"density", "electron_fraction", "n_newton"})
        if(opts.count(key)) ignored += (ignored.empty() ? "" : ", ") + std::string(key);
    static bool warned_matter_opts = false;
    if(!ignored.empty() && !warned_matter_opts) {
        warned_matter_opts = true;
        log<LOG_WARNING>(L"%1% || LBL_3+1-vacuum_angles ignores the matter-only <model> options [%2%].") % __func__ % ignored.c_str();
    }
    log<LOG_INFO>(L"%1% || LBL 3+1 model in VACUUM (%2%), truth-bin averaging of fast oscillations %3%.") % __func__
        % (input_mode == LBLInputMode::RatioLE ? "signed per-event L/E"
           : input_mode == LBLInputMode::PairLE ? "per-event L and E" : "per-event E at fixed baseline")
        % (bin_average ? "ON" : "OFF (bin_average=0: midpoint sampling)");
    if(input_mode == LBLInputMode::EOnly)
        log<LOG_INFO>(L"%1% || Fixed baseline L = %2% km.") % __func__ % baseline_km;

    // H_combined holds only the columns some event's model_rule uses (prob_types), which
    // cuts every FillSpectra GEMV by n_cols / prob_types.size(). get_probs matches that
    // layout on the binned grid; per-event callers index by model_rule and get all n_cols.
    std::vector<int> column(n_cols, -1);
    for(int r : prop.model_rule)
        if(r >= 0 && r < (int)n_cols) column[r] = 0;
    for(size_t r = 0; r < n_cols; ++r)
        if(column[r] == 0) { column[r] = (int)prob_types.size(); prob_types.push_back(r); }
    build_hists_and_combined(prop, true, [&](size_t, size_t i, int) {
        const int r = prop.model_rule[i];
        return r >= 0 && r < (int)n_cols ? column[r] : -1;
    });
    build_grid(config, prop);

    nparams = kNParams;
    param_names = {"dmsq_41", "sinsqt14", "sinsqt24", "sinsqt34", "delta_14", "delta_24",
                   "dmsq_21", "dmsq_31", "sinsqt12", "sinsqt13", "sinsqt23", "delta_CP"};
    pretty_param_names = {"#Delta m^{2}_{41}", "sin^{2}#theta_{14}", "sin^{2}#theta_{24}", "sin^{2}#theta_{34}",
                          "#delta_{14}", "#delta_{24}", "#Delta m^{2}_{21}", "#Delta m^{2}_{31}",
                          "sin^{2}#theta_{12}", "sin^{2}#theta_{13}", "sin^{2}#theta_{23}", "#delta_{CP}"};
    pretty_param_units = {"eV^{2}", "", "", "", "rad", "rad", "eV^{2}", "eV^{2}", "", "", "", "rad"};
    is_log10 = {true, true, true, true, false, false, false, false, false, false, false, false};
    build_param_index();
    // For <model><prior name="dmsq_32">: dm^2_32 = dm^2_31 - dm^2_21.
    derived_quantities["dmsq_32"] = {{kDm31, 1.0f}, {kDm21, -1.0f}};

    const float inf = std::numeric_limits<float>::infinity();
    const float pi = M_PI;
    lb = Eigen::VectorXf(kNParams);
    ub = Eigen::VectorXf(kNParams);
    default_val = Eigen::VectorXf(kNParams);
    // Sterile block as SBL_3+1_angles but to 1000 eV²; the 3nu block as PROLBL except wider
    // |Δm²₃₁| and θ23 over (almost) its full range, symmetric about maximal mixing so the
    // octant mirror of any point is in range: large sterile mixing moves the minima to
    // θ23 ≈ θ34 ≈ π/2 with |Δm²₃₂| up to ~5e-3. sin²θ23 stops 1e-4 short of 0 and 1, where
    // the linear-sin² chain rule 1/(2sc) diverges.
    lb          << -4.0f, -inf,   -inf,   -inf,   -pi, -pi, 6e-5f,   -6e-3f, 0.2f, 0.01f,  1e-4f,        -pi;
    ub          <<  3.0f, -1e-4f, -1e-4f, -1e-4f,  pi,  pi, 9e-5f,    6e-3f, 0.4f, 0.04f,  1.0f - 1e-4f,  pi;
    default_val <<  0.0f, -8.0f,  -8.0f,  -8.0f,  0.0f, 0.0f, 7.5e-5f, 1e-3f, 0.3f, 0.025f, 0.5f, 0.0f;
}

std::vector<Eigen::VectorXf> PROLBL3p1::degenerate_variants(const Eigen::VectorXf &phys) const {
    return lbl_ordering_octant_variants(phys, kDm21, kDm31, kS23);
}

double PROLBL3p1::signed_le(double v0, double v1) const {
    switch(input_mode) {
        case LBLInputMode::RatioLE: return v0;
        case LBLInputMode::EOnly:   return v0 == 0 ? 0 : baseline_km / v0;
        case LBLInputMode::PairLE:  return v1 == 0 ? 0 : v0 / v1;
    }
    return 0;
}

void PROLBL3p1::build_grid(const PROconfig &config, const PROpeller &prop) {
    // Half-width of every global truth bin of each ivar, in the [mode][detector][channel]
    // [subchannel][bin] order PROcreate gives variable_midbin.
    const size_t N = ivars.size();
    std::vector<std::vector<double>> half(N);
    std::vector<size_t> sizes(N);
    for(size_t k = 0; k < N; ++k) {
        for(size_t im = 0; im < config.m_num_modes; ++im)
            for(size_t id = 0; id < config.m_num_detectors; ++id)
                for(size_t ic = 0; ic < config.m_num_channels; ++ic) {
                    const std::vector<float> edges = config.m_channel_variable_bins.at(ic)[ivars[k]].Edges();
                    for(size_t sc = 0; sc < config.m_num_subchannels.at(ic); ++sc)
                        for(size_t j = 0; j + 1 < edges.size(); ++j)
                            half[k].push_back(0.5 * std::fabs((double)edges[j + 1] - edges[j]));
                }
        sizes[k] = prop.variable_midbin[ivars[k]].size();
        if(half[k].size() != sizes[k]) {
            log<LOG_ERROR>(L"%1% || Truth binning of variable %2% has %3% global bins in the config but %4% in the MC store.")
                % __func__ % ivars[k] % half[k].size() % sizes[k];
            throw std::runtime_error("LBL 3+1: config / PROpeller truth-binning mismatch");
        }
    }

    // The flat grid exactly as FillSpectra builds it (row-major, last ivar fastest), so
    // get_probs recognises it by value; identical points are evaluated once.
    grid_.assign(N, std::vector<float>(n_phys_bins));
    grid_index_.resize(n_phys_bins);
    grid_points_.clear();
    std::map<std::array<double, 3>, int> seen;
    for(long int flat = 0; flat < n_phys_bins; ++flat) {
        std::array<size_t, 2> bin{};
        long int rem = flat;
        for(int k = (int)N - 1; k >= 0; --k) {
            bin[k] = rem % (long int)sizes[k];
            grid_[k][flat] = prop.variable_midbin[ivars[k]][bin[k]];
            rem /= (long int)sizes[k];
        }
        const double v0 = grid_[0][flat], w0 = half[0][bin[0]];
        const double v1 = N > 1 ? grid_[1][flat] : 0, w1 = N > 1 ? half[1][bin[1]] : 0;
        OscPoint pt{signed_le(v0, v1), {0, 0}};
        if(bin_average && pt.le != 0) {
            switch(input_mode) {
                case LBLInputMode::RatioLE: pt.hw[0] = w0; break;
                case LBLInputMode::EOnly:   pt.hw[0] = baseline_km * w0 / (v0 * v0); break;
                case LBLInputMode::PairLE:  pt.hw[0] = w0 / std::fabs(v1); pt.hw[1] = std::fabs(v0) * w1 / (v1 * v1); break;
            }
        }
        auto it = seen.emplace(std::array<double, 3>{pt.le, pt.hw[0], pt.hw[1]}, (int)grid_points_.size());
        if(it.second) grid_points_.push_back(pt);
        grid_index_[flat] = it.first->second;
    }
    log<LOG_INFO>(L"%1% || Physics grid: %2% bins, %3% distinct oscillation points.") % __func__ % n_phys_bins % grid_points_.size();
}

std::vector<PROLBL3p1::OscPoint> PROLBL3p1::exact_points(const std::vector<std::vector<float>> &var_arrs) const {
    static std::atomic<bool> noted{false};
    if(!noted.exchange(true))
        log<LOG_INFO>(L"%1% || Evaluating off the binned grid (event-by-event): exact per-event probabilities, no bin averaging.") % __func__;
    std::vector<OscPoint> pts(var_arrs[0].size(), OscPoint{0, {0, 0}});
    for(size_t i = 0; i < pts.size(); ++i)
        pts[i].le = signed_le(var_arrs[0][i], var_arrs.size() > 1 ? var_arrs[1][i] : 0);
    return pts;
}

void PROLBL3p1::evaluate(const Eigen::VectorXf &phys, const std::vector<OscPoint> &pts,
                         Eigen::MatrixXf &P, std::vector<Eigen::MatrixXf> *dP) const {
    auto value = [&](int p) { return is_log10[p] ? std::pow(10.0, (double)phys(p)) : (double)phys(p); };

    // U and its coefficient matrix, once per call. dth = d(theta)/d(internal parameter):
    // 1/(2sc) for a linear sin^2, ln10 s/(2c) for a log10 one (finite at sin^2 = 0).
    std::array<double, 6> s, c, delta, dth;
    std::array<Mat4c, 6> F;
    for(size_t f = 0; f < kRotations.size(); ++f) {
        const Rotation &r = kRotations[f];
        const double x = value(r.theta);
        s[f] = std::sqrt(x);
        c[f] = std::sqrt(1 - x);
        delta[f] = r.delta < 0 ? 0 : value(r.delta);
        dth[f] = is_log10[r.theta] ? kLn10 * s[f] / (2 * c[f]) : 1 / (2 * s[f] * c[f]);
        F[f] = rotation(r, s[f], c[f], delta[f], Deriv::None);
    }
    std::array<Mat4c, 7> pre;  // pre[f] = F_0 ... F_{f-1}
    pre[0].setIdentity();
    for(size_t f = 0; f < 6; ++f) pre[f + 1] = pre[f] * F[f];
    const Mat4c &U = pre[6];
    Coef C;
    fill_coefs(U, nullptr, C);

    const std::array<double, 4> m2 = {0, value(kDm21), value(kDm31), value(kDm41)};
    std::array<double, kNPair> dmsq;
    for(int q = 0; q < kNPair; ++q) dmsq[q] = m2[kPairI[q]] - m2[kPairJ[q]];

    // Gradient set-up: dC per mixing parameter, and d(dmsq_q)/d(internal) per splitting.
    std::array<Coef, kNParams> dC;
    std::array<Terms, 3> incidence;
    if(dP) {
        std::array<Mat4c, 6> suf;  // suf[f] = F_{f+1} ... F_5
        suf[5].setIdentity();
        for(int f = 5; f > 0; --f) suf[f - 1] = F[f] * suf[f];
        for(size_t f = 0; f < kRotations.size(); ++f) {
            const Rotation &r = kRotations[f];
            const Mat4c dU = pre[f] * rotation(r, s[f], c[f], delta[f], Deriv::Theta) * suf[f] * dth[f];
            fill_coefs(U, &dU, dC[r.theta]);
            if(r.delta < 0) continue;
            const Mat4c dUd = pre[f] * rotation(r, s[f], c[f], delta[f], Deriv::Delta) * suf[f];
            fill_coefs(U, &dUd, dC[r.delta]);
        }
        for(int k = 0; k < 3; ++k) {
            const int p = kMassParam[k];
            const double chain = is_log10[p] ? kLn10 * m2[k + 1] : 1.0;
            for(int q = 0; q < kNPair; ++q) {
                const double inc = chain * ((kPairI[q] == k + 1) - (kPairJ[q] == k + 1));
                incidence[k](q) = incidence[k](kNPair + q) = inc;
            }
        }
    }

    const size_t n = pts.size();
    P.resize(n_cols, n);
    P.row(0).setOnes();
    if(dP) dP->assign(nparams, Eigen::MatrixXf::Zero(n_cols, n));
    const Eigen::Map<const Rows> kdelta(kDelta.data());

    for(size_t p = 0; p < n; ++p) {
        const OscPoint &pt = pts[p];
        Terms v, dv;  // [<sin^2 X_q> ; <sin 2X_q>] and their derivatives in dmsq_q
        const double dphi = 2 * kPhase * pt.le;
        for(int q = 0; q < kNPair; ++q) {
            const double phi = dphi * dmsq[q];
            const double sp = std::sin(phi), cp = std::cos(phi);
            double S = 1, dS = 0;
            for(double hw : pt.hw) {
                if(hw == 0) continue;
                const double k = 2 * kPhase * hw;
                double ds = 0;
                const double sa = sinc(k * dmsq[q], dP ? &ds : nullptr);
                dS = dS * sa + S * ds * k;
                S *= sa;
            }
            v(q) = 0.5 * (1 - cp * S);
            v(kNPair + q) = sp * S;
            if(dP) {
                dv(q) = 0.5 * (sp * dphi * S - cp * dS);
                dv(kNPair + q) = cp * dphi * S + sp * dS;
            }
        }
        P.col(p).tail<kNRow>() = (kdelta + C * v).cast<float>();
        if(!dP) continue;
        for(int m : kMixParam)
            (*dP)[m].col(p).tail<kNRow>() = (dC[m] * v).cast<float>();
        for(int k = 0; k < 3; ++k)
            (*dP)[kMassParam[k]].col(p).tail<kNRow>() = (C * incidence[k].cwiseProduct(dv)).cast<float>();
    }
}

Eigen::MatrixXf PROLBL3p1::expand(const Eigen::MatrixXf &P, bool on_grid) const {
    if(!on_grid) return P.transpose();
    Eigen::MatrixXf out(grid_index_.size(), prob_types.size());
    for(size_t c = 0; c < prob_types.size(); ++c)
        for(size_t f = 0; f < grid_index_.size(); ++f)
            out(f, c) = P(prob_types[c], grid_index_[f]);
    return out;
}

Eigen::MatrixXf PROLBL3p1::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    const bool on_grid = var_arrs == grid_;
    std::vector<OscPoint> exact;
    if(!on_grid) exact = exact_points(var_arrs);
    Eigen::MatrixXf P;
    evaluate(phys, on_grid ? grid_points_ : exact, P, nullptr);
    return expand(P, on_grid);
}

std::vector<Eigen::MatrixXf> PROLBL3p1::get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    const bool on_grid = var_arrs == grid_;
    std::vector<OscPoint> exact;
    if(!on_grid) exact = exact_points(var_arrs);
    Eigen::MatrixXf P;
    std::vector<Eigen::MatrixXf> dP;
    evaluate(phys, on_grid ? grid_points_ : exact, P, &dP);
    for(Eigen::MatrixXf &g : dP) g = expand(g, on_grid);
    return dP;
}

}
