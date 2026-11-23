/**
 * @file PROmodel3p1decayvis.cxx
 * @brief Implementation of the 3+1 + visible-decay model family.
 * @author PROfit Collaboration
 */
#include "PROmodels/PROmodel3p1decayvis.h"

namespace PROfit {

// ------------------------------------------------------------------
// PRO3p1_decay_vis_model1
// ------------------------------------------------------------------

PRO3p1_decay_vis_model1::PRO3p1_decay_vis_model1(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {
    // 3+1+decay to lower energy neutrinos
    // model 1 from https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.075002

    if(parameter_map.find("L") == parameter_map.end()) {
        log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L'. Make sure it's in the model section of XML.") % __func__ % __LINE__;
        throw std::runtime_error("Missing parameter: L");
    }
    if(parameter_map.find("E") == parameter_map.end()) {
        log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'E'. Make sure it's in the model section of XML.") % __func__ % __LINE__;
        throw std::runtime_error("Missing parameter: E");
    }
    ivars = {parameter_map.at("L"), parameter_map.at("E")};

    // Rules 1/2/3 hold the same-bin (parent-bin) survival contribution for
    // P_mumu / P_mue / P_ee.  Rules 4/5/6 hold the visible-decay contribution
    // arriving at lower-E daughter bins from those same flavor channels.
    prob_types = {0, 1, 2, 3, 4, 5, 6};

    model_constraint = [this](const Eigen::VectorXf &v){return this->UnitarityConstraint(v);};

    truth_overflow_cells = false; // get_counts needs the uniform truth (L, E) grid
    build_hists_and_combined(prop); // fills hists, H_combined, phys_grid_sizes
    N_truth = compute_N_truth(prop); // save N_truth, so it can be modified by flux-weighting in PROcess.cxx

    // Detect per-pair flat-bin offsets between paired osc and decay subchannels from
    // N_truth.  For each rule pair (osc r in {1,2,3}, decay r+3), the rule-(r+3)
    // carriers live at a fixed shift from rule-r in the flat grid.  Each pair is
    // allowed its own shift (so e.g. nue's osc-then-decay pair-order can differ
    // from numu's); within a pair, the shift must be self-consistent across
    // detectors — verified by checking that every non-zero rule-osc bin has a
    // matching non-zero rule-decay bin at the detected shift.
    auto first_nonzero_row = [](const Eigen::MatrixXf &M, int col) -> long int {
        for(long int i = 0; i < M.rows(); ++i) if(M(i, col) > 0.0f) return i;
        return -1;
    };
    for(int osc_r = 1; osc_r <= 3; ++osc_r) {
        const int decay_r = osc_r + 3;
        long int o = first_nonzero_row(N_truth, osc_r);
        long int d = first_nonzero_row(N_truth, decay_r);
        if(o < 0 || d < 0) {
            log<LOG_ERROR>(L"%1% || PRO3p1_decay_vis_model1: rule %2% or %3% has no MC carriers in N_truth — "
                           L"the XML must duplicate every rule-1/2/3 branch with a paired rule-4/5/6 branch.")
                % __func__ % osc_r % decay_r;
            exit(EXIT_FAILURE);
        }
        const long int shift = d - o;
        // Verify the shift is self-consistent: every non-zero rule-osc bin must map
        // to a non-zero rule-decay bin at the same offset.  Catches XML layouts that
        // pair osc/decay subchannels inconsistently across detectors.
        for(long int i = 0; i < N_truth.rows(); ++i) {
            if(N_truth(i, osc_r) <= 0.0f) continue;
            const long int j = i + shift;
            if(j < 0 || j >= N_truth.rows() || N_truth(j, decay_r) <= 0.0f) {
                log<LOG_ERROR>(L"%1% || PRO3p1_decay_vis_model1: rule %2%->%3% offset %4% is not "
                               L"self-consistent (bin %5% has rule-%2% events but bin %6% has no "
                               L"rule-%3% events). Each osc subchannel must be paired immediately "
                               L"with its decay partner using identical truth binning, in every detector.")
                    % __func__ % osc_r % decay_r % shift % i % j;
                exit(EXIT_FAILURE);
            }
        }
        decay_bin_offsets[osc_r - 1] = shift;
    }

    nparams = 4;
    param_names = {"dmsq", "Ue4^2", "Um4^2", "g_phi"};
    pretty_param_names = {"#Deltam^{2}", "|U_{e4}|^{2}", "|U_{#mu4}|^{2}", "g_{#phi}"};
    pretty_param_units = {"eV^{2}", "", "", ""};
    is_log10 = {true, true, true, false};
    build_param_index();
    lb = Eigen::VectorXf(4);
    ub = Eigen::VectorXf(4);
    default_val = Eigen::VectorXf(4);
    lb << -2, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 0;
    ub << 2, -1e-4, -1e-4, 10;
    default_val << -2, -8, -8, 0;
}

int PRO3p1_decay_vis_model1::UnitarityConstraint(const Eigen::VectorXf &v) {
    const float Ue4sq = maybe_convert_log("Ue4^2", v(param_name_to_index.at("Ue4^2")));
    const float Um4sq = maybe_convert_log("Um4^2", v(param_name_to_index.at("Um4^2")));
    const float g_phi = maybe_convert_log("g_phi", v(param_name_to_index.at("g_phi")));
    return ((Ue4sq+Um4sq)<1 && g_phi>=0 ? 1 : 0);
}

Eigen::MatrixXf PRO3p1_decay_vis_model1::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    // For this model, we calculate get_counts, which is more intuitive with energy smearing
    // (it's not exactly a probability for a single neutrino event when it depends on the behavior
    // of higher energy events), but it can be re-framed as a probability for consistency.
    // Note that this means probabilities greater than one are possible with large down-smearing!
    auto counts = get_counts(phys, var_arrs, N_truth);
    auto N_arr = N_truth.array();
    return (N_arr > 0.0f).select(counts.array() / N_arr, 0.0f);
}

// get_counts reads truth-level event counts from N_truth_vals (passed explicitly by the
// caller) rather than this->N_truth.  Callers pass either this->N_truth for the baseline
// spectrum, or a pre-reweighted copy (rows scaled by the per-truth-E flux weight) when
// evaluating pre-migration flux systematics.
Eigen::MatrixXf PRO3p1_decay_vis_model1::get_counts(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs,
                                                    const Eigen::MatrixXf &N_truth_vals) const {
    float dmsq  = maybe_convert_log("dmsq",  phys(param_name_to_index.at("dmsq")));
    float Ue4sq = maybe_convert_log("Ue4^2", phys(param_name_to_index.at("Ue4^2")));
    float Um4sq = maybe_convert_log("Um4^2", phys(param_name_to_index.at("Um4^2")));
    float g_phi = maybe_convert_log("g_phi", phys(param_name_to_index.at("g_phi")));
    float Us4sq = 1.0f - Ue4sq - Um4sq; // assuming no tau mixing (small effect on NC events)
    float sum_active = Ue4sq + Um4sq;

    float freq = 1.266932679f * dmsq;
    const auto &L_arr = var_arrs[0];
    const auto &E_arr = var_arrs[1];
    size_t n_flat = L_arr.size();
    size_t n_E    = phys_grid_sizes[1];

    float E_bin_width = E_arr[1] - E_arr[0];
    if (E_bin_width <= 0.0f) {
        log<LOG_ERROR>(L"%1% || expected positive truth-E bin width but got %2%") % __func__ % E_bin_width;
        exit(EXIT_FAILURE);
    }
    const float dE_tol = 1e-4f * E_bin_width;
    for(size_t i = 2; i < n_E; ++i) {
        float diff = E_arr[i] - E_arr[i-1];
        if(diff > 0.0f && std::abs(diff - E_bin_width) > dE_tol) {
            log<LOG_ERROR>(L"%1% || 3+1+decay requires uniform truth energy binning, "
                L"found bin width of %2% at bin %3% when expecting %4%.") % __func__ % diff % i % E_bin_width;
            exit(EXIT_FAILURE);
        }
    }

    // Truth-L bin width, needed to average rapid oscillations over the L bin.  The
    // flat grid concatenates subchannel blocks along L, so skip the non-positive
    // jumps at block boundaries (mirroring the truth-E uniformity check above).
    float L_bin_width = 0.0f;
    for(size_t i = n_E; i < n_flat; i += n_E) {
        float diff = L_arr[i] - L_arr[i - n_E];
        if(diff <= 0.0f) continue;
        if(L_bin_width == 0.0f) { L_bin_width = diff; continue; }
        if(std::abs(diff - L_bin_width) > 1e-4f * L_bin_width) {
            log<LOG_ERROR>(L"%1% || 3+1+decay requires uniform truth baseline binning, "
                L"found bin width of %2% when expecting %3%.") % __func__ % diff % L_bin_width;
            exit(EXIT_FAILURE);
        }
    }
    const auto sinc = [](float x) { return std::abs(x) < 1e-4f ? 1.0f : std::sin(x) / x; };

    // alpha_phi = g_phi^2/(4 pi)
    // Gamma_nu4 = |Us4|^2 (1 - |Us4|^2) alpha_phi/4 m_4^2/E_4
    //           = |Us4|^2 (1 - |Us4|^2) g_phi^2/(16 pi) m_4^2/E_4
    // 1 / L_dec = Gamma_nu4
    // expterm = exp(-L/(2 L_dec)) = exp(-L/2 Gamma_nu4)
    //         = exp(-L/2 |Us4|^2 (1 - |Us4|^2) g_phi^2/(16 pi) m_4^2/E_4)
    //         = exp(-[m_4^2 L/4E] |Us4|^2 (1 - |Us4|^2) g_phi^2/(8 pi))
    //         = exp(-delta |Us4|^2 * sum_active g_phi^2/(8 pi))
    //         = exp(-delta * exp_prefactor)
    const float exp_prefactor = Us4sq * sum_active * g_phi * g_phi / (8.0f * 3.14159f);

    // Mixing-only prefactors for the decay-migration term — see
    // https://github.com/kjkellyphys/muB_oscillation/blob/main/OscTools/sterile_tools.py#L261
    // In https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.075002, just after
    // equation 11, I believe they actually give the formula for the flavor projection
    // squared, not the flavor projection.
    const float mix_mumu = Um4sq * Um4sq * Us4sq / sum_active;
    const float mix_mue  = Um4sq * Ue4sq * Us4sq / sum_active;
    const float mix_ee   = Ue4sq * Ue4sq * Us4sq / sum_active;

    // Column 0 holds the no-oscillation-no-decay prediction.
    // Columns 1..3 hold the same-bin (parent-bin) oscillation contribution.
    // Columns 4..6 hold the visible-decay daughter contribution for the
    // P_mumu / P_mue / P_ee channels respectively, populated at lower-E bins.
    Eigen::MatrixXf counts = Eigen::MatrixXf::Zero(n_flat, 7);

    // looping over flattened (L, E) bins — flat_par is the parent neutrino index
    for(size_t flat_par = 0; flat_par < n_flat; ++flat_par) {
        float E_par = E_arr[flat_par];
        float L_par = L_arr[flat_par];
        float delta = freq * L_par / E_par;
        float expterm = std::exp(-delta * exp_prefactor);
        // Average the interference term over the truth bin rather than sampling it at
        // the bin midpoint: at large dmsq the phase 2*delta wraps many times within
        // one (L, E) bin, so the midpoint cos(2*delta) is an aliased pseudorandom
        // value.  Averaging cos(2*freq*L/E) analytically over the L bin, and to first
        // order in the phase over the E bin, damps the term by sinc factors that
        // -> 1 for slow oscillations and -> 0 in the fully-averaged regime.  The
        // exponential decay factor varies slowly across a bin and is left at midpoint.
        float damp = sinc(freq * L_bin_width / E_par)
                   * sinc(freq * L_par * E_bin_width / (E_par * E_par));
        float cos_mult_exp = std::cos(2.0f * delta) * expterm * damp;
        float osc_term = 1.0f - 2.0f*cos_mult_exp + expterm*expterm;

        // Oscillation contribution at flat_par (parent = daughter for non-decayed events).
        // no-osc
        counts(flat_par, 0) += N_truth_vals(flat_par, 0);
        // P_mu_mu osc
        counts(flat_par, 1) += (1.0f - 2.0f*Um4sq*(1.0f - cos_mult_exp) + Um4sq*Um4sq*osc_term) * N_truth_vals(flat_par, 1);
        // P_mu_e osc
        counts(flat_par, 2) += Ue4sq * Um4sq * osc_term * N_truth_vals(flat_par, 2);
        // P_e_e osc
        counts(flat_par, 3) += (1.0f - 2.0f*Ue4sq*(1.0f - cos_mult_exp) + Ue4sq*Ue4sq*osc_term) * N_truth_vals(flat_par, 3);

        // Decay redistribution: this parent decays into daughters at lower truth-E
        // within the same truth-L bin.  Everything that does not depend on the
        // daughter's energy is collected into kernel_base_*; multiplying by
        // s_dec(E_dst, E_par) inside the inner loop gives the contribution to each
        // daughter bin.
        float decay_frac = 1.0f - expterm * expterm;             // (1 - exp(-L/L_dec(E_par)))
        float kernel_base_mumu = mix_mumu * E_bin_width * decay_frac * N_truth_vals(flat_par, 1);
        float kernel_base_mue  = mix_mue  * E_bin_width * decay_frac * N_truth_vals(flat_par, 2);
        float kernel_base_ee   = mix_ee   * E_bin_width * decay_frac * N_truth_vals(flat_par, 3);

        size_t l_idx     = flat_par / n_E;
        size_t e_par_idx = flat_par % n_E;
        // looping over lower-E daughter bins at the same L.  flat_dst lives in the
        // OSC subchannel's bin range (same as the parent); we shift it by the
        // per-pair offset when writing to cols 4/5/6 so each decay daughter lands at
        // its paired decay subchannel's bin, where the rule-4/5/6 carrier events
        // actually live.  Out-of-range writes (from cross-subchannel iterations) are
        // filtered: at those positions, N_truth for the decay rule is zero and
        // probs = counts/N_truth clamps to zero.
        for(size_t e_dst_idx = 0; e_dst_idx < e_par_idx; ++e_dst_idx) {
            size_t flat_dst = l_idx * n_E + e_dst_idx;
            float E_dst = E_arr[flat_dst];
            float s_dec = 2.0f * E_dst / (E_par * E_par);        // dP/dE_d, 1/GeV
            // DegradationCorrection
            // Approximates sigma_CC ∝ E over the SBN range, i.e. weight by E_dst/E_par.
            // TODO: add factors here to more precisely account for the E_nu-dependent
            // cross section and efficiency effects.
            float s_dec_w = s_dec * (E_dst / E_par);
            const long int dst_mumu = (long int)flat_dst + decay_bin_offsets[0];
            const long int dst_mue  = (long int)flat_dst + decay_bin_offsets[1];
            const long int dst_ee   = (long int)flat_dst + decay_bin_offsets[2];
            if(dst_mumu >= 0 && dst_mumu < (long int)n_flat) counts(dst_mumu, 4) += kernel_base_mumu * s_dec_w;
            if(dst_mue  >= 0 && dst_mue  < (long int)n_flat) counts(dst_mue,  5) += kernel_base_mue  * s_dec_w;
            if(dst_ee   >= 0 && dst_ee   < (long int)n_flat) counts(dst_ee,   6) += kernel_base_ee   * s_dec_w;
        }
    }
    return counts;
}

// ------------------------------------------------------------------
// PRO3p1_decay_vis_model2
// ------------------------------------------------------------------

PRO3p1_decay_vis_model2::PRO3p1_decay_vis_model2(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {
    // 3+1+decay to lower energy neutrinos
    // model 2 from https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.075002

    if(parameter_map.find("L") == parameter_map.end()) {
        log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L'. Make sure it's in the model section of XML.") % __func__ % __LINE__;
        throw std::runtime_error("Missing parameter: L");
    }
    if(parameter_map.find("E") == parameter_map.end()) {
        log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'E'. Make sure it's in the model section of XML.") % __func__ % __LINE__;
        throw std::runtime_error("Missing parameter: E");
    }
    ivars = {parameter_map.at("L"), parameter_map.at("E")};

    prob_types = {0, 1, 2, 3};

    model_constraint = [this](const Eigen::VectorXf &v){return this->UnitarityConstraint(v);};

    truth_overflow_cells = false; // get_counts needs the uniform truth (L, E) grid
    build_hists_and_combined(prop); // fills hists, H_combined, phys_grid_sizes
    N_truth = compute_N_truth(prop); // save N_truth, so it can be modified by flux-weighting in PROcess.cxx

    nparams = 4;
    param_names = {"dmsq", "Ue4^2", "Um4^2", "g_e"};
    pretty_param_names = {"#Deltam^{2}", "|U_{e4}|^{2}", "|U_{#mu4}|^{2}", "g_{e}"};
    pretty_param_units = {"eV^{2}", "", "", ""};
    is_log10 = {true, true, true, false};
    build_param_index();
    lb = Eigen::VectorXf(4);
    ub = Eigen::VectorXf(4);
    default_val = Eigen::VectorXf(4);
    lb << -2, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 0;
    ub << 2, -1e-4, -1e-4, 10;
    default_val << -2, -8, -8, 0;
}

int PRO3p1_decay_vis_model2::UnitarityConstraint(const Eigen::VectorXf &v) {
    const float Ue4sq = maybe_convert_log("Ue4^2", v(param_name_to_index.at("Ue4^2")));
    const float Um4sq = maybe_convert_log("Um4^2", v(param_name_to_index.at("Um4^2")));
    const float g_e = maybe_convert_log("g_e", v(param_name_to_index.at("g_e")));
    return ((Ue4sq+Um4sq)<1 && g_e>=0 ? 1 : 0);
}

Eigen::MatrixXf PRO3p1_decay_vis_model2::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    auto counts = get_counts(phys, var_arrs, N_truth);
    auto N_arr = N_truth.array();
    return (N_arr > 0.0f).select(counts.array() / N_arr, 0.0f);
}

Eigen::MatrixXf PRO3p1_decay_vis_model2::get_counts(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs,
                                                    const Eigen::MatrixXf &N_truth_vals) const {
    float dmsq  = maybe_convert_log("dmsq",  phys(param_name_to_index.at("dmsq")));
    float Ue4sq = maybe_convert_log("Ue4^2", phys(param_name_to_index.at("Ue4^2")));
    float Um4sq = maybe_convert_log("Um4^2", phys(param_name_to_index.at("Um4^2")));
    float g_e   = maybe_convert_log("g_e",   phys(param_name_to_index.at("g_e")));

    float freq = 1.266932679f * dmsq;
    const auto &L_arr = var_arrs[0];
    const auto &E_arr = var_arrs[1];
    size_t n_flat = L_arr.size();
    size_t n_E    = phys_grid_sizes[1];

    float E_bin_width = E_arr[1] - E_arr[0];
    if (E_bin_width <= 0.0f) {
        log<LOG_ERROR>(L"%1% || expected positive truth-E bin width but got %2%") % __func__ % E_bin_width;
        exit(EXIT_FAILURE);
    }
    const float dE_tol = 1e-4f * E_bin_width;
    for(size_t i = 2; i < n_E; ++i) {
        float diff = E_arr[i] - E_arr[i-1];
        if(diff > 0.0f && std::abs(diff - E_bin_width) > dE_tol) {
            log<LOG_ERROR>(L"%1% || 3+1+decay requires uniform truth energy binning, "
                L"found bin width of %2% at bin %3% when expecting %4%.") % __func__ % diff % i % E_bin_width;
            exit(EXIT_FAILURE);
        }
    }

    // Truth-L bin width, needed to average rapid oscillations over the L bin.  The
    // flat grid concatenates subchannel blocks along L, so skip the non-positive
    // jumps at block boundaries (mirroring the truth-E uniformity check above).
    float L_bin_width = 0.0f;
    for(size_t i = n_E; i < n_flat; i += n_E) {
        float diff = L_arr[i] - L_arr[i - n_E];
        if(diff <= 0.0f) continue;
        if(L_bin_width == 0.0f) { L_bin_width = diff; continue; }
        if(std::abs(diff - L_bin_width) > 1e-4f * L_bin_width) {
            log<LOG_ERROR>(L"%1% || 3+1+decay requires uniform truth baseline binning, "
                L"found bin width of %2% when expecting %3%.") % __func__ % diff % L_bin_width;
            exit(EXIT_FAILURE);
        }
    }
    const auto sinc = [](float x) { return std::abs(x) < 1e-4f ? 1.0f : std::sin(x) / x; };

    // alpha_e = g_e^2/(4 pi)
    // Gamma_nu4 = alpha_e / 4 m_4^2/E_4
    //           = g_e^2/(16 pi) m_4^2/E_4
    // 1 / L_dec = Gamma_nu4
    // expterm = exp(-L/(2 L_dec)) = exp(-L/2 Gamma_nu4)
    //         = exp(-L/2 g_e^2/(16 pi) m_4^2/E_4)
    //         = exp(-[m_4^2 L/4E] g_e^2/(8 pi))
    //         = exp(-delta g_e^2/(8 pi))
    const float exp_prefactor = g_e * g_e / (8.0f * 3.14159f);

    // Zero-initialize counts: each cell accumulates (a) the oscillation contribution
    // when flat_par equals the cell, and (b) decay-migration from higher-E parents
    // at the same L.
    Eigen::MatrixXf counts = Eigen::MatrixXf::Zero(n_flat, 4);

    // looping over flattened (L, E) bins — flat_par is the parent index
    for(size_t flat_par = 0; flat_par < n_flat; ++flat_par) {
        float E_par = E_arr[flat_par];
        float L_par = L_arr[flat_par];
        float delta = freq * L_par / E_par;
        float expterm = std::exp(-delta * exp_prefactor);
        // Average the interference term over the truth bin rather than sampling it at
        // the bin midpoint — see the matching comment in PRO3p1_decay_vis_model1.
        float damp = sinc(freq * L_bin_width / E_par)
                   * sinc(freq * L_par * E_bin_width / (E_par * E_par));
        float cos_mult_exp = std::cos(2.0f * delta) * expterm * damp;
        float osc_term = 1.0f - 2.0f*cos_mult_exp + expterm*expterm;

        // Oscillation contribution at flat_par (same as model1, g_e drives the damping)
        counts(flat_par, 0) += N_truth_vals(flat_par, 0);
        // P_mu_mu osc
        counts(flat_par, 1) += (1.0f - 2.0f*Um4sq*(1.0f - cos_mult_exp) + Um4sq*Um4sq*osc_term) * N_truth_vals(flat_par, 1);
        // P_mu_e osc
        counts(flat_par, 2) += Ue4sq * Um4sq * osc_term * N_truth_vals(flat_par, 2);
        // P_e_e osc
        counts(flat_par, 3) += (1.0f - 2.0f*Ue4sq*(1.0f - cos_mult_exp) + Ue4sq*Ue4sq*osc_term) * N_truth_vals(flat_par, 3);

        // Decay redistribution: in model2 (g_e coupling) only fullosc and nue are affected;
        // numu disappearance (j=1) sees no decay contribution.  The decay damping
        // (1 - exp(-L/L_dec)) is evaluated at the PARENT energy E_par.  Daughter-energy
        // dependence enters only through s_dec(E_dst, E_par).
        float decay_frac = 1.0f - expterm * expterm;             // (1 - exp(-L/L_dec(E_par)))
        float kernel_base_mue = Um4sq * E_bin_width * decay_frac * N_truth_vals(flat_par, 2);
        float kernel_base_ee  = Ue4sq * E_bin_width * decay_frac * N_truth_vals(flat_par, 3);

        size_t l_idx     = flat_par / n_E;
        size_t e_par_idx = flat_par % n_E;
        // looping over lower-E daughter bins at the same L
        for(size_t e_dst_idx = 0; e_dst_idx < e_par_idx; ++e_dst_idx) {
            size_t flat_dst = l_idx * n_E + e_dst_idx;
            float E_dst = E_arr[flat_dst];
            float s_dec = 2.0f * E_dst / (E_par * E_par);
            // DegradationCorrection: weight by sigma(E_dst)/sigma(E_par) ~ E_dst/E_par
            // — see the matching comment in PRO3p1_decay_vis_model1.
            float s_dec_w = s_dec * (E_dst / E_par);
            counts(flat_dst, 2) += kernel_base_mue * s_dec_w;
            counts(flat_dst, 3) += kernel_base_ee  * s_dec_w;
        }
    }
    return counts;
}

}
