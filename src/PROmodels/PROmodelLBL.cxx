/**
 * @file PROmodelLBL.cxx
 * @brief Implementation of the PROLBL three-flavour long-baseline model (matter and vacuum).
 * @author PROfit Collaboration
 */
#include "PROmodels/PROmodelLBL.h"
#include "PROconfig.h"
#include "NuFastLBL.h"

#include <cmath>

namespace PROfit {

std::vector<Eigen::VectorXf> lbl_ordering_octant_variants(const Eigen::VectorXf &phys, int i_dm21, int i_dm31, int i_s23) {
    Eigen::VectorXf ordering = phys;
    ordering(i_dm31) = 2.0f * phys(i_dm21) - phys(i_dm31);
    Eigen::VectorXf octant = phys;
    octant(i_s23) = 1.0f - phys(i_s23);
    Eigen::VectorXf both = ordering;
    both(i_s23) = octant(i_s23);
    return {ordering, octant, both};
}

LBLInputMode resolve_lbl_inputs(const PROconfig &config, bool matter, std::vector<int> &ivars, double &baseline_km) {
    const std::map<std::string, int> &pmap = config.m_model_parameter_map;
    const bool has_L = pmap.count("L"), has_E = pmap.count("E"), has_LE = pmap.count("L/E");
    const bool has_baseline = config.m_model_options.count("baseline");

    if(has_LE && (has_L || has_E)) {
        log<LOG_ERROR>(L"%1% || Give the LBL model 'L'+'E', 'E' with baseline=, or a single 'L/E' — not a mix.") % __func__;
        throw std::runtime_error("Ambiguous LBL parameters: L/E mixed with L or E");
    }
    if(has_L && has_E) {
        if(has_baseline) {
            log<LOG_ERROR>(L"%1% || baseline= conflicts with a per-event 'L' <parameter>: use one or the other.") % __func__;
            throw std::runtime_error("LBL baseline= given alongside an L parameter");
        }
        ivars = {pmap.at("L"), pmap.at("E")};
        return LBLInputMode::PairLE;
    }
    if(has_E && !has_L) {
        if(!has_baseline) {
            log<LOG_ERROR>(L"%1% || An 'E'-only LBL model needs the fixed baseline: <model ... baseline=\"<km>\">.") % __func__;
            throw std::runtime_error("LBL E-only mode requires baseline=");
        }
        static bool warned_multi_detector = false;
        if(config.m_num_detectors > 1 && !warned_multi_detector) {
            warned_multi_detector = true;
            log<LOG_WARNING>(L"%1% || baseline= applies ONE fixed baseline to all %2% detectors. If they sit at different baselines, use per-event 'L'+'E' <parameter>s instead.")
                % __func__ % config.m_num_detectors;
        }
        baseline_km = config.m_model_options.at("baseline");
        ivars = {pmap.at("E")};
        return LBLInputMode::EOnly;
    }
    if(has_LE) {
        if(has_baseline) {
            log<LOG_ERROR>(L"%1% || baseline= is redundant with an 'L/E' <parameter>: only the ratio is physical in vacuum.") % __func__;
            throw std::runtime_error("LBL baseline= given alongside an L/E parameter");
        }
        if(matter) {
            log<LOG_ERROR>(L"%1% || The matter LBL model cannot take a bare 'L/E': the MSW potential scales with E alone. Use per-event 'L'+'E', or 'E' with baseline=. (Pre-v3.0.4 'L/E' was silently read as E at L = 1300 km.)") % __func__;
            throw std::runtime_error("Matter LBL model cannot use a bare L/E");
        }
        ivars = {pmap.at("L/E")};
        return LBLInputMode::RatioLE;
    }
    log<LOG_ERROR>(L"%1% || Missing LBL model <parameter>s: 'L'+'E' [km, GeV signed], or 'E' with baseline=%2%.") % __func__
        % (matter ? "" : ", or a single signed 'L/E' [km/GeV]");
    throw std::runtime_error("Missing LBL parameters");
}

PROLBL::PROLBL(const PROconfig &config, const PROpeller &prop, bool matter) : matter(matter), baseline_km(0) {

    // No model_functions: get_probs is overridden, so prob_types alone sets the
    // component count (same pattern as the decay models).
    prob_types = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    input_mode = resolve_lbl_inputs(config, matter, ivars, baseline_km);

    const std::map<std::string, double> &opts = config.m_model_options;
    auto opt_or = [&opts](const char *key, double fallback) {
        auto it = opts.find(key);
        return it == opts.end() ? fallback : it->second;
    };
    density           = opt_or("density",           default_density);
    electron_fraction = opt_or("electron_fraction", default_electron_fraction);
    n_newton          = (int)opt_or("n_newton",     default_n_newton);

    // Warnings are one-time: FC/AFC construct fresh models per universe.
    static bool warned_bin_average = false;
    if(opts.count("bin_average") && !warned_bin_average) {
        warned_bin_average = true;
        log<LOG_WARNING>(L"%1% || bin_average= only applies to LBL_3+1-vacuum_angles; the 3nu LBL models sample each truth bin at its midpoint. Ignored.") % __func__;
    }
    if(!matter) {
        std::string ignored;
        for(const char *key : {"density", "electron_fraction", "n_newton"})
            if(opts.count(key)) ignored += (ignored.empty() ? "" : ", ") + std::string(key);
        static bool warned_vacuum_opts = false;
        if(!ignored.empty() && !warned_vacuum_opts) {
            warned_vacuum_opts = true;
            log<LOG_WARNING>(L"%1% || LBL_3nu-vacuum_angles ignores the matter-only <model> options [%2%].") % __func__ % ignored.c_str();
        }
    } else if(matter && density == 0) {
        static bool warned_zero_density = false;
        if(!warned_zero_density) {
            warned_zero_density = true;
            log<LOG_WARNING>(L"%1% || density=\"0\" with LBL_3nu-matter_angles: consider the LBL_3nu-vacuum_angles tag instead, which uses the dedicated (cheaper, dmsq_31-sign-safe) vacuum solver.") % __func__;
        }
    }

    if(matter) {
        // NuFast's DMP matter solver is singular in a narrow band (measured width
        // < 4e-5 eV^2) around dmsq_31 = 0 and around Dmsqee = dmsq_31 - s12sq*dmsq_21 = 0,
        // both inside the fit bounds. Reject the band like 3+1 rejects non-unitarity.
        model_constraint = [](const Eigen::VectorXf &p) -> int {
            float dmsqee = p(1) - p(2) * p(0);
            return (std::fabs(dmsqee) < dmsqee_guard || std::fabs(p(1)) < dmsqee_guard) ? 0 : 1;
        };
        log<LOG_INFO>(L"%1% || LBL 3nu model in MATTER (%2%): rho = %3% g/cm^3, Ye = %4%, N_Newton = %5%.")
            % __func__ % (input_mode == InputMode::PairLE ? "per-event L and E" : "per-event E at fixed baseline")
            % density % electron_fraction % n_newton;
    } else {
        log<LOG_INFO>(L"%1% || LBL 3nu model in VACUUM (%2%).") % __func__
            % (input_mode == InputMode::RatioLE ? "signed per-event L/E"
               : input_mode == InputMode::PairLE ? "per-event L and E" : "per-event E at fixed baseline");
    }
    if(input_mode == InputMode::EOnly)
        log<LOG_INFO>(L"%1% || Fixed baseline L = %2% km.") % __func__ % baseline_km;

    build_hists_and_combined(prop);


    nparams = 6;
    param_names = {"dmsq_21", "dmsq_31", "sinsqt12", "sinsqt13", "sinsqt23", "delta_CP"};
    pretty_param_names = {"#Delta m^{2}_{21}", "#Delta m^{2}_{31}", "sin^{2}#theta_{12}",
        "sin^2#theta_{13}", "sin^{2}#theta_{23}", "delta_{CP}"};
    pretty_param_units = {"eV^{2}", "eV^{2}", "", "", "", "rad"};
    is_log10 = {false, false, false, false, false, false};
    build_param_index();
    // For <model><prior name="dmsq_32">: dm^2_32 = dm^2_31 - dm^2_21.
    derived_quantities["dmsq_32"] = {{1, 1.0f}, {0, -1.0f}};
    lb = Eigen::VectorXf(6);
    ub = Eigen::VectorXf(6);
    lb << 6e-5f, -3e-3f, 0.2f, 0.01f, 0.3f, -M_PI;
    ub << 9e-5f, 3e-3f, 0.4f, 0.04f, 0.7f, M_PI;
    default_val = Eigen::VectorXf(6);
    // Defaults set to midpoint of [lb, ub] for each parameter so the fitter starts in-range.
    default_val << 7.5e-5f, 1e-3f, 0.3f, 0.025f, 0.5f, 0.0f;
}

void PROLBL::oscillate(const Eigen::VectorXf &params, float L, float E, float (*out)[3][3]) const {
    if(E == 0) {
        // No-oscillation limit (also guards the 1/E in NuFast's phases).
        for(int from = 0; from < 3; ++from)
            for(int to = 0; to < 3; ++to)
                (*out)[from][to] = from == to ? 1.0f : 0.0f;
        return;
    }
    if(matter) {
        NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
                  params(0), params(1), L, E, density, electron_fraction, n_newton, out);
        // Gradient probes can step inside the singular band the model_constraint
        // rejects; a nudged dmsq_31 keeps them finite.
        if(!std::isfinite((*out)[0][0]) || !std::isfinite((*out)[1][1]) || !std::isfinite((*out)[2][2])) {
            float center = params(2) * params(0);
            float dm31 = center + std::copysign(2 * dmsqee_guard, params(1) - center);
            if(std::fabs(dm31) < dmsqee_guard) dm31 = std::copysign(2 * dmsqee_guard, dm31);
            NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
                      params(0), dm31, L, E, density, electron_fraction, n_newton, out);
        }
    } else {
        NuFastLBL::Probability_Vacuum_LBL(params(2), params(3), params(4), params(5),
                  params(0), params(1), L, E, out);
    }
}

Eigen::MatrixXf PROLBL::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    // Eigen matrices are column major by default so we want this layout to get a
    // contiguous probs array from each column, then transpose before returning.
    const size_t n_flat = var_arrs[0].size();
    Eigen::MatrixXf probs(prob_types.size(), n_flat);
    probs.row(0).setConstant(1);
    for(size_t i = 0; i < n_flat; ++i) {
        float L, E;
        switch(input_mode) {
            case InputMode::PairLE:  L = var_arrs[0][i]; E = var_arrs[1][i]; break;
            case InputMode::EOnly:   L = baseline_km;    E = var_arrs[0][i]; break;
            // Only L/E is physical (vacuum): feed the signed ratio as L with E = 1.
            default:                 L = var_arrs[0][i]; E = L == 0 ? 0.0f : 1.0f; break;
        }
        oscillate(phys, L, E, (float(*)[3][3])((float*)probs.col(i).data()+1));
    }
    return probs.transpose();
}

}
