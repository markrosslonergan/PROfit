#include "PROsyst.h"
#include "PROcess.h"
#include "PROpeller.h"
#include "PROconfig.h"
#include "PROcreate.h"
#include "PROlog.h"
#include "PROtocall.h"
#include <Eigen/Eigen>
#include <mutex>
#include <random>

namespace PROfit {

    bool PROsyst::shape_only = false;

    PROsyst::PROsyst( const PROpeller &prop, const PROconfig &config, const std::vector<SystStruct>& systs, bool shapeonly, int other_index, const PROmodel* model, const Eigen::VectorXf* params) : other_index(other_index) {
        shape_only = shapeonly;
        for(const auto& syst: systs) {
            log<LOG_DEBUG>(L"%1% || syst mode: %2%") % __func__ % syst.mode.c_str();
            if(syst.mode == "spline" || syst.mode == "norm" || syst.mode == "hist1d" || syst.mode == "hist2d" || syst.mode == "explicit_spline") {
                bool unmirrored = config.m_mcgen_variation_unmirrored.find(syst.systname) != config.m_mcgen_variation_unmirrored.end();
                FillSpline(syst, unmirrored);
                spline_prior_types.back() = config.GetSplinePriorType(syst.systname);
                ++n_splines;
            } else if(syst.mode == "spline_to_covariance") {
                if(model == nullptr){
                    log<LOG_ERROR>(L"%1% || spline_to_covariance requires a PROmodel to use spline2cov. "
                        L"Construct PROsyst with a model (and optional params).") % __func__;
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);
                }
                // Build spline first, then convert to covariance matrix
                bool unmirrored = config.m_mcgen_variation_unmirrored.find(syst.systname) != config.m_mcgen_variation_unmirrored.end();
                FillSpline(syst, unmirrored);
                size_t spline_idx = splines.size() - 1;

                // Build temporary priors/centers for the current spline list (including the
                // just-added spline) needed by spline2cov. Use locals so we don't clobber the
                // member variables, which are properly initialised after the full syst loop.
                Eigen::VectorXf tmp_priors  = Eigen::VectorXf::Constant(splines.size(), 1);
                Eigen::VectorXf tmp_centers = Eigen::VectorXf::Constant(splines.size(), 0);
                for(const auto &[name, prior]: config.m_mcgen_variation_prior) {
                    auto it = std::find(spline_names.begin(), spline_names.end(), name);
                    if(it != std::end(spline_names)) {
                        size_t idx = std::distance(std::begin(spline_names), it);
                        tmp_priors(idx) = prior;
                    }
                }
                for(const auto &[name, center]: config.m_mcgen_variation_prior_centers) {
                    auto it = std::find(spline_names.begin(), spline_names.end(), name);
                    if(it != std::end(spline_names)) {
                        size_t idx = std::distance(std::begin(spline_names), it);
                        tmp_centers(idx) = center;
                    }
                }
                // Temporarily assign so that spline2cov (via FillSplineRandomThrow) sees
                // the correct priors, then restore the saved state afterwards.
                Eigen::VectorXf saved_priors  = spline_priors;
                Eigen::VectorXf saved_centers = spline_centers;
                spline_priors  = tmp_priors;
                spline_centers = tmp_centers;

                Eigen::VectorXf cvparams;
                if(params != nullptr){
                    cvparams = *params;
                }else{
                    cvparams = Eigen::VectorXf::Zero(model->nparams + splines.size());
                    cvparams.segment(0, model->nparams) = model->default_val;
                }

                log<LOG_INFO>(L"%1% || Converting spline '%2%' to covariance matrix using spline2cov") % __func__ % syst.systname.c_str();
                // Disjoint seed range per converted spline (spline2cov draws 500 throws at seed..seed+499).
                Eigen::MatrixXf frac_cov = spline2cov(spline_idx, config, prop, *model, cvparams, 42u + (uint32_t)spline_idx * 500u);
                spline_priors  = saved_priors;
                spline_centers = saved_centers;
                Eigen::MatrixXf corr = GenerateCorrMatrix(frac_cov);

                // Remove the spline (it was the last one appended by FillSpline)
                syst_map.erase(syst.systname);
                splines.pop_back();
                spline_names.pop_back();
                spline_lo.pop_back();
                spline_hi.pop_back();
                spline_has_restrict.pop_back();
                spline_restrict_lo.pop_back();
                spline_restrict_hi.pop_back();
                spline_binnings.pop_back();
                spline_prior_types.pop_back();

                // Store as covariance instead
                syst_map[syst.systname] = {covmat.size(), SystType::Covariance};
                covmat.push_back(frac_cov);
                corrmat.push_back(corr);
                covar_names.push_back(syst.systname);
                ++n_covar;
            } else if(syst.mode == "covariance") {
                this->CreateMatrix(syst);
                covar_names.push_back(syst.systname);
                ++n_covar;
            } else if(syst.mode == "covariance_to_spline") {
                size_t n_before = splines.size();
                size_t n_covar_before = covar_names.size();
                FillSplinesFromCovariance(syst);
                size_t n_after = splines.size();
                size_t n_covar_after = covar_names.size();

                // Propagate parent tag + plotname to the synthesized knob entries so downstream
                // plotting code (PROplot.cxx) can group them under the same tag.
                // The maps are populated by the XML parser; here we append entries for the
                // dynamically-generated decomposition knobs and the optional residual covariance.
                PROconfig& mut_config = const_cast<PROconfig&>(config);
                auto parent_tags_it = mut_config.m_mcgen_variation_tags.find(syst.systname);
                auto parent_plotname_it = mut_config.m_mcgen_variation_plotname_map.find(syst.systname);
                auto propagate = [&](const std::string& child_name) {
                    if(parent_tags_it != mut_config.m_mcgen_variation_tags.end()) {
                        mut_config.m_mcgen_variation_tags[child_name] = parent_tags_it->second;
                    }
                    if(parent_plotname_it != mut_config.m_mcgen_variation_plotname_map.end()) {
                        // Append the "_decomp_knob_<i>" / "_resid_cov" suffix onto the parent plotname.
                        const std::string suffix = child_name.substr(syst.systname.size());
                        mut_config.m_mcgen_variation_plotname_map[child_name] = parent_plotname_it->second + suffix;
                    }
                };
                for(size_t si = n_before; si < n_after; ++si) propagate(spline_names[si]);
                for(size_t ci = n_covar_before; ci < n_covar_after; ++ci) propagate(covar_names[ci]);
            }else if(syst.mode == "flat"){
                this->CreateFlatMatrix(config, syst); 
                covar_names.push_back(syst.systname); 
                ++n_covar;
            }else if(syst.mode == "norm_to_covariance"){
                this->CreateNormMatrix(config, syst);
                covar_names.push_back(syst.systname);
                ++n_covar;
            }else if(syst.mode == "external_covariance"){
                if(other_index == syst.binning){
                    //external covariances are only created for specific binning
                    this->LoadExternalCovarianceMatrix(config,syst);
                    covar_names.push_back(syst.systname);
                    ++n_covar;
                }
            }else if(syst.mode == "external_covariance_to_spline"){
                // Like covariance_to_spline, but the fractional covariance is read from an
                // external TMatrixD instead of being built from MC universes. Splines are added
                // unconditionally (keyed to syst.binning), matching covariance_to_spline.
                Eigen::MatrixXf frac_cov = LoadExternalFractionalCovariance(config, syst);
                size_t n_before = splines.size();
                FillSplinesFromCovarianceMatrix(frac_cov, syst);
                size_t n_after = splines.size();

                // Propagate parent tag + plotname to the synthesized knob entries (see
                // covariance_to_spline branch above for the rationale).
                PROconfig& mut_config = const_cast<PROconfig&>(config);
                auto parent_tags_it = mut_config.m_mcgen_variation_tags.find(syst.systname);
                auto parent_plotname_it = mut_config.m_mcgen_variation_plotname_map.find(syst.systname);
                for(size_t si = n_before; si < n_after; ++si) {
                    const std::string& knob_name = spline_names[si];
                    if(parent_tags_it != mut_config.m_mcgen_variation_tags.end()) {
                        mut_config.m_mcgen_variation_tags[knob_name] = parent_tags_it->second;
                    }
                    if(parent_plotname_it != mut_config.m_mcgen_variation_plotname_map.end()) {
                        const std::string suffix = knob_name.substr(syst.systname.size());
                        mut_config.m_mcgen_variation_plotname_map[knob_name] = parent_plotname_it->second + suffix;
                    }
                }
            }
        }

        spline_priors = Eigen::VectorXf::Constant(n_splines, 1);
        spline_centers = Eigen::VectorXf::Constant(n_splines, 0);
        for(const auto &[name, prior]: config.m_mcgen_variation_prior) {
            auto it = std::find(spline_names.begin(), spline_names.end(), name);
            if(it != std::end(spline_names)) {
                size_t idx = std::distance(std::begin(spline_names), it);
                spline_priors(idx) = prior;
            }
        }
        for(const auto &[name, center]: config.m_mcgen_variation_prior_centers) {
            auto it = std::find(spline_names.begin(), spline_names.end(), name);
            if(it != std::end(spline_names)) {
                size_t idx = std::distance(std::begin(spline_names), it);
                spline_centers(idx) = center;
            }
        }



        if(config.m_use_mcstats){
            Eigen::MatrixXf fractional_mcstat_cov =  prop.variable_mc_stat_err[other_index].array().square().inverse().matrix().asDiagonal();
            toFiniteMatrix(fractional_mcstat_cov);
            Eigen::MatrixXf mcstat_corr = GenerateCorrMatrix(fractional_mcstat_cov);
            // Register under the systematic's XML name (like every other covariance at
            // syst_map[syst.systname] above) so downstream tag/plotname lookups match its key.
            syst_map[config.m_mcstat_systname] = {covmat.size(), SystType::Covariance};
            covmat.push_back(fractional_mcstat_cov);
            corrmat.push_back(mcstat_corr);
            ++n_covar;
        }

        if(covmat.size()==0){
            int nbins =  config.m_num_variable_bins_total[other_index];
            Eigen::MatrixXf fracM = Eigen::MatrixXf::Zero(nbins, nbins);
            covmat.push_back(fracM);
        }

        log<LOG_INFO>(L"%1% || SumMatrix for index %2% ") % __func__ % other_index;
        fractional_covariance = this->SumMatrices();
        log<LOG_INFO>(L"%1% || SumMatrix for index %2% DONE ") % __func__ % other_index;
    }

    PROsyst PROsyst::subset(const std::vector<std::string> &systs) const {
        PROsyst ret;
        log<LOG_DEBUG>(L"%1% | Creating a subset with a list of %2% systematics.") % __func__ % systs.size();
        Eigen::VectorXf tmp_priors = spline_priors;
        Eigen::VectorXf tmp_centers = spline_centers;
        for(const std::string &name: systs) {
            log<LOG_DEBUG>(L"%1% | Looking up systematic %2% from subset list.") % __func__ % name.c_str();
            const auto &[idx, stype] = syst_map.at(name);
            log<LOG_DEBUG>(L"%1% | idx %2% ") % __func__ % idx ;

            switch(stype) {
                case SystType::Spline:
                    {
                        ret.syst_map[name] = std::make_pair(ret.splines.size(), SystType::Spline);
                        ret.spline_names.push_back(name);
                        Spline spline_copy;//Create explicit deep copy of the Spline
                        spline_copy.bins = splines[idx].bins;
                        spline_copy.segments_per_bin = splines[idx].segments_per_bin;
                        spline_copy.segments = splines[idx].segments;  // vector copy
                        ret.splines.push_back(std::move(spline_copy));
                        ret.spline_hi.push_back(spline_hi[idx]);
                        ret.spline_lo.push_back(spline_lo[idx]);
                        ret.spline_has_restrict.push_back(spline_has_restrict[idx]);
                        ret.spline_restrict_lo.push_back(spline_restrict_lo[idx]);
                        ret.spline_restrict_hi.push_back(spline_restrict_hi[idx]);
                        ret.spline_binnings.push_back(spline_binnings[idx]);
                        ret.spline_prior_types.push_back(spline_prior_types[idx]);
                        tmp_priors(ret.n_splines) = spline_priors(idx);
                        tmp_centers(ret.n_splines) = spline_centers(idx);
                        ++ret.n_splines;
                        break;
                    }
                case SystType::Covariance:
                    ret.syst_map[name] = std::make_pair(ret.covmat.size(), SystType::Covariance);
                    ret.covar_names.push_back(name);
                    ret.covmat.push_back(covmat[idx]);
                    ret.corrmat.push_back(corrmat[idx]);
                    ++ret.n_covar;
                    break;
                default:
                    log<LOG_ERROR>(L"%1% || Unrecognized syst type %2% for syst %3%.") % __func__ % static_cast<int>(stype) % name.c_str();
                    break;
            }
        }
        ret.spline_priors = tmp_priors.segment(0, ret.n_splines);
        ret.spline_centers = tmp_centers.segment(0, ret.n_splines);
        ret.fractional_covariance = ret.covmat.size() ? ret.SumMatrices()
            : Eigen::MatrixXf::Constant(fractional_covariance.rows(), fractional_covariance.cols(), 0.0f);
        ret.other_index = other_index;
        ret.cov2spline_debug_info = cov2spline_debug_info;
        log<LOG_DEBUG>(L"%1% | Done Subset.") % __func__ ;
        return ret;
    }

    PROsyst PROsyst::excluding(const std::vector<std::string> &systs) const {
        PROsyst ret;
        Eigen::VectorXf tmp_priors = spline_priors;
        Eigen::VectorXf tmp_centers = spline_centers;
        // Iterate in the ORIGINAL spline/covariance order, not syst_map's
        // alphabetical order: spline position defines the parameter-vector
        // layout, so a reordered copy would silently misalign any parameter or
        // seed vector built against this object.
        for(size_t idx = 0; idx < spline_names.size(); ++idx) {
            const std::string &name = spline_names[idx];
            if(std::find(systs.begin(), systs.end(), name) != systs.end()) continue;
            ret.syst_map[name] = std::make_pair(ret.splines.size(), SystType::Spline);
            ret.spline_names.push_back(name);
            Spline spline_copy;//Create explicit deep copy of the Spline
            spline_copy.bins = splines[idx].bins;
            spline_copy.segments_per_bin = splines[idx].segments_per_bin;
            spline_copy.segments = splines[idx].segments;  // vector copy
            ret.splines.push_back(std::move(spline_copy));
            ret.spline_hi.push_back(spline_hi[idx]);
            ret.spline_lo.push_back(spline_lo[idx]);
            ret.spline_has_restrict.push_back(spline_has_restrict[idx]);
            ret.spline_restrict_lo.push_back(spline_restrict_lo[idx]);
            ret.spline_restrict_hi.push_back(spline_restrict_hi[idx]);
            ret.spline_binnings.push_back(spline_binnings[idx]);
            ret.spline_prior_types.push_back(spline_prior_types[idx]);
            tmp_priors(ret.n_splines) = spline_priors(idx);
            tmp_centers(ret.n_splines) = spline_centers(idx);
            ++ret.n_splines;
        }
        for(size_t idx = 0; idx < covar_names.size(); ++idx) {
            const std::string &name = covar_names[idx];
            if(std::find(systs.begin(), systs.end(), name) != systs.end()) continue;
            ret.syst_map[name] = std::make_pair(ret.covmat.size(), SystType::Covariance);
            ret.covar_names.push_back(name);
            ret.covmat.push_back(covmat[idx]);
            ret.corrmat.push_back(corrmat[idx]);
            ++ret.n_covar;
        }
        ret.spline_priors = tmp_priors.segment(0, ret.n_splines);
        ret.spline_centers = tmp_centers.segment(0, ret.n_splines);
        ret.fractional_covariance = ret.covmat.size() ? ret.SumMatrices()
            : Eigen::MatrixXf::Constant(fractional_covariance.rows(), fractional_covariance.cols(), 0.0f);
        ret.other_index = other_index;
        ret.cov2spline_debug_info = cov2spline_debug_info;
        return ret;
    }

    PROsyst PROsyst::allsplines2cov(const PROconfig &config, const PROpeller &prop, const PROmodel &model, const Eigen::VectorXf &params, uint32_t seed) const {
        PROsyst ret;
        for(const auto &[name, spair]: syst_map) {

            const auto &[idx, stype] = spair;
            switch(stype) {
                case SystType::Spline: {
                                           ret.syst_map[name] = std::make_pair(ret.covmat.size(), SystType::Covariance);
                                           ret.covar_names.push_back(name);
                                           // Disjoint seed range per spline (spline2cov draws 500 throws at seed..seed+499).
                                           Eigen::MatrixXf cov = spline2cov(idx, config, prop, model,params, seed + (uint32_t)idx * 500u);
                                           Eigen::MatrixXf cor = GenerateCorrMatrix(cov);
                                           ret.covmat.push_back(cov);
                                           ret.corrmat.push_back(cor);
                                           ++ret.n_covar;
                                       } break;
                case SystType::Covariance:
                                       ret.syst_map[name] = std::make_pair(ret.covmat.size(), SystType::Covariance);
                                       ret.covar_names.push_back(name);
                                       ret.covmat.push_back(covmat[idx]);
                                       ret.corrmat.push_back(corrmat[idx]);
                                       ++ret.n_covar;
                                       break;
                default:
                                       log<LOG_ERROR>(L"%1% || Unrecognized syst type %2% for syst %3%.") % __func__ % static_cast<int>(stype) % name.c_str();
                                       break;
            }
        }
        ret.fractional_covariance = ret.covmat.size() ? ret.SumMatrices()
            : Eigen::MatrixXf::Constant(fractional_covariance.rows(), fractional_covariance.cols(), 0.0f);
        ret.other_index = other_index;
        return ret;
    }

    Eigen::MatrixXf PROsyst::spline2cov(int spline, const PROconfig &config, const PROpeller &prop, const PROmodel &model, const Eigen::VectorXf &params, uint32_t seed) const {
        Eigen::MatrixXf cv = FillSpectra(config, prop, *this, model, params , true, other_index).Spec();

        std::vector<Eigen::VectorXf> specs;
        // Distinct seed per throw: FillSplineRandomThrow now uses its seed
        // argument on every call (it used to hold a function-local static RNG
        // that ignored the seed after the first-ever call).
        for(size_t i = 0; i < 500; ++i){
            specs.push_back(FillSplineRandomThrow(config, prop, *this, model, params, spline, seed + (uint32_t)i, other_index).Spec());
        }

        int nbins = config.m_num_variable_bins_total[other_index];
        Eigen::MatrixXf mat(nbins, nbins);
        mat.setZero();
        for(const auto &spec: specs){
            for(int i = 0; i < cv.size(); ++i){
                for(int j = 0; j < cv.size(); ++j){
                    mat(i, j) += (cv(i)-spec(i))*(cv(j)-spec(j));
                }
            }
        }
        mat /= specs.size();

        Eigen::MatrixXf cv_inverse = cv.asDiagonal().inverse();
        Eigen::MatrixXf frac_covar_matrix = cv_inverse * mat * cv_inverse;
        PROsyst::toFiniteMatrix(frac_covar_matrix);

        return frac_covar_matrix;
    }

    Eigen::MatrixXf PROsyst::SumMatrices() const{

        Eigen::MatrixXf sum_matrix;


        if(covmat.size()){
            int nbins = (covmat.begin())->rows();

            sum_matrix = Eigen::MatrixXf::Zero(nbins, nbins);

            for(auto& p : covmat){
                sum_matrix += p;
            }

        }else{
            log<LOG_ERROR>(L"%1% || There is no covariance available!") % __func__;
            log<LOG_ERROR>(L"%1% || Returning empty matrix") % __func__;

        }
        return sum_matrix;
    }

    Eigen::MatrixXf PROsyst::SumMatrices(const std::vector<std::string>& sysnames) const{

        Eigen::MatrixXf sum_matrix;
        if(covmat.size()){
            int nbins = (covmat.begin())->rows();

            sum_matrix = Eigen::MatrixXf::Zero(nbins, nbins);
        }
        else{
            log<LOG_ERROR>(L"%1% || There is no covariance available!!") % __func__;
            log<LOG_ERROR>(L"%1% || Returning empty matrix") % __func__;
            return sum_matrix;
        }


        for(auto& sys : sysnames){
            if(syst_map.find(sys) == syst_map.end() || syst_map.at(sys).second != SystType::Covariance){
                log<LOG_INFO>(L"%1% || No matrix in the map matches with name %2%, Skip") % __func__ % sys.c_str();
            }else{
                sum_matrix += covmat.at(syst_map.at(sys).first);
            }
        }

        return sum_matrix;
    }

    void PROsyst::CreateMatrix(const SystStruct& syst){

        std::string sysname = syst.GetSysName();

        //generate matrix only if it's not already in the map
        if(syst_map.find(sysname) == syst_map.end()){
            std::pair<Eigen::MatrixXf, Eigen::MatrixXf> matrices = PROsyst::GenerateCovarMatrices(syst);

            // If inflate is set, scale the covariance by inflate^2 (uncertainty scales by inflate).
            // The correlation matrix is unchanged by a constant scaling.
            if(syst.inflate != 1.0f) {
                log<LOG_INFO>(L"%1% || Applying inflate=%2% (covariance x %3%) for systematic %4%") % __func__ % syst.inflate % (syst.inflate * syst.inflate) % sysname.c_str();
                matrices.first *= syst.inflate * syst.inflate;
            }

            syst_map[sysname] = {covmat.size(), SystType::Covariance};
            covmat.push_back(matrices.first);
            corrmat.push_back(matrices.second);

        }

        return;
    }

    void PROsyst::CreateFlatMatrix(const PROconfig &config, const SystStruct& syst){
        std::string sysname = syst.GetSysName();
        log<LOG_INFO>(L"%1% || Generating a FLAT norm covariance matrix.") % __func__ ;
        int nbins = config.m_num_variable_bins_total[other_index];
        Eigen::MatrixXf fracM = Eigen::MatrixXf::Zero(nbins, nbins);
        Eigen::MatrixXf corrM = Eigen::MatrixXf::Identity(nbins, nbins);

        // Split on the LAST colon: the percent never contains one, and the
        // pattern may (regex constructs like (?:...) or [[:alpha:]]).
        size_t colonPos = sysname.rfind(':');
        if (colonPos == std::string::npos) {
            log<LOG_ERROR>(L"%1% || ERROR, you asked for a flat systematic but its not in NAME:percentate format %2%") % __func__  % sysname.c_str();
            exit(EXIT_FAILURE);
        }

        std::string wild = sysname.substr(0, colonPos);
        std::string sflat_percent  = sysname.substr(colonPos + 1);
        float flat_percent = std::stof(sflat_percent);


        log<LOG_INFO>(L"%1% || Wildcard %2% (and percent %3%) which matches: ") % __func__  % wild.c_str() % flat_percent;
        // Unanchored regex (plain substrings behave as before); see PROconfig.h.
        std::vector<std::string> flatnames = MatchNames(config.m_fullnames, wild, "flat systematic '" + sysname + "'");
        if(flatnames.empty()) {
            log<LOG_ERROR>(L"%1% || ERROR: flat systematic '%2%' pattern '%3%' matches NO subchannel fullname. Fullnames are <mode>_<detector>_<channel>_<subchannel>; matching is an unanchored regex (plain substrings work).") % __func__ % sysname.c_str() % wild.c_str();
            exit(EXIT_FAILURE);
        }
        log<LOG_INFO>(L"%1% || %2% . ") % __func__  % flatnames;

        std::vector<size_t> flatbins;
        for(auto &name: flatnames){
            size_t is = config.GetSubchannelIndex(name);     
            size_t ic = config.GetLocalChannelIndexFromGlobalSubchannelIndex(is);     


            size_t start = config.GetGlobalVariableBinStart(is, other_index);
            for(size_t b = 0; b < config.m_channel_variable_bins[ic][other_index].NBins(); b++) {
                fracM(start+b,start+b)=flat_percent*flat_percent;
                flatbins.push_back(start+b);
            }
        }
        log<LOG_INFO>(L"%1% || and fills bins  %2%  .") % __func__  %  flatbins;

        syst_map[sysname] = {covmat.size(), SystType::Covariance};

        covmat.push_back(fracM);
        corrmat.push_back(corrM);

        return;
    }

    void PROsyst::CreateNormMatrix(const PROconfig &config, const SystStruct& syst){
        std::string sysname = syst.GetSysName();
        log<LOG_INFO>(L"%1% || Generating a correlated normalization covariance matrix for %2%.") % __func__ % sysname.c_str();

        int nbins = config.m_num_variable_bins_total[other_index];
        Eigen::MatrixXf fracM = Eigen::MatrixXf::Zero(nbins, nbins);
        Eigen::MatrixXf corrM = Eigen::MatrixXf::Zero(nbins, nbins);
        const float covariance = syst.norm_value * syst.norm_value * syst.inflate * syst.inflate;

        for(int row : syst.norm_bins){
            for(int col : syst.norm_bins){
                fracM(row, col) = covariance;
                corrM(row, col) = 1.0;
            }
        }

        syst_map[sysname] = {covmat.size(), SystType::Covariance};
        covmat.push_back(fracM);
        corrmat.push_back(corrM);
    }

    Eigen::MatrixXf PROsyst::LoadExternalFractionalCovariance(const PROconfig &config, const SystStruct& syst){
        //this is matrix name
        std::string matrixname = syst.GetSysName();
        std::string filename = syst.external_filename;
        log<LOG_INFO>(L"%1% || Loading a TMatrix from %2% named %3%") % __func__ % filename.c_str() % matrixname.c_str();
        // The external matrix is generated in the systematic's own binning, so size against syst.binning.
        // (For the "external_covariance" mode this is gated to equal other_index by the caller.)
        int nbins = config.m_num_variable_bins_total[syst.binning];
        Eigen::MatrixXf fracM = Eigen::MatrixXf::Zero(nbins, nbins);

        TFile* file = TFile::Open(filename.c_str(), "READ");
        if(!file || file->IsZombie()){
            log<LOG_ERROR>(L"%1% || Failed to open file: %2%") % __func__ % filename.c_str();
            exit(EXIT_FAILURE);
        }

        TMatrixD* tmatrix = dynamic_cast<TMatrixD*>(file->Get(matrixname.c_str()));
        if(!tmatrix){
            log<LOG_ERROR>(L"%1% || Failed to load TMatrixD named %2% from file %3%") % __func__ % matrixname.c_str() % filename.c_str();
            file->Close();
            delete file;
            exit(EXIT_FAILURE);
        }

        // Check dimensions match expected size
        int nrows = tmatrix->GetNrows();
        int ncols = tmatrix->GetNcols();
        if(nrows != nbins || ncols != nbins){
            log<LOG_ERROR>(L"%1% || Dimension mismatch: TMatrixD is %2%x%3% but expected %4%x%4%") 
                % __func__ % nrows % ncols % nbins;
            file->Close();
            delete file;
            exit(EXIT_FAILURE);
        }

        // Copy TMatrixD values into Eigen matrix
        for(int i = 0; i < nbins; ++i){
            for(int j = 0; j < nbins; ++j){
                fracM(i, j) = static_cast<float>((*tmatrix)(i, j));
            }
        }

        // Clean up ROOT objects
        file->Close();
        delete file;

        // Zero out any NaN or infinite values
        PROsyst::toFiniteMatrix(fracM);

        // Check if matrix is positive semi-definite
        if(!PROsyst::isPositiveSemiDefinite_WithTolerance(fracM, 2.0*Eigen::NumTraits<float>::dummy_precision())){
            log<LOG_WARNING>(L"%1% || External covariance matrix %2% is not positive semi-definite!")
                % __func__ % matrixname.c_str();
        }

        log<LOG_INFO>(L"%1% || Successfully loaded %2%x%3% covariance matrix %4%")
            % __func__ % nbins % nbins % matrixname.c_str();

        return fracM;
    }

    void PROsyst::LoadExternalCovarianceMatrix(const PROconfig &config, const SystStruct& syst){
        std::string matrixname = syst.GetSysName();
        Eigen::MatrixXf fracM = LoadExternalFractionalCovariance(config, syst);

        // If inflate is set, scale the covariance by inflate^2 (uncertainty scales by inflate).
        if(syst.inflate != 1.0f) {
            log<LOG_INFO>(L"%1% || Applying inflate=%2% (covariance x %3%) for systematic %4%") % __func__ % syst.inflate % (syst.inflate * syst.inflate) % matrixname.c_str();
            fracM *= syst.inflate * syst.inflate;
        }

        // Generate correlation matrix from fractional covariance
        Eigen::MatrixXf corrM = PROsyst::GenerateCorrMatrix(fracM);

        syst_map[matrixname] = {covmat.size(), SystType::Covariance};
        covmat.push_back(fracM);
        corrmat.push_back(corrM);

        return;
    }

    std::pair<Eigen::MatrixXf, Eigen::MatrixXf>  PROsyst::GenerateCovarMatrices(const SystStruct& sys_obj){
        //get fractional covar
        Eigen::MatrixXf frac_covar_matrix = PROsyst::GenerateFracCovarMatrix(sys_obj);

        //get fractional covariance matrix
        Eigen::MatrixXf corr_covar_matrix = PROsyst::GenerateCorrMatrix(frac_covar_matrix);

        return std::pair<Eigen::MatrixXf, Eigen::MatrixXf>({frac_covar_matrix, corr_covar_matrix});
    }

    Eigen::MatrixXf PROsyst::GenerateFullCovarMatrix(const SystStruct& sys_obj){
        int n_universe = sys_obj.GetNUniverse(); 
        std::string sys_name = sys_obj.GetSysName();

        const PROspec& cv_spec = sys_obj.CV();
        int nbins = cv_spec.GetNbins();
        float cv_integral = cv_spec.Spec().sum(); 

        //build full covariance matrix 
        Eigen::MatrixXf full_covar_matrix = Eigen::MatrixXf::Zero(nbins, nbins);
        for(int i = 0; i != n_universe; ++i){

            PROspec spec_diff;
            if(shape_only){
                spec_diff = cv_spec - sys_obj.Variation(i)*(cv_integral/sys_obj.Variation(i).Spec().sum());
            }else{
                spec_diff = cv_spec - sys_obj.Variation(i);
            }
            full_covar_matrix += (spec_diff.Spec() * spec_diff.Spec().transpose() ) / static_cast<float>(n_universe);
        }

        return full_covar_matrix;
    }

    Eigen::MatrixXf PROsyst::GenerateFracCovarMatrix(const SystStruct& sys_obj){

        //build full covariance matrix 
        Eigen::MatrixXf full_covar_matrix = PROsyst::GenerateFullCovarMatrix(sys_obj);

        //build fractional covariance matrix 
        //first, get the matrix with diagonal being reciprocal of CV spectrum prdiction
        const PROspec& cv_spec = sys_obj.CV();
        int nbins = cv_spec.GetNbins();
        Eigen::MatrixXf cv_spec_matrix =  Eigen::MatrixXf::Identity(nbins, nbins);
        for(int i =0; i != nbins; ++i)
            cv_spec_matrix(i, i) = 1.0/cv_spec.GetBinContent(i);

        //second, get fractioal covar
        Eigen::MatrixXf frac_covar_matrix = cv_spec_matrix * full_covar_matrix * cv_spec_matrix;

        //zero out nans 
        PROsyst::toFiniteMatrix(frac_covar_matrix);

        //check if it's good
        if(!PROsyst::isPositiveSemiDefinite_WithTolerance(frac_covar_matrix,10.0*Eigen::NumTraits<float>::dummy_precision())){
            log<LOG_ERROR>(L"%1% || Fractional Covariance Matrix is not positive semi-definite!") % __func__;
            log<LOG_ERROR>(L"Terminating.");
            log<LOG_ERROR>(L" Matrix is %1% .") % frac_covar_matrix;
            exit(EXIT_FAILURE);
        }

        return frac_covar_matrix;
    }

    Eigen::MatrixXf PROsyst::GenerateCorrMatrix(const Eigen::MatrixXf& frac_matrix){
        int nbins = frac_matrix.rows();
        Eigen::MatrixXf corr_covar_matrix = frac_matrix;

        Eigen::MatrixXf error_reciprocal_matrix = Eigen::MatrixXf::Zero(nbins, nbins);
        for(int i = 0; i != nbins; ++i){
            if(frac_matrix(i,i) != 0){
                float temp = sqrt(frac_matrix(i,i));
                error_reciprocal_matrix(i,i) = 1.0/temp;
            }
            else
                error_reciprocal_matrix(i,i) = 1.0;
        }


        corr_covar_matrix = error_reciprocal_matrix * corr_covar_matrix * error_reciprocal_matrix;

        //zero out nans 
        PROsyst::toFiniteMatrix(corr_covar_matrix);
        return corr_covar_matrix;
    }


    void PROsyst::toFiniteMatrix(Eigen::MatrixXf& in_matrix){
        if(!PROsyst::isFiniteMatrix(in_matrix)){
            log<LOG_DEBUG>(L"%1% || Changing Nan/inf values to 0.0") % __func__;
            in_matrix = in_matrix.unaryExpr([](float v) -> float { return std::isfinite(v) ? v : 0.0f; });

        }
        return;
    }

    bool PROsyst::isFiniteMatrix(const Eigen::MatrixXf& in_matrix){

        //check for nan and infinite
        if(!in_matrix.allFinite()){
            log<LOG_DEBUG>(L"%1% || Matrix has Nan or non-finite values.") % __func__ ;
            return false;
        }
        return true;
    }

    bool PROsyst::isPositiveSemiDefinite(const Eigen::MatrixXf& in_matrix){

        //first, check if it's symmetric 
        if(!in_matrix.isApprox(in_matrix.transpose(), 10.0f*Eigen::NumTraits<float>::dummy_precision())){
            log<LOG_ERROR>(L"%1% || Covariance matrix is not symmetric, with tolerance of %2%") % __func__ % float(10.0f*Eigen::NumTraits<float>::dummy_precision());
            return false;
        }

        //second, check if it's positive semi-definite;
        Eigen::LDLT<Eigen::MatrixXf> llt(in_matrix);
        if((llt.info() == Eigen::NumericalIssue ) || (!llt.isPositive()) )
            return false;

        return true;

    }

    bool PROsyst::isPositiveSemiDefinite_WithTolerance(const Eigen::MatrixXf& in_matrix, float tolerance ){

        //first, check if it's symmetric 
        if(!in_matrix.isApprox(in_matrix.transpose(), Eigen::NumTraits<float>::dummy_precision())){
            log<LOG_ERROR>(L"%1% || Covariance matrix is not symmetric, with tolerance of %2%") % __func__ % Eigen::NumTraits<float>::dummy_precision();
            return false;
        }


        //second, check if it's positive semi-definite;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(in_matrix);
        if(eigensolver.info() != Eigen::Success){
            log<LOG_ERROR>(L"%1% || Failing to get eigenvalues..") % __func__ ;
            return false;
        }

        Eigen::VectorXf eigenvals = eigensolver.eigenvalues();
        for(auto val : eigenvals ){
            if(val < 0 && fabs(val) > tolerance){
                log<LOG_ERROR>(L"%1% || Matrix is not PSD. Found negative eigenvalues beyond tolerance (%2%): %3%") % __func__ % tolerance % val;
                return false;
            }
        }
        return true;

    }


    float PROsyst::GetSplineShift(int spline_num, float shift, int bin) const {
        const Spline& spline = splines[spline_num];
        if (bin < 0 || bin >= spline.bins) return -1;

        // Find the right segment; segments are sorted by knob value.
        int offset = bin * spline.segments_per_bin;
        const SplineSegment* segs = &spline.segments[offset];
        const SplineSegment* end = segs + spline.segments_per_bin;
        const SplineSegment* seg = std::upper_bound(segs, end, shift, [](float x, const SplineSegment& s) { return x < s.knot; });
        if (seg != segs) --seg; // seg is now the last k_i such that k_i < shift

        float hi = seg + 1 < end ? seg[1].knot : spline_hi[spline_num];
        float x = (shift - seg->knot) / (hi - seg->knot); // normalize to [0, 1]
        const auto& c = seg->coeffs;
        return c[0] + x*(c[1] + x*(c[2] + x*c[3]));
    }

    float PROsyst::GetSplineShiftDeriv(int spline_num, float shift, int bin) const {
        const Spline& spline = splines[spline_num];
        if (bin < 0 || bin >= spline.bins) return 0;

        // Same segment lookup as GetSplineShift so value and derivative always
        // come from the same cubic.
        int offset = bin * spline.segments_per_bin;
        const SplineSegment* segs = &spline.segments[offset];
        const SplineSegment* end = segs + spline.segments_per_bin;
        const SplineSegment* seg = std::upper_bound(segs, end, shift, [](float x, const SplineSegment& s) { return x < s.knot; });
        if (seg != segs) --seg;

        float hi = seg + 1 < end ? seg[1].knot : spline_hi[spline_num];
        float width = hi - seg->knot;
        float x = (shift - seg->knot) / width;
        const auto& c = seg->coeffs;
        // d/dshift of c0 + x(c1 + x(c2 + x c3)) with x = (shift-knot)/width
        return (c[1] + x*(2.0f*c[2] + 3.0f*x*c[3])) / width;
    }

    void PROsyst::FillSpline(const SystStruct& syst, bool unmirrored) {
        std::vector<PROspec> ratios;
        ratios.reserve(syst.p_multi_spec.size());
        float cv_integral = syst.p_cv->Spec().sum();

        bool found0 = false;
        int knob0_index = -1;  // Index of knobval=0 in ratios vector

        std::vector<float> knobvals;
        for (size_t i = 0; i < syst.p_multi_spec.size(); ++i) {
            if (syst.knobval[i] > 0 && !found0) {
                ratios.push_back(*syst.p_cv / *syst.p_cv);
                knobvals.push_back(0);
                knob0_index = ratios.size() - 1;
                found0 = true;
            }
            if (syst.knobval[i] == 0) {
                found0 = true;
                knob0_index = ratios.size();  // Will be set after push_back below
            }

            float mod = shape_only ? cv_integral / syst.p_multi_spec[i]->Spec().sum() : 1.0;
            /*
               if (mod < 0) {
               log<LOG_ERROR>(L"%1% || Spline shift weight is negative with value %2% for systematic %3%") % __func__ % mod % syst.systname.c_str();
               log<LOG_ERROR>(L"Terminating.");
               exit(EXIT_FAILURE);
               }
               */
            ratios.push_back(((*syst.p_multi_spec[i]) * mod) / *syst.p_cv);
            knobvals.push_back(syst.knobval[i]);
        }
        if (!found0) {
            ratios.push_back(*syst.p_cv / *syst.p_cv);
            knobvals.push_back(0);
            knob0_index = ratios.size() - 1;
        }

        // If force_0_cv is set, normalize all ratios by the ratio at knob=0
        // This ensures that at shift=0, the spline returns exactly 1.0 (no change to CV)
        if (syst.force_0_cv && knob0_index >= 0) {
            log<LOG_INFO>(L"%1% || Applying force_0_cv normalization for systematic %2%") % __func__ % syst.systname.c_str();
            // IMPORTANT: Make a copy, not a reference! Otherwise we modify the divisor during the loop.
            PROspec ratio_at_0 = ratios[knob0_index];
            for (size_t i = 0; i < ratios.size(); ++i) {
                ratios[i] = ratios[i] / ratio_at_0;
            }
        }

        // If inflate is set, scale the spline shifts about 1 (ratio -> 1 + inflate*(ratio - 1))
        // before interpolation, inflating the uncertainty while keeping the no-shift value of 1 fixed.
        if (syst.inflate != 1.0f) {
            log<LOG_INFO>(L"%1% || Applying inflate=%2% to spline shifts for systematic %3%") % __func__ % syst.inflate % syst.systname.c_str();
            for (PROspec& ratio : ratios) {
                Eigen::VectorXf& v = ratio.Spec();
                v = (1.0f + syst.inflate * (v.array() - 1.0f)).matrix();
            }
        }

        int nbins = syst.p_cv->GetNbins();
        Spline spline;
        spline.bins = nbins;
        spline.segments_per_bin = knobvals.size(); 

        if(syst.knobval.size() != syst.p_multi_spec.size()){
            log<LOG_ERROR>(L"%1% || number of knobvals specified (%2%) does not match number of weight universes in file (%3%) for systematic %4%!") % __func__ % syst.knobval.size() % syst.p_multi_spec.size() % syst.systname.c_str();
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }

        std::vector<SplineSegment> all_segments;

        for (int i = 0; i < nbins; ++i) {
            std::vector<SplineSegment> bin_segments;

            // This comment is copy-pasted from CAFAna:
            // This is cubic interpolation. For each adjacent set of four points we
            // determine coefficients for a cubic which will be the curve between the
            // center two. We constrain the function to match the two center points
            // and to have the right mean gradient at them. This causes this patch to
            // match smoothly with the next one along. The resulting function is
            // continuous and first and second differentiable. At the ends of the
            // range we fit a quadratic instead with only one constraint on the
            // slope. The coordinate conventions are that point y1 sits at x=0 and y2
            // at x=1. The matrices are simply the inverses of writing out the
            // constraints expressed above.


            if (ratios.size() < 3) {
                const float y1 = ratios[0].GetBinContent(i);
                const float y2 = ratios[1].GetBinContent(i);
                const float slope = (y2 - y1) / (knobvals[1] - knobvals[0]);
                if(unmirrored)
                    bin_segments.push_back(SplineSegment{(float)(-knobvals[1]), {slope * (-knobvals[1]) + y1, slope, 0, 0}});
                else
                    bin_segments.push_back(SplineSegment{(float)(-knobvals[1]), {y2, -slope, 0, 0}});
                bin_segments.push_back(SplineSegment{(float)knobvals[0], {slope * (float)knobvals[0] + y1, slope, 0, 0}});
            } else {
                {
                    const float y1 = ratios[0].GetBinContent(i);
                    const float y2 = ratios[1].GetBinContent(i);
                    const float y3 = ratios[2].GetBinContent(i);
                    const Eigen::Vector3f v{y1, y2, (y3 - y1) / 2};
                    const Eigen::Matrix3f m{{1, -1, 1},
                        {-2, 2, -1},
                        {1, 0, 0}};
                    const Eigen::Vector3f res = m * v;
                    bin_segments.push_back(SplineSegment{(float)knobvals[0], {res(2), res(1), res(0), 0}});
                }
                for (unsigned int shiftIdx = 1; shiftIdx < ratios.size() - 2; ++shiftIdx) {
                    const float y0 = ratios[shiftIdx - 1].GetBinContent(i);
                    const float y1 = ratios[shiftIdx].GetBinContent(i);
                    const float y2 = ratios[shiftIdx + 1].GetBinContent(i);
                    const float y3 = ratios[shiftIdx + 2].GetBinContent(i);
                    const Eigen::Vector4f v{y1, y2, (y2 - y0) / 2, (y3 - y1) / 2};
                    const Eigen::Matrix4f m{{2, -2, 1, 1},
                        {-3, 3, -2, -1},
                        {0, 0, 1, 0},
                        {1, 0, 0, 0}};
                    const Eigen::Vector4f res = m * v;
                    float knobval = knobvals[shiftIdx];
                    if (!found0 && knobval >= 0)
                        knobval = knobvals[shiftIdx] == 1 ? 0 : knobvals[shiftIdx - 1];
                    bin_segments.push_back(SplineSegment{knobval, {res(3), res(2), res(1), res(0)}});
                }
                {
                    const float y4 = ratios[ratios.size() - 3].GetBinContent(i);
                    const float y5 = ratios[ratios.size() - 2].GetBinContent(i);
                    const float y6 = ratios[ratios.size() - 1].GetBinContent(i);
                    const Eigen::Vector3f vp{y5, y6, (y6 - y4) / 2};
                    const Eigen::Matrix3f mp{{-1, 1, -1},
                        {0, 0, 1},
                        {1, 0, 0}};
                    const Eigen::Vector3f resp = mp * vp;
                    bin_segments.push_back(SplineSegment{(float)knobvals[knobvals.size() - 2], {resp(2), resp(1), resp(0), 0}});
                }
            }

            all_segments.insert(all_segments.end(), bin_segments.begin(), bin_segments.end());
        }

        // If all bins have the same number of segments, keep as knobvals.size(); else update
        spline.segments_per_bin = all_segments.size() / nbins;
        spline.segments = std::move(all_segments);

        syst_map[syst.systname] = {splines.size(), SystType::Spline};
        splines.push_back(std::move(spline));
        spline_names.push_back(syst.systname);
        spline_lo.push_back(knobvals[0]);
        spline_hi.push_back(knobvals.back());
        spline_has_restrict.push_back(syst.has_restrict);
        spline_restrict_lo.push_back(syst.restrict_lo);
        spline_restrict_hi.push_back(syst.restrict_hi);
        spline_binnings.push_back(syst.binning);
        spline_prior_types.push_back(SplinePriorType::Gaussian);

    }

    void PROsyst::FillSplinesFromCovariance(const SystStruct& syst) {
        Eigen::MatrixXf frac_cov = PROsyst::GenerateFracCovarMatrix(syst);
        FillSplinesFromCovarianceMatrix(frac_cov, syst);
    }

    void PROsyst::FillSplinesFromCovarianceMatrix(Eigen::MatrixXf frac_cov, const SystStruct& syst) {
        // Capture pre-symmetrization asymmetry as a sanity number for debug plots.
        const float pre_symm_asymmetry = (frac_cov - frac_cov.transpose()).norm();
        // symmetrize to kill any float-asymmetry before eigendecomposition
        frac_cov = 0.5f * (frac_cov + frac_cov.transpose());

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(frac_cov);
        if(solver.info() != Eigen::Success) {
            log<LOG_ERROR>(L"%1% || Eigendecomposition failed for covariance_to_spline systematic %2%") % __func__ % syst.systname.c_str();
            exit(EXIT_FAILURE);
        }

        const Eigen::VectorXf eigvals = solver.eigenvalues();  // ascending
        const Eigen::MatrixXf eigvecs = solver.eigenvectors(); // columns

        const int nbins = frac_cov.rows();
        // List strictly-positive eigenvalues in descending order (near-zero/negative are noise).
        const float tol = 10.0f * Eigen::NumTraits<float>::dummy_precision();
        std::vector<int> positive_indices; // indices into eigvals, descending eigenvalue
        for(int i = nbins - 1; i >= 0; --i) {
            if(eigvals(i) > tol) positive_indices.push_back(i);
        }
        const int n_pos = static_cast<int>(positive_indices.size());
        int K = n_pos;
        if(syst.num_decomp_knobs > 0 && syst.num_decomp_knobs < n_pos) {
            K = syst.num_decomp_knobs;
        }
        if(K == 0) {
            log<LOG_WARNING>(L"%1% || covariance_to_spline systematic %2% has no positive-eigenvalue modes; skipping.") % __func__ % syst.systname.c_str();
            return;
        }
        // Top-K positive eigenpairs become spline knobs; the remaining positive eigenpairs
        // (if any, and if requested) are folded into a residual covariance.
        std::vector<int> kept_indices(positive_indices.begin(), positive_indices.begin() + K);
        std::vector<int> residual_indices(positive_indices.begin() + K, positive_indices.end());

        log<LOG_INFO>(L"%1% || covariance_to_spline %2%: keeping %3% of %4% eigenvectors as splines (largest eigenvalue=%5%, smallest kept=%6%), %7% residual mode(s)")
            % __func__ % syst.systname.c_str() % K % nbins % eigvals(nbins - 1) % eigvals(kept_indices[K - 1]) % static_cast<int>(residual_indices.size());

        const float lo = -3.0f, hi = 3.0f;
        const int n_segments = 6; // knots at -3, -2, -1, 0, 1, 2

        Cov2SplineDebugInfo dbg;
        dbg.original_frac_cov = frac_cov;
        dbg.pre_symm_asymmetry = pre_symm_asymmetry;
        dbg.eigenvalues = eigvals;
        dbg.eigenvectors = eigvecs;
        dbg.kept_indices = kept_indices;
        dbg.binning = syst.binning;
        dbg.knob_names.reserve(K);

        for(int k = 0; k < K; ++k) {
            const int ev_idx = kept_indices[k];
            const float lambda = eigvals(ev_idx);
            const float sqrt_lambda = std::sqrt(lambda);
            const Eigen::VectorXf vec = eigvecs.col(ev_idx);

            // Per-bin fractional response to knob x is: alpha_b * x, where alpha_b = sqrt(lambda) * vec[b].
            // Full ratio: f_b(x) = 1 + alpha_b * x.
            // Represent as 6 unit-width linear segments with knots at -3..2 so GetSplineShift's clamp works.
            Spline spline;
            spline.bins = nbins;
            spline.segments_per_bin = n_segments;
            spline.segments.reserve(static_cast<size_t>(nbins) * n_segments);
            for(int b = 0; b < nbins; ++b) {
                const float alpha = sqrt_lambda * vec(b);
                for(int s = 0; s < n_segments; ++s) {
                    const float knot = static_cast<float>(s - 3); // -3..2
                    const float c0 = 1.0f + alpha * knot;
                    const float c1 = alpha;
                    spline.segments.push_back(SplineSegment{knot, {c0, c1, 0.0f, 0.0f}});
                }
            }

            const std::string knob_name = syst.systname + "_decomp_knob_" + std::to_string(k);
            syst_map[knob_name] = {splines.size(), SystType::Spline};
            splines.push_back(std::move(spline));
            spline_names.push_back(knob_name);
            spline_prior_types.push_back(SplinePriorType::Gaussian);
            spline_lo.push_back(lo);
            spline_hi.push_back(hi);
            // Keep the restrict bookkeeping vectors the SAME length as splines/spline_lo,
            // otherwise per-spline loops (e.g. pseudo-experiment throws) read OOB.
            spline_has_restrict.push_back(false);
            spline_restrict_lo.push_back(lo);
            spline_restrict_hi.push_back(hi);
            spline_binnings.push_back(syst.binning);
            dbg.knob_names.push_back(knob_name);
            ++n_splines;
        }

        // Fold the un-kept (smaller) positive eigenpairs into a residual covariance so the
        // rank-K truncation does not throw away variance. Default on; disabled with
        // include_resid_cov="false" in the XML.
        // Only register the matrix when this PROsyst's binning matches the systematic's binning,
        // mirroring external_covariance: covmat entries are summed assuming a common dimension,
        // while the splines above stay binning-tagged and are safe across variables.
        if(syst.include_resid_cov && !residual_indices.empty() && other_index == syst.binning) {
            Eigen::MatrixXf residual_cov = Eigen::MatrixXf::Zero(nbins, nbins);
            for(int idx : residual_indices) {
                const float lambda = eigvals(idx);
                const Eigen::VectorXf vec = eigvecs.col(idx);
                residual_cov.noalias() += lambda * (vec * vec.transpose());
            }
            toFiniteMatrix(residual_cov);

            const std::string resid_name = syst.systname + "_resid_cov";
            syst_map[resid_name] = {covmat.size(), SystType::Covariance};
            covmat.push_back(residual_cov);
            corrmat.push_back(GenerateCorrMatrix(residual_cov));
            covar_names.push_back(resid_name);
            ++n_covar;

            dbg.has_residual = true;
            dbg.n_residual_modes = static_cast<int>(residual_indices.size());
            dbg.residual_cov = std::move(residual_cov);
            dbg.residual_cov_name = resid_name;

            log<LOG_INFO>(L"%1% || covariance_to_spline %2%: retained %3% residual mode(s) as covariance %4%")
                % __func__ % syst.systname.c_str() % dbg.n_residual_modes % resid_name.c_str();
        } else if(!residual_indices.empty()) {
            if(!syst.include_resid_cov) {
                log<LOG_INFO>(L"%1% || covariance_to_spline %2%: dropping %3% residual mode(s) (include_resid_cov=false)")
                    % __func__ % syst.systname.c_str() % static_cast<int>(residual_indices.size());
            } else {
                log<LOG_INFO>(L"%1% || covariance_to_spline %2%: residual covariance not registered for binning %3% (built only at the systematic's binning %4%)")
                    % __func__ % syst.systname.c_str() % other_index % syst.binning;
            }
        }

        cov2spline_debug_info[syst.systname] = std::move(dbg);
    }

    float PROsyst::GetSplineShift(std::string name, float shift, int bin) const {
        if(syst_map.count(name) == 0) {
            log<LOG_ERROR>(L"%1% || Unrecognized systematic %2%") % __func__ % name.c_str();
            return 1;
        }
        return GetSplineShift(syst_map.at(name).first, shift, bin);
    }

    PROspec PROsyst::GetSplineShiftedSpectrum(const PROconfig& config, const PROpeller& prop, std::string name, float shift) const {
        int nbins = config.m_num_variable_bins_total[other_index];
        int binning = spline_binnings[syst_map.at(name).first];
        PROspec ret(nbins);
        for(size_t i = 0; i < prop.NEvent(); ++i) {
            const int spline_bin = prop.VariableBinIndex(binning, i);
            const int reco_bin = prop.VariableBinIndex(other_index, i);
            ret.Fill(reco_bin, GetSplineShift(name, shift, spline_bin) * prop.added_weights[i]);
        }
        return ret;
    }

    PROspec PROsyst::GetSplineShiftedSpectrum(const PROconfig& config, const PROpeller& prop, int syst_num, float shift) const {
        int nbins = config.m_num_variable_bins_total[other_index];
        int binning = spline_binnings[syst_num];
        PROspec ret(nbins);
        for(size_t i = 0; i < prop.NEvent(); ++i) {
            const int spline_bin = prop.VariableBinIndex(binning, i);
            const int reco_bin = prop.VariableBinIndex(other_index, i);
            ret.Fill(reco_bin, GetSplineShift(syst_num, shift, spline_bin) * prop.added_weights[i]);
        }
        return ret;
    }

    PROspec PROsyst::GetSplineShiftedSpectrum(const PROconfig& config, const PROpeller& prop, std::vector<std::string> names, std::vector<float> shifts) const {
        assert(names.size() == shifts.size());
        int nbins = config.m_num_variable_bins_total[other_index];
        PROspec ret(nbins);
        for(size_t i = 0; i < prop.NEvent(); ++i) {
            const int reco_bin = prop.VariableBinIndex(other_index, i);
            float weight = 1;
            for(size_t j = 0; j < names.size(); ++j) {
                int binning = spline_binnings[syst_map.at(names[j]).first];
                const int spline_bin = prop.VariableBinIndex(binning, i);
                weight *= GetSplineShift(names[j], shifts[j], spline_bin);
            }
            ret.Fill(reco_bin, weight * prop.added_weights[i]);
        }
        return ret;
    }

    PROspec PROsyst::GetSplineShiftedSpectrum(const PROconfig& config, const PROpeller& prop, std::vector<int> syst_nums, std::vector<float> shifts) const {
        assert(syst_nums.size() == shifts.size());
        int nbins = config.m_num_variable_bins_total[other_index];
        PROspec ret(nbins);
        for(size_t i = 0; i < prop.NEvent(); ++i) {
            const int reco_bin = prop.VariableBinIndex(other_index, i);
            float weight = 1;
            for(size_t j = 0; j < syst_nums.size(); ++j) {
                int binning = spline_binnings[syst_nums[j]];
                const int spline_bin = prop.VariableBinIndex(binning, i);
                weight *= GetSplineShift(syst_nums[j], shifts[j], spline_bin);
            }
            ret.Fill(reco_bin, weight * prop.added_weights[i]);
        }
        return ret;
    }

    PROspec PROsyst::GetSplineShiftedSpectrum(const PROconfig& config, const PROpeller& prop, std::vector<float> shifts) const {
        assert(shifts.size() == splines.size());
        int nbins = config.m_num_variable_bins_total[other_index];
        PROspec ret(nbins);
        for(size_t i = 0; i < prop.NEvent(); ++i) {
            const int reco_bin = prop.VariableBinIndex(other_index, i);
            float weight = 1;
            for(size_t j = 0; j < shifts.size(); ++j) {
                int binning = spline_binnings[j];
                const int spline_bin = prop.VariableBinIndex(binning, i);
                weight *= GetSplineShift(j, shifts[j], spline_bin);
            }
            ret.Fill(reco_bin, weight * prop.added_weights[i]);
        }
        return ret;
    }

    Eigen::MatrixXf PROsyst::GrabMatrix(const std::string& sys) const{
        if(syst_map.find(sys) != syst_map.end())
            return covmat.at(syst_map.at(sys).first);	
        else{
            log<LOG_ERROR>(L"%1% || Systematic you asked for : %2% doesn't have matrix saved yet..") % __func__ % sys.c_str();
            log<LOG_ERROR>(L"%1% || Return empty matrix .") % __func__ ;
            return Eigen::MatrixXf();
        }
    }

    Eigen::MatrixXf PROsyst::GrabCorrMatrix(const std::string& sys) const{
        if(syst_map.find(sys) != syst_map.end())
            return corrmat.at(syst_map.at(sys).first);	
        else{
            log<LOG_ERROR>(L"%1% || Systematic you asked for : %2% doesn't have matrix saved yet..") % __func__ % sys.c_str();
            log<LOG_ERROR>(L"%1% || Return empty matrix .") % __func__ ;
            return Eigen::MatrixXf();
        }
    }

    Spline PROsyst::GrabSpline(const std::string& sys) const{
        if(syst_map.find(sys) != syst_map.end())
            return splines.at(syst_map.at(sys).first);	
        else{
            log<LOG_ERROR>(L"%1% || Systematic you asked for : %2% doesn't have spline saved yet..") % __func__ % sys.c_str();
            return {};
        }
    }

    PROsyst::SystType PROsyst::GetSystType(const std::string &syst) const {
        return syst_map.at(syst).second;
    }

    Eigen::MatrixXf PROsyst::DecomposeFractionalCovariance(const PROconfig &config, const Eigen::VectorXf &cv_vec) const {
        // Spline-only (no covariance systs): fractional_covariance is the
        // ctor's all-zero placeholder, whose SVD has no singular value above
        // tolerance. A zero factor is the correct no-op throw shift.
        if(n_covar == 0) {
            size_t nbins = config.m_num_variable_bins_total_collapsed[other_index < 0 ? (int)config.i_prime : other_index];
            return Eigen::MatrixXf::Zero(nbins, nbins);
        }
        // The mutable last_decomp_* cache is written from this const method;
        // metric clones and throw helpers share PROsyst objects across
        // threads, so guard the cache. Function-local mutex keeps PROsyst
        // copyable; contention is irrelevant on this cold path (the SVD below
        // dominates).
        static std::mutex decomp_cache_mutex;
        {
            std::lock_guard<std::mutex> lk(decomp_cache_mutex);
            if(cv_vec.size() == last_decomp_spec.size() && cv_vec == last_decomp_spec)
                return last_decomp_mat;
        }
        Eigen::MatrixXf full_cov = cv_vec.asDiagonal() * fractional_covariance * cv_vec.asDiagonal();
        Eigen::MatrixXf coll = other_index < 0 ? CollapseMatrix(config, full_cov) : CollapseMatrix(config, full_cov, other_index);
        /*Eigen::LDLT<Eigen::MatrixXf> ldlt(coll);
          Eigen::MatrixXf L = ldlt.matrixL(); 
          Eigen::VectorXf D_sqrt = ldlt.vectorD().array().sqrt();  
          Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P(ldlt.transpositionsP());

          if (ldlt.info() != Eigen::Success) {
          log<LOG_ERROR>(L"%1% | Eigen LLT has failed!") % __func__ ;
          Eigen::FullPivLU<Eigen::MatrixXf> lu_decomp(coll);
          int rank = lu_decomp.rank();
          int size = coll.rows();
          if (!coll.isApprox(coll.transpose())) {
          log<LOG_ERROR>(L"%1% | Matrix is not symmetric! Rank %2% and size %3%") % __func__ % rank % size ;
          }
          Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(coll);
          if (eigensolver.eigenvalues().minCoeff() <= 0) {
          log<LOG_ERROR>(L"%1% | Matrix is not positive semi definite, minCoeff is %2%. Rank %3% and size %4% ") % __func__ % eigensolver.eigenvalues().minCoeff() % rank % size;
          }
          Eigen::JacobiSVD<Eigen::MatrixXf> svd(coll);
          log<LOG_ERROR>(L"%1% | Singular values: %2% ") % __func__ % svd.singularValues();

          Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n", "", "", "", "");
          std::ostringstream oss;
          oss << coll.format(fmt);
          log<LOG_ERROR>(L"%1% | Matrix is %2% ") % __func__ % oss.str().c_str();
          exit(EXIT_FAILURE);
          }
          return P * L * D_sqrt.asDiagonal();*/
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(coll, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto& U = svd.matrixU();
        const auto& S = svd.singularValues();

        log<LOG_DEBUG>(L"%1% | Singular values: %2% ") % __func__ % svd.singularValues();

        Eigen::FullPivLU<Eigen::MatrixXf> lu_decomp(coll);
        int rank = lu_decomp.rank();
        int size = coll.rows();
        log<LOG_DEBUG>(L"%1% | Matrix is Rank %2% and size %3%") % __func__ % rank % size ;

        float tol = 1e-8f * S.maxCoeff(); // Some cutoff? is this value impactful on out matricies? need to test
        std::vector<int> keep;
        for (int i = 0; i < S.size(); ++i) {
            if (S(i) > tol) keep.push_back(i);
        }

        if (keep.empty()) {
            log<LOG_ERROR>(L"%1% | All singular values are below tolerance, cannot sample. Blarg.") % __func__;
            exit(EXIT_FAILURE);
        }

        //going to keep only the singular values that give meaningful variance
        Eigen::MatrixXf fallback_sampler = Eigen::MatrixXf::Zero(coll.rows(), coll.cols());
        for (size_t i = 0; i < keep.size(); ++i) {
            fallback_sampler.col(i) = U.col(keep[i]) * std::sqrt(S(keep[i]));
        }

        {
            std::lock_guard<std::mutex> lk(decomp_cache_mutex);
            last_decomp_spec = cv_vec;
            last_decomp_mat = fallback_sampler;
        }
        return fallback_sampler;

    }

    Eigen::MatrixXf PROsyst::DecomposeFractionalCovarianceFull(const PROconfig &config, const Eigen::VectorXf &cv_vec) const {
        (void)config;
        // Spline-only: same zero-factor short-circuit as the collapsed version above.
        if(n_covar == 0)
            return Eigen::MatrixXf::Zero(cv_vec.size(), cv_vec.size());
        // Same cache/mutex pattern as DecomposeFractionalCovariance above.
        static std::mutex decomp_full_cache_mutex;
        {
            std::lock_guard<std::mutex> lk(decomp_full_cache_mutex);
            if(cv_vec.size() == last_decomp_full_spec.size() && cv_vec == last_decomp_full_spec)
                return last_decomp_full_mat;
        }
        Eigen::MatrixXf full_cov = cv_vec.asDiagonal() * fractional_covariance * cv_vec.asDiagonal();

        // full_cov is symmetric PSD by construction, so a self-adjoint
        // eigendecomposition gives the same tolerance-clipped sampler as the
        // JacobiSVD used for the (smaller) collapsed matrix, at lower cost on
        // the full-bin dimension.
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(full_cov);
        if(es.info() != Eigen::Success) {
            log<LOG_ERROR>(L"%1% | Eigendecomposition of full-space covariance failed.") % __func__;
            exit(EXIT_FAILURE);
        }
        const Eigen::VectorXf &evals = es.eigenvalues();
        const Eigen::MatrixXf &evecs = es.eigenvectors();

        float tol = 1e-8f * evals.maxCoeff();
        std::vector<int> keep;
        for(int i = 0; i < evals.size(); ++i) {
            if(evals(i) > tol) keep.push_back(i);
        }

        if(keep.empty()) {
            log<LOG_ERROR>(L"%1% | All eigenvalues are below tolerance, cannot sample. Blarg.") % __func__;
            exit(EXIT_FAILURE);
        }

        Eigen::MatrixXf sampler = Eigen::MatrixXf::Zero(full_cov.rows(), full_cov.cols());
        for(size_t i = 0; i < keep.size(); ++i) {
            sampler.col(i) = evecs.col(keep[i]) * std::sqrt(evals(keep[i]));
        }

        {
            std::lock_guard<std::mutex> lk(decomp_full_cache_mutex);
            last_decomp_full_spec = cv_vec;
            last_decomp_full_mat = sampler;
        }
        return sampler;
    }

    void PROsyst::PrintSplines(){
        std::cout << "=== NEW FLAT SPLINE STRUCTURE ===\n";
        for (size_t spline_idx = 0; spline_idx < splines.size(); ++spline_idx) {
            const auto &spline = splines[spline_idx];
            std::cout << "Spline #" << spline_idx << ":\n";
            for (int bin_idx = 0; bin_idx < spline.bins; ++bin_idx) {
                std::cout << "  Bin " << bin_idx << ":\n";
                for (int seg_idx = 0; seg_idx < spline.segments_per_bin; ++seg_idx) {
                    const auto &seg = spline.segments[bin_idx * spline.segments_per_bin + seg_idx];
                    std::cout << std::fixed << std::setprecision(4)
                        << "    Segment " << seg_idx
                        << ": knot=" << seg.knot
                        << ", coeffs=[";
                    for (int c = 0; c < 4; ++c) {
                        std::cout << seg.coeffs[c];
                        if (c < 3) std::cout << ", ";
                    }
                    std::cout << "]\n";
                }
            }
        }
    }

};
