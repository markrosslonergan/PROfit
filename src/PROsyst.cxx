#include "PROsyst.h"
#include "PROcess.h"
#include "PROpeller.h"
#include "PROconfig.h"
#include "PROcreate.h"
#include "PROlog.h"
#include "PROtocall.h"
#include <Eigen/Eigen>
#include <random>

namespace PROfit {

    bool PROsyst::shape_only = false;

    PROsyst::PROsyst( const PROpeller &prop, const PROconfig &config, const std::vector<SystStruct>& systs, bool shapeonly, int other_index, const PROmodel* model, const Eigen::VectorXf* params) : other_index(other_index) {
        shape_only = shapeonly;
        for(const auto& syst: systs) {
            log<LOG_DEBUG>(L"%1% || syst mode: %2%") % __func__ % syst.mode.c_str();
            if(syst.mode == "spline" || syst.mode == "norm" || syst.mode == "hist1d" || syst.mode == "hist2d") {
                FillSpline(syst);
                ++n_splines;
            } else if(syst.mode == "spline_to_covariance") {
                if(model == nullptr){
                    log<LOG_ERROR>(L"%1% || spline_to_covariance requires a PROmodel to use spline2cov. "
                        L"Construct PROsyst with a model (and optional params).") % __func__;
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);
                }
                // Build spline first, then convert to covariance matrix
                FillSpline(syst);
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
                Eigen::MatrixXf frac_cov = spline2cov(spline_idx, config, prop, *model, cvparams, 42);
                spline_priors  = saved_priors;
                spline_centers = saved_centers;
                Eigen::MatrixXf corr = GenerateCorrMatrix(frac_cov);

                // Remove the spline (it was the last one appended by FillSpline)
                syst_map.erase(syst.systname);
                splines.pop_back();
                spline_names.pop_back();
                spline_lo.pop_back();
                spline_hi.pop_back();
                spline_binnings.pop_back();

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
            }else if(syst.mode == "flat"){
                this->CreateFlatMatrix(config, syst); 
                covar_names.push_back(syst.systname); 
                ++n_covar;
            }else if(syst.mode == "external_covariance"){
                if(other_index == syst.binning){
                    //external covariances are only created for specific binning
                    this->LoadExternalCovarianceMatrix(config,syst);
                    covar_names.push_back(syst.systname);
                    ++n_covar;
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
            syst_map["mcstat"] = {covmat.size(), SystType::Covariance};
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
                        ret.spline_binnings.push_back(spline_binnings[idx]);
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
        log<LOG_DEBUG>(L"%1% | Done Subset.") % __func__ ;
        return ret;
    }

    PROsyst PROsyst::excluding(const std::vector<std::string> &systs) const {
        PROsyst ret;
        Eigen::VectorXf tmp_priors = spline_priors;
        Eigen::VectorXf tmp_centers = spline_centers;
        for(const auto &[name, spair]: syst_map) {
            if(std::find(systs.begin(), systs.end(), name) != systs.end()) continue;
            const auto &[idx, stype] = spair;
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
                        ret.spline_binnings.push_back(spline_binnings[idx]);
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
                                           Eigen::MatrixXf cov = spline2cov(idx, config, prop, model,params, seed);
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
        for(size_t i = 0; i < 500; ++i){
            specs.push_back(FillSplineRandomThrow(config, prop, *this, model, params, spline, seed, other_index).Spec());
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

            int ii=0;
            for(auto& p : covmat){
                sum_matrix += p;
                ii++;
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

        size_t colonPos = sysname.find(':');
        if (colonPos == std::string::npos) {
            log<LOG_ERROR>(L"%1% || ERROR, you asked for a flat systematic but its not in NAME:percentate format %2%") % __func__  % sysname.c_str();
            exit(EXIT_FAILURE);
        }

        std::string wild = sysname.substr(0, colonPos);
        std::string sflat_percent  = sysname.substr(colonPos + 1);
        float flat_percent = std::stof(sflat_percent);


        log<LOG_INFO>(L"%1% || Wildcard %2% (and percent %3%) which matches: ") % __func__  % wild.c_str() % flat_percent;
        std::vector<std::string> flatnames;
        for(auto & name: config.m_fullnames){
            if(name.find(wild)!=std::string::npos){
                flatnames.push_back(name); 
            }
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
    void PROsyst::LoadExternalCovarianceMatrix(const PROconfig &config, const SystStruct& syst){
        //this is matrix name
        std::string matrixname = syst.GetSysName();
        std::string filename = syst.external_filename;
        log<LOG_INFO>(L"%1% || Loading a TMatrix from %2% named %3%") % __func__ % filename.c_str() % matrixname.c_str();
        int nbins = config.m_num_variable_bins_total[other_index];
        Eigen::MatrixXf fracM = Eigen::MatrixXf::Zero(nbins, nbins);
        Eigen::MatrixXf corrM = Eigen::MatrixXf::Identity(nbins, nbins);

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

        // Generate correlation matrix from fractional covariance
        corrM = PROsyst::GenerateCorrMatrix(fracM);

        syst_map[matrixname] = {covmat.size(), SystType::Covariance};
        covmat.push_back(fracM);
        corrmat.push_back(corrM);

        log<LOG_INFO>(L"%1% || Successfully loaded %2%x%3% covariance matrix %4%") 
            % __func__ % nbins % nbins % matrixname.c_str();

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

        // Find the right segment with egments are sorted by knob value
        int offset = bin * spline.segments_per_bin;
        const SplineSegment* segs = &spline.segments[offset];

        float lowest_knobval = segs[0].knot;
        int shiftBin = std::clamp(int(shift - lowest_knobval), 0, spline.segments_per_bin - 1);

        const SplineSegment& seg = segs[shiftBin];
        float x = shift - seg.knot;
        const auto& c = seg.coeffs;
        return c[0] + x*(c[1] + x*(c[2] + x*c[3]));
    }

    void PROsyst::FillSpline(const SystStruct& syst) {
        std::vector<PROspec> ratios;
        ratios.reserve(syst.p_multi_spec.size());
        float cv_integral = syst.p_cv->Spec().sum();

        bool found0 = false;
        int knob0_index = -1;  // Index of knobval=0 in ratios vector

        std::vector<float> knobvals;
        for (size_t i = 0; i < syst.p_multi_spec.size(); ++i) {
            // Skip duplicate consecutive knob values (syst.knobval is sorted; the fill loop
            // already populated the correct p_multi_spec slot via last-occurrence semantics)
            if (i > 0 && syst.knobval[i] == syst.knobval[i-1]) continue;
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
        spline_binnings.push_back(syst.binning);

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
        if(cv_vec.size() == last_decomp_spec.size() && cv_vec == last_decomp_spec) 
          return last_decomp_mat;
        Eigen::MatrixXf diag = cv_vec.asDiagonal();
        Eigen::MatrixXf full_cov = diag * fractional_covariance * diag;
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

        last_decomp_spec = cv_vec;
        last_decomp_mat = fallback_sampler;
        return fallback_sampler;

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
