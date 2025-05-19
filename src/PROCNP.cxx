#include "PROCNP.h"
#include "PROcess.h"
#include "PROdata.h"
#include "PROlog.h"
#include "PROmetric.h"
#include "PROtocall.h"

#include <Eigen/Eigen>

using namespace PROfit;


PROCNP::PROCNP(const std::string tag, const PROconfig &conin, const PROpeller &pin, const PROsyst *systin, const PROmodel &modelin, const PROdata &datain, EvalStrategy strat, std::vector<float> physics_param_fixed) : PROmetric(), model_tag(tag), config(conin), peller(pin), syst(systin), model(modelin), data(datain), strat(strat), physics_param_fixed(physics_param_fixed), correlated_systematics(false) {
    last_value = 0.0; last_param = Eigen::VectorXf::Zero(model.nparams+syst->GetNSplines()); 
    fixed_index = -999;

    // Build the correlation matrix between priors if configured to
    if (conin.m_mcgen_correlations.size()) {
        correlated_systematics = true;
        prior_covariance = Eigen::MatrixXf::Identity(syst->GetNSplines(), syst->GetNSplines());
        for (auto const &t: conin.m_mcgen_correlations) {
          auto itA = std::find(systin->spline_names.begin(), systin->spline_names.end(), std::get<0>(t));
          if (itA == systin->spline_names.end()) {
            log<LOG_WARNING>(L"%1% || Systematic correlation %2% not in list. Skipping.") % __func__ % std::get<0>(t).c_str();
            continue;
          }

          auto itB = std::find(systin->spline_names.begin(), systin->spline_names.end(), std::get<1>(t));
          if (itB == systin->spline_names.end()) {
            log<LOG_WARNING>(L"%1% || Systematic correlation %2% not in list. Skipping.") % __func__ % std::get<1>(t).c_str();
            continue;
          }
         
          int iA = std::distance(systin->spline_names.begin(), itA);
          int iB = std::distance(systin->spline_names.begin(), itB);

          // set correlations
          prior_covariance(iA, iB) = std::get<2>(t);
          prior_covariance(iB, iA) = std::get<2>(t);
        }
    }
}

float PROCNP::Pull(const Eigen::VectorXf &systs) {
    // No correlations: sum of squares
    if (!correlated_systematics) return systs.array().square().sum();

    // Otherwise dot onto covariance
    return systs.dot(prior_covariance.inverse() * systs);
}

void PROCNP::fixSpline(int fix, float valin){
    fixed_index=fix;
    fixed_val=valin;
    return;
}
float PROCNP::operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient){
    return PROCNP::operator()(param, gradient, true);
}


float PROCNP::operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool rungradient){
    //size_t nparams = model->nparams+syst->GetNSplines();
    //size_t nsyst = syst->GetNSplines();

    // Get Spectra from FillRecoSpectra
    Eigen::VectorXf subvector1 = param.segment(0, model.nparams);
    log<LOG_DEBUG>(L"%1% || Created physics subvector with size %2%") % __func__ % subvector1.size();
    Eigen::VectorXf subvector2 = param.segment(model.nparams, syst->GetNSplines());
    log<LOG_DEBUG>(L"%1% || Created spline subvector with size %2%") % __func__ % subvector2.size();

    PROspec result = FillRecoSpectra(config, peller, *syst, model, param, strat == BinnedChi2);


    Eigen::MatrixXf inverted_collapsed_full_covariance(config.m_num_bins_total_collapsed,config.m_num_bins_total_collapsed);
    PROspec cv = FillRecoSpectra(config, peller, *syst, model, subvector1, strat != EventByEvent);
    Eigen::MatrixXf collapsed_data_stat_covariance = data.Spec().array().matrix().asDiagonal();
    Eigen::MatrixXf mc_stat_covariance = cv.Spec().array().matrix().asDiagonal();
    Eigen::MatrixXf collapsed_mc_stat_covariance = CollapseMatrix(config, mc_stat_covariance);
    
    Eigen::MatrixXf collapsed_stat_covariance = 3 * (collapsed_data_stat_covariance.inverse() + 2 * collapsed_mc_stat_covariance.inverse()).inverse();
    Eigen::MatrixXf diag = result.Spec().array().matrix().asDiagonal(); 
    Eigen::MatrixXf full_covariance = diag*(syst->fractional_covariance)*diag;
    
    Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance); 
    inverted_collapsed_full_covariance = (collapsed_stat_covariance+ collapsed_full_covariance).inverse();

    // Calculate Chi^2  value
    Eigen::VectorXf delta  = CollapseMatrix(config,result.Spec()) - data.Spec(); 

    float pull = Pull(subvector2);
    float dmsq_penalty = 0;
    float covar_portion = (delta.transpose())*inverted_collapsed_full_covariance*(delta);
    float value = covar_portion + dmsq_penalty + pull;


    if(rungradient){
        float dval = 1e-4;
        for (size_t i = 0; i < model.nparams+syst->GetNSplines(); i++) {
            //Eigen::VectorXf tmpParams = last_param;
            Eigen::VectorXf tmpParams = param;
            int sgn = ((param(i) - last_param(i)) > 0) - ((param(i) - last_param(i)) < 0);
            if(!sgn) sgn = 1;
            //if(fitparams.size() != 0 && i == 1 && param(i) < -4 + dval) sgn = 1;
            //else if(fitparams.size() != 0 && i == 1 && param(i) > 0 - dval) sgn = -1;
            tmpParams(i) = /*param(i) != last_param(i) ? param(i) :*/ param(i) + sgn * dval;
            
            Eigen::VectorXf subvector1 = tmpParams.segment(0, model.nparams);
            log<LOG_DEBUG>(L"%1% || Created physics subvector with size %2%") % __func__ % subvector1.size();
            Eigen::VectorXf subvector2 = tmpParams.segment(model.nparams, syst->GetNSplines());
            log<LOG_DEBUG>(L"%1% || Created spline subvector with size %2%") % __func__ % subvector2.size();
            PROspec result = FillRecoSpectra(config, peller, *syst, model, tmpParams, strat != EventByEvent);
            // Calcuate Full Covariance matrix
            Eigen::MatrixXf inverted_collapsed_full_covariance(config.m_num_bins_total_collapsed,config.m_num_bins_total_collapsed);

            Eigen::MatrixXf new_collapsed_stat_covariance = collapsed_stat_covariance;
            if(i < model.nparams) {
                PROspec cv = FillRecoSpectra(config, peller, *syst, model, subvector1, strat != EventByEvent);
                Eigen::MatrixXf collapsed_data_stat_covariance = data.Spec().array().matrix().asDiagonal();
                Eigen::MatrixXf mc_stat_covariance = cv.Spec().array().matrix().asDiagonal();
                Eigen::MatrixXf collapsed_mc_stat_covariance = CollapseMatrix(config, mc_stat_covariance);
                new_collapsed_stat_covariance = 3 * (collapsed_data_stat_covariance.inverse() + 2 * collapsed_mc_stat_covariance.inverse()).inverse();
            }

            Eigen::MatrixXf diag = result.Spec().array().matrix().asDiagonal(); 
            Eigen::MatrixXf full_covariance = diag*(syst->fractional_covariance)*diag;
            
            Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance); 
            inverted_collapsed_full_covariance = (collapsed_stat_covariance+ collapsed_full_covariance).inverse();
           
            // Calculate Chi^2  value
            Eigen::VectorXf delta  = CollapseMatrix(config,result.Spec()) - data.Spec(); 

            float pull = Pull(subvector2);
            float value_grad = (delta.transpose())*inverted_collapsed_full_covariance*(delta) + pull;
            
            gradient(i) = (value_grad-value)/(tmpParams(i) - param(i));
        }
    }
    //std::cout<<"Grad: "<<gradient<<std::endl;

    //log<LOG_DEBUG>(L"%1% || value %2%, last_value %3%, pull") % __func__ % value  % last_value % pull;
    log<LOG_DEBUG>(L"%1% || FINISHED ITERATION got vals: %2% %3%") % __func__ % value % last_value ;

    //Update last param
    last_param = param;
    last_value = value;

    return value;
}

float PROCNP::getSingleChannelChi(size_t global_channel_index) {
    PROspec cv = FillCVSpectrum(config, peller,strat == BinnedChi2);

    size_t nbin =  config.m_channel_num_bins[config.GetLocalChannelIndex(global_channel_index)];
    size_t startBin = config.GetCollapsedGlobalBinStart(global_channel_index);


    Eigen::MatrixXf inverted_collapsed_full_covariance(nbin,nbin);


    Eigen::MatrixXf collapsed_data_stat_covariance = (data.Spec().array().matrix().asDiagonal());
    collapsed_data_stat_covariance = collapsed_data_stat_covariance.block(startBin,startBin,nbin,nbin);
    Eigen::MatrixXf mc_stat_covariance = cv.Spec().array().matrix().asDiagonal();
    Eigen::MatrixXf collapsed_mc_stat_covariance = CollapseMatrix(config, mc_stat_covariance).block(startBin,startBin,nbin,nbin);
    Eigen::MatrixXf sub_collapsed_stat_covariance = 3 * (collapsed_data_stat_covariance.inverse() + 2 * collapsed_mc_stat_covariance.inverse()).inverse();

    //only calculate a syst covariance if we have any covariance parameters as defined in the xml
    if(syst->GetNCovar()){
        // Calculate Full Syst Covariance matrix
        Eigen::MatrixXf diag =  cv.Spec().array().matrix().asDiagonal(); 
        Eigen::MatrixXf full_covariance =  diag*(syst->fractional_covariance)*diag;

        // Collapse Covariance and Spectra 
        Eigen::MatrixXf collapsed_full_covariance =  CollapseMatrix(config,full_covariance);
        Eigen::MatrixXf sub_collapsed_full_covariance =  collapsed_full_covariance.block(startBin,startBin,nbin,nbin);

        // Invert Collaped Matrix Matrix 
        inverted_collapsed_full_covariance = (sub_collapsed_full_covariance+sub_collapsed_stat_covariance).inverse();
    } else {
        inverted_collapsed_full_covariance = (sub_collapsed_stat_covariance).inverse();
    }

    Eigen::VectorXf delta  = (CollapseMatrix(config, cv.Spec()) - data.Spec()).segment(startBin,nbin);
    //float pull = Pull(subvector2);
    float covar_portion = (delta.transpose())*inverted_collapsed_full_covariance*(delta);
    float value = covar_portion;//pull;

    return value;
}


