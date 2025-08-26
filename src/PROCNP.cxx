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
        prior_covariance = systin->spline_priors.asDiagonal() * prior_covariance * systin->spline_priors.asDiagonal();
    }
}

float PROCNP::Pull(const Eigen::VectorXf &systs) {
    // No correlations: sum of squares
    Eigen::VectorXf centered = systs - syst->spline_centers;
    if (!correlated_systematics) {
        return (centered.array().square() / syst->spline_priors.array().square()).sum();
    }

    // Otherwise dot onto covariance
    return centered.dot(prior_covariance.inverse() * centered);
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
    call_count++;

    // Get Spectra from FillRecoSpectra
    Eigen::VectorXf subvector1 = param.segment(0, model.nparams);
    //log<LOG_DEBUG>(L"%1% || Created physics subvector with size %2%") % __func__ % subvector1.size();
    Eigen::VectorXf subvector2 = param.segment(model.nparams, syst->GetNSplines());
    //log<LOG_DEBUG>(L"%1% || Created spline subvector with size %2%") % __func__ % subvector2.size();

    PROspec result = FillRecoSpectra(config, peller, *syst, model, param, strat == BinnedChi2);


    Eigen::MatrixXf inverted_collapsed_full_covariance(config.m_num_bins_total_collapsed,config.m_num_bins_total_collapsed);
    Eigen::VectorXf collapsed_cv = CollapseMatrix(config, FillRecoSpectra(config, peller, *syst, model, subvector1, strat != EventByEvent).Spec());
    Eigen::MatrixXf collapsed_stat_covariance = Eigen::MatrixXf::Zero(data.Spec().size(), data.Spec().size());
    for(long i = 0; i < data.Spec().size(); ++i)
        collapsed_stat_covariance(i,i) = data.Spec()(i) == 0 ? collapsed_cv(i)/2 :
            3 / (1.0 / data.Spec()(i) + 2.0 / collapsed_cv(i));

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

    if(std::isnan(value)) {
        log<LOG_WARNING>(L"%1% || WARNING: CNP chi2 is NaN. This is very bad.\n"
                L"covar_portion: %2%\npull: %3%\ndelta: %4%\n"
                L"mc spec: %5%\ndata spec: %6%")
            % __func__ % covar_portion % pull % delta % CollapseMatrix(config, result.Spec())
            % data.Spec();
        throw std::runtime_error("CNP chi2 is nan.");
    }


    if(rungradient){
        for (size_t i = 0; i < model.nparams+syst->GetNSplines(); i++) {

            if(is_fixed.size()>0){
                if(is_fixed.at(i)) {
                    gradient(i) = 0.0f;
                    continue;  
                }
            }

            float h;
            h = 1e-4f;
            if(i < model.nparams) {
                h = 1e-3f;
            }

            float boundary_tol = 2.0f*std::numeric_limits<float>::epsilon();
            bool at_lower = (param(i) - lb(i)) < boundary_tol;
            bool at_upper = (ub(i) - param(i)) < boundary_tol;
        
            if(at_lower && at_upper){//shouldnt happen, if ix fixed working
                    gradient(i) = 0.0f;
                    continue;
            }

            int sign = (at_lower? 1 : (at_upper ? -1 : -999) );

            if(at_lower || at_upper) {
                Eigen::VectorXf param_plus = param;
                param_plus(i) = param(i) + sign*h;

                float chi2_oneside;
                // Calculate chi2_plus or chi2_minus, depending on boundary
                PROspec result = FillRecoSpectra(config, peller, *syst, model, param_plus, strat != EventByEvent);

                Eigen::MatrixXf new_collapsed_stat_covariance = collapsed_stat_covariance;
                if(i < model.nparams) {
                    Eigen::VectorXf subvector1 = param_plus.segment(0, model.nparams);
                    Eigen::VectorXf collapsed_cv = CollapseMatrix(config, FillRecoSpectra(config, peller, *syst, model, subvector1, strat != EventByEvent).Spec());
                    for(long j = 0; j < data.Spec().size(); ++j)
                        new_collapsed_stat_covariance(j,j) = data.Spec()(j) == 0 ? collapsed_cv(j)/2 :
                            3 / (1.0 / data.Spec()(j) + 2.0 / collapsed_cv(j));
                }

                Eigen::MatrixXf diag = result.Spec().array().matrix().asDiagonal();
                Eigen::MatrixXf full_covariance = diag*(syst->fractional_covariance)*diag;
                Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance);
                Eigen::MatrixXf inverted_collapsed_full_covariance = (new_collapsed_stat_covariance + collapsed_full_covariance).inverse();

                Eigen::VectorXf delta = CollapseMatrix(config,result.Spec()) - data.Spec();
                Eigen::VectorXf subvector2 = param_plus.segment(model.nparams, syst->GetNSplines());
                float pull = Pull(subvector2);
                chi2_oneside = (delta.transpose())*inverted_collapsed_full_covariance*(delta) + pull;

                gradient(i) = sign*(chi2_oneside - value) / h;

                // If gradient suggests going further out of bounds, set to zero
                if(sign*gradient(i) > 0) {
                    gradient(i) = 0.0f;//-sign*h/2.0f;  // Minimum is at boundary, really (small bounce)
                }
            }else{

                // Central difference method
                Eigen::VectorXf param_plus = param;
                Eigen::VectorXf param_minus = param;
                param_plus(i) = param(i) + h;
                param_minus(i) = param(i) - h;

                // Calculate chi2 at both points
                float chi2_plus, chi2_minus;

                // Plus point
                {
                    PROspec result = FillRecoSpectra(config, peller, *syst, model, param_plus, strat != EventByEvent);

                    Eigen::MatrixXf new_collapsed_stat_covariance = collapsed_stat_covariance;
                    if(i < model.nparams) {
                        Eigen::VectorXf subvector1 = param_plus.segment(0, model.nparams);
                        Eigen::VectorXf collapsed_cv = CollapseMatrix(config, FillRecoSpectra(config, peller, *syst, model, subvector1, strat != EventByEvent).Spec());
                        for(long j = 0; j < data.Spec().size(); ++j)
                            new_collapsed_stat_covariance(j,j) = data.Spec()(j) == 0 ? collapsed_cv(j)/2 :
                                3 / (1.0 / data.Spec()(j) + 2.0 / collapsed_cv(j));
                    }

                    Eigen::MatrixXf diag = result.Spec().array().matrix().asDiagonal();
                    Eigen::MatrixXf full_covariance = diag*(syst->fractional_covariance)*diag;
                    Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance);
                    Eigen::MatrixXf inverted_collapsed_full_covariance = (new_collapsed_stat_covariance + collapsed_full_covariance).inverse();

                    Eigen::VectorXf delta = CollapseMatrix(config,result.Spec()) - data.Spec();
                    Eigen::VectorXf subvector2 = param_plus.segment(model.nparams, syst->GetNSplines());
                    float pull = Pull(subvector2);
                    chi2_plus = (delta.transpose())*inverted_collapsed_full_covariance*(delta) + pull;
                }

                // Minus point
                {
                    PROspec result = FillRecoSpectra(config, peller, *syst, model, param_minus, strat != EventByEvent);

                    Eigen::MatrixXf new_collapsed_stat_covariance = collapsed_stat_covariance;
                    if(i < model.nparams) {
                        Eigen::VectorXf subvector1 = param_minus.segment(0, model.nparams);
                        Eigen::VectorXf collapsed_cv = CollapseMatrix(config, FillRecoSpectra(config, peller, *syst, model, subvector1, strat != EventByEvent).Spec());
                        for(long j = 0; j < data.Spec().size(); ++j)
                            new_collapsed_stat_covariance(j,j) = data.Spec()(j) == 0 ? collapsed_cv(j)/2 :
                                3 / (1.0 / data.Spec()(j) + 2.0 / collapsed_cv(j));
                    }

                    Eigen::MatrixXf diag = result.Spec().array().matrix().asDiagonal();
                    Eigen::MatrixXf full_covariance = diag*(syst->fractional_covariance)*diag;
                    Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance);
                    Eigen::MatrixXf inverted_collapsed_full_covariance = (new_collapsed_stat_covariance + collapsed_full_covariance).inverse();

                    Eigen::VectorXf delta = CollapseMatrix(config,result.Spec()) - data.Spec();
                    Eigen::VectorXf subvector2 = param_minus.segment(model.nparams, syst->GetNSplines());
                    float pull = Pull(subvector2);
                    chi2_minus = (delta.transpose())*inverted_collapsed_full_covariance*(delta) + pull;
                }

                // Central difference formula
                gradient(i) = (chi2_plus - chi2_minus) / (2.0f * h);
            }
            // Sanity check
            if(!std::isfinite(gradient(i))) {
                gradient(i) = 0.0f;
            }
        }
    }

    //log<LOG_DEBUG>(L"%1% || value %2%, last_value %3%, pull") % __func__ % value  % last_value % pull;
    //log<LOG_DEBUG>(L"%1% || FINISHED ITERATION got vals: %2% %3%") % __func__ % value % last_value ;

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

int PROCNP::checkData(){
    int zeroCount = (data.Spec().array() == 0.0f).count();
    if(zeroCount>0){
        log<LOG_ERROR>(L"%1% || ERROR: You asked to check data, and there is  %2% zero bins. Check binning?") % __func__ % zeroCount;
        throw std::invalid_argument("Zero elements in data when checked.");
    }

    return 0;
}

