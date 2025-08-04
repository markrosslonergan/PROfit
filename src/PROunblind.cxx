#include "PROunblind.h"

namespace PROfit{


    int PROunblind_Stage1(PROmetric *metric ){

        //### 1 Number of Empty Bins in Data
        try{
            metric->checkData();
        }catch (const std::exception& e){
            log<LOG_ERROR>(L"%1% || ERROR. Check 2 unblinding Stage 1: %2% ") % __func__ % e.what();
        }

        //### 2 Covartiance Matrix at CV Positive SemiDefinite. 
        if(! metric->GetSysts().isPositiveSemiDefinite_WithTolerance(metric->GetSysts().fractional_covariance ,Eigen::NumTraits<float>::dummy_precision())){
            log<LOG_ERROR>(L"%1% || Fractional Covariance Matrix is not positive semi-definite!") % __func__;
            throw std::domain_error(std::string("Fractional Covariance Matrix is not positive semi-definite: "));
        }

        //PROfitterConfig fitconfig2("unblind",false);
        PROfitterConfig fitconfig2("good",false);

        size_t nparams = metric->GetModel().nparams + metric->GetSysts().GetNSplines();
        size_t nphys = metric->GetModel().nparams;
        Eigen::VectorXf lb = Eigen::VectorXf::Constant(nparams, -3.0);
        Eigen::VectorXf ub = Eigen::VectorXf::Constant(nparams, 3.0);
        for(size_t i = 0; i < nphys; ++i) {
            lb(i) = metric->GetModel().lb(i);
            ub(i) = metric->GetModel().ub(i);
        }
        for(size_t i = nphys; i < nparams; ++i) {
            lb(i) = metric->GetSysts().spline_lo[i-nphys];
            ub(i) = metric->GetSysts().spline_hi[i-nphys];
        }

        PROfitter fitter(ub, lb, fitconfig2,2);

        log<LOG_INFO>(L"%1% || ########### Starting Global Best Fit Minimizing ############") % __func__;

        float chi2 = fitter.Fit(*metric); 
        Eigen::VectorXf best_fit = fitter.best_fit;
        Eigen::MatrixXf post_covar = fitter.Covariance();

        //### 3 Chi is sennsible? 
        if(std::isnan(chi2) || chi2!=chi2 || std::isinf(chi2) || chi2 < 0){ 
            log<LOG_ERROR>(L"%1% || Resulting chi is NAN,INF or negative at best fit (nan %2%) (!= %3%) (inf %4%) (neg %5%)") % __func__ % std::isnan(chi2) % bool(chi2!=chi2) % std::isinf(chi2) % bool(chi2 < 0);
            throw std::domain_error(std::string("Resulting chi is NAN or INF or negative at best fit"));
        }

        //### 4 best fit params sensible 
        for(auto &v: best_fit){
            if(std::isnan(v) || v!=v || std::isinf(v) ){ 
                log<LOG_ERROR>(L"%1% || At least one resulting bf value is NAN or INF at best fit (nan %2%) (!= %3%) (inf %4%)") % __func__ % std::isnan(v) % bool(v!=v) % std::isinf(v);
                throw std::domain_error(std::string("Resulting BF value  is NAN or INF at best fit"));
            }
        }

        //### 5 best fit params boundary count
        int boundary= 0;
        for (size_t i = 0; i < best_fit.size(); ++i){
            float v = best_fit(i);
            
            if(std::isinf(lb(i)) || std::isinf(ub(i)))continue;
            float range = std::abs(lb(i)-ub(i));
            float tol = 1e-4;
            if(fabs(v-lb(i))< tol || fabs(v-ub(i))<tol) boundary++;
        }
        if(boundary>0){ 
            log<LOG_WARNING>(L"%1% || WARNING : At least one resulting bf value is at the boundary (actuall its %2% params)") % __func__ % boundary;
        }else{
            log<LOG_INFO>(L"%1% || INFO: None of the resulting bf value is at the boundary.") % __func__;
        }

        //### 6 

        log<LOG_INFO>(L"%1% || ################################################") % __func__;
        log<LOG_INFO>(L"%1% || ########### Global Best Fit Results ############") % __func__;
        log<LOG_INFO>(L"%1% || ################################################") % __func__;
        log<LOG_INFO>(L"%1% || Global Best Fit chi^2: %2%") %__func__ % chi2;
        log<LOG_INFO>(L"%1% || at paramters: ") % __func__;

        for(size_t i = 0; i< nparams; i++){

            if(i<nphys){
                log<LOG_INFO>(L"%1% || %2%  :  %3% ") % __func__ % metric->GetModel().pretty_param_names[i].c_str() % best_fit(i);
            }else{
                log<LOG_INFO>(L"%1% || %2%  :  %3% ") % __func__ % metric->GetSysts().spline_names[i-nphys].c_str() % best_fit(i);
            }
        }
        log<LOG_INFO>(L"%1% || ################################################") % __func__;


        return 0;
    }

}
