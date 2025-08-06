#include "PROunblind.h"

namespace PROfit{

    void getConfirmation(std::string first, std::string second){

        while (true) {
             log<LOG_ERROR>(L"%1% ||%2% ") % __func__ % first.c_str();;
             log<LOG_ERROR>(L"%1% || If you want to proceed please type \"proceed\" or type \"exit\" to quit.") % __func__;
             std::string input;
             std::getline(std::cin, input);
             if (input == "exit") throw std::domain_error(std::string("Manually exited on confirmation command."));
             if (input == "proceed") {
                log<LOG_ERROR>(L"%1% || Proceeding..you have 3s to ctrl-c this!") % __func__;
                std::this_thread::sleep_for(std::chrono::seconds(3));                                   
                break;
             } else {
                log<LOG_ERROR>(L"%1% || Thats not one of the two options, try again!") % __func__;
             }
       

        }

       log<LOG_ERROR>(L"%1% ||%2% ") % __func__ % second.c_str();;
       return;
    }

    int PROunblind_Stage1( const PROconfig &config, const PROpeller &prop, PROmetric *metric , PROseed &myseed, size_t nthread, std::string final_output_tag){
   
        //manually remove any print outs
        GLOBAL_LEVEL=LOG_WARNING;

        log<LOG_ERROR>(L"%1% || ################################################") % __func__;
        //### 1 Number of Empty Bins in Data
        try{
            metric->checkData();
        }catch (const std::exception& e){
            log<LOG_ERROR>(L"%1% || ERROR. Check 2 unblinding Stage 1: %2% ") % __func__ % e.what();
        }
        log<LOG_ERROR>(L"%1% || CHECK 1: Data had no empty bins. [ --Passed-- ] ") % __func__;


        //### 2 Covartiance Matrix at CV Positive SemiDefinite. 
        if(! metric->GetSysts().isPositiveSemiDefinite_WithTolerance(metric->GetSysts().fractional_covariance ,Eigen::NumTraits<float>::dummy_precision())){
            log<LOG_ERROR>(L"%1% || Fractional Covariance Matrix is not positive semi-definite!") % __func__;
            throw std::domain_error(std::string("Fractional Covariance Matrix is not positive semi-definite: "));
        }
        log<LOG_ERROR>(L"%1% || CHECK 2: Frac Covariance passed all positive semi definite-ness checks. [ --Passed-- ] ") % __func__;

        //PROfitterConfig fitconfig2("unblind",false);
        PROfitterConfig fitconfig2("good",false);
        PROfitterConfig scanfitconfig2("good",true);

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

        log<LOG_ERROR>(L"%1% || ########### Starting Global Best Fit Minimizing ############") % __func__;
        log<LOG_ERROR>(L"%1% || ####### (Note LOG level manually set to WARNING only) ######") % __func__; 

        float chi2 = fitter.Fit(*metric); 
        Eigen::VectorXf best_fit = fitter.best_fit;
        Eigen::MatrixXf post_covar = fitter.Covariance();

        //### 3 Chi is sennsible? 
        if(std::isnan(chi2) || chi2!=chi2 || std::isinf(chi2) || chi2 < 0){ 
            log<LOG_ERROR>(L"%1% || Resulting chi is NAN,INF or negative at best fit (nan %2%) (!= %3%) (inf %4%) (neg %5%)") % __func__ % std::isnan(chi2) % bool(chi2!=chi2) % std::isinf(chi2) % bool(chi2 < 0);
            throw std::domain_error(std::string("Resulting chi is NAN or INF or negative at best fit"));
        }
        log<LOG_ERROR>(L"%1% || CHECK 3: Resulting BF Chi2 was positve, non-nan, non-inf. [ --Passed-- ]  ") % __func__;

        //### 4 best fit params sensible 
        for(auto &v: best_fit){
            if(std::isnan(v) || v!=v || std::isinf(v) ){ 
                log<LOG_ERROR>(L"%1% || At least one resulting bf value is NAN or INF at best fit (nan %2%) (!= %3%) (inf %4%)") % __func__ % std::isnan(v) % bool(v!=v) % std::isinf(v);
                throw std::domain_error(std::string("Resulting BF value  is NAN or INF at best fit"));
            }
        }
        log<LOG_ERROR>(L"%1% || CHECK 4: No BF params were NAN or INF. [ --Passed-- ]  ") % __func__;

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
            log<LOG_ERROR>(L"%1% || CHECK 5: At least one resulting bf value is at the boundary (actually it's %2% params) [ --INFO-- ]") % __func__ % boundary;
        }else{
            log<LOG_ERROR>(L"%1% || CHECK 5: None of the resulting bf value is at the boundary. [ --INFO--]") % __func__;
        }

        //### 6 
        size_t counts = metric->getCallCount();
       log<LOG_ERROR>(L"%1% || CHECK 6: We had %2% total PROmetric calls during this fit. Make sure thats sensible. [ --INFO-- ]") % __func__ % counts;

       //#### 7
       if (fitter.exception_string_map.empty()) {
               log<LOG_ERROR>(L"%1% || CHECK 7: No exceptions were caught from LBFGSB [ --INFO-- ]") % __func__;
       } else {
               log<LOG_ERROR>(L"%1% || CHECK 7: Some exceptions were caught in LBFGSB [ --INFO-- ]") % __func__;
               for (const auto &[msg, count] : fitter.exception_string_map) {
                    log<LOG_ERROR>(L"%1% || CHECK 7: -- Exception \"%2%\" occurred %3% time(s)") % __func__ % msg.c_str() % count;
               }
       }

       log<LOG_ERROR>(L"%1% || ################################################") % __func__;
        getConfirmation("Passed all auto checks, but please review the above info.","Proceed to BF chi value reveal?");


        log<LOG_ERROR>(L"%1% || ################################################") % __func__;
        log<LOG_ERROR>(L"%1% || ########### Global Best Fit Results ############") % __func__;
        log<LOG_ERROR>(L"%1% || ################################################") % __func__;
        log<LOG_ERROR>(L"%1% || Global Best Fit chi^2: %2%") %__func__ % chi2;
        
        getConfirmation("############################################","Proceed to begin frequentist pval calc?");

        //manually remove any print outs
        GLOBAL_LEVEL=LOG_INFO;

        //get it all from feldman FC, modified to gof also        
        size_t nuniv = 100;
        log<LOG_ERROR>(L"%1% || -- Calculating frequentist pvalue using %2% samples   ") % __func__ % nuniv;
        size_t FCthreads = nthread > nuniv ? nuniv : nthread;
        Eigen::MatrixXf cv_vec = FillCVSpectrum(config, prop, false).Spec();
        Eigen::MatrixXf L = metric->GetSysts().DecomposeFractionalCovariance(config, cv_vec);

        std::vector<std::vector<float>> dchi2s;
        dchi2s.reserve(FCthreads);
        std::vector<std::vector<fc_out>> outs;
        outs.reserve(FCthreads);
        std::vector<std::thread> threads;
        size_t todo = nuniv/FCthreads;
        size_t addone = FCthreads - nuniv%FCthreads;
        bool gof_mode = true;
        for(size_t i = 0; i < nthread; i++) {
            dchi2s.emplace_back();
            outs.emplace_back();
            fc_args args{todo + (i >= addone), &dchi2s.back(), &outs.back(), config, prop, metric->GetSysts(), "PROCNP", best_fit, L, scanfitconfig2,(*myseed.getThreadSeeds())[i], (int)i, false,gof_mode};


            threads.emplace_back([args]() {
                    PROfit::fc_worker(args);
                    });
        }
        for(auto&& t: threads) {
            t.join();
        }

        log<LOG_ERROR>(L"%1% || Finished throws. %2%") % __func__ % __LINE__;
        {
            TFile fout((final_output_tag+"_unblind_BF_GOF.root").c_str(), "RECREATE");
            fout.cd();
            float chi2_osc, chi2_syst, best_dmsq, best_sinsq2t;
            std::map<std::string, float> best_systs_osc, best_systs, syst_throw;
            TTree tree("tree", "tree");
            tree.Branch("chi2_osc", &chi2_osc); 
            //tree.Branch("chi2_syst", &chi2_syst); 
            tree.Branch("best_dmsq", &best_dmsq); 
            tree.Branch("best_sinsq2t", &best_sinsq2t); 
            tree.Branch("best_systs_osc", &best_systs_osc); 
            //tree.Branch("best_systs", &best_systs); 
            tree.Branch("syst_throw", &syst_throw);

            for(const auto &out: outs) {
                for(const auto &fco: out) {
                    chi2_osc = fco.chi2_osc;
                    //chi2_syst = fco.chi2_syst;
                    best_dmsq = fco.dmsq;
                    best_sinsq2t = fco.sinsq2tmm;
                    for(size_t i = 0; i < metric->GetSysts().GetNSplines(); ++i) {
                        best_systs_osc[metric->GetSysts().spline_names[i]] = fco.best_fit_osc(i);
                        //best_systs[metric->GetSysts().spline_names[i]] = fco.best_fit_syst(i);
                        syst_throw[metric->GetSysts().spline_names[i]] = fco.syst_throw(i);
                    }
                    tree.Fill();
                }
            }

            tree.Write();
        }
        {
            ofstream fcout(final_output_tag+"_unblind_BF_GOF.csv");
            fcout << "chi2_osc,best_dmsq,best_sinsq2t";
            for(const std::string &name: metric->GetSysts().spline_names) {
                fcout << ",best_" << name << "_osc,best_" << name << "," << name << "_throw";
            }
            fcout << "\r\n";

            for(const auto &out: outs) {
                for(const auto &fco: out) {
                    fcout << fco.chi2_osc << "," << "," << fco.dmsq << "," << fco.sinsq2tmm;
                    for(size_t i = 0; i < metric->GetSysts().GetNSplines(); ++i) {
                        fcout << fco.best_fit_osc(i) << "," << fco.syst_throw(i);
                    }
                    fcout << "\r\n";
                }
            }
        }
        std::vector<float> flattened_gofchi2s;
        for(const auto& out: outs) for(const auto& fco: out) flattened_gofchi2s.push_back(fco.chi2_osc);
        std::sort(flattened_gofchi2s.begin(), flattened_gofchi2s.end());
        log<LOG_ERROR>(L"%1% || All: %2% ") % __func__ % flattened_gofchi2s;
        log<LOG_ERROR>(L"%1% || chi: %2% ") % __func__ % chi2;
        auto it = std::lower_bound(flattened_gofchi2s.begin(), flattened_gofchi2s.end(), chi2);
        size_t index =  std::distance(flattened_gofchi2s.begin(),it);
        size_t count_above = flattened_gofchi2s.size()-index;
        float pval = (float)count_above/(float)nuniv;
        log<LOG_ERROR>(L"%1% || Finished throws. %2% %3% %4%") % __func__ % __LINE__% index % count_above;
        log<LOG_ERROR>(L"%1% || GOF pval after throwing %2% universes is %3%") % __func__ % nuniv % pval ;
 
        /*
         for(size_t i = 0; i< nparams; i++){

            if(i<nphys){
                log<LOG_INFO>(L"%1% || %2%  :  %3% ") % __func__ % metric->GetModel().pretty_param_names[i].c_str() % best_fit(i);
            }else{
                log<LOG_INFO>(L"%1% || %2%  :  %3% ") % __func__ % metric->GetSysts().spline_names[i-nphys].c_str() % best_fit(i);
            }
        }
        */

        log<LOG_ERROR>(L"%1% || ################################################") % __func__;


        return 0;
    }

}
