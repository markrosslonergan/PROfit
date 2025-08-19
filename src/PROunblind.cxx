#include "PROunblind.h"
#include "PROsurf.h"
#include "PROplot.h"

#include "TPaveText.h"

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

    int PROunblind_Stage1( const PROconfig &config, const PROpeller &prop, PROmetric *metric , PROseed &myseed, PROdata data, size_t nthread, std::string final_output_tag){
   
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
        
        getConfirmation("Proceed to begin frequentist pval calc?","############################################");

        //manually remove any print outs
        //GLOBAL_LEVEL=LOG_INFO;

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
            fc_args args{todo + (i >= addone), &dchi2s.back(), &outs.back(), config, prop, metric->GetSysts(), "PROCNP", best_fit, L, scanfitconfig2,(*myseed.getThreadSeeds())[i], (int)i, true,gof_mode};


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
            //tree.Branch("chi2_osc", &chi2_osc); 
            tree.Branch("chi2_syst", &chi2_syst); 
            //tree.Branch("best_dmsq", &best_dmsq); 
            //tree.Branch("best_sinsq2t", &best_sinsq2t); 
            //tree.Branch("best_systs_osc", &best_systs_osc); 
            tree.Branch("best_systs", &best_systs); 
            tree.Branch("syst_throw", &syst_throw);

            for(const auto &out: outs) {
                for(const auto &fco: out) {
                    //chi2_osc = fco.chi2_osc;
                    chi2_syst = fco.chi2_syst;
                    //best_dmsq = fco.dmsq;
                    //best_sinsq2t = fco.sinsq2tmm;
                    for(size_t i = 0; i < metric->GetSysts().GetNSplines(); ++i) {
                        //best_systs_osc[metric->GetSysts().spline_names[i]] = fco.best_fit_osc(i);
                        best_systs[metric->GetSysts().spline_names[i]] = fco.best_fit_syst(i);
                        syst_throw[metric->GetSysts().spline_names[i]] = fco.syst_throw(i);
                    }
                    tree.Fill();
                }
            }

            tree.Write();
        }
        {
            ofstream fcout(final_output_tag+"_unblind_BF_GOF.csv");
            fcout << "chi2_syst,";
            for(const std::string &name: metric->GetSysts().spline_names) {
                fcout << "best_" << name << "_syst," << name << "_throw";
            }
            fcout << "\r\n";

            for(const auto &out: outs) {
                for(const auto &fco: out) {
                    fcout << fco.chi2_syst << ",";
                    for(size_t i = 0; i < metric->GetSysts().GetNSplines(); ++i) {
                        fcout << fco.best_fit_syst(i) << "," << fco.syst_throw(i);
                    }
                    fcout << "\r\n";
                }
            }
        }
        std::vector<float> flattened_gofchi2s;
        for(const auto& out: outs) for(const auto& fco: out) flattened_gofchi2s.push_back(fco.chi2_syst);
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

        getConfirmation("Proceed to begin profile with systematic results only?","############################################");

        std::vector<int> permutation(metric->GetSysts().GetNSplines(), 0);
        std::iota(permutation.begin(), permutation.end(), metric->GetModel().nparams);
        std::shuffle(permutation.begin(), permutation.end(), myseed.global_rng);
        Eigen::VectorXi perm_vec = Eigen::VectorXi::Map(permutation.data(), permutation.size());
        perm_vec = perm_vec.array() - metric->GetModel().nparams;
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(perm_vec);

        PROfile prof(config, metric->GetSysts(), metric->GetModel(), *metric, myseed, scanfitconfig2, final_output_tag, chi2, true, nthread, best_fit);

        std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());
        Metropolis mh(simple_target{*metric}, adaptive_proposal(*metric, dseed(PROseed::global_rng)), best_fit, dseed(PROseed::global_rng));

        Eigen::MatrixXf covmat = Eigen::MatrixXf::Constant(nparams, nparams, 0);
        size_t count = 0;
        const auto action = [&](const Eigen::VectorXf &value) {
            covmat += (value-best_fit) * (value-best_fit).transpose();
            count += 1; 
        };
        mh.run(fitconfig2.MCMCburn,fitconfig2.MCMCiter, action);

        covmat /= count;
        Eigen::VectorXf inv_best_fit = best_fit.array().abs().max(1e-10f).inverse();
        Eigen::MatrixXf fraccovmat = inv_best_fit.asDiagonal() * covmat * inv_best_fit.asDiagonal();

        Eigen::VectorXf inv_sqrt_diag = fraccovmat.diagonal().array().abs().max(1e-10f).sqrt().inverse();
        Eigen::MatrixXf corrmat = inv_sqrt_diag.asDiagonal() * fraccovmat * inv_sqrt_diag.asDiagonal();

        size_t nspline = metric->GetSysts().GetNSplines();
        Eigen::MatrixXf spline_corrmat = corrmat.block(metric->GetModel().nparams, metric->GetModel().nparams, nspline, nspline);
        //Eigen::MatrixXf permuted_spline_corrmat = perm * spline_corrmat * perm;


        log<LOG_ERROR>(L"%1% || Showing profile minima for nuisance parameters in random order.") % __func__ ;
        size_t above2 = 0, above3 = 0;
        for(size_t i = 0; i < permutation.size(); ++i) {
            size_t idx = permutation[i];
            float val = prof.onesig.GetPointY(idx);
            if(val >= 2 || val <= -2) above2++;
            if(val >= 3 || val <= -3) above3++;
            if(val >= 3 || val <= -3)
                log<LOG_ERROR>(L"\033[101m");
            else if(val >= 2 || val <= -2)
                log<LOG_ERROR>(L"\033[103m");
            log<LOG_ERROR>(L"%1% || Parameter %2% profile minimum: %3%") % __func__ % i % val;
            if(val >= 2 || val <= -2)
                log<LOG_ERROR>(L"\033[0m");
        }

        log<LOG_ERROR>(L"%1% || Found %2% values pulled beyond 2 sigma and %3% values pulled beyond 3 sigma.") % __func__ % above2 % above3;

        log<LOG_ERROR>(L"%1% || ################################################") % __func__;

        getConfirmation("Proceed to show +/- 1 sigma ranges for nuisance parameters?","############################################");

        log<LOG_ERROR>(L"%1% || Showing profile 1 sigma ranges for nuisance parameters in random order.") % __func__ ;
        for(size_t i = 0; i < permutation.size(); ++i) {
            size_t idx = permutation[i];
            float val = prof.onesig.GetPointY(idx);
            float p1 = prof.onesig.GetErrorYhigh(idx);
            float m1 = prof.onesig.GetErrorYlow(idx);
            log<LOG_ERROR>(L"%1% || Parameter %2% profile 1 sigma range: %3%(-%4%,+%5%)") 
                % __func__ % i % val % m1 % p1;
        }
        log<LOG_ERROR>(L"%1% || ################################################") % __func__;

        //getConfirmation("Proceed to show correlation matrix for nuisance parameters without names?","############################################");

        TCanvas c;
        // Not working right now
        //TH2D corrhist_perm("crhp", "", nspline, 0, nspline, nspline, 0, nspline);
        //for(size_t i = 0; i < nspline; ++i) {
        //    for(size_t j = 0; j < nspline; ++j) {
        //        corrhist_perm.SetBinContent(i+1, j+1, permuted_spline_corrmat(i,j));
        //    }
        //}
        //corrhist_perm.SetMaximum(1);
        //corrhist_perm.SetMinimum(-1);
        //corrhist_perm.Draw("colz");
        //c.Print((final_output_tag + "_unblinding_unnamed_spline_corr.pdf").c_str());

        getConfirmation("Proceed to show full nuisance parameter profile results with names?","############################################");

        prof.Plot(config, metric->GetSysts(), metric->GetModel(), *metric, myseed, 
                final_output_tag+"_unblinding_nuisance", true, best_fit, Eigen::VectorXf(), true); 

        TH2D corrhist("crh", "", nspline, 0, nspline, nspline, 0, nspline);
        for(size_t i = 0; i < nspline; ++i) {
            std::string label =
                config.m_mcgen_variation_plotname_map.at(metric->GetSysts().spline_names[i]).c_str();
            corrhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            corrhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            for(size_t j = 0; j < nspline; ++j) {
                corrhist.SetBinContent(i+1, j+1, spline_corrmat(i,j));
            }
        }
        corrhist.SetMaximum(1);
        corrhist.SetMinimum(-1);
        corrhist.Draw("colz");
        c.Print((final_output_tag + "_unblinding_spline_corr.pdf").c_str());

        std::vector<TH1D> priors, posteriors;
        Eigen::MatrixXf prior_covariance, spline_covariance;
        // Fix physics parameters
        std::vector<int> fixed_pars;
        for(size_t i = 0; i < metric->GetModel().nparams; ++i) fixed_pars.push_back(i);
        Metropolis mh_post(simple_target{*metric}, adaptive_proposal(*metric, dseed(PROseed::global_rng), fixed_pars), best_fit, dseed(PROseed::global_rng));
        log<LOG_INFO>(L"%1% || Starting global getPostFitErrorBand() ") % __func__;
        std::unique_ptr<TGraphAsymmErrors> post_err_band = getMCMCErrorBand(mh_post, fitconfig2.MCMCburn, fitconfig2.MCMCiter, config, prop, *metric, best_fit, posteriors, spline_covariance);

        std::string hname = "#chi^{2}/ndof = " + to_string_prec(chi2,3) + "/" + to_string(config.m_num_bins_total_collapsed);
        PROspec cv = FillCVSpectrum(config, prop, true);
        PROspec bf = FillRecoSpectra(config, prop, metric->GetSysts(), metric->GetModel(), best_fit, true);
        TH1D post_hist("psth", hname.c_str(), config.m_num_bins_total_collapsed, config.m_channel_bin_edges[0].data());
        TH1D pre_hist("preh", hname.c_str(), config.m_num_bins_total_collapsed, config.m_channel_bin_edges[0].data());
        for(size_t i = 0; i < config.m_num_bins_total_collapsed; ++i) {
            post_hist.SetBinContent(i+1, bf.Spec()(i));
            pre_hist.SetBinContent(i+1, cv.Spec()(i));
        }

        std::unique_ptr<TGraphAsymmErrors> err_band = getErrorBand(config, prop, metric->GetSysts());

        std::vector<TPaveText> texts;
        TPaveText chi2text(0.55, 0.50, 0.85, 0.58, "NDC");
        chi2text.AddText(hname.c_str());
        chi2text.SetFillColor(0);
        chi2text.SetBorderSize(0);
        chi2text.SetTextAlign(12);
        //chi2text.SetTextSize(0.035); 
        texts.push_back(chi2text);

        plot_channels((final_output_tag+"_unblinding_hist.pdf"), config, cv, bf, data, err_band.get(), post_err_band.get(), texts);

        getConfirmation("Proceed to show full BF result?","############################################");

        ofstream global_fit_out;
        global_fit_out.open(final_output_tag+"_unblinding_global_fit.txt");
        log<LOG_ERROR>(L"%1% || ################################################") % __func__;
        log<LOG_ERROR>(L"%1% || ########### Global Best Fit Results ############") % __func__;
        log<LOG_ERROR>(L"%1% || ################################################") % __func__;
        log<LOG_ERROR>(L"%1% || Global Best Fit chi^2: %2%") %__func__ % chi2;
        log<LOG_ERROR>(L"%1% || at paramters: ") % __func__;

        global_fit_out << "Global best fit:\n";

        bool use_phys = (size_t)best_fit.size() == metric->GetModel().nparams + metric->GetSysts().GetNSplines();
        for(long i = 0; i < best_fit.size(); i++){

            if(use_phys && i < (long)metric->GetModel().nparams){
                log<LOG_ERROR>(L"%1% || %2%  :  %3% ") % __func__ % metric->GetModel().pretty_param_names[i].c_str() % best_fit(i);
                global_fit_out << metric->GetModel().param_names[i]
                    << " : " << best_fit(i) << "\n";
            }else{
                long idx = use_phys ? i - metric->GetModel().nparams : i;
                log<LOG_ERROR>(L"%1% || %2%  :  %3% ") % __func__ % metric->GetSysts().spline_names[idx].c_str() % best_fit(i);
                global_fit_out << metric->GetSysts().spline_names[idx]
                    << " : " << best_fit(i) << "\n";
            }
        }
        log<LOG_ERROR>(L"%1% || ################################################") % __func__;
        global_fit_out.close();

        getConfirmation("Proceed to show full profile result?","############################################");

        prof.Plot(config, metric->GetSysts(), metric->GetModel(), *metric, myseed, 
                final_output_tag+"_unblinding", true, best_fit); 

        TH2D corrhist_full("crhf", "", nparams, 0, nparams, nparams, 0, nparams);
        for(size_t i = 0; i < nparams; ++i) {
            std::string label = i < metric->GetModel().nparams 
                ? metric->GetModel().pretty_param_names[i]
                : config.m_mcgen_variation_plotname_map.at(metric->GetSysts().spline_names[i-metric->GetModel().nparams]).c_str();
            corrhist_full.GetXaxis()->SetBinLabel(i+1, label.c_str());
            corrhist_full.GetYaxis()->SetBinLabel(i+1, label.c_str());
            for(size_t j = 0; j < nparams; ++j) {
                corrhist_full.SetBinContent(i+1, j+1, corrmat(i,j));
            }
        }
        corrhist_full.SetMaximum(1);
        corrhist_full.SetMinimum(-1);
        corrhist_full.Draw("colz");
        c.Print((final_output_tag + "_unblinding_corr.pdf").c_str());

        getConfirmation("Proceed to calculate FC pvalue?","############################################");

        log<LOG_ERROR>(L"%1% || -- Calculating FC pvalue using %2% samples   ") % __func__ % nuniv;

        std::vector<std::vector<float>> dchi2sFC;
        dchi2sFC.reserve(FCthreads);
        std::vector<std::vector<fc_out>> outsFC;
        outsFC.reserve(FCthreads);
        std::vector<std::thread> threadsFC;
        gof_mode = false;
        for(size_t i = 0; i < nthread; i++) {
            dchi2sFC.emplace_back();
            outsFC.emplace_back();
            fc_args args{todo + (i >= addone), &dchi2sFC.back(), &outsFC.back(), config, prop, metric->GetSysts(), "PROCNP", best_fit, L, scanfitconfig2,(*myseed.getThreadSeeds())[i], (int)i, true,gof_mode};

            threadsFC.emplace_back([args]() {
                    PROfit::fc_worker(args);
                    });
        }
        for(auto&& t: threadsFC) {
            t.join();
        }

        log<LOG_ERROR>(L"%1% || Finished throws. %2%") % __func__ % __LINE__;
        {
            TFile fout((final_output_tag+"_unblind_BF_FC.root").c_str(), "RECREATE");
            fout.cd();
            float chi2_osc, chi2_syst, best_dmsq, best_sinsq2t;
            std::map<std::string, float> best_systs_osc, best_systs, syst_throw;
            TTree tree("tree", "tree");
            tree.Branch("chi2_osc", &chi2_osc); 
            tree.Branch("chi2_syst", &chi2_syst); 
            tree.Branch("best_dmsq", &best_dmsq); 
            tree.Branch("best_sinsq2t", &best_sinsq2t); 
            tree.Branch("best_systs_osc", &best_systs_osc); 
            tree.Branch("best_systs", &best_systs); 
            tree.Branch("syst_throw", &syst_throw);

            for(const auto &out: outsFC) {
                for(const auto &fco: out) {
                    chi2_osc = fco.chi2_osc;
                    chi2_syst = fco.chi2_syst;
                    best_dmsq = fco.dmsq;
                    best_sinsq2t = fco.sinsq2tmm;
                    for(size_t i = 0; i < metric->GetSysts().GetNSplines(); ++i) {
                        best_systs_osc[metric->GetSysts().spline_names[i]] = fco.best_fit_osc(i);
                        best_systs[metric->GetSysts().spline_names[i]] = fco.best_fit_syst(i);
                        syst_throw[metric->GetSysts().spline_names[i]] = fco.syst_throw(i);
                    }
                    tree.Fill();
                }
            }

            tree.Write();
        }
        {
            ofstream fcout(final_output_tag+"_unblind_BF_FC.csv");
            fcout << "chi2_osc,chi2_syst,best_dmsq,best_sinsq2t";
            for(const std::string &name: metric->GetSysts().spline_names) {
                fcout << ",best_" << name << "_osc,best_" << name << "," << name << "_throw";
            }
            fcout << "\r\n";

            for(const auto &out: outsFC) {
                for(const auto &fco: out) {
                    fcout << fco.chi2_osc << "," << fco.chi2_syst << "," << fco.dmsq << "," << fco.sinsq2tmm;
                    for(size_t i = 0; i < metric->GetSysts().GetNSplines(); ++i) {
                        fcout << fco.best_fit_osc(i) << "," << fco.best_fit_syst(i) << "," << fco.syst_throw(i);
                    }
                    fcout << "\r\n";
                }
            }
        }
        std::vector<float> flattened_gofchi2sFC;
        for(const auto& out: outsFC) for(const auto& fco: out) flattened_gofchi2sFC.push_back(fco.chi2_syst);
        std::sort(flattened_gofchi2sFC.begin(), flattened_gofchi2sFC.end());
        log<LOG_ERROR>(L"%1% || All: %2% ") % __func__ % flattened_gofchi2sFC;
        log<LOG_ERROR>(L"%1% || chi: %2% ") % __func__ % chi2;
        auto itFC = std::lower_bound(flattened_gofchi2sFC.begin(), flattened_gofchi2sFC.end(), chi2);
        size_t indexFC =  std::distance(flattened_gofchi2sFC.begin(),itFC);
        size_t count_aboveFC = flattened_gofchi2sFC.size()-indexFC;
        float pvalFC = (float)count_aboveFC/(float)nuniv;
        log<LOG_ERROR>(L"%1% || Finished throws. %2% %3% %4%") % __func__ % __LINE__% indexFC % count_aboveFC;
        log<LOG_ERROR>(L"%1% || GOF pval after throwing %2% universes is %3%") % __func__ % nuniv % pvalFC ;

        getConfirmation("Proceed to calculate surface?","############################################");

        log<LOG_ERROR>(L"Surface not implemented yet.");

        return 0;
    }

}
