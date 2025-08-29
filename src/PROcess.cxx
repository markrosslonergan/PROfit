#include "PROcess.h"
#include "PROlog.h"
#include "PROspec.h"
#include "PROsyst.h"
#include "PROtocall.h"
#include "TH2D.h"

#include <Eigen/Eigen>

#include <algorithm>
#include <random>


namespace PROfit {
    PROspec FillCVSpectrum(const PROconfig &inconfig, const PROpeller &inprop, bool binned){
        PROspec myspectrum(inconfig.m_num_bins_total);

        if(binned) {
            for(int i = 0; i < inprop.hist.rows(); ++i) {
                for(size_t k = 0; k < myspectrum.GetNbins(); ++k) {
                    myspectrum.Fill(k, inprop.hist(i, k));
                }
            }
        } else {
            for(size_t i = 0; i<inprop.trueLE.size(); ++i){
                float add_w = inprop.added_weights[i]; 
                myspectrum.Fill(inprop.bin_indices[i], add_w);
            }
        }
        return myspectrum;
    }

    PROspec FillOtherCVSpectrum(const PROconfig &inconfig, const PROpeller &inprop, size_t other_index){
        PROspec myspectrum(inconfig.m_num_other_bins_total[other_index]);
        for(size_t i = 0; i<inprop.trueLE.size(); ++i){
            float add_w = inprop.added_weights[i]; 
            if(inprop.other_bin_indices[i][other_index] >= 0)
                myspectrum.Fill(inprop.other_bin_indices[i][other_index], add_w);
        }
        return myspectrum;
    }

  PROspec FillRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, const std::vector<TH2D*> inweighthists, bool binned){
        PROspec myspectrum(inconfig.m_num_bins_total);
        Eigen::VectorXf phys   = params.segment(0, inmodel.nparams);
        Eigen::VectorXf shifts = params.segment(inmodel.nparams, params.size() - inmodel.nparams);

        if(binned) {
            Eigen::VectorXf systw = Eigen::VectorXf::Constant(inconfig.m_num_bins_total, 1);
            for(int i = 0; i < shifts.size(); ++i) {
                int binning = insyst.spline_binnings[i];
                const Eigen::MatrixXf &hist =
                    binning == -2 || binning == -1 ? inprop.hist 
                                                   : inprop.other_hists[binning];
                for(size_t k = 0; k < inconfig.m_num_bins_total; ++k) {
                    if(binning == -1) systw(k) *= insyst.GetSplineShift(i, shifts(i), k);
                    else {
                        float val = 0, unweighted = 0;
                        for(long int j = 0; j < hist.rows(); ++j) {
                            float binsystw = insyst.GetSplineShift(i, shifts(i), j);
                            val += binsystw * hist(j, k);
                            unweighted += hist(j,k);
                        }
                        if(unweighted > 0) systw(k) *= val/unweighted;
                    }
                }
            }

            for(long int i = 0; i < inprop.hist.rows(); ++i) {
                float le = inprop.histLE[i];

		//Reweighting from histogram here
		float hist_w = 1.0;
		if (inweighthists.size() ) {
		  size_t subchan = inconfig.GetSubchannelIndexFromGlobalTrueBin(inprop.true_bin_indices[i]);
		  std::string name = inconfig.m_fullnames[subchan];
		  log<LOG_DEBUG>(L"%1% || subchan = %2%") % __func__ % subchan;
		  if (name == "nu_ICARUS_numu_numucc") {
		    float pmom = static_cast<float>(inprop.pmom[i]);
		    float pcosth = static_cast<float>(inprop.pcosth[i]);
		    log<LOG_DEBUG>(L"%1% || i = %2%, pmom = %3%, pcosth = %4%") % __func__ % i % pmom % pcosth;
		    
		    for (size_t j = 0; j<inweighthists.size(); ++j){
		      TH2D h = *inweighthists[j];
		      int bin = h.FindBin(pmom,pcosth);
		      hist_w *= h.GetBinContent(bin);
		      log<LOG_DEBUG>(L"%1% || j = %2%, pmom = %3%, pcosth = %4%, hist_w = %5%") % __func__ % j % pmom % pcosth %  hist_w ;
		    }
		  }
		}
				  
                for(size_t j = 0; j < inmodel.model_functions.size(); ++j) {
                    float oscw = inmodel.model_functions[j](phys, le);
                    for(size_t k = 0; k < myspectrum.GetNbins(); ++k) {
                        myspectrum.Fill(k, systw(k) * oscw * hist_w * inmodel.hists[j](i, k));
                    }
                }
            }
        } else {
            for(size_t i = 0; i<inprop.trueLE.size(); ++i){

	      //Reweighting from histogram here	      
	      float hist_w = 1.0;
	      if (inweighthists.size() ) {
		size_t subchan = inconfig.GetSubchannelIndexFromGlobalTrueBin(inprop.true_bin_indices[i]);
		std::string name = inconfig.m_fullnames[subchan];
		if (name == "nu_ICARUS_numu_numucc") {
		  float pmom = static_cast<float>(inprop.pmom[i]);
		  float pcosth = static_cast<float>(inprop.pcosth[i]);
		  for (size_t j = 0; j<inweighthists.size(); ++j){
		    TH2D h = *inweighthists[j];
		    int bin = h.FindBin(pmom,pcosth);
		    hist_w *= h.GetBinContent(bin);
		    log<LOG_DEBUG>(L"%1% || j = %2%, pmom = %3%, pcosth = %4%, hist_w = %5%") % __func__ % j % pmom % pcosth %  hist_w ;
		  }
		}
	      }

	      float oscw  =  inmodel.model_functions[inprop.model_rule[i]](phys, inprop.trueLE[i]);
	      float add_w = inprop.added_weights[i]; 
	      const int reco_bin = inprop.bin_indices[i];

	      float systw = 1;
	      for(int j = 0; j < shifts.size(); ++j) {
		int binning = insyst.spline_binnings[j];
		const int spline_bin = 
		  binning == -2 ? inprop.true_bin_indices[i] :
		  binning == -1 ? inprop.bin_indices[i]
		  : inprop.other_bin_indices[i][binning];
		systw *= insyst.GetSplineShift(j, shifts[j], spline_bin);
	      }

	      float finalw = oscw * systw * add_w * hist_w;

	      myspectrum.Fill(reco_bin, finalw);
	    }
	}
    
	return myspectrum;
  }

  //Version without the reweight option
  PROspec FillRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, bool binned){
        PROspec myspectrum(inconfig.m_num_bins_total);
        Eigen::VectorXf phys   = params.segment(0, inmodel.nparams);
        Eigen::VectorXf shifts = params.segment(inmodel.nparams, params.size() - inmodel.nparams);

	if(binned) {
            Eigen::VectorXf systw = Eigen::VectorXf::Constant(inconfig.m_num_bins_total, 1);
            for(int i = 0; i < shifts.size(); ++i) {
                int binning = insyst.spline_binnings[i];
		const Eigen::MatrixXf &hist =
		  binning == -2 || binning == -1 ? inprop.hist
		  : inprop.other_hists[binning];
                for(size_t k = 0; k < inconfig.m_num_bins_total; ++k) {
		  if(binning == -1) systw(k) *= insyst.GetSplineShift(i, shifts(i), k);
		  else {
		    float val = 0, unweighted = 0;
		    for(long int j = 0; j < hist.rows(); ++j) {
		      float binsystw = insyst.GetSplineShift(i, shifts(i), j);
		      val += binsystw * hist(j, k);
		      unweighted += hist(j,k);
		    }
		    if(unweighted > 0) systw(k) *= val/unweighted;
		  }
                }
            }
            for(long int i = 0; i < inprop.hist.rows(); ++i) {
                float le = inprop.histLE[i];
		for(size_t j = 0; j < inmodel.model_functions.size(); ++j) {
		  float oscw = inmodel.model_functions[j](phys, le);
		  for(size_t k = 0; k < myspectrum.GetNbins(); ++k) {
		    myspectrum.Fill(k, systw(k) * oscw * inmodel.hists[j](i, k));
		  }
                }
            }
        } else {
	  for(size_t i = 0; i<inprop.trueLE.size(); ++i){
	    float oscw  =  inmodel.model_functions[inprop.model_rule[i]](phys, inprop.trueLE[i]);
	    float add_w = inprop.added_weights[i];
	    const int reco_bin = inprop.bin_indices[i];

	    float systw = 1;
	    for(int j = 0; j < shifts.size(); ++j) {
	      int binning = insyst.spline_binnings[j];
	      const int spline_bin =
		binning == -2 ? inprop.true_bin_indices[i] :
		binning == -1 ? inprop.bin_indices[i]
		: inprop.other_bin_indices[i][binning];
	      systw *= insyst.GetSplineShift(j, shifts[j], spline_bin);
	    }

	    float finalw = oscw * systw * add_w;

	    myspectrum.Fill(reco_bin, finalw);
	  }
        }
        return myspectrum;
  }
	    
			    ///////			    
  PROspec FillOtherRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, size_t other_index, std::vector<TH2D*> inweighthists){
        PROspec myspectrum(inconfig.m_num_other_bins_total[other_index]);
        Eigen::VectorXf phys   = params.segment(0, inmodel.nparams);
        Eigen::VectorXf shifts = params.segment(inmodel.nparams, params.size() - inmodel.nparams);

        for(size_t i = 0; i<inprop.trueLE.size(); ++i){

	  //Reweighting from histogram here                                                                                                                        
	  float hist_w = 1.0;
	  if (inweighthists.size() ) {
	    size_t subchan = inconfig.GetSubchannelIndexFromGlobalTrueBin(inprop.true_bin_indices[i]);
	    std::string name = inconfig.m_fullnames[subchan];
	    if (name == "nu_ICARUS_numu_numucc") {
	      float pmom = static_cast<float>(inprop.pmom[i]);
	      float pcosth = static_cast<float>(inprop.pcosth[i]);
	      for (size_t j = 0; j<inweighthists.size(); ++j){
		TH2D h = *inweighthists[j];
		int bin = h.FindBin(pmom,pcosth);
		hist_w *= h.GetBinContent(bin);
		log<LOG_DEBUG>(L"%1% || in other: j = %2%, pmom = %3%, pcosth = %4%, hist_w = %5%") % __func__ % j % pmom % pcosth %  hist_w ;
	      }
	    }
	  }

	  float oscw  =  inmodel.model_functions[inprop.model_rule[i]](phys, inprop.trueLE[i]);
	  float add_w = inprop.added_weights[i]; 
	  float systw = 1;
	  for(int j = 0; j < shifts.size(); ++j) {
	    int binning = insyst.spline_binnings[j];
	    const int spline_bin = 
	      binning == -2 ? inprop.true_bin_indices[i] :
	      binning == -1 ? inprop.bin_indices[i]
	      : inprop.other_bin_indices[i][binning];
	    systw *= insyst.GetSplineShift(j, shifts[j], spline_bin);
	  }
	  
	  float finalw = oscw * systw * add_w * hist_w;

	  if(inprop.other_bin_indices[i][other_index] >= 0)
	    myspectrum.Fill(inprop.other_bin_indices[i][other_index], finalw);
        }
        return myspectrum;
}

    PROspec FillSystRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, uint32_t seed, int other_index) {
        int nbins = other_index < 0 ? inconfig.m_num_bins_total : inconfig.m_num_other_bins_total[other_index],
            nbins_collapsed = other_index < 0 ? inconfig.m_num_bins_total_collapsed : inconfig.m_num_other_bins_total_collapsed[other_index];
        Eigen::VectorXf spec = Eigen::VectorXf::Constant(nbins, 0);
        Eigen::VectorXf cvspec = Eigen::VectorXf::Constant(nbins, 0);

        // TODO: We should think about centralizing rng in a thread-safe/thread-aware way
        static std::mt19937 rng{seed};
        std::vector<std::normal_distribution<float>> d_spline;
        for(size_t i = 0; i < insyst.GetNSplines(); ++i)
            d_spline.emplace_back(insyst.spline_centers(i), insyst.spline_priors(i));
        std::normal_distribution<float> d_cov;
        std::vector<float> throws;
        //Eigen::VectorXf throwC = Eigen::VectorXf::Constant(inconfig.m_num_bins_total, 0);
        Eigen::VectorXf throwC = Eigen::VectorXf::Constant(nbins_collapsed, 0);
        for(size_t i = 0; i < insyst.GetNSplines(); i++) {
            throws.push_back(d_spline[i](rng));
        }
        for(int i = 0; i < nbins_collapsed; i++)
            throwC(i) = d_cov(rng);


        if(other_index < 0) {
            Eigen::VectorXf systw = Eigen::VectorXf::Constant(nbins, 1);
            for(size_t i = 0; i < throws.size(); ++i) {
                int binning = insyst.spline_binnings[i];
                const Eigen::MatrixXf &hist =
                    binning == -2 || binning == -1 ? inprop.hist 
                                                   : inprop.other_hists[binning];
                for(int k = 0; k < nbins; ++k) {
                    if(binning == -1) systw(k) *= insyst.GetSplineShift(i, throws[i], k);
                    else {
                        float val = 0, unweighted = 0;
                        for(long int j = 0; j < hist.rows(); ++j) {
                            float binsystw = insyst.GetSplineShift(i, throws[i], j);
                            val += binsystw * hist(j, k);
                            unweighted += hist(j,k);
                        }
                        if(unweighted > 0) systw(k) *= val/unweighted;
                    }
                }
            }
            for(long int i = 0; i < inprop.hist.rows(); ++i) {
                for(int k = 0; k < nbins; ++k) {
                    spec(k) += systw(k) * inprop.hist(i, k);
                    cvspec(k) += inprop.hist(i, k);
                }
            }
        } else {
            for(size_t i = 0; i<inprop.trueLE.size(); ++i){
                float add_w = inprop.added_weights[i]; 
                float systw = 1;
                for(size_t j = 0; j < throws.size(); ++j) {
                    int binning = insyst.spline_binnings[j];
                    const int spline_bin = 
                        binning == -2 ? inprop.true_bin_indices[i] :
                        binning == -1 ? inprop.bin_indices[i]
                                      : inprop.other_bin_indices[i][binning];
                    systw *= insyst.GetSplineShift(j, throws[j], spline_bin);
                }
                if(inprop.other_bin_indices[i][other_index] >= 0) {
                    float finalw = systw * add_w;
                    spec(inprop.other_bin_indices[i][other_index]) += finalw;
                    cvspec(inprop.other_bin_indices[i][other_index]) += add_w;
                }
            }
        }

        if(insyst.GetNCovar() == 0) {
            Eigen::VectorXf final_spec = other_index < 0 ? CollapseMatrix(inconfig, spec) : CollapseMatrix(inconfig, spec, other_index);
            return PROspec(final_spec, final_spec.array().sqrt());
        }

        Eigen::MatrixXf decomp_cov = insyst.DecomposeFractionalCovariance(inconfig, cvspec);
        Eigen::VectorXf collapsed_spec = other_index < 0 ? CollapseMatrix(inconfig, spec) : CollapseMatrix(inconfig, spec, other_index);
        Eigen::VectorXf final_spec = collapsed_spec + decomp_cov * throwC;

        //std::vector<float> stdVec(final_spec.data(), final_spec.data() + final_spec.size());
        //log<LOG_INFO>(L"%1% | final_spec is %2% ") % __func__ % stdVec;


        return PROspec(final_spec, final_spec.array().sqrt());
    }

    PROspec FillSplineRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, int spline, uint32_t seed, int other_index) {
        int nbins = other_index < 0 ? inconfig.m_num_bins_total : inconfig.m_num_other_bins_total[other_index];
        Eigen::VectorXf spec = Eigen::VectorXf::Constant(nbins, 0);

        // TODO: We should think about centralizing rng in a thread-safe/thread-aware way
        static std::mt19937 rng{seed};
        std::normal_distribution<float> d(insyst.spline_centers(spline), insyst.spline_priors(spline));
        float spline_throw = d(rng);
        int binning = insyst.spline_binnings[spline];

        if(other_index < 0) {
            const Eigen::MatrixXf &hist =
                binning == -2 || binning == -1 ? inprop.hist 
                                               : inprop.other_hists[binning];
            for(long int i = 0; i < hist.rows(); ++i) {
                float systw = 1.0;
                if(binning != -1) systw = insyst.GetSplineShift(spline, spline_throw, i);
                for(int k = 0; k < nbins; ++k) {
                    float binsystw = systw;
                    if(binning == -1) binsystw *= insyst.GetSplineShift(spline, spline_throw, k);
                    spec(k) += binsystw * hist(i, k);
                }
            }
        } else {
            for(size_t i = 0; i<inprop.trueLE.size(); ++i){
                float add_w = inprop.added_weights[i]; 
                const int spline_bin = 
                    binning == -2 ? inprop.true_bin_indices[i] :
                    binning == -1 ? inprop.bin_indices[i]
                                  : inprop.other_bin_indices[i][binning];
                float systw = insyst.GetSplineShift(spline, spline_throw, spline_bin);
                float finalw = systw * add_w;
                if(inprop.other_bin_indices[i][other_index] >= 0) {
                    spec(inprop.other_bin_indices[i][other_index]) += finalw;
                }
            }
        }

        return PROspec(spec, spec.array().sqrt());
    }
};
