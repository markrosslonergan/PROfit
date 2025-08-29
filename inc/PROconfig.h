#ifndef PROCONFIG_H_
#define PROCONFIG_H_

// STANDARD
#include <Eigen/Eigen>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <unordered_map>
#include <climits>
#include <cstdlib>
#include <numeric>
#include <stdexcept>

// TINYXML2
#include "tinyxml2.h"

// EIGEN
#include <Eigen/Eigen>

//PROfit
#include "PROlog.h"
#include "MurmurHash3.h"

//ROOT
#include "TTreeFormula.h"
#include "TColor.h"


/*eweight_type here to switch between uboone style "float" and SBNcode style "float"  */
//#define TYPE_FLOAT
#ifdef TYPE_FLOAT  
typedef float eweight_type;
#else
typedef double eweight_type;
#endif

namespace PROfit{

    /*typedef the base "eventweight_class" branch that reweightable systematics are stored in in uboonestyle systematics 
    */
    typedef std::map<std::string, std::vector<eweight_type>> eweight_map;


    /* Struct: Branch variable is a SBNfit era class to load using TTReeFormula a givem variable (or function of variables) 
     * Note: was originally split between float/int, but moved to TTreeFormula
     */

    struct BranchVariable{
      std::string name;
      std::string type;
      std::string associated_hist;
      std::string associated_systematic;
      bool central_value;

      std::shared_ptr<TTreeFormula> branch_formula=nullptr;
      std::shared_ptr<TTreeFormula> branch_monte_carlo_weight_formula = nullptr;
      std::shared_ptr<TTreeFormula> branch_true_value_formula=nullptr;
      std::shared_ptr<TTreeFormula> branch_true_L_formula=nullptr;
      std::shared_ptr<TTreeFormula> branch_true_pdg_formula=nullptr;
      std::shared_ptr<TTreeFormula> branch_true_proton_mom_formula=nullptr;
      std::shared_ptr<TTreeFormula> branch_true_proton_costh_formula=nullptr;
      std::vector<std::shared_ptr<TTreeFormula>> branch_other_values_formulas;

      std::string true_param_name;
      std::string true_L_name;
      std::string pdg_name;
      int model_rule;
      int include_systematics;

      std::vector<std::string> other_param_names;

      bool hist_reweight;
      std::string true_proton_mom_name;
      std::string true_proton_costh_name;

      //constructor
      BranchVariable(std::string n, std::string t, std::string a) : name(n), type(t), associated_hist(a), central_value(false), model_rule(-9), include_systematics(1), hist_reweight(false){}
      BranchVariable(std::string n, std::string t, std::string a_hist, std::string a_syst, bool cv) : name(n), type(t), associated_hist(a_hist), associated_systematic(a_syst), central_value(cv), model_rule(-9), include_systematics(1), hist_reweight(false){}
 
      /* Function: Return the TTreeformula for branch 'name', usually it's the reconstructed variable */
      std::shared_ptr<TTreeFormula> GetFormula(){
	return branch_formula;
      }

      void SetTrueParam(const std::string& true_parameter_def){ true_param_name = true_parameter_def; return;}
      void SetPDG(const std::string& pdg_def){ pdg_name = pdg_def; return;}
      void SetTrueL(const std::string& true_L_def){true_L_name = true_L_def; return;}
      void SetModelRule(const std::string & model_rule_def){model_rule = std::stoi(model_rule_def); return;}
      void SetIncludeSystematics(int insyst){include_systematics = insyst; return;} 

      void SetOtherParams(const std::string &other_parameter_def) { 
          size_t start = 0;
          while(start < other_parameter_def.size()) {
              size_t semicolon = other_parameter_def.find_first_of(';', start);
              if(semicolon == std::string::npos) {
                  other_param_names.push_back(other_parameter_def.substr(start, other_parameter_def.size()-start));
                  start = other_parameter_def.size();
              } else {
                  other_param_names.push_back(other_parameter_def.substr(start, semicolon - start));
                  start = semicolon + 1;
              }
          }
      }
      
      void SetReweight(bool inbool){ hist_reweight = inbool; return;}
      bool GetReweight() const {return hist_reweight;}
      void SetTrueLeadingProtonP(const std::string& true_protonp_def){ true_proton_mom_name = true_protonp_def; return;}
      void SetTrueLeadingProtonCosth(const std::string& true_protoncosth_def){ true_proton_costh_name = true_protoncosth_def; return;}

      //Function: evaluate branch "pdg", and return the value. Usually it's the pdg value of the particle
      //Note: when called, if the corresponding TreeFormula is not linked to a TTree, value of ZERO (0) will be returned.
      template <typename T = int>
	T GetTruePDG() const;

      int GetModelRule() const{
	return model_rule;
      };

      int GetIncludeSystematics() const{
	return include_systematics;
      };


      // Function: evaluate additional weight setup in the branch and return in floating precision 
      // Note: if no additional weight is set, value of 1.0 will be returned.
      inline
      float GetMonteCarloWeight() const{
	if(branch_monte_carlo_weight_formula){ 

	  return (float)branch_monte_carlo_weight_formula->EvalInstance();
	}
	return 1.0;
      }

      
      //Function: evaluate branch 'name' and return the value. Usually its reconstructed quantity
      //Note: when called, if the corresponding TreeFormula is not linked to a TTree, value of ZERO (0) will be returned.
      template <typename T=float>
	T GetValue() const;


      //Function: evaluate formula 'true_L_name' and return the value. Usually it's true baseline.
      //Note: when called, if the corresponding TreeFormula is not linked to a TTree, value of ZERO (0) will be returned.
      template <typename T=float>
	T GetTrueL() const;
      

      //Function: evaluate formula 'true_param_name' and return the value. Usually it's true energy  
      //Note: when called, if the corresponding TreeFormula is not linked to a TTree, value of ZERO (0) will be returned.
      template <typename T=float>
	T GetTrueValue() const;

      //Function: evaluate formula 'true_proton_mom_name' and return the value. Used for reweighting
      //Note: when called, if the corresponding TreeFormula is not linked to a TTree, value of ZERO (0) will be returned.
      template <typename T=float>
	T GetTrueLeadProtonMom() const;


      //Function: evaluate formula 'true_proton_costh_name' and return the value. Used for reweighting
      //Note: when called, if the corresponding TreeFormula is not linked to a TTree, value of ZERO (0) will be returned.
      template <typename T=float>
	T GetTrueLeadProtonCosth() const;

      template<typename T=float>
    std::vector<T> GetOtherValues() const;

    };

 

    /* 
     * Class: Primary booking class for loading in info from XML and saving all information on mode_detector_channel_subchannel information.
     * Note:
     *  PROconfig to be created once and only once per PROfit executable and then passed by reference to various more complex functions, and ignored when not needed. 
     *  Contains information and collapsing matricies for collapsing subchannels->channels also. Filled once, on construction and then used by later classes when needed.
     */


    class PROconfig {
        private:

            //indicator of whether each channel/detector/subchannel is used
            std::vector<bool> m_mode_bool;
            std::vector<bool> m_detector_bool;
            std::vector<bool> m_channel_bool;
            std::vector<std::vector<bool>>  m_subchannel_bool;


            //map from subchannel name/index to global index and channel index
            std::unordered_map<std::string, size_t> m_map_fullname_subchannel_index;
            std::vector<size_t> m_vec_subchannel_index; //vector of global subchannel index, in increasing order
            std::vector<size_t> m_vec_channel_index;    //vector of corresponding channel index
            std::vector<size_t> m_vec_global_reco_index_start;  //vector of global reco bin index, in increasing order
            std::vector<size_t> m_vec_global_true_index_start;  //vector of global true bin index, in increasing order
            std::vector<std::vector<size_t>> m_vec_global_other_index_start;  //vector of global true bin index, in increasing order


            //---- PRIVATE FUNCTION ------

            /* Function: construct a matrix T, which will be used to collapse matrix and vectors */
            void construct_collapsing_matrix();

            /* Function: remove any mode/detector/channel/subchannels in the configuration xml that are not used from consideration
            */
            void remove_unused_channel();


            /* Function: ignore any file that is associated with unused channels 
            */
            void remove_unused_files();


            /* Function: fill in mapping between subchannel name/index to global indices */
            void generate_index_map();


            /* Function: given an input vector that's sorted in ascending order, and input val, return the index of elmeent which is equal to val
             * Note: it gives exception when input value is not present in the vector 
             */
            size_t find_equal_index(const std::vector<size_t>& input_vec, size_t val) const;

            /* Function: given an input vector that's sorted in ascending order, and input val, return the index of the closest element which is equal or smaller than val */
            size_t find_less_or_equal_index(const std::vector<size_t>& input_vec, size_t val) const;

            /* Function: given global bin index, return associated global subchannel index 
             * Note: not used anymore 
             */
            size_t find_global_subchannel_index_from_global_bin(size_t global_index, const std::vector<size_t>& num_subchannel_in_channel, const std::vector<size_t>& num_bins_in_channel, size_t num_channels, size_t num_bins_total) const;


        public:

            PROconfig() {}; //always have an empty constructor?

            /* Constructor Function: Need a string passed which is the filename (with path) of the configuration xml */
            PROconfig(const std::string &xmlname,bool rate_only=false);

            /*
             * Function: Use TinyXML2 to load XML */
            int LoadFromXML(const std::string & filename);
            uint32_t hash;

            static bool SameChannels(const PROconfig &one, const PROconfig &two);

            std::string m_xmlname;	
            float m_plot_pot;
            std::vector<std::string> m_fullnames;

            size_t m_num_detectors;
            size_t m_num_channels;
            size_t m_num_modes;
            size_t m_num_other_vars;

            /*Vectors of length num_channels. Unless specificed all refer to fittable (reco) variables*/
            std::vector<size_t> m_num_subchannels; 
            std::vector<size_t> m_channel_num_bins;
            std::vector<std::vector<float> > m_channel_bin_edges;
            std::vector<std::vector<float> > m_channel_bin_widths;

            /* New true bins to save the truth level variables in addition.*/
            std::vector<size_t> m_channel_num_truebins;
            std::vector<std::vector<float> > m_channel_truebin_edges;
            std::vector<std::vector<float> > m_channel_truebin_widths;

            // Same but for "other" vars
            std::vector<std::vector<size_t>> m_channel_num_other_bins;
            std::vector<std::vector<std::vector<float>>> m_channel_other_bin_edges;
            std::vector<std::vector<std::vector<float>>> m_channel_other_bin_widths;

            bool m_has_oscillation_patterns;

            //the xml names are the way we track which channels and subchannels we want to use later
            std::vector<std::string> m_mode_names; 			
            std::vector<std::string> m_mode_plotnames; 			

            std::vector<std::string> m_detector_names; 		
            std::vector<std::string> m_detector_plotnames; 		

            std::vector<std::string> m_channel_names; 		
            std::vector<std::string> m_channel_plotnames; 		
            std::vector<std::string> m_channel_units; 		
            std::vector<std::vector<std::string>> m_channel_other_units;

            std::vector<std::vector<std::string >> m_subchannel_names; 
            std::vector<std::vector<std::string >> m_subchannel_plotnames; 
            std::vector<std::vector<std::string >> m_subchannel_colors; 
            std::vector<std::vector<size_t >> m_subchannel_datas; 

            size_t m_num_bins_detector_block;
            size_t m_num_bins_mode_block;
            size_t m_num_bins_total;

            size_t m_num_truebins_detector_block;
            size_t m_num_truebins_mode_block;
            size_t m_num_truebins_total;

            std::vector<size_t> m_num_other_bins_detector_block;
            std::vector<size_t> m_num_other_bins_mode_block;
            std::vector<size_t> m_num_other_bins_total;

            size_t m_num_bins_detector_block_collapsed;
            size_t m_num_bins_mode_block_collapsed;
            size_t m_num_bins_total_collapsed;

            std::vector<size_t> m_num_other_bins_detector_block_collapsed;
            std::vector<size_t> m_num_other_bins_mode_block_collapsed;
            std::vector<size_t> m_num_other_bins_total_collapsed;

            /* Eigen Matrix for collapsing subchannels->channels*/
            Eigen::MatrixXf collapsing_matrix;
            std::vector<Eigen::MatrixXf> other_collapsing_matrices;

            //This section entirely for montecarlo generation of a covariance matrix or PROspec 
            bool m_write_out_variation;
            bool m_form_covariance;
            std::string m_write_out_tag;
            int m_num_variation_type_covariance = 0;
            int m_num_variation_type_spline = 0;
            int m_num_variation_type_flat = 0;
            int m_num_variation_type_norm = 0;

            int m_num_mcgen_files;
            std::vector<std::string> m_mcgen_tree_name;	
            std::vector<std::string> m_mcgen_file_name;	
            std::vector<long int> m_mcgen_maxevents;	
            std::vector<float> m_mcgen_pot;	
            std::vector<float> m_mcgen_scale;	
            std::vector<int> m_mcgen_numfriends;	
            std::vector<bool> m_mcgen_fake;
            std::map<std::string,std::vector<std::string>> m_mcgen_file_friend_map;
            std::map<std::string,std::vector<std::string>> m_mcgen_file_friend_treename_map;
            std::vector<std::vector<std::string>> m_mcgen_additional_weight_name;
            std::vector<std::vector<bool>> m_mcgen_additional_weight_bool;
            std::vector<std::vector<std::shared_ptr<BranchVariable>>> m_branch_variables;
            std::vector<std::vector<std::string>> m_mcgen_eventweight_branch_names;
            std::vector<std::vector<int>> m_mcgen_eventweight_branch_syst;

            //specific bits for covairiancegeneration
            bool m_use_mcstats = false;
            std::vector<std::string> m_mcgen_weightmaps_formulas;
            std::vector<bool> m_mcgen_weightmaps_uses;
            std::vector<std::string> m_mcgen_weightmaps_patterns;
            std::vector<std::string> m_mcgen_weightmaps_mode;
            std::vector<std::string> m_mcgen_variation_allowlist;
            std::vector<std::string> m_mcgen_variation_denylist;
            std::vector<std::string> m_mcgen_variation_type;
            std::map<std::string, std::string> m_mcgen_variation_type_map;
            std::map<std::string, std::string> m_mcgen_variation_plotname_map;
            std::map<std::string, int> m_mcgen_variation_binning_map;
            std::map<std::string, std::vector<double>> m_mcgen_variation_knobval_override;
            std::map<std::string, std::vector<std::string>> m_mcgen_variation_tags;
            std::map<std::string, std::vector<std::string>> m_mcgen_shapeonly_listmap; //a map of shape-only systematic and corresponding subchannels
            std::vector<std::tuple<std::string, std::string, float>> m_mcgen_correlations;
            std::map<std::string, float> m_mcgen_variation_prior;
            std::map<std::string, float> m_mcgen_variation_prior_centers;
      
            //FIX skepic
            std::vector<std::string> systematic_name;

            //Some model infomation
            std::string m_model_tag;
            std::vector<int> m_model_rule_index;
            std::vector<std::string> m_model_rule_names;


            bool m_bool_rate_only;
            //----- PUBLIC FUNCTIONS ------
            //


            /* Function: return matrix T, of size (m_num_bins_total, m_num_bins_total_collapsed), which will be used to collapse matrix and vectors 
             * Note: To collapse a full matrix M, please do T.transpose() * M * T
             * 	     To collapse a full vector V, please do T.transpose() * V
             */
            inline 
                Eigen::MatrixXf GetCollapsingMatrix() const {return collapsing_matrix; }
            inline
                Eigen::MatrixXf GetCollapsingMatrix(int other_index) const {return other_collapsing_matrices[other_index]; }

            /* Function: Calculate how big each mode block and decector block are, for any given number of channels/subchannels, before and after the collapse
             * Note: only consider mode/detector/channel/subchannels that are actually used 
             */
            void CalcTotalBins();


            /* Function: given subchannel full name, return global subchannel index 
             * Note: index start from 0, not 1
             */
            size_t GetSubchannelIndex(const std::string& fullname) const;

            /* Function: given global index (in the full vector), return global subchannel index of associated subchannel
             * Note: returns a 0-based index 
             */
            size_t GetSubchannelIndexFromGlobalBin(size_t global_index) const;

            /* Function: given global true index , return global subchannel index of associated subchannel
             * Note: returns a 0-based index 
             */
            size_t GetSubchannelIndexFromGlobalTrueBin(size_t global_trueindex) const;

            /* Function: given subchannel global index, return corresponding channel index 
             * Note: index start from 0, not 1
             */
            size_t GetChannelIndex(size_t subchannel_index) const;


            /* Function: given subchannel global index, return corresponding global bin start
             * Note: global bin index start from 0, not 1
             */
            size_t GetGlobalBinStart(size_t subchannel_index) const;

            size_t GetCollapsedGlobalBinStart(size_t channel_index) const;

            /* Function: given channel index, return list of bin edges for this channel */
            const std::vector<float>& GetChannelBinEdges(size_t channel_index) const;

            /* Function: given channel index, return number of true bins for this channel */
            size_t GetChannelNTrueBins(size_t channel_index) const;

            /* Function: given subchannel global index, return corresponding global bin start
             * Note: global bin index start from 0, not 1
             */
            size_t GetGlobalTrueBinStart(size_t subchannel_index) const;

            /* Function: given channel index, return list of bin edges for this channel */
            const std::vector<float>& GetChannelTrueBinEdges(size_t channel_index) const;

            /* Function: given channel index, return number of other bins for this channel and other var*/
            size_t GetChannelNOtherBins(size_t channel_index, size_t other_index) const;

            /* Function: given subchannel global index, return corresponding global bin start
             * Note: global bin index start from 0, not 1
             */
            size_t GetGlobalOtherBinStart(size_t subchannel_index, size_t other_index) const;

            size_t GetCollapsedGlobalOtherBinStart(size_t channel_index, size_t other_index) const;

            /* Function: given channel index, return list of bin edges for this channel */
            const std::vector<float>& GetChannelOtherBinEdges(size_t channel_index, size_t other_index) const;

            /* Function: Given a global channel index return the local channel index */
            size_t GetLocalChannelIndex(size_t global_channel_index) const; 

            /* Function: Hex to int*/
            int HexToROOTColor(const std::string& hexColor) const;

            /* Calculate hash of unique properties of XML config for PROpeller */
            uint32_t CalcHash() const;
                   

            Eigen::VectorXf GetCollapsedBinCenters() const;
            Eigen::VectorXf GetCollapsedOtherBinCenters(int index) const;
            Eigen::VectorXf GetCollapsedBinWidths() const;
            Eigen::VectorXf GetCollapsedOtherBinWidths(int index) const;




    };


    //----------- BELOW: Definition of BranchVariable templated member function. Please don't move it elsewhere !! ---------------
    //----------- BELOW: Definition of BranchVariable templated member function. Please don't move it elsewhere !! ---------------
    //----------- BELOW: Definition of BranchVariable templated member function. Please don't move it elsewhere !! ---------------

    template <typename T>
        T BranchVariable::GetTruePDG() const{
            if(branch_true_pdg_formula == NULL) return static_cast<T>(0);
            else{
                return static_cast<T>(branch_true_pdg_formula->EvalInstance());
            }
        }


    template <typename T>
        T BranchVariable::GetValue() const{
            if(branch_formula == NULL) return static_cast<T>(0);
            else{
                return static_cast<T>(branch_formula->EvalInstance());
            }
        }

    template <typename T>
        T BranchVariable::GetTrueL() const{
            if(branch_true_L_formula == NULL) return static_cast<T>(0);
            else{
                return static_cast<T>(branch_true_L_formula->EvalInstance());
            }
        }

    template <typename T>
        T BranchVariable::GetTrueValue() const{
            if(branch_true_value_formula == NULL) return static_cast<T>(0);
            else{
                return static_cast<T>(branch_true_value_formula->EvalInstance());
            }
        }

    template <typename T>
        T BranchVariable::GetTrueLeadProtonMom() const{
            if(branch_true_proton_mom_formula == NULL) return static_cast<T>(0);
            else{
                return static_cast<T>(branch_true_proton_mom_formula->EvalInstance());
            }
        }

    template <typename T>
        T BranchVariable::GetTrueLeadProtonCosth() const{
            if(branch_true_proton_costh_formula == NULL) return static_cast<T>(0);
            else{
                            return static_cast<T>(branch_true_proton_costh_formula->EvalInstance());
            }
        }

    template <typename T>
        std::vector<T> BranchVariable::GetOtherValues() const {
            std::vector<T> ret;
            for(const auto &formula: branch_other_values_formulas) {
                ret.push_back(formula->EvalInstance());
            }
            return ret;
        }

    //----------- ABOVE: Definition of BranchVariable templated member function. END ---------------

}
#endif
