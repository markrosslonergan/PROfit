/**
 * @file PROconfig.h
 * @brief Central configuration class for the PROfit neutrino oscillation analysis framework.
 * @author PROfit Collaboration
 *
 * @details PROconfig is the master booking class that parses the analysis XML file and stores
 * all information about the mode/detector/channel/subchannel hierarchy, binning definitions,
 * collapsing matrices, MC file lists, systematic variation definitions, and detector-variation
 * (DetVar) file management.  It is constructed once per executable and then passed by
 * (const) reference to all downstream PROfit objects.
 *
 * Also defines the supporting types BranchVariable (a per-subchannel ROOT/flat-file variable
 * reader) and ROOTFormula (a TTreeFormula-based concrete implementation of BranchVariable::Formula).
 */
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
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <climits>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <regex>

// TINYXML2
#include "tinyxml2.h"

// EIGEN
#include <Eigen/Eigen>

//PROfit
#include "PROlog.h"
#include "MurmurHash3.h"

//ROOT
#include "TTreeFormula.h"
#include "TH1.h"
#include "TH2.h"
#include "TColor.h"


/*eweight_type here to switch between uboone style "float" and SBNcode style "float"  */
//#define TYPE_FLOAT
#ifdef TYPE_FLOAT  
typedef float eweight_type;
#else
typedef double eweight_type;
#endif

namespace PROfit{

    /** @brief Prior model applied to a spline nuisance parameter. */
    enum class SplinePriorType {
        Gaussian, ///< Gaussian pull term using the configured center and width.
        Uniform   ///< No pull term; the allowed interval is set by restrict.
    };

    /**
     * @brief Typedef for the event-weight map used by MicroBooNE-style (uboonestyle) systematics.
     * @details Each entry maps a systematic name to a vector of per-universe event weights.
     */
    typedef std::map<std::string, std::vector<eweight_type>> eweight_map;


    /**
     * @brief Per-subchannel variable reader backed by TTreeFormula or another source.
     * @details BranchVariable encapsulates all information needed to read one set of analysis
     * variables and event weights from an input file for a given subchannel.  Originally split
     * between float/int branches, it was unified around TTreeFormula evaluation.  It owns a
     * list of weight formulas and a list of variable formulas.
     */
    struct BranchVariable{
      std::string associated_hist;       ///< Name of the histogram/subchannel this branch fills.
      std::string associated_systematic; ///< Name of the associated systematic (if any).
      bool central_value;                ///< True if this branch provides the CV spectrum.

      /**
       * @brief Return value from a BranchVariable formula evaluation.
       * @details May contain a single float or a list of floats (e.g. for multi-particle events).
       */
      struct Value {
          std::vector<float> v;

          Value() {}
          Value(const std::vector<float> &v): v(v) {}
          Value(float v): v({v}) {}

          // Coerce the Value object into a single number
          float first() {return v[0];}
          size_t size() {return v.size();}
      };

      /**
       * @brief Abstract interface for loading event data from an input source.
       * @details Defined as an abstract class so future input backends (e.g., flat ROOT files,
       * HDF5, or direct arrays) can be added without modifying BranchVariable.
       */
      class Formula {
        public:
          virtual Value EvalInstance() = 0;
          virtual void LoadEvent(unsigned evtno) = 0;
          virtual std::string FormulaName() const = 0;
          virtual std::set<std::string> GetNeededBranchNames() const { return {}; }
          virtual ~Formula() {}
      };

      std::vector<std::shared_ptr<Formula>> branch_weight_formulas;   ///< Ordered list of weight formulas; product gives the total event weight.
      std::vector<std::shared_ptr<Formula>> branch_variable_formulas; ///< Ordered list of analysis variable formulas.

      int model_rule;          ///< Maps this subchannel's events to a physics probability type (-9 = unset).
      int include_systematics; ///< Flag controlling whether systematics are applied to this subchannel (1 = yes).

      std::vector<std::string> variable_names; ///< Names of the analysis variables read by this branch.

      bool hist_reweight; ///< If true, apply per-event weights from a 2D histogram instead of the formula.

      //constructor
      BranchVariable(std::string a) : associated_hist(a), central_value(false), model_rule(-9), include_systematics(1), hist_reweight(false){}
      BranchVariable(std::string a_hist, std::string a_syst, bool cv) :  associated_hist(a_hist), associated_systematic(a_syst), central_value(cv), model_rule(-9), include_systematics(1), hist_reweight(false){}
 

      void SetModelRule(const std::string & model_rule_def){model_rule = std::stoi(model_rule_def); return;}
      void SetIncludeSystematics(int insyst){include_systematics = insyst; return;} 

      //main loader for all variables
      void AddVariable(const std::string& var_in){ variable_names.push_back(var_in); return;}
     
      void SetReweight(bool inbool){ hist_reweight = inbool; return;}
      bool GetReweight() const {return hist_reweight;}

       int GetModelRule() const{
        return model_rule;
      };

      int GetIncludeSystematics() const{
        return include_systematics;
      };


      // Function: get individual weight by 0-based index. Returns 1.0 if index out of range.
      inline
      float GetWeight(int i) const{
        if(i >= 0 && i < (int)branch_weight_formulas.size() && branch_weight_formulas[i]){
          return branch_weight_formulas[i]->EvalInstance().first();
        }
        return 1.0;
      }

      // Function: get total weight (product of all defined weights). Returns 1.0 if no weights defined.
      inline
      float GetTotalWeight() const{
        float product = 1.0;
        for(const auto& f : branch_weight_formulas){
          if(f) product *= f->EvalInstance().first();
        }
        return product;
      }

      // Function: get product of weights at specified 1-based indices. Returns 1.0 if indices empty.
      inline
      float GetWeightProduct(const std::vector<int>& indices) const{
        float product = 1.0;
        for(int idx : indices){
          int i = idx - 1; // convert 1-based to 0-based
          if(i >= 0 && i < (int)branch_weight_formulas.size() && branch_weight_formulas[i]){
            product *= branch_weight_formulas[i]->EvalInstance().first();
          }
        }
        return product;
      }

      // Function: get number of defined weights
      inline
      int NumWeights() const{
        return (int)branch_weight_formulas.size();
      }

      std::vector<BranchVariable::Value> GetVariables() const {
          std::vector<BranchVariable::Value> ret;
          for(const auto &formula: branch_variable_formulas) {
              ret.push_back(formula->EvalInstance());
          }
          return ret;
      }


    };


    /**
     * @brief Concrete BranchVariable::Formula implementation backed by ROOT TTreeFormula.
     * @details Reads one or more TTreeFormula expressions from a TTree.  A single ROOTFormula
     * may wrap multiple TTreeFormula objects to support comma-separated multi-variable expressions.
     */
    class ROOTFormula: public BranchVariable::Formula {
      public:
        ROOTFormula(const std::string &name, const std::string &formula, TTree *t);
        BranchVariable::Value EvalInstance() override;
        void LoadEvent(unsigned eventno) override;
        std::string FormulaName() const override;
        // Returns the names of all TBranches referenced by this formula.
        std::set<std::string> GetNeededBranchNames() const;
        // Adds branches referenced by a single TTreeFormula to an existing set.
        static void AddFormulaBranches(const TTreeFormula* f, std::set<std::string>& result);
        // Extracts identifier tokens from a formula expression string, skipping
        // known ROOT/C keywords that can never be branch names.
        static void ExtractExprTokens(const std::string& expr, std::set<std::string>& result);
        virtual ~ROOTFormula() {}

      private:
        std::vector<std::unique_ptr<TTreeFormula>> fs;
        int treeNumber;
    };
 

    /**
     * @brief Primary configuration class: parses the analysis XML and stores the full
     *        mode/detector/channel/subchannel hierarchy, binning, and collapsing matrices.
     * @details PROconfig must be created once and only once per PROfit executable and then
     * passed by const reference to all downstream objects.  It reads the XML file via TinyXML2
     * and populates all internal members including:
     *   - mode/detector/channel/subchannel names and booleans,
     *   - multi-dimensional binning objects (Binning),
     *   - collapsing matrices (summing subchannels into channels),
     *   - MC file lists, tree names, and branch variables,
     *   - systematic variation parameters, knob definitions, and covariance settings, and
     *   - optional embedded data and detector-variation (DetVar) sections.
     * All members are set during construction and must not be modified afterwards.
     */


    class PROconfig {
        private:

            //map from subchannel name/index to global index and channel index
            std::unordered_map<std::string, size_t> m_map_fullname_subchannel_index;
            std::vector<size_t> m_vec_subchannel_index; //vector of global subchannel index, in increasing order
            std::vector<size_t> m_vec_channel_index;    //vector of corresponding channel index
            std::vector<std::vector<size_t>> m_vec_global_variable_index_start;  //vector of global true bin index, in increasing order

            //---- PRIVATE FUNCTION ------

            /* Function: construct a matrix T, which will be used to collapse matrix and vectors */
            void construct_variable_collapsing_matrices();

            /* Function: finalize mode/detector/channel/subchannel counts and build the list of
             * subchannel fullnames. (The old `use="false"` disable mechanism has been removed.)
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

            /**
             * @brief Find the variable index marked as the fitting variable in the XML.
             * @details Scans every <channel> for a <bins>/<bins2D> element carrying `fit="true"`,
             * walking that channel's binnings in the SAME order LoadFromXML assigns variable
             * indices (all <bins2D> first, then all <bins>). At most one binning per channel may
             * be marked, and every channel that marks one must agree on the resulting index —
             * the variable list is global, so a disagreement is always a config error.
             * @return The marked index, or 0 (the first variable) if no channel marks one, which
             * reproduces the historical behaviour of XMLs written before `fit=` existed.
             */
            int ResolveFitVariableFromXML(tinyxml2::XMLDocument &doc) const;

            /**
             * @brief Sanity-check the resolved i_prime once the variable and model tables exist.
             * @details Called at the end of LoadFromXML. Rejects an out-of-range index and an
             * index that is a model parameter's kinematic variable (e.g. the truth L/E binning) —
             * fitting the truth spectrum is never intended. Warns for a `plot="false"` variable.
             */
            void ValidateFitVariable() const;


        public:

            PROconfig() {}; //always have an empty constructor?

            /**
             * @brief Load an analysis configuration from an XML file.
             * @param xmlname      Filename (with path) of the configuration XML.
             * @param rate_only    Collapse the fitting variable to a single bin (--rateonly).
             * @param fit_variable Explicit override for the fitting variable index (`i_prime`).
             *                     -1 (the default) means "take it from the XML" — the binning
             *                     marked `fit="true"`, or variable 0 if none is marked.
             *                     A value >= 0 wins over the XML; it is what --fit-variable
             *                     passes, and what BuildDataConfig()/BuildDetVarConfig() pass so
             *                     that child configs always fit the same variable as their parent.
             */
            PROconfig(const std::string &xmlname,bool rate_only=false,int fit_variable=-1);

            /*
             * Function: Use TinyXML2 to load XML */
            int LoadFromXML(const std::string & filename);
            SplinePriorType GetSplinePriorType(const std::string &systematic) const {
                auto it = m_mcgen_variation_prior_types.find(systematic);
                return it == m_mcgen_variation_prior_types.end()
                    ? SplinePriorType::Gaussian
                    : it->second;
            }
            uint32_t hash;
            uint32_t detvar_hash;

            static bool SameChannels(const PROconfig &one, const PROconfig &two);

            /**
             * @brief N-dimensional binning descriptor.
             * @details Stores bin-edge vectors for each dimension.  1D binning is the common
             * case; 2D binning is used for multi-variable analyses.  Provides projection utilities
             * to reduce N-D spectra to 1D and to bin input values.
             */
            struct Binning {
              std::vector<std::vector<float>> bin_edges;
              Binning() {}
              Binning(const std::vector<std::vector<float>> &b): bin_edges(b) {}
              Binning(const std::vector<float> &b): bin_edges({b}) {}

              // Helper functions
              size_t NDim() const {return bin_edges.size();}
              // Total number of bins
              size_t NBins() const;

              // Project an input index across the full binning to a 1D index across the input dimension
              size_t ProjectIndex(size_t ind, size_t dim) const;
              Eigen::VectorXf ProjectSpectra(const Eigen::VectorXf &in, size_t dim) const;
              Eigen::VectorXf ProjectSpectraErrors(const Eigen::VectorXf &in, size_t dim) const;

              // Bin an input value
              int Bin(const std::vector<float> &v) const;
              int Bin(float v) const {return Bin(std::vector<float>({v}));}
              int Bin(const BranchVariable::Value &v) const {return Bin(v.v);}

              // number of bins along a dimension
              size_t NBinsAlong(unsigned dim) const {return (bin_edges[dim].size() == 0) ? 0 : bin_edges[dim].size()-1;}
              // number of bin edges along a dimension
              size_t NBinEdgesAlong(unsigned dim) const {return bin_edges[dim].size();}
              // return edges along a dimension
              std::vector<float> Edges(unsigned dim = 0) const {return bin_edges[dim];}
              // return widths along a dimension
              std::vector<float> Widths(unsigned dim = 0) const;
            };

            std::string m_xmlname;             ///< Path to the XML configuration file.
            std::vector<double> m_det_pot;     ///< Proton-on-target exposure per detector.
            std::vector<std::string> m_fullnames; ///< Fully-qualified subchannel names (mode_detector_channel_subchannel).

            size_t m_num_detectors; ///< Number of active detectors.
            size_t m_num_channels;  ///< Number of active analysis channels.
            size_t m_num_modes;     ///< Number of beam/run modes.
            size_t m_num_variables; ///< Number of analysis variables defined in the XML.

            std::vector<size_t> m_num_subchannels; ///< Number of subchannels per channel.

            /**
             * @brief Index of the fitting variable — the one variable the analysis actually fits.
             * @details Selects the default collapsing matrix (GetCollapsingMatrix()), the PROsyst
             * and PROdata handed to the metric, and the spectrum every metric builds. Set once, at
             * the top of LoadFromXML, from the binning marked `fit="true"` or from the
             * constructor's `fit_variable` override; 0 when neither is given. Two later parse
             * steps read it, which is why it must be resolved first: the `--rateonly` rebinning,
             * and the default `binning="reco"` of a <systematic> (which resolves to this index).
             * Not part of the config hash, so switching it does NOT invalidate the .bin caches:
             * PROpeller stores bin indices for every variable and PROcreate builds a SystStruct
             * vector per variable, so all variables are already in the cached binaries.
             */
            size_t i_prime = 0;

            /// Constructor-supplied override for i_prime; -1 = "resolve from the XML". See PROconfig().
            int m_requested_fit_variable = -1;

            // New
            std::vector<bool> m_channel_variable_plot_bool;
            std::vector<std::vector<PROconfig::Binning>> m_channel_variable_bins;

            //the xml names are the way we track which channels and subchannels we want to use later
            std::vector<std::string> m_mode_names; 			
            std::vector<std::string> m_mode_plotnames; 			

            std::vector<std::string> m_detector_names; 		
            std::vector<std::string> m_detector_plotnames; 		

            std::vector<std::string> m_channel_names;
            std::vector<std::string> m_channel_plotnames;
            std::vector<std::string> m_channel_xaxis_labels;
            std::vector<std::string> m_channel_units;

            std::vector<std::vector<std::string>> m_channel_variable_xaxis_labels;
            std::vector<std::vector<std::string>> m_channel_variable_units;
            std::vector<std::vector<int>> m_channel_variable_dims;

            std::vector<std::vector<std::string >> m_subchannel_names; 
            std::vector<std::vector<std::string >> m_subchannel_plotnames; 
            std::vector<std::vector<std::string >> m_subchannel_colors; 
            std::vector<std::vector<size_t >> m_subchannel_datas; 

            std::vector<size_t> m_num_variable_bins_detector_block;
            std::vector<size_t> m_num_variable_bins_mode_block;
            std::vector<size_t> m_num_variable_bins_total;
            std::vector<size_t> m_num_variable_bins_detector_block_collapsed; 
            std::vector<size_t> m_num_variable_bins_mode_block_collapsed; 
            std::vector<size_t> m_num_variable_bins_total_collapsed; 

            std::vector<std::vector<std::pair<float,float>>> m_variable_bin_to_edges;

            /** @brief Runtime-only fit region in COLLAPSED bin space, one mask per variable.
             *  @details Empty outer/inner vector = no mask = every bin active. Set via
             *  SetActiveBins() BEFORE any PROmetric is constructed: metrics snapshot the
             *  mask in their constructors, so later changes are not seen by existing
             *  metrics (by design — the mask is read-only during fitting and may be read
             *  concurrently by FC/MCMC worker threads). Deliberately NOT hashed and NOT
             *  serialized: it is per-run analysis state (PROjector, future bin-off
             *  studies), not part of the analysis definition. */
            std::vector<std::vector<char>> m_variable_active_bins_collapsed;

            std::vector<Eigen::MatrixXf> variable_collapsing_matrices;
            // Sparse companions to variable_collapsing_matrices (one nonzero per row).
            // Used by CollapseMatrix in the chi^2 inner loop; built once in construct_variable_collapsing_matrices.
            std::vector<Eigen::SparseMatrix<float>> variable_collapsing_matrices_sparse;
            std::vector<Eigen::VectorXf> collapsed_bin_widths;

            //This section entirely for montecarlo generation of a covariance matrix or PROspec 
            bool m_write_out_variation;
            bool m_form_covariance;
            std::string m_write_out_tag;
            int m_num_variation_type_covariance = 0;
            int m_num_variation_type_covariance_to_spline = 0;
            int m_num_variation_type_external_covariance = 0;
            int m_num_variation_type_external_covariance_to_spline = 0;
            int m_num_variation_type_spline = 0;
            int m_num_variation_type_spline_to_covariance = 0;
            int m_num_variation_type_flat = 0;
            int m_num_variation_type_norm = 0;
            int m_num_variation_type_norm_to_covariance = 0;
            int m_num_variation_type_hist1d = 0;
            int m_num_variation_type_hist2d = 0;
            int m_num_variation_type_explicit = 0;

            int m_num_mcgen_files;
            std::vector<std::string> m_mcgen_tree_name;	
            std::vector<std::string> m_mcgen_file_name;	
            std::vector<long int> m_mcgen_maxevents;	
            std::vector<float> m_mcgen_pot;	
            std::vector<float> m_mcgen_scale;	
            std::vector<int> m_mcgen_numfriends;	
            std::vector<bool> m_mcgen_fake;
            std::vector<float> m_mcgen_partial_load_frac;
            std::map<std::string,std::vector<std::string>> m_mcgen_file_friend_map;
            std::map<std::string,std::vector<std::string>> m_mcgen_file_friend_treename_map;
            std::vector<std::vector<std::vector<std::string>>> m_mcgen_weight_names; // file × branch × weight_index
            std::vector<std::vector<int>> m_mcgen_num_weights; // file × branch → count of weights
            std::vector<std::vector<std::shared_ptr<BranchVariable>>> m_branch_variables;
            std::vector<std::vector<std::string>> m_mcgen_eventweight_branch_names;
            std::vector<std::vector<int>> m_mcgen_eventweight_branch_syst;

            //specific bits for covairiancegeneration
            bool m_use_mcstats = false;
            std::string m_mcstat_systname = "mcstat";  ///< XML name of the mcstat systematic; the key its covariance is registered under (defaults to legacy "mcstat")
            std::vector<std::string> m_mcgen_weightmaps_formulas;
            std::vector<bool> m_mcgen_weightmaps_uses;
            std::vector<std::string> m_mcgen_weightmaps_patterns;
            std::vector<std::string> m_mcgen_weightmaps_mode;
            std::vector<std::string> m_mcgen_variation_allowlist;
            std::vector<std::string> m_mcgen_variation_denylist;
            std::vector<std::string> m_mcgen_variation_type;
            std::set<std::string> m_mcgen_variation_unmirrored;
            std::map<std::string, std::vector<double>> m_mcgen_explicit_weights;
            std::map<std::string, std::string> m_mcgen_variation_external_filename_map;
            std::map<std::string, std::array<int, 2>> m_mcgen_variation_histaxisvars_map;
            std::map<std::string, std::vector<TH1*>> m_mcgen_variation_hist1d_map;
            std::map<std::string, std::vector<TH2*>> m_mcgen_variation_hist2d_map;
            std::map<std::string, std::vector<std::pair<std::string, std::string>>> m_histvar_files_map; // map of histvar name to vector of (filename, histname) pairs
            std::map<std::string, std::vector<double>> m_histvar_knobvals_map; // map of histvar name to vector of knob values
            std::map<std::string, std::set<std::string>> m_histvar_subchannels_map; // map of histvar name to set of subchannels
            std::map<std::string, std::string> m_mcgen_variation_type_map;
            std::map<std::string, std::string> m_mcgen_variation_plotname_map;
            std::map<std::string, int> m_mcgen_variation_binning_map;
            std::map<std::string, std::vector<double>> m_mcgen_variation_knobval_override;
            std::map<std::string, std::vector<std::string>> m_mcgen_variation_tags;
            std::map<std::string, std::vector<std::string>> m_mcgen_shapeonly_listmap; //a map of shape-only systematic and corresponding subchannels
            std::vector<std::tuple<std::string, std::string, float>> m_mcgen_correlations;
            std::map<std::string, float> m_mcgen_variation_prior;
            std::map<std::string, float> m_mcgen_variation_prior_centers;
            std::map<std::string, SplinePriorType> m_mcgen_variation_prior_types; ///< Explicit per-spline prior models; absent means Gaussian.
            std::map<std::string, bool> m_mcgen_variation_force_0_cv; //map of systematics with force_0_cv=true (normalize shifts by shift at knob=0)
            std::map<std::string, std::vector<int>> m_mcgen_variation_include_only_weights; //map of systematics with include_only_weights (1-based indices of which weights to include in spline universes)
            std::map<std::string, std::pair<float,float>> m_mcgen_variation_restrict; //map of systematics with restrict="lo, hi" (clamp knob value during evaluation and fitting)
            std::map<std::string, float> m_mcgen_variation_scale; //map of systematics with scale factor to apply to weights (e.g., 0.001 for weights stored as x1000)
            std::map<std::string, float> m_mcgen_variation_inflate; //map of systematics with inflate factor: spline shifts are scaled about 1 (ratio -> 1 + inflate*(ratio-1)) before interpolation; covariance matrices are scaled by inflate^2
            std::map<std::string, int> m_mcgen_variation_num_decomp_knobs; //map of covariance_to_spline systematics to the number of eigenpairs to keep (-1 or missing = keep all)
            std::map<std::string, bool> m_mcgen_variation_include_resid_cov; //map of covariance_to_spline systematics to whether the un-kept eigenpairs are retained as a residual covariance (missing = true)
            std::map<std::string, std::string> m_mcgen_variation_apply_to_subchannel; //map of systematics with apply_to_subchannel="<wildcard>" (unanchored regex against subchannel fullnames; plain substrings work as-is): the systematic is only applied to matching subchannels, and its weight branch is only required in MCFiles that fill a matching subchannel
      
            //FIX skepic
            std::vector<std::string> systematic_name;

            //Some model infomation
            std::string m_model_tag;
            std::vector<int> m_model_rule_index;
            std::vector<std::string> m_model_rule_names;
            std::vector<int> m_model_parameter_index;
            std::vector<std::string> m_model_parameter_names;
            std::map<std::string,int> m_model_parameter_map;
            /// Optional per-model-parameter min/max bounds, read from the <parameter> tag's
            /// "min"/"max" attributes. Used by normalization-style models (e.g. template_fit)
            /// where each <parameter> names a subchannel and min/max are its scale bounds.
            std::vector<float> m_model_parameter_min;
            std::vector<float> m_model_parameter_max;

            bool m_bool_rate_only;
            //----- PUBLIC FUNCTIONS ------
            //


            /* Function: return matrix T, of size (m_num_bins_total, m_num_bins_total_collapsed), which will be used to collapse matrix and vectors 
             * Note: To collapse a full matrix M, please do T.transpose() * M * T
             * 	     To collapse a full vector V, please do T.transpose() * V
             */
            inline
                const Eigen::MatrixXf& GetCollapsingMatrix() const {return variable_collapsing_matrices[i_prime]; }
            inline
                const Eigen::MatrixXf& GetCollapsingMatrix(int other_index) const {return variable_collapsing_matrices[other_index]; }
            inline
                const Eigen::SparseMatrix<float>& GetCollapsingMatrixSparse() const {return variable_collapsing_matrices_sparse[i_prime]; }
            inline
                const Eigen::SparseMatrix<float>& GetCollapsingMatrixSparse(int other_index) const {return variable_collapsing_matrices_sparse[other_index]; }

            /* Function: Calculate how big each mode block and decector block are, for any given number of channels/subchannels, before and after the collapse
             * Note: only consider mode/detector/channel/subchannels that are actually used 
             */
            void CalcTotalBins();


            /* Function: given subchannel full name, return global subchannel index 
             * Note: index start from 0, not 1
             */
            size_t GetSubchannelIndex(const std::string& fullname) const;

            /* Function: given global subchannel index, return fullname
             * Note: index start from 0, not 1
             */
            std::string GetSubchannelName(size_t index) const;


            /* Function: given global index (in the full vector), return global subchannel index of associated subchannel
             * Note: returns a 0-based index 
             */
            size_t GetSubchannelIndexFromVariableGlobalBin(size_t global_index, size_t var_index) const;

            /* Function: given subchannel global index, return corresponding channel index 
             * Note: index start from 0, not 1
             */
            size_t GetLocalChannelIndexFromGlobalSubchannelIndex(size_t global_subchannel_index) const;

            /* Function: Given a global channel index return the local channel index */
            size_t GetLocalChannelIndexFromGlobalChannelIndex(size_t global_channel_index) const; 


            /* Function: given subchannel global index, return corresponding global bin start
             * Note: global bin index start from 0, not 1
             */
            size_t GetGlobalVariableBinStart(size_t subchannel_index, size_t other_index) const;

            size_t GetCollapsedGlobalVariableBinStart(size_t channel_index, size_t other_index) const;

            /* Function: given channel index, return list of bin edges for this channel */
            const Binning& GetChannelVariableBins(size_t channel_index, size_t other_index) const;

            /* Function: build the X-axis title for a channel as "label [unit]",
             * omitting either part if empty. For 2D variables, the combined
             * "xtitle;ytitle" is split and the first part is returned.
            */
            std::string GetChannelXAxisTitle(size_t channel_index) const;
            std::string GetChannelXAxisTitle(size_t channel_index, size_t other_index) const;
            std::string GetChannelAxisTitle(size_t channel_index, size_t other_index, size_t dim) const;

            /**
             * @brief Install a runtime fit-region mask over the collapsed bins of one variable.
             * @details Fatal error if the mask size does not match the variable's collapsed bin
             * count or if every bin is inactive. Must be called before metric construction to
             * take effect (metrics snapshot the mask in their constructors).
             * @param var_index  Variable (binning) index.
             * @param mask       One entry per collapsed bin; nonzero = active (included in fits).
             */
            void SetActiveBins(size_t var_index, const std::vector<char> &mask);

            /** @brief Remove all runtime fit-region masks (every bin active again). */
            void ClearActiveBins();

            /** @brief True if a fit-region mask has been installed for this variable. */
            bool HasActiveBins(size_t var_index) const;

            /** @brief True if the collapsed bin is in the fit region (always true when no mask is set). */
            bool IsBinActive(size_t var_index, size_t collapsed_bin) const;

            /** @brief Number of active collapsed bins for this variable (= total bins when no mask is set). Use for dof counting. */
            size_t NActiveBins(size_t var_index) const;

            /* Function: return the unit string for a channel's variable
             * (e.g. "MeV"), preferring the per-variable <bins unit="..."> entry
             * and falling back to the channel-level <channel unit="..."> entry.
             * Returns "" if neither is set, or for 2D variables (whose units
             * field stores the legacy combined "xtitle;ytitle" string).
             */
            std::string GetChannelUnit(size_t channel_index, size_t other_index) const;

            /* Function: Hex to int*/
            int HexToROOTColor(const std::string& hexColor) const;

            /* Calculate hash of unique properties of XML config for PROpeller */
            uint32_t CalcHash() const;
            uint32_t CalcDetVarHash() const;

            //---- Embedded data support ----
            // Whether a <data> section was found in the XML
            bool m_has_data_section = false;
            // Self-contained XML string for building a data-only PROconfig
            std::string m_data_xml_string;
            // Serialized <bins> XML for each channel (used to reconstruct data XML)
            std::vector<std::string> m_channel_bins_xml_strings;

            /* Build a data-only PROconfig from the embedded <data> section.
             * Returns a PROconfig with one "data" subchannel per channel
             * and MCFiles from the <data> block. */
            PROconfig BuildDataConfig() const;

            //---- Detector Variation (DetVar) support ----
            struct DetVarFile {
                std::string filename;
                std::string treename;
                std::string name;  // "cv" / "cv_N" or variation name like "Recomb2"
                float pot;
                float partial_load_frac = 1.0f;
                bool is_cv;
                size_t section_index;  // which DetVarSection this file belongs to
                int knobval = 0;
            };

            bool m_has_detvar_section = false;
            std::vector<DetVarFile> m_detvar_files;
            // Per-section subchannel lists (one inner vector per DetVarSection)
            std::vector<std::vector<std::string>> m_detvar_subchannels_per_section;
            // Per-section include_only_weights: 1-based indices of weight_N to use (empty = use all)
            std::vector<std::vector<int>> m_detvar_include_only_weights_per_section;
            // Per-section extra weights: additional weight expressions appended after inherited weights
            std::vector<std::vector<std::string>> m_detvar_extra_weights_per_section;
            // Per-section branch names used to match events between CV and variation files.
            // Parsed from cv_variation_matching_vars="run,subrun,event" on <DetVarSection>.
            std::vector<std::vector<std::string>> m_detvar_matching_vars_per_section;
            std::set<std::string> m_detvar_variation_names;  // variation names for lookup during systematics parsing
            // Per-section XML templates for building per-file DetVar configs
            // Each template contains channel/subchannel/model definitions and an MCFile template
            std::vector<std::string> m_detvar_xml_templates;
            // Set on DetVar mini-configs (via BuildDetVarConfig) to carry matching var branch
            // names into PROcess_CAFAna without round-tripping through the XML template.
            std::vector<std::string> m_detvar_matching_vars;

            /* Build a PROconfig for a single DetVar file (CV or variation).
             * file_index indexes into m_detvar_files. */
            PROconfig BuildDetVarConfig(size_t file_index) const;

            /* Get number of DetVar files (across all sections) */
            size_t GetNumDetVarFiles() const { return m_detvar_files.size(); }

            /* Get number of DetVarSection blocks */
            size_t GetNumDetVarSections() const { return m_detvar_xml_templates.size(); }


    };

    /* User-supplied pattern matching (subchannel fullnames, systematic names).
     * Patterns are ECMAScript regexes matched UNANCHORED via std::regex_search,
     * so a plain substring behaves exactly like the historical
     * std::string::find convention; anchor with ^...$ for a full-name match.
     * CompilePattern is fatal (logged exit) on an invalid pattern; `context`
     * names the caller/feature in that error message. */
    std::regex CompilePattern(const std::string &pattern, const std::string &context);
    inline bool PatternMatches(const std::string &name, const std::regex &re) {
        return std::regex_search(name, re);
    }
    /* All names matching pattern (compiled once); input order preserved. */
    std::vector<std::string> MatchNames(const std::vector<std::string> &names, const std::string &pattern, const std::string &context);
}
#endif
