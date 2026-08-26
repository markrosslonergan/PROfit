/**
 * @file PROcreate.h
 * @brief MC event-processing functions and systematic-structure definitions for PROfit.
 * @author PROfit Collaboration
 *
 * @details Defines SystStruct (the per-systematic container holding CV and variation spectra)
 * and CAFweightHelper (a flat-array helper for reading SBNcode/CAFAna weight branches).
 * Declares the top-level MC loading function PROcess_CAFAna() and supporting utilities for
 * creating PROdata objects, processing per-event weights, and saving/loading DetVar
 * PROpeller maps.
 */
#ifndef PROCREATE_H_
#define PROCREATE_H_

// STANDARD
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <map>
#include <ctime>
#include <cmath>

// EIGEN
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

//PROfit
#include "PROlog.h"
#include "PROconfig.h"
#include "PROdata.h"
#include "PROtocall.h"
#include "PROspec.h"
#include "PROpeller.h"

//CAFana
#include "sbnanaobj/StandardRecord/SRGlobal.h"
#include "sbnanaobj/StandardRecord/SRWeightPSet.h"

//Boost
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/version.hpp>

namespace PROfit{

    /**
     * @brief Per-systematic container holding the CV spectrum and all variation (multi-universe) spectra.
     * @details SystStruct is the fundamental unit of systematic information.  One SystStruct is created
     * per systematic variation defined in the XML.  During MC processing (PROcess_CAFAna) it is filled
     * event-by-event; afterwards PROsyst reads it to build splines or covariance matrices.
     *
     * Supported modes (stored in the `mode` field): "multisim", "minmax", "covar", "spline",
     * "external", "flat", "norm", "norm_to_covariance", "hist1d", "hist2d",
     * and "spline_to_covariance".
     */
    struct SystStruct {

        std::string systname;         ///< Name of this systematic (must match the XML tag).
        int n_univ;                   ///< Number of universes (weight variations) for this systematic.
        std::string mode;             ///< Type string: "covar", "spline", "external", "multisim", etc.
        std::string weight_formula;   ///< Formula string (or "1" if weight is taken directly from the branch).
        std::string external_filename;///< Path to an external covariance matrix file (for "external" mode).

        std::vector<eweight_type> knobval;   ///< Knob values (sigma shifts) associated with each universe.
        std::vector<eweight_type> knob_index;///< Universe index for each knob value.

        int index;    ///< Index of this systematic within the PROsyst ordering.
        uint32_t hash;///< MurmurHash3 of the originating PROconfig.
        std::vector<std::vector<std::array<float, 4>>> spline_coeffs; ///< Pre-computed spline coefficients (bin × segment × 4 coefficients).

        /// Binning-scheme index (-1 = use default binning).
        int binning = -1;
        std::shared_ptr<PROspec> p_cv;   ///< Shared pointer to the central-value spectrum.
        std::vector<std::shared_ptr<PROspec>> p_multi_spec; ///< Shared pointers to per-universe variation spectra.

        std::vector<int> norm_bins; ///< Bin indices used for shape-normalisation (empty = normalise all bins).
        float norm_value = 0.0f;    ///< Normalisation target integral (used in shape-only mode; only set for "norm" systematics).
        bool force_0_cv = false;    ///< If true, normalise spline shifts by the shift at knob=0.
        std::vector<int> include_only_weights; ///< 1-based indices of weight universes to include; empty = all.
        float scale = 1.0f;         ///< Scale factor applied to all weights (e.g. 0.001 for weights stored as x1000).
        float inflate = 1.0f;       ///< Uncertainty inflation factor: spline ratios are scaled about 1 (ratio -> 1 + inflate*(ratio-1)) before interpolation; covariance matrices are scaled by inflate^2.
        int num_decomp_knobs = -1;  ///< For "covariance_to_spline": number of top eigenpairs to keep as spline knobs (-1 = keep all).
        bool include_resid_cov = true; ///< For "covariance_to_spline": if true, retain the un-kept (smaller) eigenpairs as a "<systname>_resid_cov" covariance matrix instead of dropping them.
        bool has_restrict = false;  ///< If true, clamp the knob value to [restrict_lo, restrict_hi] during evaluation and fitting.
        float restrict_lo = 0.0f;   ///< Lower clamp bound (used only when has_restrict is true).
        float restrict_hi = 0.0f;   ///< Upper clamp bound (used only when has_restrict is true).
        std::string apply_to_subchannel; ///< Substring wildcard from XML apply_to_subchannel=; empty = systematic applies to every subchannel. Events in non-matching subchannels fill all universes at the CV weight (flat spline / zero covariance there).
        std::vector<std::string> apply_to_subchannel_names; ///< Resolved subchannel fullnames matched by apply_to_subchannel (bookkeeping/logging).

        //boost serialization
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version) {
            ar & systname;
            ar & n_univ;
            ar & mode;
            ar & weight_formula;
            ar & external_filename;
            ar & knobval;
            ar & knob_index;
            ar & index;
            ar & spline_coeffs;
            ar & binning;
            ar & p_cv;
            ar & p_multi_spec;
            ar & hash;
            ar & norm_bins;
            ar & norm_value;
            ar & force_0_cv;
            ar & include_only_weights;
            ar & scale;
            if (version >= 1) {
                ar & has_restrict;
                ar & restrict_lo;
                ar & restrict_hi;
            }
            if (version >= 2) {
                ar & num_decomp_knobs;
            }
            if (version >= 3) {
                ar & include_resid_cov;
            }
            if (version >= 4) {
                ar & inflate;
            }
            if (version >= 5) {
                ar & apply_to_subchannel;
                ar & apply_to_subchannel_names;
            }
        }


        SystStruct() = default;

        /*Function: Constructor for a blank systematic*/
        SystStruct(const std::string& in_systname, const int in_n_univ): SystStruct(in_systname, in_n_univ, "multisim", "1",{},{},0){hash=-1;}

        /*Function: Constructor for a systematic from knobs*/
        SystStruct(const std::string& in_systname, const int in_n_univ, const std::string& in_mode, const std::string& in_formula, const std::vector<eweight_type>& in_knobval, const std::vector<eweight_type>& in_knob_index, const int in_index): systname(in_systname), n_univ(in_n_univ), mode(in_mode), weight_formula(in_formula), knobval(in_knobval), knob_index(in_knob_index), index(in_index){}

        inline
            void SetMode(const std::string& in_mode){mode = in_mode; return;}

        inline
            void SetWeightFormula(const std::string& in_formula){weight_formula = in_formula; return;}


        std::vector<std::vector<eweight_type>> GetCovVec();
        std::vector<eweight_type> GetKnobs(int index, std::string variation);

        //----- Spectrum related functions ---
        //----- Spectrum related functions ---

        /* Function: clean up all the member spectra (but ONLY spectra) */
        void CleanSpecs();


        /* Function: create EMPTY spectra with given length 
        */ 
        void CreateSpecs(int num_bins);

        /* Function: given global bin index, and event weight, fill the central value spectrum */
        void FillCV(int global_bin, float event_weight);

        /* Function: given global bin index, and event weight, fill the spectrum of given universe */
        void FillUniverse(int universe, int global_bin, float event_weight);

        /* Function: return CV spectrum in PROspec */
        const PROspec& CV() const;

        /*Function: return the spectrum for variation at given universe */
        const PROspec& Variation(int universe) const;

        //---------- Helper Functions --------
        //---------- Helper Functions --------

        /*Function to set hash*/
        inline
            void SetHash(uint32_t inhash){ hash=inhash;}


        /* Return number of universes for this systematic */
        inline
            int GetNUniverse() const {return n_univ;}

        /* Return string of systematic name */
        inline 
            const std::string& GetSysName() const {return systname;}

        /* Check if weight formula is set for this ysstematic */
        inline 
            bool HasWeightFormula() const {return weight_formula == "1";}

        /* Return a string of weight formula for this systematic */
        inline 
            const std::string& GetWeightFormula() const {return weight_formula;}

        /* Function: check if num of universes of this systematics matches with its type 
         * Note: multisim mode can have many universes, while minmax mode can only have 2
         */
        void SanityCheck() const;
        void Print() const;

    };


    void saveSystStructVector(const std::vector<std::vector<SystStruct>> &structs, const std::string &filename);
    void loadSystStructVector(std::vector<std::vector<SystStruct>> &structs, const std::string &filename);




    /**
     * @brief Flat-array helper for reading SBNcode/CAFAna weight branches from a TTree.
     * @details Manages the flat C-array buffers needed to read the variable-length
     * weight universe arrays stored in the SBNcode StandardRecord format.
     * @note Array sizes are currently hardcoded; removing hardcoded limits is a known
     * improvement item.
     * @todo Remove hardcoded int array sizes.
     */
    struct CAFweightHelper{
        int i_wgt_univ_size;   ///< Total size of the flattened universe weight array (rec.mc.nu.wgt.univ..totarraysize).
        int i_wgt_size;        ///< Number of weight objects (rec.slc..length).
        int i_wgt_totsize;     ///< Total size of the weight array (rec.mc.nu.wgt..totalarraysize).

        float v_wgt_univ[100000];      ///< Flattened universe weight values.
        int v_wgt_univ_idx[50000];     ///< Start index of each systematic's universe weights.
        int v_wgt_idx[5000];           ///< Start index of each slice's weight list.
        int v_wgt_univ_length[5000];   ///< Number of universes for each weight object.
        int v_truth_index[100];        ///< Truth-level neutrino index for each slice.

        CAFweightHelper(){
            i_wgt_univ_size=0;
            i_wgt_size =0;
            i_wgt_totsize=0;
        };

        float GetUniverseWeight(int which_index , int which_uni){
            for(int s = 0; s<i_wgt_size;s++){
                if(v_truth_index[s]==0){

                    return v_wgt_univ[v_wgt_univ_idx[v_wgt_idx[s] + which_index] + which_uni];
                }
            }

            return 0;
        };

        /**
         * @brief Return the universe weight for a given neutrino, systematic, and local universe index.
         * @param nu_index    Neutrino truth index within this event.
         * @param syst_index  Systematic index within the weight list.
         * @param uni_index   Local universe index (0-based) within this systematic.
         * @return Weight value.
         */
        float GetUniverseWeight(int nu_index, int syst_index , int uni_index){
            size_t index = v_wgt_univ_idx[v_wgt_idx[nu_index] + syst_index] + uni_index;
            if(index > 100000)
                log<LOG_ERROR>(L"%1% || array size is too small to contain all universe weights. Try to access index: %2% ")%__func__% index;	
            return v_wgt_univ[index];
        }


    };


    /*----------Function to load from files------------------*/
    /*----------One per FILE-tyle being loaded---------------*/


    /**
     * @brief Main MC loading function: processes CAFAna output trees into PROpeller and SystStruct vectors.
     * @details Reads all MC files listed in @p inconfig, evaluates branch variables and weight formulas,
     * fills the PROpeller event store, and populates per-universe variation spectra in @p syst_vector.
     * Supports both XRootD and local file access.
     * @param inconfig    Analysis configuration specifying files, branches, and systematics.
     * @param syst_vector Output vector of SystStruct vectors (one inner vector per file); filled in-place.
     * @param inprop      Output PROpeller; filled in-place with per-event data.
     * @param noxrootd    If true, disable XRootD URL rewriting and use local paths directly.
     * @return 0 on success; non-zero on failure.
     */
    int PROcess_CAFAna(const PROconfig &inconfig, std::vector<std::vector<SystStruct>> &syst_vector, PROpeller &inprop, bool noxrootd = false);


    /**
     * @brief Create a vector of PROdata objects (one per channel) from the embedded data section.
     * @param configin  Analysis configuration containing the embedded <data> section.
     * @return Vector of collapsed PROdata objects, one per channel defined in the XML.
     */
    std::vector<PROdata> CreatePROdata(const PROconfig& configin);


    /* Function: assume currently reading one entry of a file, update systematic variation spectrum 
     * Note: designed to be called internally by PROcess_SBNfit() function
     *
     * Arguments: 
     * 		branch: pointer to branch variable, each corresponding to one subchannel 
     * 		eventweight_map: a map between systematic string to list of variation weights
     * 		subchannel_index: index associated with current branch/subchannel
     *		syst_vector: list of SystStruct TO BE UPDATED, each stores all variation spectra of one systematic
     *		syst_additional_weight: additional weight applied to systematic variation
     */

    void process_cafana_event(const PROconfig &inconfig, const std::shared_ptr<BranchVariable>& branch, const std::map<std::string, std::vector<eweight_type>*>& eventweight_map, float mcpot, int subchannel_index, std::vector<std::vector<SystStruct>> &syst_vector, const std::vector<float>& syst_additional_weight, const std::vector<char>& syst_applies, PROpeller& inprop);

    /**
     * @brief Convert a local or EOS file path to an XRootD URL.
     * @param fname_orig  Original file path.
     * @return XRootD-prefixed URL string suitable for use with TFile::Open().
     */
    std::string convertToXRootD(std::string fname_orig);

    /**
     * @brief Serialise a map of DetVar PROpeller objects to a single binary file.
     * @details The detvar_hash is stored in the file header and checked on load to detect
     * configuration mismatches.
     * @param props        Map from DetVar variation name to PROpeller (CV and all variations).
     * @param detvar_hash  Hash of the DetVar section of the analysis configuration.
     * @param filename     Output file path.
     */
    void saveDetVarProps(const std::map<std::string, PROpeller>& props, uint32_t detvar_hash, const std::string& filename);

    /**
     * @brief Deserialise a map of DetVar PROpeller objects from a binary file.
     * @param props    Output map to fill.
     * @param filename Input file path.
     * @return The detvar_hash stored in the file (for validation against the current config).
     */
    uint32_t loadDetVarProps(std::map<std::string, PROpeller>& props, const std::string& filename);


};

BOOST_CLASS_VERSION(PROfit::SystStruct, 5)

#endif
