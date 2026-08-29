/**
 * @file PROpeller.h
 * @brief MC event store for oscillation-weight computation in the PROfit framework.
 * @author PROfit Collaboration
 *
 * @details Defines PROpeller, the per-event Monte Carlo container that drives
 * oscillation weight calculations in PROfit (hence "propeller").  Each MC event
 * carries a central-value weight, a model-rule tag (mapping it to an oscillation
 * probability type), and per-variable values and bin indices.  Pre-binned 2D
 * histograms (PROhistStorage) are kept for fast spectrum filling.
 */
#ifndef PROPELLER_H_
#define PROPELLER_H_

#include "PROconfig.h"
#include "PROserial.h"
#include <Eigen/Eigen>
// STANDARD
#include <Eigen/src/Core/Matrix.h>
#include <vector>
#include <chrono>
namespace PROfit{
    /**
     * @brief Compact upper-triangular storage for per-variable 2D histogram pairs.
     * @details Stores an n×n collection of (n_reco × n_phys) Eigen matrices indexed by
     * two variable indices (i, j).  Only the upper triangle (i ≤ j) is stored explicitly;
     * accessing (i, j) with i > j returns the transpose of (j, i).  This saves memory for
     * multi-variable analyses while preserving symmetric access semantics.
     */
    class PROhistStorage {
        private:
            size_t n_vars = 0;             ///< Number of analysis variables.
            std::vector<Eigen::MatrixXf> data; ///< Flattened upper-triangle storage.

            /// Compute flat storage index for pair (i, j) with i ≤ j.
            size_t compute_index(size_t i, size_t j) const {
                return (i * n_vars) - (i * (i - 1)) / 2 + (j - i);
            }

            friend class boost::serialization::access;

            template<class Archive>
                void serialize(Archive& ar, const unsigned int version) {
                    (void)version;
                    ar & n_vars;

                    if (Archive::is_loading::value) {
                        if (n_vars > 0) {
                            data.resize(n_vars * (n_vars + 1) / 2);
                        } else {
                            data.clear();
                        }
                    }

                    if (n_vars > 0) {
                        for (auto& mat : data) {
                            ar & mat;
                        }
                    }
                }
        public:
            /** @brief Default constructor — creates empty storage. */
            PROhistStorage() {}
            /**
             * @brief Construct storage for @p n variables.
             * @param n Number of analysis variables; allocates n*(n+1)/2 matrices.
             */
            PROhistStorage(size_t n) {init(n);}

            /**
             * @brief Initialise storage for @p n variables.
             * @param n Number of analysis variables.
             */
            void init(size_t n) { n_vars = n;data.resize(n * (n + 1) / 2);}


            /**
             * @brief Read-only access to the 2D histogram matrix for variable pair (i, j).
             * @param i  Row variable index.
             * @param j  Column variable index.
             * @return Const reference (or transposed view) of the stored matrix.
             */
            Eigen::Ref<const Eigen::MatrixXf> operator()(size_t i, size_t j) const {
                if (i <= j) {
                    return data[compute_index(i, j)]; 
                } else {
                    return data[compute_index(j, i)].transpose();
                }
            }

            /**
             * @brief Read-only element access for bin (l, m) in the matrix for variable pair (rowvar, colvar).
             * @param rowvar  Row variable index.
             * @param colvar  Column variable index.
             * @param l       Row bin index within the matrix.
             * @param m       Column bin index within the matrix.
             * @return Element value as a float.
             */
            float operator()(size_t rowvar, size_t colvar, size_t l, size_t m) const {
                if (rowvar <= colvar) {
                    return data[compute_index(rowvar, colvar)](l, m);
                } else {
                    return data[compute_index(colvar, rowvar)](m, l); 
                }
            }


            /**
             * @brief w^T-style weighted sum over the logical histogram H(rowvar, colvar):
             * returns per-colvar-bin totals with @p w weighting the rowvar bins.
             * @details Avoids operator()(rowvar, colvar) for the lower triangle: binding an
             * Eigen::Ref<const MatrixXf> to the transposed view there forces a full-matrix
             * copy on every access; here the transpose is folded into the GEMV instead.
             */
            Eigen::VectorXf WeightedColSum(size_t rowvar, size_t colvar, const Eigen::VectorXf &w) const {
                if (rowvar <= colvar) return data[compute_index(rowvar, colvar)].transpose() * w;
                return data[compute_index(colvar, rowvar)] * w;
            }

            /// Column sums of the logical histogram H(rowvar, colvar) (per-colvar-bin totals),
            /// transpose-free like WeightedColSum.
            Eigen::VectorXf UnweightedColSum(size_t rowvar, size_t colvar) const {
                if (rowvar <= colvar) return data[compute_index(rowvar, colvar)].colwise().sum().transpose();
                return data[compute_index(colvar, rowvar)].rowwise().sum();
            }

            /**
             * @brief Mutable access to the matrix for variable pair (i, j) for filling.
             * @param i  Row variable index; must satisfy i ≤ j.
             * @param j  Column variable index.
             * @return Mutable reference to the stored matrix.
             * @warning Calling with i > j is a fatal error; the upper-triangle convention must be respected.
             */
            Eigen::MatrixXf& set(size_t i, size_t j) {
                if (i > j){
                    log<LOG_ERROR>(L"%1% || If your seeing this, something went wrong. dont access PROhistStorage out of order.") % __func__;
                    exit(EXIT_FAILURE);
                }
                return data[compute_index(i, j)];
            }


            /** @brief Return the number of variables for which storage was initialised. */
            size_t size() const { return n_vars; }
    };

    /**
     * @brief Per-event Monte Carlo container that drives oscillation weight calculations.
     * @details PROpeller (the "propeller" that moves the analysis forward) stores one entry
     * per simulated neutrino event.  It holds:
     *   - central-value event weights (`added_weights`),
     *   - a model-rule index per event mapping it to an oscillation probability type,
     *   - per-variable values and pre-computed bin indices for fast histogram filling,
     *   - pre-binned 2D histogram matrices (PROhistStorage) used by FillSpectra, and
     *   - optional per-event matching variable values for detector-variation alignment.
     *
     * The object is serialisable via Boost.Serialization and supports POT-scaling via scale().
     */
    class PROpeller {

        private:
            friend class boost::serialization::access;

            // Serialization function for boost that will allow for save state of propeller
            template <class Archive>
                void serialize(Archive& ar, const unsigned int version) {
                    ar & added_weights;
                    ar & model_rule;
                    ar & variable_mc_stat_err;
                    ar & variable_bin_indices;
                    ar & variable_hist_storage;
                    ar & variable_midbin;
                    ar & variable_values;
                    ar & hash;
                    if(version >= 1) {
                        ar & has_matching_vars;
                        if(has_matching_vars) {
                            ar & matching_var_values;
                        }
                    }
                }

        public:

            /** @brief Default constructor — creates an empty PROpeller with invalid hash. */
            PROpeller(){
                variable_values.clear();
                added_weights.clear();
                model_rule.clear();
                hash = -1;
            };

            /*Function: Primary Constructor from raw std::vectors of MC values */ 
            PROpeller( std::vector<std::vector<float>> &invars, std::vector<float> &inadded_weights,  std::vector<int> &inmodel_rule) :  added_weights(inadded_weights),  model_rule(inmodel_rule), variable_values(invars){
                //for(size_t i = 0; i < bin_indices.size(); ++i)
            };

            /* the Core MC is saved in these vectors.*/

            std::vector<float> added_weights;  ///< Per-event central-value weights (product of all CV weights and POT normalisation).
            std::vector<int>   model_rule;     ///< Per-event index mapping each event to an oscillation probability type (e.g., 0=NC, 1=CC).
            /// True when per-event matching variables are populated for detector-variation alignment.
            bool has_matching_vars = false;
            /// Per-event matching variable values; outer index = variable, inner index = event.
            /// Only populated when `m_detvar_matching_vars` is set on the mini-config.
            std::vector<std::vector<float>> matching_var_values;
            /// Per-event bin indices for each analysis variable; outer = variable, inner = event.
            std::vector<std::vector<int>> variable_bin_indices;
            /// Per-event values for each analysis variable; outer = variable, inner = event.
            std::vector<std::vector<float>> variable_values;
            /// Per-variable MC-stat effective-count vectors: sqrt(N_eff) = Sum(w)/sqrt(Sum(w^2))
            /// per bin, so 1/this^2 is the exact fractional MC-stat variance for weighted events.
            std::vector<Eigen::VectorXf> variable_mc_stat_err;
            /// Per-variable bin centre value vectors.
            std::vector<Eigen::VectorXf> variable_midbin;
            /// Pre-binned 2D histogram storage for all (variable, variable) pairs.
            PROhistStorage variable_hist_storage;

            uint32_t hash; ///< MurmurHash3 of the PROconfig used to create this PROpeller; checked during serialisation.

            /**
             * @brief Return the value of a given analysis variable for a given event.
             * @details Provided for clarity; direct access to variable_values[][] is equally valid.
             * @param i_variable  0-based variable index.
             * @param i_event     0-based event index.
             * @return Variable value as a float.
             * @throws std::runtime_error if @p i_variable is out of bounds.
             */
            float VariableValue(size_t i_variable, size_t i_event) const {
                if(i_variable >= variable_values.size()) {
                    log<LOG_ERROR>(L"%1% || Variable index %2% is out of bounds. "
                        L"You have %3% variables defined (indices 0-%4%). "
                        L"Check that the 'variable_index' in your <parameter> tag matches your <bins>/<variable> definitions in the XML.")
                        % __func__ % i_variable % variable_values.size() % (variable_values.size() > 0 ? variable_values.size() - 1 : 0);
                    throw std::runtime_error("Variable index out of bounds in VariableValue");
                }
                return variable_values[i_variable][i_event];
            }
            /**
             * @brief Return the pre-computed bin index of a given analysis variable for a given event.
             * @param i_variable  0-based variable index.
             * @param i_event     0-based event index.
             * @return Global bin index; -1 indicates the event is out of the defined range.
             * @throws std::runtime_error if @p i_variable is out of bounds.
             */
            int VariableBinIndex(size_t i_variable, size_t i_event) const {
                if(i_variable >= variable_bin_indices.size()) {
                    log<LOG_ERROR>(L"%1% || Variable index %2% is out of bounds. "
                        L"You have %3% variables defined (indices 0-%4%). "
                        L"Check that the 'variable_index' in your <parameter> tag matches your <bins>/<variable> definitions in the XML.")
                        % __func__ % i_variable % variable_bin_indices.size() % (variable_bin_indices.size() > 0 ? variable_bin_indices.size() - 1 : 0);
                    throw std::runtime_error("Variable index out of bounds in VariableBinIndex");
                }
                return variable_bin_indices[i_variable][i_event];
            }

            /** @brief Return the number of analysis variables stored in this PROpeller. */
            size_t NVariable() const {return variable_values.size();}
            /** @brief Return the number of MC events stored in this PROpeller. */
            size_t NEvent() const {return added_weights.size();}


            /**
             * @brief Serialise this PROpeller to a binary file using Boost.Serialization.
             * @param filename  Output file path.
             */
            void save(const std::string& filename) const {
                auto start = std::chrono::high_resolution_clock::now();
                std::ofstream ofs(filename, std::ios::binary);
                boost::archive::binary_oarchive oa(ofs);
                oa << *this;
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                log<LOG_INFO>(L"%1% || Serialization save of PROpeller into file  %2% took %3% seconds") % __func__ % filename.c_str() % elapsed.count();
            }

            /**
             * @brief Deserialise this PROpeller from a binary file using Boost.Serialization.
             * @param filename  Input file path.
             */
            void load(const std::string& filename) {
                auto start = std::chrono::high_resolution_clock::now();
                std::ifstream ifs(filename,std::ios::binary);
                boost::archive::binary_iarchive ia(ifs);
                ia >> *this;
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                log<LOG_INFO>(L"%1% || Serialization load of PROpeller from file  %2% took %3% seconds") % __func__ % filename.c_str() %elapsed.count();
            }

            /**
             * @brief Scale per-event weights and pre-binned histograms for POT-normalisation studies.
             * @details For each detector/subchannel name pattern found in @p scaling_map, all
             * matching events in `added_weights` and all matching bins in `variable_hist_storage`
             * are multiplied by the corresponding scale factor.  Useful for comparing different
             * POT exposures without re-running the full event loop.
             * @param inconfig      The PROconfig describing channel/subchannel names.
             * @param scaling_map   Map from subchannel name pattern (wildcard substring) to scale factor.
             * @note Scale factors must be strictly positive; a value ≤ 0 is a fatal error.
             */
            void scale(const PROconfig &inconfig, std::map<std::string, float> scaling_map){
                for (const auto& [detector, value] : scaling_map) {

                    if (value <= 0.0f) {
                        log<LOG_ERROR>(L"%1% || Scale factor %2% for '%3%' is invalid. Must be > 0.")
                            % __func__ % value % detector.c_str();
                        exit(EXIT_FAILURE);
                    }

                    log<LOG_INFO>(L"%1% || Wildcard '%2%' with scaling factor %3% matches:")
                        % __func__ % detector.c_str() % value;

                    // Unanchored regex (plain substrings behave as before); see PROconfig.h.
                    std::vector<std::string> scalenames = MatchNames(inconfig.m_fullnames, detector, "--scale pattern");

                    log<LOG_INFO>(L"%1% || %2% . ") % __func__  % scalenames;


                    std::vector<std::vector<int>> scaleotherbins;
                    for(size_t io =0; io<inconfig.m_num_variables; io++){
                        std::vector<int> tmpbins;
                        for(auto &name: scalenames){
                            size_t is = inconfig.GetSubchannelIndex(name);     
                            size_t ic = inconfig.GetLocalChannelIndexFromGlobalSubchannelIndex(is); 
                            size_t start = inconfig.GetGlobalVariableBinStart(is,io); 
                            for(size_t b = 0; b < inconfig.m_channel_variable_bins[ic][io].NBins(); b++){
                                tmpbins.push_back((int)(start+b));
                            }
                        }
                        scaleotherbins.push_back(tmpbins);

                    }


                    //Scale the binned bits first
                    for(size_t io =0; io<inconfig.m_num_variables; io++){
                        for(size_t jo =io; jo<inconfig.m_num_variables; jo++){
                            log<LOG_INFO>(L"%1% || and scales other bins  %2% .") % __func__  %  scaleotherbins[io];
                            for (int o : scaleotherbins[io]) {
                                for (int j : scaleotherbins[jo]) {
                                    variable_hist_storage.set(io,jo)(o, j) *= value;
                                }
                            }
                        }
                    }
                    //And then the unbinned weights
                    for (size_t i = 0; i < NEvent(); ++i) {
                        for(size_t io =0; io<inconfig.m_num_variables; io++){
                            if(io>0)break;//Hmm, we scale of the reco and not other bins. That seems fine, but might want to rethink
                            int bin = VariableBinIndex(io, i);
                            if (std::find(scaleotherbins[io].begin(), scaleotherbins[io].end(), bin) != scaleotherbins[io].end()) {
                                added_weights[i] *= value;
                            }
                        }
                    }

                    log<LOG_INFO>(L"%1% || Applied %2% scaling for '%3%'")
                        % __func__ % value % detector.c_str();

                }

            }

    };

}

BOOST_CLASS_VERSION(PROfit::PROpeller, 1)

#endif
