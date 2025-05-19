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


    /*Class: The PROpeller, which moves the analysis forward. A class to keep all MC events for oscllation event-by-event.
    */
    class PROpeller {

        private:
            friend class boost::serialization::access;
            int nevents;

            // Serialization function for boost that will allow for save state of propeller
            template <class Archive>
                void serialize(Archive& ar, [[maybe_unused]] const unsigned int version) {
                    ar & nevents;
                    ar & pcosth;
                    ar & pmom;
                    ar & trueLE;
                    ar & added_weights;
                    ar & bin_indices;
                    ar & model_rule;
                    ar & true_bin_indices;
                    ar & other_bin_indices;
                    ar & hist;
                    ar & other_hists;
                    ar & histLE;
                    ar & mcStatErr;
                    ar & otherMCStatErr;
                    ar & hash;
                }

        public:

            //Empty Constructor
            PROpeller(){
                nevents = -1;
                pmom.clear();
                pcosth.clear();
                trueLE.clear();
                added_weights.clear();
                bin_indices.clear();
                model_rule.clear();
                true_bin_indices.clear();
                hash = -1;
            };

            /*Function: Primary Constructor from raw std::vectors of MC values */ 
            PROpeller(const PROconfig &config, std::vector<float> &intruth, std::vector<float> &inpmom, std::vector<float> &inpcosth, std::vector<float> &inadded_weights, std::vector<int> &inbin_indices, std::vector<int> &inmodel_rule, std::vector<int> &intrue_bin_indices) : trueLE(intruth), added_weights(inadded_weights), bin_indices(inbin_indices), model_rule(inmodel_rule), true_bin_indices(intrue_bin_indices), pmom(inpmom), pcosth(inpcosth) {
                nevents = trueLE.size();
                hist = Eigen::MatrixXf::Constant(config.m_num_truebins_total, config.m_num_bins_total, 0);
                for(size_t i = 0; i < bin_indices.size(); ++i)
                    hist(true_bin_indices[i], bin_indices[i]) += added_weights[i];
                hash = config.hash;
            };

            /* the Core MC is saved in these vectors.*/

            std::vector<float> trueLE;
            std::vector<float> added_weights;
            std::vector<int>   bin_indices;        /*Precalculated Bin index*/
            std::vector<int>   model_rule;
            std::vector<int>   true_bin_indices;
            std::vector<float> pmom;
            std::vector<float> pcosth;
            std::vector<std::vector<int>> other_bin_indices;
            Eigen::MatrixXf    hist;
            std::vector<Eigen::MatrixXf> other_hists;
            Eigen::VectorXf    histLE;
            Eigen::VectorXf    mcStatErr;
            std::vector<Eigen::VectorXf> otherMCStatErr;
            uint32_t           hash;

            // boost serialize save to file
            void save(const std::string& filename) const {
                auto start = std::chrono::high_resolution_clock::now();
                std::ofstream ofs(filename, std::ios::binary);
                boost::archive::binary_oarchive oa(ofs);
                oa << *this;
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                log<LOG_INFO>(L"%1% || Serialization save of PROpeller into file  %2% took %3% seconds") % __func__ % filename.c_str() % elapsed.count();
            }

            // Load from file
            void load(const std::string& filename) {
                auto start = std::chrono::high_resolution_clock::now();
                std::ifstream ifs(filename,std::ios::binary);
                boost::archive::binary_iarchive ia(ifs);
                ia >> *this;
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                log<LOG_INFO>(L"%1% || Serialization load of PROpeller from file  %2% took %3% seconds") % __func__ % filename.c_str() %elapsed.count();
            }

            // Scale detector weights for POT studies
            void scale(const PROconfig &inconfig, std::map<std::string, float> scaling){
                for (const auto& [detector, value] : scaling) {

                    if (value <= 0.0f) {
                        log<LOG_ERROR>(L"%1% || Scale factor %2% for '%3%' is invalid. Must be > 0.")
                           % __func__ % value % detector.c_str();
                        exit(EXIT_FAILURE);
                    }

                    log<LOG_INFO>(L"%1% || Wildcard '%2%' with scaling factor %3% matches:")
                        % __func__ % detector.c_str() % value;

                    std::vector<std::string> scalenames;
                    for (const auto& name : inconfig.m_fullnames) {
                        if (name.find(detector) != std::string::npos) {
                            scalenames.push_back(name);
                        }
                    }

                    log<LOG_INFO>(L"%1% || %2% . ") % __func__  % scalenames;

                    std::vector<int> scalerecobins;
                    for(auto &name: scalenames){
                        size_t is = inconfig.GetSubchannelIndex(name);     
                        size_t ic = inconfig.GetChannelIndex(is); 

                        size_t start = inconfig.GetGlobalBinStart(is); 
                        for(size_t b = 0; b < inconfig.m_channel_num_bins[ic] ; b++){
                            scalerecobins.push_back((int)(start+b));
                        }
                    }

                    std::vector<int> scaletruebins;
                    for(auto &name: scalenames){
                        size_t is = inconfig.GetSubchannelIndex(name);     
                        size_t ic = inconfig.GetChannelIndex(is); 
                        size_t start = inconfig.GetGlobalTrueBinStart(is); 
                        for(size_t b = 0; b < inconfig.m_channel_num_truebins[ic] ; b++){
                            scaletruebins.push_back((int)(start+b));
                        }
                    }

                    log<LOG_INFO>(L"%1% || and scales reco bins  %2% and true bins %3%.") % __func__  %  scalerecobins %  scaletruebins;

                    for (int r : scaletruebins) {
                        //histLE(r) *= value;
                        for (int c : scalerecobins) {
                            hist(r, c) *= value;
                        }
                    }

                    for (int c : scalerecobins) {
                        mcStatErr(c) *= value;
                    }

                    for (size_t i = 0; i < added_weights.size(); ++i) {
                        int bin = bin_indices[i];
                        if (std::find(scalerecobins.begin(), scalerecobins.end(), bin) != scalerecobins.end()) {
                            added_weights[i] *= value;
                        }
                    }


                    log<LOG_INFO>(L"%1% || Applied %2% scaling for '%3%'")
                        % __func__ % value % detector.c_str();
                
                }
    
            }

    };

}
#endif
