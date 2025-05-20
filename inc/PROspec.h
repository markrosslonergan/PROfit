#ifndef PROSPEC_H_
#define PROSPEC_H_

// STANDARD
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

// ROOT
#include "TFile.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "THStack.h"
#include "TLegend.h"

// EIGEN
#include <Eigen/Dense>
#include <Eigen/SVD>

// PROfit
#include "PROconfig.h"
#include "PROserial.h"

namespace PROfit{

    /* Class: Primary class for storing final spectra and associated errors.
     * Note: 
     *  A barebones class, as all binning, collapsing..etc.. is handeled by PROconfig. 
     *  Essentially two Eigen::VectorXf and associated functionality
     * Todo: 
     *  The toRoot/toTH1Dnot implemented. Do we need them?
     * */

    class PROspec {

        private:
            //Base
            size_t nbins;
            Eigen::VectorXf spec;
            Eigen::VectorXf error;
            

            //---- private helper function --------
            // Function: given two eigenvector of same dimension, calculate element-wise calculation of sqrt(a**2 + b**2) 
            Eigen::VectorXf eigenvector_sqrt_quadrature_sum(const Eigen::VectorXf& a, const Eigen::VectorXf& b) const;

            // Function: given two eigenvector of same dimension, calculate element-wise division a/b 
            Eigen::VectorXf eigenvector_division(const Eigen::VectorXf& a, const Eigen::VectorXf& b) const;

            // Function: given two eigenvector of same dimension, calculate element-wise multiplication a*b 
            Eigen::VectorXf eigenvector_multiplication(const Eigen::VectorXf& a, const Eigen::VectorXf& b) const;

        public:

            uint32_t hash;
            template<class Archive>
            void serialize(Archive &ar, [[maybe_unused]] const unsigned int version) {
                ar & nbins;
                ar & spec;
                ar & error;
                ar & hash;

            }


            //Constructors
            PROspec():nbins(0) {}
            PROspec(const Eigen::VectorXf &in_spec, const Eigen::VectorXf &in_error) : nbins(in_spec.size()), spec(in_spec), error(in_error){}

            /* Function: create PROspec of given size */
            PROspec(size_t num_bins);

            //PROspec(PROconfig const & configin); //Load in config file EMPTY hists
            //PROspec(std::string &xmlname); //Load directly from XML 

            /* Function: Create a new PROspec with Poisson fluctuations applied. */
            static PROspec PoissonVariation(const PROspec &s, uint32_t seed);


            /* Function: given subchannel name/index, generate TH1D histogram of corresponding spectrum */
            TH1D toTH1D(const PROconfig& inconfig, int subchannel_index, int other_index = -1) const;
            TH1D toTH1D(const PROconfig& inconfig, const std::string& subchannel_fullname, int other_index = -1) const;

            TH1D toTH1D_Collapsed(const PROconfig& inconfig, int channel_index) const;


            /* Function: save TH1Ds of all subchannels into a root file */
            void toROOT(const PROconfig& inconfig, const std::string& output_name);

            /*Function: Create a pdf of all channels, with stacked subchannels*/
            void plotSpectrum(const PROconfig& inconfig, const std::string& output_name) const;

            /* Function: fill given bin with provided weight 
             * Note: 
             * 	 Both function do NOT check whether the given bin is out of range or not 
             * 	 Fill() updates the bin content and error, while QuickFill() only updates bin content and doesn't care error.
             * 	 Care when using QuickFill!! 
             *
             */
            void Fill(int bin_index, float weight);
            void QuickFill(int bin_index, float weight);

            /* Function: zero out the spectrum and error, but keep the dimension */
            void Zero();

            /* Function: Print out spec*/
            void Print() const;

            /* Function: Print out spec*/
            void PrintAll(const PROconfig & config) const;


            /*Return number of bins in spectrum */
            size_t GetNbins() const;



            /* Function:  Return the content of spectrum at given bin
             * Note: bin index starts at 0
             */
            inline
                float GetBinContent(int bin) const{
                    return spec(bin);
                }

            /* Function:  Return the bin error  at given bin
             * Note: bin index starts at 0
             */
            inline
                float GetBinError(int bin) const{
                    return error(bin);
                }

            /*Return reference to the core specturm */ 
            inline
                const Eigen::VectorXf& Spec() const{
                    return spec;
                }

            /*Return reference to the error specturm */ 
            inline
                const Eigen::VectorXf& Error() const{
                    return error;
                }


            /* Return true if two PROspec have the same dimension (number of bins */
            static bool SameDim(const PROspec& a, const PROspec& b);

            /* Save to binary file*/
            inline
            void save(const PROconfig& config, const std::string& filename) {
                hash = config.hash;
                auto start = std::chrono::high_resolution_clock::now();
                std::ofstream ofs(filename, std::ios::binary);
                boost::archive::binary_oarchive oa(ofs);
                oa << *this;
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                log<LOG_INFO>(L"%1% || Serialization save of PROspec data into file  %2% took %3% seconds") % __func__ % filename.c_str() % elapsed.count();
            }

            // Load from file
            inline
            void load(const std::string& filename) {
                auto start = std::chrono::high_resolution_clock::now();
                std::ifstream ifs(filename,std::ios::binary);
                boost::archive::binary_iarchive ia(ifs);
                ia >> *this;
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                log<LOG_INFO>(L"%1% || Serialization load of PRospec from file  %2% took %3% seconds") % __func__ % filename.c_str() %elapsed.count();
            }




            //----- Arithmetic Operations ---------
            //addition 
            PROspec operator+(const PROspec& b) const;
            //addition assignment
            PROspec& operator+=(const PROspec& b);
            //subtraction
            PROspec operator-(const PROspec& b) const;
            //subtraction assignment
            PROspec& operator-=(const PROspec& b);
            //division
            PROspec operator/(const PROspec& b) const;
            //division assignment
            PROspec& operator/=(const PROspec& b);
            //scaling (multiply with constant)
            PROspec operator*(float scale) const;
            //scaling assignmnet 
            PROspec& operator*=(float scale);
    };

}


#endif
