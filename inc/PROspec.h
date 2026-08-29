/**
 * @file PROspec.h
 * @brief Spectrum storage and manipulation for the PROfit neutrino oscillation framework.
 * @author PROfit Collaboration
 *
 * @details Defines the PROspec class, which is the primary container for binned event
 * spectra and their associated statistical uncertainties within the PROfit framework.
 * PROspec holds a flat Eigen::VectorXf of bin contents and a matching vector of bin
 * errors; all binning, channel layout, and collapsing logic lives in PROconfig.
 * Also defines the PROerrorbar helper struct for asymmetric error band information.
 */
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
#include "TH2D.h"
#include "THStack.h"
#include "TLegend.h"

// EIGEN
#include <Eigen/Dense>
#include <Eigen/SVD>

// PROfit
#include "PROconfig.h"
#include "PROserial.h"

namespace PROfit{

    /**
     * @brief Primary class for storing a binned spectrum and its statistical uncertainties.
     * @details A barebones container: all binning, collapsing, and channel layout are handled
     * by PROconfig.  Internally, PROspec holds two Eigen::VectorXf objects (bin contents and
     * bin errors) together with a small set of arithmetic and ROOT-conversion utilities.
     * @note All binning, collapsing, and channel definitions live in PROconfig; PROspec is
     * agnostic to the physics meaning of each bin.
     * @todo The toRoot/toTH1D conversions are not fully implemented. Evaluate whether they
     * are needed for future workflow.
     */

    /**
     * @brief Asymmetric error bar information for a spectrum, including a covariance matrix.
     * @details Used to represent pre- or post-fit uncertainty bands computed from systematic
     * throws or MCMC sampling.  All vectors have length equal to the number of collapsed bins.
     */
    struct PROerrorbar {
            Eigen::VectorXf error_down;   ///< Down (low) uncertainty per bin, measured from error_point + center_shift.
            Eigen::VectorXf error_up;     ///< Up (high) uncertainty per bin, measured from error_point + center_shift.
            Eigen::VectorXf error_point;  ///< Central value per bin (may be scaled).
            Eigen::VectorXf center_shift; ///< Offset of the band center from error_point (nonzero only for data-constrained bands, where the posterior prediction is pulled away from the best-fit spectrum).
            Eigen::MatrixXf covariance;   ///< Bin-to-bin covariance matrix.
            /**
             * @brief Construct a PROerrorbar with all vectors/matrices zeroed.
             * @param size Number of bins in the collapsed spectrum.
             */
            PROerrorbar(size_t size){
                error_down = Eigen::VectorXf::Zero(size);
                error_up = Eigen::VectorXf::Zero(size);
                error_point = Eigen::VectorXf::Zero(size);
                center_shift = Eigen::VectorXf::Zero(size);
                covariance = Eigen::MatrixXf::Zero(size, size);
            };
    };

    class PROspec {

        private:
            //Base
            size_t nbins;           ///< Total number of bins in the flat spectrum.
            Eigen::VectorXf spec;   ///< Flat vector of bin contents.
            Eigen::VectorXf error;  ///< Flat vector of per-bin statistical errors.
            

            //---- private helper function --------
            // Function: given two eigenvector of same dimension, calculate element-wise calculation of sqrt(a**2 + b**2) 
            Eigen::VectorXf eigenvector_sqrt_quadrature_sum(const Eigen::VectorXf& a, const Eigen::VectorXf& b) const;

            // Function: given two eigenvector of same dimension, calculate element-wise division a/b 
            Eigen::VectorXf eigenvector_division(const Eigen::VectorXf& a, const Eigen::VectorXf& b) const;

            // Function: given two eigenvector of same dimension, calculate element-wise multiplication a*b 
            Eigen::VectorXf eigenvector_multiplication(const Eigen::VectorXf& a, const Eigen::VectorXf& b) const;

        public:

            uint32_t hash = 0; ///< MurmurHash3 of the PROconfig used to create this spectrum; used for serialisation consistency checks. Zero-initialized: PROspecs serialized inside SystStructs (p_cv/p_multi_spec) never call save(), and an uninitialized hash writes nondeterministic bytes into _syst.bin.

            /** @brief Boost serialisation support — serialises nbins, spec, error, and hash. */
            template<class Archive>
            void serialize(Archive &ar, [[maybe_unused]] const unsigned int version) {
                ar & nbins;
                ar & spec;
                ar & error;
                ar & hash;

            }


            //Constructors
            /** @brief Default constructor — creates an empty (zero-bin) spectrum. */
            PROspec():nbins(0) {}
            /**
             * @brief Construct from pre-filled spectrum and error vectors.
             * @param in_spec  Bin-content vector.
             * @param in_error Per-bin error vector; must have the same length as @p in_spec.
             */
            PROspec(const Eigen::VectorXf &in_spec, const Eigen::VectorXf &in_error) : nbins(in_spec.size()), spec(in_spec), error(in_error){}

            /**
             * @brief Construct a zero-initialised spectrum with a given number of bins.
             * @param num_bins Number of bins.
             */
            PROspec(size_t num_bins);

            //PROspec(PROconfig const & configin); //Load in config file EMPTY hists
            //PROspec(std::string &xmlname); //Load directly from XML 

            /**
             * @brief Return a new PROspec whose bin contents have been Poisson-fluctuated.
             * @param s    Input spectrum to fluctuate.
             * @param seed Random seed for reproducibility.
             * @return A new PROspec with each bin drawn from a Poisson distribution
             *         whose mean is the corresponding bin content of @p s.
             */
            static PROspec PoissonVariation(const PROspec &s, uint32_t seed);


            /**
             * @brief Convert the spectrum for a given subchannel to a ROOT TH1D.
             * @param inconfig          The PROconfig describing binning and channel layout.
             * @param subchannel_index  Global 0-based subchannel index.
             * @param other_index       Variable index for multi-variable analyses (default 0).
             * @param dim               Projection dimension for multi-dimensional binnings (default 0).
             * @return A TH1D representing this subchannel's spectrum.
             */
            TH1D toTH1D(const PROconfig& inconfig, int subchannel_index, int other_index = 0, int dim = 0) const;
            /** @brief Convert subchannel spectrum to a TH1D projected along individual slices. */
            TH1D toTH1DSlices(const PROconfig& inconfig, int subchannel_index, int other_index = 0, int dim = 0) const;
            /** @brief Convert subchannel spectrum to a 2D ROOT TH2D. */
            TH2D toTH2D(const PROconfig& inconfig, int subchannel_index, int other_index = 0, int dim = 0) const;
            /**
             * @brief Convert the spectrum for a named subchannel to a ROOT TH1D.
             * @param inconfig             The PROconfig describing binning and channel layout.
             * @param subchannel_fullname  Full name string identifying the subchannel.
             * @param other_index          Variable index (default 0).
             * @param dim                  Projection dimension (default 0).
             * @return A TH1D for the named subchannel.
             */
            TH1D toTH1D(const PROconfig& inconfig, const std::string& subchannel_fullname, int other_index = 0, int dim = 0) const;
            /** @brief Convert named subchannel spectrum to a TH1D projected along slices. */
            TH1D toTH1DSlices(const PROconfig& inconfig, const std::string& subchannel_fullname, int other_index = 0, int dim = 0) const;
            /** @brief Convert named subchannel spectrum to a ROOT TH2D. */
            TH2D toTH2D(const PROconfig& inconfig, const std::string& subchannel_fullname, int other_index = 0, int dim = 0) const;

            /**
             * @brief Convert a collapsed (summed-over-subchannels) channel spectrum to a ROOT TH1D.
             * @param inconfig      The PROconfig describing binning and channel layout.
             * @param channel_index Global channel index.
             * @param var_index     Variable index (default 0).
             * @param dim           Projection dimension (default 0).
             * @return A TH1D of the collapsed channel spectrum.
             */
            TH1D toTH1D_Collapsed(const PROconfig& inconfig, int channel_index, size_t var_index=0, int dim = 0) const;


            /**
             * @brief Write TH1D histograms for all subchannels into a ROOT file.
             * @param inconfig     The PROconfig describing the channel/subchannel layout.
             * @param output_name  Path of the output ROOT file.
             */
            void toROOT(const PROconfig& inconfig, const std::string& output_name);

            /**
             * @brief Produce a PDF with stacked subchannel plots for every channel.
             * @param inconfig     The PROconfig describing the channel/subchannel layout.
             * @param output_name  Base name for the output PDF file.
             */
            void plotSpectrum(const PROconfig& inconfig, const std::string& output_name) const;

            /**
             * @brief Fill the given bin, updating both the bin content and its error.
             * @param bin_index  0-based global bin index.  No range check is performed.
             * @param weight     Event weight to accumulate into the bin.
             * @note Neither Fill() nor QuickFill() perform bounds checking.
             */
            void Fill(int bin_index, float weight);
            /**
             * @brief Fill the given bin, updating only the bin content (not the error).
             * @param bin_index  0-based global bin index.  No range check is performed.
             * @param weight     Event weight to accumulate into the bin.
             * @warning The error vector is not updated; use Fill() when errors matter.
             */
            void QuickFill(int bin_index, float weight);

            /**
             * @brief Zero out all bin contents and errors while preserving the number of bins.
             */
            void Zero();

            /**
             * @brief Print the spectrum contents to the log.
             */
            void Print() const;

            /**
             * @brief Return the total number of bins in the flat spectrum.
             * @return Number of bins.
             */
            size_t GetNbins() const;



            /**
             * @brief Return the content of a single bin.
             * @param bin  0-based bin index.
             * @return Bin content as a float.
             */
            inline
                float GetBinContent(int bin) const{
                    return spec(bin);
                }

            /**
             * @brief Return the statistical error for a single bin.
             * @param bin  0-based bin index.
             * @return Bin error as a float.
             */
            inline
                float GetBinError(int bin) const{
                    return error(bin);
                }

            /**
             * @brief Return a const reference to the underlying spectrum vector.
             * @return Const reference to the Eigen::VectorXf of bin contents.
             */
            inline
                const Eigen::VectorXf& Spec() const{
                    return spec;
                }

            /**
             * @brief Return a (non-const) reference to the underlying spectrum vector.
             * @return (Non-const) Reference to the Eigen::VectorXf of bin contents.
             */
            inline Eigen::VectorXf& Spec() {
                return spec;
            }

            /**
             * @brief Return a const reference to the underlying error vector.
             * @return Const reference to the Eigen::VectorXf of per-bin errors.
             */
            inline
                const Eigen::VectorXf& Error() const{
                    return error;
                }

            /**
             * @brief Return a (non-const) reference to the underlying error vector.
             * @return (Non-const) Reference to the Eigen::VectorXf of per-bin errors.
             */
            inline Eigen::VectorXf& Error() {
                return error;
            }

            /**
             * @brief Check whether two PROspec objects have the same number of bins.
             * @param a  First spectrum.
             * @param b  Second spectrum.
             * @return True if both spectra have the same bin count.
             */
            static bool SameDim(const PROspec& a, const PROspec& b);

            /**
             * @brief Serialise this spectrum to a binary file using Boost.Serialization.
             * @param config    The PROconfig whose hash is stored alongside the data.
             * @param filename  Output file path.
             */
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

            /**
             * @brief Deserialise this spectrum from a binary file using Boost.Serialization.
             * @param filename  Input file path.
             */
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
            /** @brief Element-wise addition of two spectra (errors added in quadrature). */
            PROspec operator+(const PROspec& b) const;
            /** @brief In-place element-wise addition (errors added in quadrature). */
            PROspec& operator+=(const PROspec& b);
            /** @brief Element-wise subtraction of two spectra (errors added in quadrature). */
            PROspec operator-(const PROspec& b) const;
            /** @brief In-place element-wise subtraction (errors added in quadrature). */
            PROspec& operator-=(const PROspec& b);
            /** @brief Element-wise division of two spectra. */
            PROspec operator/(const PROspec& b) const;
            /** @brief In-place element-wise division. */
            PROspec& operator/=(const PROspec& b);
            /**
             * @brief Scale the spectrum by a constant factor.
             * @param scale  Multiplicative scale factor.
             */
            PROspec operator*(float scale) const;
            /** @brief In-place scalar multiplication. */
            PROspec& operator*=(float scale);
    };

}


#endif
