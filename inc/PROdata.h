/**
 * @file PROdata.h
 * @brief Observed data spectrum container for the PROfit fitting framework.
 * @author PROfit Collaboration
 *
 * @details Defines PROdata, which mirrors PROspec in structure but represents the measured
 * (observed) event spectrum used as the "data" in chi-squared calculations.  PROdata is
 * always stored in the collapsed (channel-level) bin space, unlike PROspec which may be
 * in the full subchannel space.  Arithmetic operations, serialisation, and ROOT histogram
 * conversion are provided for convenience.
 */
#ifndef PRODATA_H_
#define PRODATA_H_

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
#include "PROtocall.h"
#include "PROspec.h"

namespace PROfit{

/**
 * @brief Observed data spectrum in the collapsed (channel-level) bin space.
 * @details PROdata is structurally identical to PROspec but semantically represents measured
 * data rather than a Monte Carlo prediction.  It is always kept in the collapsed bin space
 * and constructed either directly from vectors or by collapsing a PROspec using PROconfig.
 * Supports the same arithmetic operations, serialisation, and ROOT histogram conversion as PROspec.
 */
class PROdata {
private:
    size_t nbins;           ///< Number of collapsed bins.
    Eigen::VectorXf spec;   ///< Observed event counts per bin.
    Eigen::VectorXf error;  ///< Statistical uncertainty per bin (sqrt of observed counts for Poisson data).


    //---- private helper function --------
    // Function: given two eigenvector of same dimension, calculate element-wise calculation of sqrt(a**2 + b**2) 
    Eigen::VectorXf eigenvector_sqrt_quadrature_sum(const Eigen::VectorXf& a, const Eigen::VectorXf& b) const;

    // Function: given two eigenvector of same dimension, calculate element-wise division a/b 
    Eigen::VectorXf eigenvector_division(const Eigen::VectorXf& a, const Eigen::VectorXf& b) const;

    // Function: given two eigenvector of same dimension, calculate element-wise multiplication a*b 
    Eigen::VectorXf eigenvector_multiplication(const Eigen::VectorXf& a, const Eigen::VectorXf& b) const;

public:
    uint32_t hash = 0; ///< MurmurHash3 of the PROconfig used to create this data; checked during serialisation. Zero-initialized so copies made before save() never carry (or serialize) indeterminate bytes.
    /// Observed centres of the <model> physics priors (one per PROmodel::physics_priors
    /// entry), i.e. the external measurement as seen by this dataset. Empty = the XML centres
    /// (real data, Asimov); pseudo-experiments set ToyPhysicsPriorCenters. Metrics built from
    /// this data read it. Not serialized.
    std::vector<float> physics_prior_centers;

    /** @brief Boost serialisation support — serialises nbins, spec, error, and hash. */
    template<class Archive>
        void serialize(Archive &ar, [[maybe_unused]] const unsigned int version) {
            ar & nbins;
            ar & spec;
            ar & error;
            ar & hash;

        }

    //Constructors
    /** @brief Default constructor — creates an empty (zero-bin) data object. */
    PROdata():nbins(0) {}
    /**
     * @brief Construct from pre-filled spectrum and error vectors.
     * @param in_spec   Observed counts per (collapsed) bin.
     * @param in_error  Per-bin statistical errors.
     */
    PROdata(const Eigen::VectorXf &in_spec, const Eigen::VectorXf &in_error) : nbins(in_spec.size()), spec(in_spec), error(in_error){}
    /**
     * @brief Construct a collapsed PROdata from a full-space PROspec.
     * @details Applies the collapsing matrix from @p c to collapse @p s from subchannel to channel space.
     * Errors are propagated correctly under the linear collapsing transformation.
     * @param c  The PROconfig providing the collapsing matrix.
     * @param s  The full-space PROspec to collapse.
     */
    PROdata(const PROconfig &c, const PROspec &s):
      nbins(s.Spec().size()),
      spec(CollapseMatrix(c, s.Spec())),
      error(CollapseMatrix(c, Eigen::VectorXf(s.Error().array().square().matrix())).array().sqrt())
   {} 

    /* Function: create PROspec of given size */
    PROdata(size_t num_bins);

    // other_index defaults to 0 (the first variable), matching PROspec::toTH1D.
    // The old default of -1 indexed m_channel_variable_bins[...][-1] — UB for
    // every caller that used the default (toROOT, plotSpectrum).
    TH1D toTH1D(const PROconfig& inconfig, int channel_index, int other_index = 0, int dim = 0) const;

    void toROOT(const PROconfig& inconfig, const std::string& output_name);


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

    /*Return number of bins in spectrum */
    size_t GetNbins() const;

    /* Function:  Return the content of spectrum at given bin
     * Note: bin index starts at 0
     */
    inline float GetBinContent(int bin) const{
        return spec(bin);
    }

    /* Function:  Return the bin error  at given bin
     * Note: bin index starts at 0
     */
    inline float GetBinError(int bin) const{
        return error(bin);
    }

    /*Return reference to the core specturm */ 
    inline const Eigen::VectorXf& Spec() const{
        return spec;
    }

    /*Return reference to the error specturm */ 
    inline const Eigen::VectorXf& Error() const{
        return error;
    }

    /* Save to binary file*/
    inline void save(const PROconfig& config, const std::string& filename) {
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
    inline void load(const std::string& filename) {
        auto start = std::chrono::high_resolution_clock::now();
        std::ifstream ifs(filename,std::ios::binary);
        boost::archive::binary_iarchive ia(ifs);
        ia >> *this;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        log<LOG_INFO>(L"%1% || Serialization load of PRospec from file  %2% took %3% seconds") % __func__ % filename.c_str() %elapsed.count();
    }

    /**
     * @brief Serialise a vector of PROdata objects to a single binary file.
     * @param config    The PROconfig whose hash is stored in each element.
     * @param data      The vector of PROdata objects to serialise.
     * @param filename  Output file path.
     */
    static void saveVector(const PROconfig &config, std::vector<PROdata> &data, std::string &filename) {
        for(auto &d: data) 
            d.hash = config.hash;
        auto start = std::chrono::high_resolution_clock::now();
        std::ofstream ofs(filename, std::ios::binary);
        boost::archive::binary_oarchive oa(ofs);
        oa << data;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        log<LOG_INFO>(L"%1% || Serialization save of PROspec data into file  %2% took %3% seconds") % __func__ % filename.c_str() % elapsed.count();
    }

    /**
     * @brief Deserialise a vector of PROdata objects from a binary file.
     * @param data      The vector to fill.
     * @param filename  Input file path.
     */
    static void loadVector(std::vector<PROdata> &data, std::string &filename) {
        auto start = std::chrono::high_resolution_clock::now();
        std::ifstream ifs(filename,std::ios::binary);
        boost::archive::binary_iarchive ia(ifs);
        ia >> data;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        log<LOG_INFO>(L"%1% || Serialization load of PRospec from file  %2% took %3% seconds") % __func__ % filename.c_str() %elapsed.count();
    }

    /**
     * @brief Check whether two PROdata objects have the same number of bins.
     * @param a  First data object.
     * @param b  Second data object.
     * @return True if both have the same bin count.
     */
    static bool SameDim(const PROdata& a, const PROdata& b);

    //----- Arithmetic Operations ---------
    /** @brief Element-wise addition of two data spectra (errors added in quadrature). */
    PROdata operator+(const PROdata& b) const;
    /** @brief In-place element-wise addition (errors added in quadrature). */
    PROdata& operator+=(const PROdata& b);
    /** @brief Element-wise subtraction (errors added in quadrature). */
    PROdata operator-(const PROdata& b) const;
    /** @brief In-place element-wise subtraction (errors added in quadrature). */
    PROdata& operator-=(const PROdata& b);
    /** @brief Element-wise division. */
    PROdata operator/(const PROdata& b) const;
    /** @brief In-place element-wise division. */
    PROdata& operator/=(const PROdata& b);
    /**
     * @brief Scale the data by a constant factor.
     * @param scale  Multiplicative scale factor.
     */
    PROdata operator*(float scale) const;
    /** @brief In-place scalar multiplication. */
    PROdata& operator*=(float scale);
};

}


#endif
