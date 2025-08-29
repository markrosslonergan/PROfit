#ifndef PRO_TO_CALL_H
#define PRO_TO_CALL_H

// C++ include 
#include <algorithm>
#include <unordered_map>
#include <string>
#include <iomanip>
#include <stdexcept>

// PROfit include 
#include "PROlog.h"
#include "PROconfig.h"

// Root includes
namespace PROfit{


    /* Function: given a value for reconstructed variable, figure out which local bin in the histogram it belongs to
     * Note: bin index start from 0, not 1
     * Note: return value of -1 means the reco value is out of range
     */
    int FindLocalBin(const PROconfig &inconfig, float reco_value, int channel_index);
    /* Function: given a value for reconstructed variable, figure out which global bin it belongs to
     * Note: bin index start from 0, not 1
     * Note: if  the reco value is out of range, then return value of -1
     */
    int FindGlobalBin(const PROconfig &inconfig, float reco_value, int subchannel_index);
    int FindGlobalBin(const PROconfig &inconfig, float reco_value, const std::string& subchannel_fullname);

    /* Function: given a value for true variable, figure out which local bin in the histogram it belongs to
     * Note: bin index start from 0, not 1
     * Note: return value of -1 means the true value is out of range
     */
    int FindLocalTrueBin(const PROconfig &inconfig, float true_value, int channel_index);

    /* Function: given a value for truenstructed variable, figure out which global bin it belongs to
     * Note: bin index start from 0, not 1
     * Note: if  the true value is out of range, then return value of -1
     */
    int FindGlobalTrueBin(const PROconfig &inconfig, float true_value, int subchannel_index);
    int FindGlobalTrueBin(const PROconfig &inconfig, float true_value, const std::string& subchannel_fullname);

    /* Function: given a value for true variable, figure out which local bin in the histogram it belongs to
     * Note: bin index start from 0, not 1
     * Note: return value of -1 means the true value is out of range
     */
    int FindLocalOtherBin(const PROconfig &inconfig, float other_value, int channel_index, int other_index);

    /* Function: given a value for truenstructed variable, figure out which global bin it belongs to
     * Note: bin index start from 0, not 1
     * Note: if  the true value is out of range, then return value of -1
     */
    int FindGlobalOtherBin(const PROconfig &inconfig, float other_value, int subchannel_index, int other_index);
    int FindGlobalOtherBin(const PROconfig &inconfig, float other_value, const std::string& subchannel_fullname, int other_index);


    /* Function: given a global bin index in the full vector, return the index of the subchannle this bin belongs to
     * Parameter:
     * 	 	inconfig:     a reference to PROconfig object, needed for calculating index 
     * 	 	global_bin:   global bin index. It can be a global true bin index, or global reco bin index
     * 	 	reco_bin:     boolean. If set to true, function will assume given bin is global reco bin, and return associated global subchannel index. Otherwise return subchnnale index for true bin.
     * 	 		      Default to true.
     */
    int FindSubchannelIndexFromGlobalBin(const PROconfig &inconfig, int global_bin, bool reco_bin = true);

    /* Function: given a full matrix, collapse the matrix */
    Eigen::MatrixXf CollapseMatrix(const PROconfig &inconfig, const Eigen::MatrixXf& full_matrix);

    /* Function: given a full vector (that contains reco), collapse the vector */
    Eigen::VectorXf CollapseMatrix(const PROconfig &inconfig, const Eigen::VectorXf& full_vector);

    /* Function: given a full matrix, collapse the matrix */
    Eigen::MatrixXf CollapseMatrix(const PROconfig &inconfig, const Eigen::MatrixXf& full_matrix, int other_index);

    /* Function: given a full vector (that contains reco), collapse the vector */
    Eigen::VectorXf CollapseMatrix(const PROconfig &inconfig, const Eigen::VectorXf& full_vector, int other_index);

    std::string getIcon();

    template <typename T>
    std::string to_string_prec(const T a_value, const int n = 6)    {
      std::ostringstream out;
      out <<std::fixed<< std::setprecision(n) << a_value;
      return out.str();
    }

    Eigen::MatrixXf ComputeSquareRootCovariance(const Eigen::MatrixXf& covariance, bool use_ldlt = true, float svd_tol = -1.0);

};

#endif
