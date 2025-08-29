#ifndef PROSYST_H_
#define PROSYST_H_

//C++ include 
#include <Eigen/Eigen>
#include <string>
#include <vector>
#include <map>
#include <cmath>

// Our include
#include "PROconfig.h"
#include "PROcreate.h"
#include "PROlog.h"

namespace PROfit {

    /*Struct: Class that groups all systematics (each with a SystStruct) and manages their formation and effect on PROspecs
    */
    class PROsyst {
        public:
            using Spline = std::vector<std::vector<std::pair<float, std::array<float, 4>>>>;
            
            enum class SystType {
                Spline, Covariance,  MFA
            };

            //Empty constructor
            PROsyst(){}

            /*Function: Primary constructor from a vector of SystStructs  */
            PROsyst(const PROpeller &prop, const PROconfig &config, const std::vector<SystStruct>& systs, bool shapeonly=false, int other_index = -1);

            PROsyst subset(const std::vector<std::string> &systs) const;
            PROsyst excluding(const std::vector<std::string> &systs) const;
            PROsyst allsplines2cov(const PROconfig &config, const PROpeller &prop, uint32_t seed) const;

            Eigen::MatrixXf spline2cov(int idx, const PROconfig &config, const PROpeller &prop, uint32_t seed) const;

            /* Function: given the systematic name, return corresponding fractional covariance matrix */
            Eigen::MatrixXf GrabMatrix(const std::string& sys) const;
            Eigen::MatrixXf GrabCorrMatrix(const std::string& sys) const;

            /* Function: given the systematic name, return corresponding Spline */
            Spline GrabSpline(const std::string& sys) const;

            /* Function: given systematic name, return type of systematic */
            SystType GetSystType(const std::string& syst) const;

            size_t GetNSplines() const { return splines.size(); }

            size_t GetNSplines() { return splines.size(); }

            size_t GetNCovar() const { return n_covar; }

            //----- Spline and Covariance matrix related ---
            //----- Spline and Covariance matrix related ---

            Eigen::MatrixXf SumMatrices() const;
            Eigen::MatrixXf SumMatrices(const std::vector<std::string>& sysnames) const;

            /* Function: given a SystStruct with cv and variation spectra, build full covariance matrix for the systematics, and return it
             * Note: it assumes the SystStruct is filled 
             */
            static Eigen::MatrixXf GenerateFullCovarMatrix(const SystStruct& sys_obj);

            /* Function: Given a SystStruct, generate fractinal covariance matrix, and correlation matrix, and add matrices to covmat_map and corrtmat_map
             * Note: this function is lazy. It wouldn't do anything if it found covariance matrix with the same name already in the map.
             */
            void CreateMatrix(const SystStruct& syst);

            /* Function: Given a SystStruct, generate a FLAT norm fractinal covariance matrix, and correlation matrix, and add matrices to covmat_map and corrtmat_map
             * Note: this function is lazy. It wouldn't do anything if it found covariance matrix with the same name already in the map.
             */
            void CreateFlatMatrix(const PROconfig& config, const SystStruct& syst);

            /* Function: given a syst struct with cv and variation spectra, build fractional covariance matrix for the systematics, as well as correlation matrix 
             * Return: {fractional covariance matrix, correlation covariance matrix}
             */
            static std::pair<Eigen::MatrixXf, Eigen::MatrixXf> GenerateCovarMatrices(const SystStruct& sys_obj);

            /* Function: given a SystStruct with cv and variation spectra, build fractional covariance matrix for the systematics, and return it
             * Note: it assumes the SystStruct is filled 
             */
            static Eigen::MatrixXf GenerateFracCovarMatrix(const SystStruct& sys_obj);

            /* Given fractional covariance matrix, calculate the correlation matrix */
            static Eigen::MatrixXf GenerateCorrMatrix(const Eigen::MatrixXf& frac_matrix);

            /* Function: check if matrix has nan, or infinite value */
            static bool isFiniteMatrix(const Eigen::MatrixXf& in_matrix);

            /* Function: if matrix has nan/inf values, change to 0. 
             * Note: this modifies the matrix !! 
             */
            static void toFiniteMatrix(Eigen::MatrixXf& in_matrix);

            /* Function: check if given matrix is positive semi-definite with tolerance. UST THIS ONE!!*/
            static bool isPositiveSemiDefinite_WithTolerance(const Eigen::MatrixXf& in_matrix, float tolerance=1.0e-16);

            /* Function: check if given matrix is positive semi-definite, no tolerance at all (besides precision error from Eigen) */
            static bool isPositiveSemiDefinite(const Eigen::MatrixXf& in_matrix);

            /* Function: Fill splines assuming p_cv and p_multi_spec have been filled in the SystStruct*/
            void FillSpline(const SystStruct& syst);

            /* Function: Get weight for bin for a given shift using spline */
            float GetSplineShift(int syst_num, float shift, int bin) const;
            float GetSplineShift(std::string name, float shift, int bin) const;

            /* Function: Get cv spectrum shifted using spline */
            PROspec GetSplineShiftedSpectrum(const PROconfig& config, const PROpeller& prop, std::string name, float shift) const;
            PROspec GetSplineShiftedSpectrum(const PROconfig& config, const PROpeller& prop, int syst_num, float shift) const;
            PROspec GetSplineShiftedSpectrum(const PROconfig& config, const PROpeller& prop, std::vector<std::string> names, std::vector<float> shifts) const;
            PROspec GetSplineShiftedSpectrum(const PROconfig& config, const PROpeller& prop, std::vector<int> syst_nums, std::vector<float> shifts) const;
            PROspec GetSplineShiftedSpectrum(const PROconfig& config, const PROpeller& prop, std::vector<float> shifts) const;

            Eigen::MatrixXf DecomposeFractionalCovariance(const PROconfig &config, const Eigen::VectorXf &cv_vec) const;


            /* the fractional covariance that is the sum of all during constructor*/
            Eigen::MatrixXf fractional_covariance;

            /* names of all systs*/
            std::vector<std::string> spline_names;
            std::vector<std::string> covar_names;
            std::vector<float> spline_lo, spline_hi;
            std::vector<int> spline_binnings;
            Eigen::VectorXf spline_priors;
            Eigen::VectorXf spline_centers;
        private:
            std::map<std::string, std::pair<size_t, SystType>> syst_map;
            std::vector<Spline> splines;
            [[maybe_unused]] size_t n_splines = 0;
            size_t n_covar = 0;
            std::vector<Eigen::MatrixXf> covmat;
            std::vector<Eigen::MatrixXf> corrmat;
            int other_index;
            static bool shape_only;
    };

};

#endif
