#ifndef PROSURF_H
#define PROSURF_H

#include "PROfitter.h"
#include "PROconfig.h"
#include "PROsyst.h"
#include "PROseed.h"
#include "PROmetric.h"

#include <Eigen/Eigen>

#include "TGraphAsymmErrors.h"

namespace PROfit {

struct surfOut{
    std::vector<int> grid_index;
    std::vector<float> grid_val;
    Eigen::VectorXf best_fit;
    float chi;
};

struct profOut{
    std::vector<float> knob_vals;
    std::vector<float> knob_chis;
    float chi;
    
    void sort(){

        std::vector<std::pair<float, float>> combined;
        for (size_t i = 0; i < knob_vals.size(); ++i) {
                combined.emplace_back(knob_vals[i], knob_chis[i]);
        }

        std::sort(combined.begin(), combined.end());
         for (size_t i = 0; i < combined.size(); ++i) {
             knob_vals[i] = combined[i].first;
             knob_chis[i] = combined[i].second;
        }
        return;
    }
};

class PROfile {

        public:
	PROmetric &metric;
    TGraphAsymmErrors onesig;
    std::vector<std::unique_ptr<TGraph>> graphs; 
    // Hack: Save these into class for now
    std::vector<float> bfvalues;
    std::vector<float> barvalues;
    std::vector<float> values1_up;
    std::vector<float> values1_down;

  PROfile(const PROconfig &config, const PROsyst &systs, const PROmodel &model, PROmetric &metric, PROseed &proseed, const PROfitterConfig &fitconfig, std::string filename, float minchi = 0, bool with_osc = false, int nThreads = 1, const Eigen::VectorXf& init_seed = Eigen::VectorXf(), const Eigen::VectorXf& true_params = Eigen::VectorXf() ) ;

  void Plot(const PROconfig &config, const PROsyst &systs, const PROmodel &model, PROmetric &metric, PROseed &proseed, std::string filename, bool with_osc = false, const Eigen::VectorXf& init_seed = Eigen::VectorXf(), const Eigen::VectorXf& true_params = Eigen::VectorXf(), bool mask_osc = false) ;

    	std::vector<profOut> PROfilePointHelper(const PROsyst *systs, const PROfitterConfig &fitconfig, int offset, int stride, float minchi, bool with_osc, const Eigen::VectorXf& init_seed = Eigen::VectorXf(), uint32_t seed=0);
};

class PROsurf {
public:
    PROmetric &metric;
    size_t x_idx, y_idx, nbinsx, nbinsy;
    Eigen::VectorXf edges_x, edges_y;
    Eigen::MatrixXf surface;
    
    struct SurfPointResult {
        int binx, biny;
        Eigen::VectorXf best_fit;
        float chi2;
    };

    std::vector<SurfPointResult> results;

    enum LogLin {
        LinAxis,
        LogAxis,
    };

    PROsurf(PROmetric &metric,  size_t x_idx, size_t y_idx, size_t nbinsx, const Eigen::VectorXf &edges_x, size_t nbinsy, const Eigen::VectorXf &edges_y) : metric(metric), x_idx(x_idx), y_idx(y_idx), nbinsx(nbinsx), nbinsy(nbinsy), edges_x(edges_x), edges_y(edges_y), surface(nbinsx, nbinsy) { }

    PROsurf(PROmetric &metric, size_t x_idx, size_t y_idx, size_t nbinsx, LogLin llx, float x_lo, float x_hi, size_t nbinsy, LogLin lly, float y_lo, float y_hi);

    std::vector<surfOut> PointHelper(const PROfitterConfig &fitconfig, std::vector<surfOut> multi_physics_params, int start, int end, uint32_t seed);

    void FillSurfaceStat(const PROconfig &config, const PROfitterConfig &fitconfig, std::string filename);
    void FillSurface(const PROfitterConfig &fitconfig, std::string filename, PROseed & proseed, int nthreads = 1);

};

}

#endif

