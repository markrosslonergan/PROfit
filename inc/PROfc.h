#ifndef PRO_FC_H
#define PRO_FC_H

#include "PROfitter.h"
#include "PROconfig.h"
#include "PROsyst.h"
#include "PROseed.h"
#include "PROmetric.h"
#include "PROcess.h"
#include "PROtocall.h"
#include "PROchi.h"
#include "PROCNP.h"
#include "PROpoisson.h"
#include "PROgress.h"

#include <Eigen/Eigen>


namespace PROfit {

    struct fc_out{
        float chi2_syst, chi2_osc, dmsq, sinsq2tmm;
        Eigen::VectorXf best_fit_syst, best_fit_osc, syst_throw;
    };

    struct fc_args {
        const size_t todo;
        std::vector<float>* dchi2s;
        std::vector<fc_out>* out;
        const PROconfig config;
        const PROpeller prop;
        const PROsyst systs;
        std::string chi2;
        const Eigen::VectorXf phy_params;
        const Eigen::MatrixXf L;
        PROfitterConfig fitconfig;
        uint32_t seed;
        const int thread;
        const bool binned;
        const bool gof_mode;
    };

    void fc_worker(fc_args arg, MultiPROgressBar& progressbar);



}
#endif

