#ifndef PROUNBLIND_H
#define PROUNBLIND_H

// PROfit include 
#include "PROlog.h"
#include "PROconfig.h"
#include "PROspec.h"
#include "PROsyst.h"
#include "PROMCMC.h"
#include "PROtocall.h"
#include "PROseed.h"
#include "PROcess.h"
#include "PROversion.h"
#include "PROmetric.h"
#include "PROfitter.h"

#include <Eigen/Eigen>
#include "LBFGSB.h"
#include <thread>
#include <chrono>
namespace PROfit {


  /* Function: Unblinding Proceedure v1 ICARUS numu dis
     * Note: Proceed in stages. Stage1 is a Global Fit. 
     */
    int PROunblind_Stage1( PROmetric *metric );

}
#endif
