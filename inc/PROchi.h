#ifndef PROCHI_H_
#define PROCHI_H_

// STANDARD
#include <string>
#include <vector>

#include <Eigen/Eigen>

// OUR INCLUDES
#include "PROconfig.h"
#include "PROdata.h"
#include "PROsyst.h"
#include "PROpeller.h"
#include "PROmodel.h"
#include "PROmetric.h"

namespace PROfit{

    /* 
     * Class: Class that gathers the MC (PROpeller), Systematics (PROsyst) and model (PROsc) and forms a function calculating a chi^2 that can be minimized over
     * Note:
     *  the PROconfig,PROpeller..etc need to be accessable by the class so that the function operato "()" when passed to minimizer can access them. 
     *  Saved as pointers to the objects created in the primary executable.
     * Todo:
     *  Add capability to define function externally?
     *  Improve gradient calculation
     *  */

    class PROchi : public PROmetric
    {
        private:

        public:
            // TODO: How much of this should be in PROmetric instead?

            std::string model_tag;

            const PROconfig &config;
            const PROpeller &peller;
            const PROsyst *syst; 
            const PROmodel &model;
            const PROdata data;
            EvalStrategy strat;
            bool shape_only;
            std::vector<float> physics_param_fixed;
                        //Do we want to fix any param?
            int fixed_index;
            float fixed_val;

            //Save last values for gradient calculation
            Eigen::VectorXf last_param;
            float last_value;

            bool correlated_systematics;
            Eigen::MatrixXf prior_covariance;
            Eigen::MatrixXf collapsed_stat_covariance;


            /*Function: Constructor bringing all objects together*/
            PROchi(const std::string tag, const PROconfig &conin, const PROpeller &pin, const PROsyst *systin, const PROmodel &modelin, const PROdata &datain, EvalStrategy strat = EventByEvent, bool shape_only = false, std::vector<float> physics_param_fixed = std::vector<float>());


            /*Function: operator() is what is passed to minimizer.*/
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient);
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool nograd);

            virtual void reset() {
                physics_param_fixed.clear();
                last_value = 0;
                last_param = Eigen::VectorXf::Constant(last_param.size(), 0);
            }

            virtual PROmetric *Clone() const {
                return new PROchi(*this);
            }

            virtual const PROmodel &GetModel() const {
                return model;
            }

            virtual const PROsyst &GetSysts() const {
                return *syst;
            }

            virtual void override_systs(const PROsyst &new_syst) {
                syst = &new_syst;
            }
            
            virtual float Pull(const Eigen::VectorXf &systs);

            void fixSpline(int fix, float valin);

            float getSingleChannelChi(size_t channel_index) ;
                
    };
}
#endif
