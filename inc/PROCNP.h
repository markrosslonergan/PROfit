#ifndef PROCNP_H_
#define PROCNP_H_

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
#include "PROcess.h"
#include "PROmetric.h"

namespace PROfit{

    class PROCNP : public PROmetric
    {
        // TODO: How much of this should be in PROmetric instead?
        private:
            std::string model_tag;

            const PROconfig config;
            const PROpeller peller;
            const PROsyst *syst; 
            const PROmodel model;
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

        public:

            /*Function: Constructor bringing all objects together*/
            PROCNP(const std::string tag, const PROconfig &conin, const PROpeller &pin, const PROsyst *systin, const PROmodel &modelin, const PROdata &datain, EvalStrategy strat = EventByEvent, bool shape_only = false, std::vector<float> physics_param_fixed = std::vector<float>());

            /*Function: operator() is what is passed to minimizer.*/
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient);
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool nograd);

            PROmetric *Clone() const {
                return new PROCNP(*this);
            }

            virtual const PROmodel &GetModel() const {
                return model;
            }

            virtual const PROsyst &GetSysts() const {
                return *syst;
            }

            void reset() {
                physics_param_fixed.clear();
                last_value = 0;
                last_param = Eigen::VectorXf::Constant(last_param.size(), 0);
            }

            void override_systs(const PROsyst &new_syst) {
                syst = &new_syst;
            }

            virtual float Pull(const Eigen::VectorXf &systs);

            float getSingleChannelChi(size_t channel_index);

            void fixSpline(int fix, float valin);
    };


}
#endif
