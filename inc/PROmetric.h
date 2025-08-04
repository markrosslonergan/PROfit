#ifndef PROMETRIC_H
#define PROMETRIC_H

#include "PROsyst.h"
#include "PROmodel.h"

#include <Eigen/Eigen>

namespace PROfit {

class PROmetric {
public:
    enum EvalStrategy {
        EventByEvent,
        BinnedGrad,
        BinnedChi2
    };

    virtual void override_systs(const PROsyst &new_syst) = 0;
    virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient) = 0;
    virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool nograd) = 0;
    virtual void reset() = 0;
    virtual PROmetric *Clone() const = 0;
    virtual const PROmodel &GetModel() const = 0;
    virtual const PROsyst  &GetSysts() const = 0;
    virtual float getSingleChannelChi(size_t channel_index) = 0;
    virtual ~PROmetric() {}
    virtual void fixSpline(int,float)  = 0;
    virtual float Pull(const Eigen::VectorXf &systs) = 0;
    virtual int checkData()  = 0;

    size_t getCallCount() const { return call_count; }
    void resetCallCount() { call_count = 0; }


protected:
    mutable std::atomic<size_t> call_count{0};
};

};

#endif

