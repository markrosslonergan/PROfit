#ifndef PROMODEL_H
#define PROMODEL_H

#include "PROconfig.h"
#include "PROpeller.h"

#include <Eigen/Eigen>

#include <Eigen/src/Core/Matrix.h>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace PROfit {

class PROmodel {
public:
    size_t nparams;
    int ivar; //TODO, this should be a string like "true" and then in config we have a map to that variable. error if not found. 
    std::vector<std::string> param_names;
    std::vector<std::string> pretty_param_names;
    std::vector<std::string> pretty_param_units;
    Eigen::VectorXf lb, ub, default_val;
    std::vector<std::function<float(const Eigen::VectorXf&, float)>> model_functions;
    std::function<int(const Eigen::VectorXf&)> model_constraint;
    std::vector<std::vector<Eigen::MatrixXf>> hists; //2D hists for binned oscilattion, one for each model function, and the N-variables 
                                        //Todo: make this a vector of length n_variables, and fill them all. For now 1 is "special". 
};

class NullModel : public PROmodel {
public:
    NullModel(const PROpeller &prop) {
        nparams = 0;
        ivar = 1;
        model_functions.push_back([](const Eigen::VectorXf &, float){ return 1.0f; });
       
        size_t nvar = prop.variable_mc_stat_err.size();
        hists.resize(nvar);
        for(size_t v = 0; v <nvar ;v++){
            for(size_t m = 0; m < model_functions.size(); ++m) {
                hists.at(v).emplace_back(Eigen::MatrixXf::Constant(prop.variable_hist_storage(ivar,v).rows(), prop.variable_hist_storage(ivar,v).cols(),0.0));
                Eigen::MatrixXf &h = hists.at(v).back();
                for(size_t i = 0; i < prop.NEvent(); ++i) {
                    int tbin = prop.VariableBinIndex(ivar, i), rbin = prop.VariableBinIndex(v, i);
                    if(tbin<0 || rbin<0)continue;
                    h(tbin, rbin) += prop.added_weights[i];
                }
            }
        }
    }
};

class PROnumudis : public PROmodel {
public:
    PROnumudis(const PROpeller &prop,const std::map<std::string,int> &parameter_map) {
        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0;});
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),le);});
        

        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'.Make sure its in your model section of XML. ") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivar = parameter_map.at("L/E");

        size_t nvar = prop.variable_mc_stat_err.size();
        hists.resize(nvar);
        for(size_t v = 0; v <nvar ;v++){
            for(size_t m = 0; m < model_functions.size(); ++m) {
                hists.at(v).emplace_back(Eigen::MatrixXf::Constant(prop.variable_hist_storage(ivar,v).rows(), prop.variable_hist_storage(ivar,v).cols(),0.0));
                Eigen::MatrixXf &h = hists.at(v).back();
                for(size_t i = 0; i < prop.NEvent(); ++i) {
                    if(prop.model_rule[i] != (int)m) continue;
                    int tbin = prop.VariableBinIndex(ivar, i), rbin = prop.VariableBinIndex(v, i);
                    if(tbin<0 || rbin<0) continue;
                    h(tbin, rbin) += prop.added_weights[i];
                }
            }
        }
        nparams = 2;
        param_names = {"dmsq", "sinsq2thmm"}; 
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mu#mu}"}; 
        pretty_param_units = {"eV^{2}", ""}; 
        lb = Eigen::VectorXf(2);
        ub = Eigen::VectorXf(2);
        default_val = Eigen::VectorXf(2);
        lb << -2, -std::numeric_limits<float>::infinity();
        ub << 2, 0;
        default_val << -10, -10;

    };

    /* Function: 3+1 numu->numue disapperance prob in SBL approx */
    float Pmumu(float dmsq, float sinsq2thmumu, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        sinsq2thmumu = std::pow(10.0f, sinsq2thmumu);

        if(sinsq2thmumu > 1) {
            //log<LOG_ERROR>(L"%1% || sinsq2thmumu is %2% which is greater than 1. Setting to 1.")     % __func__ % sinsq2thmumu;
            sinsq2thmumu = 1;
        }
        if(sinsq2thmumu < 0) {
            log<LOG_ERROR>(L"%1% || sinsq2thmumu is %2% which is less than 0. Setting to 0.")
                % __func__ % sinsq2thmumu;
            sinsq2thmumu = 0;
        }

        float sinterm = std::sin(1.27f*dmsq*(le));
        float prob    = 1.0f - (sinsq2thmumu*sinterm*sinterm);

        if(prob<0.0 || prob >1.0 ){;//|| std::isnan(prob)){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                           L"dmsq = %3%, sinsq2thmumu = %4%, L/E = %5%")
                % __func__ % prob % dmsq % sinsq2thmumu % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }
};

class PROnueapp : public PROmodel {
public:
    PROnueapp(const PROpeller &prop,const std::map<std::string,int> &parameter_map) {
        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0;});
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),le);});
        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivar = parameter_map.at("L/E");

        size_t nvar = prop.variable_mc_stat_err.size();
        hists.resize(nvar);
        for(size_t v = 0; v <nvar ;v++){
            for(size_t m = 0; m < model_functions.size(); ++m) {
                hists.at(v).emplace_back(Eigen::MatrixXf::Constant(prop.variable_hist_storage(ivar,v).rows(), prop.variable_hist_storage(ivar,v).cols(),0.0));
                Eigen::MatrixXf &h = hists.at(v).back();
                for(size_t i = 0; i < prop.NEvent(); ++i) {
                    if(prop.model_rule[i] != (int)m) continue;
                    int tbin = prop.VariableBinIndex(ivar, i), rbin = prop.VariableBinIndex(v, i);
                    if(tbin<0 || rbin<0)continue;
                    h(tbin, rbin) += prop.added_weights[i];
                }
            }
        }
         nparams = 2;
        param_names = {"dmsq", "sinsq2thme"}; 
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mue}"}; 
        pretty_param_units = {"eV^{2}", ""}; 
        lb = Eigen::VectorXf(2);
        ub = Eigen::VectorXf(2);
        default_val = Eigen::VectorXf(2);
        lb << -2, -10; //-std::numeric_limits<float>::infinity();
        ub << 2, 0;
        //default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
        default_val << -10, -10; //std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
    
        log<LOG_INFO>(L"%1% || setting up a model nueapp, with  %2% params.")     % __func__ % nparams;
        for(size_t i=0; i< nparams;i++){
            log<LOG_INFO>(L"%1% || Param %2% is %3% with lower bound/upper bound of %4%/%5% and default %6%")     % __func__ % i % param_names[i].c_str() % lb[i] % ub[i] % default_val[i];
        }

    };

    float Pmue(float dmsq, float sinsq2thmue, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        sinsq2thmue = std::pow(10.0f, sinsq2thmue);

        if(sinsq2thmue > 1) {
            //log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is greater than 1. Setting to 1.")  % __func__ % sinsq2thmue;
            sinsq2thmue = 1;
        }
        if(sinsq2thmue < 0) {
            log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is less than 0. Setting to 0.")
                % __func__ % sinsq2thmue;
            sinsq2thmue = 0;
        }

        float sinterm = std::sin(1.27f*dmsq*(le));
        float prob    = sinsq2thmue*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                           L"dmsq = %3%, sinsq2thmue = %4%, L/E = %5%")
                % __func__ % prob % dmsq % sinsq2thmue % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }
};

class PROnuedis : public PROmodel {
public:
    PROnuedis(const PROpeller &prop,const std::map<std::string,int> &parameter_map) {
        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0;});
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),le);});
        

        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'.Make sure its in your model section of XML. ") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivar = parameter_map.at("L/E");

        size_t nvar = prop.variable_mc_stat_err.size();
        hists.resize(nvar);
        for(size_t v = 0; v <nvar ;v++){
            for(size_t m = 0; m < model_functions.size(); ++m) {
                hists.at(v).emplace_back(Eigen::MatrixXf::Constant(prop.variable_hist_storage(ivar,v).rows(), prop.variable_hist_storage(ivar,v).cols(),0.0));
                Eigen::MatrixXf &h = hists.at(v).back();
                for(size_t i = 0; i < prop.NEvent(); ++i) {
                    if(prop.model_rule[i] != (int)m) continue;
                    int tbin = prop.VariableBinIndex(ivar, i), rbin = prop.VariableBinIndex(v, i);
                    if(tbin<0 || rbin<0) continue;
                    h(tbin, rbin) += prop.added_weights[i];
                }
            }
        }
        nparams = 2;
        param_names = {"dmsq", "sinsq2thee"}; 
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{ee}"}; 
        pretty_param_units = {"eV^{2}", ""}; 
        lb = Eigen::VectorXf(2);
        ub = Eigen::VectorXf(2);
        default_val = Eigen::VectorXf(2);
        lb << -2, -std::numeric_limits<float>::infinity();
        ub << 2, 0;
        default_val << -10, -10;

    };

    /* Function: 3+1 nue->nue disapperance prob in SBL approx */
    float Pee(float dmsq, float sinsq2thee, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        sinsq2thee = std::pow(10.0f, sinsq2thee);

        if(sinsq2thee > 1) {
            //log<LOG_ERROR>(L"%1% || sinsq2thee is %2% which is greater than 1. Setting to 1.")     % __func__ % sinsq2thee;
            sinsq2thee = 1;
        }
        if(sinsq2thee < 0) {
            log<LOG_ERROR>(L"%1% || sinsq2thee is %2% which is less than 0. Setting to 0.")
                % __func__ % sinsq2thee;
            sinsq2thee = 0;
        }

        float sinterm = std::sin(1.27f*dmsq*(le));
        float prob    = 1.0f - (sinsq2thee*sinterm*sinterm);

        if(prob<0.0 || prob >1.0 ){;//|| std::isnan(prob)){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                           L"dmsq = %3%, sinsq2thee = %4%, L/E = %5%")
                % __func__ % prob % dmsq % sinsq2thee % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }
};


class PRO3p1 : public PROmodel {
public:
    PRO3p1(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0; });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),v(2),le); });
        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivar = parameter_map.at("L/E");

        //constraints
        model_constraint = [this](const Eigen::VectorXf &v){return this->UnitarityConstraint(v);};


         size_t nvar = prop.variable_mc_stat_err.size();
        hists.resize(nvar);
        for(size_t v = 0; v <nvar ;v++){
            for(size_t m = 0; m < model_functions.size(); ++m) {
                hists.at(v).emplace_back(Eigen::MatrixXf::Constant(prop.variable_hist_storage(ivar,v).rows(), prop.variable_hist_storage(ivar,v).cols(),0.0));
                Eigen::MatrixXf &h = hists.at(v).back();
                for(size_t i = 0; i < prop.NEvent(); ++i) {
                    if(prop.model_rule[i] != (int)m) continue;
                    int tbin = prop.VariableBinIndex(ivar, i), rbin = prop.VariableBinIndex(v, i);
                    if(tbin<0 || rbin<0)continue;
                    h(tbin, rbin) += prop.added_weights[i];
                }
            }
        }


        nparams = 3;
        param_names = {"dmsq", "Ue4^2", "Um4^2"}; 
        pretty_param_names = {"#Deltam^{2}", "|U_{e4}|^{2}", "|U_{#mu4}|^{2}"}; 
        pretty_param_units = {"eV^{2}", "",""}; 
        lb = Eigen::VectorXf(3);
        ub = Eigen::VectorXf(3);
        default_val = Eigen::VectorXf(3);
        lb << -2, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
        ub << 2, -1e-4, -1e-4;
        //default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
        default_val << -2, -8, -8;
    };

    int UnitarityConstraint(const Eigen::VectorXf &v){
        return   (pow(10,v(1))+pow(10,v(2))<1 ? 1 : 0);      
    }

    float Pmue(float dmsq, float Ue4sq, float Um4sq, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        Ue4sq = std::pow(10.0f, Ue4sq);
        Um4sq = std::pow(10.0f, Um4sq);

        if(Ue4sq > 1) {
            log<LOG_ERROR>(L"%1% || Ue4sq is %2% which is greater than 1. Setting to 1.") 
                % __func__ % Ue4sq;
            Ue4sq = 1;
        }
        if(Ue4sq < 0) {
            log<LOG_ERROR>(L"%1% || Ue4sq is %2% which is less than 0. Setting to 0.")
                % __func__ % Ue4sq;
            Ue4sq = 0;
        }
        if(Um4sq > 1) {
            log<LOG_ERROR>(L"%1% || Um4sq is %2% which is greater than 1. Setting to 1.") 
                % __func__ % Um4sq;
            Um4sq = 1;
        }
        if(Um4sq < 0) {
            log<LOG_ERROR>(L"%1% || Um4sq is %2% which is less than 0. Setting to 0.")
                % __func__ % Um4sq;
            Um4sq = 0;
        }

        float sinterm = std::sin(1.27f*dmsq*(le));
        float prob    = 4.0f*Ue4sq*Um4sq*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                           L"dmsq = %3%, Ue4sq = %4%, Um4sq = %5%, L/E = %6%")
                % __func__ % prob % dmsq % Ue4sq % Um4sq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    float Pmumu(float dmsq, [[maybe_unused]]float Ue4sq, float Um4sq, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        Um4sq = std::pow(10.0f, Um4sq);

        if(Um4sq > 1) {
            log<LOG_ERROR>(L"%1% || Um4sq is %2% which is greater than 1. Setting to 1.")
                % __func__ % Um4sq;
            Um4sq = 1;
        }
        if(Um4sq < 0) {
            log<LOG_ERROR>(L"%1% || Um4sq is %2% which is less than 0. Setting to 0.")
                % __func__ % Um4sq;
            Um4sq = 0;
        }

        float sinterm = std::sin(1.27*dmsq*(le));
        float prob    = 1.0f - 4.0f*Um4sq*(1.0f-Um4sq)*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Um4sq = %4%, L/E = %5%") % __func__ % prob % dmsq % Um4sq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    float Pee(float dmsq, float Ue4sq, [[maybe_unused]]float Um4sq, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        Ue4sq = std::pow(10.0f, Ue4sq);

        if(Ue4sq > 1) {
            log<LOG_ERROR>(L"%1% || Ue4sq is %2% which is greater than 1. Setting to 1.")
                % __func__ % Ue4sq;
            Ue4sq = 1;
        }
        if(Ue4sq < 0) {
            log<LOG_ERROR>(L"%1% || Ue4sq is %2% which is less than 0. Setting to 0.")
                % __func__ % Ue4sq;
            Ue4sq = 0;
        }

        float sinterm = std::sin(1.27*dmsq*(le));
        float prob    = 1.0f - 4.0f*Ue4sq*(1.0f-Ue4sq)*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Ue4sq = %4%, L/E = %5%") % __func__ % prob % dmsq % Ue4sq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }
};


class PRO3p1_3C : public PROmodel {
public:
    PRO3p1_3C(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0; });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),v(2),le); });
        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivar = parameter_map.at("L/E");

        //constraints
        model_constraint = [this](const Eigen::VectorXf &v){return this->UnitarityConstraint(v);};


         size_t nvar = prop.variable_mc_stat_err.size();
        hists.resize(nvar);
        for(size_t v = 0; v <nvar ;v++){
            for(size_t m = 0; m < model_functions.size(); ++m) {
                hists.at(v).emplace_back(Eigen::MatrixXf::Constant(prop.variable_hist_storage(ivar,v).rows(), prop.variable_hist_storage(ivar,v).cols(),0.0));
                Eigen::MatrixXf &h = hists.at(v).back();
                for(size_t i = 0; i < prop.NEvent(); ++i) {
                    if(prop.model_rule[i] != (int)m) continue;
                    int tbin = prop.VariableBinIndex(ivar, i), rbin = prop.VariableBinIndex(v, i);
                    if(tbin<0 || rbin<0)continue;
                    h(tbin, rbin) += prop.added_weights[i];
                }
            }
        }


        nparams = 3;
        param_names = {"dmsq", "sinsq2thmue", "xi"}; 
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mue}", "#xi"}; 
        lb = Eigen::VectorXf(3);
        ub = Eigen::VectorXf(3);
        default_val = Eigen::VectorXf(3);
        lb << -2, -std::numeric_limits<float>::infinity(), -10;
        ub << 2, -1e-3, 10;
        //default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
        default_val << -2, -8, 0;
    };

    int UnitarityConstraint(const Eigen::VectorXf &v){
        return   (sqrt(pow(10,v(1)))*cosh(v(2))<0.999 ? 1 : 0);      
    }

    float Pmue(float dmsq, float sinsq2thmue, [[maybe_unused]]float xi, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        sinsq2thmue = std::pow(10.0f, sinsq2thmue);
        
        if(sinsq2thmue > 1) {
            log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is greater than 1. Setting to 1.")
                % __func__ % sinsq2thmue;
            sinsq2thmue = 1;
        }
        if(sinsq2thmue < 0) {
            log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is less than 0. Setting to 0.")
                % __func__ % sinsq2thmue;
            sinsq2thmue = 0;
        }

        float sinterm = std::sin(1.27f*dmsq*(le));
        float prob    = sinsq2thmue*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                           L"dmsq = %3%, sinsq2thmue = %4%, L/E = %5%")
                % __func__ % prob % dmsq % sinsq2thmue % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    float Pmumu(float dmsq, float sinsq2thmue, float xi, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        sinsq2thmue = std::pow(10.0f, sinsq2thmue);

        float Um4sq=(std::exp(xi) * std::sqrt(sinsq2thmue)) / 2.0;

        float sinterm = std::sin(1.27*dmsq*(le));
        float prob    = 1.0f - 4.0f*Um4sq*(1.0f-Um4sq)*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Um4sq = %4%, sinsq2thmue = %5%, xi = %6%, L/E = %7%") % __func__ % prob % dmsq % Um4sq % sinsq2thmue % xi % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    float Pee(float dmsq, float sinsq2thmue, float xi, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        sinsq2thmue = std::pow(10.0f, sinsq2thmue);

        float Ue4sq=(std::exp(-xi) * std::sqrt(sinsq2thmue)) / 2.0;

        float sinterm = std::sin(1.27*dmsq*(le));
        float prob    = 1.0f - 4.0f*Ue4sq*(1.0f-Ue4sq)*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Ue4sq = %4%, sinsq2thmue = %5%, xi = %6%, L/E = %7%") % __func__ % prob % dmsq % Ue4sq % sinsq2thmue % xi % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }
};

class PRO3p1_3B : public PROmodel {
public:
    PRO3p1_3B(const PROpeller &prop,
              const std::map<std::string,int> &parameter_map) {

        // -----------------------------------------
        // 1) Model functions:
        //    0: constant (unosc / NC-like)
        //    1: Pmumu
        //    2: Pmue
        //    3: Pee
        // -----------------------------------------
        model_functions.push_back(
            [this]([[maybe_unused]] const Eigen::VectorXf &v, float) {
                (void)this;
                return 1.0f;
            }
        );
        model_functions.push_back(
            [this](const Eigen::VectorXf &v, float le) {
                return this->Pmumu(v(0), v(1), v(2), le);
            }
        );
        model_functions.push_back(
            [this](const Eigen::VectorXf &v, float le) {
                return this->Pmue(v(0), v(1), v(2), le);
            }
        );
        model_functions.push_back(
            [this](const Eigen::VectorXf &v, float le) {
                return this->Pee(v(0), v(1), v(2), le);
            }
        );

        // -----------------------------------------
        // 2) L/E variable index
        // -----------------------------------------
        if (parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(
                L"%1%, %2% || Missing expected parameter: 'L/E'. "
                L"Make sure it's in your model section of the XML."
            ) % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivar = parameter_map.at("L/E");

        // -----------------------------------------
        // 3) Build histograms for each model component
        // -----------------------------------------
        size_t nvar = prop.variable_mc_stat_err.size();
        hists.resize(nvar);
        for (size_t v = 0; v < nvar; ++v) {
            for (size_t m = 0; m < model_functions.size(); ++m) {
                hists.at(v).emplace_back(
                    Eigen::MatrixXf::Constant(
                        prop.variable_hist_storage(ivar, v).rows(),
                        prop.variable_hist_storage(ivar, v).cols(),
                        0.0f
                    )
                );
                Eigen::MatrixXf &h = hists.at(v).back();
                for (size_t i = 0; i < prop.NEvent(); ++i) {
                    if (prop.model_rule[i] != static_cast<int>(m)) continue;
                    int tbin = prop.VariableBinIndex(ivar, i);
                    int rbin = prop.VariableBinIndex(v, i);
                    if (tbin < 0 || rbin < 0) continue;
                    h(tbin, rbin) += prop.added_weights[i];
                }
            }
        }

        // -----------------------------------------
        // 4) Parameters and bounds
        // v(0) = dmsq_log      = log10(Δm²_41)
        // v(1) = s2mumu_log    = log10(sin²2θ_μμ)
        // v(2) = sB            ∈ [0,1]
        // -----------------------------------------
        nparams = 3;
        param_names        = {"dmsq", "sinsq2thmumu", "sB"};
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mu#mu}", "s^{B}"};

        lb          = Eigen::VectorXf(3);
        ub          = Eigen::VectorXf(3);
        default_val = Eigen::VectorXf(3);

        // Δm²: 10^[-2,2] eV²
        lb(0) = -2.0f;
        ub(0) =  2.0f;

        // sin²2θ_μμ: log10, < 1 (so < 10^0); cap at -1e-3 for safety
        lb(1) = -std::numeric_limits<float>::infinity();
        ub(1) = -1e-3f;

        // sB ∈ [0,1]
        lb(2) = 0.001f;
        ub(2) = 1.0f;

        // Some reasonable defaults
        default_val << -2.0f, -8.0f, 0.0f;

        // -----------------------------------------
        // 5) Unitarity / physicality constraint
        // -----------------------------------------
        model_constraint = [this](const Eigen::VectorXf &v) {
            return this->UnitarityConstraint(v);
        };
    }

   

    // ---------------------------------------------
    // Unitarity / physicality constraint:
    // 0 ≤ sB ≤ 1 and Ue4² + Uμ4² < 1
    // ---------------------------------------------
    int UnitarityConstraint(const Eigen::VectorXf &v) {
        float s2mumu = std::pow(10.0f, v(1));  // sin²2θμμ
        float sB     = v(2);                   // ratio parameter

        float rad = 1.0f - s2mumu;
        float Um4sq = (1.0f - std::sqrt(rad)) / 2.0f;
        float Ue4sq = sB * (1.0f - Um4sq);     // from definition of sB

        return Um4sq + Ue4sq < 0.999 ? 1 :0;  // allowed
    }

    // ---------------------------------------------
    // νμ → νμ disappearance
    // ---------------------------------------------
    float Pmumu(float dmsq_log, float s2mumu_log, float sB, float le) const {
        float dmsq   = std::pow(10.0f, dmsq_log);
        float s2mumu = std::pow(10.0f, s2mumu_log);


        float sinterm = std::sin(1.27f * dmsq * le);
        float prob    = 1.0f - s2mumu * sinterm * sinterm;

        if (prob < 0.0f || prob > 1.0f) {
            log<LOG_ERROR>(
                L"%1% || Pmumu %2% outside [0,1]. "
                L"dmsq = %3% L/E = %5%"
            ) % __func__ % prob % dmsq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    // ---------------------------------------------
    // νμ → νe appearance
    // ---------------------------------------------
    float Pmue(float dmsq_log, float s2mumu_log, float sB, float le) const {
        float dmsq   = std::pow(10.0f, dmsq_log);
        float s2mumu = std::pow(10.0f, s2mumu_log);

        float rad = 1.0f - s2mumu;
        float Um4sq = (1.0f - std::sqrt(rad)) / 2.0f;
        float Ue4sq = sB * (1.0f - Um4sq);

        float sinterm = std::sin(1.27f * dmsq * le);
        float prob    = 4.0f * Um4sq * Ue4sq * sinterm * sinterm;

        if (prob < 0.0f || prob > 1.0f) {
            log<LOG_ERROR>(
                L"%1% || Pmue %2% outside [0,1]. "
                L"dmsq = %3%, Um4sq = %4%, Ue4sq = %5%, L/E = %6%"
            ) % __func__ % prob % dmsq % Um4sq % Ue4sq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    // ---------------------------------------------
    // νe → νe disappearance
    // ---------------------------------------------
    float Pee(float dmsq_log, float s2mumu_log, float sB, float le) const {
        float dmsq   = std::pow(10.0f, dmsq_log);
        float s2mumu = std::pow(10.0f, s2mumu_log);

        float rad = 1.0f - s2mumu;
        float Um4sq = (1.0f - std::sqrt(rad)) / 2.0f;
        float Ue4sq = sB * (1.0f - Um4sq);  


        float sinterm = std::sin(1.27f * dmsq * le);
        float prob    = 1.0f - 4.0f * Ue4sq * (1.0f - Ue4sq) * sinterm * sinterm;

        if (prob < 0.0f || prob > 1.0f) {
            log<LOG_ERROR>(
                L"%1% || Pee %2% outside [0,1]. "
                L"dmsq = %3%, Ue4sq = %4%, L/E = %5%"
            ) % __func__ % prob % dmsq % Ue4sq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }
};

class PRO3p1_3A : public PROmodel {
public:
    PRO3p1_3A(const PROpeller &prop,
              const std::map<std::string,int> &parameter_map) {

        // ----------------------------------------------------
        // 1) Model functions: NC, Pmumu, Pmue, Pee
        // ----------------------------------------------------
        model_functions.push_back(
            [this]([[maybe_unused]] const Eigen::VectorXf &v, float) {
                return 1.0f;
            }
        );
        model_functions.push_back(
            [this](const Eigen::VectorXf &v, float le) {
                return this->Pmumu(v(0), v(1), v(2), le);
            }
        );
        model_functions.push_back(
            [this](const Eigen::VectorXf &v, float le) {
                return this->Pmue(v(0), v(1), v(2), le);
            }
        );
        model_functions.push_back(
            [this](const Eigen::VectorXf &v, float le) {
                return this->Pee(v(0), v(1), v(2), le);
            }
        );

        // ----------------------------------------------------
        // 2) L/E assignment
        // ----------------------------------------------------
        if (parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1% || Missing parameter 'L/E'.") % __func__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivar = parameter_map.at("L/E");

        // ----------------------------------------------------
        // 3) Histogram building
        // ----------------------------------------------------
        size_t nvar = prop.variable_mc_stat_err.size();
        hists.resize(nvar);

        for (size_t v = 0; v < nvar; ++v) {
            for (size_t m = 0; m < model_functions.size(); ++m) {
                hists.at(v).emplace_back(
                    Eigen::MatrixXf::Constant(
                        prop.variable_hist_storage(ivar, v).rows(),
                        prop.variable_hist_storage(ivar, v).cols(),
                        0.0f
                    )
                );
                Eigen::MatrixXf &h = hists.at(v).back();

                for (size_t i = 0; i < prop.NEvent(); ++i) {
                    if (prop.model_rule[i] != (int)m) continue;
                    int tbin = prop.VariableBinIndex(ivar, i);
                    int rbin = prop.VariableBinIndex(v, i);
                    if (tbin < 0 || rbin < 0) continue;
                    h(tbin, rbin) += prop.added_weights[i];
                }
            }
        }

        // ----------------------------------------------------
        // 4) Parameters
        //     v(0) = log10(dmsq)
        //     v(1) = log10(sin²2θ_{ee})
        //     v(2) = rA  (ratio parameter)
        // ----------------------------------------------------
        nparams = 3;
        param_names        = {"dmsq", "sinsq2thee", "rA"};
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{ee}", "r^{A}"};

        lb = Eigen::VectorXf(3);
        ub = Eigen::VectorXf(3);
        default_val = Eigen::VectorXf(3);

        lb << -2.0f, -std::numeric_limits<float>::infinity(),   0.001f;
        ub <<  2.0f,   -1e-3f,    1.0f;

        default_val << -2.0f, -8.0f, 0.0f;

        // ----------------------------------------------------
        // 5) Constraint (no clamping)
        // ----------------------------------------------------
        model_constraint = [this](const Eigen::VectorXf &v) {
            return this->UnitarityConstraint(v);
        };
    }

    // --------------------------------------------------------
    // Physicality / unitarity constraint (no clamping)
    // --------------------------------------------------------
    int UnitarityConstraint(const Eigen::VectorXf &v) {

        float sinsq2thee = std::pow(10.0f, v(1));
        float rA         = v(2);


        // |Ue4|² from sin²2θ_ee  -> eq. (44)
        float rad = 1.0f - sinsq2thee;

        float root  = std::sqrt(rad);

        // |Uμ4|² from rA  -> eq. (40)
        float Um4sq = rA * (1.0f + root) / 2.0f;
        float Ue4sq = 1- (Um4sq/rA);

        // Unitarity (no τ component assumed)
        return Ue4sq + Um4sq < 0.999 ? 1 : 0;
    }

    // --------------------------------------------------------
    // νμ → νμ disappearance
    // --------------------------------------------------------
    float Pmumu(float dmsq_log, float sinsq2thee_log, float rA, float le) const {

        float dmsq        = std::pow(10.0f, dmsq_log);
        float sinsq2thee  = std::pow(10.0f, sinsq2thee_log);

        float rad   = 1.0f - sinsq2thee;
        float root  = std::sqrt(rad);
        float Ue4sq = (1.0f - root) / 2.0f;
        float Um4sq = rA * (1.0f - Ue4sq);

        float sinterm = std::sin(1.27f * dmsq * le);
        float prob    = 1.0f - 4.0f * Um4sq * (1.0f - Um4sq) * sinterm * sinterm;

        if (prob < 0.0f || prob > 1.0f) {
            log<LOG_ERROR>(L"%1% || Pmumu=%2% out of [0,1].") % __func__ % prob;
            exit(EXIT_FAILURE);
        }
        return prob;
    }

    // --------------------------------------------------------
    // νμ → νe appearance
    // --------------------------------------------------------
    float Pmue(float dmsq_log, float sinsq2thee_log, float rA, float le) const {

        float dmsq        = std::pow(10.0f, dmsq_log);
        float sinsq2thee  = std::pow(10.0f, sinsq2thee_log);

        float rad   = 1.0f - sinsq2thee;
        float root  = std::sqrt(rad);
        float Ue4sq = (1.0f - root) / 2.0f;
        float Um4sq = rA * (1.0f - Ue4sq);

        float sinterm = std::sin(1.27f * dmsq * le);
        float prob    = 4.0f * Ue4sq * Um4sq * sinterm * sinterm;

        if (prob < 0.0f || prob > 1.0f) {
            log<LOG_ERROR>(L"%1% || Pmue=%2% out of [0,1].") % __func__ % prob;
            exit(EXIT_FAILURE);
        }
        return prob;
    }

    // --------------------------------------------------------
    // νe → νe disappearance
    // --------------------------------------------------------
    float Pee(float dmsq_log, float sinsq2thee_log, float rA, float le) const {

        float dmsq        = std::pow(10.0f, dmsq_log);
        float sinsq2thee  = std::pow(10.0f, sinsq2thee_log);

        float sinterm = std::sin(1.27f * dmsq * le);
        float prob    = 1.0f - sinsq2thee * sinterm * sinterm;

        if (prob < 0.0f || prob > 1.0f) {
            log<LOG_ERROR>(L"%1% || Pee=%2% out of [0,1].") % __func__ % prob;
            exit(EXIT_FAILURE);
        }
        return prob;
    }
};


class PRO3p2 : public PROmodel {
public:
    PRO3p2(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

        // model functions: 0 = null, 1 = numu->numu, 2 = numu->nue, 3 = nue->nue
        model_functions.push_back(
            [this]([[maybe_unused]] const Eigen::VectorXf &v, float) { (void)this; return 1.0f; });
        model_functions.push_back(
            [this](const Eigen::VectorXf &v, float le) { return this->Pmumu(v, le); });
        model_functions.push_back(
            [this](const Eigen::VectorXf &v, float le) { return this->Pmue(v, le); });
        model_functions.push_back(
            [this](const Eigen::VectorXf &v, float le) { return this->Pee(v, le); });

        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.")
                % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivar = parameter_map.at("L/E");

        // Unitarity constraints for e and mu rows
        model_constraint = [this](const Eigen::VectorXf &v){ return this->UnitarityConstraint(v); };

        // build histograms as in other models
        size_t nvar = prop.variable_mc_stat_err.size();
        hists.resize(nvar);
        for(size_t v = 0; v < nvar; ++v) {
            for(size_t m = 0; m < model_functions.size(); ++m) {
                hists.at(v).emplace_back(
                    Eigen::MatrixXf::Constant(
                        prop.variable_hist_storage(ivar, v).rows(),
                        prop.variable_hist_storage(ivar, v).cols(), 0.0));
                Eigen::MatrixXf &h = hists.at(v).back();
                for(size_t i = 0; i < prop.NEvent(); ++i) {
                    if(prop.model_rule[i] != (int)m) continue;
                    int tbin = prop.VariableBinIndex(ivar, i);
                    int rbin = prop.VariableBinIndex(v, i);
                    if(tbin < 0 || rbin < 0) continue;
                    h(tbin, rbin) += prop.added_weights[i];
                }
            }
        }

        // parameters: dmsq41, dmsq51, log10(Ue4^2), log10(Um4^2), log10(Ue5^2), log10(Um5^2), phi54
        nparams = 7;
        param_names = {"dmsq41", "dmsq51", "Ue4^2", "Um4^2", "Ue5^2", "Um5^2", "phi54"};
        pretty_param_names = {
            "#Delta m^{2}_{41}", "#Delta m^{2}_{51}",
            "|U_{e4}|^{2}",      "|U_{#mu4}|^{2}",
            "|U_{e5}|^{2}",      "|U_{#mu5}|^{2}",
            "#phi_{54}"
        };
        pretty_param_units = {"eV^{2}", "eV^{2}", "", "", "", "", "rad"};

        lb = Eigen::VectorXf(nparams);
        ub = Eigen::VectorXf(nparams);
        default_val = Eigen::VectorXf(nparams);

        // log10(Δm^2) between 10^-2 and 10^2, mixings between ~10^-8 and 1, phi in [-pi,pi]
        lb << -2, -2,
              -5, -5,
              -5, -5,
               0.0f;
        ub <<  2,  2,
              -0.01,  -0.01,
              -0.01,  -0.01,
              6.28318f;

        // some reasonable defaults
        default_val << -1, 0, -4, -4, -4, -4, 0.0f;
    }

    // Enforce |Ue4|^2 + |Ue5|^2 <= 1 and |Um4|^2 + |Um5|^2 <= 1
    int UnitarityConstraint(const Eigen::VectorXf &v) {
        float Ue4sq = std::pow(10.0f, v(2));
        float Um4sq = std::pow(10.0f, v(3));
        float Ue5sq = std::pow(10.0f, v(4));
        float Um5sq = std::pow(10.0f, v(5));

        if(Ue4sq + Ue5sq >= 1.0f) return 0;
        if(Um4sq + Um5sq >= 1.0f) return 0;
        // more careful with unitarity since tau and steriles are not specified and checking the max value of probability
        if(4*Um4sq*Ue4sq + 4*Um5sq*Ue5sq + 8*std::sqrt(Um4sq*Ue4sq*Um5sq*Ue5sq) >= 1.0f) return 0;

        if(1-4*(1-Ue4sq- Ue5sq)*(Ue4sq + Ue5sq) - 4*Ue4sq*Ue5sq >= 1.0f) return 0;

        if(1-4*(1-Um4sq- Um5sq)*(Um4sq + Um5sq) - 4*Um4sq*Um5sq >= 1.0f) return 0;


        return 1;
    }

    // Convenience to unpack parameters

    // ---------- Appearance: νμ → νe (α=μ, β=e) ----------
    float Pmue(const Eigen::VectorXf &v, float le) const {

    // Convert log10 parameters to physical values
    float dm41  = std::pow(10.0f, v(0));
    float dm51  = std::pow(10.0f, v(1));
    float Ue4sq = std::pow(10.0f, v(2));
    float Um4sq = std::pow(10.0f, v(3));
    float Ue5sq = std::pow(10.0f, v(4));
    float Um5sq = std::pow(10.0f, v(5));
    float phi54 = v(6);

    // Oscillation phases
    float x41 = 1.27f * dm41 * le;
    float x51 = 1.27f * dm51 * le;
    float x54 = 1.27f * (dm51 - dm41) * le;

    // Standard 3+2 appearance terms
    float term1 = 4.0f * Um4sq * Ue4sq * std::sin(x41) * std::sin(x41);
    float term2 = 4.0f * Um5sq * Ue5sq * std::sin(x51) * std::sin(x51);
    float term3 = 8.0f * std::sqrt(Um4sq * Ue4sq * Um5sq * Ue5sq)
                        * std::sin(x41) * std::sin(x51)
                        * std::cos(x54 - phi54);

    float prob = term1 + term2 + term3;

    const float eps = 1e-3f;

    if (prob < 0.0f && prob > -eps) prob = 0.0f;

    if (prob > 1.0f && prob < 1.0f + eps) prob = 1.0f;

    if(prob < 0.0f || prob > 1.0f || std::isnan(prob)) {
        log<LOG_ERROR>(L"%1% || Bad Pmue = %2%  (le=%3%)"
            L"\ndm41=%4% dm51=%5%  Ue4sq=%6% Um4sq=%7%  Ue5sq=%8% Um5sq=%9%  phi54=%10% term1=%11% term2=%12% term3=%13%"
             ) % __func__ % prob % le
             % dm41 % dm51 % Ue4sq % Um4sq % Ue5sq % Um5sq % phi54% term1 % term2 % term3;
            exit(EXIT_FAILURE);
        }
        return prob;
    }


    // ---------- Disappearance: νμ → νμ (α=μ) ----------
   float Pmumu(const Eigen::VectorXf &v, float le) const {

    float dm41  = std::pow(10.0f, v(0));
    float dm51  = std::pow(10.0f, v(1));
    float Ue4sq = std::pow(10.0f, v(2));
    float Um4sq = std::pow(10.0f, v(3));
    float Ue5sq = std::pow(10.0f, v(4));
    float Um5sq = std::pow(10.0f, v(5));
    float phi54 = v(6);

    float x41 = 1.27f * dm41 * le;
    float x51 = 1.27f * dm51 * le;
    float x54 = 1.27f * (dm51 - dm41) * le;

    float one_minus = 1.0f - Um4sq - Um5sq;

    float s41 = std::sin(x41);
    float s51 = std::sin(x51);
    float s54 = std::sin(x54);

// Individual components
    float term1 = 1.0f;

    float term2 = -4.0f * one_minus *
              (Um4sq * s41 * s41 +
               Um5sq * s51 * s51);

    float term3 = -4.0f * Um4sq * Um5sq *
              (s54 * s54);

// Full probability
    float prob = term1 + term2 + term3;

    
    const float eps = 1e-3f;

    if (prob < 0.0f && prob > -eps) prob = 0.0f;

    if (prob > 1.0f && prob < 1.0f + eps) prob = 1.0f;

    if(prob < 0.0f || prob > 1.0f || std::isnan(prob)) {
        log<LOG_ERROR>(L"%1% || Bad Pmumu = %2%  (le=%3%)"
            L"\ndm41=%4% dm51=%5%  Ue4sq=%6% Um4sq=%7%  Ue5sq=%8% Um5sq=%9%  phi54=%10% term1=%11% term2=%12% term3=%13%"
        ) % __func__ % prob % le
          % dm41 % dm51 % Ue4sq % Um4sq % Ue5sq % Um5sq % phi54 % term1 % term2 % term3;
        exit(EXIT_FAILURE);
    }
    return prob;
    }


    // ---------- Disappearance: νe → νe (α=e) ----------
   float Pee(const Eigen::VectorXf &v, float le) const {

    float dm41  = std::pow(10.0f, v(0));
    float dm51  = std::pow(10.0f, v(1));
    float Ue4sq = std::pow(10.0f, v(2));
    float Um4sq = std::pow(10.0f, v(3));
    float Ue5sq = std::pow(10.0f, v(4));
    float Um5sq = std::pow(10.0f, v(5));
    float phi54 = v(6);

    float x41 = 1.27f * dm41 * le;
    float x51 = 1.27f * dm51 * le;
    float x54 = 1.27f * (dm51 - dm41) * le;

    float one_minus = 1.0f - Ue4sq - Ue5sq;

    float s41 = std::sin(x41);
    float s51 = std::sin(x51);
    float s54 = std::sin(x54);

    float term1 = 1.0f;
    float term2 = -4.0f * one_minus * (Ue4sq * s41 * s41 + Ue5sq * s51 * s51);
    float term3 = -4.0f * Ue4sq * Ue5sq * s54 * s54;

    float prob = term1 + term2 + term3;

    const float eps = 1e-2f;

    if (prob < 0.0f && prob > -eps) prob = 0.0f;

    if (prob > 1.0f && prob < 1.0f + eps) prob = 1.0f;

    if(prob < 0.0f || prob > 1.0f || std::isnan(prob)) {
        log<LOG_ERROR>(L"%1% || Bad Pee = %2%  (le=%3%)"
            L"\ndm41=%4% dm51=%5%  Ue4sq=%6% Um4sq=%7%  Ue5sq=%8% Um5sq=%9%  phi54=%10% term1=%11% term2=%12% term3=%13%"
        ) % __func__ % prob % le
          % dm41 % dm51 % Ue4sq % Um4sq % Ue5sq % Um5sq % phi54 % term1 % term2 % term3;
        exit(EXIT_FAILURE);
    }
    return prob;
    }

};

// Main interface to different models
static inline
std::unique_ptr<PROmodel> get_model_from_string(const PROconfig& config, const PROpeller &prop) {
     std::string name = config.m_model_tag;

     if(name == "numudis") {
        return std::unique_ptr<PROmodel>(new PROnumudis(prop,config.m_model_parameter_map));
    } else if(name == "nueapp") {
        return std::unique_ptr<PROmodel>(new PROnueapp(prop,config.m_model_parameter_map));
    } else if(name == "nuedis") {
        return std::unique_ptr<PROmodel>(new PROnuedis(prop,config.m_model_parameter_map));
    } else if(name == "3+1") {
        return std::unique_ptr<PROmodel>(new PRO3p1(prop,config.m_model_parameter_map));
    } else if(name == "3+1_3C") {
        return std::unique_ptr<PROmodel>(new PRO3p1_3C(prop,config.m_model_parameter_map));
    }
    log<LOG_ERROR>(L"%1% || Unrecognized model name %2%. Try numudis, nueapp, 3+1 or 3+1_3C for now. Terminating.") % __func__ % name.c_str();
    exit(EXIT_FAILURE);
}

}

#endif