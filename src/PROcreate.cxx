#include "PROcreate.h"
#include "PROdata.h"
#include "PROlog.h"
#include "PROpeller.h"
#include "PROtocall.h"
#include "TTree.h"
#include "TFile.h"
#include "TFriendElement.h"
#include "TChain.h"
#include "TLeaf.h"
#include "TBranch.h"
#include "TTreeFormula.h"
#include <Eigen/Eigen>
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
namespace PROfit {



    void saveSystStructVector(const std::vector<std::vector<SystStruct>> &structs, const std::string &filename) {
        std::ofstream ofs(filename, std::ios::binary);
        boost::archive::binary_oarchive oa(ofs);
        oa & structs;  
    }

    void loadSystStructVector(std::vector<std::vector<SystStruct>> &structs, const std::string &filename) {
        std::ifstream ifs(filename, std::ios::binary);
        boost::archive::binary_iarchive ia(ifs);
        ia & structs;
    }

    void saveDetVarProps(const std::map<std::string, PROpeller>& props, uint32_t detvar_hash, const std::string& filename) {
        auto start = std::chrono::high_resolution_clock::now();
        std::ofstream ofs(filename, std::ios::binary);
        if(!ofs.is_open()) {
            log<LOG_ERROR>(L"%1% || ERROR: Could not open file for writing: %2%") % __func__ % filename.c_str();
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }
        boost::archive::binary_oarchive oa(ofs);
        oa & detvar_hash;
        oa & props;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        log<LOG_INFO>(L"%1% || Saved combined DetVar props (%2% entries) to %3% in %4% seconds") % __func__ % props.size() % filename.c_str() % elapsed.count();
    }

    uint32_t loadDetVarProps(std::map<std::string, PROpeller>& props, const std::string& filename) {
        auto start = std::chrono::high_resolution_clock::now();
        std::ifstream ifs(filename, std::ios::binary);
        if(!ifs.is_open()) {
            log<LOG_ERROR>(L"%1% || ERROR: Could not open file for reading: %2%") % __func__ % filename.c_str();
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }
        boost::archive::binary_iarchive ia(ifs);
        uint32_t detvar_hash;
        ia & detvar_hash;
        ia & props;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        log<LOG_INFO>(L"%1% || Loaded combined DetVar props (%2% entries) from %3% in %4% seconds") % __func__ % props.size() % filename.c_str() % elapsed.count();
        return detvar_hash;
    }


    std::string convertToXRootD(std::string fname_orig){
        std::string fname_use = fname_orig;
        // Search for "/pnfs" directly: testing for "pnfs" but erasing at
        // find("/pnfs") threw std::out_of_range for paths containing "pnfs"
        // without the leading slash.
        const std::string p = "/pnfs";
        std::string::size_type i = fname_orig.find(p);
        if(i != std::string::npos){
            fname_orig.erase(i, p.length());
            fname_use = "root://fndca1.fnal.gov:1094/pnfs/fnal.gov/usr"+fname_orig;
        }
        return fname_use;
    }


    void SystStruct::CleanSpecs(){
        if(p_cv)  p_cv.reset();
        p_multi_spec.clear();
        return;
    }

    void SystStruct::CreateSpecs(int num_bins){
        this->CleanSpecs();
        log<LOG_DEBUG>(L"%1% || Creating multi-universe spectrum with dimension (%2% x %3%)") % __func__ % n_univ % num_bins;

        p_cv = std::make_shared<PROspec>(num_bins);
        for(int i = 0; i != n_univ; ++i){
            p_multi_spec.push_back(std::make_shared<PROspec>(num_bins));
        }
        return;
    }


    const PROspec& SystStruct::CV() const{
        return *p_cv;
    }

    const PROspec& SystStruct::Variation(int universe) const{
        return *(p_multi_spec.at(universe));
    }

    void SystStruct::FillCV(int global_bin, float event_weight){
        p_cv->Fill(global_bin, event_weight);
        return;
    }

    void SystStruct::FillUniverse(int universe, int global_bin, float event_weight){
        
        /*
        if (event_weight < 0) {
            log<LOG_ERROR>(L"%1% || Event weight is negative with value %2% for systematic %3%") % __func__ % event_weight % systname.c_str();
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }
        */
        
        p_multi_spec.at(universe)->QuickFill(global_bin, event_weight);
        return;
    }

    void SystStruct::SanityCheck() const{
        if(mode == "minmax" && n_univ != 2){
            log<LOG_ERROR>(L"%1% || Systematic variation %2% is tagged as minmax mode, but has %3% universes (can only be 2)") % __func__ % systname.c_str() % n_univ;
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }

        log<LOG_DEBUG>(L"%1% || Systematic variation %2% has %3% universes, and is in %4% mode with weight formula: %5%") % __func__ % systname.c_str() % n_univ % mode.c_str() % weight_formula.c_str();
        return;
    }

    void SystStruct::Print() const{
        log<LOG_INFO>(L"%1% || Printing %2%") % __func__ % systname.c_str();
        int i=0;
        for(auto &spec: p_multi_spec){
            log<LOG_INFO>(L"%1% || On Universe %2% for knob %3% ") % __func__ % i % knobval[i];
            spec->Print();
            i++;
        }

        return;
    }

    int PROcess_CAFAna(const PROconfig &inconfig, std::vector<std::vector<SystStruct>> &syst_vector, PROpeller& inprop,bool noxrootd){

        log<LOG_DEBUG>(L"%1% || Loading Monte Carlo Files") % __func__ ;

        int num_files = inconfig.m_num_mcgen_files;

        log<LOG_DEBUG>(L"%1% || Using a total of %2% individual files") % __func__  % num_files;

        std::vector<long int> nentries(num_files,0);
        std::vector<std::unique_ptr<TFile>> files(num_files);
        //std::vector<TTree*> trees(num_files,nullptr);//keep as bare pointers because of ROOT :(
        std::vector<TChain*> chains(num_files,nullptr);
        std::vector<std::vector<TChain*>> friendChains(num_files) ;


        std::vector<std::vector<std::map<std::string, std::vector<eweight_type>*>>> f_event_weights(num_files);
        std::vector<std::vector<std::map<std::string, std::vector<eweight_type>*>>> f_knob_vals(num_files);
        std::map<std::string, int> map_systematic_num_universe;
        std::map<std::string, std::vector<eweight_type>> map_systematic_knob_vals;
        std::map<std::string, std::string> map_systematic_type;

        //open files, and link trees and branches
        int good_event = 0;
        bool useXrootD = !noxrootd;


        for(int fid=0; fid < num_files; ++fid) {
            const auto& fn = inconfig.m_mcgen_file_name.at(fid);


            std::vector<std::string> filesForChain;

            if (fn.find(".root") != std::string::npos) {
                log<LOG_INFO>(L"%1% || Starting a (single) TChain, loading file %2%") % __func__  % fn.c_str();
                filesForChain.push_back(useXrootD ? convertToXRootD(fn) : fn);
            }else{

                log<LOG_INFO>(L"%1% || Starting a TCHain, loading from filelist %2%") % __func__  % fn.c_str();
                std::ifstream infile(fn.c_str()); 
                if (!infile) {
                    log<LOG_ERROR>(L"%1% || Failed to open input filelist %2%") % __func__  % fn.c_str();
                    exit(EXIT_FAILURE);
                }

                std::string line;
                while (std::getline(infile, line)) {
                    if(useXrootD) line = convertToXRootD(line);
                    log<LOG_INFO>(L"%1% || Loading file %2% into TChain") %__func__ % line.c_str();
                    filesForChain.push_back(line);
                }

                infile.close();
            }

            chains[fid] = new TChain(inconfig.m_mcgen_tree_name.at(fid).c_str());
            if (inconfig.m_mcgen_numfriends[fid] > 0) {
                friendChains[fid].resize(inconfig.m_mcgen_numfriends[fid], nullptr);
            }

            for(auto &file: filesForChain){ 

                chains[fid]->Add(file.c_str()); 

                if (inconfig.m_mcgen_numfriends[fid] > 0) {
                    auto mcgen_file_friend_treename_iter = inconfig.m_mcgen_file_friend_treename_map.find(fn);
                    if (mcgen_file_friend_treename_iter != inconfig.m_mcgen_file_friend_treename_map.end()) {
                        auto mcgen_file_friend_iter = inconfig.m_mcgen_file_friend_map.find(fn);
                        if (mcgen_file_friend_iter == inconfig.m_mcgen_file_friend_map.end()) {
                            log<LOG_ERROR>(L"%1% || Friend TTree provided but no friend file??") % __func__;
                            log<LOG_ERROR>(L"Terminating.");
                            exit(EXIT_FAILURE);
                        }

                        for (size_t k = 0; k < mcgen_file_friend_treename_iter->second.size(); k++) {
                            std::string treefriendname = mcgen_file_friend_treename_iter->second.at(k);
                            //std::string treefriendfile = mcgen_file_friend_iter->second.at(k);//not used

                            if (!friendChains[fid][k]) {
                                friendChains[fid][k] = new TChain(treefriendname.c_str());
                            }

                            // Add the file to the friend chain
                            log<LOG_DEBUG>(L"%1% || Adding friend tree %2% from file %3%") % __func__ % treefriendname.c_str() % file.c_str();
                            friendChains[fid][k]->Add(file.c_str());
                        }

                    }
                } // End friend initialization
            } // End chain filling

            if (inconfig.m_mcgen_numfriends[fid] > 0) {
                for (size_t k = 0; k < friendChains[fid].size(); k++) {
                    log<LOG_DEBUG>(L"%1% || Adding friend chain %2% to main chain %3%") % __func__ % k % fid;
                    chains[fid]->AddFriend(friendChains[fid][k]);
                }
            }


            nentries[fid] = chains[fid]->GetEntries();



            // grab branches 
            int num_branch = inconfig.m_branch_variables[fid].size();
            f_event_weights[fid].resize(1);//was num_branch
            f_knob_vals[fid].resize(1);//was num_brach
            bool gotWeights = false;

            for(int ib = 0; ib != num_branch; ++ib) {

                std::shared_ptr<BranchVariable> branch_variable = inconfig.m_branch_variables[fid][ib];

                //quick check that this branch associated subchannel is in the known chanels;
                int is_valid_subchannel = 0;
                for(const auto &name: inconfig.m_fullnames){
                    if(branch_variable->associated_hist==name){
                        log<LOG_DEBUG>(L"%1% || Found a valid subchannel for this branch %2%") % __func__  % name.c_str();
                        ++is_valid_subchannel;
                    }
                }
                if(is_valid_subchannel==0){
                    log<LOG_ERROR>(L"%1% || This branch did not match one defined in the .xml : %2%") % __func__ % inconfig.m_xmlname.c_str();
                    log<LOG_ERROR>(L"%1% || There is probably a typo somehwhere in xml!") % __func__;
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);

                }else if(is_valid_subchannel>1){
                    log<LOG_ERROR>(L"%1% || This branch matched more than 1 subchannel!: %2%") % __func__ %  branch_variable->associated_hist.c_str();
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);
                }

                //branch_variable->branch_formula = std::make_shared<TTreeFormula>(("branch_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->name.c_str(), chains[fid]);
                //log<LOG_INFO>(L"%1% || Setting up reco variable for this branch: %2%") % __func__ %  branch_variable->name.c_str();
                //branch_variable->branch_true_L_formula = std::make_shared<TTreeFormula>(("branch_L_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->true_L_name.c_str(), chains[fid]);
                //log<LOG_INFO>(L"%1% || Setting up L variable for this branch: %2%") % __func__ %  branch_variable->true_L_name.c_str();
                //branch_variable->branch_true_value_formula = std::make_shared<TTreeFormula>(("branch_true_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->true_param_name.c_str(), chains[fid]);
                //log<LOG_INFO>(L"%1% || Setting up true E variable for this branch: %2%") % __func__ %  branch_variable->true_param_name.c_str();
                //branch_variable->branch_true_pdg_formula = std::make_shared<TTreeFormula>(("branch_pdg_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->pdg_name.c_str(), chains[fid]);
                //log<LOG_INFO>(L"%1% || Setting up PDG variable for this branch: %2%") % __func__ %  branch_variable->pdg_name.c_str();
                //if (branch_variable->hist_reweight) {
                //branch_variable->branch_true_proton_mom_formula = std::make_shared<TTreeFormula>(("branch_pmom_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->true_proton_mom_name.c_str(), chains[fid]);
                //log<LOG_INFO>(L"%1% || Setting up leading proton momentum variable for this branch: %2%") % __func__ %  branch_variable->true_proton_mom_name.c_str();
                //branch_variable->branch_true_proton_costh_formula = std::make_shared<TTreeFormula>(("branch_costh_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->true_proton_costh_name.c_str(), chains[fid]);
                //log<LOG_INFO>(L"%1% || Setting up leading proton costh variable for this branch: %2%") % __func__ %  branch_variable->true_proton_costh_name.c_str();
                //}
                int other_count = 0;
                for(const auto &name: branch_variable->variable_names) {
                    log<LOG_INFO>(L"%1% || Setting up variable formula for file %2%, branch %3% (%4%): %5%") % __func__ % fid % ib % branch_variable->associated_hist.c_str() % name.c_str();
                    branch_variable->branch_variable_formulas.push_back(std::make_shared<ROOTFormula>(
                        "branch_variable_form_"+std::to_string(fid) +"_" + std::to_string(ib)+"_"+std::to_string(other_count),
                        name, chains[fid]));
                    other_count++;
                }


                //grab weight formulas (weight_1, weight_2, ...)
                for(int wi = 0; wi < inconfig.m_mcgen_num_weights[fid][ib]; ++wi){
                    const std::string& wname = inconfig.m_mcgen_weight_names[fid][ib][wi];
                    log<LOG_INFO>(L"%1% || Setting up weight_%2% for file %3%, branch %4% (%5%): %6%") % __func__ % (wi+1) % fid % ib % branch_variable->associated_hist.c_str() % wname.c_str();
                    branch_variable->branch_weight_formulas.push_back(std::make_shared<ROOTFormula>(
                        "branch_weight_"+std::to_string(wi+1)+"_"+std::to_string(fid)+"_" + std::to_string(ib),
                        wname, chains[fid]));
                    log<LOG_INFO>(L"%1% || Successfully set up weight_%2% for this branch: %3%") % __func__ % (wi+1) % wname.c_str();
                }


                // check to make sure if its in allowlist, it IS in files
                std::vector<std::string> allowlist_check;

                //grab eventweight branch
                if(!gotWeights){
                    for(const TObject* branch: *chains[fid]->GetListOfBranches()) {
                        log<LOG_DEBUG>(L"%1% || Checking if branch %2% is in allowlist") % __func__ %  branch->GetName();

                        if (std::find(inconfig.m_mcgen_variation_allowlist.begin(), inconfig.m_mcgen_variation_allowlist.end(), branch->GetName()) != inconfig.m_mcgen_variation_allowlist.end()) {
                            log<LOG_INFO>(L"%1% || Setting up eventweight map for this branch: %2% for fid %3%") % __func__ %  branch->GetName() % fid;
                            chains[fid]->SetBranchAddress(branch->GetName(), &(f_event_weights[fid][0][branch->GetName()]));
                            allowlist_check.push_back(branch->GetName());
                        } else if(strlen(branch->GetName()) > 6 && strcmp(branch->GetName() + strlen(branch->GetName()) - 6, "_sigma") == 0) {
                            log<LOG_INFO>(L"%1% || Setting up knob val list using branch %2% for fid %3%") % __func__ % branch->GetName() % fid;
                            chains[fid]->SetBranchAddress(branch->GetName(), &(f_knob_vals[fid][0][branch->GetName()]));
                            allowlist_check.push_back(branch->GetName());
                        }
                    }
                    if(inconfig.m_mcgen_numfriends[fid]>0){
                        log<LOG_DEBUG>(L"%1% || Some Friend Processing") % __func__ ;
                        for(const TObject* friend_: *chains[fid]->GetListOfFriends()) {
                            for(const TObject* branch: *((TFriendElement*)friend_)->GetTree()->GetListOfBranches()) {
                                log<LOG_DEBUG>(L"%1% || Checking if branch %2% is in allowlist") % __func__ %  branch->GetName();

                                if (std::find(inconfig.m_mcgen_variation_allowlist.begin(), inconfig.m_mcgen_variation_allowlist.end(), branch->GetName()) != inconfig.m_mcgen_variation_allowlist.end()) {
                                    if(branch_variable->GetIncludeSystematics()){
                                        log<LOG_INFO>(L"%1% || Setting up eventweight map for this branch: %2%") % __func__ %  branch->GetName();

                                        chains[fid]->SetBranchAddress(branch->GetName(), &(f_event_weights[fid][0][branch->GetName()]));
                                        allowlist_check.push_back(branch->GetName());


                                    }else{
                                        log<LOG_INFO>(L"%1% || EXPLICITLY NOT Setting up eventweight map for this branch: %2%") % __func__ %  branch->GetName();
                                    }
                                } else if(strlen(branch->GetName()) > 6 && strcmp(branch->GetName() + strlen(branch->GetName()) - 6, "_sigma") == 0) {
                                    log<LOG_INFO>(L"%1% || Setting up knob val list using branch %2%") % __func__ % branch->GetName();
                                    chains[fid]->SetBranchAddress(branch->GetName(), &(f_knob_vals[fid][0][branch->GetName()]));
                                    allowlist_check.push_back(branch->GetName());
                                }
                            }
                        }
                    }
                    gotWeights = true;
                    if(branch_variable->GetIncludeSystematics()){
                        for(const auto &variation: inconfig.m_mcgen_variation_allowlist){
                            std::string type = inconfig.m_mcgen_variation_type_map.at(variation);
                            if (std::find(allowlist_check.begin(), allowlist_check.end(), variation  ) == allowlist_check.end() && (type=="covariance" || type=="covariance_to_spline" || type=="spline" || type=="spline_to_covariance")) {
                                log<LOG_ERROR>(L"%1% || ERROR! You have a variation named %2% in your allowlist, so you definitely want it, but its NOT found in the files. Is this a typo? FileID %3%") % __func__ % variation.c_str() %fid  ;
                                throw std::runtime_error("Allowlist variation not in file.");
                            }
                        }
                    }
                }//


                log<LOG_INFO>(L"%1% || This mcgen file has %2% friends.") % __func__ %  inconfig.m_mcgen_numfriends[fid];

            } //end of branch loop


            //calculate how many universes each systematic has.
            log<LOG_INFO>(L"%1% || Start calculating number of universes for systematics") % __func__;
            chains.at(fid)->GetEntry(good_event);

            for(int ib = 0; ib != num_branch; ++ib) {
                const auto& branch_variable = inconfig.m_branch_variables[fid][ib];
                auto& f_weight = f_event_weights[fid][0];
                auto& f_knob = f_knob_vals[fid][0];

                if(branch_variable->GetIncludeSystematics()){
                    for(const auto& it : f_weight){
                        log<LOG_DEBUG>(L"%1% || On systematic: %2%") % __func__ % it.first.c_str();

                        int count = std::count(inconfig.m_mcgen_variation_allowlist.begin(), inconfig.m_mcgen_variation_allowlist.end(), it.first);
                        if(count==0){
                            log<LOG_DEBUG>(L"%1% || Skip systematic: %2% as its not in the AllowList!!") % __func__ % it.first.c_str();
                            continue;
                        }

                        count = std::count(inconfig.m_mcgen_variation_denylist.begin(), inconfig.m_mcgen_variation_denylist.end(), it.first);
                        if(count>0){
                            log<LOG_DEBUG>(L"%1% || Skip systematic: %2% as it is in the DenyList!!") % __func__ % it.first.c_str();
                            continue;
                        }

                        log<LOG_INFO>(L"%1% || %2% has %3% montecarlo variations in branch %4%") % __func__ % it.first.c_str() % it.second->size() % branch_variable->associated_hist.c_str();

                        map_systematic_num_universe[it.first] = std::max((int)map_systematic_num_universe[it.first], (int)it.second->size());
                        if(f_knob.find(it.first+"_sigma") != std::end(f_knob)) {
                            map_systematic_knob_vals[it.first] = *f_knob[it.first+"_sigma"];
                        }
                    }
                }
            }
        } // end fid

        //Do we have any flat systeatics?
        if(inconfig.m_num_variation_type_flat>0){
            for(auto& allow_sys : inconfig.m_mcgen_variation_type_map){
                if(allow_sys.second=="flat"){
                    map_systematic_num_universe[allow_sys.first] = -1;
                }
            }
        }

        // Do we have any norm systematics?
        if(inconfig.m_num_variation_type_norm>0 || inconfig.m_num_variation_type_norm_to_covariance>0){
            for(auto& allow_sys : inconfig.m_mcgen_variation_type_map){
                if(allow_sys.second=="norm"){
                    map_systematic_num_universe[allow_sys.first] = 7;
                }else if(allow_sys.second=="norm_to_covariance"){
                    map_systematic_num_universe[allow_sys.first] = 0;
                }
            }
        }

        //Do we have any external systeatics?
        if(inconfig.m_num_variation_type_external_covariance>0 || inconfig.m_num_variation_type_external_covariance_to_spline>0){
            for(auto& allow_sys : inconfig.m_mcgen_variation_type_map){
                if(allow_sys.second=="external_covariance" || allow_sys.second=="external_covariance_to_spline"){
                    map_systematic_num_universe[allow_sys.first] = -1;
                }
            }
        }
        
        //Do we have any external systeatics?
        if(inconfig.m_num_variation_type_hist1d>0 || inconfig.m_num_variation_type_hist2d>0){
            for(auto& allow_sys : inconfig.m_mcgen_variation_type_map){
                if(allow_sys.second=="hist1d" || allow_sys.second=="hist2d"){
                    map_systematic_num_universe[allow_sys.first] = 1;
                }
            }
        }
        
        //Do we have any external systeatics?
        if(inconfig.m_num_variation_type_explicit>0){
            for(auto& allow_sys : inconfig.m_mcgen_variation_type_map){
                if(allow_sys.second=="explicit_spline"){
                    bool override_knobs = inconfig.m_mcgen_variation_knobval_override.find(allow_sys.first) != inconfig.m_mcgen_variation_knobval_override.end();
                    map_systematic_num_universe[allow_sys.first] = override_knobs ? inconfig.m_mcgen_variation_knobval_override.at(allow_sys.first).size() : 7;
                }
            }
        }

        size_t total_num_systematics = map_systematic_num_universe.size();
        log<LOG_INFO>(L"%1% || Found %2% unique variations") % __func__ % total_num_systematics;
        for(auto& sys_pair : map_systematic_num_universe){
            log<LOG_DEBUG>(L"%1% || Variation: %2% --> %3% universes") % __func__ % sys_pair.first.c_str() % sys_pair.second;
        }

        //syst_vector.emplace_back();// One for each variable. "reco" is no longer special
        for(size_t io = 0; io < inconfig.m_num_variables; ++io)
            syst_vector.emplace_back();

        //constuct object for each systematic variation, and grab weight maps
        log<LOG_INFO>(L"%1% || Now start to grab related weightmaps") % __func__;
        for(auto& sys_pair : map_systematic_num_universe){

            const std::string& sys_name = sys_pair.first;
            std::string sys_weight_formula = "1";
            std::string sys_mode = inconfig.m_mcgen_variation_type_map.at(sys_name);
            int binningindex = inconfig.m_mcgen_variation_binning_map.at(sys_name);
        
            for (size_t iv = 0; iv < syst_vector.size(); ++iv) {
                auto& sv = syst_vector[iv];
                sv.emplace_back(sys_name, sys_pair.second);

                log<LOG_INFO>(L"%1% || found mode %2% for systematic %3%: ") % __func__ % sys_mode.c_str() % sys_name.c_str();
                if(sys_weight_formula != "1" || sys_mode !=""){
                    sv.back().SetWeightFormula(sys_weight_formula);
                    sv.back().SetMode(sys_mode);
                }
                // Check if scale is set for this systematic
                if(inconfig.m_mcgen_variation_scale.find(sys_name) != inconfig.m_mcgen_variation_scale.end()) {
                    sv.back().scale = inconfig.m_mcgen_variation_scale.at(sys_name);
                    log<LOG_INFO>(L"%1% || Setting scale=%2% for systematic %3%") % __func__ % sv.back().scale % sys_name.c_str();
                }
                // Check if inflate is set for this systematic
                if(inconfig.m_mcgen_variation_inflate.find(sys_name) != inconfig.m_mcgen_variation_inflate.end()) {
                    sv.back().inflate = inconfig.m_mcgen_variation_inflate.at(sys_name);
                    log<LOG_INFO>(L"%1% || Setting inflate=%2% for systematic %3%") % __func__ % sv.back().inflate % sys_name.c_str();
                }
                if(sys_mode == "spline" || sys_mode == "spline_to_covariance" || sys_mode == "explicit_spline") {
                    bool override_knobs = inconfig.m_mcgen_variation_knobval_override.find(sys_name) != inconfig.m_mcgen_variation_knobval_override.end();
                    if(!override_knobs && map_systematic_knob_vals.find(sys_name) == map_systematic_knob_vals.end()) {
                        log<LOG_WARNING>(L"%1% || Expected %2% to have knob vals associated with it, but couldn't find any. Will use -3 to +3 as default.") % __func__ % sys_name.c_str();
                        map_systematic_knob_vals[sys_name] = {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
                        if(sys_mode == "explicit_spline" && inconfig.m_mcgen_explicit_weights.at(sys_name).size() != 7) {
                            log<LOG_ERROR>(L"%1% || Expected exactly 7 weights if no knob values are given for explicit_spline type systematic. Found %2%. Aborting.")
                                % __func__ % inconfig.m_mcgen_explicit_weights.at(sys_name).size();
                        }
                    }
                    sv.back().knob_index = override_knobs ? inconfig.m_mcgen_variation_knobval_override.at(sys_name) : map_systematic_knob_vals[sys_name];
                    sv.back().knobval = sv.back().knob_index;
                    std::sort(sv.back().knobval.begin(), sv.back().knobval.end());
                    sv.back().binning = binningindex;
                    // Check if force_0_cv is set for this systematic
                    if(inconfig.m_mcgen_variation_force_0_cv.find(sys_name) != inconfig.m_mcgen_variation_force_0_cv.end()) {
                        sv.back().force_0_cv = inconfig.m_mcgen_variation_force_0_cv.at(sys_name);
                        log<LOG_INFO>(L"%1% || Setting force_0_cv=true for systematic %2%") % __func__ % sys_name.c_str();
                    }
                    // Check if include_only_weights is set for this systematic
                    if(inconfig.m_mcgen_variation_include_only_weights.find(sys_name) != inconfig.m_mcgen_variation_include_only_weights.end()) {
                        sv.back().include_only_weights = inconfig.m_mcgen_variation_include_only_weights.at(sys_name);
                        log<LOG_INFO>(L"%1% || Setting include_only_weights for systematic %2% (%3% entries)") % __func__ % sys_name.c_str() % sv.back().include_only_weights.size();
                    }
                    // Check if restrict is set for this systematic
                    if(inconfig.m_mcgen_variation_restrict.find(sys_name) != inconfig.m_mcgen_variation_restrict.end()) {
                        sv.back().has_restrict = true;
                        sv.back().restrict_lo = inconfig.m_mcgen_variation_restrict.at(sys_name).first;
                        sv.back().restrict_hi = inconfig.m_mcgen_variation_restrict.at(sys_name).second;
                        log<LOG_INFO>(L"%1% || Setting restrict=[%2%, %3%] for systematic %4%") % __func__ % sv.back().restrict_lo % sv.back().restrict_hi % sys_name.c_str();
                    }
                }
                if(sys_mode == "covariance_to_spline"){
                    sv.back().binning = binningindex;
                    auto it_nk = inconfig.m_mcgen_variation_num_decomp_knobs.find(sys_name);
                    if(it_nk != inconfig.m_mcgen_variation_num_decomp_knobs.end()) {
                        sv.back().num_decomp_knobs = it_nk->second;
                        log<LOG_INFO>(L"%1% || Setting num_decomp_knobs=%2% for systematic %3%") % __func__ % sv.back().num_decomp_knobs % sys_name.c_str();
                    }
                    auto it_rc = inconfig.m_mcgen_variation_include_resid_cov.find(sys_name);
                    if(it_rc != inconfig.m_mcgen_variation_include_resid_cov.end()) {
                        sv.back().include_resid_cov = it_rc->second;
                    }
                    log<LOG_INFO>(L"%1% || Setting include_resid_cov=%2% for systematic %3%") % __func__ % sv.back().include_resid_cov % sys_name.c_str();
                    log<LOG_INFO>(L"%1% || Systematic variation %2% is a match for a covariance_to_spline systematic. Processing as such. ") % __func__ % sys_name.c_str();
                }
                if(sys_mode == "flat"){
                    log<LOG_INFO>(L"%1% || Systematic variation %2% is a match for a flat covariance systematic. Processing a such. ") % __func__ % sys_name.c_str();
                }
                if(sys_mode == "external_covariance"){
                    sv.back().external_filename = inconfig.m_mcgen_variation_external_filename_map.at(sys_name);
                    sv.back().binning = binningindex;
                    log<LOG_INFO>(L"%1% || Systematic variation %2% is a match for an externally loaded covariance systematic. Processing a such. ") % __func__ % sys_name.c_str();
                    log<LOG_INFO>(L"%1% || External filename:  %2%, External Matrix :%3% . Use for variable number %4%") % __func__ % sv.back().external_filename.c_str() % sys_name.c_str() % sv.back().binning;
                }
                if(sys_mode == "external_covariance_to_spline"){
                    sv.back().external_filename = inconfig.m_mcgen_variation_external_filename_map.at(sys_name);
                    sv.back().binning = binningindex;
                    auto it_nk = inconfig.m_mcgen_variation_num_decomp_knobs.find(sys_name);
                    if(it_nk != inconfig.m_mcgen_variation_num_decomp_knobs.end()) {
                        sv.back().num_decomp_knobs = it_nk->second;
                        log<LOG_INFO>(L"%1% || Setting num_decomp_knobs=%2% for systematic %3%") % __func__ % sv.back().num_decomp_knobs % sys_name.c_str();
                    }
                    log<LOG_INFO>(L"%1% || Systematic variation %2% is a match for an externally loaded covariance-to-spline systematic. Processing a such. ") % __func__ % sys_name.c_str();
                    log<LOG_INFO>(L"%1% || External filename:  %2%, External Matrix :%3% . Use for variable number %4%") % __func__ % sv.back().external_filename.c_str() % sys_name.c_str() % sv.back().binning;
                }
                if(sys_mode == "hist1d" || sys_mode == "hist2d") {
                    map_systematic_knob_vals[sys_name] = {1.0f};
                    sv.back().knob_index = map_systematic_knob_vals[sys_name];
                    sv.back().knobval = sv.back().knob_index;
                    sv.back().binning = binningindex;
                }
                if(sys_mode == "norm" || sys_mode == "norm_to_covariance") {
                    log<LOG_INFO>(L"%1% || Systematic variation %2% is a match for a normalization systematic. Processing as such. ") % __func__ % sys_name.c_str();
                    map_systematic_knob_vals[sys_name] = {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
                    sv.back().knob_index = map_systematic_knob_vals[sys_name];
                    sv.back().knobval = sv.back().knob_index;
                    std::sort(sv.back().knobval.begin(), sv.back().knobval.end());

                    size_t colonPos = sys_name.find(':');
                    if (colonPos == std::string::npos) {
                        log<LOG_ERROR>(L"%1% || ERROR, you asked for a norm spline systematic but its not in NAME:percentate format %2%") % __func__  % sys_name.c_str();
                        exit(EXIT_FAILURE);
                    }

                    std::string wild = sys_name.substr(0, colonPos);
                    std::string sflat_percent  = sys_name.substr(colonPos + 1);
                    float flat_percent = std::stof(sflat_percent);

                    // norm_to_covariance should be able to handle >= 0.33
                    if(sys_mode == "norm" && flat_percent >= 0.33333){
                        log<LOG_ERROR>(L"%1% || Currently norm takes +-3,2,1 sigma. Greater than 33.33% norm error isn't allowed. You entered %2%. Dont.  ") % __func__  %  flat_percent;
                        exit(EXIT_FAILURE);
                    }

                    log<LOG_INFO>(L"%1% || Wildcard %2% (and percent %3%) which matches: ") % __func__  % wild.c_str() % flat_percent;
                    std::vector<std::string> flatnames;
                    for(auto & name: inconfig.m_fullnames){
                        if(name.find(wild)!=std::string::npos){
                            flatnames.push_back(name); 
                        }
                    }
                    log<LOG_INFO>(L"%1% || %2% . ") % __func__  % flatnames;

                    std::vector<int> flatbins;
                    for(auto &name: flatnames){
                        size_t is = inconfig.GetSubchannelIndex(name);     
                        size_t ic = inconfig.GetLocalChannelIndexFromGlobalSubchannelIndex(is);     

                        size_t start = inconfig.GetGlobalVariableBinStart(is,iv); 
                        for(size_t b = 0; b < inconfig.m_channel_variable_bins[ic][iv].NBins(); b++) {
                            flatbins.push_back((int)(start+b));
                        }
                    }
                    log<LOG_INFO>(L"%1% || and fills bins  %2%  .") % __func__  %  flatbins;

                    sv.back().norm_bins=flatbins;
                    sv.back().norm_value = flat_percent;
                    sv.back().binning = binningindex;

                }

                for(size_t i = 0 ; i != inconfig.m_mcgen_weightmaps_patterns.size(); ++i){
                    if (inconfig.m_mcgen_weightmaps_uses[i] && sys_name.find(inconfig.m_mcgen_weightmaps_patterns[i]) != std::string::npos) {
                        sys_weight_formula = sys_weight_formula + "*(" + inconfig.m_mcgen_weightmaps_formulas[i]+")";
                        sys_mode           = inconfig.m_mcgen_weightmaps_mode[i];


                        log<LOG_INFO>(L"%1% || Systematic variation %2% is a match for pattern %3%") % __func__ % sys_name.c_str() % inconfig.m_mcgen_weightmaps_patterns[i].c_str();
                        log<LOG_INFO>(L"%1% || Corresponding weight is : %2%") % __func__ % inconfig.m_mcgen_weightmaps_formulas[i].c_str();
                        log<LOG_INFO>(L"%1% || Corresponding mode is : %2%") % __func__ % inconfig.m_mcgen_weightmaps_mode[i].c_str();
                    }
                }
            }
        }


        //sanity check 
        for(auto& sv : syst_vector){
            for(auto &s: sv) {
                s.SanityCheck();
                s.SetHash(inconfig.hash);
            }
        }


        //create 2D multi-universe spec.
        for(size_t i = 0; i < syst_vector.size(); ++i){
            auto &sv = syst_vector[i];
            for(auto &s: sv) {
                if(s.mode=="flat" || s.mode=="external_covariance" || s.mode=="external_covariance_to_spline")
                    continue;
                //Get these binnings right
                s.CreateSpecs( (s.mode == "covariance" ) ? inconfig.m_num_variable_bins_total[i] : inconfig.m_num_variable_bins_total[s.binning]);
                // use the binnign from the variable if covariance, otherise use the binning fdefined for the spline
            }
        }


        for(size_t i = 0; i < inconfig.m_num_variables; ++i) {
            inprop.variable_mc_stat_err.push_back(Eigen::VectorXf::Constant(inconfig.m_num_variable_bins_total[i], 0));
        }

        //setup and initilizte the hists
        inprop.variable_hist_storage.init(inconfig.m_num_variables);
        for(size_t i = 0; i < inconfig.m_num_variables; ++i) {
            for(size_t j = i; j < inconfig.m_num_variables; ++j) {
                log<LOG_DEBUG>(L"%1% || Init Storage hists (i,j): (%2%,%3%) of size (%4%,%5%).") % __func__ % i % j %  inconfig.m_num_variable_bins_total[i] % inconfig.m_num_variable_bins_total[j]; 
                inprop.variable_hist_storage.set(i,j)= Eigen::MatrixXf::Constant(inconfig.m_num_variable_bins_total[i],inconfig.m_num_variable_bins_total[j],0.0);
            }
        }

        // setup variable storage
        for(size_t i = 0; i < inconfig.m_num_variables; ++i) {
            inprop.variable_values.push_back({});
            inprop.variable_bin_indices.push_back({});
        }

        for(size_t i = 0; i < inconfig.m_num_variables; ++i) {

            inprop.variable_midbin.emplace_back(Eigen::VectorXf::Constant(inconfig.m_num_variable_bins_total[i], 0));
            size_t LE_bin = 0;
            for(size_t im = 0; im < inconfig.m_num_modes; im++){
                for(size_t id =0; id < inconfig.m_num_detectors; id++){
                    for(size_t ic = 0; ic < inconfig.m_num_channels; ic++){
                        std::vector<float> edges = inconfig.m_channel_variable_bins.at(ic)[i].Edges(); // TODO: handle N-dim?
                        for(size_t sc = 0; sc < inconfig.m_num_subchannels.at(ic); sc++){
                            for(size_t j = 0; j < edges.size() - 1; ++j){
                                inprop.variable_midbin.back()(LE_bin++) = (edges[j+1] + edges[j])/2;
                            }
                        }
                    }
                }
            }
        }

        time_t start_time = time(nullptr), time_stamp = time(nullptr);
        log<LOG_INFO>(L"%1% || Start reading the files..") % __func__;
        for(int fid=0; fid < num_files; ++fid) {
            const auto& fn = inconfig.m_mcgen_file_name.at(fid);
            long int nevents = std::min(inconfig.m_mcgen_maxevents[fid], (long int)(nentries[fid] * inconfig.m_mcgen_partial_load_frac[fid]));
            log<LOG_DEBUG>(L"%1% || Start @files: %2% which has %3% total events, loading fraction %4% (%5% events)") % __func__ % fn.c_str() % nentries[fid] % inconfig.m_mcgen_partial_load_frac[fid] % nevents;
            if(inconfig.m_mcgen_partial_load_frac[fid] != 1.0)
                log<LOG_INFO>(L"%1% || File %2% -- loading %3%%% of events (%4% / %5%)") % __func__ % fid % (inconfig.m_mcgen_partial_load_frac[fid] * 100) % nevents % nentries[fid];


            // set up systematic weight formula
            std::vector<float> sys_weight_value(total_num_systematics, 1.0);
            std::vector<std::unique_ptr<TTreeFormula>> sys_weight_formula;
            for(const auto& sv : syst_vector){
                for(const auto &s: sv) {
                    if(s.HasWeightFormula())
                        sys_weight_formula.push_back(std::make_unique<TTreeFormula>(("weightMapsFormulas_"+std::to_string(fid)+"_"+ s.GetSysName()).c_str(), s.GetWeightFormula().c_str(),chains[fid]));
                    else
                        sys_weight_formula.push_back(nullptr);
                }
            }
            log<LOG_DEBUG>(L"%1% || Finished setting up systematic weight formula") % __func__;

            // Set up matching variable formulas (used for DetVar CV/variation event alignment)
            const bool has_matching_vars = !inconfig.m_detvar_matching_vars.empty();
            std::vector<std::unique_ptr<TTreeFormula>> matching_var_formulas;
            if(has_matching_vars) {
                if(inprop.matching_var_values.empty()) {
                    inprop.matching_var_values.resize(inconfig.m_detvar_matching_vars.size());
                    inprop.has_matching_vars = true;
                }
                for(size_t iv = 0; iv < inconfig.m_detvar_matching_vars.size(); ++iv) {
                    const std::string& vname = inconfig.m_detvar_matching_vars[iv];
                    matching_var_formulas.push_back(std::make_unique<TTreeFormula>(
                        ("matchingVar_" + std::to_string(fid) + "_" + vname).c_str(),
                        vname.c_str(), chains[fid]));
                    log<LOG_INFO>(L"%1% || Set up matching var formula '%2%' for file %3%") % __func__ % vname.c_str() % fid;
                }
            }

            // grab the subchannel index
            int num_branch = inconfig.m_branch_variables[fid].size();
            auto& branches = inconfig.m_branch_variables[fid];
            std::vector<int> subchannel_index(num_branch, 0);
            log<LOG_DEBUG>(L"%1% || This file includes %2% branch/subchannels") % __func__ % num_branch;
            for(int ib = 0; ib != num_branch; ++ib) {

                const std::string& subchannel_name = inconfig.m_branch_variables[fid][ib]->associated_hist;
                subchannel_index[ib] = inconfig.GetSubchannelIndex(subchannel_name);
                log<LOG_DEBUG>(L"%1% || Subchannel: %2% maps to index: %3%") % __func__ % subchannel_name.c_str() % subchannel_index[ib];
            }


            // Prune unused branches: disable everything, then re-enable only what our
            // formulas and eventweight maps actually need.  This prevents loading large
            // unused friend-tree columns on each GetEntry.
            //
            // NOTE: We extract branch names from the formula expression strings directly
            // rather than via TTreeFormula::GetNcodes()/GetLeaf(), because the TTreeFormula
            // objects created in the first file-setup loop may have stale internal state
            // (ROOT can close/remap files when processing multiple TChains, invalidating
            // leaf pointers stored inside those objects).
            {
                std::set<std::string> needed;

                // Extract branch-name tokens from all formula expression strings.
                // Uses ROOTFormula::ExtractExprTokens (the canonical skip-list lives there)
                // rather than probing TTreeFormula objects, which may have stale internal
                // state from ROOT closing/remapping files across multiple TChains.
                log<LOG_DEBUG>(L"%1% || Branch pruning fid=%2%: extracting tokens from formula strings") % __func__ % fid;
                for(int ib = 0; ib < num_branch; ib++) {
                    for(const auto& expr : branches[ib]->variable_names)
                        ROOTFormula::ExtractExprTokens(expr, needed);
                    for(const auto& wname : inconfig.m_mcgen_weight_names[fid][ib])
                        ROOTFormula::ExtractExprTokens(wname, needed);
                }
                for(const auto& sv : syst_vector)
                    for(const auto& s : sv)
                        if(s.HasWeightFormula())
                            ROOTFormula::ExtractExprTokens(s.GetWeightFormula(), needed);
                for(const auto& vname : inconfig.m_detvar_matching_vars)
                    ROOTFormula::ExtractExprTokens(vname, needed);

                // Branches bound via SetBranchAddress (eventweight maps)
                for(const auto& [name, _] : f_event_weights[fid][0]) needed.insert(name);
                for(const auto& [name, _] : f_knob_vals[fid][0])     needed.insert(name);

                log<LOG_DEBUG>(L"%1% || Branch pruning fid=%2%: needed set has %3% entries, calling SetBranchStatus(*,0)") % __func__ % fid % needed.size();
                chains[fid]->SetBranchStatus("*", 0);
                for(const auto& name : needed) {
                    UInt_t found = 0;
                    chains[fid]->SetBranchStatus(name.c_str(), 1, &found);
                }
                log<LOG_INFO>(L"%1% || Branch pruning fid=%2%: keeping %3% branches") % __func__ % fid % needed.size();
            }

            chains[fid]->SetCacheSize(100000000); // 100 MB read-ahead cache
            chains[fid]->AddBranchToCache("*", kTRUE); // cache all active branches

            // loop over all entries
            size_t to_print = nevents > 5 ? nevents / 5 : 1;
            if(to_print>50000)to_print=50000;
            int currentTreeNumber = -1;
            bool file_name_logged = false;

            for(long int i=0; i < nevents; ++i) {
                if(i%to_print==0){
                    time_t time_passed = time(nullptr) - time_stamp;
                    if(!file_name_logged){
                        log<LOG_INFO>(L"%1% || File %2% (%6%) -- uni : %3% / %4%  took %5% seconds") % __func__ % fid % i % nevents % time_passed % fn.c_str();
                        file_name_logged = true;
                    }
                    else {
                        log<LOG_INFO>(L"%1% || File %2% -- uni : %3% / %4%  took %5% seconds") % __func__ % fid % i % nevents % time_passed;
                    }
                    time_stamp = time(nullptr);
                }
                chains[fid]->GetEntry(i);

                // update the branches to the current event
                for(int ib = 0; ib != num_branch; ++ib) {
                    for(auto &wf: branches[ib]->branch_weight_formulas) {
                        wf->LoadEvent(i);
                    }
                    for(auto &b: branches[ib]->branch_variable_formulas) {
                        b->LoadEvent(i);
                    }
                }

                // Refresh the systematics loaders
                if (chains[fid]->GetTreeNumber() != currentTreeNumber) {
                    currentTreeNumber = chains[fid]->GetTreeNumber();

                   for(size_t is = 0; is != total_num_systematics; ++is){
                        if(syst_vector[0][is].HasWeightFormula()){
                            sys_weight_formula[is]->UpdateFormulaLeaves();
                            sys_weight_formula[is]->GetNdata();
                        }
                    }//end sys
                    for(auto& mf : matching_var_formulas) {
                        mf->UpdateFormulaLeaves();
                        mf->GetNdata();
                    }
                    TObjArray* tbranches = chains[fid]->GetListOfBranches();
                    for (int i = 0; i < tbranches->GetEntries(); i++) {
                        TBranch* branch = (TBranch*)tbranches->At(i);
                        const char* branchName = branch->GetName();
                        bool isActive = chains[fid]->GetBranchStatus(branchName);
                        log<LOG_DEBUG>(L"%1% || BRANCH %2% is %3%") % __func__ % branchName % (isActive ? "active" : "inactive");
                    }
                }

                // grab additional weight for systematics
                for(size_t is = 0; is != total_num_systematics; ++is){
                    if(syst_vector[0][is].HasWeightFormula()){
                        sys_weight_value[is] = sys_weight_formula[is]->EvalInstance();
                    }
                }

                // Evaluate matching var values once per TTree entry (same for all branches)
                std::vector<float> current_matching_vals;
                if(has_matching_vars) {
                    for(auto& mf : matching_var_formulas)
                        current_matching_vals.push_back(mf->EvalInstance());
                }

                //branch loop
                for(int ib = 0; ib != num_branch; ++ib) {
                    const size_t prop_size_before = inprop.NEvent();
                    process_cafana_event(inconfig, branches[ib], f_event_weights[fid][0], inconfig.m_mcgen_pot[fid] * inconfig.m_mcgen_partial_load_frac[fid], subchannel_index[ib], syst_vector, sys_weight_value, inprop);
                    // Store matching vars only if process_cafana_event actually added an entry
                    // (it skips zero-weight events without pushing to added_weights).
                    if(has_matching_vars && inprop.NEvent() > prop_size_before) {
                        for(size_t iv = 0; iv < current_matching_vals.size(); ++iv)
                            inprop.matching_var_values[iv].push_back(current_matching_vals[iv]);
                    }
                }

            } //end of entry loop

        } //end of file loop

        //ensure hash is correctly assigned
        inprop.hash=inconfig.hash;
        // Convert the accumulated per-bin sum of squared weights into the sqrt of the
        // Kish effective count, sqrt(N_eff) = Sum(w)/sqrt(Sum(w^2)), using the CV Sum(w)
        // from the pre-binned hist diagonal. PROsyst's square().inverse() then gives the
        // exact fractional MC-stat variance Sum(w^2)/Sum(w)^2 for arbitrary event weights
        // (reduces to 1/N when all weights in a bin are equal).
        for(size_t i = 0; i < inconfig.m_num_variables; ++i) {
            Eigen::ArrayXf sumw  = inprop.variable_hist_storage(i,i).diagonal().array();
            Eigen::ArrayXf sumw2 = inprop.variable_mc_stat_err[i].array();
            inprop.variable_mc_stat_err[i] = (sumw2 > 0).select(sumw / sumw2.sqrt(), 0.0f).matrix();
        }


        time_t time_took = time(nullptr) - start_time;
        log<LOG_INFO>(L"%1% || Finish reading files, it took %2% seconds..") % __func__ % time_took;
        log<LOG_INFO>(L"%1% || DONE") %__func__ ;

        // Cleanup TChain objects to avoid ROOT/Cling JIT crash during global cleanup
        log<LOG_DEBUG>(L"%1% || Cleaning up TChain objects...") % __func__;
        for(int fid = 0; fid < num_files; ++fid) {
            // Delete friend chains first (they were added to main chain)
            for(auto* friendChain : friendChains[fid]) {
                if(friendChain) {
                    delete friendChain;
                }
            }
            friendChains[fid].clear();
            
            // Delete main chain
            if(chains[fid]) {
                delete chains[fid];
                chains[fid] = nullptr;
            }
        }
        log<LOG_DEBUG>(L"%1% || TChain cleanup complete.") % __func__;

        return 0;
    }

    std::vector<PROdata> CreatePROdata(const PROconfig& inconfig){
        log<LOG_INFO>(L"%1% || Start generating data spectrum") % __func__ ;

        int num_files = inconfig.m_num_mcgen_files;
        log<LOG_DEBUG>(L"%1% || Starting to read file and build spectrum!") % __func__ ;
        log<LOG_DEBUG>(L"%1% || Using a total of %2% individual files") % __func__  % num_files;

        std::vector<long int> nentries(num_files,0);
        std::vector<float> pot_scale(num_files, 1.0);
        std::vector<TChain*> chains(num_files, nullptr);
        std::vector<std::vector<TChain*>> friendChains(num_files);

        for(int fid=0; fid < num_files; ++fid) {
            const auto& fn = inconfig.m_mcgen_file_name.at(fid);

            std::vector<std::string> filesForChain;

            if (fn.find(".root") != std::string::npos) {
                log<LOG_INFO>(L"%1% || Starting a (single) TChain, loading file %2%") % __func__  % fn.c_str();
                filesForChain.push_back(fn);
            }else{
                log<LOG_INFO>(L"%1% || Starting a TChain, loading from filelist %2%") % __func__  % fn.c_str();
                std::ifstream infile(fn.c_str());
                if (!infile) {
                    log<LOG_ERROR>(L"%1% || Failed to open input filelist %2%") % __func__  % fn.c_str();
                    exit(EXIT_FAILURE);
                }

                std::string line;
                while (std::getline(infile, line)) {
                    log<LOG_INFO>(L"%1% || Loading file %2% into TChain") %__func__ % line.c_str();
                    filesForChain.push_back(line);
                }

                infile.close();
            }

            chains[fid] = new TChain(inconfig.m_mcgen_tree_name.at(fid).c_str());
            if (inconfig.m_mcgen_numfriends[fid] > 0) {
                friendChains[fid].resize(inconfig.m_mcgen_numfriends[fid], nullptr);
            }

            for(auto &file: filesForChain){
                chains[fid]->Add(file.c_str());

                if (inconfig.m_mcgen_numfriends[fid] > 0) {
                    auto mcgen_file_friend_treename_iter = inconfig.m_mcgen_file_friend_treename_map.find(fn);
                    if (mcgen_file_friend_treename_iter != inconfig.m_mcgen_file_friend_treename_map.end()) {
                        auto mcgen_file_friend_iter = inconfig.m_mcgen_file_friend_map.find(fn);
                        if (mcgen_file_friend_iter == inconfig.m_mcgen_file_friend_map.end()) {
                            log<LOG_ERROR>(L"%1% || Friend TTree provided but no friend file??") % __func__;
                            log<LOG_ERROR>(L"Terminating.");
                            exit(EXIT_FAILURE);
                        }

                        for (size_t k = 0; k < mcgen_file_friend_treename_iter->second.size(); k++) {
                            std::string treefriendname = mcgen_file_friend_treename_iter->second.at(k);

                            if (!friendChains[fid][k]) {
                                friendChains[fid][k] = new TChain(treefriendname.c_str());
                            }

                            log<LOG_DEBUG>(L"%1% || Adding friend tree %2% from file %3%") % __func__ % treefriendname.c_str() % file.c_str();
                            friendChains[fid][k]->Add(file.c_str());
                        }
                    }
                }
            }

            // Add friend chains to main chain
            if (inconfig.m_mcgen_numfriends[fid] > 0) {
                for (size_t k = 0; k < friendChains[fid].size(); k++) {
                    log<LOG_DEBUG>(L"%1% || Adding friend chain %2% to main chain %3%") % __func__ % k % fid;
                    chains[fid]->AddFriend(friendChains[fid][k]);
                }
            }

            nentries[fid] = (long int)chains[fid]->GetEntries();
            log<LOG_INFO>(L"%1% || Total Entries: %2%") % __func__ %  nentries[fid];

            // grab branches 
            int num_branch = inconfig.m_branch_variables[fid].size();
            for(int ib = 0; ib != num_branch; ++ib) {

                std::shared_ptr<BranchVariable> branch_variable = inconfig.m_branch_variables[fid][ib];

                //calculate POT scale factor                  
                const std::string& subchannel_name = inconfig.m_branch_variables[fid][ib]->associated_hist;
                int subchannel_index = inconfig.GetSubchannelIndex(subchannel_name);
                int channel_group = subchannel_index / std::accumulate(inconfig.m_num_subchannels.begin(), inconfig.m_num_subchannels.end(), 0);
                int det = channel_group % inconfig.m_num_detectors;

                float spec_pot = inconfig.m_det_pot[det];
                //log<LOG_INFO>(L"%1% || Spectrum will be generated with %2% POT") % __func__ % spec_pot;
                
                if(inconfig.m_mcgen_pot.at(fid) != -1){
                    pot_scale[fid] = spec_pot/inconfig.m_mcgen_pot.at(fid);
                }
                pot_scale[fid] *= inconfig.m_mcgen_scale[fid];
                pot_scale[fid] /= inconfig.m_mcgen_partial_load_frac[fid];
                log<LOG_INFO>(L"%1% || Branch POT: %2%, additional scale: %3%") % __func__ %  inconfig.m_mcgen_pot.at(fid) % inconfig.m_mcgen_scale[fid];
                log<LOG_INFO>(L"%1% || POT scale factor: %2%") % __func__ %  pot_scale[fid];


                //quick check that this branch associated subchannel is in the known chanels;
                int is_valid_subchannel = 0;
                for(const auto &name: inconfig.m_fullnames){
                    if(branch_variable->associated_hist==name){
                        log<LOG_DEBUG>(L"%1% || Found a valid subchannel for this branch %2%") % __func__  % name.c_str();
                        ++is_valid_subchannel;
                    }
                }
                if(is_valid_subchannel==0){
                    log<LOG_ERROR>(L"%1% || This branch did not match one defined in the .xml : %2%") % __func__ % inconfig.m_xmlname.c_str();
                    log<LOG_ERROR>(L"%1% || There is probably a typo somehwhere in xml!") % __func__;
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);

                }else if(is_valid_subchannel>1){
                    log<LOG_ERROR>(L"%1% || This branch matched more than 1 subchannel!: %2%") % __func__ %  branch_variable->associated_hist.c_str();
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);
                }


                int other_count = 0;
                for(const auto &name: branch_variable->variable_names) {
                    branch_variable->branch_variable_formulas.push_back(std::make_shared<ROOTFormula>(
                        "branch_variabler_form_"+std::to_string(fid) +"_" + std::to_string(ib)+"_"+std::to_string(other_count),
                        name, chains[fid]));
                    other_count++;
                }

                //grab weight formulas (weight_1, weight_2, ...)
                for(int wi = 0; wi < inconfig.m_mcgen_num_weights[fid][ib]; ++wi){
                    const std::string& wname = inconfig.m_mcgen_weight_names[fid][ib][wi];
                    log<LOG_INFO>(L"%1% || Setting up weight_%2% for this branch: %3%") % __func__ % (wi+1) % wname.c_str();
                    branch_variable->branch_weight_formulas.push_back(std::make_shared<ROOTFormula>(
                        "branch_weight_"+std::to_string(wi+1)+"_"+std::to_string(fid)+"_" + std::to_string(ib),
                        wname, chains[fid]));
                }

            } //end of branch loop
        } // end fid

        time_t start_time = time(nullptr);
        std::vector<PROdata> data;
        for(size_t io = 0; io < inconfig.m_num_variables; ++io)
            data.emplace_back(inconfig.m_num_variable_bins_total[io]);
        log<LOG_INFO>(L"%1% || Start reading the files..") % __func__;
        for(int fid=0; fid < num_files; ++fid) {
            const auto& fn = inconfig.m_mcgen_file_name.at(fid);
            long int nevents = std::min(inconfig.m_mcgen_maxevents[fid], (long int)(nentries[fid] * inconfig.m_mcgen_partial_load_frac[fid]));
            log<LOG_DEBUG>(L"%1% || Start @files: %2% which has %3% total events, loading fraction %4% (%5% events)") % __func__ % fn.c_str() % nentries[fid] % inconfig.m_mcgen_partial_load_frac[fid] % nevents;
            if(inconfig.m_mcgen_partial_load_frac[fid] != 1.0)
                log<LOG_INFO>(L"%1% || File %2% -- loading %3%%% of events (%4% / %5%)") % __func__ % fid % (inconfig.m_mcgen_partial_load_frac[fid] * 100) % nevents % nentries[fid];

            // grab the subchannel index
            int num_branch = inconfig.m_branch_variables[fid].size();
            auto& branches = inconfig.m_branch_variables[fid];
            std::vector<std::string> branch_fullname;
            branch_fullname.reserve(num_branch);
            log<LOG_INFO>(L"%1% || This file includes %2% branch/channels") % __func__ % num_branch;
            for(int ib = 0; ib != num_branch; ++ib) {
                const std::string& hist_name = inconfig.m_branch_variables[fid][ib]->associated_hist;
                branch_fullname.push_back(hist_name);
            }

            // Prune unused branches: disable everything, then re-enable only what our
            // formulas actually need.
            //
            // NOTE: extract tokens from formula strings directly rather than via
            // TTreeFormula::GetNcodes()/GetLeaf(), because TTreeFormula objects created
            // during the first fid loop may have stale internal state (ROOT can
            // close/remap files when processing multiple TChains, invalidating leaf
            // pointers stored inside those objects).
            {
                std::set<std::string> needed;
                log<LOG_DEBUG>(L"%1% || Branch pruning fid=%2%: extracting tokens from formula strings") % __func__ % fid;
                for(int ib = 0; ib < num_branch; ib++) {
                    for(const auto& expr : branches[ib]->variable_names)
                        ROOTFormula::ExtractExprTokens(expr, needed);
                    for(const auto& wname : inconfig.m_mcgen_weight_names[fid][ib])
                        ROOTFormula::ExtractExprTokens(wname, needed);
                }
                log<LOG_DEBUG>(L"%1% || Branch pruning fid=%2%: needed set has %3% entries, calling SetBranchStatus(*,0)") % __func__ % fid % needed.size();
                chains[fid]->SetBranchStatus("*", 0);
                for(const auto& name : needed) {
                    UInt_t found = 0;
                    chains[fid]->SetBranchStatus(name.c_str(), 1, &found);
                }
                log<LOG_INFO>(L"%1% || Branch pruning fid=%2%: keeping %3% branches") % __func__ % fid % needed.size();
            }

            chains[fid]->SetCacheSize(100000000); // 100 MB read-ahead cache
            chains[fid]->AddBranchToCache("*", kTRUE); // cache all active branches

            // loop over all entries
            for(long int i=0; i < nevents; ++i) {
                if(i%1000==0)	log<LOG_INFO>(L"%1% || -- uni : %2% / %3%") % __func__ % i % nevents;
                chains[fid]->GetEntry(i);

                // update the formulas to the current event (handles TChain file transitions)
                for(int ib = 0; ib != num_branch; ++ib) {
                    for(auto &wf: branches[ib]->branch_weight_formulas) {
                        wf->LoadEvent(i);
                    }
                    for(auto &b: branches[ib]->branch_variable_formulas) {
                        b->LoadEvent(i);
                    }
                }

                //branch loop
                for(int ib = 0; ib != num_branch; ++ib) {
                    std::vector<BranchVariable::Value> vars = branches[ib]->GetVariables();
                    float additional_weight = branches[ib]->GetTotalWeight();
                    additional_weight *= pot_scale[fid];

                    if(additional_weight == 0) //skip on event failing cuts
                        continue;

                    std::vector<int> var_bin_indices;
                    for(size_t i = 0; i < vars.size(); ++i) {
                        var_bin_indices.push_back(FindGlobalVariableBin(inconfig, vars[i], branch_fullname[ib], i));
                    }


                    for(size_t io = 0; io < inconfig.m_num_variables; ++io)
                        if(var_bin_indices[io] >= 0)
                            data[io].Fill(var_bin_indices[io], additional_weight);//from io+1
                    
                }  //end of branch loop
            } //end of entry loop
        } //end of file loop
        time_t time_took = time(nullptr) - start_time;
        log<LOG_INFO>(L"%1% || Generating data spectrum took %2% seconds..") % __func__ % time_took;

        // Cleanup TChain objects
        for(int fid = 0; fid < num_files; ++fid) {
            for(auto* friendChain : friendChains[fid]) {
                delete friendChain;
            }
            if(chains[fid]) {
                delete chains[fid];
                chains[fid] = nullptr;
            }
        }

        return data;
    }


    void process_cafana_event(const PROconfig &inconfig, const std::shared_ptr<BranchVariable>& branch, const std::map<std::string, std::vector<eweight_type>*>& eventweight_map, float mcpot, int subchannel_index, std::vector<std::vector<SystStruct>> &syst_vector, const std::vector<float>& syst_additional_weight, PROpeller& inprop){



        int total_num_sys = syst_vector[0].size(); 
        std::vector<BranchVariable::Value> vars = branch->GetVariables();

        int run_syst = branch->GetIncludeSystematics();

        // Compute individual weight values for potential use with include_only_weights
        int num_weights = branch->NumWeights();
        std::vector<float> weight_vals(num_weights);
        for(int wi = 0; wi < num_weights; ++wi) {
            weight_vals[wi] = branch->GetWeight(wi);
        }

        int channel_group = subchannel_index / std::accumulate(inconfig.m_num_subchannels.begin(), inconfig.m_num_subchannels.end(), 0);
        int det = channel_group % inconfig.m_num_detectors;

        float pot_scale = inconfig.m_det_pot[det] / mcpot;
        float mc_weight = branch->GetTotalWeight() * pot_scale;

        int model_rule = branch->GetModelRule();

        std::vector<int> var_bin_indices;
        for(size_t i = 0; i < vars.size(); ++i) {
            var_bin_indices.push_back(FindGlobalVariableBin(inconfig, vars[i], subchannel_index, i));
            //log<LOG_INFO>(L"%1% || GURP  var %2%  value: %3% bin: %4% : sbindex %5% : weight %6%") % __func__ %   i % vars[i].first() % FindGlobalVariableBin(inconfig, vars[i], subchannel_index, i) % subchannel_index % mc_weight;
        }


        if(mc_weight == 0)
            return;

        inprop.added_weights.push_back(mc_weight);
        inprop.model_rule.push_back((int)model_rule);

        // Save the variables
        for (unsigned i_var = 0; i_var < inprop.NVariable(); i_var++) {
            inprop.variable_bin_indices[i_var].push_back(var_bin_indices[i_var]);
            inprop.variable_values[i_var].push_back(vars[i_var].first());
        }

        for(size_t io = 0; io < inconfig.m_num_variables; ++io) {
            if(var_bin_indices[io] >= 0) {
                inprop.variable_mc_stat_err[io](var_bin_indices[io]) += mc_weight * mc_weight;
                for(size_t jo = io; jo < inconfig.m_num_variables; ++jo) {

                    if(var_bin_indices[jo]<0)continue;
                    inprop.variable_hist_storage.set(io,jo)(var_bin_indices[io], var_bin_indices[jo]) += mc_weight; 
                }
            }

        }

        if(!run_syst) return;

        for(int i = 0; i != total_num_sys; ++i){
            std::vector<SystStruct*> var_syst_objs;
            for(size_t io = 0; io < inconfig.m_num_variables; ++io)
                var_syst_objs.push_back(&syst_vector[io][i]);

            float additional_weight = syst_additional_weight.at(i); // extra per-systematic weights, unrelated to the per-event additional_weight set in the xml file
            auto map_iter = eventweight_map.find(var_syst_objs.front()->GetSysName());
            // The spline/covariance/covariance_to_spline paths below dereference
            // map_iter; a missing weight name (branch typo, absent friend tree)
            // must fail loudly here instead of dereferencing the end iterator.
            const std::string &sys_mode = var_syst_objs.front()->mode;
            const bool needs_weights = (sys_mode == "spline" || sys_mode == "spline_to_covariance" ||
                                        sys_mode == "covariance" || sys_mode == "covariance_to_spline");
            if(needs_weights && map_iter == eventweight_map.end()){
                log<LOG_ERROR>(L"%1% || ERROR: systematic '%2%' (mode %3%) has no entry in the event weight map. "
                               L"Check that the variation name matches a weight branch in the input files.")
                    % __func__ % var_syst_objs.front()->GetSysName().c_str() % sys_mode.c_str();
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }
            int spline_bin = (var_syst_objs.front()->mode == "covariance") ? -1: var_bin_indices[var_syst_objs.front()->binning];

            if(var_syst_objs.front()->mode == "spline" || var_syst_objs.front()->mode == "spline_to_covariance") {
                if(spline_bin < 0) continue;
                for(auto so: var_syst_objs)
                    so->FillCV(spline_bin, mc_weight);

                for(int is = 0; is < var_syst_objs.front()->GetNUniverse(); ++is){
                    size_t u = 0;
                    for(; u < var_syst_objs.front()->knobval.size(); ++u)
                        if(var_syst_objs.front()->knobval[u] == var_syst_objs.front()->knob_index[is]) break;
                    
                    float w = static_cast<float>(map_iter->second->at(is));
                    if(std::isnan(w) || std::isinf(w)) {
                        log<LOG_WARNING>(L"%1% || Encountered a bad weight (%2%) for syst %3%. Setting to 1 instead.")
                            % __func__ % w % map_iter->first.c_str();
                        w = 1;
                    } else if(w > 30) {
                        log<LOG_WARNING>(L"%1% || Encountered a very large weight (%2%) for syst %3%. Setting to 1 instead.")
                            % __func__ % w % map_iter->first.c_str();
                        w = 1;
                    }
                    for(auto so: var_syst_objs){
                        if (!so->include_only_weights.empty()) {
                            // Compute weight using only the included weights (avoids divide-by-zero)
                            float included_weight = 1.0;
                            for(int idx : so->include_only_weights) {
                                int wi = idx - 1; // convert 1-based to 0-based
                                if(wi >= 0 && wi < num_weights) {
                                    included_weight *= weight_vals[wi];
                                }
                            }
                            so->FillUniverse(u, spline_bin, included_weight * pot_scale * additional_weight * w);
                        } else {
                            so->FillUniverse(u, spline_bin, mc_weight * additional_weight * w);
                        }
                    }
                }

                continue;

            }else if(var_syst_objs.front()->mode == "covariance"){

                for(size_t io = 0; io < inconfig.m_num_variables; ++io) {
                    if(var_bin_indices[io] >= 0){

                        var_syst_objs[io]->FillCV(var_bin_indices[io], mc_weight);
                    }
                }
                for(int iuni = 0; iuni < var_syst_objs.front()->GetNUniverse(); ++iuni){
                    float raw_weight = static_cast<float>(map_iter->second->at(iuni));
                    // Same non-finite guard the spline path applies at its w:
                    // one NaN/inf universe weight would silently NaN the whole
                    // covariance (later zeroed by toFiniteMatrix, hiding the
                    // bad input). Warn so the input problem is visible.
                    if(std::isnan(raw_weight) || std::isinf(raw_weight)){
                        log<LOG_WARNING>(L"%1% || Non-finite universe weight %2% for covariance systematic '%3%' universe %4%; using 1.")
                            % __func__ % raw_weight % var_syst_objs.front()->GetSysName().c_str() % iuni;
                        raw_weight = 1;
                    }
                    float scaled_weight = raw_weight * var_syst_objs.front()->scale; // apply scale factor (default 1.0)
                    float sys_wei = run_syst ? additional_weight * scaled_weight :  1.0;
                    if(std::isnan(sys_wei) || std::isinf(sys_wei)) {
                        log<LOG_WARNING>(L"%1% || Encountered a bad weight (%2%) for syst %3%. Setting to 1 instead.")
                            % __func__ % sys_wei % map_iter->first.c_str();
                        sys_wei = 1;
                    } else if(sys_wei > 30) {
                        log<LOG_WARNING>(L"%1% || Encountered a very large weight (%2%) for syst %3%. Setting to 1 instead.")
                            % __func__ % sys_wei % map_iter->first.c_str();
                        sys_wei = 1;
                    }
                    for(size_t io = 0; io < inconfig.m_num_variables; ++io) {
                        if(var_bin_indices[io] >= 0){
                            var_syst_objs[io]->FillUniverse(iuni, var_bin_indices[io], mc_weight * sys_wei);
                        }
                    }
                }
            }else if(var_syst_objs.front()->mode == "covariance_to_spline"){
                if(spline_bin < 0) continue;
                for(auto so: var_syst_objs)
                    so->FillCV(spline_bin, mc_weight);
                for(int iuni = 0; iuni < var_syst_objs.front()->GetNUniverse(); ++iuni){
                    float raw_weight = static_cast<float>(map_iter->second->at(iuni));
                    if(std::isnan(raw_weight) || std::isinf(raw_weight)){
                        log<LOG_WARNING>(L"%1% || Non-finite universe weight %2% for covariance_to_spline systematic '%3%' universe %4%; using 1.")
                            % __func__ % raw_weight % var_syst_objs.front()->GetSysName().c_str() % iuni;
                        raw_weight = 1;
                    }
                    float scaled_weight = raw_weight * var_syst_objs.front()->scale;
                    float sys_wei = run_syst ? additional_weight * scaled_weight : 1.0;
                    if(std::isnan(sys_wei) || std::isinf(sys_wei)) {
                        log<LOG_WARNING>(L"%1% || Encountered a bad weight (%2%) for syst %3%. Setting to 1 instead.")
                            % __func__ % sys_wei % map_iter->first.c_str();
                        sys_wei = 1;
                    } else if(sys_wei > 30) {
                        log<LOG_WARNING>(L"%1% || Encountered a very large weight (%2%) for syst %3%. Setting to 1 instead.")
                            % __func__ % sys_wei % map_iter->first.c_str();
                        sys_wei = 1;
                    }
                    for(auto so: var_syst_objs){
                        so->FillUniverse(iuni, spline_bin, mc_weight * sys_wei);
                    }
                }
                continue;
            } else  if( var_syst_objs.front()->mode == "norm") {
                if(spline_bin < 0) continue;
                for(auto so: var_syst_objs)
                    so->FillCV(spline_bin, mc_weight);
                
                for(int is = 0; is < var_syst_objs.front()->GetNUniverse(); ++is){
                    size_t ivar=0;
                    for(auto so: var_syst_objs){
                        float norm_shift_percentage = 0.0;
                        if( std::find(var_syst_objs.front()->norm_bins.begin(), var_syst_objs.front()->norm_bins.end(),var_bin_indices[ivar])!=var_syst_objs.front()->norm_bins.end()){
                            norm_shift_percentage =  var_syst_objs.front()->norm_value;
                       }
                       so->FillUniverse(is, spline_bin, mc_weight * additional_weight * (1+var_syst_objs.front()->knobval[is]*norm_shift_percentage) );
                       ivar++;
                    }
                }
                continue;
            } else if(var_syst_objs.front()->mode == "hist1d") {
                if(spline_bin < 0) continue;
                int var_num = inconfig.m_mcgen_variation_histaxisvars_map.at(var_syst_objs.front()->systname)[0];
                float val = vars[var_num].first();
                TH1 *h = inconfig.m_mcgen_variation_hist1d_map.at(var_syst_objs.front()->systname);
                int bin = h->FindBin(val);
                float wgt = h->GetBinContent(bin);
                if(std::isnan(val) || std::isinf(val)) wgt = 1;
                if(val < h->GetXaxis()->GetXmin() || val > h->GetXaxis()->GetXmax()) wgt = 1;

                // Only filling 1 sigma, so just combine CV and Universe filling
                for(auto so: var_syst_objs) {
                    so->FillCV(spline_bin, mc_weight);
                    so->FillUniverse(0, spline_bin, wgt*mc_weight);
                }
                
            } else if(var_syst_objs.front()->mode == "hist2d") {
                if(spline_bin < 0) continue;
                int xvar_num = inconfig.m_mcgen_variation_histaxisvars_map.at(var_syst_objs.front()->systname)[0];
                int yvar_num = inconfig.m_mcgen_variation_histaxisvars_map.at(var_syst_objs.front()->systname)[1];
                float xval = vars[xvar_num].first();
                float yval = vars[yvar_num].first();
                TH2 *h = inconfig.m_mcgen_variation_hist2d_map.at(var_syst_objs.front()->systname);
                int bin = h->FindBin(xval, yval);
                float wgt = h->GetBinContent(bin);
                if(std::isnan(xval) || std::isnan(yval) || std::isinf(xval) || std::isinf(yval)) wgt = 1;
                if(xval < h->GetXaxis()->GetXmin() || xval > h->GetXaxis()->GetXmax()
                    || yval < h->GetYaxis()->GetXmin() || yval > h->GetYaxis()->GetXmax()) wgt = 1;

                // Only filling 1 sigma, so just combine CV and Universe filling
                for(auto so: var_syst_objs) {
                    so->FillCV(spline_bin, mc_weight);
                    so->FillUniverse(0, spline_bin, wgt*mc_weight);
                }
            } else if(var_syst_objs.front()->mode == "explicit_spline") {
                if(spline_bin < 0) continue;
                for(auto so: var_syst_objs)
                    so->FillCV(spline_bin, mc_weight);

                for(int is = 0; is < var_syst_objs.front()->GetNUniverse(); ++is){
                    size_t u = 0;
                    for(; u < var_syst_objs.front()->knobval.size(); ++u)
                        if(var_syst_objs.front()->knobval[u] == var_syst_objs.front()->knob_index[is]) break;
                    
                    float w = inconfig.m_mcgen_explicit_weights.at(var_syst_objs.front()->systname)[is];
                    if(std::isnan(w) || std::isinf(w)) {
                        log<LOG_WARNING>(L"%1% || Encountered a bad weight (%2%) for syst %3%. Setting to 1 instead.")
                            % __func__ % w % map_iter->first.c_str();
                        w = 1;
                    } else if(w > 30) {
                        log<LOG_WARNING>(L"%1% || Encountered a very large weight (%2%) for syst %3%. Setting to 1 instead.")
                            % __func__ % w % map_iter->first.c_str();
                        w = 1;
                    }
                    for(auto so: var_syst_objs){
                        if (!so->include_only_weights.empty()) {
                            // Compute weight using only the included weights (avoids divide-by-zero)
                            float included_weight = 1.0;
                            for(int idx : so->include_only_weights) {
                                int wi = idx - 1; // convert 1-based to 0-based
                                if(wi >= 0 && wi < num_weights) {
                                    included_weight *= weight_vals[wi];
                                }
                            }
                            so->FillUniverse(u, spline_bin, included_weight * pot_scale * additional_weight * w);
                        } else {
                            so->FillUniverse(u, spline_bin, mc_weight * additional_weight * w);
                        }
                    }
                }
            }
        }

    }



}//namespace

