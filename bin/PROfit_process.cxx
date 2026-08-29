#include "PROfit_common.h"

// Unique key for DetVar propeller maps (names can be reused across sections).
std::string DetVarKey(const PROconfig& config, size_t file_index) {
    const auto& dv = config.m_detvar_files[file_index];
    return "sec" + std::to_string(dv.section_index) + "::" + dv.name + "." + std::to_string(dv.knobval);
}

// Build a collision-free composite key for event i_event from its matching_var_values.
// Returns a vector of integer-cast values, one per matching variable (e.g. run, subrun, event).
std::vector<int> DetVarMatchingKey(const PROpeller& prop, size_t i_event) {
    std::vector<int> key;
    key.reserve(prop.matching_var_values.size());
    for(const auto& vals : prop.matching_var_values)
        key.push_back(static_cast<int>(std::round(vals[i_event])));
    return key;
}

// Build PROspec objects for CV and variation using only events whose matching keys appear
// in both propellers. var_idx selects which variable's bin indices to use.
// Returns false (leaving out_cv/out_var unchanged) if either propeller lacks matching vars.
bool BuildDetVarMatchedSpecs(
        const PROpeller& cvprop, const std::map<int, const PROpeller*> &varprop,
        int var_idx, int spec_size,
        PROspec& out_cv, std::map<int, PROspec> &out_var) {

    if(!cvprop.has_matching_vars || 
            !std::all_of(varprop.begin(), varprop.end(), [](const auto &p){ return p.second->has_matching_vars;})) 
        return false;
    if(!std::all_of(varprop.begin(), varprop.end(), 
                [&cvprop](const auto &p){ 
                    return cvprop.matching_var_values.size() == p.second->matching_var_values.size(); 
                }))
        return false;

    // Step 1: build lookup key -> list of event indices for CV.
    // Use std::map with vector<int> keys for guaranteed collision-free RSE matching.
    std::map<std::vector<int>, std::vector<size_t>> cv_key_map;
    for(size_t i = 0; i < cvprop.NEvent(); ++i)
        cv_key_map[DetVarMatchingKey(cvprop, i)].push_back(i);

    // Step 2: find the set of keys present in both CV and variation;
    // track unique var keys to compute CV-only / var-only / overlapping counts.
    std::map<int, std::set<std::vector<int>>> var_key_set;
    std::set<std::vector<int>> common_keys;
    for(const auto &[kv, prop] : varprop) {
        for(size_t j = 0; j < prop->NEvent(); ++j) {
            auto k = DetVarMatchingKey(*prop, j);
            var_key_set[kv].insert(k);
        }
    }
    for(const auto &cv_key : cv_key_map) {
        if(std::all_of(var_key_set.begin(), var_key_set.end(),
                    [&cv_key](const auto &s) {
                        return s.second.count(cv_key.first) > 0;
                    }))
            common_keys.insert(cv_key.first);
    }

    const size_t n_cv_unique    = cv_key_map.size();
    const size_t n_overlapping  = common_keys.size();
    const size_t n_cv_only      = n_cv_unique - n_overlapping;
    // Count total unique keys across all variations
    std::set<std::vector<int>> all_var_keys;
    for(const auto &[kv, ks] : var_key_set) all_var_keys.insert(ks.begin(), ks.end());
    const size_t n_var_unique   = all_var_keys.size();
    const size_t n_var_only     = n_var_unique > n_overlapping ? n_var_unique - n_overlapping : 0;
    log<LOG_INFO>(L"DetVar matching: total CV unique keys: %1%, total var unique keys: %2%")
        % n_cv_unique % n_var_unique;
    log<LOG_INFO>(L"DetVar matching: num CV only: %1%, num var only: %2%, num overlapping: %3%")
        % n_cv_only % n_var_only % n_overlapping;

    // Step 3: fill matched CV spec
    PROspec matched_cv(spec_size);
    size_t n_cv_prop_matched = 0;
    for(size_t i = 0; i < cvprop.NEvent(); ++i) {
        if(!common_keys.count(DetVarMatchingKey(cvprop, i))) continue;
        ++n_cv_prop_matched;
        int bin = cvprop.variable_bin_indices[var_idx][i];
        if(bin >= 0) matched_cv.QuickFill(bin, cvprop.added_weights[i]);
    }

    // Step 4: fill matched var spec
    std::map<int, PROspec> matched_var;
    Eigen::VectorXf n_var_prop_matched = Eigen::VectorXf::Zero(varprop.size());
    Eigen::VectorXf n_var_evt = Eigen::VectorXf::Zero(varprop.size());
    size_t prop_i = 0;
    for(const auto &[kv, prop] : varprop) {
        matched_var[kv] = PROspec(spec_size);
        n_var_evt(prop_i) = prop->NEvent();
        for(size_t j = 0; j < prop->NEvent(); ++j) {
            if(!common_keys.count(DetVarMatchingKey(*prop, j))) continue;
            n_var_prop_matched(prop_i) += 1;
            int bin = prop->variable_bin_indices[var_idx][j];
            if(bin >= 0) matched_var[kv].QuickFill(bin, prop->added_weights[j]);
        }
        prop_i++;
    }
    log<LOG_INFO>(L"DetVar matching: matched propeller events CV: %1%, var: %2% (total propeller events CV: %3%, var: %4%)")
        % n_cv_prop_matched % n_var_prop_matched % cvprop.NEvent() % n_var_evt;


    out_cv  = std::move(matched_cv);
    out_var = std::move(matched_var);
    return true;
}

void run_process(PROpeller &prop, std::vector<std::vector<SystStruct>> &systsstructs, const PROconfig &config, PROpt &options) {
    //input/output logic
    std::string propBinName = options.analysis_tag+"_prop.bin";
    std::string systBinName = options.analysis_tag+"_syst.bin";

    bool need_main_process = (*options.process_command) || (!std::filesystem::exists(systBinName) || !std::filesystem::exists(propBinName));

    if(need_main_process){
        log<LOG_INFO>(L"%1% || Processing PROpeller and PROsysts from XML defined root files, and saving to binary output also: %2%") % __func__ % propBinName.c_str();
        //Process the CAF files to grab and fill all SystStructs and PROpeller
        PROcess_CAFAna(config, systsstructs, prop, options.noxrootd);
        prop.save(propBinName);
        saveSystStructVector(systsstructs, systBinName);
        log<LOG_INFO>(L"%1% || Done processing PROpeller and PROsysts from XML defined root files, and saving to binary output also: %2%") % __func__ % propBinName.c_str();

    }else{
        log<LOG_INFO>(L"%1% || Loading PROpeller and PROsysts from precalc binary input: %2%") % __func__ % propBinName.c_str();
        prop.load(propBinName);
        loadSystStructVector(systsstructs, systBinName);

        //is hash right for PROpeller first?
        log<LOG_INFO>(L"%1% || Done loading. Config hash (%2%) and binary loaded PROpeller (%3%) are here. ") % __func__ %  config.hash % prop.hash ;
        if(config.hash!=prop.hash){
            if(options.force){
                log<LOG_WARNING>(L"%1% || WARNING config hash (%2%) and binary loaded PROpeller (%3%)  not compatable! ") % __func__ %  config.hash % prop.hash ;
                log<LOG_WARNING>(L"%1% || WARNING But we are forcing ahead, be SUPER clear and happy you understand what your doing.  ") % __func__;
            }else{
                log<LOG_ERROR>(L"%1% || ERROR config hash (%2%) and binary loaded PROpeller (%3%)  not compatable! ") % __func__ %  config.hash % prop.hash ;
                exit(1);
            }
        }
        //Now check syststructs, if there is any!
        if(systsstructs.front().size()>0){
            log<LOG_INFO>(L"%1% || Done loading. Config hash (%2%) and binary loaded PROsyst hash(%3%) are here. ") % __func__ %  config.hash % systsstructs[0][0].hash;
            if( config.hash!=systsstructs.front().front().hash){
                if(options.force){
                    log<LOG_WARNING>(L"%1% || WARNING config hash (%2%) and binary loaded PROsyst hash(%3%) not compatable! ") % __func__ %  config.hash %  systsstructs.front().front().hash;
                    log<LOG_WARNING>(L"%1% || WARNING But we are forcing ahead, be SUPER clear and happy you understand what your doing.  ") % __func__;
                }else{
                    log<LOG_ERROR>(L"%1% || ERROR config hash (%2%) and binary loaded PROsyst hash(%3%) not compatable! ") % __func__ %  config.hash %  systsstructs.front().front().hash;
                    exit(1);
                }
            }
        }

    }

    // Combined DetVar propeller binary: one file for all DetVar files, keyed by section+name.
    // Uses detvar_hash (binning + DetVar section only), so changes to <DetVarFiles> or
    // top-level binning trigger reprocessing without invalidating the main prop/syst binaries.
    if(config.m_has_detvar_section) {
        std::string dvAllPropsBin = options.analysis_tag + "_detvar_props.bin";
        bool need_detvar_process = (*options.process_command) || !std::filesystem::exists(dvAllPropsBin);

        std::map<std::string, PROpeller> dvprops;

        if(need_detvar_process) {
            log<LOG_INFO>(L"%1% || Processing all DetVar files into combined binary: %2%") % __func__ % dvAllPropsBin.c_str();
            for(size_t idv = 0; idv < config.GetNumDetVarFiles(); ++idv) {
                const std::string& name = config.m_detvar_files[idv].name;
                const std::string key = DetVarKey(config, idv);
                log<LOG_INFO>(L"%1% || Processing DetVar file '%2%'") % __func__ % name.c_str();
                PROconfig dvconfig = config.BuildDetVarConfig(idv);
                PROpeller dvprop;
                std::vector<std::vector<SystStruct>> dvsystsstructs;
                PROcess_CAFAna(dvconfig, dvsystsstructs, dvprop, options.noxrootd);
                dvprops[key] = std::move(dvprop);
                log<LOG_INFO>(L"%1% || Done processing DetVar file '%2%'") % __func__ % name.c_str();
            }
            saveDetVarProps(dvprops, config.detvar_hash, dvAllPropsBin);
        } else {
            log<LOG_INFO>(L"%1% || Loading DetVar props from combined binary: %2%") % __func__ % dvAllPropsBin.c_str();
            uint32_t loaded_detvar_hash = loadDetVarProps(dvprops, dvAllPropsBin);
            log<LOG_INFO>(L"%1% || Config detvar_hash (%2%) and binary detvar_hash (%3%).") % __func__ % config.detvar_hash % loaded_detvar_hash;
            if(config.detvar_hash != loaded_detvar_hash) {
                if(options.force) {
                    log<LOG_WARNING>(L"%1% || WARNING config detvar_hash (%2%) and binary detvar_hash (%3%) not compatible!") % __func__ % config.detvar_hash % loaded_detvar_hash;
                } else {
                    log<LOG_ERROR>(L"%1% || ERROR config detvar_hash (%2%) and binary detvar_hash (%3%) not compatible!") % __func__ % config.detvar_hash % loaded_detvar_hash;
                    exit(1);
                }
            }
        }

        // Build DetVar SystStructs in memory from dvprops (not stored in syst.bin —
        // they live in the DetVar binary so either binary can be regenerated independently).
        log<LOG_INFO>(L"%1% || Building DetVar SystStructs from DetVar props...") % __func__;
        PROsyst emptySyst;

        for(size_t isec = 0; isec < config.GetNumDetVarSections(); ++isec) {

            // Find CV index for this section
            size_t cv_idx = SIZE_MAX;
            for(size_t i = 0; i < config.m_detvar_files.size(); ++i) {
                if(config.m_detvar_files[i].is_cv && config.m_detvar_files[i].section_index == isec) {
                    cv_idx = i; break;
                }
            }
            if(cv_idx == SIZE_MAX) {
                log<LOG_ERROR>(L"%1% || ERROR: No CV file found for DetVar section %2%") % __func__ % isec;
                continue;
            }

            PROpeller& cvprop = dvprops.at(DetVarKey(config, cv_idx));
            PROconfig cvconfig = config.BuildDetVarConfig(cv_idx);
            NullModel cvmodel(cvprop);
            Eigen::VectorXf cvparams = Eigen::VectorXf::Constant(cvmodel.nparams, 0);
            int cv_binning = cvconfig.i_prime;
            if(cv_binning < 0 || cv_binning >= (int)config.m_num_variables)
                cv_binning = config.i_prime;
            PROspec cvSpec = FillSpectra(cvconfig, cvprop, emptySyst, cvmodel, cvparams, true, cv_binning);

            std::vector<size_t> skip;
            for(size_t idv = 0; idv < config.m_detvar_files.size(); ++idv) {
                if(skip.size() && std::find(skip.begin(), skip.end(), idv) != skip.end()) continue;
                if(config.m_detvar_files[idv].section_index != isec) continue;
                if(config.m_detvar_files[idv].is_cv) continue;

                const std::string& varName = config.m_detvar_files[idv].name;

                if(config.m_mcgen_variation_type_map.count(varName) == 0) {
                    log<LOG_INFO>(L"%1% || Skipping DetVar '%2%' -- no matching entry in <systematics> section.") % __func__ % varName.c_str();
                    continue;
                }
                std::map<int, size_t> syst_files;
                auto find_fn = [&varName](const PROconfig::DetVarFile &dvf) { return dvf.name == varName; };
                auto it = config.m_detvar_files.begin() + idv;
                while((it = std::find_if(it, config.m_detvar_files.end(), find_fn))
                        != std::end(config.m_detvar_files)) {
                    size_t i = std::distance(config.m_detvar_files.begin(), it);
                    syst_files[it->knobval] = i;
                    skip.push_back(i);
                    it++;
                }

                const std::string& systType = config.m_mcgen_variation_type_map.at(varName);
                int binningIndex = config.m_mcgen_variation_binning_map.count(varName) ? config.m_mcgen_variation_binning_map.at(varName) : config.i_prime;
                if(binningIndex < 0 || binningIndex >= (int)config.m_num_variables)
                    binningIndex = config.i_prime;

                PROspec cvSpec = FillSpectra(cvconfig, cvprop, emptySyst, cvmodel, cvparams, true, binningIndex);
                std::map<int, PROspec> specs;
                std::map<int, const PROpeller*> props;
                for(const auto &[k, i] : syst_files) {
                    PROpeller& dvprop = dvprops.at(DetVarKey(config, i));
                    PROconfig dvconfig = config.BuildDetVarConfig(i);
                    NullModel dvmodel(dvprop);
                    Eigen::VectorXf dvparams = Eigen::VectorXf::Constant(dvmodel.nparams, 0);
                    specs[k] = FillSpectra(dvconfig, dvprop, emptySyst, dvmodel, dvparams, true, binningIndex);
                    props[k] = &dvprop;
                }

                // Attempt to build matched specs using only common (run,subrun,event) events.
                // If both propellers have matching vars stored, replace cvSpec/varSpec for this pair.
                PROspec matchedCvSpec = cvSpec;
                const bool matched = BuildDetVarMatchedSpecs(
                    cvprop, props, binningIndex, (int)config.m_num_variable_bins_total[binningIndex],
                    matchedCvSpec, specs);
                if(matched) {
                    log<LOG_INFO>(L"%1% || DetVar '%2%': using event-matched spectra for spline building") % __func__ % varName.c_str();
                    // When cv_variation_matching_vars is used, undo the POT scaling that was
                    // applied during propeller filling so both CV and variation matched spectra
                    // are in raw event-weight units. The spline ratio then reflects only detector
                    // shape/efficiency effects, not any POT normalization artifact.
                    const double det_pot = config.m_det_pot[0];
                    const double cv_pot_dv = config.m_detvar_files[cv_idx].pot;
                    if(det_pot > 0.0 && cv_pot_dv > 0.0) {
                        const float cv_unscale = (float)(cv_pot_dv / det_pot);
                        matchedCvSpec.Spec() *= cv_unscale;
                        matchedCvSpec.Error() *= cv_unscale;
                        for(const auto &[kv, var_file_idx] : syst_files) {
                            const double var_pot_dv = config.m_detvar_files[var_file_idx].pot;
                            if(var_pot_dv > 0.0) {
                                const float var_unscale = (float)(var_pot_dv / det_pot);
                                specs[kv].Spec() *= var_unscale;
                                specs[kv].Error() *= var_unscale;
                            }
                        }
                    }
                } else {
                    log<LOG_INFO>(L"%1% || DetVar '%2%': no matching vars stored, using full spectra") % __func__ % varName.c_str();
                }

                {
                    // Zero out bins where CV is 0 to avoid division by zero when constructing splines
                    Eigen::ArrayXf mask = (matchedCvSpec.Spec().array() != 0.0f).cast<float>();
                    for(auto &[_, spec] : specs) {
                        spec.Spec() = spec.Spec().array() * mask;
                        spec.Error() = spec.Error().array() * mask;
                    }
                }

                {
                    std::vector<eweight_type> knobvals;
                    std::transform(specs.begin(), specs.end(), std::back_inserter(knobvals),
                            [](const auto &p){ return p.first; });
                    std::sort(knobvals.begin(), knobvals.end());
                    SystStruct ss(varName, specs.size(), systType, "1",
                                  knobvals, knobvals, 0);
                    ss.binning = binningIndex;
                    ss.CreateSpecs(matchedCvSpec.Spec().size());
                    ss.p_cv = std::make_shared<PROspec>(matchedCvSpec);
                    for(const auto &[kv, spec] : specs) {
                        size_t idx = std::distance(knobvals.begin(), std::find(knobvals.begin(), knobvals.end(), kv));
                        ss.p_multi_spec[idx] = std::make_shared<PROspec>(specs[kv]);
                    }
                    ss.SetHash(config.hash);
                    for(auto &ssv : systsstructs) ssv.push_back(ss);
                }
                log<LOG_INFO>(L"%1% || Added DetVar SystStruct '%2%' (section %3%, binning=%4%, mode=%5%)") % __func__ % varName.c_str() % isec % binningIndex % systType.c_str();
            }
        }
    }

    // For process-only command, exit early after MC processing is complete
    // This avoids unnecessary setup and potential cleanup issues with ROOT
    //if(*process_command && !*profile_command && !*surface_command && !*protest_command && !*proglobal_command && !*proplot_command && !*profc_command) {
    //    log<LOG_WARNING>(L"%1% || Process command complete. Binary files saved successfully.") % __func__;
    //    return 0;
    //}

    //Scale events by some percentage of total detector POT
    if(options.scale_arg.size()) {
        if (options.scale_arg.size() % 2 != 0) {
            log<LOG_ERROR>(L"%1% || Expected pairs of detector and scaling values (e.g., ICARUS 0.5)") % __func__;
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < options.scale_arg.size(); i += 2) {
            options.scale_map[options.scale_arg[i]] = std::stof(options.scale_arg[i + 1]);
        }
        prop.scale(config, options.scale_map);
    }

}
