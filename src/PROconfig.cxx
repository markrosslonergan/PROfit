#include "PROconfig.h"
#include "PROlog.h"
#include <cctype>
#include <cstdlib>
#include <ctype.h>
#include <numeric>
using namespace PROfit;


PROconfig::PROconfig(const std::string &xml, bool rate_only): 
    m_xmlname(xml), 
    m_plot_pot(1.0),
    m_num_detectors(0),
    m_num_channels(0),
    m_num_modes(0),
    m_num_other_vars(0),
    m_num_bins_detector_block(0),
    m_num_bins_mode_block(0),
    m_num_bins_total(0),
    m_num_truebins_detector_block(0),
    m_num_truebins_mode_block(0),
    m_num_truebins_total(0),
    m_num_bins_detector_block_collapsed(0),
    m_num_bins_mode_block_collapsed(0),
    m_num_bins_total_collapsed(0),
    m_write_out_variation(false), 
    m_form_covariance(true),
    m_write_out_tag("UNSET_DEFAULT"),
    m_num_mcgen_files(0),
    m_bool_rate_only(rate_only)
{

    LoadFromXML(m_xmlname);

    hash = PROconfig::CalcHash();
    construct_collapsing_matrix();

}

bool PROconfig::SameChannels(const PROconfig &one, const PROconfig &two) {
    if(one.m_num_modes != two.m_num_modes) {
        log<LOG_WARNING>(L"%1% || Found different number of modes %2% vs %3%")
            % __func__ % one.m_num_modes % two.m_num_modes;
        return false;
    }
    for(size_t i = 0; i < one.m_num_modes; ++i) {
        if(one.m_mode_names[i] != two.m_mode_names[i]) {
            log<LOG_WARNING>(L"%1% || Found different mode names %2% vs %3%")
                % __func__ % one.m_mode_names[i].c_str() % two.m_mode_names[i].c_str();
            return false;
        }
    }
    if(one.m_num_detectors != two.m_num_detectors) {
        log<LOG_WARNING>(L"%1% || Found different number of detectors %2% vs %3%")
            % __func__ % one.m_num_detectors % two.m_num_detectors;
        return false;
    }
    for(size_t i = 0; i < one.m_num_detectors; ++i) {
        if(one.m_detector_names[i] != two.m_detector_names[i]) {
            log<LOG_WARNING>(L"%1% || Found different detector names %2% vs %3%")
                % __func__ % one.m_detector_names[i].c_str() % two.m_detector_names[i].c_str();
            return false;
        }
    }
    if(one.m_num_channels != two.m_num_channels) {
        log<LOG_WARNING>(L"%1% || Found different number of channels %2% vs %3%")
            % __func__ % one.m_num_channels % two.m_num_channels;
        return false;
    }
    for(size_t i = 0; i < one.m_num_channels; ++i) {
        if(one.m_channel_names[i] != two.m_channel_names[i]) {
            log<LOG_WARNING>(L"%1% || Found different channel names %2% vs %3%")
                % __func__ % one.m_channel_names[i].c_str() % two.m_channel_names[i].c_str();
            return false;
        }
        if(one.m_channel_num_bins[i] != two.m_channel_num_bins[i]) {
            log<LOG_WARNING>(L"%1% || Found different number of channel bins %2% vs %3%")
                % __func__ % one.m_channel_num_bins[i] % two.m_channel_num_bins[i];
            return false;
        }
        for(size_t j = 0; j < one.m_channel_num_bins[i]+1; ++j) {
            if(one.m_channel_bin_edges[i][j] != two.m_channel_bin_edges[i][j]) {
                log<LOG_WARNING>(L"%1% || Found different bin edge for bin %2% in channel %3%. %4% vs %5%")
                    % __func__ % j % one.m_channel_names[i].c_str() % one.m_channel_bin_edges[i][j] % two.m_channel_bin_edges[i][j];
                return false;
            }
        }
    }

    return true;
}


int PROconfig::LoadFromXML(const std::string &filename){


    //Setup TiXml documents
    tinyxml2::XMLDocument doc;
    doc.LoadFile(filename.c_str());
    bool loadOkay = !doc.ErrorID();

    bool use_universe = 1; //FIX
    try{
        if(loadOkay) log<LOG_INFO>(L"%1% || Correctly loaded and parsed XML, continuing") % __func__;
        else throw 404;    
    }
    catch (int ernum) {
        log<LOG_ERROR>(L"%1% || ERROR: Failed to load XML configuration file names %4%. @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__ % filename.c_str();
        log<LOG_ERROR>(L"This generally means broken brackets or attribute syntax in xml itself.");
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }

    tinyxml2::XMLHandle hDoc(&doc);

    tinyxml2::XMLElement *pMode, *pDet, *pChan, *pPOT;

    std::vector<std::string> allowed_elements = {"mode", "detector", "channel", "plotpot", "MCFile","WeightMaps","model","variation_list","correlation","varied_spectrum", "ShapeOnlyUncertainty"   };
    for (tinyxml2::XMLElement* elem = doc.FirstChildElement(); elem; elem = elem->NextSiblingElement()) {
        std::string name = elem->Name();
        if (std::find(allowed_elements.begin(), allowed_elements.end(), name) == allowed_elements.end()) {
                    log<LOG_ERROR>(L"%1% || ERROR! Top Level Element [%2%] in the XML is not expected.") % __func__ % name.c_str()  ;
                    log<LOG_ERROR>(L"%1% || -- Check spelling: allowed elements are %2%") % __func__ % allowed_elements ;
                    throw std::invalid_argument(std::string("Top level <element> not allowed : ") + name);
        }
    }


    //max subchannels 100? Can we avoid this
    m_subchannel_plotnames.resize(100);
    m_subchannel_colors.resize(100);
    m_subchannel_datas.resize(100);
    m_subchannel_names.resize(100);
    char *end;

    //Grab the first element. Note very little error checking here! make sure they exist.
    pMode = doc.FirstChildElement("mode");
    pDet =  doc.FirstChildElement("detector");
    pChan = doc.FirstChildElement("channel");
    pPOT = doc.FirstChildElement("plotpot");


    while(pPOT){

        //check for known attributes
        const std::vector<std::string> expected_attrs = {"value"};
        for (const tinyxml2::XMLAttribute* attr = pPOT->FirstAttribute(); attr; attr = attr->Next()) {
            std::string name = attr->Name();
            if (std::find(expected_attrs.begin(), expected_attrs.end(), name) == expected_attrs.end()) {
                log<LOG_ERROR>(L"%1% || ERROR! Attribute [%2%] in the <plotpot> element is not expected.") % __func__ % name.c_str()  ;
                log<LOG_ERROR>(L"%1% || -- Check spelling: allowed attributes are %2%") % __func__ % expected_attrs ;
                throw std::invalid_argument(std::string("<plotpot> attribute not allowed : ") + name);
            }
        }

        const char* inplotpot = pPOT->Attribute("value");
        if(inplotpot){
            m_plot_pot = strtod(inplotpot,&end);
        }
        pPOT = pPOT->NextSiblingElement("plotpot");
    }


    if(!pMode){
        log<LOG_ERROR>(L"%1% || ERROR: Need at least 1 mode defined in xml.@ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }else{
        while(pMode){
            // What modes are we running in (e.g nu, nu bar, horn current=XXvolts....) Can have as many as we want

            const std::vector<std::string> expected_attrs = {"name","plotname","use"};
            for (const tinyxml2::XMLAttribute* attr = pMode->FirstAttribute(); attr; attr = attr->Next()) {
                std::string name = attr->Name();
                if (std::find(expected_attrs.begin(), expected_attrs.end(), name) == expected_attrs.end()) {
                    log<LOG_ERROR>(L"%1% || ERROR! Attribute [%2%] in the <mode> element is not expected.") % __func__ % name.c_str()  ;
                    log<LOG_ERROR>(L"%1% || -- Check spelling: allowed attributes are %2%") % __func__ % expected_attrs ;
                    throw std::invalid_argument(std::string("<mode> attribute not allowed : ") + name);
                }
            }




            const char* mode_name= pMode->Attribute("name");
            if(mode_name==NULL){
                log<LOG_ERROR>(L"%1% || Modes need a name! Please define a name attribute for all modes. @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }else{
                m_mode_names.push_back(mode_name);
            }

            const char* mode_plotname= pMode->Attribute("plotname");
            if(mode_plotname==NULL){
                m_mode_plotnames.push_back(m_mode_names.back());
            }else{
                m_mode_plotnames.push_back(mode_plotname);
            }

            const char* mode_use = pMode->Attribute("use");
            if(mode_use == NULL || std::string(mode_use) == "true")
                m_mode_bool.push_back(true);
            else
                m_mode_bool.push_back(false);

            pMode = pMode->NextSiblingElement("mode");
            log<LOG_DEBUG>(L"%1% || Loading Mode %2%  ") % __func__ % m_mode_names.back().c_str() ;

        }
    }

    // How many detectors do we want!
    if(!pDet){
        log<LOG_ERROR>(L"%1% || ERROR: Need at least 1 detector defined in xml.@ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);

    }else{

        while(pDet){

            const std::vector<std::string> expected_attrs = {"name","plotname","use"};
            for (const tinyxml2::XMLAttribute* attr = pDet->FirstAttribute(); attr; attr = attr->Next()) {
                std::string name = attr->Name();
                if (std::find(expected_attrs.begin(), expected_attrs.end(), name) == expected_attrs.end()) {
                    log<LOG_ERROR>(L"%1% || ERROR! Attribute [%2%] in the <detector> element is not expected.") % __func__ % name.c_str()  ;
                    log<LOG_ERROR>(L"%1% || -- Check spelling: allowed attributes are %2%") % __func__ % expected_attrs ;
                    throw std::invalid_argument(std::string("<detector> attribute not allowed : ") + name);
                }
            }



            const char* detector_name= pDet->Attribute("name");
            if(detector_name==NULL){
                log<LOG_ERROR>(L"%1% || ERROR: Need all detectors to have a name attribute @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }else{
                m_detector_names.push_back(detector_name);
            }

            const char* detector_plotname = pDet->Attribute("plotname");
            if(detector_plotname==NULL){
                m_detector_plotnames.push_back(m_detector_names.back());
            }else{
                m_detector_plotnames.push_back(detector_plotname);
            }

            const char* detector_use = pDet->Attribute("use");
            if(detector_use==NULL || std::string(detector_use) == "true")
                m_detector_bool.push_back(true);
            else
                m_detector_bool.push_back(false);

            pDet = pDet->NextSiblingElement("detector");
            log<LOG_DEBUG>(L"%1% || Loading Det %2%  ") % __func__ % m_detector_names.back().c_str();

        }
    }

    //How many channels do we want! At the moment each detector must have all channels
    int nchan = 0;
    if(!pChan){
        log<LOG_ERROR>(L"%1% || ERROR: Need at least 1 channel defined in xml.@ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }else{

        std::vector<std::string> expected_attrs = {"name","plotname","use","unit"};
        for (const tinyxml2::XMLAttribute* attr = pChan->FirstAttribute(); attr; attr = attr->Next()) {
            std::string name = attr->Name();
            if (std::find(expected_attrs.begin(), expected_attrs.end(), name) == expected_attrs.end()) {
                log<LOG_ERROR>(L"%1% || ERROR! Attribute [%2%] in the <channel> element is not expected.") % __func__ % name.c_str()  ;
                log<LOG_ERROR>(L"%1% || -- Check spelling: allowed attributes are %2%") % __func__ % expected_attrs ;
                throw std::invalid_argument(std::string("<channel>attribute not allowed : ") + name);
            }
        }



        while(pChan){
            // Read in how many bins this channel uses

            const char* channel_name= pChan->Attribute("name");
            if(channel_name==NULL){
                log<LOG_ERROR>(L"%1% || ERROR: Need all channels to have names in xml.@ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }else{
                m_channel_names.push_back(channel_name);
            }

            const char* channel_plotname= pChan->Attribute("plotname");
            if(channel_plotname==NULL){
                m_channel_plotnames.push_back(m_channel_names.back());
            }else{
                m_channel_plotnames.push_back(channel_plotname);
            }


            const char* channel_use = pChan->Attribute("use");
            if(channel_use==NULL || std::string(channel_use) == "true")
                m_channel_bool.push_back(true);
            else
                m_channel_bool.push_back(false);


            const char* channel_unit= pChan->Attribute("unit");
            if(channel_unit==NULL){
                m_channel_units.push_back("");
            }else{
                m_channel_units.push_back(channel_unit);
            }

            log<LOG_DEBUG>(L"%1% || Loading Channel %2% with   ") % __func__ % m_channel_names.back().c_str() ;


            // What are the bin edges and bin widths (bin widths just calculated from edges now)
            tinyxml2::XMLElement *pBin = pChan->FirstChildElement("bins");

            log<LOG_DEBUG>(L"%1% || This variable has a Reco Binning.   ") % __func__  ;

            expected_attrs = {"min","max","nbins","edges"};
            for (const tinyxml2::XMLAttribute* attr = pBin->FirstAttribute(); attr; attr = attr->Next()) {
                std::string name = attr->Name();
                if (std::find(expected_attrs.begin(), expected_attrs.end(), name) == expected_attrs.end()) {
                    log<LOG_ERROR>(L"%1% || ERROR! Attribute [%2%] in the <channel/bins> element is not expected.") % __func__ % name.c_str()  ;
                    log<LOG_ERROR>(L"%1% || -- Check spelling: allowed attributes are %2%") % __func__ % expected_attrs ;
                    throw std::invalid_argument(std::string("<channel/bins> attribute not allowed : ") + name);
                }
            }


            const char* rmin = pBin->Attribute("min");
            const char* rmax = pBin->Attribute("max");
            const char* rnbins = pBin->Attribute("nbins");
            const char* redges =pBin->Attribute("edges");

            int nbinsp;
            std::vector<float> binedge, binwidth;

            // use edges if defined, otherwise use min-max-nbins 
            if(redges != NULL){

                std::stringstream reco_iss(redges);
                float number;
                while ( reco_iss >> number ){
                    binedge.push_back(number);
                }

                nbinsp = binedge.size() - 1;
                for(int i = 0; i != nbinsp; ++i){
                    binwidth.push_back(binedge[i+1] - binedge[i]);
                }

                log<LOG_DEBUG>(L"%1% || This variable has a Reco Binning with  %2% bins, Edges defined as %3%    ") % __func__ % nbinsp % binedge ;

                if(m_bool_rate_only){
                    m_channel_num_bins.push_back(1);
                    std::vector<float> counting_exp_bin = {binedge.front(), binedge.back()};
                    m_channel_bin_edges.push_back(counting_exp_bin);
                    std::vector<float> counting_exp_width = {binedge.front()-binedge.back()};
                    m_channel_bin_widths.push_back(counting_exp_width);
                }else{
                    m_channel_num_bins.push_back(nbinsp);
                    m_channel_bin_edges.push_back(binedge);
                    m_channel_bin_widths.push_back(binwidth);
                }

            }else if (rmin!=NULL && rmax!=NULL && rnbins!=NULL ){

                float minp = strtod(rmin, &end);
                float maxp = strtod(rmax, &end);
                nbinsp = (int)strtod(rnbins, &end);
                float step = (maxp-minp)/(float)nbinsp;
                for(int i=0; i<nbinsp; i++){
                    binedge.push_back(minp+i*step);
                }
                binedge.push_back(maxp);
                binwidth.resize(nbinsp, step);
                log<LOG_DEBUG>(L"%1% || This variable has a Reco Binning with min %2%, max %3% and nbins %4%   ") % __func__ % minp % maxp % nbinsp ;
                log<LOG_DEBUG>(L"%1% || Which corresponds to edges %2%   ") % __func__ % binedge ;

                if(m_bool_rate_only){
                    m_channel_num_bins.push_back(1);
                    std::vector<float> counting_exp_bin = {binedge.front(), binedge.back()};
                    m_channel_bin_edges.push_back(counting_exp_bin);
                    std::vector<float> counting_exp_width = {binedge.front()-binedge.back()};
                    m_channel_bin_widths.push_back(counting_exp_width);
                }else{
                    m_channel_num_bins.push_back(nbinsp);
                    m_channel_bin_edges.push_back(binedge);
                    m_channel_bin_widths.push_back(binwidth);
                }
            }else{
                log<LOG_ERROR>(L"%1% || ERROR: You need to define a reco binning using either edges or min/max/nsteps @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }


            tinyxml2::XMLElement *pBinT = pChan->FirstChildElement("truebins");
            if(pBinT){

                expected_attrs = {"min","max","nbins","edges"};
                for (const tinyxml2::XMLAttribute* attr = pBinT->FirstAttribute(); attr; attr = attr->Next()) {
                    std::string name = attr->Name();
                    if (std::find(expected_attrs.begin(), expected_attrs.end(), name) == expected_attrs.end()) {
                        log<LOG_ERROR>(L"%1% || ERROR! Attribute [%2%] in the <channel/truebins> element is not expected.") % __func__ % name.c_str()  ;
                        log<LOG_ERROR>(L"%1% || -- Check spelling: allowed attributes are %2%") % __func__ % expected_attrs ;
                        throw std::invalid_argument(std::string("<channel/truebins> attribute not allowed : ") + name);
                    }
                }

                const char* tmin = pBinT->Attribute("min");
                const char* tmax = pBinT->Attribute("max");
                const char* tnbins = pBinT->Attribute("nbins");
                const char* tedges =pBinT->Attribute("edges");
                if(tmin==NULL && tmax==NULL && tnbins==NULL && tedges == NULL)
                {
                    log<LOG_DEBUG>(L"%1% || This variable has a NO truth binning (or attribute min,max,nbins)  ") % __func__ ;
                    m_channel_num_truebins.push_back(0);
                    m_channel_truebin_edges.push_back(std::vector<float>());
                    m_channel_truebin_widths.push_back(std::vector<float>());
                }else{

                    log<LOG_DEBUG>(L"%1% || This variable has a Truth Binning.   ") % __func__  ;

                    int nbinsp;
                    std::vector<float> binedge, binwidth;

                    // use edges if defined, otherwise use min-max-nbins 
                    if(tedges != NULL){

                        std::stringstream true_iss(tedges);
                        float number;
                        while ( true_iss >> number ){
                            binedge.push_back(number );
                        }

                        nbinsp = binedge.size() - 1;
                        for(int i = 0; i != nbinsp; ++i){
                            binwidth.push_back(binedge[i+1] - binedge[i]);
                        }

                        log<LOG_DEBUG>(L"%1% || This variable has a Truth Binning with  %2% bins, Edges defined as %3%    ") % __func__ % nbinsp % binedge ;


                    }else{
                        float minp = strtod(tmin, &end);
                        float maxp = strtod(tmax, &end);
                        nbinsp = (int)strtod(tnbins, &end);
                        float step = (maxp-minp)/(float)nbinsp;
                        for(int i=0; i<nbinsp; i++){
                            binedge.push_back(minp+i*step);
                        }
                        binedge.push_back(maxp);
                        binwidth.resize(nbinsp, step);
                        log<LOG_DEBUG>(L"%1% || This variable has a Truth Binning with min %2%, max %3% and nbins %4%   ") % __func__ % minp % maxp % nbinsp ;
                        log<LOG_DEBUG>(L"%1% || Which corresponds to edges %2%   ") % __func__ % binedge ;

                    }

                    m_channel_num_truebins.push_back(nbinsp);
                    m_channel_truebin_edges.push_back(binedge);
                    m_channel_truebin_widths.push_back(binwidth);


                }
            }else{
                m_channel_num_truebins.push_back(0);
                m_channel_truebin_edges.push_back(std::vector<float>());
                m_channel_truebin_widths.push_back(std::vector<float>());
            }

            tinyxml2::XMLElement *pBinO = pChan->FirstChildElement("otherbins");
            m_channel_num_other_bins.push_back({});
            m_channel_other_bin_edges.push_back({});
            m_channel_other_bin_widths.push_back({});
            m_channel_other_units.push_back({});
            while(pBinO){

                expected_attrs = {"min","max","nbins","edges","unit"};
                for (const tinyxml2::XMLAttribute* attr = pBinO->FirstAttribute(); attr; attr = attr->Next()) {
                    std::string name = attr->Name();
                    if (std::find(expected_attrs.begin(), expected_attrs.end(), name) == expected_attrs.end()) {
                        log<LOG_ERROR>(L"%1% || ERROR! Attribute [%2%] in the <channel/otherbins> element is not expected.") % __func__ % name.c_str()  ;
                        log<LOG_ERROR>(L"%1% || -- Check spelling: allowed attributes are %2%") % __func__ % expected_attrs ;
                        throw std::invalid_argument(std::string("<channel/otherbins> attribute not allowed : ") + name);
                    }
                }

                const char* omin = pBinO->Attribute("min");
                const char* omax = pBinO->Attribute("max");
                const char* onbins = pBinO->Attribute("nbins");
                const char* oedges = pBinO->Attribute("edges");
                const char* ounits = pBinO->Attribute("unit");
                if(omin==NULL && omax==NULL && onbins==NULL && oedges == NULL) {
                    log<LOG_DEBUG>(L"%1% || This variable has a NO other binning (or attribute min,max,nbins)  ") % __func__ ;
                    m_channel_num_other_bins.back().push_back(0);
                    m_channel_other_bin_edges.back().push_back(std::vector<float>());
                    m_channel_other_bin_widths.back().push_back(std::vector<float>());
                    m_channel_other_units.back().push_back("");
                }else{
                    log<LOG_DEBUG>(L"%1% || This variable has an Other Binning.   ") % __func__  ;

                    int nbinsp;
                    std::vector<float> binedge, binwidth;

                    // use edges if defined, otherwise use min-max-nbins 
                    if(oedges != NULL){
                        std::stringstream other_iss(oedges);
                        float number;
                        while (other_iss >> number){
                            binedge.push_back(number);
                        }

                        nbinsp = binedge.size() - 1;
                        for(int i = 0; i != nbinsp; ++i){
                            binwidth.push_back(binedge[i+1] - binedge[i]);
                        }

                        log<LOG_DEBUG>(L"%1% || This variable has a Truth Binning with  %2% bins, Edges defined as %3%    ") % __func__ % nbinsp % binedge ;
                    }else{
                        float minp = strtod(omin, &end);
                        float maxp = strtod(omax, &end);
                        nbinsp = (int)strtod(onbins, &end);
                        float step = (maxp-minp)/(float)nbinsp;
                        for(int i=0; i<nbinsp; i++){
                            binedge.push_back(minp+i*step);
                        }
                        binedge.push_back(maxp);
                        binwidth.resize(nbinsp, step);
                        log<LOG_DEBUG>(L"%1% || This variable has a Truth Binning with min %2%, max %3% and nbins %4%   ") % __func__ % minp % maxp % nbinsp ;
                        log<LOG_DEBUG>(L"%1% || Which corresponds to edges %2%   ") % __func__ % binedge ;
                    }

                    m_channel_num_other_bins.back().push_back(nbinsp);
                    m_channel_other_bin_edges.back().push_back(binedge);
                    m_channel_other_bin_widths.back().push_back(binwidth);
                    m_channel_other_units.back().push_back(ounits ? ounits : "");
                }
                pBinO = pBinO->NextSiblingElement("otherbins");
            }

            // Now loop over all this channels subchanels. Not the names must be UNIQUE!!
            tinyxml2::XMLElement *pSubChan;
            m_subchannel_bool.push_back({});
            pSubChan = pChan->FirstChildElement("subchannel");
            while(pSubChan){

                std::vector<std::string> expected_attrs = {"name","plotname","color","use","data"};
                for (const tinyxml2::XMLAttribute* attr = pSubChan->FirstAttribute(); attr; attr = attr->Next()) {
                    std::string name = attr->Name();
                    if (std::find(expected_attrs.begin(), expected_attrs.end(), name) == expected_attrs.end()) {
                        log<LOG_ERROR>(L"%1% || ERROR! Attribute [%2%] in the <subchannel> element is not expected.") % __func__ % name.c_str()  ;
                        log<LOG_ERROR>(L"%1% || -- Check spelling: allowed attributes are %2%") % __func__ % expected_attrs ;
                        throw std::invalid_argument(std::string("<subchannel> attribute not allowed : ") + name);
                    }
                }

                const char* subchannel_name= pSubChan->Attribute("name");
                if(subchannel_name==NULL){
                    log<LOG_ERROR>(L"%1% || ERROR: Subchannels need a name in xml.@ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);

                }else{
                    m_subchannel_names[nchan].push_back(subchannel_name);
                    log<LOG_DEBUG>(L"%1% || Subchannel Starting:  %2%") % __func__ % m_subchannel_names.at(nchan).back().c_str() ;

                }


                const char* subchannel_plotname= pSubChan->Attribute("plotname");
                if(subchannel_plotname==NULL){
                    m_subchannel_plotnames[nchan].push_back(m_subchannel_names[nchan].back());
                }else{
                    m_subchannel_plotnames[nchan].push_back(subchannel_plotname);
                }

                const char* subchannel_color= pSubChan->Attribute("color");
                if(subchannel_color==NULL){
                    m_subchannel_colors[nchan].push_back("NONE");
                }else{
                    m_subchannel_colors[nchan].push_back(subchannel_color);
                }

                const char* subchannel_use = pSubChan->Attribute("use");
                if(subchannel_use==NULL || std::string(subchannel_use) == "true")
                    m_subchannel_bool.back().push_back(true);
                else
                    m_subchannel_bool.back().push_back(false);

                const char* subchannel_data= pSubChan->Attribute("data");
                if(subchannel_data==NULL){
                    m_subchannel_datas[nchan].push_back(0);
                }else{
                    m_subchannel_datas[nchan].push_back(1);
                }

                pSubChan = pSubChan->NextSiblingElement("subchannel");
            }

            nchan++;
            pChan = pChan->NextSiblingElement("channel");
        }
    }//end channel loop
    // Assume all channels have the same number of "other" vars
    m_num_other_vars = m_channel_other_bin_edges[0].size();
    m_num_other_bins_total = std::vector<size_t>(m_num_other_vars, 0);
    m_num_other_bins_total_collapsed = std::vector<size_t>(m_num_other_vars, 0);
    m_num_other_bins_mode_block = std::vector<size_t>(m_num_other_vars, 0);
    m_num_other_bins_mode_block_collapsed = std::vector<size_t>(m_num_other_vars, 0);
    m_num_other_bins_detector_block = std::vector<size_t>(m_num_other_vars, 0);
    m_num_other_bins_detector_block_collapsed = std::vector<size_t>(m_num_other_vars, 0);

    //Now onto mcgen, for CV specs or for covariance generation
    tinyxml2::XMLElement *pMC, *pWeiMaps, *pList, *pCorrelations, *pSpec, *pShapeOnlyMap;
    pMC   = doc.FirstChildElement("MCFile");
    pWeiMaps = doc.FirstChildElement("WeightMaps");
    pList = doc.FirstChildElement("variation_list");
    pCorrelations = doc.FirstChildElement("correlation");
    pSpec = doc.FirstChildElement("varied_spectrum");
    pShapeOnlyMap = doc.FirstChildElement("ShapeOnlyUncertainty");



    if(pMC){//Skip if not in XML
        while(pMC)
        {


            std::vector<std::string> expected_attrs = {"treename","filename","maxevents","scale","pot","fake"};
            for (const tinyxml2::XMLAttribute* attr = pMC->FirstAttribute(); attr; attr = attr->Next()) {
                std::string name = attr->Name();
                if (std::find(expected_attrs.begin(), expected_attrs.end(), name) == expected_attrs.end()) {
                    log<LOG_ERROR>(L"%1% || ERROR! Attribute [%2%] in the <MCFile> element is not expected.") % __func__ % name.c_str()  ;
                    log<LOG_ERROR>(L"%1% || -- Check spelling: allowed attributes are %2%") % __func__ % expected_attrs ;
                    throw std::invalid_argument(std::string("<MCfile> attribute not allowed : ") + name);
                }
            }

            const char* tree = pMC->Attribute("treename");
            if(tree==NULL){
                log<LOG_ERROR>(L"%1% || ERROR: You must have an associated root TTree name for all MonteCarloFile tags.. eg. treename='events' @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }else{
                m_mcgen_tree_name.push_back(tree);
            }

            const char* file = pMC->Attribute("filename");
            if(file==NULL){
                log<LOG_ERROR>(L"%1% || ERROR: You must have an associated root TFile name for all MonteCarloFile tags.. eg. filename='my.root' @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }else{
                m_mcgen_file_name.push_back(file);
            }


            const char* maxevents = pMC->Attribute("maxevents");
            if(maxevents==NULL){
                m_mcgen_maxevents.push_back(1e16);
            }else{
                m_mcgen_maxevents.push_back(strtod(maxevents,&end) );
            }

            //arbitray scaling you can have
            const char* scale = pMC->Attribute("scale");
            if(scale==NULL){
                m_mcgen_scale.push_back(1.0);
            }else{
                m_mcgen_scale.push_back(strtod(scale,&end) );
            }

            const char* inpot = pMC->Attribute("pot");
            if(inpot==NULL){
                m_mcgen_pot.push_back(-1.0);
            }else{
                m_mcgen_pot.push_back(strtod(inpot,&end) );
            }

            //Is this useful? 
            const char* isfake = pMC->Attribute("fake");
            if(isfake==NULL){
                m_mcgen_fake.push_back(false);
            }else{
                m_mcgen_fake.push_back(true);
            }

            log<LOG_DEBUG>(L"%1% || MultisimFile %2%, treename: %3%  ") % __func__ % m_mcgen_file_name.back().c_str() % m_mcgen_tree_name.back().c_str();

            m_mcgen_numfriends.push_back(0);

            //Here we can grab some friend tree information
            tinyxml2::XMLElement *pFriend;
            pFriend = pMC->FirstChildElement("friend");
            while(pFriend){

                expected_attrs = {"filename","treename"};
                for (const tinyxml2::XMLAttribute* attr = pFriend->FirstAttribute(); attr; attr = attr->Next()) {
                    std::string name = attr->Name();
                    if (std::find(expected_attrs.begin(), expected_attrs.end(), name) == expected_attrs.end()) {
                        log<LOG_ERROR>(L"%1% || ERROR! Attribute [%2%] in the <MCFile/friend> element is not expected.") % __func__ % name.c_str()  ;
                        log<LOG_ERROR>(L"%1% || -- Check spelling: allowed attributes are %2%") % __func__ % expected_attrs ;
                        throw std::invalid_argument(std::string("<MCfile/friend> attribute not allowed : ") + name);
                    }
                }


                std::string ffname;
                const char* friend_filename = pFriend->Attribute("filename");
                if(friend_filename==NULL){
                    ffname = m_mcgen_file_name.back();
                }else{
                    ffname = friend_filename;
                }


                m_mcgen_file_friend_treename_map[m_mcgen_file_name.back()].push_back( pFriend->Attribute("treename") );
                m_mcgen_file_friend_map[m_mcgen_file_name.back()].push_back(ffname);
                m_mcgen_numfriends.back()+=1;
                pFriend = pFriend->NextSiblingElement("friend");
            }//END of friend loop


            tinyxml2::XMLElement *pBranch;
            pBranch = pMC->FirstChildElement("branch");



            std::vector<bool> TEMP_additional_weight_bool;
            std::vector<std::string> TEMP_additional_weight_name;
            std::vector<std::string> TEMP_eventweight_branch_names;
            std::vector<bool> TEMP_hist_weight_bool;
            std::vector<std::string> TEMP_hist_weight_name;
            std::vector<int> TEMP_eventweight_branch_syst;

            std::vector<std::shared_ptr<BranchVariable>> TEMP_branch_variables;
            while(pBranch){

                expected_attrs = {"name","incl_systematics","associated_subchannel","associated_systematic","central_value","eventweight_branch_name","additional_weight","true_param_name","other_param_names","pdg_name","model_rule","true_L_name","true_param_name","hist_reweight","true_proton_mom_name","true_proton_costh_name"};
                for (const tinyxml2::XMLAttribute* attr = pBranch->FirstAttribute(); attr; attr = attr->Next()) {
                    std::string name = attr->Name();
                    if (std::find(expected_attrs.begin(), expected_attrs.end(), name) == expected_attrs.end()) {
                        log<LOG_ERROR>(L"%1% || ERROR! Attribute [%2%] in the <MCFile/branch> element is not expected.") % __func__ % name.c_str()  ;
                        log<LOG_ERROR>(L"%1% || -- Check spelling: allowed attributes are %2%") % __func__ % expected_attrs ;
                        throw std::invalid_argument(std::string("<MCfile/branch> attribute not allowed : ") + name);
                    }
                }



                const char* bnam = pBranch->Attribute("name");
                const char* bincsyst = pBranch->Attribute("incl_systematics");
                const char* bhist = pBranch->Attribute("associated_subchannel");
                const char* bsyst = pBranch->Attribute("associated_systematic");
                const char* bcentral = pBranch->Attribute("central_value");
                const char* bwname = pBranch->Attribute("eventweight_branch_name");
                const char* badditional_weight = pBranch->Attribute("additional_weight");

                if(bwname== NULL){
                    //log<LOG_WARNING>(L"%1% || WARNING: No eventweight branch name passed, defaulting to 'weights' @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                    TEMP_eventweight_branch_names.push_back("weights");
                }else{
                    log<LOG_DEBUG>(L"%1% || Setting eventweight branch name %2%") %__func__ % bnam;
                    TEMP_eventweight_branch_names.push_back(std::string(bwname));
                }

                if(bnam == NULL){
                    log<LOG_ERROR>(L"%1% || ERROR!: Each branch must include the name of the branch to use. @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                    log<LOG_ERROR>(L"%1% || ERROR!: e.g name = 'ereco' @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);
                }
                log<LOG_DEBUG>(L"%1% || Branch name %2%") %__func__ % bnam;		

                if(bincsyst== NULL || strcmp(bincsyst, "true") == 0){
                    log<LOG_DEBUG>(L"%1% ||Apply systemtics to this file (default) ' @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                    TEMP_eventweight_branch_syst.push_back(1);
                }else{
                    log<LOG_DEBUG>(L"%1% || DO NOT apply systemtics to this file (e.g for cosmics) ' @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                    TEMP_eventweight_branch_syst.push_back(0);
                }

                if(bhist == NULL){
                    log<LOG_ERROR>(L"%1% || Each branch must have an associated_subchannel to fill! On branch %4% : @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__ % bnam;
                    log<LOG_ERROR>(L"%1% || e.g associated_subchannel='mode_det_chan_subchannel ") % __func__ % __LINE__  % __FILE__;
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);
                }
                log<LOG_DEBUG>(L"%1% || Branch subchannel %2%") %__func__ % bhist;				


                if(bsyst == NULL){
                    if(use_universe == false){
                        log<LOG_WARNING>(L"%1% || WARNING: No root file with unique systematic variation is provided ") % __func__;
                        log<LOG_ERROR>(L"%1% || ERROR! please provide what systematic variation this file correpsonds to!") % __func__;
                        log<LOG_ERROR>(L"Terminating.");
                        exit(EXIT_FAILURE);
                    }
                    systematic_name.push_back("");
                }else{
                    systematic_name.push_back(bsyst);	

                }
                //log<LOG_DEBUG>(L"%1% || Branch syst %2%") %__func__ % bsyst;						

                //std::string chk_wei = badditional_weight;
                if(badditional_weight == NULL || strcmp(badditional_weight, "") == 0){ 
                    TEMP_additional_weight_bool.push_back(0);
                    TEMP_additional_weight_name.push_back("1");
                    log<LOG_DEBUG>(L"%1% || Setting NO additional weight for branch %2% (1)") % __func__ % bnam ;
                }else{
                    TEMP_additional_weight_name.push_back(badditional_weight);
                    TEMP_additional_weight_bool.push_back(1);
                    log<LOG_DEBUG>(L"%1% || Setting an additional weight for branch %2% using the branch %3% as a reweighting.") % __func__ % bnam %badditional_weight;

                }


                if(use_universe){
                    TEMP_branch_variables.push_back( std::make_shared<BranchVariable>(bnam, "float", bhist ) );
                } else  if((std::string)bcentral == "true"){
                    TEMP_branch_variables.push_back( std::make_shared<BranchVariable>(bnam, "float", bhist,bsyst, true) );
                    log<LOG_DEBUG>(L"%1% || Setting as  CV for det sys.") % __func__ ;
                } else {
                    TEMP_branch_variables.push_back( std::make_shared<BranchVariable>(bnam, "float", bhist,bsyst, false) );
                    log<LOG_DEBUG>(L"%1% || Setting as individual (not CV) for det sys.") % __func__ ;
                }

                TEMP_branch_variables.back()->SetIncludeSystematics(TEMP_eventweight_branch_syst.back());

                if(pBranch->Attribute("true_param_name")) {
                    TEMP_branch_variables.back()->SetTrueParam(pBranch->Attribute("true_param_name"));
                }
                if(pBranch->Attribute("other_param_names")) {
                    log<LOG_DEBUG>(L"%1% || Found other_param_names %2%") % __func__ % pBranch->Attribute("other_param_names");
                    TEMP_branch_variables.back()->SetOtherParams(pBranch->Attribute("other_param_names"));
                }
                if(pBranch->Attribute("pdg_name")) {
                    TEMP_branch_variables.back()->SetPDG(pBranch->Attribute("pdg_name"));
                }
                if(pBranch->Attribute("model_rule")) {
                    TEMP_branch_variables.back()->SetModelRule(pBranch->Attribute("model_rule"));
                }
                if(pBranch->Attribute("model_rule")) {
                    log<LOG_DEBUG>(L"%1% || Branch has Model Rule  %2% ") % __func__ % pBranch->Attribute("model_rule") ;
                }

                if(pBranch->Attribute("true_L_name") != NULL){
                    //for oscillation that needs both E and L
                    TEMP_branch_variables.back()->SetTrueL(pBranch->Attribute("true_L_name"));
                    log<LOG_DEBUG>(L"%1% || Oscillations using true param name:   %2% and baseline %3% ") % __func__ % pBranch->Attribute("true_param_name") % pBranch->Attribute("true_L_name") ;
                }else if(pBranch->Attribute("true_param_name")){
                    //for oscillations that only needs E, such as an energy-dependent scaling for single photon NCpi0!
                    log<LOG_DEBUG>(L"%1% || Oscillations using  Energy only dependent oscillation ( or shift/normalization)  %2% ") % __func__ % pBranch->Attribute("true_param_name") ;
                }

                std::string hist_reweight = "false";
                if(pBranch->Attribute("hist_reweight")!=NULL){
                    hist_reweight=pBranch->Attribute("hist_reweight");
                }

                if(hist_reweight == "false"){
                    log<LOG_DEBUG>(L"%1% || Histogram reweighting is OFF ") % __func__ ;
                    TEMP_branch_variables.back()->SetReweight(false);
                }
                else if (hist_reweight=="true"){
                    log<LOG_DEBUG>(L"%1% || Histogram reweighting is ON ") % __func__;
                    TEMP_branch_variables.back()->SetReweight(true);
                    log<LOG_DEBUG>(L"%1% || Successfully setreweight ") % __func__;
                    TEMP_branch_variables.back()->SetTrueLeadingProtonP(pBranch->Attribute("true_proton_mom_name"));
                    log<LOG_DEBUG>(L"%1% || Successfully set trueleadingp: %2% ") % __func__ % pBranch->Attribute("true_proton_mom_name");				 TEMP_branch_variables.back()->SetTrueLeadingProtonCosth(pBranch->Attribute("true_proton_costh_name"));
                    log<LOG_DEBUG>(L"%1% || Successfully set trueleadingcosth: %2% ") % __func__ % pBranch->Attribute("true_proton_costh_name");				  				   
                }

                log<LOG_DEBUG>(L"%1% || Associated subchannel: %2% ") % __func__ % bhist;

                pBranch = pBranch->NextSiblingElement("branch");
            }

            m_mcgen_additional_weight_name.push_back(TEMP_additional_weight_name);
            m_mcgen_additional_weight_bool.push_back(TEMP_additional_weight_bool);
            m_branch_variables.push_back(TEMP_branch_variables);
            m_mcgen_eventweight_branch_names.push_back(TEMP_eventweight_branch_names);
            m_mcgen_eventweight_branch_syst.push_back(TEMP_eventweight_branch_syst);
            //next file
            pMC=pMC->NextSiblingElement("MCFile");
        }
    }

    if(!pList){
        log<LOG_DEBUG>(L"%1% || No Allowlist or Denylist set, including ALL variations by default.") % __func__  ;
    }else{
        while(pList){

            tinyxml2::XMLElement *pAllowList = pList->FirstChildElement("allowlist");
            while(pAllowList){
                const char *text = pAllowList->GetText();
                std::string wt = "null";
                if(text) wt = std::string(text);

                //check for known attributes
                const std::vector<std::string> expected_attrs = {"type", "plotname", "binning", "knobvals", "tag", "prior", "center"};
                for (const tinyxml2::XMLAttribute* attr = pAllowList->FirstAttribute(); attr; attr = attr->Next()) {
                    std::string name = attr->Name();
                    if (std::find(expected_attrs.begin(), expected_attrs.end(), name) == expected_attrs.end()) {
                        log<LOG_ERROR>(L"%1% || ERROR! Attribute [%2%] in the <allowlist> element is not expected.") % __func__ % name.c_str()  ;
                        log<LOG_ERROR>(L"%1% || -- Check spelling: allowed attributes are %2%") % __func__ % expected_attrs ;
                        throw std::invalid_argument(std::string("<allowlist> attribute not allowed : ") + name);
                    }
                }

                const char *variation_type = pAllowList->Attribute("type");
                const char *plot_name = pAllowList->Attribute("plotname");
                const char *binning = pAllowList->Attribute("binning");
                const char *knobs = pAllowList->Attribute("knobvals");
                const char *tags = pAllowList->Attribute("tag");
                const char *prior = pAllowList->Attribute("prior");
                const char *center = pAllowList->Attribute("center");



                m_mcgen_variation_type.push_back(variation_type);
                m_mcgen_variation_type_map[wt] = variation_type;
                m_mcgen_variation_allowlist.push_back(wt);
                if(prior) m_mcgen_variation_prior[wt] = std::strtof(prior, NULL);
                if(center) m_mcgen_variation_prior_centers[wt] = std::strtof(center, NULL);
                m_mcgen_variation_plotname_map[wt] = plot_name ? plot_name : wt;
                if(!binning || strcmp(binning, "reco") == 0) {
                    m_mcgen_variation_binning_map[wt] = -1;
                } else if(strcmp(binning, "truth") == 0) {
                    m_mcgen_variation_binning_map[wt] = -2;
                } else if(strncmp(binning, "other", 5) == 0) {
                    size_t l = strlen(binning);
                    bool all_numbers = true;
                    for(size_t i = 5; i < l; ++i) {
                        if(!isdigit(binning[i])) {
                            all_numbers = false;
                            break;
                        }
                    }
                    if(all_numbers) {
                        int binning_num;
                        sscanf(binning, "other%i", &binning_num);
                        m_mcgen_variation_binning_map[wt] = binning_num;
                    } else {
                        log<LOG_WARNING>(L"%1% || Unrecognized binning '%2%' for systematic %3%. Defaulting to reco bins.") 
                            % __func__ % binning % wt.c_str();
                        m_mcgen_variation_binning_map[wt] = -1;
                    }
                } else {
                    log<LOG_WARNING>(L"%1% || Unrecognized binning '%2%' for systematic %3%. Defaulting to reco bins.") 
                        % __func__ % binning % wt.c_str();
                    m_mcgen_variation_binning_map[wt] = -1;
                }
                if(knobs) {
                    std::vector<double> knobs_vec;
                    const char *c = knobs, *begin = NULL;
                    while(*c) {
                        if(begin && isspace(*c)) {
                            knobs_vec.push_back(strtod(begin, NULL));
                            begin = NULL;
                        } else if(!begin && !isspace(*c)) begin = c;
                        ++c;
                    }
                    knobs_vec.push_back(strtod(begin, NULL));
                    m_mcgen_variation_knobval_override[wt] = knobs_vec;
                }
                if(tags) {
                    std::vector<std::string> tags_vec;
                    const char *c = tags, *begin = NULL;
                    while(*c) {
                        if(begin && *c == ',') {
                            tags_vec.push_back(std::string(begin, c));
                            begin = NULL;
                        } else if(!begin && !isspace(*c)) begin = c;
                        ++c;
                    }
                    if(begin) tags_vec.push_back(std::string(begin, c));
                    m_mcgen_variation_tags[wt] = tags_vec;
                }
                log<LOG_DEBUG>(L"%1% || Allowlisting variations: %2%") % __func__ % wt.c_str() ;
                pAllowList = pAllowList->NextSiblingElement("allowlist");
            }

            tinyxml2::XMLElement *pDenyList = pList->FirstChildElement("denylist");
            while(pDenyList){
                std::string bt = std::string(pDenyList->GetText());
                m_mcgen_variation_denylist.push_back(bt); 
                log<LOG_DEBUG>(L"%1% || Denylisting variations: %2%") % __func__ % bt.c_str() ;
                pDenyList = pDenyList->NextSiblingElement("denylist");
            }
            pList = pList->NextSiblingElement("variation_list");
        }
    }

    // Correlations between systematics
    while (pCorrelations) {
        std::stringstream tup(pCorrelations->GetText());
        std::string s;
        std::vector<std::string> split;
        while (getline(tup, s, ' ')) split.push_back(s);

        if (split.size() != 3) {
            throw std::invalid_argument(std::string("Correlations should be formed as <Systematic A> <Systematic B> <Correlation>. Could not parse: ") + std::string(pCorrelations->GetText()));
        }

        m_mcgen_correlations.push_back(std::make_tuple(split[0], split[1], std::stof(split[2])));

        pCorrelations = pCorrelations->NextSiblingElement("correlation");
    }


    //weightMaps
    if(!pWeiMaps){
        log<LOG_DEBUG>(L"%1% || WeightMaps not set, all weights for all variations are 1 (individual branch weights still apply)") % __func__  ;
    }else{
        while(pWeiMaps){


            tinyxml2::XMLElement *pVariation;
            pVariation = pWeiMaps->FirstChildElement("variation");

            while(pVariation){


                //check for known attributes
                const std::vector<std::string> expected_attrs = {"pattern", "weight_formula", "use", "mode"};
                for (const tinyxml2::XMLAttribute* attr = pVariation->FirstAttribute(); attr; attr = attr->Next()) {
                    std::string name = attr->Name();
                    if (std::find(expected_attrs.begin(), expected_attrs.end(), name) == expected_attrs.end()) {
                        log<LOG_ERROR>(L"%1% || ERROR! Attribute [%2%] in the <variation> element is not expected.") % __func__ % name.c_str()  ;
                        log<LOG_ERROR>(L"%1% || -- Check spelling: allowed attributes are %2%") % __func__ % expected_attrs ;
                        throw std::invalid_argument(std::string("<variation> attribute not allowed : ") + name);
                    }
                }

                const char* w_pattern = pVariation->Attribute("pattern");
                const char* w_formula = pVariation->Attribute("weight_formula");
                const char* w_use = pVariation->Attribute("use");
                const char* w_mode = pVariation->Attribute("mode");

                if(w_pattern== NULL){
                    log<LOG_ERROR>(L"%1% || ERROR! No pattern passed for this variation in WeightMaps. @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);
                }else{
                    log<LOG_DEBUG>(L"%1% || Loading WeightMaps Variation Pattern: %2%") %__func__ % w_pattern;
                    m_mcgen_weightmaps_patterns.push_back(std::string(w_pattern));
                }


                if(w_formula== NULL){
                    log<LOG_WARNING>(L"%1% || Warning, No formula passed for this variation in WeightMaps. Setting to 1. Make sure this is wanted behaviour.") %__func__ ;
                    m_mcgen_weightmaps_formulas.push_back("1");
                }else{
                    log<LOG_DEBUG>(L"%1% || Loading WeightMaps Variation Formula: %2%") %__func__ % w_formula;
                    m_mcgen_weightmaps_formulas.push_back(std::string(w_formula));
                }

                if(w_use== NULL || std::string(w_use) == "true"){
                    m_mcgen_weightmaps_uses.push_back(true);
                }else{
                    m_mcgen_weightmaps_uses.push_back(false);
                }

                if(w_mode== NULL){
                    log<LOG_WARNING>(L"%1% || Warning, No mode passed for this variaiton in  WeightMaps. Assuming default multisim.  Make sure this is wanted behaviour.") %__func__ ;
                    m_mcgen_weightmaps_mode.push_back("multisim");
                }else{

                    std::string mode = std::string(w_mode);
                    if(mode=="multisim" || mode=="minmax"){
                        m_mcgen_weightmaps_mode.push_back(mode);
                    }else{
                        log<LOG_ERROR>(L"%1% || ERROR! The mode passed in is %4% but only allowed is multisim or minmax. @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__ % w_mode;
                        log<LOG_ERROR>(L"Terminating.");
                        exit(EXIT_FAILURE);
                    }
                }

                pVariation = pVariation->NextSiblingElement("variation");
            }

            pWeiMaps=pWeiMaps->NextSiblingElement("WeightMaps");
        }
    }



    while(pShapeOnlyMap){

        log<LOG_WARNING>(L"%1% || Warning!  Setting up for shape-only covariance matrix generation. MAKE SURE this is what you want if you're generating covariance matrix!!!") % __func__;

        const std::vector<std::string> expected_attrs = {"name", "use"};
        for (const tinyxml2::XMLAttribute* attr = pShapeOnlyMap->FirstAttribute(); attr; attr = attr->Next()) {
            std::string name = attr->Name();
            if (std::find(expected_attrs.begin(), expected_attrs.end(), name) == expected_attrs.end()) {
                log<LOG_ERROR>(L"%1% || ERROR! Attribute [%2%] in the <shapeonlymap> element is not expected.") % __func__ % name.c_str()  ;
                log<LOG_ERROR>(L"%1% || -- Check spelling: allowed attributes are %2%") % __func__ % expected_attrs ;
                throw std::invalid_argument(std::string("<shapeonlymap> attribute not allowed : ") + name);
            }
        }

        std::string pshapeonly_systematic_name = std::string(pShapeOnlyMap->Attribute("name"));
        const char* pshapeonly_systematic_use = pShapeOnlyMap->Attribute("use");
        bool pshapeonly_systematic_use_bool = true;

        if(pshapeonly_systematic_use == NULL || std::string(pshapeonly_systematic_use) == "true"){
            std::cout << "" << pshapeonly_systematic_name << std::endl;
            log<LOG_DEBUG>(L"%1% || Setting up shape-only covariance matrix for systematic: %2% ") % __func__ % pshapeonly_systematic_name.c_str();

        }else if(std::string(pshapeonly_systematic_use) == "false"){
            log<LOG_DEBUG>(L"%1% || Setting up shape-only covariance matrix for systematic: %2% ? False ") % __func__ % pshapeonly_systematic_name.c_str();
            pshapeonly_systematic_use_bool = false;
        }else{
            log<LOG_WARNING>(L"%1% || INVALID argument received for Attribute use of ShapeOnlyUncertainty element for systematic: %2% . Default it to true ") % __func__ % pshapeonly_systematic_name.c_str();
        }

        tinyxml2::XMLElement *pSubchannel;
        pSubchannel = pShapeOnlyMap->FirstChildElement("subchannel");	

        while(pshapeonly_systematic_use_bool && pSubchannel){

            std::string pshapeonly_subchannel_name = std::string(pSubchannel->Attribute("name"));
            std::string pshapeonly_subchannel_use = std::string(pSubchannel->Attribute("use"));

            if(pshapeonly_subchannel_use == "false" ){
                log<LOG_DEBUG>(L"%1% || Not include subchannel: %2% for shape-only covariance matrix") % __func__ % pshapeonly_subchannel_name.c_str();
            }else{
                log<LOG_DEBUG>(L"%1% || Include subchannel: %2% for shape-only covariance matrix") % __func__ % pshapeonly_subchannel_name.c_str();
                m_mcgen_shapeonly_listmap[pshapeonly_systematic_name].push_back(pshapeonly_subchannel_name);
            }

            pSubchannel = pSubchannel->NextSiblingElement("subchannel");
        }

        pShapeOnlyMap = pShapeOnlyMap->NextSiblingElement("ShapeOnlyUncertainty");

    }


    while(pSpec){
        const char* swrite_out = pSpec->Attribute("writeout");
        const char* swrite_out_tag = pSpec->Attribute("writeout_tag");
        const char* sform_matrix = pSpec->Attribute("form_matrix");	

        if( std::string(swrite_out) == "true"){
            m_write_out_variation = true;
            log<LOG_DEBUG>(L"%1% || Setting up to write out spectra for variations") % __func__;
        }

        if(m_write_out_variation){
            if(swrite_out_tag) 
                m_write_out_tag = std::string(swrite_out_tag);
        }

        if( std::string(sform_matrix) == "false"){
            m_form_covariance = false;
            log<LOG_DEBUG>(L"%1% || Explicitly to ask to not generate covariance matrix") % __func__;
        }
        pSpec = pSpec->NextSiblingElement("varied_spectrum");
    }


    //**** Model Loading ****

    tinyxml2::XMLElement *pModel;
    pModel   = doc.FirstChildElement("model");

    if(pModel){
        // Read in how many bins this channel uses

        const std::vector<std::string> expected_attrs = {"tag","name","index"};
        for (const tinyxml2::XMLAttribute* attr = pModel->FirstAttribute(); attr; attr = attr->Next()) {
            std::string name = attr->Name();
            if (std::find(expected_attrs.begin(), expected_attrs.end(), name) == expected_attrs.end()) {
                log<LOG_ERROR>(L"%1% || ERROR! Attribute [%2%] in the <variation> element is not expected.") % __func__ % name.c_str()  ;
                log<LOG_ERROR>(L"%1% || -- Check spelling: allowed attributes are %2%") % __func__ % expected_attrs ;
                throw std::invalid_argument(std::string("<variation> attribute not allowed : ") + name);
            }
        }

        const char* model_tag= pModel->Attribute("tag");
        if(model_tag==NULL){
            m_model_tag = "null";
        }else{
            m_model_tag  = model_tag;
        }

        // Now loop over all this models rules

        tinyxml2::XMLElement *pModelRule;
        pModelRule = pModel->FirstChildElement("rule");
        while(pModelRule){
            const char* model_rule_name= pModelRule->Attribute("name");
            if(model_rule_name==NULL){
                log<LOG_ERROR>(L"%1% || ERROR: Model Rules need a name in xml.@ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }else{
                m_model_rule_names.push_back(model_rule_name);
            }


            const char* model_rule_index= pModelRule->Attribute("index");
            if(model_rule_index==NULL){
                log<LOG_ERROR>(L"%1% || ERROR: Model Rules need an index in xml.@ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }else{
                m_model_rule_index.push_back(strtod(model_rule_index, &end));
            }

            log<LOG_DEBUG>(L"%1% || Model Rule Name :  %2% and index %3% ") % __func__ % m_model_rule_names.back().c_str() % m_model_rule_index.back()  ;
            pModelRule = pModelRule->NextSiblingElement("rule");
        }
    }//end model

    for(size_t i = 0 ; i<m_mcgen_variation_type.size(); ++i){
        if(m_mcgen_variation_type[i] == "spline"){
            m_num_variation_type_spline+=1;
        }

        else if(m_mcgen_variation_type[i] == "covariance"){
            m_num_variation_type_covariance+=1;
        }

        else if(m_mcgen_variation_type[i] == "flat"){
            m_num_variation_type_flat+=1;
        }
        else if(m_mcgen_variation_type[i] == "norm"){
            m_num_variation_type_norm+=1;
        }else if(m_mcgen_variation_type[i] == "mcstat"){
            m_mcgen_variation_allowlist[i] = "mcstat";
            m_mcgen_variation_type_map["mcstat"] = "mcstat";
            m_use_mcstats = true;
        }

    }

    log<LOG_INFO>(L"%1% || num_variation_type_covariance: %2% ") % __func__ % m_num_variation_type_covariance;
    log<LOG_INFO>(L"%1% || num_variation_type_flat: %2% ") % __func__ % m_num_variation_type_flat;
    log<LOG_INFO>(L"%1% || num_variation_type_norm: %2% ") % __func__ % m_num_variation_type_norm;
    log<LOG_INFO>(L"%1% || num_variation_type_spline: %2% ") % __func__ % m_num_variation_type_spline; 
    if(m_use_mcstats){
        log<LOG_INFO>(L"%1% || Using MC intrinsic stat uncertainty. ") % __func__  ;
    }else{
        log<LOG_INFO>(L"%1% || Not using MC intrinsic stat uncertainty. Check this is what you want. ") % __func__ ; 
    }



    this->CalcTotalBins();

    log<LOG_INFO>(L"%1% || Checking number of Mode/Detector/Channel/Subchannels and BINs") % __func__;
    log<LOG_INFO>(L"%1% || num_modes: %2% ") % __func__ % m_num_modes;
    log<LOG_INFO>(L"%1% || num_detectors: %2% ") % __func__ % m_num_detectors;
    log<LOG_INFO>(L"%1% || num_channels: %2% ") % __func__ % m_num_channels;
    for(size_t i = 0 ; i!=m_num_channels; ++i){
        log<LOG_INFO>(L"%1% || num of subchannels: %2% ") % __func__ % m_num_subchannels[i];
        log<LOG_INFO>(L"%1% || num of bins: %2% ") % __func__ % m_channel_num_bins[i];

    }
    log<LOG_INFO>(L"%1% || num_bins_detector_block: %2%") % __func__ % m_num_bins_detector_block;
    log<LOG_INFO>(L"%1% || num_truebins_detector_block: %2%") % __func__ % m_num_truebins_detector_block;
    log<LOG_INFO>(L"%1% || num_bins_detector_block_collapsed: %2%") % __func__ % m_num_bins_detector_block_collapsed;
    log<LOG_INFO>(L"%1% || num_bins_mode_block: %2%") % __func__ % m_num_bins_mode_block;
    log<LOG_INFO>(L"%1% || num_true_bins_mode_block: %2%") % __func__ % m_num_truebins_mode_block;
    log<LOG_INFO>(L"%1% || num_bins_mode_block_collapsed: %2%") % __func__ % m_num_bins_mode_block_collapsed;
    log<LOG_INFO>(L"%1% || num_bins_total: %2%") % __func__ % m_num_bins_total;
    log<LOG_INFO>(L"%1% || num_true_bins_total: %2%") % __func__ % m_num_truebins_total;
    log<LOG_INFO>(L"%1% || num_bins_total_collapsed: %2%") % __func__ % m_num_bins_total_collapsed;


    log<LOG_INFO>(L"%1% || Done reading the xmls") % __func__;
    return 0;
}


void PROconfig::CalcTotalBins(){
    this->remove_unused_channel();

    log<LOG_INFO>(L"%1% || Calculating number of bins involved") % __func__;
    for(size_t i = 0; i != m_num_channels; ++i){
        m_num_bins_detector_block += m_num_subchannels[i]*m_channel_num_bins[i];
        m_num_truebins_detector_block += m_num_subchannels[i]*m_channel_num_truebins[i];
        m_num_bins_detector_block_collapsed += m_channel_num_bins[i];
        for(size_t io = 0; io < m_num_other_vars; ++io) {
            m_num_other_bins_detector_block[io] += m_num_subchannels[i]*m_channel_num_other_bins[i][io];
            m_num_other_bins_detector_block_collapsed[io] += m_channel_num_other_bins[i][io];
        }
    }

    m_num_bins_mode_block = m_num_bins_detector_block *  m_num_detectors;
    m_num_truebins_mode_block = m_num_truebins_detector_block *  m_num_detectors;
    m_num_bins_mode_block_collapsed = m_num_bins_detector_block_collapsed * m_num_detectors;
    for(size_t io = 0; io < m_num_other_vars; ++io) {
        m_num_other_bins_mode_block[io] = m_num_other_bins_detector_block[io] * m_num_detectors;
        m_num_other_bins_mode_block_collapsed[io] = m_num_other_bins_detector_block_collapsed[io] * m_num_detectors;
    }

    m_num_bins_total = m_num_bins_mode_block * m_num_modes;
    m_num_truebins_total = m_num_truebins_mode_block * m_num_modes;
    m_num_bins_total_collapsed = m_num_bins_mode_block_collapsed * m_num_modes;
    for(size_t io = 0; io < m_num_other_vars; ++io) {
        m_num_other_bins_total[io] = m_num_other_bins_mode_block[io] * m_num_modes;
        m_num_other_bins_total_collapsed[io] = m_num_other_bins_mode_block_collapsed[io] * m_num_modes;
    }

    this->generate_index_map();
    return;
}

size_t PROconfig::GetSubchannelIndex(const std::string& fullname) const{
    auto pos_iter = m_map_fullname_subchannel_index.find(fullname);
    if(pos_iter == m_map_fullname_subchannel_index.end()){
        log<LOG_ERROR>(L"%1% || Subchannel name: %2% does not exist in the indexing map!") % __func__ % fullname.c_str();
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }
    return pos_iter->second;
}

size_t PROconfig::GetChannelIndex(size_t subchannel_index) const{
    size_t index = this->find_equal_index(m_vec_subchannel_index, subchannel_index);
    return m_vec_channel_index[index];
}

size_t PROconfig::GetGlobalBinStart(size_t subchannel_index) const{
    size_t index = this->find_equal_index(m_vec_subchannel_index, subchannel_index);
    return m_vec_global_reco_index_start[index];
}

size_t PROconfig::GetCollapsedGlobalBinStart(size_t channel_index) const{
    if(channel_index >= m_num_channels*m_num_modes*m_num_detectors) {
        size_t tot= m_num_channels*m_num_modes*m_num_detectors;
        log<LOG_ERROR>(L"%1% || Requested bin start of channel %2%, but only %3% total channels are known (chat*mode*det).")
            % __func__ % channel_index % tot;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }
    size_t index = 0;
    for(size_t i = 0; i < channel_index; ++i) index += m_channel_num_bins[GetLocalChannelIndex(i)];
    return index;
}

size_t PROconfig::GetGlobalTrueBinStart(size_t subchannel_index) const{
    size_t index = this->find_equal_index(m_vec_subchannel_index, subchannel_index);
    return m_vec_global_true_index_start[index];
}

size_t PROconfig::GetGlobalOtherBinStart(size_t subchannel_index, size_t other_index) const{
    size_t index = this->find_equal_index(m_vec_subchannel_index, subchannel_index);
    return m_vec_global_other_index_start[other_index][index];
}

size_t PROconfig::GetCollapsedGlobalOtherBinStart(size_t channel_index, size_t other_index) const{
    if(channel_index >= m_num_channels*m_num_modes*m_num_detectors) {
        size_t tot= m_num_channels*m_num_modes*m_num_detectors;
        log<LOG_ERROR>(L"%1% || Requested bin start of channel %2%, but only %3% total channels are known (chat*mode*det).")
            % __func__ % channel_index % tot;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }    
    size_t index = 0;
    for(size_t i = 0; i < channel_index; ++i) index += m_channel_num_other_bins[GetLocalChannelIndex(i)][other_index];
    return index;
}

size_t PROconfig::GetSubchannelIndexFromGlobalBin(size_t global_reco_index) const {
    size_t index = this->find_less_or_equal_index(m_vec_global_reco_index_start, global_reco_index); 
    return m_vec_subchannel_index[index];
}

size_t PROconfig::GetSubchannelIndexFromGlobalTrueBin(size_t global_trueindex) const{
    size_t index = this->find_less_or_equal_index(m_vec_global_true_index_start, global_trueindex);
    return m_vec_subchannel_index[index];
}

size_t PROconfig::GetLocalChannelIndex(size_t global_channel_index) const{
    return global_channel_index%m_num_channels; 
}

const std::vector<float>& PROconfig::GetChannelBinEdges(size_t channel_index) const{

    if(channel_index >= m_num_channels*m_num_modes*m_num_detectors) {
        size_t tot= m_num_channels*m_num_modes*m_num_detectors;
        log<LOG_ERROR>(L"%1% || Given channel index: %2% is out of bound") % __func__ % channel_index;
        log<LOG_ERROR>(L"%1% || Total number of channels : %2%") % __func__ % tot;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }

    return m_channel_bin_edges[ GetLocalChannelIndex(channel_index)];
}

size_t PROconfig::GetChannelNTrueBins(size_t channel_index) const{
    if(channel_index >= m_num_channels*m_num_modes*m_num_detectors) {
        size_t tot= m_num_channels*m_num_modes*m_num_detectors;
        log<LOG_ERROR>(L"%1% || Given channel index: %2% is out of bound") % __func__ % channel_index;
        log<LOG_ERROR>(L"%1% || Total number of channels : %2%") % __func__ % tot;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }
    return m_channel_num_truebins[GetLocalChannelIndex(channel_index)];
}

const std::vector<float>& PROconfig::GetChannelTrueBinEdges(size_t channel_index) const{

    if(channel_index >= m_num_channels*m_num_modes*m_num_detectors) {
        size_t tot= m_num_channels*m_num_modes*m_num_detectors;
        log<LOG_ERROR>(L"%1% || Given channel index: %2% is out of bound") % __func__ % channel_index;
        log<LOG_ERROR>(L"%1% || Total number of channels : %2%") % __func__ % tot;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }

    return m_channel_truebin_edges[GetLocalChannelIndex(channel_index)];
}

size_t PROconfig::GetChannelNOtherBins(size_t channel_index, size_t other_index) const{

    if(channel_index >= m_num_channels*m_num_modes*m_num_detectors) {
        size_t tot= m_num_channels*m_num_modes*m_num_detectors;
        log<LOG_ERROR>(L"%1% || Given channel index: %2% is out of bound") % __func__ % channel_index;
        log<LOG_ERROR>(L"%1% || Total number of channels : %2%") % __func__ % tot;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }
    return m_channel_num_other_bins[GetLocalChannelIndex(channel_index)][other_index];
}

const std::vector<float>& PROconfig::GetChannelOtherBinEdges(size_t channel_index, size_t other_index) const{

    if(channel_index >= m_num_channels*m_num_modes*m_num_detectors) {
        size_t tot= m_num_channels*m_num_modes*m_num_detectors;
        log<LOG_ERROR>(L"%1% || Given channel index: %2% is out of bound") % __func__ % channel_index;
        log<LOG_ERROR>(L"%1% || Total number of channels : %2%") % __func__ % tot;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }    return m_channel_other_bin_edges[GetLocalChannelIndex(channel_index)][other_index];
}



//------------ Start of private function ------------------
//------------ Start of private function ------------------
//------------ Start of private function ------------------

void PROconfig::remove_unused_channel(){

    log<LOG_INFO>(L"%1% || Remove any used channels and subchannels...") % __func__;

    m_num_modes = std::count(m_mode_bool.begin(), m_mode_bool.end(), true);
    m_num_detectors = std::count(m_detector_bool.begin(), m_detector_bool.end(), true);
    m_num_channels = std::count(m_channel_bool.begin(), m_channel_bool.end(), true);

    //update mode-info
    if(m_num_modes != m_mode_bool.size()){
        log<LOG_DEBUG>(L"%1% || Found unused modes!! Clean it up...") % __func__;
        std::vector<std::string> temp_mode_names(m_num_modes), temp_mode_plotnames(m_num_modes);
        for(size_t i = 0, mode_index = 0; i != m_mode_bool.size(); ++i){
            if(m_mode_bool[i]){
                temp_mode_names[mode_index] = m_mode_names[i];
                temp_mode_plotnames[mode_index] = m_mode_plotnames[i];

                ++mode_index;
            }    
        }
        m_mode_names = temp_mode_names;
        m_mode_plotnames = temp_mode_plotnames;
    }

    ///update detector-info
    if(m_num_detectors != m_detector_bool.size()){
        log<LOG_DEBUG>(L"%1% || Found unused detectors!! Clean it up...") % __func__;
        std::vector<std::string> temp_detector_names(m_num_detectors), temp_detector_plotnames(m_num_detectors);
        for(size_t i = 0, det_index = 0; i != m_detector_bool.size(); ++i){
            if(m_detector_bool[i]){
                temp_detector_names[det_index] = m_detector_names[i];
                temp_detector_plotnames[det_index] = m_detector_plotnames[i];

                ++det_index;
            }
        }
        m_detector_names = temp_detector_names;
        m_detector_plotnames = temp_detector_plotnames;
    }

    if(m_num_channels != m_channel_bool.size()){
        log<LOG_DEBUG>(L"%1% || Found unused channels!! Clean the messs up...") % __func__;

        //update channel-related info
        std::vector<size_t> temp_channel_num_bins(m_num_channels, 0);
        std::vector<std::vector<float>> temp_channel_bin_edges(m_num_channels, std::vector<float>());
        std::vector<std::vector<float>> temp_channel_bin_widths(m_num_channels, std::vector<float>());

        std::vector<size_t> temp_channel_num_truebins(m_num_channels, 0);
        std::vector<std::vector<float>> temp_channel_truebin_edges(m_num_channels, std::vector<float>());
        std::vector<std::vector<float>> temp_channel_truebin_widths(m_num_channels, std::vector<float>());

        std::vector<std::vector<size_t>> temp_channel_num_other_bins(m_num_channels);
        std::vector<std::vector<std::vector<float>>> temp_channel_other_bin_edges(m_num_channels);
        std::vector<std::vector<std::vector<float>>> temp_channel_other_bin_widths(m_num_channels);

        std::vector<std::string> temp_channel_names(m_num_channels);
        std::vector<std::string> temp_channel_plotnames(m_num_channels);
        std::vector<std::string> temp_channel_units(m_num_channels);
        std::vector<std::vector<std::string>> temp_channel_other_units(m_num_channels);
        for(size_t i=0, chan_index = 0; i< m_channel_bool.size(); ++i){
            if(m_channel_bool[i]){
                temp_channel_num_bins[chan_index] = m_channel_num_bins[i];
                temp_channel_bin_edges[chan_index] = m_channel_bin_edges[i];
                temp_channel_bin_widths[chan_index] = m_channel_bin_widths[i];

                temp_channel_num_truebins[chan_index] = m_channel_num_truebins[i];
                temp_channel_truebin_edges[chan_index] = m_channel_truebin_edges[i];
                temp_channel_truebin_widths[chan_index] = m_channel_truebin_widths[i];

                temp_channel_names[chan_index] = m_channel_names[i];
                temp_channel_plotnames[chan_index] = m_channel_plotnames[i];
                temp_channel_units[chan_index] = m_channel_units[i];

                temp_channel_num_other_bins[chan_index] = m_channel_num_other_bins[i];
                temp_channel_other_bin_edges[chan_index] = m_channel_other_bin_edges[i];
                temp_channel_other_bin_widths[chan_index] = m_channel_other_bin_widths[i];
                temp_channel_other_units[chan_index] = m_channel_other_units[i];

                ++chan_index;
            }
        }

        m_channel_num_bins = temp_channel_num_bins;
        m_channel_bin_edges = temp_channel_bin_edges;
        m_channel_bin_widths = temp_channel_bin_widths;
        m_channel_num_truebins = temp_channel_num_truebins;
        m_channel_truebin_edges = temp_channel_truebin_edges;
        m_channel_truebin_widths = temp_channel_truebin_widths;
        m_channel_names = temp_channel_names;
        m_channel_plotnames = temp_channel_plotnames;
        m_channel_units = temp_channel_units;
    }

    {

        //update subchannel-related info
        m_num_subchannels.resize(m_num_channels);
        std::vector<std::vector<std::string >> temp_subchannel_names(m_num_channels), temp_subchannel_plotnames(m_num_channels), temp_subchannel_colors(m_num_channels);
        std::vector<std::vector<size_t >> temp_subchannel_datas(m_num_channels), temp_subchannel_model_rules(m_num_channels);
        for(size_t i=0, chan_index = 0; i< m_channel_bool.size(); ++i){
            if(m_channel_bool.at(i)){
                m_num_subchannels[chan_index]= 0;
                for(size_t j=0; j< m_subchannel_bool[i].size(); ++j){ 
                    if(m_subchannel_bool[i][j]){
                        ++m_num_subchannels[chan_index];
                        temp_subchannel_names[chan_index].push_back(m_subchannel_names[i][j]);
                        temp_subchannel_plotnames[chan_index].push_back(m_subchannel_plotnames[i][j]);	
                        temp_subchannel_colors[chan_index].push_back(m_subchannel_colors[i][j]);	
                        temp_subchannel_datas[chan_index].push_back(m_subchannel_datas[i][j]);

                    }
                }


                ++chan_index;
            }
        }

        m_subchannel_names = temp_subchannel_names;
        m_subchannel_plotnames = temp_subchannel_plotnames;
        m_subchannel_colors = temp_subchannel_colors;
        m_subchannel_datas = temp_subchannel_datas;

    }

    //grab list of fullnames used.
    log<LOG_DEBUG>(L"%1% || Sweet, now generating fullnames of all channels used...") % __func__;
    m_fullnames.clear();
    for(size_t im = 0; im < m_num_modes; im++){
        for(size_t id =0; id < m_num_detectors; id++){
            for(size_t ic = 0; ic < m_num_channels; ic++){
                for(size_t sc = 0; sc < m_num_subchannels.at(ic); sc++){

                    std::string temp_name  = m_mode_names.at(im) +"_" +m_detector_names.at(id)+"_"+m_channel_names.at(ic)+"_"+m_subchannel_names.at(ic).at(sc);
                    log<LOG_INFO>(L"%1% || fullname of subchannel: %2% ") % __func__ % temp_name.c_str();
                    m_fullnames.push_back(temp_name);
                }
            }
        }
    }

    this->remove_unused_files();
    return;
}


void PROconfig::remove_unused_files(){


    //ignore any files not associated with used channels 
    //clean up branches not associated with used channels 
    size_t num_all_branches = 0;
    for(auto& br : m_branch_variables)
        num_all_branches += br.size();

    log<LOG_DEBUG>(L"%1% || Deubg: BRANCH VARIABLE size: %2% ") % __func__ % m_branch_variables.size();;
    log<LOG_DEBUG>(L"%1% || Check for any files associated with unused subchannels ....") % __func__;
    log<LOG_DEBUG>(L"%1% || Total number of %2% active subchannels..") % __func__ % m_fullnames.size();
    log<LOG_DEBUG>(L"%1% || Total number of %2% branches listed in the xml....") % __func__ % num_all_branches;

    //update file info
    //loop over all branches, and ignore ones not used  
    if(num_all_branches != m_fullnames.size()){

        std::unordered_set<std::string> set_all_names(m_fullnames.begin(), m_fullnames.end());

        std::vector<std::string> temp_tree_name;
        std::vector<std::string> temp_file_name;
        std::vector<long int> temp_maxevents;
        std::vector<float> temp_pot;
        std::vector<float> temp_scale;
        std::vector<int> temp_numfriends;
        std::vector<bool> temp_fake;
        std::map<std::string,std::vector<std::string>> temp_file_friend_map;
        std::map<std::string,std::vector<std::string>> temp_file_friend_treename_map;
        std::vector<std::vector<std::string>> temp_additional_weight_name;
        std::vector<std::vector<bool>> temp_additional_weight_bool;
        std::vector<std::vector<std::shared_ptr<BranchVariable>>> temp_branch_variables;
        std::vector<std::vector<std::string>> temp_eventweight_branch_names;
        std::vector<std::vector<int>> temp_eventweight_branch_syst;

        for(size_t i = 0; i != m_mcgen_file_name.size(); ++i){
            log<LOG_DEBUG>(L"%1% || Check on @%2% th file: %3%...") % __func__ % i % m_mcgen_file_name[i].c_str();
            bool this_file_needed = false;

            std::vector<std::string> this_file_additional_weight_name;
            std::vector<bool> this_file_additional_weight_bool;
            std::vector<std::shared_ptr<BranchVariable>> this_file_branch_variables;
            std::vector<std::string> this_file_eventweight_branch_names;
            std::vector<int> this_file_eventweight_branch_syst;
            for(size_t j = 0; j != m_branch_variables[i].size(); ++j){

                if(set_all_names.find(m_branch_variables[i][j]->associated_hist) == set_all_names.end()){
                }else{

                    set_all_names.erase(m_branch_variables[i][j]->associated_hist);
                    this_file_needed = true;

                    this_file_additional_weight_name.push_back(m_mcgen_additional_weight_name[i][j]);
                    this_file_additional_weight_bool.push_back(m_mcgen_additional_weight_bool[i][j]);
                    this_file_branch_variables.push_back(m_branch_variables[i][j]);
                    this_file_eventweight_branch_names.push_back(m_mcgen_eventweight_branch_names[i][j]);
                    this_file_eventweight_branch_syst.push_back(m_mcgen_eventweight_branch_syst[i][j]);
                }
            }

            if(this_file_needed){
                log<LOG_DEBUG>(L"%1% || This file is active, keep it!") % __func__ ;
                temp_tree_name.push_back(m_mcgen_tree_name[i]);
                temp_file_name.push_back(m_mcgen_file_name[i]);
                temp_maxevents.push_back(m_mcgen_maxevents[i]);
                temp_pot.push_back(m_mcgen_pot[i]);
                temp_scale.push_back(m_mcgen_scale[i]);
                temp_numfriends.push_back(m_mcgen_numfriends[i]);
                temp_fake.push_back(m_mcgen_fake[i]);
                temp_file_friend_map[m_mcgen_file_name[i]] = m_mcgen_file_friend_map[m_mcgen_file_name[i]];		
                temp_file_friend_treename_map[m_mcgen_file_name[i]] = m_mcgen_file_friend_treename_map[m_mcgen_file_name[i]];

                temp_additional_weight_name.push_back(this_file_additional_weight_name);
                temp_additional_weight_bool.push_back(this_file_additional_weight_bool);
                temp_branch_variables.push_back(this_file_branch_variables);
                temp_eventweight_branch_names.push_back(this_file_eventweight_branch_names);
            }
        }

        m_mcgen_file_name = temp_file_name;
        m_mcgen_tree_name = temp_tree_name;
        m_mcgen_maxevents = temp_maxevents;
        m_mcgen_pot = temp_pot;
        m_mcgen_scale = temp_scale;
        m_mcgen_numfriends = temp_numfriends;
        m_mcgen_fake = temp_fake;
        m_mcgen_file_friend_map =temp_file_friend_map;
        m_mcgen_file_friend_treename_map = temp_file_friend_treename_map;
        m_mcgen_additional_weight_name = temp_additional_weight_name;
        m_mcgen_additional_weight_bool = temp_additional_weight_bool;
        m_branch_variables = temp_branch_variables;
        m_mcgen_eventweight_branch_names = temp_eventweight_branch_names;
        m_mcgen_eventweight_branch_syst = temp_eventweight_branch_syst;
    }

    m_num_mcgen_files = m_mcgen_file_name.size();
    log<LOG_DEBUG>(L"%1% || Finish cleaning up, total of %2% files left.") % __func__ % m_num_mcgen_files;
    return;
}

size_t PROconfig::find_equal_index(const std::vector<size_t>& input_vec, size_t val) const{
    auto pos_iter = std::lower_bound(input_vec.begin(), input_vec.end(), val);
    if(pos_iter == input_vec.end() || (*pos_iter) != val){
        log<LOG_ERROR>(L"%1% || Input value: %2% does not exist in the vector! Max element available: %3%") % __func__ % val % input_vec.back();
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }
    size_t index = pos_iter - input_vec.begin();
    return index;
}


size_t PROconfig::find_less_or_equal_index(const std::vector<size_t>& input_vec, size_t val) const{
    auto pos_iter = std::lower_bound(input_vec.begin(), input_vec.end(), val);
    if(pos_iter == input_vec.end() || (*pos_iter) != val){
        return pos_iter - input_vec.begin() - 1;
    } else {
        return pos_iter - input_vec.begin();
    }
    return -1;
}


void PROconfig::generate_index_map(){
    log<LOG_INFO>(L"%1% || Generate map between subchannel and global indices..") % __func__;
    m_map_fullname_subchannel_index.clear();
    m_vec_subchannel_index.clear();
    m_vec_channel_index.clear();
    m_vec_global_reco_index_start.clear();
    m_vec_global_true_index_start.clear();
    m_vec_global_other_index_start.clear();

    for(size_t io = 0; io < m_num_other_vars; ++io) {
        m_vec_global_other_index_start.emplace_back();
    }

    size_t global_subchannel_index = 0;
    for(size_t im = 0; im < m_num_modes; im++){

        size_t mode_bin_start = im*m_num_bins_mode_block;
        size_t mode_truebin_start = im*m_num_truebins_mode_block;
        std::vector<size_t> mode_other_start;
        for(size_t io = 0; io < m_num_other_vars; ++io) {
            mode_other_start.push_back(im*m_num_other_bins_mode_block[io]);
        }

        for(size_t id =0; id < m_num_detectors; id++){

            size_t detector_bin_start = id*m_num_bins_detector_block;
            size_t channel_bin_start = 0;

            size_t detector_truebin_start = id*m_num_truebins_detector_block;
            size_t channel_truebin_start = 0;

            std::vector<size_t> detector_other_start;
            std::vector<size_t> channel_other_start;
            for(size_t io = 0; io < m_num_other_vars; ++io) {
                detector_other_start.push_back(id*m_num_other_bins_detector_block[io]);
                channel_other_start.push_back(0);
            }

            for(size_t ic = 0; ic < m_num_channels; ic++){
                for(size_t sc = 0; sc < m_num_subchannels[ic]; sc++){

                    std::string temp_name  = m_mode_names[im] +"_" +m_detector_names[id]+"_"+m_channel_names[ic]+"_"+m_subchannel_names[ic][sc];
                    size_t global_bin_index = mode_bin_start + detector_bin_start + channel_bin_start + sc*m_channel_num_bins[ic];
                    size_t global_truebin_index = mode_truebin_start + detector_truebin_start + channel_truebin_start + sc*m_channel_num_truebins[ic];

                    m_map_fullname_subchannel_index[temp_name] = global_subchannel_index;
                    m_vec_subchannel_index.push_back(global_subchannel_index);
                    m_vec_channel_index.push_back(ic);
                    m_vec_global_reco_index_start.push_back(global_bin_index);
                    m_vec_global_true_index_start.push_back(global_truebin_index);

                    for(size_t io = 0; io < m_num_other_vars; ++io) {
                        size_t global_other_index = mode_other_start[io] + detector_other_start[io] + channel_other_start[io] + sc*m_channel_num_other_bins[ic][io];
                        m_vec_global_other_index_start[io].push_back(global_other_index);
                    }

                    ++global_subchannel_index;
                }
                channel_bin_start += m_channel_num_bins[ic]*m_num_subchannels[ic];
                channel_truebin_start += m_channel_num_truebins[ic]*m_num_subchannels[ic];
                for(size_t io = 0; io < m_num_other_vars; ++io) {
                    channel_other_start[io] += m_channel_num_other_bins[ic][io]*m_num_subchannels[ic];
                }
            }
        }
    }
    return;
}



size_t PROconfig::find_global_subchannel_index_from_global_bin(size_t global_index, const std::vector<size_t>& num_subchannel_in_channel, const std::vector<size_t>& num_bins_in_channel, size_t num_channels, size_t num_bins_total) const{

    //check for out of bound
    if( global_index >= num_bins_total){
        log<LOG_ERROR>(L"%1% || Given index: %2% is out of bound") % __func__ % global_index;
        log<LOG_ERROR>(L"%1% || Total number of bins : %2%") % __func__ % num_bins_total;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }

    // get number of bins per detector block 
    size_t num_bins_per_detector_block = 0;
    for( size_t ic = 0; ic != num_channels; ++ic)
        num_bins_per_detector_block += num_subchannel_in_channel[ic] * num_bins_in_channel[ic];
    if(num_bins_per_detector_block == 0){
        log<LOG_ERROR>(L"%1% || There is zero bins for each detector!! Provided global bin index: %2% ") % __func__ % global_index;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }

    // get number of subchannels in detector block  
    size_t num_subchannel_in_detector_block = std::accumulate(num_subchannel_in_channel.begin(), num_subchannel_in_channel.end(), 0);
    size_t subchannel_index = (global_index / num_bins_per_detector_block) * num_subchannel_in_detector_block;

    global_index %= num_bins_per_detector_block;   //get the index inside a block 
    //check for each channel
    for( size_t ic = 0; ic != num_channels; ++ic){
        size_t total_bins_in_channel = num_subchannel_in_channel[ic] * num_bins_in_channel[ic];
        if(global_index >= total_bins_in_channel){
            global_index -= total_bins_in_channel;
            subchannel_index += num_subchannel_in_channel[ic];
        }
        else{
            subchannel_index += global_index / num_bins_in_channel[ic];
            break;
        }

    }
    return subchannel_index;
}

void PROconfig::construct_collapsing_matrix(){

    collapsing_matrix = Eigen::MatrixXf::Zero(m_num_bins_total, m_num_bins_total_collapsed);
    log<LOG_INFO>(L"%1% || Creating Collapsing Matrix. m_num_bins_total, m_num_bins_total_collapsed:  %2%  %3%") % __func__ % m_num_bins_total % m_num_bins_total_collapsed;

    //construct the matrix by detector block
    Eigen::MatrixXf block_collapser = Eigen::MatrixXf::Zero(m_num_bins_detector_block, m_num_bins_detector_block_collapsed);

    size_t channel_row_start = 0, channel_col_start = 0;
    for(size_t ic =0; ic != m_num_channels; ++ic){

        //first, build matrix for each channel block
        size_t total_num_bins_channel = m_num_subchannels[ic] * m_channel_num_bins[ic];

        Eigen::MatrixXf channel_collapser = Eigen::MatrixXf::Zero(total_num_bins_channel, m_channel_num_bins[ic]);
        for(size_t col = 0; col != m_channel_num_bins[ic]; ++col){
            for(size_t subch = 0; subch != m_num_subchannels[ic]; ++subch){
                size_t row = subch * m_channel_num_bins[ic] + col;
                channel_collapser(row, col) = 1.0;
            }
        }

        // now, copy this matrix to detector block
        block_collapser(Eigen::seqN(channel_row_start, total_num_bins_channel), Eigen::seqN(channel_col_start, m_channel_num_bins[ic])) = channel_collapser;
        channel_row_start += total_num_bins_channel;
        channel_col_start += m_channel_num_bins[ic];
    }

    //okay! now stuff every detector block size_to the final collapse matrix
    for(size_t im = 0; im != m_num_modes; ++im){
        for(size_t id =0; id != m_num_detectors; ++id){
            size_t row_block_start = im * m_num_bins_mode_block + id * m_num_bins_detector_block;
            size_t col_block_start = im * m_num_bins_mode_block_collapsed + id * m_num_bins_detector_block_collapsed;
            collapsing_matrix(Eigen::seqN(row_block_start, m_num_bins_detector_block), Eigen::seqN(col_block_start, m_num_bins_detector_block_collapsed)) = block_collapser;
        }
    }
    for(size_t io = 0; io < m_num_other_vars; ++io) {
        other_collapsing_matrices.push_back(Eigen::MatrixXf::Zero(m_num_other_bins_total[io], m_num_other_bins_total_collapsed[io]));
        log<LOG_INFO>(L"%1% || Creating Other %2% Collapsing Matrix. m_num_bins_total, m_num_bins_total_collapsed:  %3%  %4%") % __func__ % io % m_num_other_bins_total[io] % m_num_other_bins_total_collapsed[io];

        //construct the matrix by detector block
        Eigen::MatrixXf block_collapser = Eigen::MatrixXf::Zero(m_num_other_bins_detector_block[io], m_num_other_bins_detector_block_collapsed[io]);

        size_t channel_row_start = 0, channel_col_start = 0;
        for(size_t ic =0; ic != m_num_channels; ++ic){

            //first, build matrix for each channel block
            size_t total_num_bins_channel = m_num_subchannels[ic] * m_channel_num_other_bins[ic][io];

            Eigen::MatrixXf channel_collapser = Eigen::MatrixXf::Zero(total_num_bins_channel, m_channel_num_other_bins[ic][io]);
            for(size_t col = 0; col != m_channel_num_other_bins[ic][io]; ++col){
                for(size_t subch = 0; subch != m_num_subchannels[ic]; ++subch){
                    size_t row = subch * m_channel_num_other_bins[ic][io] + col;
                    channel_collapser(row, col) = 1.0;
                }
            }

            // now, copy this matrix to detector block
            block_collapser(Eigen::seqN(channel_row_start, total_num_bins_channel), Eigen::seqN(channel_col_start, m_channel_num_other_bins[ic][io])) = channel_collapser;
            channel_row_start += total_num_bins_channel;
            channel_col_start += m_channel_num_other_bins[ic][io];
        }

        //okay! now stuff every detector block size_to the final collapse matrix
        for(size_t im = 0; im != m_num_modes; ++im){
            for(size_t id =0; id != m_num_detectors; ++id){
                size_t row_block_start = im * m_num_other_bins_mode_block[io] + id * m_num_other_bins_detector_block[io];
                size_t col_block_start = im * m_num_other_bins_mode_block_collapsed[io] + id * m_num_other_bins_detector_block_collapsed[io];
                other_collapsing_matrices.back()(Eigen::seqN(row_block_start, m_num_other_bins_detector_block[io]), Eigen::seqN(col_block_start, m_num_other_bins_detector_block_collapsed[io])) = block_collapser;
            }
        }

    }
    return;
}

int PROconfig::HexToROOTColor(const std::string& hexColor) const{
    if (hexColor.length() != 7 || hexColor[0] != '#') {
        throw std::invalid_argument("Invalid hex color format. It should be in the format #RRGGBB.");
    }
    int r, g, b;
    std::stringstream ss;
    ss << std::hex << hexColor.substr(1, 2); 
    ss >> r;
    ss.clear();
    ss << std::hex << hexColor.substr(3, 2); 
    ss >> g;
    ss.clear();
    ss << std::hex << hexColor.substr(5, 2);
    ss >> b;
    return TColor::GetColor(r, g, b);
}

uint32_t PROconfig::CalcHash() const{
    int fixed_seed = 404;
    uint32_t hash;
    std::ostringstream unique_string;

    //Very quik hash, not including all important bits but a good start for now
    auto vecToString = [](const auto& vec) -> std::string {
        std::ostringstream oss;
        for (const auto& v : vec) {
            oss << v;
        }
        return oss.str();
    };

    unique_string << vecToString(m_fullnames);
    for (const auto& vec : m_channel_bin_edges) 
        unique_string << vecToString(vec);

    for (const auto& vec : m_channel_truebin_edges) 
        unique_string << vecToString(vec);

    for (const auto& vec : m_mcgen_additional_weight_name) 
        unique_string << vecToString(vec);

    for (const auto& vec : m_mcgen_eventweight_branch_names) 
        unique_string << vecToString(vec);

    unique_string << vecToString(m_mcgen_variation_allowlist);

    for(const auto& vec: m_branch_variables){
        for(const auto& br: vec){
            unique_string << br->name << br->associated_hist << br->associated_systematic << br->true_param_name<< br->true_L_name << br->model_rule;
        }
    }

    log<LOG_DEBUG>(L"%1% || MurmurHash input uniue string %2% ") % __func__ % unique_string.str().c_str();

    MurmurHash3_x86_32(unique_string.str().c_str(), unique_string.str().size(), fixed_seed, &hash);

    log<LOG_INFO>(L"%1% || MurmurHash output hash %2% ") % __func__ % hash;

    return hash;
}


