#include "PROconfig.h"
#include "PROlog.h"
#include <cctype>
#include <cstdlib>
#include <ctype.h>
#include <numeric>
#include <sstream>
#include <fstream>
#include <filesystem>
#include "TFile.h"
#include "TH1.h"
#include "TTree.h"
#include "TTreeFormula.h"
#include "TFriendElement.h"
using namespace PROfit;

namespace {
    // Trim trailing whitespace from a string (in place).
    void RTrim(std::string &s) {
        while(!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    }

    // Resolve the (label, unit) pair from the optional `xaxislabel` and `unit`
    // XML attributes. New form: both attributes are passed through. Legacy
    // form (only `unit` set): if it ends with a trailing "[...]", split into
    // label = everything before the bracket, unit = bracket contents;
    // otherwise label = the whole string, unit = "".
    void ResolveAxisLabelAndUnit(const char* xaxislabel_attr, const char* unit_attr,
                                 std::string &label_out, std::string &unit_out) {
        if(xaxislabel_attr != nullptr) {
            label_out = xaxislabel_attr;
            unit_out = unit_attr ? unit_attr : "";
            return;
        }
        if(unit_attr == nullptr) {
            label_out = "";
            unit_out = "";
            return;
        }
        std::string s(unit_attr);
        std::string trimmed = s;
        RTrim(trimmed);
        if(!trimmed.empty() && trimmed.back() == ']') {
            auto open = trimmed.rfind('[');
            if(open != std::string::npos) {
                std::string label = trimmed.substr(0, open);
                RTrim(label);
                unit_out = trimmed.substr(open + 1, trimmed.size() - open - 2);
                label_out = label;
                return;
            }
        }
        label_out = s;
        unit_out = "";
    }

    // The `use` attribute on <mode>/<detector>/<channel>/<subchannel> is
    // deprecated: disabling entries via use="false" left the per-channel
    // binning and per-detector POT arrays misaligned and is no longer
    // supported. use="true" is tolerated (with a warning); anything else is a
    // hard configuration error.
    void RejectDeprecatedUseAttribute(const char* use_attr, const char* element) {
        if(use_attr == nullptr) return;
        if(std::string(use_attr) == "true") {
            log<LOG_WARNING>(L"%1% || The 'use' attribute on <%2%> is deprecated and ignored; please remove it from your XML.")
                % __func__ % element;
            return;
        }
        log<LOG_ERROR>(L"%1% || ERROR: <%2% use=\"%3%\"> found. Disabling elements via use=\"false\" is deprecated and no longer supported: remove the whole element (or the 'use' attribute) from your XML.")
            % __func__ % element % use_attr;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }

    // Build uniform bin edges from min/max/nbins XML attributes with full
    // validation. A partially-specified <bins> used to reach strtod(NULL) —
    // undefined behavior — and nbins=0 divided by zero.
    std::vector<float> BuildUniformEdges(const char* omin, const char* omax, const char* onbins, const char* element) {
        if(omin == nullptr || omax == nullptr || onbins == nullptr) {
            log<LOG_ERROR>(L"%1% || ERROR: <%2%> needs all of min, max and nbins when edges are not given (got min=%3%, max=%4%, nbins=%5%).")
                % __func__ % element % (omin ? omin : "<missing>") % (omax ? omax : "<missing>") % (onbins ? onbins : "<missing>");
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }
        char* end = nullptr;
        const float minp = strtod(omin, &end);
        const float maxp = strtod(omax, &end);
        const int nbinsp = (int)strtod(onbins, &end);
        if(nbinsp < 1 || !(maxp > minp)) {
            log<LOG_ERROR>(L"%1% || ERROR: <%2%> has invalid binning min=%3%, max=%4%, nbins=%5% (need max > min and nbins >= 1).")
                % __func__ % element % minp % maxp % nbinsp;
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }
        std::vector<float> edges;
        edges.reserve(nbinsp + 1);
        const float step = (maxp - minp) / (float)nbinsp;
        for(int i = 0; i < nbinsp; ++i) edges.push_back(minp + i * step);
        edges.push_back(maxp);
        return edges;
    }
}


PROconfig::PROconfig(const std::string &xml, bool rate_only, int fit_variable):
    m_xmlname(xml),
    m_det_pot(),
    m_num_detectors(0),
    m_num_channels(0),
    m_num_modes(0),
    m_num_variables(0),
    m_num_variable_bins_detector_block(0),
    m_num_variable_bins_mode_block(0),
    m_num_variable_bins_total(0),
    m_num_variable_bins_detector_block_collapsed(0),
    m_num_variable_bins_mode_block_collapsed(0),
    m_num_variable_bins_total_collapsed(0),
    m_write_out_variation(false), 
    m_form_covariance(true),
    m_write_out_tag("UNSET_DEFAULT"),
    m_num_mcgen_files(0),
    m_bool_rate_only(rate_only)
{
    // Must be set before LoadFromXML: it resolves i_prime as its very first action.
    m_requested_fit_variable = fit_variable;

    LoadFromXML(m_xmlname);

    hash = PROconfig::CalcHash();
    detvar_hash = PROconfig::CalcDetVarHash();
    construct_variable_collapsing_matrices();

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
        if(one.m_det_pot[i] != two.m_det_pot[i]) {
            log<LOG_WARNING>(L"%1% || Found different POTs for detector %2%, %3% vs %4%")
                % __func__ % one.m_detector_names[i].c_str() % one.m_det_pot[i] % two.m_det_pot[i];
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
        if(one.m_channel_variable_bins[i][one.i_prime].NBins() != two.m_channel_variable_bins[i][two.i_prime].NBins()) {
            log<LOG_WARNING>(L"%1% || Found different number of channel bins %2% vs %3%")
                % __func__ % one.m_channel_variable_bins[i][one.i_prime].NBins() % two.m_channel_variable_bins[i][two.i_prime].NBins();
            return false;
        }
        if (one.m_channel_variable_bins[i][one.i_prime].NDim() != two.m_channel_variable_bins[i][two.i_prime].NDim()) {
            log<LOG_WARNING>(L"%1% || Found different number of channel variable dimensions %2% vs %3%")
                % __func__ % one.m_channel_variable_bins[i][one.i_prime].NDim() % two.m_channel_variable_bins[i][two.i_prime].NDim();
            return false;
        }

        for (size_t jdim = 0; jdim < one.m_channel_variable_bins[i][one.i_prime].NDim(); jdim++) {
            for (size_t k = 0; k < one.m_channel_variable_bins[i][one.i_prime].NBinEdgesAlong(jdim); k++) {
                if(one.m_channel_variable_bins[i][one.i_prime].Edges(jdim)[k] != two.m_channel_variable_bins[i][two.i_prime].Edges(jdim)[k]) {
                    log<LOG_WARNING>(L"%1% || Found different bin edge for bin %2% in channel %3% dimension %4%. %5% vs %6%")
                        % __func__ % k % i % jdim % one.m_channel_variable_bins[i][one.i_prime].Edges(jdim)[k] % two.m_channel_variable_bins[i][two.i_prime].Edges(jdim)[k];
                    return false;
                }
            }
        }
    }

    return true;
}


int PROconfig::ResolveFitVariableFromXML(tinyxml2::XMLDocument &doc) const {

    int resolved = -1;            //index marked fit="true", -1 until one is seen
    std::string resolved_channel; //channel that set it, for a useful error message
    size_t ichan = 0;
    //__func__ inside the scan lambda below is "operator()", so name the function once here.
    const char *fname = __func__;

    for(tinyxml2::XMLElement *pChanF = doc.FirstChildElement("channel"); pChanF;
            pChanF = pChanF->NextSiblingElement("channel"), ++ichan) {

        const char *cname = pChanF->Attribute("name");
        const std::string channel_name = cname ? cname : ("channel#" + std::to_string(ichan));

        // Walk this channel's binnings in the same order LoadFromXML numbers them:
        // every <bins2D> first, then every <bins>. `ivar` is shared across both passes.
        int ivar = 0, marked = -1;
        auto scan_bins = [&](const char *tag) {
            for(tinyxml2::XMLElement *pB = pChanF->FirstChildElement(tag); pB;
                    pB = pB->NextSiblingElement(tag), ++ivar) {
                const char *fit = pB->Attribute("fit");
                if(!fit) continue;
                const std::string fitstr(fit);
                if(fitstr == "false") continue;
                if(fitstr != "true") {
                    log<LOG_ERROR>(L"%1% || ERROR: <%2% fit=\"%3%\"> in channel %4% is not valid; fit must be \"true\" or \"false\".")
                        % fname % tag % fitstr.c_str() % channel_name.c_str();
                    throw std::invalid_argument(std::string("<") + tag + "> fit attribute must be true or false, got: " + fitstr);
                }
                if(marked >= 0) {
                    log<LOG_ERROR>(L"%1% || ERROR: channel %2% marks both var%3% and var%4% with fit=\"true\". Exactly one variable is fitted.")
                        % fname % channel_name.c_str() % marked % ivar;
                    throw std::invalid_argument("More than one binning marked fit=\"true\" in channel " + channel_name);
                }
                marked = ivar;
            }
        };
        scan_bins("bins2D");
        scan_bins("bins");

        if(marked < 0) continue;

        if(resolved >= 0 && resolved != marked) {
            log<LOG_ERROR>(L"%1% || ERROR: channel %2% marks var%3% as the fitting variable but channel %4% marks var%5%.")
                % __func__ % channel_name.c_str() % marked % resolved_channel.c_str() % resolved;
            log<LOG_ERROR>(L"%1% || -- Variables are numbered globally, so every channel must agree on which one is fitted.") % __func__;
            throw std::invalid_argument("Channels disagree on which variable has fit=\"true\"");
        }
        resolved = marked;
        resolved_channel = channel_name;
    }

    if(resolved < 0) {
        log<LOG_DEBUG>(L"%1% || No binning marked fit=\"true\"; defaulting to the first variable (var0).") % __func__;
        return 0;
    }

    log<LOG_DEBUG>(L"%1% || Channel %2% marks var%3% as the fitting variable.")
        % __func__ % resolved_channel.c_str() % resolved;
    return resolved;
}

void PROconfig::ValidateFitVariable() const {

    if(i_prime >= m_num_variables) {
        log<LOG_ERROR>(L"%1% || ERROR: fitting variable var%2% does not exist; this XML defines %3% variable(s) (var0..var%4%).")
            % __func__ % i_prime % m_num_variables % (m_num_variables ? m_num_variables - 1 : 0);
        throw std::invalid_argument("Fitting variable index out of range: " + std::to_string(i_prime));
    }

    // A model's kinematic variable (e.g. the truth L/E binning behind <parameter name="L/E"
    // variable_index="N">) is a physics grid, not a reco observable. Fitting it is never
    // intended and would silently produce nonsense, so reject it outright.
    for(const auto &[pname, pindex] : m_model_parameter_map) {
        if(pindex >= 0 && static_cast<size_t>(pindex) == i_prime) {
            log<LOG_ERROR>(L"%1% || ERROR: var%2% is the kinematic variable of model parameter \"%3%\", it cannot also be the fitting variable.")
                % __func__ % i_prime % pname.c_str();
            log<LOG_ERROR>(L"%1% || -- Mark a reconstructed observable with fit=\"true\" instead.") % __func__;
            throw std::invalid_argument("Fitting variable var" + std::to_string(i_prime) +
                                        " is model parameter \"" + pname + "\"'s kinematic variable");
        }
    }

    // Not fatal: PROfit builds the PROsyst and the data spectrum for i_prime regardless of
    // plot=, but nothing will be drawn for it.
    if(i_prime < m_channel_variable_plot_bool.size() && !m_channel_variable_plot_bool[i_prime])
        log<LOG_WARNING>(L"%1% || Fitting variable var%2% has plot=\"false\"; it will be fitted but not plotted.")
            % __func__ % i_prime;
}

int PROconfig::LoadFromXML(const std::string &filename){


    //Setup TiXml documents
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLError loadResult = doc.LoadFile(filename.c_str());

    bool use_universe = 1; //FIX
    if(loadResult == tinyxml2::XML_SUCCESS) {
        log<LOG_INFO>(L"%1% || Correctly loaded and parsed XML, continuing") % __func__;
    } else {
        log<LOG_ERROR>(L"%1% || ERROR: Failed to load XML configuration file: %2%") % __func__ % filename.c_str();
        if(loadResult == tinyxml2::XML_ERROR_FILE_NOT_FOUND) {
            log<LOG_ERROR>(L"The XML file was not found. Check the file path and name.");
        } else if(loadResult == tinyxml2::XML_ERROR_FILE_COULD_NOT_BE_OPENED) {
            log<LOG_ERROR>(L"The XML file could not be opened. Check file permissions.");
        } else if(loadResult == tinyxml2::XML_ERROR_FILE_READ_ERROR) {
            log<LOG_ERROR>(L"Error reading the XML file.");
        } else {
            log<LOG_ERROR>(L"XML parsing error: %1%") % doc.ErrorStr();
            log<LOG_ERROR>(L"This generally means broken brackets or attribute syntax in the XML itself.");
        }
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }

    tinyxml2::XMLHandle hDoc(&doc);

    tinyxml2::XMLElement *pMode, *pDet, *pChan;

    //**** Fitting variable (i_prime) ****
    // Resolved before ANYTHING else is parsed, because two later parse steps read it:
    // the --rateonly rebinning of the fitting variable, and the default binning of a
    // <systematic> (binning="reco" means "the fitting variable's binning").
    {
        const int xml_fit_variable = ResolveFitVariableFromXML(doc);
        if(m_requested_fit_variable >= 0) {
            if(m_requested_fit_variable != xml_fit_variable)
                log<LOG_WARNING>(L"%1% || Fitting variable overridden to var%2%; the XML asks for var%3%.")
                    % __func__ % m_requested_fit_variable % xml_fit_variable;
            i_prime = static_cast<size_t>(m_requested_fit_variable);
        } else {
            i_prime = static_cast<size_t>(xml_fit_variable);
        }
        log<LOG_INFO>(L"%1% || Fitting variable is var%2% (i_prime=%2%). All other variables are carried along but not fitted.")
            % __func__ % i_prime;
    }

    std::vector<std::string> allowed_elements ={"mode", "detector", "channel", "MCFile","WeightMaps","model","variation_list","systematics","correlation","varied_spectrum", "ShapeOnlyUncertainty", "data", "plotpot", "DetVarFiles"   };
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

            RejectDeprecatedUseAttribute(pMode->Attribute("use"), "mode");

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

            const std::vector<std::string> expected_attrs = {"name","plotname","use","pot"};
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

            RejectDeprecatedUseAttribute(pDet->Attribute("use"), "detector");

            const char* detector_pot= pDet->Attribute("pot");
            if (detector_pot == nullptr) {
                log<LOG_ERROR>(L"%1% || ERROR: Need POT defined for detector @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            } else {
                m_det_pot.push_back(std::strtod(detector_pot,&end));
            }

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

        std::vector<std::string> expected_attrs = {"name","plotname","use","unit","xaxislabel"};
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


            RejectDeprecatedUseAttribute(pChan->Attribute("use"), "channel");


            const char* channel_xaxislabel = pChan->Attribute("xaxislabel");
            const char* channel_unit= pChan->Attribute("unit");
            {
                std::string label, unit;
                ResolveAxisLabelAndUnit(channel_xaxislabel, channel_unit, label, unit);
                m_channel_xaxis_labels.push_back(label);
                m_channel_units.push_back(unit);
            }

            log<LOG_DEBUG>(L"%1% || Loading Channel %2% with   ") % __func__ % m_channel_names.back().c_str() ;

            m_channel_variable_bins.push_back({});
            m_channel_variable_xaxis_labels.push_back({});
            m_channel_variable_units.push_back({});
            m_channel_variable_dims.push_back({});


            tinyxml2::XMLElement *pBin2DO = pChan->FirstChildElement("bins2D"); // 2D Bins
            while(pBin2DO){
                const char* omin_x = pBin2DO->Attribute("minx");
                const char* omax_x = pBin2DO->Attribute("maxx");
                const char* onbins_x = pBin2DO->Attribute("nbinsx");
                const char* oedges_x = pBin2DO->Attribute("edgesx");

                const char* omin_y = pBin2DO->Attribute("miny");
                const char* omax_y = pBin2DO->Attribute("maxy");
                const char* onbins_y = pBin2DO->Attribute("nbinsy");
                const char* oedges_y = pBin2DO->Attribute("edgesy");

                const char* ounits = pBin2DO->Attribute("unit");
                const char* oplot = pBin2DO->Attribute("plot");

                if((omin_x == NULL && omax_x == NULL && onbins_x == NULL && oedges_x == NULL) ||
                        (omin_y == NULL && omax_y == NULL && onbins_y == NULL && oedges_y == NULL)) {
                    log<LOG_DEBUG>(L"%1% || This variable has a NO other binning (or attribute min,max,nbins)  ") % __func__ ;
                    m_channel_variable_bins.back().push_back(PROconfig::Binning());
                    m_channel_variable_xaxis_labels.back().push_back("");
                    m_channel_variable_units.back().push_back("");
                    m_channel_variable_dims.back().push_back(2);
                    m_channel_variable_plot_bool.push_back(true);
                }else{
                    log<LOG_DEBUG>(L"%1% || This variable has an Variable Binning.   ") % __func__  ;

                    if(oplot == NULL || std::string(oplot)=="true"){
                        m_channel_variable_plot_bool.push_back(true);
                    }else{
                        m_channel_variable_plot_bool.push_back(false);
                    }

                    std::vector<float> binedge_x;
                    // use edges if defined, otherwise use min-max-nbins 
                    if(oedges_x != NULL){
                        std::stringstream other_iss(oedges_x);
                        float number;
                        while (other_iss >> number){
                            binedge_x.push_back(number);
                        }
                        log<LOG_DEBUG>(L"%1% || This 2D variable has a Variable Binning in X with  %2% bins, Edges defined as %3%    ") % __func__ % binedge_x.size() % binedge_x;
                    }else{
                        binedge_x = BuildUniformEdges(omin_x, omax_x, onbins_x, "bins2D (x)");
                        log<LOG_DEBUG>(L"%1% || This 2D variable has a Variable Binning in X with edges %2%   ") % __func__ % binedge_x ;
                    }

                    std::vector<float> binedge_y;
                    // use edges if defined, otherwise use min-max-nbins 
                    if(oedges_y != NULL){
                        std::stringstream other_iss(oedges_y);
                        float number;
                        while (other_iss >> number){
                            binedge_y.push_back(number);
                        }
                        log<LOG_DEBUG>(L"%1% || This 2D variable has a Variable Binning in X with  %2% bins, Edges defined as %3%    ") % __func__ % binedge_y.size() % binedge_y;
                    }else{
                        binedge_y = BuildUniformEdges(omin_y, omax_y, onbins_y, "bins2D (y)");
                        log<LOG_DEBUG>(L"%1% || This 2D variable has a Variable Binning in Y with edges %2%   ") % __func__ % binedge_y ;
                    }

                    m_channel_variable_bins.back().push_back(PROconfig::Binning(std::vector<std::vector<float>>({binedge_x, binedge_y})));
                    m_channel_variable_xaxis_labels.back().push_back("");
                    m_channel_variable_units.back().push_back(ounits ? ounits : "");
                    m_channel_variable_dims.back().push_back(2);
                }
                pBin2DO = pBin2DO->NextSiblingElement("bins2D");
            }
            // 3D bins? 

            tinyxml2::XMLElement *pBinO = pChan->FirstChildElement("bins"); // 1D Bins
            while(pBinO){
                expected_attrs = {"min","max","nbins","edges","unit","xaxislabel","plot","fit"};
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
                const char* oxaxislabel = pBinO->Attribute("xaxislabel");
                const char* oplot = pBinO->Attribute("plot");
                if(omin==NULL && omax==NULL && onbins==NULL && oedges == NULL) {
                    log<LOG_DEBUG>(L"%1% || This variable has a NO other binning (or attribute min,max,nbins)  ") % __func__ ;
                    m_channel_variable_bins.back().push_back(PROconfig::Binning());
                    m_channel_variable_xaxis_labels.back().push_back("");
                    m_channel_variable_units.back().push_back("");
                    m_channel_variable_dims.back().push_back(1);
                    m_channel_variable_plot_bool.push_back(true);
                }else{
                    log<LOG_DEBUG>(L"%1% || This variable has an Variable Binning.   ") % __func__  ;

                    std::vector<float> binedge;

                    if(oplot == NULL || std::string(oplot)=="true"){
                        m_channel_variable_plot_bool.push_back(true);
                    }else{
                        m_channel_variable_plot_bool.push_back(false);
                    }

                    // use edges if defined, otherwise use min-max-nbins 
                    if(oedges != NULL){
                        std::stringstream other_iss(oedges);
                        float number;
                        while (other_iss >> number){
                            binedge.push_back(number);
                        }
                        log<LOG_DEBUG>(L"%1% || This variable has a Variable Binning with  %2% bins, Edges defined as %3%    ") % __func__ % binedge.size() % binedge ;
                    }else{
                        binedge = BuildUniformEdges(omin, omax, onbins, "bins");
                        log<LOG_DEBUG>(L"%1% || This variable has a Variable Binning with edges %2%   ") % __func__ % binedge ;
                    }

                    m_channel_variable_bins.back().push_back({binedge});
                    {
                        std::string label, unit;
                        ResolveAxisLabelAndUnit(oxaxislabel, ounits, label, unit);
                        m_channel_variable_xaxis_labels.back().push_back(label);
                        m_channel_variable_units.back().push_back(unit);
                    }
                    m_channel_variable_dims.back().push_back(1);
                }
                pBinO = pBinO->NextSiblingElement("bins");
            }

            if(m_bool_rate_only ){
                std::vector<float> edges = m_channel_variable_bins.back()[i_prime].Edges();
                m_channel_variable_bins.back()[i_prime] = std::vector<float>({ edges.front(), edges.back()});
            }

            // Now loop over all this channels subchanels. Not the names must be UNIQUE!!
            tinyxml2::XMLElement *pSubChan;
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

                RejectDeprecatedUseAttribute(pSubChan->Attribute("use"), "subchannel");

                const char* subchannel_data= pSubChan->Attribute("data");
                if(subchannel_data==NULL){
                    m_subchannel_datas[nchan].push_back(0);
                }else{
                    m_subchannel_datas[nchan].push_back(1);
                }

                pSubChan = pSubChan->NextSiblingElement("subchannel");
            }

            // Serialize bins XML for this channel (used to build the data and DetVar configs).
            // Emit <bins2D> before <bins>, matching the order the parser above numbers variables
            // in, so a child config built from this string gives every variable the same index
            // as its parent. Attributes (including fit=) are copied verbatim.
            {
                std::string channelBinsXml;
                auto serialize_bins = [&channelBinsXml, pChan](const char *tag) {
                    for(tinyxml2::XMLElement* pBinSer = pChan->FirstChildElement(tag); pBinSer;
                            pBinSer = pBinSer->NextSiblingElement(tag)) {
                        tinyxml2::XMLPrinter printer;
                        pBinSer->Accept(&printer);
                        channelBinsXml += "\t";
                        channelBinsXml += printer.CStr();
                        channelBinsXml += "\n";
                    }
                };
                serialize_bins("bins2D");
                serialize_bins("bins");
                m_channel_bins_xml_strings.push_back(channelBinsXml);
            }

            nchan++;
            pChan = pChan->NextSiblingElement("channel");
        }
    }//end channel loop
    // Assume all channels have the same number of "other" vars
    m_num_variables = m_channel_variable_bins[0].size();
    m_num_variable_bins_total = std::vector<size_t>(m_num_variables, 0);
    m_num_variable_bins_total_collapsed = std::vector<size_t>(m_num_variables, 0);
    m_num_variable_bins_mode_block = std::vector<size_t>(m_num_variables, 0);
    m_num_variable_bins_mode_block_collapsed = std::vector<size_t>(m_num_variables, 0);
    m_num_variable_bins_detector_block = std::vector<size_t>(m_num_variables, 0);
    m_num_variable_bins_detector_block_collapsed = std::vector<size_t>(m_num_variables, 0);

    //Now onto mcgen, for CV specs or for covariance generation
    tinyxml2::XMLElement *pMC, *pWeiMaps, *pList, *pCorrelations, *pSpec, *pShapeOnlyMap;
    pMC   = doc.FirstChildElement("MCFile");
    pWeiMaps = doc.FirstChildElement("WeightMaps");
    pList = doc.FirstChildElement("variation_list");
    if(!pList) pList = doc.FirstChildElement("systematics"); // alternative name for variation_list
    pCorrelations = doc.FirstChildElement("correlation");
    pSpec = doc.FirstChildElement("varied_spectrum");
    pShapeOnlyMap = doc.FirstChildElement("ShapeOnlyUncertainty");



    if(pMC){//Skip if not in XML
        while(pMC)
        {


            std::vector<std::string> expected_attrs = {"treename","filename","maxevents","scale","pot","fake","partial_load_frac"};
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
                m_mcgen_pot.push_back(1.0);
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

            const char* partial_load_frac = pMC->Attribute("partial_load_frac");
            if(partial_load_frac==NULL){
                m_mcgen_partial_load_frac.push_back(1.0);
            }else{
                m_mcgen_partial_load_frac.push_back(strtod(partial_load_frac,&end));
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



            std::vector<std::vector<std::string>> TEMP_weight_names; // per-branch list of weight formulas
            std::vector<int> TEMP_num_weights; // per-branch count of weights
            std::vector<std::string> TEMP_eventweight_branch_names;
            std::vector<bool> TEMP_hist_weight_bool;
            std::vector<std::string> TEMP_hist_weight_name;
            std::vector<int> TEMP_eventweight_branch_syst;

            std::vector<std::shared_ptr<BranchVariable>> TEMP_branch_variables;
            while(pBranch){

                expected_attrs = {"incl_systematics","associated_subchannel","associated_systematic","central_value","eventweight_branch_name","additional_weight", "model_rule"};
                for (const tinyxml2::XMLAttribute* attr = pBranch->FirstAttribute(); attr; attr = attr->Next()) {
                    std::string name = attr->Name();
                    // Allow weight_1, weight_2, ... weight_N dynamically
                    bool is_weight_attr = (name.substr(0, 7) == "weight_" && name.size() > 7);
                    if(is_weight_attr) {
                        bool all_digits = true;
                        for(size_t ci = 7; ci < name.size(); ++ci) {
                            if(!isdigit(name[ci])) { all_digits = false; break; }
                        }
                        is_weight_attr = all_digits;
                    }
                    if (!is_weight_attr && std::find(expected_attrs.begin(), expected_attrs.end(), name) == expected_attrs.end()) {
                        log<LOG_ERROR>(L"%1% || ERROR! Attribute [%2%] in the <MCFile/branch> element is not expected.") % __func__ % name.c_str()  ;
                        log<LOG_ERROR>(L"%1% || -- Check spelling: allowed attributes are %2% and weight_1, weight_2, ...") % __func__ % expected_attrs ;
                        throw std::invalid_argument(std::string("<MCfile/branch> attribute not allowed : ") + name);
                    }
                }


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
                    log<LOG_DEBUG>(L"%1% || Setting eventweight branch name %2%") %__func__ % bwname;
                    TEMP_eventweight_branch_names.push_back(std::string(bwname));
                }

                if(bincsyst== NULL || strcmp(bincsyst, "true") == 0){
                    log<LOG_DEBUG>(L"%1% ||Apply systematics to this file (default) ' @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                    TEMP_eventweight_branch_syst.push_back(1);
                }else{
                    log<LOG_DEBUG>(L"%1% || DO NOT apply systematics to this file (e.g for cosmics) ' @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                    TEMP_eventweight_branch_syst.push_back(0);
                }

                if(bhist == NULL){
                    log<LOG_ERROR>(L"%1% || Each branch must have an associated_subchannel to fill! On branch : @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__ ;
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

                // Parse generic weight attributes: weight_1, weight_2, ..., weight_N
                // Also accept additional_weight as backwards-compatible alias for weight_1
                std::vector<std::string> branch_weights;
                const char* bweight_1 = pBranch->Attribute("weight_1");
                if(bweight_1 != NULL && strcmp(bweight_1, "") != 0) {
                    branch_weights.push_back(bweight_1);
                } else if(badditional_weight != NULL && strcmp(badditional_weight, "") != 0) {
                    // additional_weight is backwards-compatible alias for weight_1
                    branch_weights.push_back(badditional_weight);
                    log<LOG_DEBUG>(L"%1% || Using additional_weight as alias for weight_1") % __func__;
                }
                // Parse weight_2, weight_3, ... (weight_1 already handled above)
                for(int wi = 2; ; ++wi) {
                    std::string attr_name = "weight_" + std::to_string(wi);
                    const char* wval = pBranch->Attribute(attr_name.c_str());
                    if(wval == NULL || strcmp(wval, "") == 0) break;
                    // If weight_1 was not set but weight_2+ exists, pad with "1" for weight_1
                    while((int)branch_weights.size() < wi - 1) {
                        branch_weights.push_back("1");
                    }
                    branch_weights.push_back(wval);
                }
                TEMP_weight_names.push_back(branch_weights);
                TEMP_num_weights.push_back((int)branch_weights.size());
                if(branch_weights.empty()) {
                    log<LOG_DEBUG>(L"%1% || Setting NO weights for branch ") % __func__;
                } else {
                    for(size_t wi = 0; wi < branch_weights.size(); ++wi) {
                        log<LOG_DEBUG>(L"%1% || Setting weight_%2% for branch: %3%") % __func__ % (wi+1) % branch_weights[wi].c_str();
                    }
                }


                if(use_universe){
                    TEMP_branch_variables.push_back( std::make_shared<BranchVariable>( bhist ) );
                } else  if((std::string)bcentral == "true"){
                    TEMP_branch_variables.push_back( std::make_shared<BranchVariable>( bhist,bsyst, true) );
                    log<LOG_DEBUG>(L"%1% || Setting as  CV for det sys.") % __func__ ;
                } else {
                    TEMP_branch_variables.push_back( std::make_shared<BranchVariable>( bhist,bsyst, false) );
                    log<LOG_DEBUG>(L"%1% || Setting as individual (not CV) for det sys.") % __func__ ;
                }


                //Load all variables
                tinyxml2::XMLElement *pVariable2D;
                // variable2D's and variable's are, for now, handled identically at config. 
                // (The parsing of the input string is done by the Formula class constructed at runtime).
                // This may change in the future, so keep the blocks separate
                pVariable2D = pBranch->FirstChildElement("variable2D"); // 2D variable
                int  nvar = 0;
                while(pVariable2D){
                    std::string var_text = std::string(pVariable2D->GetText());
                    log<LOG_DEBUG>(L"%1% || Setting branch Variable num %2%, Formula: %3%") %__func__ % nvar % var_text.c_str();
                    TEMP_branch_variables.back()->AddVariable(var_text);
                    nvar++;
                    pVariable2D = pVariable2D->NextSiblingElement("variable2D");
                } // 3D variable?

                tinyxml2::XMLElement *pVariable;
                pVariable = pBranch->FirstChildElement("variable"); // 1D variable
                while(pVariable){
                    std::string var_text = std::string(pVariable->GetText());
                    log<LOG_DEBUG>(L"%1% || Setting branch Variable num %2%, Formula: %3%") %__func__ % nvar % var_text.c_str();
                    TEMP_branch_variables.back()->AddVariable(var_text);
                    nvar++;
                    pVariable = pVariable->NextSiblingElement("variable");
                }

                if(nvar == 0){
                    log<LOG_ERROR>(L"%1% || ERROR: Need at least 1 variable passed in. You passed zero ") % __func__;
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);
                }else if((size_t)nvar!=m_num_variables){
                    log<LOG_ERROR>(L"%1% || ERROR: The number of variables in this MCFile %2%, is not the same as n_num_variables in bookeeping of XML %3%. ") % __func__ % nvar % m_num_variables;
                    log<LOG_ERROR>(L"%1% || ERROR: They need to be the same for now, Sorry. ") % __func__;
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);
                }




                TEMP_branch_variables.back()->SetIncludeSystematics(TEMP_eventweight_branch_syst.back());

                if(pBranch->Attribute("model_rule")) {
                    TEMP_branch_variables.back()->SetModelRule(pBranch->Attribute("model_rule"));
                }
                if(pBranch->Attribute("model_rule")) {
                    log<LOG_DEBUG>(L"%1% || Branch has Model Rule  %2% ") % __func__ % pBranch->Attribute("model_rule") ;
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
                    //TEMP_branch_variables.back()->SetTrueLeadingProtonP(pBranch->Attribute("true_proton_mom_name"));
                    //log<LOG_DEBUG>(L"%1% || Successfully set trueleadingp: %2% ") % __func__ % pBranch->Attribute("true_proton_mom_name");				 TEMP_branch_variables.back()->SetTrueLeadingProtonCosth(pBranch->Attribute("true_proton_costh_name"));
                    //log<LOG_DEBUG>(L"%1% || Successfully set trueleadingcosth: %2% ") % __func__ % pBranch->Attribute("true_proton_costh_name");				  				   
                }

                log<LOG_DEBUG>(L"%1% || Associated subchannel: %2% ") % __func__ % bhist;

                pBranch = pBranch->NextSiblingElement("branch");
            }

            m_mcgen_weight_names.push_back(TEMP_weight_names);
            m_mcgen_num_weights.push_back(TEMP_num_weights);
            m_branch_variables.push_back(TEMP_branch_variables);
            m_mcgen_eventweight_branch_names.push_back(TEMP_eventweight_branch_names);
            m_mcgen_eventweight_branch_syst.push_back(TEMP_eventweight_branch_syst);
            //next file
            pMC=pMC->NextSiblingElement("MCFile");
        }
    }

    // Parse DetVarFiles section (container of one or more DetVarSection blocks)
    tinyxml2::XMLElement *pDetVarContainer = doc.FirstChildElement("DetVarFiles");
    if(pDetVarContainer) {
        m_has_detvar_section = true;
        log<LOG_INFO>(L"%1% || Found <DetVarFiles> section in XML, parsing detector variation files...") % __func__;

        tinyxml2::XMLElement *pDetVar = pDetVarContainer->FirstChildElement("DetVarSection");
        if(!pDetVar) {
            log<LOG_ERROR>(L"%1% || ERROR: <DetVarFiles> must contain at least one <DetVarSection> element") % __func__;
            exit(EXIT_FAILURE);
        }

        while(pDetVar) {
            const size_t section_idx = m_detvar_xml_templates.size();
            m_detvar_subchannels_per_section.push_back({});
            std::vector<std::string>& section_subchannels = m_detvar_subchannels_per_section.back();

            log<LOG_INFO>(L"%1% || Parsing <DetVarSection> index %2%") % __func__ % section_idx;

            const char* dv_treename = pDetVar->Attribute("treename");
            bool have_tree = dv_treename;
            const char* dv_filename = pDetVar->Attribute("filename");
            bool have_file = dv_filename;

            const char* dv_scale_str = pDetVar->Attribute("scale");
            std::string dv_scale = dv_scale_str ? dv_scale_str : "1.0";

            // Parse include_only_weights for this section (1-based weight indices, comma-separated)
            const char* dv_iow_str = pDetVar->Attribute("include_only_weights");
            std::vector<int> section_iow;
            if(dv_iow_str) {
                const char *c = dv_iow_str, *begin = NULL;
                while(*c) {
                    if(begin && (isspace((unsigned char)*c) || *c == ',')) {
                        section_iow.push_back(atoi(begin));
                        begin = NULL;
                    } else if(!begin && !isspace((unsigned char)*c) && *c != ',') begin = c;
                    ++c;
                }
                if(begin) section_iow.push_back(atoi(begin));
                log<LOG_INFO>(L"%1% || Parsed include_only_weights for DetVar section %2%: %3% indices") % __func__ % section_idx % section_iow.size();
            }
            m_detvar_include_only_weights_per_section.push_back(section_iow);

            // Parse extra_weight attribute on <DetVarSection> (same attribute style as weight_N on <branch>)
            std::vector<std::string> section_extra_weights;
            const char* ew_attr = pDetVar->Attribute("extra_weight");
            if(ew_attr && strcmp(ew_attr, "") != 0) {
                section_extra_weights.push_back(ew_attr);
                log<LOG_INFO>(L"%1% || Parsed extra_weight for DetVar section %2%: '%3%'") % __func__ % section_idx % ew_attr;
            }
            m_detvar_extra_weights_per_section.push_back(section_extra_weights);

            // Parse cv_variation_matching_vars for this section (comma-separated branch names)
            const char* dv_mvars_str = pDetVar->Attribute("cv_variation_matching_vars");
            std::vector<std::string> section_matching_vars;
            if(dv_mvars_str) {
                std::istringstream mvars_ss(dv_mvars_str);
                std::string tok;
                while(std::getline(mvars_ss, tok, ',')) {
                    tok.erase(0, tok.find_first_not_of(" \t"));
                    if(!tok.empty()) tok.erase(tok.find_last_not_of(" \t") + 1);
                    if(!tok.empty()) section_matching_vars.push_back(tok);
                }
                log<LOG_INFO>(L"%1% || Parsed cv_variation_matching_vars for DetVar section %2%: %3% vars") % __func__ % section_idx % section_matching_vars.size();
            }
            m_detvar_matching_vars_per_section.push_back(section_matching_vars);

            // CV file: name is "cv" for section 0, "cv_N" for subsequent sections
            tinyxml2::XMLElement *pCV = pDetVar->FirstChildElement("cv");
            if(!pCV) {
                log<LOG_ERROR>(L"%1% || ERROR: <DetVarSection> must have a <cv> element") % __func__;
                exit(EXIT_FAILURE);
            }
            if(!have_file) dv_filename = pCV->Attribute("filename");
            if(!dv_filename) {
                log<LOG_ERROR>(L"%1% || ERROR: Require filename attribute in either DetVarSection or cv.") % __func__;
                exit(EXIT_FAILURE);
            }
            const char* cv_pot_str = pCV->Attribute("pot");
            if(!cv_pot_str) {
                log<LOG_ERROR>(L"%1% || ERROR: <cv> must have pot attribute") % __func__;
                exit(EXIT_FAILURE);
            }
            if(!have_tree) dv_treename = pCV->Attribute("treename");
            if(!dv_treename) {
                log<LOG_ERROR>(L"%1% || ERROR: Require a treename in either the DetVarSection or cv elements.") % __func__;
                exit(EXIT_FAILURE);
            }
            DetVarFile cv_file;
            cv_file.filename = dv_filename;
            cv_file.treename = dv_treename;
            cv_file.name = (section_idx == 0) ? "cv" : ("cv_" + std::to_string(section_idx));
            cv_file.pot = strtod(cv_pot_str, &end);
            cv_file.is_cv = true;
            cv_file.section_index = section_idx;
            { const char* frac = pCV->Attribute("partial_load_frac");
              cv_file.partial_load_frac = frac ? (float)strtod(frac, nullptr) : 1.0f; }
            m_detvar_files.push_back(cv_file);
            log<LOG_INFO>(L"%1% || DetVar CV file (section %2%): %3%, POT: %4%") % __func__ % section_idx % dv_filename % cv_file.pot;

            // Parse variation files
            tinyxml2::XMLElement *pVar = pDetVar->FirstChildElement("variation");
            while(pVar) {
                if(!have_file) dv_filename = pVar->Attribute("filename");
                if(!dv_filename) {
                    log<LOG_ERROR>(L"%1% || ERROR: Require filename attribute in either DetVarSection or variation.") % __func__;
                    exit(EXIT_FAILURE);
                }
                const char* var_name = pVar->Attribute("name");
                const char* var_pot_str = pVar->Attribute("pot");
                if(!var_name || !var_pot_str) {
                    log<LOG_ERROR>(L"%1% || ERROR: <variation> must have name and pot attributes") % __func__;
                    exit(EXIT_FAILURE);
                }
                if(!have_tree) dv_treename = pVar->Attribute("treename");
                if(!dv_treename) {
                    log<LOG_ERROR>(L"%1% || ERROR: Require a treename in either the DetVarSection or variation elements.") % __func__;
                    exit(EXIT_FAILURE);
                }
                const char *knobval = pVar->Attribute("knobval");
                DetVarFile var_file;
                var_file.filename = dv_filename;
                var_file.treename = dv_treename;
                var_file.name = var_name;
                var_file.pot = strtod(var_pot_str, &end);
                var_file.is_cv = false;
                var_file.knobval = knobval ? strtod(knobval, &end) : 1;
                var_file.section_index = section_idx;
                { const char* frac = pVar->Attribute("partial_load_frac");
                  var_file.partial_load_frac = frac ? (float)strtod(frac, nullptr) : 1.0f; }
                m_detvar_files.push_back(var_file);
                m_detvar_variation_names.insert(var_name);
                log<LOG_INFO>(L"%1% || DetVar variation '%2%' file (section %3%): %4%, POT: %5%") % __func__ % var_name % section_idx % dv_filename % var_file.pot;

                pVar = pVar->NextSiblingElement("variation");
            }

            // Parse subchannel list for this section
            tinyxml2::XMLElement *pSC = pDetVar->FirstChildElement("subchannel");
            while(pSC) {
                const char* sc_text = pSC->GetText();
                if(sc_text) {
                    section_subchannels.push_back(sc_text);
                    log<LOG_INFO>(L"%1% || DetVar subchannel (section %2%): %3%") % __func__ % section_idx % sc_text;
                }
                pSC = pSC->NextSiblingElement("subchannel");
            }
            if(section_subchannels.empty()) {
                log<LOG_ERROR>(L"%1% || ERROR: <DetVarSection> must have at least one <subchannel> element") % __func__;
                exit(EXIT_FAILURE);
            }

            // Build XML template for this section's DetVar configs
            // Contains mode/detector/channel definitions from main config, model, and MCFile template
            // Branches are auto-inherited from MCFile entries with matching associated_subchannel
            {
                std::ostringstream dvXml;
                dvXml << "<?xml version=\"1.0\" ?>\n\n";

                // Mode(s) - use m_mode_names.size() since m_num_modes isn't set yet
                for(size_t im = 0; im < m_mode_names.size(); im++) {
                    dvXml << "<mode name=\"" << m_mode_names[im] << "\" />\n";
                }
                dvXml << "\n";

                // Detector(s) - use m_detector_names.size() since m_num_detectors isn't set yet
                for(size_t id = 0; id < m_detector_names.size(); id++) {
                    dvXml << "<detector name=\"" << m_detector_names[id] << "\" pot=\"";
                    dvXml << std::scientific << m_det_pot[id] << "\" />\n";
                }
                dvXml << "\n";

                // Channels with same subchannels as main config
                for(size_t ic = 0; ic < m_channel_names.size(); ic++) {
                    dvXml << "<channel name=\"" << m_channel_names[ic] << "\"";
                    if(!m_channel_plotnames[ic].empty()) {
                        dvXml << " plotname=\"" << m_channel_plotnames[ic] << "\"";
                    }
                    dvXml << ">\n";
                    dvXml << m_channel_bins_xml_strings[ic];
                    for(size_t sc = 0; sc < m_subchannel_names[ic].size(); sc++) {
                        dvXml << "\t<subchannel name=\"" << m_subchannel_names[ic][sc] << "\"";
                        if(!m_subchannel_plotnames[ic][sc].empty()) {
                            dvXml << " plotname=\"" << m_subchannel_plotnames[ic][sc] << "\"";
                        }
                        if(!m_subchannel_colors[ic][sc].empty()) {
                            dvXml << " color=\"" << m_subchannel_colors[ic][sc] << "\"";
                        }
                        dvXml << "/>\n";
                    }
                    dvXml << "</channel>\n";
                }
                dvXml << "\n";

                // Model (serialize from XML directly)
                tinyxml2::XMLElement* pModelSer = doc.FirstChildElement("model");
                if(pModelSer) {
                    tinyxml2::XMLPrinter printer;
                    pModelSer->Accept(&printer);
                    dvXml << printer.CStr() << "\n\n";
                }

                // MCFile template with placeholders
                dvXml << "<MCFile treename=\"__DETVAR_TREENAME__\" filename=\"__DETVAR_FILENAME__\" scale=\"" << dv_scale << "\" pot=\"__DETVAR_POT__\" partial_load_frac=\"__DETVAR_PARTIAL_LOAD_FRAC__\">\n";

                // Serialize friend trees from this DetVarSection
                tinyxml2::XMLElement *pDVFriend = pDetVar->FirstChildElement("friend");
                while(pDVFriend) {
                    tinyxml2::XMLPrinter printer;
                    pDVFriend->Accept(&printer);
                    dvXml << "\t" << printer.CStr() << "\n";
                    pDVFriend = pDVFriend->NextSiblingElement("friend");
                }

                // Auto-inherit branches from MCFile entries by matching associated_subchannel
                for(const auto& sc_name : section_subchannels) {
                    bool found_any = false;
                    for(size_t fi = 0; fi < m_branch_variables.size(); fi++) {
                        for(size_t bi = 0; bi < m_branch_variables[fi].size(); bi++) {
                            if(m_branch_variables[fi][bi]->associated_hist == sc_name) {
                                found_any = true;
                                // Reconstruct <branch> XML from stored data
                                dvXml << "\t<branch associated_subchannel=\"" << sc_name << "\"";
                                if(m_branch_variables[fi][bi]->model_rule >= 0) {
                                    dvXml << " model_rule=\"" << m_branch_variables[fi][bi]->model_rule << "\"";
                                }
                                if(m_mcgen_eventweight_branch_syst[fi][bi] == 0) {
                                    dvXml << " incl_systematics=\"false\"";
                                }
                                // Weights (filtered by include_only_weights if set for this section)
                                const std::vector<int>& sec_iow = m_detvar_include_only_weights_per_section[section_idx];
                                int out_wi = 1;
                                for(size_t wi = 0; wi < m_mcgen_weight_names[fi][bi].size(); wi++) {
                                    if(!sec_iow.empty() && std::find(sec_iow.begin(), sec_iow.end(), (int)(wi+1)) == sec_iow.end()) continue;
                                    dvXml << " weight_" << out_wi++ << "=\"" << m_mcgen_weight_names[fi][bi][wi] << "\"";
                                }
                                // Append extra weights defined only for this DetVarSection
                                for(const auto& ew : m_detvar_extra_weights_per_section[section_idx]) {
                                    dvXml << " weight_" << out_wi++ << "=\"" << ew << "\"";
                                }
                                dvXml << ">\n";
                                // Variables
                                for(const auto& vname : m_branch_variables[fi][bi]->variable_names) {
                                    dvXml << "\t\t<variable>" << vname << "</variable>\n";
                                }
                                dvXml << "\t</branch>\n";
                            }
                        }
                    }
                    if(!found_any) {
                        log<LOG_ERROR>(L"%1% || ERROR: DetVar subchannel '%2%' not found in any MCFile branch") % __func__ % sc_name.c_str();
                        exit(EXIT_FAILURE);
                    }
                }

                dvXml << "</MCFile>\n";

                m_detvar_xml_templates.push_back(dvXml.str());
                log<LOG_INFO>(L"%1% || DetVar XML template for section %2% built successfully (%3% bytes)") % __func__ % section_idx % m_detvar_xml_templates.back().size();
                log<LOG_DEBUG>(L"%1% || DetVar XML template section %2% content:\n%3%") % __func__ % section_idx % m_detvar_xml_templates.back().c_str();
            }

            pDetVar = pDetVar->NextSiblingElement("DetVarSection");
        }
        // Note: DetVar systematic registration happens during <systematics> parsing
        // when a variation name matches an allowlist entry
    }

    if(!pList){
        log<LOG_DEBUG>(L"%1% || No Allowlist or Denylist set, including ALL variations by default.") % __func__  ;
    }else{
        while(pList){

            // Support both old naming (allowlist) and new naming (systematic)
            tinyxml2::XMLElement *pAllowList = pList->FirstChildElement("allowlist");
            if(!pAllowList) pAllowList = pList->FirstChildElement("systematic");
            while(pAllowList){
                const char *text = pAllowList->GetText();
                std::string wt = "null";
                if(text) {
                    wt = std::string(text);
                } else {
                    // Support name attribute as fallback (e.g. <allowlist name="Recomb2" ... />)
                    const char *name_attr = pAllowList->Attribute("name");
                    if(name_attr) wt = std::string(name_attr);
                }

                //check for known attributes
                const std::vector<std::string> expected_attrs = {"name", "type", "plotname", "binning", "knobvals", "tag", "prior", "center", "prior_type", "force_0_cv", "include_only_weights", "scale","filename", "xvar", "yvar", "restrict", "mirror", "num_decomp_knobs", "include_resid_cov", "inflate", "weights", "apply_to_subchannel"};
                for (const tinyxml2::XMLAttribute* attr = pAllowList->FirstAttribute(); attr; attr = attr->Next()) {
                    std::string name = attr->Name();
                    if (std::find(expected_attrs.begin(), expected_attrs.end(), name) == expected_attrs.end()) {
                        log<LOG_ERROR>(L"%1% || ERROR! Attribute [%2%] in the <allowlist>/<systematic> element is not expected.") % __func__ % name.c_str()  ;
                        log<LOG_ERROR>(L"%1% || -- Check spelling: allowed attributes are %2%") % __func__ % expected_attrs ;
                        throw std::invalid_argument(std::string("<allowlist>/<systematic> attribute not allowed : ") + name);
                    }
                }

                const char *variation_type = pAllowList->Attribute("type");
                const char *plot_name = pAllowList->Attribute("plotname");
                const char *binning = pAllowList->Attribute("binning");
                const char *knobs = pAllowList->Attribute("knobvals");
                const char *tags = pAllowList->Attribute("tag");
                const char *prior = pAllowList->Attribute("prior");
                const char *center = pAllowList->Attribute("center");
                const char *prior_type = pAllowList->Attribute("prior_type");
                const char *force_0_cv = pAllowList->Attribute("force_0_cv");
                const char *include_only_weights_str = pAllowList->Attribute("include_only_weights");
                const char *restrict_str = pAllowList->Attribute("restrict");
                const char *scale = pAllowList->Attribute("scale");
                const char *inflate = pAllowList->Attribute("inflate");
                const char *filename = pAllowList->Attribute("filename");
                const char *xvar = pAllowList->Attribute("xvar");
                const char *yvar = pAllowList->Attribute("yvar");
                const char *mirrored = pAllowList->Attribute("mirror");
                const char *num_decomp_knobs = pAllowList->Attribute("num_decomp_knobs");
                const char *include_resid_cov = pAllowList->Attribute("include_resid_cov");
                const char *weights = pAllowList->Attribute("weights");
                const char *apply_to_subchannel = pAllowList->Attribute("apply_to_subchannel");


                m_mcgen_variation_type.push_back(variation_type);
                m_mcgen_variation_type_map[wt] = variation_type;
                if(prior_type) {
                    const std::string parsed_prior_type(prior_type);
                    static const std::map<std::string, SplinePriorType> supported_prior_types = {
                        {"gaussian", SplinePriorType::Gaussian},
                        {"uniform", SplinePriorType::Uniform}
                    };
                    auto parsed = supported_prior_types.find(parsed_prior_type);
                    if(parsed == supported_prior_types.end()) {
                        throw std::invalid_argument(
                            std::string("Systematic '") + wt +
                            "' has unsupported prior_type='" + parsed_prior_type +
                            "' (expected 'gaussian' or 'uniform')"
                        );
                    }
                    if(!variation_type || std::string(variation_type) != "spline") {
                        throw std::invalid_argument(
                            std::string("prior_type is only supported for type='spline' systematics; got '") +
                            (variation_type ? variation_type : "unspecified") + "' for '" + wt + "'"
                        );
                    }
                    if(parsed->second == SplinePriorType::Uniform) {
                        if(!restrict_str) {
                            throw std::invalid_argument(
                                std::string("Uniform-prior spline '") + wt +
                                "' requires a finite restrict='lo, hi' range"
                            );
                        }
                        if(prior || center) {
                            throw std::invalid_argument(
                                std::string("Uniform-prior spline '") + wt +
                                "' cannot also specify Gaussian prior= or center= attributes"
                            );
                        }
                    }
                    m_mcgen_variation_prior_types[wt] = parsed->second;
                }
                // mcstat's covariance is registered in PROsyst under the systematic's name; remember
                // that name here so the covariance key matches the tag/plotname maps (both keyed by wt).
                if(variation_type && std::string(variation_type) == "mcstat") m_mcstat_systname = wt;

                // DetVar variations are handled separately (not weight branches in MC files),
                // so don't add them to the allowlist that PROcess_CAFAna uses.
                bool is_detvar = m_detvar_variation_names.count(wt) > 0;
                if(!is_detvar) {
                    m_mcgen_variation_allowlist.push_back(wt);
                } else {
                    log<LOG_INFO>(L"%1% || Systematic '%2%' matches a DetVar variation; skipping weight-branch allowlist.") % __func__ % wt.c_str();
                }
                if(prior) m_mcgen_variation_prior[wt] = std::strtof(prior, NULL);
                if(center) m_mcgen_variation_prior_centers[wt] = std::strtof(center, NULL);
                if(filename) m_mcgen_variation_external_filename_map[wt] = filename;
                m_mcgen_variation_plotname_map[wt] = plot_name ? plot_name : wt;
                if(variation_type && (strcmp(variation_type, "hist1d") == 0 || strcmp(variation_type, "hist2d") == 0)) {
                    if(!xvar) {
                        log<LOG_ERROR>(L"%1% || Expected xvar attribute for %2% systematic.")
                            % __func__ % variation_type;
                        exit(EXIT_FAILURE);
                    }
                    int v = atoi(xvar+3);
                    m_mcgen_variation_histaxisvars_map[wt][0] = v;
                    TFile fin(filename);
                    if(strcmp(variation_type, "hist1d") == 0) {
                        m_mcgen_variation_hist1d_map[wt] = (TH1*)fin.Get<TH1>(wt.c_str())->Clone();
                        m_mcgen_variation_hist1d_map[wt]->SetDirectory(0);
                    } else { 
                        m_mcgen_variation_hist2d_map[wt] = (TH2*)fin.Get<TH2>(wt.c_str())->Clone();
                        m_mcgen_variation_hist2d_map[wt]->SetDirectory(0);
                    }
                }
                if(variation_type && strcmp(variation_type, "hist2d") == 0) {
                    if(!yvar) {
                        log<LOG_ERROR>(L"%1% || Expected yvar attribute for hist2d systematic.")
                            % __func__;
                        exit(EXIT_FAILURE);
                    }
                    int v = atoi(yvar+3);
                    m_mcgen_variation_histaxisvars_map[wt][1] = v;
                }
                if(!binning || strcmp(binning, "reco") == 0) {
                    m_mcgen_variation_binning_map[wt] = i_prime;
                } else if(strncmp(binning, "var", 3) == 0) {
                    size_t l = strlen(binning);
                    bool all_numbers = true;
                    for(size_t i = 3; i < l; ++i) {
                        if(!isdigit(binning[i])) {
                            all_numbers = false;
                            break;
                        }
                    }
                    if(all_numbers) {
                        int binning_num;
                        sscanf(binning, "var%i", &binning_num);
                        m_mcgen_variation_binning_map[wt] = binning_num;
                    } else {
                        log<LOG_WARNING>(L"%1% || Unrecognized binning '%2%' for systematic %3%. Defaulting to reco bins.") 
                            % __func__ % binning % wt.c_str();
                        m_mcgen_variation_binning_map[wt] = i_prime;
                    }
                } else {
                    log<LOG_WARNING>(L"%1% || Unrecognized binning '%2%' for systematic %3%. Defaulting to reco bins.") 
                        % __func__ % binning % wt.c_str();
                    m_mcgen_variation_binning_map[wt] = i_prime;
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
                if(weights) {
                    std::vector<double> weights_vec;
                    const char *c = weights, *begin = NULL;
                    while(*c) {
                        if(begin && isspace(*c)) {
                            weights_vec.push_back(strtod(begin, NULL));
                            begin = NULL;
                        } else if(!begin && !isspace(*c)) begin = c;
                        ++c;
                    }
                    weights_vec.push_back(strtod(begin, NULL));
                    m_mcgen_explicit_weights[wt] = weights_vec;
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
                if(force_0_cv && strcmp(force_0_cv, "true") == 0) {
                    m_mcgen_variation_force_0_cv[wt] = true;
                    log<LOG_INFO>(L"%1% || Parsed force_0_cv=true for systematic %2%") % __func__ % wt.c_str();
                }
                if(include_only_weights_str) {
                    std::vector<int> iow_vec;
                    const char *c = include_only_weights_str, *begin = NULL;
                    while(*c) {
                        if(begin && (isspace(*c) || *c == ',')) {
                            iow_vec.push_back(atoi(begin));
                            begin = NULL;
                        } else if(!begin && !isspace(*c) && *c != ',') begin = c;
                        ++c;
                    }
                    if(begin) iow_vec.push_back(atoi(begin));
                    m_mcgen_variation_include_only_weights[wt] = iow_vec;
                    log<LOG_INFO>(L"%1% || Parsed include_only_weights for systematic %2%: %3% entries") % __func__ % wt.c_str() % iow_vec.size();
                }
                if(restrict_str) {
                    char *end;
                    float rlo = std::strtof(restrict_str, &end);
                    if(end == restrict_str)
                        throw std::invalid_argument(std::string("restrict attribute for systematic '") + wt + "' must be two numbers, e.g. restrict=\"-1, 1\"");
                    while(*end == ' ' || *end == ',') ++end;
                    float rhi = std::strtof(end, nullptr);
                    if(rlo > rhi) {
                        log<LOG_WARNING>(L"%1% || restrict for systematic %2% given as [%3%, %4%] with lo>hi; swapping. "
                                         L"An inverted range would otherwise hang rejection-sampling (pseudo-experiments).")
                            % __func__ % wt.c_str() % rlo % rhi;
                        const float t = rlo; rlo = rhi; rhi = t;
                    }
                    m_mcgen_variation_restrict[wt] = {rlo, rhi};
                    log<LOG_INFO>(L"%1% || Parsed restrict=[%2%, %3%] for systematic %4%") % __func__ % rlo % rhi % wt.c_str();
                }
                if(scale) {
                    m_mcgen_variation_scale[wt] = std::strtof(scale, NULL);
                    log<LOG_INFO>(L"%1% || Parsed scale=%2% for systematic %3%") % __func__ % m_mcgen_variation_scale[wt] % wt.c_str();
                }
                if(inflate) {
                    char *end;
                    float inflate_val = std::strtof(inflate, &end);
                    if(end == inflate || inflate_val <= 0) {
                        log<LOG_ERROR>(L"%1% || ERROR! inflate attribute for systematic %2% must be a positive number, got '%3%'") % __func__ % wt.c_str() % inflate;
                        throw std::invalid_argument(std::string("inflate attribute for systematic '") + wt + "' must be a positive number");
                    }
                    m_mcgen_variation_inflate[wt] = inflate_val;
                    log<LOG_INFO>(L"%1% || Parsed inflate=%2% for systematic %3%") % __func__ % inflate_val % wt.c_str();
                    const std::vector<std::string> inflatable_types = {"spline", "spline_to_covariance", "covariance", "external_covariance", "norm", "norm_to_covariance", "hist1d", "hist2d"};
                    if(!variation_type || std::find(inflatable_types.begin(), inflatable_types.end(), variation_type) == inflatable_types.end()) {
                        log<LOG_WARNING>(L"%1% || inflate is not supported for systematic %2% (type %3%); it will have no effect.")
                            % __func__ % wt.c_str() % (variation_type ? variation_type : "unspecified");
                    }
                }
                if(mirrored) {
                    if(strcmp(mirrored, "false") == 0 || strcmp(mirrored, "no") == 0 || strcmp(mirrored, "0") == 0)
                        m_mcgen_variation_unmirrored.insert(wt);
                }
                if(num_decomp_knobs) {
                    m_mcgen_variation_num_decomp_knobs[wt] = atoi(num_decomp_knobs);
                    log<LOG_INFO>(L"%1% || Parsed num_decomp_knobs=%2% for systematic %3%") % __func__ % m_mcgen_variation_num_decomp_knobs[wt] % wt.c_str();
                }
                if(include_resid_cov) {
                    bool keep_resid = !(strcmp(include_resid_cov, "false") == 0 || strcmp(include_resid_cov, "no") == 0 || strcmp(include_resid_cov, "0") == 0);
                    m_mcgen_variation_include_resid_cov[wt] = keep_resid;
                    log<LOG_INFO>(L"%1% || Parsed include_resid_cov=%2% for systematic %3%") % __func__ % keep_resid % wt.c_str();
                }
                if(apply_to_subchannel) {
                    std::string pattern = apply_to_subchannel;
                    // trim leading/trailing whitespace so apply_to_subchannel="nu_SBND " behaves
                    size_t b = pattern.find_first_not_of(" \t");
                    size_t e = pattern.find_last_not_of(" \t");
                    pattern = (b == std::string::npos) ? "" : pattern.substr(b, e - b + 1);
                    if(pattern.empty()) {
                        log<LOG_ERROR>(L"%1% || ERROR! apply_to_subchannel attribute for systematic %2% is empty.") % __func__ % wt.c_str();
                        throw std::invalid_argument(std::string("apply_to_subchannel attribute for systematic '") + wt + "' is empty");
                    }
                    m_mcgen_variation_apply_to_subchannel[wt] = pattern;
                    log<LOG_INFO>(L"%1% || Parsed apply_to_subchannel='%2%' for systematic %3% (substring match against subchannel fullnames)") % __func__ % pattern.c_str() % wt.c_str();
                }
                log<LOG_DEBUG>(L"%1% || Allowlisting variations: %2%") % __func__ % wt.c_str() ;
                tinyxml2::XMLElement *pNext = pAllowList->NextSiblingElement("allowlist");
                if(!pNext) pNext = pAllowList->NextSiblingElement("systematic");
                pAllowList = pNext;
            }

            tinyxml2::XMLElement *pDenyList = pList->FirstChildElement("denylist");
            while(pDenyList){
                std::string bt = std::string(pDenyList->GetText());
                m_mcgen_variation_denylist.push_back(bt); 
                log<LOG_DEBUG>(L"%1% || Denylisting variations: %2%") % __func__ % bt.c_str() ;
                pDenyList = pDenyList->NextSiblingElement("denylist");
            }
            tinyxml2::XMLElement *pNextList = pList->NextSiblingElement("variation_list");
            if(!pNextList) pNextList = pList->NextSiblingElement("systematics");
            pList = pNextList;
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

        const auto is_uniform = [this](const std::string &name) {
            return GetSplinePriorType(name) == SplinePriorType::Uniform;
        };
        if(is_uniform(split[0]) || is_uniform(split[1])) {
            throw std::invalid_argument(
                std::string("Uniform-prior splines cannot appear in Gaussian <correlation> entries: ") +
                split[0] + " " + split[1]
            );
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

        // Now loop over all this models Parameters
        tinyxml2::XMLElement *pModelParam;
        pModelParam = pModel->FirstChildElement("parameter");
        while(pModelParam){
            const char* model_parameter_name= pModelParam->Attribute("name");
            if(model_parameter_name==NULL){
                log<LOG_ERROR>(L"%1% || ERROR: Model Params need a name in xml.@ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }else{
                m_model_parameter_names.push_back(model_parameter_name);
            }


            // variable_index is optional: oscillation models require it (it maps the parameter
            // to a kinematic PROpeller variable, e.g. "L/E"), but normalization models such as
            // template_fit name a subchannel instead and carry no kinematic variable, so a
            // missing variable_index defaults to -1 rather than being a hard error.
            const char* model_parameter_index= pModelParam->Attribute("variable_index");
            if(model_parameter_index==NULL){
                m_model_parameter_index.push_back(-1);
            }else{
                m_model_parameter_index.push_back(strtod(model_parameter_index, &end));
            }

            // Optional scale bounds (used by template_fit-style normalization models). Default
            // to [0, 10] when absent so a generic <parameter> tag without them is still valid.
            const char* model_parameter_min = pModelParam->Attribute("min");
            const char* model_parameter_max = pModelParam->Attribute("max");
            m_model_parameter_min.push_back(model_parameter_min==NULL ? 0.0f  : (float)strtod(model_parameter_min, &end));
            m_model_parameter_max.push_back(model_parameter_max==NULL ? 10.0f : (float)strtod(model_parameter_max, &end));

            log<LOG_DEBUG>(L"%1% || Model Param Name :  %2% and index %3% (min %4%, max %5%) ") % __func__ % m_model_parameter_names.back().c_str() % m_model_parameter_index.back() % m_model_parameter_min.back() % m_model_parameter_max.back()  ;
            m_model_parameter_map[m_model_parameter_names.back()]=m_model_parameter_index.back();
            pModelParam = pModelParam->NextSiblingElement("parameter");
        }

    }//end model

    for(size_t i = 0 ; i<m_mcgen_variation_type.size(); ++i){
        if(m_mcgen_variation_type[i] == "spline"){
            m_num_variation_type_spline+=1;
        }

        else if(m_mcgen_variation_type[i] == "covariance"){
            m_num_variation_type_covariance+=1;
        }

        else if(m_mcgen_variation_type[i] == "covariance_to_spline"){
            m_num_variation_type_covariance_to_spline+=1;
        }

        else if(m_mcgen_variation_type[i] == "flat"){
            m_num_variation_type_flat+=1;
        } else if(m_mcgen_variation_type[i] == "norm"){
            m_num_variation_type_norm+=1;
        } else if (m_mcgen_variation_type[i] == "norm_to_covariance") {
            m_num_variation_type_norm_to_covariance+=1;
        }else if(m_mcgen_variation_type[i] == "spline_to_covariance"){
            m_num_variation_type_spline_to_covariance+=1;
        }else if(m_mcgen_variation_type[i] == "mcstat"){
            m_mcgen_variation_allowlist[i] = "mcstat";
            m_mcgen_variation_type_map["mcstat"] = "mcstat";
            m_use_mcstats = true;
        }else if(m_mcgen_variation_type[i] == "external_covariance"){
            m_num_variation_type_external_covariance+=1;
        }else if(m_mcgen_variation_type[i] == "external_covariance_to_spline"){
            m_num_variation_type_external_covariance_to_spline+=1;
        } else if(m_mcgen_variation_type[i] == "hist1d"){
            m_num_variation_type_hist1d+=1;
        } else if(m_mcgen_variation_type[i] == "hist2d"){
            m_num_variation_type_hist2d+=1;
        } else if(m_mcgen_variation_type[i] == "explicit_spline"){
            m_num_variation_type_explicit+=1;
        } else {
            log<LOG_ERROR>(L"%1% || Unrecognized variation type %2%") % __func__ % m_mcgen_variation_type[i].c_str();
        }

    }

    log<LOG_INFO>(L"%1% || num_variation_type_covariance: %2% ") % __func__ % m_num_variation_type_covariance;
    log<LOG_INFO>(L"%1% || num_variation_type_covariance_to_spline: %2% ") % __func__ % m_num_variation_type_covariance_to_spline;
    log<LOG_INFO>(L"%1% || num_variation_type_external_ovariance: %2% ") % __func__ % m_num_variation_type_external_covariance;
    log<LOG_INFO>(L"%1% || num_variation_type_external_covariance_to_spline: %2% ") % __func__ % m_num_variation_type_external_covariance_to_spline;
    log<LOG_INFO>(L"%1% || num_variation_type_flat: %2% ") % __func__ % m_num_variation_type_flat;
    log<LOG_INFO>(L"%1% || num_variation_type_norm: %2% ") % __func__ % m_num_variation_type_norm;
    log<LOG_INFO>(L"%1% || num_variation_type_norm_to_covariance: %2% ") % __func__ % m_num_variation_type_norm_to_covariance;
    log<LOG_INFO>(L"%1% || num_variation_type_spline: %2% ") % __func__ % m_num_variation_type_spline;
    log<LOG_INFO>(L"%1% || num_variation_type_spline_to_covariance: %2% ") % __func__ % m_num_variation_type_spline_to_covariance;
    log<LOG_INFO>(L"%1% || num_variation_type_explicit_spline: %2% ") % __func__ % m_num_variation_type_explicit;
    if(m_use_mcstats){
        log<LOG_INFO>(L"%1% || Using MC intrinsic stat uncertainty. ") % __func__  ;
    }else{
        log<LOG_INFO>(L"%1% || Not using MC intrinsic stat uncertainty. Check this is what you want. ") % __func__ ; 
    }



    // i_prime was resolved at the top of this function; only now are m_num_variables and the
    // model parameter map filled in, so this is the first point it can be checked.
    this->ValidateFitVariable();

    this->CalcTotalBins();

    log<LOG_INFO>(L"%1% || Checking number of Mode/Detector/Channel/Subchannels and BINs") % __func__;
    log<LOG_INFO>(L"%1% || num_modes: %2% ") % __func__ % m_num_modes;
    log<LOG_INFO>(L"%1% || num_detectors: %2% ") % __func__ % m_num_detectors;
    log<LOG_INFO>(L"%1% || num_channels: %2% ") % __func__ % m_num_channels;
    for(size_t i = 0 ; i!=m_num_channels; ++i){
        log<LOG_INFO>(L"%1% || num of subchannels: %2% ") % __func__ % m_num_subchannels[i];
    }
    for(size_t io = 0; io < m_num_variables; ++io) {
        log<LOG_INFO>(L"%1% || variable %2% num_bins_total: %3%") % __func__ % io % m_num_variable_bins_total[io];
        log<LOG_INFO>(L"%1% || variable %2% num_bins_total_collapsed: %3%") % __func__ % io % m_num_variable_bins_total_collapsed[io];
    }

    // Parse embedded <data> section if present
    tinyxml2::XMLElement* pData = doc.FirstChildElement("data");
    if(pData) {
        m_has_data_section = true;
        log<LOG_INFO>(L"%1% || Found embedded <data> section in XML, building data config string...") % __func__;

        // Build a self-contained data XML string
        std::ostringstream dataXml;
        dataXml << "<?xml version=\"1.0\" ?>\n\n";

        // Mode(s)
        for(size_t im = 0; im < m_num_modes; im++) {
            dataXml << "<mode name=\"" << m_mode_names[im] << "\" />\n";
        }
        dataXml << "\n";

        // Detector(s) — format pot in scientific notation to preserve precision
        for(size_t id = 0; id < m_num_detectors; id++) {
            dataXml << "<detector name=\"" << m_detector_names[id] << "\" pot=\"";
            dataXml << std::scientific << m_det_pot[id] << "\" />\n";
        }
        dataXml << "\n";

        // Channels with a single "data" subchannel, reusing the original bins XML
        for(size_t ic = 0; ic < m_num_channels; ic++) {
            dataXml << "<channel name=\"" << m_channel_names[ic] << "\"";
            if(!m_channel_plotnames[ic].empty()) {
                dataXml << " plotname=\"" << m_channel_plotnames[ic] << "\"";
            }
            dataXml << ">\n";
            dataXml << m_channel_bins_xml_strings[ic];
            dataXml << "\t<subchannel name=\"data\" plotname=\"Data\" color=\"#99CCFF\"/>\n";
            dataXml << "</channel>\n";
        }
        dataXml << "\n";

        // Serialize MCFile elements from inside the <data> section
        tinyxml2::XMLElement* pDataMC = pData->FirstChildElement("MCFile");
        while(pDataMC) {
            const char* frac = pDataMC->Attribute("partial_load_frac");
            if(frac != nullptr && strtod(frac, nullptr) != 1.0) {
                log<LOG_ERROR>(L"%1% || ERROR: partial_load_frac is not supported on data MCFiles (found value: %2%). Partial loading only applies to MC/EXT.") % __func__ % frac;
                throw std::invalid_argument("partial_load_frac is not supported on data MCFiles");
            }
            tinyxml2::XMLPrinter printer;
            pDataMC->Accept(&printer);
            dataXml << printer.CStr() << "\n";
            pDataMC = pDataMC->NextSiblingElement("MCFile");
        }

        m_data_xml_string = dataXml.str();
        log<LOG_INFO>(L"%1% || Data config XML string built successfully (%2% bytes)") % __func__ % m_data_xml_string.size();
    }

    log<LOG_INFO>(L"%1% || Done reading the xmls") % __func__;
    return 0;
}


void PROconfig::CalcTotalBins(){
    this->remove_unused_channel();

    log<LOG_INFO>(L"%1% || calculating number of bins involved") % __func__;
    for(size_t i = 0; i != m_num_channels; ++i){
        for(size_t io = 0; io < m_num_variables; ++io) {
            m_num_variable_bins_detector_block[io] += m_num_subchannels[i]*m_channel_variable_bins[i][io].NBins();
            m_num_variable_bins_detector_block_collapsed[io] += m_channel_variable_bins[i][io].NBins();
        }
    }

    for(size_t io = 0; io < m_num_variables; ++io) {
        m_num_variable_bins_mode_block[io] = m_num_variable_bins_detector_block[io] * m_num_detectors;
        m_num_variable_bins_mode_block_collapsed[io] = m_num_variable_bins_detector_block_collapsed[io] * m_num_detectors;
    }

    for(size_t io = 0; io < m_num_variables; ++io) {
        m_num_variable_bins_total[io] = m_num_variable_bins_mode_block[io] * m_num_modes;
        m_num_variable_bins_total_collapsed[io] = m_num_variable_bins_mode_block_collapsed[io] * m_num_modes;
    }

    log<LOG_INFO>(L"%1% || Generating Index maps for convenience") % __func__;
    this->generate_index_map();

    //some internal cals

    for(size_t io = 0; io < m_num_variables; ++io) {
        std::vector<float> tmp;
        std::vector<std::pair<float,float>> tmpe;
        int global_channel_index=0;
        for(size_t mode = 0; mode < m_num_modes; ++mode) {
            for(size_t det = 0; det < m_num_detectors; ++det) {
                for(size_t channel = 0; channel < m_num_channels; ++channel) {
                    std::vector<float> widths =  GetChannelVariableBins(global_channel_index, io).Widths();
                    const std::vector<float>& edges = GetChannelVariableBins(global_channel_index, io).Edges();

                    for(size_t sc = 0; sc < m_num_subchannels[channel]; sc++){           
                        for(size_t i = 0; i < edges.size() - 1; ++i) {
                            tmpe.push_back(std::make_pair(edges[i], edges[i+1]));
                        }
                    }

                    global_channel_index++;
                    tmp.insert(tmp.end(), widths.begin(), widths.end());
                }
            }
        }
        // One entry per VARIABLE (consumers index this by variable index).
        // This push used to sit inside the mode loop with a never-cleared
        // accumulator: with >1 mode the vector held nvars*nmodes cumulatively
        // growing entries and per-variable lookups read the wrong edges.
        m_variable_bin_to_edges.push_back(tmpe);
        Eigen::VectorXf coll_bin_widths = Eigen::Map<Eigen::VectorXf>(tmp.data(),tmp.size());
        collapsed_bin_widths.push_back(coll_bin_widths);
        log<LOG_INFO>(L"%1% || On variable %2% bin widths are size %3% and  %4% ") % __func__ % io % coll_bin_widths.size() % coll_bin_widths;

    }


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

std::string PROconfig::GetSubchannelName(size_t index) const{
    return m_fullnames.at(index); 

}

size_t PROconfig::GetLocalChannelIndexFromGlobalSubchannelIndex(size_t subchannel_index) const{
    size_t index = this->find_equal_index(m_vec_subchannel_index, subchannel_index);
    return m_vec_channel_index[index];
}




size_t PROconfig::GetGlobalVariableBinStart(size_t subchannel_index, size_t other_index) const{
    size_t index = this->find_equal_index(m_vec_subchannel_index, subchannel_index);
    return m_vec_global_variable_index_start[other_index][index];
}
size_t PROconfig::GetCollapsedGlobalVariableBinStart(size_t channel_index, size_t other_index) const{
    if(channel_index >= m_num_channels*m_num_modes*m_num_detectors) {
        size_t tot= m_num_channels*m_num_modes*m_num_detectors;
        log<LOG_ERROR>(L"%1% || Requested bin start of channel %2%, but only %3% total channels are known (chat*mode*det).")
            % __func__ % channel_index % tot;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }    
    size_t index = 0;
    for(size_t i = 0; i < channel_index; ++i) index += m_channel_variable_bins[GetLocalChannelIndexFromGlobalChannelIndex(i)][other_index].NBins();
    return index;
}


size_t PROconfig::GetSubchannelIndexFromVariableGlobalBin(size_t global_reco_index, size_t var_index) const {
    size_t index = this->find_less_or_equal_index(m_vec_global_variable_index_start[var_index], global_reco_index); 
    return m_vec_subchannel_index[index];
}


size_t PROconfig::GetLocalChannelIndexFromGlobalChannelIndex(size_t global_channel_index) const{
    return global_channel_index%m_num_channels; 
}

const PROconfig::Binning& PROconfig::GetChannelVariableBins(size_t channel_index, size_t other_index) const {
    if(channel_index >= m_num_channels*m_num_modes*m_num_detectors) {
        size_t tot= m_num_channels*m_num_modes*m_num_detectors;
        log<LOG_ERROR>(L"%1% || Given channel index: %2% is out of bound") % __func__ % channel_index;
        log<LOG_ERROR>(L"%1% || Total number of channels : %2%") % __func__ % tot;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }
    return m_channel_variable_bins[GetLocalChannelIndexFromGlobalChannelIndex(channel_index)][other_index];
}

void PROconfig::SetActiveBins(size_t var_index, const std::vector<char> &mask) {
    if(var_index >= m_num_variables) {
        log<LOG_ERROR>(L"%1% || Variable index %2% out of range (%3% variables).") % __func__ % var_index % m_num_variables;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }
    if(mask.size() != m_num_variable_bins_total_collapsed[var_index]) {
        log<LOG_ERROR>(L"%1% || Active-bin mask for variable %2% has %3% entries but the variable has %4% collapsed bins.")
            % __func__ % var_index % mask.size() % m_num_variable_bins_total_collapsed[var_index];
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }
    size_t n_active = 0;
    for(char m : mask) n_active += (m != 0);
    if(n_active == 0) {
        log<LOG_ERROR>(L"%1% || Active-bin mask for variable %2% has no active bins; refusing to install it.") % __func__ % var_index;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }
    if(m_variable_active_bins_collapsed.size() < m_num_variables)
        m_variable_active_bins_collapsed.resize(m_num_variables);
    m_variable_active_bins_collapsed[var_index] = mask;
    log<LOG_INFO>(L"%1% || Installed fit-region mask for variable %2%: %3% of %4% collapsed bins active.")
        % __func__ % var_index % n_active % mask.size();
}

void PROconfig::ClearActiveBins() {
    m_variable_active_bins_collapsed.clear();
}

bool PROconfig::HasActiveBins(size_t var_index) const {
    return var_index < m_variable_active_bins_collapsed.size()
        && !m_variable_active_bins_collapsed[var_index].empty();
}

bool PROconfig::IsBinActive(size_t var_index, size_t collapsed_bin) const {
    if(!HasActiveBins(var_index)) return true;
    const std::vector<char> &mask = m_variable_active_bins_collapsed[var_index];
    return collapsed_bin < mask.size() && mask[collapsed_bin] != 0;
}

size_t PROconfig::NActiveBins(size_t var_index) const {
    if(!HasActiveBins(var_index)) return m_num_variable_bins_total_collapsed[var_index];
    size_t n = 0;
    for(char m : m_variable_active_bins_collapsed[var_index]) n += (m != 0);
    return n;
}

namespace {
    std::string FormatLabelUnit(const std::string &label, const std::string &unit) {
        if(label.empty() && unit.empty()) return "";
        if(unit.empty()) return label;
        if(label.empty()) return "[" + unit + "]";
        return label + " [" + unit + "]";
    }
}

std::string PROconfig::GetChannelXAxisTitle(size_t channel_index) const {
    if(channel_index >= m_channel_units.size()) return "";
    return FormatLabelUnit(m_channel_xaxis_labels[channel_index], m_channel_units[channel_index]);
}

std::string PROconfig::GetChannelXAxisTitle(size_t channel_index, size_t other_index) const {
    if(channel_index >= m_channel_variable_units.size()) return "";
    if(other_index >= m_channel_variable_units[channel_index].size()) return "";
    // 2D variables keep the legacy combined "xtitle;ytitle" string.
    if(channel_index < m_channel_variable_dims.size() &&
       other_index < m_channel_variable_dims[channel_index].size() &&
       m_channel_variable_dims[channel_index][other_index] == 2) {
        return m_channel_variable_units[channel_index][other_index];
    }
    return FormatLabelUnit(m_channel_variable_xaxis_labels[channel_index][other_index],
                           m_channel_variable_units[channel_index][other_index]);
}

std::string PROconfig::GetChannelAxisTitle(size_t channel_index, size_t other_index, size_t dim) const {
    const std::string title = GetChannelXAxisTitle(channel_index, other_index);
    if(channel_index >= m_channel_variable_dims.size() ||
       other_index >= m_channel_variable_dims[channel_index].size() ||
       m_channel_variable_dims[channel_index][other_index] != 2) {
        return dim == 0 ? title : "";
    }

    const size_t separator = title.find(';');
    if(dim == 0) return title.substr(0, separator);
    if(dim == 1 && separator != std::string::npos) return title.substr(separator + 1);
    return "";
}

std::string PROconfig::GetChannelUnit(size_t channel_index, size_t other_index) const {
    // 2D variables keep the legacy combined "xtitle;ytitle" string in their
    // units slot, so it isn't a real unit -- skip straight to the channel-level.
    bool is_2d = (channel_index < m_channel_variable_dims.size() &&
                  other_index < m_channel_variable_dims[channel_index].size() &&
                  m_channel_variable_dims[channel_index][other_index] == 2);
    if(!is_2d &&
       channel_index < m_channel_variable_units.size() &&
       other_index < m_channel_variable_units[channel_index].size() &&
       !m_channel_variable_units[channel_index][other_index].empty()) {
        return m_channel_variable_units[channel_index][other_index];
    }
    if(channel_index < m_channel_units.size()) {
        return m_channel_units[channel_index];
    }
    return "";
}

//------------ Start of private function ------------------
//------------ Start of private function ------------------
//------------ Start of private function ------------------

void PROconfig::remove_unused_channel(){

    // The old `use="false"` disable mechanism is gone (it left the
    // per-channel binning and per-detector POT arrays misaligned); every
    // parsed mode/detector/channel/subchannel is in use, so the counts come
    // straight from the parsed arrays.
    m_num_modes = m_mode_names.size();
    m_num_detectors = m_detector_names.size();
    m_num_channels = m_channel_names.size();

    // The subchannel arrays are over-allocated (to 100) before parsing; trim
    // them to the real channel count so their size matches m_num_channels.
    m_subchannel_names.resize(m_num_channels);
    m_subchannel_plotnames.resize(m_num_channels);
    m_subchannel_colors.resize(m_num_channels);
    m_subchannel_datas.resize(m_num_channels);

    m_num_subchannels.resize(m_num_channels);
    for(size_t i = 0; i < m_num_channels; ++i)
        m_num_subchannels[i] = m_subchannel_names[i].size();

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
        std::vector<std::vector<std::vector<std::string>>> temp_weight_names;
        std::vector<std::vector<int>> temp_num_weights;
        std::vector<std::vector<std::shared_ptr<BranchVariable>>> temp_branch_variables;
        std::vector<std::vector<std::string>> temp_eventweight_branch_names;
        std::vector<std::vector<int>> temp_eventweight_branch_syst;

        for(size_t i = 0; i != m_mcgen_file_name.size(); ++i){
            log<LOG_DEBUG>(L"%1% || Check on @%2% th file: %3%...") % __func__ % i % m_mcgen_file_name[i].c_str();
            bool this_file_needed = false;

            std::vector<std::vector<std::string>> this_file_weight_names;
            std::vector<int> this_file_num_weights;
            std::vector<std::shared_ptr<BranchVariable>> this_file_branch_variables;
            std::vector<std::string> this_file_eventweight_branch_names;
            std::vector<int> this_file_eventweight_branch_syst;
            for(size_t j = 0; j != m_branch_variables[i].size(); ++j){

                if(set_all_names.find(m_branch_variables[i][j]->associated_hist) == set_all_names.end()){
                }else{

                    set_all_names.erase(m_branch_variables[i][j]->associated_hist);
                    this_file_needed = true;

                    this_file_weight_names.push_back(m_mcgen_weight_names[i][j]);
                    this_file_num_weights.push_back(m_mcgen_num_weights[i][j]);
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

                temp_weight_names.push_back(this_file_weight_names);
                temp_num_weights.push_back(this_file_num_weights);
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
        m_mcgen_weight_names = temp_weight_names;
        m_mcgen_num_weights = temp_num_weights;
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
        if(input_vec.empty())
            log<LOG_ERROR>(L"%1% || Input value: %2% does not exist in the vector (vector is empty)!") % __func__ % val;
        else
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
    m_vec_global_variable_index_start.clear();

    for(size_t io = 0; io < m_num_variables; ++io) {
        m_vec_global_variable_index_start.emplace_back();
    }

    size_t global_subchannel_index = 0;
    for(size_t im = 0; im < m_num_modes; im++){

        std::vector<size_t> mode_variable_start;
        for(size_t io = 0; io < m_num_variables; ++io) {
            mode_variable_start.push_back(im*m_num_variable_bins_mode_block[io]);
        }

        for(size_t id =0; id < m_num_detectors; id++){


            std::vector<size_t> detector_variable_start;
            std::vector<size_t> channel_variable_start;
            for(size_t io = 0; io < m_num_variables; ++io) {
                detector_variable_start.push_back(id*m_num_variable_bins_detector_block[io]);
                channel_variable_start.push_back(0);
            }

            for(size_t ic = 0; ic < m_num_channels; ic++){
                for(size_t sc = 0; sc < m_num_subchannels[ic]; sc++){

                    std::string temp_name  = m_mode_names[im] +"_" +m_detector_names[id]+"_"+m_channel_names[ic]+"_"+m_subchannel_names[ic][sc];

                    m_map_fullname_subchannel_index[temp_name] = global_subchannel_index;
                    m_vec_subchannel_index.push_back(global_subchannel_index);
                    m_vec_channel_index.push_back(ic);

                    for(size_t io = 0; io < m_num_variables; ++io) {
                        size_t global_variable_index = mode_variable_start[io] + detector_variable_start[io] + channel_variable_start[io] + sc*m_channel_variable_bins[ic][io].NBins();
                        m_vec_global_variable_index_start[io].push_back(global_variable_index);
                    }

                    ++global_subchannel_index;
                }
                for(size_t io = 0; io < m_num_variables; ++io) {
                    channel_variable_start[io] += m_channel_variable_bins[ic][io].NBins()*m_num_subchannels[ic];
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

void PROconfig::construct_variable_collapsing_matrices(){

    log<LOG_INFO>(L"%1% || Creating Collapsing Matrices ") % __func__; 

    for(size_t io = 0; io < m_num_variables; ++io) {
        variable_collapsing_matrices.push_back(Eigen::MatrixXf::Zero(m_num_variable_bins_total[io], m_num_variable_bins_total_collapsed[io]));
        log<LOG_INFO>(L"%1% || Creating Variable %2% Collapsing Matrix. m_num_variable_bins_total[io], m_num_variable_bins_total[io]_collapsed:  %3%  %4%") % __func__ % io % m_num_variable_bins_total[io] % m_num_variable_bins_total_collapsed[io];

        //construct the matrix by detector block
        Eigen::MatrixXf block_collapser = Eigen::MatrixXf::Zero(m_num_variable_bins_detector_block[io], m_num_variable_bins_detector_block_collapsed[io]);

        size_t channel_row_start = 0, channel_col_start = 0;
        for(size_t ic =0; ic != m_num_channels; ++ic){

            //first, build matrix for each channel block
            size_t total_num_bins_channel = m_num_subchannels[ic] * m_channel_variable_bins[ic][io].NBins();

            Eigen::MatrixXf channel_collapser = Eigen::MatrixXf::Zero(total_num_bins_channel, m_channel_variable_bins[ic][io].NBins());
            for(size_t col = 0; col != m_channel_variable_bins[ic][io].NBins(); ++col){
                for(size_t subch = 0; subch != m_num_subchannels[ic]; ++subch){
                    size_t row = subch * m_channel_variable_bins[ic][io].NBins() + col;
                    channel_collapser(row, col) = 1.0;
                }
            }

            // now, copy this matrix to detector block
            block_collapser(Eigen::seqN(channel_row_start, total_num_bins_channel), Eigen::seqN(channel_col_start, m_channel_variable_bins[ic][io].NBins())) = channel_collapser;
            channel_row_start += total_num_bins_channel;
            channel_col_start += m_channel_variable_bins[ic][io].NBins();
        }

        //okay! now stuff every detector block size_to the final collapse matrix
        for(size_t im = 0; im != m_num_modes; ++im){
            for(size_t id =0; id != m_num_detectors; ++id){
                size_t row_block_start = im * m_num_variable_bins_mode_block[io] + id * m_num_variable_bins_detector_block[io];
                size_t col_block_start = im * m_num_variable_bins_mode_block_collapsed[io] + id * m_num_variable_bins_detector_block_collapsed[io];
                variable_collapsing_matrices.back()(Eigen::seqN(row_block_start, m_num_variable_bins_detector_block[io]), Eigen::seqN(col_block_start, m_num_variable_bins_detector_block_collapsed[io])) = block_collapser;
            }
        }

    }

    // Build sparse companions. Each row of T has exactly one 1.0, so nnz = m_num_variable_bins_total[io].
    variable_collapsing_matrices_sparse.clear();
    variable_collapsing_matrices_sparse.reserve(m_num_variables);
    for(size_t io = 0; io < m_num_variables; ++io) {
        variable_collapsing_matrices_sparse.emplace_back(variable_collapsing_matrices[io].sparseView());
        variable_collapsing_matrices_sparse.back().makeCompressed();
    }
    return;
}

int PROconfig::HexToROOTColor(const std::string& hexColor) const{
    if (hexColor.length() != 7 || hexColor[0] != '#') {
        throw std::invalid_argument("Invalid hex color format. It should be in the format #RRGGBB.");
    }
    int r = 0, g = 0, b = 0;
    std::stringstream ss;
    ss << std::hex << hexColor.substr(1, 2);
    ss >> r;
    ss.clear();
    ss << std::hex << hexColor.substr(3, 2);
    ss >> g;
    ss.clear();
    ss << std::hex << hexColor.substr(5, 2);
    ss >> b;
    if(ss.fail()) {
        throw std::invalid_argument("Invalid hex color '" + hexColor + "': components must be hexadecimal.");
    }
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
    for (const auto& vec1 : m_channel_variable_bins){
        for (const auto& vec2 : vec1){ 
            for (const auto& vec3 : vec2.bin_edges) {
                unique_string << vecToString(vec3);
            }
        }
    }
    for (const auto& vec : m_mcgen_weight_names)
        for (const auto& vec2 : vec)
            unique_string << vecToString(vec2);

    for (const auto& vec : m_mcgen_eventweight_branch_names) 
        unique_string << vecToString(vec);

    unique_string << vecToString(m_mcgen_variation_allowlist);

    // apply_to_subchannel wildcards change which bins a systematic populates in the
    // cached SystStructs; empty map appends nothing so pre-existing hashes are unchanged.
    for (const auto& [sysname, pattern] : m_mcgen_variation_apply_to_subchannel)
        unique_string << sysname << "->" << pattern;

    for(const auto& vec: m_branch_variables){
        for(const auto& br: vec){
            unique_string << br->associated_hist << br->associated_systematic << br->model_rule;
            for(const auto& v: br->variable_names){
                unique_string << v;
            }
        }
    }

    log<LOG_DEBUG>(L"%1% || MurmurHash input uniue string %2% ") % __func__ % unique_string.str().c_str();

    MurmurHash3_x86_32(unique_string.str().c_str(), unique_string.str().size(), fixed_seed, &hash);

    log<LOG_INFO>(L"%1% || MurmurHash output hash %2% ") % __func__ % hash;

    return hash;
}

uint32_t PROconfig::CalcDetVarHash() const{
    int fixed_seed = 404;
    uint32_t hash;
    std::ostringstream unique_string;

    auto vecToString = [](const auto& vec) -> std::string {
        std::ostringstream oss;
        for (const auto& v : vec) {
            oss << v;
        }
        return oss.str();
    };

    // Binning and channel structure — DetVar propellers are filled into these bins,
    // so any change to channels or bin edges requires reprocessing.
    unique_string << vecToString(m_fullnames);
    for (const auto& vec1 : m_channel_variable_bins){
        for (const auto& vec2 : vec1){
            for (const auto& vec3 : vec2.bin_edges) {
                unique_string << vecToString(vec3);
            }
        }
    }

    // DetVar section content: filenames, POTs, names, section indices, and template content
    for(const auto& dv : m_detvar_files) {
        unique_string << dv.section_index << dv.name << dv.filename << dv.pot << dv.is_cv;
    }
    for(const auto& tmpl : m_detvar_xml_templates)
        unique_string << tmpl;

    // Include matching vars so that adding/removing cv_variation_matching_vars forces reprocessing
    for(const auto& sec_vars : m_detvar_matching_vars_per_section)
        for(const auto& v : sec_vars)
            unique_string << v;

    MurmurHash3_x86_32(unique_string.str().c_str(), unique_string.str().size(), fixed_seed, &hash);

    log<LOG_INFO>(L"%1% || MurmurHash detvar_hash output %2% ") % __func__ % hash;

    return hash;
}

PROconfig PROconfig::BuildDataConfig() const {
    if(!m_has_data_section) {
        log<LOG_ERROR>(L"%1% || BuildDataConfig called but no <data> section was found in XML!") % __func__;
        throw std::runtime_error("No <data> section in XML");
    }

    // Write the data XML string to a temporary file and load it as a PROconfig
    std::string tmpdir = std::filesystem::temp_directory_path().string();
    std::string tmpfile = tmpdir + "/profit_data_config_tmp_" + std::to_string(getpid()) + ".xml";

    {
        std::ofstream ofs(tmpfile);
        ofs << m_data_xml_string;
    }

    log<LOG_INFO>(L"%1% || Loading data config from temporary XML: %2%") % __func__ % tmpfile.c_str();
    // Pass i_prime explicitly: the child must fit the same variable as this config even when
    // that came from --fit-variable rather than from a fit="true" in the serialized bins.
    PROconfig dataconfig(tmpfile, m_bool_rate_only, static_cast<int>(i_prime));
    std::filesystem::remove(tmpfile);

    return dataconfig;
}

PROconfig PROconfig::BuildDetVarConfig(size_t file_index) const {
    if(!m_has_detvar_section) {
        log<LOG_ERROR>(L"%1% || BuildDetVarConfig called but no <DetVarFiles> section was found in XML!") % __func__;
        throw std::runtime_error("No <DetVarFiles> section in XML");
    }
    if(file_index >= m_detvar_files.size()) {
        log<LOG_ERROR>(L"%1% || BuildDetVarConfig called with file_index %2% but only %3% DetVar files exist") % __func__ % file_index % m_detvar_files.size();
        throw std::runtime_error("DetVar file index out of range");
    }

    const auto& dvfile = m_detvar_files[file_index];

    // Replace placeholders in the template for this file's section
    std::string xml_str = m_detvar_xml_templates[dvfile.section_index];
    {
        std::string fn_placeholder = "__DETVAR_FILENAME__";
        auto pos = xml_str.find(fn_placeholder);
        if(pos != std::string::npos) {
            xml_str.replace(pos, fn_placeholder.size(), dvfile.filename);
        }

        std::string tr_placeholder = "__DETVAR_TREENAME__";
        pos = xml_str.find(tr_placeholder);
        if(pos != std::string::npos) {
            xml_str.replace(pos, tr_placeholder.size(), dvfile.treename);
        }

        std::string pot_placeholder = "__DETVAR_POT__";
        pos = xml_str.find(pot_placeholder);
        if(pos != std::string::npos) {
            std::ostringstream pot_ss;
            pot_ss << std::scientific << dvfile.pot;
            xml_str.replace(pos, pot_placeholder.size(), pot_ss.str());
        }

        std::string frac_placeholder = "__DETVAR_PARTIAL_LOAD_FRAC__";
        pos = xml_str.find(frac_placeholder);
        if(pos != std::string::npos) {
            xml_str.replace(pos, frac_placeholder.size(), std::to_string(dvfile.partial_load_frac));
        }
    }

    // Write to temp file and load as PROconfig
    std::string tmpdir = std::filesystem::temp_directory_path().string();
    std::string tmpfile = tmpdir + "/profit_detvar_config_tmp_" + std::to_string(getpid()) + "_" + std::to_string(file_index) + ".xml";

    {
        std::ofstream ofs(tmpfile);
        ofs << xml_str;
    }

    log<LOG_INFO>(L"%1% || Loading DetVar config for '%2%' from temporary XML: %3%") % __func__ % dvfile.name.c_str() % tmpfile.c_str();
    // Inherit the parent's fitting variable (see BuildDataConfig) — PROfit.cxx falls back to
    // the parent's i_prime when the DetVar config's is out of range, so keep them identical.
    PROconfig dvconfig(tmpfile, m_bool_rate_only, static_cast<int>(i_prime));
    std::filesystem::remove(tmpfile);

    // Propagate matching var branch names directly onto the mini-config so PROcess_CAFAna can read them.
    dvconfig.m_detvar_matching_vars = m_detvar_matching_vars_per_section[dvfile.section_index];

    return dvconfig;
}

ROOTFormula::ROOTFormula(const std::string &name, const std::string &formula, TTree *t) {
    std::stringstream formula_reader(formula);
    // split the formula at a ";" into multiple values. used to be "," but that breaks arguments
    std::string this_formula;
    while(std::getline(formula_reader, this_formula, ';')) {
        log<LOG_DEBUG>(L"%1% || Compiling TTreeFormula '%2%' with name '%3%' on tree '%4%'") % __func__ % this_formula.c_str() % name.c_str() % t->GetName();
        auto f = std::make_unique<TTreeFormula>(name.c_str(), this_formula.c_str(), t);
        log<LOG_DEBUG>(L"%1% || TTreeFormula compiled: GetNdim()=%2%, GetNcodes()=%3%, GetNdata()=%4%") % __func__ % f->GetNdim() % f->GetNcodes() % f->GetNdata();
        // Check if formula compiled successfully
        if (f->GetNdim() == 0 && f->GetNcodes() == 0) {
            log<LOG_ERROR>(L"%1% || ERROR: TTreeFormula not compiled correctly for formula: %2%") % __func__ % this_formula.c_str();
            log<LOG_ERROR>(L"%1% || -- Tree name: %2%, Tree entries: %3%") % __func__ % t->GetName() % t->GetEntries();
            // List available branches for debugging
            log<LOG_ERROR>(L"%1% || -- Available branches in tree:") % __func__;
            if(t->GetListOfBranches()){
                for(int ib = 0; ib < std::min(t->GetListOfBranches()->GetEntries(), (int)50); ++ib){
                    log<LOG_ERROR>(L"%1% ||    branch: %2%") % __func__ % t->GetListOfBranches()->At(ib)->GetName();
                }
                if(t->GetListOfBranches()->GetEntries() > 50){
                    log<LOG_ERROR>(L"%1% ||    ... and %2% more branches") % __func__ % (t->GetListOfBranches()->GetEntries() - 50);
                }
            }
            // List friend tree branches too
            if(t->GetListOfFriends()){
                for(const TObject* fr : *t->GetListOfFriends()){
                    TTree* ftree = ((TFriendElement*)fr)->GetTree();
                    if(ftree && ftree->GetListOfBranches()){
                        log<LOG_ERROR>(L"%1% || -- Friend tree '%2%' branches:") % __func__ % ftree->GetName();
                        for(int ib = 0; ib < std::min(ftree->GetListOfBranches()->GetEntries(), (int)20); ++ib){
                            log<LOG_ERROR>(L"%1% ||    branch: %2%") % __func__ % ftree->GetListOfBranches()->At(ib)->GetName();
                        }
                    }
                }
            }
            exit(EXIT_FAILURE);
        }
        fs.push_back(std::move(f));
    }
    log<LOG_DEBUG>(L"%1% || Successfully compiled %2% formula(s) for '%3%'") % __func__ % fs.size() % name.c_str();
    treeNumber = -1;
}

BranchVariable::Value ROOTFormula::EvalInstance() {
    BranchVariable::Value ret;
    for (const std::unique_ptr<TTreeFormula> &f: fs) {
        ret.v.push_back(f->EvalInstance());
    }
    return ret;
}

void ROOTFormula::LoadEvent(unsigned eventno) {
    if (fs.size() == 0) return;

    const TTree *tree = fs[0]->GetTree();
    if (tree == nullptr) {
        log<LOG_ERROR>(L"%1% || ERROR: TTreeFormula::GetTree() returned nullptr at eventno=%2%, stored treeNumber=%3%") % __func__ % eventno % treeNumber;
        log<LOG_ERROR>(L"%1% || This formula has %2% sub-formula(s):") % __func__ % fs.size();
        for (size_t fi = 0; fi < fs.size(); ++fi) {
            log<LOG_ERROR>(L"%1% ||   [%2%] expression='%3%'  GetNdim=%4%  GetNcodes=%5%") % __func__ % fi
                % fs[fi]->GetTitle() % fs[fi]->GetNdim() % fs[fi]->GetNcodes();
        }
        log<LOG_ERROR>(L"%1% || ERROR: Maybe the TTreeFormula is using a variable that is not in the file?") % __func__;
        exit(EXIT_FAILURE);
    }

    int this_tree_number = tree->GetTreeNumber();

    // if we're on a new tree, refresh the formulas
    if (this_tree_number != treeNumber) {
        treeNumber = this_tree_number;
        for (const std::unique_ptr<TTreeFormula> &f: fs) {
            f->UpdateFormulaLeaves();
            f->GetNdata();
        }
    }
}

void ROOTFormula::ExtractExprTokens(const std::string& expr, std::set<std::string>& result) {
    // Canonical skip-list: ROOT aggregate functions, C math functions, ROOT constants,
    // and C++ keywords that can never be branch names.
    static const std::set<std::string> kSkipTokens = {
        "Sum", "Min", "Max", "Alt", "Count", "Iteration",
        "MinIf", "MaxIf", "SumIf",
        "abs", "fabs", "sqrt", "sin", "cos", "tan",
        "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh",
        "exp", "log", "log10", "pow",
        "ceil", "floor", "round", "fmod",
        "TMath",
        "kTRUE", "kFALSE",
        "true", "false",
        "int", "float", "double", "bool", "void",
        "if", "else", "for", "while", "do", "return",
    };
    const char* p = expr.c_str();
    while (*p) {
        if (isalpha((unsigned char)*p) || *p == '_') {
            const char* start = p;
            while (isalnum((unsigned char)*p) || *p == '_') p++;
            std::string token(start, p);
            if (kSkipTokens.find(token) == kSkipTokens.end())
                result.insert(std::move(token));
        } else {
            ++p;
        }
    }
}

void ROOTFormula::AddFormulaBranches(const TTreeFormula* f, std::set<std::string>& result) {
    if (!f) return;

    // Standard path: scalar branches registered via GetNcodes() / GetLeaf().
    for (int n = 0; n < f->GetNcodes(); n++) {
        TLeaf* leaf = f->GetLeaf(n);
        if (!leaf) continue;
        TBranch* br = leaf->GetBranch();
        if (br) {
            result.insert(std::string(br->GetName()));
            TBranch* mother = br->GetMother();
            if (mother && mother != br) result.insert(std::string(mother->GetName()));
        }
    }

    // Supplemental path: token-extract the expression string to catch vector branches
    // and Sum$()/Min$()/Max$() sub-expressions not reachable via GetLeaf().
    const char* title = f->GetTitle();
    if (title) ExtractExprTokens(std::string(title), result);
}

std::set<std::string> ROOTFormula::GetNeededBranchNames() const {
    std::set<std::string> result;
    for (const std::unique_ptr<TTreeFormula>& f : fs)
        AddFormulaBranches(f.get(), result);
    return result;
}

std::string ROOTFormula::FormulaName() const {
    std::string ret;
    bool delim = false;
    for (const std::unique_ptr<TTreeFormula> &f: fs) {
        if (delim) ret += ';';
        ret += f->PrintValue();
        delim = true;
    }
    return ret;
}

size_t PROconfig::Binning::NBins() const {
    if (bin_edges.size() == 0) return 0;
    size_t ret = 1;
    for (const std::vector<float> &b: bin_edges) {
        if (b.size() < 2) return 0;
        ret *= (b.size() - 1);
    }
    return ret;
}

// Project an input index across the full binning to a 1D index across the input dimension
size_t PROconfig::Binning::ProjectIndex(size_t ind, size_t dim) const {
    size_t div = 1;
    size_t stride = NBinsAlong(dim);
    for (unsigned i_vec = NDim()-1; i_vec > dim; i_vec--) {
        div *= NBinsAlong(i_vec);
        stride *= NBinsAlong(i_vec);
    }

    return (ind % stride) / div;
}

Eigen::VectorXf PROconfig::Binning::ProjectSpectra(const Eigen::VectorXf &in, size_t dim) const {
    if (in.size() % NBins() != 0) {
        log<LOG_ERROR>(L"%1% || Mismatch between input spectrum length (%2%) and number of bins (%3%). Returning empty array.") % __func__ % in.size() % NBins();
        return Eigen::VectorXf();
    }
    // This should be equal to num_detectors * num_subchannels. TODO: check?
    unsigned n_copies = in.size() / NBins();

    Eigen::VectorXf ret = Eigen::VectorXf::Zero(NBinsAlong(dim)*n_copies);

    for (unsigned i = 0; i < in.size(); i++) {
        unsigned i_copy = i / NBins();
        unsigned i_bin = i % NBins();
        ret(ProjectIndex(i_bin, dim) + i_copy*NBinsAlong(dim)) += in(i);
    }

    return ret;
}

Eigen::VectorXf PROconfig::Binning::ProjectSpectraErrors(const Eigen::VectorXf &in, size_t dim) const {
    if (in.size() % NBins() != 0) {
        log<LOG_ERROR>(L"%1% || Mismatch between input spectrum length (%2%) and number of bins (%3%). Returning empty array.") % __func__ % in.size() % NBins();
        return Eigen::VectorXf();
    }
    // This should be equal to num_detectors * num_subchannels. TODO: check?
    unsigned n_copies = in.size() / NBins();

    Eigen::VectorXf ret = Eigen::VectorXf::Zero(NBinsAlong(dim)*n_copies);

    for (unsigned i = 0; i < in.size(); i++) {
        unsigned i_copy = i / NBins();
        unsigned i_bin = i % NBins();
        ret(ProjectIndex(i_bin, dim) + i_copy*NBinsAlong(dim)) += in(i)*in(i); // sum of squares
    }

    return ret.array().sqrt();
}

// Bin an input value
int PROconfig::Binning::Bin(const std::vector<float> &v) const {
    if (NDim() == 0) return 0;

    if (v.size() != NDim()) {
        log<LOG_ERROR>(L"%1% || Mismatch between input variable dimensions (%2%) and bin dimensions (%3%). Returning -1.") % __func__ % v.size() % NDim(); 
        return -1;
    }

    size_t ret = 0;
    size_t stride = 1;
    for (int i_vec = NDim()-1; i_vec >=0; i_vec--) {
        const std::vector<float> &thisbin = bin_edges[i_vec];
        auto pos_iter = std::upper_bound(thisbin.begin(), thisbin.end(), v[i_vec]);
        size_t local_bin = pos_iter - thisbin.begin() - 1;

        if(pos_iter == thisbin.end() || pos_iter == thisbin.begin()){
            static int underflow_overflow_count = 0;
            underflow_overflow_count++;
            if(underflow_overflow_count <= 10) {
                log<LOG_DEBUG>(L"%1% || Value: %2% in Dim: %3% is in underflow or overflow bins, return bin of -1") % __func__ % v[i_vec] % i_vec;
                log<LOG_DEBUG>(L"%1% || Binning has bin lower edge: %2% and bin upper edge: %3%") % __func__ % thisbin.front() % thisbin.back();
                if(underflow_overflow_count == 10)
                    log<LOG_DEBUG>(L"%1% || (suppressing further underflow/overflow messages)") % __func__;
            }
            return -1;
        }

        ret += local_bin * stride;
        stride *= NBinsAlong(i_vec);
    }
    return ret;
}

// return widths along a dimension
std::vector<float> PROconfig::Binning::Widths(unsigned dim) const {
    std::vector<float> ret;
    for (int i = 0; i < (int)bin_edges[dim].size() - 1; i++) ret.push_back(bin_edges[dim][i+1] - bin_edges[dim][i]);
    return ret;
}
