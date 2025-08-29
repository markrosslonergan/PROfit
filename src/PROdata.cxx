#include "PROdata.h"
#include "PROtocall.h"

using namespace PROfit;

PROdata::PROdata(size_t num_bins):
    nbins(num_bins),
    spec(Eigen::VectorXf::Zero(num_bins)),
    error(Eigen::VectorXf::Zero(num_bins)){
    }

size_t PROdata::GetNbins() const{
    return nbins;
}

void PROdata::Zero(){
    log<LOG_INFO>(L"%1% || Zero out data spectrum") % __func__ ;
    spec.setZero();
    error.setZero();
    return;
}

void PROdata::Print() const {
    std::string spec_string = "";
    for(auto &f : spec) spec_string+=" "+std::to_string(f); 
    log<LOG_INFO>(L"%1% || %2%" ) % __func__ % spec_string.c_str();
    return;
}


void PROdata::Fill(int bin_index, float weight){
    //Removed to help speed up filling
    spec(bin_index) += weight;
    float tmp_err = error(bin_index);
    error(bin_index) = std::sqrt(tmp_err*tmp_err + weight*weight);

    return;
}

void PROdata::QuickFill(int bin_index, float weight){
    spec(bin_index) += weight;
    return;
}

TH1D PROdata::toTH1D(const PROconfig &inconfig, int global_channel_index, int other_index) const {
    int local_channel_index = inconfig.GetLocalChannelIndex(global_channel_index);

    log<LOG_DEBUG>(L"%1% || in A ") % __func__ ;
    int global_bin_start = other_index < 0 ? inconfig.GetCollapsedGlobalBinStart(global_channel_index) : inconfig.GetCollapsedGlobalOtherBinStart(global_channel_index, other_index);
    //set up hist specs
    log<LOG_DEBUG>(L"%1% || in B ") % __func__ ;
    int nbins = other_index < 0 ? inconfig.m_channel_num_bins[local_channel_index] : inconfig.m_channel_num_other_bins[local_channel_index][other_index];
    log<LOG_DEBUG>(L"%1% || in C ") % __func__ ;
    const std::vector<float>& bin_edges = other_index < 0 ? inconfig.GetChannelBinEdges(local_channel_index) : inconfig.GetChannelOtherBinEdges(local_channel_index, other_index);
    log<LOG_DEBUG>(L"%1% || in D tt %2% of %3%  with chan in %4%") % __func__ % local_channel_index % inconfig.m_channel_names.size() % global_channel_index;
    std::string hist_name = inconfig.m_channel_names[local_channel_index] + " Data";
    log<LOG_DEBUG>(L"%1% || in E ") % __func__ ;
    std::string xaxis_title = other_index < 0 ? inconfig.m_channel_units[local_channel_index] : inconfig.m_channel_other_units[local_channel_index][other_index];

    log<LOG_DEBUG>(L"%1% || in F ") % __func__ ;
    //fill 1D hist
    TH1D hSpec(hist_name.c_str(),hist_name.c_str(), nbins, &bin_edges[0]); 
    hSpec.GetXaxis()->SetTitle(xaxis_title.c_str());
    for(int i = 1; i <= nbins; ++i){
        hSpec.SetBinContent(i, spec(global_bin_start + i -1));
        hSpec.SetBinError(i, error(global_bin_start + i -1));
    }

    return hSpec;
}

void PROdata::toROOT(const PROconfig& inconfig, const std::string& output_name){
    TFile f(output_name.c_str(), "recreate");
    const std::vector<std::string>& all_channels = inconfig.m_channel_names;
    for(size_t i = 0; i!= all_channels.size(); ++i){
        TH1D h = this->toTH1D(inconfig, i);
        h.Write();
    }
    f.Write();
}

bool PROdata::SameDim(const PROdata& a, const PROdata& b){
    if(a.nbins != b.nbins){
        log<LOG_ERROR>(L"%1% || Two spectra have different bins: %2 vs. %3") % __func__ % a.nbins % b.nbins;
        return false;
    }
    return true;
}

PROdata PROdata::operator+(const PROdata& b) const{
    //check if dimension 
    if(!PROdata::SameDim(*this, b)){
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }


    Eigen::VectorXf sum_spec = this->spec + b.spec;
    Eigen::VectorXf error_spec = this->eigenvector_sqrt_quadrature_sum(this->error, b.error); 
    return PROdata(sum_spec, error_spec);
}

PROdata& PROdata::operator+=(const PROdata& b){
    //check if dimension 
    if(!PROdata::SameDim(*this, b)){
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }


    this->spec += b.spec;
    this->error = this->eigenvector_sqrt_quadrature_sum(this->error, b.error);

    return *this;
}

PROdata PROdata::operator-(const PROdata& b) const{
    //check if dimension 
    if(!PROdata::SameDim(*this, b)){
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }


    Eigen::VectorXf sum_spec = this->spec - b.spec;
    Eigen::VectorXf error_spec = this->eigenvector_sqrt_quadrature_sum(this->error, b.error); 
    return PROdata(sum_spec, error_spec);
}

PROdata& PROdata::operator-=(const PROdata& b){
    //check if dimension 
    if(!PROdata::SameDim(*this, b)){
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }


    this->spec -= b.spec;
    this->error = this->eigenvector_sqrt_quadrature_sum(this->error, b.error);

    return *this;
}

PROdata PROdata::operator/(const PROdata& b) const{
    //check if dimension 
    if(!PROdata::SameDim(*this, b)){
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }


    Eigen::VectorXf ratio_spec = this->eigenvector_division(this->spec, b.spec);

    //calculate relative error, and sqrt of quadratic sum of relative error
    Eigen::VectorXf this_relative_error = this->eigenvector_division(this->error, this->spec), b_relative_error = this->eigenvector_division(b.error, b.spec);
    Eigen::VectorXf ratio_relative_error = this->eigenvector_sqrt_quadrature_sum(this_relative_error, b_relative_error);

    Eigen::VectorXf ratio_error = this->eigenvector_multiplication(ratio_spec, ratio_relative_error);
    return PROdata(ratio_spec, ratio_error); 
}

PROdata& PROdata::operator/=(const PROdata& b) {
    //check if dimension 
    if(!PROdata::SameDim(*this, b)){
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }



    this->spec =  this->eigenvector_division(this->spec, b.spec);
    //calculate relative error, and sqrt of quadratic sum of relative error
    Eigen::VectorXf this_relative_error = this->eigenvector_division(this->error, this->spec), b_relative_error = this->eigenvector_division(b.error, b.spec);
    Eigen::VectorXf ratio_relative_error = this->eigenvector_sqrt_quadrature_sum(this_relative_error, b_relative_error);

    this->error  = this->eigenvector_multiplication(this->spec, ratio_relative_error);
    return *this; 
}

PROdata PROdata::operator*(float scale) const{

    Eigen::VectorXf scaled_spec = scale * this->spec, scaled_error = scale * this->error;
    return PROdata(scaled_spec, scaled_error);
}

PROdata& PROdata::operator*=(float scale){

    this->spec *= scale; 
    this->error *= scale;
    return *this;
}

Eigen::VectorXf PROdata::eigenvector_sqrt_quadrature_sum(const Eigen::VectorXf& a, const Eigen::VectorXf& b) const{
    int nbin = a.size();
    Eigen::VectorXf error_spec = Eigen::VectorXf::Zero(nbin); 
    error_spec = ((a.array()).square() + (b.array()).square()).sqrt();
    return error_spec;
}

Eigen::VectorXf PROdata::eigenvector_division(const Eigen::VectorXf& a, const Eigen::VectorXf& b) const{
    int nbin = a.size();
    Eigen::VectorXf ratio_spec = Eigen::VectorXf::Zero(nbin);
    for(int i = 0; i != nbin; ++i){
        if(b(i) == 0){
            if(a(i) !=0 ){
                log<LOG_ERROR>(L"%1% || Divide by Zero. Numerator: %2%, denominator: %3% ") % __func__ % a(i) % b(i);
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }else{
                //log<LOG_DEBUG>(L"%1% || Both numerator and denominator are zero, setting the ratio to 1.") % __func__;
                ratio_spec(i) = 1.0;
            }
        }else
            ratio_spec(i) = a(i) / b(i);
    }
    return ratio_spec;
}

Eigen::VectorXf PROdata::eigenvector_multiplication(const Eigen::VectorXf& a, const Eigen::VectorXf& b) const{
    int nbin = a.size();
    Eigen::VectorXf ratio_spec = Eigen::VectorXf::Zero(nbin);
    for(int i = 0; i != nbin; ++i){
        ratio_spec(i) = a(i) * b(i);
    }
    return ratio_spec;
}

void PROdata::plotSpectrum(const PROconfig& inconfig, const std::string& output_name) const{
    bool div_bin = true;
    int n_subplots = inconfig.m_num_channels*inconfig.m_num_modes*inconfig.m_num_detectors;
    log<LOG_DEBUG>(L"%1% || Creatign canvas with  %2% subplots") % __func__ % n_subplots;

    TCanvas *c =  new TCanvas(output_name.c_str(), output_name.c_str(), 800*n_subplots, 600);
    c->Divide(n_subplots,1);

    std::vector<std::unique_ptr<TH1D>> hists;  
    std::vector<std::unique_ptr<TLegend>> legs;

    size_t global_channel_index = 0;
    for(size_t im = 0; im < inconfig.m_num_modes; im++){
        for(size_t id =0; id < inconfig.m_num_detectors; id++){
            for(size_t ic = 0; ic < inconfig.m_num_channels; ic++){
                c->cd(global_channel_index+1);

                std::unique_ptr<TLegend> leg = std::make_unique<TLegend>(0.59,0.89,0.59,0.89);
                leg->SetFillStyle(0);
                leg->SetLineWidth(0);
                legs.push_back(std::move(leg));

                std::unique_ptr<TH1D> htmp = std::make_unique<TH1D>(toTH1D(inconfig, ic));
                htmp->SetLineWidth(3);
                htmp->SetLineColor(kBlack);
                hists.push_back(std::move(htmp));  // Move the unique_ptr into the vector

                if(div_bin){
                    hists.back()->Scale(1, "width");
                }

                hists.back()->SetTitle((inconfig.m_mode_names[im]  +" "+ inconfig.m_detector_names[id]+" "+ inconfig.m_channel_names[ic]).c_str());
                hists.back()->GetXaxis()->SetTitle(inconfig.m_channel_units[ic].c_str());
                hists.back()->GetYaxis()->SetTitle("Events");
                hists.back()->Draw("hist");
                ++global_channel_index;
            }//end chan
        }//end det
    }//end mode

    c->SaveAs(("PROplot_"+output_name+".pdf").c_str(),"pdf");

    delete c;
    return;
}

Eigen::VectorXf PROdata::Normalize(const PROconfig &inconfig, const PROspec &target_spec) const{
    //Single channel version
    //Eigen::VectorXf test =  spec* target_spec.Spec().array().sum() / spec.array().sum();

    Eigen::VectorXf collapsed_target  = CollapseMatrix(inconfig,target_spec.Spec());
    Eigen::VectorXf output(spec.size());
    size_t global_channel_index = 0;
    for(size_t im = 0; im < inconfig.m_num_modes; im++){
        for(size_t id =0; id < inconfig.m_num_detectors; id++){
            for(size_t ic = 0; ic < inconfig.m_num_channels; ic++){

                size_t nbin =  inconfig.m_channel_num_bins[inconfig.GetLocalChannelIndex(global_channel_index)];
                size_t startCollBin = inconfig.GetCollapsedGlobalBinStart(global_channel_index);

                float target_chan_sum = collapsed_target.segment(startCollBin,nbin).sum();
                float data_chan_sum = spec.segment(startCollBin,nbin).sum();
                //log<LOG_INFO>(L"%1% || sums %2% data %3%") % __func__ % target_chan_sum % data_chan_sum;
                output.segment(startCollBin,nbin).setConstant(target_chan_sum/data_chan_sum);
                global_channel_index++;
            }
        }
    }


    return spec.array()*output.array();
}

