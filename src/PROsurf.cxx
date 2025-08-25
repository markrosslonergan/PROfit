#include "PROsurf.h"
#include "PROfitter.h"
#include "PROlog.h"

#include <Eigen/Eigen>

#include <cmath> 
#include <future>
#include <algorithm>

#include "TGraph.h"
#include "TLatex.h"
#include "TLine.h"
#include "TMarker.h"

using namespace PROfit;

std::vector<float> linspace(float start, float end, int N, bool endpoint = true) {
    std::vector<float> result;
    result.reserve(N);
    if (N == 0) return result;
    if (N == 1) {
        result.push_back(start);
        return result;
    }

    float step = (end - start) / (endpoint ? (N - 1) : N);
    for (int i = 0; i < N; ++i) {
        result.push_back(start + i * step);
    }
    return result;
}
std::vector<float> combined_sparse_dense(float Amin, float Amax, float CV, int Nsparse, int Ndense, float dense_width) {
    std::vector<float> sparse = linspace(Amin, Amax, Nsparse);  // global scan
    std::vector<float> dense = linspace(CV - dense_width / 2.0, CV + dense_width / 2.0, Ndense-1);  // local dense
    dense.push_back(CV);

    sparse.insert(sparse.end(), dense.begin(), dense.end());
    std::sort(sparse.begin(), sparse.end());

    sparse.erase(std::unique(sparse.begin(), sparse.end(),[](float a, float b) { return std::fabs(a - b) < 1e-8; }), sparse.end());

    return sparse;
}

PROsurf::PROsurf(PROmetric &metric,  size_t x_idx, size_t y_idx, size_t nbinsx, LogLin llx, float x_lo, float x_hi, size_t nbinsy, LogLin lly, float y_lo, float y_hi) : metric(metric), x_idx(x_idx), y_idx(y_idx), nbinsx(nbinsx), nbinsy(nbinsy), edges_x(Eigen::VectorXf::Constant(nbinsx + 1, 0)), edges_y(Eigen::VectorXf::Constant(nbinsy + 1, 0)), surface(nbinsx, nbinsy) {
    if(llx == LogAxis) {
        x_lo = std::log10(x_lo);
        x_hi = std::log10(x_hi);
    }
    if(lly == LogAxis) {
        y_lo = std::log10(y_lo);
        y_hi = std::log10(y_hi);
    }
    for(size_t i = 0; i < nbinsx + 1; i++)
        edges_x(i) = x_lo + i * (x_hi - x_lo) / nbinsx;
    for(size_t i = 0; i < nbinsy + 1; i++)
        edges_y(i) = y_lo + i * (y_hi - y_lo) / nbinsy;
}

void PROsurf::FillSurfaceStat(const PROconfig &config, const PROfitterConfig &fitconfig, std::string filename) {
    std::ofstream chi_file;
    if(!filename.empty()){
        chi_file.open(filename);
        chi_file << "Dimensions: " << nbinsx << " " << nbinsy << "\n";
        chi_file << "Fixed indices: " << x_idx << " " << y_idx << "\n";
        chi_file << "Parameters:\n";
        for(const auto &name: metric.GetModel().param_names) chi_file << name << "\n";

        chi_file << "xval yval chi2";
        // TODO: Not saving this info for stats only right now (but we should)
        //for(size_t i = 0; i < metric.GetModel().nparams; ++i)
        //    chi_file << " p" << i;
        chi_file << "\n";
    }

    // I think this will be needed for stat fits with more than 2 physics parameters
    (void)fitconfig;

    PROsyst dummy_syst;
    dummy_syst.fractional_covariance = Eigen::MatrixXf::Constant(config.m_num_bins_total, config.m_num_bins_total, 0);
    Eigen::VectorXf empty_vec;

    PROmetric *local_metric = metric.Clone();
    local_metric->override_systs(dummy_syst);
    float min_chi = 1e9;

    for(size_t i = 0; i < nbinsx; i++) {
        for(size_t j = 0; j < nbinsy; j++) {
            Eigen::VectorXf physics_params{{(float)edges_y(j), (float)edges_x(i)}};
            float fx = (*local_metric)(physics_params, empty_vec, false);
            if(fx < min_chi) min_chi = fx;
            surface(i, j) = fx;
        }
    }
    for(size_t i = 0; i < nbinsx; ++i) {
        for(size_t j = 0; j < nbinsy; ++j) {
            float fx = surface(i,j); 
            fx -= min_chi;
            surface(i,j) = fx;
            if(!filename.empty()){
                chi_file<<"\n"<<edges_x(i)<<" "<<edges_y(j)<<" "<<fx<<std::flush;
            }
        }
    }
    delete local_metric;
}

std::vector<profOut> PROfile::PROfilePointHelper(const PROsyst *systs, const PROfitterConfig &fitconfig, int offset, int stride, float minchi, bool with_osc, const Eigen::VectorXf& init_seed, uint32_t seed) {

    std::vector<std::vector<float>> raw_data = {
        {0.485, -0.86471, 0.0170137, 0.104528, -0.0283426, 0.00279879, 0.0208215, -0.0926952, 0.222069, 0.198824, 0.231268, 0.251005, -0.0184288, -0.000513931, -0.0361243, 0.0232172, 0.0802458, 0.441158, -0.345993, -0.411753, 0.0819521, -0.0645341, -0.107321, -0.0734786, 0.107736, -0.111059, 0.0244807, 0.022668},
        {0.474766, -0.867831, 0.0260973, 0.0810398, -0.0269792, 0.00807371, 0.02131, -0.111147, 0.250043, 0.214654, 0.243961, 0.239005, -0.00528075, -0.00149055, -0.0319146, 0.0213539, 0.0949611, 0.45352, -0.402601, -0.369121, 0.0633344, -0.0444957, -0.0889923, -0.0803323, 0.171943, -0.0863389, 0.0164291, 0.0227805},
        {0.829999, -0.976741, 0.0725738, 1.19297e-08, 0.0422631, 0.0385649, 0.00911757, -0.0399184, 0.0365365, 0.0400661, 0.291621, -0.0422943, 0.0236677, 0.0579808, -0.0353748, 0.0479033, 0.121751, 0.171561, -0.102017, -0.152211, 0.229242, 0.111926, -0.0964324, -0.185962, 0.0305081, 0.0656443, 0.0297324, 0.0355304},
        {0.839703, -0.948661, 0.0648656, 1.19297e-08, 0.0360992, 0.0328609, 0.0143282, -0.0345832, 0.0497163, 0.0354029, 0.297695, -0.0461125, 0.0162662, 0.0669471, -0.0533917, 0.0632727, 0.111775, 0.160662, -0.110152, -0.161047, 0.274411, 0.14283, -0.0906258, -0.192496, 0.0547937, 0.0610332, 0.0250901, 0.0301965},
        {0.994999, -0.941675, 0.0637537, 0.000121893, 8.16941e-05, 0.094999, 0.0456049, 0.0380098, -0.0224773, -0.0557426, 0.357454, 0.0275775, 0.00659798, 0.0682764, -0.0702659, 0.0859955, 0.186077, 0.0418417, -0.251211, -0.124622, 0.259458, 0.00545461, -0.0250737, 0.00102504, -0.146749, 0.00960102, 0.0214561, 0.024309},
        {0.995789, -0.941741, 0.0639269, 0, 5.62678e-05, 0.0950947, 0.0454555, 0.0377245, -0.0225568, -0.0554885, 0.357673, 0.0276098, 0.00677591, 0.0686042, -0.0703565, 0.0862388, 0.186091, 0.0418193, -0.251303, -0.124678, 0.259392, 0.00547226, -0.025135, 0.00105721, -0.146828, 0.00960091, 0.0214454, 0.0243171},
        {1.145, -0.57456, 0.067645, 1.04671e-05, 0.019199, 0.0483296, 0.0714033, -0.00604484, -0.0197749, -0.00255066, 0.689515, 0.105238, 0.00103528, 0.213119, -0.227853, 0.252444, -0.188227, -0.0596296, -0.172342, -0.176507, 0.263451, -0.0678924, -0.0391068, -0.0533925, 0.0969578, 0.0361359, 0.0518776, 0.104654},
        {1.14779, -0.556797, 0.0679164, 0, 0.0113129, 0.0580228, 0.0703737, -0.0114061, -0.0250086, 0.00371503, 0.707514, 0.108362, 0.00590898, 0.223448, -0.233248, 0.254173, -0.233123, -0.114031, -0.1604, -0.185797, 0.265368, -0.0446821, -0.0355603, -0.053471, 0.104471, 0.037458, 0.0548756, 0.114481},
        {1.24, -0.89031, 0.0698619, 0.000338879, -0.0478376, 0.0590649, 0.041511, 0.0337675, 0.00500578, -0.0287995, 0.339434, -0.0514084, -0.0246182, 0.0779438, -0.0930988, 0.112924, 0.22947, 0.167529, -0.349196, -0.18623, 0.327277, -0.0495066, -0.0482322, -0.0909566, 0.027103, 0.0142797, 0.0243903, 0.0404437},
        {1.23475, -0.88486, 0.0601797, 4.19329e-06, -0.0207065, 0.0614393, 0.0237653, 0.0412123, -0.0157849, -0.0449422, 0.356689, -0.0461959, 0.000408558, 0.0918443, -0.0993713, 0.109325, 0.216192, 0.198985, -0.352962, -0.18233, 0.338463, -0.0730607, -0.0597843, -0.0862176, 0.029396, 0.0189039, 0.0323959, 0.0403107},
        {1.325, -1.06888, 0.04121, 0, -0.0203858, 0.0520344, 0.00352051, 0.0177859, -0.0738371, -0.0345644, 0.249239, -0.0640054, 0.0248875, 0.0550519, -0.0550401, 0.0531485, 0.109395, 0.189289, -0.286025, -0.166324, 0.348242, -0.0730666, -0.0485811, -0.0611362, 0.0129237, 0.0278722, 0.0149234, 0.0285333},
        {1.32317, -1.06876, 0.0414288, 1.85708e-05, -0.0203373, 0.0520528, 0.00340401, 0.0176855, -0.0737865, -0.0346089, 0.249347, -0.0641049, 0.0248497, 0.05502, -0.0550107, 0.0531097, 0.109234, 0.18934, -0.28598, -0.166389, 0.348115, -0.0731993, -0.0485469, -0.0612793, 0.0129357, 0.027736, 0.0148973, 0.0285797},
        {1.36, -1.8751, 0.0378774, 0.00178928, -0.029671, 0.0607809, -0.0558929, 0.0675719, -0.101047, -0.107309, 0.163373, -0.094194, 0.0140625, -0.0581972, 0.0263483, -0.0392082, 0.162169, 0.182046, -0.31067, -0.136799, 0.349084, -0.0975921, -0.0402057, -0.0543526, -0.0273017, -0.0275676, 0.017591, -0.0129855},
        {1.39013, -1.87082, 0.0299344, 0.000422631, -0.0318158, 0.0576696, -0.0344629, 0.063662, -0.0861597, -0.087804, 0.144644, -0.100031, 0.00871589, -0.027407, 0.0253222, -0.0291903, 0.160625, 0.188065, -0.312189, -0.145754, 0.346673, -0.0865825, -0.0432711, -0.0691682, -0.0441005, -0.0175064, 0.00949213, -0.0178814},
        {1.395, -1.24523, 0.0337391, 0, -0.00844986, 0.0726176, -0.0203947, 0.0170948, -0.0412499, -0.0860204, 0.216865, -0.0826365, 0.0241319, 0.0216808, -0.0267606, 0.0440713, 0.135467, 0.204551, -0.294179, -0.162828, 0.333821, -0.0708865, -0.0514734, -0.0743465, -0.0252889, 0.0141299, 0.00657769, 0.0132076},
        {1.1478, -0.554229, 0.10318, 0, 0.0451436, 0.0703757, 0.121169, -0.0285836, 0.0331938, 0.0383888, 0.685639, 0.0987032, 0.0253443, 0.256263, -0.233728, 0.245331, -0.246089, -0.10546, -0.154191, -0.188906, 0.27629, -0.0443139, -0.0352279, -0.0446396, 0.0904858, 0.0500175, 0.0464356, 0.0688263},
        {1.435, -1.37003, 0.0608502, 0.000130616, 0.00355303, 0.053271, 0.00905853, 0.0504211, -0.0555714, -0.0681423, 0.208575, -0.0804202, -0.00339181, 0.0279455, -0.0285475, -0.00863772, 0.16109, 0.178251, -0.315403, -0.158008, 0.354755, -0.0934149, -0.0372077, -0.0647316, -0.03523, 0.00216871, -0.0119553, -0.0276224},
        {1.4393, -1.37164, 0.0574526, 4.59743e-12, -1.23028e-05, 0.0524512, 0.00584779, 0.0507838, -0.0597198, -0.0682003, 0.209914, -0.0789494, -0.00313852, 0.0260898, -0.0274945, -0.0100591, 0.163897, 0.18102, -0.31408, -0.157533, 0.35459, -0.0930033, -0.0388525, -0.0658024, -0.0344873, 0.00296245, -0.0108327, -0.0293229}
    };

    std::vector<Eigen::VectorXf> hack_seed;
    for (const auto& row : raw_data) {
        Eigen::VectorXf vec(row.size());
        for (size_t i = 0; i < row.size(); ++i) {
            vec(i) = row[i];
        }
        hack_seed.push_back(vec);
    }



    std::vector<profOut> outs;
    // Make a local copy for this thread
    PROmetric *local_metric = metric.Clone();
    int nparams = local_metric->GetModel().nparams + systs->GetNSplines();
    int nstep = 18;

    Eigen::VectorXf ub, lb, tub, tlb;

    if(with_osc) {
        lb = Eigen::VectorXf::Constant(nparams, -3.0);
        ub = Eigen::VectorXf::Constant(nparams, 3.0);
        size_t nphys = local_metric->GetModel().nparams;
        //set physics to correct values
        for(size_t j=0; j<nphys; j++){
            ub(j) = local_metric->GetModel().ub(j);
            lb(j) = local_metric->GetModel().lb(j); 
        }
        //upper lower bounds for splines
        for(int j = nphys; j < nparams; ++j) {
            lb(j) = systs->spline_lo[j-nphys];
            ub(j) = systs->spline_hi[j-nphys];
        }
    } else {
        ub = Eigen::VectorXf::Map(systs->spline_hi.data(), systs->spline_hi.size());
        lb = Eigen::VectorXf::Map(systs->spline_lo.data(), systs->spline_lo.size());
        nparams = systs->GetNSplines();
    }

    //loop over this threads todo list
    for(int i=offset; i<nparams;i+=stride) {
        tlb = lb;
        tub = ub;

        local_metric->reset();

        size_t which_spline= i;
        bool isphys = which_spline < local_metric->GetModel().nparams;
        profOut output;

        log<LOG_INFO>(L"%1% || THREADS %2% in this batch if ( %3%,%4% )") % __func__ %  i % offset % stride;


        Eigen::VectorXf last_bf;
        if(init_seed.norm()>0) last_bf= init_seed;
        int reset = -1;

        //first get what values to sample
        std::vector<float> test_values;

        //if not physis do normal
        if(!isphys){
            for (int j = 0; j <= nstep; ++j) {
                int k;
                if (j <= nstep - nstep / 2) {
                    k = nstep / 2 + j;  // Forward direction
                } else {
                    if(reset<0){
                        reset = test_values.size();
                    }
                    k = nstep - j;  // Backward direction
                }
                float which_value =  std::isinf(lb(which_spline)) ? -3 + (ub(which_spline) - (-3)) * k / (float)nstep :   lb(which_spline) + (ub(which_spline) - lb(which_spline)) * k / (float)nstep;
                test_values.push_back(which_value);       
            }
        }else{
            //if its physics, 
            test_values = combined_sparse_dense(std::isinf(lb(which_spline)) ? -3 : lb(which_spline), ub(which_spline), init_seed(which_spline), nstep, nstep*0.8, 0.2 );
            if(which_spline==0){
                test_values.clear();
                //for(float ll=0.4; ll<1.5; ll+=0.01){
                for(float ll=0.4; ll<1.5; ll+=0.025){
                    test_values.push_back(ll);
                }
            }
            if(which_spline==1){
            test_values.clear();
            for(float ll=-1.6; ll<=0; ll+=0.02){
                test_values.push_back(ll);
            }
            }
        }


        //and minimize
        int cnt=0;
        for(auto &v: test_values){
            float which_value = v;    
            float fx;
            output.knob_vals.push_back(which_value);

            tlb[which_spline] = which_value;
            tub[which_spline] = which_value;

            local_metric->setBounds(tlb,tub);
            local_metric->fixSpline(which_spline,which_value);

            if(reset==cnt) last_bf = init_seed;
            PROfitter fitter(tub, tlb, fitconfig, seed+i);
            std::vector<Eigen::VectorXf> hack_seeds2 = hack_seed;
            hack_seeds2.push_back(last_bf);
            if(last_bf.size()>0){
                //fx = fitter.Fit(*local_metric,last_bf);
                fx = fitter.Fit(*local_metric,hack_seeds2);
            }else{
                fx = fitter.Fit(*local_metric,hack_seeds2);
            }
            output.knob_chis.push_back(fx - minchi);
            last_bf = fitter.best_fit;
            local_metric->freeParams(); 


            std::string spec_string = "";
            for(auto &f : fitter.best_fit) spec_string+=" "+std::to_string(f); 
            log<LOG_INFO>(L"%1% || Fixed value of spline # %2% is value %3%, has a chi post of : %4% (i %5% nstep %6% ") % __func__ % which_spline % which_value % fx % i % nstep;
            log<LOG_INFO>(L"%1% || at a BF param value of @ %2%") % __func__ %  spec_string.c_str();

            cnt++;
        }    //end step loop        
        output.sort();
        outs.push_back(output);

    }//end thread

    delete local_metric;

    return outs;
}

std::vector<surfOut> PROsurf::PointHelper(const PROfitterConfig &fitconfig, std::vector<surfOut> multi_physics_params, int start, int end, uint32_t seed){

    std::vector<surfOut> outs;

    // Make a local copy for this thread
    PROmetric *local_metric = metric.Clone();

    for(int i=start; i<end;i++){
        local_metric->reset();

        surfOut output;
        std::vector<float> physics_params = multi_physics_params[i].grid_val;
        output.grid_val = physics_params;
        output.grid_index = multi_physics_params[i].grid_index;

        int nparams = local_metric->GetModel().nparams + local_metric->GetSysts().GetNSplines() - 2;

        if(nparams == 0) {
            Eigen::VectorXf empty_vec, 
                params = Eigen::VectorXf::Map(physics_params.data(), physics_params.size());
            output.chi = (*local_metric)(params, empty_vec, false);
            output.best_fit = params; 
            outs.push_back(output);
            continue;
        }

        Eigen::VectorXf lb(nparams+2);
        lb << local_metric->GetModel().lb, Eigen::VectorXf::Map(local_metric->GetSysts().spline_lo.data(), local_metric->GetSysts().spline_lo.size());
        Eigen::VectorXf ub(nparams+2);
        ub << local_metric->GetModel().ub, Eigen::VectorXf::Map(local_metric->GetSysts().spline_hi.data(), local_metric->GetSysts().spline_hi.size());

        lb(x_idx) = multi_physics_params[i].grid_val[1];
        ub(x_idx) = multi_physics_params[i].grid_val[1];
        lb(y_idx) = multi_physics_params[i].grid_val[0];
        ub(y_idx) = multi_physics_params[i].grid_val[0];


        PROfitter fitter(ub, lb, fitconfig, seed+i);
        if(i!=start){
            output.chi = fitter.Fit(*local_metric, outs.back().best_fit);
        }else{
            output.chi = fitter.Fit(*local_metric);
        }
        output.best_fit = fitter.best_fit;
        outs.push_back(output);
    }

    delete local_metric;

    return outs;
}


void PROsurf::FillSurface(const PROfitterConfig &fitconfig, std::string filename, PROseed &proseed, int nThreads) {
    std::ofstream chi_file;
    if(!filename.empty()){
        chi_file.open(filename);
    }

    std::vector<surfOut> grid;
    for(size_t i = 0; i < nbinsx; i++) {
        for(size_t j = 0; j < nbinsy; j++) {
            std::vector<int> grid_pts = {(int)i,(int)j};
            std::vector<float> physics_params = {(float)edges_y(j), (float)edges_x(i)};  //deltam^2, sin^22thetamumu
            surfOut pt; pt.grid_val = physics_params; pt.grid_index = grid_pts;
            grid.push_back(pt);
        }
    }

    int loopSize = grid.size();
    int chunkSize = loopSize / nThreads;

    std::vector<std::future<std::vector<surfOut>>> futures; 

    log<LOG_INFO>(L"%1% || Starting THREADS  : %2% , Loops %3%, Chunks %4%") % __func__ % nThreads % loopSize % chunkSize;

    for (int t = 0; t < nThreads; ++t) {
        int start = t * chunkSize;
        int end = (t == nThreads - 1) ? loopSize : start + chunkSize;
        futures.emplace_back(std::async(std::launch::async, [&, start, end]() {
                    return this->PointHelper(fitconfig, grid, start, end, proseed.getThreadSeeds()->at(t));
                    }));

    }

    std::vector<surfOut> combinedResults;
    for (auto& fut : futures) {
        std::vector<surfOut> result = fut.get();
        combinedResults.insert(combinedResults.end(), result.begin(), result.end());
    }

    if(filename != "") {
        chi_file << "Dimensions: " << nbinsx << " " << nbinsy << "\n";
        chi_file << "Fixed indices: " << x_idx << " " << y_idx << "\n";
        chi_file << "Parameters:\n";
        for(const auto &name: metric.GetModel().param_names) chi_file << name << "\n";
        for(const auto &name: metric.GetSysts().spline_names) chi_file << name << "\n";

        chi_file << "\nxval yval chi2";
        for(size_t i = 0; i < metric.GetModel().nparams + metric.GetSysts().GetNSplines(); ++i)
            chi_file << " p" << i;
    }
    float min_chi = 1e9;
    for(const auto &item: combinedResults) {
        if(item.chi < min_chi) min_chi = item.chi;
    }
    for (const auto& item : combinedResults) {
        log<LOG_INFO>(L"%1% || Finished  : %2% %3% %4%") % __func__ % item.grid_val[1] % item.grid_val[0] % (item.chi - min_chi);
        surface(item.grid_index[0], item.grid_index[1]) = item.chi - min_chi;
        results.push_back({item.grid_index[0], item.grid_index[1], item.best_fit, (item.chi-min_chi)});
        if(filename != "") {
            chi_file<<"\n"<<item.grid_val[1]<<" "<<item.grid_val[0]<<" "<<(item.chi-min_chi);
            for(float val: item.best_fit)
                chi_file << " " << val;
        }
    }
}

std::vector<float> findMinAndBounds(TGraph *g, float val, float lo, float hi) {
    float step = 0.001;
    int n = g->GetN();
    float minY = 1e9, minX = 0;
    for (int i = 0; i < n; ++i) {
        double x, y;
        g->GetPoint(i, x,y);
        if (y < minY) {
            minY = y;
            minX = x;
        }
    }
    //..ok so minX is the min and Currentl minY is the chi^2. Want this to be delta chi^2

    float leftX = minX, rightX = minX;

    // Search to the left of the minimum
    for (float x = minX; x >= lo; x -= step) {
        float y = g->Eval(x) - minY; //DeltaChi^2
        if (y >= val) {
            leftX = x;
            break;
        } else if(x - step < lo) {
            // If at end of loop and haven't found left side
            leftX = lo;
        }
    }

    // Search to the right of the minimum
    for (float x = minX; x <= hi; x += step) {
        float y = g->Eval(x)-minY;
        if (y >= val) {
            rightX = x;
            break;
        } else if(x + step > hi) {
            // If at end of loop and haven't found right side
            rightX = hi;
        }
    }

    return {minX,leftX,rightX};
}


PROfile::PROfile(const PROconfig &config, const PROsyst &systs, const PROmodel &model, PROmetric &metric, PROseed &proseed, const PROfitterConfig &fitconfig, std::string filename, float minchi, bool with_osc, int nThreads, const Eigen::VectorXf & init_seed, const Eigen::VectorXf & true_params) : metric(metric) {
    LBFGSpp::LBFGSBSolver<float> solver(fitconfig.param);
    int nparams = systs.GetNSplines() + model.nparams*with_osc;
    std::vector<float> physics_params; 

    //hack
    std::vector<float> priorX;
    std::vector<float> priorY;

    for(int i=0; i<=30;i++){
        float which_value = -3.0+0.2*i;
        priorX.push_back(which_value);
        priorY.push_back(which_value*which_value);

    }
    std::unique_ptr<TGraph> gprior = std::make_unique<TGraph>(priorX.size(), priorX.data(), priorY.data());

    std::vector<std::string> names;
    if(with_osc) for(const auto& name: model.pretty_param_names) names.push_back(name);
    for(const auto &name: systs.spline_names) names.push_back(name);

    int loopSize = nparams;
    if(nThreads>loopSize){
        nThreads = loopSize;
        log<LOG_INFO>(L"%1% || nThreads is < loopSize (nparams) : %2% <  %3%. Setting equal ") % __func__ % nThreads % loopSize ;
    }

    int chunkSize = loopSize / nThreads;

    std::vector<std::future<std::vector<profOut>>> futures; 

    log<LOG_INFO>(L"%1% || Starting THREADS  : %2% , Loops %3%, Chunks %4%") % __func__ % nThreads % loopSize % chunkSize;

    for (int t = 0; t < nThreads; ++t) {
        std::string  strD = "";
        for(int i=t; i<nparams;i+=nThreads) {
            strD+=std::to_string(i);
        }
        log<LOG_INFO>(L"%1% || THREAD #%2% runs pts: %3% ") % __func__ % t % strD.c_str();

        futures.emplace_back(std::async(std::launch::async, [&, t]() {
                    return this->PROfilePointHelper(&systs, fitconfig, t, nThreads, minchi, with_osc, init_seed, proseed.getThreadSeeds()->at(t));
                    }));

    }

    std::vector<profOut> combinedResults(nparams);
    int offset = 0;
    int stride = nThreads;
    for (auto& fut : futures) {
        std::vector<profOut> result = fut.get();
        for(size_t i = 0; i < result.size(); ++i)
            combinedResults.at(offset+i*stride) = result.at(i);
        ++offset;
    }


    //create all graphs, used directly in first setion
    for(auto & out: combinedResults){
        log<LOG_INFO>(L"%1% || Knob Values: %2%") % __func__ %  out.knob_vals;
        log<LOG_INFO>(L"%1% || Knob Chis: %2%") % __func__ %  out.knob_chis;
        std::unique_ptr<TGraph> g = std::make_unique<TGraph>(out.knob_vals.size(), out.knob_vals.data(), out.knob_chis.data());
        graphs.push_back(std::move(g));
    }

    //Analyze them, used in later section
    //plot 2sigma also? default no, as its messier
    bool twosig = false;

    std::vector<float> values1_errup;
    std::vector<float> values1_errdown;

    std::vector<float> barvalues_err;

    std::vector<float> values2_up;
    std::vector<float> values2_down;


    log<LOG_INFO>(L"%1% || Getting BF, +/- one sigma ranges. Is Two sigma turned on? : %2% ") % __func__ % twosig;

    size_t count = 0;
    for(auto &g:graphs){
        //if(metric->GetModel().nparams)continue;
        float lo = count < metric.GetModel().nparams ? metric.GetModel().lb(count) :
            metric.GetSysts().spline_lo[count - metric.GetModel().nparams];
        if(std::isinf(lo)) lo = lo < 0 ? -5 : 5;
        float hi = count < metric.GetModel().nparams ? metric.GetModel().ub(count) :
            metric.GetSysts().spline_hi[count - metric.GetModel().nparams];
        if(std::isinf(hi)) hi = hi < 0 ? -5 : 5;
        std::vector<float> tmp = findMinAndBounds(g.get(),1.0, lo, hi);
        barvalues.push_back(float(count)+0.5);
        barvalues_err.push_back(0.3);
        bfvalues.push_back(tmp[0]);
        values1_down.push_back(tmp[1]);
        values1_up.push_back(tmp[2]);
        values1_errdown.push_back(abs(tmp[1]-tmp[0]));
        values1_errup.push_back(abs(tmp[2]-tmp[0]));
        log<LOG_DEBUG>(L"%1% || Results of findMinAndBounds : %2% %3% %4% ") % __func__ % tmp[0] % tmp[1] % tmp[2];
        log<LOG_DEBUG>(L"%1% || Barvalues : %2% %3% %4% %5%") % __func__ % count % barvalues[count] % barvalues_err[count] % barvalues_err[count];
        log<LOG_DEBUG>(L"%1% || Bfvalues : %2% %3% ") % __func__ % count % bfvalues[count];
        log<LOG_DEBUG>(L"%1% || RangeValues : %2% %3% %4% ") % __func__ % count % values1_down[count] % values1_up[count];
        log<LOG_DEBUG>(L"%1% || ErrValues : %2% %3% %4% ") % __func__ % count % values1_errdown[count] % values1_errup[count];
        if(twosig){
            std::vector<float> tmp2 = findMinAndBounds(g.get(),4.0,lo, hi);
            values2_down.push_back(abs(tmp2[1]-tmp[0]));
            values2_up.push_back(abs(tmp2[2]-tmp[0]));
        }
        count++;
    }
    onesig = TGraphAsymmErrors(barvalues.size(),barvalues.data(), bfvalues.data(), barvalues_err.data(), barvalues_err.data(), values1_errdown.data(), values1_errup.data());

}

void PROfile::Plot(const PROconfig &config, const PROsyst &systs, const PROmodel &model, PROmetric &metric, PROseed &proseed, std::string filename, bool with_osc, const Eigen::VectorXf& init_seed, const Eigen::VectorXf & true_params, bool mask_osc) {

    int nparams = systs.GetNSplines() + model.nparams*with_osc;
    int nBins = nparams;
    std::vector<std::string> names;
    if(with_osc) for(const auto& name: model.pretty_param_names) names.push_back(name);
    for(const auto &name: systs.spline_names) names.push_back(name);

    std::vector<float> priorX;
    std::vector<float> priorY;

    for(int i=0; i<=30;i++){
        float which_value = -3.0+0.2*i;
        priorX.push_back(which_value);
        priorY.push_back(which_value*which_value);

    }
    std::unique_ptr<TGraph> gprior = std::make_unique<TGraph>(priorX.size(), priorX.data(), priorY.data());

    //First plot
    int depth = std::ceil((nparams+model.nparams)/4.0);
    TCanvas *c =  new TCanvas(filename.c_str(), filename.c_str() , 350*4, 350*depth);
    c->Divide(4,depth);


    size_t zoom_shift = 0;
    for(size_t w = 0; w< graphs.size(); w++ ){
        if(mask_osc && w < model.nparams) continue;

        c->cd(w+1+zoom_shift);
        std::string xval = w < model.nparams ? "Log_{10}(" + model.pretty_param_names[w]+")" :"#sigma Shift"  ;
        std::string tit = (w < model.nparams ? names[w] :config.m_mcgen_variation_plotname_map.at(names[w]))+ ";"+xval+"; #Delta#Chi^{2}";
        graphs[w]->SetTitle(tit.c_str());
        graphs[w]->Draw("AL");
        graphs[w]->SetLineWidth(2);
        graphs[w]->GetYaxis()->SetTitleSize(0.04);             
        graphs[w]->GetYaxis()->SetLabelSize(0.04);            
        graphs[w]->GetXaxis()->SetTitleSize(0.04);             
        graphs[w]->GetXaxis()->SetLabelSize(0.04);            
        graphs[w]->GetYaxis()->SetRangeUser(0, graphs[w]->GetHistogram()->GetMaximum());

        TLine* line = new TLine(graphs[w]->GetXaxis()->GetXmin(), 1, graphs[w]->GetXaxis()->GetXmax(), 1);
        line->SetLineStyle(3);  // Dotted line style (1 is solid, 2 is dashed, 3 is dotted)
        line->SetLineWidth(1);  // Thin line
        line->SetLineColor(kBlack);  // Set color (black for visibility)
        line->Draw();

        if(w<model.nparams) graphs[w]->SetLineColor(kBlue-7);

        if(w>=model.nparams){
            gprior->Draw("L same");
            gprior->SetLineStyle(2);
            gprior->SetLineWidth(2);
            gprior->SetLineColor(kRed-7);
            graphs[w]->GetYaxis()->SetRangeUser(0, std::min(graphs[w]->GetHistogram()->GetMaximum(),10.0));
        }

        if(w==model.nparams-1){
            //on past physics param, lets do a quick zoom, stepping back though the physics param
            for(int zs = model.nparams-1; zs>=0; zs--){
                c->cd(w+1+zs+1);
                TGraph * graphClone = new TGraph(*graphs[w-zs]);
                graphClone->Draw("AL");
                std::string newTitle = std::string(graphClone->GetTitle()) + " Zoomed 1#sigma";
                graphClone->SetTitle(newTitle.c_str());
                graphClone->SetLineColor(kViolet);
                float vd = std::min(values1_down[w-zs],values1_up[w-zs]) ;
                float vu = std::max(values1_down[w-zs],values1_up[w-zs]) ;
                float pd = (vd>0 ? vd*0.9 : vd*1.1);
                float pu = (vu >0 ? vu*1.1 : vu*0.9);
                graphClone->GetXaxis()->SetLimits(pd,pu); 
                graphClone->GetYaxis()->SetRangeUser(0, std::max(graphClone->Eval(pu),graphClone->Eval(pd))*1.1) ;
                graphClone->GetYaxis()->SetTitleSize(0.04);             
                graphClone->GetYaxis()->SetLabelSize(0.04);            
                graphClone->GetXaxis()->SetTitleSize(0.04);             
                graphClone->GetXaxis()->SetLabelSize(0.04);            

                log<LOG_INFO>(L"%1% || Zoom boundaries X %2% %3% Y %4% %5%  ") % __func__ % pd % pu % 0.0 % (std::max(graphClone->Eval(pu),graphClone->Eval(pd))*1.1)  ;

                TLine *line1 = new TLine(pd, 1, pu, 1);
                line1->SetLineStyle(3);  
                line1->SetLineWidth(1);  
                line1->SetLineColor(kBlack); 
                line1->Draw();

                TLine* line2 = new TLine(vd, graphClone->Eval(vd) ,vd, 0);
                line2->SetLineStyle(3);  
                line2->SetLineWidth(1);  
                line2->SetLineColor(kBlack); 
                line2->Draw();

                TLine *line3 = new TLine(vu, graphClone->Eval(vu) ,vu, 0);
                line3->SetLineStyle(3);  
                line3->SetLineWidth(1);  
                line3->SetLineColor(kBlack); 
                line3->Draw();
            }
            zoom_shift=model.nparams;
        }
    }

    c->SaveAs((filename+".pdf").c_str(),"pdf");

    delete c;

    //Next version
    TCanvas *c2 =  new TCanvas((filename+"1sigma").c_str(), (filename+"1sigma").c_str() , 20*nparams, 400);
    c2->cd();
    c2->SetBottomMargin(0.25);
    c2->SetRightMargin(0.05);

    log<LOG_DEBUG>(L"%1% || Are all lines the same : %2% %3% %4% %5% %6%") % __func__ % nBins % barvalues.size() % bfvalues.size() % values1_down.size() % values1_up.size() ;

    float minVal = *std::min_element(values1_down.begin(), values1_down.end());
    float maxVal = *std::max_element(values1_up.begin(), values1_up.end());

    onesig.SetFillColor(kBlue-7);
    onesig.SetStats(0);
    //onesig.SetMinimum(min(-1.2,minVal*1.2));
    onesig.SetMinimum(minVal*1.1);
    onesig.SetMaximum(maxVal*1.1);

    onesig.GetXaxis()->SetNdivisions(barvalues.size());  // Set number of tick marks
    onesig.GetXaxis()->SetLabelSize(0);  // Hide default numerical labels

    onesig.SetTitle("");
    TGraphAsymmErrors todraw = onesig;
    if(mask_osc) {
        for(size_t i = 0; i < model.nparams; ++i) {
            todraw.SetPoint(i, 0,0);
            todraw.SetPointError(i, 0, 0, 0, 0);
        }
    }
    todraw.Draw("A2");
    //onesig.Draw("A2");
    //onesig.GetYaxis()->SetTitle("#sigma Shift");
    todraw.GetYaxis()->SetTitle("Posterior 1#sigma Error");
    todraw.GetYaxis()->SetTitleOffset(0.8);

    float y_min = todraw.GetMinimum();
    for (size_t i = 0; i < barvalues.size(); ++i) {
        std::string label = i < model.nparams ? "Log_{10}(" + model.pretty_param_names[i]+")" : config.m_mcgen_variation_plotname_map.at(names[i]);
        TLatex* text = new TLatex(barvalues[i], y_min - 0.05, label.c_str());  // Position text below axis
        text->SetTextAlign(13);  
        text->SetTextSize(0.03); 
        text->SetTextAngle(-45); 
        text->Draw();
    }

    c2->Update();

    //if (twosig) {
    //    TGraphAsymmErrors *h2 = new TGraphAsymmErrors(barvalues.size(),barvalues.data(), bfvalues.data(), barvalues_err.data(), barvalues_err.data(), values2_down.data(), values2_up.data());
    //    h2->SetFillColor(38);
    //    h2->SetStats(0);
    //    h2->SetTitle("");
    //    h2->Draw("A2");
    //    h2->GetYaxis()->SetTitle("");
    //}


    TLine l(0,0,nBins+0.5,0);
    l.SetLineStyle(2);
    l.SetLineColor(kBlack);
    l.SetLineWidth(1);
    l.Draw();
    TLine l2(0,-1,nBins+0.5,-1);
    l2.SetLineStyle(3);
    l2.SetLineColor(kBlack);
    l2.SetLineWidth(1);
    l2.Draw();
    TLine l3(0,1,nBins+0.5,1);
    l3.SetLineStyle(3);
    l3.SetLineColor(kBlack);
    l3.SetLineWidth(1);
    l3.Draw();



    for (int i = 0; i < nBins; ++i) {
        TMarker* initstar = new TMarker(i+0.5, init_seed[i], 29);
        initstar->SetMarkerSize(0.6); 
        initstar->SetMarkerColor(kBlue); 
        initstar->Draw();

        if (i < true_params.size()) {

            TMarker* truestar = new TMarker(i+0.5, true_params[i], 29);
            truestar->SetMarkerSize(0.5); 
            truestar->SetMarkerColor(kRed); 
            truestar->Draw();
        }

        TMarker* star = new TMarker(i+0.5, bfvalues[i], 29);
        star->SetMarkerSize(0.5); 
        star->SetMarkerColor(kBlack); 
        star->Draw();
    }



    c2->SaveAs((filename+"_1sigma.pdf").c_str(),"pdf");
    delete c2;

    return;
}

