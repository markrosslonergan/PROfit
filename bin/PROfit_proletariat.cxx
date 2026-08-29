#include "PROfit_common.h"

int run_proletariat(PROpt &options) {
    options.grid_opts.xml              = options.xmlname;
    options.grid_opts.analysis_tag     = options.analysis_tag;
    options.grid_opts.final_output_tag = options.final_output_tag;
    if(options.grid_sl7) options.grid_opts.singularity_image = PROletariatOptions::kImageSL7;
    if(options.grid_backend_str == "slurm") {
        options.grid_opts.backend = PROletariatOptions::Backend::Slurm;
    } else if(options.grid_backend_str != "jobsub") {
        log<LOG_ERROR>(L"%1% || Unknown --backend '%2%' (expected jobsub or slurm).") % __func__ % options.grid_backend_str.c_str();
        return 1;
    }
    PROletariat submitter(options.grid_opts);
    return submitter.Run();
}
