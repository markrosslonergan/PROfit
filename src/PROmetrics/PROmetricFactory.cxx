#include "PROmetrics/PROmetricFactory.h"
#include "PROmetrics/PROchi_neyman.h"
#include "PROmetrics/PROchi_pearson.h"
#include "PROmetrics/PROchi_CNP.h"
#include "PROmetrics/PROpoisson.h"
#include "PROlog.h"

#include <stdexcept>

namespace PROfit {

    std::unique_ptr<PROmetric> MakeMetric(const std::string &chi2_kind,
                                          const PROconfig &config, const PROpeller &prop,
                                          const PROsyst *systs, const PROmodel &model,
                                          const PROdata &data, PROmetric::EvalStrategy strat,
                                          bool shape_only,
                                          std::vector<float> physics_param_fixed) {
        // Defensive canonicalization: idempotent for PROfit-binary callers (main already
        // canonicalized), protects direct library callers.
        const std::string kind = PROmetric::canonicalizeMetricName(chi2_kind);
        // Statistical overlap (PROoverlap.h) is Neyman-only, and once the MC carries it every
        // dataset must come with its cells, or that fit would silently ignore the overlap.
        if(data.has_overlap && kind != "neyman") {
            log<LOG_ERROR>(L"%1% || ERROR: statistical overlap between analyses is only supported by the neyman metric (-c neyman), not %2%.") % __func__ % kind.c_str();
            throw std::runtime_error("statistical overlap needs -c neyman");
        }
        if(prop.overlap && !data.has_overlap) {
            log<LOG_ERROR>(L"%1% || ERROR: the MC carries statistical-overlap information but this dataset does not (its toy generator does not throw overlap cells, or the real data lacks overlap_matching_vars). Refusing an uncorrelated fit.") % __func__;
            throw std::runtime_error("dataset without overlap cells");
        }
        if(kind == "neyman")
            return std::unique_ptr<PROmetric>(new PROchi("", config, prop, systs, model, data, strat, shape_only, physics_param_fixed));
        if(kind == "pearson")
            return std::unique_ptr<PROmetric>(new PROchi_pearson("", config, prop, systs, model, data, strat, shape_only, physics_param_fixed));
        if(kind == "CNP")
            return std::unique_ptr<PROmetric>(new PROCNP("", config, prop, systs, model, data, strat, shape_only, physics_param_fixed));
        if(kind == "poisson")
            return std::unique_ptr<PROmetric>(new PROpoisson("", config, prop, systs, model, data, strat, shape_only, physics_param_fixed));
        log<LOG_ERROR>(L"%1% || Unrecognized chi2 function %2%. Options: neyman (default), pearson, CNP, poisson (legacy aliases: PROchi, PROCNP, Poisson).") % __func__ % chi2_kind.c_str();
        return nullptr;
    }

}
