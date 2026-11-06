// check_spline_response.C — assertion macro for the shared-DetVar suite test.
//
// Asserts a spline's response at knob value `knob` bin by bin, using the
// Splines/<name>_fixedpts_<b> TGraphs of a PROplot ROOT file (plot --with-splines).
//
//   root -l -b -q 'tests/check_spline_response.C("<tag>_PROplot.root", "<specs>", 1.0)'
//
// <specs> is a comma-separated list of  name;lo-hi;expected  where [lo,hi) are
// uncollapsed bin indices. Every bin in range must respond with `expected` (relative
// tolerance 1e-4) or be exactly flat at 1 (a bin with no CV events). Unless expected
// is 1, at least one bin per range must respond. Exits 0 iff all pass.

#include <cmath>
#include <string>
#include <vector>

static TGraph *find_graph(TDirectory *d, const std::string &name) {
    if (!d) return nullptr;
    TIter it(d->GetListOfKeys());
    while (TKey *k = (TKey *)it())
        if (name == k->GetName()) return dynamic_cast<TGraph *>(k->ReadObj());
    return nullptr;
}

int check_spline_response(const char *fname, const char *specs, double knob = 1.0) {
    TFile f(fname);
    if (f.IsZombie()) { printf("FAIL: cannot open %s\n", fname); return 1; }
    TDirectory *dir = (TDirectory *)f.Get("Splines");
    if (!dir) { printf("FAIL: no Splines/ in %s\n", fname); return 1; }
    int bad = 0;

    std::string all(specs);
    size_t pos = 0;
    while (pos <= all.size()) {
        size_t end = all.find(',', pos);
        if (end == std::string::npos) end = all.size();
        const std::string spec = all.substr(pos, end - pos);
        pos = end + 1;
        if (spec.empty()) continue;
        const size_t s1 = spec.find(';'), s2 = spec.find(';', s1 + 1);
        int lo = 0, hi = 0;
        if (s1 == std::string::npos || s2 == std::string::npos ||
            sscanf(spec.substr(s1 + 1, s2 - s1 - 1).c_str(), "%d-%d", &lo, &hi) != 2) {
            printf("FAIL: bad spec '%s'\n", spec.c_str()); ++bad; continue;
        }
        const std::string name = spec.substr(0, s1);
        const double expected = atof(spec.substr(s2 + 1).c_str());

        int n_resp = 0, n_flat = 0, n_wrong = 0;
        for (int b = lo; b < hi; ++b) {
            TGraph *g = find_graph(dir, name + "_fixedpts_" + std::to_string(b));
            if (!g) { printf("FAIL: no Splines/%s_fixedpts_%d\n", name.c_str(), b); ++n_wrong; continue; }
            int ik = -1;
            for (int i = 0; i < g->GetN(); ++i) if (fabs(g->GetX()[i] - knob) < 1e-6) ik = i;
            if (ik < 0) { printf("FAIL: %s bin %d has no point at knob %g\n", name.c_str(), b, knob); ++n_wrong; continue; }
            const double y = g->GetY()[ik];
            if (fabs(y - expected) <= 1e-4 * fabs(expected)) ++n_resp;
            else if (y == 1.0) ++n_flat;
            else { if (n_wrong < 5) printf("  %s bin %d: response %.7g, expected %.7g\n", name.c_str(), b, y, expected); ++n_wrong; }
        }
        const bool ok = n_wrong == 0 && (n_resp > 0 || expected == 1.0);
        printf("%s %s [%d,%d) expect %g: %d match, %d flat, %d wrong\n", ok ? "PASS" : "FAIL",
               name.c_str(), lo, hi, expected, n_resp, n_flat, n_wrong);
        if (!ok) ++bad;
    }
    return bad ? 1 : 0;
}
