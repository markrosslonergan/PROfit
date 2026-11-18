// check_applyto.C — assertion macro for the apply_to_subchannel suite tests.
//
// Asserts that a systematic's response is EXACTLY confined to a bin range of the
// uncollapsed spectrum, using the PROplot ROOT file (run_short_tests.sh section 9).
//
//   root -l -b -q 'tests/check_applyto.C("<tag>_PROplot.root", "<specs>")'
//
// <specs> is a comma-separated list of  name;kind;range  where
//   name  = the systematic's XML name (for flat/norm that is the NAME:percent text)
//   kind  = cov     -> Covariance/<name>_cov TH2D: nonzero only inside range x range
//           total   -> Covariance/total_frac_cov, same check (every syst restricted)
//           spline  -> Splines/<name>_curve_<b> TGraphs: flat at 1 outside range,
//                      at least one varying curve inside
//   range = ND | FD | lo-hi   (uncollapsed bin indices, [lo,hi); ND/FD assume the
//           fake_sbn_v2.xml layout: one mode, detectors ND then FD with identical
//           channel structure, so ND is exactly the first half of the bins)
// The default specs reproduce the original t26aptzero assertion. Exits 0 iff all pass.

#include <cstring>
#include <string>
#include <vector>

static TObject *find_in_dir(TDirectory *d, const std::string &name) {
    if (!d) return nullptr;
    TIter it(d->GetListOfKeys());
    while (TKey *k = (TKey *)it())
        if (name == k->GetName()) return k->ReadObj();   // key names may contain ':'
    return nullptr;
}

int check_applyto(const char *fname, const char *specs = "RPA_CCQE;cov;FD,DetSys1;spline;ND") {
    TFile f(fname);
    if (f.IsZombie()) { printf("FAIL: cannot open %s\n", fname); return 1; }
    int bad = 0;

    // Bin count from the total covariance (always present with --with-covar).
    TH2D *tot = (TH2D *)find_in_dir((TDirectory *)f.Get("Covariance"), "total_frac_cov");
    if (!tot) { printf("FAIL: no Covariance/total_frac_cov in %s\n", fname); return 1; }
    const int N = tot->GetNbinsX(), H = N / 2;

    std::string all(specs);
    size_t pos = 0;
    while (pos <= all.size()) {
        size_t end = all.find(',', pos);
        if (end == std::string::npos) end = all.size();
        std::string spec = all.substr(pos, end - pos);
        pos = end + 1;
        if (spec.empty()) continue;
        std::vector<std::string> fld;
        size_t p = 0;
        for (int i = 0; i < 3; ++i) {
            size_t q = (i < 2) ? spec.find(';', p) : std::string::npos;
            fld.push_back(spec.substr(p, q == std::string::npos ? std::string::npos : q - p));
            if (q == std::string::npos) break;
            p = q + 1;
        }
        if (fld.size() != 3) { printf("FAIL: bad spec '%s'\n", spec.c_str()); ++bad; continue; }
        const std::string &name = fld[0], &kind = fld[1], &range = fld[2];
        int lo = 0, hi = N;
        if (range == "ND") { lo = 0; hi = H; }
        else if (range == "FD") { lo = H; hi = N; }
        else if (sscanf(range.c_str(), "%d-%d", &lo, &hi) != 2) { printf("FAIL: bad range '%s'\n", range.c_str()); ++bad; continue; }
        auto in = [&](int b0) { return b0 >= lo && b0 < hi; };   // 0-based bin

        if (kind == "cov" || kind == "total") {
            TH2D *cov = (kind == "total") ? tot
                : (TH2D *)find_in_dir((TDirectory *)f.Get("Covariance"), name + "_cov");
            if (!cov) { printf("FAIL: no Covariance/%s_cov in %s\n", name.c_str(), fname); ++bad; continue; }
            double inside = 0, outside = 0;
            for (int i = 1; i <= N; ++i)
                for (int j = 1; j <= N; ++j) {
                    const double v = fabs(cov->GetBinContent(i, j));
                    if (in(i - 1) && in(j - 1)) inside += v; else outside += v;
                }
            printf("%s (%s [%d,%d)): sum|cov| inside=%.6g outside=%.6g\n", name.c_str(), kind.c_str(), lo, hi, inside, outside);
            if (outside != 0) { printf("FAIL: %s has nonzero covariance outside its scope\n", name.c_str()); ++bad; }
            if (inside <= 0) { printf("FAIL: %s has zero covariance inside its scope (systematic lost entirely?)\n", name.c_str()); ++bad; }
        } else if (kind == "spline") {
            TDirectory *sd = (TDirectory *)f.Get("Splines");
            if (!sd) { printf("FAIL: no Splines directory in %s\n", fname); ++bad; continue; }
            int nbins = 0, in_varying = 0, out_varying = 0;
            for (int b = 0;; ++b) {
                TGraph *g = (TGraph *)sd->Get(Form("%s_curve_%d", name.c_str(), b));
                if (!g) { nbins = b; break; }
                double dev = 0;
                for (int q = 0; q < g->GetN(); ++q) dev = std::max(dev, fabs(g->GetY()[q] - 1.0));
                if (dev > 1e-6) (in(b) ? in_varying : out_varying)++;
            }
            printf("%s (spline [%d,%d)): %d bins, varying inside=%d outside=%d\n", name.c_str(), lo, hi, nbins, in_varying, out_varying);
            if (nbins != N) { printf("FAIL: %s spline bin count %d != covariance dim %d\n", name.c_str(), nbins, N); ++bad; }
            if (out_varying != 0) { printf("FAIL: %s has non-flat splines outside its scope\n", name.c_str()); ++bad; }
            if (in_varying <= 0) { printf("FAIL: %s has no varying splines inside its scope (systematic lost entirely?)\n", name.c_str()); ++bad; }
        } else { printf("FAIL: unknown kind '%s'\n", kind.c_str()); ++bad; }
    }

    printf(bad ? "check_applyto: FAIL (%d)\n" : "check_applyto: OK\n", bad);
    return bad ? 1 : 0;
}
