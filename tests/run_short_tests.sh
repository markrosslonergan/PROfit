#!/usr/bin/env bash
#
# PROfit short test suite.
#
# Runs `process` and then one short instance of every major PROfit workflow
# (global fits in all three metrics, profile, surface incl. AMR, plot variants,
# FC, the full adaptive-FC pipeline, MCMC, benchmarks, and the PROjector
# two-stage pre-fit/projected fit) against a fixed fake ND+FD SBN config.
# Everything is seeded and single-threaded so two runs of the same code are
# bit-reproducible and two tags can be compared with compare_tags.sh.
#
# Usage:
#   tests/run_short_tests.sh <TAG> [XML]
#
#   TAG   Analysis tag; all outputs land in tests/runs/<TAG>/.
#   XML   Config to use. Default:
#         working_dir/Neutrino2026/fake_sbn_v2.xml
#
# Environment overrides:
#   PROFIT_BIN            PROfit executable   (default: <repo>/build/bin/PROfit)
#   PROFIT_TEST_MCDIR     Directory holding fake_sbn_mc_{ND,FD}.root referenced
#                         by the XML (default: directory containing the XML).
#                         The XML's hardcoded /exp/... path is rewritten to it.
#   PROFIT_TEST_OUTDIR    Where run directories go (default: <repo>/tests/runs)
#   PROFIT_TEST_TIMEOUT   Per-test timeout in seconds (default: 1800)
#
# Exit code: number of failed tests (0 = all passed).

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAG="${1:?usage: run_short_tests.sh <TAG> [XML]}"
XML_IN="${2:-$REPO/working_dir/Neutrino2026/fake_sbn_v2.xml}"
BIN="${PROFIT_BIN:-$REPO/build/bin/PROfit}"
MCDIR="${PROFIT_TEST_MCDIR:-$(cd "$(dirname "$XML_IN")" && pwd)}"
OUTBASE="${PROFIT_TEST_OUTDIR:-$REPO/tests/runs}"
TIMEOUT="${PROFIT_TEST_TIMEOUT:-1800}"
RUNDIR="$OUTBASE/$TAG"

[ -x "$BIN" ]    || { echo "ERROR: PROfit binary not found/executable: $BIN"; exit 99; }
[ -f "$XML_IN" ] || { echo "ERROR: XML not found: $XML_IN"; exit 99; }

mkdir -p "$RUNDIR/logs"
cd "$RUNDIR"

# Localize the XML: the reference config points its MCFile entries at a fixed
# /exp/... path; rewrite that directory to wherever the fake MC actually lives.
# Both tags of a comparison must use the same MCDIR or the config hash differs.
sed "s|/exp/uboone/data/users/markross|$MCDIR|g" "$XML_IN" > local_test.xml

for f in "$MCDIR/fake_sbn_mc_ND.root" "$MCDIR/fake_sbn_mc_FD.root"; do
    [ -f "$f" ] || { echo "ERROR: fake MC file missing: $f (set PROFIT_TEST_MCDIR)"; exit 99; }
done

# Fixed, deterministic base arguments. -n 1 is REQUIRED for bit-reproducibility
# (thread scheduling changes AMR warm-start ordering); --preset fast keeps every
# individual fit sub-second (a single value sets both the global and scan
# presets; do NOT pass two values — CLI11's greedy vector parsing would eat the
# subcommand name).
COMMON=(-x local_test.xml -t "$TAG" -n 1 -v 2 --seed 405 --preset fast)

# Physics axes for the nueapp model of this config.
AXES=(--xvar sinsq2thme --yvar dmsq --xlo 1e-4 --xhi 1 --ylo 1e-2 --yhi 1e2)

PASS=0; FAIL=0
SUMMARY="$RUNDIR/summary.txt"
: > "$SUMMARY"

note() { echo "$*" | tee -a "$SUMMARY"; }

# run_test <name> <args...>  — expects exit 0.
run_test() {
    local name="$1"; shift
    local t0=$SECONDS
    if timeout "$TIMEOUT" "$BIN" "${COMMON[@]}" -o "$name" "$@" > "logs/$name.log" 2>&1; then
        local nerr; nerr=$(grep -c "ERROR" "logs/$name.log" || true)
        note "PASS  $name  ($((SECONDS-t0))s, ${nerr} ERROR lines)"
        PASS=$((PASS+1))
    else
        local rc=$?
        note "FAIL  $name  ($((SECONDS-t0))s, exit $rc) -- see logs/$name.log"
        tail -n 5 "logs/$name.log" | sed 's/^/        /'
        FAIL=$((FAIL+1))
    fi
}

# run_test_tag <name> <otag> <args...> — run_test with an explicit -o, for
# multi-stage workflows whose artifacts are keyed by -o (fc-adaptive).
run_test_tag() {
    local name="$1" otag="$2"; shift 2
    local t0=$SECONDS
    if timeout "$TIMEOUT" "$BIN" "${COMMON[@]}" -o "$otag" "$@" > "logs/$name.log" 2>&1; then
        local nerr; nerr=$(grep -c "ERROR" "logs/$name.log" || true)
        note "PASS  $name  ($((SECONDS-t0))s, ${nerr} ERROR lines)"
        PASS=$((PASS+1))
    else
        local rc=$?
        note "FAIL  $name  ($((SECONDS-t0))s, exit $rc) -- see logs/$name.log"
        tail -n 5 "logs/$name.log" | sed 's/^/        /'
        FAIL=$((FAIL+1))
    fi
}

# expect_fail <name> <args...> — passes only if PROfit exits NONzero (validation
# paths must refuse bad input loudly, not limp on).
expect_fail() {
    local name="$1"; shift
    if timeout "$TIMEOUT" "$BIN" "${COMMON[@]}" -o "$name" "$@" > "logs/$name.log" 2>&1; then
        note "FAIL  $name  (expected a refusal but PROfit exited 0)"
        FAIL=$((FAIL+1))
    else
        note "PASS  $name  (correctly refused)"
        PASS=$((PASS+1))
    fi
}

# expect_fail_tag <name> <otag> <args...> — expect_fail with an explicit -o.
expect_fail_tag() {
    local name="$1" otag="$2"; shift 2
    if timeout "$TIMEOUT" "$BIN" "${COMMON[@]}" -o "$otag" "$@" > "logs/$name.log" 2>&1; then
        note "FAIL  $name  (expected a refusal but PROfit exited 0)"
        FAIL=$((FAIL+1))
    else
        note "PASS  $name  (correctly refused)"
        PASS=$((PASS+1))
    fi
}

note "PROfit short test suite"
note "  tag: $TAG"
note "  xml: $XML_IN"
note "  bin: $BIN"
note "  git: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) $(git -C "$REPO" diff --quiet 2>/dev/null || echo '(dirty)')"
note "  date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
note "----------------------------------------------------------------------"

# --- 0. MC processing (creates <TAG>_prop.bin / <TAG>_syst.bin) --------------
run_test t00process process
[ -f "${TAG}_prop.bin" ] || { note "FATAL t00process produced no ${TAG}_prop.bin; aborting."; exit 98; }

# --- 1. Global fits across metrics and data options --------------------------
run_test t01global        --use-fake-data global
run_test t02globalcnp     --use-fake-data -c PROCNP global
# Poisson has no covariance term: it refuses covariance systs (incl. MCStat) unless excluded.
run_test t03globalpoisson --use-fake-data -c Poisson --exclude-systs MCStat RPA_CCQE NormCCMEC CoulombCCQE MaCCRES MvCCRES global
expect_fail t03cpoissoncov --use-fake-data -c poisson global
run_test t03bglobalpearson --use-fake-data -c pearson global
run_test t04statonly      --use-fake-data --statonly global
run_test t05inject        --use-fake-data -i dmsq 1 sinsq2thme 0.01 global
run_test t06pseudoexp     --use-fake-data --pseudo-experiment global

# --- 2. Profile (legacy 18-point scan, and the PRObe adaptive scan) -----------
run_test t07profile       --use-fake-data profile
run_test t07probe         --use-fake-data profile --probe

# --- 3. Surfaces --------------------------------------------------------------
run_test t08surface       --use-fake-data surface -g 4 "${AXES[@]}"
run_test t09surfaceamr    --use-fake-data surface -g 4 "${AXES[@]}" --surface-amr --amr-initial 4 --amr-levels 1

# --- 4. Plotting variants -----------------------------------------------------
# t10 opts into the covariance plots (off by default since --with-covar);
# t11/t12 run the default (no Covar.pdf / ROOT Covariance dir).
run_test t10plot          --use-fake-data plot --with-splines --with-covar
run_test t11plotwidth     --use-fake-data --scale-by-width plot
run_test t12plotbkgsub    --use-fake-data plot --bkg-subtract background

# --- 4b. Shape-only (prediction rescaled onto the data per channel; every
# systematic shape-projected). Covers all three chi2 families that differ in
# their statistical term, the scan warm-start path, FC (flag must reach the
# per-universe metrics), and the width+area plot combination.
run_test t12sglobal       --use-fake-data --shapeonly global
run_test t12sglobalcnp    --use-fake-data --shapeonly -c CNP global
run_test t12sglobalpois   --use-fake-data --shapeonly -c poisson --exclude-systs MCStat RPA_CCQE NormCCMEC CoulombCCQE MaCCRES MvCCRES global
run_test t12sprofile      --use-fake-data --shapeonly profile
run_test t12sfc           --use-fake-data --shapeonly fc -u 2
run_test t12splotwidth    --use-fake-data --shapeonly --scale-by-width plot

# --- 5. Feldman-Cousins -------------------------------------------------------
run_test t13fc            --use-fake-data fc -u 2

# --- 6. Adaptive FC pipeline (stages share -o so artifacts chain together) ----
AFC=(fc-adaptive --throws 2 --prepass-amr-initial 4 4 --prepass-amr-levels 1
     --p-thresh 0.33 --baseline-level 1 "${AXES[@]}")
run_test_tag t14afcmesh   t14afc --use-fake-data "${AFC[@]}" --mode build-mesh
run_test_tag t14afcbank   t14afc --use-fake-data "${AFC[@]}" --mode init-bank --n-pe-min 1 --n-pe-max 2
run_test_tag t14afcprint  t14afc --use-fake-data "${AFC[@]}" --mode print-bank
run_test_tag t14afcasimov t14afc --use-fake-data "${AFC[@]}" --mode asimov
run_test_tag t14afcbrazil t14afc --use-fake-data "${AFC[@]}" --mode brazil --n-brazil-throws 2

# --- 7. MCMC + benchmark smoke ------------------------------------------------
run_test t15mcmc          --use-fake-data mcmc --nchains 1
run_test t16scaletest     --use-fake-data scale-test -N 50 --tests fillspectra,metric

# --- 7b. Systematic selection (--exclude-systs / --syst-list) -----------------
# By name + tag (MC-stat must stay in the fit), by plotname (drops MC-stat on
# request), a tag-based --syst-list, and a typo'd name, which must be refused.
run_test t40exclude       --use-fake-data --poisson-throw --exclude-systs RPA_CCQE flux global
run_test t40bexclmcstat   --use-fake-data --poisson-throw --exclude-systs "MC Stats" global
run_test t41systlist      --use-fake-data --poisson-throw --syst-list xsec MCStat global
expect_fail t42excltypo   --use-fake-data --exclude-systs NotASyst global

# --- 8. PROjector two-stage pre-fit / projected fit ---------------------------
run_test t17pjprefit      --use-fake-data --projector-prefit "_ND_" global
CONSTRAINT="${TAG}_t17pjprefit_PROjector_constraint.bin"
if [ -f "$CONSTRAINT" ]; then
    run_test t18pjglobal  --use-fake-data --projector "$CONSTRAINT" global
    run_test t19pjfc      --use-fake-data --projector "$CONSTRAINT" fc -u 2
else
    note "FAIL  t18pjglobal (missing $CONSTRAINT)"; FAIL=$((FAIL+1))
    note "FAIL  t19pjfc     (missing $CONSTRAINT)"; FAIL=$((FAIL+1))
fi
# Partial-channel pattern (fullosc is one subchannel of the nue channel) and a
# match-everything pattern must both be refused.
expect_fail t20pjpartial  --use-fake-data --projector-prefit "fullosc" global
expect_fail t21pjall      --use-fake-data --projector-prefit "nu_" global

# --- 9. apply_to_subchannel (per-subchannel systematic scoping) ---------------
# DetSys1 (spline) restricted to ND, RPA_CCQE (covariance) restricted to FD.
# process must not require (or read) their weight branches in non-matching
# files; outside the match splines are flat at 1 and covariance blocks exactly
# zero — asserted numerically by check_applyto.C on the t23 plot output.
# Also: a flat systematic restricted to FD and a norm_to_covariance restricted to
# the ND numu channel (both matrix-only types, scoped by the post-build stage
# PROsyst::ApplySubchannelScopes rather than at fill time).
sed -e 's|plotname="DetSys1" tag="det"|plotname="DetSys1" tag="det" apply_to_subchannel="_ND_"|' \
    -e 's|plotname="RPA_CCQE" tag="QE-MEC"|plotname="RPA_CCQE" tag="QE-MEC" apply_to_subchannel="_FD_"|' \
    local_test.xml \
  | awk '{print} /FiducialVol_FD/ && !d {print "    <allowlist type=\"flat\" plotname=\"FlatFD\" tag=\"other\" apply_to_subchannel=\"_FD_\">nu_:0.05</allowlist>"; print "    <allowlist type=\"norm_to_covariance\" plotname=\"NormCovNDnumu\" tag=\"other\" apply_to_subchannel=\"_ND_numu\">nu_:0.03</allowlist>"; d=1}' \
  > local_applyto.xml
SAVED_COMMON=("${COMMON[@]}")
COMMON=(-x local_applyto.xml -t "${TAG}apt" -n 1 -v 2 --seed 405 --preset fast)
run_test t22aptprocess process
# --with-covar: t26aptzero asserts on the Covariance dir in this ROOT file.
run_test t23aptplot   --use-fake-data plot --with-splines --with-covar
run_test t24aptglobal --use-fake-data global
# A wildcard matching no subchannel fullname must be refused loudly.
sed 's|apply_to_subchannel="_ND_"|apply_to_subchannel="_TYPO_"|' local_applyto.xml > local_applyto_typo.xml
COMMON=(-x local_applyto_typo.xml -t "${TAG}apttypo" -n 1 -v 2 --seed 405 --preset fast)
expect_fail t25apttypo    process
COMMON=("${SAVED_COMMON[@]}")
# Numeric assertion: non-matching covariance blocks exactly zero, non-matching
# splines exactly flat (needs root; skipped silently if unavailable).
ROOTEXE="${ROOTEXE:-$(command -v root || true)}"
[ -z "$ROOTEXE" ] && [ -x /usr/local/root/root/bin/root ] && ROOTEXE=/usr/local/root/root/bin/root
if [ -n "$ROOTEXE" ]; then
    # ND numu block = bins [64,166): nue 4x16 then numu 3x34 in the ND half.
    if "$ROOTEXE" -l -b -q "$REPO/tests/check_applyto.C(\"${TAG}apt_t23aptplot_PROplot.root\",\"RPA_CCQE;cov;FD,DetSys1;spline;ND,nu_:0.05;cov;FD,nu_:0.03;cov;64-166\")" > logs/t26aptzero.log 2>&1; then
        note "PASS  t26aptzero  (non-matching cov blocks zero, splines flat)"
        PASS=$((PASS+1))
    else
        note "FAIL  t26aptzero -- see logs/t26aptzero.log"
        FAIL=$((FAIL+1))
    fi
else
    note "SKIP  t26aptzero  (no root executable for the numeric assertion)"
fi

# --- 9b. apply_to_subchannel: every type, DetVar, and branch order --------------
# (a) EVERY systematic (splines, covariances, mcstat, norm, flat, norm_to_covariance)
#     restricted to _FD_: the TOTAL fractional covariance must have an exactly-zero
#     ND block. mcstat is not plotted on its own, so this is what covers it.
sed 's|\(<allowlist [^>]*\)>|\1 apply_to_subchannel="_FD_">|' local_test.xml \
  | awk '{print} /FiducialVol_FD/ && !d {print "    <allowlist type=\"flat\" plotname=\"FlatFD\" tag=\"other\" apply_to_subchannel=\"_FD_\">nu_:0.05</allowlist>"; print "    <allowlist type=\"norm_to_covariance\" plotname=\"NormCovFD\" tag=\"other\" apply_to_subchannel=\"_FD_\">nu_:0.03</allowlist>"; d=1}' \
  > local_applyto_fd.xml
COMMON=(-x local_applyto_fd.xml -t "${TAG}aptfd" -n 1 -v 2 --seed 405 --preset fast)
run_test t26bfdprocess process
run_test t26cfdplot    --use-fake-data plot --with-splines --with-covar
if [ -n "$ROOTEXE" ]; then
    if "$ROOTEXE" -l -b -q "$REPO/tests/check_applyto.C(\"${TAG}aptfd_t26cfdplot_PROplot.root\",\"total_frac_cov;total;FD,nu_:0.05;cov;FD,nu_:0.03;cov;FD,Flux1;spline;FD\")" > logs/t26dfdzero.log 2>&1; then
        note "PASS  t26dfdzero  (all-FD scope: total covariance ND block exactly zero)"
        PASS=$((PASS+1))
    else
        note "FAIL  t26dfdzero -- see logs/t26dfdzero.log"
        FAIL=$((FAIL+1))
    fi
fi
# (b) DetVar systematic: the same ND file as CV and as a variation at half the POT
#     (ratio 2 everywhere the section covers), restricted to the ND nue channel via
#     apply_to_subchannel: spline varying only in bins [0,64), flat everywhere else.
awk -v mc="$MCDIR" '
  /<variation_list>/ && !dv {
    print "<DetVarFiles>";
    print "  <DetVarSection treename=\"events/selected\" scale=\"1.0\">";
    print "    <cv filename=\"" mc "/fake_sbn_mc_ND.root\" pot=\"1e+21\"/>";
    print "    <variation name=\"DetVarTest\" filename=\"" mc "/fake_sbn_mc_ND.root\" pot=\"5e+20\"/>";
    print "    <subchannel>nu_ND_nue_intrinsic</subchannel>";
    print "    <subchannel>nu_ND_nue_background</subchannel>";
    print "    <subchannel>nu_ND_nue_fullosc</subchannel>";
    print "    <subchannel>nu_ND_numu_signal</subchannel>";
    print "    <subchannel>nu_ND_numu_background</subchannel>";
    print "  </DetVarSection>";
    print "</DetVarFiles>";
    dv=1 }
  {print}
  /FiducialVol_FD/ && !sy { print "    <allowlist type=\"spline\" name=\"DetVarTest\" plotname=\"DetVarTest\" tag=\"det\" apply_to_subchannel=\"_ND_nue_\"/>"; sy=1 }
' local_applyto.xml > local_applyto_detvar.xml
COMMON=(-x local_applyto_detvar.xml -t "${TAG}aptdv" -n 1 -v 2 --seed 405 --preset fast)
run_test t26edvprocess process
run_test t26fdvplot    --use-fake-data plot --with-splines --with-covar
if [ -n "$ROOTEXE" ]; then
    if "$ROOTEXE" -l -b -q "$REPO/tests/check_applyto.C(\"${TAG}aptdv_t26fdvplot_PROplot.root\",\"DetVarTest;spline;0-64,RPA_CCQE;cov;FD\")" > logs/t26gdvzero.log 2>&1; then
        note "PASS  t26gdvzero  (DetVar systematic scoped to the ND nue channel)"
        PASS=$((PASS+1))
    else
        note "FAIL  t26gdvzero -- see logs/t26gdvzero.log"
        FAIL=$((FAIL+1))
    fi
fi
# (c) Branch order: an incl_systematics="false" branch listed FIRST in a file whose
#     later branches carry systematics (zero-weight, so the physics is unchanged).
#     The friend-tree weight binding used to be decided by that first branch and
#     every later branch then died at fill time; the fit must match t24aptglobal.
awk '/<friend treename="events\/variationTree" \/>/ && !d { print;
       print "    <branch associated_subchannel=\"nu_ND_nue_cosmic\" incl_systematics=\"false\" model_rule=\"0\" additional_weight=\"0.0*mcweight\">";
       print "        <variable>reco_visible_energy</variable>";
       print "        <variable>true_baseline/(1000*true_neutrino_energy)</variable>";
       print "        <variable>true_neutrino_energy</variable>";
       print "        <variable>random_value</variable>";
       print "    </branch>"; d=1; next } {print}' local_applyto.xml > local_applyto_order.xml
COMMON=(-x local_applyto_order.xml -t "${TAG}aptord" -n 1 -v 2 --seed 405 --preset fast)
run_test t26hordprocess process
run_test t26iordglobal  --use-fake-data global
if cmp -s "${TAG}aptord_t26iordglobal_global_fit.txt" "${TAG}apt_t24aptglobal_global_fit.txt"; then
    note "PASS  t26jordsame  (cosmic-first branch order gives the identical global fit)"
    PASS=$((PASS+1))
else
    note "FAIL  t26jordsame  (global fit differs from t24aptglobal)"
    FAIL=$((FAIL+1))
fi
# (d) Escaped characters in a DetVar-inherited branch: tinyxml2 decodes &lt;/&amp; on
#     parse, and the DetVar child XML used to be written back unescaped, so a '<' in a
#     <variable> broke the child parse. Both cuts are no-ops (category is an integer,
#     random_value lies in [0,1)).
sed -e '0,/5\*mcweight\*(category == 0)/s//5*mcweight*(category \&gt; -1 \&amp;\&amp; category \&lt; 1)/' \
    -e '0,/<variable>reco_visible_energy<\/variable>/s//<variable>reco_visible_energy*(random_value \&lt; 2 \&amp;\&amp; random_value \&gt; -1)<\/variable>/' \
    local_applyto_detvar.xml > local_applyto_detvar_esc.xml
COMMON=(-x local_applyto_detvar_esc.xml -t "${TAG}aptesc" -n 1 -v 2 --seed 405 --preset fast)
run_test t26kescprocess process
# (e) One DetVar name in several <DetVarSection>s is ONE systematic: each section gives the
#     response in its own subchannels, and where sections overlap their responses average,
#     weighted by each section's POT-normalised CV. Section A (event-matched, all ND, same
#     file: ratio 1, CV at 5e20 POT = weight 2), B (FD, ratio 4) and C (ND numu only,
#     ratio 4, weight 1) give ND nue 1, ND numu (2*1+1*4)/3 = 2, FD 4. This used to build
#     three same-named parameters, each from the last section's variation file.
dvs_section() {  # <ND|FD> <cv pot> <var pot> <knobval> <event-matched 0|1> <subchannel>...
    local det=$1 cvpot=$2 varpot=$3 knob=$4 match=$5; shift 5
    printf '  <DetVarSection treename="events/selected" scale="1.0"%s>\n' "$([ "$match" = 1 ] && echo ' cv_variation_matching_vars="Run,Subrun,Evt"')"
    printf '    <cv filename="%s/fake_sbn_mc_%s.root" pot="%s"/>\n' "$MCDIR" "$det" "$cvpot"
    printf '    <variation name="DetVarShared" filename="%s/fake_sbn_mc_%s.root" pot="%s" knobval="%s"/>\n' "$MCDIR" "$det" "$varpot" "$knob"
    printf '    <subchannel>%s</subchannel>\n' "$@"
    printf '  </DetVarSection>\n'
}
dvs_xml() {  # <out.xml> <knobval of the FD section>
    local nd="nu_ND_nue_intrinsic nu_ND_nue_background nu_ND_nue_fullosc nu_ND_numu_signal nu_ND_numu_background"
    local blk
    blk="<DetVarFiles>
$(dvs_section ND 5e+20 2.5e+20 +1 1 $nd)
$(dvs_section FD 1e+21 2.5e+20 "$2" 0 ${nd//_ND_/_FD_})
$(dvs_section ND 1e+21 2.5e+20 +1 0 nu_ND_numu_signal nu_ND_numu_background)
</DetVarFiles>"
    awk -v blk="$blk" '/<variation_list>/ && !dv { print blk; dv=1 } {print}
      /FiducialVol_FD/ && !sy { print "    <allowlist type=\"spline\" name=\"DetVarShared\" plotname=\"DetVarShared\" tag=\"det\"/>"; sy=1 }' \
      local_applyto.xml > "$1"
}
dvs_xml local_detvar_shared.xml +1
dvs_xml local_detvar_shared_knobs.xml -1
COMMON=(-x local_detvar_shared.xml -t "${TAG}dvs" -n 1 -v 2 --seed 405 --preset fast)
run_test t26ldvsprocess process
run_test t26mdvsplot    --use-fake-data plot --with-splines
if [ -n "$ROOTEXE" ]; then
    if "$ROOTEXE" -l -b -q "$REPO/tests/check_spline_response.C(\"${TAG}dvs_t26mdvsplot_PROplot.root\",\"DetVarShared;0-48;1,DetVarShared;64-132;2,DetVarShared;166-214;4,DetVarShared;230-298;4\")" > logs/t26ndvsresp.log 2>&1; then
        note "PASS  t26ndvsresp  (shared DetVar name: per-section responses, CV-weighted overlap)"
        PASS=$((PASS+1))
    else
        note "FAIL  t26ndvsresp -- see logs/t26ndvsresp.log"
        FAIL=$((FAIL+1))
    fi
fi
# ...and a shared name must carry the same knob values in every section.
COMMON=(-x local_detvar_shared_knobs.xml -t "${TAG}dvsk" -n 1 -v 2 --seed 405 --preset fast)
expect_fail t26odvsknobs process
COMMON=("${SAVED_COMMON[@]}")

# --- 10. regex wildcards (patterns are unanchored ECMAScript regexes) ---------
# Plain substrings keep their old meaning (every test above covers that); here
# a genuine regex alternation must be accepted end-to-end. NAME:percent splits
# on the LAST colon, so regex constructs containing ':' survive too.
sed 's#>nu_ND_numu:0.02<#>nu_(ND|FD)_numu:0.02<#' local_test.xml > local_regex.xml
SAVED_COMMON=("${COMMON[@]}")
COMMON=(-x local_regex.xml -t "${TAG}rgx" -n 1 -v 2 --seed 405 --preset fast)
run_test t27regexprocess process
run_test t28regexglobal --use-fake-data global
# An invalid regex must be refused loudly (CompilePattern fatal)...
sed 's#>nu_ND:0.01<#>*bad:0.01<#' local_test.xml > local_regex_bad.xml
COMMON=(-x local_regex_bad.xml -t "${TAG}rgxbad" -n 1 -v 2 --seed 405 --preset fast)
expect_fail t29badregex   process
# ...and so must a valid regex that matches no subchannel (zero-match fatal).
sed 's#>nu_ND:0.01<#>^nomatch$:0.01<#' local_test.xml > local_regex_none.xml
COMMON=(-x local_regex_none.xml -t "${TAG}rgxnone" -n 1 -v 2 --seed 405 --preset fast)
expect_fail t30nomatch    process
COMMON=("${SAVED_COMMON[@]}")

# --- 11. LBL 3nu models (matter + vacuum) -------------------------------------
# Matter needs per-event L and E <parameter>s, so its XML replaces the L/E
# variable with a true-baseline one (hash changes -> own tag + process). The
# L/E and E binnings are cut down hard: the model grid is n_L x n_E GLOBAL bins
# x 10 components x every variable's reco bins, which OOMs at the base 200x20.
# Vacuum's single signed L/E swaps only the model block (not hashed) and reuses
# the t00 caches. Binaries predating the vacuum tag fail t31c by design.
sed -e 's|<bins unit="True L/E \[km/GeV\]" min="0" max="2.5" nbins="200" plot="false"/>|<bins unit="True Baseline [km]" min="0" max="1" nbins="4" plot="false"/>|' \
    -e 's|<bins unit="True Neutrino Energy \[GeV\]" min="0" max="3" nbins="20" />|<bins unit="True Neutrino Energy [GeV]" min="0" max="3" nbins="5" />|' \
    -e 's|<variable>true_baseline/(1000\*true_neutrino_energy)</variable>|<variable>true_baseline/1000</variable>|' \
    -e 's|<model tag="nueapp">|<model tag="LBL_3nu-matter_angles">|' \
    -e 's|<parameter name="L/E" variable_index="1"/>|<parameter name="L" variable_index="1"/><parameter name="E" variable_index="2"/>|' \
    local_test.xml > local_lbl_matter.xml
sed 's|tag="LBL_3nu-matter_angles"|tag="LBL_3nu-matter_angles" density="3" electron_fraction="0.5" n_newton="0"|' local_lbl_matter.xml > local_lbl_explicit.xml
sed 's|<model tag="nueapp">|<model tag="LBL_3nu-vacuum_angles">|' local_test.xml > local_lbl_vacuum.xml
sed -e 's|<model tag="nueapp">|<model tag="LBL_3nu-matter_angles" baseline="1300">|' \
    -e 's|<parameter name="L/E" variable_index="1"/>|<parameter name="E" variable_index="2"/>|' \
    local_test.xml > local_lbl_eonly.xml
sed 's|<model tag="nueapp">|<model tag="nueapp" density="3">|'    local_test.xml > local_lbl_sblopt.xml
COMMON=(-x local_lbl_matter.xml -t "${TAG}lbl" -n 1 -v 2 --seed 405 --preset fast)
run_test t31lblprocess process
run_test t31almatter   --use-fake-data global
COMMON=(-x local_lbl_explicit.xml -t "${TAG}lbl" -n 1 -v 2 --seed 405 --preset fast)
run_test t31blexplicit --use-fake-data global
# Explicit legacy-default attributes must be bitwise identical to no attributes.
if cmp -s "${TAG}lbl_t31almatter_global_fit.txt" "${TAG}lbl_t31blexplicit_global_fit.txt"; then
    note "PASS  t31dlbldefault  (explicit default attributes bitwise-identical)"
    PASS=$((PASS+1))
else
    note "FAIL  t31dlbldefault  (explicit LBL defaults changed the fit)"
    FAIL=$((FAIL+1))
fi
COMMON=(-x local_lbl_vacuum.xml -t "$TAG" -n 1 -v 2 --seed 405 --preset fast)
run_test t31clvacuum   --use-fake-data global
# E-only + baseline= (fixed-baseline mode, 1D grid; also reuses the t00 caches).
COMMON=(-x local_lbl_eonly.xml -t "$TAG" -n 1 -v 2 --seed 405 --preset fast)
run_test t31eleonly    --use-fake-data global
# Model options on a non-LBL tag must be refused loudly.
COMMON=(-x local_lbl_sblopt.xml -t "$TAG" -n 1 -v 2 --seed 405 --preset fast)
expect_fail t32lbadopt  --use-fake-data global
COMMON=("${SAVED_COMMON[@]}")

# --- 12. template model: shared subchannels= parameter + default= -------------
# The <model> block is not hashed, so every variant reuses the t00 caches. One
# subchannels= regex drives several subchannels with a single scale (the use
# case: one signal strength over nu+nubar); here "mu" floats the ND and FD nue
# fullosc templates together, against the legacy form (one exact-name parameter
# per subchannel). With EVERY parameter pinned (--fix at --inject-cv /
# --inject-systs-cv) a global fit is one chi2 evaluation, and the two forms at
# the same scale are the same prediction, so their chi2 must agree; unpinned,
# the shared fit must recover the injected mu. Binaries predating subchannels=
# fail t33*.
tmpl_xml() {  # <out.xml> <model tag> <parameter element>...
    local out=$1 tag=$2; shift 2
    awk -v tag="$tag" -v p="$(printf '    %s\n' "$@")" '
        /<model tag="nueapp">/ { print "<model tag=\"" tag "\">"; print p; skip=1; next }
        skip { if (/<\/model>/) { print; skip=0 } next }
        { print }' local_test.xml > "$out"
}
TSEP=('<parameter name="nu_ND_nue_fullosc" min="0" max="1"/>' '<parameter name="nu_FD_nue_fullosc" min="0" max="1"/>')
TSH='<parameter name="mu" subchannels="^nu_(ND|FD)_nue_fullosc$" min="0" max="1" default="0"/>'
tmpl_xml local_tmpl_sep.xml      template "${TSEP[@]}"
tmpl_xml local_tmpl_shared.xml   template "$TSH"
tmpl_xml local_tmpl_grad.xml     template '<parameter name="mu" subchannels="^nu_(ND|FD)_nue_fullosc$" min="0.045" max="0.055" default="0.05"/>'
tmpl_xml local_tmpl_none.xml     template '<parameter name="mu" subchannels="^nu_(ND|FD)_nue_nomatch$" min="0" max="1"/>'
tmpl_xml local_tmpl_twice.xml    template "$TSH" "${TSEP[0]}"
tmpl_xml local_tmpl_baddef.xml   template '<parameter name="mu" subchannels="fullosc" min="0" max="1" default="2"/>'
tmpl_xml local_tmpl_nontmpl.xml  nueapp   '<parameter name="L/E" variable_index="1" subchannels="fullosc"/>'

# global_chi2 <test name>: the INFO-level best-fit chi2 from the test's file log.
global_chi2() { sed -n 's/.*Global Best Fit chi^2: *\([-+0-9.eE]*\).*/\1/p' "logs/$1.full.log" | tail -n 1; }
tmpl_equal() {  # <check name> <test A> <test B>
    local a b; a=$(global_chi2 "$2"); b=$(global_chi2 "$3")
    if [ -n "$a" ] && [ -n "$b" ] && awk -v a="$a" -v b="$b" 'BEGIN { d=a-b; if (d<0) d=-d; m=(a<0?-a:a); exit !(d <= 1e-5*(m>1?m:1)) }'; then
        note "PASS  $1  (shared vs separate, all pinned, chi2: $a vs $b)"
        PASS=$((PASS+1))
    else
        note "FAIL  $1  (shared vs separate pinned chi2 differ: '$a' vs '$b')"
        FAIL=$((FAIL+1))
    fi
}
SEPNAMES=(nu_ND_nue_fullosc nu_FD_nue_fullosc)
SPLINES=(CrossSection1 CrossSection2 CrossSection3 CrossSection4 DetSys1 DetSys2 DetSys3
         Flux1 Flux2 Flux3 FiducialVol_FD FluxNorm_FD FiducialVol_ND FluxNorm_ND)
SPLINE_CV=(--inject-systs-cv CrossSection1 0.7 CrossSection2 -1.1 DetSys1 0.4 Flux2 -0.3 FluxNorm_ND 0.9)
COMMON=(-x local_tmpl_sep.xml -t "$TAG" -n 1 -v 2 --seed 405 --preset fast)
run_test t33atmplsep02    -l logs/t33atmplsep02.full.log -w 3 --use-fake-data --fix "${SEPNAMES[@]}" "${SPLINES[@]}" "${SPLINE_CV[@]}" \
    -i nu_ND_nue_fullosc 0.05 nu_FD_nue_fullosc 0.05 --inject-cv nu_ND_nue_fullosc 0.02 nu_FD_nue_fullosc 0.02 global
run_test t33ctmplsep08    -l logs/t33ctmplsep08.full.log -w 3 --use-fake-data --fix "${SEPNAMES[@]}" "${SPLINES[@]}" "${SPLINE_CV[@]}" \
    -i nu_ND_nue_fullosc 0.05 nu_FD_nue_fullosc 0.05 --inject-cv nu_ND_nue_fullosc 0.8 nu_FD_nue_fullosc 0.8 global
COMMON=(-x local_tmpl_shared.xml -t "$TAG" -n 1 -v 2 --seed 405 --preset fast)
run_test t33btmplshared02 -l logs/t33btmplshared02.full.log -w 3 --use-fake-data --fix mu "${SPLINES[@]}" "${SPLINE_CV[@]}" \
    -i mu 0.05 --inject-cv mu 0.02 global
run_test t33dtmplshared08 -l logs/t33dtmplshared08.full.log -w 3 --use-fake-data --fix mu "${SPLINES[@]}" "${SPLINE_CV[@]}" \
    -i mu 0.05 --inject-cv mu 0.8 global
tmpl_equal t33etmplequal02 t33atmplsep02 t33btmplshared02
tmpl_equal t33ftmplequal08 t33ctmplsep08 t33dtmplshared08
run_test t33gtmplrecover  --use-fake-data -i mu 0.05 global
mu_fit=$(sed -n 's/^mu : *//p' "${TAG}_t33gtmplrecover_global_fit.txt" 2>/dev/null)
if [ -n "$mu_fit" ] && awk -v m="$mu_fit" 'BEGIN { d=m-0.05; if (d<0) d=-d; exit !(d < 1e-3) }'; then
    note "PASS  t33htmplmu  (shared mu recovered: $mu_fit, injected 0.05)"
    PASS=$((PASS+1))
else
    note "FAIL  t33htmplmu  (shared mu not recovered: '$mu_fit', injected 0.05)"
    FAIL=$((FAIL+1))
fi
# gradcheck draws mu uniformly in [min,max] around an Asimov at default=: keep the box
# tight, since the fullosc template is so large that a wide box puts every point at a
# chi2 where the float central-FD reference itself is noise.
COMMON=(-x local_tmpl_grad.xml -t "$TAG" -n 1 -v 2 --seed 405 --preset fast)
run_test t33itmplgrad     -l logs/t33itmplgrad.full.log -w 3 --use-fake-data scale-test --tests gradcheck -N 200
grad_rel=$(sed -n 's/.*\[GRADCHECK\] mode=analytic .*mean_rel=\([-+0-9.eE]*\).*/\1/p' logs/t33itmplgrad.full.log | tail -n 1)
if [ -n "$grad_rel" ] && awk -v r="$grad_rel" 'BEGIN { exit !(r < 0.2) }'; then
    note "PASS  t33itmplgradrel  (analytic vs central-full mean_rel $grad_rel)"
    PASS=$((PASS+1))
else
    note "FAIL  t33itmplgradrel  (analytic vs central-full mean_rel '$grad_rel', want < 0.2)"
    FAIL=$((FAIL+1))
fi
# Refused: zero-match pattern, a subchannel claimed twice, default= outside [min,max],
# and subchannels= on a non-template model.
COMMON=(-x local_tmpl_none.xml    -t "$TAG" -n 1 -v 2 --seed 405 --preset fast)
expect_fail t33jtmplnomatch  --use-fake-data global
COMMON=(-x local_tmpl_twice.xml   -t "$TAG" -n 1 -v 2 --seed 405 --preset fast)
expect_fail t33ktmpltwice    --use-fake-data global
COMMON=(-x local_tmpl_baddef.xml  -t "$TAG" -n 1 -v 2 --seed 405 --preset fast)
expect_fail t33ltmplbaddef   --use-fake-data global
COMMON=(-x local_tmpl_nontmpl.xml -t "$TAG" -n 1 -v 2 --seed 405 --preset fast)
expect_fail t33mtmplnontmpl  --use-fake-data global
COMMON=("${SAVED_COMMON[@]}")

# --- 13. 3+1 FC: three physics parameters, two of them axes -------------------
# The appearance parameterisation (dmsq, sinsq2thmue, xi) with the fake sample
# routed through the 3+1 rules: fullosc -> mue (2), numu signal -> mumu (1),
# nue intrinsic -> ee (3). model_rule is hashed, so this config has its own
# process step. Keep xhi <= 0.3: the unitarity constraint sqrt(s)*cosh(xi) < 1
# walls off larger sinsq2thmue. AFC needs an explicit treatment of the
# non-axis xi: without one it is refused, with --fix xi it runs the xi slice.
sed 's|<model tag="nueapp">|<model tag="SBL_3+1_sinsq2thmue">|' local_test.xml | awk '
  /associated_subchannel/ { r = /_nue_fullosc"/ ? 2 : /_numu_signal"/ ? 1 : /_nue_intrinsic"/ ? 3 : -1 }
  /model_rule/ && r >= 0 { sub(/"[0-9]+"/, "\"" r "\"") } { print }' > local_3p1.xml
AX3=(--xvar sinsq2thmue --yvar dmsq --xlo 1e-4 --xhi 0.3 --ylo 1e-2 --yhi 1e2)
AFC3=(fc-adaptive --throws 2 --prepass-amr-initial 3 3 --prepass-amr-levels 1
      --p-thresh 0.33 --baseline-level 1 "${AX3[@]}")
COMMON=(-x local_3p1.xml -t "${TAG}3p1" -n 1 -v 2 --seed 405 --preset fast)
run_test t50p3process process
run_test t51p3fc      --use-fake-data -i dmsq 1 sinsq2thmue 0.01 fc -u 2
expect_fail t52p3afcpin --use-fake-data "${AFC3[@]}" --mode build-mesh
run_test_tag t53p3afcmesh   t53p3afc --use-fake-data --fix xi "${AFC3[@]}" --mode build-mesh
run_test_tag t53p3afcbank   t53p3afc --use-fake-data --fix xi "${AFC3[@]}" --mode init-bank --n-pe-min 1 --n-pe-max 1
run_test_tag t53p3afcasimov t53p3afc --use-fake-data --fix xi "${AFC3[@]}" --mode asimov
run_test_tag t53p3afcbrazil t53p3afc --use-fake-data --fix xi "${AFC3[@]}" --mode brazil --n-brazil-throws 2
# Profile FC (xi profiled at the tested point): toy truth from the data's
# conditional best fit (+ the observed statistic for --pval), and from -i.
run_test t54p3fcprof    --use-fake-data -i dmsq 1 sinsq2thmue 0.01 fc -u 2 --profile-params xi --profile-truth data --pval
run_test t55p3fcprofinj --use-fake-data -i dmsq 1 sinsq2thmue 0.01 xi 0.3 fc -u 2 --profile-params xi --profile-truth inject
# A --fix'd parameter cannot be profiled: --profile-params must then fall back to
# the plain FC, bitwise.
run_test t56afcfixxi     --use-fake-data --fix xi -i dmsq 1 sinsq2thmue 0.01 fc -u 2
run_test t56bfcfixxiprof --use-fake-data --fix xi -i dmsq 1 sinsq2thmue 0.01 fc -u 2 --profile-params xi
if cmp -s "${TAG}3p1_t56afcfixxi_FC.csv" "${TAG}3p1_t56bfcfixxiprof_FC.csv"; then
    note "PASS  t56cfcfixsame  (--profile-params of a --fix'd parameter gives the plain FC)"
    PASS=$((PASS+1))
else
    note "FAIL  t56cfcfixsame  (--fix xi FC changed when xi is also listed in --profile-params)"
    FAIL=$((FAIL+1))
fi
# Adaptive profile FC: xi profiled, each cell's toys at the conditional best fit
# to the fake data. The bank is tied to that dataset, so asimov on other data
# (a different -i) must be refused.
AFC3P=("${AFC3[@]}" --profile-params xi)
run_test_tag t57p3afcpmesh   t57p3afcp --use-fake-data "${AFC3P[@]}" --mode build-mesh
run_test_tag t57p3afcpbank   t57p3afcp --use-fake-data "${AFC3P[@]}" --mode init-bank --n-pe-min 1 --n-pe-max 1
run_test_tag t57p3afcpasimov t57p3afcp --use-fake-data "${AFC3P[@]}" --mode asimov
run_test_tag t57p3afcpbrazil t57p3afcp --use-fake-data "${AFC3P[@]}" --mode brazil --n-brazil-throws 2
expect_fail_tag t58p3afcpfp t57p3afcp --use-fake-data -i dmsq 1 sinsq2thmue 0.01 "${AFC3P[@]}" --mode asimov
COMMON=("${SAVED_COMMON[@]}")

# --- 14. Statistical overlap between analyses (overlap_matching_vars) ----------
# nueB is the (oscillating) nue channel again at half weight, from the same MC
# entries, so it carries no information nue lacks: with the overlap declared, the
# stat-only Asimov surface must equal the plain config's exactly. The key adds
# category so an offbeam entry never matches a beam entry with the same Run/Subrun/Evt.
awk '
    /<channel name="nue"/ { inb=1; blk="" }
    inb { blk = blk $0 "\n"; if ($0 ~ /<\/channel>/) { printf "%s", blk; b=blk; gsub(/name="nue"/, "name=\"nueB\"", b); printf "%s", b; inb=0 }; next }
    /<MCFile / { sub(/>$/, " overlap_matching_vars=\"Run,Subrun,Evt,category\">") }
    /<branch/ { inbr=1; br="" }
    inbr { br = br $0 "\n"; if ($0 ~ /<\/branch>/) { printf "%s", br; if (br ~ /_nue_/) { c=br; gsub(/_nue_/, "_nueB_", c); sub(/additional_weight *= *"/, "&0.5*(", c); sub(/0\.5\*\([^"]*/, "&)", c); printf "%s", c }; inbr=0 }; next }
    { print }' local_test.xml > local_overlap.xml
sed 's| overlap_matching_vars="Run,Subrun,Evt,category"||' local_overlap.xml > local_overlap_none.xml
SAVED_COMMON=("${COMMON[@]}")
COMMON=(-x local_overlap.xml -t "${TAG}ov" -n 1 -v 2 --seed 405 --preset fast)
run_test t60ovprocess process
run_test t61ovglobal  --use-fake-data global
run_test t62ovpe      --use-fake-data --pseudo-experiment global
run_test t63ovfc      --use-fake-data -i dmsq 1 sinsq2thme 0.01 fc -u 2
expect_fail t64ovcnp  --use-fake-data -c CNP global
run_test t65ovsurf    --use-fake-data --statonly surface -g 4 "${AXES[@]}"
AFCOV=(fc-adaptive --throws 2 --prepass-amr-initial 3 3 --prepass-amr-levels 1
       --p-thresh 0.33 --baseline-level 1 "${AXES[@]}")
run_test_tag t66ovafcmesh   t66ovafc --use-fake-data "${AFCOV[@]}" --mode build-mesh
run_test_tag t66ovafcbank   t66ovafc --use-fake-data "${AFCOV[@]}" --mode init-bank --n-pe-min 1 --n-pe-max 1
run_test_tag t66ovafcbrazil t66ovafc --use-fake-data "${AFCOV[@]}" --mode brazil --n-brazil-throws 2
COMMON=(-x local_overlap_none.xml -t "${TAG}ovn" -n 1 -v 2 --seed 405 --preset fast)
run_test t67ovnsurf   --use-fake-data --statonly surface -g 4 "${AXES[@]}"
COMMON=("${SAVED_COMMON[@]}")
run_test t68asurf     --use-fake-data --statonly surface -g 4 "${AXES[@]}"
# A+B with the overlap matches A alone (float noise); without it, it does not.
surf_match() {
    awk 'FNR==1{f++} FNR>10{k=$1"_"$2; v[f,k]=$3; keys[k]=1}
         END{bad=0; for(k in keys){d=v[1,k]-v[2,k]; if(d<0)d=-d; if(d > 1e-4 + 1e-3*v[1,k]) bad++} exit bad>0}' "$1" "$2"
}
if surf_match "${TAG}_t68asurf_statonly_surface.txt" "${TAG}ov_t65ovsurf_statonly_surface.txt" \
   && ! surf_match "${TAG}_t68asurf_statonly_surface.txt" "${TAG}ovn_t67ovnsurf_statonly_surface.txt"; then
    note "PASS  t69ovclosure  (overlap A+B surface equals A alone; ignoring the overlap does not)"
    PASS=$((PASS+1))
else
    note "FAIL  t69ovclosure  (A+B with overlap differs from A alone, or ignoring it does not)"
    FAIL=$((FAIL+1))
fi

note "----------------------------------------------------------------------"
note "RESULT: $PASS passed, $FAIL failed  (outputs in $RUNDIR)"
exit "$FAIL"
