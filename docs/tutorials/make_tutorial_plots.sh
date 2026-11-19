#!/usr/bin/env bash
#
# make_tutorial_plots.sh — regenerate every plot referenced in
# docs/tutorials/PROfit_Tutorial_v2.md.
#
# Runs each command exactly as shown in the tutorial (tag TUT, seed 405)
# against the fake ND+FD SBN config working_dir/Neutrino2026/fake_sbn_v2.xml.
# Outputs (PDFs/ROOT/txt) land in one run directory; every "PLOT PLACEHOLDER"
# in the tutorial names a file this script produces.
#
# Usage:
#   docs/tutorials/make_tutorial_plots.sh
#   RUN_EXPENSIVE=1 docs/tutorials/make_tutorial_plots.sh   # + Brazil bands etc.
#
# Environment overrides:
#   PROFIT_BIN         PROfit executable      (default: <repo>/build/bin/PROfit)
#   PROFIT_XML         Reference XML config   (default: <repo>/docs/tutorials/fake_sbn_v2.xml)
#   PROFIT_TEST_MCDIR  Directory holding fake_sbn_mc_{ND,FD}.root
#                      (default: directory containing the XML; the XML's
#                      hardcoded /exp/... path is rewritten to it)
#   TUTORIAL_OUTDIR    Run directory (default: docs/tutorials/tutorial_run)
#   NTHREADS           Threads for fits (default: 8)
#   SEED               RNG seed (default: 405, as in the tutorial)
#   PRESET             Optional --preset (e.g. fast) applied to every fit;
#                      unset = PROfit default (good). Use fast for a quick
#                      smoke run; leave unset for tutorial-quality plots.
#   RUN_EXPENSIVE      1 = also run the heavy extras (Brazil-band surface,
#                      no-flux comparison surface). Default 0.
#
# Exit code: number of failed steps (0 = all passed).

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

XML_IN="${PROFIT_XML:-$REPO/docs/tutorials/fake_sbn_v2.xml}"
BIN="${PROFIT_BIN:-$REPO/build/bin/PROfit}"
# realpath: absolutize before the cd into RUNDIR below, so relative
# PROFIT_TEST_MCDIR values resolve from the invocation directory.
MCDIR="$(realpath "${PROFIT_TEST_MCDIR:-$(dirname "$XML_IN")}")"
RUNDIR="${TUTORIAL_OUTDIR:-$REPO/docs/tutorials/tutorial_run}"
NTHREADS="${NTHREADS:-8}"
SEED="${SEED:-405}"
RUN_EXPENSIVE="${RUN_EXPENSIVE:-0}"
TAG="TUT"

[ -x "$BIN" ]    || { echo "ERROR: PROfit binary not found/executable: $BIN (set PROFIT_BIN)"; exit 99; }
[ -f "$XML_IN" ] || { echo "ERROR: XML not found: $XML_IN"; exit 99; }

mkdir -p "$RUNDIR/logs"
cd "$RUNDIR"

# Localize the XML: point every MCFile entry at $MCDIR, regardless of what
# path (bare filename, /exp/..., etc.) the source XML uses.
sed -E "s|filename=\"[^\"]*(fake_sbn_mc_[NF]D\.root)\"|filename=\"$MCDIR/\1\"|g" "$XML_IN" > tutorial.xml

for f in "$MCDIR/fake_sbn_mc_ND.root" "$MCDIR/fake_sbn_mc_FD.root"; do
    [ -f "$f" ] || { echo "ERROR: fake MC file missing: $f (set PROFIT_TEST_MCDIR)"; exit 99; }
done

COMMON=(-x tutorial.xml -t "$TAG" --seed "$SEED" -n "$NTHREADS")
[ -n "${PRESET:-}" ] && COMMON+=(--preset "$PRESET")

# Physics axes for the nueapp model of this config (surface/fc-adaptive).
AXES=(--xvar sinsq2thme --yvar dmsq --xlo 1e-4 --xhi 1 --ylo 1e-2 --yhi 1e2)

INJECT=(-i dmsq 1 sinsq2thme 0.01)

PASS=0; FAIL=0
SUMMARY="$RUNDIR/summary.txt"
: > "$SUMMARY"
note() { echo "$*" | tee -a "$SUMMARY"; }

# run_step <name> <args...> — one tutorial command; -o <name> is added.
run_step() {
    local name="$1"; shift
    local t0=$SECONDS
    note "RUN   $name: PROfit ${COMMON[*]} -o $name $*"
    if "$BIN" "${COMMON[@]}" -o "$name" "$@" > "logs/$name.log" 2>&1; then
        note "PASS  $name  ($((SECONDS-t0))s)"
        PASS=$((PASS+1))
    else
        note "FAIL  $name  ($((SECONDS-t0))s, exit $?) — see logs/$name.log"
        FAIL=$((FAIL+1))
    fi
}

note "PROfit tutorial plot generation"
note "  bin: $BIN"
note "  xml: $XML_IN  (localized to $RUNDIR/tutorial.xml)"
note "  out: $RUNDIR"
note "  threads: $NTHREADS  seed: $SEED  preset: ${PRESET:-<default>}  expensive: $RUN_EXPENSIVE"
note "  git: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
note "----------------------------------------------------------------------"

# --- Section 3: process (creates TUT_prop.bin / TUT_syst.bin) -----------------
run_step process process
[ -f "${TAG}_prop.bin" ] || { note "FATAL: process produced no ${TAG}_prop.bin; aborting."; exit 98; }

# --- Section 4: plot -----------------------------------------------------------
run_step plotcv   plot
run_step plotinj  "${INJECT[@]}" plot
run_step plotsyst "${INJECT[@]}" --inject-systs Flux1 1.0 DetSys2 -2.0 plot
run_step plotspl  plot --with-splines
run_step plotbsub plot --bkg-subtract background

# --- Section 5: global fits -----------------------------------------------------
run_step glob1   "${INJECT[@]}" global
run_step globcnp -c CNP global
run_step globpoi -c poisson global
# Pull distributions need a fluctuated dataset (Asimov pulls are all 0).
run_step globpe  --pseudo-experiment global

# --- Section 6: profile ---------------------------------------------------------
run_step prof1  "${INJECT[@]}" profile
run_step probe1 "${INJECT[@]}" profile --probe
run_step profso --syst-only profile --probe

# --- Section 7: surfaces --------------------------------------------------------
run_step surf1    surface -g 30 "${AXES[@]}"
run_step surfstat --statonly surface -g 30 "${AXES[@]}"
run_step surfamr  surface "${AXES[@]}" --surface-amr --amr-initial 10 --amr-levels 3 \
                  --amr-levels-chi2 2.30 5.99
run_step curve1   surface "${AXES[@]}" -g 20 --curve-mode -1 -3 1 -1
if [ "$RUN_EXPENSIVE" = "1" ]; then
    run_step surfnoflux --exclude-systs Flux1 Flux2 Flux3 surface -g 30 "${AXES[@]}"
    run_step surfbrz    surface "${AXES[@]}" --surface-amr --amr-initial 10 --amr-levels 2 \
                        --brazil-band
else
    note "SKIP  surfnoflux, surfbrz (set RUN_EXPENSIVE=1)"
fi

# --- Section 8: FC and adaptive FC ----------------------------------------------
run_step fc1 "${INJECT[@]}" fc -u 500

# All fc-adaptive stages of one study must share the same -o (artifacts chain).
AFC=(fc-adaptive --throws 25 --prepass-amr-initial 8 8 --prepass-amr-levels 2 "${AXES[@]}")
run_step afc         "${AFC[@]}" --mode build-mesh
mv -f "logs/afc.log" "logs/afc_mesh.log" 2>/dev/null
run_step afc         "${AFC[@]}" --mode init-bank --n-pe-min 25 --n-pe-max 400
mv -f "logs/afc.log" "logs/afc_bank.log" 2>/dev/null
run_step afc         "${AFC[@]}" --mode print-bank
mv -f "logs/afc.log" "logs/afc_print.log" 2>/dev/null
run_step afc         "${AFC[@]}" --mode asimov
mv -f "logs/afc.log" "logs/afc_asimov.log" 2>/dev/null
run_step afc         "${AFC[@]}" --mode brazil --n-brazil-throws 50
mv -f "logs/afc.log" "logs/afc_brazil.log" 2>/dev/null

# --- Section 9: PROjector -------------------------------------------------------
run_step pj --projector-prefit "_ND_" global
CONSTRAINT="${TAG}_pj_PROjector_constraint.bin"
if [ -f "$CONSTRAINT" ]; then
    run_step pjglob --projector "$CONSTRAINT" global
    run_step pjprof --projector "$CONSTRAINT" profile --probe
    run_step pjsurf --projector "$CONSTRAINT" surface "${AXES[@]}" \
                    --surface-amr --amr-initial 10 --amr-levels 3
else
    note "SKIP  pjglob/pjprof/pjsurf (no constraint file from pj stage)"
    FAIL=$((FAIL+3))
fi

# --- Section 11: fit regions ----------------------------------------------------
run_step fr      "${INJECT[@]}" --fit-region "_numu_" global
run_step frblind "${INJECT[@]}" --fit-region "_numu_" --blind-eval global
run_step frplot  "${INJECT[@]}" --fit-region "_numu_" plot

note "----------------------------------------------------------------------"
note "Done: $PASS passed, $FAIL failed. Outputs in $RUNDIR"
note "Tutorial placeholders map to: ${TAG}_<step>_*.pdf"
exit "$FAIL"
