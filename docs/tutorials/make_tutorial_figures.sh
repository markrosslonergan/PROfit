#!/usr/bin/env bash
#
# make_tutorial_figures.sh — render the tutorial's referenced PDF pages to
# PNGs in docs/tutorials/figures/, for inline embedding in
# PROfit_Tutorial_v2.md (GitHub markdown cannot embed PDFs).
#
# Reads the outputs of make_tutorial_plots.sh from TUTORIAL_OUTDIR and writes
# one PNG per tutorial figure. Missing PDFs are skipped with a note, so this
# can be re-run after generating the remaining plots (fc-adaptive, PROjector,
# Brazil band) to fill in the missing figures.
#
# Multi-page handling:
#   * 4-page per-(detector x channel) plots  -> 2x2 montage
#   * 2-page PROfile canvases                -> vertical stack
#   * posteriors (one parameter per page)    -> 2x2 montage of a
#     representative set (CrossSection1, DetSys1, Flux1, FiducialVol_FD)
#   * everything else                        -> the single relevant page
#
# Requirements: pdftoppm (poppler-utils), montage (ImageMagick).
#
# Environment overrides:
#   TUTORIAL_OUTDIR  Where the TUT_*.pdf live (default: docs/tutorials/tutorial_run)
#   FIGDIR           Output directory (default: docs/tutorials/figures)
#   DPI              Render resolution for single pages (default 130;
#                    montage panels render at 2/3 of this)
#
# Exit code: number of missing/failed figures (0 = complete set).

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNDIR="${TUTORIAL_OUTDIR:-$REPO/docs/tutorials/tutorial_run}"
FIGDIR="${FIGDIR:-$REPO/docs/tutorials/figures}"
DPI="${DPI:-130}"
MDPI=$((DPI * 2 / 3))

command -v pdftoppm >/dev/null || { echo "ERROR: pdftoppm not found (install poppler-utils)"; exit 99; }
command -v montage  >/dev/null || { echo "ERROR: montage not found (install imagemagick)"; exit 99; }
[ -d "$RUNDIR" ] || { echo "ERROR: run directory not found: $RUNDIR (run make_tutorial_plots.sh first)"; exit 99; }

mkdir -p "$FIGDIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

MISS=0; DONE=0

# render <dpi> <pdf-basename> <page> <out-prefix> — one page to
# $TMP/<out-prefix>.png, whitespace-trimmed (ROOT pages carry big margins).
render() {
    pdftoppm -png -r "$1" -f "$3" -l "$3" "$RUNDIR/$2.pdf" "$TMP/$4" && \
        mv "$TMP/$4"-*.png "$TMP/$4.png" && \
        mogrify -trim +repage -bordercolor white -border 12 "$TMP/$4.png"
}

# page <name> <page>          — single page -> figures/<name>.png
page() {
    local name="$1" p="${2:-1}"
    if [ ! -f "$RUNDIR/$name.pdf" ]; then echo "SKIP  $name.png ($name.pdf missing)"; MISS=$((MISS+1)); return; fi
    if render "$DPI" "$name" "$p" one && mv "$TMP/one.png" "$FIGDIR/$name.png"; then
        echo "OK    $name.png (page $p)"; DONE=$((DONE+1))
    else echo "FAIL  $name.png"; MISS=$((MISS+1)); fi
}

# grid <name> <tile> <pages...> — montage of pages -> figures/<name>.png
grid() {
    local name="$1" tile="$2"; shift 2
    if [ ! -f "$RUNDIR/$name.pdf" ]; then echo "SKIP  $name.png ($name.pdf missing)"; MISS=$((MISS+1)); return; fi
    local parts=() i=0 p
    for p in "$@"; do
        render "$MDPI" "$name" "$p" "part$i" || { echo "FAIL  $name.png (page $p)"; MISS=$((MISS+1)); return; }
        parts+=("$TMP/part$i.png"); i=$((i+1))
    done
    if montage "${parts[@]}" -tile "$tile" -geometry +8+8 -background white "$FIGDIR/$name.png"; then
        echo "OK    $name.png (pages $*, tile $tile)"; DONE=$((DONE+1))
    else echo "FAIL  $name.png"; MISS=$((MISS+1)); fi
    rm -f "$TMP"/part*.png
}

# --- Section 4: plot ----------------------------------------------------------
grid TUT_plotcv_PROplot_Variable_0_CV        2x2  1 2 3 4
grid TUT_plotcv_PROplot_Variable_0_ErrorBand 2x2  1 2 3 4
page TUT_plotcv_PROplot_Covar                1     # collapsed correlation matrix
page TUT_plotcv_fractional_systematics       1     # fitting variable
grid TUT_plotinj_PROplot_Osc                 2x2  1 2 3 4
grid TUT_plotinj_PROplot_Variable_0_ErrorBand  2x2  1 2 3 4
grid TUT_plotsyst_PROplot_Variable_0_ErrorBand 2x2  1 2 3 4
page TUT_plotspl_PROplot_Spline              1     # example page
grid TUT_plotbsub_PROplot_Variable_0_ErrorBand 2x2  1 2 3 4

# --- Section 5: global --------------------------------------------------------
grid TUT_glob1_PROglobal_hists               2x2  1 2 3 4
page TUT_glob1_PROglobal_postfit_correlation_matrix
grid TUT_glob1_PROglobal_postfit_posteriors  2x2  1 5 8 11
grid TUT_globpe_PROglobal_postfit_pull_distributions 1x2  1 2   # splines + covariance modes

# --- Section 6: profile -------------------------------------------------------
grid TUT_prof1_PROfile                       1x2  1 2
page TUT_prof1_PROfile_1sigma
grid TUT_prof1_PROfile_hists                 2x2  1 2 3 4
grid TUT_probe1_PROfile                      1x2  1 2

# --- Section 7: surfaces ------------------------------------------------------
page TUT_surf1_surface
page TUT_curve1_PROcurve
page TUT_surfamr_surface
page TUT_surfamr_amr_mesh
page TUT_surfbrz_surface

# --- Section 8: FC / adaptive FC ----------------------------------------------
page TUT_afc_metamesh
page TUT_afc_throws
page TUT_afc_bank_summary
page TUT_afc_asimov_contour
page TUT_afc_asimov_verdict
page TUT_afc_brazil_band

# --- Section 9: PROjector -----------------------------------------------------
page TUT_pjprof_PROfile_1sigma
page TUT_pjsurf_surface

# --- Section 11: fit regions --------------------------------------------------
page TUT_fr_PROglobal_FitRegion 2

echo "----------------------------------------------------------------------"
echo "Done: $DONE figures written to $FIGDIR, $MISS missing/failed."
exit "$MISS"
