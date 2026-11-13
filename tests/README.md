# PROfit short test suite

## Sparse response product regression

This header-only test requires Eigen 3.4 and a C++17 compiler, without ROOT or
MC inputs. From the repository root (adjust the Eigen include path if needed):

```bash
c++ -std=c++17 -O3 -DNDEBUG -pthread -Iinc -Ibuild/_deps/eigen-src \
    tests/test_CSRDoubleProduct.cxx -o /tmp/test_CSRDoubleProduct
/tmp/test_CSRDoubleProduct
```

It checks mapped probability/derivative inputs, cancellation-sensitive double
accumulation, agreement with a dense64 reference, unchanged float coefficients,
concurrent calls, and empty response rows. The API still returns float vectors;
bitwise agreement with the former float-accumulated dense products is not
expected. Use the workflow suite below to assess changes to fitted results.

Two scripts for fast, deterministic, end-to-end regression testing of PROfit —
not CI, just something you run by hand before/after a change.

## Quick start

```bash
# Run the suite once per code state you want to compare:
tests/run_short_tests.sh ref          # e.g. on the base branch build
tests/run_short_tests.sh mybranch     # after rebuilding with your changes

# Compare the two:
tests/compare_tags.sh ref mybranch
# -> tests/runs/compare_ref_vs_mybranch.md
```

## What it runs

`run_short_tests.sh <TAG> [XML]` processes the fake ND+FD SBN MC
(`working_dir/Neutrino2026/fake_sbn_v2.xml` by default; the hardcoded
`/exp/...` MC paths are rewritten to sit next to the XML) and then runs one
short, seeded (`--seed 405 -n 1 --preset fast fast`) instance of each workflow:

| Tests | Coverage |
|---|---|
| t00 | `process` (MC → `_prop.bin`/`_syst.bin`) |
| t01–t06 | `global` with neyman (PROchi) / CNP (t02, legacy spelling `-c PROCNP` — covers the alias path) / poisson (t03, legacy `-c Poisson`, covariance systs excluded; t03c must refuse poisson with covariances selected) / pearson (t03b) / `--statonly` / `--inject` / `--pseudo-experiment` |
| t07 | `profile` (all parameters) |
| t08–t09 | `surface` dense grid and `--surface-amr` |
| t10–t12 | `plot` with `--with-splines`, `--scale-by-width`, `--bkg-subtract` |
| t12s* | `--shapeonly`: global (neyman/CNP/poisson), profile, fc, and `--scale-by-width plot` |
| t13 | `fc` (2 universes) |
| t14 (×5) | full `fc-adaptive` chain: build-mesh → init-bank → print-bank → asimov → brazil, all stages on one `-o t14afc` (`run_test_tag`; before 2026-10-13 each stage had its own `-o`, so every stage after build-mesh failed to load its input and exited 0 — the chain compared nothing) |
| t15–t16 | `mcmc` (1 chain), `scale-test` benchmark smoke |
| t40–t42 | systematic selection: `--exclude-systs` by name+tag (MC-stat kept) and by plotname (MC-stat dropped), a tag-based `--syst-list`, and a typo'd name refused |
| t17–t19 | PROjector: ND pre-fit → projected `global` → projected `fc` |
| t20–t21 | PROjector negative tests (partial-channel and match-everything patterns must be refused) |
| t22–t26 | `apply_to_subchannel`: process/plot/global with a spline restricted to ND, a covariance to FD, a flat to FD and a norm_to_covariance to ND numu; t25 refuses a zero-match pattern; t26 asserts exact zero blocks / flat splines (`check_applyto.C`) |
| t26b–d | every systematic (incl. mcstat, norm, flat, norm_to_covariance) restricted to FD ⇒ total fractional covariance ND block exactly zero |
| t26e–g | DetVar systematic (same ND file as CV and half-POT variation) restricted to the ND nue channel |
| t26h–j | branch-order regression: an `incl_systematics="false"` branch first in a systematics-carrying MCFile must process and give the identical global fit |
| t26l–o | one DetVar name in three sections (event-matched ND, FD, overlapping ND numu) is ONE systematic: per-section responses, CV-weighted average where sections overlap (`check_spline_response.C`); t26o refuses differing knob sets |
| t27–t30 | regex patterns: alternation accepted; invalid regex and zero-match patterns refused |
| t31–t32 | LBL 3ν models: matter `global` with per-event L+E parameters (own tag + process — its XML replaces the L/E variable with a lean 4-bin L and cuts E to 5 bins, since the 2D model grid is n_L×n_E global bins ×10 components and OOMs at the base binning); explicit legacy-default attributes (`density="3" electron_fraction="0.5" n_newton="0"`) bitwise-identical to no attributes; vacuum `global` from the single signed L/E and matter `global` in fixed-baseline E-only mode (`baseline="1300"`), both reusing the t00 caches; model options on a non-LBL tag refused. Fails by design on binaries predating `LBL_3nu-vacuum_angles` |
| t33 | `template` model (reuses the t00 caches — the `<model>` block is not hashed): one `subchannels=` regex parameter `mu` (ND+FD nue fullosc, `default="0"`) vs the legacy one-exact-name-parameter-per-subchannel form. With every parameter pinned (`--fix` at `--inject-cv`/`--inject-systs-cv`) each global fit is a single χ² evaluation; the two forms must agree at mu=0.02 and 0.8 (t33e/f, χ² parsed from `-l … -w 3` file logs). The unpinned shared fit must recover the injected mu (t33h); `scale-test --tests gradcheck` in a tight box around the Asimov must give analytic mean_rel < 0.2 (t33i — a wide box on this huge template puts the float FD reference at noise-level χ²); zero-match pattern, doubly-claimed subchannel, `default=` outside `[min,max]` and `subchannels=` on a non-template tag refused. Fails by design on binaries predating `subchannels=` |
| t50–t58 | 3+1 (`SBL_3+1_sinsq2thmue`: dmsq, sinsq2thmue, xi) on a copy of the config with `model_rule` remapped (fullosc→2 μe, numu signal→1 μμ, nue intrinsic→3 ee; hashed, so its own process step t50). t51 `fc` at the full 3D point; t52 fc-adaptive refused (non-axis `xi` neither fixed nor profiled); t53 fc-adaptive chain with `--fix xi` (an FC slice); t54/t55 profile FC (`--profile-params xi`, toy truth from the data / from `-i`, `--pval`); t56c `--fix xi --profile-params xi` must equal plain `--fix xi` byte for byte (a fixed parameter cannot be profiled); t57 fc-adaptive profile chain (`--profile-params xi`: data-fitted per-cell truths, PEBank v3 / BrazilArchive v2); t58 asimov on another dataset must be refused (bank fingerprint) |
| t60–t69 | statistical overlap (`overlap_matching_vars`, Appendix G of the tutorial) on a copy of the config with a `nueB` channel: the oscillating nue channel again at half weight, from the same MC entries, every MCFile flagged with key `Run,Subrun,Evt,category` (own tag + process). t61–t63 `global` Asimov, `--pseudo-experiment` and `fc` with overlap-correlated toys; t64 `-c CNP` refused; t66 fc-adaptive chain (bank v4 / archive v3); t69 asserts the exact closure: the stat-only Asimov surface with the overlap (t65) equals the plain config's (t68), and without it (t67, attributes stripped) it does not (1.5×). Fails by design on binaries predating `overlap_matching_vars` |

Outputs land in `tests/runs/<TAG>/` with per-test logs in `logs/` and a
PASS/FAIL `summary.txt`. Exit code = number of failures.

Everything is single-threaded and fixed-seed, so **two runs of identical code
produce byte-identical `.bin`/`.txt` artifacts** and semantically identical
ROOT files (ROOT file bytes always differ via embedded timestamps).

## What the comparison does

`compare_tags.sh <TAG1> <TAG2>` walks the two run directories and writes a
markdown report (`tests/runs/compare_<TAG1>_vs_<TAG2>.md`):

- `.bin` / `.txt` / `.xml` — byte-level `cmp` (expected IDENTICAL for
  identical physics),
- `.root` — dumped to text with `tests/dump_root.C` (histogram cells, graph
  points, per-leaf tree statistics) and diffed (MATCH/DIFFER),
- `.pdf` — size note only (never comparable, never fails the compare),
- `summary.txt` — PASS/FAIL verdicts compared, timings ignored,
- missing/extra files flagged.

Exit code = number of DIFFER + MISSING findings, so it can gate a script.

## Knobs

Environment variables (see script headers for the full list):
`PROFIT_BIN` (default `build/bin/PROfit`), `PROFIT_TEST_MCDIR` (where the
`fake_sbn_mc_{ND,FD}.root` files live), `PROFIT_TEST_OUTDIR`,
`PROFIT_TEST_TIMEOUT` (per test, default 1800 s), `ROOTEXE` (for the dumper;
auto-detected from PATH or `/usr/local/root/root/bin/root`).

Caveats: comparisons are only meaningful between runs on the **same machine
with the same MC directory** (the localized XML path enters the config hash),
and `-n 1` must not be raised (thread scheduling breaks AMR determinism).
