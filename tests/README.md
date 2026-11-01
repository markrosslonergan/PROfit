# PROfit short test suite

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
| t70–t74 | `LBL_3+1-vacuum_angles` on a copy of the config with `model_rule` remapped to its active-flavour columns (fullosc→4 Pμe, numu signal→5 Pμμ, nue intrinsic→1 Pee; hashed, so its own process step t70; the E-only/3ν variants reuse it). t71 Asimov at a sterile point (3ν block, θ34 and the phases pinned) must recover log10 dmsq_41 and log10 sinsqt24 (t71b); t72 gradcheck analytic mean_rel < 0.2; t73 E-only at `baseline="1300"` with truth-bin averaging, 3ν atmospheric parameters free (smoke/regression, no recovery check — fast presets can land in the wrong mass ordering); t74 every parameter pinned at 1300 km (pinned NO vs IO data): `LBL_3nu-vacuum_angles` and the 3+1 model at zero sterile mixing with `bin_average="0"` must give the same χ² (t74c). Fails by design on binaries predating `LBL_3+1-vacuum_angles` |
| t75–t77 | `<model><prior>` physics priors on copies of the t70/t00 configs (the `<model>` block is not hashed, so they reuse those caches). With the data equal to the pinned prediction the χ² is the prior term alone: t75b `abs` prior on `dmsq_31` at −2.4e-3 → 0.04, t75d signed → 96.04, t75h the LBL derived quantity `dmsq_32` = `dmsq_31` − `dmsq_21` → 0.002704, t75f a log10 parameter (SBL `dmsq`, physical units) → 25. t76a `fc` with a prior (toys re-observe its centre); t76b `fc --reuse` of that file without the prior must be refused (`fc_meta`); t76c a prior on a non-parameter refused. t77 turning the carried-only true-L/E variable to `plot="true"` on the t00 cache must be refused (process stored no matrices with it). Fails by design on binaries predating physics priors |
| t78 | Pearson analytic gradient: `-c pearson scale-test --tests gradcheck` (t78a) and with `--shapeonly` (t78c); analytic vs central-full mean_rel < 0.2 and no fallback warning (t78b/d). On binaries predating it, analytic silently fell back to one-sided-full, so t78b/d fail there by design |
| t79 | Preconditioned L-BFGS-B descents (fit option `precondition`): the same `--pseudo-experiment` on the E-only `LBL_3+1-vacuum_angles` config (t70 cache, `-p grad-fast`, Δm²₃₁/θ₂₃/θ₁₃/Δm²₄₁/θ₂₄ free) with `precondition 0` (t79a) vs the default auto = on for LBL (t79b); t79c requires the preconditioned global χ² to be > 0.1 below the unscaled one and the log to say the descents were preconditioned. t79d `precondition 2` refused; t79e SBN smoke with it forced on under `--shapeonly` (Gauss-Newton rescale chain rule). Binaries predating the option fail t79b–d by design |
| t80–t82 | `--lndet` / `--frozen-cov` / `--cov-reference`. t80 exact: pearson, covariance systs excluded, one template scale `mu` on every subchannel (`subchannels="."`) pinned at 2 against Asimov data at 1, reference at 1 ⇒ M = diag(k μ): t80e the ln det term equals N ln 2 (N from the ndf line), t80f frozen χ² = 2 × dynamic and a reference at the tested point leaves χ² unchanged. t81 gradcheck (analytic mean_rel < 0.2, no fallback): neyman `--lndet` (t81a), + `--shapeonly` (t81c), pearson `--lndet` and CNP `--frozen-cov` with the splines pinned (a frozen M does not grow with the prediction, so at random spline points χ² is huge and the float FD reference is noise). t82 `fc --lndet` (fc_meta) whose `--reuse` without it is refused, poisson + `--lndet`, an unknown `--cov-reference` name and a PROjector stage 2 with `--lndet` on t17's plain constraint all refused. Binaries predating the options fail t80–t82 by design |
| t83 | `--ordering-octant-seeds`: t83a/b an SBN global fit with the flag must be byte-identical to t01 (the nueapp model has no mirrors). t83c/d/e `LBL_3nu-vacuum_angles` E-only, Asimov at normal ordering / upper octant (Δm²₃₁ 2.45e-3, sin²θ₂₃ 0.6), only Δm²₃₁ and θ₂₃ free, one start at inverted / lower (`--inject-cv`) and no Latin multistarts (`--fit-options n_latin_points 1 n_localfit 1`): without the flag the fit stays in the wrong basin (χ² > 1), with it a mirror of the start reaches χ² < 1e-3. t83f/g/h the same case with `--ordering-octant-polish` (variants of each fit's best point) alone and together with the seeds: both reach χ² < 1e-3; t83i/j an SBN fit with the polish flag is byte-identical to t01. Binaries predating the flags fail t83 by design |

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
