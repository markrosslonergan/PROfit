<p align="center">
<img src="/other/PRofit_mono.png" width="350">
</p>

<h3 align="center">A PROfessional, PROductive, PROficient and PROfound<br>neutrino-oscillation and BSM fitting framework.</h3>

<p align="center">
<a href="https://github.com/markrosslonergan/PROfit/actions/workflows/cmake-single-platform.yml"><img src="https://github.com/markrosslonergan/PROfit/actions/workflows/cmake-single-platform.yml/badge.svg" alt="Build"></a>
<a href="https://markrosslonergan.github.io/PROfit/"><img src="https://github.com/markrosslonergan/PROfit/actions/workflows/doxygen-pages.yml/badge.svg" alt="Docs"></a>
<a href="https://github.com/markrosslonergan/PROfit/tags"><img src="https://img.shields.io/github/v/tag/markrosslonergan/PROfit?label=release" alt="Latest release"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
</p>

---

**PROfit** is a fast, modern C++ framework for frequentist fits of
short-baseline neutrino oscillation and BSM physics models, developed for the
SBN/ICARUS program. You describe your entire analysis, be it MC files, selections,
binning, oscillation model, and systematic uncertainties, in a single XML
file, and PROfit turns it into spectra, covariance matrices, response splines,
and statistically rigorous fits and confidence regions.

## Highlights

- **One XML, one binary.** A single `PROfit` executable with subcommands for
  every stage of an analysis: `process`, `plot`, `global`, `profile`,
  `surface`, `fc`, `fc-adaptive`, `mcmc`, `scale-test`.
- **Event-by-event MC reweighting** with cached binary event stores — read
  your ROOT ntuples once, then iterate on fits in seconds. Caches are
  hash-validated against the XML, so stale inputs are caught automatically.
- **Flexible systematics**: CAFAna-style response splines, SBNfit-style
  fractional covariances, flat normalizations, detector-variation samples,
  MC-stat errors, and covariance↔spline conversion via eigenmode
  decomposition,  all freely mixed, with optional Gaussian priors and
  correlations.
- **Three χ² metrics** : standard covariance (`PROchi`), combined
  Neyman-Pearson (`PROCNP`), and Poisson  likelihood — behind
  one common interface, with analytic and Gauss-Newton gradient modes.
- **A robust global fitter**: Latin-hypercube sampling → particle-swarm
  optimization → multi-start L-BFGS-B, with tunable presets from `fast` to
  `overkill` and harmonic-oscillation seeding for Δm²-like parameters.
- **Multithreaded 1D profiles and 2D surfaces** with cross-thread warm-start
  seed banks, plus adaptive mesh refinement (AMR) for surfaces.
- **Feldman-Cousins done properly**: classic per-point FC and an adaptive
  Feldman-Cousins (`fc-adaptive`) pipeline that builds a meta-mesh, grows
  pseudo-experiment banks level-by-level, and produces sensitivity/Brazil
  bands at a fraction of the brute-force cost.
- **PROjector two-stage fits**: fit a constraining sample (e.g. a near
  detector) once, save the posterior as a correlated prior, and re-use it in
  downstream fits, profiles, surfaces, and FC — projection without
  approximation [Under Development!]
- **Multiple physics models** out of the box (3+1, 3+2, two-flavor, LBL in
  matter or vacuum with configurable baseline and density, simple scaling)
  with a clean base class for adding your own.
- **Deterministic and tested**: seeded runs are bit-reproducible, and a
  regression test suite covering every major workflow ships in
  [`tests/`](tests/README.md).

Built on [Eigen](https://gitlab.com/libeigen/eigen) for all internal linear algebra,
[LBFGSpp](https://github.com/yixuan/LBFGSpp),
 and ROOT for input/output and some plot making. Long-baseline oscillation
probabilities use the vendored
[NuFast-LBL](https://github.com/PeterDenton/NuFast-LBL) algorithm (see
[Acknowledgements & citations](#acknowledgements--citations)).

## Documentation

| Resource | What it covers |
|---|---|
| [**Walk-through tutorial**](docs/tutorials/PROfit_Tutorial_v2.md) | The place to start. A complete v2.X walk-through: concepts, the XML format, every subcommand, Feldman-Cousins, and PROjector — with regenerable plots. |
| [**API reference (Doxygen)**](https://markrosslonergan.github.io/PROfit/) | Auto-generated class documentation, rebuilt on every push. Build locally with `make docs` (requires Doxygen). |
| [**Test suite guide**](tests/README.md) | Deterministic regression tests for before/after comparisons of physics-touching changes. |

Questions? Join **#profit** on the SBN Slack, mail
`profit@listserv.fnal.gov`, or open an
[issue](https://github.com/markrosslonergan/PROfit/issues).

## Installation

PROfit needs **ROOT**, **Boost**, **HDF5**, and **CMake ≥ 3.x** available in
your environment (via `apt-get`, `homebrew`, or your experiment's software
stack). Smaller dependencies (Eigen, LBFGSpp, CLI11, TinyXML2, …) are fetched
and built automatically by CMake.

```bash
git clone https://github.com/markrosslonergan/PROfit.git
cd PROfit/build
cmake ..
make -j4                     # → build/bin/PROfit
export PATH=$PATH:$PWD/bin   # optional
```

<details>
<summary>Setting up dependencies on the FNAL gpvms</summary>

In an AL9 enviroment (Prefered)
```bash
    source /cvmfs/larsoft.opensciencegrid.org/spack-packages/setup-env.sh
    spack load gcc@12.2.0
    spack load cmake@3.27.7
    spack load boost/iylabz2
    spack load hdf5@1.14.3
    spack load root@6.28.12
    spack load ifdhc@2.7.2
```


Inside an SL7 container:

```bash
source /cvmfs/larsoft.opensciencegrid.org/setup_larsoft.sh
setup root v6_28_12 -q e26:p3915:prof
setup cmake v3_27_4
setup hdf5 v1_12_2b -q e26:prof
setup boost v1_82_0 -q e26:prof
```

PROfit aims to stay generic across versions, but the above combination is
confirmed to compile and run on the gpvms.

</details>

<details>
<summary>Python interface (pybind11)</summary>

With ROOT, HDF5, and Boost available globally:

```bash
python -m venv env && . env/bin/activate
pip install --upgrade pip
pip install wheel setuptools pybind11 numpy==2.0.2
pip install git+https://github.com/markrosslonergan/PROfit
```

Then `import profit` in a Python shell, use the Python executables
(`PROsurf.py`, …), or run the binaries via the `PRO` helper (`PRO PROsurf`).
For development, clone the repo and `pip install .` from the checkout,
re-running it after changes.

</details>

## Quick start

Every PROfit command takes an XML config (`-x`) and an analysis tag (`-t`):

```bash
# One-time: read the MC ROOT files and cache event store + systematics
PROfit -x analysis.xml -t MyTag process

# Plot CV spectra with error bands and covariances
PROfit -x analysis.xml -t MyTag plot

# Global best fit
PROfit -x analysis.xml -t MyTag global

# 1D profiled Δχ² for every physics and nuisance parameter (8 threads)
PROfit -x analysis.xml -t MyTag -n 8 profile

# 2D Δχ² surface over the physics parameters
PROfit -x analysis.xml -t MyTag -n 8 surface

# Feldman-Cousins for an injected signal
PROfit -x analysis.xml -t MyTag -n 8 --inject dmsq 1.0 fc
```

`PROfit --help` and `PROfit <subcommand> --help` document every option;
`--fit-help` explains the fitter presets. The
[tutorial](docs/tutorials/PROfit_Tutorial_v2.md) walks through all of this on
a realistic two-detector example.

## How it works

```
XML config
   │  (parsed by PROconfig — owns all binning/bookkeeping)
   ▼
CAF/MC ROOT files ──► process ──► <TAG>_prop.bin + <TAG>_syst.bin
   │                             (event store + systematics, read ONCE,
   │                              hash-checked against the XML forever after)
   ▼
PROsyst (splines + fractional covariances)     PROpeller (per-event MC)
   │                                                │
   └──────────────────┬─────────────────────────────┘
                      ▼
   χ² metric (PROchi / PROCNP / Poisson) ⇐ binds config, MC, systs, model, data
                      ▼
   PROfitter (Latin hypercube → particle swarm → L-BFGS-B)
                      ▼
   global fit / profile / surface / FC / adaptive-FC / PROjector
                      ▼
   <TAG>_<out>_*.root, *.pdf, *.txt
```

PROfit is deliberately a *fitting* framework, not an end-to-end analysis
framework: it expects a final-stage selection as input and gives you fast,
reproducible statistical inference on top of it.

## Acknowledgements & citations

PROfit builds on the following third-party code. Vendored files retain their
original license headers in-tree; the remaining dependencies are fetched and
built automatically by CMake.

| Component | How it's used | License |
|---|---|---|
| [NuFast-LBL](https://github.com/PeterDenton/NuFast-LBL) | Vendored in [`src/NuFastLBL.cxx`](src/NuFastLBL.cxx) — fast long-baseline oscillation probabilities in matter and vacuum | MIT (© 2024 Peter B. Denton) |
| [LBFGSpp](https://github.com/yixuan/LBFGSpp) | Fetched ([our fork](https://github.com/markrosslonergan/LBFGSpp)) — the L-BFGS-B bounded local minimizer at the heart of every fit | MIT (© Yixuan Qiu) |
| [Eigen](https://gitlab.com/libeigen/eigen) 3.4.0 | Fetched — all internal linear algebra | MPL2 |
| [CLI11](https://github.com/CLIUtils/CLI11) 2.3.2 | Vendored in [`inc/CLI11.h`](inc/CLI11.h) — command-line interface | 3-clause BSD (University of Cincinnati / Henry Schreiner) |
| [TinyXML2](https://github.com/leethomason/tinyxml2) 9.0.0 | Fetched — XML configuration parsing | zlib |
| [MurmurHash3](https://github.com/aappleby/smhasher) | Vendored in [`inc/MurmurHash3.h`](inc/MurmurHash3.h) — XML hashing for binary-cache validation | Public domain (Austin Appleby) |

**Please cite:** if you use the LBL oscillation model in a publication, cite
the NuFast paper, [arXiv:2405.02400](https://arxiv.org/abs/2405.02400) (the
vendored source header points there too).

## Versioning

Development happens on the **`project-SBN-dev`** branch (the v2 release
line). The v1→v2 update made breaking XML changes — v1.x configs will not
work with v2 binaries, and bugfixes are not back-ported to v1.1. Use the
latest v2 tag for anything new.

## Contributing

Bug reports and pull requests are welcome. Before opening a PR that touches
physics code, run the deterministic regression suite
([`tests/README.md`](tests/README.md)):

```bash
tests/run_short_tests.sh ref      # baseline
# ...build your change...
tests/run_short_tests.sh mine
tests/compare_tags.sh ref mine    # semantic diff of all outputs
```

## License

PROfit is released under the [MIT License](LICENSE). Vendored third-party
code retains its original licenses — see
[Acknowledgements & citations](#acknowledgements--citations).
