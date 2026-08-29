# PROfit v2.X Walk-Through Tutorial

This tutorial walks through the full PROfit v2.X workflow, from the conceptual
building blocks and XML configuration all the way to Feldman-Cousins and the
PROjector two-stage fit. Every plot shown here can be regenerated with the
companion script [`make_tutorial_plots.sh`](make_tutorial_plots.sh) in this
directory; the embedded images live in [`figures/`](figures/) and are rendered
from those PDFs by [`make_tutorial_figures.sh`](make_tutorial_figures.sh) —
each figure caption names the exact output file it came from.

> **Note:** This walk-through was developed and tested on the `project-SBN-dev`
> branch of PROfit v2.X. If commands here disagree with `PROfit --help` on your
> checkout, trust the binary.

Useful contacts and links:

* GitHub: [markrosslonergan/Elephant_Vanishes](https://github.com/markrosslonergan/Elephant_Vanishes)
* Slack: **#profit** (shortbaseline/SBN workspace)
* Listserv: profit@listserv.fnal.gov

> **Versioning:** this tutorial targets the **v2 release line** (development
> branch `project-SBN-dev`). Breaking XML changes were made in the v1→v2
> update, so v1.x XMLs will **not** work with v2 binaries, and bugfixes are
> not back-ported to the v1.1 line — use v2.1.1+ for anything new.

### Getting set up

PROfit needs ROOT, Boost, HDF5, and CMake (on the FNAL gpvms — AL9 since the
2024/25 migration — set these up from cvmfs with `spack load`; on legacy SL7
containers use UPS `setup`; on your own machine apt-get or homebrew versions
are fine). Then:

```bash
git clone https://github.com/markrosslonergan/Elephant_Vanishes.git
cd Elephant_Vanishes/build
cmake ..
make -j4            # → build/bin/PROfit
export PATH=$PATH:$PWD/bin   # optional
```

The configuration used throughout is `working_dir/Neutrino2026/fake_sbn_v2.xml`:
a **fake** two-detector (ND + FD) SBN-style setup with a νe appearance search
(`nueapp` model: `dmsq`, `sinsq2thme`), two channels (`nue`, `numu`), and a
realistic mix of spline, covariance, flat-normalisation, and MC-stat
systematics. The MC files (`fake_sbn_mc_ND.root`, `fake_sbn_mc_FD.root`, ~1 GB
each) live alongside the XML, or ping Mark Ross-Lonergan on Slack for a
tarball. **These are toy files for teaching — do not use for physics results.**


All commands below use `tutorial.xml`, the tag `TUT`, and a fixed seed so your
numbers should match.

---

## Table of contents

1. [PROfit conceptual introduction](#1-profit-conceptual-introduction)
2. [The PROfit XML](#2-the-profit-xml)
3. [General arguments and how stuff works](#3-general-arguments-and-how-stuff-works)
4. [Subcommand `process` — loading your files](#4-subcommand-process--loading-your-files)
5. [Subcommand `plot` — exploring your spectra](#5-subcommand-plot--exploring-your-spectra)
6. [Subcommand `global` — fitting and fitter configuration](#6-subcommand-global--fitting-and-fitter-configuration)
7. [Subcommand `profile` — 1D profiled Δχ²](#7-subcommand-profile--1d-profiled-χ²)
8. [Subcommand `surface` — 2D Wilks surfaces and AMR](#8-subcommand-surface--2d-wilks-surfaces-and-amr)
9. [Feldman-Cousins: `fc` and `fc-adaptive`](#9-feldman-cousins-fc-and-fc-adaptive)
10. [PROjector — two-stage pre-fit / projected fits](#10-projector--two-stage-pre-fit--projected-fits)
11. [PROletariat — grid submission: `proletariat`](#11-proletariat--grid-submission-proletariat)

Appendices:

* [Appendix A: regenerating every plot in this tutorial](#appendix-a-regenerating-every-plot-in-this-tutorial)
* [Appendix B: available physics models](#appendix-b-available-physics-models-incpromodelh)
* [Appendix C: the pre-fit and post-fit error bands, in full](#appendix-c-the-pre-fit-and-post-fit-error-bands-in-full)

---

# 1. PROfit conceptual introduction

PROfit is a frequentist fitting framework for short-baseline neutrino
oscillation analyses. You describe your entire analysis — MC files, event
selections, binning, oscillation model, and systematics — in a single XML
file, and PROfit turns that into spectra, covariance matrices, response
splines, and fits.

In a nutshell, PROfit's job is two-fold:

1. Build, work with, and understand the **systematic uncertainty** of a
   Monte-Carlo prediction — a.k.a. *the Error Bar*.
2. Given data (or fake data) and those systematics, **fit a physics model**,
   so we can quantify what the data tells us about the model *within* that
   uncertainty.

Equally important is what PROfit is **not**: a full end-to-end analysis
framework. It expects a *final-stage* selection as input — arbitrary cuts,
binnings, and variables are all handled well, but they are meant for studying
sub-categories of a finished selection, not for developing one from scratch.

The data flow, start to finish:

```
XML config
   │  (parsed by PROconfig — owns all binning/bookkeeping)
   ▼
CAF/MC ROOT files  ──►  process  ──►  <TAG>_prop.bin + <TAG>_syst.bin
   │                              (event store + systematics, read ONCE,
   │                               hash-checked against the XML forever after)
   ▼
PROsyst (splines + fractional covariances)      PROpeller (per-event MC)
   │                                                 │
   └───────────────┬─────────────────────────────────┘
                   ▼
   χ² metric (PROchi / PROCNP / Poisson)  ⇐ binds config, MC, systs, model, data
                   ▼
   PROfitter (Latin hypercube → particle swarm → L-BFGS-B)
                   ▼
   global fit / profile / surface / FC / adaptive-FC / PROjector
                   ▼
   <TAG>_<out>_*.root, *.pdf, *.txt
```

A few concepts you should internalize before anything else:

**Channels, subchannels, and fullnames.** A *channel* is a selection with a
binning (e.g. the reconstructed-energy νμ selection); a *subchannel* is a
truth-level component stacked inside it (signal, background, cosmics, …). Each
subchannel has a *fullname* of the form

```
<mode>_<detector>_<channel>_<subchannel>     e.g.  nu_FD_numu_signal
```

Everywhere PROfit accepts a "pattern" (background subtraction, POT scaling,
PROjector channel selection, flat/norm systematics, `apply_to_subchannel=`)
it matches it against these fullnames as an **unanchored regex**
(`std::regex_search`, ECMAScript). A plain substring is a valid regex and
behaves exactly as before: `"_ND_"` matches everything in the near detector;
`background` matches every background subchannel in every channel and
detector. Regex gives you more when you need it — `"nu_(ND|FD)_numu"`,
`"^nu_ND_numu_signal$"` (anchored full match). Two caveats: in XML, `&` and
`<` must be written `&amp;`/`&lt;` (standard XML escaping — all other regex
metacharacters pass through untouched), and on the command line quote the
pattern so the shell doesn't expand `*`, `(`, `|`, etc. A pattern that is an
invalid regex, or that matches nothing (flat/norm/`apply_to_subchannel`), is
a fatal, clearly-logged error.

**Variables.** Each channel can carry several binnings (`<bins>` entries):
reconstructed energy, true L/E, true energy, and so on. Exactly one of them is
the *fitting variable* — the one the χ² is actually computed in; the others are
used for the oscillation model (true L/E) and diagnostics. Plots are made for
all of them. Mark the fitting variable with `fit="true"` on its `<bins>` entry,
or leave every entry unmarked to fit the first one (the historical default).
`--fit-variable N` overrides the XML at run time, and needs **no re-`process`**:
all variables live in the cached `_prop.bin`/`_syst.bin` already, and the
fitting variable is not part of the config hash.

**The parameter vector.** A fit point is
`[physics parameters, one parameter per spline systematic]`, in that order.
Physics parameters are mostly in **log10 space** (for `nueapp`:
`log10(Δm²) ∈ [-2, 2]`, `log10(sin²2θμe) ∈ [-10, 0]`); spline parameters are
in **σ units** of their knobs. Covariance-type systematics are *not* fit
parameters — they are marginalized analytically inside the χ² covariance
matrix.

**The χ².** For the default `PROchi`:

```
χ² = Δᵀ M⁻¹ Δ  +  Σ pulls(θ)
M  = stat  +  (collapsed, prediction-scaled fractional covariance)
```

with a Gaussian pull term on every spline parameter (priors and centers
configurable per systematic in the XML). `PROCNP` swaps the statistical term
for the combined-Neyman-Pearson variance, and `Poisson` uses the
Baker-Cousins likelihood-ratio sum (and ignores covariance systematics — it
warns).

**Asimov vs fake data.** Unless you say otherwise, the "data" in every fit is
the central-value expectation itself (Asimov). You can inject an oscillation
signal, shift systematics, Poisson-fluctuate, or generate a full FC-style
pseudo-experiment — all from the command line, no XML edits needed.

---

# 2. The PROfit XML

The XML defines the *entire* analysis. Let's walk through
`fake_sbn_v2.xml` block by block. (For deeper background see the
[XML configuration wiki page](https://github.com/markrosslonergan/Elephant_Vanishes/wiki/Minimizing-PROfit:-XML-configuration).)

### Mode and detectors

```xml
<mode name="nu" />
<detector name="ND" pot="1e+21"/>
<detector name="FD" pot="1e+21"/>
```

Multi-detector configs are first-class: each detector gets its own contiguous
block of bins, and every channel is replicated per detector. The `pot` sets
the target exposure each MC file is scaled to.

Three things to keep in mind:

* `mode` and `detector` are **just bookkeeping labels** — they organize the
  spectra and never change the physics. You can use them for run periods
  (`run1`/`run2`) or TPC halves just as well as for ν/ν̄ running.
* A channel is replicated with the **same binning** in every detector. If you
  need different binnings per detector, define separate channels inside one
  dummy detector instead (e.g. `sbndnumu` + `icarusnumu`) — at the cost of
  handling the relative POT scaling yourself in the branch weights.
* **Never put underscores inside a name.** The fullname is built by joining
  the four levels with `_`; an underscore inside a name breaks the pattern
  bookkeeping.

### Channels, binnings, and subchannels

```xml
<channel name="nue" plotname="Fake CC #nu_{e} Selection">
    <bins unit="Reconstructed Neutrino Energy [GeV]" min="0.1" max="3.0" nbins="16"/>
    <bins unit="True L/E [km/GeV]" min="0" max="2.5" nbins="200" plot="false"/>
    <bins unit="True Neutrino Energy [GeV]" min="0" max="3" nbins="20" />
    <bins unit="Random Value" min="0" max="1" nbins="50"/>
    <subchannel name="intrinsic" plotname="Intrinsic #nu_{e} CC" color="#34A853"/>
    <subchannel name="background" plotname="#nu_{e} Backgrounds" color="#FF6961"/>
    <subchannel name="fullosc" plotname="#nu_{#mu}#rightarrow#nu_{e} (full osc)" color="#4285F4"/>
    <subchannel name="cosmic" plotname="Cosmics" color="#E37400"/>
</channel>
```

Each `<bins>` entry is one *variable*, numbered from 0 in the order they appear
(any `<bins2D>` entries come first, then the 1D `<bins>`). With no `fit=`
attribute anywhere, the **first** entry is the fitting variable (reconstructed
energy here); the rest are extra binnings that PROfit tracks and plots for you
(`plot="false"` suppresses plotting of a variable, useful for the 200-bin
true-L/E binning that only exists for the oscillation model). Subchannel
`plotname` and `color` control the stacked histograms in `plot`.

To fit a different variable, put `fit="true"` on its `<bins>` entry:

```xml
    <bins unit="Reconstructed Neutrino Energy [GeV]" min="0.1" max="3.0" nbins="16"/>
    <bins unit="True L/E [km/GeV]" min="0" max="2.5" nbins="200" plot="false"/>
    <bins unit="True Neutrino Energy [GeV]" min="0" max="3" nbins="20" fit="true"/>
    <bins unit="Random Value" min="0" max="1" nbins="50"/>
```

At most one entry per channel may be marked, and every channel that marks one
must land on the same index — variables are numbered globally, so a
disagreement is a config error and PROfit refuses to start. A variable that a
`<parameter variable_index=...>` claims for the oscillation model (true L/E
here) is also refused: that binning is a physics grid, not an observable.
`--fit-variable N` overrides the XML for a single run without touching the file
and without re-running `process`.

### The oscillation model

```xml
<model tag="nueapp">
    <rule index="0" name="No Osc"/>
    <rule index="1" name="Nue Appearance"/>
    <parameter name="L/E" variable_index="1"/>
</model>
```

`tag` selects the physics model implemented in `inc/PROmodel.h`
(`numudis` = 3+1 νμ disappearance with parameters `dmsq`, `sinsq2thmm`;
`nueapp` = 3+1 νμ→νe appearance with `dmsq`, `sinsq2thme`; also available:
`nuedis`, the full three-parameter `3+1` (|Ue4|², |Uμ4|², Δm² with a
unitarity constraint), and several re-parameterized `3+1_*` variants — see
section 7 for why those exist, and the model appendix at the bottom of this
document for every model's parameters, bounds, and rules). Each *rule* is an
oscillation-weight
function; **rule 0 always means "no oscillation" and is the default**. Every
MC branch declares which rule applies to it, so e.g. intrinsic backgrounds
sit on rule 0 while the fullosc component oscillates. The `<parameter>` line
tells the model which channel variable holds true L/E (variable index 1,
i.e. the second `<bins>` entry). The XML model block is pure bookkeeping —
the actual oscillation logic lives in `PROmodel.h`.

### MC files and branches

```xml
<MCFile treename="events/selected" filename=".../fake_sbn_mc_ND.root" scale="1.0" pot="1e+21">
    <friend treename="events/multisigmaTree" />
    <friend treename="events/multisimTree" />
    <friend treename="events/variationTree" />
    <branch
        associated_subchannel = "nu_ND_nue_intrinsic"
        model_rule            = "0"
        additional_weight     = "5*mcweight*(category == 0)"
    >
        <variable>reco_visible_energy</variable>
        <variable>true_baseline/(1000*true_neutrino_energy)</variable>
        <variable>true_neutrino_energy</variable>
        <variable>random_value</variable>
    </branch>
    ...
</MCFile>
```

Each `<branch>` fills one subchannel: `additional_weight` is an arbitrary
TTree formula (here also doing the truth-category selection — anything a
`TTreeFormula` can do is allowed), `model_rule` picks the oscillation rule,
and the `<variable>` list maps one formula to each `<bins>` entry of the
channel, **in order**. `<friend>` trees carry the systematic weight branches
(multisigma/multisim/variation trees). Branches with
`incl_systematics="false"` (the cosmics here) get no systematic variations
at all.

Two MCFile details worth knowing: `filename` can be a single ROOT file **or a
plain-text filelist** of ROOT files, and the `pot` attribute is the
*generated* POT of that file — events are scaled from it up to the detector
`pot` declared at the top of the XML.

### Systematics: the `<variation_list>`

```xml
<variation_list>
    <allowlist type="mcstat" plotname="MC Stats" tag="other">MCStat</allowlist>
    <allowlist type="spline" binning="var0" plotname="Flux1" tag="flux">Flux1</allowlist>
    ...
    <allowlist type="covariance" plotname="RPA_CCQE" tag="QE-MEC">RPA_CCQE</allowlist>
    ...
    <allowlist type="norm" plotname="FluxNorm_ND" tag="other">nu_ND_numu:0.02</allowlist>
</variation_list>
```

The five main `type`s:

| type | Source | Becomes | Fit parameter? |
|---|---|---|---|
| `covariance` | multisim universes | fractional covariance matrix | no — marginalized analytically in the χ² |
| `spline` | multisigma knobs (±1,2,3σ universes) | per-bin cubic response spline | yes, one per systematic, in σ units |
| `mcstat` | finite MC statistics | diagonal MC-stat covariance | no |
| `norm` | you, in the XML (`pattern:fraction`) | ONE spline parameter giving a flat ±fraction normalisation on every subchannel whose fullname contains the pattern — 100% correlated inside the match, uncorrelated outside | yes (a spline; fraction must be < 0.333) |
| `flat` | you, in the XML (`pattern:fraction`) | diagonal-only covariance — the error is uncorrelated bin-by-bin | no |

`mcstat` you will almost always want on (finite MC statistics IS a
systematic on the prediction); `norm` is the right tool for flux
normalisations, POT/fiducial-volume errors and flat detector systematics
(note the `nu_ICARUS:0.02`-style entries showing up as fit parameters in the
global-fit tables later); `flat` is mostly for sensitivity tests and
worst-case scenarios.

The `<allowlist>` attributes:

* `type` — the method, as above.
* `plotname` — label used on plots.
* `tag` — a grouping label so whole sets of systematics can be
  included/excluded and, importantly, grouped in the fractional-systematics
  plots (comma-separated multi-tags are allowed but a few functions dislike
  them).
* `binning` — which variable's bins the systematic response is built on
  (`var0`, `var1`, ... in the order of the `<bins>`/`<variable>` entries; the
  default, `reco`, means *the fitting variable*, so it follows `fit="true"`).
  E.g. cross-section splines are often better built in true energy, detector
  systematics in the reco variable. Note this default is baked into
  `_syst.bin` at `process` time: if you later switch the fitting variable, an
  unqualified systematic keeps the binning it was built with. Write
  `binning="varN"` explicitly if you want that pinned down.
* `knobvals` — for splines, the knob values if not stored in the file, as a
  space-separated list (default `-3 -2 -1 0 1 2 3`).
* `prior=` / `center=` — override the default N(0,1) Gaussian pull;
  `<correlation>` blocks make spline priors correlated.
* `mode="covariance_to_spline"` with `num_decomp_knobs=` promotes a
  covariance to its leading eigenmode splines (the same machinery PROjector
  uses — see section 9). `restrict` bounds a spline's allowed range.
* `apply_to_subchannel="pattern"` — restrict a weight-based systematic
  (`spline`, `covariance`, `covariance_to_spline`, `hist1d/2d`, ...) to the
  subchannels whose fullname matches the pattern (same unanchored-regex
  matching as `norm`/`flat` — plain substrings work as-is, e.g.
  `apply_to_subchannel="nu_SBND"` or `"_ND_"`, and regex like
  `"nu_(ND|FD)"` too). Non-matching subchannels get exactly no response (flat spline
  at 1 / zero covariance block), and the systematic's weight branch is only
  required — or even looked for — in MCFiles that fill a matching
  subchannel. This is how per-detector systematics work in multi-detector
  fits where each detector's MC carries a different set of weight branches.

---

# 3. General arguments and how stuff works

The standard invocation is always

```bash
PROfit -x tutorial.xml -t TUT [GLOBAL OPTIONS] SUBCOMMAND [SUBCOMMAND OPTIONS]
```

Run `PROfit --help` for the full list. The subcommands in v2.X:

```
Subcommands:
  process      PROcess the MC and systematics in root files into binary data for future rapid loading.
  surface      Make a 2D surface scan of two physics parameters, profiling over all others.
  profile      Make a 1D profiled chi2 for each physics and nuisence parameter.
  plot         Make plots of CV, or injected point with error bars and covariance.
  fc           Run Feldman-Cousins for this injected signal
  fc-adaptive  Adaptive Feldman-Cousins. Sub-modes: build-mesh, init-bank, print-bank, print-mesh, asimov, brazil, brazil-cleanup, merge-mesh, merge-bank.
  global       Just do a single global fit.
  mcmc         Get bayesian posteriors using MCMC
  scale-test   Run timing benchmarks for FillSpectra / metric / fit hot paths.
```

### Tags, outputs, and the binary cache

- `-t/--tag TUT` names one *processing* of the ROOT files. The first time any
  subcommand runs it creates `TUT_prop.bin` (the event store) and
  `TUT_syst.bin` (the systematics) in the current directory; every later run
  with the same tag loads these instead of touching ROOT files. You almost
  never need to run `process` explicitly.
- A MurmurHash of the XML is stored inside the binaries. Change the XML and
  keep the tag, and PROfit refuses to load with a hash-mismatch ERROR — rerun
  `process` (or change the tag). `--force` overrides the check; be careful.
- `-o/--output v1` is a *secondary* label that goes into every output
  filename (`<tag>_<out>_...`) but does **not** touch the binaries. Use it to
  run many studies off one processing without overwriting results.

```bash
PROfit -x tutorial.xml -t TUT --log process.log process
# → TUT_prop.bin  TUT_syst.bin   (~a minute for this config, once)
```

### Data, injection, and fake-data options

By default the fit data is the Asimov CV. You can build essentially any fake
dataset from the command line:

| Option | What it does |
|---|---|
| `-i/--inject dmsq 1 sinsq2thme 0.01` | inject an oscillation signal as truth (**name value pairs** in v2.X, in linear units) |
| `--inject-systs Flux1 1.0 DetSys2 -2.0` | shift spline systematics (in σ) into the fake data |
| `--inject-cv` / `--inject-systs-cv` | same, but shift the *prediction* CV instead of the data |
| `--poisson-throw` | Poisson-fluctuate the fake data |
| `--pseudo-experiment` | full FC-style throw: spline pulls + covariance Cholesky shift + Poisson stats (combines with `--inject`) |
| `-d/--data file` | load real data from a separate file/XML (plot subcommand) |
| `--use-fake-data` | ignore any embedded/external data and force MC fake data |
| `--scale ND 0.5` | scale POT of subchannels matching a pattern |
| `--seed 405` | fix the RNG seed (default -1 = hardware random) |

Reproducibility rule of thumb: `--seed N` plus `-n 1` is bit-reproducible;
multithreaded runs are statistically equivalent but not byte-identical.

### Controlling the systematics and parameters

| Option | What it does |
|---|---|
| `--syst-list Flux1 Flux2` | use ONLY these systematics |
| `--exclude-systs RPA_CCQE` | use everything except these |
| `--fix dmsq Flux1` | fix parameters at CV (physics or splines) |
| `--syst-only` | fix ALL physics parameters (nuisance-only fit) |
| `--statonly` | drop systematics entirely |
| `--shapeonly` / `--rateonly` | shape-only or single-bin-normalisation analysis |
| `-c/--chi2 PROchi\|PROCNP\|Poisson` | χ² metric (default PROchi) |
| `--grad-mode central-lin` | gradient strategy: `central-full` (most accurate) / `one-sided-full` / `central-lin` (default, Gauss-Newton, 5-10× faster) / `one-sided-lin` |

`--grad-mode central-lin` is exact at minima and fine for scans; use
`central-full` for final publication-quality runs.

### Logging and housekeeping

`-v` sets terminal verbosity (1 Error → 4 Debug), `-l/--log file` saves all
messages to a log file *in addition* to the terminal, `-w/--file-verbosity`
sets the log file's verbosity independently, and `-b/--progress` draws
progress bars where applicable (messy combined with high verbosity). A
common pattern — quiet terminal, full log, progress bars:

```bash
PROfit -x tutorial.xml -t TUT -v 1 -w 3 --log fit.log --progress ... profile
```

Log files are cheap and strongly encouraged: the full fitter configuration
and the global best-fit tables are printed there, which makes checking on
past runs painless. Fair warning: Info/Debug output from multithreaded scans
interleaves and can be hard to read.

`-n/--nthread N` parallelizes all fitting code. `-m/--max N` truncates the
MC event loop (quick tests only).

---

# 4. Subcommand `process` — loading your files

```
Usage: PROfit process [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
```

The process command loops through events in the input data, prediction, and systematic files and processes them to efficiently store only the necessary information described in the xml. 
The output is _prop.bin and _syst.bin files which are efficiently loaded by the remaining PROfit commands.

---

# 5. Subcommand `plot` — exploring your spectra

```
Usage: PROfit plot [OPTIONS]
Options:
  --with-splines              Include graphs of splines in output.
  --bkg-subtract TEXT         Substring pattern; that background's CV is subtracted
                              from data and CV at plot time (publication convention).
```

Plus the relevant global options: `--area-norm`, `--scale-by-width`,
`--plot-bounds ymax 100 ratmin 0.5 ratmax 1.5`, and all the injection
machinery from section 3.

### The CV and error band

```bash
PROfit -x tutorial.xml -t TUT -o plotcv --seed 405 plot
```

Outputs (one `Variable_<i>` set per plotted binning of each channel):

* `TUT_plotcv_PROplot_Variable_0_CV.pdf` — stacked CV spectra, fitting variable
* `TUT_plotcv_PROplot_Variable_0_ErrorBand.pdf` — CV + full systematic band
* `TUT_plotcv_PROplot_Variable_2_*.pdf`, ... — same for the other variables
* `TUT_plotcv_PROplot_Covar.pdf` — all covariance matrices, per systematic and total
* `TUT_plotcv_fractional_systematics.pdf` — fractional uncertainty per bin, one panel per systematic `tag` plus a summary (the `tag=` attributes in the variation list control this grouping — without them you get one unreadable 30-line legend)
* `TUT_plotcv_ratio_fractional_systematics.pdf` — same as a ratio
* `TUT_plotcv_PROplot.root` — everything above as ROOT objects

<img src="figures/TUT_plotcv_PROplot_Variable_0_CV.png" width="800"/>

*`TUT_plotcv_PROplot_Variable_0_CV.pdf` — stacked CV, reconstructed energy, ND+FD νe and νμ channels.*

<img src="figures/TUT_plotcv_PROplot_Variable_0_ErrorBand.png" width="800"/>

*`TUT_plotcv_PROplot_Variable_0_ErrorBand.pdf` — CV + total systematic error band; "data" points are the Asimov CV since we injected nothing.*

<img src="figures/TUT_plotcv_PROplot_Covar.png" width="600"/>

*`TUT_plotcv_PROplot_Covar.pdf` (page 1) — the total collapsed correlation matrix.*

<img src="figures/TUT_plotcv_fractional_systematics.png" width="600"/>

*`TUT_plotcv_fractional_systematics.pdf` — per-bin fractional uncertainty by systematic tag: flux / xsec / det / QE-MEC / RES / other.*

### Injecting a signal

Inject a 1 eV² sterile with sin²2θμe = 0.01 (note the v2.X **name value**
pair syntax, linear units):

```bash
PROfit -x tutorial.xml -t TUT -o plotinj --seed 405 -i dmsq 1 sinsq2thme 0.01 plot
```

You now additionally get `TUT_plotinj_PROplot_Osc.pdf` (oscillated vs
unoscillated spectra — a handy debugging view of the raw oscillation
pattern) and the "data" points in the error-band plot become the injected
fake data.

A note on the two injection families: `--inject` (and `--inject-systs`)
modify the **fake data** while the prediction stays at CV — the "can I
recover an injected signal?" workflow used throughout this tutorial.
`--inject-cv` (and `--inject-systs-cv`) instead modify the **prediction's
central value**, e.g. to make an oscillated spectrum the null hypothesis.
They can be combined — inject one point as truth and a different one as the
CV — to study fitting under a wrong model.

<img src="figures/TUT_plotinj_PROplot_Osc.png" width="800"/>

*`TUT_plotinj_PROplot_Osc.pdf` — oscillated vs unoscillated spectra for the injected point.*

<img src="figures/TUT_plotinj_PROplot_Variable_0_ErrorBand.png" width="800"/>

*`TUT_plotinj_PROplot_Variable_0_ErrorBand.pdf` — error band with injected-signal fake data.*

### Injecting systematic shifts

Shift `Flux1` up by 1σ and `DetSys2` down by 2σ on top of the signal:

```bash
PROfit -x tutorial.xml -t TUT -o plotsyst --seed 405 \
    -i dmsq 1 sinsq2thme 0.01 --inject-systs Flux1 1.0 DetSys2 -2.0 plot
```

<img src="figures/TUT_plotsyst_PROplot_Variable_0_ErrorBand.png" width="800"/>

*`TUT_plotsyst_PROplot_Variable_0_ErrorBand.pdf` — fake data now includes both the signal and the systematic shifts.*

### Looking at the splines themselves

```bash
PROfit -x tutorial.xml -t TUT -o plotspl --seed 405 plot --with-splines
```

adds `TUT_plotspl_PROplot_Spline.pdf`: per-bin response splines for every
spline systematic, with the knob points overlaid — the single most useful
plot for debugging a suspicious multisigma input.

<img src="figures/TUT_plotspl_PROplot_Spline.png" width="700"/>

*`TUT_plotspl_PROplot_Spline.pdf` (example page) — per-bin response splines with the knob points overlaid.*

### Background subtraction

`--bkg-subtract PATTERN` applies the publication convention: the matched
background's CV is subtracted from both data and prediction, the error band
becomes signal-only (each throw's own background is subtracted, so background
variations cancel), and the background's uncertainty moves onto the data
points as √(N + σ²_bkg-syst + σ²_bkg-MCstat).

```bash
PROfit -x tutorial.xml -t TUT -o plotbsub --seed 405 plot --bkg-subtract background
```

<img src="figures/TUT_plotbsub_PROplot_Variable_0_ErrorBand.png" width="800"/>

*`TUT_plotbsub_PROplot_Variable_0_ErrorBand.pdf` — background-subtracted spectra; compare with the plotcv version.*

Other useful toggles to try yourself: `--area-norm`, `--scale-by-width`,
`--scale FD 0.5` (half the far-detector POT), `--poisson-throw`. And when
comparing plots from different runs, PROfit's auto-ranging gets in the way —
pin the axes with `--plot-bounds ymax 1800 ratmin 0.5 ratmax 1.5` so the two
PDFs are directly comparable by eye.

---

# 6. Subcommand `global` — fitting and fitter configuration

`global` performs one full global best fit of all physics + spline parameters
and draws the post-fit results. It takes no subcommand options of its own —
everything is controlled by the global arguments.

```bash
PROfit -x tutorial.xml -t TUT -o glob1 --seed 405 -n 8 \
    -i dmsq 1 sinsq2thme 0.01 --log glob1.log global
```

Outputs:

* `TUT_glob1_global_fit.txt` — best-fit χ² and every parameter value in a plain-text table
* `TUT_glob1_PROglobal_hists.pdf` — pre-fit (blue/gray) vs post-fit (red) spectra + error bands, with a data/fit ratio panel
* `TUT_glob1_PROglobal_postfit_correlation_matrix.pdf` (+ `_nuisance_only` version) — post-fit parameter correlations
* `TUT_glob1_PROglobal_postfit_posteriors.pdf` — post-fit parameter constraints
* `TUT_glob1_PROglobal.root` — all of the above as ROOT objects

<img src="figures/TUT_glob1_PROglobal_hists.png" width="800"/>

*`TUT_glob1_PROglobal_hists.pdf` — pre-fit vs post-fit spectra with ratio panel — your at-a-glance goodness-of-fit.*

<img src="figures/TUT_glob1_PROglobal_postfit_correlation_matrix.png" width="600"/>

*`TUT_glob1_PROglobal_postfit_correlation_matrix.pdf` — post-fit parameter correlations.*

<img src="figures/TUT_glob1_PROglobal_postfit_posteriors.png" width="700"/>

*`TUT_glob1_PROglobal_postfit_posteriors.pdf` (representative pages: CrossSection1, DetSys1, Flux1, FiducialVol_FD) — post-fit parameter constraints, one parameter per page.*

The text output looks like:

```
################################################
########### Global Best Fit Results ############
################################################
Global Best Fit chi^2: <...>
at paramters:
#Deltam^{2}          :  <log10 value>
sin^{2}2#theta_{#mue}:  <log10 value>
Flux1                :  <sigma>
...
```

Remember the physics values are printed in **log10 space** for this model.

### Where the time goes

The minimization itself is usually the *fast* part of `global`. The **pre-fit
error band** is built by throwing a few thousand spectra from the priors and
covariances (random — fix `--seed`, and bump the sample count for
publication-quality plots, since sparsely-filled bins jitter). The **post-fit
error band, correlation matrix, and posteriors** come from an MCMC sampled
around the best fit (tens of thousands of samples after burn-in): post-fit
pulls can be strongly correlated, so marginal errors must be sampled
simultaneously rather than read off one parameter at a time. This MCMC
covariance is also the *trusted* post-fit covariance — the L-BFGS internal
Hessian is only a limited-memory approximation. Everything the fitter used
(preset values, seeds, exceptions) is printed to the log, so past fits stay
auditable.

### How the fit actually works

Every fit in PROfit is a three-stage pipeline:

1. **Latin hypercube sampling** (`n_latin_points`): random-but-space-filling
   starting points across all parameters, filtered for diversity
   (`latin_diversity_factor`).
2. **Particle swarm optimization** (`n_swarm_particles`,
   `n_swarm_iterations`): the best hypercube points explore globally.
3. **L-BFGS-B local fits** (`n_localfit`): full gradient-based minimizations
   from the best swarm point and each seed point; best result wins.

After the global fit, PROfit runs a **harmonic seed search** over Δm² (the χ²
is quasi-periodic in log Δm², so degenerate local minima are found by a
frequency scan) — those seeds are handed to every subsequent profile/surface
fit, which is a big part of why scans are robust to the multi-modal
oscillation landscape.

### Fit presets and fit options

There are two independent fitter configurations: the **global** fit (done
once — be careful) and the **scan** fit (done thousands of times in
profile/surface — be fast). Configure them with:

```bash
--preset fast              # ONE value sets both global and scan config
--preset good overkill     # first = global, second = scan
--fit-options n_latin_points 2000 max_iterations 5000     # global fit knobs
--scan-fit-options n_localfit 2 ...                        # scan fit knobs
```

Presets are `fast`, `good` (default), `overkill`, and `sensitivity`.

> ⚠️ **CLI11 trap:** if you pass a single preset (`--preset fast`) make sure
> the *next* token is a flag or the subcommand is protected — greedy vector
> parsing can eat a following bare word. `--preset fast --seed 405 ... global`
> is safe; when in doubt put `--preset` before other options.

Run `PROfit --fit-help -x anything` for the full annotated parameter list; the
highlights:

```
------ PROfitter Specific Parameters ------
  n_latin_points          : Number of Latin hypercube points sampled across all parameters
  latin_diversity_factor  : 0 = no distance weighting, 1 = most diverse points
  n_localfit              : Total number of L-BFGS-B fits after PSO
  n_max_local_retries     : Retries if L-BFGS-B throws
------ Particle Swarm Optimization ------
  n_swarm_particles, n_swarm_iterations, n_swarm_max_stagnent_iterations,
  swarm_inertia_start/end, swarm_cognitive_score, swarm_social_score, ...
------ Harmonic Seed Search ------
  harmonic_min/max_num_seeds, harmonic_num_test_points, ...
------ L-BFGS-B ------
  m, epsilon, epsilon_rel, past, delta, max_iterations, max_linesearch, ...
```

(L-BFGS-B parameter meanings: see the
[optimizer wiki page](https://github.com/markrosslonergan/Elephant_Vanishes/wiki/L%E2%80%90BFGS%E2%80%90B-Optimizer-Parameter-Descriptions).)

One benign scary-looking thing you WILL see in logs: L-BFGS-B throwing
`"line search step became smaller than minimum"` on near-optimal starting
points. This is routine (the seed's own χ² is kept as a candidate), not a
failure.

### Metric choice matters

```bash
PROfit -x tutorial.xml -t TUT -o globcnp --seed 405 -n 8 -c PROCNP global
PROfit -x tutorial.xml -t TUT -o globpoi --seed 405 -n 8 -c Poisson global
```

`PROchi` is the classic Pearson χ² (statistical + systematic covariance plus
Gaussian pulls); `PROCNP` swaps the statistical variance for the combined
Neyman-Pearson form `3/(1/d + 2/μ)` ([X. Ji et al.](https://arxiv.org/pdf/1903.07185))
and is the **recommended** choice whenever bins can be low-statistics;
`Poisson` is the Baker-Cousins likelihood-ratio sum and ignores
covariance-type systematics entirely (it will warn you). Adding a new metric
is deliberately easy — they all implement the same `PROmetric` interface.

---

# 7. Subcommand `profile` — 1D profiled Δχ²

```
Usage: PROfit profile [OPTIONS]
Options:
  --mcmc-prefit        Use MCMC to sample the systematic priors for the pre-fit error band.
  --probe              Use PRObe adaptive importance sampling instead of the legacy 18-uniform scan.
  --probe-chunks INT   With --probe, split each physics scan into N parallel chunks.
  --profile-timing     Emit a scan-timing summary (diagnostic).
```

`profile` first runs the full global fit (identical to `global`), then scans
**every parameter one at a time** — each physics parameter and each spline —
profiling (re-minimizing) over all the others at each scan point. Do the
arithmetic before you hit enter: the legacy scan is ~18 points per parameter,
so this config's 2 physics + 14 nuisance parameters mean roughly **300 full
minimizations** on top of the global fit. This is where `-n` starts to
matter (the harmonic seeds and the cross-thread warm-start bank are what
keep each individual scan fit cheap).

```bash
PROfit -x tutorial.xml -t TUT -o prof1 --seed 405 -n 8 \
    -i dmsq 1 sinsq2thme 0.01 --log prof1.log profile
```

Outputs:

* `TUT_prof1_PROfile.pdf` — the Δχ² curve for every parameter (physics first, then all nuisances with the prior in dashed red)
* `TUT_prof1_PROfile_1sigma.pdf` (+ `_1sigma_detailed.pdf`) — post-fit ±1σ summary for all nuisance parameters at a glance (red star = injected truth, blue = global best fit, black = the per-profile minima on the scan grid)
* `TUT_prof1_PROfile_hists.pdf` — pre-fit vs post-fit spectra (same as global's)
* `TUT_prof1_PROfile_postfit_correlation_matrix.pdf`, `_postfit_posteriors.pdf`
* `TUT_prof1_PROfile.root` — every profile as a TGraph, plus the 1σ summary
* `TUT_prof1_global_fit.txt` — the global best fit table

<img src="figures/TUT_prof1_PROfile.png" width="700"/>

*`TUT_prof1_PROfile.pdf` — per-parameter profiled Δχ²; top row = physics, rest = nuisances vs their priors.*

<img src="figures/TUT_prof1_PROfile_1sigma.png" width="800"/>

*`TUT_prof1_PROfile_1sigma.pdf` — ±1σ nuisance summary; star = injected, black = best fit.*

<img src="figures/TUT_prof1_PROfile_hists.png" width="800"/>

*`TUT_prof1_PROfile_hists.pdf` — pre-fit vs post-fit spectra.*

How to read `PROfile.pdf`: each nuisance panel shows the profiled Δχ²
(black) against the dashed-red prior (a 1σ Gaussian pull by construction),
with the dotted Δχ²=1 line marking the "1σ error". Black narrower than red
means the data **constrains** that systematic beyond its prior; black on top
of red means you are effectively **insensitive** to it (an injected shift in
such a systematic will simply not be recovered — that's physics, not a fit
failure); a shifted minimum means the fit **pulled** it (watch for
correlated pairs sharing a pull, e.g. two normalisations covering for each
other). An occasional "kink" in a black curve is the scan fit landing in a
local minimum (usually a Δm² harmonic) at one point — a stronger preset
generally smooths it out.

Two benign log messages you may notice: the routine L-BFGS-B line-search
throws mentioned earlier, and a `Warning. A lower global best fit was found
during PROfile...`. The scan runs hundreds more fits than `global` did, so
occasionally it stumbles on a marginally lower χ² in degenerate phase space
— tiny differences are expected, large ones mean your global preset is too
weak.

### PRObe: the adaptive scan

The legacy scan evaluates a fixed 18-point grid per parameter. `--probe`
replaces this with PRObe, an adaptive algorithm that anchors the minimum,
fits a quadratic surrogate, and bisects to the Δχ²=1 crossings directly —
usually fewer fits *and* more accurate 1σ bounds, especially for strongly
constrained nuisances. For physics parameters (which can be spiky in Δm²) it
uses a spike-safe coarse grid + refinement instead.

```bash
PROfit -x tutorial.xml -t TUT -o probe1 --seed 405 -n 8 \
    -i dmsq 1 sinsq2thme 0.01 profile --probe
```

<img src="figures/TUT_probe1_PROfile.png" width="700"/>

*`TUT_probe1_PROfile.pdf` — PRObe version — compare point placement with the legacy scan above.*

If the two physics-parameter scans are your wall-time bottleneck and you have
threads to spare, `--probe-chunks 4` splits each physics scan across threads
(the cross-thread seed bank keeps the chunks cooperating).

### Nuisance-only profiling

`--syst-only` fixes the physics at CV and profiles just the nuisances —
useful for constraint studies:

```bash
PROfit -x tutorial.xml -t TUT -o profso --seed 405 -n 8 --syst-only profile --probe
```

---

# 8. Subcommand `surface` — 2D Wilks surfaces and AMR

`surface` maps Δχ² over a 2D grid of two physics parameters, profiling over
everything else at each point. Contours at Wilks-theorem critical values
(Δχ²=2.30/5.99/... for 2 dof) give your confidence regions — see section 8
for when Wilks isn't good enough.

Selected options (run `PROfit ... surface --help` for all):

```
-g,--grid INT [40]        Grid size (one value = square, two = rectangular)
--xvar / --yvar           Which physics parameters on which axes
--xlo --xhi --ylo --yhi   Axis ranges (or --xlims/--ylims)
--logx/--linx --logy/--liny   Axis scaling (default log)
--xlabel --ylabel         Axis labels
--brazil-band             1000 stats+systs throws → median ±1σ/±2σ sensitivity bands
  --stat-throws / --single-throw / --only-throw / --from-many
--curve-mode FLOAT ...    1D PROcurve from param A to param B
--surface-amr             Adaptive mesh refinement instead of the dense grid
  --amr-initial INT [10]  AMR coarsest grid (NxN)
  --amr-levels INT [3]    Refinement depth (resolution ≈ initial * 2^levels)
  --amr-delta FLOAT [0.5] Straddle-band widening in chi^2 units
  --amr-levels-chi2 ...   Target Delta-chi^2 contours (default 5.99)
```

**For the `nueapp` model you must set the axes** — the built-in defaults are
for the νμ-disappearance model:

```bash
AXES="--xvar sinsq2thme --yvar dmsq --xlo 1e-4 --xhi 1 --ylo 1e-2 --yhi 1e2"
```

### A dense Asimov sensitivity surface

```bash
PROfit -x tutorial.xml -t TUT -o surf1 --seed 405 -n 8 --log surf1.log \
    surface -g 30 $AXES
```

This is 900 profiled fits, so it's the first genuinely slow thing in the
tutorial — grid size and thread count are your levers. Outputs:

* `TUT_surf1_surface.txt` — plain-text grid. The header records the grid
  dimensions, the `Fixed indices` (which parameter-vector entries are the y
  and x axes), and the full parameter order; then one row per grid point:
  `xval yval chi2 p0 p1 ...` with the best fit of **every** parameter at that
  point. Everything needed to reconstruct the surface externally.
* `TUT_surf1_surf.root` — the Δχ² TH2D plus a `BestFitTree` whose
  `map<string,float>` branch holds the per-point best fits (heads-up:
  `TTree::Draw` on such split-object leaves returns garbage — read the branch
  objects properly)
* `TUT_surf1_surface.pdf` — a quick-look contour plot
* `TUT_surf1_global_fit.txt` — the global best fit

<img src="figures/TUT_surf1_surface.png" width="600"/>

*`TUT_surf1_surface.pdf` — Asimov sensitivity Δχ² surface, 30×30 grid.*

We stress: PROfit's job is to give you the *surface data*; ROOT is not the
place to make pretty contour plots. The `.txt` output loads trivially into
matplotlib — see `working_dir/Tutorial_V2.0.1_allincl/PROfit_Contour_plotter.ipynb`
for a ready-made notebook, and everything you learned in the old v1 tutorial
about overlaying `--syst-list` / `--exclude-systs` / `--statonly` variants
still applies:

```bash
PROfit -x tutorial.xml -t TUT -o surfstat --seed 405 -n 8 --statonly surface -g 30 $AXES
PROfit -x tutorial.xml -t TUT -o surfnoflux --seed 405 -n 8 --exclude-systs Flux1 Flux2 Flux3 surface -g 30 $AXES
```

### PROcurve: watching the pulls along a 1D path

`--curve-mode param1start param2start param1end param2end` (values in the axes' native — here log10 — space)
replaces the 2D scan with a 1D walk from point A to point B across the
(param1, param2) plane, fitting the nuisances at each step and plotting how every pull
evolves along the path. It's the quickest way to see *which* systematics
bend to absorb an oscillation signal as you approach it. `-g` sets the
number of points on the path.

```bash
PROfit -x tutorial.xml -t TUT -o curve1 --seed 405 -n 8 \
    surface $AXES -g 20 --curve-mode -1 -3 1 -1
```

<img src="figures/TUT_curve1_PROcurve.png" width="800"/>

*`TUT_curve1_PROcurve.pdf` — path across the plane + every nuisance parameter's best-fit value along it, here from (Δm², sin²2θμe) = (0.1 eV², 10⁻³) to (10 eV², 10⁻¹).*

### Adaptive mesh refinement: `--surface-amr`

A dense grid wastes almost all its fits far from the contour you care about.
`--surface-amr` starts from a coarse grid and recursively refines only the
cells that straddle (within `--amr-delta`) one of your target Δχ² levels:

```bash
PROfit -x tutorial.xml -t TUT -o surfamr --seed 405 -n 8 --log surfamr.log \
    surface $AXES --surface-amr --amr-initial 10 --amr-levels 3 \
    --amr-levels-chi2 2.30 5.99
```

Effective resolution along the contour is `amr_initial × 2^amr_levels`
(here 80×80) for roughly the cost of the coarse grid plus a band around the
contours — typically a **6-8× wall-time win** at equivalent contour quality.
The scan writes `TUT_surfamr_surface_amr.txt` (same column format, one row
per evaluated mesh point) alongside the usual `_surf.root` / `_surface.pdf`,
plus a `TUT_surfamr_amr_mesh.pdf` mesh diagnostic.

<img src="figures/TUT_surfamr_surface.png" width="600"/>

*`TUT_surfamr_surface.pdf` — AMR surface; note the refined mesh hugging the 1σ/2σ contours.*

<img src="figures/TUT_surfamr_amr_mesh.png" width="600"/>

*`TUT_surfamr_amr_mesh.pdf` — the mesh itself: cells refined only along the two
target Δχ² contours (overlaid curves), with per-level fit counts. Here the
80×80-equivalent contour resolution cost 1067 fits instead of 6400.*

### Brazil bands

`--brazil-band` repeats the sensitivity estimate over 1000 statistical +
systematic throws of the data and draws the median and ±1σ/±2σ envelope of
the resulting contours — the classic sensitivity band. This is ~1000× the
cost of one surface, so use AMR, threads, and patience (or `--stat-throws`
for stat-only bands, `--single-throw` to test the machinery, and
`--from-many file1 file2 ...` to merge throw files from separate jobs run in
parallel on a cluster).

```bash
PROfit -x tutorial.xml -t TUT -o surfbrz --seed 405 -n 16 --log surfbrz.log \
    surface $AXES --surface-amr --amr-initial 10 --amr-levels 2 --brazil-band
```

This command does not make a Brazil band itself, but outputs information that could be used to make a Brazil band externally.


### A note on models: parameterize in the variable you plot

The scan axes are the *model's* parameters — so pick the model whose
parameters ARE the plot you want. The naive way to make a
(sin²2θμe, Δm²) contour from a full 3+1 fit is a 3D grid in
(|Ue4|², |Uμ4|², Δm²) projected down afterwards, which is both brutally
expensive (a 60³ grid is days of CPU where 60² is hours) and messy: many 3D
points map to the same sin²2θμe, the grid spacing doesn't map smoothly, and
for FC you risk the global best fit not lying on your grid at all — which
invalidates the calibration. That's why `PROmodel.h` carries re-parameterized
3+1 variants (`3+1_angles` and effective-amplitude parameterizations like the
sin²2θμe-focused one): the plotted quantity becomes a direct 2D scan axis and
PROfit's minimizer profiles smoothly over the leftover mixing freedom at each
point. A full 3D scan is still the right tool for *understanding* 3+1 — but
if the end goal is a contour in an effective angle, fit in that angle.

---

# 9. Feldman-Cousins: `fc` and `fc-adaptive`

Wilks' theorem (Δχ² cuts of 2.30/5.99/...) assumes Gaussian-land: no physical
boundaries, no degenerate minima. Oscillation fits violate both, so for
publication contours you calibrate the Δχ² cut empirically with
Feldman-Cousins pseudo-experiments.

### Classic FC: `fc`

```
Usage: PROfit fc [OPTIONS]
Options:
  -u,--universes UINT [1000]  Number of Feldman Cousins universes to throw
  --gof                       Get GOF pvalue
  --pval                      Get FC pvalue
```

At the injected point (`-i`, or CV if none), `fc` throws `-u` universes —
each one a full pseudo-experiment: Gaussian spline pulls sampled from the
priors, covariance-systematic shifts via Cholesky decomposition, and Poisson
statistics — and for each universe runs both a nuisance-only fit and a free
fit. The distribution of `Δχ² = χ²(fixed) - χ²(free)` is your empirical
critical-value distribution at that point.

```bash
PROfit -x tutorial.xml -t TUT -o fc1 --seed 405 -n 8 \
    -i dmsq 1 sinsq2thme 0.01 --log fc1.log fc -u 500
```

Output is `TUT_fc1_FC.root` containing a TTree with, per universe, the two
χ² values and the best-fit parameters — from which you extract the
90%/95% quantiles and compare to the Wilks values. This is the honest but
brute-force approach: to calibrate a whole *contour* you would repeat it at
every grid point, which is exactly what `fc-adaptive` automates.

### Adaptive FC: `fc-adaptive`

The adaptive-FC pipeline concentrates pseudo-experiments where they matter —
near the contour — instead of uniformly across the plane. It runs in stages
that communicate through binary artifacts, **all keyed by the `-o` output
tag**, so every stage of one study must share the same `-o`:

```
build-mesh  →  <tag>_<out>_mesh.bin      (Wilks prepass: N throws, each an AMR mesh;
                                          cells that many throws refine form the meta-mesh)
init-bank   →  <tag>_<out>_bank.bin      (pseudo-experiment bank: PEs per meta-mesh cell,
                                          doubling with refinement level; re-running ADDS PEs)
print-bank  →  summary PDFs               (bank occupancy diagnostics)
print-mesh  →  mesh PDFs                  (<tag>_mesh.bin, or any mesh files via --merge-input)
asimov      →  FC contour + verdict PDFs  (classify the Asimov data against the bank)
brazil      →  Brazil-band PDFs           (throw pseudo-data, classify each against the bank)
merge-mesh  →  <tag>_<out>_mesh.bin       (union-merge mesh binaries from separate runs)
merge-bank  →  <tag>_<out>_bank.bin       (harvest PEs from separate bank binaries onto that mesh)
brazil-cleanup → <tag>_<out>_cleanup_mesh.bin  (mesh densified at the Brazil ±2σ contours; no fits)
```

A full small-scale run (bump `--throws` and `--n-pe-min` for real studies):

```bash
AXES="--xvar sinsq2thme --yvar dmsq --xlo 1e-4 --xhi 1 --ylo 1e-2 --yhi 1e2"
AFC="fc-adaptive --throws 25 --prepass-amr-initial 8 8 --prepass-amr-levels 2 $AXES"

# Stage 1: Wilks prepass → meta-mesh
PROfit -x tutorial.xml -t TUT -o afc --seed 405 -n 8 $AFC --mode build-mesh
# Stage 2: fill the PE bank (repeat to add more PEs, capped at --n-pe-max)
PROfit -x tutorial.xml -t TUT -o afc --seed 405 -n 8 $AFC --mode init-bank --n-pe-min 25 --n-pe-max 400
# Stage 3: inspect
PROfit -x tutorial.xml -t TUT -o afc --seed 405 -n 8 $AFC --mode print-bank
# Stage 4: FC-corrected Asimov contour
PROfit -x tutorial.xml -t TUT -o afc --seed 405 -n 8 $AFC --mode asimov
# Stage 5: FC-corrected Brazil band
PROfit -x tutorial.xml -t TUT -o afc --seed 405 -n 8 $AFC --mode brazil --n-brazil-throws 50
```

Key knobs: `--p-thresh` (fraction of prepass throws that must refine a cell
for it to enter the meta-mesh), `--baseline-level` (levels always kept),
`--cl` (target CLs), `--update-layer` / `--update-only-layer` (target which
refinement layers get new PEs on an `init-bank` re-run), `--stat-only-throws`.

### Merging meshes and banks: `merge-mesh` and `merge-bank`

The PE bank is by far the most expensive artifact in the pipeline, so PROfit
can combine meshes and banks produced by *separate* runs — different `-o`
tags, different machines, different days — into one study. `--merge-input`
takes explicit filenames or (quoted) glob patterns; the merged artifacts are
written under the normal `-t`/`-o` naming, and every downstream stage
(`print-bank`, `asimov`, `brazil`, `init-bank` top-up) consumes them as if
they had been produced in one run.

```bash
# Union-merge two meshes (wherever the inputs disagree, the finer tiling wins)
PROfit -x tutorial.xml -t TUT -o merged --seed 405 -n 8 $AFC \
    --mode merge-mesh --merge-input TUT_runA_mesh.bin TUT_runB_mesh.bin

# Harvest the PEs from both banks onto the merged mesh
PROfit -x tutorial.xml -t TUT -o merged --seed 405 -n 8 $AFC \
    --mode merge-bank --merge-input 'TUT_run[AB]_bank.bin'

# Top up: every cell to 400 PEs (cells already at/above 400 are left alone)
PROfit -x tutorial.xml -t TUT -o merged --seed 405 -n 8 $AFC \
    --mode init-bank --n-pe-min 400 --n-pe-max 400
```

**Where this is useful:**

1. **Grid / cluster PE production.** Build the mesh once, ship it to N jobs,
   run `init-bank` in each job under its own `-o` tag **with a distinct
   `--seed`**, copy the banks back, and merge (the shipping half is what
   the [`proletariat` subcommand](#10-proletariat--grid-submission-proletariat)
   automates):

   ```bash
   # one job, i = 0..N-1  (each job has its own copy of the shared mesh
   # as TUT_grid<i>_mesh.bin — identical copies merge to themselves)
   PROfit -x tutorial.xml -t TUT -o grid$i --seed $((1000+i)) -n 8 $AFC \
       --mode init-bank --n-pe-min 100 --n-pe-max 5000

   # back home:
   PROfit ... -o merged $AFC --mode merge-mesh --merge-input 'TUT_grid*_mesh.bin'
   PROfit ... -o merged $AFC --mode merge-bank --merge-input 'TUT_grid*_bank.bin'
   ```

2. **Updating the mesh without losing the bank.** Re-running `build-mesh`
   with more throws or different `--p-thresh` changes the meta-mesh, and a
   plain `init-bank` would then discard the old bank as footprint-mismatched.
   Instead: `merge-mesh` the old and new meshes, `merge-bank` the old bank
   onto the union, then `init-bank` to generate PEs only where the mesh
   actually changed. Every cell whose footprint survived keeps its PEs.

3. **Combining independent studies of the same configuration** — e.g. a
   coarse exploratory bank and a later refined one.

**Rules and gotchas:**

* PEs are Δχ² samples thrown at a cell **center**, so `merge-bank` carries a
  cell over only when the merged mesh has the *exact same footprint*
  (same position and size). PEs from cells that changed refinement are at the
  wrong truth points and are dropped — the log reports carried / dropped /
  still-empty counts per input.
* All inputs must share the finest grid resolution and axis bounds
  (`--prepass-amr-initial`, `--prepass-amr-levels`, `--xlo/xhi/ylo/yhi`);
  mismatches are refused loudly.
* Bitwise-identical PEs (same per-PE seed *and* Δχ²) are deduplicated, so
  merging two banks generated with the same `--seed` silently costs you the
  overlap — you get a warning, but the wasted CPU already happened. Give
  every grid job a distinct seed.
* The binaries carry no XML/fit-config provenance, so the merge **cannot**
  verify that inputs used the same XML, χ² metric (`-c`), `--preset`, and
  `--grad-mode`. Mixing those pools different Δχ² distributions — that
  discipline is on you.
* The brazil archive (`_brazil.bin`) is *not* merged: throws are cheap
  relative to banks and are simply re-thrown against the merged bank.

### Sharpening the band edges: `brazil-cleanup`

The Wilks prepass refines the mesh around the *Asimov* contour, but the
Brazil ±2σ band edges — the P(included) = 0.025 and 0.975 contours of the
throw ensemble — spread beyond it, often into coarse baseline cells where
the band looks blocky. `brazil-cleanup` closes that loop: it reads the
bank (grid geometry) and the **saved contour curves in
`<tag>_<out>_brazil.root`** (**no fits, no throws, nothing recomputed — it
runs in seconds**), rasterizes the requested quantile curves
(`--cleanup-quantiles`, default `0.025 0.975`; any of the five saved levels
0.025/0.16/0.5/0.84/0.975 works) onto the finest grid, and writes
`<tag>_<out>_cleanup_mesh.bin`: finest cells along the curves ±
`--cleanup-halo` bins (default 1), coarsest tiling elsewhere. Because the
curves come from the brazil ROOT artifact itself, the refined region is
*by construction* the same contours the band PDF drew — run it on the
`_brazil.root` from the same brazil invocation as the band you are looking
at. It shares the finest grid with the original mesh by construction,
so it union-merges with it:

```bash
# 1. band-edge refinement mesh from the finished brazil run (tag "afc")
PROfit -x tutorial.xml -t TUT -o afc --seed 405 -n 8 $AFC --mode brazil-cleanup

# 2. union with the original mesh, harvest the original bank, top up, re-throw
PROfit -x tutorial.xml -t TUT -o afc2 --seed 405 -n 8 $AFC \
    --mode merge-mesh --merge-input TUT_afc_mesh.bin TUT_afc_cleanup_mesh.bin
PROfit -x tutorial.xml -t TUT -o afc2 --seed 405 -n 8 $AFC \
    --mode merge-bank --merge-input TUT_afc_bank.bin
PROfit -x tutorial.xml -t TUT -o afc2 --seed 406 -n 8 $AFC \
    --mode init-bank --n-pe-min 25 --n-pe-max 400
PROfit -x tutorial.xml -t TUT -o afc2 --seed 405 -n 8 $AFC \
    --mode brazil --n-brazil-throws 50
```

Every PE from the original bank survives wherever the cell footprint is
unchanged (everywhere except the newly refined band-edge cells), so the
top-up only pays for the new fine cells. Iterate if the edges are still
coarse. To eyeball any mesh along the way (the cleanup mesh, the merged
mesh, ...): `--mode print-mesh --merge-input TUT_afc_cleanup_mesh.bin`
renders each given file as `<same name>.pdf`; with no `--merge-input` it
plots the current tag's `_mesh.bin`. A decided cell sitting inside the band whose neighbour is
*undecidable* is also refined — that is exactly where the contour runs off
into unsampled territory, and the top-up is what makes it decidable.

Outputs along the way:

<img src="figures/TUT_afc_metamesh.png" width="600"/>
*`TUT_afc_metamesh.pdf` — the meta-mesh: cell refinement levels, concentrated where throws put the contour.*

<img src="figures/TUT_afc_throws.png" width="600"/>
*`TUT_afc_throws.pdf` — the Wilks-prepass throw contours that built the mesh.*

<img src="figures/TUT_afc_bank_summary.png" width="600"/>
*`TUT_afc_bank_summary.pdf` — PE bank occupancy per level.*

<img src="figures/TUT_afc_asimov_contour.png" width="600"/>
*`TUT_afc_asimov_contour.pdf` — FC-corrected contour vs the Wilks contour.*

<img src="figures/TUT_afc_asimov_verdict.png" width="600"/>
*`TUT_afc_asimov_verdict.pdf` — per-cell FC vs Wilks verdict map.*

<img src="figures/TUT_afc_brazil_band.png" width="600"/>
*`TUT_afc_brazil_band.pdf` — FC-corrected Brazil band from the bank.*

Determinism note: with `-n 1` and a fixed `--seed` the entire pipeline is
bit-reproducible; multithreaded runs are statistically equivalent.

---

# 10. PROjector — two-stage pre-fit / projected fits

PROjector answers "what does my near detector buy me?" properly. Instead of
fitting ND and FD simultaneously every time, you (1) fit **only** the ND
channels once and save the nuisance posterior, then (2) run any FD study with
those channels masked out and the saved posterior installed as a correlated
prior. Same statistical content as the joint fit (with a Gaussian
approximation and a no-near-detector-oscillation approximation), at a fraction 
of the per-fit cost — which matters enormously for FC studies.

### Stage 1: the pre-fit

```bash
PROfit -x tutorial.xml -t TUT -o pj --seed 405 -n 8 \
    --projector-prefit "_ND_" global
# → TUT_pj_PROjector_constraint.bin
```

What happens: every subchannel matching `_ND_` (unanchored regex — substrings work — and matches must
cover **whole channels** — χ² lives in collapsed space) is selected; all
covariance-type systematics are *promoted* to their eigenmode splines so the
fit has explicit parameters for them (`--projector-knobs N` limits to the top
N modes + a residual covariance; the default -1 keeps all modes, exact);
physics is fixed at CV; and the fit's posterior — best-fit point θ̂ plus the
full correlated covariance Σ from a finite-difference Hessian — is written to
the constraint file.

Useful stage-1 options:

* `--projector-knobs 20` — promote only the leading 20 eigenmodes per covariance (remainder stays as a residual covariance)
* `--projector-keep-cov MCStat DetSys...` — covariances to *not* promote (e.g. detector-local systematics with no ND/FD correlation; MC-stat is never promoted)
* `--projector-float-physics` — float physics in the pre-fit; the saved posterior is then the physics-marginalized nuisance covariance

### Stage 2: the projected fit

Any subcommand — `global`, `profile`, `surface`, `fc`, `fc-adaptive` — can
run projected:

```bash
PROfit -x tutorial.xml -t TUT -o pjglob --seed 405 -n 8 \
    --projector TUT_pj_PROjector_constraint.bin global

PROfit -x tutorial.xml -t TUT -o pjprof --seed 405 -n 8 \
    --projector TUT_pj_PROjector_constraint.bin profile --probe

PROfit -x tutorial.xml -t TUT -o pjsurf --seed 405 -n 8 \
    --projector TUT_pj_PROjector_constraint.bin \
    surface --surface-amr --amr-initial 10 --amr-levels 3 \
    --xvar sinsq2thme --yvar dmsq --xlo 1e-4 --xhi 1 --ylo 1e-2 --yhi 1e2
```

Stage 2 re-derives the identical eigenmode promotion (name-checked against
the constraint file), masks the pre-fit channels OUT of the χ² (active-bins
mask + zeroed data), and installs (θ̂, Σ) as a fully correlated Gaussian prior
on the promoted spline parameters.

<img src="figures/TUT_pjprof_PROfile_1sigma.png" width="800"/>

*`TUT_pjprof_PROfile_1sigma.pdf` — projected nuisance constraints — compare against the joint-fit `TUT_prof1_PROfile_1sigma.pdf`.*

<img src="figures/TUT_pjsurf_surface.png" width="800"/>

*`TUT_pjsurf_surface.pdf` — projected FD-only sensitivity with the ND constraint as prior, vs the joint surface `TUT_surfamr_surface.pdf`.*

### Rules and closure checks

* **Same XML, same binaries, same systematic selection in both stages** —
  hash-enforced; the constraint file also records the χ² metric (`-c`) and
  refuses a mismatch.
* The pattern must cover whole channels: `--projector-prefit fullosc` (one
  subchannel) or `nu_` (everything) are refused loudly. Detector patterns
  like `"_ND_"` are the intended use.
* Closure signatures worth checking on any new setup: a projected Asimov
  `global` fit gives χ² = 0 at CV; a pre-fit with only FD-blind systematics
  gives posterior widths of exactly 1.
* FC/Brazil throws in projected mode sample the *marginal* widths of the
  constraint only — correlations enter the pull term, not the throws (PROfit
  prints a runtime warning to remind you).

Note that autocorrelation and harmonic scan plots will not be filled properly with the projector option.

---

# 11. PROletariat — grid submission: `proletariat`

Everything above runs on one machine. For the artifacts that are genuinely
expensive at scale — FC PE banks above all — the workflow is: ship PROfit and
its inputs to N FermiGrid jobs, run a worker script in each, copy the outputs
back, and merge (section 8). The shipping half used to be a hand-maintained
shell script (`grid/maketar_submit_v2.4.sh`, now deprecated); it is now the
`proletariat` subcommand, implemented by the `PROletariat` class
(`inc/PROletariat.h`).

`proletariat` does three things:

1. **Stages** a fresh tarball: the running PROfit binary itself (located via
   `/proc/self/exe` — the exact executable you invoked is what ships), your
   `-x` XML, any analysis artifacts it finds in the current directory
   (see below), and any `--input` extras, all under a directory literally
   named `grid_dir/`.
2. **Tars** it to `grid_dir.tar` in the current directory and prints the
   contents.
3. **Submits** N copies of your worker script with `jobsub_submit`, attaching
   the tarball via the dropbox mechanism. On the worker node, jobsub unpacks
   it at `$INPUT_TAR_DIR_LOCAL/grid_dir/`.

Unlike every other subcommand, `proletariat` dispatches *before* the XML is
parsed or any binaries are loaded — it only needs the file paths and tags, so
it runs in seconds even for heavy configurations.

### Quick start

```bash
# from the directory holding your XML and TUT_prop.bin / TUT_syst.bin:
PROfit -x tutorial.xml -t TUT proletariat \
    -N 500 --script ../../grid/runFC_v2.4_v4_AL9.sh \
    --lifetime 2d --memory 4000 --disk 10000 \
    --dry-run          # drop this to actually submit
```

`--dry-run` does the full staging and tarring, prints the exact
`jobsub_submit` command it *would* run (copy-paste ready), and stops —
useful for checking the tarball contents on a dev box that has no jobsub
client. The staging directory is always cleaned up; the tarball is left
behind on purpose.

### The container: AL9 by default, `--sl7` for legacy

Grid jobs run inside an Apptainer/Singularity container, and the OS inside
it decides how the worker script sets up its environment:

* **AL9 (default)** — the `fnal-wn-el9` image; software comes from the CVMFS
  **Spack** distribution (`spack load root@6.28.12` etc.). This matches the
  migrated FermiGrid worker nodes and gpvms.
* **SL7 (legacy)** — pass `--sl7` to submit with the old `fnal-wn-sl7` image
  instead; software comes from CVMFS **UPS** (`setup root v6_28_12 -q ...`).
  Use this only if you need to reproduce an old campaign or your worker
  script predates the migration.

An explicit `--singularity-image <path>` overrides the choice entirely and
is mutually exclusive with `--sl7` (passing both is a parse error). The
reference worker script (below) detects the OS at runtime from
`/etc/os-release` and picks Spack or UPS itself, so the same script works
under either image — the submitter's flag is the only switch you touch.

### What gets bundled

Always (missing = hard error):

* the PROfit binary, staged under the literal name `PROfit` (worker scripts
  invoke `./PROfit`) — override which binary ships with `--profit-bin`;
* the worker script (`--script`);
* the `-x` XML;
* every `--input` file (repeatable).

Automatically, if present in the current directory (missing = an INFO line,
not an error):

* `<tag>_prop.bin` and `<tag>_syst.bin` — the binary caches, so workers skip
  the expensive `process` step;
* `<tag>_<output>_mesh.bin` and `<tag>_<output>_bank.bin` — AFC artifacts
  matching your `-t`/`-o` tags, for `init-bank` grid production.

The tarball layout is flat (`grid_dir/<basename>`), so two inputs with the
same basename from different directories are refused loudly rather than
silently clobbering each other.

### The worker script

The payload is still a shell script you own — `grid/runFC_v2.4_v4_AL9.sh` is
the reference implementation and worth reading in full (the older
`runFC_v2.4_v2.sh` is its UPS/SL7-only predecessor). Its contract:

* inputs appear at `$INPUT_TAR_DIR_LOCAL/grid_dir/`; copy them into
  `$_CONDOR_SCRATCH_DIR` and run there;
* the binary is `./PROfit`; the script sets up ROOT/Boost/etc. from CVMFS
  before touching it — Spack on AL9, UPS on SL7, chosen at runtime from the
  container's `/etc/os-release` — and sanity-checks with `ldd` and
  `./PROfit --help` so "missing library" and "bad physics" fail
  distinguishably;
* `$PROCESS` (0..N-1) is the job's identity, but it **restarts at 0 in every
  submission** — a seed or `-o` tag built from it alone collides across
  batches. Fold in `$CLUSTER` (unique per `jobsub_submit`), e.g.
  `SEED=$(( (CLUSTER % 1000000) * 1000 + PROCESS ))` and
  `-o fc_${CLUSTER}_$((PROCESS+1))`, and run with `-n 1`. Colliding seeds
  are not just statistically dubious: with `-n 1` the duplicate jobs are
  bitwise identical, merge-bank silently dedupes their PEs (section 8), and
  the CPU is wasted; colliding `-o` tags additionally give different
  campaigns identical output *filenames*, which then can't share a
  directory at `--merge-input` time;
* copy outputs back with `ifdh cp` to a `/pnfs` scratch area, namespaced by
  `$CLUSTER` so a resubmission doesn't clobber the last campaign.

### Arguments

| Option | Default | What it does |
|---|---|---|
| `--script` | *(required)* | Worker script executed on each grid node. |
| `-N, --n-jobs` | `2` | Number of jobs (`jobsub_submit -N`). Each sees its own `$PROCESS`. |
| `--lifetime` | `2d` | `--expected-lifetime`; 3d is the FermiGrid ceiling before rejection. |
| `--memory` | `4000` | Requested memory in MB. |
| `--disk` | `10000` | Requested scratch disk in MB. |
| `--input` | — | Extra file(s) for the tarball (repeatable). Missing file = hard error. |
| `--dry-run` | off | Stage + tar + print the jobsub command; do not submit. |
| `--backend` | `jobsub` | Scheduler backend: `jobsub` or `slurm` (SLURM is a stub for now and errors out). |
| `--group` | `sbnd` | Experiment group (`jobsub_submit -G`). |
| `--role` | `Analysis` | `--role`. |
| `--singularity-image` | `fnal-wn-el9:latest` (CVMFS path) | Apptainer/Singularity image the jobs run in. Excludes `--sl7`. |
| `--sl7` | off | Use the legacy `fnal-wn-sl7:latest` image instead of AL9. Excludes `--singularity-image`. |
| `--resource-provides` | `usage_model=DEDICATED,OPPORTUNISTIC,OFFSITE` | Usage model. |
| `--lines` | the three `+FERMIHTC_*` classads | Condor classad `--lines` entries. **Replaces** the defaults when given — to *append*, use `--jobsub-arg` instead. |
| `--jobsub-arg` | — | Raw argument passed to `jobsub_submit` verbatim (repeatable) — the escape hatch for anything not covered above. |
| `--profit-bin` | this executable | Override the PROfit binary to ship. |

The default `--lines` are `+FERMIHTC_AutoRelease=True`,
`+FERMIHTC_GraceMemory=4000`, `+FERMIHTC_GraceLifetime=7200` — auto-release
held jobs with a memory/lifetime grace margin. The submission also pins
`(TARGET.HAS_SINGULARITY=?=true)` as a condor requirement.

### End-to-end: a grid FC bank campaign

```bash
# 1. build the mesh locally (section 8)
PROfit -x tutorial.xml -t TUT -o grid --seed 405 -n 8 $AFC --mode build-mesh

# 2. ship it: TUT_prop.bin, TUT_syst.bin and TUT_grid_mesh.bin are picked up
#    automatically from the cwd; check first with --dry-run
PROfit -x tutorial.xml -t TUT -o grid proletariat \
    -N 500 --script runFC_v2.4_v4_AL9.sh --lifetime 2d

# 3. ...wait; fetch outputs from /pnfs to a local dir...

# 4. merge and continue exactly as in section 8
PROfit -x tutorial.xml -t TUT -o merged $AFC --mode merge-mesh --merge-input 'TUT_fc_*_mesh.bin'
PROfit -x tutorial.xml -t TUT -o merged $AFC --mode merge-bank --merge-input 'TUT_fc_*_bank.bin'
```

### Rules and gotchas

* **Run it where your artifacts live.** Auto-bundling searches the *current
  directory* for `<tag>_prop.bin` etc.; the tarball also lands there. Anything
  elsewhere needs an explicit `--input path/to/file`.
* **The tarball is big.** The PROfit binary alone is ~250 MB; with the syst
  cache a typical tarball is 300–400 MB. The size is logged before
  submission — jobsub's dropbox handles it, but don't be surprised.
* Submission needs a working **jobsub client** (a GPVM, valid token/proxy).
  Everything up to and including `--dry-run` works anywhere.
* **Seeds are your job.** `proletariat` submits N identical scripts; the
  worker script must diversify `--seed`/`-o` from `$PROCESS` **and**
  `$CLUSTER` — `$PROCESS` alone repeats across submissions. Identical
  seeds silently dedupe at merge-bank time (section 8).
* **Match the image to the script.** A worker script that only knows UPS
  (`setup <prod>`) dies during environment setup under the default AL9
  image, and a Spack-only script dies under `--sl7`. The reference script
  detects the OS and handles both; if you maintain your own, either make it
  do the same or submit with the image it expects.
* The old `grid/maketar_submit_v2.4.sh` is kept for reference but
  deprecated; it had a latent bug (the `file://` script URL broke unless the
  script sat in the submission directory) that the subcommand fixes.

---

# Appendix A: regenerating every plot in this tutorial

```bash
# from the repository root, after building:
docs/tutorials/make_tutorial_plots.sh
# heavy extras (Brazil bands, larger FC banks):
RUN_EXPENSIVE=1 docs/tutorials/make_tutorial_plots.sh
```

The script localizes the XML, processes once, and runs every command shown
above with fixed seeds. Outputs land in `docs/tutorials/tutorial_run/` (set
`TUTORIAL_OUTDIR` to change). See the script header for the environment
overrides (`PROFIT_BIN`, `PROFIT_TEST_MCDIR`, `NTHREADS`, `SEED`).

The images embedded in this document are PNG renders of those PDFs (GitHub
markdown cannot display PDFs inline), produced by:

```bash
docs/tutorials/make_tutorial_figures.sh   # PDFs → docs/tutorials/figures/*.png
```

It needs `pdftoppm` (poppler-utils) and `montage` (ImageMagick), picks the
relevant page of each multi-page PDF (e.g. the collapsed correlation page of
`_Covar.pdf`), tiles the per-(detector × channel) pages into 2×2 montages,
and skips any PDF not yet generated — so after running the missing heavy
steps you can re-run it and just uncomment the corresponding `<img>` tags in
the remaining placeholders. Commit `figures/`, not `tutorial_run/` (which
holds ~GB binaries).

---

# Appendix B: available physics models (`inc/PROmodel.h`)

The `<model tag="...">` in your XML selects one of the models below. A few
conventions apply to all of them:

* **Parameter names** (`param_names`) are what you use everywhere on the
  command line: `-i/--inject`, `--inject-cv`, `--fix`, `--xvar/--yvar`.
  Injection values are given in **linear** units (`-i dmsq 1` = 1 eV²).
* **Fit space**: parameters marked *log10* are fitted (and printed in
  best-fit tables) as log10 of the physical value; the bounds and defaults
  below are quoted in linear units for readability.
* **Rules**: `model_rule` on each MC `<branch>` picks which probability
  function weights that branch's events. **Rule 0 is always "no oscillation"
  (weight 1)** — the default for intrinsic/background components.
* **Constraints**: some models carry a `model_constraint` that rejects
  unphysical parameter combinations (e.g. 3+1 unitarity) during the fit.
* All oscillation models need the `<parameter name="L/E" .../>` entry in the
  model block pointing at the true-L/E variable (exception: `numudisTEST`
  and `template_fit`, noted below).

### `nullmodel`

No physics parameters at all — rule 0 only. For pure-systematics setups
(constraint studies, template validation) where nothing oscillates.

### `numudis` — 3+1 νμ disappearance

P(νμ→νμ) = 1 − sin²2θμμ · sin²(1.267 Δm² L/E).

| # | name | meaning | fit space | bounds | default |
|---|---|---|---|---|---|
| 0 | `dmsq` | Δm²₄₁ [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `sinsq2thmm` | sin²2θμμ | log10 | 0 – 1 | 10⁻¹⁰ |

Rules: **0** = no osc, **1** = νμ survival (apply to your νμ-CC signal).

### `numudisTEST` — two-variable validation twin

Identical physics and parameters to `numudis`, but built on two separate
model variables `L` and `E` (both must be `<parameter>` entries) instead of
one `L/E`. Exists to validate the machinery — the two should produce
identical spectra.

### `nueapp` — 3+1 νμ→νe appearance

P(νμ→νe) = sin²2θμe · sin²(1.267 Δm² L/E). The model used throughout this
tutorial.

| # | name | meaning | fit space | bounds | default |
|---|---|---|---|---|---|
| 0 | `dmsq` | Δm²₄₁ [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `sinsq2thme` | sin²2θμe | log10 | 10⁻¹⁰ – 1 | 10⁻¹⁰ |

Rules: **0** = no osc, **1** = νμ→νe appearance (apply to the fullosc
subchannel).

### `nuedis` — 3+1 νe disappearance

| # | name | meaning | fit space | bounds | default |
|---|---|---|---|---|---|
| 0 | `dmsq` | Δm²₄₁ [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `sinsq2thee` | sin²2θee | log10 | 0 – 1 | 10⁻¹⁰ |

Rules: **0** = no osc, **1** = νe survival.

### `3+1` — full three-parameter 3+1 (mixing elements)

The most physical parameterization: all three SBL channels driven
simultaneously by the extended-PMNS elements, with the **unitarity
constraint |Ue4|² + |Uμ4|² < 1** enforced via `model_constraint`.

| # | name | meaning | fit space | bounds | default |
|---|---|---|---|---|---|
| 0 | `dmsq` | Δm²₄₁ [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `Ue4^2` | \|Ue4\|² | log10 | 0 – ~1 | 10⁻⁸ |
| 2 | `Um4^2` | \|Uμ4\|² | log10 | 0 – ~1 | 10⁻⁸ |

Rules: **0** = no osc, **1** = νμ→νμ, **2** = νμ→νe, **3** = νe→νe — so one
XML can simultaneously disappear the νμ-CC sample (rule 1), appear the
fullosc component (rule 2), and disappear the intrinsic νe (rule 3).

### `3+1_angles` — 3+1 in mixing angles

Same physics and rules (0–3) as `3+1`, re-parameterized in angles, with the
unitarity constraint.

| # | name | meaning | fit space | bounds | default |
|---|---|---|---|---|---|
| 0 | `dmsq` | Δm²₄₁ [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `sinsq2th14` | sin²2θ₁₄ | log10 | 0 – ~1 | 10⁻⁸ |
| 2 | `sinsqth24` | sin²θ₂₄ | log10 | 0 – ~1 | 10⁻⁸ |

### `3+1_3A` / `3+1_3B` / `3+1_3C` — effective-amplitude 3+1 variants

Full 3+1 physics (rules **0–3** as above), re-parameterized so that the
*effective amplitude you want to plot* is a direct scan axis and the
leftover mixing freedom is a single nuisance-like third parameter that
PROfit profiles smoothly at each grid point (see the surface-section note on
parameterizations).

**`3+1_3A`** — sin²2θee-focused; ratio parameter is exactly sin²θ₂₄
(= sin²2θμe / sin²2θee). No extra constraint needed.

| # | name | fit space | bounds | default |
|---|---|---|---|---|
| 0 | `dmsq` [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `sinsq2thee` | log10 | 0 – ~1 | 10⁻⁸ |
| 2 | `sinsqth24` | log10 | 0 – ~1 | 10⁻⁸ |

**`3+1_3B`** — sin²2θμμ-focused; `sB` is the ratio
sin²2θμe / sin²2θμμ (= |Ue4|² / (1−|Uμ4|²)). Unitarity constraint enforced.

| # | name | fit space | bounds | default |
|---|---|---|---|---|
| 0 | `dmsq` [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `sinsq2thmumu` | log10 | 0 – ~1 | 10⁻⁸ |
| 2 | `sB` | log10 | 0 – ~1 | 10⁻⁸ |

**`3+1_3C`** — sin²2θμe-focused (the one to use for direct appearance
contours); `xi` is the log geometric ratio ξ = ½·log(|Ue4|²/|Uμ4|²)
(ξ = 0 ⇒ equal mixing, ξ > 0 ⇒ electron-dominated). Unitarity constraint
√(sin²2θμe)·cosh(ξ) < 1.

| # | name | fit space | bounds | default |
|---|---|---|---|---|
| 0 | `dmsq` [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `sinsq2thmue` | log10 | 0 – ~1 | 10⁻⁸ |
| 2 | `xi` | **linear** | −10 – 10 | 0 |

### `3+1_decay_invis` — 3+1 with invisible sterile decay

The `3+1` mixing-element model extended with an invisible-decay coupling;
combined unitarity + positivity constraint. Rules **0–3** as in `3+1`.

| # | name | meaning | fit space | bounds | default |
|---|---|---|---|---|---|
| 0 | `dmsq` | Δm²₄₁ [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `Ue4^2` | \|Ue4\|² | log10 | 0 – ~1 | 10⁻⁸ |
| 2 | `Um4^2` | \|Uμ4\|² | log10 | 0 – ~1 | 10⁻⁸ |
| 3 | `g2` | decay coupling g² | **linear** | 0 – 10 | 0 |

### `3+2` — two-sterile model

Seven parameters with a CP phase, so ν and ν̄ appearance differ. Unitarity
constraints |Ue4|²+|Ue5|² ≤ 1 and |Uμ4|²+|Uμ5|² ≤ 1.

| # | name | meaning | fit space | bounds | default |
|---|---|---|---|---|---|
| 0 | `dmsq41` | Δm²₄₁ [eV²] | log10 | 10⁻² – 10² | 10⁻¹ |
| 1 | `dmsq51` | Δm²₅₁ [eV²] | log10 | 10⁻² – 10² | 1 |
| 2 | `Ue4sq` | \|Ue4\|² | log10 | 10⁻⁵ – ~0.98 | 10⁻⁴ |
| 3 | `Um4sq` | \|Uμ4\|² | log10 | 10⁻⁵ – ~0.98 | 10⁻⁴ |
| 4 | `Ue5sq` | \|Ue5\|² | log10 | 10⁻⁵ – ~0.98 | 10⁻⁴ |
| 5 | `Um5sq` | \|Uμ5\|² | log10 | 10⁻⁵ – ~0.98 | 10⁻⁴ |
| 6 | `phi54` | CP phase φ₅₄ [rad] | **linear** | 0 – 2π | 0 |

Rules: **0** = no osc, **1** = νμ→νμ, **2** = νμ→νe, **3** = νe→νe,
**4** = ν̄μ→ν̄e (the CP-conjugate appearance — give your antineutrino
fullosc branch rule 4).

### `LBL` — full three-flavour long-baseline (NuFastLBL)

Standard 3ν oscillations, all parameters fitted in **linear** space with
bounds spanning the global-fit allowed ranges.

| # | name | meaning | bounds | default |
|---|---|---|---|---|
| 0 | `dmsq_21` | Δm²₂₁ [eV²] | 6×10⁻⁵ – 9×10⁻⁵ | 7.5×10⁻⁵ |
| 1 | `dmsq_31` | Δm²₃₁ [eV²] | −3×10⁻³ – 3×10⁻³ | 10⁻³ |
| 2 | `sinsqt12` | sin²θ₁₂ | 0.2 – 0.4 | 0.3 |
| 3 | `sinsqt13` | sin²θ₁₃ | 0.01 – 0.04 | 0.025 |
| 4 | `sinsqt23` | sin²θ₂₃ | 0.3 – 0.7 | 0.5 |
| 5 | `delta_CP` | δ_CP [rad] | −π – π | 0 |

Rules cover the full 3×3 matrix: **0** = no osc, **1** = Pee, **2** = Peμ,
**3** = Peτ, **4** = Pμe, **5** = Pμμ, **6** = Pμτ, **7** = Pτe,
**8** = Pτμ, **9** = Pττ.

### `template` / `template_fit` — per-subchannel normalization fit

Not an oscillation model: each `<parameter name="...">` in the model block
names a **subchannel** whose normalization floats as one linear scale
parameter (bounds from the parameter's `min`/`max` attributes, default 1);
every non-floated subchannel stays fixed at 1. `model_rule` is ignored —
events are routed by subchannel membership. Needs no L/E variable. Useful
for sideband/template fits and cross-section-style normalization studies.

---

# Appendix C: the pre-fit and post-fit error bands, in full

This appendix is the complete, self-contained recipe for the shaded error
bands PROfit draws around the prediction. There are two of them, and they
answer two different questions:

* the **pre-fit band** answers *"before we look at any data, how uncertain is
  our prediction?"* — it is a picture of the systematic priors, nothing more;
* the **post-fit band** answers *"after the fit, how uncertain is the
  prediction, and where does it actually sit?"* — the data has now constrained
  both the spline nuisances *and* the covariance-encoded systematics, so this
  band is (usually much) narrower, and its center can move.

Below we work out the exact maths in the form the code computes it, name the
function that implements each step, and state every assumption. Section C.4
at the end *derives* the key formulas from scratch, so nothing here needs to
be taken on faith.

### C.0 Notation and shared ingredients

Everything happens in the **collapsed bin space of the fitting variable**
(`config.i_prime`) — the same space the χ² lives in, after subchannels have
been summed into channels. Throughout, write:

```
θ = (φ, s)        the parameter vector: physics φ, then one param per spline s
P(θ)              the collapsed predicted spectrum at θ (FillSpectra + collapse)
θ_CV              the central-value parameters (physics at CV, splines at their centers)
θ̂  = (φ̂, ŝ)      the global best fit
d                 the collapsed data spectrum (Asimov, fake, or real)
```

**Covariance-type systematics** (including MC-stat) are not fit parameters:
their summed *fractional* covariance `F` (`PROsyst::fractional_covariance`)
is folded into the χ² analytically, as described in section 1. The band
machinery needs two things built from `F` once, at a reference spectrum
(`PROsyst::DecomposeFractionalCovariance`): the *absolute* covariance, and a
"square root" of it that turns unit Gaussian random numbers into correlated
spectrum fluctuations. Note that `F` lives in the *uncollapsed* bin space, so
the absolute covariance is built there and collapsed afterwards:

```
Σ = collapse( diag(P_unc)·F·diag(P_unc) )   absolute covariance; P_unc is the
                                            UNCOLLAPSED spectrum matching F's dims
Σ = U S Uᵀ                                  (eigendecomposition)
L = U·√S        with modes below tolerance dropped (their columns are zero)
```

so that `Σ = L·Lᵀ` (up to the dropped below-tolerance modes), and a random
spectrum fluctuation with exactly the covariance Σ is simply `L·g` with
`g ~ N(0, 1)` per component. `L` is n_bins × n_bins with `k ≤ n_bins`
non-zero columns (the rank). Each non-zero column of `L` is one independent
"mode" of correlated systematic variation — that picture matters in C.2 and
C.4, where each mode becomes one effective parameter.

One approximation to note now: `L` is **frozen at its reference spectrum**.
Inside the χ² the covariance is rescaled by the *current* prediction at every
single evaluation, but the band machinery builds `L` once — the
**linear-response approximation**. The default pre-fit band references the CV
spectrum; the post-fit band, the `--mcmc-prefit` variant, and both degenerate
shortcuts (below) reference the best-fit spectrum, because that is the
spectrum they are drawn around.

**Spline systematics** have Gaussian priors `s_j ~ N(c_j, σ_j)` (XML
`center=`/`prior=`, defaults 0 and 1), and are genuine fit parameters — the
fit moves them, so their post-fit spread must come from sampling the fit's
posterior, not from a formula.

All bands are reported per bin as **16/84 percentiles** of an ensemble of
sampled spectra — a 68% interval that keeps any real asymmetry, rather than
forcing a symmetric ±σ — plus a full bin-to-bin covariance of the ensemble
for anything downstream that needs correlations (2D→1D projections,
ratio-error propagation).

### C.1 The pre-fit band — sampling the prior

*(Implemented in `getErrorBand` → `FillSystRandomThrow` (src/PROcess.cxx);
the per-variable bands from `plot` and the default global pre-fit band both
use this. `--mcmc-prefit` swaps in a prior-only Metropolis chain through the
same machinery as C.2 with zero data — note that variant, like the post-fit
band, references the best-fit spectrum rather than the CV.)*

The idea is the simplest thing you could do: **throw every systematic from
its prior many times, rebuild the spectrum each time, and look at the spread
of spectra you get.** No data is involved anywhere. Concretely, for each of
`N = 2500` throws `i`:

1. **Throw every spline from its prior**: `s_j⁽ⁱ⁾ ~ N(c_j, σ_j)`,
   independently per spline (marginals only — XML correlations enter the fit
   through the pull term, not these throws). Physics stays at the CV.
2. **Rebuild the spectrum through the full (nonlinear) spline response**:
   `S⁽ⁱ⁾ = P(φ_CV, s⁽ⁱ⁾)` — each event/bin is reweighted through its response
   spline at the thrown knob values, then collapsed. This is where the
   nonlinearity of the spline systematics is kept honestly: a +1σ throw and a
   −1σ throw need not have equal and opposite effects.
3. **Add a covariance throw on top**: `S⁽ⁱ⁾ ← S⁽ⁱ⁾ + L·g⁽ⁱ⁾`, `g⁽ⁱ⁾ ~ N(0,1)`,
   with `L` built at the CV spectrum. This single vector of Gaussian numbers,
   pushed through `L`, fluctuates all bins together with exactly the
   covariance Σ.

The band in each bin `b` is then read off the ensemble:

```
e⁺_b = q84_b − P_b(θ_CV)        e⁻_b = P_b(θ_CV) − q16_b
```

with `q16/q84` the percentiles of the 2500 values of `S_b⁽ⁱ⁾`, quoted about
the **CV prediction** — which is the ensemble center to first order (spline
nonlinearity can shift the throw mean slightly off the CV; it is the same
effect that makes e⁺ ≠ e⁻). The stored covariance is
`(1/N)·Σᵢ (S⁽ⁱ⁾−P_CV)(S⁽ⁱ⁾−P_CV)ᵀ`.

That is the whole story before data: **prior widths, centered on the CV.**

### C.2 The post-fit band — sampling the posterior

*(Implemented in `getMCMCErrorBand` (inc/PROplot.h), called by
`run_global_fit` (bin/PROfit_fit.cxx) with the data spectrum; drawn by
`plot_channels`.)*

After the fit, the two kinds of systematic need different treatment, because
the fit treated them differently. The **splines** were genuine fit
parameters, so their post-fit spread comes from sampling the fit's posterior
directly (Step 1). The **covariance systematics** were never fit parameters —
the fit marginalized them analytically — so their post-fit spread has to be
*reconstructed* from a formula (Step 2). The two mechanisms compose exactly,
and section C.4 proves the formula used in Step 2.

**Step 1 — sample the spline posterior with MCMC.** A Metropolis chain runs
over the *free* spline parameters (physics is held at `φ̂`, as are any
`--fix`'d splines), targeting the fit likelihood itself,

```
exp( −χ²(φ̂, s | d) / 2 )
```

with the full fit metric. Two things ride along for free because the target
is the real χ²: the covariance-marginalized `M = stat + Σ(θ)` (where the
covariance is rescaled by the *current* prediction at every evaluation — the
frozen `L` of C.0 is only a band-reconstruction approximation, never a fit
approximation), and the spline pull terms including any configured
correlations. In plain words: the chain wanders through spline space visiting
each point in proportion to how well it describes the data, given everything
else the fit knew. Defaults: 25,000 burn-in + 20,000 kept steps
(`--fit-options MCMCburn/MCMCiter`). Each kept step `i` gives a spectrum

```
S⁽ⁱ⁾ = P(φ̂, s⁽ⁱ⁾)        (collapsed; NO prior covariance throw here)
```

— note the covariance systematics contribute *nothing* yet; adding a prior
`L·g` throw here would be wrong, because the data has constrained them.

**Step 2 — reconstruct the covariance-systematic posterior per sample.**
Here is the key fact (derived in C.4): because the covariance systematics
shift the spectrum *linearly* and have Gaussian priors, their posterior given
the data and a fixed spline point is **exactly Gaussian, with a mean and
covariance you can write down**. So instead of fitting them, we compute the
Gaussian and draw from it — once per chain step.

First, restrict to the **contributing bins** `B`: bins that are active
(`PROconfig::SetActiveBins` mask) **and** have `d_b > 0`. These are exactly
the bins PROchi uses (its statistical term is `diag(max(d,1))` and zero-data
bins are marginalized away), so the reconstruction is constrained by the same
information the fit was — PROjector-masked channels, for example, cannot pull
on anything. On those bins define:

```
C   = diag( max(d_b, 1) )                 statistical covariance, b ∈ B
L_B = rows of L in B, zero columns dropped   (n_B × k)
A   = 1_k + L_Bᵀ C⁻¹ L_B                  (k × k, factorized once, in double)
```

Then for every chain step `i`, with residual `u⁽ⁱ⁾ = d − S⁽ⁱ⁾` on `B`:

```
α⁽ⁱ⁾ = A⁻¹ L_Bᵀ C⁻¹ u⁽ⁱ⁾  +  δ⁽ⁱ⁾ ,      δ⁽ⁱ⁾ ~ N(0, A⁻¹)
S⁽ⁱ⁾ ← S⁽ⁱ⁾ + L·α⁽ⁱ⁾                      (full rows: the shift touches every bin,
                                           the constraint is informed only by B)
```

In plain words: the first term is the *best-fit pull* — how far the data
drags each covariance mode, given where this particular spline sample left
the prediction — and the second term is the *left-over uncertainty* of that
pull. (Numerically, `δ` is drawn as `U⁻¹v` with `A = UᵀU` the Cholesky factor
and `v ~ N(0,1_k)`, which has covariance exactly `A⁻¹` without ever forming a
matrix inverse.)

Pushed into spectrum space (via the identity `A⁻¹L_BᵀC⁻¹ = L_Bᵀ(C+Σ_BB)⁻¹`,
proved in C.4), the two pieces become the familiar constraint formulas:

```
mean pull:      L·α_min = Σ[:,B] (C + Σ_BB)⁻¹ u        (the "constrained" shift)
fluctuation:    cov( L·δ ) = Σ − Σ[:,B] (C + Σ_BB)⁻¹ Σ[B,:]   (the shrunk width)
```

— i.e. exactly the conditional-Gaussian update every "ND-constrains-FD"-style
analysis uses, applied per posterior sample. This is provably identical
(C.4) to promoting the covariance to eigen-knob spline parameters
(`covariance_to_spline`) and fitting them, up to two caveats: the linear
response of `L` (frozen at the best-fit spectrum) and the promoted knobs'
±3σ spline range, which the analytic pull does not have.

**Step 3 — extract the band.** Per bin `b`, sort the `S_b⁽ⁱ⁾` and take

```
m_b  = q50_b                          the sample median
e⁺_b = q84_b − m_b ,   e⁻_b = m_b − q16_b
shift_b = m_b − P_b(θ̂)               stored as center_shift
```

Why the **median** and not the best-fit spectrum? After Step 2 the sample
cloud genuinely sits *away* from `P(θ̂)` — the data pulled it. If we quoted
`|quantile − P(θ̂)|` instead, a thin band sitting next to the best fit would
get folded into a fat band straddling it, which is both wrong and misleading.
So the code measures the width about where the cloud actually is (the
median), and records *how far the cloud moved* separately (`center_shift`).
The stored covariance is made **central** for the same reason (second moment
minus the mean-shift outer product), so that projected widths do not
double-count the displacement.

**Step 4 — what is drawn.** The red curve in every post-fit plot is the
**constrained best fit**

```
P_constrained = P(θ̂) + shift
```

— the spline best fit *plus* the covariance pull — folded into the best-fit
histogram at construction, so the main stack view, 2D maps, slices,
projections, and all ratio panels use it consistently. The band `[m − e⁻,
m + e⁺]` rides exactly on that curve. The payoff of this convention: an
all-spline analysis and an all-covariance analysis of the same systematics
produce the *same* red line and the *same* band (that is the equivalence of
C.4 made visible); and for a spline-only fit `shift ≡ 0`, so nothing moves
and the plots look exactly as they always did. Legends label the curve
`Best-Fit ± 1σ (post-fit)` in the main view and `Constrained Best-Fit` on
the ratio pages.

**Degenerate shortcut.** If the chain would have zero free parameters
(covariance-only systematics, or everything `--fix`'d), there is nothing to
sample — the conditional Gaussian of Step 2 *is* the entire posterior — so no
MCMC runs at all. `getCovarianceOnlyErrorBand` evaluates the two closed-form
lines once: shift `Σ[:,B](C+Σ_BB)⁻¹u`, covariance
`Σ − Σ[:,B](C+Σ_BB)⁻¹Σ[B,:]`, symmetric errors `√diag`, centered on
`P(θ̂) + shift`. (Called without data it returns the prior `√diag(Σ)` band —
the pre-fit degenerate case.)

### C.3 Properties & checks

Closure properties you can test:

* **Shrinkage**: the covariance part of the posterior strictly shrinks —
  `Σ_post = Σ − Σ(C+Σ)⁻¹Σ` is smaller than `Σ` as a matrix. Bin-by-bin on a
  plot the guarantee holds at matched prediction: the pre-fit `L` is scaled
  at the CV spectrum and the post-fit `L` at the best-fit spectrum, so if the
  fit moved the prediction a lot in some bin, the absolute widths being
  compared reference different event counts there. In practice (best fit ≈
  CV) the post-fit band is narrower everywhere, and in the high-statistics
  limit `Σ ≫ C` it approaches the pure statistical width — the data has
  simply measured the bins.
* **Asimov closure**: with `d = P(θ̂)` the per-sample pulls average to zero,
  `shift → 0`, and the constrained best fit coincides with the plain best
  fit.
* **The pull is a shrinkage of the residual — as a vector, not per bin**:
  the shift `Σ(C+Σ)⁻¹·u` scales each *mode* of Σ by a factor between 0 and 1
  (large where the systematic budget dominates the statistics, small where
  statistics dominate). Because the modes correlate bins, an individual bin
  can legitimately be pulled *beyond* its own residual, or even against it,
  by its correlated neighbours — seeing that in a plot is not a bug.
* **Representation independence**: converting a `covariance` systematic to
  `covariance_to_spline` (all modes kept) must reproduce the same band and
  the same constrained best fit, up to MC noise, the linearity approximation,
  and pulls beyond 3σ.

Some things to note, which may not be obvious:

* The constraint assumes the covariance systematics act **linearly and
  unboundedly** on the spectrum. The same assumption their presence in the
  χ² covariance already makes, so this is NOT the same as splines with a 3 sigma cutoff (but should be close)
* `C = diag(max(d,1))` matches **PROchi**; for `PROCNP`/`Poisson` fits the
  reconstruction is an approximation to the corresponding stat model.
* PROjector-masked channels (active-bins mask, zeroed data) are excluded
  from `B` automatically. Aka masked bins cannot pull on the systematics.
* The post-fit band is a posterior **conditioned on the very data drawn on
  top of it**. Agreement of data with the shrunken band is partly by
  construction: goodness-of-fit comes from the fit χ², never from this plot.
* For 2D fitting variables everything above happens per flattened (x,y) bin;
  the 1D projection bands sum the shift linearly and take widths from the
  (central) covariance summed over the projected block, while the per-slice
  pages use the per-bin percentile widths directly.

### C.4 Why this works — deriving the pull and its covariance

Nothing in Step 2 of C.2 needs to be taken on faith: both formulas fall out
of ordinary calculus on the χ², in about a page. What we can show is equivalance of pulls and covariance, aka *putting a systematic in the covariance matrix and fitting it as an
explicit pull parameter are the **same fit**, and the pull's best-fit value
and uncertainty can be recovered exactly even when you chose the covariance
route.* (This is the result of G. Putnam's SBN note "How to Obtain Pull Terms
for Systematic Uncertainties Embedded in a Covariance Matrix", written here
in PROfit's variables. Need to upload to DocDB)

**Setup.** Take the covariance systematics and make them explicit fit
parameters for a moment. C.0 built `Σ = L·Lᵀ`, so each non-zero column of `L`
is one independent mode of correlated variation; give each mode a knob
`α_j`, so the prediction becomes `P(θ) + L·α`, and give each knob a unit
Gaussian prior (that is what "the mode has size √S" already encoded into
`L`), contributing a pull term `αᵀα`. With `C` the statistical covariance and
`u ≡ d − P(θ)` the residual, the χ² with everything explicit is

```
χ²(θ, α) = (u − L·α)ᵀ C⁻¹ (u − L·α)  +  αᵀα  +  pulls(s)
```

In plain words: how far is the (shifted) prediction from the data in units
of the statistical error, plus how far did we bend each systematic knob in
units of its prior.

**Step 1 — minimize over α.** Expand the first term and collect powers of α:

```
χ² = uᵀC⁻¹u  −  2αᵀ(LᵀC⁻¹u)  +  αᵀ(LᵀC⁻¹L + 1)α  +  pulls(s)
```

Setting the derivative with respect to α to zero:

```
∂χ²/∂α = −2 LᵀC⁻¹u + 2 (1 + LᵀC⁻¹L) α  =  0
```

and naming `A ≡ 1 + LᵀC⁻¹L` (the same `A` as C.2), the best-fit pull is

```
α_min = A⁻¹ LᵀC⁻¹ u
```

The second derivative is `∂²χ²/∂α∂α = 2A`, a constant — the χ² is *exactly*
a parabola in α (this is what "linear systematic" buys you). A quadratic χ²
means a Gaussian likelihood, and a curvature of `2A` means its covariance is

```
cov(α) = A⁻¹        (the "restricted" posterior covariance of the pulls)
```

Those are precisely the two objects Step 2 of C.2 uses: draw
`α = α_min + δ` with `δ ~ N(0, A⁻¹)`.

**Step 2 — complete the square.** Because the χ² is an exact parabola in α,
it can be rewritten with no approximation as its minimum plus the quadratic
around it:

```
χ²(θ, α) = χ²(θ, α_min)  +  (α − α_min)ᵀ A (α − α_min)
```

(Multiply out the right-hand side and use the stationarity condition
`A·α_min = LᵀC⁻¹u`; every cross term cancels.) This one line *is* the whole
theorem: for any fixed θ and data, the α-dependence of the likelihood is
exactly the Gaussian `N(α_min, A⁻¹)` — so sampling α from that Gaussian, per
posterior sample of θ, reproduces the full joint posterior of (θ, α). That
is literally what the code does.

**Step 3 — recover the covariance-matrix form.** What is the minimum value
`χ²(θ, α_min)`? Substituting α_min into the expanded χ² (two of the three
α-terms merge via stationarity):

```
χ²(θ, α_min) = uᵀC⁻¹u − uᵀC⁻¹L·A⁻¹·LᵀC⁻¹u + pulls(s)
```

Now use the Woodbury matrix identity, which states exactly that

```
(C + L·Lᵀ)⁻¹  =  C⁻¹ − C⁻¹L·A⁻¹·LᵀC⁻¹
```

(you can verify it by multiplying both sides by `C + LLᵀ`). The two terms
collapse into one:

```
χ²(θ, α_min)  =  uᵀ (C + Σ)⁻¹ u  +  pulls(s)
```

which is PROfit's actual χ² — the one with `M = C + Σ` that the fitter
minimizes and section 1 describes. So: **fitting the knobs explicitly and
profiling them out gives the identical χ²(θ) as never introducing them and
putting Σ in the covariance matrix.** (For Gaussians, profiling and
marginalizing differ only by a θ-independent constant, so the statement holds
either way you read it.) The two representations are the same fit; the
covariance route just discards the record of where the knobs went — and
Steps 1–2 above are the recipe for getting that record back.

**Step 4 — the spectrum-space forms.** The band code applies the pull to the
spectrum, so translate both objects with `Σ = LLᵀ`. First a small identity:

```
LᵀC⁻¹(C + Σ) = Lᵀ + LᵀC⁻¹LLᵀ = (1 + LᵀC⁻¹L)Lᵀ = A·Lᵀ
      ⇒   A⁻¹LᵀC⁻¹ = Lᵀ(C + Σ)⁻¹
```

Applying it to the mean and to the covariance (`A⁻¹ = 1 − Lᵀ(C+Σ)⁻¹L`
follows from the same line):

```
L·α_min   =  L·Lᵀ(C+Σ)⁻¹u        =  Σ (C+Σ)⁻¹ u
cov(L·δ)  =  L·A⁻¹·Lᵀ            =  Σ − Σ (C+Σ)⁻¹ Σ
```

— the "constrained shift" and "shrunk width" quoted in C.2. In the fit
itself only the contributing bins `B` enter, which is the same derivation
with `L` replaced by `L_B` and the shift `L·α` still applied to every bin —
exactly the restriction Step 2 of C.2 makes.

**Reading the result.** `Σ(C+Σ)⁻¹` is a matrix "fraction" — systematics over
(statistics + systematics). Where the systematic budget dominates, the
fraction approaches 1 and the data pulls the prediction essentially all the
way onto itself; where statistics dominate, it approaches 0 and the
prediction barely moves. The posterior width `Σ − Σ(C+Σ)⁻¹Σ` is the prior
width minus what the data pinned down — always smaller, shrinking to the
statistical floor in the high-statistics limit. Everything the post-fit band
does is these two lines, evaluated once per MCMC sample.

---

*Why is the repository called "Elephant Vanishes"? Old habit — all of my git repos are named after the book I was reading when I created them. This one was Haruki Murakami's* The Elephant Vanishes.
