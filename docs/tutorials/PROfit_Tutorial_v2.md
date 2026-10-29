# PROfit v2.X Walk-Through Tutorial

This tutorial walks through the full PROfit v2.X workflow, from the conceptual
building blocks and XML configuration all the way to Feldman-Cousins, the
PROjector two-stage fit, and sideband-only fit regions. Every plot shown here can be regenerated with the
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

> **Versioning:** this tutorial targets the **v2 release line and its v3
> continuation** (development branch `project-SBN-dev`; the project version
> was bumped to **3.0.0** in August 2026 — same XML schema, but changed
> fitter defaults and post-fit covariance conventions, see the fitting and
> plotting sections). Breaking XML changes were made in the v1→v2 update, so
> v1.x XMLs will **not** work with v2/v3 binaries, and bugfixes are not
> back-ported to the v1.1 line — use v2.1.1+ for anything new.

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
(`SBL_2flav_nueapp` model: `dmsq`, `sinsq2thme`), two channels (`nue`, `numu`), and a
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
11. [Fit regions — sideband-only fits and constrained predictions](#11-fit-regions--sideband-only-fits-and-constrained-predictions)
12. [PROletariat — grid submission: `proletariat`](#12-proletariat--grid-submission-proletariat)

Appendices:

* [Appendix A: regenerating every plot in this tutorial](#appendix-a-regenerating-every-plot-in-this-tutorial)
* [Appendix B: available physics models](#appendix-b-available-physics-models-incpromodelh)
* [Appendix C: the pre-fit and post-fit error bands, in full](#appendix-c-the-pre-fit-and-post-fit-error-bands-in-full)
* [Appendix D: shape-only fits (`--shapeonly`), in full](#appendix-d-shape-only-fits---shapeonly-in-full)
* [Appendix E: counting degrees of freedom (ndf)](#appendix-e-counting-degrees-of-freedom-ndf)
* [Appendix F: post-fit pull distributions, in full](#appendix-f-post-fit-pull-distributions-in-full)

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
   global fit / profile / surface / FC / adaptive-FC / PROjector / fit regions
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
PROjector and `--fit-region` channel selection, flat/norm systematics, `apply_to_subchannel=`)
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
Physics parameters are mostly in **log10 space** (for `SBL_2flav_nueapp`:
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
Baker-Cousins likelihood-ratio sum. It has no covariance term, so it refuses
to run while any covariance-type systematic (MC-stat included) is selected;
drop them with `--exclude-systs`.

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
<model tag="SBL_2flav_nueapp">
    <rule index="0" name="No Osc"/>
    <rule index="1" name="Nue Appearance"/>
    <parameter name="L/E" variable_index="1"/>
</model>
```

`tag` selects the physics model implemented in `inc/PROmodel.h`. Tags follow a
fixed `regime_model_parameterization(_NC)` convention
(`SBL_2flav_numudis` = 3+1 νμ disappearance with parameters `dmsq`, `sinsq2thmm`;
`SBL_2flav_nueapp` = 3+1 νμ→νe appearance with `dmsq`, `sinsq2thme`; also
available: `SBL_2flav_nuedis`, the full three-parameter `SBL_3+1_Usq`
(|Ue4|², |Uμ4|², Δm² with a unitarity constraint), several re-parameterized
`SBL_3+1_*` variants, and NC-disappearance extensions (`*_NC`) — see
section 7 for why those exist, and the model appendix at the bottom of this
document for the convention itself plus every model's parameters, bounds, and
rules; legacy pre-v3.1 tags like `nueapp` or `3+1` keep working as deprecated
aliases). Each *rule* is an
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

Three MCFile details worth knowing: `filename` can be a single ROOT file, a
**TChain wildcard** (any name containing `.root` is handed to `TChain::Add`,
which expands the glob itself — a pattern matching no files is a fatal error),
**or a plain-text filelist** of ROOT files; several `<MCFile>` blocks may share
one `filename`/wildcard string (e.g. to split branches across blocks); and the
`pot` attribute is the *generated* POT of that file — summed over every file
the wildcard/filelist pulls in — from which events are scaled up to the
detector `pot` declared at the top of the XML. One wildcard caveat: the config
hash sees only the pattern string, so new files appearing in a globbed
directory do not invalidate the `_prop.bin`/`_syst.bin` caches — re-`process`
with `--force`.

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
| `binned_unconstrained` | you, in the XML (a name; scope via `apply_to_subchannel`) | ONE free, un-pulled normalisation parameter **per bin** of the `binning` variable, shared by every matched subchannel — the bins float independently inside `scale_range` (default ×0 to ×10) with a uniform prior | yes, N of them, named `<name>_bin0..<name>_bin(N-1)`, in units of (scale − 1) |
| `covariance_to_spline_uniform` | the SUM of the `covariance` entries matched by `sources="regex"` (which are then not built on their own) | the `num_decomp_knobs` leading eigenmodes of the fractional covariance become **free, un-pulled** linear splines inside `restrict` (default ±10σ of the mode); the remaining modes stay a Gaussian residual covariance (`include_resid_cov`, default on) | yes, N of them, named `<name>_decomp_knob_0..(N-1)`, in σ units of each mode |
| `spline_cross_quad` | a weight branch holding, per event, the CV weight and the joint +1σ shift of every pair of the `spline` entries listed in `splines="A, B, ..."` | the ηᵢηⱼ cross coefficients those splines' product lacks; the group's factor becomes `1 + Σ(sᵢ−1) + Σ eᵢⱼηᵢηⱼ`, exact for a response quadratic in the group | no — it couples the listed splines |

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
* `prior_type="gaussian|uniform"` — `uniform` (spline type only) removes the
  Gaussian pull entirely: the parameter floats freely inside its
  `restrict="lo,hi"` range, which becomes **mandatory** (and `prior=`,
  `center=`, and `<correlation>` entries are forbidden for it). Use it for
  a genuinely unconstrained scale factor you want measured, not pulled.
  FC/Brazil/fc-adaptive pseudo-experiment throws draw such splines flat
  over the `restrict` range, matching the prior. (PROplot's pre-fit band
  still throws them as N(0,1); that is display only.)
* `mode="covariance_to_spline"` with `num_decomp_knobs=` promotes a
  covariance to its leading eigenmode splines (the same machinery PROjector
  uses — see section 9). `restrict` bounds a spline's allowed range.
* `apply_to_subchannel="pattern"` — restrict ANY systematic to the
  subchannels whose fullname matches the pattern (same unanchored-regex
  matching as `norm`/`flat` — plain substrings work as-is, e.g.
  `apply_to_subchannel="nu_SBND"` or `"_ND_"`, and regex like
  `"nu_(ND|FD)"` too). Non-matching subchannels get exactly no response (flat
  spline at 1 / zero covariance rows and columns). It is honoured by every
  type: weight/universe-based ones (`spline`, `covariance`,
  `covariance_to_spline`, `spline_to_covariance`, `hist1d/2d`,
  `explicit_spline`, `norm`) are scoped while the MC is read — non-matching
  events fill every universe at the CV weight, and the weight branch is only
  required (or even looked for) in MCFiles that fill a matching subchannel —
  and, independently of how a systematic was built, PROsyst re-asserts the
  scope after ALL systematics exist (`PROsyst::ApplySubchannelScopes`), which
  is what makes it work for `flat`, `norm_to_covariance`, `external_covariance`,
  `external_covariance_to_spline`, `mcstat`, DetVar (`<DetVarFiles>`) and
  HistVar systematics too. Notes: for `norm`/`flat` the `NAME:percent` pattern
  and `apply_to_subchannel` intersect; a `<HistVarSection><subchannel>` list is
  an exact-name scope that is ANDed with the regex; `incl_systematics="false"`
  on a branch overrides it; the pattern is keyed by subchannel, so it cannot
  exempt one of two MCFiles that fill the same subchannel (use
  `incl_systematics="false"` on that branch). This is how per-detector
  systematics work in multi-detector fits where each detector's MC carries a
  different set of weight branches. The converse, ONE detector-variation knob
  shared by several detectors, is a DetVar `<variation name=…>` repeated across
  `<DetVarSection>`s: it is one systematic (one fit parameter), each section
  supplying the response in its own subchannels from its own `<cv>`. Every
  section must give the same `knobval`s. Where two sections fill the same
  subchannel their responses are averaged, weighted by each section's
  POT-normalised CV, so set their `pot=`/`scale=` as for the matching MC files.
* `scale_range="lo, hi"` — `binned_unconstrained` only: the multiplicative
  range every free bin may take (default `0, 10`; must contain 1).
* `smooth="sigma"` — smooth this systematic's variation/CV ratios across
  neighbouring reco bins (Gaussian kernel, `sigma` in bins; `"1"` is a good
  start) to suppress MC statistical fluctuations, e.g. in DetVars. Each
  subchannel is smoothed separately, keeping its rate change; ratios never go
  negative. Applies to `spline`, `spline_to_covariance` and `hist1d/2d`, not
  `spline_cross_quad` members. Part of the config hash: re-run `process` after
  changing it.

### Freeing a whole distribution bin-by-bin: `binned_unconstrained`

```xml
<bins unit="True E_{#nu} [GeV]" edges="0 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.5 2.0 3.0 20.0" plot="false"/>
...
<allowlist type="binned_unconstrained" binning="var2" apply_to_subchannel="numu_(cc|numunc|othernc)"
           scale_range="0, 10" tag="FreeFlux" plotname="Free flux">free_flux</allowlist>
```

One line, N fit parameters: for each local bin `j` of the `binning` variable
PROfit synthesizes a spline named `free_flux_bin<j>` that multiplies bin `j`
of **every** matched subchannel (here, all neutrino subchannels of all three
detectors — so one parameter per true-energy bin is shared between near and
far, which is what a "flux shape fully unconstrained" study wants; write two
entries with different `apply_to_subchannel` patterns to decorrelate them).
The parameter is `scale − 1` so the CV sits at 0 like every other spline,
its box is `[lo−1, hi−1]`, and there is **no Gaussian pull** (it is a
`prior_type="uniform"` spline under the hood). Things to know:

* the response is exactly linear, so nothing is read from the MC: no weight
  branch, no universes, and the splines are rebuilt from the cached
  `SystStruct` on every run (`process` is still needed once because the
  binning variable must be in the `_prop.bin`);
* every matched subchannel must share the binning of `binning` (always true
  within one channel; PROfit refuses mixed binnings — use one entry per
  channel instead);
* when `binning` is not the fitting variable the per-bin factors are folded
  into the reco spectrum through the CV migration matrix, exactly as for any
  other true-binned spline (see `binning` above);
* the parent name works everywhere a name is accepted for
  `--syst-list`/`--exclude-systs` (it expands to its children), while
  `--fix` and `--inject-systs` take the individual `_bin<j>` names;
* `prior=`, `center=`, `prior_type=`, `restrict=` and `<correlation>` are
  refused for these entries; FC/Brazil throws draw them flat over their
  range, like any uniform-prior spline (see above).

### Coupling quadratic splines exactly: `spline_cross_quad`

```xml
<systematic type="spline" binning="var0" knobvals="-3, -2, -1, 0, 1, 2, 3" force_0_cv="true">FA_PCA1</systematic>
<systematic type="spline" binning="var0" knobvals="-3, -2, -1, 0, 1, 2, 3" force_0_cv="true">FA_PCA2</systematic>
<systematic type="spline_cross_quad" binning="var0" force_0_cv="true" splines="FA_PCA1, FA_PCA2">FA_CROSS</systematic>
```

Every `type="spline"` systematic is one fit parameter, in σ units of its knob.
Write the parameters of the two splines above as `η₁` (FA_PCA1) and `η₂`
(FA_PCA2). From its ±1, 2, 3σ universes PROfit builds, per bin, a response curve
`sᵢ(ηᵢ)` for each spline on its own. Near the CV that curve is
`sᵢ(ηᵢ) = 1 + bᵢηᵢ + dᵢηᵢ² + …`, Here, we assume `dᵢηᵢ²` is the highest non-zero term. 
When several splines are active, PROfit multiplies their curves, so the prediction for the bin is scaled by
`s₁(η₁) · s₂(η₂)`.

That product is only correct when the two parameters act independently. It
fails whenever the true response contains a term proportional to `η₁η₂`:
multiplying out `s₁ · s₂` gives `η₁η₂` the coefficient `b₁b₂`, whereas the
physics can give it any value. A case of this is the axial form factor. The fit
parameters shift `F_A` linearly, `F_A = F_A^CV + c₁η₁ + c₂η₂` for fixed
functions `c₁`, `c₂` of `Q²`. We expect the event weight to be quadratic in `F_A`. Squaring
the sum produces a `2c₁c₂η₁η₂` piece in the per-event weight that no product of
one-parameter curves reproduces.

For the splines listed in `splines=` (the "group"), the per-bin response to
moving all of the group's parameters at the same time is, to second order,

```
1 + Σᵢ bᵢηᵢ + Σᵢ dᵢηᵢ² + Σᵢ<ⱼ eᵢⱼηᵢηⱼ
```

The member splines already fix `bᵢ` and `dᵢ`. The `spline_cross_quad` weight
branch supplies what is missing, `eᵢⱼ`, through one extra universe per pair of
members. Entry 0 holds the CV weight. The pairs are enumerated as `i<j` over the
`splines=` list, so with `splines="A, B, C"` entry 1 is A and B both shifted to
+1σ, entry 2 is A and C, and entry 3 is B and C. `N` members give `N(N−1)/2`
pairs and the branch must hold exactly `1 + N(N−1)/2` entries. From each pair
universe PROfit solves for `eᵢⱼ` exactly, and the group's weight becomes
`1 + Σᵢ(sᵢ−1) + Σᵢ<ⱼ eᵢⱼηᵢηⱼ`, evaluated additively instead of as `Π sᵢ`.

The `spline_cross_quad` entry is not a fit parameter and never appears in a
PROsyst on its own. Its name is a shorthand for its members:
`--syst-list FA_CROSS` is the same as `--syst-list FA_PCA1 FA_PCA2`, and
`--exclude-systs FA_CROSS` removes both. If a selection keeps only some of a
group's members, the coupling is dropped and the survivors are combined
multiplicatively as ordinary splines (PROfit warns when this happens). Members
must be `type="spline"` on the same `binning`, carry no `inflate=`, use evenly
spaced `knobvals`, belong to no other group, and all carry the same
`apply_to_subchannel` pattern or none. The entry itself inherits that pattern
when it has none. The entry and its members must carry the same
`include_only_weights` (or none), since `eᵢⱼ` is extracted from ratios of
universes that have to be filled with the same event weights. The group is not
supported with `--shape-only`, where the per-channel normalised response is no
longer quadratic. Members must also have `force_0_cv="true"` (or no `0` in
their `knobvals`) so that `sᵢ(0) = 1`, which the additive form assumes. 
The additive form is exact only when the response is at most
quadratic in the group's parameters, which holds for the `F_A²` case above; 
PROfit checks each member against a quadratic at its knots and warns if the 
departure is large.

### Freeing the leading modes of a covariance: `covariance_to_spline_uniform`

```xml
<!-- the 13 flux multisims stay declared as usual ... -->
<allowlist type="covariance" tag="Flux" plotname="piplus">multisim_piplus_Flux</allowlist>
...
<!-- ... and this entry sums their covariances and frees the 5 leading modes of the sum -->
<allowlist type="covariance_to_spline_uniform" num_decomp_knobs="5" sources="multisim_.*_Flux"
           tag="FreeFluxModes" plotname="Flux mode">flux_modes</allowlist>
```

The same eigen-decomposition as `covariance_to_spline`, applied to the
**sum** of the `type="covariance"` entries whose names match `sources`
(unanchored regex, like every other pattern site; anchor it, e.g.
`sources="^multisim_piplus_Flux$"`, to decompose a single entry). The
retained knobs `<name>_decomp_knob_0..(N-1)` carry **no Gaussian pull**:
each floats freely inside `restrict="lo, hi"` (σ units of its mode; default
±10σ). The un-kept modes are folded into `<name>_resid_cov` as usual
(`include_resid_cov="false"` drops them). This asks "what if the shape of
the existing uncertainty were right but its size were unknown" — the middle
ground between trusting the covariance as a prior and freeing every bin
independently (`binned_unconstrained`).

The source entries are what gets read from the files (so `scale=`,
`inflate=` and `apply_to_subchannel=` belong on them); they are then **not**
built as covariances of their own — their variance lives only in the modes
plus the residual covariance, so nothing is counted twice. The sum is formed
in the fitting variable's bin space, so the modes are exactly the ones the χ²
sees. Notes:

* `sources` and `num_decomp_knobs` are both **mandatory** (freeing *all*
  modes of a reco-space matrix is almost never what you want);
* `prior=`, `center=`, `prior_type=`, `binning=` and `<correlation>` are
  refused for this entry;
* the parent name expands to its children in `--syst-list`/`--exclude-systs`
  (this now also holds for plain `covariance_to_spline`); `--fix` and
  `--inject-systs` take the `_decomp_knob_<k>` names; a consumed source entry
  is no longer a fit object, so its own name resolves nowhere;
* the usual uniform-spline caveat: PROplot's pre-fit band throws these knobs
  as N(0,1) (not meaningful); FC/Brazil throws draw them flat over the knob box.

### Asymmetric errors from histogram sources: `<HistVarFiles>`

The `hist1d`/`hist2d` systematic types reweight events by looking up a
ratio histogram in a kinematic variable (`xvar=`/`yvar=`). Historically one
histogram gave a single mirrored ±1σ universe; since v3.0 a systematic can
carry **several independently-measured variation histograms**, each tagged
with a knob value — the natural way to feed in *asymmetric* errors (a −1σ
and a +1σ histogram measured separately):

```xml
<variation_list>
  <allowlist type="hist1d" xvar="var0" mirror="false">my_asym_syst</allowlist>
</variation_list>
<HistVarFiles>
  <HistVarSection name="my_asym_syst">
    <variation filename="syst_dn.root" histname="ratio_m1sigma" knobval="-1"/>
    <variation filename="syst_up.root" histname="ratio_p1sigma" knobval="1"/>
    <subchannel>nu_ICARUS_numu_signal</subchannel>  <!-- optional scope; EXACT fullname, no regex -->
  </HistVarSection>
</HistVarFiles>
```

The section `name` must match a `hist1d`/`hist2d` entry in the variation
list, which must set `mirror="false"` when more than one variation is
declared. Each histogram becomes one universe at its `knobval`; the CV
(ratio 1 at knob 0) is inserted automatically, and the standard per-bin
response spline is built through the knots — the asymmetry lives in the
spline shape while the fit pull stays a symmetric Gaussian (standard
vertical-morphing practice). Things to get right:

* declaration order is free (sorted by knobval internally), but **knob
  values must be unique** — duplicates currently produce silently corrupt
  splines rather than an error;
* the fit parameter *and* FC throws are bounded by the extreme knob values,
  so a ±1σ-only pair clamps the parameter to [−1, 1] — supply ±2/±3σ
  histograms, or set `restrict=`, if the fit may want to pull further;
* `<subchannel>` entries are **exact fullnames** (unlike every pattern site)
  and are not checked for existence — a typo silently disables the
  systematic (all events fall out of scope, flat spline);
* lookups happen on each histogram's own binning; out-of-range events get
  weight 1;
* the config hash covers the filenames and knob values but **not the
  histogram contents** — after editing a variation file in place, re-run
  `process --force`.

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
  protest      Testing ground for rapid quick tests (developer scratch space).
  scale-test   Run timing benchmarks for FillSpectra / metric / fit hot paths.
  proletariat  Stage, tar, and submit grid jobs (section 12).
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
| `--pseudo-experiment` | full FC-style throw: spline pulls + covariance shift at the toy's own prediction + Poisson stats (combines with `--inject`) |
| `-d/--data file` | load real data from a separate file/XML (plot subcommand) |
| `--use-fake-data` | ignore any embedded/external data and force MC fake data |
| `--scale ND 0.5` | scale POT of subchannels matching a pattern |
| `--seed 405` | fix the RNG seed (default -1 = hardware random) |

Reproducibility rule of thumb: `--seed N` plus `-n 1` is bit-reproducible;
multithreaded runs are statistically equivalent but not byte-identical.

### Controlling the systematics and parameters

| Option | What it does |
|---|---|
| `--syst-list Flux1 Flux2` | use ONLY these systematics (the MC-stat covariance too only if its XML name is listed) |
| `--exclude-systs RPA_CCQE` | use everything except these (MC-stat stays unless its XML name, e.g. `MCStat`, is listed) |
| `--fix dmsq Flux1` | fix parameters at CV (physics or splines) |
| `--syst-only` | fix ALL physics parameters (nuisance-only fit) |
| `--statonly` | drop systematics entirely |
| `--shapeonly` (alias `--shape-only`) / `--rateonly` | shape-only or single-bin-normalisation analysis. Shape-only: in every collapsed channel (per mode × detector × channel) the *prediction* is rescaled onto the data's integral before the χ² (the data is never touched, so the statistical term is fixed and the χ² is exactly invariant under an overall rate change); that rescale removes the normalisation part of every spline, and the covariance systematics are projected onto shape (the MiniBooNE M_shape) about the current prediction at every evaluation. The systematics themselves stay physical. One dof per channel is lost; `fc`/brazil/`fc-adaptive` inherit the flag. Implies `--area-norm` for plots. Full details: [Appendix D](#appendix-d-shape-only-fits---shapeonly-in-full). |
| `-c/--chi2 neyman\|pearson\|CNP\|poisson` | χ² metric (default `neyman`; legacy aliases `PROchi`/`PROCNP`/`Poisson`) |
| `--grad-mode analytic` | gradient strategy: `analytic` (default, alias `exact`) / `central-full` / `one-sided-full` / `central-lin` (Gauss-Newton) / `one-sided-lin` |

`--syst-list` and `--exclude-systs` accept XML names, plotnames or tags (a tag
selects every systematic that carries it), and a `covariance_to_spline` /
`external_covariance_to_spline` / `binned_unconstrained` parent name selects all
of its derived parameters. A name that matches nothing is fatal, so a typo cannot
silently fit the full set.

The **default gradient is now `analytic`** (closed-form spectrum Jacobian
through the oscillation models and splines, *plus* the exact
covariance-scaling term the old Gauss-Newton `-lin` modes dropped): as
accurate as `central-full` at the cost of roughly one evaluation, so it is
the right choice for scans *and* publication runs. It applies to the
`neyman` (PROchi) binned strategies; `CNP`, `poisson`, and
`--eventbyevent` automatically
fall back to `central-lin`, and `pearson` to `one-sided-full` (its
statistical variance IS the prediction, so the term Gauss-Newton drops is
first-order and the lin modes can stall), each with a one-time warning. `central-full` remains
the finite-difference gold standard for cross-checks (and `scale-test
--tests gradcheck` validates every mode against it). Note the default change
(from `central-lin`) means fits are **not bit-reproducible against
pre-August-2026 seeded references**; pass `--grad-mode central-lin` to
reproduce old numbers.

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

### Keeping the arguments in a file: `--config`

Long command lines get hard to reproduce. Any argument from this tutorial can
go in a TOML (or INI) file instead, passed with `--config`:

```toml
# tut_profile.toml
xml      = "tutorial.xml"
tag      = "TUT"
output   = "probe_v1"
nthread  = 8
log      = "profile.log"
progress = true
inject   = ["dmsq", 1, "sinsq2thme", 0.01]
preset   = ["grad-good", "grad-fast"]

[profile]
probe        = true
probe-chunks = 2
```

```bash
PROfit --config tut_profile.toml                  # runs the profile above
PROfit --config tut_profile.toml -n 16 -o probe_v2  # command line overrides the file
```

- A key is the option's long name without the dashes (`nthread`,
  `grad-mode`, `fit-options`). Flags take `true`/`false`, and multi-value
  options take a TOML array.
- Global options go at the top. Subcommand options go under a
  `[subcommand]` section (`[profile]`, `[surface]`, `[fc-adaptive]`, ...).
- A `[subcommand]` section also **runs** that subcommand, even if you don't
  name it on the command line. If you name a different subcommand on the
  command line, both run.
- Anything given on the command line wins over the file. One run card can
  therefore serve a whole study, with `-o`, `--seed`, `-i` etc. varied per
  job.
- INI syntax (`key=value`, with space-separated lists such as
  `inject=dmsq 1 sinsq2thme 0.01`) is also accepted.
- To tie a run card to a build, add `required-version = "v3.0.5"` (or pass
  `--required-version v3.0.5`). PROfit then stops straight away unless it is
  exactly that version, the one printed in the log header and on every plot.
  A development build only matches its full string (`v3.0.5-dev`), not the
  release it leads up to.
- **Misspelled keys are silently ignored**: `nthreads = 8` leaves you on
  one thread. The first log line echoes the xml, tag, output and nthread
  PROfit actually picked up, which is a quick check.

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
  --with-covar                Include the covariance/correlation matrix plots
                              (off by default since v3.0 — they are the slow part).
  --no-frac-syst              Skip the fractional-systematics breakdown PDFs.
  --band-throws INT [2500]    Number of throws used to build the error band.
  --bkg-subtract TEXT         Pattern (unanchored regex; plain substrings work);
                              that background's CV is subtracted from data and CV
                              at plot time (publication convention).
```

Plus the relevant global options: `--area-norm`, `--scale-by-width`,
`--plot-bounds ymax 100 ratmin 0.5 ratmax 1.5`, and all the injection
machinery from section 3.

> **Changed default (v3.0):** `plot` no longer produces the covariance
> plots (`_PROplot_Covar.pdf` and the ROOT `Covariance/` directory) unless
> you pass `--with-covar` — that flag alone restores the old output exactly
> (byte-identical bands in seeded runs). `--no-frac-syst` additionally drops
> the fractional-systematics PDFs; with *both* the covariance plots off and
> `--no-frac-syst`, the slow spline→covariance conversion is skipped
> entirely (a big speedup for spline-heavy configs) and the χ²/ndf labels
> disappear from the error-band pages. If a downstream script consumes
> `_PROplot_Covar.pdf` or the `Covariance/` ROOT directory, add
> `--with-covar` to its plot command.

### The CV and error band

```bash
PROfit -x tutorial.xml -t TUT -o plotcv --seed 405 plot --with-covar
```

Outputs (one `Variable_<i>` set per plotted binning of each channel):

* `TUT_plotcv_PROplot_Variable_0_CV.pdf` — stacked CV spectra, fitting variable
* `TUT_plotcv_PROplot_Variable_0_ErrorBand.pdf` — CV + full systematic band
* `TUT_plotcv_PROplot_Variable_2_*.pdf`, ... — same for the other variables
* `TUT_plotcv_PROplot_Covar.pdf` — all covariance matrices, per systematic and total (`--with-covar` only)
* `TUT_plotcv_fractional_systematics.pdf` — fractional uncertainty per bin, one panel per systematic `tag` plus a summary (the `tag=` attributes in the variation list control this grouping — without them you get one unreadable 30-line legend; suppressed by `--no-frac-syst`)
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
and draws the post-fit results. Its only subcommand option is
`--legacy-postfit-error` (also on `profile`), which reverts the post-fit band
to the pre-v3.0 recipe — posterior spline throws around the best fit, but
*prior* (unconstrained) covariance-systematic throws with no center shift (see
Appendix C.2). Everything else is controlled by the global arguments.

```bash
PROfit -x tutorial.xml -t TUT -o glob1 --seed 405 -n 8 \
    -i dmsq 1 sinsq2thme 0.01 --log glob1.log global
```

Outputs:

* `TUT_glob1_global_fit.txt` — best-fit χ² and every parameter value in a plain-text table
* `TUT_glob1_PROglobal_hists.pdf` — pre-fit (blue/gray) vs post-fit (red) spectra + error bands, with a data/fit ratio panel
* `TUT_glob1_PROglobal_postfit_correlation_matrix.pdf` (+ `_nuisance_only` version) — post-fit parameter correlations
* `TUT_glob1_PROglobal_postfit_posteriors.pdf` — post-fit parameter constraints
* `TUT_glob1_PROglobal_postfit_pull_distributions.pdf` — histograms of all nuisance pulls, which should look like N(0,1) when the model describes the data (all 0 for this Asimov fit). See [Are the pulls healthy?](#are-the-pulls-healthy-pull-distributions) below.
* `TUT_glob1_PROglobal.root` — all of the above as ROOT objects

The `global #chi^{2}/ndf` label on the post-fit pages uses
**ndf = (bins entering the χ²) − (free physics parameters) − (free
uniform-prior splines) − (one per channel under `--shapeonly`)**; the log
prints the breakdown. See Appendix D for what each term counts.

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

The default `grad-*` presets (below) rebalance this pipeline around the
analytic gradient: the Latin hypercube samples **physics parameters only**
(`latin_phys_only` — nuisances start at their centers, where the pull term
puts them anyway), the particle-swarm stage is effectively disabled (one
iteration), and the budget goes into **many L-BFGS-B multistarts** instead —
with the exact gradient, local descents are cheap and reliable enough that
they beat swarm exploration. The classic three-stage behavior is still
available via the older presets (`good`, `overkill`, ...).

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
--preset grad-fast         # ONE value sets both global and scan config
--preset good overkill     # first = global, second = scan
--fit-options n_latin_points 2000 max_iterations 5000     # global fit knobs
--scan-fit-options n_localfit 2 ...                        # scan fit knobs
```

Presets come in two families:

* **`grad-fast` / `grad-good` / `grad-deep` / `grad-overkill`** — the
  analytic-gradient-era presets and the **current defaults**
  (`grad-good` for the global fit, `grad-fast` for scans): physics-only
  Latin hypercube, no effective PSO, escalating numbers of L-BFGS-B
  multistarts (roughly 8 / 16 / 50 / 100; the deeper two also warm-start
  the nuisances at each multistart via `localfit_warm_nuisance`).
* **`fast` / `good` / `overkill` / `sensitivity`** — the classic
  LHS→PSO→L-BFGS-B presets, kept for compatibility and for metrics/modes
  where the analytic gradient doesn't apply.

> **Reproducing pre-v3.0 numbers:** the defaults used to be
> `good`/`fast` with `--grad-mode central-lin`. Passing `-p good fast` alone
> does **not** restore them — the gradient mode is independent of the
> preset — you need `-p good fast --grad-mode central-lin` for
> bit-comparable results against old seeded runs.

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
  latin_phys_only         : 1 = LHS samples physics only, nuisances start at centers (grad-* presets)
  n_localfit              : Total number of L-BFGS-B fits after PSO
  localfit_warm_nuisance  : 1 = warm-start nuisances at each multistart (grad-deep/overkill)
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
PROfit -x tutorial.xml -t TUT -o globcnp --seed 405 -n 8 -c CNP global
PROfit -x tutorial.xml -t TUT -o globpea --seed 405 -n 8 -c pearson global
PROfit -x tutorial.xml -t TUT -o globpoi --seed 405 -n 8 -c poisson global
```

`neyman` (the default; class `PROchi`) is the classic Neyman χ² — the
statistical variance of each bin is the *observed data* count (zero-data
bins drop out = Gaussian marginalization) — plus the scaled systematic
covariance and Gaussian pulls; `pearson` is the textbook Pearson χ², with
the statistical variance instead the *predicted* count μ at the current
parameter point (zero-data bins are kept); `CNP` swaps the statistical
variance for the combined
Neyman-Pearson form `3/(1/d + 2/μ)` ([X. Ji et al.](https://arxiv.org/pdf/1903.07185))
and is the **recommended** choice whenever bins can be low-statistics;
`poisson` is the Baker-Cousins likelihood-ratio sum. It has no covariance
term, so it is a fatal error to run it with any covariance-type systematic
(MC-stat included) still selected: remove them with `--exclude-systs`. Under
PROjector every covariance must be promoted (`--projector-knobs -1`, no
`--projector-keep-cov`). The legacy
spellings `PROchi`, `PROCNP`, and `Poisson` still work as deprecated
aliases (mapping to `neyman`/`CNP`/`poisson` with a one-time warning).
Only `neyman` has the closed-form analytic gradient — `CNP`/`poisson`
fall back to `central-lin`, `pearson` to `one-sided-full`, and all three
use the legacy `good`/`fast` presets. Adding a
new metric
is deliberately easy — they all implement the same `PROmetric` interface.

### Are the pulls healthy? Pull distributions

The posteriors and the profile 1σ plots show the systematics one at a time.
The pull distribution asks about all of them at once: *if the systematic
model and its priors are right, did the fit move each nuisance by a
believable amount?* When the answer is yes, every pull is one draw from a
unit Gaussian, so a histogram of all of them should look like N(0,1).

Every `global` and `profile` run writes it. An Asimov fit is a poor
example, because the fit never moves anything and every pull is exactly 0.
So here is a pseudo-experiment, i.e. fake data thrown from the model itself:

```bash
PROfit -x tutorial.xml -t TUT -o globpe --seed 405 -n 8 --pseudo-experiment global
```

<img src="figures/TUT_globpe_PROglobal_postfit_pull_distributions.png" width="750"/>

*`TUT_globpe_PROglobal_postfit_pull_distributions.pdf`, pages 1 and 2. Shaded:
the pulls, one colour per page; dashed: the N(0,1) they should follow; red: a Gaussian fitted to
them. The side panel compares the fit with what the model predicts for this
histogram.*

**What is plotted.** Not the familiar (θ̂ − θ₀)/σ_prior. That raw pull is
narrower than N(0,1) even when everything is right, because the data only
resolves part of each prior. PROfit plots the Demortier–Lyons pull instead

```
z = (θ̂ − θ₀) / √(σ²_prior − σ²_post)
```

which *is* N(0,1) when the model is right. It also has a plain reading: z is
the tension between the prior and what the data alone say about that
parameter, in units of their combined uncertainty. Appendix F derives both
statements and shows how PROfit gets σ_post.

**The four pages** sort the parameters by how they enter the fit:

| Page | Parameters | Treatment |
|---|---|---|
| 1 | Gaussian-prior splines whose fit range covers θ₀ ± 3σ (ordinary −3…3 knobs, `covariance_to_spline` knobs) | z as above |
| 2 | The covariance systematics, as eigenmodes of their summed covariance, each with prior N(0,1) | z of the data-constrained mode pull (the same modes as `_postfit_covariance_pulls.pdf`, Appendix C.4) |
| 3 | Gaussian-prior splines with a fit bound inside 3σ (0…1 knobs, `NAME:f` norms with f > 1/3, narrow `restrict=`) | A knob stopped at its bound is drawn hatched and treated as *censored*; the fit accounts for it |
| 4 | Uniform-prior splines (`prior_type="uniform"`, `binned_unconstrained`) | Their estimate is mapped onto a score that is also N(0,1) under the model |

Parameters the data barely constrains are left out and counted under "Not
shown", as are `--fix`'d ones (listed on page 4). "Barely" means the data
reduced the prior variance by less than 5%, so z would be 0/0. In this config
that removes the two FiducialVol knobs and 55 of the 100 covariance modes.

**Reading the side panel:**

* **Gaussian fit:** mean and width of the histogram (an unbinned maximum-
  likelihood fit, which is binning-independent with a few dozen entries).
* **If the model is right:** what to compare those against. Pulls of related
  systematics are correlated, so the mean of one fit's histogram scatters by
  more than 1/√N, and the histogram is *narrower* than 1. PROfit computes
  both from the correlations (Appendix F.6). Here the 12 spline pulls predict
  μ = 0 ± 0.47 and σ ≈ 0.88.
* **Whitened χ²/ndf:** a single goodness-of-fit for the page that does handle
  the correlations. Its p-value is the number to quote.
* **Largest |z|:** which parameters drive any problem.

**What a deviation tells you:**

* *Too wide, or a large whitened χ²:* the data pulled the systematics further
  than their priors allow. Priors that are too small, a missing systematic,
  or data the model cannot describe. For example, `--pseudo-experiment` with
  `--exclude-systs` deliberately keeps the excluded systematics in the fake
  data. The fit cannot absorb them, and a Poisson fit of this config without
  its covariances comes out 2.8 wide.
* *Clearly narrower than the "if the model is right" width:* the priors are
  more generous than the data supports.
* *A shifted mean:* a group of correlated knobs moved together. Look at the
  largest-|z| list and the post-fit correlation matrix.
* *One outlier:* one systematic in tension with the data.

The example shows one of these. The spline mean is +0.84 against
0 ± 0.47, and the whitened p is 0.012, while the covariance modes on page 2
are fine. That is not bad luck. Over 30 pseudo-experiments (physics fixed
with `--syst-only`) this config's spline pulls average +0.56 with the right
width (1.03). The offset is the
same with `-c CNP`, and stays when the toys hold every spline at zero. It
lives in a few weakly constrained knobs that move with the ND normalisation
(CrossSection3/4, DetSys2/3, Flux3, FluxNorm_ND), whose
fitted values are themselves pushed up. So the fit, not the priors, shifts
them. A likely cause is that the χ² scales the systematic covariance with the
current prediction, which rewards a slightly higher prediction when the data
fluctuate. Over the same toys the covariance-mode pulls have widths of 1.00
(CNP) and 1.04 (neyman). Pull distributions test the whole chain, fitter
included.

**Pooling many fits.** One fit is one draw. For a sharper test, run many
pseudo-experiments and pool them, e.g. `-o pe1 --seed 1` up to
`-o pe30 --seed 30`. The per-parameter
values are in the ROOT tree `postfit_pulls` (`name`, `category` = page,
`value`, `center`, `sigma_prior`, `sigma_post`, `var_reduction`, `raw_pull`,
`pull`, `censored`, `used`), and the per-page fits are in
`postfit_pull_fits`:

```cpp
TChain ch("postfit_pulls");
ch.Add("TUT_pe*_PROglobal.root");
ch.Draw("pull", "used && category==1");   // page-1 pulls of every toy
```

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
* `TUT_prof1_PROfile_postfit_correlation_matrix.pdf`, `_postfit_posteriors.pdf`, `_postfit_pull_distributions.pdf` (as for `global`; redrawn at the new minimum if the scan finds a lower one)
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

**For the `SBL_2flav_nueapp` model you must set the axes** — the built-in defaults are
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
3+1 variants (`SBL_3+1_angles` and headline-amplitude parameterizations like
`SBL_3+1_sinsq2thmue`): the plotted quantity becomes a direct 2D scan axis and
`surface` profiles smoothly over the leftover mixing freedom at each point. A
full 3D scan is still the right tool for *understanding* 3+1 — but if the end
goal is a contour in an effective angle, fit in that angle. For an
FC-calibrated contour the third parameter must be profiled explicitly
(`--profile-params`): see
[Three physics parameters: profile FC](#three-physics-parameters-profile-fc).

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
  --reuse                     Recompute the p-values from an existing
                              <tag>_<out>_FC.root instead of throwing universes
  --stat-only-throws          Poisson fluctuations only (no systematic throws)
  --syst-only-throws          Systematic throws only (no Poisson fluctuation)
  --profile-params TEXT ...   Physics parameters to profile at the tested point
  --profile-truth TEXT [data] Toy truth of the profiled parameters: data | inject
  --fc-point TEXT ...         Tested point, if different from the -i point
```

At the tested point (`-i`, or CV if none; `--fc-point` overrides it), `fc`
throws `-u` universes — each one a full pseudo-experiment: Gaussian spline
pulls sampled from the priors, a covariance-systematic shift thrown at the
universe's own prediction, and Poisson statistics — and for each universe
runs both a fit at the tested point (nuisances only) and a free fit. The
distribution of `Δχ² = χ²(fixed) - χ²(free)` is your empirical
critical-value distribution at that point. The log prints its 68.3/90/95.4%
quantiles; `--pval` compares the data's own Δχ² at the tested point against
them.

```bash
PROfit -x tutorial.xml -t TUT -o fc1 --seed 405 -n 8 \
    -i dmsq 1 sinsq2thme 0.01 --log fc1.log fc -u 500
```

Output is `TUT_fc1_FC.root` containing a TTree with, per universe, the two
χ² values and the best-fit parameters — from which you extract the
90%/95% quantiles and compare to the Wilks values — plus an `fc_meta` tree
(also written as `TUT_fc1_FC_meta.txt`) recording the tested point, fixed and
profiled parameters, metric, model, config hash, seed and throw mode. This is the honest but
brute-force approach: to calibrate a whole *contour* you would repeat it at
every grid point, which is exactly what `fc-adaptive` automates.

### Reusing a saved distribution: `--reuse`

`fc --reuse` reads a previously produced `<tag>_<out>_FC.root` back and
re-runs only the observed fits on the (real or injected) data plus the
empirical p-value calculation — no universes are thrown, so a `--gof`/
`--pval` re-evaluation against an expensive stored Δχ² distribution takes
minutes instead of days. Rules of the road:

* the file's `fc_meta` must match this run's tested point, fixed and
  profiled parameters, toy truth, χ² metric, model and config hash, or the
  run refuses (`--force` overrides). Files written before `fc_meta` existed
  are accepted with a warning — for those the match is on you. The
  systematic selection is not recorded: keep it the same;
* `-u/--universes` is ignored under `--reuse` (the sample size is the file's
  entry count; there is no top-up mechanism);
* the reused FC.root and its CSV are left untouched (only fresh throws are
  ever written);
* `--pval --reuse` on a file produced with only `--gof` is refused (the
  needed branches are absent) — it falls back to fresh throws with a
  warning if the file is missing entirely;
* if you plan to `hadd` FC.root files from parallel jobs into one pooled
  distribution, give every job a **distinct `--seed`** — identical seeds
  produce identical throw sets and there is no deduplication — and never
  mix `--gof`-only files into a `--pval` pool.

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

### Three physics parameters: profile FC

With a 3+1 model you plot two parameters and the third, *z*, is a nuisance
(ξ in `SBL_3+1_sinsq2thmue`, `sB` in `SBL_3+1_sinsq2thmumu`, `sinsqth24` in
`SBL_3+1_sinsq2thee`). Without further options `fc` pins all three at the
tested point, which tests the full 3D point: its Δχ² has three degrees of
freedom (Wilks 6.25 at 90%), and its contours are not 2D contours. For a 2D
plane, profile z instead (the NOvA "profile construction", also used by
MINOS+ and MicroBooNE):

* the test statistic minimises over z at the tested point:
  `t = min_z χ²(μ, z) − χ²_min`, asymptotically 2 dof (4.61 at 90%);
* the toys put z at its **conditional best fit to the analysed data** at
  that point, ẑ(μ); the splines are thrown from their priors as before
  (so this is NOvA's "hybrid": profile for physics nuisances,
  Highland–Cousins for systematics).

Here `tutorial_3p1.xml` is the tutorial config with
`<model tag="SBL_3+1_sinsq2thmue">` and every `<branch>`'s `model_rule` mapped
to the 3+1 rules (1 = νμ→νμ, 2 = νμ→νe, 3 = νe→νe; see the
[model reference](#sbl_31_sinsq2thee--sbl_31_sinsq2thmumu--sbl_31_sinsq2thmue-legacy-31_3a--31_3b--31_3c--headline-amplitude-31-variants));
`tests/run_short_tests.sh` builds one this way from the fake SBN config.

```bash
# one point, z = xi profiled, toy truth from the data's conditional best fit
PROfit -x tutorial_3p1.xml -t T3 -o pfc --seed 405 -n 8 --use-fake-data \
    fc -u 500 --profile-params xi --pval --fc-point sinsq2thmue 0.01 dmsq 1
```

`--profile-truth inject` throws the toys at the `-i` value of z instead (the
FC distribution at a fixed true z). If the data cannot tell z values apart
(pinning z at its default costs < 0.01 in χ²), ẑ is snapped to the default and
the log says so. Per toy, `cbest_<z>` records where the constrained fit put z.
`--fix xi` pins z instead: an FC slice at that value of z.

The adaptive pipeline takes the same flag, and every stage must get it:

```bash
AFC="fc-adaptive --xvar sinsq2thmue --yvar dmsq --xlo 1e-4 --xhi 0.3 --profile-params xi"
for m in build-mesh init-bank asimov brazil; do
  PROfit -x tutorial_3p1.xml -t T3 -o pafc --seed 405 -n 8 --use-fake-data $AFC --mode $m
done
```

What changes with a profile bank:

* `init-bank` first fits the dataset at every cell (`<tag>_bank_truth_<z>.pdf`
  maps ẑ) and throws that cell's toys there. The bank is therefore tied to
  **one dataset**: `asimov` refuses data whose fingerprint differs from the
  bank's (`--force` overrides), and the real data needs a bank of its own,
  built with the real-data options.
* `brazil` classifies every throw against the bank built at the Asimov's
  ẑ, not at the throw's own: the approximate, NOvA-style band.
* Banks are written as version 3 (v2 payload plus profile manifest, per-cell
  truths and per-PE ẑ) and archives as version 2; older binaries cannot read
  them. The manifest (profiled parameters, dataset, config, metric) must match
  for top-up, `merge-bank` and `merge-brazil`.
* Toy truths depend only on the cell footprint and the dataset, not on
  `--seed` or `-n`, so grid jobs with distinct seeds produce mergeable banks.
  Compute them once with `init-bank --n-pe-min 0` and ship that bank to the
  jobs if you want to skip the per-job truth fits.
* Without either `--profile-params` or `--fix`, a non-axis physics parameter is
  refused for models with more than two physics parameters (old banks pinned
  it at its default in the bank but profiled it in the prepass). Pass
  `--legacy-nuisance-pin` to reproduce those banks.

Two cautions. A profiled 3+1 parameter can have near-degenerate branches
(|U_e4|² below or above ½ give the same νe-disappearance amplitude), so ẑ can
jump between neighbouring cells, and two fitters can pick different branches
where they differ by ~0.01 in χ². Each profiled fit is seeded in every basin
of a coarse scan of z, but a very narrow basin against a unitarity wall can
still be missed. And every toy is dearer: the constrained fit now has a free
physics parameter, about 4x the pinned cost.

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

# 11. Fit regions — sideband-only fits and constrained predictions

A classic analysis check: fit **only** the sideband channels, then ask how
well the sideband-constrained prediction describes the channels the fit never
saw (the signal region, or a validation sample). `--fit-region` does this in
one step. Unlike PROjector it keeps the data outside the fit region, adds no
parameters (covariance systematics are constrained analytically, not promoted
to splines), and needs no second stage.

### Fitting only the sidebands

```bash
PROfit -x tutorial.xml -t TUT -o fr --seed 405 -n 8 \
    -i dmsq 1 sinsq2thme 0.01 \
    --fit-region "_numu_" global
# → TUT_fr_PROglobal_FitRegion.pdf (+ the usual TUT_fr_PROglobal_* outputs)
```

What happens: the channels matching `_numu_` (same pattern rules as
PROjector — unanchored regex, **whole channels** only) become the fit region,
and every χ² is restricted to their bins. The other channels' data is kept,
and physics parameters are fixed at CV (pass `--fit-region-float-physics` to
float them). The mask is honored everywhere — `profile`, `surface`, `fc`,
`fc-adaptive` all run as sideband-only fits — but only `global` and `plot`
produce the constraint report below.

Here the injected νe appearance signal lives only in the νe channels, so the
νμ sideband fit cannot see it; the νe channels are where it should show up.

### The constrained prediction and its χ²

After the fit, the default *evaluation region* (every channel outside the fit
region, here the two νe channels) is tested with the Gaussian conditional
constraint. Linearizing the prediction about the best fit, with the
Gaussian-prior splines and the covariance-type systematics together giving a
systematic covariance V and a prior prediction m, the evaluation bins S are
conditioned on the fit-region data d_B:

```
m_S'  = m_S + V_SB (C_B + V_BB)^-1 (d_B - m_B)      constrained prediction
V_SS' = V_SS - V_SB (C_B + V_BB)^-1 V_BS            constrained systematic covariance
χ²    = (d_S - m_S')^T (C_S + V_SS')^-1 (d_S - m_S')     ndf = |S|
```

C is the metric's own statistical variance (neyman: the data; CNP, pearson
as usual). Since the evaluation data never entered the fit, the χ² has one
degree of freedom per evaluated bin (minus one per channel with
`--shapeonly`). The log reports it before and after the constraint, overall
and per channel:

```
Conditioned on 68 fit-region bins; 14 Gaussian-prior spline(s) propagated, 0 held at the fit point.
Fit region, post-fit       : chi2 = 0 (ndf = 68 bins - 0 phys - 0 uniform - 0 shape = 68, p = 1)
Fit region, linearised     : chi2 = 0 (matches the fit chi2 when the model is ~linear about the fit point)
Eval region, pre-constraint : chi2 = 8.53389 / 32 (p = 0.999989)
Eval region, post-constraint: chi2 = 8.98349 / 32 (p = 0.99998)
  nu_ND_nue : pre 2.56499 / 16 (p = 0.999941)   post 2.82587 / 16 (p = 0.999887)
  nu_FD_nue : pre 5.39055 / 16 (p = 0.993444)   post 5.84422 / 16 (p = 0.989693)
```

This is Asimov fake data, so without the injection every χ² would be 0: the
evaluation χ² is purely the cost of the injected signal. It grows after the
constraint because the sideband shrinks the systematic band under the excess.
The same effect is visible bin by bin in the plot:

<img src="figures/TUT_fr_PROglobal_FitRegion.png" width="800"/>

*`TUT_fr_PROglobal_FitRegion.pdf`, page 2 — FD νe data against the
pre-constraint (hatched) and sideband-constrained (blue) prediction. The ratio
panel is data / constrained prediction. One page per evaluation channel, plus
a final page comparing the pre- and post-constraint systematic covariances.*

The histograms and both covariances are also written to `TUT_fr_PROglobal.root`
(`fit_region_*`). The ordinary post-fit plots (`TUT_fr_PROglobal_hists.pdf`)
show the constrained band in the evaluation channels too, since their post-fit
band is conditioned on the fit-region data.

Two built-in checks:

* **Linearity.** The "linearised" fit-region χ² should track the actual fit
  χ². A large gap means the splines are strongly non-linear about the fit point
  and the Gaussian constraint is approximate.
* **Closure.** With physics fixed, χ²_post-constraint = χ²_min(joint fit) −
  χ²_min(fit-region fit) for a linear model. Running the same data once with
  `--fit-region` and once without (plus `--fix` on the physics parameters) is a
  quick end-to-end check on a new setup.

### Options

* `--eval-region "FD_nue"` — evaluate only these channels (must not overlap
  the fit region; the default is everything outside it).
* `--blind-eval` — zero the data outside the fit region and report only the
  constrained prediction, e.g. before unblinding:

  ```
  nu_FD_nue : total events pre 58.4618 +/- 7.74117 (syst) -> post 58.4618 +/- 6.62346 (syst)
  ```
* `--fit-region-float-physics` — float physics in the sideband fit; the
  constraint is then evaluated at the fitted physics point.
* `plot` instead of `global` — no fit at all: the same constraint is evaluated
  analytically about the CV and drawn to `<TAG>_<out>_PROplot_FitRegion.pdf`.
  For a linear model this equals the fitted result.

### Rules and caveats

* Patterns must select whole channels; a pattern matching every channel, or a
  channel in both `--fit-region` and `--eval-region`, is refused.
* Splines that are `--fix`ed or have a uniform prior are held at their fit
  values; their uncertainty is not propagated into the evaluation region (a
  warning lists them).
* Bins with zero statistical variance (zero data under neyman) are drawn but
  left out of the χ², exactly as the fit metric does.
* `-c poisson` has no covariance term, so the evaluation χ² is Gaussian with
  pearson-style (prediction) statistical errors.
* PROjector's stage-1 pre-fit uses the same fit-region machinery with the
  remaining data blinded; use PROjector when you need the saved posterior as a
  prior for downstream FC/surface studies, `--fit-region` for a sideband →
  signal-region comparison.

---

# 12. PROletariat — grid submission: `proletariat`

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

# Appendix B: available physics models (`inc/PROmodel.h`, `inc/PROmodels/`)

> Implementation note (September 2026): every single-Δm² sine-kernel model below
> (the 2-flavour family, all `SBL_3+1_*` variants except `SBL_3+1+decay_invis`,
> and the NC-disappearance models) is defined as a *recipe* — parameter/channel
> tables plus amplitude and analytic-gradient formulas — evaluated by one shared
> `PROsineModel` class (`inc/PROmodels/PROmodelSine.h`). This changed no
> parameters or numbers (verified bitwise), but it is where to look for, or add
> to, these models' code. All of them support the default analytic-gradient fit
> mode with closed-form derivatives.

The `<model tag="...">` in your XML selects one of the models below.

### The tag naming convention

Model tags follow a fixed scheme:

```
regime_model_parameterization(_NC)
```

* **regime** — the oscillation approximation the model lives in: `SBL`
  (short-baseline, single-Δm² sine kernel, no matter effects) or `LBL`
  (long-baseline three-flavour). Regime-free utility models (`null`,
  `template`) carry no prefix.
* **model** — the physics hypothesis: `2flav` (one effective two-flavour-like
  amplitude per channel), `3+1`, `3+2`, `3+1+decay` (3+1 plus sterile decay),
  `3nu-matter` (standard three-flavour with Earth-matter effects) or
  `3nu-vacuum` (the same three-flavour physics in vacuum). A hyphen
  qualifies *within* a slot (`3nu-matter` vs `3nu-vacuum`), just as `+`
  does in `3+1`.
* **parameterization** — which parameter set the *same* physics is fitted in:
  `Usq` (squared mixing-matrix elements |Ue4|², |Uμ4|², …), `angles`
  (sin²2θ₁₄, sin²θ₂₄, …), or the name of the headline amplitude that becomes a
  direct scan axis (`sinsq2thee`, `sinsq2thmumu`, `sinsq2thmue`). For the
  2-flavour family this slot names the channel (`numudis`, `nueapp`, `nuedis`).
* **`_NC` suffix** — the NC-disappearance extension of the same
  model/parameterization: extra νs/ντ probability channels (and, for the 3+1
  family, one extra sterile/tau-split parameter).

Legacy pre-v3.1 tags (`numudis`, `nueapp`, `3+1`, `3+1_angles`, `3+1_3A/3B/3C`,
`NCnumudisapp`, `NCdisapp`, the `_NC` and decay spellings, `nullmodel`,
`template_fit`, `LBL`, `3+2`) **keep working as deprecated aliases**: they are
mapped to the canonical tag with a one-time warning (see
`canonicalize_model_tag` in `src/PROmodel.cxx` for the full table). New configs
should use the canonical names; each model section below quotes its legacy
alias.

A few conventions apply to all models:

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
  model block pointing at the true-L/E variable (exceptions: `numudisTEST`
  takes `name="L"` + `name="E"`, the LBL models take {L,E}, E+`baseline=`,
  or — vacuum only — a signed L/E, and `template` takes none; see below).
  For the LBL models negative E (or L/E) marks an antineutrino event.

### `null` *(legacy: `nullmodel`)*

No physics parameters at all — rule 0 only. For pure-systematics setups
(constraint studies, template validation) where nothing oscillates. This is
also the default when the `<model>` block has no `tag=` attribute.

### `SBL_2flav_numudis` *(legacy: `numudis`)* — 3+1 νμ disappearance

P(νμ→νμ) = 1 − sin²2θμμ · sin²(1.267 Δm² L/E).

| # | name | meaning | fit space | bounds | default |
|---|---|---|---|---|---|
| 0 | `dmsq` | Δm²₄₁ [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `sinsq2thmm` | sin²2θμμ | log10 | 0 – 1 | 10⁻¹⁰ |

Rules: **0** = no osc, **1** = νμ survival (apply to your νμ-CC signal).

### `numudisTEST` — two-variable validation twin

Identical physics and parameters to `SBL_2flav_numudis`, but built on two separate
model variables `L` and `E` (both must be `<parameter>` entries) instead of
one `L/E`. Exists to validate the machinery — the two should produce
identical spectra.

### `SBL_2flav_nueapp` *(legacy: `nueapp`)* — 3+1 νμ→νe appearance

P(νμ→νe) = sin²2θμe · sin²(1.267 Δm² L/E). The model used throughout this
tutorial.

| # | name | meaning | fit space | bounds | default |
|---|---|---|---|---|---|
| 0 | `dmsq` | Δm²₄₁ [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `sinsq2thme` | sin²2θμe | log10 | 10⁻¹⁰ – 1 | 10⁻¹⁰ |

Rules: **0** = no osc, **1** = νμ→νe appearance (apply to the fullosc
subchannel).

### `SBL_2flav_nuedis` *(legacy: `nuedis`)* — 3+1 νe disappearance

| # | name | meaning | fit space | bounds | default |
|---|---|---|---|---|---|
| 0 | `dmsq` | Δm²₄₁ [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `sinsq2thee` | sin²2θee | log10 | 0 – 1 | 10⁻¹⁰ |

Rules: **0** = no osc, **1** = νe survival.

### `SBL_3+1_Usq` *(legacy: `3+1`)* — full three-parameter 3+1 (mixing elements)

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

### `SBL_3+1_angles` *(legacy: `3+1_angles`)* — 3+1 in mixing angles

Same physics and rules (0–3) as `SBL_3+1_Usq`, re-parameterized in angles, with
the unitarity constraint.

> ⚠️ **Results from before August 2026 are invalid for this model.** Its
> P(νμ→νμ) was missing the factor 4 in sin²2θμμ = 4|Uμ4|²(1−|Uμ4|²), so any
> pre-fix fit/contour/FC study involving νμ disappearance had a 4× weaker
> disappearance amplitude (old sin²θ₂₄ exclusion curves ~×4 too weak). The
> binary-cache hash only covers the XML, **not the code**, so old
> `_mesh.bin`/`_bank.bin`/`_brazil.bin` and PROjector constraint files made
> with this model load silently into fixed builds — regenerate them all.
> (`_prop.bin`/`_syst.bin` contain no probabilities and are fine; the other
> 3+1 variants and the 2-flavor models were verified unaffected.)

| # | name | meaning | fit space | bounds | default |
|---|---|---|---|---|---|
| 0 | `dmsq` | Δm²₄₁ [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `sinsq2th14` | sin²2θ₁₄ | log10 | 0 – ~1 | 10⁻⁸ |
| 2 | `sinsqth24` | sin²θ₂₄ | log10 | 0 – ~1 | 10⁻⁸ |

### `SBL_3+1_sinsq2thee` / `SBL_3+1_sinsq2thmumu` / `SBL_3+1_sinsq2thmue` *(legacy: `3+1_3A` / `3+1_3B` / `3+1_3C`)* — headline-amplitude 3+1 variants

Full 3+1 physics (rules **0–3** as above), re-parameterized so that the
*effective amplitude you want to plot* — the one each tag is named after —
is a direct scan axis and the leftover mixing freedom is a single
nuisance-like third parameter that PROfit profiles smoothly at each grid
point (see the surface-section note on parameterizations).

**`SBL_3+1_sinsq2thee`** — sin²2θee-focused; ratio parameter is exactly sin²θ₂₄
(= sin²2θμe / sin²2θee). No extra constraint needed.

| # | name | fit space | bounds | default |
|---|---|---|---|---|
| 0 | `dmsq` [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `sinsq2thee` | log10 | 0 – ~1 | 10⁻⁸ |
| 2 | `sinsqth24` | log10 | 0 – ~1 | 10⁻⁸ |

**`SBL_3+1_sinsq2thmumu`** — sin²2θμμ-focused; `sB` is the ratio
sin²2θμe / sin²2θμμ (= |Ue4|² / (1−|Uμ4|²)). Unitarity constraint enforced.

| # | name | fit space | bounds | default |
|---|---|---|---|---|
| 0 | `dmsq` [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `sinsq2thmumu` | log10 | 0 – ~1 | 10⁻⁸ |
| 2 | `sB` | log10 | 0 – ~1 | 10⁻⁸ |

**`SBL_3+1_sinsq2thmue`** — sin²2θμe-focused (the one to use for direct appearance
contours); `xi` is the log geometric ratio ξ = ½·log(|Ue4|²/|Uμ4|²)
(ξ = 0 ⇒ equal mixing, ξ > 0 ⇒ electron-dominated). Unitarity constraint
√(sin²2θμe)·cosh(ξ) < 1.

| # | name | fit space | bounds | default |
|---|---|---|---|---|
| 0 | `dmsq` [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `sinsq2thmue` | log10 | 0 – ~1 | 10⁻⁸ |
| 2 | `xi` | **linear** | −10 – 10 | 0 |

### NC-disappearance models (September 2026)

NC samples measure the **total active-neutrino rate**: oscillation to ντ still
produces NC events, so only the *sterile* fraction of the disappeared flux is a
real NC deficit. The `_NC` variants below extend their siblings' rules 0–3 with
four NC/tau channels:

* **4** = νμ→νs (visible NC deficit, `1 − 4|Uμ4|²|Us4|² sin²Δ`) — apply to a
  νμ-sourced NC sample; **5** = νe→νs (same for νe-sourced NC);
* **6** = νμ→ντ appearance (`4|Uμ4|²|Uτ4|² sin²Δ`) and **7** = νe→ντ — for
  CC-ντ samples, if you have any.

Since |Us4|² + |Uτ4|² = 1 − |Ue4|² − |Uμ4|² exactly, each family fits the
sterile/tau **split** with one extra parameter in its own language:

| model *(legacy)* | base params (identical to sibling) | split parameter | fit space | bounds | default |
|---|---|---|---|---|---|
| `SBL_3+1_Usq_NC` *(3+1_NC)* | `dmsq`, `Ue4^2`, `Um4^2` | `Uta4sq` = \|Uτ4\|² | log10 | 0 – ~1 | 10⁻⁸ |
| `SBL_3+1_angles_NC` *(3+1_angles_NC)* | `dmsq`, `sinsq2th14`, `sinsqth24` | `cosq34` = cos²θ₃₄ | **linear** | 0 – 1 | **1** |
| `SBL_3+1_sinsq2thee_NC` *(3+1_3A_NC)* | `dmsq`, `sinsq2thee`, `sinsqth24` | `cosq34` | **linear** | 0 – 1 | **1** |
| `SBL_3+1_sinsq2thmumu_NC` *(3+1_3B_NC)* | `dmsq`, `sinsq2thmumu`, `sB` | `cosq34` | **linear** | 0 – 1 | **1** |
| `SBL_3+1_sinsq2thmue_NC` *(3+1_3C_NC)* | `dmsq`, `sinsq2thmue`, `xi` | `Uta4sq` | log10 | 0 – ~1 | 10⁻⁸ |

`cosq34 = 1` (the default) sends all of the off-diagonal strength to sterile —
maximal NC deficit, no ντ appearance; `cosq34 = 0` is the opposite. In
`SBL_3+1_Usq_NC` / `SBL_3+1_sinsq2thmue_NC` the tau element is fitted directly and
|Us4|² = max(0, 1 − Σ|Uα4|²); their unitarity constraints extend to include
|Uτ4|². An `_NC` model whose XML routes **no** branch to rules 4–7 reproduces
its non-NC sibling's spectra exactly (verified bitwise), so you can switch a
config to the `_NC` tag before the NC sample exists without changing results —
only the parameter count (and thus fit trajectories/seeds) differs.

Two simpler **phenomenological** NC models treat the deficit amplitudes as free
parameters (no joint unitarity between them — fine for limit plots, not an
exact 3+1 sub-model):

**`SBL_2flav_numudis_NC`** *(legacy: `NCnumudisapp`)* — single channel,
P(νμ→active) = 1 − sin²2θμs·sin²Δ.
Rules: **0** = no osc, **1** = NC deficit.

| # | name | fit space | bounds | default |
|---|---|---|---|---|
| 0 | `dmsq` [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `sinsq2thms` | log10 | 0 – 1 | 10⁻¹⁰ |

**`SBL_2flav_nudis_NC`** *(legacy: `NCdisapp`)* — independent νμ- and νe-sourced
deficits. Rules: **0** = no osc, **1** = νμ NC deficit, **2** = νe NC deficit.

| # | name | fit space | bounds | default |
|---|---|---|---|---|
| 0 | `dmsq` [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `sinsq2thms` | log10 | 0 – 1 | 10⁻¹⁰ |
| 2 | `sinsq2thes` | log10 | 0 – 1 | 10⁻¹⁰ |

### `SBL_3+1+decay_invis` *(legacy: `3+1_decay_invis`)* — 3+1 with invisible sterile decay

The `SBL_3+1_Usq` mixing-element model extended with an invisible-decay
coupling; combined unitarity + positivity constraint (g² = 0, the pure-3+1
limit, is accepted since August 2026 — it was previously rejected by a strict
inequality, which biased fits near the no-decay boundary). Rules **0–3** as
in `SBL_3+1_Usq`. Visible-decay variants exist as `SBL_3+1+decay_vis1`
*(legacy: `3+1_decay_vis_model1`; coupling `g_phi`, paired osc/decay
subchannels)* and `SBL_3+1+decay_vis2` *(legacy: `3+1_decay_vis_model2`;
coupling `g_e`, needs separate `L` and `E` parameters)*.

| # | name | meaning | fit space | bounds | default |
|---|---|---|---|---|---|
| 0 | `dmsq` | Δm²₄₁ [eV²] | log10 | 10⁻² – 10² | 10⁻² |
| 1 | `Ue4^2` | \|Ue4\|² | log10 | 0 – ~1 | 10⁻⁸ |
| 2 | `Um4^2` | \|Uμ4\|² | log10 | 0 – ~1 | 10⁻⁸ |
| 3 | `g2` | decay coupling g² | **linear** | 0 – 10 | 0 |

### `SBL_3+2_Usq` *(legacy: `3+2`)* — two-sterile model

Seven parameters with a CP phase, so ν and ν̄ appearance differ. Unitarity
constraints: |Ue4|²+|Ue5|² ≤ 1, |Uμ4|²+|Uμ5|² ≤ 1, and the e and μ rows
orthogonal, |Ue4Uμ4* + Ue5Uμ5*|² ≤ (1−|Ue4|²−|Ue5|²)(1−|Uμ4|²−|Uμ5|²)
(φ₅₄-dependent; the 3+1 limit is |Ue4|²+|Uμ4|² ≤ 1). The older appearance
cap 4(|Ue4||Uμ4| + |Ue5||Uμ5|)² < 1 is also still applied; it is stricter
than unitarity when cos φ₅₄ < 0.

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

### `LBL_3nu-matter_angles` *(legacy: `LBL`)* and `LBL_3nu-vacuum_angles` — full three-flavour long-baseline (NuFastLBL)

Standard 3ν oscillations, either **including Earth-matter effects**
(NuFastLBL's `Probability_Matter_LBL` with a constant-density profile) or
**in vacuum** (`Probability_Vacuum_LBL` — cheaper, and numerically safe as
Δm²₃₁ crosses zero, where the matter solver has a singular point near
Δm²ee = 0). Both tags share one implementation and one parameter set, all
fitted in **linear** space with bounds spanning the global-fit allowed ranges,
so you can flip a config between them to compare matter and vacuum directly.

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

The kinematic inputs are **per-event `<variable>`s**, like every other
oscillation model; E is always **signed** (negative = antineutrino, flipping
δ_CP and the MSW potential), and E = 0 or L/E = 0 is the no-oscillation
limit. Three layouts are accepted:

| `<parameter>`s | matter | vacuum | `baseline=` attribute |
|---|---|---|---|
| `L` [km] + `E` [GeV], both per-event | exact 2D | works | **conflict — fatal** |
| `E` only | L fixed at `baseline=`, 1D grid | ratio = baseline/E | **required** |
| `L/E` [km/GeV, signed] only | fatal (MSW ∝ E; the ratio underdetermines it) | exact 1D | fatal (redundant) |

Most fixed-baseline experiments want the **E-only** form — it is exact for a
constant baseline and keeps the evaluation grid one-dimensional. Use the
`{L, E}` pair only when the baseline genuinely varies across the MC.

⚠ *Before v3.0.4-dev the model took one `"L/E"` parameter that was silently
read as E [GeV] at a fixed L = 1300 km — older LBL configs must migrate to
the inputs above.*

⚠ **Grid memory (`{L, E}` pair only)**: with two kinematic variables the
binned-evaluation grid is n_L × n_E **global** bins (bin count × number of
subchannels for each) × 10 probability components × every variable's reco
bins. Keep the L and E truth binnings lean — a 200×20-bin pair on a
many-subchannel config allocates tens of GB and will OOM; ~10 bins each is
usually plenty. The E-only and L/E forms have no such blow-up.

`<model>` tag attributes (on a non-LBL tag they are a config error; the
vacuum tag ignores the three matter-only ones with a warning so a matter
config can be re-tagged unchanged):

| attribute | meaning | allowed | default |
|---|---|---|---|
| `baseline=` | fixed baseline L [km] (E-only mode; both regimes) | > 0 | — (required with E-only) |
| `density=` | constant matter density ρ [g/cm³] (matter only) | ≥ 0 | 3 |
| `electron_fraction=` | electron fraction Yₑ (matter only) | (0, 1] | 0.5 |
| `n_newton=` | NuFast Newton iterations (matter only; 1 recommended for many-year DUNE/HK precision) | integer 0–10 | 0 |

```xml
<model tag="LBL_3nu-matter_angles" baseline="1300" density="2.848" n_newton="1">
    <parameter name="E" variable_index="2"/>
    ...
</model>
```

Notes:

* NuFast is a **constant-density** solver: there is no layered/PREM profile.
  Use your experiment's line-averaged crustal density — commonly ≈2.6 g/cm³
  for T2K/HK, ≈2.84 for NOvA, 2.848 for DUNE (the default 3 is the legacy
  hardcoded value, kept for continuity). `density="0"` works but warns —
  prefer the vacuum tag, which takes a different (safer) code path.
* The **matter** solver (DMP zeroth order) is singular in a narrow band of
  dmsq_31 around 0 and around Δm²ee = 0 (≈2×10⁻⁵ eV²); the model rejects it
  via `model_constraint` (like 3+1 unitarity), so fits and scans skip those
  points automatically. The vacuum solver has no such band.
* Like every model setting, these attributes are **not** part of the XML
  hash: changing ρ re-uses the `_prop/_syst.bin` caches (correct — the
  model never affects them) but silently *invalidates* any FC.root, AFC
  `_mesh/_bank/_brazil.bin` or PROjector constraint made with the old values
  (the same artifact class as gotcha 12 in CLAUDE.md) — regenerate those.

### `template` *(legacy alias: `template_fit`)* — per-subchannel normalization fit

Not an oscillation model: each `<parameter>` in the model block is one
linear scale parameter (bounds from its `min`/`max` attributes, default 1)
multiplying the normalization of one or more **subchannels**; every
non-floated subchannel stays fixed at 1. `model_rule` is ignored — events are
routed by subchannel membership. Needs no L/E variable. Useful for
sideband/template fits, cross-section-style normalization studies, and
signal-strength fits of a predicted signal template.

Which subchannels a parameter scales:

* **No `subchannels=` attribute** (the original form): `name` *is* the exact
  subchannel fullname, and the parameter floats that one subchannel.
* **`subchannels="<regex>"`**: the parameter floats **every** subchannel whose
  fullname the regex matches (unanchored, like every pattern in PROfit —
  anchor with `^…$` for exact names), all driven by the *same* scale. `name`
  is then just the parameter's name, the one `--fix`, `--inject`,
  `--inject-cv` and `surface --xvar/--yvar` use.

```xml
<model tag="template">
  <!-- one signal strength for both beam modes: the fit tests the predicted nubar/nu ratio -->
  <parameter name="mu_signal" subchannels="^(nu|nubar)_MiniBooNE_nue_oscnue$" min="0" max="50" default="0"/>
  <!-- the original exact-name form still works alongside it -->
  <parameter name="nu_MiniBooNE_numu_background" min="0.5" max="2"/>
</model>
```

`default="<value>"` (must lie in `[min, max]`) moves the parameter's central
value off 1. With `default="0"` the CV, the background-only fixed seed of
`global`/`profile`, the `--fix` pin and the default Asimov fake data all sit
at the **no-signal** hypothesis — the natural null for a signal strength.
`--inject mu_signal 1` then makes the nominal-signal Asimov.

Refused at parse or model construction: a `subchannels=` pattern that
matches no subchannel, an empty pattern, a subchannel claimed by two
parameters (by pattern or by exact name), two parameters with the same
name, a `default=` outside `[min, max]`, and `subchannels=`/`default=` on
any model other than `template`. As with every model setting the `<model>`
block is not part of the `_prop/_syst.bin` hash, so editing it reuses those
caches.

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
Σ = U S Uᵀ                                  (eigendecomposition, double precision)
L = U·√S        largest mode first; rounding-level modes dropped (zero columns)
```

The only modes dropped are those below `n·ε_double·max(S)`, i.e. numerical
zeros of a rank-deficient Σ (and the slightly negative noise eigenvalues
rounding produces). Σ's genuine spectrum spans ~10⁻¹² of its largest
eigenvalue — MC-stat and systematics on low-count bins sit next to
high-count ones — so a looser cutoff would silently delete real variance in
exactly the bins where the statistical error is smallest. With that, `Σ = L·Lᵀ`
to rounding, and a random
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
`N = 2500` throws `i` (configurable with `plot --band-throws N`):

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

`C` is the fit metric's **own** statistical covariance (its per-bin variance,
`PROmetric::GetStatVariances`), so the reconstruction uses the same
statistical model the fit did:

| `-c` | `C_b` | in the chain |
|---|---|---|
| `neyman` | `d_b` | constant |
| `CNP` | `3/(1/d_b + 2/μ_b)`, `μ_b/2` if `d_b = 0` (μ = physics-only CV) | constant (physics is pinned in the band chain) |
| `pearson` | `S_b`, the sample's own prediction | re-evaluated at every chain point |

Restrict to the **contributing bins** `B`: bins that are active
(`PROconfig::SetActiveBins` mask) **and** have `C_b > 0` — exactly the bins
the fit's χ² uses (neyman drops zero-data bins; CNP and pearson keep every
active bin), so the reconstruction is constrained by the same information the
fit was — PROjector-masked channels, for example, cannot pull on anything. On
those bins define:

```
C   = diag( C_b )                         statistical covariance, b ∈ B
L_B = rows of L in B, zero columns dropped   (n_B × k)
A   = 1_k + L_Bᵀ C⁻¹ L_B                  (k × k, in double; refactorized only when C changes)
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

**Legacy variant.** `global --legacy-postfit-error` (also on `profile`) skips
Step 2 entirely and reverts to the pre-v3.0 recipe: the spline posterior of
Step 1 is kept, but each sample gets a *prior* covariance throw `L·g`
(exactly as in C.1), the center shift is zero, and the curve/band sit on the
raw best-fit spectrum with the plain `Best-Fit` label, drawn **green**
(RGB 52,168,83) instead of the usual post-fit red so a legacy plot is
recognizable at a glance. This double-counts the
data's constraint on the covariance systematics (the χ² already marginalized
them) and is **not statistically correct** — it exists for systematics
studies and for reproducing pre-v3.0 plots.

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
* `C` is the fit metric's own statistical variance, so neyman, CNP and pearson
  bands all condition with the χ² they were fitted with. With no free splines
  (the analytic path) the logged `uᵀ(C+Σ)⁻¹u` at the best fit reproduces that
  fit χ² (to float rounding).
  `poisson` has no covariance term; covariance systematics are refused there.
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
`L`), contributing a pull term `αᵀα`. With `C` the statistical covariance
(it may depend on θ, as pearson's does, but never on α — so everything below
holds at each fixed θ) and `u ≡ d − P(θ)` the residual, the χ² with
everything explicit is

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

# Appendix D: shape-only fits (`--shapeonly`), in full

A shape-only analysis asks: *does the data have the right spectral shape,
whatever its overall rate?* Every channel's normalisation is thrown away, and
only the distribution of events across bins is compared.

```bash
PROfit -x cfg.xml -t TUT --shapeonly global     # also profile / surface / fc / fc-adaptive / plot
```

The short version: **the systematics stay physical, and two operations inside
the fit do the shape-only work.**
1. The prediction is rescaled onto the data in every channel. This handles the
   splines and the physics.
2. The covariance is projected onto shape (the MiniBooNE M_shape). This handles
   the covariance systematics.

Both are recomputed at every χ² evaluation, about the prediction being tested.

### D.0 What "normalisation" means here

- **Per collapsed channel.** Normalisation is removed separately for every
  (mode, detector, channel) block, the same blocks the χ² is built from. Each
  block has its own free normalisation.
- **ND and FD are separate blocks.** Their relative rate therefore carries no
  information in a shape-only fit. In SBN that ratio is most of the
  νμ-disappearance power, so decide whether that is what you want. The same
  applies between channels: the νe/νμ rate ratio is discarded too.
- **Everything in a channel counts.** That includes beam subchannels,
  cosmic/offbeam subchannels (`incl_systematics="false"`), and oscillated
  samples at their *current* oscillated weight. A systematic that moves only
  the beam events still changes the beam-vs-cosmic *mix*, and that mix is
  shape.
- **One degree of freedom is lost per normalised channel.**

### D.1 The χ², step by step

Notation:

```
s(θ)      the uncollapsed prediction at parameters θ (physics + spline knobs)
P = Tᵀs   the collapsed prediction;  d = data;  c = one channel
D_c = Σ_{b∈c} d_b      the data total in channel c
S_c = Σ_{b∈c} P_b      the prediction total in channel c
```

1. **Predict.** `s(θ)` = event weights × oscillation probability × the product
   of every spline's response `S_k(θ_k, bin)`. This is exactly the non-shape
   prediction.
2. **Rescale onto the data** (`ShapeRescaleToData`, `inc/PROtocall.h`). Every
   bin of channel c, in every subchannel, is multiplied by the same factor:
   `s̃ = s · D_c / S_c`. The data is never touched.
3. **Covariance** (D.2). `M = stat + R·Σ(s̃)·Rᵀ`, where `Σ(s̃)` is the usual
   systematic covariance scaled by the rescaled prediction.
4. **χ²** `= (P̃ − d)ᵀ M⁻¹ (P̃ − d) + Σ_k ((θ_k − c_k)/σ_k)²`. The second term
   is the spline pulls.

Consequences:
- The χ² is **exactly** invariant under s → k·s, so the overall rate is
  invisible.
- The statistical term never moves with the fit.
- The analytic gradient carries both the rescale and the projector derivatives,
  and is checked against finite differences by `scale-test --tests gradcheck`.

### D.2 Covariance systematics: the MiniBooNE decomposition

A covariance matrix M can be split into normalisation, mixed and shape parts:
`M = M_shape + M_mixed + M_norm`. Shape-only keeps only `M_shape`. With
`R = I − P̃·1ᵀ/S_c` on each channel block (`ShapeProjectorCollapsed`):

```
(R M Rᵀ)_ij = M_ij − (P_i/S) Σ_k M_kj − (P_j/S) Σ_k M_ik + (P_i P_j/S²) Σ_kl M_kl  =  M_shape
```

It is rebuilt about the **current** rescaled prediction at every evaluation, as
MiniBooNE rebuilt it at each oscillation point. The projection covers every
covariance source: universes, flat, `norm_to_covariance`, external, MC-stat,
and the residuals of covariance→spline decompositions.

The statistical term is *not* projected, and it does not need to be. The
residual `P̃ − d` sums to zero in every channel, and the conditional-Gaussian
identity then makes `stat + M_shape` give the correct χ² for the renormalised
comparison. For `-c pearson` (and `poisson`) this is exact: the multinomial
statistics. For `neyman` it is the usual approximation.

### D.3 Spline systematics: the rescale does it

Splines are **never projected**. They act on the prediction in step 1, and the
rescale in step 2 removes their normalisation part, exactly and at every knob
value. To first order in a knob change,

```
δs̃ = (D/S) · R · δs        (R = I − s·1ᵀ/S)
```

This is the same projector D.2 applies to the covariance. So splines and
covariances get one consistent definition of "shape"; for splines it is simply
exact rather than linearised.

| Knob kind | What the fit sees |
|---|---|
| Pure normalisation of a whole channel (cosmics included) | Nothing: `s̃` does not move. Only its pull acts, so it sits at its prior. |
| Scales the beam events only | The beam-vs-cosmic mix shifts. That is kept as shape. |
| Normalisation + shape | The shape part is kept exactly; the normalisation part is gone. |
| Restricted to some subchannels (`apply_to_subchannel`) | The composition shift is kept. |

A worked example. Take one channel with bin 1 = 90 beam events and bin 2 = 10
cosmic events, and a data total of 100.

| Knob | Raw prediction | After rescale to 100 | Effect vs CV (90, 10) |
|---|---|---|---|
| beam +10% | (99, 10) | (90.8, 9.2) | 0.8 events move into bin 1: shape, kept |
| everything +10% | (99, 11) | (90, 10) | none: pure normalisation |

### D.4 Error bands under shape-only (and `--area-norm`)

An area-normalised plot has no normalisation freedom, so its band must carry
shape uncertainty only. `--shapeonly` implies `--area-norm`, and the band
builders then do the following.

- **Pre-fit band** (`getErrorBand`):
  1. Throw every knob `θ_k ~ N(c_k, σ_k)`, build the spectrum, and add a
     correlated covariance throw.
  2. **Rescale that throw per channel onto the CV's channel totals.**
  3. Take the 16/84% quantiles per bin as usual (Appendix C).

  A knob that only scales a whole channel contributes no width; its shape
  effect contributes.
- **Post-fit band, Constrained Best-Fit, covariance-pull plot**
  (`getMCMCErrorBand`, `getCovarianceOnlyErrorBand`,
  `plotCovariancePosteriorPulls`). They condition *exactly as the fit does*:
  - the best fit and every posterior sample are rescaled onto the data;
  - the covariance is taken at that scale and projected onto shape;
  - the conditioning residual is `d − P̃`.

  Each finished sample is then renormalised for display.
- **`--area-norm` without `--shapeonly`.** The fit is absolute, but the plot is
  area-normalised. The bands are shape bands as above, and the χ² label, which
  is still the absolute χ², gets a small grey **"absolute χ²"** tag. Under
  `--shapeonly` the tag reads **"shape-only"**.
- **Detector and channel ratio pages** (`ND / FD`, and `--plot-ratios`).
  Under `--area-norm` (so also `--shapeonly`) each spectrum is normalised to
  unit area per channel *before* dividing, for the CV, the best fit and the
  data alike, and the axis reads "(area normalized)". The bands are shape
  bands, so a ratio of raw rates would show offsets that neither the band nor
  a shape-only χ² accounts for. The ratio therefore sits around 1 by
  construction: only its energy dependence is tested, not the ND/FD rate.
- **Fractional-systematics breakdowns and covariance plots.** These are the
  `_fractional_systematics.pdf` family, `_PROplot_Covar.pdf` and the ROOT
  `Covariance/` directory. Under `--shapeonly`, `plot` projects every
  systematic's matrix onto shape about the CV first, so they show exactly the
  M_shape the fit uses. Every such page carries a light-grey "shape-only" note.
  Expect big changes relative to the full plots:
  - normalisation-dominated systematics shrink;
  - a systematic confined to part of the spectrum grows in the bins it never
    touches, because renormalising pushes them the opposite way;
  - each systematic has pivot bins where its shape error passes close to zero;
  - the covariance acquires negative (anti-correlated) entries.

  Under `--area-norm` alone these plots stay full, because the fit is absolute.

### D.5 What the systematics object does, and what it used to do

The `PROsyst` is built **physical**, identical to a non-shape run, with one
exception. When a covariance is decomposed into spline knobs
(`covariance_to_spline`, `_uniform`, external, PROjector promotion), it is
first projected onto shape about the physical null prediction: every event,
cosmics included, at the model's default oscillation. Knobs are then never
spent on, or left free along, a normalisation the fit cannot see.


# Appendix E: counting degrees of freedom (ndf)

```
ndf = n_bins − n_free_physics − n_free_uniform − n_shape
```

Computed by `PROmetric::GetNdof()` (`inc/PROmetric.h`) at the post-fit
minimum:

* **`n_bins`**: the bins that actually enter the χ² sum, i.e. active
  (fit-region mask, including PROjector's) and with positive statistical
  variance. That depends on the metric:
  * `neyman`: zero-data bins drop out, since their variance *is* the data;
  * `CNP`, `pearson`, `poisson`: every active bin counts.
* **`n_free_physics`**: physics parameters that are not fixed.
* **`n_free_uniform`**: free `prior_type="uniform"` splines. They have no
  pull term, so each one is a genuine free parameter.
* **Gaussian-prior splines net zero**, whether the prior comes from XML
  `prior=`, `<correlation>`s or a PROjector constraint. The pull term is one
  pseudo-measurement that cancels the parameter.
* **Covariance systematics count zero.** They are marginalised inside the
  covariance matrix and are not fit parameters.
* **"Fixed"** means pinned by `--fix`, `--syst-only` or a scan, i.e. zero-width
  bounds.
* **`n_shape`** (`--shapeonly` only): one per channel with at least one
  contributing bin, for its lost normalisation. Channels are per detector, so
  shape-only also gives up the ND/FD rate ratio.

The log line reads e.g. `ndf = 100 bins - 2 phys - 0 uniform - 0 shape = 98`.
Only the global post-fit label is a χ²/ndf. The per-channel and 2D
projection labels are fixed-point comparisons with no pull and no parameter
accounting, so they read χ²/nbins. A pathological setup can give ndf ≤ 0.

---

# Appendix F: post-fit pull distributions, in full

This appendix explains what the pull-distribution pages
([section 6](#are-the-pulls-healthy-pull-distributions)) plot and why. The
central result is due to L. Demortier and L. Lyons, *"Everything you always
wanted to know about pulls"*, CDF/ANAL/PUBLIC/5776 (2002). F.1–F.3 derive it
three ways. F.4–F.5 extend it to PROfit's many correlated parameters and show
how σ_post is computed, F.6 covers reading a whole histogram, F.7–F.8 the
bounded and uniform pages, and F.9 the limits. The code is
`computePostfitPulls` / `plotPostfitPullDistributions` (src/PROpulls.cxx).

### F.0 What a pull should measure

A nuisance θ has prior θ₀ ± σ_prior, and the fit returns θ̂ ± σ_post. The
question is whether θ̂ landed where we should expect it. The naive pull is
`(θ̂ − θ₀)/σ_prior`. Look at its two extremes:

* **The data says nothing about θ.** The fit leaves θ at θ₀ every time, so
  the naive pull is identically 0: a spike, not a Gaussian. Here
  σ_post = σ_prior.
* **The data measures θ perfectly.** Then θ̂ is the true value, and if the
  prior is honest the true value is a draw from N(θ₀, σ_prior). The naive
  pull is N(0,1). Here σ_post → 0.

Real systematics sit in between, so each naive pull has its own width
somewhere between 0 and 1, set by how well the data constrains that
parameter. A histogram of naive pulls has no fixed expected shape. Dividing
each one by the width it actually has fixes that, and that width turns out to
be √(σ²_prior − σ²_post).

### F.1 The prior is a measurement

Demortier and Lyons' reading: a Gaussian prior is the same thing as an
auxiliary measurement. Someone measured θ, got `a = θ₀`, and quoted the error
σ_prior. The fit combines `a` with the data, and θ̂ ± σ_post is the best
combination. The pull compares the auxiliary measurement with the combined
result, `a − θ̂`. These two are correlated, because θ̂ contains `a`, so the
error on their difference is not √(σ²_prior + σ²_post).

The key fact is that for the best (minimum-variance) combination, the
leftover `a − θ̂` is uncorrelated with θ̂. The proof is one line. The quantity
`θ̂ + λ(a − θ̂)` is still an unbiased estimate of θ (since `a − θ̂` averages to
zero), and its variance is

```
Var(θ̂ + λ(a − θ̂)) = Var(θ̂) + 2λ·Cov(θ̂, a − θ̂) + λ²·Var(a − θ̂)
```

If the covariance were non-zero, a small λ of the right sign would beat θ̂,
contradicting that θ̂ is the best combination. So `Cov(θ̂, a − θ̂) = 0`.

Now `a` splits into two uncorrelated pieces, `a = θ̂ + (a − θ̂)`, and their
variances add, like the sides of a right triangle:

```
Var(a)    = Var(θ̂) + Var(a − θ̂)
σ²_prior  = σ²_post + Var(a − θ̂)
       ⇒   Var(a − θ̂) = σ²_prior − σ²_post
```

So the pull with unit width divides by √(σ²_prior − σ²_post). PROfit plots it
with the sign flipped, so that positive z means the fit moved the parameter
up:

```
z = (θ̂ − θ₀) / √(σ²_prior − σ²_post)
```

This holds when θ̂ really is the best combination and σ_post its error. That
is exactly true of a χ² minimum in a model linear in its parameters with
Gaussian errors (the Gauss–Markov theorem), which is the approximation used
throughout (F.9).

### F.2 Checking it by hand, and what z means

Take one nuisance and let the data alone measure it as x ± σ_d. σ_d is how
well the data would pin θ with no prior at all. The combination is

```
1/σ²_post = 1/σ²_prior + 1/σ²_d
θ̂         = σ²_post · (θ₀/σ²_prior + x/σ²_d)
⇒  θ̂ − θ₀ = (σ²_post/σ²_d) · (x − θ₀)
```

x and θ₀ are independent measurements, so `Var(x − θ₀) = σ²_d + σ²_prior`.
Using `σ²_post = σ²_prior·σ²_d / (σ²_prior + σ²_d)`:

```
Var(θ̂ − θ₀)       = (σ²_post/σ²_d)² · (σ²_prior + σ²_d)       = σ⁴_prior / (σ²_prior + σ²_d)
σ²_prior − σ²_post = σ²_prior − σ²_prior·σ²_d/(σ²_prior + σ²_d) = σ⁴_prior / (σ²_prior + σ²_d)   ✓
```

Divide one by the other and z simplifies to

```
z = (x − θ₀) / √(σ²_d + σ²_prior)
```

In plain words, **z is the difference between what the data alone says (x)
and what the prior says (θ₀), in units of their combined error.** It measures
the tension between data and prior, which is exactly what a pull should
measure. The two extremes of F.0 come out right. As σ_d → ∞ the pull is 0/0:
there is no information and nothing to test, and PROfit leaves such
parameters out (F.4). As σ_d → 0 it becomes the naive pull. In between, the
naive pull is narrower than z by the factor √(1 − σ²_post/σ²_prior).

### F.3 The same result from the Bayesian side

`--pseudo-experiment` does not move the auxiliary measurement. It draws the
*truth* from the prior and the data around the truth. The law of total
variance then gives the same identity in one line. For a Gaussian linear
model, the posterior width σ_post is the same for every dataset and the
posterior mean is θ̂, so

```
Var(θ) = E[ Var(θ | data) ] + Var( E[θ | data] )
σ²_prior =       σ²_post     +      Var(θ̂)
```

Of the prior variance, σ²_post is what the data leaves unresolved. The rest,
σ²_prior − σ²_post, is what the data resolves, and only that part makes θ̂
move. Whichever way the pseudo-experiments are generated, z ~ N(0,1).

### F.4 Many correlated parameters

PROfit fits many nuisances together, plus physics parameters with no prior.
Both arguments carry over with matrices. The leftover is uncorrelated with the
estimate (F.1), or the total covariance splits (F.3):

```
V ≡ Cov(θ̂ − θ₀) = P − Σ_post
```

Here P is the prior covariance of the nuisances: diag(σ²), or the full matrix
`Pull()` uses when XML `<correlation>`s or a PROjector constraint are present.
Σ_post is the post-fit covariance of the same nuisances. Two details matter:

* **Σ_post must be taken with every free parameter floating, physics
  included**, not at fixed physics. A floating physics parameter can absorb
  part of what a nuisance would otherwise explain. That widens Σ_post, and
  the estimate θ̂ loses exactly that information too, so V shrinks by the same
  amount. A Σ_post at fixed physics would overstate V and make every pull
  look too small.
* **`--fix`'d splines with correlated priors.** The prior of the free splines
  is then the one *conditional* on the fixed values, with its centre and width
  shifted. PROfit uses that conditional prior. Without correlations nothing
  changes.

Each pull uses its diagonal element, `z_i = (θ̂_i − θ₀_i)/√V_ii`. The ratio
`r_i = V_ii/P_ii` is the fraction of the prior variance the data resolved.
Below r = 5% the pull is 0/0 noise and is left out as "weakly constrained".
In a linear model Σ_post depends on the response, the statistical variance
and the priors, but not on the data values. The cut therefore picks the same
parameters whatever the data, and cannot bias the histogram. (Neyman's
C = data adds a mild data dependence.)

### F.5 How PROfit computes Σ_post

PROfit linearises the fit around the best fit and inverts the curvature of
the χ², for all free parameters at once. Following Appendix C.4, each
covariance mode (Σ = L·Lᵀ) gets a knob α_j with prior N(0,1), so the
covariance systematics get pulls too:

```
x       = (φ, s, α)               free physics, free splines, one knob per covariance mode
J       = ∂P/∂x at the best fit   collapsed prediction; φ and s by finite differences
                                  (central, one-sided at a bound); the α columns are L
C       = the fit metric's own statistical variance over the bins in the χ²
          (data for neyman, μ for pearson and poisson, 3/(1/d + 2/μ) for CNP)
Λ       = prior precision:  P⁻¹ for Gaussian splines (as Pull() applies it)
                            1   for each α
                            12/W² for physics and uniform splines (W = fit range)
H       = Jᵀ C⁻¹ J + Λ
Σ_post  = H⁻¹
```

This is the Gauss–Newton curvature of the χ², the same approximation as the
`lin` gradient modes, and it is exact for a linear model.

**The 12/W² term.** Physics parameters and uniform-prior splines have flat
priors over their fit range, which has variance W²/12. With exactly zero
there, a direction the data barely resolves could absorb its whole overlap
with the nuisances, however tiny its slope. Δm² when sin²2θ ≈ 0 is the
standard example. The box variance keeps such directions bounded, as the real
fit range does.

**The covariance-mode pulls** are `α̂ = A⁻¹LᵀC⁻¹u` from C.4, evaluated at the
best fit (the joint minimum over (θ, α) puts α at exactly that conditional
value). Their width is not C.4's A⁻¹, though. A⁻¹ is the width at *fixed* θ,
while V needs the width with θ floating, and H⁻¹ supplies that.

**Under `--shapeonly`**, J is the derivative of the per-channel rescaled
prediction and L is projected onto shape, exactly as the metric does
(Appendix D). A pure normalisation knob then has J = 0 and is reported as
weakly constrained, not given a spurious pull.

### F.6 Reading a whole histogram

Each z_i is N(0,1), but the z's are correlated, because knobs that do similar
things to the spectrum move together:

```
ρ_ij = V_ij / √(V_ii · V_jj)
```

That changes what one fit's histogram should look like. For the mean of n
pulls, and for the spread of the histogram about that mean:

```
Var(z̄)                   = (1/n²) Σ_ij ρ_ij        (= 1/n if uncorrelated)
E[ (1/n) Σ_i (z_i − z̄)² ] = 1 − Var(z̄)
```

The second line follows from `(1/n)Σ(z_i − z̄)² = (1/n)Σz_i² − z̄²`, with
`E[(1/n)Σz_i²] = 1` and `E[z̄²] = Var(z̄)`. These are the "If the model is
right" numbers on each page: μ = 0 ± √Var(z̄) and σ ≈ √(1 − Var(z̄)).
Strongly correlated pulls give a large Var(z̄) and a narrow histogram, like
the 12 spline pulls of section 6's example (σ ≈ 0.88).

The test that uses the correlations fully is the **whitened χ²**. Change to
coordinates in which the pulls are independent with unit variance. That
transformation turns correlated noise into white noise, hence the name:

```
d = θ̂ − θ₀,     y = V^(−1/2) · d    ⇒    each y_i ~ N(0,1), independent
q = yᵀy = dᵀ V⁻¹ d  ~  χ²(k)
```

PROfit evaluates q in the eigenbasis of ρ. It drops directions with
eigenvalue below 0.01, which are combinations the model pins so tightly that
the fit's convergence tolerance, not the data, sets them. k counts the rest.
q is not computed on a page with censored entries (F.7), whose z are bounds
rather than values.

The fitted Gaussian is an unbinned maximum-likelihood fit: for uncensored
pulls, simply their mean and RMS. Its quoted errors, σ/√n and σ/√(2n),
assume independent entries. Judge the mean against √Var(z̄), not against
them.

### F.7 Bounded and one-sided splines (page 3)

A knob whose fit range ends within 3σ of its centre can run into that end.
Take a 0…1 knob with its prior centred on 0. It hits 0 whenever the data
would prefer a negative value, which for an honest prior is about half the
time. The fit then returns the bound, not the value the data wanted.

Model this with a latent, unbounded estimate θ̃, whose z is the one from F.4
(the linearisation ignores the bound), and let the fit return
θ̂ = clamp(θ̃, lo, hi). At a bound we only know that the latent z lies beyond
it:

```
θ̂ = lo   ⇒   z ≤ z_lo = (lo − θ₀)/√V        censored below
θ̂ = hi   ⇒   z ≥ z_hi = (hi − θ₀)/√V        censored above
```

Censored entries are drawn hatched at their bound, on the side where the
latent value lies. The Gaussian fit uses the censored likelihood (φ and Φ are
the unit-Gaussian density and CDF):

```
L(μ, σ) =  Π_interior φ((z − μ)/σ)/σ  ×  Π_below Φ((z_lo − μ)/σ)  ×  Π_above [1 − Φ((z_hi − μ)/σ)]
```

It is maximised by expectation–maximisation. Each censored entry is replaced
by the mean and variance of the current Gaussian truncated beyond its bound,
for example below b, with a = (b − μ)/σ and λ = φ(a)/Φ(a):

```
E[z | z ≤ b]   = μ − σλ
Var[z | z ≤ b] = σ²·(1 − aλ − λ²)
```

Then μ and σ are re-estimated, and the two steps repeat until nothing
changes. The errors come from the curvature of ln L. A pile of hatched
entries at one bound is therefore not a problem by itself; compare the fitted
latent μ and σ with 0 and 1. Page-1 knobs that stop at ±3 are handled the
same way.

### F.8 Uniform-prior splines (page 4)

A uniform prior has no pull term and no σ_prior, so F.1–F.4 do not apply.
Ask the question directly instead. If the truth is anywhere in the box
[lo, hi] with equal probability, and the data measures θ with error σ_d,
where should θ̂ land? It lands at the truth smeared by N(0, σ²_d): a uniform
convolved with a Gaussian, whose CDF is

```
F(t) = (1/W) ∫_lo^hi Φ((t − a)/σ_d) da  =  (σ_d/W) · [ G((t − lo)/σ_d) − G((t − hi)/σ_d) ]
G(x) = x·Φ(x) + φ(x)                     (an integral of Φ: G′ = Φ)
```

Any continuous quantity passed through its own CDF is uniform on (0, 1). This
is the probability integral transform. Therefore

```
z = Φ⁻¹( F(θ̂) )  ~  N(0, 1)
```

which puts page 4 on the same footing as the others. σ_d is the data-only
width with everything else marginalised. From F.5, remove the parameter's own
box term: 1/σ²_d = 1/Σ_post,ii − 12/W². A box edge is censored as in F.7. The
parameter counts as weakly constrained unless σ²_d is below 95% of the box
variance W²/12.

### F.9 Limits

* **Linearised at the best fit.** Exact when the prediction is linear in the
  parameters and the errors are Gaussian. Strongly nonlinear splines,
  physics near a boundary (Δm² is undefined when the mixing vanishes) and
  very low-count bins (where neyman's C = data is a poor stand-in for
  Poisson) all bend it.
* **A pseudo-experiment tests the whole chain.** A deviation can come from
  the fit itself rather than the priors. The shifted spline mean in section
  6's example is such a case.
* **An Asimov fit gives z ≡ 0** for everything: nothing fluctuated, so
  nothing moved.
* **One fit is one draw.** Pool pseudo-experiments for precision (section 6).

---

*Why is the repository called "Elephant Vanishes"? Old habit — all of my git repos are named after the book I was reading when I created them. This one was Haruki Murakami's* The Elephant Vanishes.
