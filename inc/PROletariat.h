/**
 * @file PROletariat.h
 * @brief Grid job submitter: stage inputs, build grid_dir.tar, submit via jobsub.
 * @author PROfit Collaboration
 *
 * @details Replaces grid/maketar_submit_v2.4.sh as the "submitter half" of the
 * grid workflow. Jobs run in the AL9 (el9) container by default; the --sl7
 * CLI flag swaps in the legacy SL7 image (an explicit --singularity-image
 * always wins and excludes --sl7). The payload is still a user-supplied worker
 * script (e.g. grid/runFC_v2.4_v4_AL9.sh, which detects the container OS and
 * uses Spack on AL9 / UPS on SL7); PROletariat stages the PROfit binary (self-located
 * via /proc/self/exe, override with --profit-bin), the analysis XML,
 * auto-detected analysis artifacts in the current directory
 * (<tag>_prop.bin, <tag>_syst.bin, <tag>_data.bin, <tag>_overlap_{mc,data}.bin,
 * <tag>_<out>_mesh.bin, <tag>_<out>_bank.bin)
 * and any --input extras into a fresh temp dir, tars them under the literal
 * root directory grid_dir/ (the worker-script contract: workers read
 * $INPUT_TAR_DIR_LOCAL/grid_dir/), and invokes jobsub_submit as a subprocess
 * with argv built directly — no shell, so the +SingularityImage=\"...\"
 * escaped-quote token and paths with spaces need no quoting. --dry-run stages
 * and tars, then prints the exact command without submitting.
 *
 * Backends: jobsub (implemented), slurm (stub, errors out; to implement, add
 * a BuildSlurmCommand() and a switch arm in Submit()).
 *
 * Errors are reported via log<LOG_ERROR> + nonzero return codes rather than
 * exit(): exit() would skip the RAII cleanup of the staging directory.
 */
#ifndef PROLETARIAT_H
#define PROLETARIAT_H

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace PROfit {

/// All knobs for one grid submission. Defaults mirror grid/maketar_submit_v2.4.sh.
struct PROletariatOptions {
    enum class Backend { Jobsub, Slurm };
    Backend backend = Backend::Jobsub;

    // Job shape
    int         njobs     = 2;
    std::string lifetime  = "2d";   ///< jobsub --expected-lifetime (3d is the FermiGrid ceiling).
    int         memory_mb = 4000;
    int         disk_mb   = 10000;

    // Payload
    std::string script;                     ///< Worker script run on each node (required).
    std::string xml;                        ///< -x XML, always bundled.
    std::string analysis_tag;               ///< For <tag>_prop.bin / <tag>_syst.bin auto-bundle.
    std::string final_output_tag;           ///< For <tag>_<out>_mesh.bin / _bank.bin auto-bundle.
    std::vector<std::string> extra_inputs;  ///< --input files; a missing one is a hard error.
    std::string profit_bin;                 ///< Override binary to ship; empty => /proc/self/exe.

    // Container images: AL9 (el9) is the default now that FermiGrid nodes have
    // migrated; --sl7 at the CLI swaps in the legacy SL7 image. Worker scripts
    // detect the OS at runtime and use Spack (AL9) or UPS (SL7) accordingly.
    static constexpr const char *kImageAL9 =
        "/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-el9:latest";
    static constexpr const char *kImageSL7 =
        "/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-sl7:latest";

    // jobsub knobs (defaults = current shell-script values)
    std::string group             = "sbnd";
    std::string role              = "Analysis";
    std::string singularity_image = kImageAL9;
    std::string resource_provides = "usage_model=DEDICATED,OPPORTUNISTIC,OFFSITE";
    std::vector<std::string> condor_lines = {
        "+FERMIHTC_AutoRelease=True", "+FERMIHTC_GraceMemory=4000", "+FERMIHTC_GraceLifetime=7200"};
    std::vector<std::string> extra_jobsub_args;  ///< Raw passthrough, appended before the file:// URL.

    bool dry_run = false;
};

class PROletariat {
public:
    explicit PROletariat(PROletariatOptions opts) : opts_(std::move(opts)) {}

    /// Stage + tar + submit (or dry-run print). Returns 0 on success,
    /// nonzero on any failure (suitable as a process exit code).
    int Run();

private:
    int  Stage();                                        ///< Copy binary/script/xml/inputs into grid_dir_.
    int  MakeTarball();                                  ///< tar cf grid_dir.tar -C stage grid_dir; list contents.
    std::vector<std::string> BuildJobsubCommand() const; ///< Full jobsub_submit argv.
    int  Submit();                                       ///< Dispatch on backend; runs or (dry-run) prints the command.

    int  AddInput(const std::filesystem::path &src, bool required,
                  const std::string &stage_as = "");     ///< Copy into grid_dir_ (basename-collision checked).
    static std::string LocateSelf();                     ///< readlink /proc/self/exe; "" on failure.
    static int  RunCommand(const std::vector<std::string> &argv); ///< posix_spawnp, inherited stdio; child exit code, or -1 on launch failure.
    static std::string ShellQuote(const std::string &s); ///< Quote for copy-paste display only.

    PROletariatOptions opts_;
    std::filesystem::path stage_dir_;   ///< .grid_stage.XXXXXX (absolute).
    std::filesystem::path grid_dir_;    ///< stage_dir_/grid_dir.
    std::filesystem::path tarball_;     ///< <cwd>/grid_dir.tar (absolute).
    std::string script_abs_;            ///< Canonical absolute path of opts_.script (used for the file:// URL).
    std::map<std::string, std::filesystem::path> staged_sources_; ///< staged basename -> source path (collision check).
};

}
#endif
