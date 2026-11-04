#include "PROletariat.h"
#include "PROlog.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>

#include <limits.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

namespace PROfit {

namespace {
// Removes the staging directory on every exit path (success, error return,
// exception). Best-effort: uses the error_code overload so it never throws.
struct StageGuard {
    std::filesystem::path path;
    ~StageGuard() {
        if(path.empty()) return;
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};
}

std::string PROletariat::LocateSelf() {
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if(n <= 0) return "";
    buf[n] = '\0';
    return std::string(buf);
}

std::string PROletariat::ShellQuote(const std::string &s) {
    if(!s.empty() && s.find_first_not_of(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._/:=+,-") == std::string::npos)
        return s;
    std::string out = "'";
    for(char c : s) {
        if(c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

int PROletariat::RunCommand(const std::vector<std::string> &argv) {
    std::vector<char*> cargv;
    cargv.reserve(argv.size() + 1);
    for(const auto &a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
    cargv.push_back(nullptr);

    pid_t pid;
    // Child inherits stdout/stderr so tar listings and jobsub progress stream live.
    int rc = posix_spawnp(&pid, cargv[0], nullptr, nullptr, cargv.data(), environ);
    if(rc != 0) {
        log<LOG_ERROR>(L"%1% || Failed to launch '%2%': %3%") % __func__ % argv[0].c_str() % strerror(rc);
        return -1;
    }
    int status = 0;
    if(waitpid(pid, &status, 0) < 0) {
        log<LOG_ERROR>(L"%1% || waitpid failed for '%2%': %3%") % __func__ % argv[0].c_str() % strerror(errno);
        return -1;
    }
    if(WIFSIGNALED(status)) {
        log<LOG_ERROR>(L"%1% || '%2%' killed by signal %3%") % __func__ % argv[0].c_str() % WTERMSIG(status);
        return 128 + WTERMSIG(status);
    }
    return WEXITSTATUS(status);
}

int PROletariat::AddInput(const std::filesystem::path &src, bool required, const std::string &stage_as) {
    std::error_code ec;
    if(!std::filesystem::is_regular_file(src, ec)) {
        if(required) {
            log<LOG_ERROR>(L"%1% || Required input file missing: %2%") % __func__ % src.c_str();
            return 1;
        }
        log<LOG_INFO>(L"%1% || Optional artifact not found, skipping: %2%") % __func__ % src.c_str();
        return 0;
    }

    std::string name = stage_as.empty() ? src.filename().string() : stage_as;
    std::filesystem::path dst = grid_dir_ / name;
    if(std::filesystem::exists(dst)) {
        // The same file reachable twice (e.g. --input naming an auto-bundled
        // artifact) is fine — skip. Two DIFFERENT sources with one basename
        // would silently clobber each other on the worker node: refuse.
        auto it = staged_sources_.find(name);
        if(it != staged_sources_.end() && std::filesystem::equivalent(src, it->second, ec)) {
            log<LOG_INFO>(L"%1% || Already staged, skipping duplicate: %2%") % __func__ % src.c_str();
            return 0;
        }
        log<LOG_ERROR>(L"%1% || Basename collision in tarball: '%2%' already staged from '%3%', refusing to overwrite with '%4%'")
            % __func__ % name.c_str() % (it != staged_sources_.end() ? it->second.c_str() : "?") % src.c_str();
        return 1;
    }
    staged_sources_[name] = std::filesystem::absolute(src);
    if(!std::filesystem::copy_file(src, dst, ec) || ec) {
        log<LOG_ERROR>(L"%1% || Failed to copy '%2%' into staging dir: %3%") % __func__ % src.c_str() % ec.message().c_str();
        return 1;
    }
    log<LOG_INFO>(L"%1% || Staged %2% (%3% bytes)") % __func__ % dst.filename().c_str() % std::filesystem::file_size(dst, ec);
    return 0;
}

int PROletariat::Stage() {
    // Binary: --profit-bin override, else this very executable.
    std::string bin = opts_.profit_bin.empty() ? LocateSelf() : opts_.profit_bin;
    if(bin.empty()) {
        log<LOG_ERROR>(L"%1% || Cannot locate the PROfit binary via /proc/self/exe; pass --profit-bin explicitly.") % __func__;
        return 1;
    }

    // Worker script: canonicalize now, both to validate it and because the
    // jobsub file:// URL must point at the real path (the old shell script
    // wrongly used $PWD/$(basename script)).
    std::error_code ec;
    std::filesystem::path script_canon = std::filesystem::canonical(opts_.script, ec);
    if(ec) {
        log<LOG_ERROR>(L"%1% || Worker script not found: %2%") % __func__ % opts_.script.c_str();
        return 1;
    }
    script_abs_ = script_canon.string();

    // Required inputs. The binary is always staged under the literal name
    // "PROfit": worker scripts invoke ./PROfit.
    if(AddInput(bin, true, "PROfit")) return 1;
    if(AddInput(script_canon, true)) return 1;
    if(AddInput(opts_.xml, true)) return 1;
    for(const auto &f : opts_.extra_inputs)
        if(AddInput(f, true)) return 1;

    // Auto-detected analysis artifacts in the current directory, optional.
    const std::vector<std::string> auto_artifacts = {
        opts_.analysis_tag + "_prop.bin",
        opts_.analysis_tag + "_syst.bin",
        opts_.analysis_tag + "_data.bin",
        opts_.analysis_tag + "_overlap_mc.bin",
        opts_.analysis_tag + "_overlap_data.bin",
        opts_.final_output_tag + "_mesh.bin",
        opts_.final_output_tag + "_bank.bin",
    };
    for(const auto &f : auto_artifacts)
        if(AddInput(std::filesystem::current_path() / f, false)) return 1;

    // Executable bits for the binary and the worker script.
    using std::filesystem::perms;
    for(const auto &name : {std::string("PROfit"), script_canon.filename().string()}) {
        std::filesystem::permissions(grid_dir_ / name,
            perms::owner_exec | perms::group_exec | perms::others_exec,
            std::filesystem::perm_options::add, ec);
        if(ec) {
            log<LOG_ERROR>(L"%1% || chmod +x failed on staged '%2%': %3%") % __func__ % name.c_str() % ec.message().c_str();
            return 1;
        }
    }
    return 0;
}

int PROletariat::MakeTarball() {
    tarball_ = std::filesystem::current_path() / "grid_dir.tar";
    std::error_code ec;
    std::filesystem::remove(tarball_, ec);

    if(int rc = RunCommand({"tar", "cf", tarball_.string(), "-C", stage_dir_.string(), "grid_dir"})) {
        log<LOG_ERROR>(L"%1% || tar failed with exit code %2%") % __func__ % rc;
        return rc > 0 ? rc : 1;
    }

    uintmax_t sz = std::filesystem::file_size(tarball_, ec);
    log<LOG_INFO>(L"%1% || Tarball %2% (%3% MB), contents:") % __func__ % tarball_.c_str() % (sz / (1024 * 1024));
    RunCommand({"tar", "tvf", tarball_.string()});
    return 0;
}

std::vector<std::string> PROletariat::BuildJobsubCommand() const {
    std::vector<std::string> cmd = {
        "jobsub_submit",
        "-G", opts_.group,
        "-N", std::to_string(opts_.njobs),
        "--role=" + opts_.role,
        "--expected-lifetime=" + opts_.lifetime,
        "--memory=" + std::to_string(opts_.memory_mb) + "MB",
        "--disk="   + std::to_string(opts_.disk_mb)   + "MB"};
    for(const auto &l : opts_.condor_lines) {
        cmd.push_back("--lines");
        cmd.push_back(l);
    }
    cmd.push_back("--resource-provides=" + opts_.resource_provides);
    cmd.push_back("-l");
    // jobsub wants the literal bytes +SingularityImage=\"<image>\" (backslash
    // and quote included) — same as the shell script's \\\" expansion.
    cmd.push_back("+SingularityImage=\\\"" + opts_.singularity_image + "\\\"");
    cmd.push_back("--append_condor_requirements=(TARGET.HAS_SINGULARITY=?=true)");
    cmd.push_back("--tar_file_name");
    cmd.push_back("dropbox://" + tarball_.string());
    for(const auto &a : opts_.extra_jobsub_args) cmd.push_back(a);
    cmd.push_back("file://" + script_abs_);
    return cmd;
}

int PROletariat::Submit() {
    switch(opts_.backend) {
    case PROletariatOptions::Backend::Slurm:
        log<LOG_ERROR>(L"%1% || SLURM backend not yet implemented. Use --backend jobsub.") % __func__;
        return 1;

    case PROletariatOptions::Backend::Jobsub: {
        std::vector<std::string> cmd = BuildJobsubCommand();
        std::string display;
        for(const auto &a : cmd) {
            if(!display.empty()) display += " ";
            display += ShellQuote(a);
        }
        log<LOG_INFO>(L"%1% || Submission command: %2%") % __func__ % display.c_str();

        if(opts_.dry_run) {
            log<LOG_INFO>(L"%1% || Dry run: not submitted. Tarball left at %2%") % __func__ % tarball_.c_str();
            return 0;
        }

        log<LOG_INFO>(L"%1% || Submitting %2% job(s), lifetime %3%, %4%MB memory, %5%MB disk")
            % __func__ % opts_.njobs % opts_.lifetime.c_str() % opts_.memory_mb % opts_.disk_mb;
        int rc = RunCommand(cmd);
        if(rc == -1 || rc == 127) {
            log<LOG_ERROR>(L"%1% || jobsub_submit not found or failed to launch. Run from a GPVM with the jobsub client set up, or use --dry-run.") % __func__;
            return 1;
        }
        if(rc != 0)
            log<LOG_ERROR>(L"%1% || jobsub_submit exited with code %2%") % __func__ % rc;
        return rc;
    }
    }
    return 1;
}

int PROletariat::Run() {
    // SLURM is a stub: fail before doing any staging work.
    if(opts_.backend == PROletariatOptions::Backend::Slurm) return Submit();

    // Fresh staging dir every run so nothing from a previous submission leaks
    // into the tarball; the guard removes it on every exit path.
    std::string tmpl = (std::filesystem::current_path() / ".grid_stage.XXXXXX").string();
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if(!mkdtemp(buf.data())) {
        log<LOG_ERROR>(L"%1% || Failed to create staging dir '%2%': %3%") % __func__ % tmpl.c_str() % strerror(errno);
        return 1;
    }
    stage_dir_ = buf.data();
    StageGuard guard{stage_dir_};

    grid_dir_ = stage_dir_ / "grid_dir";
    std::error_code ec;
    if(!std::filesystem::create_directory(grid_dir_, ec)) {
        log<LOG_ERROR>(L"%1% || Failed to create %2%: %3%") % __func__ % grid_dir_.c_str() % ec.message().c_str();
        return 1;
    }

    if(int rc = Stage()) return rc;
    if(int rc = MakeTarball()) return rc;
    return Submit();
}

}
