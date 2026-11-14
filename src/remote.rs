//! Remote execution through rcargo: instance selection, project sync and SSH are all rcargo's job;
//! rblender only adds "make sure Blender is there", "run it", "bring the results back".

use std::error::Error;
use std::io::Write;
use std::path::Path;
use std::process::{Command, ExitStatus, Stdio};

use crate::assets;
use crate::util::{base64, base64_decode, sh_quote};

pub type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

pub struct Remote {
    pub instance: Option<String>,
    pub version: String,
}

/// What to execute inside Blender.
pub enum Source<'a> {
    /// A file inside the synced project (path relative to the project root).
    ProjectFile(&'a str),
    /// A script deployed with rblender itself (demo.py, rig.py).
    Bundled(&'a str),
    /// Inline Python.
    Inline(&'a str),
}

pub struct RunOpts {
    pub args: Vec<String>,
    pub keep_out: bool,
    pub xvfb: bool,
    pub verbose: bool,
}

/// Remote bash program that runs one script in background Blender.
pub fn run_script(src: &Source, o: &RunOpts) -> String {
    let mut s = String::from("set -uo pipefail\nexport RBLENDER_OUT=\"$PWD/out\"\n");
    if !o.keep_out {
        s += "rm -rf \"$PWD/out\"\n";
    }
    s += "mkdir -p \"$PWD/out\"\n";
    let script = match src {
        Source::ProjectFile(p) => sh_quote(p),
        Source::Bundled(n) => format!("\"$HOME/.rblender/py/{n}\""),
        Source::Inline(code) => {
            s += &format!(
                "echo {} | base64 -d > \"$HOME/.rblender/inline.py\"\n",
                base64(code.as_bytes())
            );
            "\"$HOME/.rblender/inline.py\"".to_string()
        }
    };
    let pre = if o.xvfb { "xvfb-run -a " } else { "" };
    let args: String = o.args.iter().map(|a| format!(" {}", sh_quote(a))).collect();
    let filter = if o.verbose {
        String::new()
    } else {
        " | { grep --line-buffered -v '^Fra:' || true; }".to_string()
    };
    s += &format!(
        "{pre}\"$HOME/blender/blender\" -b --factory-startup --python-exit-code 1 \
         --python-expr \"import sys; sys.path[:0]=['$HOME/.rblender/py', '$PWD']\" \
         -P {script} -- {args} 2>&1{filter}\nexit ${{PIPESTATUS[0]}}\n"
    );
    s
}

/// Remote bash for raw blender arguments (`rblender exec -- ...`).
pub fn exec_script(args: &[String], xvfb: bool) -> String {
    let pre = if xvfb { "xvfb-run -a " } else { "" };
    let args: String = args.iter().map(|a| format!(" {}", sh_quote(a))).collect();
    format!("export RBLENDER_OUT=\"$PWD/out\"; mkdir -p \"$PWD/out\"\n{pre}exec \"$HOME/blender/blender\"{args}\n")
}

impl Remote {
    fn rcargo(&self, no_sync: bool, quiet: bool) -> Command {
        let mut c = Command::new("rcargo");
        if let Some(i) = &self.instance {
            c.args(["-i", i]);
        }
        if quiet {
            c.arg("-q");
        }
        if no_sync {
            c.arg("--no-sync");
        }
        c.args(["remote", "shell", "--", "bash", "-c"]);
        c
    }

    fn spawn_err(e: std::io::Error) -> Box<dyn Error> {
        if e.kind() == std::io::ErrorKind::NotFound {
            "rcargo is not installed or not on PATH. Install: \
             curl -fsSL https://raw.githubusercontent.com/igun997/rcargo/main/install.sh | sh"
                .into()
        } else {
            e.into()
        }
    }

    /// Run a bash program on the remote workspace with inherited stdio.
    pub fn shell(&self, script: &str, no_sync: bool, quiet: bool) -> Result<ExitStatus> {
        self.rcargo(no_sync, quiet)
            .arg(script)
            .status()
            .map_err(Self::spawn_err)
    }

    /// Silent probe: true iff the remote bash program exits 0.
    fn probe(&self, script: &str) -> Result<bool> {
        Ok(self
            .rcargo(false, true)
            .arg(script)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(Self::spawn_err)?
            .success())
    }

    /// Install Blender + helpers if the remote is missing or out of date. Cheap when current.
    pub fn ensure(&self) -> Result<()> {
        if self.probe(&assets::check_script(&self.version))? {
            return Ok(());
        }
        eprintln!(
            "rblender: provisioning Blender {} on the remote machine (first use, a minute or two)",
            self.version
        );
        let st = self.shell(&assets::deploy_script(&self.version), false, false)?;
        if !st.success() {
            return Err(format!("remote provisioning failed ({st})").into());
        }
        Ok(())
    }

    /// Copy the remote `out/` directory into `dest`. rcargo's shell stream is text-oriented and
    /// corrupts raw binary, so the archive travels as gzip + base64.
    pub fn pull(&self, dest: &Path) -> Result<u64> {
        std::fs::create_dir_all(dest)?;
        let res = self
            .rcargo(true, true)
            .arg("cd \"$PWD/out\" 2>/dev/null && tar -czf - . | base64 -w0")
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .output()
            .map_err(Self::spawn_err)?;
        if !res.status.success() {
            return Err("remote has no out/ directory (nothing was produced)".into());
        }
        let tarball =
            base64_decode(&res.stdout).ok_or("corrupt transfer from remote (bad base64)")?;
        let mut sink = Command::new("tar")
            .args(["-xzf", "-", "-C"])
            .arg(dest)
            .stdin(Stdio::piped())
            .spawn()?;
        sink.stdin.take().ok_or("no stdin")?.write_all(&tarball)?;
        if !sink.wait()?.success() {
            return Err("failed to unpack results".into());
        }
        Ok(dir_size(dest))
    }
}

fn dir_size(p: &Path) -> u64 {
    std::fs::read_dir(p)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| match e.metadata() {
            Ok(m) if m.is_dir() => dir_size(&e.path()),
            Ok(m) => m.len(),
            Err(_) => 0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(args: &[&str]) -> RunOpts {
        RunOpts {
            args: args.iter().map(|s| s.to_string()).collect(),
            keep_out: false,
            xvfb: false,
            verbose: false,
        }
    }

    #[test]
    fn project_file_with_hostile_args_is_quoted() {
        let s = run_script(
            &Source::ProjectFile("scenes/my scene.py"),
            &opts(&["--name", "a'b; rm -rf /"]),
        );
        assert!(s.contains("-P 'scenes/my scene.py' --"));
        assert!(s.contains("'a'\\''b; rm -rf /'"));
        assert!(s.contains("rm -rf \"$PWD/out\""));
    }

    #[test]
    fn keep_out_skips_clean_and_verbose_skips_filter() {
        let mut o = opts(&[]);
        o.keep_out = true;
        o.verbose = true;
        let s = run_script(&Source::Bundled("demo.py"), &o);
        assert!(!s.contains("rm -rf"));
        assert!(!s.contains("grep"));
        assert!(s.contains("$HOME/.rblender/py/demo.py"));
        assert!(s.ends_with("exit ${PIPESTATUS[0]}\n"));
    }

    #[test]
    fn inline_code_travels_as_base64_not_in_shell_text() {
        let s = run_script(&Source::Inline("print('x\"; boom')"), &opts(&[]));
        assert!(!s.contains("boom"));
        assert!(s.contains("inline.py"));
    }
}
