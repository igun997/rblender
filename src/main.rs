use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use rblender::assets::DEFAULT_VERSION;
use rblender::remote::{exec_script, run_script, Remote, Result, RunOpts, Source};
use rblender::verify;

const HELP: &str = "\
rblender: headless Blender on a remote sandbox, zero setup (built on rcargo)

USAGE: rblender [-i INSTANCE] <command> [args]

  init                 make this folder an rblender project (Cargo.toml stub + rcargo init)
  setup                install Blender + helpers on the remote now (otherwise done on first use)
  doctor               check rcargo, the remote machine and Blender
  run SCRIPT.py [ARGS] run a Blender Python script from this project (bpy available)
  eval 'PYTHON'        run inline Python inside Blender
  demo [--quick]       PoC: texture, animate, render, export (glb/fbx/obj/usd/blend)
  rig  [--quick]       PoC: model + rig (armature, skin weights) + walk cycle + export glb
  exec -- BLENDER_ARGS raw blender command line (e.g. -- -b scene.blend -a)
  pull [DIR]           copy remote out/ to DIR (default ./out)
  verify [DIR] [--skin] [--anim] [--pbr]
                       prove the artifacts in DIR (default ./out) are real
  down                 pause the remote machine (wakes on next use)

Flags for run/eval/demo/rig:
  --no-pull   do not copy out/ back     --keep   keep previous remote out/ contents
  --xvfb      run under a virtual display (needed by some EEVEE setups)
  --verbose   show Blender's per-sample progress lines
Scripts write results with rb.out(\"name.ext\") (or into $RBLENDER_OUT); they are pulled to ./out.
Environment: BLENDER_VERSION (default 4.5.16), RCARGO_INSTANCE.";

fn main() -> ExitCode {
    match real_main() {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("rblender: {e}");
            ExitCode::from(1)
        }
    }
}

fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    let before = args.len();
    args.retain(|a| a != flag);
    args.len() != before
}

fn real_main() -> Result<u8> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut instance = std::env::var("RCARGO_INSTANCE")
        .ok()
        .filter(|s| !s.is_empty());
    if matches!(args.first().map(String::as_str), Some("-i" | "--instance")) {
        if args.len() < 2 {
            return Err("-i needs an instance name".into());
        }
        instance = Some(args[1].clone());
        args.drain(..2);
    }
    if matches!(
        args.first().map(String::as_str),
        Some("--version" | "-V" | "version")
    ) {
        println!("rblender {}", env!("CARGO_PKG_VERSION"));
        return Ok(0);
    }
    if args.is_empty() || matches!(args[0].as_str(), "help" | "-h" | "--help") {
        println!("{HELP}");
        return Ok(0);
    }
    let cmd = args.remove(0);
    let remote = Remote {
        instance,
        version: std::env::var("BLENDER_VERSION").unwrap_or_else(|_| DEFAULT_VERSION.into()),
    };
    let mut o = RunOpts {
        keep_out: take_flag(&mut args, "--keep"),
        xvfb: take_flag(&mut args, "--xvfb"),
        verbose: take_flag(&mut args, "--verbose"),
        args: Vec::new(),
    };
    let no_pull = take_flag(&mut args, "--no-pull");

    match cmd.as_str() {
        "init" => init(),
        "setup" => {
            remote.ensure()?;
            println!("rblender: Blender {} ready on the remote", remote.version);
            Ok(0)
        }
        "doctor" => doctor(&remote),
        "down" => status(Command::new("rcargo").args(["remote", "down"]).status()?),
        "pull" => {
            let dest = PathBuf::from(args.first().map_or("out", String::as_str));
            let bytes = remote.pull(&dest)?;
            println!("rblender: pulled {} to {}", human(bytes), dest.display());
            Ok(0)
        }
        "verify" => {
            let req = verify::Require {
                skin: take_flag(&mut args, "--skin"),
                anim: take_flag(&mut args, "--anim"),
                pbr: take_flag(&mut args, "--pbr"),
            };
            let dir = PathBuf::from(args.first().map_or("out", String::as_str));
            verify_dir(&dir, req)
        }
        "exec" => {
            let raw: Vec<String> = args.into_iter().skip_while(|a| a == "--").collect();
            remote.ensure()?;
            status(remote.shell(&exec_script(&raw, o.xvfb), false, false)?)
        }
        "run" | "eval" | "demo" | "rig" => {
            if args.is_empty() && matches!(cmd.as_str(), "run" | "eval") {
                return Err(format!(
                    "usage: rblender {cmd} <{}>",
                    if cmd == "run" { "script.py" } else { "python" }
                )
                .into());
            }
            let first = if matches!(cmd.as_str(), "run" | "eval") {
                Some(args.remove(0))
            } else {
                None
            };
            o.args = args;
            let script = match (cmd.as_str(), &first) {
                ("run", Some(p)) => run_script(&Source::ProjectFile(p), &o),
                ("eval", Some(code)) => run_script(&Source::Inline(code), &o),
                ("demo", _) => run_script(&Source::Bundled("demo.py"), &o),
                _ => run_script(&Source::Bundled("rig.py"), &o),
            };
            remote.ensure()?;
            let st = remote.shell(&script, false, false)?;
            if !st.success() {
                eprintln!("rblender: Blender exited with {st}; results not pulled (use `rblender pull` to fetch partial output)");
                return Ok(st.code().unwrap_or(1).clamp(1, 255) as u8);
            }
            if !no_pull {
                if !o.keep_out {
                    let _ = std::fs::remove_dir_all("out");
                }
                let bytes = remote.pull(Path::new("out"))?;
                println!("\nrblender: pulled {} to ./out", human(bytes));
                list_out(Path::new("out"));
            }
            Ok(0)
        }
        other => Err(format!("unknown command `{other}` (see `rblender help`)").into()),
    }
}

fn status(st: std::process::ExitStatus) -> Result<u8> {
    Ok(if st.success() {
        0
    } else {
        st.code().unwrap_or(1).clamp(1, 255) as u8
    })
}

fn human(b: u64) -> String {
    if b >= 1 << 20 {
        format!("{:.1} MiB", b as f64 / (1 << 20) as f64)
    } else {
        format!("{:.1} KiB", b as f64 / 1024.0)
    }
}

fn list_out(dir: &Path) {
    let mut v: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_file())
        .map(|e| {
            (
                e.file_name().to_string_lossy().into_owned(),
                e.metadata().map(|m| m.len()).unwrap_or(0),
            )
        })
        .collect();
    v.sort();
    for (n, len) in v {
        println!("  {:<24}{:>10}", n, human(len));
    }
}

fn verify_dir(dir: &Path, req: verify::Require) -> Result<u8> {
    let findings = verify::dir(dir, req)?;
    if findings.is_empty() {
        return Err(format!("no verifiable artifacts in {}", dir.display()).into());
    }
    let mut bad = 0;
    for f in &findings {
        println!(
            "[{}] {:<28} {}",
            if f.ok { " ok " } else { "FAIL" },
            f.file,
            f.detail
        );
        bad += usize::from(!f.ok);
    }
    println!("\n{} checked, {} failed", findings.len(), bad);
    Ok(u8::from(bad > 0))
}

fn doctor(remote: &Remote) -> Result<u8> {
    let st = Command::new("rcargo").args(["remote", "doctor"]).status()?;
    if !st.success() {
        return status(st);
    }
    remote.ensure()?;
    status(remote.shell(
        "\"$HOME/blender/blender\" -b --version | head -1; nproc | sed 's/^/cpus: /'; \
         command -v xvfb-run >/dev/null && echo 'xvfb: yes' || echo 'xvfb: no'",
        true,
        true,
    )?)
}

fn init() -> Result<u8> {
    let cwd = std::env::current_dir()?;
    if !cwd.ancestors().any(|d| d.join("Cargo.toml").exists()) {
        let name: String = cwd
            .file_name()
            .map_or("scene".into(), |n| n.to_string_lossy().into_owned())
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() {
                    c.to_ascii_lowercase()
                } else {
                    '-'
                }
            })
            .collect();
        // rcargo needs a cargo project to anchor sync; this stub makes any folder one.
        std::fs::write(
            "Cargo.toml",
            format!("[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n"),
        )?;
        std::fs::create_dir_all("src")?;
        std::fs::write("src/lib.rs", "")?;
        println!("rblender: created Cargo.toml stub ({name})");
    }
    let gi = std::fs::read_to_string(".gitignore").unwrap_or_default();
    if !gi.lines().any(|l| l.trim_matches('/') == "out") {
        std::fs::write(
            ".gitignore",
            format!(
                "{gi}{}/out\n/target\n",
                if gi.is_empty() || gi.ends_with('\n') {
                    ""
                } else {
                    "\n"
                }
            ),
        )?;
    }
    for dir in [".claude/skills/rblender", ".agents/skills/rblender"] {
        std::fs::create_dir_all(dir)?;
        std::fs::write(format!("{dir}/SKILL.md"), rblender::assets::SKILL_MD)?;
    }
    println!("rblender: installed agent skill (.claude/skills/rblender, .agents/skills/rblender)");
    if !Path::new("scenes/hello.py").exists() {
        std::fs::create_dir_all("scenes")?;
        std::fs::write("scenes/hello.py", rblender::assets::HELLO_PY)?;
        println!("rblender: wrote scenes/hello.py (try: rblender run scenes/hello.py)");
    }
    status(Command::new("rcargo").arg("init").status()?)
}
