//! Everything the remote machine needs, embedded in the binary: the Blender installer and the
//! Python helper/demo scripts. `deploy_script` produces one bash program that installs it all.

use crate::util::{base64, fnv1a, sh_quote};

pub const DEFAULT_VERSION: &str = "4.5.16";

pub const INSTALL_SH: &str = include_str!("../scripts/install-blender.sh");
pub const SKILL_MD: &str = include_str!("../skills/rblender/SKILL.md");
pub const HELLO_PY: &str = include_str!("../scenes/hello.py");
pub const PY_FILES: &[(&str, &str)] = &[
    ("rb.py", include_str!("../py/rb.py")),
    ("demo.py", include_str!("../py/demo.py")),
    ("rig.py", include_str!("../py/rig.py")),
];

/// Changes whenever the Blender version or any embedded file changes, so a new rblender build
/// re-deploys itself on first use.
pub fn stamp(version: &str) -> String {
    let mut parts: Vec<&[u8]> = vec![version.as_bytes(), INSTALL_SH.as_bytes()];
    for (name, body) in PY_FILES {
        parts.push(name.as_bytes());
        parts.push(body.as_bytes());
    }
    format!("{version}-{:016x}", fnv1a(&parts))
}

/// Remote bash: exit 0 iff Blender and the helper files are installed and current.
pub fn check_script(version: &str) -> String {
    format!(
        "[ -x \"$HOME/blender/blender\" ] && [ \"$(cat \"$HOME/.rblender/stamp\" 2>/dev/null)\" = {} ]",
        sh_quote(&stamp(version))
    )
}

/// Remote bash: write helper files, install Blender (idempotent), record the stamp.
pub fn deploy_script(version: &str) -> String {
    let mut s = String::from("set -euo pipefail\nD=\"$HOME/.rblender\"; mkdir -p \"$D/py\"\n");
    s += &format!(
        "echo {} | base64 -d > \"$D/install-blender.sh\"\n",
        base64(INSTALL_SH.as_bytes())
    );
    for (name, body) in PY_FILES {
        s += &format!(
            "echo {} | base64 -d > \"$D/py/{name}\"\n",
            base64(body.as_bytes())
        );
    }
    s += &format!(
        "BLENDER_VERSION={} bash \"$D/install-blender.sh\" >&2\necho {} > \"$D/stamp\"\n",
        sh_quote(version),
        sh_quote(&stamp(version))
    );
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamp_depends_on_version() {
        assert_ne!(stamp("4.5.16"), stamp("4.5.15"));
        assert!(stamp("4.5.16").starts_with("4.5.16-"));
    }

    #[test]
    fn deploy_script_embeds_every_file() {
        let s = deploy_script(DEFAULT_VERSION);
        for (name, _) in PY_FILES {
            assert!(s.contains(&format!("py/{name}")), "{name} missing");
        }
        assert!(s.contains("install-blender.sh"));
    }
}
