//! `rblender verify`: prove that pulled artifacts are real, not just non-empty files.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use serde_json::Value;

pub struct Finding {
    pub file: String,
    pub ok: bool,
    pub detail: String,
}

#[derive(Default, Clone, Copy)]
pub struct Require {
    pub skin: bool,
    pub anim: bool,
}

pub fn dir(path: &Path, req: Require) -> std::io::Result<Vec<Finding>> {
    let mut names: Vec<_> = fs::read_dir(path)?
        .flatten()
        .filter(|e| e.path().is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    Ok(names
        .into_iter()
        .filter_map(|n| {
            let bytes = fs::read(path.join(&n)).ok()?;
            let ext = n.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
            let (ok, detail) = match ext.as_str() {
                "png" => png(&bytes),
                "glb" => glb(&bytes, req),
                "mp4" => magic(&bytes, 4, b"ftyp", "MP4 container"),
                "fbx" => magic(&bytes, 0, b"Kaydara FBX Binary", "binary FBX"),
                "blend" => blend(&bytes),
                "usdc" => magic(&bytes, 0, b"PXR-USDC", "USD crate"),
                "obj" => text_has(&bytes, "\nv ", "OBJ vertices"),
                "json" => json(&bytes),
                _ => return None,
            };
            Some(Finding {
                file: format!("{n} ({} B)", bytes.len()),
                ok,
                detail,
            })
        })
        .collect())
}

fn magic(b: &[u8], at: usize, m: &[u8], what: &str) -> (bool, String) {
    let ok = b.len() >= at + m.len() && &b[at..at + m.len()] == m;
    (
        ok,
        if ok {
            what.into()
        } else {
            format!("not a valid {what}")
        },
    )
}

fn blend(b: &[u8]) -> (bool, String) {
    if b.starts_with(b"BLENDER") {
        return (true, "Blender file (raw)".into());
    }
    // Blender saves zstd-compressed .blend files by default.
    if b.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
        return (true, "Blender file (zstd)".into());
    }
    (false, "not a Blender file".into())
}

fn text_has(b: &[u8], needle: &str, what: &str) -> (bool, String) {
    let s = String::from_utf8_lossy(b);
    let n = s.matches(needle).count();
    (n > 0, format!("{n} {what}"))
}

fn json(b: &[u8]) -> (bool, String) {
    match serde_json::from_slice::<Value>(b) {
        Ok(_) => (true, "valid JSON".into()),
        Err(e) => (false, format!("invalid JSON: {e}")),
    }
}

fn png(b: &[u8]) -> (bool, String) {
    let mut dec = png::Decoder::new(std::io::Cursor::new(b));
    dec.set_transformations(png::Transformations::EXPAND);
    let mut reader = match dec.read_info() {
        Ok(r) => r,
        Err(e) => return (false, format!("bad PNG: {e}")),
    };
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = match reader.next_frame(&mut buf) {
        Ok(i) => i,
        Err(e) => return (false, format!("bad PNG data: {e}")),
    };
    let ch = info.color_type.samples();
    let px = &buf[..info.buffer_size()];
    let mut colors = HashSet::new();
    let mut lum = 0u64;
    for p in px.chunks_exact(ch) {
        let (r, g, bl) = if ch >= 3 {
            (p[0], p[1], p[2])
        } else {
            (p[0], p[0], p[0])
        };
        colors.insert((r as u32) << 16 | (g as u32) << 8 | bl as u32);
        lum += (r as u64 + g as u64 + bl as u64) / 3;
    }
    let n = (px.len() / ch).max(1) as u64;
    let mean = lum / n;
    let ok = colors.len() > 64 && mean > 2 && mean < 253;
    (
        ok,
        format!(
            "{}x{} PNG, {} distinct colors, mean luma {}/255{}",
            info.width,
            info.height,
            colors.len(),
            mean,
            if ok { "" } else { " (looks blank!)" }
        ),
    )
}

fn glb(b: &[u8], req: Require) -> (bool, String) {
    if b.len() < 20 || &b[0..4] != b"glTF" {
        return (false, "missing glTF magic".into());
    }
    let ver = u32::from_le_bytes(b[4..8].try_into().unwrap());
    let len = u32::from_le_bytes(b[12..16].try_into().unwrap()) as usize;
    if &b[16..20] != b"JSON" || b.len() < 20 + len {
        return (false, "missing JSON chunk".into());
    }
    let doc: Value = match serde_json::from_slice(&b[20..20 + len]) {
        Ok(v) => v,
        Err(e) => return (false, format!("bad glTF JSON: {e}")),
    };
    let count = |k: &str| doc.get(k).and_then(Value::as_array).map_or(0, Vec::len);
    let joints = doc
        .get("skins")
        .and_then(Value::as_array)
        .and_then(|s| s.first())
        .and_then(|s| s.get("joints"))
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let (meshes, skins, anims) = (count("meshes"), count("skins"), count("animations"));
    let mut ok = ver == 2 && meshes > 0;
    let mut why = String::new();
    if req.skin && (skins == 0 || joints == 0) {
        ok = false;
        why += " [no skin/joints]";
    }
    if req.anim && anims == 0 {
        ok = false;
        why += " [no animation]";
    }
    (
        ok,
        format!(
            "glTF {ver}: {meshes} mesh, {} material, {} image, {} node, {skins} skin ({joints} joints), {anims} animation{why}",
            count("materials"),
            count("images"),
            count("nodes"),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glb_bytes(json: &str) -> Vec<u8> {
        let mut j = json.as_bytes().to_vec();
        while !j.len().is_multiple_of(4) {
            j.push(b' ');
        }
        let mut v = b"glTF".to_vec();
        v.extend(2u32.to_le_bytes());
        v.extend((20 + j.len() as u32).to_le_bytes());
        v.extend((j.len() as u32).to_le_bytes());
        v.extend(b"JSON");
        v.extend(j);
        v
    }

    #[test]
    fn glb_skin_and_animation_are_counted() {
        let g = glb_bytes(r#"{"meshes":[{}],"skins":[{"joints":[0,1,2]}],"animations":[{}]}"#);
        let (ok, d) = glb(
            &g,
            Require {
                skin: true,
                anim: true,
            },
        );
        assert!(ok, "{d}");
        assert!(d.contains("3 joints"));
    }

    #[test]
    fn glb_without_skin_fails_when_skin_required() {
        let g = glb_bytes(r#"{"meshes":[{}]}"#);
        assert!(glb(&g, Require::default()).0);
        assert!(
            !glb(
                &g,
                Require {
                    skin: true,
                    anim: false
                }
            )
            .0
        );
    }

    #[test]
    fn garbage_is_rejected() {
        assert!(!glb(b"nope", Require::default()).0);
        assert!(!png(b"not a png").0);
        assert!(!magic(b"xxxx", 0, b"ftyp", "MP4").0);
    }

    #[test]
    fn flat_png_is_reported_blank() {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, 16, 16);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            let mut w = enc.write_header().unwrap();
            w.write_image_data(&vec![0u8; 16 * 16 * 3]).unwrap();
        }
        let (ok, d) = png(&out);
        assert!(!ok, "{d}");
    }
}
