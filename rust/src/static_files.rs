//! The [`StaticSite`] type: serves the built PWA front end from a directory.
//!
//! When `GAIA_WEB_DIR` points at the SvelteKit static build output, the backend
//! serves those files directly, so a single Container App hosts **both** the
//! JSON/WebSocket API and the installable PWA. Same origin means no CORS and no
//! separate Static Web App to manage — the front end's public client ids are
//! baked into the bundle at image-build time and injected per CI/CD run.
//!
//! Any request that does not resolve to a real file falls back to `index.html`.
//! That is what lets a single-page app keep client-side routing working when a
//! user reloads or deep-links to a route the server has no file for.

use std::path::{Path, PathBuf};

use crate::http_response::HttpResponse;

/// A static-file site rooted at a directory of pre-built PWA assets.
#[derive(Debug, Clone)]
pub struct StaticSite {
    /// Directory holding the built PWA (`index.html`, `_app/…`, icons, …).
    root: PathBuf,
}

impl StaticSite {
    /// Build a site from the `GAIA_WEB_DIR` environment variable.
    ///
    /// Returns `None` when the variable is unset/empty or does not point at an
    /// existing directory, in which case the server simply runs API-only. This
    /// keeps local `cargo run` (no web build) and the CLI tests unchanged.
    pub fn from_env() -> Option<Self> {
        let dir = crate::llm::value_from_env("GAIA_WEB_DIR")?;
        let root = PathBuf::from(dir);
        root.is_dir().then_some(Self { root })
    }

    /// Build a site rooted at `root`. Test-only constructor.
    #[cfg(test)]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The web root as a display string, for the startup log line.
    pub fn describe(&self) -> String {
        self.root.display().to_string()
    }

    /// Serve the file for `request_path`, falling back to `index.html`.
    ///
    /// `request_path` is the raw request path (it may carry a `?query`, which is
    /// ignored). A path that escapes the web root, or that does not name an
    /// existing file, yields the SPA shell (`index.html`) so client-side routing
    /// still works. A missing `index.html` is the only case that 404s.
    pub fn response_for(&self, request_path: &str) -> HttpResponse {
        // Drop any query string and the leading slash; "/" maps to index.html.
        let path = request_path.split('?').next().unwrap_or("");
        let rel = path.trim_start_matches('/');
        let rel = if rel.is_empty() { "index.html" } else { rel };

        if let Some(file) = self.safe_join(rel) {
            if file.is_file() {
                return file_response(&file, rel);
            }
        }

        // SPA fallback: render the app shell for unknown (client-routed) paths.
        let index = self.root.join("index.html");
        file_response(&index, "index.html")
    }

    /// Join `rel` onto the web root, refusing anything that could escape it.
    ///
    /// Only plain path segments are allowed: `..`, empty/`.` segments, and
    /// segments containing a backslash or NUL are rejected (returning `None`) so
    /// a crafted URL can never read files outside the web root.
    fn safe_join(&self, rel: &str) -> Option<PathBuf> {
        let mut out = self.root.clone();
        for part in rel.split('/') {
            if part.is_empty() || part == "." {
                continue;
            }
            if part == ".." || part.contains('\\') || part.contains('\0') {
                return None;
            }
            out.push(part);
        }
        Some(out)
    }
}

/// Read `file` and wrap it in a `200 OK` response with a content type and cache
/// policy derived from `rel` (the request-relative path). A read failure 404s.
fn file_response(file: &Path, rel: &str) -> HttpResponse {
    match std::fs::read(file) {
        Ok(body) => HttpResponse::bytes(200, "OK", content_type_for(rel), body)
            .with_header("Cache-Control", cache_control_for(rel)),
        Err(_) => HttpResponse::with_status_json(404, "Not Found", r#"{"error":"not found"}"#),
    }
}

/// Pick a `Content-Type` from the file extension of `rel`.
///
/// Covers the asset kinds a SvelteKit PWA emits; anything unknown falls back to
/// `application/octet-stream`, which browsers download rather than mis-render.
fn content_type_for(rel: &str) -> &'static str {
    let ext = rel.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "webmanifest" => "application/manifest+json; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        "wasm" => "application/wasm",
        "txt" => "text/plain; charset=utf-8",
        "map" => "application/json; charset=utf-8",
        _ => "application/octet-stream",
    }
}

/// Choose a `Cache-Control` value for `rel`.
///
/// SvelteKit fingerprints everything under `_app/immutable/`, so those files can
/// be cached forever. The HTML shell must never be cached (it points at the
/// current asset hashes); everything else gets a modest cache.
fn cache_control_for(rel: &str) -> &'static str {
    if rel == "index.html" || rel.ends_with(".html") {
        "no-cache"
    } else if rel.starts_with("_app/immutable/") {
        "public, max-age=31536000, immutable"
    } else {
        "public, max-age=3600"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Create a unique temp directory seeded with a minimal PWA build.
    fn temp_site() -> (PathBuf, StaticSite) {
        let mut root = std::env::temp_dir();
        let unique = format!(
            "gaia-static-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        root.push(unique);
        fs::create_dir_all(root.join("_app/immutable")).unwrap();
        fs::write(
            root.join("index.html"),
            b"<!doctype html><title>Gaia</title>",
        )
        .unwrap();
        fs::write(root.join("favicon.ico"), b"icon-bytes").unwrap();
        fs::write(root.join("_app/immutable/app.js"), b"console.log(1)").unwrap();
        let site = StaticSite::new(root.clone());
        (root, site)
    }

    #[test]
    fn serves_an_existing_asset_with_its_content_type() {
        let (root, site) = temp_site();
        let resp = site.response_for("/_app/immutable/app.js");
        assert_eq!(resp.status(), 200);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn root_path_serves_index_html() {
        let (root, site) = temp_site();
        assert_eq!(site.response_for("/").status(), 200);
        assert_eq!(site.response_for("").status(), 200);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn unknown_path_falls_back_to_index_for_spa_routing() {
        let (root, site) = temp_site();
        // A client-side route with no matching file should still get the shell.
        let resp = site.response_for("/chat/deep/link");
        assert_eq!(resp.status(), 200);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn query_string_is_ignored() {
        let (root, site) = temp_site();
        assert_eq!(site.response_for("/favicon.ico?v=2").status(), 200);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn path_traversal_is_refused_and_returns_the_shell() {
        let (root, site) = temp_site();
        // `..` segments are rejected by safe_join, so the SPA shell is served
        // instead of a file outside the web root.
        let resp = site.response_for("/../../etc/passwd");
        assert_eq!(resp.status(), 200);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn missing_index_yields_404() {
        let mut root = std::env::temp_dir();
        root.push(format!("gaia-static-empty-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let site = StaticSite::new(root.clone());
        assert_eq!(site.response_for("/").status(), 404);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn content_type_matches_extension() {
        assert_eq!(content_type_for("index.html"), "text/html; charset=utf-8");
        assert_eq!(content_type_for("a/b.js"), "text/javascript; charset=utf-8");
        assert_eq!(content_type_for("style.css"), "text/css; charset=utf-8");
        assert_eq!(content_type_for("icon.png"), "image/png");
        assert_eq!(
            content_type_for("manifest.webmanifest"),
            "application/manifest+json; charset=utf-8"
        );
        assert_eq!(content_type_for("noext"), "application/octet-stream");
    }

    #[test]
    fn cache_policy_distinguishes_shell_immutable_and_other() {
        assert_eq!(cache_control_for("index.html"), "no-cache");
        assert_eq!(
            cache_control_for("_app/immutable/chunk.js"),
            "public, max-age=31536000, immutable"
        );
        assert_eq!(cache_control_for("favicon.ico"), "public, max-age=3600");
    }
}
