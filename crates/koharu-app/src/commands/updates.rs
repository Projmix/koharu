use std::time::Duration;

use semver::Version;
use serde::{Deserialize, Serialize};
use specta::Type;

use super::Error;

// Last integrated upstream: koharu-rs/koharu@52bb4f55e2266f72c1523562a583c827ec191ceb.
// Advance this baseline only when integrating upstream, independently of fork releases.
const INTEGRATED_UPSTREAM_VERSION: &str = "0.83.5";

#[derive(Debug, Serialize, Type)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UpstreamUpdate {
    version: String,
    integrated_version: String,
    url: String,
}

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    draft: bool,
    prerelease: bool,
}

impl Release {
    fn newer_than_integrated(self) -> Option<UpstreamUpdate> {
        if self.draft || self.prerelease {
            return None;
        }
        let version =
            Version::parse(self.tag_name.strip_prefix('v').unwrap_or(&self.tag_name)).ok()?;
        let integrated =
            Version::parse(INTEGRATED_UPSTREAM_VERSION).expect("valid upstream baseline");
        if !version.pre.is_empty() || !version.cmp_precedence(&integrated).is_gt() {
            return None;
        }
        Some(UpstreamUpdate {
            version: version.to_string(),
            integrated_version: INTEGRATED_UPSTREAM_VERSION.to_owned(),
            // Use the canonical repository, never an arbitrary URL from release metadata.
            url: format!(
                "https://github.com/koharu-rs/koharu/releases/tag/{}",
                self.tag_name
            ),
        })
    }
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn check_upstream_update() -> Result<Option<UpstreamUpdate>, Error> {
    let client = reqwest::Client::builder()
        .user_agent(concat!("Projmix-koharu/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(8))
        .build()?;
    let release = client
        .get("https://api.github.com/repos/koharu-rs/koharu/releases/latest")
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()
        .await?
        .error_for_status()?
        .json::<Release>()
        .await?;
    Ok(release.newer_than_integrated())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compares_stable_releases_with_integrated_upstream_not_the_fork_version() {
        for tag in [
            "0.82.1",
            "v0.83.5",
            "0.83.5+fork.20",
            "0.84.0-beta.1",
            "not-a-version",
        ] {
            assert!(
                Release {
                    tag_name: tag.to_owned(),
                    draft: false,
                    prerelease: false
                }
                .newer_than_integrated()
                .is_none(),
                "{tag}"
            );
        }
        for tag in ["0.83.6", "v0.84.0", "0.100.0", "1.0.0"] {
            let update = Release {
                tag_name: tag.to_owned(),
                draft: false,
                prerelease: false,
            }
            .newer_than_integrated()
            .expect(tag);
            assert_eq!(update.integrated_version, "0.83.5");
            assert_eq!(
                update.url,
                format!("https://github.com/koharu-rs/koharu/releases/tag/{tag}")
            );
        }
        for (draft, prerelease) in [(true, false), (false, true)] {
            assert!(
                Release {
                    tag_name: "1.0.0".to_owned(),
                    draft,
                    prerelease
                }
                .newer_than_integrated()
                .is_none()
            );
        }
    }
}
