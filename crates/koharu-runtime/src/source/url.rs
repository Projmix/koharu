use std::path::PathBuf;

use anyhow::{Context, ensure};
use sha2::{Digest, Sha256};

use crate::{Store, download};

/// A file outside Hugging Face, pinned by the SHA-256 its publisher lists.
/// The digest is checked before the file enters the store, so a changed
/// upstream file fails instead of loading.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct UrlFile<'a> {
    url: &'a str,
    sha256: &'a str,
    filename: &'a str,
}

impl<'a> UrlFile<'a> {
    #[must_use]
    pub const fn pinned(url: &'a str, sha256: &'a str, filename: &'a str) -> Self {
        Self {
            url,
            sha256,
            filename,
        }
    }

    #[must_use]
    pub fn exists(self) -> bool {
        self.path().is_file()
    }

    #[tracing::instrument(skip_all)]
    pub async fn resolve(self) -> anyhow::Result<PathBuf> {
        Store::file(self.path(), move |stage| async move {
            download::fetch(self.url, &stage).await?;
            let bytes = tokio::fs::read(&stage)
                .await
                .with_context(|| format!("failed to read {}", stage.display()))?;
            let digest = Sha256::digest(&bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            ensure!(
                digest == self.sha256,
                "{} has SHA-256 {digest}, expected {}",
                self.url,
                self.sha256
            );
            Ok(())
        })
        .await
    }

    fn path(self) -> PathBuf {
        Store::root()
            .join("sha256")
            .join(self.sha256)
            .join(self.filename)
    }
}
