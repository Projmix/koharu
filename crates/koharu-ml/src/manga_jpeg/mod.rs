//! MangaJPEG (bunzero, 2021): 1x ESRGAN models that remove JPEG artifacts from
//! greyscale manga while keeping screentone, one per JPEG quality range.
//!
//! https://openmodeldb.info/models/1x-MangaJPEGHQ. The weights are
//! CC BY-NC-SA 4.0: the fork's owner accepted that license for this
//! non-commercial fork, and they are downloaded on first use, never bundled.

mod model;
mod processor;

use anyhow::{Context, Result};
use image::GrayImage;
use koharu_runtime::UrlFile;
use koharu_torch::Device;

use crate::backend::TryIntoDevice;

use self::model::Model;

/// The published models, named after the JPEG quality range each was trained
/// for: HQPlus 95-80, HQ 80-50, MQ 60-30, LQ 25-5.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MangaJpegVariant {
    HqPlus,
    Hq,
    Mq,
    Lq,
}

impl MangaJpegVariant {
    pub const ALL: [Self; 4] = [Self::HqPlus, Self::Hq, Self::Mq, Self::Lq];

    /// The model for a libjpeg quality. Deliberately one step lighter than
    /// the published ranges: on a screentone crop recompressed at 85/60/40/20
    /// the best model was HQPlus/HQPlus/MQ/MQ, and LQ added halos at every
    /// quality (docs/fork/scan-restoration.md).
    #[must_use]
    pub const fn for_quality(quality: u8) -> Self {
        match quality {
            55.. => Self::HqPlus,
            45..=54 => Self::Hq,
            12..=44 => Self::Mq,
            _ => Self::Lq,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::HqPlus => "MangaJPEG HQPlus",
            Self::Hq => "MangaJPEG HQ",
            Self::Mq => "MangaJPEG MQ",
            Self::Lq => "MangaJPEG LQ",
        }
    }

    // URLs and SHA-256 digests from OpenModelDB's records for these models.
    const fn weights(self) -> UrlFile<'static> {
        match self {
            Self::HqPlus => UrlFile::pinned(
                "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/1x-MangaJPEGHQPlus.pth",
                "3126339603a3876f541d054bd40cdad417141a32c14e91fd6adaeaf04677be34",
                "1x-MangaJPEGHQPlus.pth",
            ),
            Self::Hq => UrlFile::pinned(
                "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/1x-MangaJPEGHQ.pth",
                "e2898bfba74083025a2bd2326b6c159031b882a890bd1cc30fd6910cda2ce4ee",
                "1x-MangaJPEGHQ.pth",
            ),
            Self::Mq => UrlFile::pinned(
                "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/1x-MangaJPEGMQ.pth",
                "42529335f43ac04f615af3442b2570d23dc64a206e2da4741bf5f835206354ae",
                "1x-MangaJPEGMQ.pth",
            ),
            Self::Lq => UrlFile::pinned(
                "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/1x-MangaJPEGLQ.pth",
                "af11ec91b6a38e0109cbb926b19e8a01975f947aacb3490f008dcab06e9b922d",
                "1x-MangaJPEGLQ.pth",
            ),
        }
    }
}

#[derive(Debug)]
pub struct MangaJpeg {
    variant: MangaJpegVariant,
    model: Model,
    device: Device,
}

impl MangaJpeg {
    pub async fn load(variant: MangaJpegVariant, device: crate::Device) -> Result<Self> {
        let device: Device = device.try_into_device()?;
        let path = variant
            .weights()
            .resolve()
            .await
            .with_context(|| format!("failed to resolve {} weights", variant.name()))?;
        let mut model = Model::new(device);
        model
            .load(&path)
            .with_context(|| format!("failed to load {}", path.display()))?;
        Ok(Self {
            variant,
            model,
            device,
        })
    }

    #[must_use]
    pub const fn variant(&self) -> MangaJpegVariant {
        self.variant
    }

    /// Restores a greyscale image at its own size. `stopped` is checked
    /// between tiles; a stop returns `None`.
    pub fn inference(
        &self,
        image: &GrayImage,
        stopped: &dyn Fn() -> bool,
    ) -> Result<Option<GrayImage>> {
        koharu_torch::no_grad(|| processor::restore(&self.model, self.device, image, stopped))
    }
}
