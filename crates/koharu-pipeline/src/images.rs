use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use anyhow::{Context as _, Result};
use koharu_scene::{AssetRole, BlobId, EntityId, Snapshot};

#[derive(Default)]
pub(crate) struct ImageCache {
    decoded: Mutex<HashMap<BlobId, Arc<tokio::sync::OnceCell<Arc<image::DynamicImage>>>>>,
}

impl ImageCache {
    /// The page as later stages read it: the visible scan enhancement layer
    /// when the page has one, otherwise the imported source.
    pub(crate) async fn page_image(
        &self,
        scene: &Snapshot,
        page: EntityId,
    ) -> Result<Option<Arc<image::DynamicImage>>> {
        match crate::stages::visible_enhancement_layer(scene, page)? {
            Some(layer) => self.get(scene, layer, "source").await,
            None => self.get(scene, page, "source").await,
        }
    }

    pub(crate) async fn get(
        &self,
        scene: &Snapshot,
        entity: EntityId,
        role: &str,
    ) -> Result<Option<Arc<image::DynamicImage>>> {
        let role = AssetRole::new(role)?;
        let Some(asset) = scene.asset(entity, &role)? else {
            return Ok(None);
        };
        let cell = self
            .decoded
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .entry(asset.blob)
            .or_default()
            .clone();
        let image = cell
            .get_or_try_init(|| async {
                let bytes = scene.read_blob(asset.blob).await?;
                let role = role.clone();
                tokio_rayon::spawn(move || {
                    image::load_from_memory(&bytes)
                        .map(Arc::new)
                        .with_context(|| {
                            format!(
                                "failed to decode {} image for entity {entity}",
                                role.as_str()
                            )
                        })
                })
                .await
            })
            .await?
            .clone();
        Ok(Some(image))
    }
}
