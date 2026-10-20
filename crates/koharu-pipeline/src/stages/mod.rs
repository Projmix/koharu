mod detection;
mod fal;
mod inpainting;
mod markup;
mod ocr;
mod openrouter_image;
mod replicate;
mod translation;

use std::{collections::BTreeSet, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use koharu_scene::{Edit, EntityId, Generation, Patch, ProducerId, Snapshot};

pub use detection::{KoharuLayoutRFDetrSeg2XLConfig, text_layer_placement};
pub(crate) use fal::FalModel;
pub(crate) use fal::normalize_cleanup_prompt;
pub use fal::{ImageEditConfig, InpaintingApplyMode};
pub use inpainting::{Flux2KleinConfig, RoremMixedConfig};
pub use ocr::ocr_unverified;
pub(crate) use replicate::ReplicateModel;
pub use translation::ChapterTranslation;

use crate::{
    Bounds, ImageCache, InpaintingIntent, InpaintingMask, InpaintingModelChoice,
    InpaintingProvider, PipelineConfig, Stage, StageTarget, StopToken,
};

pub(crate) async fn inpainting_models() -> Result<Vec<InpaintingModelChoice>> {
    let discovered = match koharu_runtime::http_client() {
        Ok(client) => match openrouter_image::models(&client).await {
            Ok(models) => models,
            Err(error) => {
                tracing::warn!(%error, "failed to list OpenRouter image models");
                Vec::new()
            }
        },
        Err(error) => {
            tracing::warn!(%error, "failed to create the OpenRouter image catalog client");
            Vec::new()
        }
    };
    Ok(merge_inpainting_models(discovered))
}

fn merge_inpainting_models(
    discovered: Vec<openrouter_image::OpenRouterImageModelChoice>,
) -> Vec<InpaintingModelChoice> {
    let mut models: Vec<_> = FalModel::ALL
        .into_iter()
        .map(|model| InpaintingModelChoice {
            provider: InpaintingProvider::Fal,
            model: model.id().to_owned(),
            name: model.name().to_owned(),
        })
        .collect();
    models.extend(
        ReplicateModel::ALL
            .into_iter()
            .map(|model| InpaintingModelChoice {
                provider: InpaintingProvider::Replicate,
                model: model.id().to_owned(),
                name: model.name().to_owned(),
            }),
    );
    models.extend([
        InpaintingModelChoice {
            provider: InpaintingProvider::OpenRouter,
            model: "microsoft/mai-image-2.5".to_owned(),
            name: "Microsoft MAI Image 2.5".to_owned(),
        },
        InpaintingModelChoice {
            provider: InpaintingProvider::OpenRouter,
            model: "microsoft/mai-image-2.5-pro".to_owned(),
            name: "Microsoft MAI Image 2.5 Pro".to_owned(),
        },
    ]);

    for model in discovered {
        if !models.iter().any(|choice| {
            choice.provider == InpaintingProvider::OpenRouter && choice.model == model.model
        }) {
            models.push(InpaintingModelChoice {
                provider: InpaintingProvider::OpenRouter,
                model: model.model,
                name: model.name,
            });
        }
    }
    models
}

#[derive(Clone)]
pub(crate) struct StageInput {
    scene: koharu_scene::Snapshot,
    target: StageTarget,
    pages: Arc<[EntityId]>,
    entities: Option<Arc<BTreeSet<EntityId>>>,
    region: Option<Bounds>,
    images: Arc<ImageCache>,
    inpainting_mask: Option<InpaintingMask>,
    image_edit_reviewer: Option<Arc<dyn crate::ImageEditReviewer>>,
    stop: StopToken,
    ocr_pass: crate::stage::OcrPass,
    shorten: bool,
}

impl StageInput {
    pub(crate) fn new(
        scene: Snapshot,
        page: EntityId,
        entities: Option<Arc<BTreeSet<EntityId>>>,
        region: Option<Bounds>,
        images: Arc<ImageCache>,
        inpainting_mask: Option<InpaintingMask>,
    ) -> Self {
        Self {
            scene,
            target: StageTarget::Page(page),
            pages: Arc::from([page]),
            entities,
            region,
            images,
            inpainting_mask,
            image_edit_reviewer: None,
            stop: StopToken::default(),
            ocr_pass: crate::stage::OcrPass::Final,
            shorten: false,
        }
    }

    pub(crate) fn chapter(scene: Snapshot, pages: Arc<[EntityId]>) -> Self {
        Self {
            scene,
            target: StageTarget::Chapter,
            pages,
            entities: None,
            region: None,
            images: Arc::new(ImageCache::default()),
            inpainting_mask: None,
            image_edit_reviewer: None,
            stop: StopToken::default(),
            ocr_pass: crate::stage::OcrPass::Final,
            shorten: false,
        }
    }

    pub(crate) fn with_stop(mut self, stop: StopToken) -> Self {
        self.stop = stop;
        self
    }

    pub(crate) fn with_image_edit_reviewer(
        mut self,
        reviewer: Option<Arc<dyn crate::ImageEditReviewer>>,
    ) -> Self {
        self.image_edit_reviewer = reviewer;
        self
    }

    pub(crate) fn with_ocr_pass(mut self, pass: crate::stage::OcrPass) -> Self {
        self.ocr_pass = pass;
        self
    }

    pub(crate) fn with_shorten(mut self, shorten: bool) -> Self {
        self.shorten = shorten;
        self
    }

    pub(crate) fn target(&self) -> StageTarget {
        self.target
    }

    pub(crate) fn page(&self) -> Result<EntityId> {
        self.target
            .page()
            .ok_or_else(|| anyhow::anyhow!("chapter work has no page"))
    }

    pub(crate) fn pages(&self) -> &[EntityId] {
        &self.pages
    }

    fn inpainting_intent(&self) -> Option<InpaintingIntent> {
        self.inpainting_mask.as_ref().map(|mask| mask.intent)
    }

    fn contains_entity(&self, entity: EntityId) -> Result<bool> {
        crate::scope::contains_entity(
            &self.scene,
            self.page()?,
            self.entities.as_deref(),
            self.region,
            entity,
        )
    }
}

#[async_trait]
trait StageProcessor: Send + Sync {
    fn model(&self, input: &StageInput) -> String;
    fn recovers_from_memory_pressure(&self, _input: &StageInput) -> bool {
        false
    }
    fn skip(&self, _input: &StageInput) -> Result<bool> {
        Ok(false)
    }
    fn unload(&self) -> bool;
    async fn load(&self, input: &StageInput) -> Result<()>;
    async fn process(&self, input: StageInput) -> Result<Patch>;
}

pub(crate) struct Stages {
    detection: detection::Processor,
    ocr: ocr::Processor,
    markup: markup::Processor,
    translation: translation::Processor,
    inpainting: inpainting::Processor,
    manual_inpainting: Option<inpainting::Processor>,
    restoration_inpainting: inpainting::Processor,
}

impl Stages {
    pub(crate) fn new(
        config: &PipelineConfig,
        translator: koharu_translator::Translator,
        device: &koharu_ml::Device,
    ) -> Result<Self> {
        let automatic_inpainting = config.inpainting()?;
        let manual_inpainting = config.manual_inpainting()?;
        let restoration_inpainting = config.restoration_inpainting()?;
        let manual_inpainting = (manual_inpainting != automatic_inpainting)
            .then(|| inpainting::Processor::new(manual_inpainting, device.clone()))
            .transpose()?;
        Ok(Self {
            detection: detection::Processor::new(
                config.detection()?,
                config.translation.source_language,
                device.clone(),
            ),
            ocr: ocr::Processor::new(
                config.ocr.clone(),
                config.translation.source_language,
                translator.clone(),
                device.clone(),
            ),
            markup: markup::Processor::new(
                config.markup.clone(),
                config.translation.source_language,
                translator.clone(),
            ),
            translation: translation::Processor::new(config.translation.clone(), translator),
            inpainting: inpainting::Processor::new(automatic_inpainting, device.clone())?,
            manual_inpainting,
            restoration_inpainting: inpainting::Processor::new(
                restoration_inpainting,
                device.clone(),
            )?,
        })
    }

    fn processor(&self, stage: Stage, input: &StageInput) -> &dyn StageProcessor {
        match stage {
            Stage::Detection => &self.detection,
            Stage::Ocr => &self.ocr,
            Stage::Markup => &self.markup,
            Stage::Translation => &self.translation,
            Stage::Inpainting if input.inpainting_intent() == Some(InpaintingIntent::Remove) => {
                self.manual_inpainting.as_ref().unwrap_or(&self.inpainting)
            }
            Stage::Inpainting if input.inpainting_intent() == Some(InpaintingIntent::Restore) => {
                &self.restoration_inpainting
            }
            Stage::Inpainting => &self.inpainting,
        }
    }

    pub(crate) fn model(&self, stage: Stage, input: &StageInput) -> String {
        self.processor(stage, input).model(input)
    }

    pub(crate) fn skip(&self, stage: Stage, input: &StageInput) -> Result<bool> {
        self.processor(stage, input).skip(input)
    }

    pub(crate) fn recovers_from_memory_pressure(&self, stage: Stage, input: &StageInput) -> bool {
        self.processor(stage, input)
            .recovers_from_memory_pressure(input)
    }

    pub(crate) async fn load(&self, stage: Stage, input: &StageInput) -> Result<()> {
        self.processor(stage, input).load(input).await
    }

    pub(crate) async fn process(&self, stage: Stage, input: StageInput) -> Result<Patch> {
        self.processor(stage, &input).process(input).await
    }

    pub(crate) fn unload(&self, stage: Stage) -> bool {
        match stage {
            Stage::Detection => self.detection.unload(),
            Stage::Ocr => self.ocr.unload(),
            Stage::Markup => self.markup.unload(),
            Stage::Translation => self.translation.unload(),
            Stage::Inpainting => {
                let automatic = self.inpainting.unload();
                self.manual_inpainting
                    .as_ref()
                    .is_some_and(StageProcessor::unload)
                    || self.restoration_inpainting.unload()
                    || automatic
            }
        }
    }
}

fn generation(producer: &str, model: &str) -> Result<Generation> {
    let mut generation = Generation::new(ProducerId::new(producer)?);
    generation.model = Some(model.to_owned());
    Ok(generation)
}

fn finish(edit: Edit) -> Result<Patch> {
    edit.finish().map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inpainting_catalog_pins_fal_edits_and_openrouter_fallbacks() {
        let models = merge_inpainting_models(Vec::new());

        assert_eq!(
            models
                .iter()
                .map(|choice| (choice.provider, choice.model.as_str()))
                .collect::<Vec<_>>(),
            [
                (InpaintingProvider::Fal, "microsoft/mai-image-2.5/edit"),
                (InpaintingProvider::Fal, "microsoft/mai-image-2.5-pro/edit"),
                (InpaintingProvider::Fal, "fal-ai/flux-kontext/dev"),
                (InpaintingProvider::Fal, "fal-ai/flux-pro/kontext"),
                (InpaintingProvider::Fal, "fal-ai/flux-pro/kontext/max"),
                (InpaintingProvider::Fal, "fal-ai/flux-2/edit"),
                (InpaintingProvider::Fal, "fal-ai/flux-2-pro/edit"),
                (InpaintingProvider::Fal, "fal-ai/flux-2-flex/edit"),
                (InpaintingProvider::Fal, "fal-ai/flux-2/klein/4b/edit"),
                (InpaintingProvider::Fal, "fal-ai/flux-2/klein/9b/edit"),
                (InpaintingProvider::Fal, "fal-ai/qwen-image-edit-2511"),
                (
                    InpaintingProvider::Fal,
                    "fal-ai/bytedance/seedream/v4.5/edit"
                ),
                (
                    InpaintingProvider::Replicate,
                    "black-forest-labs/flux-2-dev"
                ),
                (
                    InpaintingProvider::Replicate,
                    "black-forest-labs/flux-2-pro"
                ),
                (
                    InpaintingProvider::Replicate,
                    "black-forest-labs/flux-2-max"
                ),
                (
                    InpaintingProvider::Replicate,
                    "black-forest-labs/flux-2-flex"
                ),
                (
                    InpaintingProvider::Replicate,
                    "black-forest-labs/flux-2-klein-4b"
                ),
                (
                    InpaintingProvider::Replicate,
                    "black-forest-labs/flux-2-klein-9b"
                ),
                (
                    InpaintingProvider::Replicate,
                    "black-forest-labs/flux-2-klein-4b-base"
                ),
                (
                    InpaintingProvider::Replicate,
                    "black-forest-labs/flux-2-klein-9b-base"
                ),
                (
                    InpaintingProvider::Replicate,
                    "black-forest-labs/flux-kontext-dev"
                ),
                (
                    InpaintingProvider::Replicate,
                    "black-forest-labs/flux-kontext-pro"
                ),
                (
                    InpaintingProvider::Replicate,
                    "black-forest-labs/flux-kontext-max"
                ),
                (InpaintingProvider::OpenRouter, "microsoft/mai-image-2.5"),
                (
                    InpaintingProvider::OpenRouter,
                    "microsoft/mai-image-2.5-pro"
                ),
            ]
        );
    }

    #[test]
    fn inpainting_catalog_does_not_duplicate_pinned_fallbacks() {
        let models = merge_inpainting_models(vec![
            openrouter_image::OpenRouterImageModelChoice {
                model: "microsoft/mai-image-2.5".to_owned(),
                name: "Microsoft: MAI-Image-2.5".to_owned(),
            },
            openrouter_image::OpenRouterImageModelChoice {
                model: "provider/other-edit".to_owned(),
                name: "Other Edit".to_owned(),
            },
        ]);

        assert_eq!(
            models
                .iter()
                .filter(|choice| {
                    choice.provider == InpaintingProvider::OpenRouter
                        && choice.model == "microsoft/mai-image-2.5"
                })
                .count(),
            1
        );
        assert_eq!(models.last().unwrap().model, "provider/other-edit");
    }

    fn asset(bytes: &'static [u8]) -> koharu_scene::AssetInput {
        koharu_scene::AssetInput::new(
            bytes,
            "image/png",
            koharu_scene::AssetMetadata {
                width: Some(1),
                height: Some(1),
                attributes: std::collections::BTreeMap::new(),
            },
        )
    }

    #[tokio::test]
    async fn registry_contains_every_stage() {
        let translator = koharu_translator::Translator::from_config(
            koharu_ml::Device::cpu(),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        )
        .unwrap();
        let stages = Stages::new(
            &PipelineConfig::default(),
            translator,
            &koharu_ml::Device::cpu(),
        )
        .unwrap();
        let session = koharu_scene::Session::memory().await.unwrap();
        let input = StageInput::new(
            session.snapshot(),
            EntityId::new(),
            None,
            None,
            Arc::new(ImageCache::default()),
            None,
        );

        assert_eq!(
            Stage::ALL.map(|stage| stages.model(stage, &input)),
            [
                "koharu-layout-rfdetr-seg-2xl",
                "paddleocr-vl-1.6",
                "openrouter: google/gemini-3.7-flash",
                "local",
                "lama",
            ]
        );
    }

    #[tokio::test]
    async fn inpainting_intents_select_independent_automatic_remove_and_restore_processors() {
        let translator = koharu_translator::Translator::from_config(
            koharu_ml::Device::cpu(),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        )
        .unwrap();
        let mut config = PipelineConfig::default();
        config.inpainting.method = crate::InpaintingMethod::Api;
        config.inpainting.api.model = "microsoft/mai-image-2.5-pro/edit".to_owned();
        let stages = Stages::new(&config, translator, &koharu_ml::Device::cpu()).unwrap();
        let session = koharu_scene::Session::memory().await.unwrap();
        let page = EntityId::new();
        let automatic = StageInput::new(
            session.snapshot(),
            page,
            None,
            None,
            Arc::new(ImageCache::default()),
            None,
        );
        let manual = StageInput::new(
            session.snapshot(),
            page,
            None,
            None,
            Arc::new(ImageCache::default()),
            Some(crate::InpaintingMask {
                page,
                png: Arc::from([]),
                intent: crate::InpaintingIntent::Remove,
            }),
        );
        let restore = StageInput::new(
            session.snapshot(),
            page,
            None,
            None,
            Arc::new(ImageCache::default()),
            Some(crate::InpaintingMask {
                page,
                png: Arc::from([]),
                intent: crate::InpaintingIntent::Restore,
            }),
        );

        assert_eq!(
            stages.model(Stage::Inpainting, &automatic),
            "microsoft/mai-image-2.5-pro/edit"
        );
        assert_eq!(stages.model(Stage::Inpainting, &manual), "lama");
        assert_eq!(
            stages.model(Stage::Inpainting, &restore),
            "microsoft/mai-image-2.5/edit"
        );
    }

    #[tokio::test]
    async fn memory_pressure_recovery_is_limited_to_locally_managed_models() {
        let translator = koharu_translator::Translator::from_config(
            koharu_ml::Device::cpu(),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        )
        .unwrap();
        let session = koharu_scene::Session::memory().await.unwrap();
        let page = EntityId::new();
        let input = StageInput::new(
            session.snapshot(),
            page,
            None,
            None,
            Arc::new(ImageCache::default()),
            None,
        );

        let local = Stages::new(
            &PipelineConfig::default(),
            translator.clone(),
            &koharu_ml::Device::cpu(),
        )
        .unwrap();
        assert_eq!(
            Stage::ALL.map(|stage| local.recovers_from_memory_pressure(stage, &input)),
            [true, true, false, true, true]
        );

        let mut remote_config = PipelineConfig::default();
        remote_config.ocr.method = crate::OcrMethod::Api;
        remote_config.translation.page.model.provider = koharu_translator::Provider::OpenRouter;
        remote_config.translation.chapter.model.provider = koharu_translator::Provider::OpenRouter;
        remote_config.inpainting.method = crate::InpaintingMethod::Api;
        let remote = Stages::new(&remote_config, translator, &koharu_ml::Device::cpu()).unwrap();
        assert_eq!(
            Stage::ALL.map(|stage| remote.recovers_from_memory_pressure(stage, &input)),
            [true, false, false, false, false]
        );
        let chapter = StageInput::chapter(session.snapshot(), Arc::from([page]));
        assert!(!remote.recovers_from_memory_pressure(Stage::Translation, &chapter));
    }

    #[tokio::test]
    async fn translation_and_inpainting_compose_without_weakening_text_guards() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(
                koharu_scene::PageDraft::new("page", 1.0, 1.0),
                koharu_scene::At::End,
            )
            .unwrap();
        let text = setup.add_text_content(page, koharu_scene::At::End).unwrap();
        setup
            .set(
                text,
                &koharu_scene::SourceText {
                    text: koharu_scene::Authored::user("before".to_owned()),
                    language: None,
                },
            )
            .unwrap();
        setup
            .set_asset(
                page,
                &koharu_scene::AssetRole::new("source").unwrap(),
                asset(b"source"),
            )
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let base = session.snapshot();

        let mut text_edit = base.edit();
        text_edit.observe::<koharu_scene::SourceText>(text).unwrap();
        text_edit
            .observe::<koharu_scene::Translation>(text)
            .unwrap();
        text_edit
            .set(
                text,
                &koharu_scene::Translation {
                    text: koharu_scene::Authored::user("after".to_owned()),
                    language: None,
                },
            )
            .unwrap();
        let text_patch = text_edit.finish().unwrap();

        let mut image_edit = base.edit();
        image_edit.observe_assets(page).unwrap();
        let cleanup = image_edit
            .add_entity(page, koharu_scene::At::Start)
            .unwrap();
        image_edit
            .set(
                cleanup,
                &koharu_scene::RasterLayer {
                    origin: koharu_scene::Origin::User,
                    name: "Cleanup".to_owned(),
                    kind: koharu_scene::RasterLayerKind::Cleanup,
                },
            )
            .unwrap();
        image_edit
            .set_asset(
                cleanup,
                &koharu_scene::AssetRole::new("source").unwrap(),
                asset(b"clean"),
            )
            .unwrap();
        let image_patch = image_edit.finish().unwrap();

        let image_first = base.preview([&image_patch]).unwrap();
        assert!(text_patch.rebase_on(&image_first).is_ok());
        let text_first = base.preview([&text_patch]).unwrap();
        assert!(image_patch.rebase_on(&text_first).is_ok());

        let changed_source = base
            .patch(|edit| {
                edit.set(
                    text,
                    &koharu_scene::SourceText {
                        text: koharu_scene::Authored::user("changed".to_owned()),
                        language: None,
                    },
                )
            })
            .unwrap();
        let changed_source = base.preview([&changed_source]).unwrap();
        assert!(text_patch.rebase_on(&changed_source).is_err());
    }
}
