use anyhow::{Result, bail};
use koharu_translator::{
    GenerationConfig, Language, ModelSelection as TranslationModelSelection, Provider,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer, ser::SerializeMap as _};
use specta::Type;

use crate::stages::{
    Flux2KleinConfig, ImageEditConfig, KoharuLayoutRFDetrSeg2XLConfig, RoremMixedConfig,
    normalize_cleanup_prompt,
};

#[derive(Clone, Debug, PartialEq, Type)]
pub struct PipelineConfig {
    pub detection: DetectionModel,
    pub ocr: OcrConfig,
    pub translation: TranslationConfig,
    pub inpainting: InpaintingConfig,
    /// Settings for every model are kept independently of the active model.
    /// The active stage fields above only select which profile is used.
    pub processor: ProcessorConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
struct PipelineFile {
    detection: ModelSelection,
    ocr: OcrConfig,
    translation: TranslationConfig,
    inpainting: InpaintingFile,
    #[serde(default)]
    processor: ProcessorConfig,
}

impl Default for PipelineFile {
    fn default() -> Self {
        Self {
            detection: ModelSelection {
                model: "koharu-layout-rfdetr-seg-2xl".to_owned(),
            },
            ocr: OcrConfig::default(),
            translation: TranslationConfig::default(),
            inpainting: InpaintingFile::default(),
            processor: ProcessorConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
struct ModelSelection {
    model: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
struct InpaintingFile {
    method: InpaintingMethod,
    local_model: ModelSelection,
    #[serde(default = "lama_selection")]
    manual_model: ModelSelection,
    api: ApiInpaintingConfig,
}

fn lama_selection() -> ModelSelection {
    ModelSelection {
        model: "lama".to_owned(),
    }
}

impl Default for InpaintingFile {
    fn default() -> Self {
        Self {
            method: InpaintingMethod::Local,
            local_model: ModelSelection {
                model: "lama".to_owned(),
            },
            manual_model: lama_selection(),
            api: ApiInpaintingConfig::default(),
        }
    }
}

impl Serialize for PipelineConfig {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let detection = match &self.detection {
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_) => "koharu-layout-rfdetr-seg-2xl",
        };
        let local_model = local_inpainting_model_id(&self.inpainting.local_model);
        let manual_model = local_inpainting_model_id(&self.inpainting.manual_model);
        let mut processor = self.processor.clone();
        let DetectionModel::KoharuLayoutRFDetrSeg2XL(config) = &self.detection;
        processor
            .koharu_layout_rfdetr_seg_2xl
            .get_or_insert_with(|| config.clone());
        remember_local_inpainting_profile(&mut processor, &self.inpainting.local_model);
        remember_local_inpainting_profile(&mut processor, &self.inpainting.manual_model);
        if self.inpainting.api.provider == InpaintingProvider::Fal {
            let profile = ImageEditConfig {
                prompt: self.inpainting.api.prompt.clone(),
                apply_mode: self.inpainting.api.apply_mode,
            };
            match self.inpainting.api.model.as_str() {
                "microsoft/mai-image-2.5/edit" => {
                    processor.mai_image_2_5_edit = Some(profile);
                }
                "microsoft/mai-image-2.5-pro/edit" => {
                    processor.mai_image_2_5_pro_edit = Some(profile);
                }
                _ => {}
            }
        }
        PipelineFile {
            detection: ModelSelection {
                model: detection.to_owned(),
            },
            ocr: self.ocr.clone(),
            translation: self.translation.clone(),
            inpainting: InpaintingFile {
                method: self.inpainting.method,
                local_model: ModelSelection {
                    model: local_model.to_owned(),
                },
                manual_model: ModelSelection {
                    model: manual_model.to_owned(),
                },
                api: self.inpainting.api.clone(),
            },
            processor,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PipelineConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let file = PipelineFile::deserialize(deserializer)?;
        let mut processor = file.processor;
        for profile in [
            processor.mai_image_2_5_edit.as_mut(),
            processor.mai_image_2_5_pro_edit.as_mut(),
        ]
        .into_iter()
        .flatten()
        {
            if normalize_cleanup_prompt(&mut profile.prompt) {
                profile.apply_mode = crate::InpaintingApplyMode::FullPage;
            }
        }
        let detection = match file.detection.model.as_str() {
            "koharu-layout-rfdetr-seg-2xl" => DetectionModel::KoharuLayoutRFDetrSeg2XL(
                processor
                    .koharu_layout_rfdetr_seg_2xl
                    .clone()
                    .unwrap_or_default(),
            ),
            model => {
                return Err(serde::de::Error::custom(format!(
                    "unsupported detection model {model}"
                )));
            }
        };
        let local_model =
            restore_local_inpainting_model(&file.inpainting.local_model.model, &processor)
                .map_err(serde::de::Error::custom)?;
        let manual_model =
            restore_local_inpainting_model(&file.inpainting.manual_model.model, &processor)
                .map_err(serde::de::Error::custom)?;
        let mut api = file.inpainting.api;
        if normalize_cleanup_prompt(&mut api.prompt) {
            api.apply_mode = crate::InpaintingApplyMode::FullPage;
        }
        if api.provider == InpaintingProvider::Fal {
            let profile = match api.model.as_str() {
                "microsoft/mai-image-2.5/edit" => processor.mai_image_2_5_edit.as_ref(),
                "microsoft/mai-image-2.5-pro/edit" => processor.mai_image_2_5_pro_edit.as_ref(),
                _ => None,
            };
            if let Some(profile) = profile {
                api.prompt.clone_from(&profile.prompt);
                api.apply_mode = profile.apply_mode;
            }
        }
        Ok(Self {
            detection,
            ocr: file.ocr,
            translation: file.translation,
            inpainting: InpaintingConfig {
                method: file.inpainting.method,
                local_model,
                manual_model,
                api,
            },
            processor,
        })
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            detection: DetectionModel::KoharuLayoutRFDetrSeg2XL(
                KoharuLayoutRFDetrSeg2XLConfig::default(),
            ),
            ocr: OcrConfig::default(),
            translation: TranslationConfig::default(),
            inpainting: InpaintingConfig::default(),
            processor: ProcessorConfig::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Type)]
pub struct TranslationConfig {
    /// Language expected in source text and sent to API OCR/translation prompts.
    #[specta(type = String)]
    pub source_language: Language,
    #[specta(type = String)]
    pub target_language: Language,
    pub page: TranslationProfile,
    pub chapter: TranslationProfile,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct TranslationProfileFile {
    model: koharu_translator::ModelSelection,
    generation: GenerationConfig,
    instructions: Option<String>,
    unit_policy: Option<TranslationUnitPolicy>,
}

impl TranslationProfileFile {
    fn into_profile(self, default_policy: TranslationUnitPolicy) -> TranslationProfile {
        TranslationProfile {
            model: self.model,
            generation: self.generation,
            instructions: self.instructions,
            unit_policy: self.unit_policy.unwrap_or(default_policy),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
struct TranslationConfigFile {
    #[serde(default = "default_source_language")]
    source_language: Language,
    #[serde(default = "default_target_language")]
    target_language: Language,
    page: Option<TranslationProfileFile>,
    chapter: Option<TranslationProfileFile>,
}

impl Default for TranslationConfigFile {
    fn default() -> Self {
        Self {
            source_language: default_source_language(),
            target_language: default_target_language(),
            page: None,
            chapter: None,
        }
    }
}

fn default_source_language() -> Language {
    Language::Japanese
}

fn default_target_language() -> Language {
    Language::English
}

impl<'de> Deserialize<'de> for TranslationConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let file = TranslationConfigFile::deserialize(deserializer)?;
        Ok(Self {
            source_language: file.source_language,
            target_language: file.target_language,
            page: file
                .page
                .unwrap_or_default()
                .into_profile(TranslationUnitPolicy::PageOnly),
            chapter: file
                .chapter
                .unwrap_or_default()
                .into_profile(TranslationUnitPolicy::AdaptiveV1),
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
pub struct TranslationProfile {
    pub model: koharu_translator::ModelSelection,
    pub generation: GenerationConfig,
    pub instructions: Option<String>,
    #[serde(default)]
    pub unit_policy: TranslationUnitPolicy,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum TranslationUnitPolicy {
    #[default]
    PageOnly,
    AdaptiveV1,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            source_language: Language::Japanese,
            target_language: Language::English,
            page: TranslationProfile {
                unit_policy: TranslationUnitPolicy::PageOnly,
                ..TranslationProfile::default()
            },
            chapter: TranslationProfile {
                unit_policy: TranslationUnitPolicy::AdaptiveV1,
                ..TranslationProfile::default()
            },
        }
    }
}

impl PipelineConfig {
    pub fn load() -> anyhow::Result<koharu_config::Config<Self>> {
        koharu_config::load("pipeline")
    }

    pub fn detection(&self) -> Result<DetectionModel> {
        match &self.detection {
            DetectionModel::KoharuLayoutRFDetrSeg2XL(config) => {
                Ok(DetectionModel::KoharuLayoutRFDetrSeg2XL(
                    self.processor
                        .koharu_layout_rfdetr_seg_2xl
                        .clone()
                        .unwrap_or_else(|| config.clone()),
                ))
            }
        }
    }

    pub(crate) fn inpainting(&self) -> Result<InpaintingModel> {
        if self.inpainting.method == InpaintingMethod::Local {
            return Ok(configured_local_inpainting_model(
                &self.inpainting.local_model,
            ));
        }

        let config = ImageEditConfig {
            prompt: self.inpainting.api.prompt.clone(),
            apply_mode: self.inpainting.api.apply_mode,
        };
        match self.inpainting.api.provider {
            InpaintingProvider::Fal => match self.inpainting.api.model.as_str() {
                "microsoft/mai-image-2.5/edit" => Ok(InpaintingModel::MaiImage2_5Edit(config)),
                "microsoft/mai-image-2.5-pro/edit" => {
                    Ok(InpaintingModel::MaiImage2_5ProEdit(config))
                }
                model => bail!("unsupported Fal.ai inpainting model {model}"),
            },
            InpaintingProvider::OpenRouter => Ok(InpaintingModel::OpenRouterImageEdit {
                model: self.inpainting.api.model.clone(),
                config,
            }),
        }
    }

    pub(crate) fn manual_inpainting(&self) -> Result<InpaintingModel> {
        Ok(configured_local_inpainting_model(
            &self.inpainting.manual_model,
        ))
    }

    pub fn validate(&self) -> Result<()> {
        let _ = self.detection()?;
        let _ = self.inpainting()?;
        let _ = self.manual_inpainting()?;
        if self.ocr.method == OcrMethod::Api {
            if !matches!(
                self.ocr.api.model.provider,
                Provider::OpenAi | Provider::OpenAiCompatible | Provider::OpenRouter
            ) {
                bail!("API OCR supports only OpenAI, OpenAI-compatible, and OpenRouter providers")
            }
            if !self.ocr.api.model.vision {
                bail!("API OCR requires a vision model")
            }
            if self
                .ocr
                .api
                .model
                .model
                .as_deref()
                .is_none_or(str::is_empty)
            {
                bail!("API OCR requires a selected model")
            }
        }
        if self.inpainting.method == InpaintingMethod::Api {
            if self.inpainting.api.model.trim().is_empty() {
                bail!("API inpainting requires a selected model")
            }
            if self.inpainting.api.prompt.contains('\0') {
                bail!("API inpainting prompt contains NUL")
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize, Type)]
#[serde(default)]
pub struct ProcessorConfig {
    #[serde(rename = "koharu-layout-rfdetr-seg-2xl")]
    pub koharu_layout_rfdetr_seg_2xl: Option<KoharuLayoutRFDetrSeg2XLConfig>,
    #[serde(rename = "flux2-klein")]
    pub flux2_klein: Option<Flux2KleinConfig>,
    #[serde(rename = "rorem-mixed")]
    pub rorem_mixed: Option<RoremMixedConfig>,
    #[serde(rename = "microsoft/mai-image-2.5/edit")]
    pub mai_image_2_5_edit: Option<ImageEditConfig>,
    #[serde(rename = "microsoft/mai-image-2.5-pro/edit")]
    pub mai_image_2_5_pro_edit: Option<ImageEditConfig>,
}

impl Serialize for ProcessorConfig {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let values = [
            self.koharu_layout_rfdetr_seg_2xl.is_some(),
            self.flux2_klein.is_some(),
            self.rorem_mixed.is_some(),
            self.mai_image_2_5_edit.is_some(),
            self.mai_image_2_5_pro_edit.is_some(),
        ];
        let mut map =
            serializer.serialize_map(Some(values.into_iter().filter(|set| *set).count()))?;
        if let Some(config) = &self.koharu_layout_rfdetr_seg_2xl {
            map.serialize_entry("koharu-layout-rfdetr-seg-2xl", config)?;
        }
        if let Some(config) = &self.flux2_klein {
            map.serialize_entry("flux2-klein", config)?;
        }
        if let Some(config) = &self.rorem_mixed {
            map.serialize_entry("rorem-mixed", config)?;
        }
        if let Some(config) = &self.mai_image_2_5_edit {
            map.serialize_entry("microsoft/mai-image-2.5/edit", config)?;
        }
        if let Some(config) = &self.mai_image_2_5_pro_edit {
            map.serialize_entry("microsoft/mai-image-2.5-pro/edit", config)?;
        }
        map.end()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "model")]
pub enum DetectionModel {
    #[serde(rename = "koharu-layout-rfdetr-seg-2xl")]
    KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "model")]
pub enum OcrModel {
    #[serde(rename = "paddleocr-vl-1.6")]
    PaddleOcrVl1_6,
    #[serde(rename = "manga-ocr")]
    MangaOcr,
    #[serde(rename = "baberu-ocr")]
    BaberuOcr,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "kebab-case")]
pub enum OcrMethod {
    #[default]
    Local,
    Api,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct OcrConfig {
    pub method: OcrMethod,
    pub local_model: OcrModel,
    pub api: ApiOcrConfig,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            method: OcrMethod::Local,
            local_model: OcrModel::PaddleOcrVl1_6,
            api: ApiOcrConfig::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct ApiOcrConfig {
    pub model: TranslationModelSelection,
    pub generation: GenerationConfig,
    pub instructions: Option<String>,
}

impl Default for ApiOcrConfig {
    fn default() -> Self {
        Self {
            model: TranslationModelSelection {
                provider: Provider::OpenRouter,
                model: Some("qwen/qwen3-vl-32b-instruct".to_owned()),
                quantization: None,
                vision: true,
                reasoning: true,
                reasoning_required: false,
            },
            generation: GenerationConfig {
                temperature: Some(0.0),
                max_tokens: Some(1024),
                vision: Some(true),
                reasoning: Some(false),
                ..GenerationConfig::default()
            },
            instructions: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "kebab-case")]
pub enum InpaintingMethod {
    #[default]
    Local,
    Api,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, Type)]
pub enum InpaintingProvider {
    #[default]
    #[serde(rename = "fal")]
    Fal,
    #[serde(rename = "openrouter")]
    OpenRouter,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Type)]
pub struct InpaintingModelChoice {
    pub provider: InpaintingProvider,
    pub model: String,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct InpaintingConfig {
    pub method: InpaintingMethod,
    pub local_model: LocalInpaintingModel,
    pub manual_model: LocalInpaintingModel,
    pub api: ApiInpaintingConfig,
}

fn default_manual_inpainting_model() -> LocalInpaintingModel {
    LocalInpaintingModel::LaMa {}
}

impl Default for InpaintingConfig {
    fn default() -> Self {
        Self {
            method: InpaintingMethod::Local,
            local_model: LocalInpaintingModel::LaMa {},
            manual_model: default_manual_inpainting_model(),
            api: ApiInpaintingConfig::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct ApiInpaintingConfig {
    pub provider: InpaintingProvider,
    pub model: String,
    pub prompt: String,
    pub apply_mode: crate::InpaintingApplyMode,
}

impl Default for ApiInpaintingConfig {
    fn default() -> Self {
        Self {
            provider: InpaintingProvider::Fal,
            model: "microsoft/mai-image-2.5/edit".to_owned(),
            prompt: ImageEditConfig::default().prompt,
            apply_mode: crate::InpaintingApplyMode::FullPage,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "model")]
pub enum LocalInpaintingModel {
    #[serde(rename = "lama")]
    LaMa {},
    #[serde(rename = "aot-inpainting")]
    AotInpainting {},
    #[serde(rename = "flux2-klein")]
    Flux2Klein(Flux2KleinConfig),
    #[serde(rename = "rorem-mixed")]
    RoremMixed(RoremMixedConfig),
}

fn local_inpainting_model_id(model: &LocalInpaintingModel) -> &'static str {
    match model {
        LocalInpaintingModel::LaMa {} => "lama",
        LocalInpaintingModel::AotInpainting {} => "aot-inpainting",
        LocalInpaintingModel::Flux2Klein(_) => "flux2-klein",
        LocalInpaintingModel::RoremMixed(_) => "rorem-mixed",
    }
}

fn remember_local_inpainting_profile(
    processor: &mut ProcessorConfig,
    model: &LocalInpaintingModel,
) {
    match model {
        LocalInpaintingModel::Flux2Klein(config) => {
            processor.flux2_klein = Some(config.clone());
        }
        LocalInpaintingModel::RoremMixed(config) => {
            processor.rorem_mixed = Some(config.clone());
        }
        LocalInpaintingModel::LaMa {} | LocalInpaintingModel::AotInpainting {} => {}
    }
}

fn restore_local_inpainting_model(
    model: &str,
    processor: &ProcessorConfig,
) -> Result<LocalInpaintingModel> {
    match model {
        "lama" => Ok(LocalInpaintingModel::LaMa {}),
        "aot-inpainting" => Ok(LocalInpaintingModel::AotInpainting {}),
        "flux2-klein" => Ok(LocalInpaintingModel::Flux2Klein(
            processor.flux2_klein.clone().unwrap_or_default(),
        )),
        "rorem-mixed" => Ok(LocalInpaintingModel::RoremMixed(
            processor.rorem_mixed.clone().unwrap_or_default(),
        )),
        model => bail!("unsupported local inpainting model {model}"),
    }
}

fn configured_local_inpainting_model(model: &LocalInpaintingModel) -> InpaintingModel {
    match model {
        LocalInpaintingModel::LaMa {} => InpaintingModel::LaMa {},
        LocalInpaintingModel::AotInpainting {} => InpaintingModel::AotInpainting {},
        LocalInpaintingModel::Flux2Klein(config) => InpaintingModel::Flux2Klein(config.clone()),
        LocalInpaintingModel::RoremMixed(config) => InpaintingModel::RoremMixed(config.clone()),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum InpaintingModel {
    LaMa {},
    AotInpainting {},
    Flux2Klein(Flux2KleinConfig),
    RoremMixed(RoremMixedConfig),
    MaiImage2_5Edit(ImageEditConfig),
    MaiImage2_5ProEdit(ImageEditConfig),
    OpenRouterImageEdit {
        model: String,
        config: ImageEditConfig,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_select_one_processor_for_each_phase() {
        let config = PipelineConfig::default();

        assert!(matches!(
            config.detection,
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_)
        ));
        assert_eq!(config.ocr.method, OcrMethod::Local);
        assert_eq!(config.ocr.local_model, OcrModel::PaddleOcrVl1_6);
        assert_eq!(config.translation.source_language, Language::Japanese);
        assert_eq!(config.inpainting.method, InpaintingMethod::Local);
        assert!(matches!(
            config.inpainting.local_model,
            LocalInpaintingModel::LaMa {}
        ));
        assert!(matches!(
            config.inpainting.manual_model,
            LocalInpaintingModel::LaMa {}
        ));
    }

    #[test]
    fn manual_remove_defaults_to_lama_independently_of_automatic_cleanup() {
        let mut config = PipelineConfig::default();
        config.inpainting.method = InpaintingMethod::Api;

        assert!(matches!(
            config.inpainting().unwrap(),
            InpaintingModel::MaiImage2_5Edit(_)
        ));
        assert!(matches!(
            config.manual_inpainting().unwrap(),
            InpaintingModel::LaMa {}
        ));
    }

    #[test]
    fn api_inpainting_defaults_to_removing_all_visible_text() {
        let config = ApiInpaintingConfig::default();

        assert_eq!(config.apply_mode, crate::InpaintingApplyMode::FullPage);
        assert!(config.prompt.contains("all visible text"));
        for removable in ["watermarks", "credits", "signatures", "logos"] {
            assert!(config.prompt.contains(removable), "missing {removable}");
        }
        assert!(config.prompt.contains("stylized onomatopoeia"));
        assert!(config.prompt.contains("Do not leave any text behind"));
        assert!(!config.prompt.contains("creator signatures"));
        let preserve = config.prompt.find("Preserve every speech bubble").unwrap();
        let reconstruct = config.prompt.find("reconstruct the artwork").unwrap();
        assert!(preserve < reconstruct);
    }

    #[test]
    fn replaces_the_previous_cleanup_default_when_loading_preferences() {
        const PREVIOUS_PROMPT: &str = "Remove all visible text from the image, including dialogue, captions, sound-effect lettering, watermarks, credits, signatures, logos, and re-upload notices. Preserve every speech bubble and caption box exactly as drawn: do not remove, redraw, reshape, move, resize, recolor, or change their outlines, tails, fills, or textures. Only after preserving those containers, reconstruct the artwork and background hidden behind the removed text. Do not alter characters, panel borders, or any non-text artwork.";

        let mut config = PipelineConfig::default();
        config.inpainting.api.prompt = PREVIOUS_PROMPT.to_owned();
        config.processor.mai_image_2_5_edit = Some(ImageEditConfig {
            prompt: PREVIOUS_PROMPT.to_owned(),
            ..ImageEditConfig::default()
        });

        let restored =
            toml::from_str::<PipelineConfig>(&toml::to_string(&config).unwrap()).unwrap();
        let expected = ImageEditConfig::default().prompt;
        assert_eq!(restored.inpainting.api.prompt, expected);
        assert_eq!(
            restored
                .processor
                .mai_image_2_5_edit
                .as_ref()
                .unwrap()
                .prompt,
            expected
        );
        assert_eq!(
            restored.inpainting.api.apply_mode,
            crate::InpaintingApplyMode::FullPage
        );
        assert_eq!(
            restored
                .processor
                .mai_image_2_5_edit
                .as_ref()
                .unwrap()
                .apply_mode,
            crate::InpaintingApplyMode::FullPage
        );
    }

    #[test]
    fn parses_phase_keyed_processor_configuration() {
        let config: PipelineConfig = toml::from_str(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"

                [ocr]
                method = "local"

                [ocr.local_model]
                model = "baberu-ocr"

                [ocr.api.model]
                provider = "openrouter"
                model = "qwen/qwen3.8-27b"
                vision = true

                [ocr.api.generation]

                [inpainting]
                method = "local"

                [inpainting.local_model]
                model = "rorem-mixed"

                [processor."rorem-mixed"]
                prompt = "Remove the lettering."
                negative_prompt = "letters, words"
            "#,
        )
        .unwrap();

        assert!(matches!(
            config.detection,
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_)
        ));
        assert_eq!(config.ocr.local_model, OcrModel::BaberuOcr);
        assert!(matches!(
            config.inpainting(),
            Ok(InpaintingModel::RoremMixed(config))
                if config.prompt == "Remove the lettering."
                    && config.negative_prompt == "letters, words"
        ));
    }

    #[test]
    fn missing_slots_use_defaults() {
        let config = toml::from_str::<PipelineConfig>("").unwrap();

        assert_eq!(config, PipelineConfig::default());
    }

    #[test]
    fn ocr_keeps_local_and_api_settings_independently() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [ocr]
                method = "api"

                [ocr.local_model]
                model = "manga-ocr"

                [ocr.api]
                instructions = "Preserve furigana."

                [ocr.api.model]
                provider = "openrouter"
                model = "qwen/qwen3-vl-32b-instruct"
                vision = true

                [ocr.api.generation]
                temperature = 0.1
                max_tokens = 2048
                vision = true
            "#,
        )
        .unwrap();

        assert_eq!(config.ocr.method, OcrMethod::Api);
        assert_eq!(config.ocr.local_model, OcrModel::MangaOcr);
        assert_eq!(
            config.ocr.api.model.model.as_deref(),
            Some("qwen/qwen3-vl-32b-instruct")
        );
        assert_eq!(
            config.ocr.api.instructions.as_deref(),
            Some("Preserve furigana.")
        );
        assert_eq!(config.ocr.api.generation.max_tokens, Some(2048));
        config.validate().unwrap();

        let document = toml::to_string(&config).unwrap();
        assert_eq!(
            toml::from_str::<PipelineConfig>(&document).unwrap().ocr,
            config.ocr
        );
    }

    #[test]
    fn api_ocr_rejects_providers_without_a_vision_ocr_backend() {
        let mut config = PipelineConfig::default();
        config.ocr.method = OcrMethod::Api;
        config.ocr.api.model = TranslationModelSelection {
            provider: Provider::Gemini,
            model: Some("gemini-2.5-pro".to_owned()),
            quantization: None,
            vision: true,
            reasoning: true,
            reasoning_required: false,
        };

        let error = config.validate().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("OpenAI, OpenAI-compatible, and OpenRouter")
        );
    }

    #[test]
    fn ignores_unknown_model_configuration_fields() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"
                legacy_threshold = 0.5

                [ocr]
                method = "local"
                legacy_language = "ja"

                [ocr.local_model]
                model = "paddleocr-vl-1.6"

                [ocr.api.model]
                provider = "openrouter"
                model = "qwen/qwen3.8-27b"
                vision = true

                [ocr.api.generation]

                [inpainting]
                method = "local"
                legacy_resolution = 1024

                [inpainting.local_model]
                model = "lama"
            "#,
        )
        .unwrap();

        assert!(matches!(
            config.detection,
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_)
        ));
        assert_eq!(config.ocr.local_model, OcrModel::PaddleOcrVl1_6);
        assert!(matches!(config.inpainting(), Ok(InpaintingModel::LaMa {})));
    }

    #[test]
    fn parses_detection_and_generative_inpainting_options() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"

                [inpainting]
                method = "local"

                [inpainting.local_model]
                model = "flux2-klein"

                [processor."koharu-layout-rfdetr-seg-2xl"]
                text_threshold = 0.25
                bubble_threshold = 0.45
                panel_threshold = 0.55

                [processor."flux2-klein"]
                prompt = "Reconstruct the illustration without text."
            "#,
        )
        .unwrap();

        assert!(matches!(
            config.detection().unwrap(),
            DetectionModel::KoharuLayoutRFDetrSeg2XL(config)
                if config.text_threshold == Some(0.25)
                    && config.bubble_threshold == Some(0.45)
                    && config.panel_threshold == Some(0.55)
        ));
        assert!(matches!(
            config.inpainting().unwrap(),
            InpaintingModel::Flux2Klein(config)
                if config.prompt == "Reconstruct the illustration without text."
        ));
    }

    #[test]
    fn keeps_profiles_separate_from_active_stage_selection() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"

                [inpainting]
                method = "local"

                [inpainting.local_model]
                model = "flux2-klein"

                [processor."flux2-klein"]
                prompt = "saved prompt"
            "#,
        )
        .unwrap();

        let InpaintingModel::Flux2Klein(config) = config.inpainting().unwrap() else {
            panic!("expected FLUX profile")
        };
        assert_eq!(config.prompt, "saved prompt");
    }

    #[test]
    fn serializes_model_profiles_under_processor() {
        let config = PipelineConfig {
            detection: DetectionModel::KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig {
                text_threshold: Some(0.25),
                ..Default::default()
            }),
            ocr: OcrConfig::default(),
            translation: TranslationConfig::default(),
            inpainting: InpaintingConfig {
                local_model: LocalInpaintingModel::Flux2Klein(Flux2KleinConfig {
                    prompt: "Keep the line art.".to_owned(),
                }),
                ..InpaintingConfig::default()
            },
            processor: ProcessorConfig::default(),
        };
        let document = toml::to_string(&config).unwrap();
        assert!(document.contains("[detection]\nmodel = \"koharu-layout-rfdetr-seg-2xl\""));
        assert!(document.contains("[processor.koharu-layout-rfdetr-seg-2xl]"));
        assert!(document.contains("[processor.flux2-klein]"));
        assert!(document.contains("[translation]"));
        assert!(!document.contains("prompt = \"Keep the line art.\"\n[inpainting]"));

        let restored = toml::from_str::<PipelineConfig>(&document).unwrap();
        assert!(matches!(
            restored.inpainting().unwrap(),
            InpaintingModel::Flux2Klein(config) if config.prompt == "Keep the line art."
        ));
    }

    #[test]
    fn translation_profiles_keep_independent_settings() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [translation]
                source_language = "ja-JP"
                target_language = "ru"

                [translation.page]
                unit_policy = "page_only"

                [translation.page.model]
                provider = "openrouter"
                model = "page-model"

                [translation.page.generation]
                temperature = 0.2

                [translation.chapter]
                unit_policy = "adaptive_v1"

                [translation.chapter.model]
                provider = "openrouter"
                model = "chapter-model"

                [translation.chapter.generation]
                temperature = 0.8

                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"

                [inpainting]
                method = "local"

                [inpainting.local_model]
                model = "lama"
            "#,
        )
        .unwrap();

        assert_eq!(
            config.translation.page.model.model.as_deref(),
            Some("page-model")
        );
        assert_eq!(config.translation.page.generation.temperature, Some(0.2));
        assert_eq!(
            config.translation.chapter.model.model.as_deref(),
            Some("chapter-model")
        );
        assert_eq!(config.translation.chapter.generation.temperature, Some(0.8));
        assert_eq!(
            config.translation.page.unit_policy,
            TranslationUnitPolicy::PageOnly
        );
        assert_eq!(
            config.translation.chapter.unit_policy,
            TranslationUnitPolicy::AdaptiveV1
        );
    }

    #[test]
    fn missing_chapter_policy_defaults_to_adaptive_v1() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [translation]
                source_language = "ja-JP"
                target_language = "ru"

                [translation.chapter.model]
                provider = "openrouter"
                model = "chapter-model"

                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"

                [inpainting]
                method = "local"

                [inpainting.local_model]
                model = "lama"
            "#,
        )
        .unwrap();

        assert_eq!(
            config.translation.chapter.unit_policy,
            TranslationUnitPolicy::AdaptiveV1
        );
    }

    #[test]
    fn fal_models_keep_independent_settings() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [inpainting]
                method = "api"

                [inpainting.local_model]
                model = "lama"

                [inpainting.api]
                provider = "fal"
                model = "microsoft/mai-image-2.5/edit"
                prompt = "Clean the page."
                apply_mode = "mask"

                [processor."microsoft/mai-image-2.5/edit"]
                prompt = "Clean the page."
                apply_mode = "mask"

                [processor."microsoft/mai-image-2.5-pro/edit"]
                prompt = "Rebuild the page."
                apply_mode = "full-page"
            "#,
        )
        .unwrap();

        assert!(matches!(
            config.inpainting().unwrap(),
            InpaintingModel::MaiImage2_5Edit(settings)
                if settings.prompt == "Clean the page."
                    && settings.apply_mode == crate::InpaintingApplyMode::Mask
        ));
        assert_eq!(
            config
                .processor
                .mai_image_2_5_pro_edit
                .as_ref()
                .unwrap()
                .prompt,
            "Rebuild the page."
        );

        let document = toml::to_string(&config).unwrap();
        let restored = toml::from_str::<PipelineConfig>(&document).unwrap();
        let edit = restored.processor.mai_image_2_5_edit.unwrap();
        assert_eq!(edit.prompt, "Clean the page.");
        assert_eq!(edit.apply_mode, crate::InpaintingApplyMode::Mask);
        let pro = restored.processor.mai_image_2_5_pro_edit.unwrap();
        assert_eq!(pro.prompt, "Rebuild the page.");
        assert_eq!(pro.apply_mode, crate::InpaintingApplyMode::FullPage);
    }

    #[test]
    fn api_inpainting_uses_provider_specific_mai_model_ids() {
        let mut config = PipelineConfig::default();
        config.inpainting.method = InpaintingMethod::Api;
        config.inpainting.api.prompt = "Remove every letter.".to_owned();

        config.inpainting.api.provider = InpaintingProvider::Fal;
        config.inpainting.api.model = "microsoft/mai-image-2.5/edit".to_owned();
        assert!(matches!(
            config.inpainting().unwrap(),
            InpaintingModel::MaiImage2_5Edit(_)
        ));
        config.inpainting.api.model = "microsoft/mai-image-2.5-pro/edit".to_owned();
        assert!(matches!(
            config.inpainting().unwrap(),
            InpaintingModel::MaiImage2_5ProEdit(_)
        ));

        config.inpainting.api.provider = InpaintingProvider::OpenRouter;
        config.inpainting.api.model = "microsoft/mai-image-2.5".to_owned();
        assert!(matches!(
            config.inpainting().unwrap(),
            InpaintingModel::OpenRouterImageEdit { model, config }
                if model == "microsoft/mai-image-2.5"
                    && config.prompt == "Remove every letter."
        ));
        config.inpainting.api.model = "microsoft/mai-image-2.5-pro".to_owned();
        assert!(matches!(
            config.inpainting().unwrap(),
            InpaintingModel::OpenRouterImageEdit { model, .. }
                if model == "microsoft/mai-image-2.5-pro"
        ));
    }

    #[test]
    fn inpainting_keeps_local_and_api_selections_independently() {
        let mut config = PipelineConfig::default();
        config.inpainting.method = InpaintingMethod::Api;
        config.inpainting.local_model = LocalInpaintingModel::AotInpainting {};
        config.inpainting.api.provider = InpaintingProvider::OpenRouter;
        config.inpainting.api.model = "microsoft/mai-image-2.5-pro".to_owned();
        config.inpainting.api.prompt = "Rebuild only the lettering regions.".to_owned();
        config.inpainting.api.apply_mode = crate::InpaintingApplyMode::Mask;

        let document = toml::to_string(&config).unwrap();
        let restored = toml::from_str::<PipelineConfig>(&document).unwrap();

        assert_eq!(restored.inpainting.method, InpaintingMethod::Api);
        assert!(matches!(
            restored.inpainting.local_model,
            LocalInpaintingModel::AotInpainting {}
        ));
        assert_eq!(restored.inpainting.api, config.inpainting.api);
        restored.validate().unwrap();
    }
}
