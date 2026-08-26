//! Translation through local and hosted providers.

mod backend;
mod error;
mod language;
mod local;
mod model;
mod prompt;
mod provider;
mod remote;

use std::sync::Arc;

use anyhow::Context as _;
use koharu_ml::Device;

use error::{Error, Result};
use local::LocalTranslator;

const MAX_ESTIMATED_CHAPTER_INPUT_TOKENS: usize = 100_000;
const MAX_ESTIMATED_CHAPTER_OUTPUT_TOKENS: usize = 65_536;
const CHAPTER_OUTPUT_BASE_TOKENS: usize = 16_384;
const CHAPTER_OUTPUT_TOKENS_PER_SEGMENT: usize = 192;

fn chapter_output_budget(request: &TranslationRequest) -> anyhow::Result<u32> {
    let source_bytes = request.segments.iter().map(String::len).sum::<usize>();
    let estimated_tokens = CHAPTER_OUTPUT_BASE_TOKENS
        .saturating_add(
            request
                .segments
                .len()
                .saturating_mul(CHAPTER_OUTPUT_TOKENS_PER_SEGMENT),
        )
        .saturating_add(source_bytes.div_ceil(2));
    // This is a request budget, not a prediction of the model's actual output.
    // Providers cap one completion at 65,536 tokens; an estimate above that
    // must be clamped instead of rejecting a valid request before HTTP.
    Ok(estimated_tokens.min(MAX_ESTIMATED_CHAPTER_OUTPUT_TOKENS) as u32)
}

pub use backend::{
    OcrRequest, TranslationContext, TranslationRequest, TranslationSegmentMetadata, TranslationUnit,
};
pub use language::Language;
pub use model::{GenerationConfig, Model, ModelSelection, Quantization};
pub(crate) use model::{ModelGeneration, QuantizationDefinition, display_name};
pub use prompt::parse_translation_response;
pub use provider::{Provider, ProviderConfig, ProvidersConfig};

#[derive(Clone)]
pub struct Translator {
    providers: koharu_config::Config<ProvidersConfig>,
    local: Arc<tokio::sync::Mutex<Option<LoadedLocal>>>,
    client: reqwest::Client,
    device: Device,
}

struct LoadedLocal {
    model: Option<String>,
    quantization: Option<String>,
    translator: Arc<LocalTranslator>,
}

impl LoadedLocal {
    fn matches(&self, selection: &ModelSelection) -> bool {
        self.model == selection.model && self.quantization == selection.quantization
    }
}

impl Translator {
    /// Build the exact OpenRouter chat-completions JSON body without adding an
    /// authentication header. The resulting value is safe to save and submit
    /// manually; it contains no credentials.
    pub fn openrouter_request(
        selection: &ModelSelection,
        generation: GenerationConfig,
        mut request: TranslationRequest,
    ) -> anyhow::Result<serde_json::Value> {
        anyhow::ensure!(
            selection.provider == Provider::OpenRouter,
            "chapter export requires the OpenRouter translation provider"
        );
        let model = selection
            .model
            .as_deref()
            .context("OpenRouter requires a selected translation model")?;
        let mut generation = generation.for_model(selection);
        request.remove_image();
        if request.is_chapter() {
            let (_, payload) = prompt::prompts(&request)?;
            let estimated_tokens = payload.len().div_ceil(4);
            anyhow::ensure!(
                estimated_tokens <= MAX_ESTIMATED_CHAPTER_INPUT_TOKENS,
                "chapter translation request is estimated at {estimated_tokens} input tokens, above the conservative limit of {MAX_ESTIMATED_CHAPTER_INPUT_TOKENS}; {}",
                request.chapter_split_hint()
            );
            let output_budget = chapter_output_budget(&request)?;
            generation.max_tokens = Some(
                generation
                    .max_tokens
                    .unwrap_or(output_budget)
                    .min(MAX_ESTIMATED_CHAPTER_OUTPUT_TOKENS as u32),
            );
        }
        remote::openrouter_translation_request_body(model, &generation, &request)
            .map_err(Into::into)
    }

    pub fn from_config(
        device: Device,
        providers: koharu_config::Config<ProvidersConfig>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            providers,
            local: Arc::new(tokio::sync::Mutex::new(None)),
            client: koharu_runtime::http_client()?,
            device,
        })
    }

    #[must_use]
    pub fn model(selection: &ModelSelection) -> &'static str {
        selection.provider.into()
    }

    #[must_use]
    pub fn supports_vision(selection: &ModelSelection, generation: &GenerationConfig) -> bool {
        generation.vision.unwrap_or(false)
            && (selection.provider != Provider::Local || local::supports_vision(selection))
    }

    #[must_use]
    pub fn loaded(&self, selection: &ModelSelection) -> bool {
        if selection.provider != Provider::Local {
            return true;
        }
        self.local
            .try_lock()
            .map(|loaded| {
                loaded
                    .as_ref()
                    .is_some_and(|loaded| loaded.matches(selection))
            })
            .unwrap_or(true)
    }

    pub fn unload(&self) -> bool {
        self.local
            .try_lock()
            .map(|mut loaded| loaded.take().is_some())
            .unwrap_or(false)
    }

    #[tracing::instrument(skip_all)]
    pub async fn load_model(&self, selection: &ModelSelection) -> anyhow::Result<()> {
        if selection.provider == Provider::Local {
            self.local(selection).await?;
        }
        Ok(())
    }

    #[tracing::instrument(
        target = "koharu_metrics",
        name = "model_run",
        skip_all,
        fields(
            stage = "translation",
            provider = %selection.provider,
            model = selection.model.as_deref().unwrap_or("provider_default"),
            target_language = request.target_language.tag(),
            outcome = tracing::field::Empty,
        ),
    )]
    pub async fn translate(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        mut request: TranslationRequest,
    ) -> anyhow::Result<(&'static str, Vec<String>)> {
        let _metric = tracing::info_span!(
            target: "koharu_metrics",
            "translation_request",
            provider = %selection.provider,
            model = selection.model.as_deref().unwrap_or("provider_default"),
            target_language = request.target_language.tag(),
        );
        let provider = selection.provider;
        let provider_id: &'static str = provider.into();
        if request.segments.is_empty() {
            tracing::Span::current().record("outcome", "skipped");
            return Ok((provider_id, request.segments));
        }

        let mut generation = generation.for_model(selection);

        if Self::supports_vision(selection, &generation) {
            request.prepare_image()?;
        } else {
            request.remove_image();
        }

        if request.is_chapter() {
            let (_, payload) = prompt::prompts(&request)?;
            let estimated_tokens = payload.len().div_ceil(4);
            anyhow::ensure!(
                estimated_tokens <= MAX_ESTIMATED_CHAPTER_INPUT_TOKENS,
                "chapter translation request is estimated at {estimated_tokens} input tokens, above the conservative limit of {MAX_ESTIMATED_CHAPTER_INPUT_TOKENS}; {}",
                request.chapter_split_hint()
            );
            let output_budget = chapter_output_budget(&request)?;
            generation.max_tokens = Some(
                generation
                    .max_tokens
                    .unwrap_or(output_budget)
                    .min(MAX_ESTIMATED_CHAPTER_OUTPUT_TOKENS as u32),
            );
        }

        let expected = request.segments.len();
        let translated = if provider == Provider::Local {
            self.local(selection)
                .await?
                .translate(request, generation)
                .await?
        } else {
            let providers = self.providers.read()?.clone();
            remote::translate(&self.client, &providers, selection, &generation, &request).await?
        };
        if translated.len() != expected {
            return Err(Error::SegmentCount {
                provider: provider_id,
                expected,
                actual: translated.len(),
            }
            .into());
        }
        tracing::Span::current().record("outcome", "completed");
        Ok((provider_id, translated))
    }

    #[tracing::instrument(
        target = "koharu_metrics",
        name = "model_run",
        skip_all,
        fields(
            stage = "ocr",
            provider = %selection.provider,
            model = selection.model.as_deref().unwrap_or("provider_default"),
            outcome = tracing::field::Empty,
        ),
    )]
    pub async fn recognize(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        mut request: OcrRequest,
    ) -> anyhow::Result<(&'static str, String)> {
        anyhow::ensure!(
            selection.provider != Provider::Local,
            "API OCR requires a remote provider"
        );
        anyhow::ensure!(
            selection.vision,
            "the selected OCR model does not accept images"
        );
        let model = selection.model.as_deref().ok_or_else(|| {
            anyhow::anyhow!("{} requires a selected OCR model", selection.provider)
        })?;
        request.prepare_image()?;
        let generation = generation.for_model(selection);
        let providers = self.providers.read()?.clone();
        let text = remote::recognize(
            &self.client,
            &providers,
            selection.provider,
            model,
            &generation,
            &request,
        )
        .await?;
        let provider: &'static str = selection.provider.into();
        tracing::Span::current().record("outcome", "completed");
        Ok((provider, text))
    }

    #[tracing::instrument(skip_all)]
    pub async fn models() -> anyhow::Result<Vec<Model>> {
        let providers = ProvidersConfig::load()?;
        let providers = providers.read()?.clone();
        let client = koharu_runtime::http_client()?;
        let mut models = local::models();
        models.extend(remote::models(&client, &providers).await);
        Ok(models)
    }

    async fn local(&self, selection: &ModelSelection) -> Result<Arc<LocalTranslator>> {
        let mut loaded = self.local.lock().await;
        if loaded
            .as_ref()
            .is_none_or(|loaded| !loaded.matches(selection))
        {
            *loaded = Some(LoadedLocal {
                model: selection.model.clone(),
                quantization: selection.quantization.clone(),
                translator: Arc::new(LocalTranslator::load(self.device.clone(), selection).await?),
            });
        }
        Ok(Arc::clone(
            &loaded
                .as_ref()
                .expect("local translator was loaded")
                .translator,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local_selection(model: &str) -> ModelSelection {
        ModelSelection {
            provider: Provider::Local,
            model: Some(model.to_owned()),
            quantization: None,
            vision: true,
            reasoning: true,
            reasoning_required: false,
        }
    }

    #[test]
    fn local_vision_requires_capability_and_generation_setting() {
        assert!(Translator::supports_vision(
            &local_selection("gemma4-e2b-it"),
            &GenerationConfig {
                vision: Some(true),
                ..GenerationConfig::default()
            }
        ));
        assert!(!Translator::supports_vision(
            &local_selection("gemma4-e2b-it"),
            &GenerationConfig {
                vision: Some(false),
                ..GenerationConfig::default()
            }
        ));
        assert!(!Translator::supports_vision(
            &local_selection("lfm2.5-1.2b-instruct"),
            &GenerationConfig {
                vision: Some(true),
                ..GenerationConfig::default()
            }
        ));
    }

    #[test]
    fn chapter_output_budget_scales_with_segment_count_and_text() {
        let request = TranslationRequest::new(["猫"; 50], Language::Russian)
            .with_page_numbers([1; 50])
            .unwrap();

        assert_eq!(chapter_output_budget(&request).unwrap(), 26_059);
    }

    #[test]
    fn chapter_output_budget_clamps_to_one_completion() {
        let request = TranslationRequest::new(["x"; 300], Language::Russian)
            .with_page_numbers([1; 300])
            .unwrap();

        assert_eq!(
            chapter_output_budget(&request).unwrap(),
            MAX_ESTIMATED_CHAPTER_OUTPUT_TOKENS as u32
        );
    }

    #[test]
    fn exported_openrouter_request_matches_network_contract_without_credentials() {
        let request = TranslationRequest::new(["text"; 300], Language::Russian)
            .with_page_numbers([1; 300])
            .unwrap();
        let selection = ModelSelection {
            provider: Provider::OpenRouter,
            model: Some("google/gemini-test".to_owned()),
            quantization: None,
            vision: false,
            reasoning: true,
            reasoning_required: true,
        };

        let body = Translator::openrouter_request(&selection, GenerationConfig::default(), request)
            .unwrap();

        assert_eq!(body["model"], "google/gemini-test");
        assert_eq!(body["max_tokens"], 65_536);
        assert_eq!(body["reasoning"]["enabled"], true);
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["role"], "user");
        assert!(body.get("authorization").is_none());
        assert!(!body.to_string().contains("api_key"));
    }

    #[tokio::test]
    async fn oversized_chapter_is_rejected_before_provider_request() {
        let translator = Translator::from_config(
            Device::cpu(),
            koharu_config::Config::memory(ProvidersConfig::default()),
        )
        .unwrap();
        let request = TranslationRequest::new(["x".repeat(450_000)], Language::English)
            .with_page_numbers([1])
            .unwrap();
        let selection = ModelSelection {
            provider: Provider::OpenRouter,
            model: Some("test/model".to_owned()),
            quantization: None,
            vision: false,
            reasoning: false,
            reasoning_required: false,
        };

        let error = translator
            .translate(&selection, GenerationConfig::default(), request)
            .await
            .unwrap_err()
            .to_string();

        assert!(error.contains("above the conservative limit"), "{error}");
        assert!(error.contains("Try two chapter parts"), "{error}");
    }
}
