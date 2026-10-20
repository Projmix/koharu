// https://openrouter.ai/docs/api/reference/overview
// https://openrouter.ai/docs/api/api-reference/models/get-models
// https://openrouter.ai/docs/guides/best-practices/reasoning-tokens

use std::collections::HashMap;

use anyhow::Context;
use koharu_secrets::ExposeSecret;
use reqwest::Client;
use serde::Deserialize;

use super::openai_compatible::{ChatBackend, ReasoningConfig, ResponseMode};
use crate::{
    GenerationConfig, MarkupOutcome, MarkupRequest, Model, OcrRequest, Provider, Result,
    TranslationRequest,
};

const CHAT_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const MODELS_URL: &str = "https://openrouter.ai/api/v1/models";

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(default)]
pub struct OpenRouterConfig {}

pub(super) async fn translate(
    client: &Client,
    _config: &OpenRouterConfig,
    model: &str,
    generation: &GenerationConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let api_key =
        koharu_secrets::get("openrouter")?.context("openrouter API key is not configured")?;
    let api_key = api_key.expose_secret();
    let reasoning = reasoning(client, api_key, model, generation.reasoning).await?;
    let backend = translation_backend(Some(api_key), model, generation, reasoning);
    super::openai_compatible::translate(client, backend, request).await
}

fn translation_backend<'a>(
    api_key: Option<&'a str>,
    model: &'a str,
    generation: &GenerationConfig,
    reasoning: Option<ReasoningConfig>,
) -> ChatBackend<'a> {
    ChatBackend {
        require_parameters: true,
        reasoning,
        ..ChatBackend::new(
            "openrouter",
            CHAT_URL,
            api_key,
            model,
            generation,
            ResponseMode::JsonSchema,
        )
    }
}

pub(super) async fn recognize(
    client: &Client,
    _config: &OpenRouterConfig,
    model: &str,
    generation: &GenerationConfig,
    request: &OcrRequest,
) -> Result<String> {
    let api_key =
        koharu_secrets::get("openrouter")?.context("openrouter API key is not configured")?;
    // Enforce the OCR schema natively, as translation does; without it some
    // models (Gemini Flash among them) wrap the JSON in a Markdown fence.
    let api_key = api_key.expose_secret();
    let reasoning = reasoning(client, api_key, model, generation.reasoning).await?;
    let backend = translation_backend(Some(api_key), model, generation, reasoning);
    super::openai_compatible::recognize(client, backend, request).await
}

pub(super) async fn mark_up(
    client: &Client,
    _config: &OpenRouterConfig,
    model: &str,
    generation: &GenerationConfig,
    request: &MarkupRequest,
) -> Result<MarkupOutcome> {
    let api_key =
        koharu_secrets::get("openrouter")?.context("openrouter API key is not configured")?;
    let api_key = api_key.expose_secret();
    let reasoning = reasoning(client, api_key, model, generation.reasoning).await?;
    let backend = translation_backend(Some(api_key), model, generation, reasoning);
    super::openai_compatible::mark_up(client, backend, request).await
}

/// OpenRouter rejects a disabled flag for a model whose listing marks
/// reasoning mandatory, so "off" becomes the lowest effort that model lists.
/// The live listing decides rather than the saved selection, whose capability
/// flags can predate the model's mandatory marking.
async fn reasoning(
    client: &Client,
    api_key: &str,
    model: &str,
    enabled: Option<bool>,
) -> Result<Option<ReasoningConfig>> {
    if enabled != Some(false) {
        return Ok(enabled.map(|enabled| ReasoningConfig::Toggle { enabled }));
    }
    static LISTING: tokio::sync::OnceCell<HashMap<String, ReasoningCapabilities>> =
        tokio::sync::OnceCell::const_new();
    let listing = LISTING
        .get_or_try_init(|| async {
            let listed: ModelsResponse =
                super::send_json("openrouter", client.get(MODELS_URL).bearer_auth(api_key)).await?;
            Ok::<_, crate::Error>(
                listed
                    .data
                    .into_iter()
                    .filter_map(|model| Some((model.id, model.reasoning?)))
                    .collect(),
            )
        })
        .await?;
    Ok(disabled_reasoning(listing.get(model)))
}

fn disabled_reasoning(capabilities: Option<&ReasoningCapabilities>) -> Option<ReasoningConfig> {
    match capabilities {
        Some(capabilities) if capabilities.mandatory => ["minimal", "low", "medium", "high"]
            .into_iter()
            .find(|effort| {
                capabilities
                    .supported_efforts
                    .iter()
                    .any(|listed| listed == effort)
            })
            .map(|effort| ReasoningConfig::Effort { effort }),
        _ => Some(ReasoningConfig::Toggle { enabled: false }),
    }
}

pub(super) async fn models(client: &Client) -> Result<Vec<Model>> {
    let Some(api_key) = koharu_secrets::get("openrouter")? else {
        return Ok(Vec::new());
    };
    let discovered: Result<ModelsResponse> = super::send_json(
        "openrouter",
        client.get(MODELS_URL).bearer_auth(api_key.expose_secret()),
    )
    .await;
    Ok(match discovered {
        Ok(discovered) => discovered
            .data
            .into_iter()
            .map(ListedModel::into_model)
            .collect(),
        Err(error) => {
            tracing::warn!(%error, "failed to list OpenRouter models");
            Vec::new()
        }
    })
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ListedModel>,
}

#[derive(Deserialize)]
struct ListedModel {
    id: String,
    name: String,
    architecture: Architecture,
    supported_parameters: Vec<String>,
    reasoning: Option<ReasoningCapabilities>,
}

impl ListedModel {
    fn into_model(self) -> Model {
        Model {
            provider: Provider::OpenRouter,
            name: self.name,
            model: Some(self.id),
            quantizations: Vec::new(),
            vision: self
                .architecture
                .input_modalities
                .iter()
                .any(|modality| modality == "image"),
            reasoning: self.reasoning.is_some()
                || self
                    .supported_parameters
                    .iter()
                    .any(|parameter| parameter == "reasoning"),
            reasoning_required: self
                .reasoning
                .as_ref()
                .is_some_and(|reasoning| reasoning.mandatory),
        }
    }
}

#[derive(Deserialize)]
struct ReasoningCapabilities {
    #[serde(default)]
    mandatory: bool,
    #[serde(default)]
    supported_efforts: Vec<String>,
}

#[derive(Deserialize)]
struct Architecture {
    input_modalities: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translation_requests_strict_structured_output() {
        let request = TranslationRequest::new(["こんにちは"], crate::Language::Russian);
        let backend = translation_backend(None, "test/model", &GenerationConfig::default(), None);
        let body =
            super::super::openai_compatible::translation_request_body(&backend, &request).unwrap();
        assert_eq!(body["provider"]["require_parameters"], true);
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
        assert_eq!(
            body["response_format"]["json_schema"]["schema"],
            crate::prompt::output_schema()
        );
    }

    #[test]
    fn reads_reasoning_capability_from_model_list() {
        let response: ModelsResponse = serde_json::from_value(serde_json::json!({
            "data": [
                {
                    "id": "provider/reasoning-model",
                    "name": "Reasoning Model",
                    "architecture": { "input_modalities": ["text"] },
                    "supported_parameters": ["reasoning"],
                    "reasoning": {
                        "mandatory": true,
                        "supported_efforts": ["high", "medium", "low"]
                    }
                },
                {
                    "id": "provider/chat-model",
                    "name": "Chat Model",
                    "architecture": { "input_modalities": ["text"] },
                    "supported_parameters": []
                }
            ]
        }))
        .unwrap();
        let models = response
            .data
            .into_iter()
            .map(ListedModel::into_model)
            .collect::<Vec<_>>();
        assert!(models[0].reasoning);
        assert!(models[0].reasoning_required);
        assert!(!models[1].reasoning);
        assert!(!models[1].reasoning_required);
    }

    #[test]
    fn disabled_reasoning_uses_lowest_effort_when_mandatory() {
        let mandatory = |efforts: &[&str]| ReasoningCapabilities {
            mandatory: true,
            supported_efforts: efforts.iter().map(|effort| (*effort).to_owned()).collect(),
        };
        assert_eq!(
            disabled_reasoning(Some(&mandatory(&["high", "medium", "low"]))),
            Some(ReasoningConfig::Effort { effort: "low" })
        );
        assert_eq!(
            disabled_reasoning(Some(&mandatory(&["high", "minimal", "low"]))),
            Some(ReasoningConfig::Effort { effort: "minimal" })
        );
        assert_eq!(disabled_reasoning(Some(&mandatory(&[]))), None);
        let optional = ReasoningCapabilities {
            mandatory: false,
            supported_efforts: vec!["none".to_owned()],
        };
        for capabilities in [Some(&optional), None] {
            assert_eq!(
                disabled_reasoning(capabilities),
                Some(ReasoningConfig::Toggle { enabled: false })
            );
        }
    }
}
