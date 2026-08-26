// https://openrouter.ai/docs/api/reference/overview
// https://openrouter.ai/docs/api/api-reference/models/get-models
// https://openrouter.ai/docs/guides/best-practices/reasoning-tokens

use anyhow::Context;
use koharu_secrets::ExposeSecret;
use reqwest::Client;
use serde::Deserialize;

use super::openai_compatible::{ChatBackend, ResponseMode};
use crate::{GenerationConfig, Model, OcrRequest, Provider, Result, TranslationRequest};

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
    let backend = ChatBackend {
        reasoning: generation.reasoning,
        ..ChatBackend::new(
            "openrouter",
            CHAT_URL,
            Some(api_key.expose_secret()),
            model,
            generation,
            ResponseMode::JsonSchema,
        )
    };
    super::openai_compatible::translate(client, backend, request).await
}

pub(super) fn translation_request_body(
    model: &str,
    generation: &GenerationConfig,
    request: &TranslationRequest,
) -> Result<serde_json::Value> {
    let backend = ChatBackend {
        reasoning: generation.reasoning,
        ..ChatBackend::new(
            "openrouter",
            CHAT_URL,
            None,
            model,
            generation,
            ResponseMode::JsonSchema,
        )
    };
    super::openai_compatible::translation_request_body(&backend, request)
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
    let backend = ChatBackend {
        reasoning: generation.reasoning,
        ..ChatBackend::new(
            "openrouter",
            CHAT_URL,
            Some(api_key.expose_secret()),
            model,
            generation,
            ResponseMode::PromptOnly,
        )
    };
    super::openai_compatible::recognize(client, backend, request).await
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
        let source = include_str!("openrouter.rs");
        let translation = source
            .split_once("pub(super) async fn translate")
            .unwrap()
            .1
            .split_once("pub(super) async fn recognize")
            .unwrap()
            .0;

        assert!(
            translation.contains("ResponseMode::JsonSchema"),
            "OpenRouter translation currently relies only on prompt wording, so a model may return markdown or prose that the strict importer cannot parse"
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
}
