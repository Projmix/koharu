mod caiyun;
mod claude;
mod deepl;
mod deepseek;
mod gemini;
mod google_cloud;
mod grok;
mod lm_studio;
mod minimax;
mod openai;
mod openai_compatible;
mod openrouter;
mod qwen_ocr;

use anyhow::Context;
use futures::{FutureExt, future::BoxFuture, future::join_all};
use reqwest::{Client, RequestBuilder, StatusCode};
use serde::de::DeserializeOwned;

pub use caiyun::CaiyunConfig;
pub use claude::ClaudeConfig;
pub use deepl::DeepLConfig;
pub use deepseek::DeepSeekConfig;
pub use gemini::GeminiConfig;
pub use google_cloud::GoogleCloudConfig;
pub use grok::GrokConfig;
pub use lm_studio::LmStudioConfig;
pub use minimax::MiniMaxConfig;
pub use openai::OpenAiConfig;
pub use openai_compatible::OpenAiCompatibleConfig;
pub use openrouter::OpenRouterConfig;
pub use qwen_ocr::QwenOcrConfig;
pub use qwen_ocr::default_model as default_qwen_ocr_model;

use crate::{
    Error, GenerationConfig, MarkupOutcome, MarkupRequest, Model, ModelSelection, OcrRequest,
    Provider, ProvidersConfig, Result, TranslationRequest,
};

pub(crate) async fn translate(
    client: &Client,
    providers: &ProvidersConfig,
    selection: &ModelSelection,
    generation: &GenerationConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    if request.is_chapter() && !supports_structured_chapter(selection.provider) {
        return Err(anyhow::anyhow!(
            "{} does not support strict chapter translation responses; choose a structured-output provider",
            selection.provider
        )
        .into());
    }
    // A shortening rewrite reads the current translation and its limit from
    // the prompt; machine translation would only translate the source again.
    if request.has_shorten() && !supports_structured_chapter(selection.provider) {
        return Err(anyhow::anyhow!(
            "{} cannot shorten a translation; choose a language-model provider",
            selection.provider
        )
        .into());
    }
    let model = || {
        selection
            .model
            .as_deref()
            .with_context(|| format!("{} requires a selected model", selection.provider))
    };
    match selection.provider {
        Provider::OpenAi => {
            openai::translate(client, &providers.openai, model()?, generation, request).await
        }
        Provider::Gemini => {
            gemini::translate(client, &providers.gemini, model()?, generation, request).await
        }
        Provider::Claude => {
            claude::translate(client, &providers.claude, model()?, generation, request).await
        }
        Provider::Grok => {
            grok::translate(client, &providers.grok, model()?, generation, request).await
        }
        Provider::MiniMax => {
            minimax::translate(client, &providers.minimax, model()?, generation, request).await
        }
        Provider::DeepSeek => {
            deepseek::translate(client, &providers.deepseek, model()?, generation, request).await
        }
        Provider::OpenAiCompatible => {
            openai_compatible::compatible(
                client,
                &providers.openai_compatible,
                model()?,
                generation,
                request,
            )
            .await
        }
        Provider::OpenRouter => {
            openrouter::translate(client, &providers.openrouter, model()?, generation, request)
                .await
        }
        Provider::LmStudio => {
            lm_studio::translate(client, &providers.lm_studio, model()?, generation, request).await
        }
        Provider::DeepL => deepl::translate(client, &providers.deepl, request).await,
        Provider::GoogleCloudTranslation => {
            google_cloud::translate(client, &providers.google_cloud_translation, request).await
        }
        Provider::Caiyun => caiyun::translate(client, &providers.caiyun, request).await,
        Provider::QwenOcr => Err(anyhow::anyhow!("Qwen OCR supports recognition only").into()),
        Provider::Local => unreachable!("local translation has its own backend"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positional_providers_are_not_allowed_for_chapter_requests() {
        let request = TranslationRequest::new(["one"], crate::Language::English)
            .with_page_numbers([1])
            .unwrap();
        assert!(request.is_chapter());
        for provider in [
            Provider::DeepL,
            Provider::GoogleCloudTranslation,
            Provider::Caiyun,
        ] {
            assert!(
                !supports_structured_chapter(provider),
                "{provider} must not be treated as an ID-addressable chapter provider"
            );
        }
    }
}

fn supports_structured_chapter(provider: Provider) -> bool {
    !matches!(
        provider,
        Provider::DeepL | Provider::GoogleCloudTranslation | Provider::Caiyun | Provider::QwenOcr
    )
}

pub(crate) async fn recognize(
    client: &Client,
    providers: &ProvidersConfig,
    provider: Provider,
    model: &str,
    generation: &GenerationConfig,
    request: &OcrRequest,
) -> Result<String> {
    match provider {
        Provider::QwenOcr => {
            qwen_ocr::recognize(client, &providers.qwen_ocr, model, generation, request).await
        }
        Provider::OpenRouter => {
            openrouter::recognize(client, &providers.openrouter, model, generation, request).await
        }
        Provider::OpenAi => {
            openai::recognize(client, &providers.openai, model, generation, request).await
        }
        Provider::OpenAiCompatible => {
            openai_compatible::compatible_ocr(
                client,
                &providers.openai_compatible,
                model,
                generation,
                request,
            )
            .await
        }
        _ => Err(anyhow::anyhow!("{provider} does not support API OCR").into()),
    }
}

pub(crate) async fn mark_up(
    client: &Client,
    providers: &ProvidersConfig,
    provider: Provider,
    model: &str,
    generation: &GenerationConfig,
    request: &MarkupRequest,
) -> Result<MarkupOutcome> {
    match provider {
        Provider::OpenRouter => {
            openrouter::mark_up(client, &providers.openrouter, model, generation, request).await
        }
        Provider::OpenAi => {
            openai::mark_up(client, &providers.openai, model, generation, request).await
        }
        Provider::OpenAiCompatible => {
            openai_compatible::compatible_markup(
                client,
                &providers.openai_compatible,
                model,
                generation,
                request,
            )
            .await
        }
        _ => Err(anyhow::anyhow!("{provider} does not support scene markup").into()),
    }
}

pub(crate) async fn models(client: &Client, providers: &ProvidersConfig) -> Vec<Model> {
    let mut models = qwen_ocr::models();
    let pending: Vec<BoxFuture<'_, Result<Vec<Model>>>> = vec![
        openai::models(client).boxed(),
        gemini::models(client).boxed(),
        claude::models(client).boxed(),
        grok::models(client).boxed(),
        minimax::models(client).boxed(),
        deepseek::models(client).boxed(),
        openai_compatible::models(client, &providers.openai_compatible).boxed(),
        openrouter::models(client).boxed(),
        lm_studio::models(client, &providers.lm_studio).boxed(),
        deepl::models().boxed(),
        google_cloud::models().boxed(),
        caiyun::models().boxed(),
    ];
    for result in join_all(pending).await {
        append_models(&mut models, result);
    }
    models.sort_by(|left, right| {
        left.provider
            .to_string()
            .cmp(&right.provider.to_string())
            .then_with(|| left.model.cmp(&right.model))
    });
    models.dedup_by(|left, right| left.provider == right.provider && left.model == right.model);
    models
}

fn append_models(models: &mut Vec<Model>, result: Result<Vec<Model>>) {
    match result {
        Ok(mut provider_models) => models.append(&mut provider_models),
        Err(error) => tracing::warn!(%error, "failed to list translation models"),
    }
}

pub(super) async fn send_json<T: DeserializeOwned>(
    provider: &'static str,
    request: RequestBuilder,
) -> Result<T> {
    let response = request
        .send()
        .await
        .with_context(|| format!("{provider} request failed"))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .with_context(|| format!("failed to read {provider} response"))?;

    if !status.is_success() {
        let lower = text.to_ascii_lowercase();
        if status == StatusCode::TOO_MANY_REQUESTS
            || lower.contains("insufficient_quota")
            || lower.contains("resource_exhausted")
            || lower.contains("rate limit exceeded")
            || lower.contains("credit balance is too low")
        {
            return Err(Error::QuotaExceeded { provider });
        }
        return Err(Error::Api {
            provider,
            status: status.as_u16(),
            message: text.chars().take(4096).collect(),
        });
    }

    serde_json::from_str(&text)
        .with_context(|| format!("failed to decode {provider} response"))
        .map_err(Into::into)
}
