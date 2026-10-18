use std::{future::Future, io::Cursor, time::Duration};

use anyhow::{Context as _, Result, bail, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures::StreamExt as _;
use image::{DynamicImage, ImageFormat, imageops::FilterType};
use koharu_secrets::ExposeSecret as _;
use reqwest::{Response, StatusCode, Url, header::CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use specta::Type;

use crate::StopToken;

#[cfg(test)]
pub(super) use tests::submit_reviewed_image;

const MAX_IMAGE_BYTES: usize = 64 * 1024 * 1024;
const MAX_ERROR_BYTES: usize = 16 * 1024;
const MAX_ERROR_DETAIL_CHARS: usize = 2 * 1024;
const POLL_INTERVAL: Duration = Duration::from_millis(750);
const READ_ATTEMPTS: u32 = 4;
const READ_RETRY_DELAY: Duration = Duration::from_secs(1);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
pub(crate) const DEFAULT_CLEANUP_PROMPT: &str = "Remove all visible text from the image. Do not leave any text behind. This includes dialogue, captions, labels, credits, signatures, logos, watermarks, re-upload notices, and especially stylized onomatopoeia and sound-effect lettering, even when it is rotated, distorted, hand-drawn, outlined, partly occluded, or outside speech bubbles. Preserve every speech bubble and caption box exactly as drawn: do not remove, redraw, reshape, move, resize, recolor, or change their outlines, tails, fills, or textures. Only after preserving those containers, reconstruct the artwork and background hidden behind the removed text. Do not alter characters, panel borders, or any non-text artwork.";
const PREVIOUS_CLEANUP_PROMPTS: [&str; 3] = [
    "Remove all visible text from the image, including dialogue, captions, sound-effect lettering, watermarks, credits, signatures, logos, and re-upload notices. Preserve every speech bubble and caption box exactly as drawn: do not remove, redraw, reshape, move, resize, recolor, or change their outlines, tails, fills, or textures. Only after preserving those containers, reconstruct the artwork and background hidden behind the removed text. Do not alter characters, panel borders, or any non-text artwork.",
    "Remove dialogue and sound-effect lettering from the detected text regions, plus visible watermarks and re-upload notices. Preserve every speech bubble and caption box exactly as drawn: do not remove, redraw, reshape, move, resize, recolor, or change their outlines, tails, fills, or textures. Only after preserving those containers, reconstruct the artwork hidden behind the removed lettering, watermarks, or notices. Do not alter characters, panel borders, creator signatures, logos, or any other original artwork.",
    "Clean only dialogue and sound-effect lettering from the detected text regions and reconstruct the artwork behind it. Do not alter anything outside those regions.",
];

pub(crate) fn normalize_cleanup_prompt(prompt: &mut String) -> bool {
    if PREVIOUS_CLEANUP_PROMPTS.contains(&prompt.as_str()) {
        *prompt = DEFAULT_CLEANUP_PROMPT.to_owned();
        true
    } else {
        false
    }
}
const CONTENT_POLICY_DETAIL: &str = "Fal.ai blocked this image or prompt under its content policy. \
    Koharu did not retry the paid job. Review the page and prompt against Fal.ai's policy before \
    trying again";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "kebab-case")]
pub enum InpaintingApplyMode {
    #[default]
    FullPage,
    Mask,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct ImageEditConfig {
    pub prompt: String,
    pub apply_mode: InpaintingApplyMode,
}

impl Default for ImageEditConfig {
    fn default() -> Self {
        Self {
            prompt: DEFAULT_CLEANUP_PROMPT.to_owned(),
            apply_mode: InpaintingApplyMode::FullPage,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FalModel {
    MaiImage2_5Edit,
    MaiImage2_5ProEdit,
    FluxKontextDev,
    FluxKontextPro,
    FluxKontextMax,
    Flux2Dev,
    Flux2Pro,
    Flux2Flex,
    Flux2Klein4B,
    Flux2Klein9B,
    QwenImageEdit2511,
    Seedream4_5Edit,
}

impl FalModel {
    pub(crate) const ALL: [Self; 12] = [
        Self::MaiImage2_5Edit,
        Self::MaiImage2_5ProEdit,
        Self::FluxKontextDev,
        Self::FluxKontextPro,
        Self::FluxKontextMax,
        Self::Flux2Dev,
        Self::Flux2Pro,
        Self::Flux2Flex,
        Self::Flux2Klein4B,
        Self::Flux2Klein9B,
        Self::QwenImageEdit2511,
        Self::Seedream4_5Edit,
    ];

    pub(crate) fn from_id(id: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|model| model.id() == id)
            .with_context(|| format!("unsupported Fal.ai image editing model {id}"))
    }

    pub(crate) const fn id(self) -> &'static str {
        match self {
            Self::MaiImage2_5Edit => "microsoft/mai-image-2.5/edit",
            Self::MaiImage2_5ProEdit => "microsoft/mai-image-2.5-pro/edit",
            Self::FluxKontextDev => "fal-ai/flux-kontext/dev",
            Self::FluxKontextPro => "fal-ai/flux-pro/kontext",
            Self::FluxKontextMax => "fal-ai/flux-pro/kontext/max",
            Self::Flux2Dev => "fal-ai/flux-2/edit",
            Self::Flux2Pro => "fal-ai/flux-2-pro/edit",
            Self::Flux2Flex => "fal-ai/flux-2-flex/edit",
            Self::Flux2Klein4B => "fal-ai/flux-2/klein/4b/edit",
            Self::Flux2Klein9B => "fal-ai/flux-2/klein/9b/edit",
            Self::QwenImageEdit2511 => "fal-ai/qwen-image-edit-2511",
            Self::Seedream4_5Edit => "fal-ai/bytedance/seedream/v4.5/edit",
        }
    }

    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::MaiImage2_5Edit => "Microsoft MAI Image 2.5 Edit",
            Self::MaiImage2_5ProEdit => "Microsoft MAI Image 2.5 Pro Edit",
            Self::FluxKontextDev => "FLUX.1 Kontext Dev",
            Self::FluxKontextPro => "FLUX.1 Kontext Pro",
            Self::FluxKontextMax => "FLUX.1 Kontext Max",
            Self::Flux2Dev => "FLUX.2 Dev Edit",
            Self::Flux2Pro => "FLUX.2 Pro Edit",
            Self::Flux2Flex => "FLUX.2 Flex Edit",
            Self::Flux2Klein4B => "FLUX.2 Klein 4B Edit",
            Self::Flux2Klein9B => "FLUX.2 Klein 9B Edit",
            Self::QwenImageEdit2511 => "Qwen Image Edit 2511",
            Self::Seedream4_5Edit => "ByteDance Seedream 4.5 Edit",
        }
    }
}

#[derive(Clone)]
pub(super) struct Fal {
    client: reqwest::Client,
    queue_base: Url,
    poll_interval: Duration,
    retry_delay: Duration,
}

impl Fal {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            client: koharu_runtime::http_client()?,
            queue_base: Url::parse("https://queue.fal.run/")?,
            poll_interval: POLL_INTERVAL,
            retry_delay: READ_RETRY_DELAY,
        })
    }

    pub(super) async fn edit(
        &self,
        model: FalModel,
        config: &ImageEditConfig,
        image: &DynamicImage,
        stop: &StopToken,
    ) -> Result<Option<DynamicImage>> {
        let secret = koharu_secrets::get("fal")?.context("Fal.ai credential is not configured")?;
        self.edit_with_key(model, config, image, stop, secret.expose_secret())
            .await
    }

    async fn edit_with_key(
        &self,
        model: FalModel,
        config: &ImageEditConfig,
        image: &DynamicImage,
        stop: &StopToken,
        key: &str,
    ) -> Result<Option<DynamicImage>> {
        let key = normalize_key(key)?;
        let mut encoded = Cursor::new(Vec::new());
        image
            .write_to(&mut encoded, ImageFormat::Png)
            .context("failed to encode the Fal.ai input image")?;
        let data_uri = format!(
            "data:image/png;base64,{}",
            STANDARD.encode(encoded.into_inner())
        );
        let submit_url = self.queue_base.join(model.id())?;
        // A lost submission response can still represent an accepted, billable job.
        // Only reads of the returned job URLs may be retried below.
        let submission = tokio::select! {
            biased;
            () = stop.cancelled() => return Ok(None),
            result = async {
                let response = self.client
                    .post(submit_url)
                    .timeout(REQUEST_TIMEOUT)
                    .header("Authorization", format!("Key {key}"))
                    .json(&request_body(model, config, &data_uri))
                    .send()
                    .await
                    .context("Fal.ai queue submission could not be confirmed. Check Fal.ai Requests before retrying; Koharu did not automatically resubmit the paid job")?;
                let response = authenticated_response(response, "Fal.ai queue submission failed", key).await?;
                response.json::<QueueSubmission>().await
                    .context("Fal.ai queue acknowledgement could not be read. Check Fal.ai Requests before retrying; Koharu did not automatically resubmit the paid job")
            } => result?,
        };
        for url in [
            &submission.status_url,
            &submission.response_url,
            &submission.cancel_url,
        ] {
            self.require_https(url)?;
        }

        loop {
            tokio::select! {
                () = stop.cancelled() => {
                    self.cancel(&submission.cancel_url, key);
                    return Ok(None);
                }
                () = tokio::time::sleep(self.poll_interval) => {}
            }
            let Some(status) = self
                .get(
                    &submission.status_url,
                    stop,
                    &submission.cancel_url,
                    key,
                    true,
                    |response| async move {
                        let response = authenticated_response(response, "Fal.ai status request failed", key).await?;
                        Ok(response.json::<QueueStatus>().await?)
                    },
                )
                .await
                .with_context(|| format!("Fal.ai status could not be retrieved; existing job: {}. The paid job was not resubmitted", submission.status_url))?
            else {
                return Ok(None);
            };
            match status.status.as_str() {
                "IN_QUEUE" | "IN_PROGRESS" => {}
                "COMPLETED" => break,
                status => bail!("Fal.ai returned unexpected queue status {status}"),
            }
        }

        let Some(result) = self
            .get(
                &submission.response_url,
                stop,
                &submission.cancel_url,
                key,
                true,
                |response| async move {
                    let response = authenticated_response(response, "Fal.ai result request failed", key).await?;
                    Ok(response.json::<FalResult>().await?)
                },
            )
            .await
            .with_context(|| format!("Fal.ai result could not be retrieved; existing job: {}. The paid job was not resubmitted", submission.response_url))?
        else {
            return Ok(None);
        };
        ensure!(
            !result.has_nsfw_concepts.iter().any(|flagged| *flagged),
            "Fal.ai blocked the generated image with its safety checker; the result was not applied"
        );
        let [output] = result.images.as_slice() else {
            bail!(
                "Fal.ai returned {} images, expected exactly one",
                result.images.len()
            );
        };
        self.require_https(&output.url)?;
        if let Some(content_type) = output.content_type.as_deref() {
            ensure!(
                content_type.starts_with("image/"),
                "Fal.ai result has invalid content type {content_type}"
            );
        }

        let Some(bytes) = self
            .get(
                &output.url,
                stop,
                &submission.cancel_url,
                key,
                false,
                read_image,
            )
            .await
            .with_context(|| format!("Fal.ai image download failed. The image was already generated; download it from {}. The paid job was not resubmitted", output.url))?
        else {
            return Ok(None);
        };
        let decoded =
            image::load_from_memory(&bytes).context("Fal.ai returned an invalid image")?;
        Ok(Some(decoded.resize_exact(
            image.width(),
            image.height(),
            FilterType::Lanczos3,
        )))
    }

    fn require_https(&self, url: &Url) -> Result<()> {
        let test_http = cfg!(test)
            && self.queue_base.scheme() == "http"
            && url.scheme() == "http"
            && self.queue_base.host() == url.host();
        ensure!(
            url.scheme() == "https" || test_http,
            "Fal.ai returned a non-HTTPS URL"
        );
        Ok(())
    }

    async fn get<T, F, Fut>(
        &self,
        url: &Url,
        stop: &StopToken,
        cancel_url: &Url,
        key: &str,
        authenticated: bool,
        read: F,
    ) -> Result<Option<T>>
    where
        F: Fn(Response) -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        for attempt in 1..=READ_ATTEMPTS {
            let mut request = self.client.get(url.clone()).timeout(REQUEST_TIMEOUT);
            if authenticated {
                request = request.header("Authorization", format!("Key {key}"));
            }
            let result = tokio::select! {
                biased;
                () = stop.cancelled() => {
                    self.cancel(cancel_url, key);
                    return Ok(None);
                }
                result = async {
                    let response = request.send().await?;
                    if retryable_status(response.status()) {
                        return Err(response.error_for_status().unwrap_err().into());
                    }
                    read(response).await
                } => result,
            };
            match result {
                Ok(value) => return Ok(Some(value)),
                Err(error) if retryable_read(&error) && attempt < READ_ATTEMPTS => {
                    tracing::warn!(
                        attempt,
                        "Fal.ai read failed; retrying the same URL without resubmitting the job"
                    );
                }
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("Fal.ai read failed after {attempt} attempt(s)"));
                }
            }
            tokio::select! {
                biased;
                () = stop.cancelled() => {
                    self.cancel(cancel_url, key);
                    return Ok(None);
                }
                () = tokio::time::sleep(self.retry_delay * (1 << (attempt - 1))) => {}
            };
        }
        unreachable!("the final read attempt always returns")
    }

    fn cancel(&self, cancel_url: &Url, key: &str) {
        let client = self.client.clone();
        let cancel_url = cancel_url.clone();
        let authorization = format!("Key {key}");
        drop(tokio::spawn(async move {
            match client
                .put(cancel_url)
                .timeout(Duration::from_secs(10))
                .header("Authorization", authorization)
                .send()
                .await
            {
                Ok(response) if response.status().is_success() => {}
                Ok(response) => {
                    tracing::warn!(status = %response.status(), "Fal.ai cancellation was rejected")
                }
                Err(error) => tracing::warn!(%error, "Fal.ai cancellation failed"),
            }
        }));
    }
}

fn retryable_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::REQUEST_TIMEOUT
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT
    )
}

fn retryable_read(error: &anyhow::Error) -> bool {
    // Reqwest wraps interrupted JSON downloads in a Decode error whose source
    // is a Body error. A well-downloaded but invalid JSON document is not retried.
    error.chain().any(|source| {
        source
            .downcast_ref::<reqwest::Error>()
            .is_some_and(|error| {
                error.is_connect()
                    || error.is_timeout()
                    || error.is_request()
                    || error.is_body()
                    || error.status().is_some_and(retryable_status)
            })
    })
}

async fn read_image(response: Response) -> Result<Vec<u8>> {
    let response = response.error_for_status()?;
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .context("Fal.ai image response has no valid content type")?;
    ensure!(
        content_type.starts_with("image/"),
        "Fal.ai image response has invalid content type {content_type}"
    );
    if let Some(length) = response.content_length() {
        ensure!(
            length <= MAX_IMAGE_BYTES as u64,
            "Fal.ai image exceeds the 64 MiB limit"
        );
    }
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        ensure!(
            bytes.len().saturating_add(chunk.len()) <= MAX_IMAGE_BYTES,
            "Fal.ai image exceeds the 64 MiB limit"
        );
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

async fn authenticated_response(
    response: Response,
    context: &'static str,
    key: &str,
) -> Result<Response> {
    let status = response.status();
    let request_id = response
        .headers()
        .get("x-fal-request-id")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("not provided");
    let request_id = redact_sensitive(request_id, key);
    match status {
        StatusCode::UNAUTHORIZED => bail!(
            "Fal.ai rejected the configured credential (HTTP 401 Unauthorized). Verify that the \
             saved FAL_KEY is active and was copied completely from the Fal.ai dashboard (Fal \
             request ID: {request_id})"
        ),
        StatusCode::FORBIDDEN => bail!(
            "Fal.ai denied access for the configured credential (HTTP 403 Forbidden). Verify that \
             the key has API scope and its account or team can use the selected model (Fal request \
             ID: {request_id})"
        ),
        status @ (StatusCode::BAD_GATEWAY
        | StatusCode::SERVICE_UNAVAILABLE
        | StatusCode::GATEWAY_TIMEOUT) => bail!(
            "{context}: Fal.ai or its model provider is temporarily unavailable ({status}; Fal \
             request ID: {request_id}). Retry this stage later; Koharu did not automatically \
             resubmit the image job because that could create a duplicate paid request"
        ),
        _ if status.is_success() => Ok(response),
        _ => {
            let detail = safe_error_detail(response, key).await;
            bail!("{context}: {status}; Fal request ID: {request_id}; detail: {detail}")
        }
    }
}

async fn safe_error_detail(response: Response, key: &str) -> String {
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    let mut truncated = false;
    while let Some(chunk) = stream.next().await {
        let Ok(chunk) = chunk else {
            return "response body could not be read".to_owned();
        };
        let remaining = MAX_ERROR_BYTES.saturating_sub(body.len());
        if chunk.len() > remaining {
            body.extend_from_slice(&chunk[..remaining]);
            truncated = true;
            break;
        }
        body.extend_from_slice(&chunk);
        if body.len() == MAX_ERROR_BYTES {
            truncated = true;
            break;
        }
    }

    if body
        .windows(b"content_policy_violation".len())
        .any(|window| window == b"content_policy_violation")
    {
        return CONTENT_POLICY_DETAIL.to_owned();
    }

    let detail = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|value| {
            value
                .get("detail")
                .or_else(|| value.get("message"))
                .or_else(|| value.get("error"))
                .map(safe_json_detail)
        })
        .unwrap_or_else(|| {
            if truncated || matches!(body.first(), Some(b'{') | Some(b'[')) {
                "response body could not be safely decoded".to_owned()
            } else {
                String::from_utf8_lossy(&body)
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        });
    let detail = redact_sensitive(&detail, key);
    let mut detail = detail
        .chars()
        .take(MAX_ERROR_DETAIL_CHARS)
        .collect::<String>();
    if detail.is_empty() {
        detail.push_str("response body was empty");
    } else if truncated || detail.chars().count() == MAX_ERROR_DETAIL_CHARS {
        detail.push('…');
    }
    detail
}

fn redact_sensitive(value: &str, key: &str) -> String {
    let mut value = value.replace(key, "[redacted FAL_KEY]");
    while let Some(start) = value
        .as_bytes()
        .windows(b"data:image/".len())
        .position(|window| window.eq_ignore_ascii_case(b"data:image/"))
    {
        let payload = value[start..]
            .find(";base64,")
            .map(|offset| start + offset + ";base64,".len())
            .unwrap_or(start);
        let end = value[payload..]
            .find(|character: char| {
                !character.is_ascii_alphanumeric()
                    && !matches!(character, '+' | '/' | '_' | '-' | '=')
            })
            .map_or(value.len(), |offset| payload + offset);
        value.replace_range(start..end, "[redacted data URI]");
    }

    let mut redacted = String::with_capacity(value.len());
    let mut copied = 0;
    let mut token_start = None;
    for (index, character) in value
        .char_indices()
        .chain(std::iter::once((value.len(), ' ')))
    {
        let token_character = character.is_ascii_alphanumeric()
            || matches!(character, '_' | '-' | '+' | '/' | '=' | '.' | ':');
        match (token_start, token_character) {
            (None, true) => token_start = Some(index),
            (Some(start), false) if index - start >= 64 => {
                redacted.push_str(&value[copied..start]);
                redacted.push_str("[redacted token]");
                copied = index;
                token_start = None;
            }
            (Some(_), false) => token_start = None,
            _ => {}
        }
    }
    redacted.push_str(&value[copied..]);
    redacted
}

fn safe_json_detail(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Array(values) => values
            .iter()
            .map(safe_json_detail)
            .collect::<Vec<_>>()
            .join("; "),
        Value::Object(value)
            if value.get("type").and_then(Value::as_str) == Some("content_policy_violation") =>
        {
            CONTENT_POLICY_DETAIL.to_owned()
        }
        Value::Object(value) => ["loc", "msg", "message", "type", "code"]
            .into_iter()
            .filter_map(|key| {
                value
                    .get(key)
                    .map(|value| format!("{key}={}", safe_json_detail(value)))
            })
            .collect::<Vec<_>>()
            .join(", "),
        value => value.to_string(),
    }
}

fn normalize_key(credential: &str) -> Result<&str> {
    let credential = credential.trim();
    ensure!(!credential.is_empty(), "Fal.ai credential is empty");
    let mut parts = credential.splitn(2, char::is_whitespace);
    let prefix = parts
        .next()
        .expect("a non-empty credential has a first part");
    let key = if prefix.eq_ignore_ascii_case("key") {
        parts.next().unwrap_or_default().trim()
    } else {
        credential
    };
    ensure!(!key.is_empty(), "Fal.ai credential has no FAL_KEY value");
    ensure!(
        !key.chars().any(char::is_whitespace),
        "Fal.ai credential contains whitespace inside the FAL_KEY value"
    );
    Ok(key)
}

fn request_body(model: FalModel, config: &ImageEditConfig, data_uri: &str) -> Value {
    let mut body = json!({
        "prompt": config.prompt,
    });
    match model {
        FalModel::MaiImage2_5ProEdit
        | FalModel::FluxKontextDev
        | FalModel::FluxKontextPro
        | FalModel::FluxKontextMax => body["image_url"] = json!(data_uri),
        _ => body["image_urls"] = json!([data_uri]),
    }
    if !matches!(model, FalModel::Seedream4_5Edit) {
        body["output_format"] = json!("png");
    }
    if !matches!(model, FalModel::Flux2Pro | FalModel::Flux2Flex) {
        body["num_images"] = json!(1);
    }
    // Each edit endpoint owns its sizing defaults; MAI's "auto" is invalid for Kontext.
    match model {
        FalModel::MaiImage2_5Edit | FalModel::MaiImage2_5ProEdit => {
            body["aspect_ratio"] = json!("auto");
        }
        FalModel::FluxKontextDev => body["resolution_mode"] = json!("match_input"),
        FalModel::Seedream4_5Edit => {
            body["max_images"] = json!(1);
            body["image_size"] = json!("auto_2K");
        }
        _ => {}
    }
    body
}

#[derive(Deserialize)]
struct QueueSubmission {
    status_url: Url,
    response_url: Url,
    cancel_url: Url,
}

#[derive(Deserialize)]
struct QueueStatus {
    status: String,
}

#[derive(Deserialize)]
struct FalResult {
    images: Vec<FalImage>,
    #[serde(default)]
    has_nsfw_concepts: Vec<bool>,
}

#[derive(Deserialize)]
struct FalImage {
    url: Url,
    content_type: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    use tokio::sync::Notify;

    #[test]
    fn published_model_endpoints_use_their_exact_image_fields() {
        let config = ImageEditConfig {
            prompt: "  exact restoration prompt  ".to_owned(),
            ..ImageEditConfig::default()
        };
        let normal = request_body(
            FalModel::MaiImage2_5Edit,
            &config,
            "data:image/png;base64,x",
        );
        let pro = request_body(
            FalModel::MaiImage2_5ProEdit,
            &config,
            "data:image/png;base64,x",
        );

        assert_eq!(
            FalModel::MaiImage2_5Edit.id(),
            "microsoft/mai-image-2.5/edit"
        );
        assert_eq!(
            FalModel::MaiImage2_5ProEdit.id(),
            "microsoft/mai-image-2.5-pro/edit"
        );
        assert_eq!(normal["image_urls"][0], "data:image/png;base64,x");
        assert!(normal.get("image_url").is_none());
        assert_eq!(pro["image_url"], "data:image/png;base64,x");
        assert!(pro.get("image_urls").is_none());
        for body in [&normal, &pro] {
            assert_eq!(body["prompt"], "  exact restoration prompt  ");
            assert_eq!(body["num_images"], 1);
            assert_eq!(body["aspect_ratio"], "auto");
            assert_eq!(body["output_format"], "png");
        }
    }

    #[test]
    fn editing_requests_match_each_documented_fal_schema() {
        let config = ImageEditConfig {
            prompt: "Restore this crop.".to_owned(),
            ..ImageEditConfig::default()
        };
        for (endpoints, expected) in [
            (
                vec!["fal-ai/flux-kontext/dev"],
                json!({"prompt": "Restore this crop.", "image_url": "data:image/png;base64,x",
                    "num_images": 1, "output_format": "png", "resolution_mode": "match_input"}),
            ),
            (
                vec!["fal-ai/flux-pro/kontext", "fal-ai/flux-pro/kontext/max"],
                json!({"prompt": "Restore this crop.", "image_url": "data:image/png;base64,x",
                    "num_images": 1, "output_format": "png"}),
            ),
            (
                vec![
                    "fal-ai/flux-2/edit",
                    "fal-ai/flux-2/klein/4b/edit",
                    "fal-ai/flux-2/klein/9b/edit",
                    "fal-ai/qwen-image-edit-2511",
                ],
                json!({"prompt": "Restore this crop.", "image_urls": ["data:image/png;base64,x"],
                    "num_images": 1, "output_format": "png"}),
            ),
            (
                vec!["fal-ai/flux-2-pro/edit", "fal-ai/flux-2-flex/edit"],
                json!({"prompt": "Restore this crop.", "image_urls": ["data:image/png;base64,x"],
                    "output_format": "png"}),
            ),
            (
                vec!["fal-ai/bytedance/seedream/v4.5/edit"],
                json!({"prompt": "Restore this crop.", "image_urls": ["data:image/png;base64,x"],
                    "num_images": 1, "max_images": 1, "image_size": "auto_2K"}),
            ),
        ] {
            for endpoint in endpoints {
                let model = FalModel::from_id(endpoint).unwrap();
                assert_eq!(model.id(), endpoint);
                assert_eq!(
                    request_body(model, &config, "data:image/png;base64,x"),
                    expected,
                    "{endpoint}"
                );
            }
        }
    }

    #[tokio::test]
    async fn flagged_result_does_not_download_or_apply_a_black_placeholder() {
        let server = MockFal::start(MockMode::FlaggedImage).await;
        let result = server
            .client(Duration::from_millis(1))
            .edit_with_key(
                FalModel::Flux2Klein9B,
                &ImageEditConfig::default(),
                &DynamicImage::new_rgb8(3, 2),
                &StopToken::default(),
                "test-key",
            )
            .await;
        assert!(
            result.is_err(),
            "a flagged placeholder must not reach the canvas"
        );
        assert!(
            !server
                .requests
                .lock()
                .unwrap()
                .iter()
                .any(|request| request == "GET /image")
        );
    }

    #[test]
    fn rejects_non_https_urls() {
        let fal = Fal::new().unwrap();
        assert!(
            fal.require_https(&Url::parse("http://example.com/image.png").unwrap())
                .is_err()
        );
        assert!(
            fal.require_https(&Url::parse("https://example.com/image.png").unwrap())
                .is_ok()
        );
    }

    #[tokio::test]
    async fn follows_queue_and_resizes_the_single_png_result() {
        let server = MockFal::start(MockMode::Success).await;
        let fal = server.client(Duration::from_millis(1));
        let result = fal
            .edit_with_key(
                FalModel::MaiImage2_5Edit,
                &ImageEditConfig::default(),
                &DynamicImage::new_rgb8(3, 2),
                &StopToken::default(),
                "test-key",
            )
            .await
            .unwrap()
            .unwrap();

        assert_eq!((result.width(), result.height()), (3, 2));
        let requests = server.requests.lock().unwrap();
        assert_eq!(requests[0], "POST /microsoft/mai-image-2.5/edit");
        assert_eq!(
            requests
                .iter()
                .filter(|request| *request == "GET /status")
                .count(),
            2
        );
        assert!(requests.contains(&"GET /result".to_owned()));
        assert!(requests.contains(&"GET /image".to_owned()));
        assert_eq!(
            server.authorizations.lock().unwrap().as_slice(),
            [
                "Key test-key",
                "Key test-key",
                "Key test-key",
                "Key test-key"
            ]
        );
    }

    #[tokio::test]
    async fn normalizes_the_header_and_reports_auth_failures_without_exposing_the_key() {
        for (mode, expected) in [
            (MockMode::Unauthorized, "HTTP 401 Unauthorized"),
            (MockMode::Forbidden, "HTTP 403 Forbidden"),
        ] {
            let server = MockFal::start(mode).await;
            let key = "private-test-key";
            let error = server
                .client(Duration::from_millis(1))
                .edit_with_key(
                    FalModel::MaiImage2_5Edit,
                    &ImageEditConfig::default(),
                    &DynamicImage::new_rgb8(1, 1),
                    &StopToken::default(),
                    "  kEy \t private-test-key  ",
                )
                .await
                .unwrap_err();
            let message = format!("{error:#}");

            assert!(message.contains(expected));
            assert!(!message.contains(key));
            assert_eq!(
                server.authorizations.lock().unwrap().as_slice(),
                ["Key private-test-key"]
            );
        }
    }

    #[tokio::test]
    async fn transient_read_failures_recover_without_resubmitting_the_paid_job() {
        for (path, failure) in [
            ("/status", ReadFailure::Disconnect),
            ("/result", ReadFailure::Truncated),
            ("/image", ReadFailure::Disconnect),
            ("/image", ReadFailure::Truncated),
            ("/image", ReadFailure::Unavailable),
            ("/image", ReadFailure::Stall),
        ] {
            let server = MockFal::start(MockMode::ReadFailure {
                path,
                failure,
                recover: true,
            })
            .await;
            let result = server
                .client(Duration::from_millis(1))
                .edit_with_key(
                    FalModel::Flux2Flex,
                    &ImageEditConfig::default(),
                    &DynamicImage::new_rgb8(3, 2),
                    &StopToken::default(),
                    "test-key",
                )
                .await
                .unwrap_or_else(|error| panic!("{path} {failure:?}: {error:#}"))
                .unwrap();
            assert_eq!((result.width(), result.height()), (3, 2));
            let requests = server.requests.lock().unwrap();
            assert_eq!(
                requests.iter().filter(|r| r.starts_with("POST ")).count(),
                1
            );
            assert_eq!(
                requests
                    .iter()
                    .filter(|r| *r == &format!("GET {path}"))
                    .count(),
                if path == "/status" { 3 } else { 2 }
            );
        }
    }

    #[tokio::test]
    async fn exhausted_download_retries_report_the_existing_image_url() {
        for failure in [
            ReadFailure::Disconnect,
            ReadFailure::Truncated,
            ReadFailure::Unavailable,
        ] {
            let server = MockFal::start(MockMode::ReadFailure {
                path: "/image",
                failure,
                recover: false,
            })
            .await;
            let error = server
                .client(Duration::from_millis(1))
                .edit_with_key(
                    FalModel::Flux2Flex,
                    &ImageEditConfig::default(),
                    &DynamicImage::new_rgb8(1, 1),
                    &StopToken::default(),
                    "test-key",
                )
                .await
                .unwrap_err();
            let message = format!("{error:#}");
            assert!(message.contains("already generated"), "{message}");
            assert!(
                message.contains(server.base.join("image").unwrap().as_str()),
                "{message}"
            );
            assert!(message.contains("4 attempt(s)"), "{message}");
            let requests = server.requests.lock().unwrap();
            assert_eq!(
                requests.iter().filter(|r| r.starts_with("POST ")).count(),
                1
            );
            assert_eq!(requests.iter().filter(|r| *r == "GET /image").count(), 4);
        }
    }

    #[tokio::test]
    async fn invalid_json_and_rejected_reads_are_not_retried() {
        for failure in [ReadFailure::InvalidJson, ReadFailure::Unauthorized] {
            let server = MockFal::start(MockMode::ReadFailure {
                path: "/result",
                failure,
                recover: false,
            })
            .await;
            let error = server
                .client(Duration::from_millis(1))
                .edit_with_key(
                    FalModel::Flux2Flex,
                    &ImageEditConfig::default(),
                    &DynamicImage::new_rgb8(1, 1),
                    &StopToken::default(),
                    "test-key",
                )
                .await
                .unwrap_err();
            let message = format!("{error:#}");
            assert!(!message.contains("test-key"), "{message}");
            let requests = server.requests.lock().unwrap();
            assert_eq!(
                requests.iter().filter(|r| r.starts_with("POST ")).count(),
                1
            );
            assert_eq!(requests.iter().filter(|r| *r == "GET /result").count(), 1);
            assert!(!requests.iter().any(|r| r == "GET /image"));
        }
    }

    #[tokio::test]
    async fn lost_submission_acknowledgements_are_not_retried() {
        for truncated in [false, true] {
            let server = MockFal::start(MockMode::SubmissionFailure { truncated }).await;
            let error = server
                .client(Duration::from_millis(1))
                .edit_with_key(
                    FalModel::QwenImageEdit2511,
                    &ImageEditConfig::default(),
                    &DynamicImage::new_rgb8(1, 1),
                    &StopToken::default(),
                    "test-key",
                )
                .await
                .unwrap_err();
            let message = format!("{error:#}");
            assert!(message.contains("could not be"), "{message}");
            assert!(
                message.contains("did not automatically resubmit"),
                "{message}"
            );
            assert_eq!(
                server.requests.lock().unwrap().as_slice(),
                ["POST /fal-ai/qwen-image-edit-2511"]
            );
        }
    }

    #[tokio::test]
    async fn stop_interrupts_read_backoff_and_a_stalled_body() {
        for failure in [ReadFailure::Unavailable, ReadFailure::Stall] {
            let server = MockFal::start(MockMode::ReadFailure {
                path: "/image",
                failure,
                recover: false,
            })
            .await;
            let mut fal = server.client(Duration::from_millis(1));
            fal.retry_delay = Duration::from_secs(10);
            let stop = StopToken::default();
            let task = tokio::spawn({
                let stop = stop.clone();
                async move {
                    fal.edit_with_key(
                        FalModel::Flux2Flex,
                        &ImageEditConfig::default(),
                        &DynamicImage::new_rgb8(1, 1),
                        &stop,
                        "test-key",
                    )
                    .await
                }
            });
            tokio::time::timeout(Duration::from_secs(1), server.read_failed.notified())
                .await
                .unwrap();
            stop.stop();
            assert!(
                tokio::time::timeout(Duration::from_secs(1), task)
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap()
                    .is_none()
            );
            tokio::time::timeout(Duration::from_secs(1), server.cancelled.notified())
                .await
                .unwrap();
            let requests = server.requests.lock().unwrap();
            assert_eq!(
                requests.iter().filter(|r| r.starts_with("POST ")).count(),
                1
            );
            assert_eq!(requests.iter().filter(|r| *r == "GET /image").count(), 1);
        }
    }

    #[tokio::test]
    async fn reports_upstream_failure_without_resubmitting_the_paid_job() {
        let server = MockFal::start(MockMode::UpstreamFailure).await;
        let error = server
            .client(Duration::from_millis(1))
            .edit_with_key(
                FalModel::MaiImage2_5Edit,
                &ImageEditConfig::default(),
                &DynamicImage::new_rgb8(1, 1),
                &StopToken::default(),
                "test-key",
            )
            .await
            .unwrap_err();

        let message = format!("{error:#}");
        assert!(message.contains("temporarily unavailable (502 Bad Gateway"));
        assert!(message.contains("Fal request ID: mock-request"));
        assert_eq!(
            server
                .requests
                .lock()
                .unwrap()
                .iter()
                .filter(|request| request.starts_with("POST "))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn reports_content_policy_rejection_without_exposing_or_retrying_the_input() {
        let server = MockFal::start(MockMode::ResultUnprocessable).await;
        let config = ImageEditConfig {
            prompt: "Remove all text and reconstruct the original manga artwork.".to_owned(),
            ..ImageEditConfig::default()
        };
        let error = server
            .client(Duration::from_millis(1))
            .edit_with_key(
                FalModel::MaiImage2_5Edit,
                &config,
                &DynamicImage::new_rgb8(1, 1),
                &StopToken::default(),
                "test-key",
            )
            .await
            .unwrap_err();

        let message = format!("{error:#}");
        assert!(message.contains(CONTENT_POLICY_DETAIL), "{message}");
        assert!(!message.contains("private-image-data"), "{message}");
        assert_eq!(
            server
                .requests
                .lock()
                .unwrap()
                .iter()
                .filter(|request| request.starts_with("POST "))
                .count(),
            1
        );
        let bodies = server.request_bodies.lock().unwrap();
        assert_eq!(bodies[0]["prompt"], config.prompt);
    }

    #[tokio::test]
    async fn redacts_sensitive_values_from_generic_error_details() {
        let server = MockFal::start(MockMode::GenericUnprocessable).await;
        let key = "private-test-key";
        let long_token = "opaque".repeat(20);
        let error = server
            .client(Duration::from_millis(1))
            .edit_with_key(
                FalModel::MaiImage2_5Edit,
                &ImageEditConfig::default(),
                &DynamicImage::new_rgb8(1, 1),
                &StopToken::default(),
                key,
            )
            .await
            .unwrap_err();

        let message = format!("{error:#}");
        assert!(message.contains("422 Unprocessable Entity"), "{message}");
        assert!(message.contains("value_error"), "{message}");
        assert!(!message.contains("data:image"), "{message}");
        assert!(!message.contains(key), "{message}");
        assert!(!message.contains(&long_token), "{message}");
        assert!(message.contains("[redacted data URI]"), "{message}");
        assert!(message.contains("[redacted FAL_KEY]"), "{message}");
        assert!(message.contains("[redacted token]"), "{message}");
        assert_eq!(
            server
                .requests
                .lock()
                .unwrap()
                .iter()
                .filter(|request| request.starts_with("POST "))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn stop_cancels_the_queued_request() {
        let server = MockFal::start(MockMode::Success).await;
        let fal = server.client(Duration::from_millis(1));
        let stop = StopToken::default();
        let task = tokio::spawn({
            let stop = stop.clone();
            async move {
                fal.edit_with_key(
                    FalModel::MaiImage2_5ProEdit,
                    &ImageEditConfig::default(),
                    &DynamicImage::new_rgb8(1, 1),
                    &stop,
                    "test-key",
                )
                .await
            }
        });

        server.status_polled.notified().await;
        stop.stop();
        assert!(task.await.unwrap().unwrap().is_none());
        tokio::time::timeout(Duration::from_secs(1), server.cancelled.notified())
            .await
            .unwrap();
        assert!(
            server
                .requests
                .lock()
                .unwrap()
                .contains(&"PUT /cancel".to_owned())
        );
    }

    #[tokio::test]
    async fn rejects_bad_status_oversized_and_invalid_image_responses() {
        for mode in [
            MockMode::BadStatus,
            MockMode::Oversized,
            MockMode::InvalidImage,
        ] {
            let server = MockFal::start(mode).await;
            let error = server
                .client(Duration::from_millis(1))
                .edit_with_key(
                    FalModel::MaiImage2_5Edit,
                    &ImageEditConfig::default(),
                    &DynamicImage::new_rgb8(1, 1),
                    &StopToken::default(),
                    "test-key",
                )
                .await
                .unwrap_err();
            let message = format!("{error:#}");
            match mode {
                MockMode::BadStatus => assert!(message.contains("unexpected queue status")),
                MockMode::Oversized => assert!(message.contains("64 MiB")),
                MockMode::InvalidImage => assert!(message.contains("invalid image")),
                MockMode::Success
                | MockMode::FlaggedImage
                | MockMode::Unauthorized
                | MockMode::Forbidden
                | MockMode::UpstreamFailure
                | MockMode::ResultUnprocessable
                | MockMode::GenericUnprocessable
                | MockMode::SubmissionFailure { .. }
                | MockMode::ReadFailure { .. } => unreachable!(),
            }
        }
    }

    pub(in crate::stages) async fn submit_reviewed_image(
        config: &ImageEditConfig,
        image: &DynamicImage,
        stop: &StopToken,
    ) -> Value {
        let server = MockFal::start(MockMode::Success).await;
        server
            .client(Duration::from_millis(1))
            .edit_with_key(FalModel::MaiImage2_5Edit, config, image, stop, "test-key")
            .await
            .unwrap()
            .unwrap();
        server.request_bodies.lock().unwrap()[0].clone()
    }

    #[derive(Clone, Copy)]
    enum MockMode {
        SubmissionFailure {
            truncated: bool,
        },
        ReadFailure {
            path: &'static str,
            failure: ReadFailure,
            recover: bool,
        },
        FlaggedImage,
        Success,
        Unauthorized,
        Forbidden,
        UpstreamFailure,
        ResultUnprocessable,
        GenericUnprocessable,
        BadStatus,
        Oversized,
        InvalidImage,
    }

    #[derive(Clone, Copy, Debug)]
    enum ReadFailure {
        Disconnect,
        Truncated,
        Unavailable,
        InvalidJson,
        Unauthorized,
        Stall,
    }

    struct MockFal {
        base: Url,
        requests: Arc<Mutex<Vec<String>>>,
        request_bodies: Arc<Mutex<Vec<Value>>>,
        authorizations: Arc<Mutex<Vec<String>>>,
        status_polled: Arc<Notify>,
        cancelled: Arc<Notify>,
        read_failed: Arc<Notify>,
    }

    impl MockFal {
        async fn start(mode: MockMode) -> Self {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let request_bodies = Arc::new(Mutex::new(Vec::new()));
            let authorizations = Arc::new(Mutex::new(Vec::new()));
            let status_polled = Arc::new(Notify::new());
            let cancelled = Arc::new(Notify::new());
            let read_failed = Arc::new(Notify::new());
            let task_base = base.clone();
            let task_requests = requests.clone();
            let task_request_bodies = request_bodies.clone();
            let task_authorizations = authorizations.clone();
            let task_status_polled = status_polled.clone();
            let task_cancelled = cancelled.clone();
            let task_read_failed = read_failed.clone();
            tokio::spawn(async move {
                let statuses = AtomicUsize::new(0);
                loop {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    let mut buffer = vec![0; 128 * 1024];
                    let read = socket.read(&mut buffer).await.unwrap();
                    let request = String::from_utf8_lossy(&buffer[..read]);
                    let first = request.lines().next().unwrap();
                    let mut parts = first.split_whitespace();
                    let method = parts.next().unwrap();
                    let path = parts.next().unwrap();
                    if method == "POST"
                        && let Some((_, body)) = request.split_once("\r\n\r\n")
                    {
                        task_request_bodies
                            .lock()
                            .unwrap()
                            .push(serde_json::from_str(body).unwrap());
                    }
                    task_requests
                        .lock()
                        .unwrap()
                        .push(format!("{method} {path}"));
                    if let Some(authorization) = request.lines().find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("authorization")
                            .then(|| value.trim().to_owned())
                    }) {
                        task_authorizations.lock().unwrap().push(authorization);
                    }
                    if let MockMode::SubmissionFailure { truncated } = mode
                        && method == "POST"
                    {
                        if truncated {
                            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100\r\nConnection: close\r\n\r\n{").await.unwrap();
                        }
                        socket.shutdown().await.unwrap();
                        continue;
                    }
                    if let MockMode::ReadFailure {
                        path: failed_path,
                        failure,
                        recover,
                    } = mode
                        && method == "GET"
                        && path == failed_path
                    {
                        let attempts = task_requests
                            .lock()
                            .unwrap()
                            .iter()
                            .filter(|request| *request == &format!("GET {path}"))
                            .count();
                        if !recover || attempts == 1 {
                            let response = match failure {
                                ReadFailure::Disconnect => "",
                                ReadFailure::Truncated | ReadFailure::Stall => {
                                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 100\r\nConnection: close\r\n\r\nx"
                                }
                                ReadFailure::Unavailable => {
                                    "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                                }
                                ReadFailure::InvalidJson => {
                                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}"
                                }
                                ReadFailure::Unauthorized => {
                                    "HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                                }
                            };
                            socket.write_all(response.as_bytes()).await.unwrap();
                            if matches!(failure, ReadFailure::Stall) {
                                tokio::spawn(async move {
                                    let _ = socket.read(&mut [0]).await;
                                });
                            } else {
                                socket.shutdown().await.unwrap();
                            }
                            task_read_failed.notify_one();
                            continue;
                        }
                    }
                    let (status, content_type, body, content_length) = match (method, path) {
                        ("POST", _) if matches!(mode, MockMode::Unauthorized) => {
                            let body = br#"{"detail":"Unauthorized"}"#.to_vec();
                            (
                                "401 Unauthorized",
                                "application/json",
                                body.clone(),
                                body.len(),
                            )
                        }
                        ("POST", _) if matches!(mode, MockMode::Forbidden) => {
                            let body = br#"{"detail":"Forbidden"}"#.to_vec();
                            (
                                "403 Forbidden",
                                "application/json",
                                body.clone(),
                                body.len(),
                            )
                        }
                        ("POST", _) if matches!(mode, MockMode::UpstreamFailure) => {
                            let body = br#"{"detail":"upstream unavailable"}"#.to_vec();
                            (
                                "502 Bad Gateway",
                                "application/json",
                                body.clone(),
                                body.len(),
                            )
                        }
                        ("POST", _) => {
                            let body = json!({
                                "status_url": task_base.join("status").unwrap(),
                                "response_url": task_base.join("result").unwrap(),
                                "cancel_url": task_base.join("cancel").unwrap(),
                            })
                            .to_string()
                            .into_bytes();
                            ("200 OK", "application/json", body.clone(), body.len())
                        }
                        ("GET", "/status") => {
                            task_status_polled.notify_one();
                            let status = if matches!(mode, MockMode::BadStatus) {
                                "FAILED"
                            } else if statuses.fetch_add(1, Ordering::SeqCst) == 0 {
                                "IN_QUEUE"
                            } else {
                                "COMPLETED"
                            };
                            let body = json!({ "status": status }).to_string().into_bytes();
                            ("200 OK", "application/json", body.clone(), body.len())
                        }
                        ("GET", "/result") => {
                            if matches!(mode, MockMode::ResultUnprocessable) {
                                let body = json!({
                                    "detail": [{
                                        "type": "content_policy_violation",
                                        "input": {
                                            "image_urls": [format!(
                                                "data:image/png;base64,private-image-data{}",
                                                "x".repeat(MAX_ERROR_BYTES)
                                            )]
                                        }
                                    }]
                                })
                                .to_string()
                                .into_bytes();
                                (
                                    "422 Unprocessable Entity",
                                    "application/json",
                                    body.clone(),
                                    body.len(),
                                )
                            } else if matches!(mode, MockMode::GenericUnprocessable) {
                                let body = json!({
                                    "detail": [{
                                        "type": "value_error",
                                        "msg": format!(
                                            "invalid data:image/png;base64,{} FAL_KEY={} opaque={}",
                                            "a".repeat(96),
                                            "private-test-key",
                                            "opaque".repeat(20)
                                        )
                                    }]
                                })
                                .to_string()
                                .into_bytes();
                                (
                                    "422 Unprocessable Entity",
                                    "application/json",
                                    body.clone(),
                                    body.len(),
                                )
                            } else {
                                let body = json!({
                                    "images": [{
                                        "url": task_base.join("image").unwrap(),
                                        "content_type": "image/png"
                                    }],
                                    "has_nsfw_concepts": [matches!(mode, MockMode::FlaggedImage)]
                                })
                                .to_string()
                                .into_bytes();
                                ("200 OK", "application/json", body.clone(), body.len())
                            }
                        }
                        ("GET", "/image") => {
                            let body = if matches!(mode, MockMode::InvalidImage) {
                                b"not a png".to_vec()
                            } else {
                                let mut bytes = Cursor::new(Vec::new());
                                DynamicImage::new_rgb8(1, 1)
                                    .write_to(&mut bytes, ImageFormat::Png)
                                    .unwrap();
                                bytes.into_inner()
                            };
                            let length = if matches!(mode, MockMode::Oversized) {
                                MAX_IMAGE_BYTES + 1
                            } else {
                                body.len()
                            };
                            ("200 OK", "image/png", body, length)
                        }
                        ("PUT", "/cancel") => {
                            task_cancelled.notify_one();
                            ("202 Accepted", "application/json", Vec::new(), 0)
                        }
                        _ => ("404 Not Found", "text/plain", Vec::new(), 0),
                    };
                    let headers = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {content_length}\r\nX-Fal-Request-Id: mock-request\r\nConnection: close\r\n\r\n"
                    );
                    socket.write_all(headers.as_bytes()).await.unwrap();
                    socket.write_all(&body).await.unwrap();
                    socket.shutdown().await.unwrap();
                }
            });
            Self {
                base,
                requests,
                request_bodies,
                authorizations,
                status_polled,
                cancelled,
                read_failed,
            }
        }

        fn client(&self, poll_interval: Duration) -> Fal {
            Fal {
                client: reqwest::Client::builder()
                    .no_proxy()
                    .read_timeout(Duration::from_millis(200))
                    .build()
                    .unwrap(),
                queue_base: self.base.clone(),
                poll_interval,
                retry_delay: Duration::from_millis(1),
            }
        }
    }
}
