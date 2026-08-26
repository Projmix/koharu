use std::{io::Cursor, time::Duration};

use anyhow::{Context as _, Result, bail, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures::StreamExt as _;
use image::{DynamicImage, ImageFormat, imageops::FilterType};
use koharu_secrets::ExposeSecret as _;
use reqwest::{Response, StatusCode, Url, header::CONTENT_TYPE};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use specta::Type;

use crate::StopToken;

const MAX_IMAGE_BYTES: usize = 64 * 1024 * 1024;
const MAX_ERROR_BYTES: usize = 16 * 1024;
const MAX_ERROR_DETAIL_CHARS: usize = 2 * 1024;
const POLL_INTERVAL: Duration = Duration::from_millis(750);
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

#[derive(Clone, Copy)]
pub(super) enum FalModel {
    MaiImage2_5Edit,
    MaiImage2_5ProEdit,
}

impl FalModel {
    pub(super) const fn id(self) -> &'static str {
        match self {
            Self::MaiImage2_5Edit => "microsoft/mai-image-2.5/edit",
            Self::MaiImage2_5ProEdit => "microsoft/mai-image-2.5-pro/edit",
        }
    }

    const fn pro(self) -> bool {
        matches!(self, Self::MaiImage2_5ProEdit)
    }
}

#[derive(Clone)]
pub(super) struct Fal {
    client: reqwest::Client,
    queue_base: Url,
    poll_interval: Duration,
}

impl Fal {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            client: koharu_runtime::http_client()?,
            queue_base: Url::parse("https://queue.fal.run/")?,
            poll_interval: POLL_INTERVAL,
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
        let response = self
            .send(
                self.client
                    .post(submit_url)
                    .header("Authorization", format!("Key {key}"))
                    .json(&request_body(model, config, &data_uri)),
                stop,
                None,
                key,
            )
            .await?;
        let Some(response) = response else {
            return Ok(None);
        };
        let response =
            authenticated_response(response, "Fal.ai queue submission failed", key).await?;
        let Some(submission) = self
            .json::<QueueSubmission>(response, stop, None, key)
            .await?
        else {
            return Ok(None);
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
            let response = self
                .send(
                    self.client
                        .get(submission.status_url.clone())
                        .header("Authorization", format!("Key {key}")),
                    stop,
                    Some(&submission.cancel_url),
                    key,
                )
                .await?;
            let Some(response) = response else {
                return Ok(None);
            };
            let response =
                authenticated_response(response, "Fal.ai status request failed", key).await?;
            let Some(status) = self
                .json::<QueueStatus>(response, stop, Some(&submission.cancel_url), key)
                .await?
            else {
                return Ok(None);
            };
            match status.status.as_str() {
                "IN_QUEUE" | "IN_PROGRESS" => {}
                "COMPLETED" => break,
                status => bail!("Fal.ai returned unexpected queue status {status}"),
            }
        }

        let response = self
            .send(
                self.client
                    .get(submission.response_url.clone())
                    .header("Authorization", format!("Key {key}")),
                stop,
                Some(&submission.cancel_url),
                key,
            )
            .await?;
        let Some(response) = response else {
            return Ok(None);
        };
        if response.status() == StatusCode::UNPROCESSABLE_ENTITY {
            let request_id = response
                .headers()
                .get("x-fal-request-id")
                .and_then(|value| value.to_str().ok())
                .unwrap_or("not provided")
                .to_owned();
            let detail = safe_error_detail(response, key).await;
            bail!(
                "Fal.ai result request failed: 422 Unprocessable Entity; Fal request ID: {}; detail: {detail}",
                redact_sensitive(&request_id, key)
            );
        }
        let response =
            authenticated_response(response, "Fal.ai result request failed", key).await?;
        let Some(result) = self
            .json::<FalResult>(response, stop, Some(&submission.cancel_url), key)
            .await?
        else {
            return Ok(None);
        };
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

        let response = self
            .send(
                self.client.get(output.url.clone()),
                stop,
                Some(&submission.cancel_url),
                key,
            )
            .await?;
        let Some(response) = response else {
            return Ok(None);
        };
        let response = response
            .error_for_status()
            .context("Fal.ai image download failed")?;
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
        let Some(bytes) = self
            .download(response, stop, &submission.cancel_url, key)
            .await?
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

    async fn send(
        &self,
        request: reqwest::RequestBuilder,
        stop: &StopToken,
        cancel_url: Option<&Url>,
        key: &str,
    ) -> Result<Option<Response>> {
        tokio::select! {
            response = request.send() => Ok(Some(response?)),
            () = stop.cancelled() => {
                if let Some(cancel_url) = cancel_url {
                    self.cancel(cancel_url, key);
                }
                Ok(None)
            }
        }
    }

    async fn json<T: DeserializeOwned>(
        &self,
        response: Response,
        stop: &StopToken,
        cancel_url: Option<&Url>,
        key: &str,
    ) -> Result<Option<T>> {
        tokio::select! {
            result = response.json() => Ok(Some(result?)),
            () = stop.cancelled() => {
                if let Some(cancel_url) = cancel_url {
                    self.cancel(cancel_url, key);
                }
                Ok(None)
            }
        }
    }

    async fn download(
        &self,
        response: Response,
        stop: &StopToken,
        cancel_url: &Url,
        key: &str,
    ) -> Result<Option<Vec<u8>>> {
        let mut bytes = Vec::new();
        let mut stream = response.bytes_stream();
        loop {
            let chunk = tokio::select! {
                chunk = stream.next() => chunk,
                () = stop.cancelled() => {
                    self.cancel(cancel_url, key);
                    return Ok(None);
                }
            };
            let Some(chunk) = chunk else {
                break;
            };
            let chunk = chunk?;
            ensure!(
                bytes.len().saturating_add(chunk.len()) <= MAX_IMAGE_BYTES,
                "Fal.ai image exceeds the 64 MiB limit"
            );
            bytes.extend_from_slice(&chunk);
        }
        Ok(Some(bytes))
    }

    fn cancel(&self, cancel_url: &Url, key: &str) {
        let client = self.client.clone();
        let cancel_url = cancel_url.clone();
        let authorization = format!("Key {key}");
        drop(tokio::spawn(async move {
            match client
                .put(cancel_url)
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
        detail.push_str("…");
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
    let prompt = match config.prompt.trim() {
        DEFAULT_CLEANUP_PROMPT => DEFAULT_CLEANUP_PROMPT.to_owned(),
        prompt => format!("{prompt}\n\nSafety boundary: {DEFAULT_CLEANUP_PROMPT}"),
    };
    let mut body = json!({
        "prompt": prompt,
        "num_images": 1,
        "aspect_ratio": "auto",
        "output_format": "png",
    });
    if model.pro() {
        body["image_url"] = Value::String(data_uri.to_owned());
    } else {
        body["image_urls"] = Value::Array(vec![Value::String(data_uri.to_owned())]);
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
        let config = ImageEditConfig::default();
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
            assert_eq!(body["num_images"], 1);
            assert_eq!(body["aspect_ratio"], "auto");
            assert_eq!(body["output_format"], "png");
        }
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
        let prompt = bodies[0]["prompt"].as_str().unwrap();
        assert!(prompt.contains(&config.prompt), "{prompt}");
        assert!(prompt.contains("speech bubble and caption box"), "{prompt}");
        assert!(prompt.contains("all visible text"), "{prompt}");
        assert!(prompt.contains("watermarks"), "{prompt}");
        assert!(prompt.contains("signatures"), "{prompt}");
        assert!(prompt.contains("Only after preserving"), "{prompt}");
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
                | MockMode::Unauthorized
                | MockMode::Forbidden
                | MockMode::UpstreamFailure
                | MockMode::ResultUnprocessable
                | MockMode::GenericUnprocessable => unreachable!(),
            }
        }
    }

    #[derive(Clone, Copy)]
    enum MockMode {
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

    struct MockFal {
        base: Url,
        requests: Arc<Mutex<Vec<String>>>,
        request_bodies: Arc<Mutex<Vec<Value>>>,
        authorizations: Arc<Mutex<Vec<String>>>,
        status_polled: Arc<Notify>,
        cancelled: Arc<Notify>,
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
            let task_base = base.clone();
            let task_requests = requests.clone();
            let task_request_bodies = request_bodies.clone();
            let task_authorizations = authorizations.clone();
            let task_status_polled = status_polled.clone();
            let task_cancelled = cancelled.clone();
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
                                    }]
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
            }
        }

        fn client(&self, poll_interval: Duration) -> Fal {
            Fal {
                client: reqwest::Client::new(),
                queue_base: self.base.clone(),
                poll_interval,
            }
        }
    }
}
