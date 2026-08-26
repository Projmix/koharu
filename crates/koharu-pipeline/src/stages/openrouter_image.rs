use std::io::Cursor;

use anyhow::{Context as _, Result, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures::StreamExt as _;
use image::{DynamicImage, ImageFormat, imageops::FilterType};
use koharu_secrets::ExposeSecret as _;
use reqwest::{Client, RequestBuilder, Response, Url};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::StopToken;

const IMAGES_URL: &str = "https://openrouter.ai/api/v1/images";
const IMAGE_MODELS_URL: &str = "https://openrouter.ai/api/v1/images/models";
const MAX_IMAGE_BYTES: usize = 64 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = (MAX_IMAGE_BYTES / 3 + 1) * 4 + 1024 * 1024;
const MAX_ERROR_BYTES: usize = 16 * 1024;
const MAX_ERROR_DETAIL_CHARS: usize = 512;
const FIRST_PINNED_MODEL: &str = "microsoft/mai-image-2.5";
const SECOND_PINNED_MODEL: &str = "microsoft/mai-image-2.5-pro";

/// A model returned by OpenRouter's dedicated image catalog.
///
/// This intentionally contains only the fields the inpainting picker needs;
/// chat-model capabilities must not leak into the image-model selector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OpenRouterImageModelChoice {
    pub model: String,
    pub name: String,
}

#[derive(Clone)]
pub(super) struct OpenRouterImage {
    client: Client,
    images_url: Url,
}

impl OpenRouterImage {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            client: koharu_runtime::http_client()?,
            images_url: Url::parse(IMAGES_URL)?,
        })
    }

    pub(super) async fn edit(
        &self,
        model: &str,
        prompt: &str,
        image: &DynamicImage,
        stop: &StopToken,
    ) -> Result<Option<DynamicImage>> {
        let secret = koharu_secrets::get("openrouter")?
            .context("OpenRouter credential is not configured")?;
        let key = secret.expose_secret().trim();
        ensure!(!key.is_empty(), "OpenRouter credential is empty");
        self.edit_with_key(model, prompt, image, stop, key).await
    }

    async fn edit_with_key(
        &self,
        model: &str,
        prompt: &str,
        image: &DynamicImage,
        stop: &StopToken,
        key: &str,
    ) -> Result<Option<DynamicImage>> {
        if stop.stopped() {
            return Ok(None);
        }

        let mut encoded = Cursor::new(Vec::new());
        image
            .write_to(&mut encoded, ImageFormat::Png)
            .context("failed to encode the OpenRouter input image")?;
        let data_uri = format!(
            "data:image/png;base64,{}",
            STANDARD.encode(encoded.into_inner())
        );
        let response = self
            .send(
                self.client
                    .post(self.images_url.clone())
                    .bearer_auth(key)
                    .json(&request_body(model, prompt, &data_uri)),
                stop,
            )
            .await?;
        let Some(response) = response else {
            return Ok(None);
        };
        let status = response.status();
        if !status.is_success() {
            let detail = self.error_detail(response, stop, key).await?;
            if stop.stopped() {
                return Ok(None);
            }
            anyhow::bail!(
                "OpenRouter image request failed: HTTP {status}{}",
                detail.map_or_else(String::new, |detail| format!(": {detail}"))
            );
        }
        let Some(result) = self.json::<ImageResponse>(response, stop).await? else {
            return Ok(None);
        };
        if stop.stopped() {
            return Ok(None);
        }
        let [output] = result.data.as_slice() else {
            anyhow::bail!(
                "OpenRouter returned {} images, expected exactly one",
                result.data.len()
            );
        };
        if let Some(media_type) = output.media_type.as_deref() {
            ensure_raster_mime(media_type)?;
        }

        // Reject an over-sized payload before allocating its decoded bytes.
        let maximum_encoded = (MAX_IMAGE_BYTES / 3 + 1) * 4 + 4;
        ensure!(
            output.b64_json.len() <= maximum_encoded,
            "OpenRouter image exceeds the 64 MiB limit"
        );
        let bytes = STANDARD
            .decode(&output.b64_json)
            .context("OpenRouter returned invalid base64 image data")?;
        ensure!(
            bytes.len() <= MAX_IMAGE_BYTES,
            "OpenRouter image exceeds the 64 MiB limit"
        );
        if stop.stopped() {
            return Ok(None);
        }
        let decoded =
            image::load_from_memory(&bytes).context("OpenRouter returned an invalid image")?;
        if stop.stopped() {
            return Ok(None);
        }
        Ok(Some(decoded.resize_exact(
            image.width(),
            image.height(),
            FilterType::Lanczos3,
        )))
    }

    async fn send(&self, request: RequestBuilder, stop: &StopToken) -> Result<Option<Response>> {
        if stop.stopped() {
            return Ok(None);
        }
        tokio::select! {
            response = request.send() => {
                let response = response?;
                if stop.stopped() { Ok(None) } else { Ok(Some(response)) }
            }
            () = stop.cancelled() => Ok(None),
        }
    }

    async fn json<T: serde::de::DeserializeOwned>(
        &self,
        response: Response,
        stop: &StopToken,
    ) -> Result<Option<T>> {
        if let Some(length) = response.content_length() {
            ensure!(
                length <= MAX_RESPONSE_BYTES as u64,
                "OpenRouter image response exceeds the 64 MiB image limit"
            );
        }
        let mut bytes = Vec::new();
        let mut stream = response.bytes_stream();
        loop {
            let chunk = tokio::select! {
                chunk = stream.next() => chunk,
                () = stop.cancelled() => return Ok(None),
            };
            let Some(chunk) = chunk else {
                break;
            };
            let chunk = chunk?;
            ensure!(
                bytes.len().saturating_add(chunk.len()) <= MAX_RESPONSE_BYTES,
                "OpenRouter image response exceeds the 64 MiB image limit"
            );
            bytes.extend_from_slice(&chunk);
        }
        if stop.stopped() {
            return Ok(None);
        }
        Ok(Some(serde_json::from_slice(&bytes).context(
            "OpenRouter image response returned invalid JSON",
        )?))
    }

    async fn error_detail(
        &self,
        response: Response,
        stop: &StopToken,
        key: &str,
    ) -> Result<Option<String>> {
        if response
            .content_length()
            .is_some_and(|length| length > MAX_ERROR_BYTES as u64)
        {
            return Ok(Some(
                "error response body exceeded the 16 KiB limit".to_owned(),
            ));
        }

        let mut bytes = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = tokio::select! {
            chunk = stream.next() => chunk,
            () = stop.cancelled() => return Ok(None),
        } {
            let chunk = chunk?;
            let remaining = MAX_ERROR_BYTES.saturating_sub(bytes.len());
            if chunk.len() > remaining {
                bytes.extend_from_slice(&chunk[..remaining]);
                break;
            }
            bytes.extend_from_slice(&chunk);
            if bytes.len() == MAX_ERROR_BYTES {
                break;
            }
        }
        if stop.stopped() {
            return Ok(None);
        }

        let value = serde_json::from_slice::<Value>(&bytes).ok();
        let error = value.as_ref().and_then(|value| value.get("error"));
        let message = error
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .or_else(|| {
                value
                    .as_ref()
                    .and_then(|value| value.get("message"))
                    .and_then(Value::as_str)
            });
        let Some(message) = message else {
            return Ok(Some(
                "OpenRouter returned an error without a message".to_owned(),
            ));
        };
        let code = error
            .and_then(|error| error.get("code"))
            .and_then(Value::as_i64)
            .map(|code| format!(" (code {code})"))
            .unwrap_or_default();
        let mut detail = format!("{}{}", message, code);
        if !key.is_empty() {
            detail = detail.replace(key, "[redacted API key]");
        }
        while let Some(start) = detail.find("data:image/") {
            let end = detail[start..]
                .find(|character: char| character.is_whitespace() || character == '"')
                .map_or(detail.len(), |offset| start + offset);
            detail.replace_range(start..end, "[redacted data URI]");
        }
        let detail = detail.chars().take(MAX_ERROR_DETAIL_CHARS).collect();
        Ok(Some(detail))
    }

    #[cfg(test)]
    fn for_test(client: Client, base: Url) -> Self {
        Self {
            client,
            images_url: base.join("images").unwrap(),
        }
    }
}

/// Discover image-edit-capable OpenRouter models.
pub(super) async fn models(client: &Client) -> Result<Vec<OpenRouterImageModelChoice>> {
    let Some(secret) = koharu_secrets::get("openrouter")? else {
        return Ok(Vec::new());
    };
    let key = secret.expose_secret().trim();
    if key.is_empty() {
        return Ok(Vec::new());
    }
    let endpoint = Url::parse(IMAGE_MODELS_URL)?;
    models_with_key(client, endpoint, key).await
}

async fn models_with_key(
    client: &Client,
    endpoint: Url,
    key: &str,
) -> Result<Vec<OpenRouterImageModelChoice>> {
    let response = client
        .get(endpoint)
        .bearer_auth(key)
        .send()
        .await?
        .error_for_status()
        .context("OpenRouter image model catalog request failed")?;
    let response: ModelsResponse = response
        .json()
        .await
        .context("OpenRouter image model catalog returned invalid JSON")?;
    Ok(filter_and_sort_models(response.data))
}

fn request_body(model: &str, prompt: &str, data_uri: &str) -> Value {
    json!({
        "model": model,
        "prompt": prompt,
        "n": 1,
        "aspect_ratio": "auto",
        "input_references": [{
            "type": "image_url",
            "image_url": { "url": data_uri },
        }],
    })
}

fn ensure_raster_mime(media_type: &str) -> Result<()> {
    let media_type = media_type
        .split(';')
        .next()
        .map(str::trim)
        .unwrap_or_default()
        .to_ascii_lowercase();
    ensure!(
        media_type.starts_with("image/")
            && media_type != "image/svg+xml"
            && media_type != "image/svg",
        "OpenRouter returned a non-raster image MIME type {media_type}"
    );
    Ok(())
}

#[derive(Deserialize)]
struct ImageResponse {
    data: Vec<ImageData>,
}

#[derive(Deserialize)]
struct ImageData {
    b64_json: String,
    media_type: Option<String>,
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ListedImageModel>,
}

#[derive(Deserialize)]
struct ListedImageModel {
    id: String,
    name: String,
    #[serde(default)]
    architecture: Architecture,
    #[serde(default)]
    supported_parameters: Value,
}

#[derive(Default, Deserialize)]
struct Architecture {
    #[serde(default)]
    input_modalities: Vec<String>,
    #[serde(default)]
    output_modalities: Vec<String>,
}

fn filter_and_sort_models(models: Vec<ListedImageModel>) -> Vec<OpenRouterImageModelChoice> {
    let mut models: Vec<_> = models
        .into_iter()
        .filter(|model| {
            has_modality(&model.architecture.input_modalities, "image")
                && has_modality(&model.architecture.output_modalities, "image")
                && supports_input_references(&model.supported_parameters)
                && supports_raster_output(&model.supported_parameters)
        })
        .map(|model| OpenRouterImageModelChoice {
            model: model.id,
            name: model.name,
        })
        .collect();
    models.sort_by(|left, right| {
        pinned_rank(&left.model)
            .cmp(&pinned_rank(&right.model))
            .then_with(|| left.model.cmp(&right.model))
    });
    models
}

fn has_modality(modalities: &[String], expected: &str) -> bool {
    modalities
        .iter()
        .any(|modality| modality.eq_ignore_ascii_case(expected))
}

fn supports_input_references(parameters: &Value) -> bool {
    parameters
        .as_object()
        .is_some_and(|parameters| parameters.contains_key("input_references"))
}

fn supports_raster_output(parameters: &Value) -> bool {
    let Some(output_format) = parameters
        .as_object()
        .and_then(|parameters| parameters.get("output_format"))
    else {
        // Some image endpoints always return a raster and omit the optional
        // output_format capability from their coarse model record.
        return true;
    };
    let Some(values) = output_format
        .as_object()
        .and_then(|descriptor| descriptor.get("values"))
        .or_else(|| output_format.get("values"))
    else {
        return true;
    };
    let Some(values) = values.as_array() else {
        return true;
    };
    values.iter().any(|value| {
        value.as_str().is_some_and(|format| {
            matches!(
                format.to_ascii_lowercase().as_str(),
                "png" | "jpeg" | "jpg" | "webp"
            )
        })
    })
}

fn pinned_rank(model: &str) -> u8 {
    match model {
        FIRST_PINNED_MODEL => 0,
        SECOND_PINNED_MODEL => 1,
        _ => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{Arc, Mutex},
        time::Duration,
    };
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    use tokio::sync::Notify;

    #[test]
    fn request_uses_openrouter_image_edit_contract() {
        let body = request_body("provider/edit", "clean text", "data:image/png;base64,x");
        assert_eq!(body["model"], "provider/edit");
        assert_eq!(body["prompt"], "clean text");
        assert_eq!(body["n"], 1);
        assert_eq!(body["aspect_ratio"], "auto");
        assert!(body.get("stream").is_none());
        assert_eq!(body["input_references"][0]["type"], "image_url");
        assert_eq!(
            body["input_references"][0]["image_url"]["url"],
            "data:image/png;base64,x"
        );
    }

    #[test]
    fn catalog_filters_capabilities_and_pins_mai_models() {
        let model =
            |id: &str, input: &[&str], output: &[&str], parameters: Value| ListedImageModel {
                id: id.to_owned(),
                name: id.to_owned(),
                architecture: Architecture {
                    input_modalities: input.iter().map(|value| (*value).to_owned()).collect(),
                    output_modalities: output.iter().map(|value| (*value).to_owned()).collect(),
                },
                supported_parameters: parameters,
            };
        let accepted_parameters = json!({"input_references": {"type": "array"}});
        let accepted_with_formats = json!({
            "input_references": {},
            "output_format": {"type": "enum", "values": ["jpeg", "webp"]}
        });
        let rejected_format = json!({
            "input_references": {},
            "output_format": {"type": "enum", "values": ["svg"]}
        });
        let result = filter_and_sort_models(vec![
            model(
                "z/model",
                &["text", "image"],
                &["image"],
                accepted_parameters,
            ),
            model(
                SECOND_PINNED_MODEL,
                &["image"],
                &["image"],
                accepted_with_formats.clone(),
            ),
            model(
                FIRST_PINNED_MODEL,
                &["image"],
                &["image"],
                accepted_with_formats,
            ),
            model(
                "text/only",
                &["text"],
                &["image"],
                json!({"input_references": {}}),
            ),
            model("svg/model", &["image"], &["image"], rejected_format),
        ]);
        assert_eq!(
            result
                .iter()
                .map(|model| model.model.as_str())
                .collect::<Vec<_>>(),
            [FIRST_PINNED_MODEL, SECOND_PINNED_MODEL, "z/model"]
        );
    }

    #[test]
    fn rejects_vector_mime_types() {
        assert!(ensure_raster_mime("image/png").is_ok());
        assert!(ensure_raster_mime("image/jpeg; charset=binary").is_ok());
        assert!(ensure_raster_mime("image/svg+xml").is_err());
        assert!(ensure_raster_mime("application/json").is_err());
    }

    #[tokio::test]
    async fn decodes_one_result_and_resizes_it() {
        let server = MockServer::start(MockResponse::Success).await;
        let client = server.client();
        let result = client
            .edit_with_key(
                "provider/edit",
                "clean",
                &DynamicImage::new_rgb8(3, 2),
                &StopToken::default(),
                "test-key",
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!((result.width(), result.height()), (3, 2));
        let request = server.requests.lock().unwrap().clone();
        assert!(request.starts_with("POST /images "));
        assert!(request.contains("\"input_references\""));
        assert!(request.contains("Bearer test-key"));
    }

    #[tokio::test]
    async fn rejects_zero_or_multiple_images() {
        for (mode, expected) in [
            (MockResponse::Empty, "returned 0 images"),
            (MockResponse::Multiple, "returned 2 images"),
        ] {
            let server = MockServer::start(mode).await;
            let error = server
                .client()
                .edit_with_key(
                    "provider/edit",
                    "clean",
                    &DynamicImage::new_rgb8(1, 1),
                    &StopToken::default(),
                    "test-key",
                )
                .await
                .unwrap_err();
            assert!(format!("{error:#}").contains(expected));
        }
    }

    #[tokio::test]
    async fn rejects_bad_base64_invalid_raster_and_oversized_response() {
        for (mode, expected) in [
            (MockResponse::InvalidBase64, "invalid base64 image data"),
            (MockResponse::InvalidImage, "returned an invalid image"),
            (
                MockResponse::Oversized,
                "image response exceeds the 64 MiB image limit",
            ),
        ] {
            let server = MockServer::start(mode).await;
            let error = server
                .client()
                .edit_with_key(
                    "provider/edit",
                    "clean",
                    &DynamicImage::new_rgb8(1, 1),
                    &StopToken::default(),
                    "test-key",
                )
                .await
                .unwrap_err();
            assert!(format!("{error:#}").contains(expected));
        }
    }

    #[tokio::test]
    async fn preserves_openrouter_http_error_details() {
        let server = MockServer::start(MockResponse::BadRequest).await;
        let error = server
            .client()
            .edit_with_key(
                "microsoft/mai-image-2.5",
                "clean",
                &DynamicImage::new_rgb8(1, 1),
                &StopToken::default(),
                "test-key",
            )
            .await
            .unwrap_err();

        let message = format!("{error:#}");
        assert!(message.contains("HTTP 400 Bad Request"), "{message}");
        assert!(message.contains("input_references"), "{message}");
        assert!(
            message.contains("model does not support this request"),
            "{message}"
        );
    }

    #[tokio::test]
    async fn stop_interrupts_request_and_response_body_waits() {
        for mode in [
            MockResponse::HangBeforeHeaders,
            MockResponse::HangDuringBody,
        ] {
            let server = MockServer::start(mode).await;
            let stop = StopToken::default();
            let task = tokio::spawn({
                let client = server.client();
                let stop = stop.clone();
                async move {
                    client
                        .edit_with_key(
                            "provider/edit",
                            "clean",
                            &DynamicImage::new_rgb8(1, 1),
                            &stop,
                            "test-key",
                        )
                        .await
                }
            });

            let ready = match mode {
                MockResponse::HangBeforeHeaders => &server.request_received,
                MockResponse::HangDuringBody => &server.body_started,
                _ => unreachable!(),
            };
            tokio::time::timeout(Duration::from_secs(1), ready.notified())
                .await
                .expect("mock server did not receive the request");
            stop.stop();
            let result = tokio::time::timeout(Duration::from_secs(1), task)
                .await
                .expect("OpenRouter request was not cancelled")
                .unwrap()
                .unwrap();
            assert!(result.is_none());
            server.release.notify_one();
        }
    }

    #[derive(Clone, Copy)]
    enum MockResponse {
        Success,
        BadRequest,
        Empty,
        Multiple,
        InvalidBase64,
        InvalidImage,
        Oversized,
        HangBeforeHeaders,
        HangDuringBody,
    }

    struct MockServer {
        base: Url,
        requests: Arc<Mutex<String>>,
        request_received: Arc<Notify>,
        body_started: Arc<Notify>,
        release: Arc<Notify>,
    }

    impl MockServer {
        async fn start(mode: MockResponse) -> Self {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
            let requests = Arc::new(Mutex::new(String::new()));
            let request_received = Arc::new(Notify::new());
            let body_started = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            let task_requests = Arc::clone(&requests);
            let task_request_received = Arc::clone(&request_received);
            let task_body_started = Arc::clone(&body_started);
            let task_release = Arc::clone(&release);
            tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buffer = vec![0_u8; 256 * 1024];
                let read = socket.read(&mut buffer).await.unwrap();
                *task_requests.lock().unwrap() =
                    String::from_utf8_lossy(&buffer[..read]).into_owned();
                task_request_received.notify_one();
                if matches!(mode, MockResponse::HangBeforeHeaders) {
                    task_release.notified().await;
                    return;
                }
                let mut image = Cursor::new(Vec::new());
                DynamicImage::new_rgb8(1, 1)
                    .write_to(&mut image, ImageFormat::Png)
                    .unwrap();
                let image = STANDARD.encode(image.into_inner());
                let entry = || {
                    json!({
                        "b64_json": image,
                        "media_type": "image/png"
                    })
                };
                let body = match mode {
                    MockResponse::BadRequest => {
                        json!({
                            "error": {
                                "message": "model does not support this request: input_references",
                                "code": 400
                            }
                        })
                    }
                    MockResponse::Success => json!({"data": [entry()]}),
                    MockResponse::Empty => json!({"data": []}),
                    MockResponse::Multiple => json!({"data": [entry(), entry()]}),
                    MockResponse::InvalidBase64 => json!({
                        "data": [{"b64_json": "%%%", "media_type": "image/png"}]
                    }),
                    MockResponse::InvalidImage => json!({
                        "data": [{
                            "b64_json": STANDARD.encode(b"not a raster image"),
                            "media_type": "image/png"
                        }]
                    }),
                    MockResponse::Oversized
                    | MockResponse::HangBeforeHeaders
                    | MockResponse::HangDuringBody => {
                        json!({"data": []})
                    }
                }
                .to_string();
                if matches!(mode, MockResponse::HangDuringBody) {
                    let headers = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\n";
                    let _ = socket.write_all(headers.as_bytes()).await;
                    let _ = socket.write_all(b"{").await;
                    task_body_started.notify_one();
                    task_release.notified().await;
                    return;
                }
                let declared_length = if matches!(mode, MockResponse::Oversized) {
                    MAX_RESPONSE_BYTES as u64 + 1
                } else {
                    body.len() as u64
                };
                let status = if matches!(mode, MockResponse::BadRequest) {
                    "400 Bad Request"
                } else {
                    "200 OK"
                };
                let headers = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {declared_length}\r\nConnection: close\r\n\r\n"
                );
                let _ = socket.write_all(headers.as_bytes()).await;
                let _ = socket.write_all(body.as_bytes()).await;
            });
            Self {
                base,
                requests,
                request_received,
                body_started,
                release,
            }
        }

        fn client(&self) -> OpenRouterImage {
            OpenRouterImage::for_test(Client::new(), self.base.clone())
        }
    }
}
