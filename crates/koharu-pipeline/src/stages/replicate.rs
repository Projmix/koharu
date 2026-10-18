use std::{future::Future, io::Cursor, time::Duration};

use anyhow::{Context as _, Result, bail, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures::StreamExt as _;
use image::{DynamicImage, ImageFormat, imageops::FilterType};
use koharu_secrets::ExposeSecret as _;
use reqwest::{Response, StatusCode, Url, multipart};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, json};

use crate::StopToken;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReplicateModel {
    Flux2Dev,
    Flux2Pro,
    Flux2Max,
    Flux2Flex,
    Flux2Klein4B,
    Flux2Klein9B,
    Flux2Klein4BBase,
    Flux2Klein9BBase,
    FluxKontextDev,
    FluxKontextPro,
    FluxKontextMax,
}

impl ReplicateModel {
    pub(crate) const ALL: [Self; 11] = [
        Self::Flux2Dev,
        Self::Flux2Pro,
        Self::Flux2Max,
        Self::Flux2Flex,
        Self::Flux2Klein4B,
        Self::Flux2Klein9B,
        Self::Flux2Klein4BBase,
        Self::Flux2Klein9BBase,
        Self::FluxKontextDev,
        Self::FluxKontextPro,
        Self::FluxKontextMax,
    ];

    pub(crate) fn id(self) -> &'static str {
        match self {
            Self::Flux2Dev => "black-forest-labs/flux-2-dev",
            Self::Flux2Pro => "black-forest-labs/flux-2-pro",
            Self::Flux2Max => "black-forest-labs/flux-2-max",
            Self::Flux2Flex => "black-forest-labs/flux-2-flex",
            Self::Flux2Klein4B => "black-forest-labs/flux-2-klein-4b",
            Self::Flux2Klein9B => "black-forest-labs/flux-2-klein-9b",
            Self::Flux2Klein4BBase => "black-forest-labs/flux-2-klein-4b-base",
            Self::Flux2Klein9BBase => "black-forest-labs/flux-2-klein-9b-base",
            Self::FluxKontextDev => "black-forest-labs/flux-kontext-dev",
            Self::FluxKontextPro => "black-forest-labs/flux-kontext-pro",
            Self::FluxKontextMax => "black-forest-labs/flux-kontext-max",
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Flux2Dev => "FLUX.2 Dev Edit",
            Self::Flux2Pro => "FLUX.2 Pro Edit",
            Self::Flux2Max => "FLUX.2 Max Edit",
            Self::Flux2Flex => "FLUX.2 Flex Edit",
            Self::Flux2Klein4B => "FLUX.2 Klein 4B Edit",
            Self::Flux2Klein9B => "FLUX.2 Klein 9B Edit",
            Self::Flux2Klein4BBase => "FLUX.2 Klein 4B Base Edit",
            Self::Flux2Klein9BBase => "FLUX.2 Klein 9B Base Edit",
            Self::FluxKontextDev => "FLUX.1 Kontext Dev Edit",
            Self::FluxKontextPro => "FLUX.1 Kontext Pro Edit",
            Self::FluxKontextMax => "FLUX.1 Kontext Max Edit",
        }
    }

    pub(crate) fn from_id(id: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|model| model.id() == id)
            .with_context(|| format!("unsupported Replicate image editing model {id}"))
    }

    fn input(self, prompt: &str, image: &str, width: u32, height: u32) -> Value {
        // Each official /api/schema owns its fields; notably Klein uses `images`
        // and returns an array, while Kontext accepts a single `input_image`.
        let mut input = json!({
            "prompt": prompt, "aspect_ratio": "match_input_image", "output_format": "png",
        });
        match self {
            Self::Flux2Dev => {
                input["input_images"] = json!([image]);
                input["go_fast"] = json!(false);
                input["disable_safety_checker"] = json!(true);
            }
            Self::Flux2Pro | Self::Flux2Max | Self::Flux2Flex => {
                input["input_images"] = json!([image]);
                input["resolution"] = json!("match_input_image");
                input["safety_tolerance"] = json!(5);
                if self == Self::Flux2Flex {
                    input["prompt_upsampling"] = json!(false);
                }
            }
            Self::Flux2Klein4B
            | Self::Flux2Klein9B
            | Self::Flux2Klein4BBase
            | Self::Flux2Klein9BBase => {
                input["images"] = json!([image]);
                input["go_fast"] = json!(false);
                input["disable_safety_checker"] = json!(true);
                let pixels = u64::from(width) * u64::from(height);
                input["output_megapixels"] = json!(match pixels {
                    0..=250_000 => "0.25",
                    250_001..=500_000 => "0.5",
                    500_001..=1_000_000 => "1",
                    1_000_001..=2_000_000 => "2",
                    _ => "4",
                });
            }
            Self::FluxKontextDev | Self::FluxKontextPro | Self::FluxKontextMax => {
                input["input_image"] = json!(image);
                if self == Self::FluxKontextDev {
                    input["go_fast"] = json!(false);
                    input["disable_safety_checker"] = json!(true);
                } else {
                    // Kontext caps tolerance at 2 when an input image is supplied.
                    input["safety_tolerance"] = json!(2);
                    input["prompt_upsampling"] = json!(false);
                }
            }
        }
        input
    }

    fn output_url(self, output: Value) -> Result<Url> {
        match self {
            Self::Flux2Klein4B
            | Self::Flux2Klein9B
            | Self::Flux2Klein4BBase
            | Self::Flux2Klein9BBase => {
                let mut urls: Vec<Url> = serde_json::from_value(output)
                    .context("Replicate Klein returned an invalid image list")?;
                ensure!(
                    urls.len() == 1,
                    "Replicate Klein must return exactly one edited image"
                );
                Ok(urls.remove(0))
            }
            _ => serde_json::from_value(output).context("Replicate returned an invalid image URL"),
        }
    }
}
const INLINE_IMAGE_BYTES: usize = 256 * 1024;
const MAX_IMAGE_BYTES: usize = 64 * 1024 * 1024;
const MAX_ERROR_BYTES: usize = 16 * 1024;
const READ_ATTEMPTS: u32 = 4;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Clone)]
pub(super) struct Replicate {
    client: reqwest::Client,
    api_base: Url,
    poll_interval: Duration,
    retry_delay: Duration,
}

impl Replicate {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            client: koharu_runtime::http_client()?,
            api_base: Url::parse("https://api.replicate.com/v1/")?,
            poll_interval: Duration::from_secs(1),
            retry_delay: Duration::from_secs(1),
        })
    }

    pub(super) async fn edit(
        &self,
        model: ReplicateModel,
        prompt: &str,
        image: &DynamicImage,
        stop: &StopToken,
    ) -> Result<Option<DynamicImage>> {
        let secret = koharu_secrets::get("replicate")?
            .context("Replicate API token is not configured; enter it in Settings > Providers")?;
        self.edit_with_key(model, prompt, image, stop, secret.expose_secret())
            .await
    }

    async fn edit_with_key(
        &self,
        model: ReplicateModel,
        prompt: &str,
        image: &DynamicImage,
        stop: &StopToken,
        key: &str,
    ) -> Result<Option<DynamicImage>> {
        let key = key.trim();
        let key = match key.split_once(char::is_whitespace) {
            Some((scheme, value)) if scheme.eq_ignore_ascii_case("bearer") => value.trim(),
            _ => key,
        };
        ensure!(
            !key.is_empty() && !key.chars().any(char::is_whitespace),
            "Replicate API token is empty or contains whitespace"
        );
        ensure!(
            !prompt.trim().is_empty() && !prompt.contains('\0'),
            "Replicate requires a non-empty prompt without NUL"
        );
        if stop.stopped() {
            return Ok(None);
        }
        let mut png = Cursor::new(Vec::new());
        image
            .write_to(&mut png, ImageFormat::Png)
            .context("failed to encode the Replicate input PNG")?;
        let png = png.into_inner();
        ensure!(
            png.len() <= MAX_IMAGE_BYTES,
            "Replicate input exceeds the 64 MiB limit"
        );
        // Large pages use Replicate's file API; small crops can stay inline.
        let input_image = if png.len() <= INLINE_IMAGE_BYTES {
            format!("data:image/png;base64,{}", STANDARD.encode(png))
        } else {
            let part = multipart::Part::bytes(png)
                .file_name("koharu.png")
                .mime_str("image/png")?;
            let request = self
                .client
                .post(self.api_base.join("files")?)
                .bearer_auth(key)
                .multipart(multipart::Form::new().part("content", part));
            let Some(file) = self
                .once_json::<UploadedFile>(request, stop, key)
                .await
                .context("Replicate input upload failed; no prediction was submitted")?
            else {
                return Ok(None);
            };
            self.require_https(&file.urls.get)?;
            file.urls.get.to_string()
        };
        let request = self
            .client
            .post(
                self.api_base
                    .join(&format!("models/{}/predictions", model.id()))?,
            )
            .bearer_auth(key)
            .json(
                &json!({"input": model.input(prompt, &input_image, image.width(), image.height())}),
            );
        // A missing acknowledgement may still belong to an accepted paid job.
        let Some(mut prediction) = self.once_json::<Prediction>(request, stop, key).await
            .context("Replicate prediction submission could not be confirmed. Check Replicate Predictions before retrying; the paid job was not automatically resubmitted")? else { return Ok(None); };
        ensure!(
            !prediction.id.is_empty()
                && prediction
                    .id
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_'),
            "Replicate returned an invalid prediction ID"
        );
        let id = prediction.id.clone();
        // Construct authenticated endpoints ourselves, never from returned third-party URLs.
        let prediction_url = self.api_base.join(&format!("predictions/{id}"))?;
        let cancel_url = self.api_base.join(&format!("predictions/{id}/cancel"))?;
        loop {
            match prediction.status.as_str() {
                "succeeded" => break,
                "starting" | "processing" => {}
                "failed" => bail!(
                    "Replicate prediction {id} failed: {}",
                    safe_detail(prediction.error.as_ref(), key)
                ),
                "canceled" => bail!("Replicate prediction {id} was canceled"),
                _ => bail!("Replicate prediction {id} returned an unexpected status"),
            }
            if !self.pause(self.poll_interval, stop, &cancel_url, key).await {
                return Ok(None);
            }
            let Some(next) = self.get(&prediction_url, stop, &cancel_url, key, true, |response| async move {
                let response = api_response(response, key).await?;
                Ok(response.json::<Prediction>().await?)
            }).await.with_context(|| format!("Replicate result could not be retrieved; existing job: {prediction_url}. The paid job was not resubmitted"))? else { return Ok(None); };
            ensure!(
                next.id == id,
                "Replicate returned a different prediction ID while reading {id}"
            );
            prediction = next;
        }
        let output = prediction
            .output
            .context("Replicate completed the prediction without an output image")?;
        let output = model.output_url(output)?;
        self.require_https(&output)?;
        let Some(bytes) = self.get(&output, stop, &cancel_url, key, false, read_image).await
            .with_context(|| format!("Replicate image download failed. The image was already generated; download it from {output}. The paid job was not resubmitted"))? else { return Ok(None); };
        let decoded =
            image::load_from_memory(&bytes).context("Replicate returned an invalid image")?;
        Ok(Some(decoded.resize_exact(
            image.width(),
            image.height(),
            FilterType::Lanczos3,
        )))
    }

    async fn once_json<T: DeserializeOwned>(
        &self,
        request: reqwest::RequestBuilder,
        stop: &StopToken,
        key: &str,
    ) -> Result<Option<T>> {
        tokio::select! {
            biased;
            () = stop.cancelled() => Ok(None),
            result = async {
                let response = request.timeout(REQUEST_TIMEOUT).send().await?;
                let response = api_response(response, key).await?;
                Ok(Some(response.json::<T>().await?))
            } => result,
        }
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
                request = request.bearer_auth(key);
            }
            let mut delay = self.retry_delay * (1 << (attempt - 1));
            let result = tokio::select! {
                biased;
                () = stop.cancelled() => { self.cancel(cancel_url, key); return Ok(None); }
                result = async {
                    let response = request.send().await?;
                    if retryable_status(response.status()) {
                        if let Some(seconds) = response.headers().get("retry-after").and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<u64>().ok()) {
                            delay = delay.max(Duration::from_secs(seconds.min(60)));
                        }
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
                        "Replicate read failed; retrying the existing result"
                    );
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("Replicate read failed after {attempt} attempt(s)")
                    });
                }
            }
            if !self.pause(delay, stop, cancel_url, key).await {
                return Ok(None);
            }
        }
        unreachable!("the final read attempt always returns")
    }

    async fn pause(
        &self,
        duration: Duration,
        stop: &StopToken,
        cancel_url: &Url,
        key: &str,
    ) -> bool {
        tokio::select! {
            biased;
            () = stop.cancelled() => { self.cancel(cancel_url, key); false }
            () = tokio::time::sleep(duration) => true,
        }
    }

    fn cancel(&self, url: &Url, key: &str) {
        let request = self
            .client
            .post(url.clone())
            .bearer_auth(key)
            .timeout(Duration::from_secs(10));
        drop(tokio::spawn(async move {
            match request.send().await {
                Ok(response) if response.status().is_success() => {}
                Ok(response) => {
                    tracing::warn!(status = %response.status(), "Replicate cancellation was rejected")
                }
                Err(error) => tracing::warn!(%error, "Replicate cancellation failed"),
            }
        }));
    }

    fn require_https(&self, url: &Url) -> Result<()> {
        let test_http = cfg!(test)
            && self.api_base.scheme() == "http"
            && url.origin() == self.api_base.origin();
        ensure!(
            url.scheme() == "https" || test_http,
            "Replicate returned a non-HTTPS image URL"
        );
        Ok(())
    }
}

#[derive(Deserialize)]
struct Prediction {
    id: String,
    status: String,
    output: Option<Value>,
    error: Option<Value>,
}

#[derive(Deserialize)]
struct UploadedFile {
    urls: FileUrls,
}

#[derive(Deserialize)]
struct FileUrls {
    get: Url,
}

fn retryable_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::REQUEST_TIMEOUT
            | StatusCode::TOO_MANY_REQUESTS
            | StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT
    )
}

fn retryable_read(error: &anyhow::Error) -> bool {
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

async fn api_response(response: Response, key: &str) -> Result<Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    match status {
        StatusCode::UNAUTHORIZED => bail!(
            "Replicate rejected the API token (HTTP 401 Unauthorized); check the saved Replicate token"
        ),
        StatusCode::PAYMENT_REQUIRED => bail!(
            "Replicate requires available credit (HTTP 402 Payment Required); check your Replicate balance"
        ),
        StatusCode::FORBIDDEN => {
            bail!("Replicate denied access (HTTP 403 Forbidden); check model and account access")
        }
        _ => {}
    }
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let Ok(chunk) = chunk else {
            break;
        };
        let remaining = MAX_ERROR_BYTES - bytes.len();
        bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
        if bytes.len() == MAX_ERROR_BYTES {
            break;
        }
    }
    let body = serde_json::from_slice::<Value>(&bytes).ok();
    bail!(
        "Replicate API request failed: {status}; {}",
        safe_detail(body.as_ref(), key)
    )
}

fn safe_detail(value: Option<&Value>, key: &str) -> String {
    let text = value
        .and_then(|value| {
            value.as_str().or_else(|| {
                ["detail", "error", "title"]
                    .into_iter()
                    .find_map(|field| value.get(field).and_then(Value::as_str))
            })
        })
        .unwrap_or("no error details provided");
    let text = text.replace(key, "[redacted token]");
    text.split_whitespace()
        .map(|word| {
            if word.contains("data:") || word.len() > 120 {
                "[redacted]"
            } else {
                word
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(2048)
        .collect()
}

async fn read_image(response: Response) -> Result<Vec<u8>> {
    let response = response.error_for_status()?;
    let mime = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    ensure!(
        mime.starts_with("image/"),
        "Replicate image response has an invalid content type"
    );
    if let Some(length) = response.content_length() {
        ensure!(
            length <= MAX_IMAGE_BYTES as u64,
            "Replicate image exceeds the 64 MiB limit"
        );
    }
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        ensure!(
            bytes.len().saturating_add(chunk.len()) <= MAX_IMAGE_BYTES,
            "Replicate image exceeds the 64 MiB limit"
        );
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use image::{ImageFormat, Rgb, RgbImage};
    use serde_json::{Value, json};
    use std::{
        io::Cursor,
        sync::{Arc, Mutex},
    };
    use tokio::{
        io::{AsyncReadExt as _, AsyncWriteExt as _},
        sync::Notify,
    };

    #[derive(Clone, Copy)]
    enum Mode {
        Success,
        ArrayOutput,
        Unauthorized,
        PaymentRequired,
        LostCreate,
        Failed,
        Running,
        PollRetry,
        DownloadRetry,
        DownloadUnavailable,
        BadImage,
        Oversized,
    }

    struct Request {
        method: String,
        path: String,
        authorization: Option<String>,
        body: Vec<u8>,
    }

    struct Server {
        base: Url,
        requests: Arc<Mutex<Vec<Request>>>,
        polled: Arc<Notify>,
        canceled: Arc<Notify>,
        task: tokio::task::JoinHandle<()>,
    }

    impl Drop for Server {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    fn png(image: &DynamicImage) -> Vec<u8> {
        let mut out = Cursor::new(Vec::new());
        image.write_to(&mut out, ImageFormat::Png).unwrap();
        out.into_inner()
    }

    impl Server {
        async fn start(mode: Mode) -> Self {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base =
                Url::parse(&format!("http://{}/v1/", listener.local_addr().unwrap())).unwrap();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let polled = Arc::new(Notify::new());
            let canceled = Arc::new(Notify::new());
            let (base2, requests2, polled2, canceled2) = (
                base.clone(),
                requests.clone(),
                polled.clone(),
                canceled.clone(),
            );
            let task = tokio::spawn(async move {
                loop {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    let mut buffer = Vec::new();
                    let header_end = loop {
                        let mut chunk = [0; 8192];
                        let n = socket.read(&mut chunk).await.unwrap();
                        if n == 0 {
                            return;
                        }
                        buffer.extend_from_slice(&chunk[..n]);
                        if let Some(i) = buffer.windows(4).position(|x| x == b"\r\n\r\n") {
                            break i + 4;
                        }
                    };
                    let headers = String::from_utf8_lossy(&buffer[..header_end]).to_string();
                    let header = |name: &str| {
                        headers.lines().find_map(|line| {
                            let (key, value) = line.split_once(':')?;
                            key.eq_ignore_ascii_case(name)
                                .then(|| value.trim().to_owned())
                        })
                    };
                    let length =
                        header("content-length").map_or(0, |x| x.parse::<usize>().unwrap());
                    while buffer.len() < header_end + length {
                        let mut chunk = [0; 8192];
                        let n = socket.read(&mut chunk).await.unwrap();
                        if n == 0 {
                            return;
                        }
                        buffer.extend_from_slice(&chunk[..n]);
                    }
                    let mut first = headers.lines().next().unwrap().split_whitespace();
                    let method = first.next().unwrap().to_owned();
                    let path = first.next().unwrap().to_owned();
                    let attempt = {
                        let mut requests = requests2.lock().unwrap();
                        requests.push(Request {
                            method: method.clone(),
                            path: path.clone(),
                            authorization: header("authorization"),
                            body: buffer[header_end..header_end + length].to_vec(),
                        });
                        requests
                            .iter()
                            .filter(|r| r.path == path && r.method == method)
                            .count()
                    };
                    let prediction = |status: &str| {
                        json!({
                            "id": "job-1", "status": status,
                            "output": if status != "succeeded" { Value::Null }
                                else if matches!(mode, Mode::ArrayOutput) { json!([base2.join("image").unwrap()]) }
                                else { json!(base2.join("image").unwrap()) },
                            "error": if status == "failed" { json!("content policy: secret-test-token data:image/png;base64,private-pixels") } else { Value::Null },
                        })
                    };
                    let (status, mime, body, length) = if path == "/v1/files" {
                        let body =
                            json!({"urls": {"get": base2.join("files/input/download").unwrap()}})
                                .to_string()
                                .into_bytes();
                        ("201 Created", "application/json", body.clone(), body.len())
                    } else if method == "POST" && path.ends_with("/cancel") {
                        canceled2.notify_one();
                        ("200 OK", "application/json", b"{}".to_vec(), 2)
                    } else if method == "POST" {
                        if matches!(mode, Mode::LostCreate) {
                            socket.shutdown().await.unwrap();
                            continue;
                        }
                        let (status, body) = match mode {
                            Mode::Unauthorized => (
                                "401 Unauthorized",
                                json!({"detail": "secret-test-token data:image/png;base64,private-pixels"}),
                            ),
                            Mode::PaymentRequired => (
                                "402 Payment Required",
                                json!({"detail": "Insufficient credit"}),
                            ),
                            _ => ("201 Created", prediction("starting")),
                        };
                        let body = body.to_string().into_bytes();
                        (status, "application/json", body.clone(), body.len())
                    } else if path == "/v1/predictions/job-1" {
                        polled2.notify_one();
                        if matches!(mode, Mode::PollRetry) && attempt == 1 {
                            ("503 Service Unavailable", "application/json", Vec::new(), 0)
                        } else {
                            let body = prediction(match mode {
                                Mode::Failed => "failed",
                                Mode::Running => "processing",
                                _ => "succeeded",
                            })
                            .to_string()
                            .into_bytes();
                            ("200 OK", "application/json", body.clone(), body.len())
                        }
                    } else if path == "/v1/image" {
                        if matches!(mode, Mode::DownloadUnavailable) {
                            ("503 Service Unavailable", "text/plain", Vec::new(), 0)
                        } else if matches!(mode, Mode::DownloadRetry) && attempt == 1 {
                            ("200 OK", "image/png", b"partial".to_vec(), 100)
                        } else if matches!(mode, Mode::BadImage) {
                            ("200 OK", "image/png", b"invalid".to_vec(), 7)
                        } else {
                            let body = png(&DynamicImage::new_rgb8(2, 1));
                            let length = if matches!(mode, Mode::Oversized) {
                                65 * 1024 * 1024
                            } else {
                                body.len()
                            };
                            ("200 OK", "image/png", body, length)
                        }
                    } else {
                        ("404 Not Found", "text/plain", Vec::new(), 0)
                    };
                    let headers = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: {mime}\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n"
                    );
                    socket.write_all(headers.as_bytes()).await.unwrap();
                    let _ = socket.write_all(&body).await;
                    let _ = socket.shutdown().await;
                }
            });
            Self {
                base,
                requests,
                polled,
                canceled,
                task,
            }
        }

        fn client(&self) -> Replicate {
            Replicate {
                client: reqwest::Client::builder()
                    .no_proxy()
                    .read_timeout(Duration::from_secs(1))
                    .build()
                    .unwrap(),
                api_base: self.base.clone(),
                poll_interval: Duration::from_millis(1),
                retry_delay: Duration::from_millis(1),
            }
        }
    }

    #[tokio::test]
    async fn replicate_flux_variants_send_their_documented_inputs_and_read_outputs() {
        for (slug, input_field, settings, array_output) in [
            (
                "flux-2-pro",
                "input_images",
                json!({"resolution":"match_input_image", "safety_tolerance":5}),
                false,
            ),
            (
                "flux-2-max",
                "input_images",
                json!({"resolution":"match_input_image", "safety_tolerance":5}),
                false,
            ),
            (
                "flux-2-flex",
                "input_images",
                json!({"resolution":"match_input_image", "safety_tolerance":5, "prompt_upsampling":false}),
                false,
            ),
            (
                "flux-2-klein-4b",
                "images",
                json!({"go_fast":false,"disable_safety_checker":true,"output_megapixels":"0.25"}),
                true,
            ),
            (
                "flux-2-klein-9b",
                "images",
                json!({"go_fast":false,"disable_safety_checker":true,"output_megapixels":"0.25"}),
                true,
            ),
            (
                "flux-2-klein-4b-base",
                "images",
                json!({"go_fast":false,"disable_safety_checker":true,"output_megapixels":"0.25"}),
                true,
            ),
            (
                "flux-2-klein-9b-base",
                "images",
                json!({"go_fast":false,"disable_safety_checker":true,"output_megapixels":"0.25"}),
                true,
            ),
            (
                "flux-kontext-dev",
                "input_image",
                json!({"go_fast":false,"disable_safety_checker":true}),
                false,
            ),
            (
                "flux-kontext-pro",
                "input_image",
                json!({"safety_tolerance":2,"prompt_upsampling":false}),
                false,
            ),
            (
                "flux-kontext-max",
                "input_image",
                json!({"safety_tolerance":2,"prompt_upsampling":false}),
                false,
            ),
        ] {
            let id = format!("black-forest-labs/{slug}");
            let model = ReplicateModel::from_id(&id).unwrap();
            let server = Server::start(if array_output {
                Mode::ArrayOutput
            } else {
                Mode::Success
            })
            .await;
            let image = DynamicImage::new_rgb8(3, 2);
            let output = server
                .client()
                .edit_with_key(
                    model,
                    "  Exact edit.  ",
                    &image,
                    &StopToken::default(),
                    "secret-test-token",
                )
                .await
                .unwrap()
                .unwrap();
            assert_eq!((output.width(), output.height()), (3, 2), "{id}");
            let requests = server.requests.lock().unwrap();
            assert_eq!(requests[0].path, format!("/v1/models/{id}/predictions"));
            let mut expected = json!({"prompt":"  Exact edit.  ","aspect_ratio":"match_input_image","output_format":"png"});
            let image_uri = format!("data:image/png;base64,{}", STANDARD.encode(png(&image)));
            expected[input_field] = if input_field == "input_image" {
                json!(image_uri)
            } else {
                json!([image_uri])
            };
            expected
                .as_object_mut()
                .unwrap()
                .extend(settings.as_object().unwrap().clone());
            assert_eq!(
                serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
                json!({"input":expected}),
                "{id}"
            );
            assert_eq!(requests.iter().filter(|r| r.method == "POST").count(), 1);
            assert!(
                requests
                    .iter()
                    .filter(|r| r.path == "/v1/image")
                    .all(|r| r.authorization.is_none())
            );
        }
    }

    #[test]
    fn replicate_klein_preserves_page_resolution_and_rejects_ambiguous_outputs() {
        let model = ReplicateModel::Flux2Klein9B;
        for (width, height, expected) in [
            (1000, 1000, "1"),
            (1000, 1001, "2"),
            (1600, 2400, "4"),
            (4000, 6000, "4"),
        ] {
            assert_eq!(
                model.input("clean", "https://example.com/input.png", width, height)["output_megapixels"],
                expected
            );
        }
        assert!(model.output_url(json!([])).is_err());
        assert!(
            model
                .output_url(json!([
                    "https://example.com/1.png",
                    "https://example.com/2.png"
                ]))
                .is_err()
        );
        assert!(
            model
                .output_url(json!("https://example.com/1.png"))
                .is_err()
        );
        assert!(ReplicateModel::from_id("black-forest-labs/flux-schnell").is_err());
    }

    #[tokio::test]
    async fn replicate_edits_with_the_documented_payload_and_downloads_without_credentials() {
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        let server = Server::start(Mode::Success).await;
        let image = DynamicImage::new_rgb8(3, 2);
        let output = server
            .client()
            .edit_with_key(
                ReplicateModel::Flux2Dev,
                "  exact prompt\n  ",
                &image,
                &StopToken::default(),
                "Bearer secret-test-token",
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!((output.width(), output.height()), (3, 2));
        let requests = server.requests.lock().unwrap();
        let create = &requests[0];
        assert_eq!(
            create.path,
            "/v1/models/black-forest-labs/flux-2-dev/predictions"
        );
        assert_eq!(
            create.authorization.as_deref(),
            Some("Bearer secret-test-token")
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&create.body).unwrap(),
            json!({"input": {
                "prompt": "  exact prompt\n  ", "input_images": [format!("data:image/png;base64,{}", STANDARD.encode(png(&image)))],
                "aspect_ratio": "match_input_image", "output_format": "png", "go_fast": false, "disable_safety_checker": true,
            }})
        );
        assert_eq!(requests.iter().filter(|r| r.method == "POST").count(), 1);
        assert!(
            requests
                .iter()
                .filter(|r| r.path == "/v1/image")
                .all(|r| r.authorization.is_none())
        );
    }

    #[tokio::test]
    async fn replicate_uploads_large_pngs_before_creating_one_prediction() {
        let mut random = 42u32;
        let image = DynamicImage::ImageRgb8(RgbImage::from_fn(400, 400, |_, _| {
            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
            Rgb([
                (random >> 24) as u8,
                (random >> 16) as u8,
                (random >> 8) as u8,
            ])
        }));
        let encoded = png(&image);
        assert!(encoded.len() > 256 * 1024);
        let server = Server::start(Mode::Success).await;
        server
            .client()
            .edit_with_key(
                ReplicateModel::Flux2Dev,
                "cleanup",
                &image,
                &StopToken::default(),
                "secret-test-token",
            )
            .await
            .unwrap()
            .unwrap();
        let requests = server.requests.lock().unwrap();
        assert_eq!(requests[0].path, "/v1/files");
        assert!(
            requests[0]
                .body
                .windows(encoded.len())
                .any(|window| window == encoded)
        );
        let body: Value = serde_json::from_slice(&requests[1].body).unwrap();
        assert_eq!(
            body["input"]["input_images"][0],
            server.base.join("files/input/download").unwrap().as_str()
        );
        assert_eq!(
            requests
                .iter()
                .filter(|r| r.method == "POST" && r.path.ends_with("/predictions"))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn replicate_retries_reads_without_recreating_the_prediction() {
        for mode in [Mode::PollRetry, Mode::DownloadRetry] {
            let server = Server::start(mode).await;
            server
                .client()
                .edit_with_key(
                    ReplicateModel::Flux2Dev,
                    "cleanup",
                    &DynamicImage::new_rgb8(1, 1),
                    &StopToken::default(),
                    "secret-test-token",
                )
                .await
                .unwrap()
                .unwrap();
            let requests = server.requests.lock().unwrap();
            assert_eq!(requests.iter().filter(|r| r.method == "POST").count(), 1);
            let path = if matches!(mode, Mode::PollRetry) {
                "/v1/predictions/job-1"
            } else {
                "/v1/image"
            };
            assert_eq!(requests.iter().filter(|r| r.path == path).count(), 2);
        }
    }

    #[tokio::test]
    async fn replicate_rejections_and_lost_create_responses_are_not_retried() {
        for (mode, expected) in [
            (Mode::Unauthorized, "401"),
            (Mode::PaymentRequired, "402"),
            (Mode::LostCreate, "not automatically resubmitted"),
            (Mode::Failed, "content policy"),
            (Mode::BadImage, "invalid image"),
            (Mode::Oversized, "64 MiB"),
        ] {
            let server = Server::start(mode).await;
            let error = server
                .client()
                .edit_with_key(
                    ReplicateModel::Flux2Dev,
                    "cleanup",
                    &DynamicImage::new_rgb8(1, 1),
                    &StopToken::default(),
                    "secret-test-token",
                )
                .await
                .unwrap_err();
            let message = format!("{error:#}");
            assert!(message.contains(expected), "{message}");
            assert!(!message.contains("secret-test-token"), "{message}");
            assert!(!message.contains("private-pixels"), "{message}");
            assert_eq!(
                server
                    .requests
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|r| r.method == "POST")
                    .count(),
                1
            );
        }
    }

    #[tokio::test]
    async fn replicate_exhausted_download_keeps_the_result_url() {
        let server = Server::start(Mode::DownloadUnavailable).await;
        let error = server
            .client()
            .edit_with_key(
                ReplicateModel::Flux2Dev,
                "cleanup",
                &DynamicImage::new_rgb8(1, 1),
                &StopToken::default(),
                "secret-test-token",
            )
            .await
            .unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains(server.base.join("image").unwrap().as_str()),
            "{message}"
        );
        let requests = server.requests.lock().unwrap();
        assert_eq!(requests.iter().filter(|r| r.method == "POST").count(), 1);
        assert_eq!(requests.iter().filter(|r| r.path == "/v1/image").count(), 4);
    }

    #[tokio::test]
    async fn replicate_stop_cancels_an_existing_prediction() {
        let server = Server::start(Mode::Running).await;
        let stop = StopToken::default();
        let client = server.client();
        let task = tokio::spawn({
            let stop = stop.clone();
            async move {
                client
                    .edit_with_key(
                        ReplicateModel::Flux2Dev,
                        "cleanup",
                        &DynamicImage::new_rgb8(1, 1),
                        &stop,
                        "secret-test-token",
                    )
                    .await
            }
        });
        tokio::time::timeout(Duration::from_secs(2), server.polled.notified())
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
        tokio::time::timeout(Duration::from_secs(1), server.canceled.notified())
            .await
            .unwrap();
    }

    pub(in crate::stages) async fn submit_reviewed_image(
        prompt: &str,
        image: &DynamicImage,
        stop: &StopToken,
    ) -> Value {
        let server = Server::start(Mode::Success).await;
        server
            .client()
            .edit_with_key(
                ReplicateModel::Flux2Dev,
                prompt,
                image,
                stop,
                "secret-test-token",
            )
            .await
            .unwrap()
            .unwrap();
        let body: Value = serde_json::from_slice(&server.requests.lock().unwrap()[0].body).unwrap();
        body["input"].clone()
    }
}
