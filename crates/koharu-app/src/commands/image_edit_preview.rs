use std::{io::Cursor, sync::Arc};

use anyhow::{Context as _, Result, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::ImageFormat;
use koharu_pipeline::{ImageEditReview, ImageEditReviewer, StopToken};
use parking_lot::Mutex;
use serde::Serialize;
use specta::Type;
use tauri::{AppHandle, Manager as _, State, ipc::Channel};
use tauri_runtime_cef::CefRuntime;
use tokio::sync::oneshot;
use uuid::Uuid;

use super::{Error, processing::JobId};

#[derive(Clone, Debug, Serialize, Type)]
pub struct ImageEditPreview {
    pub id: Uuid,
    pub job: JobId,
    pub page: koharu_scene::EntityId,
    pub page_label: String,
    pub provider: koharu_pipeline::InpaintingProvider,
    pub model: String,
    pub prompt: String,
    pub image_url: String,
    pub width: u32,
    pub height: u32,
    pub restoration: bool,
}

struct PendingPreview {
    preview: ImageEditPreview,
    response: oneshot::Sender<Option<String>>,
    stop: StopToken,
}

#[derive(Default)]
struct PreviewState {
    channel: Option<Channel<Vec<ImageEditPreview>>>,
    pending: Vec<PendingPreview>,
}

impl PreviewState {
    fn previews(&self) -> Vec<ImageEditPreview> {
        self.pending
            .iter()
            .map(|entry| entry.preview.clone())
            .collect()
    }

    fn publish(&mut self) -> Result<()> {
        let result = self
            .channel
            .as_ref()
            .context("image preview window is not connected")?
            .send(self.previews());
        if let Err(error) = result {
            self.channel.take();
            for pending in self.pending.drain(..) {
                pending.stop.stop();
            }
            return Err(error).context("could not show the image request preview");
        }
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct ImageEditPreviews {
    inner: Mutex<PreviewState>,
}

impl ImageEditPreviews {
    pub(crate) fn subscribe(&self, channel: Channel<Vec<ImageEditPreview>>) -> Result<()> {
        let mut state = self.inner.lock();
        state.channel = Some(channel);
        state.publish()
    }

    pub(crate) fn pending(&self) -> Vec<ImageEditPreview> {
        self.inner.lock().previews()
    }

    fn begin(&self, preview: ImageEditPreview, stop: StopToken) -> Result<PendingReview<'_>> {
        let mut state = self.inner.lock();
        ensure!(!stop.stopped(), "image edit was stopped before preview");
        ensure!(
            state.channel.is_some(),
            "image preview window is not connected"
        );
        let id = preview.id;
        let (send, response) = oneshot::channel();
        state.pending.push(PendingPreview {
            preview,
            response: send,
            stop,
        });
        state.publish()?;
        Ok(PendingReview {
            response,
            _guard: PreviewGuard { previews: self, id },
        })
    }

    fn resolve(&self, id: Uuid, prompt: Option<String>) -> Result<()> {
        if let Some(prompt) = prompt.as_ref() {
            ensure!(
                !prompt.trim().is_empty(),
                "image edit prompt must not be empty"
            );
            ensure!(!prompt.contains('\0'), "image edit prompt contains NUL");
        }
        let mut state = self.inner.lock();
        let index = state
            .pending
            .iter()
            .position(|entry| entry.preview.id == id)
            .context("this image preview is no longer pending")?;
        let pending = state.pending.remove(index);
        if prompt.is_none() {
            pending.stop.stop();
        }
        let result = pending
            .response
            .send(prompt)
            .map_err(|_| anyhow::anyhow!("this image request is no longer waiting"));
        // The response is already accepted. A disconnected UI must not turn it
        // into a retryable command error or keep other jobs waiting indefinitely.
        let _ = state.publish();
        result
    }

    fn remove(&self, id: Uuid) {
        let mut state = self.inner.lock();
        if let Some(index) = state
            .pending
            .iter()
            .position(|entry| entry.preview.id == id)
        {
            state.pending.remove(index);
            let _ = state.publish();
        }
    }

    pub(crate) fn cancel_all(&self) {
        let mut state = self.inner.lock();
        for pending in state.pending.drain(..) {
            pending.stop.stop();
        }
        let _ = state.publish();
    }
}

struct PendingReview<'a> {
    response: oneshot::Receiver<Option<String>>,
    _guard: PreviewGuard<'a>,
}

struct PreviewGuard<'a> {
    previews: &'a ImageEditPreviews,
    id: Uuid,
}

impl Drop for PreviewGuard<'_> {
    fn drop(&mut self) {
        // Also runs when the pipeline drops a cancelled review future.
        self.previews.remove(self.id);
    }
}

pub(crate) struct DesktopImageEditReviewer {
    handle: AppHandle<CefRuntime>,
    job: JobId,
}

impl DesktopImageEditReviewer {
    pub(crate) fn for_job(
        handle: AppHandle<CefRuntime>,
        job: JobId,
    ) -> Result<Option<Arc<dyn ImageEditReviewer>>> {
        let enabled = koharu_pipeline::PipelineConfig::load()?
            .read()?
            .inpainting
            .preview_before_send;
        Ok(enabled.then(|| Arc::new(Self { handle, job }) as Arc<dyn ImageEditReviewer>))
    }
}

#[async_trait::async_trait]
impl ImageEditReviewer for DesktopImageEditReviewer {
    async fn review(&self, request: ImageEditReview, stop: &StopToken) -> Result<Option<String>> {
        if stop.stopped() {
            return Ok(None);
        }
        let image = Arc::clone(&request.image);
        let image_url = tokio::task::spawn_blocking(move || -> Result<String> {
            let mut png = Cursor::new(Vec::new());
            image.write_to(&mut png, ImageFormat::Png)?;
            Ok(format!(
                "data:image/png;base64,{}",
                STANDARD.encode(png.into_inner())
            ))
        })
        .await
        .context("image preview encoding failed")??;
        if stop.stopped() {
            return Ok(None);
        }
        let preview = ImageEditPreview {
            id: Uuid::new_v4(),
            job: self.job,
            page: request.page,
            page_label: request.page_label,
            provider: request.provider,
            model: request.model,
            prompt: request.prompt,
            image_url,
            width: request.image.width(),
            height: request.image.height(),
            restoration: request.restoration,
        };
        let previews = self.handle.state::<ImageEditPreviews>();
        let pending = previews.begin(preview, stop.clone())?;
        tokio::select! {
            biased;
            () = stop.cancelled() => Ok(None),
            response = pending.response => response.context("image preview closed before receiving an answer"),
        }
    }
}

#[tauri::command]
#[specta::specta]
pub(crate) fn resolve_image_edit_preview(
    id: Uuid,
    prompt: Option<String>,
    previews: State<'_, ImageEditPreviews>,
) -> Result<(), Error> {
    previews.resolve(id, prompt)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preview(job: JobId) -> ImageEditPreview {
        ImageEditPreview {
            id: Uuid::new_v4(),
            job,
            page: koharu_scene::EntityId::new(),
            page_label: "page.png".to_owned(),
            provider: koharu_pipeline::InpaintingProvider::OpenRouter,
            model: "microsoft/mai-image-2.6".to_owned(),
            prompt: "Original prompt".to_owned(),
            image_url: "data:image/png;base64,test".to_owned(),
            width: 2,
            height: 3,
            restoration: true,
        }
    }

    fn subscribed_previews() -> ImageEditPreviews {
        let previews = ImageEditPreviews::default();
        previews.subscribe(Channel::new(|_| Ok(()))).unwrap();
        previews
    }

    #[tokio::test]
    async fn preview_reply_is_exact_and_consumed_once() {
        let previews = subscribed_previews();
        let preview = preview(JobId::new());
        let id = preview.id;
        let stop = StopToken::default();
        let pending = previews.begin(preview, stop.clone()).unwrap();
        assert_eq!(previews.pending().len(), 1);
        previews
            .resolve(id, Some("  Exact\nprompt  ".to_owned()))
            .unwrap();
        assert_eq!(
            pending.response.await.unwrap(),
            Some("  Exact\nprompt  ".to_owned())
        );
        assert!(!stop.stopped());
        assert!(previews.pending().is_empty());
        assert!(previews.resolve(id, Some("Again".to_owned())).is_err());
    }

    #[tokio::test]
    async fn cancelling_preview_stops_only_its_job() {
        let previews = subscribed_previews();
        let first = preview(JobId::new());
        let id = first.id;
        let first_stop = StopToken::default();
        let second_stop = StopToken::default();
        let first = previews.begin(first, first_stop.clone()).unwrap();
        let second = previews
            .begin(preview(JobId::new()), second_stop.clone())
            .unwrap();
        previews.resolve(id, None).unwrap();
        assert_eq!(first.response.await.unwrap(), None);
        assert!(first_stop.stopped());
        assert!(!second_stop.stopped());
        assert_eq!(previews.pending().len(), 1);
        drop(second);
        assert!(previews.pending().is_empty());
    }

    #[test]
    fn dropping_pending_review_removes_the_modal() {
        let previews = subscribed_previews();
        let pending = previews
            .begin(preview(JobId::new()), StopToken::default())
            .unwrap();
        assert_eq!(previews.pending().len(), 1);
        drop(pending);
        assert!(previews.pending().is_empty());
    }

    #[test]
    fn empty_prompt_keeps_preview_pending() {
        let previews = subscribed_previews();
        let preview = preview(JobId::new());
        let id = preview.id;
        let _pending = previews.begin(preview, StopToken::default()).unwrap();
        assert!(previews.resolve(id, Some(" \n ".to_owned())).is_err());
        assert_eq!(previews.pending().len(), 1);
    }

    #[test]
    fn unavailable_preview_channel_does_not_leave_a_waiter() {
        let previews = ImageEditPreviews::default();
        assert!(
            previews
                .begin(preview(JobId::new()), StopToken::default())
                .is_err()
        );
        assert!(previews.pending().is_empty());
    }

    #[tokio::test]
    async fn closing_project_cancels_every_waiting_request() {
        let previews = subscribed_previews();
        let stop = StopToken::default();
        let pending = previews.begin(preview(JobId::new()), stop.clone()).unwrap();
        previews.cancel_all();
        assert!(stop.stopped());
        assert!(pending.response.await.is_err());
        assert!(previews.pending().is_empty());
    }

    #[tokio::test]
    async fn disconnecting_window_cancels_pending_requests() {
        let previews = subscribed_previews();
        let stop = StopToken::default();
        let pending = previews.begin(preview(JobId::new()), stop.clone()).unwrap();
        assert!(
            previews
                .subscribe(Channel::new(|_| Err(tauri::Error::FailedToReceiveMessage)))
                .is_err()
        );
        assert!(stop.stopped());
        assert!(pending.response.await.is_err());
        assert!(previews.pending().is_empty());
    }
}
