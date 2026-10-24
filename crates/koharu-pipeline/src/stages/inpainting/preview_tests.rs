use super::*;
use crate::{ImageEditReview, ImageEditReviewer, InpaintingProvider, StopToken};
use std::time::Duration;

struct Reviewer {
    seen: Mutex<Vec<ImageEditReview>>,
    answer: Option<String>,
    pending: Option<Arc<tokio::sync::Notify>>,
}

impl Reviewer {
    fn new(answer: Option<&str>) -> Arc<Self> {
        Arc::new(Self {
            seen: Mutex::new(Vec::new()),
            answer: answer.map(str::to_owned),
            pending: None,
        })
    }
}

#[async_trait]
impl ImageEditReviewer for Reviewer {
    async fn review(&self, request: ImageEditReview, _stop: &StopToken) -> Result<Option<String>> {
        self.seen.lock().unwrap().push(request);
        if let Some(entered) = &self.pending {
            entered.notify_one();
            std::future::pending::<()>().await;
        }
        Ok(self.answer.clone())
    }
}

fn png(image: DynamicImage) -> Arc<[u8]> {
    let mut bytes = Cursor::new(Vec::new());
    image.write_to(&mut bytes, ImageFormat::Png).unwrap();
    Arc::from(bytes.into_inner())
}

async fn scene_input(restoration: bool) -> StageInput {
    let mut session = koharu_scene::Session::memory().await.unwrap();
    let mut page = None;
    let patch = session
        .snapshot()
        .patch(|edit| {
            let id = edit.add_page(
                koharu_scene::PageDraft::new("Review page", 4.0, 2.0),
                At::End,
            )?;
            let asset = |image: RgbaImage| {
                AssetInput::new(
                    png(DynamicImage::ImageRgba8(image)),
                    "image/png",
                    AssetMetadata {
                        width: Some(4),
                        height: Some(2),
                        attributes: BTreeMap::new(),
                    },
                )
            };
            edit.set_asset(
                id,
                &AssetRole::new("source")?,
                asset(RgbaImage::from_pixel(4, 2, Rgba([10, 20, 30, 255]))),
            )?;
            for (kind, x, color, opacity, visible) in [
                (RasterLayerKind::Cleanup, 1, [200, 0, 0, 255], 0.5, true),
                (RasterLayerKind::Paint, 1, [0, 0, 200, 255], 0.5, true),
                (RasterLayerKind::Paint, 2, [200, 100, 0, 255], 1.0, true),
                (RasterLayerKind::Cleanup, 2, [40, 50, 60, 255], 1.0, true),
                (RasterLayerKind::Cleanup, 1, [0, 255, 0, 255], 1.0, false),
                (RasterLayerKind::Paint, 2, [255, 0, 255, 255], 1.0, false),
                (RasterLayerKind::Paint, 3, [255, 255, 0, 255], 0.0, true),
            ] {
                let entity = edit.add_entity(id, At::End)?;
                edit.set(
                    entity,
                    &RasterLayer {
                        origin: Origin::User,
                        name: "Layer".to_owned(),
                        kind,
                    },
                )?;
                edit.set(
                    entity,
                    &Visibility {
                        origin: Origin::User,
                        visible,
                        opacity,
                    },
                )?;
                let mut pixels = RgbaImage::new(4, 2);
                pixels.put_pixel(x, 0, Rgba(color));
                edit.set_asset(entity, &AssetRole::new("source")?, asset(pixels))?;
            }
            let content = edit.add_text_content(id, At::End)?;
            edit.set(
                content,
                &koharu_scene::SourceText {
                    text: koharu_scene::Authored::user(
                        "Text must not enter the image-edit input".to_owned(),
                    ),
                    language: None,
                },
            )?;
            edit.add_text_layer(
                id,
                At::End,
                content,
                &koharu_scene::TextLayout {
                    origin: Origin::User,
                    kind: koharu_scene::TextLayoutKind::Paragraph,
                    angle_degrees: None,
                },
            )?;
            page = Some(id);
            Ok(())
        })
        .unwrap();
    let scene = session.commit(patch).await.unwrap().snapshot;
    let page = page.unwrap();
    StageInput::new(
        scene,
        page,
        None,
        None,
        Arc::new(crate::ImageCache::default()),
        restoration.then(|| {
            let mut mask = GrayImage::new(4, 2);
            mask.put_pixel(1, 0, Luma([255]));
            mask.put_pixel(2, 0, Luma([255]));
            crate::InpaintingMask {
                page,
                png: png(DynamicImage::ImageLuma8(mask)),
                intent: InpaintingIntent::Restore,
            }
        }),
    )
}

#[tokio::test]
async fn image_edit_preview_approval_preserves_edited_prompt_verbatim() {
    let reviewer = Reviewer::new(Some("  reviewed prompt\nNo extra instructions.  "));
    let input = scene_input(false)
        .await
        .with_image_edit_reviewer(Some(reviewer.clone()));
    let prepared = prepare(&input, true, false, false).await.unwrap();
    let result = reviewed_prompt(
        &input,
        InpaintingProvider::Fal,
        "model",
        "final prompt",
        &prepared.image,
    )
    .await
    .unwrap();
    assert_eq!(
        result.as_deref(),
        Some("  reviewed prompt\nNo extra instructions.  ")
    );
    let seen = reviewer.seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert!(Arc::ptr_eq(&seen[0].image, &prepared.image));
    assert_eq!(seen[0].prompt, "final prompt");
    assert_eq!(seen[0].page_label, "Review page");
    assert_eq!(seen[0].page, input.page().unwrap());
    assert_eq!(seen[0].provider, InpaintingProvider::Fal);
    assert_eq!(seen[0].model, "model");
    assert!(!seen[0].restoration);
}

#[tokio::test]
async fn image_edit_preview_cancel_stops_the_run() {
    let input = scene_input(false)
        .await
        .with_image_edit_reviewer(Some(Reviewer::new(None)));
    let prepared = prepare(&input, true, false, false).await.unwrap();
    let result = reviewed_prompt(
        &input,
        InpaintingProvider::OpenRouter,
        "model",
        "prompt",
        &prepared.image,
    )
    .await
    .unwrap();
    assert!(result.is_none());
    assert!(input.stop.stopped());
}

#[tokio::test]
async fn image_edit_preview_rejects_empty_and_nul_prompts() {
    for prompt in ["", " \n\t", "edited\0prompt"] {
        let input = scene_input(false)
            .await
            .with_image_edit_reviewer(Some(Reviewer::new(Some(prompt))));
        let prepared = prepare(&input, true, false, false).await.unwrap();
        let result = reviewed_prompt(
            &input,
            InpaintingProvider::Fal,
            "model",
            "prompt",
            &prepared.image,
        )
        .await;
        assert!(
            result.is_err(),
            "invalid reviewed prompt was accepted: {prompt:?}"
        );
    }
}

#[tokio::test]
async fn image_edit_preview_stop_interrupts_a_reviewer_that_never_returns() {
    let entered = Arc::new(tokio::sync::Notify::new());
    let reviewer = Arc::new(Reviewer {
        seen: Mutex::new(Vec::new()),
        answer: None,
        pending: Some(entered.clone()),
    });
    let input = scene_input(false)
        .await
        .with_image_edit_reviewer(Some(reviewer));
    let stop = input.stop.clone();
    let task = tokio::spawn(async move {
        let prepared = prepare(&input, true, false, false).await.unwrap();
        reviewed_prompt(
            &input,
            InpaintingProvider::Fal,
            "model",
            "prompt",
            &prepared.image,
        )
        .await
    });
    tokio::time::timeout(Duration::from_secs(1), entered.notified())
        .await
        .expect("reviewer did not start");
    stop.stop();
    assert!(
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("review did not stop")
            .unwrap()
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn image_edit_preview_model_receives_final_pixels_and_prompt_before_cancellation() {
    for provider in [
        InpaintingProvider::Fal,
        InpaintingProvider::OpenRouter,
        InpaintingProvider::Replicate,
    ] {
        for restoration in [false, true] {
            let reviewer = Reviewer::new(None);
            let input = scene_input(restoration)
                .await
                .with_image_edit_reviewer(Some(reviewer.clone()));
            let config = ImageEditConfig {
                prompt: "  configured prompt  ".to_owned(),
                apply_mode: InpaintingApplyMode::FullPage,
            };
            let model = match provider {
                InpaintingProvider::Fal => Model::Fal {
                    client: Fal::new().unwrap(),
                    model: FalModel::MaiImage2_5Edit,
                    config,
                },
                InpaintingProvider::OpenRouter => Model::OpenRouter {
                    client: OpenRouterImage::new().unwrap(),
                    model: "provider/edit".to_owned(),
                    config,
                },
                InpaintingProvider::Replicate => Model::Replicate {
                    client: Replicate::new().unwrap(),
                    model: replicate::ReplicateModel::Flux2Dev,
                    config,
                },
            };
            let stop = input.stop.clone();
            // No credentials are read and no HTTP call occurs when review cancels.
            let patch = model.run(input).await.unwrap();
            assert!(patch.is_empty());
            assert!(stop.stopped());
            let seen = reviewer.seen.lock().unwrap();
            let [request] = seen.as_slice() else {
                panic!("expected one review request")
            };
            assert_eq!(request.provider, provider);
            assert_eq!(request.restoration, restoration);
            let expected_prompt = if provider == InpaintingProvider::Fal && !restoration {
                format!("configured prompt\n\nSafety boundary: {DEFAULT_CLEANUP_PROMPT}")
            } else {
                "  configured prompt  ".to_owned()
            };
            assert_eq!(request.prompt, expected_prompt);
            if restoration {
                assert_eq!(request.image.dimensions(), (2, 1));
                assert_eq!(request.image.get_pixel(0, 0), Rgba([52, 4, 107, 254]));
                assert_eq!(request.image.get_pixel(1, 0), Rgba([40, 50, 60, 255]));
            } else {
                assert_eq!(request.image.dimensions(), (4, 2));
                assert_eq!(request.image.get_pixel(0, 0), Rgba([10, 20, 30, 255]));
                // The hand-made cleanup and the paint over it, as restoration
                // sees them.
                assert_eq!(request.image.get_pixel(1, 0), Rgba([52, 4, 107, 254]));
                assert_eq!(request.image.get_pixel(2, 0), Rgba([40, 50, 60, 255]));
                assert_eq!(request.image.get_pixel(3, 0), Rgba([10, 20, 30, 255]));
            }
        }
    }
}

#[tokio::test]
async fn image_edit_preview_sends_approved_prompt_and_pixels_to_each_http_boundary() {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    for provider in [
        InpaintingProvider::Fal,
        InpaintingProvider::OpenRouter,
        InpaintingProvider::Replicate,
    ] {
        let reviewer = Reviewer::new(Some("  user edit\nverbatim  "));
        let input = scene_input(true)
            .await
            .with_image_edit_reviewer(Some(reviewer));
        let prepared = prepare(&input, true, true, false).await.unwrap();
        let prompt = reviewed_prompt(
            &input,
            provider,
            "provider/edit",
            "configured",
            &prepared.image,
        )
        .await
        .unwrap()
        .unwrap();
        let body = match provider {
            InpaintingProvider::Fal => {
                super::super::fal::submit_reviewed_image(
                    &ImageEditConfig {
                        prompt,
                        apply_mode: InpaintingApplyMode::Mask,
                    },
                    &prepared.image,
                    &input.stop,
                )
                .await
            }
            InpaintingProvider::OpenRouter => {
                super::super::openrouter_image::submit_reviewed_image(
                    "provider/edit",
                    &prompt,
                    &prepared.image,
                    &input.stop,
                )
                .await
            }
            InpaintingProvider::Replicate => {
                super::super::replicate::tests::submit_reviewed_image(
                    &prompt,
                    &prepared.image,
                    &input.stop,
                )
                .await
            }
        };
        assert_eq!(body["prompt"], "  user edit\nverbatim  ");
        let data_uri = match provider {
            InpaintingProvider::Fal => body["image_urls"][0].as_str().unwrap(),
            InpaintingProvider::OpenRouter => body["input_references"][0]["image_url"]["url"]
                .as_str()
                .unwrap(),
            InpaintingProvider::Replicate => body["input_images"][0].as_str().unwrap(),
        };
        let sent = image::load_from_memory(
            &STANDARD
                .decode(data_uri.strip_prefix("data:image/png;base64,").unwrap())
                .unwrap(),
        )
        .unwrap();
        assert_eq!(sent.dimensions(), (2, 1));
        assert_eq!(sent.get_pixel(0, 0), Rgba([52, 4, 107, 254]));
        assert_eq!(sent.get_pixel(1, 0), Rgba([40, 50, 60, 255]));
        assert!(body.get("mask").is_none());
        assert!(body.get("mask_url").is_none());
    }
}

#[tokio::test]
async fn image_edit_preview_stopped_run_skips_review_and_disabled_review_preserves_prompt() {
    let input = scene_input(false).await;
    let prepared = prepare(&input, true, false, false).await.unwrap();
    assert_eq!(
        reviewed_prompt(
            &input,
            InpaintingProvider::Fal,
            "model",
            "  unchanged  ",
            &prepared.image
        )
        .await
        .unwrap()
        .as_deref(),
        Some("  unchanged  ")
    );
    let reviewer = Reviewer::new(Some("must not send"));
    let input = input.with_image_edit_reviewer(Some(reviewer.clone()));
    input.stop.stop();
    let patch = Model::OpenRouter {
        client: OpenRouterImage::new().unwrap(),
        model: "provider/edit".to_owned(),
        config: ImageEditConfig::default(),
    }
    .run(input)
    .await
    .unwrap();
    assert!(patch.is_empty());
    assert!(reviewer.seen.lock().unwrap().is_empty());
}
