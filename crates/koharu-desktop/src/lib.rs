//! Native page preparation shared by browser presentation and export.

use std::{collections::HashSet, future::Future, sync::Arc};

use anyhow::{Context as _, Result, bail};
use koharu_rasterizer::{Rasterizer, ResourceId};
use koharu_renderer::{Frame as RenderedFrame, LayerKind, RenderDiagnostic, Renderer};
use koharu_scene::{Commit, EntityId, Geometry, Point, Revision, Snapshot};
use parking_lot::{Mutex as SyncMutex, RwLock};
use serde::{Deserialize, Serialize};
use specta::Type;
use tokio::sync::{Mutex, Notify, OnceCell};

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize, Type)]
pub struct Frame {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub angle_degrees: f32,
}

impl Frame {
    fn is_valid(self) -> bool {
        [self.x, self.y, self.width, self.height, self.angle_degrees]
            .into_iter()
            .all(f32::is_finite)
            && self.width > 0.0
            && self.height > 0.0
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Type)]
pub struct TransformFrame {
    pub element: EntityId,
    pub frame: Frame,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct CanvasState {
    pub page: Option<EntityId>,
    pub revision: Option<Revision>,
    #[specta(type = f64)]
    pub generation: u64,
    pub size: [u32; 2],
    pub element_frames: Vec<TransformFrame>,
    /// Text the renderer could not fit at a readable size on the presented page.
    pub text_fit_warnings: Vec<TextFitWarning>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Type)]
pub struct TextFitWarning {
    pub element: EntityId,
    pub kind: TextFitWarningKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum TextFitWarningKind {
    /// The text does not fit its area even at the smallest size tried.
    Overflow,
    /// The text fits only below the readable minimum size.
    BelowReadableSize,
}

#[derive(Default)]
struct PresentationState {
    generation: u64,
    frame: Option<RenderedFrame>,
    requested_preparation: u64,
}

const MAX_CACHED_PAGE_FRAMES: usize = 8;

#[derive(Default)]
struct PageFrameCache {
    clock: u64,
    entries: Vec<CachedPageFrame>,
}

struct CachedPageFrame {
    page: EntityId,
    revision: Revision,
    frame: RenderedFrame,
    last_used: u64,
}

pub struct Desktop {
    renderer: Renderer,
    rasterizer: OnceCell<Arc<Rasterizer>>,
    presentation: RwLock<PresentationState>,
    page_frames: SyncMutex<PageFrameCache>,
    preparation: Mutex<()>,
    preparation_changed: Notify,
}

impl Desktop {
    pub fn new() -> Result<Self> {
        Ok(Self {
            renderer: Renderer::new()?,
            rasterizer: OnceCell::new(),
            presentation: RwLock::new(PresentationState::default()),
            page_frames: SyncMutex::new(PageFrameCache::default()),
            preparation: Mutex::new(()),
            preparation_changed: Notify::new(),
        })
    }

    #[must_use]
    pub fn renderer(&self) -> Renderer {
        self.renderer.clone()
    }

    pub async fn rasterizer(&self) -> Result<Arc<Rasterizer>> {
        self.rasterizer
            .get_or_try_init(|| async {
                let rasterizer = tokio_rayon::spawn(Rasterizer::new).await?;
                Ok::<_, anyhow::Error>(Arc::new(rasterizer))
            })
            .await
            .cloned()
    }

    pub fn frame_manifest_bytes(&self, generation: u64) -> Result<Vec<u8>> {
        let frame = self.frame_at_generation(generation)?;
        frame.prepared().manifest()?.encode().map_err(Into::into)
    }

    pub fn frame_resource_bytes(&self, generation: u64, resource: ResourceId) -> Result<Vec<u8>> {
        self.frame_at_generation(generation)?
            .prepared()
            .resource_packet(resource)?
            .encode()
            .map_err(Into::into)
    }

    pub async fn prepare_page(&self, snapshot: &Snapshot, page: EntityId) -> Result<bool> {
        let request = self.request_preparation();
        let _preparation = self.preparation.lock().await;
        if !self.is_current_preparation(request) {
            return Ok(false);
        }
        let revision = snapshot.revision();
        let cached = self.page_frames.lock().get(page, revision);
        if cached.is_some() {
            return Ok(true);
        }
        let Some(frame) = self
            .await_preparation(request, self.renderer.render(snapshot, page))
            .await
        else {
            return Ok(false);
        };
        let frame = frame?;
        if !self.is_current_preparation(request) {
            return Ok(false);
        }
        self.page_frames.lock().insert(frame);
        Ok(true)
    }

    pub fn page_manifest_bytes(&self, page: EntityId, revision: Revision) -> Result<Vec<u8>> {
        self.page_frame(page, revision)?
            .prepared()
            .manifest()?
            .encode()
            .map_err(Into::into)
    }

    pub fn page_resource_bytes(
        &self,
        page: EntityId,
        revision: Revision,
        resource: ResourceId,
    ) -> Result<Vec<u8>> {
        self.page_frame(page, revision)?
            .prepared()
            .resource_packet(resource)?
            .encode()
            .map_err(Into::into)
    }

    fn frame_at_generation(&self, generation: u64) -> Result<RenderedFrame> {
        let presentation = self.presentation.read();
        if presentation.generation != generation {
            bail!(
                "canvas generation {generation} is stale; current generation is {}",
                presentation.generation
            );
        }
        Ok(presentation
            .frame
            .as_ref()
            .context("there is no prepared canvas frame")?
            .clone())
    }

    fn page_frame(&self, page: EntityId, revision: Revision) -> Result<RenderedFrame> {
        self.page_frames
            .lock()
            .get(page, revision)
            .with_context(|| format!("canvas page {page} at revision {revision} is not prepared"))
    }

    #[must_use]
    pub fn canvas_state(&self) -> CanvasState {
        let presentation = self.presentation.read();
        let Some(frame) = presentation.frame.as_ref() else {
            return CanvasState {
                page: None,
                revision: None,
                generation: presentation.generation,
                size: [0, 0],
                element_frames: Vec::new(),
                text_fit_warnings: Vec::new(),
            };
        };
        let (width, height) = frame.size();
        let element_frames = frame
            .layers()
            .iter()
            .filter(|layer| layer.entity() != frame.page())
            .filter(|layer| {
                let presentation = layer.presentation();
                presentation.visible && presentation.opacity > 0.0
            })
            .filter_map(|layer| {
                let value = layer.element_frame()?;
                Some(TransformFrame {
                    element: layer.entity(),
                    frame: Frame {
                        x: value.x,
                        y: value.y,
                        width: value.width,
                        height: value.height,
                        angle_degrees: value.angle_degrees,
                    },
                })
            })
            .collect();
        CanvasState {
            page: Some(frame.page()),
            revision: Some(frame.revision()),
            generation: presentation.generation,
            size: [width, height],
            element_frames,
            text_fit_warnings: text_fit_warnings(frame.diagnostics()),
        }
    }

    pub fn transform_geometries(
        &self,
        expected_revision: Revision,
        elements: &[TransformFrame],
    ) -> Result<Vec<(EntityId, Geometry, f32)>> {
        let presentation = self.presentation.read();
        let rendered = presentation
            .frame
            .as_ref()
            .context("there is no prepared canvas frame")?;
        if rendered.revision() != expected_revision {
            bail!(
                "canvas transform expected revision {expected_revision}, but the prepared frame is at {}",
                rendered.revision()
            );
        }
        let mut seen = HashSet::with_capacity(elements.len());
        let mut geometries = Vec::with_capacity(elements.len());
        for element in elements {
            if !seen.insert(element.element) {
                bail!("canvas transform repeats element {}", element.element);
            }
            if !element.frame.is_valid() {
                bail!("canvas transform frame must be finite and non-empty");
            }
            let layer = rendered.layer(element.element).with_context(|| {
                format!(
                    "canvas transform element {} is not rendered",
                    element.element
                )
            })?;
            let layer_presentation = layer.presentation();
            if !layer_presentation.visible
                || layer_presentation.opacity <= 0.0
                || !matches!(layer.kind(), LayerKind::Text(_))
            {
                bail!(
                    "canvas transform element {} is not selectable",
                    element.element
                );
            }
            let original = layer
                .element_frame()
                .context("canvas transform element has no control frame")?;
            let original = Frame {
                x: original.x,
                y: original.y,
                width: original.width,
                height: original.height,
                angle_degrees: original.angle_degrees,
            };
            let coefficients = frame_transform(original, element.frame);
            let points = layer
                .geometry()
                .points
                .iter()
                .map(|point| transform_point(coefficients, point))
                .collect::<Vec<_>>();
            // Moving or rotating keeps the balloon's shape. Resizing makes the area the
            // box the user dragged, like a paragraph text box, instead of stretching the
            // balloon outline past the balloon.
            let resized = (element.frame.width - original.width).abs() > RESIZE_EPSILON
                || (element.frame.height - original.height).abs() > RESIZE_EPSILON;
            let geometry = Geometry {
                origin: layer.geometry().origin.clone(),
                points: if resized {
                    rectangle_in_frame_axes(&points, element.frame.angle_degrees)
                } else {
                    points
                },
            };
            if geometry != *layer.geometry()
                || element.frame.angle_degrees != original.angle_degrees
            {
                geometries.push((element.element, geometry, element.frame.angle_degrees));
            }
        }
        Ok(geometries)
    }

    #[tracing::instrument(skip_all)]
    pub async fn synchronize(
        &self,
        snapshot: &Snapshot,
        page: Option<EntityId>,
        commit: &Commit,
    ) -> Result<bool> {
        let request = self.request_preparation();
        let _preparation = self.preparation.lock().await;
        if !self.is_current_preparation(request) {
            return Ok(false);
        }
        let (current_page, revision, previous) = {
            let presentation = self.presentation.read();
            let frame = presentation.frame.as_ref();
            (
                frame.map(RenderedFrame::page),
                frame.map(RenderedFrame::revision),
                frame.cloned(),
            )
        };
        if current_page != page {
            return self.show_page_locked(snapshot, page, request).await;
        }
        if revision.is_some_and(|revision| revision >= snapshot.revision()) {
            return Ok(false);
        }
        let Some(page) = page else {
            if self.replace_frame_if_current(request, None) {
                self.renderer.discard_retained_nodes();
            }
            return Ok(false);
        };
        let rendered = if commit.revision == snapshot.revision()
            && revision == Some(commit.changes.from)
            && let Some(previous) = previous.as_ref()
        {
            self.await_preparation(
                request,
                self.renderer.update(previous, snapshot, &commit.changes),
            )
            .await
        } else {
            self.await_preparation(request, self.renderer.render(snapshot, page))
                .await
        };
        let Some(rendered) = rendered else {
            return Ok(false);
        };
        let rendered = rendered?;
        if !self.replace_frame_if_current(request, Some(rendered)) {
            return Ok(false);
        }
        Ok(true)
    }

    #[tracing::instrument(skip_all)]
    pub async fn show_page(&self, snapshot: &Snapshot, page: Option<EntityId>) -> Result<bool> {
        let request = self.request_preparation();
        let _preparation = self.preparation.lock().await;
        if !self.is_current_preparation(request) {
            return Ok(false);
        }
        self.show_page_locked(snapshot, page, request).await
    }

    pub async fn clear(&self) {
        let request = self.request_preparation();
        let _preparation = self.preparation.lock().await;
        if !self.is_current_preparation(request) {
            return;
        }
        if self.replace_frame_if_current(request, None) {
            self.renderer.discard_retained_nodes();
        }
    }

    async fn show_page_locked(
        &self,
        snapshot: &Snapshot,
        page: Option<EntityId>,
        request: u64,
    ) -> Result<bool> {
        if !self.is_current_preparation(request) {
            return Ok(false);
        }
        let Some(page) = page else {
            let replaced = self.replace_frame_if_current(request, None);
            if replaced {
                self.renderer.discard_retained_nodes();
            }
            return Ok(replaced);
        };
        let cached = self.page_frames.lock().get(page, snapshot.revision());
        if let Some(frame) = cached {
            return Ok(self.replace_frame_if_current(request, Some(frame)));
        }
        let Some(frame) = self
            .await_preparation(request, self.renderer.render(snapshot, page))
            .await
        else {
            return Ok(false);
        };
        let frame = frame?;
        if !self.replace_frame_if_current(request, Some(frame)) {
            return Ok(false);
        }
        Ok(true)
    }

    fn request_preparation(&self) -> u64 {
        let request = {
            let mut presentation = self.presentation.write();
            presentation.requested_preparation = presentation.requested_preparation.wrapping_add(1);
            presentation.requested_preparation
        };
        self.preparation_changed.notify_waiters();
        request
    }

    fn is_current_preparation(&self, request: u64) -> bool {
        self.presentation.read().requested_preparation == request
    }

    async fn await_preparation<F>(&self, request: u64, future: F) -> Option<F::Output>
    where
        F: Future,
    {
        tokio::select! {
            biased;
            () = self.preparation_superseded(request) => None,
            output = future => self.is_current_preparation(request).then_some(output),
        }
    }

    async fn preparation_superseded(&self, request: u64) {
        loop {
            let changed = self.preparation_changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if !self.is_current_preparation(request) {
                return;
            }
            changed.await;
        }
    }

    fn replace_frame_if_current(&self, request: u64, frame: Option<RenderedFrame>) -> bool {
        let mut presentation = self.presentation.write();
        if presentation.requested_preparation != request {
            return false;
        }
        presentation.generation = presentation.generation.wrapping_add(1).max(1);
        presentation.frame.clone_from(&frame);
        drop(presentation);
        if let Some(frame) = frame {
            self.page_frames.lock().insert(frame);
        } else {
            self.page_frames.lock().clear();
        }
        true
    }
}

impl PageFrameCache {
    fn get(&mut self, page: EntityId, revision: Revision) -> Option<RenderedFrame> {
        self.clock = self.clock.wrapping_add(1);
        let entry = self
            .entries
            .iter_mut()
            .find(|entry| entry.page == page && entry.revision == revision)?;
        entry.last_used = self.clock;
        Some(entry.frame.clone())
    }

    fn insert(&mut self, frame: RenderedFrame) {
        self.clock = self.clock.wrapping_add(1);
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.page == frame.page() && entry.revision == frame.revision())
        {
            entry.frame = frame;
            entry.last_used = self.clock;
            return;
        }
        self.entries.push(CachedPageFrame {
            page: frame.page(),
            revision: frame.revision(),
            frame,
            last_used: self.clock,
        });
        if self.entries.len() > MAX_CACHED_PAGE_FRAMES {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(index, _)| index)
                .expect("an over-capacity page frame cache is non-empty");
            self.entries.swap_remove(oldest);
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

fn frame_transform(original: Frame, preview: Frame) -> [f64; 6] {
    let original_angle = f64::from(original.angle_degrees).to_radians();
    let preview_angle = f64::from(preview.angle_degrees).to_radians();
    let (original_sin, original_cos) = original_angle.sin_cos();
    let (preview_sin, preview_cos) = preview_angle.sin_cos();
    let scale_x = f64::from(preview.width / original.width);
    let scale_y = f64::from(preview.height / original.height);
    let a = preview_cos * scale_x * original_cos + preview_sin * scale_y * original_sin;
    let b = preview_sin * scale_x * original_cos - preview_cos * scale_y * original_sin;
    let c = preview_cos * scale_x * original_sin - preview_sin * scale_y * original_cos;
    let d = preview_sin * scale_x * original_sin + preview_cos * scale_y * original_cos;
    let original_center_x = f64::from(original.x + original.width * 0.5);
    let original_center_y = f64::from(original.y + original.height * 0.5);
    let preview_center_x = f64::from(preview.x + preview.width * 0.5);
    let preview_center_y = f64::from(preview.y + preview.height * 0.5);
    [
        a,
        b,
        c,
        d,
        preview_center_x - a * original_center_x - c * original_center_y,
        preview_center_y - b * original_center_x - d * original_center_y,
    ]
}

const RESIZE_EPSILON: f32 = 0.01;

/// One warning per text element; overflow wins because the text is cut off.
fn text_fit_warnings(diagnostics: &[RenderDiagnostic]) -> Vec<TextFitWarning> {
    let mut warnings = Vec::<TextFitWarning>::new();
    for diagnostic in diagnostics {
        let (element, kind) = match diagnostic {
            RenderDiagnostic::TextOverflow { entity, .. } => {
                (*entity, TextFitWarningKind::Overflow)
            }
            RenderDiagnostic::TextBelowReadableSize { entity, .. } => {
                (*entity, TextFitWarningKind::BelowReadableSize)
            }
            RenderDiagnostic::MissingAsset { .. } => continue,
        };
        match warnings
            .iter_mut()
            .find(|warning| warning.element == element)
        {
            Some(warning) if kind == TextFitWarningKind::Overflow => warning.kind = kind,
            Some(_) => {}
            None => warnings.push(TextFitWarning { element, kind }),
        }
    }
    warnings
}

/// The rectangle that bounds `points` along axes rotated by `angle_degrees`, as
/// top-left, top-right, bottom-right, bottom-left so its top edge keeps the angle.
fn rectangle_in_frame_axes(points: &[Point], angle_degrees: f32) -> Vec<Point> {
    let (sin, cos) = f64::from(angle_degrees).to_radians().sin_cos();
    let (mut min_u, mut min_v) = (f64::INFINITY, f64::INFINITY);
    let (mut max_u, mut max_v) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for point in points {
        let u = point.x * cos + point.y * sin;
        let v = -point.x * sin + point.y * cos;
        min_u = min_u.min(u);
        max_u = max_u.max(u);
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    [
        (min_u, min_v),
        (max_u, min_v),
        (max_u, max_v),
        (min_u, max_v),
    ]
    .into_iter()
    .map(|(u, v)| Point {
        x: u * cos - v * sin,
        y: u * sin + v * cos,
    })
    .collect()
}

fn transform_point([a, b, c, d, e, f]: [f64; 6], point: &Point) -> Point {
    Point {
        x: a * point.x + c * point.y + e,
        y: b * point.x + d * point.y + f,
    }
}

#[cfg(test)]
mod tests {
    use std::{future::pending, sync::Arc, time::Duration};

    use tokio::{sync::Notify, time::timeout};

    use koharu_scene::Point;

    use koharu_renderer::{RenderBounds, RenderDiagnostic};
    use koharu_scene::EntityId;

    use super::{
        Desktop, TextFitWarning, TextFitWarningKind, rectangle_in_frame_axes, text_fit_warnings,
    };

    #[test]
    fn overflow_outranks_a_small_size_for_the_same_text() {
        let first = EntityId::new();
        let second = EntityId::new();
        let below = |entity| RenderDiagnostic::TextBelowReadableSize {
            entity,
            font_size: 6.0,
            minimum_font_size: 9.0,
        };
        let overflow = |entity| RenderDiagnostic::TextOverflow {
            entity,
            available: RenderBounds::default(),
            actual_width: 50.0,
            actual_height: 50.0,
            font_size: 9.0,
        };

        let warnings = text_fit_warnings(&[below(first), overflow(first), below(second)]);

        assert_eq!(
            warnings,
            [
                TextFitWarning {
                    element: first,
                    kind: TextFitWarningKind::Overflow
                },
                TextFitWarning {
                    element: second,
                    kind: TextFitWarningKind::BelowReadableSize
                },
            ]
        );
    }

    #[test]
    fn resized_balloon_area_becomes_the_dragged_rectangle() {
        let ellipse = (0..16)
            .map(|step| {
                let angle = f64::from(step) * std::f64::consts::TAU / 16.0;
                Point {
                    x: 100.0 + 40.0 * angle.cos(),
                    y: 200.0 + 60.0 * angle.sin(),
                }
            })
            .collect::<Vec<_>>();

        let rectangle = rectangle_in_frame_axes(&ellipse, 0.0);

        let corners = rectangle
            .iter()
            .map(|p| (p.x.round(), p.y.round()))
            .collect::<Vec<_>>();
        assert_eq!(
            corners,
            [(60.0, 140.0), (140.0, 140.0), (140.0, 260.0), (60.0, 260.0)]
        );
    }

    #[test]
    fn rotated_area_rectangle_keeps_the_frame_angle() {
        let square =
            [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)].map(|(x, y)| Point { x, y });

        let rectangle = rectangle_in_frame_axes(&square, 90.0);

        let top = (
            rectangle[1].x - rectangle[0].x,
            rectangle[1].y - rectangle[0].y,
        );
        assert!(top.0.abs() < 1e-9 && (top.1 - 10.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn newer_request_preempts_current_preparation_and_acquires_ownership() {
        let desktop = Arc::new(Desktop::new().unwrap());
        let first_request = desktop.request_preparation();
        let started = Arc::new(Notify::new());
        let first = {
            let desktop = Arc::clone(&desktop);
            let started = Arc::clone(&started);
            tokio::spawn(async move {
                let _preparation = desktop.preparation.lock().await;
                started.notify_one();
                desktop
                    .await_preparation(first_request, pending::<()>())
                    .await
            })
        };
        started.notified().await;

        let latest_request = desktop.request_preparation();
        let _preparation = timeout(Duration::from_secs(1), desktop.preparation.lock())
            .await
            .expect("the obsolete preparation must release ownership");
        assert!(first.await.unwrap().is_none());
        assert_eq!(
            desktop
                .await_preparation(latest_request, std::future::ready(7))
                .await,
            Some(7)
        );
    }

    #[tokio::test]
    async fn obsolete_request_cannot_replace_the_presentation() {
        let desktop = Desktop::new().unwrap();
        let obsolete = desktop.request_preparation();
        let latest = desktop.request_preparation();

        assert!(!desktop.replace_frame_if_current(obsolete, None));
        assert_eq!(desktop.canvas_state().generation, 0);
        assert!(desktop.replace_frame_if_current(latest, None));
        assert_eq!(desktop.canvas_state().generation, 1);
    }
}
