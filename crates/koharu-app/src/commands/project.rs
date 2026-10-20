use std::{
    collections::{HashMap, HashSet},
    io::Cursor,
    path::PathBuf,
};

use anyhow::{Context as _, Result, anyhow, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::{DynamicImage, ImageFormat, RgbaImage};
use koharu_desktop::Frame;
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, Authored, BubbleRegion, Character, CharacterId,
    CharacterList, Commit, EntityId, EntityOrigin, FitsTo, FlowsIn, Gender,
    Geometry as SceneGeometry, Group as SceneGroup, Inside, Origin, PageDraft, PageMarkup,
    PageMarkupMissing, PageMarkupStatus, Point as ScenePoint, Presents,
    RasterLayer as SceneRasterLayer, RasterLayerKind, RecognizedFrom, Region as SceneRegion,
    RegionSpec, RemovePolicy, Revision, Session, Snapshot, SourceText as SceneSourceText,
    SpeechKind, SpeechMarkup, TextGroup as SceneTextGroup, TextLayout as SceneTextLayout,
    TextLayoutKind, TextRegion, TextRole as SceneTextRole, Translation as SceneTranslation,
    TranslationReview as SceneTranslationReview, TranslationReviewReason,
    Typography as SceneTypography, Visibility as SceneVisibility,
};
use serde::{Deserialize, Serialize};
use specta::Type;
use tokio::sync::Mutex;

use super::{
    canvas::Point,
    editing::{GeometryUpdate, TypographyUpdate},
};

const FREE_TEXT_ROLE: &str = "dev.koharu.text.free-text";
const DIALOGUE_ROLE: &str = "dev.koharu.text.dialogue";
const ONOMATOPOEIA_ROLE: &str = "dev.koharu.text.onomatopoeia";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RasterStrokeMode {
    Paint,
    Erase,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct ProjectInfo {
    pub name: String,
    pub revision: Revision,
    pub active_page: Option<EntityId>,
    pub can_undo: bool,
    pub can_redo: bool,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct ProjectSummary {
    pub name: String,
}

#[derive(Clone, Copy, Debug, Serialize, Type)]
pub struct PageSize {
    pub width: f64,
    pub height: f64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Type)]
pub struct PageWarning {
    pub stage: koharu_pipeline::Stage,
    pub model: String,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct PageSummary {
    pub id: EntityId,
    pub label: String,
    pub size: PageSize,
    pub source_asset: Option<String>,
    #[specta(type = f64)]
    pub layer_count: usize,
    pub warning: Option<PageWarning>,
    /// Why scene markup could not label this page, when it tried and failed.
    pub markup_missing: Option<PageMarkupMissing>,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct Page {
    pub id: EntityId,
    pub label: String,
    pub size: PageSize,
    pub layers: Vec<Layer>,
    pub regions: Vec<AnalysisRegion>,
}

#[derive(Clone, Debug, Serialize, Type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Layer {
    Group {
        id: EntityId,
        parent: Option<EntityId>,
        visibility: LayerVisibility,
        name: String,
        role: Option<GroupRole>,
    },
    Text {
        id: EntityId,
        parent: Option<EntityId>,
        geometry: Option<Geometry>,
        angle_degrees: Option<f32>,
        visibility: LayerVisibility,
        content: Box<TextContent>,
        typography: Option<Typography>,
        layout: TextLayoutKind,
        automatic_region: Option<EntityId>,
    },
    Raster {
        id: EntityId,
        parent: Option<EntityId>,
        visibility: LayerVisibility,
        image: Option<String>,
        name: String,
        kind: RasterLayerKind,
    },
    Image {
        id: EntityId,
        parent: Option<EntityId>,
        geometry: Geometry,
        visibility: LayerVisibility,
        image: String,
    },
    Artwork {
        id: EntityId,
        parent: Option<EntityId>,
        geometry: Geometry,
        visibility: LayerVisibility,
        image: String,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum GroupRole {
    Text,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct TextContent {
    pub id: EntityId,
    pub source: Option<SourceText>,
    pub translation: Option<Translation>,
    /// Present only while the review still applies to `translation`.
    pub translation_review: Option<TranslationReview>,
    /// The source text is a local OCR draft no API read has confirmed yet.
    pub ocr_unverified: bool,
    pub markup: Option<TextMarkup>,
    pub role: Option<String>,
    pub source_region: Option<EntityId>,
}

/// Scene markup of a text: its kind and the speaker's character id. A speaker
/// the character list no longer contains is reported as no speaker.
#[derive(Clone, Debug, Serialize, Type)]
pub struct TextMarkup {
    pub kind: SpeechKind,
    pub speaker: Option<CharacterId>,
    pub user: bool,
}

/// A character to save. A new character has no id yet; native code assigns it.
#[derive(Clone, Debug, Deserialize, Type)]
pub struct CharacterInput {
    pub id: Option<CharacterId>,
    pub name: String,
    pub gender: Gender,
    pub notes: String,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct TranslationReview {
    pub reason: TranslationReviewReason,
    pub reference: Option<String>,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct AnalysisRegion {
    pub id: EntityId,
    pub parent: Option<EntityId>,
    pub geometry: Geometry,
    pub kind: String,
    pub label: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, Type)]
pub struct Geometry {
    pub points: Vec<Point>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize, Type)]
pub struct LayerVisibility {
    pub visible: bool,
    pub opacity: f32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, Type)]
pub struct SourceText {
    pub text: String,
    pub language: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, Type)]
pub struct Translation {
    pub text: String,
    pub language: Option<String>,
    /// Written by the pipeline rather than typed by the user.
    pub generated: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, Type)]
pub struct Typography {
    pub preferred_font: Option<String>,
    pub font_weight: Option<u16>,
    pub font_style: Option<koharu_scene::FontStyle>,
    pub size: Option<f32>,
    pub auto_fit: bool,
    pub color: Option<[u8; 4]>,
    pub stroke_color: Option<[u8; 4]>,
    pub stroke_width: Option<f32>,
    pub alignment: Option<koharu_scene::TextAlignment>,
    pub writing_mode: Option<koharu_scene::WritingMode>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum TextTransferMode {
    Copy,
    Move,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, Type)]
pub struct TextClipboardRegion {
    pub id: EntityId,
    pub geometry: Geometry,
    pub label: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, Type)]
pub struct TextClipboardLayer {
    pub layer: EntityId,
    pub content: EntityId,
    pub source: Option<SourceText>,
    pub translation: Option<Translation>,
    pub role: Option<String>,
    pub geometry: Option<Geometry>,
    pub angle_degrees: Option<f32>,
    pub visibility: LayerVisibility,
    pub typography: Option<Typography>,
    pub layout: TextLayoutKind,
    pub source_region: Option<TextClipboardRegion>,
    pub automatic_region: Option<TextClipboardRegion>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, Type)]
pub struct TextClipboardPayload {
    pub version: u32,
    pub mode: TextTransferMode,
    pub source_page: EntityId,
    pub layers: Vec<TextClipboardLayer>,
}

pub(crate) struct CurrentProject {
    pub(crate) project: Mutex<Option<Project>>,
}

#[derive(Clone)]
pub(crate) struct ProjectLibrary {
    root: PathBuf,
}

impl ProjectLibrary {
    pub(crate) fn new() -> Result<Self> {
        let root = dirs::document_dir()
            .context("the Documents directory is unavailable")?
            .join("Koharu");
        std::fs::create_dir_all(&root)
            .with_context(|| format!("failed to create {}", root.display()))?;
        Ok(Self { root })
    }

    pub(crate) fn list(&self) -> Result<Vec<ProjectSummary>> {
        let mut projects = std::fs::read_dir(&self.root)
            .with_context(|| format!("failed to read {}", self.root.display()))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
            .filter_map(|entry| {
                let path = entry.path();
                let is_project_directory = path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("khrproj"))
                    && (path.join("state-a.khr").is_file() || path.join("state-b.khr").is_file());
                if !is_project_directory {
                    return None;
                }
                let last_used = ["state-a.khr", "state-b.khr"]
                    .into_iter()
                    .filter_map(|file| std::fs::metadata(path.join(file)).ok()?.modified().ok())
                    .max()
                    .unwrap_or(std::time::UNIX_EPOCH);
                Some((
                    last_used,
                    ProjectSummary {
                        name: path.file_stem()?.to_str()?.to_owned(),
                    },
                ))
            })
            .collect::<Vec<_>>();
        projects.sort_unstable_by(|(left_used, left), (right_used, right)| {
            right_used
                .cmp(left_used)
                .then_with(|| left.name.to_lowercase().cmp(&right.name.to_lowercase()))
        });
        Ok(projects.into_iter().map(|(_, project)| project).collect())
    }

    pub(crate) async fn create(&self, name: &str) -> Result<Project> {
        let (name, path) = self.resolve(name)?;
        Project::create(name, path).await
    }

    pub(crate) async fn open(&self, name: &str) -> Result<Project> {
        let (name, path) = self.resolve(name)?;
        Project::open(name, path).await
    }

    /// The character list of another project, read from its newest valid
    /// state. Opening takes the project's lock and never writes its state.
    pub(crate) async fn characters(&self, name: &str) -> Result<Vec<Character>> {
        let (_, path) = self.resolve(name)?;
        let session = Session::open(&path)
            .await
            .with_context(|| format!("failed to open {}", path.display()))?;
        Project::characters(&session.snapshot())
    }

    pub(crate) fn delete(&self, name: &str) -> Result<()> {
        let (_, path) = self.resolve(name)?;
        if !path.is_dir() {
            bail!("project {name:?} does not exist");
        }
        std::fs::remove_dir_all(&path)
            .with_context(|| format!("failed to delete {}", path.display()))
    }

    fn resolve(&self, name: &str) -> Result<(String, PathBuf)> {
        let name = validate_project_name(name)?;
        Ok((name.clone(), self.root.join(format!("{name}.khrproj"))))
    }
}

pub(crate) struct Project {
    pub(crate) session: Session,
    pub(crate) name: String,
    pub(crate) active_page: Option<EntityId>,
    pub(crate) undo: Vec<Vec<Revision>>,
    pub(crate) redo: Vec<Vec<Revision>>,
}

impl Project {
    pub(crate) async fn create(name: String, path: PathBuf) -> Result<Self> {
        let session = Session::create(&path)
            .await
            .with_context(|| format!("failed to create {}", path.display()))?;
        Ok(Self::new(session, name))
    }

    pub(crate) async fn open(name: String, path: PathBuf) -> Result<Self> {
        let session = Session::open(&path)
            .await
            .with_context(|| format!("failed to open {}", path.display()))?;
        Ok(Self::new(session, name))
    }

    fn new(session: Session, name: String) -> Self {
        let active_page = session.snapshot().pages().next().map(|page| page.id());
        Self {
            session,
            name,
            active_page,
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    pub(crate) fn snapshot(&self) -> Snapshot {
        self.session.snapshot()
    }

    pub(crate) fn revision(&self) -> Revision {
        self.snapshot().revision()
    }

    pub(crate) fn active_page(&self) -> Option<EntityId> {
        self.active_page
    }

    pub(crate) fn select_page(&mut self, page: EntityId) -> Result<()> {
        self.snapshot().page(page)?;
        self.active_page = Some(page);
        Ok(())
    }

    pub(crate) fn reconcile_page(&mut self) {
        let snapshot = self.snapshot();
        if self
            .active_page
            .is_none_or(|page| snapshot.page(page).is_err())
        {
            self.active_page = snapshot.pages().next().map(|page| page.id());
        }
    }

    pub(crate) fn info(&self) -> ProjectInfo {
        ProjectInfo {
            name: self.name.clone(),
            revision: self.revision(),
            active_page: self.active_page,
            can_undo: !self.undo.is_empty(),
            can_redo: !self.redo.is_empty(),
        }
    }

    pub(crate) async fn list_versions(&self) -> Result<Vec<koharu_scene::SavedVersion>> {
        self.session.list_versions().await.map_err(Into::into)
    }

    pub(crate) async fn save_version(&self, name: String) -> Result<koharu_scene::SavedVersion> {
        self.session.save_version(name).await.map_err(Into::into)
    }

    pub(crate) async fn delete_version(&self, id: koharu_scene::VersionId) -> Result<()> {
        self.session.delete_version(id).await.map_err(Into::into)
    }

    pub(crate) async fn restore_version(&mut self, id: koharu_scene::VersionId) -> Result<()> {
        self.session.restore_version(id).await?;
        self.undo.clear();
        self.redo.clear();
        self.reconcile_page();
        Ok(())
    }

    pub(crate) fn pages(snapshot: &Snapshot) -> Result<Vec<PageSummary>> {
        snapshot
            .pages()
            .map(|page| {
                let value = page.page()?;
                let source_asset = Self::asset_id(snapshot, page.id(), "source")?;
                let layer_count = snapshot
                    .descendants(page.id())?
                    .map(|entity| Self::is_content_layer(snapshot, entity.id()))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .filter(|present| *present)
                    .count()
                    + usize::from(source_asset.is_some());
                Ok(PageSummary {
                    id: page.id(),
                    label: value.label,
                    size: PageSize {
                        width: value.width,
                        height: value.height,
                    },
                    source_asset,
                    layer_count,
                    warning: None,
                    markup_missing: match snapshot
                        .component::<PageMarkup>(page.id())?
                        .map(|markup| markup.status)
                    {
                        Some(PageMarkupStatus::Missing(reason)) => Some(reason),
                        _ => None,
                    },
                })
            })
            .collect()
    }

    pub(crate) async fn rename_page(&mut self, page: EntityId, label: String) -> Result<Commit> {
        let snapshot = self.snapshot();
        let current = snapshot.page(page)?.page()?;
        let patch = snapshot.patch(|edit| {
            edit.set_page(page, PageDraft::new(label, current.width, current.height))
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn delete_pages(&mut self, pages: Vec<EntityId>) -> Result<Commit> {
        let snapshot = self.snapshot();
        let pages = Self::unique_roots(&snapshot, pages)?;
        let patch = snapshot.patch(|edit| {
            for page in pages {
                edit.remove_entity(page, RemovePolicy::Cascade)?;
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn duplicate_pages(
        &mut self,
        pages: Vec<EntityId>,
    ) -> Result<(Commit, Vec<EntityId>)> {
        let snapshot = self.snapshot();
        let mut duplicated = Vec::new();
        let patch = snapshot.patch(|edit| {
            duplicated = edit.duplicate_pages(&pages)?;
            Ok(())
        })?;
        let commit = self.commit(patch).await?;
        self.active_page = duplicated.first().copied();
        Ok((commit, duplicated))
    }

    fn clipboard_region(snapshot: &Snapshot, id: EntityId) -> Result<TextClipboardRegion> {
        let region = snapshot
            .component::<SceneRegion>(id)?
            .context("clipboard relation target is not a region")?;
        if region.kind != TextRegion::kind() {
            bail!("text clipboard regions must be text regions");
        }
        let region = Self::analysis_region(snapshot, id, region)?;
        Ok(TextClipboardRegion {
            id,
            geometry: region.geometry,
            label: region.label,
        })
    }

    fn clipboard_layer(snapshot: &Snapshot, layer: EntityId) -> Result<TextClipboardLayer> {
        let Layer::Text {
            geometry,
            angle_degrees,
            visibility,
            content,
            typography,
            layout,
            automatic_region,
            ..
        } = Self::layer_view(snapshot, layer)?
        else {
            bail!("text clipboard contains a non-text layer");
        };
        let source_region = content
            .source_region
            .map(|region| Self::clipboard_region(snapshot, region))
            .transpose()?;
        let automatic_region = automatic_region
            .map(|region| Self::clipboard_region(snapshot, region))
            .transpose()?;
        Ok(TextClipboardLayer {
            layer,
            content: content.id,
            source: content.source,
            translation: content.translation,
            role: content.role,
            geometry,
            angle_degrees,
            visibility,
            typography,
            layout,
            source_region,
            automatic_region,
        })
    }

    fn clipboard_layer_matches(current: &TextClipboardLayer, payload: &TextClipboardLayer) -> bool {
        current.layer == payload.layer
            && current.content == payload.content
            && current.source == payload.source
            && current.translation == payload.translation
            && current.role == payload.role
            && Self::clipboard_geometry_matches(
                current.geometry.as_ref(),
                payload.geometry.as_ref(),
            )
            && match (current.angle_degrees, payload.angle_degrees) {
                (Some(current), Some(payload)) => {
                    Self::clipboard_float_matches(f64::from(current), f64::from(payload))
                }
                (current, payload) => current.is_none() && payload.is_none(),
            }
            && current.visibility.visible == payload.visibility.visible
            && Self::clipboard_float_matches(
                f64::from(current.visibility.opacity),
                f64::from(payload.visibility.opacity),
            )
            && Self::clipboard_typography_matches(
                current.typography.as_ref(),
                payload.typography.as_ref(),
            )
            && current.layout == payload.layout
            && Self::clipboard_region_matches(
                current.source_region.as_ref(),
                payload.source_region.as_ref(),
            )
            && Self::clipboard_region_matches(
                current.automatic_region.as_ref(),
                payload.automatic_region.as_ref(),
            )
    }

    fn clipboard_region_matches(
        current: Option<&TextClipboardRegion>,
        payload: Option<&TextClipboardRegion>,
    ) -> bool {
        match (current, payload) {
            (None, None) => true,
            (Some(current), Some(payload)) => {
                current.id == payload.id
                    && current.label == payload.label
                    && Self::clipboard_geometry_matches(
                        Some(&current.geometry),
                        Some(&payload.geometry),
                    )
            }
            _ => false,
        }
    }

    fn clipboard_geometry_matches(current: Option<&Geometry>, payload: Option<&Geometry>) -> bool {
        match (current, payload) {
            (None, None) => true,
            (Some(current), Some(payload)) => {
                current.points.len() == payload.points.len()
                    && current
                        .points
                        .iter()
                        .zip(&payload.points)
                        .all(|(current, payload)| {
                            Self::clipboard_float_matches(current.x, payload.x)
                                && Self::clipboard_float_matches(current.y, payload.y)
                        })
            }
            _ => false,
        }
    }

    fn clipboard_typography_matches(
        current: Option<&Typography>,
        payload: Option<&Typography>,
    ) -> bool {
        match (current, payload) {
            (None, None) => true,
            (Some(current), Some(payload)) => {
                current.preferred_font == payload.preferred_font
                    && current.font_weight == payload.font_weight
                    && current.font_style == payload.font_style
                    && Self::clipboard_optional_float_matches(current.size, payload.size)
                    && current.auto_fit == payload.auto_fit
                    && current.color == payload.color
                    && current.stroke_color == payload.stroke_color
                    && Self::clipboard_optional_float_matches(
                        current.stroke_width,
                        payload.stroke_width,
                    )
                    && current.alignment == payload.alignment
                    && current.writing_mode == payload.writing_mode
            }
            _ => false,
        }
    }

    fn clipboard_float_matches(current: f64, payload: f64) -> bool {
        current == payload || (current - payload).abs() <= 1e-6
    }

    fn clipboard_optional_float_matches(current: Option<f32>, payload: Option<f32>) -> bool {
        match (current, payload) {
            (None, None) => true,
            (Some(current), Some(payload)) => {
                Self::clipboard_float_matches(f64::from(current), f64::from(payload))
            }
            _ => false,
        }
    }

    fn clipboard_offset(payload: &TextClipboardPayload, position: Option<Point>) -> Result<Point> {
        let Some(position) = position else {
            return Ok(Point { x: 0.0, y: 0.0 });
        };
        if !position.x.is_finite() || !position.y.is_finite() {
            bail!("text paste position must contain finite coordinates");
        }
        let mut bounds: Option<(f64, f64, f64, f64)> = None;
        for geometry in payload.layers.iter().flat_map(|block| {
            [
                block.geometry.as_ref(),
                block.source_region.as_ref().map(|region| &region.geometry),
                block
                    .automatic_region
                    .as_ref()
                    .map(|region| &region.geometry),
            ]
            .into_iter()
            .flatten()
        }) {
            for point in &geometry.points {
                bounds = Some(match bounds {
                    Some((min_x, min_y, max_x, max_y)) => (
                        min_x.min(point.x),
                        min_y.min(point.y),
                        max_x.max(point.x),
                        max_y.max(point.y),
                    ),
                    None => (point.x, point.y, point.x, point.y),
                });
            }
        }
        let Some((min_x, min_y, max_x, max_y)) = bounds else {
            return Ok(Point { x: 0.0, y: 0.0 });
        };
        Ok(Point {
            x: position.x - (min_x + max_x) * 0.5,
            y: position.y - (min_y + max_y) * 0.5,
        })
    }

    fn translated_geometry(geometry: &Geometry, offset: Point) -> Geometry {
        Geometry {
            points: geometry
                .points
                .iter()
                .map(|point| Point {
                    x: point.x + offset.x,
                    y: point.y + offset.y,
                })
                .collect(),
        }
    }

    fn scene_geometry(geometry: &Geometry) -> SceneGeometry {
        SceneGeometry {
            origin: Origin::User,
            points: geometry
                .points
                .iter()
                .map(|point| ScenePoint {
                    x: point.x,
                    y: point.y,
                })
                .collect(),
        }
    }

    fn add_clipboard_region(
        edit: &mut koharu_scene::Edit,
        page: EntityId,
        region: &TextClipboardRegion,
        offset: Point,
    ) -> Result<EntityId, koharu_scene::Error> {
        let geometry = Self::translated_geometry(&region.geometry, offset);
        edit.add_analysis_region::<TextRegion>(
            page,
            At::End,
            &Self::scene_geometry(&geometry),
            region.label.clone(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn transfer_clipboard_region(
        edit: &mut koharu_scene::Edit,
        snapshot: &Snapshot,
        target_page: EntityId,
        region: &TextClipboardRegion,
        mode: TextTransferMode,
        movable: &HashSet<EntityId>,
        copies: &mut HashMap<EntityId, EntityId>,
        relocated: &mut HashSet<EntityId>,
        offset: Point,
    ) -> Result<EntityId, koharu_scene::Error> {
        if mode == TextTransferMode::Move && movable.contains(&region.id) {
            if relocated.insert(region.id) {
                edit.move_entity(region.id, Some(target_page), At::End)?;
                for relation in snapshot
                    .relations_from_as::<Inside>(region.id)
                    .map(|relation| relation.id())
                    .collect::<Vec<_>>()
                {
                    edit.remove_relation(relation)?;
                }
                if offset.x != 0.0 || offset.y != 0.0 {
                    let geometry = Self::translated_geometry(&region.geometry, offset);
                    edit.set(region.id, &Self::scene_geometry(&geometry))?;
                }
            }
            Ok(region.id)
        } else if let Some(region) = copies.get(&region.id) {
            Ok(*region)
        } else {
            let copied = Self::add_clipboard_region(edit, target_page, region, offset)?;
            copies.insert(region.id, copied);
            Ok(copied)
        }
    }

    fn add_clipboard_content(
        edit: &mut koharu_scene::Edit,
        page: EntityId,
        block: &TextClipboardLayer,
    ) -> Result<EntityId, koharu_scene::Error> {
        let content = edit.add_text_content(page, At::End)?;
        if let Some(source) = &block.source {
            edit.set(
                content,
                &SceneSourceText {
                    text: Authored::user(source.text.clone()),
                    language: source
                        .language
                        .as_ref()
                        .map(|language| koharu_scene::LanguageTag::new(language.clone()))
                        .transpose()?,
                },
            )?;
        }
        if let Some(translation) = &block.translation {
            edit.set(
                content,
                &SceneTranslation {
                    text: Authored::user(translation.text.clone()),
                    language: translation
                        .language
                        .as_ref()
                        .map(|language| koharu_scene::LanguageTag::new(language.clone()))
                        .transpose()?,
                },
            )?;
        }
        if let Some(role) = &block.role {
            edit.set(
                content,
                &SceneTextRole {
                    origin: Origin::User,
                    role: role.clone(),
                },
            )?;
        }
        Ok(content)
    }

    fn set_clipboard_layer_components(
        edit: &mut koharu_scene::Edit,
        layer: EntityId,
        block: &TextClipboardLayer,
        offset: Point,
    ) -> Result<(), koharu_scene::Error> {
        if let Some(geometry) = &block.geometry {
            let geometry = Self::translated_geometry(geometry, offset);
            edit.set(layer, &Self::scene_geometry(&geometry))?;
        }
        edit.set(
            layer,
            &SceneVisibility {
                origin: Origin::User,
                visible: block.visibility.visible,
                opacity: block.visibility.opacity,
            },
        )?;
        if let Some(typography) = &block.typography {
            edit.set(
                layer,
                &SceneTypography {
                    origin: Origin::User,
                    preferred_font: typography.preferred_font.clone(),
                    font_weight: typography.font_weight,
                    font_style: typography.font_style,
                    size: typography.size,
                    auto_fit: typography.auto_fit,
                    color: typography.color,
                    stroke_color: typography.stroke_color,
                    stroke_width: typography.stroke_width,
                    alignment: typography.alignment,
                    writing_mode: typography.writing_mode,
                    extensions: Default::default(),
                },
            )?;
        }
        Ok(())
    }

    pub(crate) async fn paste_text_blocks(
        &mut self,
        target_page: EntityId,
        payload: TextClipboardPayload,
        position: Option<Point>,
    ) -> Result<(Commit, Vec<EntityId>)> {
        if payload.version != 1 {
            bail!("unsupported text clipboard version {}", payload.version);
        }
        if payload.layers.is_empty() {
            bail!("text clipboard is empty");
        }
        let snapshot = self.snapshot();
        snapshot.page(target_page)?;
        snapshot.page(payload.source_page)?;
        let requested = payload
            .layers
            .iter()
            .map(|block| block.layer)
            .collect::<HashSet<_>>();
        if requested.len() != payload.layers.len() {
            bail!("text clipboard contains duplicate layers");
        }
        let source_order = Self::page(&snapshot, payload.source_page)?
            .layers
            .into_iter()
            .filter_map(|layer| match layer {
                Layer::Text { id, .. } if requested.contains(&id) => Some(id),
                _ => None,
            })
            .collect::<Vec<_>>();
        if source_order
            != payload
                .layers
                .iter()
                .map(|block| block.layer)
                .collect::<Vec<_>>()
        {
            bail!("text clipboard layer order is stale or invalid");
        }
        for block in &payload.layers {
            let current = Self::clipboard_layer(&snapshot, block.layer)?;
            if !Self::clipboard_layer_matches(&current, block) {
                bail!("text clipboard contents are stale");
            }
        }
        let offset = Self::clipboard_offset(&payload, position)?;

        let moved_layers = requested;
        let moved_contents = payload
            .layers
            .iter()
            .map(|block| block.content)
            .filter(|content| {
                snapshot
                    .relations_to_as::<Presents>(*content)
                    .all(|relation| moved_layers.contains(&relation.value().source))
            })
            .collect::<HashSet<_>>();
        let candidate_regions = payload
            .layers
            .iter()
            .flat_map(|block| {
                [
                    block.source_region.as_ref(),
                    block.automatic_region.as_ref(),
                ]
                .into_iter()
                .flatten()
                .map(|region| region.id)
            })
            .collect::<HashSet<_>>();
        let moved_regions = candidate_regions
            .into_iter()
            .filter(|region| {
                snapshot
                    .relations_to_as::<RecognizedFrom>(*region)
                    .all(|relation| moved_contents.contains(&relation.value().source))
                    && snapshot
                        .relations_to_as::<FitsTo>(*region)
                        .all(|relation| moved_layers.contains(&relation.value().source))
            })
            .collect::<HashSet<_>>();
        let cross_page = payload.source_page != target_page;
        let mut transferred = Vec::new();
        let patch = snapshot.patch(|edit| {
            let group = edit.ensure_text_group(target_page)?;
            let mut content_copies = HashMap::new();
            let mut region_copies = HashMap::new();
            let mut relocated_contents = HashSet::new();
            let mut relocated_regions = HashSet::new();
            let mut linked_contents = HashSet::new();
            let mut shifted_same_page_regions = HashSet::new();

            for block in &payload.layers {
                if payload.mode == TextTransferMode::Move && !cross_page {
                    edit.move_entity(block.layer, Some(group), At::End)?;
                    if offset.x != 0.0 || offset.y != 0.0 {
                        if let Some(geometry) = &block.geometry {
                            let geometry = Self::translated_geometry(geometry, offset);
                            edit.set(block.layer, &Self::scene_geometry(&geometry))?;
                        }
                        for region in [
                            block.source_region.as_ref(),
                            block.automatic_region.as_ref(),
                        ]
                        .into_iter()
                        .flatten()
                        {
                            if moved_regions.contains(&region.id)
                                && shifted_same_page_regions.insert(region.id)
                            {
                                let geometry = Self::translated_geometry(&region.geometry, offset);
                                edit.set(region.id, &Self::scene_geometry(&geometry))?;
                            }
                        }
                    }
                    transferred.push(block.layer);
                    continue;
                }

                let content = if payload.mode == TextTransferMode::Move
                    && moved_contents.contains(&block.content)
                {
                    if relocated_contents.insert(block.content) {
                        edit.move_entity(block.content, Some(target_page), At::End)?;
                    }
                    block.content
                } else if let Some(content) = content_copies.get(&block.content) {
                    *content
                } else {
                    let content = Self::add_clipboard_content(edit, target_page, block)?;
                    content_copies.insert(block.content, content);
                    content
                };

                if let Some(region) = &block.source_region {
                    let target = Self::transfer_clipboard_region(
                        edit,
                        &snapshot,
                        target_page,
                        region,
                        payload.mode,
                        &moved_regions,
                        &mut region_copies,
                        &mut relocated_regions,
                        offset,
                    )?;
                    if linked_contents.insert(content) {
                        if content != block.content {
                            edit.relate::<RecognizedFrom>(content, target)?;
                        } else if target != region.id
                            && let Some(relation) =
                                snapshot.relation_from::<RecognizedFrom>(content)?
                        {
                            edit.remove_relation(relation.id())?;
                            edit.relate::<RecognizedFrom>(content, target)?;
                        }
                    }
                }

                let layer = if payload.mode == TextTransferMode::Move {
                    if content != block.content {
                        let relation = snapshot
                            .relation_from::<Presents>(block.layer)?
                            .ok_or_else(|| {
                                koharu_scene::Error::Invalid(
                                    "text layer has no presentation relation".to_owned(),
                                )
                            })?;
                        edit.remove_relation(relation.id())?;
                        edit.relate::<Presents>(block.layer, content)?;
                    }
                    for relation in snapshot
                        .relations_from_as::<FlowsIn>(block.layer)
                        .map(|relation| relation.id())
                        .collect::<Vec<_>>()
                    {
                        edit.remove_relation(relation)?;
                    }
                    edit.move_entity(block.layer, Some(group), At::End)?;
                    if (offset.x != 0.0 || offset.y != 0.0)
                        && let Some(geometry) = &block.geometry
                    {
                        let geometry = Self::translated_geometry(geometry, offset);
                        edit.set(block.layer, &Self::scene_geometry(&geometry))?;
                    }
                    block.layer
                } else {
                    let layer = edit.add_text_layer(
                        target_page,
                        At::End,
                        content,
                        &SceneTextLayout {
                            origin: Origin::User,
                            kind: block.layout,
                            angle_degrees: block.angle_degrees,
                        },
                    )?;
                    Self::set_clipboard_layer_components(edit, layer, block, offset)?;
                    layer
                };

                if let Some(region) = &block.automatic_region {
                    let target = Self::transfer_clipboard_region(
                        edit,
                        &snapshot,
                        target_page,
                        region,
                        payload.mode,
                        &moved_regions,
                        &mut region_copies,
                        &mut relocated_regions,
                        offset,
                    )?;
                    if payload.mode == TextTransferMode::Move
                        && let Some(relation) = snapshot.relation_from::<FitsTo>(block.layer)?
                    {
                        if relation.value().target != target {
                            edit.remove_relation(relation.id())?;
                            edit.relate::<FitsTo>(layer, target)?;
                        }
                    } else if payload.mode == TextTransferMode::Copy {
                        edit.relate::<FitsTo>(layer, target)?;
                    }
                }
                transferred.push(layer);
            }
            Ok(())
        })?;
        let commit = self.commit(patch).await?;
        self.active_page = Some(target_page);
        Ok((commit, transferred))
    }

    pub(crate) async fn move_page(&mut self, page: EntityId, index: usize) -> Result<Commit> {
        let snapshot = self.snapshot();
        let siblings = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
        let at = Self::placement(&siblings, page, index);
        let patch = snapshot.patch(|edit| edit.move_entity(page, None, at))?;
        self.commit(patch).await
    }

    pub(crate) async fn add_point_text(
        &mut self,
        page: EntityId,
        point: Point,
    ) -> Result<(Commit, EntityId)> {
        if !point.x.is_finite() || !point.y.is_finite() {
            bail!("text position must contain finite coordinates");
        }
        let value = self.snapshot().page(page)?.page()?;
        if point.x < 0.0 || point.y < 0.0 || point.x >= value.width || point.y >= value.height {
            bail!("text position must be inside the page");
        }
        self.add_text(
            page,
            Frame {
                x: point.x as f32,
                y: point.y as f32,
                width: (value.width - point.x).clamp(1.0, 320.0) as f32,
                height: (value.height - point.y).clamp(1.0, 120.0) as f32,
                angle_degrees: 0.0,
            },
            TextLayoutKind::Point,
        )
        .await
    }

    pub(crate) async fn add_text_box(
        &mut self,
        page: EntityId,
        frame: Frame,
    ) -> Result<(Commit, EntityId)> {
        self.add_text(page, frame, TextLayoutKind::Paragraph).await
    }

    pub(crate) async fn add_ocr_region(
        &mut self,
        page: EntityId,
        frame: Frame,
        source_language: koharu_translator::Language,
    ) -> Result<(Commit, EntityId, EntityId)> {
        let snapshot = self.snapshot();
        let geometry = Self::geometry_from_frame(frame)?;
        let page_value = snapshot.page(page)?.page()?;
        if geometry.points.iter().any(|point| {
            point.x < 0.0
                || point.y < 0.0
                || point.x > page_value.width
                || point.y > page_value.height
        }) {
            bail!("OCR region must be inside the page");
        }
        let bubble = Self::containing_bubble(&snapshot, page, &geometry)?;
        let placement =
            koharu_pipeline::text_layer_placement(&snapshot, page, &geometry, source_language)?;
        let mut added = None;
        let patch = snapshot.patch(|edit| {
            let region = edit.add_analysis_region::<TextRegion>(
                page,
                At::End,
                &geometry,
                Some("text".to_owned()),
            )?;
            let content = edit.add_text_content(page, At::End)?;
            let layer = edit.add_text_layer(
                page,
                placement,
                content,
                &SceneTextLayout {
                    origin: Origin::User,
                    kind: TextLayoutKind::Paragraph,
                    angle_degrees: None,
                },
            )?;
            edit.set(
                layer,
                &SceneTypography {
                    origin: Origin::User,
                    preferred_font: None,
                    font_weight: None,
                    font_style: None,
                    size: None,
                    auto_fit: true,
                    color: None,
                    stroke_color: None,
                    stroke_width: None,
                    alignment: None,
                    writing_mode: None,
                    extensions: Default::default(),
                },
            )?;
            edit.set(
                content,
                &SceneTextRole {
                    origin: Origin::User,
                    role: if bubble.is_some() {
                        DIALOGUE_ROLE
                    } else {
                        FREE_TEXT_ROLE
                    }
                    .to_owned(),
                },
            )?;
            edit.relate::<RecognizedFrom>(content, region)?;
            if let Some(bubble) = bubble {
                edit.relate::<Inside>(region, bubble)?;
                edit.relate::<FlowsIn>(layer, bubble)?;
            } else {
                edit.relate::<FitsTo>(layer, region)?;
            }
            added = Some((layer, region));
            Ok(())
        })?;
        let commit = self.commit(patch).await?;
        let (layer, region) =
            added.expect("OCR layer and region were added while building the patch");
        Ok((commit, layer, region))
    }

    async fn add_text(
        &mut self,
        page: EntityId,
        frame: Frame,
        kind: TextLayoutKind,
    ) -> Result<(Commit, EntityId)> {
        let snapshot = self.snapshot();
        let geometry = Self::geometry_from_frame(frame)?;
        let mut layer = None;
        let patch = snapshot.patch(|edit| {
            let content = edit.add_text_content(page, At::End)?;
            edit.set(
                content,
                &SceneSourceText {
                    text: Authored::user(String::new()),
                    language: None,
                },
            )?;
            let added_layer = edit.add_text_layer(
                page,
                At::End,
                content,
                &SceneTextLayout {
                    origin: Origin::User,
                    kind,
                    angle_degrees: Some(frame.angle_degrees),
                },
            )?;
            layer = Some(added_layer);
            edit.set(added_layer, &geometry)?;
            edit.set(
                added_layer,
                &SceneTypography {
                    origin: Origin::User,
                    preferred_font: None,
                    font_weight: None,
                    font_style: None,
                    size: None,
                    auto_fit: true,
                    color: None,
                    stroke_color: None,
                    stroke_width: None,
                    alignment: match kind {
                        TextLayoutKind::Point => Some(koharu_scene::TextAlignment::Start),
                        TextLayoutKind::Paragraph => None,
                    },
                    writing_mode: None,
                    extensions: Default::default(),
                },
            )?;
            Ok(())
        })?;
        Ok((
            self.commit(patch).await?,
            layer.expect("text layer was added while building the patch"),
        ))
    }

    pub(crate) async fn set_source_text(
        &mut self,
        layer: EntityId,
        text: String,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let content = Self::text_content(&snapshot, layer)?;
        let language = snapshot
            .component::<SceneSourceText>(content)?
            .and_then(|source| source.language);
        let patch = snapshot.patch(|edit| {
            edit.promote_entity_to_user(layer)?;
            edit.promote_entity_to_user(content)?;
            edit.set(
                content,
                &SceneSourceText {
                    text: Authored::user(text),
                    language,
                },
            )
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn set_text_role(&mut self, layer: EntityId, role: String) -> Result<Commit> {
        if !matches!(
            role.as_str(),
            DIALOGUE_ROLE | FREE_TEXT_ROLE | ONOMATOPOEIA_ROLE
        ) {
            bail!("unsupported text role");
        }
        let snapshot = self.snapshot();
        let content = Self::text_content(&snapshot, layer)?;
        let patch = snapshot.patch(|edit| {
            edit.promote_entity_to_user(layer)?;
            edit.promote_entity_to_user(content)?;
            edit.set(
                content,
                &SceneTextRole {
                    origin: Origin::User,
                    role,
                },
            )
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn set_translation(
        &mut self,
        layer: EntityId,
        text: Option<String>,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let content = Self::text_content(&snapshot, layer)?;
        let language = snapshot
            .component::<SceneTranslation>(content)?
            .and_then(|translation| translation.language);
        let reviewed = snapshot
            .component::<SceneTranslationReview>(content)?
            .is_some();
        let patch = snapshot.patch(|edit| {
            edit.promote_entity_to_user(layer)?;
            edit.promote_entity_to_user(content)?;
            // Writing the translation by hand is the review's answer, including
            // keeping the reviewed text unchanged.
            if reviewed {
                edit.remove::<SceneTranslationReview>(content)?;
            }
            match text {
                Some(text) => edit.set(
                    content,
                    &SceneTranslation {
                        text: Authored::user(text),
                        language,
                    },
                ),
                None => edit.remove::<SceneTranslation>(content),
            }
        })?;
        self.commit(patch).await
    }

    pub(crate) fn characters(snapshot: &Snapshot) -> Result<Vec<Character>> {
        Ok(snapshot
            .project_component::<CharacterList>()?
            .map(|list| list.characters)
            .unwrap_or_default())
    }

    pub(crate) async fn set_speech_markup(
        &mut self,
        layer: EntityId,
        kind: SpeechKind,
        speaker: Option<CharacterId>,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let content = Self::text_content(&snapshot, layer)?;
        if let Some(speaker) = &speaker {
            let known = snapshot
                .project_component::<CharacterList>()?
                .is_some_and(|list| list.character(speaker).is_some());
            if !known {
                bail!("character {speaker} is not in the character list");
            }
        }
        let patch = snapshot.patch(|edit| {
            edit.promote_entity_to_user(layer)?;
            edit.promote_entity_to_user(content)?;
            edit.set(
                content,
                &SpeechMarkup {
                    origin: Origin::User,
                    kind,
                    speaker,
                },
            )
        })?;
        self.commit(patch).await
    }

    /// Save the character list as the user's. A character that is no longer
    /// listed is also cleared from every text that names it as the speaker,
    /// in the same patch, so the edit undoes as one step.
    pub(crate) async fn set_characters(
        &mut self,
        characters: Vec<CharacterInput>,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let characters = characters
            .into_iter()
            .map(|input| {
                let id = match input.id {
                    Some(id) => id,
                    None => CharacterId::new(format!("c-{}", uuid::Uuid::now_v7().simple()))?,
                };
                Ok(Character {
                    id,
                    name: input.name.trim().to_owned(),
                    gender: input.gender,
                    notes: input.notes.trim().to_owned(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let removed = Self::characters(&snapshot)?
            .into_iter()
            .map(|character| character.id)
            .filter(|id| characters.iter().all(|character| &character.id != id))
            .collect::<Vec<_>>();
        let mut cleared = Vec::new();
        if !removed.is_empty() {
            for layer in snapshot.text_layers()? {
                let content = layer.content()?.id();
                if let Some(markup) = snapshot.component::<SpeechMarkup>(content)?
                    && markup
                        .speaker
                        .as_ref()
                        .is_some_and(|speaker| removed.contains(speaker))
                {
                    cleared.push((content, markup));
                }
            }
        }
        let patch = snapshot.patch(|edit| {
            edit.set_project(&CharacterList {
                origin: Origin::User,
                characters,
            })?;
            for (content, markup) in cleared {
                edit.set(
                    content,
                    &SpeechMarkup {
                        speaker: None,
                        ..markup
                    },
                )?;
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    /// Add characters carried over from another project as the user's, so
    /// scene markup keeps them. A character whose id or name is already
    /// listed is skipped. Returns how many were added; the import is one
    /// undo step.
    pub(crate) async fn import_characters(
        &mut self,
        imported: Vec<Character>,
    ) -> Result<(Commit, usize)> {
        let snapshot = self.snapshot();
        let mut characters = Self::characters(&snapshot)?;
        let before = characters.len();
        for character in imported {
            if characters
                .iter()
                .all(|known| known.id != character.id && known.name.trim() != character.name.trim())
            {
                characters.push(character);
            }
        }
        let added = characters.len() - before;
        let patch = snapshot.patch(|edit| {
            if added > 0 {
                edit.set_project(&CharacterList {
                    origin: Origin::User,
                    characters,
                })?;
            }
            Ok(())
        })?;
        Ok((self.commit(patch).await?, added))
    }

    /// `laid_out_size` is the font size the presented page drew a layer at.
    /// A user edit pins detected typography as the user's, so an outline the
    /// edit leaves untouched is stored at the capped width the renderer drew
    /// rather than the raw measurement.
    pub(crate) async fn set_typography(
        &mut self,
        updates: Vec<TypographyUpdate>,
        laid_out_size: impl Fn(EntityId) -> Option<f32>,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let updates = updates
            .into_iter()
            .map(|mut update| {
                let content = Self::text_content(&snapshot, update.layer)?;
                if let Some(previous) = snapshot.component::<SceneTypography>(update.layer)?
                    && matches!(previous.origin, Origin::Generated(_))
                    && let Some(width) = previous.stroke_width
                    && update.typography.stroke_width == Some(width)
                    && let Some(size) = if update.typography.auto_fit {
                        laid_out_size(update.layer)
                    } else {
                        update.typography.size
                    }
                {
                    update.typography.stroke_width =
                        Some(koharu_renderer::measured_stroke_width(width, size));
                }
                Ok((update, content))
            })
            .collect::<Result<Vec<_>>>()?;
        let patch = snapshot.patch(|edit| {
            for (update, content) in updates {
                edit.promote_entity_to_user(update.layer)?;
                edit.promote_entity_to_user(content)?;
                edit.set(
                    update.layer,
                    &SceneTypography {
                        origin: Origin::User,
                        preferred_font: update.typography.preferred_font,
                        font_weight: update.typography.font_weight,
                        font_style: update.typography.font_style,
                        size: update.typography.size.filter(|size| *size > 0.0),
                        auto_fit: update.typography.auto_fit,
                        color: update.typography.color,
                        stroke_color: update.typography.stroke_color,
                        stroke_width: update.typography.stroke_width,
                        alignment: update.typography.alignment,
                        writing_mode: update.typography.writing_mode,
                        extensions: Default::default(),
                    },
                )?;
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn set_geometry(&mut self, updates: Vec<GeometryUpdate>) -> Result<Commit> {
        let snapshot = self.snapshot();
        let updates = updates
            .into_iter()
            .map(|update| {
                if snapshot.component::<SceneRegion>(update.layer)?.is_some() {
                    if update.points.is_none() {
                        bail!("analysis regions require explicit geometry");
                    }
                    return Ok((update, None));
                }
                let layout = snapshot
                    .component::<SceneTextLayout>(update.layer)?
                    .context("only text layers or analysis regions can change geometry")?;
                let content = Self::text_content(&snapshot, update.layer)?;
                if update.points.is_none()
                    && snapshot
                        .text_layer(update.layer)?
                        .automatic_target()?
                        .is_none()
                {
                    bail!("only automatically placed text can reset its geometry");
                }
                Ok((update, Some((content, layout))))
            })
            .collect::<Result<Vec<_>>>()?;
        let patch = snapshot.patch(|edit| {
            for (update, text) in updates {
                edit.promote_entity_to_user(update.layer)?;
                if let Some((content, mut layout)) = text {
                    edit.promote_entity_to_user(content)?;
                    layout.origin = Origin::User;
                    layout.angle_degrees = None;
                    edit.set(update.layer, &layout)?;
                }
                match update.points {
                    Some(points) => edit.set(
                        update.layer,
                        &SceneGeometry {
                            origin: Origin::User,
                            points: points
                                .into_iter()
                                .map(|point| ScenePoint {
                                    x: point.x,
                                    y: point.y,
                                })
                                .collect(),
                        },
                    )?,
                    None => edit.remove::<SceneGeometry>(update.layer)?,
                }
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn set_visibility(
        &mut self,
        layers: Vec<EntityId>,
        visible: Option<bool>,
        opacity: Option<f32>,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let patch = snapshot.patch(|edit| {
            for layer in layers {
                let mut value =
                    snapshot
                        .component::<SceneVisibility>(layer)?
                        .unwrap_or(SceneVisibility {
                            origin: Origin::User,
                            visible: true,
                            opacity: 1.0,
                        });
                if let Some(visible) = visible {
                    value.visible = visible;
                }
                if let Some(opacity) = opacity {
                    value.opacity = opacity;
                }
                value.origin = Origin::User;
                edit.set(layer, &value)?;
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn delete_layers(&mut self, layers: Vec<EntityId>) -> Result<Commit> {
        let snapshot = self.snapshot();
        let mut expanded = Vec::new();
        for layer in layers {
            if snapshot.component::<SceneTextGroup>(layer)?.is_some() {
                expanded.extend(snapshot.children(layer)?);
            } else {
                expanded.push(layer);
            }
        }
        let layers = Self::unique_roots(&snapshot, expanded)?;
        let mut deleted_entities = HashSet::new();
        for layer in &layers {
            deleted_entities.extend(snapshot.subtree(*layer)?.map(|entity| entity.id()));
        }
        let mut orphaned_contents = HashSet::new();
        for layer in &layers {
            let Some(relation) = snapshot.relation_from::<Presents>(*layer)? else {
                continue;
            };
            let content = relation.value().target;
            if snapshot
                .relations_to_as::<Presents>(content)
                .all(|relation| deleted_entities.contains(&relation.value().source))
            {
                orphaned_contents.insert(content);
            }
        }
        let mut orphaned_regions = HashSet::new();
        for content in &orphaned_contents {
            let Some(relation) = snapshot.relation_from::<RecognizedFrom>(*content)? else {
                continue;
            };
            let region = relation.value().target;
            let is_text_region = snapshot
                .component::<SceneRegion>(region)?
                .is_some_and(|value| value.kind == TextRegion::kind());
            let has_live_content = snapshot
                .relations_to_as::<RecognizedFrom>(region)
                .filter(|relation| !orphaned_contents.contains(&relation.value().source))
                .any(|relation| {
                    snapshot
                        .relations_to_as::<Presents>(relation.value().source)
                        .any(|presentation| {
                            !deleted_entities.contains(&presentation.value().source)
                        })
                });
            if is_text_region && !has_live_content {
                orphaned_regions.insert(region);
            }
        }
        let patch = snapshot.patch(|edit| {
            for layer in layers {
                if snapshot.page(layer).is_ok() {
                    return Err(koharu_scene::Error::Invalid(
                        "delete pages with delete_pages".to_owned(),
                    ));
                }
                edit.remove_entity(layer, RemovePolicy::Cascade)?;
            }
            for content in orphaned_contents {
                if snapshot.entity(content).is_ok() {
                    edit.remove_entity(content, RemovePolicy::Cascade)?;
                }
            }
            for region in orphaned_regions {
                if snapshot.entity(region).is_ok() {
                    edit.remove_entity(region, RemovePolicy::Cascade)?;
                }
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    #[tracing::instrument(level = "info", skip_all, fields(layer = %layer, parent = %parent))]
    pub(crate) async fn move_layer(
        &mut self,
        layer: EntityId,
        parent: EntityId,
        index: usize,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let siblings = snapshot
            .children(parent)?
            .map(|candidate| Ok(Self::is_layer(&snapshot, candidate)?.then_some(candidate)))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let at = Self::placement(&siblings, layer, index);
        let patch = snapshot.patch(|edit| {
            edit.promote_entity_to_user(layer)?;
            if snapshot.component::<SceneTextGroup>(parent)?.is_some() {
                edit.promote_entity_to_user(parent)?;
            }
            edit.move_entity(layer, Some(parent), at)
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn reorder_layers(
        &mut self,
        parent: EntityId,
        order: Vec<EntityId>,
    ) -> Result<Commit> {
        if order.is_empty() {
            bail!("layer order must not be empty");
        }
        let snapshot = self.snapshot();
        let siblings = snapshot
            .children(parent)?
            .map(|candidate| Ok(Self::is_layer(&snapshot, candidate)?.then_some(candidate)))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let moving = order.iter().copied().collect::<HashSet<_>>();
        if moving.len() != order.len() || !moving.iter().all(|layer| siblings.contains(layer)) {
            bail!("layer order must contain unique children of the requested parent");
        }
        let mut replacements = order.iter().copied();
        let desired = siblings
            .iter()
            .map(|layer| {
                if moving.contains(layer) {
                    replacements
                        .next()
                        .expect("the replacement order has the same number of layers")
                } else {
                    *layer
                }
            })
            .collect::<Vec<_>>();
        let patch = snapshot.patch(|edit| {
            let mut current = siblings.clone();
            for (index, layer) in desired.iter().copied().enumerate() {
                if current[index] == layer {
                    continue;
                }
                let anchor = current[index];
                edit.promote_entity_to_user(layer)?;
                edit.move_entity(layer, Some(parent), At::Before(anchor))?;
                let source = current
                    .iter()
                    .position(|candidate| *candidate == layer)
                    .expect("the desired order contains the same sibling layers");
                current.remove(source);
                current.insert(index, layer);
            }
            if snapshot.component::<SceneTextGroup>(parent)?.is_some() {
                edit.promote_entity_to_user(parent)?;
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn set_geometries(
        &mut self,
        geometries: impl IntoIterator<Item = (EntityId, SceneGeometry, f32)>,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let geometries = geometries
            .into_iter()
            .map(|(element, geometry, angle_degrees)| {
                let mut layout = snapshot
                    .component::<SceneTextLayout>(element)?
                    .context("only text layers can change geometry")?;
                layout.origin = Origin::User;
                layout.angle_degrees = Some(angle_degrees);
                let content = Self::text_content(&snapshot, element)?;
                Ok((element, geometry, content, layout))
            })
            .collect::<Result<Vec<_>>>()?;
        let patch = snapshot.patch(|edit| {
            for (element, mut geometry, content, layout) in geometries {
                edit.promote_entity_to_user(element)?;
                edit.promote_entity_to_user(content)?;
                geometry.origin = Origin::User;
                edit.set(element, &geometry)?;
                edit.set(element, &layout)?;
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn apply_raster_stroke(
        &mut self,
        page: EntityId,
        layer: Option<EntityId>,
        mode: RasterStrokeMode,
        color: [u8; 4],
        diameter: f32,
        points: Vec<ScenePoint>,
    ) -> Result<(Commit, EntityId)> {
        if !diameter.is_finite() || diameter <= 0.0 || points.is_empty() {
            bail!("a raster stroke requires a positive diameter and at least one point");
        }
        if points
            .iter()
            .any(|point| !point.x.is_finite() || !point.y.is_finite())
        {
            bail!("raster stroke points must be finite");
        }
        let snapshot = self.snapshot();
        let page_value = snapshot.page(page)?.page()?;
        let width = page_value.width.round() as u32;
        let height = page_value.height.round() as u32;
        if width == 0 || height == 0 {
            bail!("page dimensions must be positive");
        }
        let layer = match layer {
            Some(layer) => Some(layer),
            None if mode == RasterStrokeMode::Erase => snapshot.children(page)?.find(|entity| {
                snapshot
                    .component::<SceneRasterLayer>(*entity)
                    .ok()
                    .flatten()
                    .is_some_and(|layer| layer.kind == RasterLayerKind::Cleanup)
                    && snapshot
                        .component::<SceneVisibility>(*entity)
                        .ok()
                        .flatten()
                        .is_none_or(|visibility| visibility.visible)
            }),
            None => None,
        };
        let mut raster_layer = None;
        let mut promote_layer = false;
        let mut image = if let Some(layer) = layer {
            if snapshot.parent(layer)? != Some(page) {
                bail!("the raster target must be a layer on the active page");
            }
            let target = snapshot
                .component::<SceneRasterLayer>(layer)?
                .context("the raster target must be a layer on the active page")?;
            promote_layer = target.origin != Origin::User
                || snapshot
                    .component::<EntityOrigin>(layer)?
                    .is_some_and(|origin| origin.origin != Origin::User);
            raster_layer = Some(target);
            match snapshot.asset(layer, &AssetRole::new("source")?)? {
                Some(asset) => {
                    let bytes = snapshot.read_blob(asset.blob).await?;
                    image::load_from_memory(&bytes)?.to_rgba8()
                }
                None => RgbaImage::new(width, height),
            }
        } else {
            if mode == RasterStrokeMode::Erase {
                bail!("eraser requires a raster layer target");
            }
            RgbaImage::new(width, height)
        };
        if image.dimensions() != (width, height) {
            bail!("raster layer dimensions must match the page");
        }
        rasterize_stroke(&mut image, mode, color, diameter, &points);
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(image).write_to(&mut bytes, ImageFormat::Png)?;
        let source = AssetRole::new("source")?;
        let name = format!(
            "Paint {}",
            snapshot
                .children(page)?
                .filter(|entity| {
                    snapshot
                        .component::<SceneRasterLayer>(*entity)
                        .ok()
                        .flatten()
                        .is_some_and(|layer| layer.kind == RasterLayerKind::Paint)
                })
                .count()
                + 1
        );
        let mut committed_layer = layer;
        let patch = snapshot.patch(|edit| {
            let layer = if let Some(layer) = committed_layer {
                if promote_layer {
                    edit.promote_entity_to_user(layer)?;
                    let raster = raster_layer
                        .as_mut()
                        .expect("an existing raster target has a layer component");
                    raster.origin = Origin::User;
                    edit.set(layer, raster)?;
                }
                layer
            } else {
                let at = snapshot
                    .page(page)?
                    .text_group()?
                    .map_or(At::End, |group| At::Before(group.id()));
                let layer = edit.add_entity(page, at)?;
                edit.set(
                    layer,
                    &SceneRasterLayer {
                        origin: Origin::User,
                        name,
                        kind: RasterLayerKind::Paint,
                    },
                )?;
                committed_layer = Some(layer);
                layer
            };
            edit.set_asset(
                layer,
                &source,
                AssetInput::new(
                    bytes.into_inner(),
                    "image/png",
                    AssetMetadata {
                        width: Some(width),
                        height: Some(height),
                        attributes: Default::default(),
                    },
                ),
            )
        })?;
        Ok((
            self.commit(patch).await?,
            committed_layer.expect("raster layer was selected or added while building the patch"),
        ))
    }

    pub(crate) async fn import_cleanup_image(
        &mut self,
        page: EntityId,
        data_url: String,
    ) -> Result<(Commit, EntityId)> {
        let (header, encoded) = data_url
            .split_once(',')
            .context("cleanup image must be a data URL")?;
        if !header.starts_with("data:image/") || !header.ends_with(";base64") {
            bail!("cleanup image must be a base64 image data URL");
        }
        let bytes = STANDARD
            .decode(encoded)
            .context("cleanup image contains invalid base64")?;
        let snapshot = self.snapshot();
        let page_value = snapshot.page(page)?.page()?;
        let width = page_value.width.round() as u32;
        let height = page_value.height.round() as u32;
        if width == 0 || height == 0 {
            bail!("page dimensions must be positive");
        }
        let decoded = image::load_from_memory(&bytes).context("clipboard image is invalid")?;
        let (source_width, source_height) = (decoded.width(), decoded.height());
        if source_width == 0 || source_height == 0 {
            bail!("clipboard image dimensions must be positive");
        }
        let source_ratio = source_width as f64 / source_height as f64;
        let page_ratio = width as f64 / height as f64;
        if ((source_ratio - page_ratio) / page_ratio).abs() > 0.01 {
            bail!("clipboard image aspect ratio does not match the page");
        }
        let image = if (source_width, source_height) == (width, height) {
            decoded.to_rgba8()
        } else {
            image::imageops::resize(
                &decoded.to_rgba8(),
                width,
                height,
                image::imageops::FilterType::Lanczos3,
            )
        };
        let mut png = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(image).write_to(&mut png, ImageFormat::Png)?;
        let cleanup = snapshot
            .children(page)?
            .filter(|entity| {
                snapshot
                    .component::<SceneRasterLayer>(*entity)
                    .ok()
                    .flatten()
                    .is_some_and(|layer| layer.kind == RasterLayerKind::Cleanup)
            })
            .collect::<Vec<_>>();
        if cleanup.len() > 1 {
            bail!("page contains multiple cleanup layers");
        }
        let mut committed_layer = cleanup.first().copied();
        let source = AssetRole::new("source")?;
        let patch = snapshot.patch(|edit| {
            let layer = if let Some(layer) = committed_layer {
                edit.promote_entity_to_user(layer)?;
                edit.set(
                    layer,
                    &SceneRasterLayer {
                        origin: Origin::User,
                        name: "Cleanup".to_owned(),
                        kind: RasterLayerKind::Cleanup,
                    },
                )?;
                layer
            } else {
                let layer = edit.add_entity(page, At::Start)?;
                edit.set(
                    layer,
                    &SceneRasterLayer {
                        origin: Origin::User,
                        name: "Cleanup".to_owned(),
                        kind: RasterLayerKind::Cleanup,
                    },
                )?;
                edit.set(
                    layer,
                    &SceneVisibility {
                        origin: Origin::User,
                        visible: true,
                        opacity: 1.0,
                    },
                )?;
                committed_layer = Some(layer);
                layer
            };
            edit.set_asset(
                layer,
                &source,
                AssetInput::new(
                    png.into_inner(),
                    "image/png",
                    AssetMetadata {
                        width: Some(width),
                        height: Some(height),
                        attributes: Default::default(),
                    },
                ),
            )
        })?;
        Ok((
            self.commit(patch).await?,
            committed_layer.expect("cleanup layer was selected or created while building patch"),
        ))
    }

    pub(crate) async fn undo(&mut self) -> Result<Commit> {
        let revisions = self.undo.pop().ok_or_else(|| anyhow!("nothing to undo"))?;
        let commit = match self.session.undo_many(revisions.iter().copied()).await {
            Ok(commit) => commit,
            Err(error) => {
                self.undo.push(revisions);
                return Err(error.into());
            }
        };
        self.redo.push(vec![commit.revision]);
        Ok(commit)
    }

    pub(crate) async fn redo(&mut self) -> Result<Commit> {
        let revisions = self.redo.pop().ok_or_else(|| anyhow!("nothing to redo"))?;
        let commit = match self.session.undo_many(revisions.iter().copied()).await {
            Ok(commit) => commit,
            Err(error) => {
                self.redo.push(revisions);
                return Err(error.into());
            }
        };
        self.undo.push(vec![commit.revision]);
        Ok(commit)
    }

    pub(crate) fn record(&mut self, revisions: Vec<Revision>) {
        if !revisions.is_empty() {
            self.undo.push(revisions);
            self.redo.clear();
        }
    }

    pub(crate) fn record_commit(&mut self, commit: &Commit) {
        if commit.changes.to != commit.changes.from {
            self.record(vec![commit.revision]);
        }
    }

    pub(crate) async fn commit_rebased(
        &mut self,
        patch: koharu_scene::Patch,
    ) -> Result<Option<Commit>> {
        let current = self.snapshot();
        let patch = match patch.rebase_on(&current) {
            Ok(patch) => patch,
            Err(
                error @ (koharu_scene::Error::PatchConflict(_)
                | koharu_scene::Error::EntityNotFound(_)
                | koharu_scene::Error::RelationNotFound(_)),
            ) => {
                tracing::debug!(%error, "pipeline output was superseded by a document edit");
                return Ok(None);
            }
            Err(error) => return Err(error.into()),
        };
        if patch.is_empty() {
            return Ok(None);
        }
        Ok(Some(self.commit(patch).await?))
    }

    async fn commit(&mut self, patch: koharu_scene::Patch) -> Result<Commit> {
        Ok(self.session.commit(patch).await?)
    }

    pub(crate) fn page(snapshot: &Snapshot, page: EntityId) -> Result<Page> {
        let value = snapshot.page(page)?.page()?;
        let source = Self::asset_id(snapshot, page, "source")?;
        let mut layers = Vec::new();
        Self::collect_layer_views(snapshot, page, &mut layers)?;
        if let Some(image) = source.clone() {
            layers.insert(
                0,
                Layer::Artwork {
                    id: page,
                    parent: None,
                    geometry: Geometry {
                        points: vec![
                            Point { x: 0.0, y: 0.0 },
                            Point {
                                x: value.width,
                                y: 0.0,
                            },
                            Point {
                                x: value.width,
                                y: value.height,
                            },
                            Point {
                                x: 0.0,
                                y: value.height,
                            },
                        ],
                    },
                    visibility: LayerVisibility {
                        visible: true,
                        opacity: 1.0,
                    },
                    image,
                },
            );
        }
        let regions = snapshot
            .descendants(page)?
            .map(|entity| {
                let id = entity.id();
                snapshot
                    .component::<SceneRegion>(id)?
                    .map(|region| Self::analysis_region(snapshot, id, region))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        Ok(Page {
            id: page,
            label: value.label,
            size: PageSize {
                width: value.width,
                height: value.height,
            },
            layers,
            regions,
        })
    }

    fn layer_view(snapshot: &Snapshot, layer: EntityId) -> Result<Layer> {
        let parent = snapshot.parent(layer)?;
        let visibility = Self::layer_visibility(snapshot, layer)?;
        if let Some(group) = snapshot.component::<SceneGroup>(layer)? {
            return Ok(Layer::Group {
                id: layer,
                parent,
                visibility,
                name: group.name,
                role: snapshot
                    .component::<SceneTextGroup>(layer)?
                    .map(|_| GroupRole::Text),
            });
        }
        if let Some(layout) = snapshot.component::<SceneTextLayout>(layer)? {
            let text_layer = snapshot.text_layer(layer)?;
            let content = text_layer.content()?;
            let ocr_unverified = koharu_pipeline::ocr_unverified(content)?;
            let source = content.source()?.map(|source| SourceText {
                text: source.text.value,
                language: source.language.map(|language| language.to_string()),
            });
            let translation = content.translation()?.map(|translation| Translation {
                generated: matches!(translation.text.origin, koharu_scene::Origin::Generated(_)),
                text: translation.text.value,
                language: translation.language.map(|language| language.to_string()),
            });
            let translation_review =
                content
                    .translation_review()?
                    .map(|review| TranslationReview {
                        reason: review.reason,
                        reference: review.reference,
                    });
            let characters = snapshot.project_component::<CharacterList>()?;
            let markup = snapshot
                .component::<SpeechMarkup>(content.id())?
                .map(|markup| TextMarkup {
                    kind: markup.kind,
                    speaker: markup.speaker.filter(|speaker| {
                        characters
                            .as_ref()
                            .is_some_and(|list| list.character(speaker).is_some())
                    }),
                    user: markup.origin == Origin::User,
                });
            let role = content.role()?.map(|role| role.role);
            let source_region = content.source_region()?.map(|region| region.id());
            let automatic_region = text_layer.automatic_target()?.map(|region| region.id());
            let typography = text_layer.typography()?.map(Self::typography_view);
            return Ok(Layer::Text {
                id: layer,
                parent,
                geometry: snapshot
                    .component::<SceneGeometry>(layer)?
                    .map(Self::geometry_view),
                angle_degrees: layout.angle_degrees,
                visibility,
                content: Box::new(TextContent {
                    id: content.id(),
                    source,
                    translation,
                    translation_review,
                    ocr_unverified,
                    markup,
                    role,
                    source_region,
                }),
                typography,
                layout: layout.kind,
                automatic_region,
            });
        }
        if let Some(raster) = snapshot.component::<SceneRasterLayer>(layer)? {
            return Ok(Layer::Raster {
                id: layer,
                parent,
                visibility,
                image: Self::asset_id(snapshot, layer, "source")?,
                name: raster.name,
                kind: raster.kind,
            });
        }
        let geometry = snapshot
            .component::<SceneGeometry>(layer)?
            .map(Self::geometry_view)
            .context("image layer has no geometry")?;
        let image = Self::asset_id(snapshot, layer, "source")?
            .context("image layer has no source asset")?;
        Ok(Layer::Image {
            id: layer,
            parent,
            geometry,
            visibility,
            image,
        })
    }

    fn analysis_region(
        snapshot: &Snapshot,
        id: EntityId,
        region: SceneRegion,
    ) -> Result<AnalysisRegion> {
        let geometry = snapshot
            .component::<SceneGeometry>(id)?
            .map(Self::geometry_view)
            .context("analysis region has no geometry")?;
        Ok(AnalysisRegion {
            id,
            parent: snapshot.parent(id)?,
            geometry,
            kind: region.kind.as_str().to_owned(),
            label: region.label,
        })
    }

    fn layer_visibility(snapshot: &Snapshot, layer: EntityId) -> Result<LayerVisibility> {
        Ok(snapshot.component::<SceneVisibility>(layer)?.map_or(
            LayerVisibility {
                visible: true,
                opacity: 1.0,
            },
            |visibility| LayerVisibility {
                visible: visibility.visible,
                opacity: visibility.opacity,
            },
        ))
    }

    fn typography_view(typography: SceneTypography) -> Typography {
        let writing_mode = if matches!(&typography.origin, Origin::User) {
            typography.writing_mode
        } else {
            None
        };
        Typography {
            preferred_font: typography.preferred_font,
            font_weight: typography.font_weight,
            font_style: typography.font_style,
            size: typography.size,
            auto_fit: typography.auto_fit,
            color: typography.color,
            stroke_color: typography.stroke_color,
            stroke_width: typography.stroke_width,
            alignment: typography.alignment,
            writing_mode,
        }
    }

    fn geometry_view(geometry: SceneGeometry) -> Geometry {
        Geometry {
            points: geometry
                .points
                .into_iter()
                .map(|point| Point {
                    x: point.x,
                    y: point.y,
                })
                .collect(),
        }
    }

    fn asset_id(snapshot: &Snapshot, entity: EntityId, role: &str) -> Result<Option<String>> {
        Ok(snapshot
            .asset(entity, &AssetRole::new(role)?)?
            .map(|asset| asset.blob.to_string()))
    }

    fn is_layer(snapshot: &Snapshot, entity: EntityId) -> Result<bool> {
        Ok(snapshot.component::<SceneGroup>(entity)?.is_some()
            || snapshot.component::<SceneTextLayout>(entity)?.is_some()
            || snapshot.component::<SceneRasterLayer>(entity)?.is_some()
            || (snapshot.component::<SceneGeometry>(entity)?.is_some()
                && snapshot
                    .asset(entity, &AssetRole::new("source")?)?
                    .is_some()))
    }

    fn is_content_layer(snapshot: &Snapshot, entity: EntityId) -> Result<bool> {
        Ok(
            snapshot.component::<SceneGroup>(entity)?.is_none()
                && Self::is_layer(snapshot, entity)?,
        )
    }

    fn collect_layer_views(
        snapshot: &Snapshot,
        parent: EntityId,
        layers: &mut Vec<Layer>,
    ) -> Result<()> {
        for child in snapshot.children(parent)? {
            if !Self::is_layer(snapshot, child)? {
                continue;
            }
            let group = snapshot.component::<SceneGroup>(child)?.is_some();
            layers.push(Self::layer_view(snapshot, child)?);
            if group {
                Self::collect_layer_views(snapshot, child, layers)?;
            }
        }
        Ok(())
    }

    fn text_content(snapshot: &Snapshot, layer: EntityId) -> Result<EntityId> {
        snapshot
            .text_layer(layer)?
            .content()
            .map(|content| content.id())
            .map_err(Into::into)
    }

    fn placement(siblings: &[EntityId], moving: EntityId, index: usize) -> At {
        siblings
            .iter()
            .copied()
            .filter(|entity| *entity != moving)
            .nth(index)
            .map_or(At::End, At::Before)
    }

    fn unique_roots(snapshot: &Snapshot, entities: Vec<EntityId>) -> Result<Vec<EntityId>> {
        let selected = entities.into_iter().collect::<HashSet<_>>();
        let mut roots = Vec::new();
        for entity in selected.iter().copied() {
            let mut parent = snapshot.parent(entity)?;
            let mut nested = false;
            while let Some(value) = parent {
                if selected.contains(&value) {
                    nested = true;
                    break;
                }
                parent = snapshot.parent(value)?;
            }
            if !nested {
                roots.push(entity);
            }
        }
        roots.sort_unstable();
        Ok(roots)
    }

    fn geometry_from_frame(frame: Frame) -> Result<SceneGeometry> {
        if !frame.x.is_finite()
            || !frame.y.is_finite()
            || !frame.width.is_finite()
            || !frame.height.is_finite()
            || !frame.angle_degrees.is_finite()
            || frame.width <= 0.0
            || frame.height <= 0.0
        {
            bail!("frame must contain finite coordinates and positive dimensions");
        }
        let center_x = f64::from(frame.x + frame.width * 0.5);
        let center_y = f64::from(frame.y + frame.height * 0.5);
        let half_width = f64::from(frame.width) * 0.5;
        let half_height = f64::from(frame.height) * 0.5;
        let (sin, cos) = f64::from(frame.angle_degrees).to_radians().sin_cos();
        Ok(SceneGeometry {
            origin: Origin::User,
            points: [
                (-half_width, -half_height),
                (half_width, -half_height),
                (half_width, half_height),
                (-half_width, half_height),
            ]
            .map(|(x, y)| ScenePoint {
                x: center_x + x * cos - y * sin,
                y: center_y + x * sin + y * cos,
            })
            .into(),
        })
    }

    fn containing_bubble(
        snapshot: &Snapshot,
        page: EntityId,
        geometry: &SceneGeometry,
    ) -> Result<Option<EntityId>> {
        let mut smallest = None;
        for entity in snapshot.descendants(page)? {
            let id = entity.id();
            let is_bubble = snapshot
                .component::<SceneRegion>(id)?
                .is_some_and(|region| region.kind == BubbleRegion::kind());
            if !is_bubble {
                continue;
            }
            let Some(candidate) = snapshot.component::<SceneGeometry>(id)? else {
                continue;
            };
            if !geometry_contains(&candidate, geometry) {
                continue;
            }
            let area = polygon_area(&candidate);
            if smallest.is_none_or(|(_, smallest_area)| area < smallest_area) {
                smallest = Some((id, area));
            }
        }
        Ok(smallest.map(|(id, _)| id))
    }
}

fn geometry_contains(parent: &SceneGeometry, child: &SceneGeometry) -> bool {
    child
        .points
        .iter()
        .all(|point| point_in_geometry(point, parent))
}

fn polygon_area(geometry: &SceneGeometry) -> f64 {
    if geometry.points.len() < 3 {
        return 0.0;
    }
    geometry
        .points
        .iter()
        .zip(geometry.points.iter().cycle().skip(1))
        .map(|(left, right)| left.x * right.y - right.x * left.y)
        .sum::<f64>()
        .abs()
        * 0.5
}

fn point_in_geometry(point: &ScenePoint, geometry: &SceneGeometry) -> bool {
    if geometry.points.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut previous = geometry.points.last().expect("geometry is non-empty");
    for current in &geometry.points {
        let cross = (point.x - current.x) * (previous.y - current.y)
            - (point.y - current.y) * (previous.x - current.x);
        let on_segment = cross.abs() <= 1e-6
            && point.x >= current.x.min(previous.x) - 1e-6
            && point.x <= current.x.max(previous.x) + 1e-6
            && point.y >= current.y.min(previous.y) - 1e-6
            && point.y <= current.y.max(previous.y) + 1e-6;
        if on_segment {
            return true;
        }
        if (current.y > point.y) != (previous.y > point.y)
            && point.x
                < (previous.x - current.x) * (point.y - current.y) / (previous.y - current.y)
                    + current.x
        {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

fn validate_project_name(name: &str) -> Result<String> {
    let name = name.trim();
    if name.is_empty() {
        bail!("project name cannot be empty");
    }
    if name.ends_with(['.', ' '])
        || name
            .chars()
            .any(|character| character.is_control() || r#"<>:"/\|?*"#.contains(character))
    {
        bail!("project name contains characters that cannot be used in a file name");
    }
    let stem = name.split('.').next().unwrap_or(name).to_ascii_uppercase();
    if matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || stem
            .strip_prefix("COM")
            .or_else(|| stem.strip_prefix("LPT"))
            .is_some_and(|number| {
                matches!(number, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
            })
    {
        bail!("project name is reserved by Windows");
    }
    Ok(name.to_owned())
}

fn rasterize_stroke(
    image: &mut RgbaImage,
    mode: RasterStrokeMode,
    color: [u8; 4],
    diameter: f32,
    points: &[ScenePoint],
) {
    let radius = f64::from(diameter) * 0.5;
    for (start, end) in points
        .iter()
        .zip(points.iter().skip(1))
        .chain(points.last().map(|point| (point, point)))
    {
        let left = (start.x.min(end.x) - radius - 0.5).floor().max(0.0) as u32;
        let top = (start.y.min(end.y) - radius - 0.5).floor().max(0.0) as u32;
        let right = (start.x.max(end.x) + radius + 0.5)
            .ceil()
            .min(f64::from(image.width())) as u32;
        let bottom = (start.y.max(end.y) + radius + 0.5)
            .ceil()
            .min(f64::from(image.height())) as u32;
        let dx = end.x - start.x;
        let dy = end.y - start.y;
        let length_squared = dx * dx + dy * dy;
        for y in top..bottom {
            for x in left..right {
                let px = f64::from(x) + 0.5;
                let py = f64::from(y) + 0.5;
                let t = if length_squared == 0.0 {
                    0.0
                } else {
                    (((px - start.x) * dx + (py - start.y) * dy) / length_squared).clamp(0.0, 1.0)
                };
                let distance =
                    ((px - (start.x + t * dx)).powi(2) + (py - (start.y + t * dy)).powi(2)).sqrt();
                let coverage = (radius + 0.5 - distance).clamp(0.0, 1.0) as f32;
                if coverage == 0.0 {
                    continue;
                }
                let pixel = image.get_pixel_mut(x, y);
                match mode {
                    RasterStrokeMode::Paint => {
                        let source_alpha = f32::from(color[3]) / 255.0 * coverage;
                        let destination_alpha = f32::from(pixel[3]) / 255.0;
                        let output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha);
                        if output_alpha > 0.0 {
                            for channel in 0..3 {
                                let source = f32::from(color[channel]) / 255.0;
                                let destination = f32::from(pixel[channel]) / 255.0;
                                pixel[channel] = (((source * source_alpha
                                    + destination * destination_alpha * (1.0 - source_alpha))
                                    / output_alpha)
                                    * 255.0)
                                    .round() as u8;
                            }
                        }
                        pixel[3] = (output_alpha * 255.0).round() as u8;
                    }
                    RasterStrokeMode::Erase => {
                        pixel[3] = (f32::from(pixel[3]) * (1.0 - coverage)).round() as u8;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use koharu_scene::{
        BubbleRegion, DetectionAnalysis, FitsTo, FlowsIn, Generation, Inside, ProducerId,
        RecognizedFrom, TextRegion, TextRole, WritingMode,
    };

    use super::*;

    #[test]
    fn only_user_authored_direction_is_projected_as_an_override() {
        let mut typography = SceneTypography {
            origin: Origin::Generated(Generation::new(
                ProducerId::new("dev.koharu.pipeline.detection").expect("valid producer"),
            )),
            preferred_font: None,
            font_weight: None,
            font_style: None,
            size: None,
            auto_fit: true,
            color: None,
            stroke_color: None,
            stroke_width: None,
            alignment: None,
            writing_mode: Some(WritingMode::Vertical),
            extensions: Default::default(),
        };

        assert_eq!(
            Project::typography_view(typography.clone()).writing_mode,
            None
        );
        typography.origin = Origin::User;
        assert_eq!(
            Project::typography_view(typography).writing_mode,
            Some(WritingMode::Vertical)
        );
    }

    #[tokio::test]
    async fn removing_a_character_clears_it_as_speaker_in_the_same_edit() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        let content = setup.add_text_content(page, At::End).unwrap();
        let layer = setup
            .add_text_layer(
                page,
                At::End,
                content,
                &SceneTextLayout {
                    origin: Origin::User,
                    kind: TextLayoutKind::Paragraph,
                    angle_degrees: None,
                },
            )
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());
        let input = |id: Option<CharacterId>, name: &str| CharacterInput {
            id,
            name: name.to_owned(),
            gender: Gender::Male,
            notes: String::new(),
        };

        project
            .set_characters(vec![input(None, "Izuku"), input(None, "Katsuki")])
            .await
            .unwrap();
        let characters = Project::characters(&project.snapshot()).unwrap();
        let (izuku, katsuki) = (characters[0].id.clone(), characters[1].id.clone());
        assert!(
            project
                .set_speech_markup(
                    layer,
                    SpeechKind::Speech,
                    Some(CharacterId::new("nobody").unwrap())
                )
                .await
                .is_err()
        );
        project
            .set_speech_markup(layer, SpeechKind::Thought, Some(izuku))
            .await
            .unwrap();

        project
            .set_characters(vec![input(Some(katsuki.clone()), "Katsuki")])
            .await
            .unwrap();

        let snapshot = project.snapshot();
        let markup = snapshot
            .component::<SpeechMarkup>(content)
            .unwrap()
            .unwrap();
        assert_eq!(markup.kind, SpeechKind::Thought);
        assert_eq!(markup.speaker, None);
        let list = snapshot
            .project_component::<CharacterList>()
            .unwrap()
            .unwrap();
        assert_eq!(list.origin, Origin::User);
        assert_eq!(
            list.characters
                .iter()
                .map(|character| &character.id)
                .collect::<Vec<_>>(),
            vec![&katsuki]
        );
    }

    #[tokio::test]
    async fn frontend_clipboard_snapshot_round_trips_without_becoming_stale() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let source_page = setup
            .add_page(PageDraft::new("source", 100.0, 100.0), At::End)
            .unwrap();
        let target_page = setup
            .add_page(PageDraft::new("target", 100.0, 100.0), At::End)
            .unwrap();
        let region = setup
            .add_analysis_region::<TextRegion>(
                source_page,
                At::End,
                &SceneGeometry::rectangle(10.0, 20.0, 30.0, 40.0),
                Some("region".to_owned()),
            )
            .unwrap();
        let content = setup.add_text_content(source_page, At::End).unwrap();
        setup.relate::<RecognizedFrom>(content, region).unwrap();
        let layer = setup
            .add_text_layer(
                source_page,
                At::End,
                content,
                &SceneTextLayout {
                    origin: Origin::User,
                    kind: TextLayoutKind::Paragraph,
                    angle_degrees: None,
                },
            )
            .unwrap();
        setup.relate::<FitsTo>(layer, region).unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let page = Project::page(&project.snapshot(), source_page).unwrap();
        let Layer::Text {
            id,
            geometry,
            angle_degrees,
            visibility,
            content,
            typography,
            layout,
            automatic_region,
            ..
        } = page
            .layers
            .into_iter()
            .find(|candidate| match candidate {
                Layer::Text { id, .. } => *id == layer,
                _ => false,
            })
            .unwrap()
        else {
            panic!("clipboard source is a text layer");
        };
        let region = |id: Option<EntityId>| {
            id.and_then(|id| page.regions.iter().find(|region| region.id == id))
                .map(|region| TextClipboardRegion {
                    id: region.id,
                    geometry: region.geometry.clone(),
                    label: region.label.clone(),
                })
        };
        let payload = TextClipboardPayload {
            version: 1,
            mode: TextTransferMode::Copy,
            source_page,
            layers: vec![TextClipboardLayer {
                layer: id,
                content: content.id,
                source: content.source,
                translation: content.translation,
                role: content.role,
                geometry,
                angle_degrees,
                visibility,
                typography,
                layout,
                source_region: region(content.source_region),
                automatic_region: region(automatic_region),
            }],
        };
        let mut payload: TextClipboardPayload =
            serde_json::from_str(&serde_json::to_string(&payload).unwrap()).unwrap();
        // Browser number formatting can move a scene coordinate by a tiny
        // amount when a Rust float crosses the JSON/JavaScript boundary.
        payload.layers[0]
            .source_region
            .as_mut()
            .unwrap()
            .geometry
            .points[2]
            .y += 1e-13;

        let (_, transferred) = project
            .paste_text_blocks(target_page, payload, Some(Point { x: 80.0, y: 70.0 }))
            .await
            .expect("a payload built from the frontend page view must not be stale");
        let target = Project::page(&project.snapshot(), target_page).unwrap();
        let Layer::Text { content, .. } = target
            .layers
            .into_iter()
            .find(|layer| matches!(layer, Layer::Text { id, .. } if *id == transferred[0]))
            .unwrap()
        else {
            panic!("pasted layer is text");
        };
        let region = target
            .regions
            .into_iter()
            .find(|region| Some(region.id) == content.source_region)
            .unwrap();
        let min_x = region
            .geometry
            .points
            .iter()
            .map(|point| point.x)
            .fold(f64::INFINITY, f64::min);
        let max_x = region
            .geometry
            .points
            .iter()
            .map(|point| point.x)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_y = region
            .geometry
            .points
            .iter()
            .map(|point| point.y)
            .fold(f64::INFINITY, f64::min);
        let max_y = region
            .geometry
            .points
            .iter()
            .map(|point| point.y)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((min_x + max_x).mul_add(0.5, -80.0).abs() < 1e-9);
        assert!((min_y + max_y).mul_add(0.5, -70.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn text_clipboard_same_page_copy_and_move_center_at_the_cursor() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        let content = setup.add_text_content(page, At::End).unwrap();
        let layer = setup
            .add_text_layer(
                page,
                At::End,
                content,
                &SceneTextLayout {
                    origin: Origin::User,
                    kind: TextLayoutKind::Paragraph,
                    angle_degrees: None,
                },
            )
            .unwrap();
        setup
            .set(layer, &SceneGeometry::rectangle(10.0, 20.0, 20.0, 10.0))
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let copied_payload = TextClipboardPayload {
            version: 1,
            mode: TextTransferMode::Copy,
            source_page: page,
            layers: vec![Project::clipboard_layer(&project.snapshot(), layer).unwrap()],
        };
        let (_, copied) = project
            .paste_text_blocks(page, copied_payload, Some(Point { x: 80.0, y: 90.0 }))
            .await
            .unwrap();
        let copied_geometry = Project::clipboard_layer(&project.snapshot(), copied[0])
            .unwrap()
            .geometry
            .unwrap();
        assert_eq!(copied_geometry.points[0], Point { x: 70.0, y: 85.0 });
        assert_eq!(copied_geometry.points[2], Point { x: 90.0, y: 95.0 });

        let moved_payload = TextClipboardPayload {
            version: 1,
            mode: TextTransferMode::Move,
            source_page: page,
            layers: vec![Project::clipboard_layer(&project.snapshot(), layer).unwrap()],
        };
        let (_, moved) = project
            .paste_text_blocks(page, moved_payload, Some(Point { x: 60.0, y: 70.0 }))
            .await
            .unwrap();
        assert_eq!(moved, [layer]);
        let moved_geometry = Project::clipboard_layer(&project.snapshot(), layer)
            .unwrap()
            .geometry
            .unwrap();
        assert_eq!(moved_geometry.points[0], Point { x: 50.0, y: 65.0 });
        assert_eq!(moved_geometry.points[2], Point { x: 70.0, y: 75.0 });
    }

    #[tokio::test]
    async fn text_clipboard_copy_preserves_visible_data_and_internal_text_relations() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let source_page = setup
            .add_page(PageDraft::new("source", 100.0, 100.0), At::End)
            .unwrap();
        let target_page = setup
            .add_page(PageDraft::new("target", 100.0, 100.0), At::End)
            .unwrap();
        let geometry = SceneGeometry {
            origin: Origin::User,
            points: vec![
                ScenePoint { x: 10.0, y: 20.0 },
                ScenePoint { x: 40.0, y: 20.0 },
                ScenePoint { x: 40.0, y: 50.0 },
                ScenePoint { x: 10.0, y: 50.0 },
            ],
        };
        let region = setup
            .add_analysis_region::<TextRegion>(source_page, At::End, &geometry, None)
            .unwrap();
        let content = setup.add_text_content(source_page, At::End).unwrap();
        setup
            .set(
                content,
                &SceneSourceText {
                    text: Authored::user("原文".to_owned()),
                    language: Some(koharu_scene::LanguageTag::new("ja").unwrap()),
                },
            )
            .unwrap();
        setup
            .set(
                content,
                &SceneTranslation {
                    text: Authored::user("Translation".to_owned()),
                    language: Some(koharu_scene::LanguageTag::new("en").unwrap()),
                },
            )
            .unwrap();
        setup
            .set(
                content,
                &SceneTextRole {
                    origin: Origin::User,
                    role: DIALOGUE_ROLE.to_owned(),
                },
            )
            .unwrap();
        setup.relate::<RecognizedFrom>(content, region).unwrap();
        let layer = setup
            .add_text_layer(
                source_page,
                At::End,
                content,
                &SceneTextLayout {
                    origin: Origin::User,
                    kind: TextLayoutKind::Paragraph,
                    angle_degrees: None,
                },
            )
            .unwrap();
        setup.set(layer, &geometry).unwrap();
        setup
            .set(
                layer,
                &SceneTypography {
                    origin: Origin::User,
                    preferred_font: Some("Noto Sans".to_owned()),
                    font_weight: Some(700),
                    font_style: None,
                    size: Some(18.0),
                    auto_fit: false,
                    color: Some([1, 2, 3, 255]),
                    stroke_color: None,
                    stroke_width: None,
                    alignment: Some(koharu_scene::TextAlignment::Center),
                    writing_mode: Some(WritingMode::Vertical),
                    extensions: Default::default(),
                },
            )
            .unwrap();
        setup.relate::<FitsTo>(layer, region).unwrap();
        let second_layer = setup
            .add_text_layer(
                source_page,
                At::End,
                content,
                &SceneTextLayout {
                    origin: Origin::User,
                    kind: TextLayoutKind::Paragraph,
                    angle_degrees: None,
                },
            )
            .unwrap();
        setup.set(second_layer, &geometry).unwrap();
        setup.relate::<FitsTo>(second_layer, region).unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());
        let payload = TextClipboardPayload {
            version: 1,
            mode: TextTransferMode::Copy,
            source_page,
            layers: vec![
                Project::clipboard_layer(&project.snapshot(), layer).unwrap(),
                Project::clipboard_layer(&project.snapshot(), second_layer).unwrap(),
            ],
        };

        let (commit, copied) = project
            .paste_text_blocks(target_page, payload.clone(), None)
            .await
            .unwrap();
        project.record_commit(&commit);
        let copied_layer = copied[0];
        let copied_block = Project::clipboard_layer(&project.snapshot(), copied_layer).unwrap();
        assert_ne!(copied_layer, layer);
        assert_ne!(copied_block.content, content);
        assert_eq!(copied_block.source, payload.layers[0].source);
        assert_eq!(copied_block.translation, payload.layers[0].translation);
        assert_eq!(copied_block.role, payload.layers[0].role);
        assert_eq!(copied_block.geometry, payload.layers[0].geometry);
        assert_eq!(copied_block.typography, payload.layers[0].typography);
        assert_ne!(
            copied_block.source_region.as_ref().unwrap().id,
            payload.layers[0].source_region.as_ref().unwrap().id
        );
        assert_eq!(
            copied_block.source_region.as_ref().unwrap().id,
            copied_block.automatic_region.as_ref().unwrap().id
        );
        assert_eq!(
            project
                .snapshot()
                .relations_from_as::<FlowsIn>(copied_layer)
                .count(),
            0
        );
        assert_eq!(
            project
                .snapshot()
                .relation_from::<Presents>(copied[1])
                .unwrap()
                .unwrap()
                .value()
                .target,
            copied_block.content
        );

        project.undo().await.unwrap();
        assert!(project.snapshot().entity(copied_layer).is_err());
    }

    #[tokio::test]
    async fn moving_one_of_two_shared_text_layers_uses_copy_on_write() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let source_page = setup
            .add_page(PageDraft::new("source", 100.0, 100.0), At::End)
            .unwrap();
        let target_page = setup
            .add_page(PageDraft::new("target", 100.0, 100.0), At::End)
            .unwrap();
        let content = setup.add_text_content(source_page, At::End).unwrap();
        setup
            .set(
                content,
                &SceneSourceText {
                    text: Authored::user("shared".to_owned()),
                    language: None,
                },
            )
            .unwrap();
        let layout = SceneTextLayout {
            origin: Origin::User,
            kind: TextLayoutKind::Paragraph,
            angle_degrees: None,
        };
        let moved = setup
            .add_text_layer(source_page, At::End, content, &layout)
            .unwrap();
        let retained = setup
            .add_text_layer(source_page, At::End, content, &layout)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());
        let payload = TextClipboardPayload {
            version: 1,
            mode: TextTransferMode::Move,
            source_page,
            layers: vec![Project::clipboard_layer(&project.snapshot(), moved).unwrap()],
        };

        let (commit, layers) = project
            .paste_text_blocks(target_page, payload, None)
            .await
            .unwrap();
        project.record_commit(&commit);
        assert_eq!(layers, [moved]);
        let snapshot = project.snapshot();
        let moved_content = snapshot
            .relation_from::<Presents>(moved)
            .unwrap()
            .unwrap()
            .value()
            .target;
        assert_ne!(moved_content, content);
        assert_eq!(
            snapshot
                .relation_from::<Presents>(retained)
                .unwrap()
                .unwrap()
                .value()
                .target,
            content
        );
        assert_eq!(
            snapshot
                .component::<SceneSourceText>(moved_content)
                .unwrap()
                .unwrap()
                .text
                .value,
            "shared"
        );

        let exclusive = TextClipboardPayload {
            version: 1,
            mode: TextTransferMode::Move,
            source_page,
            layers: vec![Project::clipboard_layer(&project.snapshot(), retained).unwrap()],
        };
        let (exclusive_commit, exclusive_layers) = project
            .paste_text_blocks(target_page, exclusive, None)
            .await
            .unwrap();
        project.record_commit(&exclusive_commit);
        assert_eq!(exclusive_layers, [retained]);
        assert_eq!(
            project
                .snapshot()
                .relation_from::<Presents>(retained)
                .unwrap()
                .unwrap()
                .value()
                .target,
            content
        );

        project.undo().await.unwrap();
        project.undo().await.unwrap();
        let snapshot = project.snapshot();
        assert_eq!(
            snapshot
                .relation_from::<Presents>(moved)
                .unwrap()
                .unwrap()
                .value()
                .target,
            content
        );
    }

    #[tokio::test]
    async fn cleanup_import_scales_reuses_the_layer_and_is_the_default_eraser_target() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 4.0, 4.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());
        let mut source = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, image::Rgba([9, 8, 7, 123])))
            .write_to(&mut source, ImageFormat::Png)
            .unwrap();
        let data_url = format!(
            "data:image/png;base64,{}",
            STANDARD.encode(source.into_inner())
        );

        let mut wide = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(RgbaImage::new(2, 1))
            .write_to(&mut wide, ImageFormat::Png)
            .unwrap();
        assert!(
            project
                .import_cleanup_image(
                    page,
                    format!(
                        "data:image/png;base64,{}",
                        STANDARD.encode(wide.into_inner())
                    ),
                )
                .await
                .is_err()
        );

        let (first, cleanup) = project
            .import_cleanup_image(page, data_url.clone())
            .await
            .unwrap();
        project.record_commit(&first);
        let first_blob = project
            .snapshot()
            .asset(cleanup, &AssetRole::new("source").unwrap())
            .unwrap()
            .unwrap()
            .blob;
        let first_image =
            image::load_from_memory(&project.snapshot().read_blob(first_blob).await.unwrap())
                .unwrap()
                .to_rgba8();
        assert!(first_image.pixels().all(|pixel| pixel[3] == 123));
        let (second, replacement) = project.import_cleanup_image(page, data_url).await.unwrap();
        project.record_commit(&second);
        assert_eq!(replacement, cleanup);
        assert_eq!(
            project
                .snapshot()
                .asset(cleanup, &AssetRole::new("source").unwrap())
                .unwrap()
                .unwrap()
                .metadata
                .width,
            Some(4)
        );
        assert_eq!(
            project
                .snapshot()
                .component::<SceneRasterLayer>(cleanup)
                .unwrap()
                .unwrap()
                .kind,
            RasterLayerKind::Cleanup
        );

        let (erased, erased_layer) = project
            .apply_raster_stroke(
                page,
                None,
                RasterStrokeMode::Erase,
                [0; 4],
                3.0,
                vec![ScenePoint { x: 2.0, y: 2.0 }],
            )
            .await
            .unwrap();
        project.record_commit(&erased);
        assert_eq!(erased_layer, cleanup);
        let asset = project
            .snapshot()
            .asset(cleanup, &AssetRole::new("source").unwrap())
            .unwrap()
            .unwrap();
        let image =
            image::load_from_memory(&project.snapshot().read_blob(asset.blob).await.unwrap())
                .unwrap()
                .to_rgba8();
        assert!(image.pixels().any(|pixel| pixel[3] == 0));

        project.undo().await.unwrap();
        assert_eq!(
            project
                .snapshot()
                .asset(cleanup, &AssetRole::new("source").unwrap())
                .unwrap()
                .unwrap()
                .blob,
            first_blob
        );
    }

    #[tokio::test]
    async fn reordering_layers_moves_a_group_atomically_and_preserves_its_internal_order() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        let layout = SceneTextLayout {
            origin: Origin::User,
            kind: TextLayoutKind::Paragraph,
            angle_degrees: None,
        };
        let mut layers = Vec::new();
        for _ in 0..4 {
            let content = setup.add_text_content(page, At::End).unwrap();
            layers.push(
                setup
                    .add_text_layer(page, At::End, content, &layout)
                    .unwrap(),
            );
        }
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());
        let group = project
            .snapshot()
            .page(page)
            .unwrap()
            .text_group()
            .unwrap()
            .unwrap()
            .id();

        let commit = project
            .reorder_layers(group, vec![layers[2], layers[3], layers[0], layers[1]])
            .await
            .unwrap();
        project.record_commit(&commit);
        let text_order = |snapshot: &Snapshot| {
            snapshot
                .page(page)
                .unwrap()
                .text_group()
                .unwrap()
                .unwrap()
                .text_layers()
                .unwrap()
                .map(|layer| layer.id())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            text_order(&project.snapshot()),
            [layers[2], layers[3], layers[0], layers[1]]
        );

        project.undo().await.unwrap();
        assert_eq!(text_order(&project.snapshot()), layers);
        project.redo().await.unwrap();
        assert_eq!(
            text_order(&project.snapshot()),
            [layers[2], layers[3], layers[0], layers[1]]
        );
    }

    #[tokio::test]
    async fn pipeline_commit_rebases_or_yields_to_the_manual_edit() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let pipeline_page = setup
            .add_page(PageDraft::new("pipeline", 100.0, 100.0), At::End)
            .unwrap();
        let manual_page = setup
            .add_page(PageDraft::new("manual", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let base = project.snapshot();
        let pipeline = base
            .patch(|edit| {
                edit.set_page(
                    pipeline_page,
                    PageDraft::new("pipeline result", 100.0, 100.0),
                )
            })
            .unwrap();
        let manual = base
            .patch(|edit| edit.set_page(manual_page, PageDraft::new("manual edit", 100.0, 100.0)))
            .unwrap();
        project.commit(manual).await.unwrap();

        let commit = project.commit_rebased(pipeline).await.unwrap().unwrap();
        assert_eq!(
            commit
                .snapshot
                .page(pipeline_page)
                .unwrap()
                .page()
                .unwrap()
                .label,
            "pipeline result"
        );
        assert_eq!(
            commit
                .snapshot
                .page(manual_page)
                .unwrap()
                .page()
                .unwrap()
                .label,
            "manual edit"
        );

        let base = project.snapshot();
        let pipeline = base
            .patch(|edit| {
                edit.set_page(
                    pipeline_page,
                    PageDraft::new("stale pipeline", 100.0, 100.0),
                )
            })
            .unwrap();
        let manual = base
            .patch(|edit| {
                edit.set_page(pipeline_page, PageDraft::new("latest manual", 100.0, 100.0))
            })
            .unwrap();
        project.commit(manual).await.unwrap();

        assert!(project.commit_rebased(pipeline).await.unwrap().is_none());
        assert_eq!(
            project
                .snapshot()
                .page(pipeline_page)
                .unwrap()
                .page()
                .unwrap()
                .label,
            "latest manual"
        );
    }

    #[tokio::test]
    async fn deleting_the_last_ocr_layer_removes_its_text_region_but_keeps_the_bubble() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        let bubble = setup
            .add_analysis_region::<BubbleRegion>(
                page,
                At::End,
                &SceneGeometry::rectangle(5.0, 5.0, 90.0, 90.0),
                None,
            )
            .unwrap();
        let region = setup
            .add_analysis_region::<TextRegion>(
                page,
                At::End,
                &SceneGeometry::rectangle(20.0, 20.0, 30.0, 20.0),
                None,
            )
            .unwrap();
        let content = setup.add_text_content(page, At::End).unwrap();
        let layer = setup
            .add_text_layer(
                page,
                At::End,
                content,
                &SceneTextLayout {
                    origin: Origin::User,
                    kind: TextLayoutKind::Paragraph,
                    angle_degrees: None,
                },
            )
            .unwrap();
        setup.relate::<RecognizedFrom>(content, region).unwrap();
        setup.relate::<Inside>(region, bubble).unwrap();
        setup.relate::<FlowsIn>(layer, bubble).unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let deletion = project.delete_layers(vec![layer]).await.unwrap();
        project.record_commit(&deletion);

        let snapshot = project.snapshot();
        assert!(snapshot.entity(layer).is_err());
        assert!(snapshot.entity(content).is_err());
        assert!(snapshot.entity(region).is_err());
        assert!(snapshot.entity(bubble).is_ok());

        project.undo().await.unwrap();
        let snapshot = project.snapshot();
        assert!(snapshot.entity(layer).is_ok());
        assert!(snapshot.entity(content).is_ok());
        assert!(snapshot.entity(region).is_ok());
        assert!(snapshot.entity(bubble).is_ok());

        project.redo().await.unwrap();
        let snapshot = project.snapshot();
        assert!(snapshot.entity(layer).is_err());
        assert!(snapshot.entity(content).is_err());
        assert!(snapshot.entity(region).is_err());
        assert!(snapshot.entity(bubble).is_ok());
    }

    #[tokio::test]
    async fn deleting_one_of_multiple_presentations_keeps_the_content_and_region() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        let region = setup
            .add_analysis_region::<TextRegion>(
                page,
                At::End,
                &SceneGeometry::rectangle(20.0, 20.0, 30.0, 20.0),
                None,
            )
            .unwrap();
        let content = setup.add_text_content(page, At::End).unwrap();
        let layout = SceneTextLayout {
            origin: Origin::User,
            kind: TextLayoutKind::Paragraph,
            angle_degrees: None,
        };
        let first = setup
            .add_text_layer(page, At::End, content, &layout)
            .unwrap();
        let second = setup
            .add_text_layer(page, At::End, content, &layout)
            .unwrap();
        setup.relate::<RecognizedFrom>(content, region).unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        project.delete_layers(vec![first]).await.unwrap();

        let snapshot = project.snapshot();
        assert!(snapshot.entity(first).is_err());
        assert!(snapshot.entity(second).is_ok());
        assert!(snapshot.entity(content).is_ok());
        assert!(snapshot.entity(region).is_ok());
    }

    #[tokio::test]
    async fn manual_ocr_region_is_empty_until_ocr_and_uses_bubble_semantics_when_contained() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        let bubble = setup
            .add_analysis_region::<BubbleRegion>(
                page,
                At::End,
                &SceneGeometry::rectangle(5.0, 5.0, 55.0, 55.0),
                None,
            )
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let (_, dialogue_layer, dialogue_region) = project
            .add_ocr_region(
                page,
                Frame {
                    x: 15.0,
                    y: 15.0,
                    width: 20.0,
                    height: 20.0,
                    angle_degrees: 0.0,
                },
                koharu_translator::Language::Japanese,
            )
            .await
            .unwrap();
        let (_, free_layer, free_region) = project
            .add_ocr_region(
                page,
                Frame {
                    x: 70.0,
                    y: 70.0,
                    width: 20.0,
                    height: 20.0,
                    angle_degrees: 0.0,
                },
                koharu_translator::Language::Japanese,
            )
            .await
            .unwrap();

        let snapshot = project.snapshot();
        assert_eq!(
            snapshot
                .page(page)
                .unwrap()
                .text_group()
                .unwrap()
                .unwrap()
                .text_layers()
                .unwrap()
                .map(|layer| layer.id())
                .collect::<Vec<_>>(),
            [free_layer, dialogue_layer]
        );
        let dialogue_content = snapshot
            .relation_from::<Presents>(dialogue_layer)
            .unwrap()
            .unwrap()
            .value()
            .target;
        assert!(
            snapshot
                .component::<SceneSourceText>(dialogue_content)
                .unwrap()
                .is_none()
        );
        assert!(
            snapshot
                .component::<DetectionAnalysis>(dialogue_region)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            snapshot
                .component::<TextRole>(dialogue_content)
                .unwrap()
                .unwrap()
                .role,
            DIALOGUE_ROLE
        );
        assert_eq!(
            snapshot
                .relation_from::<RecognizedFrom>(dialogue_content)
                .unwrap()
                .unwrap()
                .value()
                .target,
            dialogue_region
        );
        assert_eq!(
            snapshot
                .relations_from_as::<Inside>(dialogue_region)
                .next()
                .unwrap()
                .value()
                .target,
            bubble
        );
        assert_eq!(
            snapshot
                .relation_from::<FlowsIn>(dialogue_layer)
                .unwrap()
                .unwrap()
                .value()
                .target,
            bubble
        );

        let free_content = snapshot
            .relation_from::<Presents>(free_layer)
            .unwrap()
            .unwrap()
            .value()
            .target;
        assert_eq!(
            snapshot
                .component::<TextRole>(free_content)
                .unwrap()
                .unwrap()
                .role,
            FREE_TEXT_ROLE
        );
        assert_eq!(
            snapshot
                .relation_from::<FitsTo>(free_layer)
                .unwrap()
                .unwrap()
                .value()
                .target,
            free_region
        );

        project
            .set_geometry(vec![GeometryUpdate {
                layer: free_region,
                points: Some(vec![
                    Point { x: 68.0, y: 68.0 },
                    Point { x: 95.0, y: 75.0 },
                    Point { x: 88.0, y: 98.0 },
                    Point { x: 61.0, y: 91.0 },
                ]),
            }])
            .await
            .unwrap();
        let geometry = project
            .snapshot()
            .component::<SceneGeometry>(free_region)
            .unwrap()
            .unwrap();
        assert_eq!(geometry.points[0].x, 68.0);
        assert_eq!(geometry.points[3].y, 91.0);
    }

    #[tokio::test]
    async fn editing_detected_typography_keeps_the_drawn_outline_width() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        let content = setup.add_text_content(page, At::End).unwrap();
        let layer = setup
            .add_text_layer(
                page,
                At::End,
                content,
                &SceneTextLayout {
                    origin: Origin::User,
                    kind: TextLayoutKind::Paragraph,
                    angle_degrees: None,
                },
            )
            .unwrap();
        let detected = SceneTypography {
            origin: Origin::Generated(Generation::new(
                ProducerId::new("dev.koharu.pipeline.detection").expect("valid producer"),
            )),
            preferred_font: None,
            font_weight: None,
            font_style: None,
            size: None,
            auto_fit: true,
            color: Some([0, 0, 0, 255]),
            stroke_color: Some([255, 255, 255, 255]),
            stroke_width: Some(12.0),
            alignment: None,
            writing_mode: None,
            extensions: Default::default(),
        };
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut detection = session.snapshot().edit_as(Generation::new(
            ProducerId::new("dev.koharu.pipeline.detection").expect("valid producer"),
        ));
        detection.set(layer, &detected).unwrap();
        session.commit(detection.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());
        let recolored = |stroke_width| TypographyUpdate {
            layer,
            typography: Typography {
                color: Some([200, 0, 0, 255]),
                stroke_width,
                ..Project::typography_view(detected.clone())
            },
        };
        let stored_width = |project: &Project| {
            project
                .snapshot()
                .component::<SceneTypography>(layer)
                .unwrap()
                .unwrap()
                .stroke_width
        };

        let commit = project
            .set_typography(vec![recolored(Some(12.0))], |_| Some(20.0))
            .await
            .unwrap();
        project.record_commit(&commit);
        assert_eq!(stored_width(&project), Some(2.0));

        let commit = project
            .set_typography(vec![recolored(Some(5.0))], |_| Some(20.0))
            .await
            .unwrap();
        project.record_commit(&commit);
        assert_eq!(stored_width(&project), Some(5.0));
    }

    #[tokio::test]
    async fn changing_an_ocr_text_role_is_undoable_and_rejects_unknown_roles() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());
        let (_, layer, _) = project
            .add_ocr_region(
                page,
                Frame {
                    x: 10.0,
                    y: 10.0,
                    width: 30.0,
                    height: 20.0,
                    angle_degrees: 0.0,
                },
                koharu_translator::Language::Japanese,
            )
            .await
            .unwrap();
        let content = Project::text_content(&project.snapshot(), layer).unwrap();

        assert!(
            project
                .set_text_role(layer, "dev.koharu.text.unknown".to_owned())
                .await
                .is_err()
        );
        let commit = project
            .set_text_role(layer, ONOMATOPOEIA_ROLE.to_owned())
            .await
            .unwrap();
        project.record_commit(&commit);
        let role = project
            .snapshot()
            .component::<TextRole>(content)
            .unwrap()
            .unwrap();
        assert_eq!(role.role, ONOMATOPOEIA_ROLE);
        assert_eq!(role.origin, Origin::User);

        project.undo().await.unwrap();
        assert_eq!(
            project
                .snapshot()
                .component::<TextRole>(content)
                .unwrap()
                .unwrap()
                .role,
            FREE_TEXT_ROLE
        );

        project.redo().await.unwrap();
        assert_eq!(
            project
                .snapshot()
                .component::<TextRole>(content)
                .unwrap()
                .unwrap()
                .role,
            ONOMATOPOEIA_ROLE
        );
    }

    #[test]
    fn raster_strokes_are_continuous_and_erasable() {
        let mut image = RgbaImage::new(32, 16);
        let points = [
            ScenePoint { x: 4.0, y: 8.0 },
            ScenePoint { x: 28.0, y: 8.0 },
        ];
        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Paint,
            [210, 40, 20, 255],
            5.0,
            &points,
        );
        for x in 4..28 {
            assert_eq!(image.get_pixel(x, 8).0, [210, 40, 20, 255]);
        }

        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Erase,
            [0, 0, 0, 0],
            5.0,
            &[ScenePoint { x: 16.0, y: 8.0 }],
        );
        assert_eq!(image.get_pixel(16, 8)[3], 0);
        assert_eq!(image.get_pixel(4, 8)[3], 255);

        let mut white = RgbaImage::new(4, 4);
        rasterize_stroke(
            &mut white,
            RasterStrokeMode::Paint,
            [255, 255, 255, 255],
            3.0,
            &[ScenePoint { x: 2.0, y: 2.0 }],
        );
        assert_eq!(white.get_pixel(2, 2).0, [255, 255, 255, 255]);
    }

    #[tokio::test]
    async fn importing_characters_adds_missing_ones_without_touching_the_source() {
        let root = tempfile::tempdir().unwrap();
        let library = ProjectLibrary {
            root: root.path().to_owned(),
        };
        let input = |name: &str, gender: Gender| CharacterInput {
            id: None,
            name: name.to_owned(),
            gender,
            notes: format!("{name} notes"),
        };
        {
            let mut source = library.create("chapter-1").await.unwrap();
            source
                .set_characters(vec![
                    input("Izuku", Gender::Male),
                    input("Ochaco", Gender::Female),
                ])
                .await
                .unwrap();
        }
        let source_files = |name: &str| {
            ["state-a.khr", "state-b.khr"].map(|file| {
                std::fs::read(root.path().join(format!("{name}.khrproj")).join(file)).ok()
            })
        };
        let before = source_files("chapter-1");
        let mut target = library.create("chapter-2").await.unwrap();
        target
            .set_characters(vec![input("Ochaco", Gender::Unknown)])
            .await
            .unwrap();
        let undo_steps = target.undo.len();

        let imported = library.characters("chapter-1").await.unwrap();
        let (commit, added) = target.import_characters(imported.clone()).await.unwrap();
        target.record_commit(&commit);

        assert_eq!(added, 1);
        assert_eq!(target.undo.len(), undo_steps + 1);
        let list = target
            .snapshot()
            .project_component::<CharacterList>()
            .unwrap()
            .unwrap();
        assert_eq!(list.origin, Origin::User);
        assert_eq!(
            list.characters
                .iter()
                .map(|character| (character.name.as_str(), character.gender))
                .collect::<Vec<_>>(),
            [("Ochaco", Gender::Unknown), ("Izuku", Gender::Male)]
        );
        assert_eq!(list.characters[1], imported[0]);

        let revision = target.revision();
        let (commit, added) = target.import_characters(imported).await.unwrap();
        target.record_commit(&commit);
        assert_eq!(added, 0);
        assert_eq!(target.revision(), revision);
        assert_eq!(target.undo.len(), undo_steps + 1);
        assert_eq!(source_files("chapter-1"), before);

        target.undo().await.unwrap();
        assert_eq!(
            Project::characters(&target.snapshot())
                .unwrap()
                .iter()
                .map(|character| character.name.as_str())
                .collect::<Vec<_>>(),
            ["Ochaco"]
        );
    }
}
