use std::{future::Future, pin::Pin, sync::OnceLock};

use anyhow::{Context as _, Result, anyhow, bail};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use koharu_agent::{Control, Host, Invocation, Tool, ToolCall};
use koharu_desktop::{Desktop, Frame};
use koharu_pipeline::{Committer, Operation, RunStatus, Scope, Stage, StageOutput, StopToken};
use koharu_scene::{Commit, EntityId, Snapshot};
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::{Value, json};
use tauri::{AppHandle, Manager as _};
use tauri_runtime_cef::CefRuntime;

use super::typesetting;
use crate::commands::{
    ChannelExt as _,
    canvas::{CanvasChannel, Point},
    editing::{GeometryUpdate, TypographyUpdate},
    image_edit_preview::DesktopImageEditReviewer,
    lifecycle::ProjectChannel,
    output,
    preferences::Preferences,
    processing::{JobId, Processing},
    project::{CurrentProject, Project, Typography},
};

#[derive(Clone)]
pub(super) struct KoharuHost {
    handle: AppHandle<CefRuntime>,
    /// An agent outside the app (MCP) rather than the owner's own panel.
    outside: bool,
}

impl KoharuHost {
    /// The host of the owner's Agent panel: its edits are the owner's.
    pub(super) fn new(handle: AppHandle<CefRuntime>) -> Self {
        Self {
            handle,
            outside: false,
        }
    }

    /// The host of an agent outside the app, editing beside the owner.
    pub(super) fn outside(handle: AppHandle<CefRuntime>) -> Self {
        Self {
            handle,
            outside: true,
        }
    }

    async fn project_context(&self) -> Result<Value> {
        let (project, pages) = {
            let current = self.handle.state::<CurrentProject>();
            let current = current.project.lock().await;
            let project = current.as_ref().context("no project is open")?;
            let snapshot = project.snapshot();
            let pages = Project::pages(&snapshot)?
                .into_iter()
                .map(|page| Project::page(&snapshot, page.id))
                .collect::<Result<Vec<_>>>()?;
            (project.info(), pages)
        };
        let preferences = Preferences::load()?;
        let fonts = self
            .handle
            .state::<Desktop>()
            .renderer()
            .available_fonts()
            .await?;
        let providers = preferences
            .providers
            .entries
            .iter()
            .map(|provider| {
                json!({
                    "name": provider.name,
                    "config": provider.config,
                    "credential_configured": provider
                        .credential
                        .as_ref()
                        .is_some_and(|credential| credential.configured),
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({
            "project": project,
            "pages": pages,
            "configuration": {
                "pipeline": preferences.pipeline,
                "providers": providers,
                "typesetting": preferences.typesetting,
                "available_fonts": fonts.into_iter().map(|font| font.name).collect::<Vec<_>>(),
            },
        }))
    }

    async fn mutate<T>(
        &self,
        mutation: impl for<'project> FnOnce(
            &'project mut Project,
        ) -> Pin<
            Box<dyn Future<Output = Result<(Commit, T)>> + Send + 'project>,
        >,
    ) -> Result<Invocation>
    where
        T: serde::Serialize + Send,
    {
        let (commit, value, page, project) = {
            let current = self.handle.state::<CurrentProject>();
            let mut current = current.project.lock().await;
            let project = current.as_mut().context("no project is open")?;
            let (commit, value) = mutation(project).await?;
            // An agent outside the app edits beside the owner: its edits stay
            // off the owner's undo history, so the owner's Ctrl+Z takes back
            // the owner's own last action and keeps their redo.
            if !self.outside {
                project.record_commit(&commit);
            }
            project.reconcile_page();
            (commit, value, project.active_page(), project.info())
        };
        let revision = commit.revision;
        self.synchronize(commit, page).await?;
        // The UI refreshes its project, pages and layers from this, whichever
        // client (the panel or an MCP agent) made the edit.
        self.handle
            .state::<ProjectChannel>()
            .channel
            .publish(Some(project.clone()));
        Invocation::changed(json!({
            "revision": revision,
            "project": project,
            "result": value,
        }))
    }

    /// The open project's name and revision, to tell an agent when someone
    /// else changed the project between its calls.
    pub(super) async fn project_revision(&self) -> Option<(String, Value)> {
        let current = self.handle.state::<CurrentProject>();
        let current = current.project.lock().await;
        let info = current.as_ref()?.info();
        let revision = serde_json::to_value(info.revision).ok()?;
        Some((info.name, revision))
    }

    async fn snapshot(&self) -> Result<Snapshot> {
        Ok(self
            .handle
            .state::<CurrentProject>()
            .project
            .lock()
            .await
            .as_ref()
            .context("no project is open")?
            .snapshot())
    }

    async fn synchronize(&self, commit: Commit, page: Option<EntityId>) -> Result<()> {
        let desktop = self.handle.state::<Desktop>();
        desktop.synchronize(&commit.snapshot, page, &commit).await?;
        let canvas = desktop.canvas_state();
        self.handle.state::<CanvasChannel>().channel.publish(canvas);
        Ok(())
    }

    async fn run_pipeline(&self, arguments: RunPipeline, control: &Control) -> Result<Invocation> {
        let scope = arguments.scope()?;
        let operation = arguments.operation.into();
        let snapshot = self
            .handle
            .state::<CurrentProject>()
            .project
            .lock()
            .await
            .as_ref()
            .context("no project is open")?
            .snapshot();
        let job = JobId::new();
        let image_edit_reviewer = DesktopImageEditReviewer::for_job(self.handle.clone(), job)?;
        let stop = StopToken::default();
        {
            let processing = self.handle.state::<Processing>();
            let mut stops = processing.stops.lock();
            if !stops.is_empty() {
                bail!("another pipeline process is already running");
            }
            stops.insert(job, stop.clone());
        }

        let watcher = tauri::async_runtime::spawn({
            let control = control.clone();
            let stop = stop.clone();
            async move {
                control.cancelled().await;
                stop.stop();
            }
        });
        let mut committer = AgentCommitter { host: self.clone() };
        let request = koharu_pipeline::Request {
            operation,
            scope,
            stop,
            progress: None,
            inpainting_mask: None,
            image_edit_reviewer,
        };
        let result = self
            .handle
            .state::<koharu_pipeline::Pipeline>()
            .execute(snapshot, request, &mut committer)
            .await;
        watcher.abort();
        self.handle.state::<Processing>().stops.lock().remove(&job);
        let report = result.map_err(|error| anyhow!(error))?;
        if report.status == RunStatus::Stopped {
            bail!("pipeline processing was cancelled");
        }
        Invocation::changed(json!({
            "base_revision": report.base,
            "final_revision": report.final_revision,
            "completed": report.completed,
            "total": report.total,
            "elapsed_ms": report.elapsed.as_millis(),
        }))
    }
}

#[async_trait]
impl Host for KoharuHost {
    async fn context(&self) -> Result<Value> {
        self.project_context().await
    }

    fn tools(&self) -> Vec<Tool> {
        static TOOLS: OnceLock<Vec<Tool>> = OnceLock::new();
        TOOLS
            .get_or_init(|| {
                vec![
                    definition::<InspectProject>(
                        "inspect_project",
                        "Read the latest complete semantic project state after edits. This does not include page images.",
                    ),
                    definition::<ViewPage>(
                        "view_page",
                        "Render and inspect one page image. Call this only for pages whose visual appearance matters to the request.",
                    ),
                    definition::<RenamePage>("rename_page", "Rename a project page."),
                    definition::<MovePage>("move_page", "Move a page to a zero-based project index."),
                    definition::<DeletePages>("delete_pages", "Delete pages and their contents."),
                    definition::<AddTextBox>("add_text_box", "Add a paragraph text box to a page."),
                    definition::<SetText>("set_source_text", "Replace an element's source text."),
                    definition::<SetTranslation>(
                        "set_translation",
                        "Set an element's translation, or pass null to remove it.",
                    ),
                    definition::<SetTypography>(
                        "set_typography",
                        "Replace an element's typography settings. Font is a family name from available_fonts.",
                    ),
                    definition::<SetGeometry>(
                        "set_geometry",
                        "Replace an element's page-space polygon geometry.",
                    ),
                    definition::<SetVisibility>(
                        "set_visibility",
                        "Change visibility and/or opacity for elements.",
                    ),
                    definition::<DeleteElements>("delete_elements", "Delete elements and descendants."),
                    definition::<MoveElement>(
                        "move_element",
                        "Move an element under a page or element at a zero-based index.",
                    ),
                    definition::<typesetting::PageLayers>(
                        "page_layers",
                        "List a page's text layers in their numbered order: text, role, box (as set, with angle when turned), balloon box and the largest box inside the balloon around the text (measured on the page's pixels when the balloon is closed), typography, what the renderer drew (font, size, auto_fit_max: the largest size auto-fit may choose, small: below 1.2% of the page width, layout box, each line with its width and room, hyphenated words, warning and what overflowed), and lettering: whether the cleaning removed the source lettering and its tone, with check_color when the fill is the opposite tone. Also flow (balloon: laid out in the balloon's shape with no box of its own; box: in its box), the balloon's interior (box and centre of the lobe holding the text, without tail and bumps), off_centre (drawn block centre minus that centre, px) and edge_gap (narrowest gap between a line and the balloon's inside, px; negative crosses it), fits_up_to (when auto-fit stopped at its ceiling: the largest size the box takes, to set as a fixed size), empty_balloons (balloons with no text layer, for add_text_layer) and the edge lines of the neighbouring pages. Pages and layers can be named by id or by address: 45 for page 45, 45#13 for its layer 13.",
                    ),
                    definition::<typesetting::Crop>(
                        "crop",
                        "Return a PNG of part of a page as it shows: typeset (with text), clean (without its text layers) or source (as imported). compare puts the source on the left and the view on the right; outline draws every text layer's box with its number and the measured balloon interiors. Pages and layers can be named by id or by address: 45 for page 45, 45#13 for its layer 13.",
                    ),
                    definition::<typesetting::PreviewLayers>(
                        "preview_layers",
                        "Trial-render text, box and typography changes to layers of one page without committing anything. Returns each changed layer in short form (box, size, lines, warning, overflow, hyphenated, small; full for the whole layer) and a PNG of the area at the stated scale. compare puts the source on the left; outline draws the changed layers' boxes with their numbers and the measured balloon interiors. To fit text, try auto_fit in the intended box; line_breaking whole_words or manual stops auto-fit from hyphenating. Pages and layers can be named by id or by address: 45 for page 45, 45#13 for its layer 13.",
                    ),
                    definition::<typesetting::ApplyLayers>(
                        "apply_layers",
                        "Apply text, box, typography and visibility changes to any layers as one edit, one undo step. flow: balloon drops a layer's box and lays it out in the shape of the balloon that holds it (each line as wide as the balloon at its height, placed by the renderer, a joined balloon shared among its layers); a region gives it a box again. Returns each changed layer in short form (box, flow, size, lines, warning, overflow, hyphenated, small, inner_box, off_centre, edge_gap, fits_up_to); full returns them as page_layers does. Pages and layers can be named by id or by address: 45 for page 45, 45#13 for its layer 13.",
                    ),
                    definition::<typesetting::AddTextLayer>(
                        "add_text_layer",
                        "Add a text layer in a region with its translation and typography: an empty balloon from page_layers' empty_balloons, or a box from check_chapter's unboxed_text for lettering drawn in the art. Inside a balloon the text flows in the balloon's shape; elsewhere it fits the region. Without a colour the text is black, or white on a dark ground. One undo step; returns the layer as apply_layers does.",
                    ),
                    definition::<typesetting::HideUntranslated>(
                        "hide_untranslated",
                        "Hide every shown text layer that has no translation (not lettering, or left untranslated), on all pages or a page range, as one undo step. Returns how many were hidden.",
                    ),
                    definition::<typesetting::MergeTextLayers>(
                        "merge_text_layers",
                        "Merge two or more text layers of one page into the first, in reading order: one sound effect detection split in two. Sources and translations are joined (with joiner, nothing by default), the box covers all of them, and the other layers are deleted.",
                    ),
                    definition::<typesetting::CheckChapterArguments>(
                        "check_chapter",
                        "Run the pre-export chapter check: blocks with no translation, Japanese left in a translation, fit warnings, unverified OCR (cleared by writing the source text) or cleaning, translated text that is hidden and what hides it, and pages with lettering outside every text region. Pages and layers carry their numbers. One line per issue (page#layer, kinds, text) after a count by kind; kinds keeps only some (fit, translation, ocr, cleaning, hidden, unboxed); full returns JSON.",
                    ),
                    definition::<typesetting::ChapterText>(
                        "chapter_text",
                        "List the chapter's text, one line per layer: `number role [hidden] | source | translation`, line breaks as ` / `, under a `# page N label id` line per page. full returns JSON with ids instead.",
                    ),
                    definition::<RunPipeline>(
                        "run_pipeline",
                        "Run Koharu's configured processing pipeline for the whole project, selected pages, or selected text elements.",
                    ),
                ]
            })
            .clone()
    }

    async fn invoke(&self, call: ToolCall, control: &Control) -> Result<Invocation> {
        match call.name.as_str() {
            "inspect_project" => {
                let _: InspectProject = arguments(&call)?;
                Invocation::read(self.project_context().await?)
            }
            "view_page" => {
                let arguments: ViewPage = arguments(&call)?;
                let page = entity(&arguments.page)?;
                let (label, snapshot) = {
                    let current = self.handle.state::<CurrentProject>();
                    let current = current.project.lock().await;
                    let project = current.as_ref().context("no project is open")?;
                    let snapshot = project.snapshot();
                    let label = snapshot.page(page)?.page()?.label;
                    (label, snapshot)
                };
                let desktop = self.handle.state::<Desktop>();
                let renderer = desktop.renderer();
                let rasterizer = desktop.rasterizer().await?;
                let bytes =
                    output::rendered_preview(&renderer, rasterizer, &snapshot, page).await?;
                Ok(
                    Invocation::read(json!({ "page": page, "label": label }))?.with_image(
                        format!("Rendered page {label} ({page})"),
                        format!("data:image/webp;base64,{}", STANDARD.encode(bytes)),
                    ),
                )
            }
            "rename_page" => {
                let arguments: RenamePage = arguments(&call)?;
                let page = entity(&arguments.page)?;
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project.rename_page(page, arguments.label).await?,
                            json!({ "page": page }),
                        ))
                    })
                })
                .await
            }
            "move_page" => {
                let arguments: MovePage = arguments(&call)?;
                let page = entity(&arguments.page)?;
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project.move_page(page, arguments.index).await?,
                            json!({ "page": page }),
                        ))
                    })
                })
                .await
            }
            "delete_pages" => {
                let arguments: DeletePages = arguments(&call)?;
                let pages = entities(&arguments.pages)?;
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project.delete_pages(pages.clone()).await?,
                            json!({ "pages": pages }),
                        ))
                    })
                })
                .await
            }
            "add_text_box" => {
                let arguments: AddTextBox = arguments(&call)?;
                let page = entity(&arguments.page)?;
                let frame = Frame {
                    x: arguments.x,
                    y: arguments.y,
                    width: arguments.width,
                    height: arguments.height,
                    angle_degrees: arguments.angle_degrees,
                };
                self.mutate(|project| {
                    Box::pin(async move {
                        let (commit, element) = project.add_text_box(page, frame).await?;
                        Ok((commit, json!({ "page": page, "element": element })))
                    })
                })
                .await
            }
            "set_source_text" => {
                let arguments: SetText = arguments(&call)?;
                let element = entity(&arguments.element)?;
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project.set_source_text(element, arguments.text).await?,
                            json!({ "element": element }),
                        ))
                    })
                })
                .await
            }
            "set_translation" => {
                let arguments: SetTranslation = arguments(&call)?;
                let element = entity(&arguments.element)?;
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project.set_translation(element, arguments.text).await?,
                            json!({ "element": element }),
                        ))
                    })
                })
                .await
            }
            "set_typography" => {
                let arguments: SetTypography = arguments(&call)?;
                let element = entity(&arguments.element)?;
                let typography = Typography {
                    preferred_font: arguments.preferred_font,
                    font_weight: arguments.font_weight,
                    font_style: arguments.font_style.map(Into::into),
                    size: arguments.size,
                    auto_fit: arguments.auto_fit,
                    color: arguments.color,
                    stroke_color: arguments.stroke_color,
                    stroke_width: arguments.stroke_width,
                    alignment: arguments.alignment.map(Into::into),
                    writing_mode: arguments.writing_mode.map(Into::into),
                    line_breaking: arguments.line_breaking.map(Into::into),
                };
                let laid_out_size = self.handle.state::<Desktop>().presented_font_size(element);
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project
                                .set_typography(
                                    vec![TypographyUpdate {
                                        layer: element,
                                        typography,
                                    }],
                                    |_| laid_out_size,
                                )
                                .await?,
                            json!({ "element": element }),
                        ))
                    })
                })
                .await
            }
            "set_geometry" => {
                let arguments: SetGeometry = arguments(&call)?;
                let element = entity(&arguments.element)?;
                let points = arguments
                    .points
                    .into_iter()
                    .map(|point| Point {
                        x: point.x,
                        y: point.y,
                    })
                    .collect();
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project
                                .set_geometry(vec![GeometryUpdate {
                                    layer: element,
                                    points: Some(points),
                                }])
                                .await?,
                            json!({ "element": element }),
                        ))
                    })
                })
                .await
            }
            "set_visibility" => {
                let arguments: SetVisibility = arguments(&call)?;
                let elements = entities(&arguments.elements)?;
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project
                                .set_visibility(
                                    elements.clone(),
                                    arguments.visible,
                                    arguments.opacity,
                                )
                                .await?,
                            json!({ "elements": elements }),
                        ))
                    })
                })
                .await
            }
            "delete_elements" => {
                let arguments: DeleteElements = arguments(&call)?;
                let elements = entities(&arguments.elements)?;
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project.delete_layers(elements.clone()).await?,
                            json!({ "elements": elements }),
                        ))
                    })
                })
                .await
            }
            "move_element" => {
                let arguments: MoveElement = arguments(&call)?;
                let element = entity(&arguments.element)?;
                let parent = entity(&arguments.parent)?;
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project.move_layer(element, parent, arguments.index).await?,
                            json!({ "element": element, "parent": parent }),
                        ))
                    })
                })
                .await
            }
            "page_layers" => {
                let arguments: typesetting::PageLayers = arguments(&call)?;
                let snapshot = self.snapshot().await?;
                let page = typesetting::address(&snapshot, &arguments.page)?;
                let desktop = self.handle.state::<Desktop>();
                Invocation::read(typesetting::page_layers(&desktop, &snapshot, page).await?)
            }
            "crop" => {
                let arguments: typesetting::Crop = arguments(&call)?;
                let snapshot = self.snapshot().await?;
                let desktop = self.handle.state::<Desktop>();
                let (value, png) = typesetting::crop(&desktop, &snapshot, &arguments).await?;
                Ok(Invocation::read(value)?.with_image("Page crop", typesetting::image_url(png)))
            }
            "preview_layers" => {
                let arguments: typesetting::PreviewLayers = arguments(&call)?;
                let snapshot = self.snapshot().await?;
                let desktop = self.handle.state::<Desktop>();
                let (value, png) =
                    typesetting::preview_layers(&desktop, &snapshot, arguments).await?;
                Ok(
                    Invocation::read(value)?
                        .with_image("Trial render", typesetting::image_url(png)),
                )
            }
            "apply_layers" => {
                let arguments: typesetting::ApplyLayers = arguments(&call)?;
                let layers = typesetting::elements(&self.snapshot().await?, &arguments.changes)?;
                let (full, quiet) = (arguments.full, arguments.quiet);
                let desktop = self.handle.state::<Desktop>();
                let sizes = layers
                    .iter()
                    .map(|layer| (*layer, desktop.presented_font_size(*layer)))
                    .collect::<Vec<_>>();
                let mut invocation = self
                    .mutate(|project| {
                        Box::pin(async move {
                            let snapshot = project.snapshot();
                            let steps =
                                typesetting::steps(&snapshot, arguments.changes, |layer| {
                                    sizes
                                        .iter()
                                        .find(|(candidate, _)| *candidate == layer)
                                        .and_then(|(_, size)| *size)
                                })?;
                            Ok((
                                project.apply_steps(steps).await?,
                                json!({ "changed": true }),
                            ))
                        })
                    })
                    .await?;
                if !quiet {
                    let snapshot = self.snapshot().await?;
                    invocation.value["layers"] =
                        typesetting::report(&desktop, &snapshot, &layers, full).await?;
                }
                Ok(invocation)
            }
            "hide_untranslated" => {
                let arguments: typesetting::HideUntranslated = arguments(&call)?;
                self.mutate(|project| {
                    Box::pin(async move {
                        let snapshot = project.snapshot();
                        let (steps, count) =
                            typesetting::hide_untranslated_steps(&snapshot, &arguments)?;
                        if count == 0 {
                            bail!("every shown text layer already has a translation");
                        }
                        Ok((
                            project.apply_steps(steps).await?,
                            json!({ "hidden": count }),
                        ))
                    })
                })
                .await
            }
            "merge_text_layers" => {
                let arguments: typesetting::MergeTextLayers = arguments(&call)?;
                let snapshot = self.snapshot().await?;
                let layers = arguments
                    .layers
                    .iter()
                    .map(|layer| typesetting::address(&snapshot, layer))
                    .collect::<Result<Vec<_>>>()?;
                let joiner = arguments.joiner;
                self.mutate(|project| {
                    Box::pin(async move {
                        let snapshot = project.snapshot();
                        let steps = typesetting::merge_steps(&snapshot, &layers, &joiner)?;
                        project.apply_steps(steps).await?;
                        let commit = project.delete_layers(layers[1..].to_vec()).await?;
                        Ok((
                            commit,
                            json!({ "kept": layers[0], "removed": &layers[1..] }),
                        ))
                    })
                })
                .await
            }
            "add_text_layer" => {
                let arguments: typesetting::AddTextLayer = arguments(&call)?;
                let page = typesetting::address(&self.snapshot().await?, &arguments.page)?;
                let text = match (arguments.text, arguments.lines) {
                    (Some(_), Some(_)) => bail!("give either text or lines, not both"),
                    (text, lines) => text.or_else(|| lines.map(|lines| lines.join("\n"))),
                };
                let [x, y, width, height] = arguments.region;
                let frame = Frame {
                    x: x as f32,
                    y: y as f32,
                    width: width as f32,
                    height: height as f32,
                    angle_degrees: 0.0,
                };
                let source_language = koharu_pipeline::PipelineConfig::load()?
                    .read()?
                    .translation
                    .source_language;
                let typography = {
                    let snapshot = self.snapshot().await?;
                    let desktop = self.handle.state::<Desktop>();
                    typesetting::new_layer_typography(
                        &desktop,
                        &snapshot,
                        page,
                        arguments.region,
                        arguments.typography,
                    )
                    .await?
                };
                let mut invocation = self
                    .mutate(|project| {
                        Box::pin(async move {
                            let commit = project
                                .add_text_layer(page, frame, source_language, text, typography)
                                .await?;
                            let layer = commit.1;
                            Ok((commit.0, json!({ "element": layer })))
                        })
                    })
                    .await?;
                let layer = entity(
                    invocation.value["result"]["element"]
                        .as_str()
                        .unwrap_or_default(),
                )?;
                let snapshot = self.snapshot().await?;
                let desktop = self.handle.state::<Desktop>();
                invocation.value["layers"] =
                    typesetting::report(&desktop, &snapshot, &[layer], false).await?;
                Ok(invocation)
            }
            "check_chapter" => {
                let arguments: typesetting::CheckChapterArguments = arguments(&call)?;
                let (source, target) = {
                    let config = koharu_pipeline::PipelineConfig::load()?;
                    let config = config.read()?;
                    (
                        config.translation.source_language,
                        config.translation.target_language,
                    )
                };
                let snapshot = self.snapshot().await?;
                let desktop = self.handle.state::<Desktop>();
                let check = output::chapter_check(&snapshot, &desktop, source, target).await?;
                Invocation::read(typesetting::chapter_issues(&snapshot, check, &arguments)?)
            }
            "chapter_text" => {
                let arguments: typesetting::ChapterText = arguments(&call)?;
                let snapshot = self.snapshot().await?;
                Invocation::read(typesetting::chapter_text(&snapshot, &arguments)?)
            }
            "run_pipeline" => self.run_pipeline(arguments(&call)?, control).await,
            name => bail!("unknown Koharu tool {name}"),
        }
    }
}

struct AgentCommitter {
    host: KoharuHost,
}

#[async_trait]
impl Committer for AgentCommitter {
    async fn commit(&mut self, output: StageOutput) -> Result<Snapshot> {
        let (commit, page) = {
            let current = self.host.handle.state::<CurrentProject>();
            let mut current = current.project.lock().await;
            let project = current.as_mut().context("no project is open")?;
            let Some(commit) = project.commit_stage_output(output).await? else {
                return Ok(project.snapshot());
            };
            (commit, project.active_page())
        };
        let snapshot = commit.snapshot.clone();
        self.host.synchronize(commit, page).await?;
        Ok(snapshot)
    }
}

fn definition<T: JsonSchema>(name: &str, description: &str) -> Tool {
    Tool::new(
        name,
        description,
        serde_json::to_value(schema_for!(T)).expect("tool argument schema must serialize"),
    )
}

fn arguments<T: for<'de> Deserialize<'de>>(call: &ToolCall) -> Result<T> {
    serde_json::from_str(&call.arguments)
        .with_context(|| format!("invalid arguments for {}", call.name))
}

pub(super) fn entity(value: &str) -> Result<EntityId> {
    serde_json::from_value(Value::String(value.to_owned()))
        .with_context(|| format!("invalid entity ID {value}"))
}

fn entities(values: &[String]) -> Result<Vec<EntityId>> {
    values.iter().map(|value| entity(value)).collect()
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct InspectProject {}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ViewPage {
    page: String,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RenamePage {
    page: String,
    label: String,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct MovePage {
    page: String,
    index: usize,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct DeletePages {
    pages: Vec<String>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct AddTextBox {
    page: String,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    #[serde(default)]
    angle_degrees: f32,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SetText {
    element: String,
    text: String,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SetTranslation {
    element: String,
    text: Option<String>,
}

#[derive(Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(super) enum AgentTextAlignment {
    Start,
    Center,
    End,
    Justify,
}

impl From<AgentTextAlignment> for koharu_scene::TextAlignment {
    fn from(value: AgentTextAlignment) -> Self {
        match value {
            AgentTextAlignment::Start => Self::Start,
            AgentTextAlignment::Center => Self::Center,
            AgentTextAlignment::End => Self::End,
            AgentTextAlignment::Justify => Self::Justify,
        }
    }
}

#[derive(Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(super) enum AgentWritingMode {
    Horizontal,
    Vertical,
}

#[derive(Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(super) enum AgentLineBreaking {
    /// Wrap between words only, never inside one.
    WholeWords,
    /// Keep the given lines: break only at line breaks in the text. Auto-fit
    /// shrinks the text to keep them; a fixed size that cannot overflows.
    Manual,
}

impl From<AgentLineBreaking> for koharu_scene::LineBreaking {
    fn from(value: AgentLineBreaking) -> Self {
        match value {
            AgentLineBreaking::WholeWords => Self::WholeWords,
            AgentLineBreaking::Manual => Self::Manual,
        }
    }
}

#[derive(Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(super) enum AgentFontStyle {
    Normal,
    Italic,
    Oblique,
}

impl From<AgentFontStyle> for koharu_scene::FontStyle {
    fn from(value: AgentFontStyle) -> Self {
        match value {
            AgentFontStyle::Normal => Self::Normal,
            AgentFontStyle::Italic => Self::Italic,
            AgentFontStyle::Oblique => Self::Oblique,
        }
    }
}

impl From<AgentWritingMode> for koharu_scene::WritingMode {
    fn from(value: AgentWritingMode) -> Self {
        match value {
            AgentWritingMode::Horizontal => Self::Horizontal,
            AgentWritingMode::Vertical => Self::Vertical,
        }
    }
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SetTypography {
    element: String,
    preferred_font: Option<String>,
    font_weight: Option<u16>,
    font_style: Option<AgentFontStyle>,
    size: Option<f32>,
    auto_fit: bool,
    color: Option<[u8; 4]>,
    stroke_color: Option<[u8; 4]>,
    stroke_width: Option<f32>,
    alignment: Option<AgentTextAlignment>,
    writing_mode: Option<AgentWritingMode>,
    /// Where lines may break; automatic (hyphenating a word as a last
    /// resort) when absent.
    #[serde(default)]
    line_breaking: Option<AgentLineBreaking>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SetGeometry {
    element: String,
    points: Vec<AgentPoint>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct AgentPoint {
    x: f64,
    y: f64,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SetVisibility {
    elements: Vec<String>,
    visible: Option<bool>,
    opacity: Option<f32>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct DeleteElements {
    elements: Vec<String>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct MoveElement {
    element: String,
    parent: String,
    index: usize,
}

#[derive(Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum AgentPipelineOperation {
    Full,
    Detection,
    Ocr,
    Markup,
    Translation,
    Inpainting,
}

impl From<AgentPipelineOperation> for Operation {
    fn from(value: AgentPipelineOperation) -> Self {
        match value {
            AgentPipelineOperation::Full => Self::Full,
            AgentPipelineOperation::Detection => Self::Only {
                stage: Stage::Detection,
            },
            AgentPipelineOperation::Ocr => Self::Only { stage: Stage::Ocr },
            AgentPipelineOperation::Markup => Self::Only {
                stage: Stage::Markup,
            },
            AgentPipelineOperation::Translation => Self::Only {
                stage: Stage::Translation,
            },
            AgentPipelineOperation::Inpainting => Self::Only {
                stage: Stage::Inpainting,
            },
        }
    }
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RunPipeline {
    operation: AgentPipelineOperation,
    #[serde(default)]
    pages: Vec<String>,
    #[serde(default)]
    elements: Vec<String>,
}

impl RunPipeline {
    fn scope(&self) -> Result<Scope> {
        match (self.pages.is_empty(), self.elements.is_empty()) {
            (true, true) => Ok(Scope::Project),
            (false, true) => Ok(Scope::Pages(entities(&self.pages)?)),
            (true, false) => Ok(Scope::Entities(entities(&self.elements)?)),
            (false, false) => bail!("pipeline scope cannot contain both pages and elements"),
        }
    }
}
