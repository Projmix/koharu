use super::*;

#[test]
fn configuration_ignores_unknown_fields() {
    let config = toml::from_str::<PipelineConfig>("legacy_limit = 1").unwrap();
    assert_eq!(config, PipelineConfig::default());
}

#[tokio::test]
async fn stop_is_a_successful_partial_result() {
    let pipeline = pipeline(Default::default());
    let stop = StopToken::default();
    stop.stop();
    let request = Request {
        stop,
        ..Request::default()
    };
    let mut committer = RejectCommitter;

    let report = pipeline
        .execute(
            koharu_scene::Session::memory().await.unwrap().snapshot(),
            request,
            &mut committer,
        )
        .await
        .unwrap();

    assert_eq!(report.status, RunStatus::Stopped);
    assert_eq!(report.completed, 0);
}

#[tokio::test]
async fn stop_after_a_page_keeps_completed_progress() {
    let translation = TranslationConfig {
        page: crate::TranslationProfile {
            model: koharu_translator::ModelSelection {
                provider: koharu_translator::Provider::OpenAi,
                model: Some("gpt-5.6-luna".to_owned()),
                quantization: None,
                vision: true,
                reasoning: true,
                reasoning_required: false,
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let pipeline = pipeline(translation);
    let mut session = koharu_scene::Session::memory().await.unwrap();
    let mut pages = Vec::new();
    let patch = session
        .snapshot()
        .patch(|edit| {
            pages.push(edit.add_page(
                koharu_scene::PageDraft::new("one", 1.0, 1.0),
                koharu_scene::At::End,
            )?);
            pages.push(edit.add_page(
                koharu_scene::PageDraft::new("two", 1.0, 1.0),
                koharu_scene::At::End,
            )?);
            Ok(())
        })
        .unwrap();
    session.commit(patch).await.unwrap();
    let stop = StopToken::default();
    let progress_stop = stop.clone();
    let request = Request {
        operation: Operation::Only {
            stage: Stage::Translation,
        },
        scope: Scope::Pages(pages),
        stop,
        progress: Some(std::sync::Arc::new(move |event| {
            if matches!(event, Progress::Skipped { .. }) {
                progress_stop.stop();
            }
        })),
        ..Request::default()
    };
    let mut committer = RejectCommitter;

    let report = pipeline
        .execute(session.snapshot(), request, &mut committer)
        .await
        .unwrap();

    assert_eq!(report.status, RunStatus::Stopped);
    assert_eq!(report.completed, 1);
    assert_eq!(report.total, 2);
}

#[tokio::test]
async fn selected_stages_resume_by_skipping_completed_work() {
    let pipeline = pipeline(Default::default());
    let mut session = koharu_scene::Session::memory().await.unwrap();
    let mut page = None;
    let patch = session
        .snapshot()
        .patch(|edit| {
            let id = edit.add_page(
                koharu_scene::PageDraft::new("page", 100.0, 100.0),
                koharu_scene::At::End,
            )?;
            edit.add_analysis_region::<koharu_scene::TextRegion>(
                id,
                koharu_scene::At::End,
                &koharu_scene::Geometry::rectangle(10.0, 10.0, 20.0, 20.0),
                None,
            )?;
            page = Some(id);
            Ok(())
        })
        .unwrap();
    let snapshot = session.commit(patch).await.unwrap().snapshot;
    let page = page.unwrap();
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let captured = events.clone();
    let request = Request {
        operation: Operation::Stages {
            stages: vec![Stage::Detection],
        },
        scope: Scope::Pages(vec![page]),
        progress: Some(std::sync::Arc::new(move |event| {
            captured.lock().unwrap().push(event);
        })),
        ..Request::default()
    };
    let mut committer = RejectCommitter;

    let report = pipeline
        .execute(snapshot, request, &mut committer)
        .await
        .unwrap();

    assert_eq!(report.status, RunStatus::Completed);
    assert_eq!(report.completed, 1);
    assert!(events.lock().unwrap().iter().any(|event| matches!(
        event,
        Progress::Skipped {
            target: StageTarget::Page(target),
            stage: Stage::Detection,
        } if *target == page
    )));
}

struct RejectCommitter;

fn pipeline(translation: TranslationConfig) -> Pipeline {
    let config = PipelineConfig {
        translation,
        ..PipelineConfig::default()
    };
    Pipeline::from_config(
        koharu_config::Config::memory(config),
        koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        koharu_ml::Device::cpu(),
    )
    .unwrap()
}

#[async_trait::async_trait]
impl Committer for RejectCommitter {
    async fn commit(&mut self, _output: StageOutput) -> anyhow::Result<koharu_scene::Snapshot> {
        anyhow::bail!("stopped execution must not commit")
    }
}

#[test]
fn operations_expand_to_the_supported_workflows() {
    assert_eq!(
        Operation::Through {
            stage: Stage::Translation,
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Ocr, Stage::Translation],
    );
    assert_eq!(
        Operation::Through {
            stage: Stage::Inpainting,
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Inpainting],
    );
    assert_eq!(
        Operation::Only {
            stage: Stage::Translation,
        }
        .stages()
        .unwrap(),
        vec![Stage::Translation],
    );
    assert_eq!(
        Operation::Stages {
            stages: vec![Stage::Translation, Stage::Detection, Stage::Translation],
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Translation],
    );
}
