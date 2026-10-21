use std::sync::Arc;

use anyhow::{Context as _, Result};
use arc_swap::ArcSwap;
use koharu_config::Config;
use koharu_scene::Snapshot;

use crate::{
    Committer, PipelineConfig, PipelineError, Report, Request, ResourceSnapshot, Stage, StageProgress,
    execution::Execution, resources::ResourceMonitor, stage_runner::StageRunner,
};

#[derive(Clone)]
pub struct Pipeline {
    current: Arc<ArcSwap<StageRunner>>,
    translator: koharu_translator::Translator,
    resources: Arc<ResourceMonitor>,
    execution: Arc<tokio::sync::Mutex<()>>,
}

impl Pipeline {
    pub fn load(device: koharu_ml::Device) -> Result<Self> {
        Self::from_config(
            PipelineConfig::load()?,
            koharu_translator::ProvidersConfig::load()?,
            device,
        )
    }

    #[tracing::instrument(skip_all)]
    pub fn from_config(
        config: Config<PipelineConfig>,
        providers: Config<koharu_translator::ProvidersConfig>,
        device: koharu_ml::Device,
    ) -> Result<Self> {
        let translator = koharu_translator::Translator::from_config(device.clone(), providers)?;
        let resources = ResourceMonitor::new(&device);
        let runner = {
            let value = config.read()?;
            StageRunner::new(&value, translator.clone(), &device, resources.clone())?
        };
        let current = Arc::new(ArcSwap::from_pointee(runner));
        let watched = current.clone();
        let watched_resources = resources.clone();
        let stage_translator = translator.clone();
        let _watcher = tokio::runtime::Handle::try_current()
            .context("pipeline requires a Tokio runtime")?
            .spawn(async move {
                let mut changes = config.subscribe();
                while changes.changed().await.is_ok() {
                    let runner = config.read().and_then(|value| {
                        StageRunner::new(
                            &value,
                            stage_translator.clone(),
                            &device,
                            watched_resources.clone(),
                        )
                    });
                    match runner {
                        Ok(runner) => watched.store(Arc::new(runner)),
                        Err(error) => tracing::error!(%error, "failed to reload pipeline"),
                    }
                }
            });
        Ok(Self {
            current,
            translator,
            resources,
            execution: Arc::new(tokio::sync::Mutex::new(())),
        })
    }

    /// The translator shared by every stage, for work that runs outside a
    /// pipeline job, such as importing a manually translated chapter.
    pub fn translator(&self) -> &koharu_translator::Translator {
        &self.translator
    }

    pub fn subscribe_resources(&self) -> tokio::sync::watch::Receiver<ResourceSnapshot> {
        self.resources.start();
        self.resources.subscribe()
    }

    /// What each stage has left to do on a project with the current settings.
    #[must_use]
    pub fn progress(&self, snapshot: &Snapshot) -> Vec<StageProgress> {
        self.current.load().progress(snapshot)
    }

    /// The pages, in project order, that a stage still has work on.
    #[must_use]
    pub fn unfinished(&self, snapshot: &Snapshot, stage: Stage) -> Vec<koharu_scene::EntityId> {
        self.current.load().unfinished(snapshot, stage)
    }

    #[tracing::instrument(skip_all)]
    pub async fn execute(
        &self,
        snapshot: Snapshot,
        request: Request,
        committer: &mut dyn Committer,
    ) -> std::result::Result<Report, PipelineError> {
        let _execution = self.execution.lock().await;
        Execution::new(
            self.current.load_full(),
            self.resources.clone(),
            snapshot,
            request,
            committer,
        )?
        .run()
        .await
    }
}
