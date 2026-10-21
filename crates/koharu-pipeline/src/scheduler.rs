use std::collections::{BTreeMap, BTreeSet};

use koharu_scene::EntityId;

use crate::stage::OcrPass;
use crate::{Stage, StageTarget};

#[derive(Clone, Copy, Eq, PartialEq)]
enum WorkState {
    Pending,
    Running,
    Finished,
}

struct StageWork {
    stage: Stage,
    ocr_pass: OcrPass,
    state: WorkState,
}

struct PageWork {
    page: EntityId,
    stages: Vec<StageWork>,
}

impl PageWork {
    fn started(&self) -> bool {
        self.stages
            .iter()
            .any(|work| work.state != WorkState::Pending)
    }

    fn finished(&self) -> bool {
        self.stages
            .iter()
            .all(|work| work.state == WorkState::Finished)
    }

    fn ready(&self, index: usize) -> bool {
        let prerequisites = prerequisites(self.stages[index].stage);
        self.stages
            .iter()
            .filter(|work| prerequisites.contains(&work.stage))
            .all(|work| work.state == WorkState::Finished)
    }
}

pub(crate) struct Scheduler {
    pages: Vec<PageWork>,
    page_index: BTreeMap<EntityId, usize>,
    page_window: usize,
    active_pages: usize,
    head: usize,
    chapter: Option<StageWork>,
    project_scope: bool,
    parallel_inpainting: bool,
    ocr_requested: bool,
    total: usize,
}

impl Scheduler {
    pub(crate) fn new(
        pages: &[EntityId],
        stages: &[Stage],
        project_scope: bool,
        parallel_inpainting: bool,
    ) -> Self {
        let page_stages = stages
            .iter()
            .copied()
            .filter(|stage| !(project_scope && *stage == Stage::Translation))
            .collect::<Vec<_>>();
        let pages = pages
            .iter()
            .map(|page| PageWork {
                page: *page,
                stages: page_stages
                    .iter()
                    .map(|stage| StageWork {
                        stage: *stage,
                        ocr_pass: OcrPass::Final,
                        state: WorkState::Pending,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let chapter =
            (project_scope && stages.contains(&Stage::Translation)).then_some(StageWork {
                stage: Stage::Translation,
                ocr_pass: OcrPass::Final,
                state: WorkState::Pending,
            });
        let total = pages
            .len()
            .saturating_mul(page_stages.len())
            .saturating_add(usize::from(chapter.is_some()));
        let page_window = if parallel_inpainting && page_stages.contains(&Stage::Inpainting) {
            pages.len().max(1)
        } else {
            page_stages.len().max(1)
        };
        Self {
            page_index: pages
                .iter()
                .enumerate()
                .map(|(index, page)| (page.page, index))
                .collect(),
            pages,
            page_window,
            active_pages: 0,
            head: 0,
            chapter,
            project_scope,
            parallel_inpainting,
            ocr_requested: page_stages.contains(&Stage::Ocr),
            total,
        }
    }

    pub(crate) fn total(&self) -> usize {
        self.total
    }

    pub(crate) fn enable_hybrid_ocr(&mut self) {
        for page in &mut self.pages {
            if let Some(index) = page.stages.iter().position(|work| work.stage == Stage::Ocr) {
                page.stages.insert(
                    index,
                    StageWork {
                        stage: Stage::Ocr,
                        ocr_pass: OcrPass::Local,
                        state: WorkState::Pending,
                    },
                );
                self.total += 1;
            }
        }
    }

    pub(crate) fn ocr_pass(&self, target: StageTarget) -> OcrPass {
        target
            .page()
            .and_then(|page| self.page_index.get(&page))
            .and_then(|index| {
                self.pages[*index]
                    .stages
                    .iter()
                    .find(|work| work.stage == Stage::Ocr && work.state == WorkState::Running)
            })
            .map_or(OcrPass::Final, |work| work.ocr_pass)
    }

    pub(crate) const fn project_scope(&self) -> bool {
        self.project_scope
    }

    pub(crate) const fn parallel_inpainting(&self) -> bool {
        self.parallel_inpainting
    }

    pub(crate) fn start_next(
        &mut self,
        busy_stages: &BTreeSet<Stage>,
    ) -> Option<(StageTarget, Stage)> {
        if self.project_scope {
            return self.start_next_project(busy_stages);
        }

        for page_index in self.head..self.pages.len() {
            let started = self.pages[page_index].started();
            if !started && self.active_pages >= self.page_window {
                break;
            }
            let stage_index =
                self.pages[page_index]
                    .stages
                    .iter()
                    .enumerate()
                    .find_map(|(index, work)| {
                        (work.state == WorkState::Pending
                            && (self.parallel_inpainting && work.stage == Stage::Inpainting
                                || !busy_stages.contains(&work.stage))
                            && self.pages[page_index].ready(index))
                        .then_some(index)
                    });
            let Some(stage_index) = stage_index else {
                continue;
            };
            if !started {
                self.active_pages += 1;
            }
            let page = &mut self.pages[page_index];
            let work = &mut page.stages[stage_index];
            work.state = WorkState::Running;
            return Some((StageTarget::Page(page.page), work.stage));
        }

        let chapter_ready = !self.ocr_requested
            || self.pages.iter().all(|page| {
                page.stages
                    .iter()
                    .find(|work| work.stage == Stage::Ocr)
                    .is_none_or(|work| work.state == WorkState::Finished)
            });
        let chapter = self.chapter.as_mut()?;
        if chapter.state == WorkState::Pending
            && chapter_ready
            && !busy_stages.contains(&Stage::Translation)
        {
            chapter.state = WorkState::Running;
            return Some((StageTarget::Chapter, Stage::Translation));
        }
        None
    }

    fn start_next_project(
        &mut self,
        busy_stages: &BTreeSet<Stage>,
    ) -> Option<(StageTarget, Stage)> {
        let chapter_ready = self.chapter_ready();
        if chapter_ready
            && !busy_stages.contains(&Stage::Translation)
            && let Some(chapter) = self.chapter.as_mut()
            && chapter.state == WorkState::Pending
        {
            chapter.state = WorkState::Running;
            return Some((StageTarget::Chapter, Stage::Translation));
        }

        for page_index in 0..self.pages.len() {
            let stage_index =
                self.pages[page_index]
                    .stages
                    .iter()
                    .enumerate()
                    .find_map(|(index, work)| {
                        (work.state == WorkState::Pending
                            && (self.parallel_inpainting && work.stage == Stage::Inpainting
                                || !busy_stages.contains(&work.stage))
                            && self.page_stage_ready(work))
                        .then_some(index)
                    });
            let Some(stage_index) = stage_index else {
                continue;
            };
            let page = &mut self.pages[page_index];
            let work = &mut page.stages[stage_index];
            work.state = WorkState::Running;
            return Some((StageTarget::Page(page.page), work.stage));
        }
        None
    }

    fn chapter_ready(&self) -> bool {
        self.pages.iter().all(|page| {
            page.stages
                .iter()
                .filter(|work| work.stage < Stage::Translation)
                .all(|work| work.state == WorkState::Finished)
        })
    }

    fn page_stage_ready(&self, current: &StageWork) -> bool {
        let stage = current.stage;
        if stage == Stage::Inpainting
            && self
                .chapter
                .as_ref()
                .is_some_and(|chapter| chapter.state != WorkState::Finished)
        {
            return false;
        }
        self.pages.iter().all(|page| {
            page.stages
                .iter()
                .filter(|work| (work.stage, work.ocr_pass) < (stage, current.ocr_pass))
                .all(|work| work.state == WorkState::Finished)
        })
    }

    pub(crate) fn complete_stage(&mut self, target: StageTarget, stage: Stage) -> bool {
        let StageTarget::Page(page) = target else {
            if let Some(chapter) = &mut self.chapter
                && stage == Stage::Translation
            {
                chapter.state = WorkState::Finished;
            }
            return false;
        };
        let Some(&page_index) = self.page_index.get(&page) else {
            return false;
        };
        let page = &mut self.pages[page_index];
        let was_finished = page.finished();
        if let Some(work) = page
            .stages
            .iter_mut()
            .find(|work| work.stage == stage && work.state == WorkState::Running)
        {
            work.state = WorkState::Finished;
        }
        let page_finished = !was_finished && page.finished();
        if page_finished {
            self.active_pages = self.active_pages.saturating_sub(1);
            while self.head < self.pages.len() && self.pages[self.head].finished() {
                self.head += 1;
            }
        }
        page_finished
    }
}

/// The earlier stages of a page that a stage waits for when they are in the
/// same run. Translation uses whatever markup exists, so it waits for markup
/// only when the run includes it.
const fn prerequisites(stage: Stage) -> &'static [Stage] {
    match stage {
        Stage::Enhancement => &[],
        // Every later stage reads the enhanced page when the run makes one.
        Stage::Detection => &[Stage::Enhancement],
        Stage::Ocr | Stage::Inpainting => &[Stage::Enhancement, Stage::Detection],
        Stage::Markup => &[Stage::Enhancement, Stage::Ocr],
        Stage::Translation => &[Stage::Enhancement, Stage::Ocr, Stage::Markup],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pages(count: usize) -> Vec<EntityId> {
        (0..count).map(|_| EntityId::new()).collect()
    }

    #[test]
    fn hybrid_chapter_finishes_all_local_ocr_before_any_api_call() {
        let pages = pages(3);
        let mut scheduler = Scheduler::new(&pages, &[Stage::Ocr, Stage::Translation], true, false);
        scheduler.enable_hybrid_ocr();
        let busy = BTreeSet::new();
        assert_eq!(scheduler.total(), 7);
        for pass in [OcrPass::Local, OcrPass::Final] {
            for page in &pages {
                let work = scheduler.start_next(&busy).unwrap();
                assert_eq!(work, (StageTarget::Page(*page), Stage::Ocr));
                assert_eq!(scheduler.ocr_pass(work.0), pass);
                assert!(
                    scheduler
                        .start_next(&BTreeSet::from([Stage::Ocr]))
                        .is_none()
                );
                scheduler.complete_stage(work.0, work.1);
            }
        }
        assert_eq!(
            scheduler.start_next(&busy),
            Some((StageTarget::Chapter, Stage::Translation))
        );
    }

    #[test]
    fn hybrid_page_translation_waits_for_the_api_pass() {
        let pages = pages(1);
        let mut scheduler = Scheduler::new(&pages, &[Stage::Ocr, Stage::Translation], false, false);
        scheduler.enable_hybrid_ocr();
        let busy = BTreeSet::new();
        let local = scheduler.start_next(&busy).unwrap();
        scheduler.complete_stage(local.0, local.1);
        let api = scheduler.start_next(&busy).unwrap();
        assert_eq!(api.1, Stage::Ocr);
        assert_eq!(scheduler.ocr_pass(api.0), OcrPass::Final);
        assert!(
            scheduler
                .start_next(&BTreeSet::from([Stage::Ocr]))
                .is_none()
        );
        scheduler.complete_stage(api.0, api.1);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((StageTarget::Page(pages[0]), Stage::Translation))
        );
    }

    #[test]
    fn starts_pages_in_order_and_models_independently() {
        let pages = pages(2);
        let mut scheduler = Scheduler::new(&pages, &Stage::ALL, false, false);
        let mut busy = BTreeSet::new();

        let first = scheduler.start_next(&busy).unwrap();
        assert_eq!(first, (StageTarget::Page(pages[0]), Stage::Detection));
        busy.insert(Stage::Detection);
        assert!(scheduler.start_next(&busy).is_none());

        busy.clear();
        assert!(!scheduler.complete_stage(StageTarget::Page(pages[0]), Stage::Detection));
        let ocr = scheduler.start_next(&busy).unwrap();
        busy.insert(ocr.1);
        let inpainting = scheduler.start_next(&busy).unwrap();
        busy.insert(inpainting.1);
        let next_page = scheduler.start_next(&busy).unwrap();
        busy.insert(next_page.1);
        assert_eq!(ocr, (StageTarget::Page(pages[0]), Stage::Ocr));
        assert_eq!(inpainting, (StageTarget::Page(pages[0]), Stage::Inpainting));
        assert_eq!(next_page, (StageTarget::Page(pages[1]), Stage::Detection));

        assert!(!scheduler.complete_stage(StageTarget::Page(pages[0]), Stage::Ocr));
        busy.remove(&Stage::Ocr);
        let markup = scheduler.start_next(&busy).unwrap();
        assert_eq!(markup, (StageTarget::Page(pages[0]), Stage::Markup));
        assert!(busy.contains(&Stage::Detection));
        assert!(busy.contains(&Stage::Inpainting));
    }

    #[test]
    fn page_translation_waits_for_markup_only_when_it_is_in_the_run() {
        let pages = pages(1);
        let stages = [Stage::Ocr, Stage::Markup, Stage::Translation];
        let mut scheduler = Scheduler::new(&pages, &stages, false, false);
        let busy = BTreeSet::new();
        let ocr = scheduler.start_next(&busy).unwrap();
        scheduler.complete_stage(ocr.0, ocr.1);
        let markup = scheduler.start_next(&busy).unwrap();
        assert_eq!(markup, (StageTarget::Page(pages[0]), Stage::Markup));
        assert!(
            scheduler
                .start_next(&BTreeSet::from([Stage::Markup]))
                .is_none()
        );
        scheduler.complete_stage(markup.0, markup.1);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((StageTarget::Page(pages[0]), Stage::Translation))
        );

        let mut scheduler = Scheduler::new(&pages, &[Stage::Ocr, Stage::Translation], false, false);
        let ocr = scheduler.start_next(&busy).unwrap();
        scheduler.complete_stage(ocr.0, ocr.1);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((StageTarget::Page(pages[0]), Stage::Translation))
        );
    }

    #[test]
    fn chapter_translation_waits_for_markup_on_every_page() {
        let pages = pages(2);
        let mut scheduler = Scheduler::new(&pages, &Stage::ALL, true, false);
        let busy = BTreeSet::new();
        assert_eq!(scheduler.total(), 9);
        for stage in [Stage::Detection, Stage::Ocr, Stage::Markup] {
            for page in &pages {
                let work = scheduler.start_next(&busy).unwrap();
                assert_eq!(work, (StageTarget::Page(*page), stage));
                assert!(scheduler.start_next(&BTreeSet::from([stage])).is_none());
                scheduler.complete_stage(work.0, work.1);
            }
        }
        let chapter = scheduler.start_next(&busy).unwrap();
        assert_eq!(chapter, (StageTarget::Chapter, Stage::Translation));
        assert!(
            scheduler
                .start_next(&BTreeSet::from([Stage::Translation]))
                .is_none()
        );
        scheduler.complete_stage(chapter.0, chapter.1);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((StageTarget::Page(pages[0]), Stage::Inpainting))
        );
    }

    #[test]
    fn markup_alone_runs_each_page_in_project_order() {
        let pages = pages(2);
        let mut scheduler = Scheduler::new(&pages, &[Stage::Markup], true, false);
        let busy = BTreeSet::new();
        assert_eq!(scheduler.total(), 2);
        for page in &pages {
            let work = scheduler.start_next(&busy).unwrap();
            assert_eq!(work, (StageTarget::Page(*page), Stage::Markup));
            scheduler.complete_stage(work.0, work.1);
        }
        assert!(scheduler.start_next(&busy).is_none());
    }

    #[test]
    fn api_cleaning_pages_can_start_in_parallel_for_both_scopes() {
        for project_scope in [false, true] {
            let pages = pages(2);
            let mut scheduler = Scheduler::new(&pages, &[Stage::Inpainting], project_scope, true);
            let mut busy = BTreeSet::new();

            assert_eq!(
                scheduler.start_next(&busy),
                Some((StageTarget::Page(pages[0]), Stage::Inpainting))
            );
            busy.insert(Stage::Inpainting);
            assert_eq!(
                scheduler.start_next(&busy),
                Some((StageTarget::Page(pages[1]), Stage::Inpainting))
            );
        }

        let pages = pages(2);
        let mut local = Scheduler::new(&pages, &[Stage::Inpainting], false, false);
        let mut busy = BTreeSet::new();
        assert_eq!(
            local.start_next(&busy),
            Some((StageTarget::Page(pages[0]), Stage::Inpainting))
        );
        busy.insert(Stage::Inpainting);
        assert!(local.start_next(&busy).is_none());
    }

    #[test]
    fn parallel_cleaning_keeps_other_models_backpressured() {
        let pages = pages(4);
        let stages = [Stage::Detection, Stage::Ocr, Stage::Inpainting];
        let mut scheduler = Scheduler::new(&pages, &stages, false, true);
        let mut busy = BTreeSet::new();

        assert_eq!(
            scheduler.start_next(&busy),
            Some((StageTarget::Page(pages[0]), Stage::Detection))
        );
        assert!(!scheduler.complete_stage(StageTarget::Page(pages[0]), Stage::Detection));
        let ocr = scheduler.start_next(&busy).unwrap();
        busy.insert(ocr.1);
        let inpainting = scheduler.start_next(&busy).unwrap();
        busy.insert(inpainting.1);

        for page in &pages[1..] {
            assert_eq!(
                scheduler.start_next(&busy),
                Some((StageTarget::Page(*page), Stage::Detection))
            );
            assert!(!scheduler.complete_stage(StageTarget::Page(*page), Stage::Detection));
            assert_eq!(
                scheduler.start_next(&busy),
                Some((StageTarget::Page(*page), Stage::Inpainting))
            );
        }
        assert!(scheduler.start_next(&busy).is_none());

        assert!(!scheduler.complete_stage(StageTarget::Page(pages[0]), Stage::Ocr));
        assert!(scheduler.complete_stage(StageTarget::Page(pages[0]), Stage::Inpainting));
        busy.remove(&Stage::Ocr);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((StageTarget::Page(pages[1]), Stage::Ocr))
        );
    }

    #[test]
    fn schedules_one_chapter_translation_after_every_page_ocr() {
        let pages = pages(2);
        let stages = [Stage::Detection, Stage::Ocr, Stage::Translation];
        let mut scheduler = Scheduler::new(&pages, &stages, true, false);
        let busy = BTreeSet::new();

        assert_eq!(scheduler.total(), 5);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((StageTarget::Page(pages[0]), Stage::Detection))
        );
        scheduler.complete_stage(StageTarget::Page(pages[0]), Stage::Detection);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((StageTarget::Page(pages[1]), Stage::Detection))
        );
        scheduler.complete_stage(StageTarget::Page(pages[1]), Stage::Detection);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((StageTarget::Page(pages[0]), Stage::Ocr))
        );
        scheduler.complete_stage(StageTarget::Page(pages[0]), Stage::Ocr);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((StageTarget::Page(pages[1]), Stage::Ocr))
        );
        scheduler.complete_stage(StageTarget::Page(pages[1]), Stage::Ocr);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((StageTarget::Chapter, Stage::Translation))
        );
    }

    #[test]
    fn chapter_translation_precedes_parallel_project_inpainting() {
        let pages = pages(2);
        let stages = [
            Stage::Detection,
            Stage::Ocr,
            Stage::Translation,
            Stage::Inpainting,
        ];
        let mut scheduler = Scheduler::new(&pages, &stages, true, true);
        let busy = BTreeSet::new();

        for page in &pages {
            assert_eq!(
                scheduler.start_next(&busy),
                Some((StageTarget::Page(*page), Stage::Detection))
            );
            scheduler.complete_stage(StageTarget::Page(*page), Stage::Detection);
        }
        for page in &pages {
            assert_eq!(
                scheduler.start_next(&busy),
                Some((StageTarget::Page(*page), Stage::Ocr))
            );
            scheduler.complete_stage(StageTarget::Page(*page), Stage::Ocr);
        }

        assert_eq!(
            scheduler.start_next(&busy),
            Some((StageTarget::Chapter, Stage::Translation))
        );
        scheduler.complete_stage(StageTarget::Chapter, Stage::Translation);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((StageTarget::Page(pages[0]), Stage::Inpainting))
        );
        assert_eq!(
            scheduler.start_next(&busy),
            Some((StageTarget::Page(pages[1]), Stage::Inpainting))
        );
    }
}
