use std::collections::{BTreeMap, BTreeSet};

use koharu_scene::EntityId;

use crate::{Stage, StageTarget};

#[derive(Clone, Copy, Eq, PartialEq)]
enum WorkState {
    Pending,
    Running,
    Finished,
}

struct StageWork {
    stage: Stage,
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
        let Some(prerequisite) = prerequisite(self.stages[index].stage) else {
            return true;
        };
        self.stages
            .iter()
            .find(|work| work.stage == prerequisite)
            .is_none_or(|work| work.state == WorkState::Finished)
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
    ocr_requested: bool,
    total: usize,
}

impl Scheduler {
    pub(crate) fn new(pages: &[EntityId], stages: &[Stage], project_scope: bool) -> Self {
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
                        state: WorkState::Pending,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let chapter =
            (project_scope && stages.contains(&Stage::Translation)).then_some(StageWork {
                stage: Stage::Translation,
                state: WorkState::Pending,
            });
        let total = pages
            .len()
            .saturating_mul(page_stages.len())
            .saturating_add(usize::from(chapter.is_some()));
        Self {
            page_index: pages
                .iter()
                .enumerate()
                .map(|(index, page)| (page.page, index))
                .collect(),
            pages,
            page_window: page_stages.len().max(1),
            active_pages: 0,
            head: 0,
            chapter,
            project_scope,
            ocr_requested: page_stages.contains(&Stage::Ocr),
            total,
        }
    }

    pub(crate) fn total(&self) -> usize {
        self.total
    }

    pub(crate) const fn project_scope(&self) -> bool {
        self.project_scope
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
                            && !busy_stages.contains(&work.stage)
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
        if chapter_ready && !busy_stages.contains(&Stage::Translation) {
            if let Some(chapter) = self.chapter.as_mut()
                && chapter.state == WorkState::Pending
            {
                chapter.state = WorkState::Running;
                return Some((StageTarget::Chapter, Stage::Translation));
            }
        }

        for page_index in 0..self.pages.len() {
            let stage_index =
                self.pages[page_index]
                    .stages
                    .iter()
                    .enumerate()
                    .find_map(|(index, work)| {
                        (work.state == WorkState::Pending
                            && !busy_stages.contains(&work.stage)
                            && self.page_stage_ready(work.stage))
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

    fn page_stage_ready(&self, stage: Stage) -> bool {
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
                .filter(|work| work.stage < stage)
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
        if let Some(work) = page.stages.iter_mut().find(|work| work.stage == stage) {
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

const fn prerequisite(stage: Stage) -> Option<Stage> {
    match stage {
        Stage::Detection => None,
        Stage::Ocr | Stage::Inpainting => Some(Stage::Detection),
        Stage::Translation => Some(Stage::Ocr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pages(count: usize) -> Vec<EntityId> {
        (0..count).map(|_| EntityId::new()).collect()
    }

    #[test]
    fn starts_pages_in_order_and_models_independently() {
        let pages = pages(2);
        let mut scheduler = Scheduler::new(&pages, &Stage::ALL, false);
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
        let translation = scheduler.start_next(&busy).unwrap();
        assert_eq!(
            translation,
            (StageTarget::Page(pages[0]), Stage::Translation)
        );
        assert!(busy.contains(&Stage::Detection));
        assert!(busy.contains(&Stage::Inpainting));
    }

    #[test]
    fn sliding_window_backpressures_fast_upstream_models() {
        let pages = pages(4);
        let stages = [Stage::Detection, Stage::Ocr, Stage::Inpainting];
        let mut scheduler = Scheduler::new(&pages, &stages, false);
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

        for page in &pages[1..3] {
            assert_eq!(
                scheduler.start_next(&busy),
                Some((StageTarget::Page(*page), Stage::Detection))
            );
            assert!(!scheduler.complete_stage(StageTarget::Page(*page), Stage::Detection));
        }
        assert!(scheduler.start_next(&busy).is_none());

        assert!(!scheduler.complete_stage(StageTarget::Page(pages[0]), Stage::Ocr));
        assert!(scheduler.complete_stage(StageTarget::Page(pages[0]), Stage::Inpainting));
        busy.clear();
        assert_eq!(
            scheduler.start_next(&busy),
            Some((StageTarget::Page(pages[1]), Stage::Ocr))
        );
    }

    #[test]
    fn schedules_one_chapter_translation_after_every_page_ocr() {
        let pages = pages(2);
        let stages = [Stage::Detection, Stage::Ocr, Stage::Translation];
        let mut scheduler = Scheduler::new(&pages, &stages, true);
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
    fn chapter_translation_precedes_sequential_project_inpainting() {
        let pages = pages(2);
        let stages = [
            Stage::Detection,
            Stage::Ocr,
            Stage::Translation,
            Stage::Inpainting,
        ];
        let mut scheduler = Scheduler::new(&pages, &stages, true);
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
        scheduler.complete_stage(StageTarget::Page(pages[0]), Stage::Inpainting);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((StageTarget::Page(pages[1]), Stage::Inpainting))
        );
    }
}
