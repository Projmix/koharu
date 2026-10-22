//! Times page preparation on a copy of a real project: how long a page takes to
//! prepare from nothing, and how long the next page takes after a text edit on
//! the presented one, carried forward and rendered again.
//!
//! `cargo run --release -p koharu-desktop --example page_switch -- <project.khrproj>`

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result};
use koharu_desktop::Desktop;
use koharu_scene::{Authored, Session, Translation};

#[tokio::main]
async fn main() -> Result<()> {
    let source = PathBuf::from(
        std::env::args()
            .nth(1)
            .context("usage: page_switch <project.khrproj>")?,
    );
    let copy = std::env::temp_dir().join(format!("koharu-page-switch-{}", std::process::id()));
    copy_dir(&source, &copy)?;
    let result = run(&copy).await;
    let _ = std::fs::remove_dir_all(&copy);
    result
}

async fn run(project: &Path) -> Result<()> {
    let mut session = Session::open(project).await?;
    let snapshot = session.snapshot();
    let pages = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
    anyhow::ensure!(pages.len() >= 3, "the project needs at least three pages");
    let desktop = Desktop::new()?;

    // Fonts and images load on first use; one page warms them for the rest.
    desktop.show_page(&snapshot, Some(pages[0])).await?;
    let mut fresh = Vec::new();
    for &page in &pages[1..] {
        let started = Instant::now();
        desktop.renderer().render(&snapshot, page).await?;
        fresh.push(started.elapsed());
    }
    report("prepare a page from nothing", &fresh);

    let mut after_edit = Vec::new();
    let mut rendered_again = Vec::new();
    for window in pages.windows(2).take(10) {
        let (presented, next) = (window[0], window[1]);
        let snapshot = session.snapshot();
        desktop.show_page(&snapshot, Some(presented)).await?;
        desktop.prepare_page(&snapshot, next).await?;
        let Some(text) = snapshot
            .descendants(presented)?
            .find(|entity| matches!(entity.component::<Translation>(), Ok(Some(_))))
            .map(|entity| entity.id())
        else {
            continue;
        };
        let translation = snapshot
            .component::<Translation>(text)?
            .expect("found above");
        let edit = snapshot.patch(|edit| {
            edit.set(
                text,
                &Translation {
                    text: Authored::user(format!("{} ", translation.text.value)),
                    language: translation.language.clone(),
                },
            )
        })?;
        let commit = session.commit(edit).await?;
        desktop
            .synchronize(&commit.snapshot, Some(presented), &commit)
            .await?;
        let started = Instant::now();
        desktop.prepare_page(&commit.snapshot, next).await?;
        after_edit.push(started.elapsed());
        // What the preparation cost before prepared pages were carried forward.
        let started = Instant::now();
        desktop.renderer().render(&commit.snapshot, next).await?;
        rendered_again.push(started.elapsed());
    }
    report("next page after an edit, rendered again", &rendered_again);
    report("next page after an edit, carried forward", &after_edit);
    Ok(())
}

fn report(label: &str, times: &[Duration]) {
    if times.is_empty() {
        println!("{label}: no samples");
        return;
    }
    let mut ms = times
        .iter()
        .map(|time| time.as_secs_f64() * 1000.0)
        .collect::<Vec<_>>();
    ms.sort_by(f64::total_cmp);
    println!(
        "{label}: median {:.1} ms, max {:.1} ms, {} pages",
        ms[ms.len() / 2],
        ms[ms.len() - 1],
        ms.len()
    );
}

fn copy_dir(from: &Path, to: &Path) -> Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else if entry.file_name() != "project.lock" {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}
