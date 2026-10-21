use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use koharu_ml::manga_jpeg::{MangaJpeg, MangaJpegVariant};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Variant {
    HqPlus,
    Hq,
    Mq,
    Lq,
}

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,

    #[arg(short, long, value_name = "FILE")]
    output: PathBuf,

    #[arg(long, value_enum, default_value = "hq")]
    variant: Variant,

    #[arg(long, default_value_t = false)]
    cpu: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    let image = image::open(&cli.input)?.to_luma8();
    let variant = match cli.variant {
        Variant::HqPlus => MangaJpegVariant::HqPlus,
        Variant::Hq => MangaJpegVariant::Hq,
        Variant::Mq => MangaJpegVariant::Mq,
        Variant::Lq => MangaJpegVariant::Lq,
    };

    koharu_ml::init().await?;
    let model = MangaJpeg::load(variant, koharu_ml::device(cli.cpu)).await?;
    let started = std::time::Instant::now();
    let restored = model
        .inference(&image, &|| false)?
        .expect("inference was not stopped");
    tracing::info!(elapsed = ?started.elapsed(), "restored");
    restored.save(cli.output)?;
    Ok(())
}
