//! Old-architecture ESRGAN `RRDBNet` at 1x, in the key layout the MangaJPEG
//! checkpoints use.
//!
//! Mirrors `RRDBNet` in BasicSR's pre-2020 `codes/models/modules/architecture.py`
//! as spandrel's ESRGAN loader reads it
//! (https://github.com/chaiNNer-org/spandrel/blob/main/libs/spandrel/spandrel/architectures/ESRGAN/__init__.py):
//! `model.0` is the first conv, `model.1.sub.N.RDB{1,2,3}.conv{1..5}.0` the
//! dense blocks, `model.1.sub.{blocks}` the body conv inside the shortcut,
//! and without an upsampler `model.2` (conv, LeakyReLU at `model.3`) and
//! `model.4` the head. MangaJPEG: 1 channel, 32 features, growth 32, 12 blocks.

use std::path::Path;

use anyhow::Result;
use koharu_torch::{
    Device, Tensor,
    nn::{self, Module},
};

const CHANNELS: i64 = 1;
const FEATURES: i64 = 32;
const GROWTH: i64 = 32;
const BLOCKS: i64 = 12;
const RESIDUAL_SCALE: f64 = 0.2;
const NEGATIVE_SLOPE: f64 = 0.2;

fn conv(path: &nn::Path<'_>, input: i64, output: i64) -> nn::Conv2D {
    nn::conv2d(
        path,
        input,
        output,
        3,
        nn::ConvConfig {
            padding: 1,
            ..Default::default()
        },
    )
}

/// LeakyReLU with slope 0.2; the binding's `leaky_relu` has a fixed 0.01.
fn leaky(x: &Tensor) -> Tensor {
    x.maximum(&(x * NEGATIVE_SLOPE))
}

#[derive(Debug)]
struct DenseBlock {
    convs: [nn::Conv2D; 5],
}

impl DenseBlock {
    fn new(path: &nn::Path<'_>) -> Self {
        Self {
            convs: std::array::from_fn(|index| {
                let index = index as i64;
                let output = if index == 4 { FEATURES } else { GROWTH };
                conv(
                    &(path / format!("conv{}", index + 1) / 0),
                    FEATURES + index * GROWTH,
                    output,
                )
            }),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut features = vec![x.shallow_clone()];
        for conv in &self.convs[..4] {
            let next = leaky(&conv.forward(&Tensor::cat(&features, 1)));
            features.push(next);
        }
        self.convs[4].forward(&Tensor::cat(&features, 1)) * RESIDUAL_SCALE + x
    }
}

#[derive(Debug)]
struct ResidualInResidual {
    blocks: [DenseBlock; 3],
}

impl ResidualInResidual {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.shallow_clone();
        for block in &self.blocks {
            out = block.forward(&out);
        }
        out * RESIDUAL_SCALE + x
    }
}

#[derive(Debug)]
pub(super) struct Model {
    vs: nn::VarStore,
    first: nn::Conv2D,
    body: Vec<ResidualInResidual>,
    body_tail: nn::Conv2D,
    hr: nn::Conv2D,
    last: nn::Conv2D,
}

impl Model {
    pub(super) fn new(device: Device) -> Self {
        let mut vs = nn::VarStore::new(device);
        crate::backend::set_precision(&mut vs);
        let root = vs.root();
        let model = &root / "model";
        let sub = &model / 1 / "sub";
        let first = conv(&(&model / 0), CHANNELS, FEATURES);
        let body = (0..BLOCKS)
            .map(|index| ResidualInResidual {
                blocks: std::array::from_fn(|block| {
                    DenseBlock::new(&(&sub / index / format!("RDB{}", block + 1)))
                }),
            })
            .collect();
        let body_tail = conv(&(&sub / BLOCKS), FEATURES, FEATURES);
        let hr = conv(&(&model / 2), FEATURES, FEATURES);
        let last = conv(&(&model / 4), FEATURES, CHANNELS);
        vs.freeze();
        Self {
            vs,
            first,
            body,
            body_tail,
            hr,
            last,
        }
    }

    pub(super) fn load(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.vs.load(path)?;
        crate::backend::set_precision(&mut self.vs);
        Ok(())
    }

    /// `x` is `[N, 1, H, W]` in `0..=1`; the output has the same shape.
    pub(super) fn forward(&self, x: &Tensor) -> Tensor {
        let x = x.to_kind(self.vs.kind());
        let first = self.first.forward(&x);
        let mut feature = first.shallow_clone();
        for block in &self.body {
            feature = block.forward(&feature);
        }
        let feature = self.body_tail.forward(&feature) + first;
        self.last
            .forward(&leaky(&self.hr.forward(&feature)))
            .clamp(0.0, 1.0)
    }
}
