//! Tiling around the fully convolutional network. Each tile carries a margin
//! of real neighbouring pixels, so its edges see the context a whole-page
//! pass would; only the tile's core is kept. The receptive field of the
//! network is far wider than the margin, but past a few pixels the
//! contribution is negligible, which the comparison against a whole-crop pass
//! checks.

use anyhow::{Result, ensure};
use image::{GrayImage, Luma};
use koharu_torch::{Device, Kind, Tensor};

use super::model::Model;

const TILE: u32 = 512;
const MARGIN: u32 = 16;

pub(super) fn restore(
    model: &Model,
    device: Device,
    image: &GrayImage,
    stopped: &dyn Fn() -> bool,
) -> Result<Option<GrayImage>> {
    let (width, height) = image.dimensions();
    ensure!(width > 0 && height > 0, "image dimensions must be non-zero");
    let mut output = GrayImage::new(width, height);
    for top in (0..height).step_by(TILE as usize) {
        for left in (0..width).step_by(TILE as usize) {
            if stopped() {
                return Ok(None);
            }
            let right = (left + TILE).min(width);
            let bottom = (top + TILE).min(height);
            let x0 = left.saturating_sub(MARGIN);
            let y0 = top.saturating_sub(MARGIN);
            let x1 = (right + MARGIN).min(width);
            let y1 = (bottom + MARGIN).min(height);
            let tile = image::imageops::crop_imm(image, x0, y0, x1 - x0, y1 - y0).to_image();
            let input = Tensor::from_slice(tile.as_raw())
                .view([1, 1, i64::from(y1 - y0), i64::from(x1 - x0)])
                .to_device(device)
                .to_kind(Kind::Float)
                / 255.0;
            let restored = (model.forward(&input).to_kind(Kind::Float) * 255.0)
                .round()
                .to_kind(Kind::Uint8)
                .to_device(Device::Cpu)
                .contiguous()
                .view([-1]);
            let pixels = Vec::<u8>::try_from(&restored)?;
            let tile_width = (x1 - x0) as usize;
            for y in top..bottom {
                let row = (y - y0) as usize * tile_width;
                for x in left..right {
                    output.put_pixel(x, y, Luma([pixels[row + (x - x0) as usize]]));
                }
            }
        }
    }
    Ok(Some(output))
}
