use super::*;

fn png(image: DynamicImage) -> Arc<[u8]> {
    let mut bytes = Cursor::new(Vec::new());
    image.write_to(&mut bytes, ImageFormat::Png).unwrap();
    Arc::from(bytes.into_inner())
}

async fn removal_scene() -> (koharu_scene::Session, StageInput) {
    let mut session = koharu_scene::Session::memory().await.unwrap();
    let mut page = None;
    let patch = session
        .snapshot()
        .patch(|edit| {
            let id = edit.add_page(koharu_scene::PageDraft::new("Page", 4.0, 1.0), At::End)?;
            let asset = |image: RgbaImage| {
                AssetInput::new(
                    png(DynamicImage::ImageRgba8(image)),
                    "image/png",
                    AssetMetadata {
                        width: Some(4),
                        height: Some(1),
                        attributes: BTreeMap::new(),
                    },
                )
            };
            edit.set_asset(
                id,
                &AssetRole::new("source")?,
                asset(RgbaImage::from_pixel(4, 1, Rgba([10, 20, 30, 255]))),
            )?;
            for (kind, x, color, opacity, visible) in [
                (RasterLayerKind::Cleanup, 1, [0, 255, 0, 255], 1.0, false),
                (RasterLayerKind::Cleanup, 0, [200, 0, 0, 255], 0.5, true),
                (RasterLayerKind::Paint, 0, [0, 0, 200, 255], 0.5, true),
                (RasterLayerKind::Paint, 2, [200, 100, 0, 255], 1.0, true),
                (RasterLayerKind::Cleanup, 2, [40, 50, 60, 255], 1.0, true),
                (RasterLayerKind::Paint, 1, [255, 0, 255, 255], 1.0, false),
                (RasterLayerKind::Cleanup, 3, [255, 255, 0, 255], 0.0, true),
            ] {
                let entity = edit.add_entity(id, At::End)?;
                edit.set(
                    entity,
                    &RasterLayer {
                        origin: Origin::User,
                        name: "Layer".to_owned(),
                        kind,
                    },
                )?;
                edit.set(
                    entity,
                    &Visibility {
                        origin: Origin::User,
                        visible,
                        opacity,
                    },
                )?;
                let mut pixels = RgbaImage::new(4, 1);
                pixels.put_pixel(x, 0, Rgba(color));
                edit.set_asset(entity, &AssetRole::new("source")?, asset(pixels))?;
            }
            edit.ensure_text_group(id)?;
            page = Some(id);
            Ok(())
        })
        .unwrap();
    let scene = session.commit(patch).await.unwrap().snapshot;
    let page = page.unwrap();
    let input = StageInput::new(
        scene,
        page,
        None,
        None,
        Arc::new(crate::ImageCache::default()),
        Some(crate::InpaintingMask {
            page,
            png: png(DynamicImage::ImageLuma8(GrayImage::from_pixel(
                4,
                1,
                Luma([255]),
            ))),
            intent: InpaintingIntent::Remove,
        }),
    );

    (session, input)
}

#[tokio::test]
async fn removal_uses_visible_raster_layers_in_order_with_opacity() {
    let (_, input) = removal_scene().await;
    let prepared = prepare(&input, false, false).await.unwrap();
    assert_eq!(
        prepared.image.to_rgba8().as_raw(),
        &[
            52, 4, 107, 254, 10, 20, 30, 255, 40, 50, 60, 255, 10, 20, 30, 255
        ],
    );
    // Removal must cover painted pixels, unlike automatic background cleanup.
    assert!(prepared.paint.is_none());
}

#[tokio::test]
async fn removal_results_stack_above_rasters_and_preserve_existing_layers_and_unmasked_pixels() {
    let (mut session, mut input) = removal_scene().await;
    let page = input.page().unwrap();
    let original = input.scene.clone();
    let original_children = original.children(page).unwrap().collect::<Vec<_>>();
    let text_group = original
        .page(page)
        .unwrap()
        .text_group()
        .unwrap()
        .unwrap()
        .id();
    let mut new_layers = Vec::new();
    for (x, color) in [(0, [9, 8, 7, 255]), (2, [6, 5, 4, 255])] {
        let mut mask = GrayImage::new(4, 1);
        mask.put_pixel(x, 0, Luma([255]));
        input.inpainting_mask.as_mut().unwrap().png = png(DynamicImage::ImageLuma8(mask));
        let prepared = prepare(&input, false, false).await.unwrap();
        let target = CleanupTarget {
            dimensions: prepared.original.dimensions(),
            mask: prepared.mask,
            paint: prepared.paint,
            restore_bounds: prepared.restore_bounds,
            flat_fill: prepared.flat_fill,
        };
        let patch = target
            .patch(
                &input,
                "lama",
                &RgbaImage::from_pixel(4, 1, Rgba(color)),
                false,
            )
            .unwrap();
        input.scene = session.commit(patch).await.unwrap().snapshot;
        let children = input.scene.children(page).unwrap().collect::<Vec<_>>();
        let added = children
            .iter()
            .copied()
            .find(|id| !original_children.contains(id) && !new_layers.contains(id))
            .unwrap();
        new_layers.push(added);
        let layer = input
            .images
            .get(&input.scene, added, "source")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(layer.get_pixel(x, 0), Rgba(color));
        assert_eq!(layer.get_pixel(1, 0), Rgba([0, 0, 0, 0]));
        assert_eq!(children[children.len() - 2], added);
        assert_eq!(children.last(), Some(&text_group));
    }

    let prepared = prepare(&input, false, false).await.unwrap();
    assert_eq!(
        prepared.image.to_rgba8().as_raw(),
        &[9, 8, 7, 255, 10, 20, 30, 255, 6, 5, 4, 255, 10, 20, 30, 255],
    );
    for entity in original_children {
        assert_eq!(
            input
                .scene
                .asset(entity, &AssetRole::new("source").unwrap())
                .unwrap(),
            original
                .asset(entity, &AssetRole::new("source").unwrap())
                .unwrap(),
        );
        assert_eq!(
            input.scene.component::<Visibility>(entity).unwrap(),
            original.component::<Visibility>(entity).unwrap(),
        );
        assert_eq!(
            input.scene.component::<RasterLayer>(entity).unwrap(),
            original.component::<RasterLayer>(entity).unwrap(),
        );
    }
}

#[tokio::test]
async fn removal_respects_group_visibility_and_opacity() {
    let (mut session, mut input) = removal_scene().await;
    let page = input.page().unwrap();
    let paint = input.scene.children(page).unwrap().nth(2).unwrap();
    let mut group = None;
    let patch = input
        .scene
        .patch(|edit| {
            let id = edit.add_entity(page, At::Before(paint))?;
            edit.set(
                id,
                &koharu_scene::Group {
                    origin: Origin::User,
                    name: "Paint group".to_owned(),
                },
            )?;
            edit.move_entity(paint, Some(id), At::End)?;
            group = Some(id);
            Ok(())
        })
        .unwrap();
    input.scene = session.commit(patch).await.unwrap().snapshot;
    let group = group.unwrap();
    for (visible, opacity, expected) in [
        (true, 0.0, [105, 9, 14]),
        (false, 1.0, [105, 9, 14]),
        (true, 0.5, [78, 7, 60]),
    ] {
        let patch = input
            .scene
            .patch(|edit| {
                edit.set(
                    group,
                    &Visibility {
                        origin: Origin::User,
                        visible,
                        opacity,
                    },
                )
            })
            .unwrap();
        input.scene = session.commit(patch).await.unwrap().snapshot;
        let prepared = prepare(&input, false, false).await.unwrap();
        let actual = prepared.image.get_pixel(0, 0);
        for (actual, expected) in actual.0[..3].iter().zip(expected) {
            assert!(
                actual.abs_diff(expected) <= 1,
                "pixel {actual:?} differs from {expected:?}"
            );
        }
    }
}

#[tokio::test]
async fn repeated_automatic_cleanup_is_visible_and_preserves_layers_before_manual_removal() {
    let (mut session, mut input) = removal_scene().await;
    let page = input.page().unwrap();
    let original = input.scene.clone();
    let original_children = original.children(page).unwrap().collect::<Vec<_>>();
    let patch = input
        .scene
        .patch(|edit| {
            edit.set_asset(
                page,
                &AssetRole::new("text-mask")?,
                AssetInput::new(
                    png(DynamicImage::ImageLuma8(GrayImage::from_pixel(
                        4,
                        1,
                        Luma([255]),
                    ))),
                    "image/png",
                    AssetMetadata {
                        width: Some(4),
                        height: Some(1),
                        attributes: BTreeMap::new(),
                    },
                ),
            )
        })
        .unwrap();
    input.scene = session.commit(patch).await.unwrap().snapshot;
    input.inpainting_mask = None;
    for color in [[9, 8, 7, 255], [6, 5, 4, 255]] {
        let prepared = prepare(&input, true, false).await.unwrap();
        let target = CleanupTarget {
            dimensions: prepared.original.dimensions(),
            mask: prepared.mask,
            paint: prepared.paint,
            restore_bounds: prepared.restore_bounds,
            flat_fill: prepared.flat_fill,
        };
        let patch = target
            .patch(
                &input,
                "test-api",
                &RgbaImage::from_pixel(4, 1, Rgba(color)),
                false,
            )
            .unwrap();
        input.scene = session.commit(patch).await.unwrap().snapshot;
        let mut removal_input = input.clone();
        removal_input.inpainting_mask = Some(crate::InpaintingMask {
            page,
            png: png(DynamicImage::ImageLuma8(GrayImage::new(4, 1))),
            intent: InpaintingIntent::Remove,
        });
        let composite = prepare(&removal_input, false, false).await.unwrap().image;
        assert_eq!(composite.get_pixel(1, 0), Rgba(color));
        assert_eq!(composite.get_pixel(3, 0), Rgba(color));
        assert_eq!(composite.get_pixel(0, 0), Rgba([52, 4, 107, 254]));
        assert_eq!(composite.get_pixel(2, 0), Rgba([40, 50, 60, 255]));
    }
    for entity in original_children {
        assert_eq!(
            input
                .scene
                .asset(entity, &AssetRole::new("source").unwrap())
                .unwrap(),
            original
                .asset(entity, &AssetRole::new("source").unwrap())
                .unwrap()
        );
        assert_eq!(
            input.scene.component::<Visibility>(entity).unwrap(),
            original.component::<Visibility>(entity).unwrap()
        );
    }

    let mut mask = GrayImage::new(4, 1);
    mask.put_pixel(0, 0, Luma([255]));
    input.inpainting_mask = Some(crate::InpaintingMask {
        page,
        png: png(DynamicImage::ImageLuma8(mask)),
        intent: InpaintingIntent::Remove,
    });
    let prepared = prepare(&input, false, false).await.unwrap();
    let target = CleanupTarget {
        dimensions: prepared.original.dimensions(),
        mask: prepared.mask,
        paint: prepared.paint,
        restore_bounds: prepared.restore_bounds,
        flat_fill: prepared.flat_fill,
    };
    let patch = target
        .patch(
            &input,
            "lama",
            &RgbaImage::from_pixel(4, 1, Rgba([3, 2, 1, 255])),
            false,
        )
        .unwrap();
    input.scene = session.commit(patch).await.unwrap().snapshot;
    let composite = prepare(&input, false, false).await.unwrap().image;
    assert_eq!(composite.get_pixel(0, 0), Rgba([3, 2, 1, 255]));
    assert_eq!(composite.get_pixel(1, 0), Rgba([6, 5, 4, 255]));
    assert_eq!(composite.get_pixel(2, 0), Rgba([40, 50, 60, 255]));
    assert_eq!(composite.get_pixel(3, 0), Rgba([6, 5, 4, 255]));
}
