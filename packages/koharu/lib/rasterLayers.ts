'use client'

import { call } from '@/lib/backend'
import { pageKey, pagesKey, projectKey, queryClient, refresh } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import { commands, type EntityId, type Layer, type Page } from '@koharu/bridge/protocol'

export type RasterLayer = Extract<Layer, { type: 'raster' }>

/** Paint and cleanup layers; scan enhancement replaces the page source instead. */
export function isEditableRaster(layer: Layer): layer is RasterLayer {
  return layer.type === 'raster' && layer.kind !== 'enhancement'
}

/** The selected layers Merge Layers takes: one merges down, more merge together. */
export function mergeableSelection(page: Page | null | undefined, selected: EntityId[]) {
  const layers = selected.map((id) => page?.layers.find((layer) => layer.id === id))
  if (layers.length === 0 || !layers.every((layer) => layer && isEditableRaster(layer))) return null
  return selected
}

export function duplicableSelection(page: Page | null | undefined, selected: EntityId[]) {
  const layer =
    selected.length === 1 ? page?.layers.find((value) => value.id === selected[0]) : null
  return layer && isEditableRaster(layer) ? layer.id : null
}

export function duplicateLayer(layer: EntityId): void {
  void call(commands.duplicateLayer, layer)
    .then((copy) => {
      useKoharuStore.getState().selectLayers([copy])
      return refresh(projectKey, pagesKey, pageKey)
    })
    .catch(() => undefined)
}

export function mergeLayers(layers: EntityId[]): void {
  void call(commands.mergeLayers, layers)
    .then((merged) => {
      useKoharuStore.getState().selectLayers([merged])
      return refresh(projectKey, pagesKey, pageKey)
    })
    .catch(() => undefined)
}

/** Duplicate Layer (Ctrl+J) and Merge Layers (Ctrl+E) on the current selection. */
export function runLayerShortcut(key: 'j' | 'e'): boolean {
  const page = queryClient.getQueryData<Page | null>(pageKey)
  const selected = useKoharuStore.getState().selectedLayers
  if (key === 'j') {
    const layer = duplicableSelection(page, selected)
    if (layer) duplicateLayer(layer)
    return layer !== null
  }
  const layers = mergeableSelection(page, selected)
  if (layers) mergeLayers(layers)
  return layers !== null
}

export function setRevealMask(layer: EntityId, enabled: boolean): void {
  void call(commands.setRevealMask, layer, enabled)
    .then(() => refresh(projectKey, pagesKey, pageKey))
    .catch(() => undefined)
}

export function toggleLayerTint(layer: EntityId): void {
  useKoharuStore.setState((state) => ({
    tintedLayers: state.tintedLayers.includes(layer)
      ? state.tintedLayers.filter((id) => id !== layer)
      : [...state.tintedLayers, layer],
  }))
}
