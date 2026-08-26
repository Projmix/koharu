'use client'

import { useMemo, useRef } from 'react'

import { SelectionControls } from '@/components/editor/SelectionControls'
import {
  effectiveLayerVisibility,
  expandLayerSelection,
  isTextLayer,
  ocrNumbering,
} from '@/lib/document'
import {
  controlFrame,
  cssFrame,
  framePoints,
  geometryFrame,
  selectableLayer,
  type Camera,
} from '@/lib/geometry'
import type {
  AnalysisRegion,
  EntityId,
  Frame,
  Geometry,
  Page,
  Point,
  TransformFrame,
} from '@koharu/bridge/protocol'

interface CanvasOverlayProps {
  page: Page
  camera: Camera
  selected: EntityId[]
  hovered: EntityId | null
  frames: Readonly<Record<EntityId, Frame>>
  previews: Readonly<Record<EntityId, Frame>>
  draft: Frame | null
  cursor: Point | null
  brushSize: number
  showBrushCursor: boolean
  showDetectionRegions: boolean
  onTransformStart: (elements: TransformFrame[]) => void
  onTransformFrame: (elements: TransformFrame[]) => void
  onTransformEnd: () => void
}

export function CanvasOverlay({
  page,
  camera,
  selected,
  hovered,
  frames,
  previews,
  draft,
  cursor,
  brushSize,
  showBrushCursor,
  showDetectionRegions,
  onTransformStart,
  onTransformFrame,
  onTransformEnd,
}: CanvasOverlayProps) {
  const root = useRef<HTMLDivElement>(null)
  const expandedSelection = useMemo(
    () => expandLayerSelection(page.layers, selected),
    [page.layers, selected],
  )
  const selectedIds = useMemo(() => new Set(expandedSelection), [expandedSelection])
  const { regionNumbers } = useMemo(() => ocrNumbering(page.layers), [page.layers])
  const multipleSelected = expandedSelection.length > 1
  const layers = useMemo(
    () =>
      page.layers.flatMap((layer) => {
        const visibility = effectiveLayerVisibility(page.layers, layer)
        if (!visibility.visible || visibility.opacity <= 0) return []
        const frame = previews[layer.id] ?? controlFrame(layer, frames)
        return frame ? [{ layer, frame, opacity: visibility.opacity }] : []
      }),
    [page.layers, previews, frames],
  )
  const selectedLayer =
    expandedSelection.length === 1
      ? page.layers.find((layer) => layer.id === expandedSelection[0])
      : undefined
  const selectedTextLayer = selectedLayer && isTextLayer(selectedLayer) ? selectedLayer : undefined
  const automaticRegion = selectedTextLayer?.automatic_region
    ? page.regions.find((region) => region.id === selectedTextLayer.automatic_region)
    : undefined
  const sourceRegion = selectedTextLayer?.content.source_region
    ? page.regions.find((region) => region.id === selectedTextLayer.content.source_region)
    : undefined
  const sourceRegionFrame = sourceRegion
    ? (previews[sourceRegion.id] ?? geometryFrame(sourceRegion.geometry))
    : undefined
  const selectionControl = multipleSelected
    ? undefined
    : sourceRegion && sourceRegionFrame
      ? { element: sourceRegion.id, frame: sourceRegionFrame }
      : layers
          .filter(({ layer }) => selectedIds.has(layer.id) && selectableLayer(layer))
          .map(({ layer, frame }) => ({ element: layer.id, frame }))[0]
  const scale = camera.zoom / window.devicePixelRatio

  return (
    <div
      ref={root}
      data-testid='canvas-overlay'
      className='pointer-events-none absolute inset-0 overflow-hidden'
      aria-hidden
    >
      {showDetectionRegions && (
        <DetectionRegions
          regions={page.regions}
          camera={camera}
          previews={previews}
          numbers={regionNumbers}
        />
      )}
      {automaticRegion && (
        <AutomaticRegionOverlay geometry={automaticRegion.geometry} camera={camera} />
      )}
      {layers.map(({ layer, frame, opacity }) => {
        const position = cssFrame(frame, camera)
        const selected = selectedIds.has(layer.id) && selectableLayer(layer)
        const highlighted = !selected && hovered === layer.id
        return (
          <div
            key={layer.id}
            data-element={layer.id}
            className='absolute box-border bg-transparent'
            style={{
              left: position.left,
              top: position.top,
              width: position.width,
              height: position.height,
              transform: `rotate(${position.angle}deg)`,
              transformOrigin: '50% 50%',
              border:
                highlighted || (selected && multipleSelected)
                  ? '1px solid var(--canvas-selection)'
                  : undefined,
              opacity,
              willChange: selected ? 'left, top, width, height, transform' : undefined,
            }}
          />
        )
      })}

      {draft && <DraftOverlay frame={draft} camera={camera} />}
      {showBrushCursor && cursor && (
        <div
          className='absolute rounded-full border border-white/95 shadow-[0_0_0_1px_rgb(0_0_0/0.9),0_1px_3px_rgb(0_0_0/0.45)]'
          style={{
            left: cursor.x / window.devicePixelRatio - (brushSize * scale) / 2,
            top: cursor.y / window.devicePixelRatio - (brushSize * scale) / 2,
            width: brushSize * scale,
            height: brushSize * scale,
          }}
        />
      )}

      {selectionControl && (
        <SelectionControls
          container={root}
          element={selectionControl.element}
          frame={selectionControl.frame}
          camera={camera}
          edgesOnly={Boolean(selectedTextLayer && !sourceRegion)}
          onTransformStart={onTransformStart}
          onTransformFrame={onTransformFrame}
          onTransformEnd={onTransformEnd}
        />
      )}
    </div>
  )
}

function DetectionRegions({
  regions,
  camera,
  previews,
  numbers,
}: {
  regions: AnalysisRegion[]
  camera: Camera
  previews: Readonly<Record<EntityId, Frame>>
  numbers: ReadonlyMap<EntityId, number>
}) {
  if (!regions.length) return null

  return (
    <svg className='absolute inset-0 size-full overflow-hidden' data-testid='detection-regions'>
      {regions.map((region) => {
        const frame = previews[region.id] ?? geometryFrame(region.geometry)
        const points = previews[region.id]
          ? geometryPointsFromPoints(framePoints(previews[region.id]), camera)
          : geometryPoints(region.geometry, camera)
        if (!points) return null
        const number = numbers.get(region.id)
        const position = frame ? cssFrame(frame, camera) : null
        const numberX = position ? position.left + 7 : 0
        const numberY = position
          ? position.top >= 18
            ? position.top - 9
            : position.top + position.height + 9
          : 0
        const color = regionColor(region.kind)
        return (
          <g key={region.id}>
            <polygon
              data-region-kind={region.kind}
              points={points}
              fill='none'
              stroke={color}
              strokeWidth='2'
              strokeDasharray={region.kind.endsWith('.panel') ? undefined : '6 4'}
              strokeLinejoin='round'
              opacity='0.8'
              vectorEffect='non-scaling-stroke'
            />
            {number !== undefined && position && (
              <g
                data-testid={`ocr-region-number-${region.id}`}
                transform={`translate(${numberX} ${numberY})`}
              >
                <circle r='7' fill={color} opacity='0.95' />
                <text
                  fill='white'
                  fontSize='9'
                  fontWeight='700'
                  textAnchor='middle'
                  dominantBaseline='central'
                >
                  {number}
                </text>
              </g>
            )}
          </g>
        )
      })}
    </svg>
  )
}

function regionColor(kind: string) {
  if (kind === 'text' || kind.endsWith('.text')) return '#22c55e'
  if (kind === 'bubble' || kind.endsWith('.bubble')) return '#f59e0b'
  if (kind === 'panel' || kind.endsWith('.panel')) return '#a855f7'
  return 'var(--canvas-region-stroke)'
}

function geometryPoints(geometry: Geometry, camera: Camera) {
  return geometryPointsFromPoints(geometry.points, camera)
}

function geometryPointsFromPoints(points: Point[], camera: Camera) {
  const dpr = window.devicePixelRatio
  return points
    .map(
      (point) =>
        `${(point.x * camera.zoom + camera.translation[0]) / dpr},${(point.y * camera.zoom + camera.translation[1]) / dpr}`,
    )
    .join(' ')
}

function AutomaticRegionOverlay({ geometry, camera }: { geometry: Geometry; camera: Camera }) {
  const points = geometryPoints(geometry, camera)
  if (!points) return null

  return (
    <svg className='absolute inset-0 size-full overflow-hidden' data-testid='text-fit-region'>
      <polygon
        points={points}
        fill='none'
        stroke='var(--canvas-region-contrast)'
        strokeWidth='7'
        strokeDasharray='10 6'
        strokeLinecap='round'
        strokeLinejoin='round'
        vectorEffect='non-scaling-stroke'
      />
      <polygon
        points={points}
        fill='none'
        stroke='var(--canvas-region-stroke)'
        strokeWidth='3'
        strokeDasharray='10 6'
        strokeLinecap='round'
        strokeLinejoin='round'
        vectorEffect='non-scaling-stroke'
      />
    </svg>
  )
}

function DraftOverlay({ frame, camera }: { frame: Frame; camera: Camera }) {
  const position = cssFrame(frame, camera)
  return (
    <div
      className='absolute border border-dashed border-primary bg-primary/5'
      style={{
        left: position.left,
        top: position.top,
        width: position.width,
        height: position.height,
        transform: `rotate(${position.angle}deg)`,
      }}
    />
  )
}
