import type {
  AnalysisRegion,
  EntityId,
  Frame,
  Geometry,
  Layer,
  Point,
  TransformFrame,
} from '@koharu/bridge/protocol'

import { effectiveLayerVisibility, isTextLayer } from './document'

const minimumFrameSize = 1e-6

export interface Camera {
  zoom: number
  translation: [number, number]
}

export interface CssFrame {
  left: number
  top: number
  width: number
  height: number
  angle: number
}

export type ResizeHandle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w'

const resizeDirections: Record<ResizeHandle, Point> = {
  nw: { x: -1, y: -1 },
  n: { x: 0, y: -1 },
  ne: { x: 1, y: -1 },
  e: { x: 1, y: 0 },
  se: { x: 1, y: 1 },
  s: { x: 0, y: 1 },
  sw: { x: -1, y: 1 },
  w: { x: -1, y: 0 },
}

export function physicalPoint(clientX: number, clientY: number, bounds: DOMRect): Point {
  const dpr = window.devicePixelRatio
  return { x: (clientX - bounds.x) * dpr, y: (clientY - bounds.y) * dpr }
}

export function pagePoint(
  clientX: number,
  clientY: number,
  bounds: DOMRect,
  camera: Camera,
): Point {
  const point = physicalPoint(clientX, clientY, bounds)
  return {
    x: (point.x - camera.translation[0]) / camera.zoom,
    y: (point.y - camera.translation[1]) / camera.zoom,
  }
}

export function draftFrame(start: Point, end: Point): Frame {
  return {
    x: Math.min(start.x, end.x),
    y: Math.min(start.y, end.y),
    width: Math.max(1, Math.abs(end.x - start.x)),
    height: Math.max(1, Math.abs(end.y - start.y)),
    angle_degrees: 0,
  }
}

export function selectableLayer(layer: Layer): boolean {
  return layer.type === 'text' || layer.type === 'image'
}

export function layerFrame(layer: Layer): Frame | null {
  const points =
    layer.type === 'text' || layer.type === 'image' || layer.type === 'artwork'
      ? layer.geometry?.points
      : null
  return points ? frameFromPoints(points) : null
}

export function geometryFrame(geometry: Geometry): Frame | null {
  return frameFromPoints(geometry.points)
}

export function framePoints(frame: Frame): Point[] {
  const radians = (frame.angle_degrees * Math.PI) / 180
  const cos = Math.cos(radians)
  const sin = Math.sin(radians)
  const center = { x: frame.x + frame.width * 0.5, y: frame.y + frame.height * 0.5 }
  return [
    {
      x: center.x - frame.width * 0.5 * cos + frame.height * 0.5 * sin,
      y: center.y - frame.width * 0.5 * sin - frame.height * 0.5 * cos,
    },
    {
      x: center.x + frame.width * 0.5 * cos + frame.height * 0.5 * sin,
      y: center.y + frame.width * 0.5 * sin - frame.height * 0.5 * cos,
    },
    {
      x: center.x + frame.width * 0.5 * cos - frame.height * 0.5 * sin,
      y: center.y + frame.width * 0.5 * sin + frame.height * 0.5 * cos,
    },
    {
      x: center.x - frame.width * 0.5 * cos - frame.height * 0.5 * sin,
      y: center.y - frame.width * 0.5 * sin + frame.height * 0.5 * cos,
    },
  ]
}

function frameFromPoints(points: Point[]): Frame | null {
  if (!points.length || points.some((point) => !finite(point.x, point.y))) return null
  if (points.length === 4) {
    const [topLeft, topRight, bottomRight, bottomLeft] = points
    const top: [number, number] = [topRight.x - topLeft.x, topRight.y - topLeft.y]
    const right: [number, number] = [bottomRight.x - topRight.x, bottomRight.y - topRight.y]
    const bottom: [number, number] = [bottomLeft.x - bottomRight.x, bottomLeft.y - bottomRight.y]
    const left: [number, number] = [topLeft.x - bottomLeft.x, topLeft.y - bottomLeft.y]
    const width = Math.hypot(...top)
    const height = Math.hypot(...right)
    if (width > minimumFrameSize && height > minimumFrameSize) {
      const scale = Math.max(width, height, 1)
      const oppositeLengthsMatch =
        Math.abs(Math.hypot(...bottom) - width) <= scale * 1e-6 &&
        Math.abs(Math.hypot(...left) - height) <= scale * 1e-6
      const perpendicular = Math.abs(top[0] * right[0] + top[1] * right[1]) <= width * height * 1e-6
      const diagonalsBisect =
        Math.abs(topLeft.x + bottomRight.x - topRight.x - bottomLeft.x) <= scale * 1e-6 &&
        Math.abs(topLeft.y + bottomRight.y - topRight.y - bottomLeft.y) <= scale * 1e-6
      if (oppositeLengthsMatch && perpendicular && diagonalsBisect) {
        const centerX = points.reduce((sum, point) => sum + point.x, 0) * 0.25
        const centerY = points.reduce((sum, point) => sum + point.y, 0) * 0.25
        return {
          x: centerX - width * 0.5,
          y: centerY - height * 0.5,
          width,
          height,
          angle_degrees: (Math.atan2(top[1], top[0]) * 180) / Math.PI,
        }
      }
    }
  }

  const xs = points.map((point) => point.x)
  const ys = points.map((point) => point.y)
  const x = Math.min(...xs)
  const y = Math.min(...ys)
  const width = Math.max(...xs) - x
  const height = Math.max(...ys) - y
  return width > minimumFrameSize && height > minimumFrameSize
    ? { x, y, width, height, angle_degrees: 0 }
    : null
}

export function controlFrame(
  layer: Layer,
  frames: Readonly<Record<EntityId, Frame>>,
): Frame | null {
  const frame = isTextLayer(layer) ? frames[layer.id] : undefined
  return frame && validFrame(frame) ? frame : layerFrame(layer)
}

export function hitTestLayers(
  layers: Layer[],
  point: Point,
  frames: Readonly<Record<EntityId, Frame>>,
): Layer | null {
  for (let index = layers.length - 1; index >= 0; index -= 1) {
    const layer = layers[index]
    const visibility = effectiveLayerVisibility(layers, layer)
    if (!selectableLayer(layer) || !visibility.visible || visibility.opacity <= 0) continue
    const frame = controlFrame(layer, frames)
    if (frame && frameContains(frame, point)) return layer
  }
  return null
}

export function hitTestEditorSelection(
  layers: Layer[],
  regions: AnalysisRegion[],
  point: Point,
  frames: Readonly<Record<EntityId, Frame>>,
): Layer | null {
  const byId = new Map(regions.map((region) => [region.id, region]))
  type Candidate = {
    layer: Layer
    /** Source OCR geometry is more precise than a containing bubble. */
    priority: 0 | 1 | 2
    area: number
    distance: number
    /** Later entries are visually on top and win otherwise. */
    zIndex: number
  }

  const candidates: Candidate[] = []
  for (let index = 0; index < layers.length; index += 1) {
    const layer = layers[index]
    if (!isTextLayer(layer)) continue
    const visibility = effectiveLayerVisibility(layers, layer)
    if (!visibility.visible || visibility.opacity <= 0) continue

    const source = layer.content.source_region ? byId.get(layer.content.source_region) : undefined
    const automatic = layer.automatic_region ? byId.get(layer.automatic_region) : undefined
    const frame = controlFrame(layer, frames)

    // A click on the detected glyph/text polygon should always resolve to
    // that dialog, even when another bubble happens to contain the same
    // point.  OCR polygons can overlap at their edges, so the smallest
    // polygon is the most specific candidate and wins the tie-breaker below.
    if (source && polygonContains(source.geometry.points, point)) {
      candidates.push({
        layer,
        priority: 0,
        area: polygonArea(source.geometry.points),
        distance: distanceSquared(polygonCenter(source.geometry.points), point),
        zIndex: index,
      })
    }

    // Automatic regions are normally speech bubbles.  Several text layers
    // may share one bubble, so do not blindly return the last layer: choose
    // the text frame closest to the click (and use the source polygon as a
    // fallback for layers that have no rendered frame yet).
    if (automatic && polygonContains(automatic.geometry.points, point)) {
      const anchor = frame
        ? { x: frame.x + frame.width * 0.5, y: frame.y + frame.height * 0.5 }
        : source
          ? polygonCenter(source.geometry.points)
          : polygonCenter(automatic.geometry.points)
      candidates.push({
        layer,
        priority: 1,
        area: frame ? frame.width * frame.height : polygonArea(automatic.geometry.points),
        distance: distanceSquared(anchor, point),
        zIndex: index,
      })
    }

    if (frame && frameContains(frame, point)) {
      candidates.push({
        layer,
        priority: 2,
        area: frame.width * frame.height,
        distance: distanceSquared(
          { x: frame.x + frame.width * 0.5, y: frame.y + frame.height * 0.5 },
          point,
        ),
        zIndex: index,
      })
    }
  }

  candidates.sort(
    (left, right) =>
      left.priority - right.priority ||
      left.area - right.area ||
      left.distance - right.distance ||
      right.zIndex - left.zIndex,
  )
  return candidates[0]?.layer ?? hitTestLayers(layers, point, frames)
}

export function frameContains(frame: Frame, point: Point): boolean {
  const centerX = frame.x + frame.width * 0.5
  const centerY = frame.y + frame.height * 0.5
  const angle = (-frame.angle_degrees * Math.PI) / 180
  const cos = Math.cos(angle)
  const sin = Math.sin(angle)
  const x = point.x - centerX
  const y = point.y - centerY
  const localX = x * cos - y * sin
  const localY = x * sin + y * cos
  return Math.abs(localX) <= frame.width * 0.5 && Math.abs(localY) <= frame.height * 0.5
}

function polygonContains(points: Point[], point: Point): boolean {
  if (points.length < 3) return false
  let inside = false
  for (let index = 0, previous = points.length - 1; index < points.length; previous = index++) {
    const start = points[previous]
    const end = points[index]
    const cross = (point.x - start.x) * (end.y - start.y) - (point.y - start.y) * (end.x - start.x)
    if (
      Math.abs(cross) <= minimumFrameSize &&
      point.x >= Math.min(start.x, end.x) &&
      point.x <= Math.max(start.x, end.x) &&
      point.y >= Math.min(start.y, end.y) &&
      point.y <= Math.max(start.y, end.y)
    ) {
      return true
    }
    if (
      start.y > point.y !== end.y > point.y &&
      point.x < ((end.x - start.x) * (point.y - start.y)) / (end.y - start.y) + start.x
    ) {
      inside = !inside
    }
  }
  return inside
}

function polygonArea(points: Point[]): number {
  if (points.length < 3) return Number.POSITIVE_INFINITY
  let area = 0
  for (let index = 0, previous = points.length - 1; index < points.length; previous = index++) {
    const start = points[previous]
    const end = points[index]
    area += start.x * end.y - end.x * start.y
  }
  const result = Math.abs(area) * 0.5
  return Number.isFinite(result) && result > minimumFrameSize ? result : Number.POSITIVE_INFINITY
}

function polygonCenter(points: Point[]): Point {
  if (!points.length) return { x: Number.POSITIVE_INFINITY, y: Number.POSITIVE_INFINITY }
  const area = polygonSignedArea(points)
  if (Math.abs(area) <= minimumFrameSize || !Number.isFinite(area)) {
    return {
      x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
      y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
    }
  }
  let x = 0
  let y = 0
  for (let index = 0, previous = points.length - 1; index < points.length; previous = index++) {
    const start = points[previous]
    const end = points[index]
    const cross = start.x * end.y - end.x * start.y
    x += (start.x + end.x) * cross
    y += (start.y + end.y) * cross
  }
  const scale = 1 / (6 * area)
  return { x: x * scale, y: y * scale }
}

function polygonSignedArea(points: Point[]): number {
  let area = 0
  for (let index = 0, previous = points.length - 1; index < points.length; previous = index++) {
    area += points[previous].x * points[index].y - points[index].x * points[previous].y
  }
  return area * 0.5
}

function distanceSquared(left: Point, right: Point): number {
  const x = left.x - right.x
  const y = left.y - right.y
  const distance = x * x + y * y
  return Number.isFinite(distance) ? distance : Number.POSITIVE_INFINITY
}

export function cssFrame(frame: Frame, camera: Camera): CssFrame {
  const dpr = window.devicePixelRatio
  const scale = camera.zoom / dpr
  return {
    left: (frame.x * camera.zoom + camera.translation[0]) / dpr,
    top: (frame.y * camera.zoom + camera.translation[1]) / dpr,
    width: frame.width * scale,
    height: frame.height * scale,
    angle: frame.angle_degrees,
  }
}

export function resizeFrame(
  frame: Frame,
  handle: ResizeHandle,
  point: Point,
  minimumSize: number,
): Frame {
  const direction = resizeDirections[handle]
  const angle = (frame.angle_degrees * Math.PI) / 180
  const widthAxis = { x: Math.cos(angle), y: Math.sin(angle) }
  const heightAxis = { x: -widthAxis.y, y: widthAxis.x }
  const center = { x: frame.x + frame.width * 0.5, y: frame.y + frame.height * 0.5 }
  const anchor = {
    x:
      center.x -
      widthAxis.x * direction.x * frame.width * 0.5 -
      heightAxis.x * direction.y * frame.height * 0.5,
    y:
      center.y -
      widthAxis.y * direction.x * frame.width * 0.5 -
      heightAxis.y * direction.y * frame.height * 0.5,
  }
  const delta = { x: point.x - anchor.x, y: point.y - anchor.y }
  const width = direction.x
    ? Math.max(minimumSize, direction.x * dot(delta, widthAxis))
    : frame.width
  const height = direction.y
    ? Math.max(minimumSize, direction.y * dot(delta, heightAxis))
    : frame.height
  const nextCenter = {
    x:
      anchor.x +
      widthAxis.x * direction.x * width * 0.5 +
      heightAxis.x * direction.y * height * 0.5,
    y:
      anchor.y +
      widthAxis.y * direction.x * width * 0.5 +
      heightAxis.y * direction.y * height * 0.5,
  }
  return {
    ...frame,
    x: nextCenter.x - width * 0.5,
    y: nextCenter.y - height * 0.5,
    width,
    height,
  }
}

export function rotateFrame(frame: Frame, start: Point, point: Point): Frame {
  const center = { x: frame.x + frame.width * 0.5, y: frame.y + frame.height * 0.5 }
  const from = { x: start.x - center.x, y: start.y - center.y }
  const to = { x: point.x - center.x, y: point.y - center.y }
  if (
    Math.hypot(from.x, from.y) <= minimumFrameSize ||
    Math.hypot(to.x, to.y) <= minimumFrameSize
  ) {
    return frame
  }
  const delta = (Math.atan2(from.x * to.y - from.y * to.x, dot(from, to)) * 180) / Math.PI
  return { ...frame, angle_degrees: normalizeDegrees(frame.angle_degrees + delta) }
}

export function translateFrames(originals: TransformFrame[], delta: Point): TransformFrame[] {
  return originals.map(({ element, frame }) => ({
    element,
    frame: { ...frame, x: frame.x + delta.x, y: frame.y + delta.y },
  }))
}

function finite(...values: number[]): boolean {
  return values.every(Number.isFinite)
}

function dot(left: Point, right: Point): number {
  return left.x * right.x + left.y * right.y
}

function normalizeDegrees(value: number): number {
  return ((((value + 180) % 360) + 360) % 360) - 180
}

function validFrame(frame: Frame): boolean {
  return (
    finite(frame.x, frame.y, frame.width, frame.height, frame.angle_degrees) &&
    frame.width > minimumFrameSize &&
    frame.height > minimumFrameSize
  )
}
