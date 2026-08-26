import { describe, expect, it } from 'vitest'

import {
  framePoints,
  geometryFrame,
  hitTestEditorSelection,
  resizeFrame,
  rotateFrame,
} from '@/lib/geometry'
import type { AnalysisRegion, Frame, Layer } from '@koharu/bridge/protocol'

const frame: Frame = { x: 10, y: 20, width: 100, height: 50, angle_degrees: 0 }

describe('selection control geometry', () => {
  it('resizes from a handle while keeping the opposite edge fixed', () => {
    expect(resizeFrame(frame, 'e', { x: 140, y: 45 }, 1)).toEqual({
      ...frame,
      width: 130,
    })
    expect(resizeFrame(frame, 'nw', { x: 0, y: 10 }, 1)).toEqual({
      ...frame,
      x: 0,
      y: 10,
      width: 110,
      height: 60,
    })
  })

  it('resizes in the frame local axes when it is rotated', () => {
    const result = resizeFrame({ ...frame, angle_degrees: 90 }, 'e', { x: 60, y: 125 }, 1)
    expect(result.x).toBeCloseTo(-5)
    expect(result.y).toBeCloseTo(35)
    expect(result.width).toBeCloseTo(130)
    expect(result.height).toBe(50)
  })

  it('rotates around the frame center', () => {
    expect(rotateFrame(frame, { x: 60, y: -5 }, { x: 110, y: 45 })).toEqual({
      ...frame,
      angle_degrees: 90,
    })
  })

  it('round-trips rotated analysis region geometry through a control frame', () => {
    const rotated = { ...frame, angle_degrees: 27 }
    const points = framePoints(rotated)
    const result = geometryFrame({ points })
    expect(result?.x).toBeCloseTo(rotated.x)
    expect(result?.y).toBeCloseTo(rotated.y)
    expect(result?.width).toBeCloseTo(rotated.width)
    expect(result?.height).toBeCloseTo(rotated.height)
    expect(result?.angle_degrees).toBeCloseTo(rotated.angle_degrees)
  })

  it('prefers the most specific OCR polygon when detected regions overlap', () => {
    const broad = textLayer('broad', 'broad-text')
    const specific = textLayer('specific', 'specific-text')
    const regions = [
      region('broad-text', [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
      ]),
      region('specific-text', [
        [40, 40],
        [60, 40],
        [60, 60],
        [40, 60],
      ]),
    ]

    expect(hitTestEditorSelection([specific, broad], regions, { x: 50, y: 50 }, {})).toBe(specific)
  })

  it('chooses the nearest text frame when several dialogs share a bubble', () => {
    const left = textLayer('left', 'left-text', 'shared-bubble')
    const right = textLayer('right', 'right-text', 'shared-bubble')
    const regions = [
      region('shared-bubble', [
        [0, 0],
        [120, 0],
        [120, 100],
        [0, 100],
      ]),
      region('left-text', [
        [20, 20],
        [30, 20],
        [30, 30],
        [20, 30],
      ]),
      region('right-text', [
        [90, 70],
        [100, 70],
        [100, 80],
        [90, 80],
      ]),
    ]

    expect(
      hitTestEditorSelection(
        [right, left],
        regions,
        { x: 30, y: 35 },
        {
          left: { x: 20, y: 20, width: 25, height: 25, angle_degrees: 0 },
          right: { x: 85, y: 65, width: 25, height: 25, angle_degrees: 0 },
        },
      ),
    ).toBe(left)
  })
})

function textLayer(
  id: string,
  sourceRegion: string,
  automaticRegion: string | null = null,
): Extract<Layer, { type: 'text' }> {
  return {
    type: 'text',
    id,
    parent: 'page',
    geometry: null,
    visibility: { visible: true, opacity: 1 },
    content: {
      id: `${id}-content`,
      source: { text: id, language: 'ja' },
      translation: null,
      role: 'dialogue',
      source_region: sourceRegion,
    },
    typography: null,
    layout: 'paragraph',
    automatic_region: automaticRegion,
  }
}

function region(id: string, points: number[][]): AnalysisRegion {
  return {
    id,
    parent: 'page',
    geometry: { points: points.map(([x, y]) => ({ x, y })) },
    kind: id.endsWith('bubble') ? 'bubble' : 'text',
    label: null,
  }
}
