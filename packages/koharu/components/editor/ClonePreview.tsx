'use client'

import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef } from 'react'

import { call } from '@/lib/backend'
import { useKoharuStore } from '@/lib/store'
import { commands, type EntityId, type Page, type Point } from '@koharu/bridge/protocol'

/** A Clone Stamp stroke in progress, in page coordinates. */
export interface CloneStroke {
  points: Point[]
  diameter: number
  offset: Point
  /**
   * The masked layer an Eraser stroke reveals: the preview then shows that
   * layer's pixels before its mask, in place.
   */
  reveal?: EntityId
}

/**
 * The Clone Stamp stroke as it will be committed: the page as it shows, read
 * from the source offset, inside the stroke and at the stamp opacity. The
 * native commit makes the same copy; this preview stays until its frame
 * replaces it.
 */
export function ClonePreview({
  page,
  revision,
  stroke,
}: {
  page: Page
  revision: number | null
  stroke: CloneStroke | null
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const camera = useKoharuStore((state) => state.camera)
  const cloneOpacity = useKoharuStore((state) => state.cloneOpacity)
  const reveal = stroke?.reveal
  const opacity = reveal
    ? (page.layers.find((layer) => layer.id === reveal)?.visibility.opacity ?? 1)
    : cloneOpacity
  const source = useQuery({
    queryKey: ['clone-source', page.id, revision, reveal ?? null],
    queryFn: async () =>
      createImageBitmap(
        new Blob([
          new Uint8Array(
            await (reveal
              ? call(commands.getUnmaskedLayer, reveal)
              : call(commands.getCloneSource, page.id)),
          ),
        ]),
      ),
    staleTime: Number.POSITIVE_INFINITY,
    gcTime: 0,
  }).data

  useEffect(() => {
    const target = canvas.current
    const context = target?.getContext('2d')
    if (!target || !context) return
    context.clearRect(0, 0, target.width, target.height)
    if (!stroke || !source || stroke.points.length === 0) return
    // The stroke's shape first, then the source shifted by the offset kept
    // only inside it, so overlapping dabs do not build up.
    context.globalCompositeOperation = 'source-over'
    context.lineCap = 'round'
    context.lineJoin = 'round'
    context.lineWidth = stroke.diameter
    context.strokeStyle = '#000'
    context.fillStyle = '#000'
    context.beginPath()
    const [first, ...rest] = stroke.points
    context.moveTo(first!.x, first!.y)
    for (const point of rest) context.lineTo(point.x, point.y)
    if (rest.length === 0) {
      context.arc(first!.x, first!.y, stroke.diameter / 2, 0, Math.PI * 2)
      context.fill()
    } else {
      context.stroke()
    }
    context.globalCompositeOperation = 'source-in'
    context.drawImage(source, -stroke.offset.x, -stroke.offset.y)
    context.globalCompositeOperation = 'source-over'
  }, [source, stroke])

  const dpr = window.devicePixelRatio
  const scale = camera.zoom / dpr
  return (
    <canvas
      ref={canvas}
      data-testid='clone-preview'
      width={Math.round(page.size.width)}
      height={Math.round(page.size.height)}
      className='pointer-events-none absolute z-10 max-w-none'
      style={{
        left: camera.translation[0] / dpr,
        top: camera.translation[1] / dpr,
        width: page.size.width * scale,
        height: page.size.height * scale,
        opacity,
      }}
    />
  )
}
