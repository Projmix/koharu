'use client'

import { useRef, useState, type PointerEvent as ReactPointerEvent, type RefObject } from 'react'
import { useTranslation } from 'react-i18next'

import {
  convexCorners,
  cssFrame,
  frameUnitPoint,
  pagePoint,
  resizeFrame,
  rotateFrame,
  type Camera,
  type ResizeHandle,
} from '@/lib/geometry'
import type { EntityId, Frame, Point, TransformFrame } from '@koharu/bridge/protocol'

interface SelectionControlsProps {
  container: RefObject<HTMLDivElement | null>
  element: EntityId
  frame: Frame
  camera: Camera
  edgesOnly: boolean
  /** Corners, in units of the frame, a text layer is drawn onto in perspective. */
  perspective?: Point[] | null
  /** Commits new perspective corners, or none to draw the text flat again. */
  onPerspective?: (corners: Point[] | null) => void
  onTransformStart: (elements: TransformFrame[]) => void
  onTransformFrame: (elements: TransformFrame[]) => void
  onTransformEnd: () => void
}

type ControlGesture =
  | { kind: 'resize'; pointer: number; original: Frame; handle: ResizeHandle }
  | { kind: 'rotate'; pointer: number; original: Frame; start: Point }
  | { kind: 'distort'; pointer: number; corner: number; corners: Point[]; moved: boolean }

const flatCorners: Point[] = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
]

/** Ctrl+drag on a corner handle moves that corner alone, as Photoshop's Distort. */
const cornerIndex: Partial<Record<ResizeHandle, number>> = { nw: 0, ne: 1, se: 2, sw: 3 }

const handles: Array<{
  handle: ResizeHandle
  left: string
  top: string
  cursor: string
}> = [
  { handle: 'nw', left: '0%', top: '0%', cursor: 'nwse-resize' },
  { handle: 'n', left: '50%', top: '0%', cursor: 'ns-resize' },
  { handle: 'ne', left: '100%', top: '0%', cursor: 'nesw-resize' },
  { handle: 'e', left: '100%', top: '50%', cursor: 'ew-resize' },
  { handle: 'se', left: '100%', top: '100%', cursor: 'nwse-resize' },
  { handle: 's', left: '50%', top: '100%', cursor: 'ns-resize' },
  { handle: 'sw', left: '0%', top: '100%', cursor: 'nesw-resize' },
  { handle: 'w', left: '0%', top: '50%', cursor: 'ew-resize' },
]

export function SelectionControls({
  container,
  element,
  frame,
  camera,
  edgesOnly,
  perspective = null,
  onPerspective,
  onTransformStart,
  onTransformFrame,
  onTransformEnd,
}: SelectionControlsProps) {
  const { t } = useTranslation()
  const gesture = useRef<ControlGesture | null>(null)
  const [draftCorners, setDraftCorners] = useState<Point[] | null>(null)
  // The committed corners arrive with the refreshed layer; until then the
  // draft keeps showing where the drag ended.
  const [shownPerspective, setShownPerspective] = useState(perspective)
  if (shownPerspective !== perspective) {
    setShownPerspective(perspective)
    if (!gesture.current) setDraftCorners(null)
  }
  const corners = draftCorners ?? perspective
  const position = cssFrame(frame, camera)

  const eventPoint = (event: ReactPointerEvent<HTMLDivElement>): Point | null => {
    const bounds = container.current?.getBoundingClientRect()
    return bounds ? pagePoint(event.clientX, event.clientY, bounds, camera) : null
  }

  const capture = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault()
    event.stopPropagation()
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  const startDistort =
    (corner: number, from: Point[]) => (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 0 || gesture.current || !onPerspective) return
      capture(event)
      gesture.current = {
        kind: 'distort',
        pointer: event.pointerId,
        corner,
        corners: from,
        moved: false,
      }
      setDraftCorners(from)
    }

  const startResize = (handle: ResizeHandle) => (event: ReactPointerEvent<HTMLDivElement>) => {
    const corner = cornerIndex[handle]
    if (event.ctrlKey && corner !== undefined && onPerspective) {
      startDistort(corner, perspective ?? flatCorners)(event)
      return
    }
    if (event.button !== 0 || gesture.current) return
    capture(event)
    gesture.current = { kind: 'resize', pointer: event.pointerId, original: frame, handle }
    onTransformStart([{ element, frame }])
  }

  const startRotate = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || gesture.current) return
    const start = eventPoint(event)
    if (!start) return
    capture(event)
    gesture.current = { kind: 'rotate', pointer: event.pointerId, original: frame, start }
    onTransformStart([{ element, frame }])
  }

  const update = (event: ReactPointerEvent<HTMLDivElement>) => {
    const current = gesture.current
    if (!current || current.pointer !== event.pointerId) return
    event.preventDefault()
    event.stopPropagation()
    const point = eventPoint(event)
    if (!point) return
    if (current.kind === 'distort') {
      const moved = current.corners.map((corner, index) =>
        index === current.corner ? frameUnitPoint(frame, point) : corner,
      )
      // A corner dragged past its neighbours would fold the text over itself.
      if (convexCorners(moved)) {
        gesture.current = { ...current, corners: moved, moved: true }
        setDraftCorners(moved)
      }
      return
    }
    const next =
      current.kind === 'resize'
        ? resizeFrame(
            current.original,
            current.handle,
            point,
            window.devicePixelRatio / camera.zoom,
          )
        : rotateFrame(current.original, current.start, point)
    onTransformFrame([{ element, frame: next }])
  }

  const finish = (event: ReactPointerEvent<HTMLDivElement>) => {
    const current = gesture.current
    if (!current || current.pointer !== event.pointerId) return
    event.preventDefault()
    event.stopPropagation()
    gesture.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    if (current.kind === 'distort') {
      // A canceled drag commits nothing.
      if (event.type === 'pointerup' && current.moved) onPerspective?.(current.corners)
      else setDraftCorners(null)
      return
    }
    onTransformEnd()
  }

  const lostCapture = (event: ReactPointerEvent<HTMLDivElement>) => {
    const current = gesture.current
    if (!current || current.pointer !== event.pointerId) return
    gesture.current = null
    if (current.kind === 'distort') {
      setDraftCorners(null)
      return
    }
    onTransformEnd()
  }

  const pointerEvents = {
    onPointerMove: update,
    onPointerUp: finish,
    onPointerCancel: finish,
    onLostPointerCapture: lostCapture,
  }

  return (
    <div
      data-testid='selection-controls'
      className='pointer-events-none absolute box-border border border-[var(--canvas-selection)]'
      style={{
        left: position.left,
        top: position.top,
        width: position.width,
        height: position.height,
        transform: `rotate(${position.angle}deg)`,
        transformOrigin: '50% 50%',
      }}
    >
      {handles.map(({ handle, left, top, cursor }) => (
        <div
          key={handle}
          data-canvas-control
          data-resize-handle={handle}
          className='pointer-events-auto absolute grid size-3.5 -translate-x-1/2 -translate-y-1/2 touch-none place-items-center'
          style={{
            left,
            top,
            cursor,
            width:
              edgesOnly && (handle === 'n' || handle === 's')
                ? 'max(14px, calc(100% - 14px))'
                : undefined,
            height:
              edgesOnly && (handle === 'e' || handle === 'w')
                ? 'max(14px, calc(100% - 14px))'
                : undefined,
          }}
          onPointerDown={startResize(handle)}
          {...pointerEvents}
        >
          {!edgesOnly && (
            <span className='size-1.5 rounded-[1px] border border-primary-foreground bg-[var(--canvas-selection)] shadow-[0_0_0_1px_rgb(0_0_0/0.12)]' />
          )}
        </div>
      ))}

      {corners && (
        <svg
          data-testid='perspective-outline'
          className='pointer-events-none absolute inset-0 size-full overflow-visible'
          viewBox='0 0 1 1'
          preserveAspectRatio='none'
        >
          <polygon
            points={corners.map(({ x, y }) => `${x},${y}`).join(' ')}
            fill='none'
            stroke='var(--canvas-selection)'
            strokeWidth='1.5'
            strokeDasharray='5 3'
            vectorEffect='non-scaling-stroke'
          />
        </svg>
      )}
      {corners?.map((corner, index) => (
        <div
          key={index}
          data-canvas-control
          data-perspective-handle={index}
          title={t('canvas.perspective.handle')}
          className='pointer-events-auto absolute grid size-4 -translate-x-1/2 -translate-y-1/2 cursor-crosshair touch-none place-items-center'
          style={{ left: `${corner.x * 100}%`, top: `${corner.y * 100}%` }}
          onPointerDown={startDistort(index, corners)}
          onDoubleClick={(event) => {
            event.stopPropagation()
            if (perspective) onPerspective?.(null)
          }}
          {...pointerEvents}
        >
          <span className='size-2 rounded-full border border-primary-foreground bg-[var(--canvas-selection)] shadow-[0_0_0_1px_rgb(0_0_0/0.12)]' />
        </div>
      ))}

      <span className='pointer-events-none absolute top-0 left-1/2 h-2 w-px -translate-x-1/2 -translate-y-full bg-[var(--canvas-selection)]' />
      <div
        data-canvas-control
        data-rotate-handle
        className='pointer-events-auto absolute top-[-16px] left-1/2 grid size-3.5 -translate-x-1/2 -translate-y-1/2 cursor-alias touch-none place-items-center'
        onPointerDown={startRotate}
        {...pointerEvents}
      >
        <span className='size-1.5 rounded-[1px] border border-primary-foreground bg-[var(--canvas-selection)] shadow-[0_0_0_1px_rgb(0_0_0/0.12)]' />
      </div>
    </div>
  )
}
