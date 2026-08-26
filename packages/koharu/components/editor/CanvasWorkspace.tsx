'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { useColorSampling } from '@/components/controls/ColorSampling'
import { CanvasCommandBar } from '@/components/editor/CanvasCommandBar'
import { CanvasOverlay } from '@/components/editor/CanvasOverlay'
import { StatusBar } from '@/components/editor/StatusBar'
import { ToolBar } from '@/components/editor/ToolBar'
import { useCanvas } from '@/components/editor/useCanvas'
import { call } from '@/lib/backend'
import { expandLayerSelection } from '@/lib/document'
import {
  controlFrame,
  draftFrame,
  framePoints,
  geometryFrame,
  hitTestEditorSelection,
  pagePoint,
  physicalPoint,
  selectableLayer,
  translateFrames,
} from '@/lib/geometry'
import { enqueueHistory, historyShortcutKey } from '@/lib/history'
import {
  pageKey,
  pagesKey,
  preparedPageKey,
  projectKey,
  queryClient,
  refresh,
  usePage,
  usePages,
} from '@/lib/queries'
import {
  isBrushTool,
  MAX_BRUSH_DIAMETER,
  MIN_BRUSH_DIAMETER,
  receiveError,
  useKoharuStore,
  type CanvasTool,
} from '@/lib/store'
import { prefetchCanvasPages, workspaceColor, type CanvasColor } from '@koharu/bridge/canvas'
import { commands, type Frame, type Point, type TransformFrame } from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'

const BRUSH_DIAMETER_STEP = 4

const canvasCursors = {
  select: undefined,
  text: 'text',
  ocr: 'crosshair',
  draw: 'none',
  eraser: 'none',
  color_picker: 'crosshair',
  remove: 'none',
  pan: 'grab',
} as const satisfies Record<CanvasTool, string | undefined>

type Gesture =
  | { kind: 'pan'; pointer: number; start: Point; translation: [number, number] }
  | {
      kind: 'move'
      pointer: number
      start: Point
      originals: TransformFrame[]
      collapseTo: string | null
      moved: boolean
    }
  | { kind: 'text' | 'ocr'; pointer: number; start: Point; frame: Frame }
  | StrokeGesture

interface StrokeGesture {
  kind: 'paint' | 'erase' | 'inpaint'
  pointer: number
  revision: number
  layer: string | null
  points: Point[]
  diameter: number
  color?: CanvasColor
}

interface CanvasView {
  zoom: number
  translation: [number, number]
}

interface StrokeUpdate {
  kind: 'paint' | 'erase' | 'inpaint'
  points: Point[]
}

function shortcutKey(event: KeyboardEvent): string {
  if (/^Key[A-Z]$/.test(event.code)) return event.code.slice(3).toLowerCase()
  if (/^Digit[0-9]$/.test(event.code)) return event.code.slice(5)
  return event.key.toLowerCase()
}

export function CanvasWorkspace() {
  const { t } = useTranslation()
  const surface = useRef<HTMLDivElement>(null)
  const [canvasElement, setCanvasElement] = useState<HTMLCanvasElement | null>(null)
  const gesture = useRef<Gesture | null>(null)
  const previousPageIndex = useRef<number | null>(null)
  const spaceHeld = useRef(false)
  const transformActive = useRef(false)
  const transformRegion = useRef(false)
  const transformRevision = useRef<number | null>(null)
  const transformFinal = useRef<TransformFrame[]>([])
  const transformInitial = useRef<TransformFrame[]>([])
  const commitPending = useRef(false)
  const commandQueue = useRef<Promise<void>>(Promise.resolve())
  const [previews, setPreviews] = useState<Record<string, Frame>>({})
  const [draft, setDraft] = useState<Frame | null>(null)
  const [hovered, setHovered] = useState<string | null>(null)
  const [cursor, setCursor] = useState<Point | null>(null)
  const colorSampling = useColorSampling()

  const page = usePage().data
  const pages = usePages().data
  const camera = useKoharuStore((state) => state.camera)
  const canvasPage = useKoharuStore((state) => state.canvasPage)
  const canvasRevision = useKoharuStore((state) => state.canvasRevision)
  const canvasGeneration = useKoharuStore((state) => state.canvasGeneration)
  const canvasSize = useKoharuStore((state) => state.canvasSize)
  const fitCanvasRequest = useKoharuStore((state) => state.fitCanvasRequest)
  const layerFrames = useKoharuStore((state) => state.layerFrames)
  const tool = useKoharuStore((state) => state.tool)
  const brush = useKoharuStore((state) => state.brush)
  const selected = useKoharuStore((state) => state.selectedLayers)
  const selectLayers = useKoharuStore((state) => state.selectLayers)
  const setTool = useKoharuStore((state) => state.setTool)
  const setBrush = useKoharuStore((state) => state.setBrush)
  const requestCanvasFit = useKoharuStore((state) => state.requestCanvasFit)
  const canvasState = useCanvas(canvasElement, canvasRevision, canvasGeneration)
  const canvas = canvasState.canvas
  const pageId = page?.id
  const pageWidth = canvasSize[0] || page?.size.width
  const pageHeight = canvasSize[1] || page?.size.height
  const activeRaster =
    selected.length === 1
      ? page?.layers.find((layer) => layer.id === selected[0] && layer.type === 'raster')
      : undefined

  const enqueue = useCallback(<Result,>(operation: () => Promise<Result>): Promise<Result> => {
    const pending = commandQueue.current.then(operation)
    commandQueue.current = pending.then(
      () => undefined,
      () => undefined,
    )
    return pending
  }, [])

  const transformUpdates = useFrameCommand((elements: TransformFrame[]) =>
    canvas?.updateTransform(elements),
  )
  const strokeUpdates = useFrameCommand(
    ({ points }: StrokeUpdate) => canvas?.extendStroke(points),
    mergeStrokeUpdates,
  )

  const beginTransform = useCallback(
    (elements: TransformFrame[]) => {
      if (
        !canvas ||
        canvasRevision === null ||
        !elements.length ||
        transformActive.current ||
        commitPending.current
      )
        return
      transformUpdates.clear()
      transformActive.current = true
      transformRegion.current = elements.every((element) =>
        page?.regions.some((region) => region.id === element.element),
      )
      transformRevision.current = canvasRevision
      transformFinal.current = elements
      transformInitial.current = elements
      setPreviews(Object.fromEntries(elements.map(({ element, frame }) => [element, frame])))
      try {
        if (!transformRegion.current) canvas.beginTransform(elements)
      } catch (error) {
        transformActive.current = false
        transformRegion.current = false
        receiveError(errorMessage(error))
      }
    },
    [canvas, canvasRevision, page, transformUpdates],
  )

  const updateTransform = useCallback(
    (elements: TransformFrame[]) => {
      if (!transformActive.current) return
      transformFinal.current = elements
      setPreviews(Object.fromEntries(elements.map(({ element, frame }) => [element, frame])))
      if (!transformRegion.current) transformUpdates.schedule(elements)
    },
    [transformUpdates],
  )

  const finishTransform = useCallback(() => {
    if (!transformActive.current) return
    transformUpdates.commit()
    transformActive.current = false
    const revision = transformRevision.current
    const elements = transformFinal.current
    const initial = transformInitial.current
    const regionTransform = transformRegion.current
    transformRevision.current = null
    transformInitial.current = []
    transformRegion.current = false
    if (regionTransform) {
      if (sameTransformFrames(initial, elements)) {
        setPreviews({})
        return
      }
      if (revision === null) {
        setPreviews({})
        return
      }
      commitPending.current = true
      void enqueue(() =>
        call(
          commands.setGeometry,
          elements.map(({ element, frame }) => ({ layer: element, points: framePoints(frame) })),
        ),
      )
        .then(() => refresh(projectKey, pagesKey, pageKey))
        .catch(() => undefined)
        .finally(() => {
          commitPending.current = false
          setPreviews({})
        })
      return
    }
    try {
      canvas?.finishTransform()
    } catch (error) {
      receiveError(errorMessage(error))
      setPreviews({})
      return
    }
    if (revision === null) {
      canvas?.cancelTransform()
      setPreviews({})
      return
    }
    commitPending.current = true
    void enqueue(() => call(commands.commitTransform, revision, elements))
      .then((revision) => (revision === null ? undefined : refresh(projectKey, pagesKey, pageKey)))
      .catch(() => canvas?.cancelTransform())
      .finally(() => {
        commitPending.current = false
        setPreviews({})
      })
  }, [canvas, enqueue, transformUpdates])

  const cancelGesture = useCallback(() => {
    const current = gesture.current
    gesture.current = null
    if (current?.kind === 'paint' || current?.kind === 'erase' || current?.kind === 'inpaint') {
      strokeUpdates.clear()
      canvas?.cancelStroke()
    }
    if (transformActive.current) {
      transformUpdates.clear()
      transformActive.current = false
      transformRevision.current = null
      transformInitial.current = []
      if (!transformRegion.current) canvas?.cancelTransform()
      transformRegion.current = false
    }
    setDraft(null)
    setPreviews({})
  }, [canvas, strokeUpdates, transformUpdates])

  const fitCanvas = useCallback(() => {
    const element = surface.current
    if (!element || !pageId || pageWidth === undefined || pageHeight === undefined) return
    const bounds = element.getBoundingClientRect()
    const dpr = window.devicePixelRatio
    const next = containCamera(bounds.width * dpr, bounds.height * dpr, pageWidth, pageHeight)
    useKoharuStore.setState({ camera: { ...next, fitted: true } })
  }, [pageHeight, pageId, pageWidth])

  const report = useCallback(() => {
    const element = surface.current
    if (!element || !canvas) return
    const bounds = element.getBoundingClientRect()
    canvas.resize(bounds.width, bounds.height, window.devicePixelRatio, workspaceColor())
    if (useKoharuStore.getState().camera.fitted) fitCanvas()
  }, [canvas, fitCanvas])

  const setZoom = useCallback((zoom: number) => {
    const element = surface.current
    if (!element) return
    const bounds = element.getBoundingClientRect()
    const dpr = window.devicePixelRatio
    const current = useKoharuStore.getState().camera
    const center = { x: bounds.width * dpr * 0.5, y: bounds.height * dpr * 0.5 }
    const pageX = (center.x - current.translation[0]) / current.zoom
    const pageY = (center.y - current.translation[1]) / current.zoom
    useKoharuStore.setState({
      camera: {
        zoom,
        translation: [center.x - pageX * zoom, center.y - pageY * zoom],
        fitted: false,
      },
    })
  }, [])

  useEffect(() => {
    const element = surface.current
    if (!element) return
    report()
    const resize = new ResizeObserver(report)
    const theme = new MutationObserver(report)
    resize.observe(element)
    theme.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    window.addEventListener('resize', report)
    window.visualViewport?.addEventListener('resize', report)
    return () => {
      resize.disconnect()
      theme.disconnect()
      window.removeEventListener('resize', report)
      window.visualViewport?.removeEventListener('resize', report)
    }
  }, [report])

  useEffect(() => {
    canvas?.setView(camera.zoom, camera.translation)
  }, [canvas, camera])

  useEffect(() => {
    if (
      !pageId ||
      canvasPage !== pageId ||
      canvasState.generation !== canvasGeneration ||
      canvasState.status !== 'ready' ||
      !pages?.length
    )
      return
    const index = pages.findIndex((candidate) => candidate.id === pageId)
    if (index < 0) return
    const previous = previousPageIndex.current
    previousPageIndex.current = index
    const direction = previous !== null && index < previous ? -1 : 1
    const adjacent = pages[index + direction]?.id ?? pages[index - direction]?.id
    if (adjacent) {
      void prefetchCanvasPages([adjacent])
        .then((prepared) => {
          for (const page of prepared) {
            queryClient.setQueryData(preparedPageKey(page.page.id), page)
          }
        })
        .catch(() => undefined)
    }
  }, [canvasGeneration, canvasPage, canvasState.generation, canvasState.status, pageId, pages])

  useEffect(() => {
    fitCanvas()
  }, [fitCanvas, fitCanvasRequest])

  useEffect(() => cancelGesture, [cancelGesture, canvasGeneration, canvasRevision, page?.id, tool])

  useEffect(() => {
    const editable = (target: EventTarget | null) =>
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      (target instanceof HTMLElement && target.isContentEditable)

    const down = (event: KeyboardEvent) => {
      if (event.key === 'Alt') {
        event.preventDefault()
        return
      }
      const state = useKoharuStore.getState()
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        spaceHeld.current = false
        cancelGesture()
        colorSampling?.cancel()
        selectLayers([])
        setTool('select')
        if (event.target instanceof HTMLElement) event.target.blur()
        return
      }
      if (editable(event.target)) return
      if (event.code === 'Space') {
        spaceHeld.current = true
        event.preventDefault()
        return
      }
      const command = event.ctrlKey || event.metaKey
      const historyKey = historyShortcutKey(event)
      // KoharuApp owns the global listener. Keep this local fallback for an
      // embedded workspace, while honoring a parent listener that handled it.
      if (!event.defaultPrevented && page && command && historyKey !== null) {
        event.preventDefault()
        void enqueueHistory(historyKey === 'y' || event.shiftKey ? 'redo' : 'undo').catch(
          () => undefined,
        )
        return
      }
      if (command && event.key.toLowerCase() === 'a' && page) {
        event.preventDefault()
        selectLayers(page.layers.filter(selectableLayer).map((layer) => layer.id))
        return
      }
      if (
        (event.key === 'Delete' || event.key === 'Backspace') &&
        state.selectedLayers.length > 0
      ) {
        event.preventDefault()
        void call(commands.deleteLayers, state.selectedLayers)
          .then(() => refresh(projectKey, pagesKey, pageKey))
          .catch(() => undefined)
        return
      }
      const key = shortcutKey(event)
      if (command && key === state.shortcuts.fit) {
        event.preventDefault()
        requestCanvasFit()
        return
      }
      if (!command) return
      const next = (
        ['select', 'text', 'ocr', 'draw', 'eraser', 'color_picker', 'remove', 'pan'] as const
      ).find((action) => state.shortcuts[action] === key)
      if (next) {
        event.preventDefault()
        setTool(next)
      }
    }

    const up = (event: KeyboardEvent) => {
      if (event.key === 'Alt') event.preventDefault()
      if (event.code === 'Space') spaceHeld.current = false
    }
    const blur = () => {
      spaceHeld.current = false
      cancelGesture()
    }

    window.addEventListener('keydown', down, true)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', blur)
    return () => {
      window.removeEventListener('keydown', down, true)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', blur)
    }
  }, [cancelGesture, colorSampling, enqueue, page, requestCanvasFit, selectLayers, setTool])

  const clientPagePoint = (clientX: number, clientY: number) =>
    pagePoint(
      clientX,
      clientY,
      surface.current!.getBoundingClientRect(),
      useKoharuStore.getState().camera,
    )

  const clientPhysicalPoint = (clientX: number, clientY: number) =>
    physicalPoint(clientX, clientY, surface.current!.getBoundingClientRect())

  const sampleCanvasColor = (point: Point) => {
    if (!canvas) return
    void canvas
      .sampleColor(point)
      .then((color) => {
        const hex = rgbaToHex(color)
        if (!colorSampling?.complete(hex)) {
          setBrush({ ...useKoharuStore.getState().brush, color: hex })
        }
      })
      .catch((error: unknown) => receiveError(errorMessage(error)))
  }

  const framesFor = (layers: string[]): TransformFrame[] => {
    const ids = expandLayerSelection(page?.layers ?? [], layers)
    const selectedLayers = ids.map((id) => page?.layers.find((candidate) => candidate.id === id))
    if (
      selectedLayers.length > 0 &&
      selectedLayers.every(
        (layer) => layer?.type === 'text' && Boolean(layer.content.source_region),
      )
    ) {
      const regions = new Set(
        selectedLayers.flatMap((layer) =>
          layer?.type === 'text' && layer.content.source_region
            ? [layer.content.source_region]
            : [],
        ),
      )
      return [...regions].flatMap((id) => {
        const region = page?.regions.find((candidate) => candidate.id === id)
        const frame = region && geometryFrame(region.geometry)
        return frame ? [{ element: region.id, frame }] : []
      })
    }
    return ids.flatMap((id) => {
      const layer = page?.layers.find((candidate) => candidate.id === id)
      const frame = layer && selectableLayer(layer) ? controlFrame(layer, layerFrames) : null
      return frame ? [{ element: id, frame }] : []
    })
  }

  const moveGesture = (
    pointer: number,
    samples: ReadonlyArray<{ clientX: number; clientY: number }>,
  ) => {
    const current = gesture.current
    const sample = samples.at(-1)
    if (!page || !sample) return
    const physical = clientPhysicalPoint(sample.clientX, sample.clientY)
    setCursor(physical)
    if (!current || current.pointer !== pointer) {
      if (tool === 'select') {
        setHovered(
          hitTestEditorSelection(
            page.layers,
            page.regions,
            clientPagePoint(sample.clientX, sample.clientY),
            layerFrames,
          )?.id ?? null,
        )
      }
      return
    }

    if (current.kind === 'pan') {
      const bounds = surface.current!.getBoundingClientRect()
      const dpr = window.devicePixelRatio
      let translation: [number, number] = [
        current.translation[0] + physical.x - current.start.x,
        current.translation[1] + physical.y - current.start.y,
      ]
      translation = clampCameraTranslation(
        translation,
        camera.zoom,
        pageWidth ?? page.size.width,
        pageHeight ?? page.size.height,
        bounds.width * dpr,
        bounds.height * dpr,
        dpr,
      )
      useKoharuStore.setState({ camera: { zoom: camera.zoom, translation, fitted: false } })
      return
    }

    const points = samples.map((value) => clientPagePoint(value.clientX, value.clientY))
    const point = points.at(-1)!
    if (current.kind === 'move') {
      const delta = { x: point.x - current.start.x, y: point.y - current.start.y }
      current.moved ||= delta.x !== 0 || delta.y !== 0
      updateTransform(translateFrames(current.originals, delta))
    } else if (current.kind === 'text' || current.kind === 'ocr') {
      const end =
        current.kind === 'ocr' ? clampPagePoint(point, page.size.width, page.size.height) : point
      current.frame =
        current.kind === 'ocr' ? rectangleFrame(current.start, end) : draftFrame(current.start, end)
      setDraft(current.frame)
    } else if (current.kind === 'paint' || current.kind === 'erase' || current.kind === 'inpaint') {
      current.points.push(...points)
      strokeUpdates.schedule({ kind: current.kind, points })
    }
  }

  const finishGesture = () => {
    const current = gesture.current
    gesture.current = null
    if (!current || !page) return
    if (current.kind === 'move') {
      finishTransform()
      if (!current.moved && current.collapseTo) selectLayers([current.collapseTo])
    } else if (current.kind === 'text') {
      const pointText =
        current.frame.width < 4 / camera.zoom && current.frame.height < 4 / camera.zoom
      setDraft(null)
      void (
        pointText
          ? call(commands.addPointText, current.start)
          : call(commands.addTextBox, current.frame)
      )
        .then((result) => {
          selectLayers([result.layer])
          return refresh(projectKey, pagesKey, pageKey)
        })
        .catch(() => undefined)
    } else if (current.kind === 'ocr') {
      setDraft(null)
      if (
        current.frame.width < 4 / camera.zoom ||
        current.frame.height < 4 / camera.zoom ||
        hasRunningJob()
      )
        return
      commitPending.current = true
      void call(commands.addOcrRegion, current.frame)
        .then(async (result) => {
          selectLayers([result.layer])
          await refresh(projectKey, pagesKey, pageKey)
          return call(
            commands.process,
            { scope: 'entities', value: [result.region] },
            { operation: 'stages', stages: ['ocr'] },
          )
        })
        .catch(() => undefined)
        .finally(() => (commitPending.current = false))
    } else if (current.kind === 'paint' || current.kind === 'erase' || current.kind === 'inpaint') {
      strokeUpdates.commit()
      try {
        canvas?.finishStroke()
      } catch (error) {
        receiveError(errorMessage(error))
        return
      }
      const operation =
        current.kind === 'paint'
          ? enqueue(() =>
              call(commands.commitPaint, current.revision, current.layer, current.points, {
                diameter: current.diameter,
                color: current.color!,
              }),
            ).then((result) => {
              selectLayers([result.layer])
              return refresh(projectKey, pagesKey, pageKey)
            })
          : current.kind === 'erase'
            ? enqueue(() =>
                call(
                  commands.commitErase,
                  current.revision,
                  current.layer!,
                  current.points,
                  current.diameter,
                ),
              ).then((result) => {
                selectLayers([result.layer])
                return refresh(projectKey, pagesKey, pageKey)
              })
            : enqueue(() =>
                call(commands.commitInpaint, current.revision, current.points, current.diameter),
              )
      commitPending.current = true
      void operation
        .catch(() => canvas?.cancelStroke())
        .finally(() => (commitPending.current = false))
    }
  }

  return (
    <main className='relative flex h-full min-h-0 min-w-0 flex-col overflow-hidden rounded-tl-2xl bg-[var(--surface-canvas)]'>
      <CanvasCommandBar />
      <div className='relative flex min-h-0 min-w-0 flex-1'>
        <ToolBar />
        <div
          ref={surface}
          tabIndex={0}
          aria-label={t('canvas.surface')}
          aria-busy={page ? canvasState.status !== 'ready' : undefined}
          className='relative min-h-0 min-w-0 flex-1 touch-none overflow-hidden bg-[var(--surface-canvas)] outline-none'
          style={{
            cursor: page && canvasState.status === 'ready' ? canvasCursors[tool] : undefined,
          }}
          onContextMenu={(event) => event.preventDefault()}
          onPointerDown={(event) => {
            if (
              !page ||
              !canvas ||
              canvasState.status !== 'ready' ||
              canvasRevision === null ||
              commitPending.current ||
              (tool === 'ocr' && hasRunningJob()) ||
              event.button > 1
            )
              return
            if (event.target instanceof Element && event.target.closest('[data-canvas-control]'))
              return
            event.currentTarget.focus()
            event.currentTarget.setPointerCapture(event.pointerId)
            const physical = clientPhysicalPoint(event.clientX, event.clientY)
            const point = clientPagePoint(event.clientX, event.clientY)
            setCursor(physical)

            if (event.button === 1 && tool === 'draw') {
              sampleCanvasColor(physical)
            } else if (event.button === 1 || tool === 'pan' || spaceHeld.current) {
              gesture.current = {
                kind: 'pan',
                pointer: event.pointerId,
                start: physical,
                translation: camera.translation,
              }
            } else if (tool === 'select') {
              const target = hitTestEditorSelection(page.layers, page.regions, point, layerFrames)
              const additive = event.shiftKey || event.ctrlKey || event.metaKey
              if (!target) {
                if (!additive) selectLayers([])
                return
              }
              const preserveGroup = !additive && selected.length > 1 && selected.includes(target.id)
              const next = additive
                ? selected.includes(target.id)
                  ? selected.filter((id) => id !== target.id)
                  : [...selected, target.id]
                : preserveGroup
                  ? selected
                  : [target.id]
              selectLayers(next)
              if (!next.includes(target.id)) return
              const originals = framesFor(next)
              if (!originals.length) return
              gesture.current = {
                kind: 'move',
                pointer: event.pointerId,
                start: point,
                originals,
                collapseTo: preserveGroup ? target.id : null,
                moved: false,
              }
              beginTransform(originals)
            } else if (tool === 'text' || tool === 'ocr') {
              const start =
                tool === 'ocr' ? clampPagePoint(point, page.size.width, page.size.height) : point
              const frame = tool === 'ocr' ? rectangleFrame(start, start) : draftFrame(start, start)
              gesture.current = { kind: tool, pointer: event.pointerId, start, frame }
              setDraft(frame)
            } else if (tool === 'draw') {
              strokeUpdates.clear()
              const color = hexToRgba(brush.color)
              gesture.current = {
                kind: 'paint',
                pointer: event.pointerId,
                revision: canvasRevision,
                layer: activeRaster?.id ?? null,
                points: [point],
                diameter: brush.diameter,
                color,
              }
              try {
                canvas.beginStroke({
                  kind: 'paint',
                  layer: activeRaster?.id ?? null,
                  point,
                  diameter: brush.diameter,
                  color,
                })
              } catch (error) {
                gesture.current = null
                receiveError(errorMessage(error))
              }
            } else if (tool === 'eraser') {
              if (!activeRaster) {
                receiveError('Select a paint or cleanup layer before using the Eraser.')
                return
              }
              strokeUpdates.clear()
              gesture.current = {
                kind: 'erase',
                pointer: event.pointerId,
                revision: canvasRevision,
                layer: activeRaster.id,
                points: [point],
                diameter: brush.diameter,
              }
              try {
                canvas.beginStroke({
                  kind: 'erase',
                  layer: activeRaster.id,
                  point,
                  diameter: brush.diameter,
                })
              } catch (error) {
                gesture.current = null
                receiveError(errorMessage(error))
              }
            } else if (tool === 'remove') {
              strokeUpdates.clear()
              gesture.current = {
                kind: 'inpaint',
                pointer: event.pointerId,
                revision: canvasRevision,
                layer: null,
                points: [point],
                diameter: brush.diameter,
              }
              try {
                canvas.beginStroke({
                  kind: 'inpaint',
                  layer: null,
                  point,
                  diameter: brush.diameter,
                })
              } catch (error) {
                gesture.current = null
                receiveError(errorMessage(error))
              }
            } else if (tool === 'color_picker') {
              sampleCanvasColor(physical)
            }
            event.preventDefault()
          }}
          onPointerMove={(event) => {
            if (!page) return
            const coalesced = event.nativeEvent.getCoalescedEvents?.() ?? [event.nativeEvent]
            moveGesture(event.pointerId, coalesced.length ? coalesced : [event.nativeEvent])
          }}
          onPointerUp={(event) => {
            moveGesture(event.pointerId, [event.nativeEvent])
            finishGesture()
            if (event.currentTarget.hasPointerCapture(event.pointerId)) {
              event.currentTarget.releasePointerCapture(event.pointerId)
            }
          }}
          onPointerCancel={() => cancelGesture()}
          onPointerLeave={(event) => {
            if (!event.currentTarget.hasPointerCapture(event.pointerId)) {
              setHovered(null)
              setCursor(null)
            }
          }}
          onWheel={(event) => {
            if (!page) return
            event.preventDefault()
            if (event.altKey && isBrushTool(tool)) {
              if (event.deltaY !== 0) {
                const currentBrush = useKoharuStore.getState().brush
                const delta = event.deltaY < 0 ? BRUSH_DIAMETER_STEP : -BRUSH_DIAMETER_STEP
                const nextDiameter = clamp(
                  currentBrush.diameter + delta,
                  MIN_BRUSH_DIAMETER,
                  MAX_BRUSH_DIAMETER,
                )
                if (nextDiameter !== currentBrush.diameter) {
                  setBrush({ ...currentBrush, diameter: nextDiameter })
                }
              }
              return
            }
            const point = clientPhysicalPoint(event.clientX, event.clientY)
            const current = useKoharuStore.getState().camera
            let zoom = current.zoom
            let translation = current.translation
            if (event.ctrlKey) {
              zoom = clamp(current.zoom * Math.exp(-event.deltaY * 0.0015), 0.02, 16)
              const pageX = (point.x - current.translation[0]) / current.zoom
              const pageY = (point.y - current.translation[1]) / current.zoom
              translation = [point.x - pageX * zoom, point.y - pageY * zoom]
            } else {
              const dpr = window.devicePixelRatio
              let deltaX = event.deltaX
              let deltaY = event.deltaY
              if (event.shiftKey && deltaX === 0) {
                deltaX = deltaY
                deltaY = 0
              }
              translation = [
                current.translation[0] - deltaX * dpr,
                current.translation[1] - deltaY * dpr,
              ]
            }
            const bounds = event.currentTarget.getBoundingClientRect()
            const dpr = window.devicePixelRatio
            translation = clampCameraTranslation(
              translation,
              zoom,
              pageWidth ?? page.size.width,
              pageHeight ?? page.size.height,
              bounds.width * dpr,
              bounds.height * dpr,
              dpr,
            )
            useKoharuStore.setState({ camera: { zoom, translation, fitted: false } })
          }}
        >
          <canvas
            ref={setCanvasElement}
            data-testid='webgpu-canvas'
            aria-hidden
            className='pointer-events-none absolute inset-0 block size-full'
          />
          {page && canvasState.status === 'ready' && (
            <CanvasOverlay
              page={page}
              camera={camera}
              selected={selected}
              hovered={hovered}
              frames={layerFrames}
              previews={previews}
              draft={draft}
              cursor={cursor}
              brushSize={brush.diameter}
              showBrushCursor={isBrushTool(tool)}
              showDetectionRegions={tool === 'select' || tool === 'ocr'}
              onTransformStart={beginTransform}
              onTransformFrame={updateTransform}
              onTransformEnd={finishTransform}
            />
          )}
          {page && canvasState.status === 'error' && (
            <div
              role='alert'
              className={
                canvasState.hasFrame
                  ? 'absolute inset-x-0 top-3 z-20 flex justify-center px-3'
                  : 'absolute inset-0 z-20 grid place-items-center bg-[var(--surface-canvas)]/90 p-6'
              }
            >
              <div
                className={
                  canvasState.hasFrame
                    ? 'flex max-w-sm flex-col items-center gap-2 rounded-md border border-border bg-popover/95 p-3 text-center shadow-sm'
                    : 'flex max-w-sm flex-col items-center gap-2 text-center'
                }
              >
                <p className='text-sm font-medium text-foreground'>{t('canvas.unavailable')}</p>
                {canvasState.error && (
                  <p className='text-xs leading-relaxed text-muted-foreground'>
                    {canvasState.error.message}
                  </p>
                )}
                <Button
                  data-canvas-control
                  size='sm'
                  variant='outline'
                  className='mt-1'
                  onClick={canvasState.retry}
                >
                  {t('errors.tryAgain')}
                </Button>
              </div>
            </div>
          )}
          {!page && (
            <div className='pointer-events-none absolute inset-0 grid place-items-center'>
              <p className='text-[12px] text-muted-foreground'>{t('canvas.empty')}</p>
            </div>
          )}
        </div>
      </div>
      <StatusBar onZoomChange={setZoom} />
    </main>
  )
}

function hexToRgba(hex: string): [number, number, number, number] {
  const value = hex.replace('#', '')
  return [
    Number.parseInt(value.slice(0, 2), 16),
    Number.parseInt(value.slice(2, 4), 16),
    Number.parseInt(value.slice(4, 6), 16),
    255,
  ]
}

function rgbaToHex(color: [number, number, number, number]): string {
  return `#${color
    .slice(0, 3)
    .map((channel) => channel.toString(16).padStart(2, '0'))
    .join('')}`.toUpperCase()
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value))
}

function clampPagePoint(point: Point, width: number, height: number): Point {
  return { x: clamp(point.x, 0, width), y: clamp(point.y, 0, height) }
}

function rectangleFrame(start: Point, end: Point): Frame {
  return {
    x: Math.min(start.x, end.x),
    y: Math.min(start.y, end.y),
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y),
    angle_degrees: 0,
  }
}

function hasRunningJob(): boolean {
  return Object.values(useKoharuStore.getState().jobs).some((job) => job.state === 'running')
}

function sameTransformFrames(left: TransformFrame[], right: TransformFrame[]): boolean {
  return (
    left.length === right.length &&
    left.every(({ element, frame }, index) => {
      const other = right[index]
      return (
        other?.element === element &&
        other.frame.x === frame.x &&
        other.frame.y === frame.y &&
        other.frame.width === frame.width &&
        other.frame.height === frame.height &&
        other.frame.angle_degrees === frame.angle_degrees
      )
    })
  )
}

function containCamera(
  viewportWidth: number,
  viewportHeight: number,
  pageWidth: number,
  pageHeight: number,
): CanvasView {
  if (viewportWidth <= 0 || viewportHeight <= 0 || pageWidth <= 0 || pageHeight <= 0) {
    return { zoom: 1, translation: [0, 0] }
  }
  const zoom = Math.max(
    Number.EPSILON,
    Math.min(viewportWidth / pageWidth, viewportHeight / pageHeight),
  )
  return {
    zoom,
    translation: [
      (viewportWidth - pageWidth * zoom) * 0.5,
      (viewportHeight - pageHeight * zoom) * 0.5,
    ],
  }
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  return typeof error === 'string' ? error : 'The WebGPU canvas returned an unknown error.'
}

function clampCameraTranslation(
  translation: [number, number],
  zoom: number,
  pageWidth: number,
  pageHeight: number,
  viewportWidth: number,
  viewportHeight: number,
  dpr: number,
): [number, number] {
  const minOverlap = 100 * dpr
  const minX = minOverlap - pageWidth * zoom
  const minY = minOverlap - pageHeight * zoom
  const maxX = viewportWidth - minOverlap
  const maxY = viewportHeight - minOverlap

  const x =
    minX <= maxX ? clamp(translation[0], minX, maxX) : (viewportWidth - pageWidth * zoom) * 0.5
  const y =
    minY <= maxY ? clamp(translation[1], minY, maxY) : (viewportHeight - pageHeight * zoom) * 0.5
  return [x, y]
}

function mergeStrokeUpdates(current: StrokeUpdate, next: StrokeUpdate): StrokeUpdate {
  if (current.kind !== next.kind) return next
  current.points.push(...next.points)
  return current
}

function useFrameCommand<Value>(
  execute: (value: Value) => void | Promise<unknown>,
  merge?: (current: Value, next: Value) => Value,
): FrameCommand<Value> {
  const executeRef = useRef(execute)
  executeRef.current = execute
  const command = useRef<FrameCommand<Value> | null>(null)
  command.current ??= new FrameCommand((value) => executeRef.current(value), merge)

  useEffect(() => () => command.current?.clear(), [])
  return command.current
}

class FrameCommand<Value> {
  private pending: Value | undefined
  private frame: number | null = null

  constructor(
    private readonly execute: (value: Value) => void | Promise<unknown>,
    private readonly merge: (current: Value, next: Value) => Value = (_current, next) => next,
  ) {}

  schedule(value: Value): void {
    this.pending = this.pending === undefined ? value : this.merge(this.pending, value)
    if (this.frame !== null) return
    this.frame = requestAnimationFrame(() => {
      this.frame = null
      this.executePending()
    })
  }

  commit(): void {
    if (this.frame !== null) {
      cancelAnimationFrame(this.frame)
      this.frame = null
    }
    this.executePending()
  }

  clear(): void {
    if (this.frame !== null) cancelAnimationFrame(this.frame)
    this.frame = null
    this.pending = undefined
  }

  private executePending(): void {
    const value = this.pending
    if (value === undefined) return
    this.pending = undefined
    try {
      void Promise.resolve(this.execute(value)).catch((error: unknown) =>
        receiveError(errorMessage(error)),
      )
    } catch (error) {
      receiveError(errorMessage(error))
    }
  }
}
