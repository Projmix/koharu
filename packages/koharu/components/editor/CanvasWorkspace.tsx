'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { useColorSampling } from '@/components/controls/ColorSampling'
import { CanvasCommandBar } from '@/components/editor/CanvasCommandBar'
import { CanvasOverlay } from '@/components/editor/CanvasOverlay'
import { ClonePreview, type CloneStroke } from '@/components/editor/ClonePreview'
import { OriginalView } from '@/components/editor/OriginalView'
import { PagePreview } from '@/components/editor/PagePreview'
import { StatusBar } from '@/components/editor/StatusBar'
import { ToolBar } from '@/components/editor/ToolBar'
import { useCanvas } from '@/components/editor/useCanvas'
import { call } from '@/lib/backend'
import { toggleCleanupDone } from '@/lib/cleanup'
import { expandLayerSelection } from '@/lib/document'
import {
  controlFrame,
  draftFrame,
  framePoints,
  geometryFrame,
  hitTestEditorSelection,
  hitTestLayers,
  pagePoint,
  physicalPoint,
  selectableLayer,
  translateFrames,
} from '@/lib/geometry'
import { enqueueHistory, historyShortcutKey } from '@/lib/history'
import { stepPage } from '@/lib/navigation'
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
import { runLayerShortcut } from '@/lib/rasterLayers'
import {
  isBrushTool,
  maxBrushDiameter,
  MIN_BRUSH_DIAMETER,
  receiveError,
  useKoharuStore,
  type CanvasTool,
} from '@/lib/store'
import {
  prefetchCanvasPages,
  workspaceColor,
  type Canvas,
  type CanvasColor,
} from '@koharu/bridge/canvas'
import {
  commands,
  type Frame,
  type Layer,
  type Page,
  type Point,
  type TextClipboardPayload,
  type TextTransferMode,
  type TransformFrame,
  type UnboxedText,
} from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'

const BRUSH_DIAMETER_STEP = 4
const TEXT_CLIPBOARD_MIME = 'application/x-koharu-text-layers+json'

const canvasCursors = {
  select: undefined,
  text: 'text',
  ocr: 'crosshair',
  draw: 'none',
  eraser: 'none',
  clone: 'none',
  color_picker: 'crosshair',
  remove: 'none',
  restore: 'crosshair',
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
  | {
      kind: 'text' | 'ocr' | 'restore' | 'area'
      pointer: number
      revision: number
      start: Point
      frame: Frame
    }
  | StrokeGesture

type RasterLayer = Extract<Layer, { type: 'raster' }>

interface StrokeGesture {
  kind: 'paint' | 'erase' | 'inpaint' | 'clone'
  pointer: number
  revision: number
  layer: string | null
  points: Point[]
  diameter: number
  color?: CanvasColor
  /** Page offset from a Clone Stamp stroke to the pixels it copies. */
  offset?: Point
  /** An Eraser stroke that reveals a layer through its reveal mask. */
  reveal?: boolean
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

function textClipboardPayload(
  page: Page,
  selected: string[],
  mode: TextTransferMode,
): TextClipboardPayload | null {
  const selectedLayers = new Set(expandLayerSelection(page.layers, selected))
  const layers = page.layers
    .filter((layer) => layer.type === 'text' && selectedLayers.has(layer.id))
    .map((layer) => {
      if (layer.type !== 'text') throw new Error('unreachable non-text clipboard layer')
      const region = (id: string | null) => {
        const value = id ? page.regions.find((candidate) => candidate.id === id) : undefined
        return value ? { id: value.id, geometry: value.geometry, label: value.label } : null
      }
      return {
        layer: layer.id,
        content: layer.content.id,
        source: layer.content.source,
        translation: layer.content.translation,
        role: layer.content.role,
        geometry: layer.geometry,
        angle_degrees: layer.angle_degrees,
        visibility: layer.visibility,
        typography: layer.typography,
        layout: layer.layout,
        source_region: region(layer.content.source_region),
        automatic_region: region(layer.automatic_region),
      }
    })
  return layers.length ? { version: 1, mode, source_page: page.id, layers } : null
}

function clipboardPlainText(payload: TextClipboardPayload): string {
  return payload.layers
    .map((layer) => layer.translation?.text || layer.source?.text || '')
    .filter(Boolean)
    .join('\n')
}

function readDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error ?? new Error('Failed to read clipboard image'))
    reader.onload = () => resolve(String(reader.result))
    reader.readAsDataURL(file)
  })
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
  const [committedEdit, setCommittedEdit] = useState<{
    canvas: Canvas
    page: string
    revision: number
    kind: 'stroke' | 'transform'
  } | null>(null)
  const commandQueue = useRef<Promise<void>>(Promise.resolve())
  const [previews, setPreviews] = useState<Record<string, Frame>>({})
  const [draft, setDraft] = useState<Frame | null>(null)
  const [hovered, setHovered] = useState<string | null>(null)
  const [cursor, setCursor] = useState<Point | null>(null)
  const cursorPage = useRef<Point | null>(null)
  const colorSampling = useColorSampling()

  const page = usePage().data
  const compareOriginal = useKoharuStore((state) => state.compareOriginal)
  const cloneSource = useKoharuStore((state) => state.cloneSource)
  const [cloneStroke, setCloneStroke] = useState<CloneStroke | null>(null)
  const pendingPage = useKoharuStore((state) => state.pendingPage)
  const pages = usePages().data
  const canvasPage = useKoharuStore((state) => state.canvasPage)
  const canvasRevision = useKoharuStore((state) => state.canvasRevision)
  const canvasGeneration = useKoharuStore((state) => state.canvasGeneration)
  const canvasSize = useKoharuStore((state) => state.canvasSize)
  const fitCanvasRequest = useKoharuStore((state) => state.fitCanvasRequest)
  const layerFrames = useKoharuStore((state) => state.layerFrames)
  const tool = useKoharuStore((state) => state.tool)
  const textTransformTarget = useKoharuStore((state) => state.textTransformTarget)
  const editTextLayer = useKoharuStore((state) => state.editTextLayer)
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
  const selectedRaster =
    selected.length === 1
      ? page?.layers.find(
          (layer): layer is RasterLayer => layer.id === selected[0] && layer.type === 'raster',
        )
      : undefined
  const eraserRaster =
    selectedRaster ??
    page?.layers.find(
      (layer): layer is RasterLayer =>
        layer.type === 'raster' && layer.kind === 'cleanup' && layer.visibility.visible,
    )

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

  useEffect(() => {
    if (!committedEdit) return
    // The native commit can finish before its frame is installed in WebGPU.
    // Keep the preview and edit lock until that frame is usable, including
    // no-op commits that keep the same revision and never replace the frame.
    if (
      committedEdit.canvas === canvas &&
      committedEdit.page === canvasPage &&
      canvasState.status !== 'error' &&
      (canvasState.status !== 'ready' ||
        canvasState.generation !== canvasGeneration ||
        canvasRevision === null ||
        canvasRevision < committedEdit.revision)
    )
      return
    // The committed frame now shows the Clone Stamp stroke.
    setCloneStroke(null)
    try {
      if (committedEdit.kind === 'stroke') committedEdit.canvas.cancelStroke()
      else committedEdit.canvas.cancelTransform()
    } catch (error) {
      if (committedEdit.canvas === canvas) receiveError(errorMessage(error))
    } finally {
      setCommittedEdit(null)
    }
  }, [
    canvas,
    canvasGeneration,
    canvasPage,
    canvasRevision,
    canvasState.generation,
    canvasState.status,
    committedEdit,
  ])

  const beginTransform = useCallback(
    (elements: TransformFrame[]) => {
      if (
        !canvas ||
        canvasRevision === null ||
        !elements.length ||
        transformActive.current ||
        commitPending.current ||
        committedEdit ||
        canvasState.status !== 'ready' ||
        canvasState.generation !== canvasGeneration
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
    [
      canvas,
      canvasGeneration,
      canvasRevision,
      canvasState.generation,
      canvasState.status,
      committedEdit,
      page,
      transformUpdates,
    ],
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
      canvas?.cancelTransform()
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
      .then((revision) => {
        if (revision === null) {
          canvas?.cancelTransform()
          return
        }
        if (canvas && pageId)
          setCommittedEdit({ canvas, page: pageId, revision, kind: 'transform' })
        return refresh(projectKey, pagesKey, pageKey)
      })
      .catch(() => {
        canvas?.cancelTransform()
        setCommittedEdit(null)
      })
      .finally(() => {
        commitPending.current = false
        setPreviews({})
      })
  }, [canvas, enqueue, pageId, transformUpdates])

  // One path for a drawn OCR rectangle and a flagged unboxed text: add the
  // region, select its layer, then read it.
  const readRegion = useCallback(
    (frame: Frame) => {
      commitPending.current = true
      void call(commands.addOcrRegion, frame)
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
    },
    [selectLayers],
  )

  const readUnboxedText = useCallback(
    (flag: UnboxedText) => {
      if (commitPending.current) return
      readRegion({ x: flag.x, y: flag.y, width: flag.width, height: flag.height, angle_degrees: 0 })
    },
    [readRegion],
  )

  const dismissUnboxedText = useCallback(
    (flag: UnboxedText) => {
      if (!pageId) return
      void enqueue(() => call(commands.dismissUnboxedText, pageId, flag))
        .then(() => refresh(projectKey, pagesKey, pageKey))
        .catch(() => undefined)
    },
    [enqueue, pageId],
  )

  const cancelGesture = useCallback(() => {
    const current = gesture.current
    gesture.current = null
    if (
      current?.kind === 'paint' ||
      current?.kind === 'erase' ||
      current?.kind === 'inpaint' ||
      current?.kind === 'clone'
    ) {
      strokeUpdates.clear()
      canvas?.cancelStroke()
      setCloneStroke(null)
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

  // The camera reaches the canvas from the store, not from a render: a passive effect lands a
  // frame late under continuous input and halves the presented rate.
  useEffect(() => {
    if (!canvas) return
    const apply = ({ zoom, translation }: CanvasView) => {
      try {
        canvas.setView(zoom, translation)
      } catch (error) {
        receiveError(errorMessage(error))
      }
    }
    apply(useKoharuStore.getState().camera)
    return useKoharuStore.subscribe((state, previous) => {
      if (state.camera !== previous.camera) apply(state.camera)
    })
  }, [canvas])

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
    // Two pages ahead in the paging direction and one behind, so stepping
    // either way shows a prepared page at once.
    const adjacent = [index + direction, index + 2 * direction, index - direction]
      .map((offset) => pages[offset]?.id)
      .filter((id): id is string => Boolean(id))
    if (adjacent.length) {
      void prefetchCanvasPages(adjacent)
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

  // The status bar's cleanup opacity is a display override: it never changes
  // the layers, and a new frame drops overrides, so it is applied again.
  const cleanupOpacity = useKoharuStore((state) => state.cleanupOpacity)
  useEffect(() => {
    if (!canvas || !page || canvasState.status !== 'ready') return
    for (const layer of page.layers) {
      if (layer.type !== 'raster' || layer.kind !== 'cleanup' || !layer.visibility.visible) continue
      try {
        canvas.previewOpacity(
          layer.id,
          cleanupOpacity === 1 ? null : cleanupOpacity * layer.visibility.opacity,
        )
      } catch {
        // The layer is not in the presented frame yet.
      }
    }
  }, [canvas, page, canvasState.status, canvasState.generation, canvasRevision, cleanupOpacity])

  const tintedLayers = useKoharuStore((state) => state.tintedLayers)
  useEffect(() => {
    if (!canvas || canvasState.status !== 'ready') return
    canvas.setTintedLayers(tintedLayers)
  }, [canvas, canvasState.status, tintedLayers])

  useEffect(() => {
    if (pendingPage && pendingPage === canvasPage) useKoharuStore.setState({ pendingPage: null })
  }, [canvasPage, pendingPage])

  useEffect(
    () => cancelGesture,
    [cancelGesture, canvasGeneration, canvasRevision, page?.id, tool, textTransformTarget],
  )

  useEffect(() => {
    cursorPage.current = null
  }, [canvasGeneration, pageId])

  useEffect(() => {
    const editable = (target: EventTarget | null) =>
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      (target instanceof HTMLElement && target.isContentEditable)

    const down = (event: KeyboardEvent) => {
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
      // Left and right step through pages, unless a control that uses the
      // arrows itself has focus.
      if (
        (event.key === 'ArrowLeft' || event.key === 'ArrowRight') &&
        !event.altKey &&
        !event.ctrlKey &&
        !event.metaKey &&
        !event.shiftKey &&
        !(
          event.target instanceof HTMLElement &&
          event.target.closest(
            '[role=slider],[role=menu],[role=menubar],[role=listbox],[role=tablist],[role=radiogroup],[role=tree],select',
          )
        )
      ) {
        event.preventDefault()
        stepPage(event.key === 'ArrowLeft' ? -1 : 1)
        return
      }
      if (event.key === 'Alt') {
        event.preventDefault()
        return
      }
      if (
        event.code === 'Space' &&
        !event.altKey &&
        !event.ctrlKey &&
        !event.metaKey &&
        !event.shiftKey
      ) {
        spaceHeld.current = true
        event.preventDefault()
        return
      }
      const command = event.ctrlKey || event.metaKey
      const brushDelta =
        event.key === '+' || event.code === 'NumpadAdd'
          ? 1
          : event.key === '-' || event.code === 'NumpadSubtract'
            ? -1
            : null
      if (
        brushDelta !== null &&
        !event.defaultPrevented &&
        !command &&
        !event.altKey &&
        isBrushTool(state.tool)
      ) {
        event.preventDefault()
        event.stopPropagation()
        const diameter = clamp(
          state.brush.diameter + brushDelta,
          MIN_BRUSH_DIAMETER,
          maxBrushDiameter(page?.size ?? { width: 2048, height: 2048 }),
        )
        if (diameter !== state.brush.diameter) {
          state.setBrush({ ...state.brush, diameter })
        }
        return
      }
      const historyKey = historyShortcutKey(event)
      // KoharuApp owns the global listener. Keep this local fallback for an
      // embedded workspace, while honoring a parent listener that handled it.
      if (!event.defaultPrevented && page && command && !event.altKey && historyKey !== null) {
        event.preventDefault()
        void enqueueHistory(historyKey === 'y' || event.shiftKey ? 'redo' : 'undo').catch(
          () => undefined,
        )
        return
      }
      if (
        command &&
        !event.altKey &&
        !event.shiftKey &&
        (event.code === 'KeyJ' || event.code === 'KeyE') &&
        runLayerShortcut(event.code === 'KeyJ' ? 'j' : 'e')
      ) {
        event.preventDefault()
        event.stopPropagation()
        return
      }
      if (command && !event.altKey && event.key.toLowerCase() === 'a' && page) {
        event.preventDefault()
        event.stopPropagation()
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
      const toolShortcut = event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey
      if (toolShortcut && key === state.shortcuts.fit) {
        event.preventDefault()
        event.stopPropagation()
        requestCanvasFit()
        return
      }
      if (toolShortcut && key === state.shortcuts.cleanup_done) {
        event.preventDefault()
        event.stopPropagation()
        toggleCleanupDone()
        return
      }
      if (!toolShortcut) return
      const next = (
        [
          'select',
          'text',
          'ocr',
          'draw',
          'eraser',
          'color_picker',
          'clone',
          'remove',
          'restore',
          'pan',
        ] as const
      ).find((action) => state.shortcuts[action] === key)
      if (next) {
        event.preventDefault()
        event.stopPropagation()
        setTool(next)
      }
    }

    const up = (event: KeyboardEvent) => {
      if (event.key === 'Alt' && !editable(event.target)) event.preventDefault()
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

  useEffect(() => {
    const editable = (target: EventTarget | null) =>
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      (target instanceof HTMLElement && target.isContentEditable)

    const copy = (event: ClipboardEvent, mode: TextTransferMode) => {
      if (!page || editable(event.target)) return
      const payload = textClipboardPayload(page, useKoharuStore.getState().selectedLayers, mode)
      if (!payload || !event.clipboardData) return
      event.preventDefault()
      event.stopPropagation()
      event.clipboardData.setData(TEXT_CLIPBOARD_MIME, JSON.stringify(payload))
      event.clipboardData.setData('text/plain', clipboardPlainText(payload))
    }
    const paste = (event: ClipboardEvent) => {
      if (!page || editable(event.target) || !event.clipboardData) return
      const encoded = event.clipboardData.getData(TEXT_CLIPBOARD_MIME)
      if (event.clipboardData.types.includes(TEXT_CLIPBOARD_MIME)) {
        event.preventDefault()
        event.stopPropagation()
        if (!encoded) {
          receiveError('The Koharu clipboard data is damaged.')
          return
        }
        let payload: TextClipboardPayload
        try {
          payload = JSON.parse(encoded) as TextClipboardPayload
        } catch {
          receiveError('The Koharu clipboard data is damaged.')
          return
        }
        void call(commands.pasteTextBlocks, page.id, payload, cursorPage.current)
          .then((result) => {
            queryClient.setQueryData(projectKey, result.project)
            queryClient.setQueryData(pageKey, result.page)
            if (useKoharuStore.getState().canvasPage === page.id) {
              if (result.layers.length === 1 && result.layers[0]) editTextLayer(result.layers[0])
              else selectLayers(result.layers)
            }
            return refresh(pagesKey)
          })
          .catch(() => undefined)
        return
      }
      const image = [...event.clipboardData.items]
        .find((item) => item.kind === 'file' && item.type.startsWith('image/'))
        ?.getAsFile()
      if (!image) return
      event.preventDefault()
      event.stopPropagation()
      void readDataUrl(image)
        .then((dataUrl) => call(commands.importCleanupImage, dataUrl))
        .then((result) => {
          selectLayers([result.layer])
          return refresh(projectKey, pagesKey, pageKey)
        })
        .catch((error: unknown) => receiveError(errorMessage(error)))
    }
    const onCopy = (event: ClipboardEvent) => copy(event, 'copy')
    const onCut = (event: ClipboardEvent) => copy(event, 'move')
    window.addEventListener('copy', onCopy, true)
    window.addEventListener('cut', onCut, true)
    window.addEventListener('paste', paste, true)
    return () => {
      window.removeEventListener('copy', onCopy, true)
      window.removeEventListener('cut', onCut, true)
      window.removeEventListener('paste', paste, true)
    }
  }, [page, selectLayers, editTextLayer])

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

  const selectionAt = (point: Point) => {
    if (!page) return null
    if (textTransformTarget === 'text') {
      const visible = hitTestLayers(page.layers, point, layerFrames)
      if (visible?.type === 'text') return visible
    }
    return hitTestEditorSelection(page.layers, page.regions, point, layerFrames)
  }

  const framesFor = (layers: string[]): TransformFrame[] => {
    const ids = expandLayerSelection(page?.layers ?? [], layers)
    const selectedLayers = ids.map((id) => page?.layers.find((candidate) => candidate.id === id))
    if (
      textTransformTarget === 'region' &&
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
    cursorPage.current = clientPagePoint(sample.clientX, sample.clientY)
    setCursor(physical)
    if (!current || current.pointer !== pointer) {
      if (tool === 'select') {
        setHovered(selectionAt(clientPagePoint(sample.clientX, sample.clientY))?.id ?? null)
      }
      return
    }

    if (current.kind === 'pan') {
      const { zoom } = useKoharuStore.getState().camera
      const bounds = surface.current!.getBoundingClientRect()
      const dpr = window.devicePixelRatio
      let translation: [number, number] = [
        current.translation[0] + physical.x - current.start.x,
        current.translation[1] + physical.y - current.start.y,
      ]
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
      return
    }

    const points = samples.map((value) => clientPagePoint(value.clientX, value.clientY))
    const point = points.at(-1)!
    if (current.kind === 'move') {
      const delta = { x: point.x - current.start.x, y: point.y - current.start.y }
      current.moved ||= delta.x !== 0 || delta.y !== 0
      updateTransform(translateFrames(current.originals, delta))
    } else if (
      current.kind === 'text' ||
      current.kind === 'ocr' ||
      current.kind === 'restore' ||
      current.kind === 'area'
    ) {
      const end =
        current.kind === 'text' ? point : clampPagePoint(point, page.size.width, page.size.height)
      current.frame =
        current.kind === 'text'
          ? draftFrame(current.start, end)
          : rectangleFrame(current.start, end)
      setDraft(current.frame)
    } else if (
      current.kind === 'paint' ||
      current.kind === 'erase' ||
      current.kind === 'inpaint' ||
      current.kind === 'clone'
    ) {
      current.points.push(...points)
      if (current.kind === 'clone') {
        setCloneStroke({
          points: [...current.points],
          diameter: current.diameter,
          offset: current.offset!,
        })
      } else if (current.reveal) {
        setCloneStroke({
          points: [...current.points],
          diameter: current.diameter,
          offset: { x: 0, y: 0 },
          reveal: current.layer!,
        })
      } else {
        strokeUpdates.schedule({ kind: current.kind, points })
      }
    }
  }

  const finishGesture = () => {
    const current = gesture.current
    gesture.current = null
    if (!current || !page) return
    const { zoom } = useKoharuStore.getState().camera
    if (current.kind === 'move') {
      finishTransform()
      if (!current.moved && current.collapseTo) selectLayers([current.collapseTo])
    } else if (current.kind === 'text') {
      const pointText = current.frame.width < 4 / zoom && current.frame.height < 4 / zoom
      setDraft(null)
      void (
        pointText
          ? call(commands.addPointText, current.start)
          : call(commands.addTextBox, current.frame)
      )
        .then((result) => {
          if (useKoharuStore.getState().canvasPage === page.id) editTextLayer(result.layer)
          return refresh(projectKey, pagesKey, pageKey)
        })
        .catch(() => undefined)
    } else if (current.kind === 'area') {
      setDraft(null)
      const [layer] = useKoharuStore.getState().selectedLayers
      if (!layer || current.frame.width < 4 / zoom || current.frame.height < 4 / zoom) return
      commitPending.current = true
      void enqueue(() =>
        call(commands.setGeometry, [{ layer, points: framePoints(current.frame) }]),
      )
        .then(() => {
          useKoharuStore.setState({ textTransformTarget: 'text' })
          return refresh(projectKey, pagesKey, pageKey)
        })
        .catch(() => undefined)
        .finally(() => (commitPending.current = false))
    } else if (current.kind === 'ocr') {
      setDraft(null)
      if (current.frame.width < 4 / zoom || current.frame.height < 4 / zoom) return
      readRegion(current.frame)
    } else if (current.kind === 'restore') {
      setDraft(null)
      if (current.frame.width < 4 / zoom || current.frame.height < 4 / zoom) return
      commitPending.current = true
      void enqueue(() => call(commands.commitRestore, current.revision, current.frame))
        .catch(() => undefined)
        .finally(() => (commitPending.current = false))
    } else if (
      current.kind === 'paint' ||
      current.kind === 'erase' ||
      current.kind === 'inpaint' ||
      current.kind === 'clone'
    ) {
      strokeUpdates.commit()
      try {
        if (current.kind !== 'clone' && !current.reveal) canvas?.finishStroke()
      } catch (error) {
        canvas?.cancelStroke()
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
              if (canvas)
                setCommittedEdit({
                  canvas,
                  page: page.id,
                  revision: result.revision,
                  kind: 'stroke',
                })
              selectLayers([result.layer])
              return refresh(projectKey, pagesKey, pageKey)
            })
          : current.kind === 'clone'
            ? enqueue(() =>
                call(
                  commands.commitClone,
                  current.revision,
                  current.layer,
                  current.points,
                  current.diameter,
                  current.offset!,
                  useKoharuStore.getState().cloneOpacity,
                ),
              ).then((result) => {
                if (canvas)
                  setCommittedEdit({
                    canvas,
                    page: page.id,
                    revision: result.revision,
                    kind: 'stroke',
                  })
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
                  if (canvas)
                    setCommittedEdit({
                      canvas,
                      page: page.id,
                      revision: result.revision,
                      kind: 'stroke',
                    })
                  selectLayers([result.layer])
                  return refresh(projectKey, pagesKey, pageKey)
                })
              : enqueue(() =>
                  call(commands.commitInpaint, current.revision, current.points, current.diameter),
                )
      commitPending.current = true
      void operation
        .then(() => {
          if (current.kind === 'inpaint') canvas?.cancelStroke()
        })
        .catch(() => {
          canvas?.cancelStroke()
          setCommittedEdit(null)
          setCloneStroke(null)
        })
        .finally(() => (commitPending.current = false))
    }
  }

  return (
    <main className='relative flex h-full min-h-0 min-w-0 flex-col overflow-hidden rounded-tl-2xl bg-[var(--surface-canvas)]'>
      <CanvasCommandBar />
      <div className='relative flex min-h-0 min-w-0 flex-1'>
        <ToolBar />
        {compareOriginal && page && <OriginalView page={page} />}
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
              canvasState.generation !== canvasGeneration ||
              canvasRevision === null ||
              commitPending.current ||
              committedEdit ||
              gesture.current ||
              transformActive.current ||
              event.button > 1
            )
              return
            if (event.target instanceof Element && event.target.closest('[data-canvas-control]'))
              return
            event.currentTarget.focus()
            event.currentTarget.setPointerCapture(event.pointerId)
            const physical = clientPhysicalPoint(event.clientX, event.clientY)
            const point = clientPagePoint(event.clientX, event.clientY)
            cursorPage.current = point
            setCursor(physical)

            if (event.button === 1 && tool === 'draw') {
              sampleCanvasColor(physical)
            } else if (event.button === 1 || tool === 'pan' || spaceHeld.current) {
              gesture.current = {
                kind: 'pan',
                pointer: event.pointerId,
                start: physical,
                translation: useKoharuStore.getState().camera.translation,
              }
            } else if (tool === 'select' && textTransformTarget === 'area') {
              const start = clampPagePoint(point, page.size.width, page.size.height)
              const frame = rectangleFrame(start, start)
              gesture.current = {
                kind: 'area',
                pointer: event.pointerId,
                revision: canvasRevision,
                start,
                frame,
              }
              setDraft(frame)
            } else if (tool === 'select') {
              const target = selectionAt(point)
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
            } else if (tool === 'text' || tool === 'ocr' || tool === 'restore') {
              const start =
                tool === 'text' ? point : clampPagePoint(point, page.size.width, page.size.height)
              const frame =
                tool === 'text' ? draftFrame(start, start) : rectangleFrame(start, start)
              gesture.current = {
                kind: tool,
                pointer: event.pointerId,
                revision: canvasRevision,
                start,
                frame,
              }
              setDraft(frame)
            } else if (tool === 'draw') {
              strokeUpdates.clear()
              const color = hexToRgba(brush.color)
              gesture.current = {
                kind: 'paint',
                pointer: event.pointerId,
                revision: canvasRevision,
                layer: selectedRaster?.id ?? null,
                points: [point],
                diameter: brush.diameter,
                color,
              }
              try {
                // On a reveal mask the Brush hides the layer, so it previews
                // as erasing that layer.
                canvas.beginStroke(
                  selectedRaster?.masked
                    ? {
                        kind: 'erase',
                        layer: selectedRaster.id,
                        point,
                        diameter: brush.diameter,
                      }
                    : {
                        kind: 'paint',
                        layer: selectedRaster?.id ?? null,
                        point,
                        diameter: brush.diameter,
                        color,
                      },
                )
              } catch (error) {
                gesture.current = null
                receiveError(errorMessage(error))
              }
            } else if (tool === 'eraser') {
              if (!eraserRaster) {
                receiveError('Paste or select a cleanup layer before using the Eraser.')
                return
              }
              strokeUpdates.clear()
              gesture.current = {
                kind: 'erase',
                pointer: event.pointerId,
                revision: canvasRevision,
                layer: eraserRaster.id,
                points: [point],
                diameter: brush.diameter,
                reveal: eraserRaster.masked,
              }
              // On a reveal mask the Eraser reveals the layer; ClonePreview
              // shows the revealed pixels until the committed frame does.
              if (eraserRaster.masked) {
                setCloneStroke({
                  points: [point],
                  diameter: brush.diameter,
                  offset: { x: 0, y: 0 },
                  reveal: eraserRaster.id,
                })
                return
              }
              try {
                canvas.beginStroke({
                  kind: 'erase',
                  layer: eraserRaster.id,
                  point,
                  diameter: brush.diameter,
                })
              } catch (error) {
                gesture.current = null
                receiveError(errorMessage(error))
              }
            } else if (tool === 'clone') {
              // Alt+click picks the source; a stroke copies the page from it.
              if (event.altKey) {
                useKoharuStore.setState({ cloneSource: { source: point, offset: null } })
                event.preventDefault()
                return
              }
              const clone = useKoharuStore.getState().cloneSource
              if (!clone) {
                receiveError(t('tools.cloneSourceMissing'))
                return
              }
              const offset = clone.offset ?? {
                x: clone.source.x - point.x,
                y: clone.source.y - point.y,
              }
              if (!clone.offset) useKoharuStore.setState({ cloneSource: { ...clone, offset } })
              const layer = selectedRaster?.id ?? eraserRaster?.id ?? null
              strokeUpdates.clear()
              gesture.current = {
                kind: 'clone',
                pointer: event.pointerId,
                revision: canvasRevision,
                layer,
                points: [point],
                diameter: brush.diameter,
                offset,
              }
              // The copy is made natively; ClonePreview shows it meanwhile.
              setCloneStroke({ points: [point], diameter: brush.diameter, offset })
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
            if (gesture.current?.pointer !== event.pointerId) return
            moveGesture(event.pointerId, [event.nativeEvent])
            finishGesture()
            if (event.currentTarget.hasPointerCapture(event.pointerId)) {
              event.currentTarget.releasePointerCapture(event.pointerId)
            }
          }}
          onPointerCancel={(event) => {
            if (gesture.current?.pointer !== event.pointerId) return
            cursorPage.current = null
            cancelGesture()
          }}
          onLostPointerCapture={(event) => {
            if (gesture.current?.pointer !== event.pointerId) return
            cursorPage.current = null
            cancelGesture()
          }}
          onPointerLeave={(event) => {
            cursorPage.current = null
            if (!event.currentTarget.hasPointerCapture(event.pointerId)) {
              setHovered(null)
              setCursor(null)
            }
          }}
          onWheel={(event) => {
            if (!page) return
            event.preventDefault()
            if (event.altKey && isBrushTool(tool)) {
              cursorPage.current = clientPagePoint(event.clientX, event.clientY)
              setCursor(clientPhysicalPoint(event.clientX, event.clientY))
              if (event.deltaY !== 0) {
                const currentBrush = useKoharuStore.getState().brush
                const delta = event.deltaY < 0 ? BRUSH_DIAMETER_STEP : -BRUSH_DIAMETER_STEP
                const nextDiameter = clamp(
                  currentBrush.diameter + delta,
                  MIN_BRUSH_DIAMETER,
                  maxBrushDiameter(page.size),
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
          {pendingPage && pendingPage !== canvasPage && (
            <PagePreview page={pages?.find((summary) => summary.id === pendingPage)} />
          )}
          {/* Mounted with the tool, so the source is loaded before the first stroke. */}
          {page && (tool === 'clone' || cloneStroke) && (
            <ClonePreview page={page} revision={canvasRevision} stroke={cloneStroke} />
          )}
          <canvas
            ref={setCanvasElement}
            data-testid='webgpu-canvas'
            aria-hidden
            className='pointer-events-none absolute inset-0 block size-full'
          />
          {page && canvasState.status === 'ready' && (
            <CanvasOverlay
              page={page}
              selected={selected}
              hovered={hovered}
              frames={layerFrames}
              previews={previews}
              draft={draft}
              cursor={cursor}
              brushSize={brush.diameter}
              showBrushCursor={isBrushTool(tool)}
              cloneSource={tool === 'clone' ? cloneSource : null}
              showDetectionRegions={tool === 'select' || tool === 'ocr'}
              onCreateUnboxedText={readUnboxedText}
              onDismissUnboxedText={dismissUnboxedText}
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
