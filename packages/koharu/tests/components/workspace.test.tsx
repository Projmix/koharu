import { QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { CanvasWorkspace } from '@/components/editor/CanvasWorkspace'
import { pageKey, pagesKey, projectKey, queryClient } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import { commands, type Layer } from '@koharu/bridge/protocol'
import { TooltipProvider } from '@koharu/ui/components/tooltip'

const canvas = vi.hoisted(() => ({
  resize: vi.fn(),
  setView: vi.fn(),
  stageManifest: vi.fn(),
  installResource: vi.fn(),
  activateFrame: vi.fn(),
  activatePage: vi.fn(),
  clear: vi.fn(),
  previewOpacity: vi.fn(),
  beginTransform: vi.fn(),
  updateTransform: vi.fn(),
  finishTransform: vi.fn(),
  cancelTransform: vi.fn(),
  beginStroke: vi.fn(),
  extendStroke: vi.fn(),
  finishStroke: vi.fn(),
  cancelStroke: vi.fn(),
  sampleColor: vi.fn(),
  dispose: vi.fn(),
}))

const canvasState = vi.hoisted(() => ({
  canvas,
  error: null as Error | null,
  generation: 1 as number | null,
  hasFrame: true,
  retry: vi.fn(),
  status: 'ready' as 'loading' | 'switching' | 'ready' | 'recovering' | 'error',
}))
const prefetchCanvasPages = vi.hoisted(() => vi.fn(async () => []))

vi.mock('@/components/editor/useCanvas', () => ({
  useCanvas: () => canvasState,
}))
vi.mock('@koharu/bridge/canvas', () => ({
  prefetchCanvasPages,
  workspaceColor: () => [245, 245, 245],
}))

const layer: Layer = {
  type: 'image',
  id: 'element',
  parent: 'page',
  geometry: {
    points: [
      { x: 10, y: 20 },
      { x: 110, y: 20 },
      { x: 110, y: 70 },
      { x: 10, y: 70 },
    ],
  },
  visibility: { visible: true, opacity: 1 },
  image: 'image',
}

const paintLayer: Layer = {
  type: 'raster',
  id: 'paint',
  parent: 'page',
  visibility: { visible: true, opacity: 1 },
  image: 'paint-image',
  name: 'Paint 1',
  kind: 'paint',
}

let nextAnimationFrame = 1
let animationFrames = new Map<number, FrameRequestCallback>()

beforeEach(() => {
  nextAnimationFrame = 1
  animationFrames = new Map()
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    const frame = nextAnimationFrame++
    animationFrames.set(frame, callback)
    return frame
  })
  vi.stubGlobal('cancelAnimationFrame', (frame: number) => animationFrames.delete(frame))
  canvasState.error = null
  canvasState.generation = 1
  canvasState.hasFrame = true
  canvasState.status = 'ready'
  prefetchCanvasPages.mockClear()
})

afterEach(() => vi.unstubAllGlobals())

function installProject() {
  const page = {
    id: 'page',
    label: 'Page',
    size: { width: 1000, height: 1000 },
    layers: [layer],
    regions: [],
  }
  queryClient.setQueryData(projectKey, {
    name: 'Book',
    revision: 1,
    active_page: 'page',
    can_undo: false,
    can_redo: false,
  })
  queryClient.setQueryData(pagesKey, [])
  queryClient.setQueryData(pageKey, page)
  useKoharuStore.setState({ selectedLayers: [], tool: 'select' })
  useKoharuStore.setState({
    canvasPage: 'page',
    canvasRevision: 1,
    canvasGeneration: 1,
    canvasSize: [1000, 1000],
  })
}

function renderWorkspace() {
  render(
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <CanvasWorkspace />
      </TooltipProvider>
    </QueryClientProvider>,
  )
  const surface = screen.getByLabelText('Koharu canvas')
  Object.defineProperty(surface, 'getBoundingClientRect', {
    value: () => ({ x: 10, y: 20, width: 800, height: 600 }),
  })
  return surface
}

describe('canvas interaction adapter', () => {
  it('prefetches only after the authoritative canvas page and generation are active', async () => {
    installProject()
    queryClient.setQueryData(pagesKey, [
      {
        id: 'page',
        label: 'Page',
        size: { width: 1000, height: 1000 },
        source_asset: null,
        layer_count: 1,
      },
      {
        id: 'next',
        label: 'Next',
        size: { width: 1000, height: 1000 },
        source_asset: null,
        layer_count: 1,
      },
    ])
    useKoharuStore.setState({ canvasPage: 'previous' })
    renderWorkspace()

    await Promise.resolve()
    expect(prefetchCanvasPages).not.toHaveBeenCalled()

    act(() => useKoharuStore.setState({ canvasPage: 'page' }))
    await waitFor(() => expect(prefetchCanvasPages).toHaveBeenCalledWith(['next']))
  })

  it('renders an accessible browser canvas and keeps camera updates local', async () => {
    installProject()
    const surface = renderWorkspace()
    expect(screen.getByTestId('webgpu-canvas')).toBeInstanceOf(HTMLCanvasElement)

    fireEvent.wheel(surface, { clientX: 100, clientY: 100, deltaY: 4 })
    fireEvent.wheel(surface, { clientX: 100, clientY: 100, deltaY: 6 })

    await waitFor(() => expect(canvas.setView).toHaveBeenLastCalledWith(1, [0, -10]))
  })

  it('preserves a manual camera when a newer canvas generation becomes ready', () => {
    installProject()
    renderWorkspace()
    const camera = { zoom: 2, translation: [-120, -80] as [number, number], fitted: false }

    act(() => useKoharuStore.setState({ camera }))
    act(() => {
      canvasState.status = 'switching'
      useKoharuStore.setState({ canvasGeneration: 2 })
    })
    act(() => {
      canvasState.status = 'ready'
      canvasState.generation = 2
      useKoharuStore.setState({ canvasRevision: 2 })
    })

    expect(useKoharuStore.getState().camera).toEqual(camera)
  })

  it('suppresses Alt browser handling while an editor input retains focus', () => {
    installProject()
    renderWorkspace()
    const input = document.createElement('input')
    document.body.append(input)
    input.focus()

    const event = new KeyboardEvent('keydown', { key: 'Alt', bubbles: true, cancelable: true })
    input.dispatchEvent(event)

    expect(event.defaultPrevented).toBe(true)
    input.remove()
  })

  it('activates tools only through Ctrl plus the physical shortcut key', () => {
    installProject()
    renderWorkspace()

    fireEvent.keyDown(window, { key: 'o', code: 'KeyO' })
    expect(useKoharuStore.getState().tool).toBe('select')

    fireEvent.keyDown(window, { key: 'щ', code: 'KeyO', ctrlKey: true })
    expect(useKoharuStore.getState().tool).toBe('ocr')
  })

  it('leaves an editor field and resets the canvas interaction on Escape', () => {
    installProject()
    renderWorkspace()
    useKoharuStore.setState({ tool: 'draw', selectedLayers: ['element'] })
    const input = document.body.appendChild(document.createElement('input'))
    input.addEventListener('keydown', (event) => event.stopPropagation())
    input.focus()

    const event = new KeyboardEvent('keydown', {
      key: 'Escape',
      bubbles: true,
      cancelable: true,
    })
    input.dispatchEvent(event)

    expect(event.defaultPrevented).toBe(true)
    expect(input).not.toHaveFocus()
    expect(useKoharuStore.getState()).toMatchObject({ tool: 'select', selectedLayers: [] })
    input.remove()
  })

  it('announces WebGPU startup failures and offers recovery', () => {
    installProject()
    canvasState.status = 'error'
    canvasState.error = new Error('No compatible WebGPU adapter was found.')
    renderWorkspace()
    expect(screen.getByRole('alert')).toHaveTextContent('WebGPU canvas unavailable')
    expect(screen.getByRole('alert')).toHaveTextContent('No compatible WebGPU adapter was found.')
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
    expect(canvasState.retry).toHaveBeenCalledOnce()
  })

  it('previews brush input locally and sends only the durable paint commit to Rust', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'draw', brush: { diameter: 48, color: '#FFFFFF' } })
    const commit = vi
      .spyOn(commands, 'commitPaint')
      .mockResolvedValue({ revision: 2, layer: 'paint' })
    const surface = renderWorkspace()
    expect(surface).toHaveStyle({ cursor: 'none' })

    fireEvent.pointerDown(surface, { button: 0, pointerId: 7, clientX: 30, clientY: 40 })
    fireEvent.pointerMove(surface, { pointerId: 7, clientX: 55, clientY: 65 })
    fireEvent.pointerUp(surface, { pointerId: 7, clientX: 58, clientY: 70 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(canvas.beginStroke).toHaveBeenCalledWith({
      kind: 'paint',
      layer: null,
      point: { x: 20, y: 20 },
      diameter: 48,
      color: [255, 255, 255, 255],
    })
    expect(canvas.extendStroke).toHaveBeenCalledWith(expect.arrayContaining([{ x: 45, y: 45 }]))
    expect(canvas.finishStroke).toHaveBeenCalledOnce()
    expect(commit).toHaveBeenCalledWith(1, null, expect.arrayContaining([{ x: 45, y: 45 }]), {
      diameter: 48,
      color: [255, 255, 255, 255],
    })
  })

  it('temporarily samples the drawing color with the middle mouse button', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'draw', brush: { diameter: 48, color: '#FFFFFF' } })
    canvas.sampleColor.mockResolvedValue([17, 34, 51, 255])
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 1, pointerId: 71, clientX: 30, clientY: 40 })
    fireEvent.pointerUp(surface, { button: 1, pointerId: 71, clientX: 30, clientY: 40 })

    await waitFor(() => expect(useKoharuStore.getState().brush.color).toBe('#112233'))
    expect(canvas.sampleColor).toHaveBeenCalledWith({ x: 20, y: 20 })
    expect(canvas.beginStroke).not.toHaveBeenCalled()
  })

  it.each([
    ['revision', { canvasRevision: 2 }],
    ['generation', { canvasGeneration: 2 }],
  ])('cancels an active gesture when the canvas %s changes', async (_name, update) => {
    installProject()
    useKoharuStore.setState({ tool: 'draw', brush: { diameter: 48, color: '#FFFFFF' } })
    const commit = vi
      .spyOn(commands, 'commitPaint')
      .mockResolvedValue({ revision: 2, layer: 'paint' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 8, clientX: 30, clientY: 40 })
    expect(canvas.beginStroke).toHaveBeenCalledOnce()

    act(() => {
      useKoharuStore.setState(update)
    })

    await waitFor(() => expect(canvas.cancelStroke).toHaveBeenCalledOnce())
    fireEvent.pointerUp(surface, { pointerId: 8, clientX: 30, clientY: 40 })
    expect(canvas.finishStroke).not.toHaveBeenCalled()
    expect(commit).not.toHaveBeenCalled()
  })

  it('uses rendered text bounds for hit testing and semantic transforms', async () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [
        {
          type: 'text',
          id: 'element',
          parent: 'page',
          geometry: layer.geometry,
          visibility: { visible: true, opacity: 1 },
          content: {
            id: 'content',
            source: { text: 'Source', language: 'en' },
            translation: { text: 'Rendered', language: null },
            role: null,
            source_region: null,
          },
          typography: null,
          layout: 'paragraph',
          automatic_region: null,
        },
      ],
    }))
    useKoharuStore.setState({
      layerFrames: {
        element: { x: 30, y: 40, width: 50, height: 20, angle_degrees: 0 },
      },
    })
    const commit = vi.spyOn(commands, 'commitTransform').mockResolvedValue(2)
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 9, clientX: 50, clientY: 60 })
    fireEvent.pointerMove(surface, { pointerId: 9, clientX: 70, clientY: 80 })
    fireEvent.pointerUp(surface, { pointerId: 9, clientX: 70, clientY: 80 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(useKoharuStore.getState().selectedLayers).toEqual(['element'])
    expect(canvas.beginTransform).toHaveBeenCalledWith([
      {
        element: 'element',
        frame: { x: 30, y: 40, width: 50, height: 20, angle_degrees: 0 },
      },
    ])
    expect(canvas.updateTransform).toHaveBeenCalledWith(
      expect.arrayContaining([
        expect.objectContaining({
          element: 'element',
          frame: expect.objectContaining({ x: 50, y: 60 }),
        }),
      ]),
    )
    expect(canvas.finishTransform).toHaveBeenCalledOnce()
    expect(commit).toHaveBeenCalledWith(
      1,
      expect.arrayContaining([
        expect.objectContaining({
          element: 'element',
          frame: expect.objectContaining({ x: 50, y: 60 }),
        }),
      ]),
    )
  })

  it('selects a text layer through its detected text and bubble regions', () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [
        {
          type: 'text',
          id: 'dialogue',
          parent: 'page',
          geometry: null,
          visibility: { visible: true, opacity: 1 },
          content: {
            id: 'content',
            source: { text: '原文', language: 'ja' },
            translation: null,
            role: 'dialogue',
            source_region: 'text-region',
          },
          typography: null,
          layout: 'paragraph',
          automatic_region: 'bubble-region',
        },
      ],
      regions: [
        {
          id: 'bubble-region',
          parent: 'page',
          geometry: {
            points: [
              { x: 10, y: 10 },
              { x: 100, y: 10 },
              { x: 100, y: 100 },
              { x: 10, y: 100 },
            ],
          },
          kind: 'bubble',
          label: 'bubble',
        },
        {
          id: 'text-region',
          parent: 'page',
          geometry: {
            points: [
              { x: 20, y: 20 },
              { x: 40, y: 20 },
              { x: 40, y: 40 },
              { x: 20, y: 40 },
            ],
          },
          kind: 'text',
          label: 'text',
        },
      ],
    }))
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 20, clientX: 35, clientY: 45 })
    fireEvent.pointerUp(surface, { pointerId: 20, clientX: 35, clientY: 45 })
    expect(useKoharuStore.getState().selectedLayers).toEqual(['dialogue'])

    act(() => useKoharuStore.getState().selectLayers([]))
    fireEvent.pointerDown(surface, { button: 0, pointerId: 21, clientX: 90, clientY: 90 })
    fireEvent.pointerUp(surface, { pointerId: 21, clientX: 90, clientY: 90 })
    expect(useKoharuStore.getState().selectedLayers).toEqual(['dialogue'])
  })

  it('shift-selects and moves multiple detected text regions together', async () => {
    installProject()
    const ocrLayer = (id: string, region: string): Layer => ({
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
        source_region: region,
      },
      typography: null,
      layout: 'paragraph',
      automatic_region: null,
    })
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [ocrLayer('first', 'first-region'), ocrLayer('second', 'second-region')],
      regions: [
        {
          id: 'first-region',
          parent: 'page',
          geometry: {
            points: [
              { x: 20, y: 20 },
              { x: 40, y: 20 },
              { x: 40, y: 40 },
              { x: 20, y: 40 },
            ],
          },
          kind: 'dev.koharu.region.text',
          label: null,
        },
        {
          id: 'second-region',
          parent: 'page',
          geometry: {
            points: [
              { x: 80, y: 20 },
              { x: 100, y: 20 },
              { x: 100, y: 40 },
              { x: 80, y: 40 },
            ],
          },
          kind: 'dev.koharu.region.text',
          label: null,
        },
      ],
    }))
    const setGeometry = vi.spyOn(commands, 'setGeometry').mockResolvedValue(null)
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 22, clientX: 35, clientY: 45 })
    fireEvent.pointerUp(surface, { pointerId: 22, clientX: 35, clientY: 45 })
    fireEvent.pointerDown(surface, {
      button: 0,
      pointerId: 23,
      clientX: 95,
      clientY: 45,
      shiftKey: true,
    })
    fireEvent.pointerUp(surface, { pointerId: 23, clientX: 95, clientY: 45 })

    expect(useKoharuStore.getState().selectedLayers).toEqual(['first', 'second'])

    fireEvent.pointerDown(surface, { button: 0, pointerId: 24, clientX: 95, clientY: 45 })
    fireEvent.pointerMove(surface, { pointerId: 24, clientX: 115, clientY: 65 })
    fireEvent.pointerUp(surface, { pointerId: 24, clientX: 115, clientY: 65 })

    await waitFor(() => expect(setGeometry).toHaveBeenCalledOnce())
    expect(setGeometry).toHaveBeenCalledWith([
      expect.objectContaining({ layer: 'first-region', points: expect.any(Array) }),
      expect.objectContaining({ layer: 'second-region', points: expect.any(Array) }),
    ])
  })

  it('shows detected regions only while the select tool is active', () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [
        ...page.layers,
        {
          type: 'text',
          id: 'ocr-layer',
          parent: 'page',
          geometry: null,
          visibility: { visible: true, opacity: 1 },
          content: {
            id: 'ocr-content',
            source: { text: 'Source', language: 'ja' },
            translation: null,
            role: 'dialogue',
            source_region: 'text-region',
          },
          typography: null,
          layout: 'paragraph',
          automatic_region: null,
        } satisfies Layer,
      ],
      regions: [
        {
          id: 'text-region',
          parent: 'page',
          geometry: {
            points: [
              { x: 20, y: 20 },
              { x: 40, y: 20 },
              { x: 40, y: 40 },
              { x: 20, y: 40 },
            ],
          },
          kind: 'dev.koharu.region.text',
          label: null,
        },
        {
          id: 'bubble-region',
          parent: 'page',
          geometry: {
            points: [
              { x: 10, y: 10 },
              { x: 100, y: 10 },
              { x: 100, y: 100 },
              { x: 10, y: 100 },
            ],
          },
          kind: 'dev.koharu.region.bubble',
          label: null,
        },
      ],
    }))
    renderWorkspace()

    expect(screen.getByTestId('detection-regions').querySelectorAll('polygon')).toHaveLength(2)
    expect(screen.getByTestId('ocr-region-number-text-region')).toHaveTextContent('1')

    act(() => useKoharuStore.getState().setTool('eraser'))
    expect(screen.queryByTestId('detection-regions')).not.toBeInTheDocument()

    act(() => useKoharuStore.getState().setTool('ocr'))
    expect(screen.getByTestId('detection-regions')).toBeInTheDocument()
  })

  it('edits the selected text region with move, resize, and rotate controls', async () => {
    installProject()
    const region = {
      id: 'text-region',
      parent: 'page',
      geometry: {
        points: [
          { x: 20, y: 20 },
          { x: 60, y: 20 },
          { x: 60, y: 40 },
          { x: 20, y: 40 },
        ],
      },
      kind: 'dev.koharu.region.text',
      label: null,
    }
    queryClient.setQueryData(pageKey, (page: { layers: Layer[]; regions: unknown[] }) => ({
      ...page,
      layers: [
        {
          type: 'text',
          id: 'dialogue',
          parent: 'page',
          geometry: region.geometry,
          visibility: { visible: true, opacity: 1 },
          content: {
            id: 'content',
            source: { text: 'Source', language: 'ja' },
            translation: null,
            role: 'dialogue',
            source_region: 'text-region',
          },
          typography: null,
          layout: 'paragraph',
          automatic_region: null,
        },
      ],
      regions: [region],
    }))
    useKoharuStore.setState({ selectedLayers: ['dialogue'], tool: 'select' })
    const setGeometry = vi.spyOn(commands, 'setGeometry').mockResolvedValue(null)
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 34, clientX: 35, clientY: 45 })
    fireEvent.pointerMove(surface, { pointerId: 34, clientX: 55, clientY: 65 })
    fireEvent.pointerUp(surface, { pointerId: 34, clientX: 55, clientY: 65 })
    await waitFor(() => expect(setGeometry).toHaveBeenCalledOnce())
    expect(setGeometry.mock.calls[0][0][0]).toMatchObject({
      layer: 'text-region',
      points: expect.any(Array),
    })

    const handle = document.querySelector<HTMLElement>('[data-resize-handle="e"]')!
    fireEvent.pointerDown(handle, { button: 0, pointerId: 35, clientX: 70, clientY: 40 })
    fireEvent.pointerMove(handle, { pointerId: 35, clientX: 90, clientY: 40 })
    fireEvent.pointerUp(handle, { pointerId: 35, clientX: 90, clientY: 40 })

    await waitFor(() => expect(setGeometry).toHaveBeenCalledTimes(2))
    expect(setGeometry).toHaveBeenCalledWith([
      expect.objectContaining({ layer: 'text-region', points: expect.any(Array) }),
    ])
    expect(setGeometry.mock.calls[0][0][0].points).toHaveLength(4)
  })

  it('queues repeated undo commands and supports Ctrl+Y redo globally', async () => {
    installProject()
    const firstUndo = deferred<null>()
    const undo = vi
      .spyOn(commands, 'undo')
      .mockImplementationOnce(() => firstUndo.promise)
      .mockResolvedValue(null)
    const redo = vi.spyOn(commands, 'redo').mockResolvedValue(null)
    renderWorkspace()

    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    await waitFor(() => expect(undo).toHaveBeenCalledOnce())

    firstUndo.resolve(null)
    await waitFor(() => expect(undo).toHaveBeenCalledTimes(2))

    fireEvent.keyDown(window, { key: 'y', ctrlKey: true })
    await waitFor(() => expect(redo).toHaveBeenCalledOnce())
  })

  it('resizes a selected layer through Koharu selection controls', async () => {
    installProject()
    useKoharuStore.setState({ selectedLayers: ['element'] })
    const commit = vi.spyOn(commands, 'commitTransform').mockResolvedValue(2)
    renderWorkspace()
    Object.defineProperty(screen.getByTestId('canvas-overlay'), 'getBoundingClientRect', {
      value: () => ({ x: 10, y: 20, width: 800, height: 600 }),
    })
    const handle = document.querySelector<HTMLElement>('[data-resize-handle="e"]')!

    fireEvent.pointerDown(handle, { button: 0, pointerId: 10, clientX: 120, clientY: 65 })
    fireEvent.pointerMove(handle, { pointerId: 10, clientX: 140, clientY: 65 })
    fireEvent.pointerUp(handle, { pointerId: 10, clientX: 140, clientY: 65 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(canvas.beginTransform).toHaveBeenCalledWith([
      { element: 'element', frame: { x: 10, y: 20, width: 100, height: 50, angle_degrees: 0 } },
    ])
    expect(canvas.updateTransform).toHaveBeenCalledWith(
      expect.arrayContaining([
        {
          element: 'element',
          frame: { x: 10, y: 20, width: 120, height: 50, angle_degrees: 0 },
        },
      ]),
    )
    expect(commit).toHaveBeenCalledWith(
      1,
      expect.arrayContaining([
        {
          element: 'element',
          frame: { x: 10, y: 20, width: 120, height: 50, angle_degrees: 0 },
        },
      ]),
    )
  })

  it('shows the automatic region behind the selected text controls', () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [
        {
          type: 'text',
          id: 'element',
          parent: 'page',
          geometry: null,
          visibility: { visible: true, opacity: 1 },
          content: {
            id: 'content',
            source: { text: 'Source', language: 'en' },
            translation: { text: 'Rendered', language: null },
            role: null,
            source_region: null,
          },
          typography: null,
          layout: 'paragraph',
          automatic_region: 'bubble',
        },
      ],
      regions: [
        {
          id: 'bubble',
          parent: 'page',
          geometry: {
            points: [
              { x: 20, y: 30 },
              { x: 100, y: 30 },
              { x: 100, y: 90 },
              { x: 20, y: 90 },
            ],
          },
          kind: 'bubble',
          label: null,
        },
      ],
    }))
    useKoharuStore.setState({
      selectedLayers: ['element'],
      layerFrames: {
        element: { x: 30, y: 40, width: 50, height: 20, angle_degrees: 0 },
      },
    })

    renderWorkspace()

    expect(screen.getByTestId('text-fit-region').querySelector('polygon')).toHaveAttribute(
      'points',
      '20,30 100,30 100,90 20,90',
    )
  })

  it('targets the selected raster layer with the eraser', async () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, paintLayer],
    }))
    useKoharuStore.setState({ tool: 'eraser', selectedLayers: ['paint'] })
    const commit = vi
      .spyOn(commands, 'commitErase')
      .mockResolvedValue({ revision: 2, layer: 'paint' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 11, clientX: 30, clientY: 40 })
    fireEvent.pointerUp(surface, { pointerId: 11, clientX: 30, clientY: 40 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(canvas.beginStroke).toHaveBeenCalledWith({
      kind: 'erase',
      layer: 'paint',
      point: { x: 20, y: 20 },
      diameter: 48,
    })
    expect(commit).toHaveBeenCalledWith(1, 'paint', expect.arrayContaining([{ x: 20, y: 20 }]), 48)
  })

  it('maps the Remove tool to an inpainting mask gesture', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'remove' })
    const commit = vi.spyOn(commands, 'commitInpaint').mockResolvedValue('job')
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 12, clientX: 30, clientY: 40 })
    fireEvent.pointerUp(surface, { pointerId: 12, clientX: 30, clientY: 40 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(canvas.beginStroke).toHaveBeenCalledWith({
      kind: 'inpaint',
      layer: null,
      point: { x: 20, y: 20 },
      diameter: 48,
    })
    expect(commit).toHaveBeenCalledWith(1, expect.arrayContaining([{ x: 20, y: 20 }]), 48)
  })

  it('creates point text on click and paragraph text on drag', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'text' })
    const point = vi
      .spyOn(commands, 'addPointText')
      .mockResolvedValue({ revision: 2, layer: 'point-text' })
    const box = vi
      .spyOn(commands, 'addTextBox')
      .mockResolvedValue({ revision: 3, layer: 'box-text' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 13, clientX: 30, clientY: 40 })
    fireEvent.pointerUp(surface, { pointerId: 13, clientX: 30, clientY: 40 })
    await waitFor(() => expect(point).toHaveBeenCalledWith({ x: 20, y: 20 }))

    fireEvent.pointerDown(surface, { button: 0, pointerId: 14, clientX: 40, clientY: 50 })
    fireEvent.pointerMove(surface, { pointerId: 14, clientX: 140, clientY: 110 })
    fireEvent.pointerUp(surface, { pointerId: 14, clientX: 140, clientY: 110 })
    await waitFor(() => expect(box).toHaveBeenCalledOnce())
    expect(box).toHaveBeenCalledWith({
      x: 30,
      y: 30,
      width: 100,
      height: 60,
      angle_degrees: 0,
    })
  })

  it('creates a manual OCR region and runs only OCR for that region', async () => {
    installProject()
    const addRegion = vi
      .spyOn(commands, 'addOcrRegion')
      .mockResolvedValue({ revision: 2, layer: 'ocr-layer', region: 'ocr-region' })
    const process = vi.spyOn(commands, 'process').mockResolvedValue('ocr-job')
    const surface = renderWorkspace()

    fireEvent.keyDown(window, { key: 'o', code: 'KeyO', ctrlKey: true })
    expect(useKoharuStore.getState().tool).toBe('ocr')

    fireEvent.pointerDown(surface, { button: 0, pointerId: 30, clientX: 40, clientY: 50 })
    fireEvent.pointerMove(surface, { pointerId: 30, clientX: 140, clientY: 110 })
    fireEvent.pointerUp(surface, { pointerId: 30, clientX: 140, clientY: 110 })

    await waitFor(() => expect(process).toHaveBeenCalledOnce())
    expect(addRegion).toHaveBeenCalledWith({
      x: 30,
      y: 30,
      width: 100,
      height: 60,
      angle_degrees: 0,
    })
    expect(process).toHaveBeenCalledWith(
      { scope: 'entities', value: ['ocr-region'] },
      { operation: 'stages', stages: ['ocr'] },
    )
    expect(useKoharuStore.getState().selectedLayers).toEqual(['ocr-layer'])
    expect(useKoharuStore.getState().tool).toBe('ocr')
  })

  it('clamps an OCR rectangle to the page', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'ocr' })
    const addRegion = vi
      .spyOn(commands, 'addOcrRegion')
      .mockResolvedValue({ revision: 2, layer: 'ocr-layer', region: 'ocr-region' })
    vi.spyOn(commands, 'process').mockResolvedValue('ocr-job')
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 31, clientX: 0, clientY: 0 })
    fireEvent.pointerMove(surface, { pointerId: 31, clientX: 1200, clientY: 1200 })
    fireEvent.pointerUp(surface, { pointerId: 31, clientX: 1200, clientY: 1200 })

    await waitFor(() => expect(addRegion).toHaveBeenCalledOnce())
    expect(addRegion).toHaveBeenCalledWith({
      x: 0,
      y: 0,
      width: 1000,
      height: 1000,
      angle_degrees: 0,
    })
  })

  it('ignores a tiny OCR rectangle and input while another job is running', () => {
    installProject()
    useKoharuStore.setState({ tool: 'ocr' })
    const addRegion = vi.spyOn(commands, 'addOcrRegion')
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 32, clientX: 40, clientY: 50 })
    fireEvent.pointerMove(surface, { pointerId: 32, clientX: 42, clientY: 150 })
    fireEvent.pointerUp(surface, { pointerId: 32, clientX: 42, clientY: 150 })
    expect(addRegion).not.toHaveBeenCalled()

    act(() =>
      useKoharuStore.setState({
        jobs: {
          active: {
            id: 'active',
            state: 'running',
            completed: 0,
            total: 1,
            target: null,
            stage: 'ocr',
            model: 'ocr-model',
            error: null,
          },
        },
      }),
    )
    fireEvent.pointerDown(surface, { button: 0, pointerId: 33, clientX: 40, clientY: 50 })
    fireEvent.pointerMove(surface, { pointerId: 33, clientX: 140, clientY: 150 })
    fireEvent.pointerUp(surface, { pointerId: 33, clientX: 140, clientY: 150 })
    expect(addRegion).not.toHaveBeenCalled()
  })

  it('keeps a failed manual OCR region selected for retry or deletion', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'ocr' })
    vi.spyOn(commands, 'addOcrRegion').mockResolvedValue({
      revision: 2,
      layer: 'failed-ocr-layer',
      region: 'failed-ocr-region',
    })
    const process = vi.spyOn(commands, 'process').mockRejectedValue(new Error('OCR failed'))
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 34, clientX: 40, clientY: 50 })
    fireEvent.pointerMove(surface, { pointerId: 34, clientX: 140, clientY: 110 })
    fireEvent.pointerUp(surface, { pointerId: 34, clientX: 140, clientY: 110 })

    await waitFor(() => expect(process).toHaveBeenCalledOnce())
    expect(useKoharuStore.getState().selectedLayers).toEqual(['failed-ocr-layer'])
    expect(useKoharuStore.getState().tool).toBe('ocr')
  })
})

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}
