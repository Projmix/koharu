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

const cleanupLayer: Layer = {
  type: 'raster',
  id: 'cleanup',
  parent: 'page',
  visibility: { visible: true, opacity: 1 },
  image: 'cleanup-image',
  name: 'Cleanup',
  kind: 'cleanup',
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
  // Hoisted canvas mocks outlive restoreMocks, so stateful implementations must not leak.
  for (const mock of Object.values(canvas)) mock.mockReset()
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
    unboxed_text: [],
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

  it('suppresses bare Alt on the canvas but leaves editor input handling untouched', () => {
    installProject()
    const surface = renderWorkspace()
    const canvasDown = new KeyboardEvent('keydown', {
      key: 'Alt',
      bubbles: true,
      cancelable: true,
    })
    const canvasUp = new KeyboardEvent('keyup', {
      key: 'Alt',
      bubbles: true,
      cancelable: true,
    })
    surface.dispatchEvent(canvasDown)
    surface.dispatchEvent(canvasUp)
    expect(canvasDown.defaultPrevented).toBe(true)
    expect(canvasUp.defaultPrevented).toBe(true)

    const input = document.createElement('input')
    document.body.append(input)
    input.focus()
    const inputDown = new KeyboardEvent('keydown', {
      key: 'Alt',
      bubbles: true,
      cancelable: true,
    })
    const inputUp = new KeyboardEvent('keyup', {
      key: 'Alt',
      bubbles: true,
      cancelable: true,
    })
    input.dispatchEvent(inputDown)
    input.dispatchEvent(inputUp)
    expect(inputDown.defaultPrevented).toBe(false)
    expect(inputUp.defaultPrevented).toBe(false)
    input.remove()
  })

  it('activates tools only through exact Alt plus the physical shortcut key', () => {
    installProject()
    renderWorkspace()

    fireEvent.keyDown(window, { key: 'o', code: 'KeyO' })
    expect(useKoharuStore.getState().tool).toBe('select')

    fireEvent.keyDown(window, { key: 'щ', code: 'KeyO', ctrlKey: true })
    expect(useKoharuStore.getState().tool).toBe('select')

    fireEvent.keyDown(window, { key: 'щ', code: 'KeyO', altKey: true })
    expect(useKoharuStore.getState().tool).toBe('ocr')

    const fitRequest = useKoharuStore.getState().fitCanvasRequest
    const fit = new KeyboardEvent('keydown', {
      key: 'а',
      code: 'KeyF',
      altKey: true,
      bubbles: true,
      cancelable: true,
    })
    act(() => window.dispatchEvent(fit))
    expect(fit.defaultPrevented).toBe(true)
    expect(useKoharuStore.getState().fitCanvasRequest).toBe(fitRequest + 1)
  })

  it('copies, cuts, and pastes selected text layers with the Koharu MIME payload', async () => {
    installProject()
    const textLayer = (id: string, source: string, translation: string): Layer => ({
      type: 'text',
      id,
      parent: 'text-group',
      geometry: layer.geometry,
      angle_degrees: null,
      visibility: { visible: true, opacity: 1 },
      content: {
        id: `${id}-content`,
        source: { text: source, language: 'ja' },
        translation: { text: translation, language: 'en', generated: true },
        translation_review: null,
        ocr_unverified: false,
        cleaning_unverified: false,
        markup: null,
        role: 'dev.koharu.text.dialogue',
        source_region: null,
      },
      typography: null,
      layout: 'paragraph',
      automatic_region: null,
    })
    const first = textLayer('first', '一', 'One')
    const second = textLayer('second', '二', 'Two')
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [first, second],
    }))
    useKoharuStore.setState({ selectedLayers: ['second', 'first'] })
    const surface = renderWorkspace()

    const data = new Map<string, string>()
    const clipboard = {
      items: [],
      get types() {
        return [...data.keys()]
      },
      getData: (type: string) => data.get(type) ?? '',
      setData: (type: string, value: string) => {
        data.set(type, value)
      },
    }
    const dispatch = (type: 'copy' | 'cut' | 'paste') => {
      const event = new Event(type, { bubbles: true, cancelable: true }) as ClipboardEvent
      Object.defineProperty(event, 'clipboardData', { value: clipboard })
      window.dispatchEvent(event)
      return event
    }

    expect(dispatch('copy').defaultPrevented).toBe(true)
    expect(data.get('text/plain')).toBe('One\nTwo')
    const copied = JSON.parse(data.get('application/x-koharu-text-layers+json')!)
    expect(copied).toMatchObject({
      version: 1,
      mode: 'copy',
      source_page: 'page',
      layers: [{ layer: 'first' }, { layer: 'second' }],
    })

    dispatch('cut')
    const moved = JSON.parse(data.get('application/x-koharu-text-layers+json')!)
    expect(moved.mode).toBe('move')
    expect(useKoharuStore.getState().selectedLayers).toEqual(['second', 'first'])

    const paste = vi.spyOn(commands, 'pasteTextBlocks').mockResolvedValue({
      project: queryClient.getQueryData(projectKey)!,
      page: queryClient.getQueryData(pageKey)!,
      layers: ['moved'],
    })
    expect(dispatch('paste').defaultPrevented).toBe(true)
    await waitFor(() => expect(paste).toHaveBeenCalledWith('page', moved, null))
    expect(useKoharuStore.getState().selectedLayers).toEqual(['moved'])
    expect(useKoharuStore.getState().pendingTextFocus).toBe('moved')
    expect(useKoharuStore.getState().tool).toBe('select')

    fireEvent.pointerMove(surface, { pointerId: 1, clientX: 110, clientY: 220 })
    dispatch('paste')
    await waitFor(() => expect(paste).toHaveBeenLastCalledWith('page', moved, { x: 100, y: 200 }))

    fireEvent.pointerLeave(surface, { pointerId: 1 })
    dispatch('paste')
    await waitFor(() => expect(paste).toHaveBeenLastCalledWith('page', moved, null))

    data.clear()
    const input = document.body.appendChild(document.createElement('input'))
    input.focus()
    const nativeCopy = new Event('copy', { bubbles: true, cancelable: true }) as ClipboardEvent
    Object.defineProperty(nativeCopy, 'clipboardData', { value: clipboard })
    input.dispatchEvent(nativeCopy)
    expect(nativeCopy.defaultPrevented).toBe(false)
    expect(data.size).toBe(0)
    input.remove()
  })

  it('imports the first clipboard image when no Koharu MIME is present', async () => {
    installProject()
    class ClipboardFileReader {
      result: string | null = null
      error: DOMException | null = null
      onload: (() => void) | null = null
      onerror: (() => void) | null = null

      readAsDataURL() {
        this.result = 'data:image/png;base64,aW1hZ2U='
        this.onload?.()
      }
    }
    vi.stubGlobal('FileReader', ClipboardFileReader)
    const imported = vi
      .spyOn(commands, 'importCleanupImage')
      .mockResolvedValue({ revision: 2, layer: 'cleanup' })
    renderWorkspace()
    const file = new File(['image'], 'cleanup.png', { type: 'image/png' })
    const event = new Event('paste', { bubbles: true, cancelable: true }) as ClipboardEvent
    Object.defineProperty(event, 'clipboardData', {
      value: {
        types: ['Files'],
        getData: () => '',
        items: [
          {
            kind: 'file',
            type: 'image/png',
            getAsFile: () => file,
          },
        ],
      },
    })

    window.dispatchEvent(event)

    expect(event.defaultPrevented).toBe(true)
    await waitFor(() => expect(imported).toHaveBeenCalledWith('data:image/png;base64,aW1hZ2U='))
    expect(useKoharuStore.getState().selectedLayers).toEqual(['cleanup'])
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

  it('releases a committed brush preview even when the frame does not change', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'draw' })
    let active = false
    canvas.beginStroke.mockImplementation(() => {
      if (active) throw new Error('another canvas edit is already active')
      active = true
    })
    canvas.cancelStroke.mockImplementation(() => {
      active = false
    })
    const commit = vi
      .spyOn(commands, 'commitPaint')
      .mockResolvedValue({ revision: 1, layer: 'paint' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 81, clientX: 30, clientY: 40 })
    fireEvent.pointerUp(surface, { pointerId: 81, clientX: 30, clientY: 40 })
    await waitFor(() => expect(canvas.cancelStroke).toHaveBeenCalledOnce())
    fireEvent.pointerDown(surface, { button: 0, pointerId: 82, clientX: 50, clientY: 60 })
    fireEvent.pointerUp(surface, { pointerId: 82, clientX: 50, clientY: 60 })
    await waitFor(() => expect(commit).toHaveBeenCalledTimes(2))
  })

  it('waits for the committed brush frame before accepting another stroke', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'draw' })
    const commit = vi
      .spyOn(commands, 'commitPaint')
      .mockResolvedValue({ revision: 2, layer: 'paint' })
    const surface = renderWorkspace()
    fireEvent.pointerDown(surface, { button: 0, pointerId: 83, clientX: 30, clientY: 40 })
    await act(async () => {
      fireEvent.pointerUp(surface, { pointerId: 83, clientX: 30, clientY: 40 })
    })
    expect(commit).toHaveBeenCalledOnce()
    fireEvent.pointerDown(surface, { button: 0, pointerId: 84, clientX: 50, clientY: 60 })
    expect(canvas.beginStroke).toHaveBeenCalledOnce()
    expect(canvas.cancelStroke).not.toHaveBeenCalled()

    act(() => useKoharuStore.setState({ canvasRevision: 2, canvasGeneration: 2 }))
    fireEvent.pointerDown(surface, { button: 0, pointerId: 85, clientX: 50, clientY: 60 })
    expect(canvas.beginStroke).toHaveBeenCalledOnce()

    canvasState.generation = 2
    act(() => useKoharuStore.setState({ brush: { diameter: 49, color: '#FFFFFF' } }))
    await waitFor(() => expect(canvas.cancelStroke).toHaveBeenCalledOnce())
    fireEvent.pointerDown(surface, { button: 0, pointerId: 86, clientX: 50, clientY: 60 })
    fireEvent.pointerUp(surface, { pointerId: 86, clientX: 50, clientY: 60 })
    await waitFor(() => expect(commit).toHaveBeenCalledTimes(2))
    expect(commit.mock.calls[1][0]).toBe(2)
  })

  it('cancels an unfinished brush gesture when pointer capture is lost', () => {
    installProject()
    useKoharuStore.setState({ tool: 'draw' })
    const surface = renderWorkspace()
    fireEvent.pointerDown(surface, { button: 0, pointerId: 87, clientX: 30, clientY: 40 })
    fireEvent.lostPointerCapture(surface, { pointerId: 87 })
    expect(canvas.cancelStroke).toHaveBeenCalledOnce()
    fireEvent.pointerUp(surface, { pointerId: 87, clientX: 30, clientY: 40 })
    expect(canvas.finishStroke).not.toHaveBeenCalled()
    fireEvent.pointerDown(surface, { button: 0, pointerId: 88, clientX: 50, clientY: 60 })
    expect(canvas.beginStroke).toHaveBeenCalledTimes(2)
  })

  it('keeps a brush gesture owned by its original pointer', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'draw' })
    const commit = vi
      .spyOn(commands, 'commitPaint')
      .mockResolvedValue({ revision: 1, layer: 'paint' })
    const surface = renderWorkspace()
    fireEvent.pointerDown(surface, { button: 0, pointerId: 89, clientX: 30, clientY: 40 })
    fireEvent.pointerDown(surface, { button: 0, pointerId: 90, clientX: 50, clientY: 60 })
    expect(canvas.beginStroke).toHaveBeenCalledOnce()
    fireEvent.pointerUp(surface, { pointerId: 90, clientX: 50, clientY: 60 })
    fireEvent.pointerCancel(surface, { pointerId: 90 })
    expect(canvas.finishStroke).not.toHaveBeenCalled()
    expect(canvas.cancelStroke).not.toHaveBeenCalled()
    fireEvent.pointerUp(surface, { pointerId: 89, clientX: 35, clientY: 45 })
    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(commit.mock.calls[0][2]).toEqual([
      { x: 20, y: 20 },
      { x: 25, y: 25 },
    ])
  })

  it('releases the brush after a preview finalization error', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'draw' })
    canvas.finishStroke.mockImplementationOnce(() => {
      throw new Error('preview finalization failed')
    })
    const commit = vi
      .spyOn(commands, 'commitPaint')
      .mockResolvedValue({ revision: 1, layer: 'paint' })
    const surface = renderWorkspace()
    fireEvent.pointerDown(surface, { button: 0, pointerId: 91, clientX: 30, clientY: 40 })
    fireEvent.pointerUp(surface, { pointerId: 91, clientX: 30, clientY: 40 })
    expect(canvas.cancelStroke).toHaveBeenCalledOnce()
    expect(commit).not.toHaveBeenCalled()
    fireEvent.pointerDown(surface, { button: 0, pointerId: 92, clientX: 50, clientY: 60 })
    fireEvent.pointerUp(surface, { pointerId: 92, clientX: 50, clientY: 60 })
    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
  })

  it('releases a no-op transform before switching to the brush', async () => {
    installProject()
    const transform = vi.spyOn(commands, 'commitTransform').mockResolvedValue(null)
    const commit = vi
      .spyOn(commands, 'commitPaint')
      .mockResolvedValue({ revision: 1, layer: 'paint' })
    const surface = renderWorkspace()
    fireEvent.pointerDown(surface, { button: 0, pointerId: 93, clientX: 30, clientY: 45 })
    fireEvent.pointerUp(surface, { pointerId: 93, clientX: 30, clientY: 45 })
    await waitFor(() => expect(transform).toHaveBeenCalledOnce())
    await waitFor(() => expect(canvas.cancelTransform).toHaveBeenCalledOnce())
    act(() => useKoharuStore.setState({ tool: 'draw' }))
    fireEvent.pointerDown(surface, { button: 0, pointerId: 94, clientX: 50, clientY: 60 })
    fireEvent.pointerUp(surface, { pointerId: 94, clientX: 50, clientY: 60 })
    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
  })

  it.each(['draw', 'eraser', 'remove'] as const)(
    'adjusts the %s brush by one pixel with plus and minus',
    (tool) => {
      installProject()
      useKoharuStore.setState({ tool, brush: { diameter: 48, color: '#FFFFFF' } })
      renderWorkspace()

      fireEvent.keyDown(window, { key: '+', code: 'Equal', shiftKey: true })
      expect(useKoharuStore.getState().brush.diameter).toBe(49)

      fireEvent.keyDown(window, { key: '-', code: 'Minus' })
      expect(useKoharuStore.getState().brush.diameter).toBe(48)
    },
  )

  it('keeps the brush cursor synchronized with Alt+wheel and later pointer movement', () => {
    installProject()
    useKoharuStore.setState({ tool: 'remove', brush: { diameter: 48, color: '#FFFFFF' } })
    const surface = renderWorkspace()

    fireEvent.pointerMove(surface, { pointerId: 70, clientX: 55, clientY: 65 })
    const cursor = screen
      .getByTestId('canvas-overlay')
      .querySelector<HTMLElement>('.absolute.rounded-full')!
    expect(cursor).toHaveStyle({ left: '21px', top: '21px', width: '48px', height: '48px' })

    fireEvent.wheel(surface, { clientX: 110, clientY: 220, deltaY: -1, altKey: true })
    expect(cursor).toHaveStyle({ left: '74px', top: '174px', width: '52px', height: '52px' })

    fireEvent.pointerMove(surface, { pointerId: 70, clientX: 210, clientY: 320 })
    expect(cursor).toHaveStyle({ left: '174px', top: '274px', width: '52px', height: '52px' })
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
          angle_degrees: 0,
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
          angle_degrees: null,
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
    useKoharuStore.setState({ textTransformTarget: 'region' })
    const ocrLayer = (id: string, region: string): Layer => ({
      type: 'text',
      id,
      parent: 'page',
      geometry: null,
      angle_degrees: null,
      visibility: { visible: true, opacity: 1 },
      content: {
        id: `${id}-content`,
        source: { text: id, language: 'ja' },
        translation: null,
        translation_review: null,
        ocr_unverified: false,
        cleaning_unverified: false,
        markup: null,
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

  it('draws a rectangle as the selected text area and leaves drawing mode', async () => {
    installProject()
    const setGeometry = vi.spyOn(commands, 'setGeometry').mockResolvedValue(null)
    const surface = renderWorkspace()
    act(() => {
      useKoharuStore.getState().selectLayers(['dialogue'])
      useKoharuStore.setState({ textTransformTarget: 'area' })
    })

    fireEvent.pointerDown(surface, { button: 0, pointerId: 30, clientX: 20, clientY: 20 })
    fireEvent.pointerMove(surface, { pointerId: 30, clientX: 60, clientY: 70 })
    fireEvent.pointerUp(surface, { pointerId: 30, clientX: 60, clientY: 70 })

    await waitFor(() => expect(setGeometry).toHaveBeenCalledOnce())
    const [[update]] = setGeometry.mock.calls[0]
    expect(update.layer).toBe('dialogue')
    expect(update.points).toHaveLength(4)
    await waitFor(() => expect(useKoharuStore.getState().textTransformTarget).toBe('text'))
    expect(useKoharuStore.getState().selectedLayers).toEqual(['dialogue'])
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
          angle_degrees: null,
          visibility: { visible: true, opacity: 1 },
          content: {
            id: 'ocr-content',
            source: { text: 'Source', language: 'ja' },
            translation: null,
            translation_review: null,
            ocr_unverified: false,
            cleaning_unverified: false,
            markup: null,
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

  it.each(['text', 'region'] as const)(
    'moves and resizes the selected %s independently',
    async (target) => {
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
            angle_degrees: null,
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
      useKoharuStore.setState({
        selectedLayers: ['dialogue'],
        tool: 'select',
        textTransformTarget: target,
      })
      const setGeometry = vi.spyOn(commands, 'setGeometry').mockResolvedValue(null)
      const commitTransform = vi.spyOn(commands, 'commitTransform').mockResolvedValue(2)
      const surface = renderWorkspace()

      fireEvent.pointerDown(surface, { button: 0, pointerId: 34, clientX: 35, clientY: 45 })
      fireEvent.pointerMove(surface, { pointerId: 34, clientX: 55, clientY: 65 })
      fireEvent.pointerUp(surface, { pointerId: 34, clientX: 55, clientY: 65 })
      const commit = target === 'region' ? setGeometry : commitTransform
      await waitFor(() => expect(commit).toHaveBeenCalledOnce())
      if (target === 'region') {
        expect(setGeometry.mock.calls[0][0][0].layer).toBe('text-region')
        expect(commitTransform).not.toHaveBeenCalled()
      } else {
        expect(commitTransform.mock.calls[0][1][0].element).toBe('dialogue')
        expect(setGeometry).not.toHaveBeenCalled()
        canvasState.generation = 2
        act(() => useKoharuStore.setState({ canvasRevision: 2, canvasGeneration: 2 }))
        await waitFor(() => expect(canvas.cancelTransform).toHaveBeenCalledOnce())
      }

      const handle = document.querySelector<HTMLElement>('[data-resize-handle="e"]')!
      fireEvent.pointerDown(handle, { button: 0, pointerId: 35, clientX: 70, clientY: 40 })
      fireEvent.pointerMove(handle, { pointerId: 35, clientX: 90, clientY: 40 })
      fireEvent.pointerUp(handle, { pointerId: 35, clientX: 90, clientY: 40 })

      await waitFor(() => expect(commit).toHaveBeenCalledTimes(2))
      if (target === 'region') {
        expect(setGeometry.mock.calls[1][0][0].layer).toBe('text-region')
        expect(setGeometry.mock.calls[1][0][0].points).toHaveLength(4)
      } else {
        expect(commitTransform.mock.calls[1][1][0].element).toBe('dialogue')
        expect(setGeometry).not.toHaveBeenCalled()
      }
    },
  )

  it.each([
    ['text', 'A'],
    ['region', 'B'],
  ] as const)(
    'selects the visible %s when moved text overlaps another OCR region',
    (target, expected) => {
      installProject()
      const geometry = {
        points: [
          { x: 100, y: 100 },
          { x: 140, y: 100 },
          { x: 140, y: 140 },
          { x: 100, y: 140 },
        ],
      }
      queryClient.setQueryData(pageKey, (page: { layers: Layer[]; regions: unknown[] }) => ({
        ...page,
        layers: ['A', 'B'].map((id) => ({
          type: 'text',
          id,
          parent: 'page',
          geometry: null,
          angle_degrees: null,
          visibility: { visible: true, opacity: 1 },
          content: {
            id: `content-${id}`,
            source: null,
            translation: { text: id, language: null },
            role: 'dialogue',
            source_region: id === 'B' ? 'source-B' : null,
          },
          typography: null,
          layout: 'paragraph',
          automatic_region: null,
        })),
        regions: [
          { id: 'source-B', parent: 'page', geometry, kind: 'dev.koharu.region.text', label: null },
        ],
      }))
      useKoharuStore.setState({
        textTransformTarget: target,
        layerFrames: {
          A: { x: 100, y: 100, width: 40, height: 40, angle_degrees: 0 },
          B: { x: 300, y: 100, width: 40, height: 40, angle_degrees: 0 },
        },
      })
      const surface = renderWorkspace()
      fireEvent.pointerDown(surface, { button: 0, pointerId: 36, clientX: 120, clientY: 130 })
      expect(useKoharuStore.getState().selectedLayers).toEqual([expected])
      fireEvent.pointerCancel(surface, { pointerId: 36 })
    },
  )

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
          angle_degrees: null,
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

  it('falls back to the visible cleanup layer when the eraser selection is empty', async () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, cleanupLayer],
    }))
    useKoharuStore.setState({ tool: 'eraser', selectedLayers: [] })
    const commit = vi
      .spyOn(commands, 'commitErase')
      .mockResolvedValue({ revision: 2, layer: 'cleanup' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 111, clientX: 30, clientY: 40 })
    fireEvent.pointerUp(surface, { pointerId: 111, clientX: 30, clientY: 40 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(canvas.beginStroke).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'erase', layer: 'cleanup' }),
    )
    expect(commit).toHaveBeenCalledWith(
      1,
      'cleanup',
      expect.arrayContaining([{ x: 20, y: 20 }]),
      48,
    )
  })

  it('clears an accepted Remove preview before starting the next stroke', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'remove' })
    let activeStroke = false
    canvas.beginStroke.mockImplementation(() => {
      if (activeStroke) throw new Error('another canvas edit is already active')
      activeStroke = true
    })
    canvas.cancelStroke.mockImplementation(() => {
      activeStroke = false
    })
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
    expect(canvas.cancelStroke).toHaveBeenCalledOnce()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 13, clientX: 50, clientY: 60 })
    expect(canvas.beginStroke).toHaveBeenCalledTimes(2)
  })

  it('previews and commits a clamped restore rectangle with Alt+R', async () => {
    installProject()
    const commit = vi.spyOn(commands, 'commitRestore').mockResolvedValue('job')
    const surface = renderWorkspace()

    fireEvent.keyDown(window, { key: 'j', code: 'KeyJ', altKey: true })
    expect(useKoharuStore.getState().tool).toBe('remove')
    fireEvent.keyDown(window, { key: 'r', code: 'KeyR', altKey: true })
    expect(useKoharuStore.getState().tool).toBe('restore')

    fireEvent.pointerDown(surface, { button: 0, pointerId: 121, clientX: 0, clientY: 0 })
    fireEvent.pointerMove(surface, { pointerId: 121, clientX: 1200, clientY: 1200 })
    expect(screen.getByTestId('rectangle-preview')).toBeInTheDocument()
    fireEvent.pointerUp(surface, { pointerId: 121, clientX: 1200, clientY: 1200 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(commit).toHaveBeenCalledWith(1, {
      x: 0,
      y: 0,
      width: 1000,
      height: 1000,
      angle_degrees: 0,
    })
  })

  it('ignores tiny restore rectangles and accepts restore input while a job is active', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'restore' })
    const commit = vi.spyOn(commands, 'commitRestore')
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 122, clientX: 40, clientY: 50 })
    fireEvent.pointerMove(surface, { pointerId: 122, clientX: 42, clientY: 150 })
    fireEvent.pointerUp(surface, { pointerId: 122, clientX: 42, clientY: 150 })
    expect(commit).not.toHaveBeenCalled()

    act(() =>
      useKoharuStore.setState({
        jobs: {
          active: {
            id: 'active',
            state: 'running',
            completed: 0,
            total: 1,
            target: null,
            stage: 'inpainting',
            model: 'microsoft/mai-image-2.5/edit',
            error: null,
          },
        },
      }),
    )
    fireEvent.pointerDown(surface, { button: 0, pointerId: 123, clientX: 40, clientY: 50 })
    fireEvent.pointerMove(surface, { pointerId: 123, clientX: 140, clientY: 150 })
    fireEvent.pointerUp(surface, { pointerId: 123, clientX: 140, clientY: 150 })
    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(commit).toHaveBeenCalledWith(1, {
      x: 30,
      y: 30,
      width: 100,
      height: 100,
      angle_degrees: 0,
    })
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
    await waitFor(() => expect(useKoharuStore.getState().pendingTextFocus).toBe('point-text'))
    expect(useKoharuStore.getState().tool).toBe('select')
    act(() => useKoharuStore.getState().setTool('text'))

    fireEvent.pointerDown(surface, { button: 0, pointerId: 14, clientX: 40, clientY: 50 })
    fireEvent.pointerMove(surface, { pointerId: 14, clientX: 140, clientY: 110 })
    fireEvent.pointerUp(surface, { pointerId: 14, clientX: 140, clientY: 110 })
    await waitFor(() => expect(box).toHaveBeenCalledOnce())
    await waitFor(() => expect(useKoharuStore.getState().pendingTextFocus).toBe('box-text'))
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

    fireEvent.keyDown(window, { key: 'o', code: 'KeyO', altKey: true })
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

  it('reads a flagged unboxed text as an OCR region or dismisses the flag', async () => {
    installProject()
    const flag = { x: 300, y: 200, width: 80, height: 40, kind: 'sound-effect' as const }
    queryClient.setQueryData(pageKey, (page: object) => ({ ...page, unboxed_text: [flag] }))
    const addRegion = vi
      .spyOn(commands, 'addOcrRegion')
      .mockResolvedValue({ revision: 2, layer: 'ocr-layer', region: 'ocr-region' })
    const process = vi.spyOn(commands, 'process').mockResolvedValue('ocr-job')
    const dismiss = vi.spyOn(commands, 'dismissUnboxedText').mockResolvedValue(null)
    renderWorkspace()

    expect(screen.getAllByTestId('unboxed-text')).toHaveLength(1)
    fireEvent.click(screen.getByTitle('Create a text block here'))
    await waitFor(() => expect(process).toHaveBeenCalledOnce())
    expect(addRegion).toHaveBeenCalledWith({
      x: 300,
      y: 200,
      width: 80,
      height: 40,
      angle_degrees: 0,
    })
    expect(process).toHaveBeenCalledWith(
      { scope: 'entities', value: ['ocr-region'] },
      { operation: 'stages', stages: ['ocr'] },
    )

    fireEvent.click(screen.getByTitle('Not text'))
    await waitFor(() => expect(dismiss).toHaveBeenCalledWith('page', flag))
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

  it('ignores a tiny OCR rectangle and accepts manual OCR while another job is running', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'ocr' })
    const addRegion = vi
      .spyOn(commands, 'addOcrRegion')
      .mockResolvedValue({ revision: 2, layer: 'queued-ocr-layer', region: 'queued-ocr-region' })
    const process = vi.spyOn(commands, 'process').mockResolvedValue('queued-ocr-job')
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
    await waitFor(() => expect(process).toHaveBeenCalledOnce())
    expect(addRegion).toHaveBeenCalledWith({
      x: 30,
      y: 30,
      width: 100,
      height: 100,
      angle_degrees: 0,
    })
    expect(process).toHaveBeenCalledWith(
      { scope: 'entities', value: ['queued-ocr-region'] },
      { operation: 'stages', stages: ['ocr'] },
    )
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
