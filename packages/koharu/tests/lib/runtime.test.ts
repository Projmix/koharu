import { act, render, screen, waitFor } from '@testing-library/react'
import { createElement, StrictMode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import Providers from '@/app/providers'
import { call } from '@/lib/backend'
import { usePages, useProject } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import {
  commands,
  type Model,
  type Preferences,
  type ProjectInfo,
  type StartupState,
} from '@koharu/bridge/protocol'

vi.mock('@tauri-apps/api/core', () => ({
  Channel: class<T> {
    onmessage: (payload: T) => void

    constructor(handler: (payload: T) => void) {
      this.onmessage = handler
    }
  },
  invoke: vi.fn(),
}))

const preferences: Preferences = {
  pipeline: {
    detection: { model: 'koharu-layout-rfdetr-seg-2xl' },
    ocr: {
      method: 'local',
      local_model: { model: 'paddleocr-vl-1.6' },
      api: {
        model: {
          provider: 'openrouter',
          model: 'qwen/qwen3.8-27b',
          quantization: null,
          vision: true,
          reasoning: true,
        },
        generation: {
          temperature: 0,
          max_tokens: 1024,
          vision: true,
          reasoning: false,
        },
        instructions: null,
      },
    },
    translation: {
      source_language: 'ja-JP',
      target_language: 'en-US',
      page: translationProfile(),
      chapter: translationProfile(),
    },
    inpainting: {
      method: 'local',
      local_model: { model: 'lama' },
      manual_model: { model: 'lama' },
      api: {
        provider: 'fal',
        model: 'microsoft/mai-image-2.5/edit',
        prompt: 'Remove all text and reconstruct the original manga artwork.',
        apply_mode: 'full-page',
      },
    },
    processor: {},
  },
  providers: {
    fal: { configured: false, value: null, clear: false },
    entries: [],
  },
  typesetting: {
    font_families: ['CCWildWords', 'Adobe 黑体 Std'],
  },
  languages: [],
}

const project: ProjectInfo = {
  name: 'Book',
  revision: 3,
  active_page: null,
  can_undo: true,
  can_redo: false,
}

beforeEach(() => {
  vi.spyOn(commands, 'getTranslationModels').mockResolvedValue([])
  vi.spyOn(commands, 'getInpaintingModels').mockResolvedValue([])
})

const startupState = (): StartupState => ({
  preferences,
  jobs: [],
  canvas: {
    page: null,
    revision: null,
    generation: 0,
    size: [0, 0],
    element_frames: [],
  },
})

async function start() {
  const pending = deferred<StartupState>()
  const binding = vi.spyOn(commands, 'subscribe').mockReturnValue(pending.promise)
  const view = render(createElement(Providers, null, createElement('div')))
  pending.resolve(startupState())
  await waitFor(() => expect(useKoharuStore.getState().preferences).toBe(preferences))
  return { binding, dispose: view.unmount }
}

function ProjectProbe() {
  const project = useProject().data
  return createElement(
    'span',
    null,
    project === undefined ? 'Loading' : (project?.name ?? 'Closed'),
  )
}

function PagesProbe() {
  const pages = usePages().data
  return createElement(
    'span',
    null,
    pages === undefined ? 'Pages loading' : `Pages ${pages.length}`,
  )
}

describe('Tauri runtime', () => {
  it('persists mandatory reasoning for a reloaded model profile', async () => {
    const model: Model = {
      provider: 'openrouter',
      model: 'google/gemini-3.5-flash-lite',
      name: 'Gemini 3.5 Flash Lite',
      quantizations: [],
      vision: true,
      reasoning: true,
      reasoning_required: true,
    }
    const configured: Preferences = {
      ...preferences,
      pipeline: {
        ...preferences.pipeline,
        translation: {
          ...preferences.pipeline.translation,
          page: {
            ...preferences.pipeline.translation.page,
            model: {
              provider: model.provider,
              model: model.model,
              quantization: null,
              vision: true,
              reasoning: true,
            },
            generation: {
              ...preferences.pipeline.translation.page.generation,
              reasoning: false,
            },
          },
        },
      },
    }
    vi.spyOn(commands, 'getTranslationModels').mockResolvedValue([model])
    vi.spyOn(commands, 'subscribe').mockResolvedValue({
      ...startupState(),
      preferences: configured,
    })
    const save = vi
      .spyOn(commands, 'savePreferences')
      .mockImplementation(async (pipeline, providers, typesetting) => ({
        ...configured,
        pipeline,
        providers,
        typesetting,
      }))

    const view = render(createElement(Providers, null, createElement('div')))
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        expect.objectContaining({
          translation: expect.objectContaining({
            page: expect.objectContaining({
              model: expect.objectContaining({ reasoning_required: true }),
              generation: expect.objectContaining({ reasoning: true }),
            }),
          }),
        }),
        configured.providers,
        configured.typesetting,
      ),
    )
    await waitFor(() => {
      const page = useKoharuStore.getState().preferences?.pipeline.translation.page
      expect(page?.model.reasoning_required).toBe(true)
      expect(page?.generation.reasoning).toBe(true)
    })
    view.unmount()
  })

  it('keeps the project unresolved until its backend query returns', async () => {
    const projectPending = deferred<ProjectInfo | null>()
    vi.spyOn(commands, 'getProject').mockReturnValue(projectPending.promise)
    vi.spyOn(commands, 'subscribe').mockResolvedValue(startupState())
    const view = render(createElement(Providers, null, createElement(ProjectProbe)))

    expect(await screen.findByText('Loading')).toBeInTheDocument()
    projectPending.resolve(null)
    expect(await screen.findByText('Closed')).toBeInTheDocument()
    view.unmount()
  })

  it('keeps one live job channel through Strict Mode effect replay', async () => {
    const binding = vi.spyOn(commands, 'subscribe').mockResolvedValue(startupState())
    const view = render(
      createElement(StrictMode, null, createElement(Providers, null, createElement('div'))),
    )

    await waitFor(() => expect(binding).toHaveBeenCalledTimes(1))
    const [, jobChannel] = binding.mock.calls[0]
    act(() => {
      jobChannel.onmessage({
        id: 'job',
        state: 'running',
        completed: 0,
        total: 4,
        target: { target: 'page', value: 'page' },
        stage: 'detection',
        model: 'model',
        error: null,
      })
    })

    expect(useKoharuStore.getState().jobs.job).toMatchObject({ state: 'running', total: 4 })
    view.unmount()
  })

  it('passes only domain arguments to mutation commands', async () => {
    const rename = vi.spyOn(commands, 'renamePage').mockResolvedValue(null)

    await expect(call(commands.renamePage, 'page', 'Chapter 1')).resolves.toBeNull()
    expect(rename).toHaveBeenCalledWith('page', 'Chapter 1')
  })

  it('passes the managed project name to open', async () => {
    const open = vi.spyOn(commands, 'openProject').mockResolvedValue(null)
    await expect(call(commands.openProject, 'Volume 1')).resolves.toBeNull()
    expect(open).toHaveBeenCalledWith('Volume 1')
  })

  it('applies independent channel updates directly to the store', async () => {
    useKoharuStore.setState({ downloads: {}, resources: null })
    const { binding, dispose } = await start()
    const [, , downloadChannel, resourcesChannel] = binding.mock.calls[0]

    downloadChannel.onmessage({
      id: 7,
      state: 'running',
      name: 'model.bin',
      completed: 25,
      total: 100,
      error: null,
    })
    resourcesChannel.onmessage({
      process_memory: 1024,
      system_memory: 8192,
      process_cpu: 5,
      devices: [
        { name: 'GPU', selected: true, memory_budget: 8192, memory_used: 4096, utilization: 40 },
      ],
    })
    expect(useKoharuStore.getState().downloads[7]).toMatchObject({ completed: 25, total: 100 })
    expect(useKoharuStore.getState().resources).toMatchObject({
      process_cpu: 5,
      devices: [{ memory_used: 4096 }],
    })
    dispose()
  })

  it('refreshes project queries when an autonomous job commits work', async () => {
    vi.spyOn(commands, 'getProject').mockResolvedValueOnce(null).mockResolvedValue(project)
    const binding = vi.spyOn(commands, 'subscribe').mockResolvedValue(startupState())
    const view = render(
      createElement(
        Providers,
        null,
        createElement('div', null, createElement(ProjectProbe), createElement(PagesProbe)),
      ),
    )
    expect(await screen.findByText('Closed')).toBeInTheDocument()

    const [, jobChannel] = binding.mock.calls[0]
    jobChannel.onmessage({
      id: 'job',
      state: 'running',
      completed: 1,
      total: 2,
      target: { target: 'page', value: 'page' },
      stage: 'ocr',
      model: 'model',
      error: null,
    })

    expect(await screen.findByText('Book')).toBeInTheDocument()
    view.unmount()
  })

  it('refreshes page queries when a new processing job starts', async () => {
    vi.spyOn(commands, 'getProject').mockResolvedValue(project)
    const getPages = vi.spyOn(commands, 'getPages').mockResolvedValue([])
    const binding = vi.spyOn(commands, 'subscribe').mockResolvedValue(startupState())
    const view = render(
      createElement(
        Providers,
        null,
        createElement('div', null, createElement(ProjectProbe), createElement(PagesProbe)),
      ),
    )
    expect(await screen.findByText('Book')).toBeInTheDocument()

    const callsBeforeJob = getPages.mock.calls.length
    const [, jobChannel] = binding.mock.calls[0]
    act(() => {
      jobChannel.onmessage({
        id: 'new-job',
        state: 'running',
        completed: 0,
        total: 4,
        target: { target: 'page', value: 'page' },
        stage: 'inpainting',
        model: 'lama',
        error: null,
      })
    })

    await waitFor(() => expect(getPages.mock.calls.length).toBeGreaterThan(callsBeforeJob))
    view.unmount()
  })
})

function translationProfile() {
  return {
    model: {
      provider: 'local' as const,
      model: 'lfm2.5-1.2b-instruct',
      quantization: null,
      vision: false,
      reasoning: false,
    },
    generation: { vision: true, reasoning: true },
    instructions: null,
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}
