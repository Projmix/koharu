import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { Updater } from '@/components/app/Updater'

const nativeCheck = vi.hoisted(() => vi.fn())
const nativeOpenUrl = vi.hoisted(() => vi.fn(async () => undefined))
const upstreamCheck = vi.hoisted(() => vi.fn())
const nativeRelaunch = vi.hoisted(() => vi.fn(async () => undefined))

vi.mock('@koharu/bridge/protocol', () => ({
  commands: { checkUpstreamUpdate: upstreamCheck },
}))

vi.mock('@tauri-apps/plugin-opener', () => ({ openUrl: nativeOpenUrl }))
vi.mock('@tauri-apps/plugin-process', () => ({ relaunch: nativeRelaunch }))
vi.mock('@tauri-apps/plugin-updater', () => ({ check: nativeCheck }))

describe('updater', () => {
  beforeEach(() => {
    localStorage.clear()
    nativeCheck.mockResolvedValue(null)
    upstreamCheck.mockResolvedValue(null)
  })

  it('renders Markdown release notes inside a viewport-bounded scroll area', async () => {
    nativeCheck.mockResolvedValue({
      version: '0.63.0',
      body: '# Highlights\n\n- Fixed the canvas\n\n[Full changelog](https://example.com/changelog)',
      close: vi.fn(async () => undefined),
      downloadAndInstall: vi.fn(async () => undefined),
    })

    render(<Updater />)

    const heading = await screen.findByRole('heading', { level: 1, name: 'Highlights' })
    const title = screen.getByRole('heading', {
      level: 2,
      name: 'Update available',
    })
    const copy = title.parentElement
    const header = copy?.parentElement
    expect(copy).toContainElement(
      header?.querySelector('[data-slot="alert-dialog-description"]') ?? null,
    )
    expect(header).toHaveClass('flex', 'flex-row', 'items-start', 'gap-4', 'text-left')
    expect(header).not.toHaveClass('grid', 'text-center')
    expect(screen.getByRole('listitem')).toHaveTextContent('Fixed the canvas')
    expect(heading.closest('[data-slot="scroll-area"]')).toHaveClass(
      'h-64',
      'max-h-[40dvh]',
      'min-h-0',
    )
    expect(screen.getByRole('alertdialog')).toHaveClass(
      'flex',
      'max-h-[calc(100dvh-2rem)]',
      'flex-col',
      'overflow-hidden',
    )

    fireEvent.click(screen.getByRole('link', { name: 'Full changelog' }))
    expect(nativeOpenUrl).toHaveBeenCalledWith('https://example.com/changelog')
  })

  const upstream = {
    version: '0.83.0',
    integratedVersion: '0.82.1',
    url: 'https://github.com/koharu-rs/koharu/releases/tag/v0.83.0',
  }

  it('offers upstream release notes without an installation action', async () => {
    upstreamCheck.mockResolvedValue(upstream)
    render(<Updater />)

    expect(
      await screen.findByRole('heading', { name: 'Original Koharu has an update' }),
    ).toBeVisible()
    expect(screen.getByRole('alertdialog')).toHaveTextContent('0.83.0')
    expect(screen.getByRole('alertdialog')).toHaveTextContent('0.82.1')
    expect(screen.queryByRole('button', { name: 'Update' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'View original release' }))
    expect(nativeOpenUrl).toHaveBeenCalledWith(upstream.url)
  })

  it('shows the fork update first and the independent upstream notice after dismissal', async () => {
    const close = vi.fn(async () => undefined)
    nativeCheck.mockResolvedValue({ version: '2.0.0', close })
    upstreamCheck.mockResolvedValue(upstream)
    render(<Updater />)

    expect(await screen.findByRole('heading', { name: 'Update available' })).toBeVisible()
    expect(
      screen.queryByRole('heading', { name: 'Original Koharu has an update' }),
    ).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Later' }))
    expect(
      await screen.findByRole('heading', { name: 'Original Koharu has an update' }),
    ).toBeVisible()
    expect(close).toHaveBeenCalledOnce()
  })

  it('remembers upstream dismissal per version while continuing to offer fork updates', async () => {
    upstreamCheck.mockResolvedValue(upstream)
    const view = render(<Updater />)
    fireEvent.click(await screen.findByRole('button', { name: 'Dismiss this version' }))
    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
    view.unmount()

    const nextView = render(<Updater />)
    await waitFor(() => expect(upstreamCheck).toHaveBeenCalledTimes(2))
    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
    nextView.unmount()

    nativeCheck.mockResolvedValue({ version: '2.0.0', close: vi.fn(async () => undefined) })
    const forkView = render(<Updater />)
    expect(await screen.findByRole('heading', { name: 'Update available' })).toBeVisible()
    forkView.unmount()

    nativeCheck.mockResolvedValue(null)
    upstreamCheck.mockResolvedValue({ ...upstream, version: '0.84.0' })
    render(<Updater />)
    expect(
      await screen.findByRole('heading', { name: 'Original Koharu has an update' }),
    ).toBeVisible()
    expect(screen.getByRole('alertdialog')).toHaveTextContent('0.84.0')
  })

  it('still offers a fork update when the upstream request fails', async () => {
    upstreamCheck.mockRejectedValue(new Error('Rate limited'))
    nativeCheck.mockResolvedValue({ version: '2.0.0', close: vi.fn(async () => undefined) })
    render(<Updater />)
    expect(await screen.findByRole('heading', { name: 'Update available' })).toBeVisible()
    expect(nativeCheck).toHaveBeenCalledWith({ timeout: 8000 })
  })

  it('installs only the fork update when both channels have a newer release', async () => {
    const downloadAndInstall = vi.fn(async () => undefined)
    nativeCheck.mockResolvedValue({ version: '2.0.0', close: vi.fn(), downloadAndInstall })
    upstreamCheck.mockResolvedValue(upstream)
    render(<Updater />)
    fireEvent.click(await screen.findByRole('button', { name: 'Update' }))
    await waitFor(() => expect(nativeRelaunch).toHaveBeenCalledOnce())
    expect(downloadAndInstall).toHaveBeenCalledOnce()
    expect(nativeOpenUrl).not.toHaveBeenCalled()
  })

  it('keeps the notice usable if persistent browser storage is unavailable', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('Unavailable')
    })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('Unavailable')
    })
    upstreamCheck.mockResolvedValue(upstream)
    render(<Updater />)
    fireEvent.click(await screen.findByRole('button', { name: 'Dismiss this version' }))
    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
  })
})
