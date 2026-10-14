import { clearMocks, mockIPC } from '@tauri-apps/api/mocks'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ImageEditPreviewDialog } from '@/components/app/ImageEditPreviewDialog'
import { useKoharuStore } from '@/lib/store'
import type { ImageEditPreview } from '@koharu/bridge/protocol'

const invoke = vi.fn()

const preview: ImageEditPreview = {
  id: 'preview-1',
  job: 'job-1',
  page: 'page-1',
  page_label: 'Page 1',
  provider: 'fal',
  model: 'microsoft/mai-image-2.5/edit',
  prompt: 'Restore the original artwork.',
  image_url: 'data:image/png;base64,cHJldmlldw==',
  width: 1536,
  height: 2048,
  restoration: false,
}

beforeEach(() => {
  invoke.mockResolvedValue(null)
  mockIPC(invoke)
  useKoharuStore.setState({ imageEditPreviews: [preview] })
})

afterEach(() => clearMocks())

describe('image edit preview', () => {
  it.each(['  \n ', 'invalid\0prompt'])('does not send an invalid prompt %j', async (value) => {
    const user = userEvent.setup()
    render(<ImageEditPreviewDialog />)
    fireEvent.change(screen.getByRole('textbox', { name: 'Prompt for this request' }), {
      target: { value },
    })
    const send = screen.getByRole('button', { name: 'Send to API' })
    expect(send).toBeDisabled()
    await user.click(send)
    expect(invoke).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Cancel job' })).toBeEnabled()
  })

  it('shows the outgoing image and sends the exact edited prompt for this request only', async () => {
    const user = userEvent.setup()
    render(<ImageEditPreviewDialog />)

    const image = screen.getByRole('img', { name: 'Outgoing image' })
    expect(image).toHaveAttribute('src', preview.image_url)
    expect(image).toHaveAttribute('width', '1536')
    expect(image).toHaveAttribute('height', '2048')
    expect(screen.getByText('Page 1')).toBeInTheDocument()
    expect(screen.getByText('1536 × 2048 px')).toBeInTheDocument()
    expect(screen.getByText('Fal.ai')).toBeInTheDocument()
    expect(screen.getByText('microsoft/mai-image-2.5/edit')).toBeInTheDocument()

    const prompt = screen.getByRole('textbox', { name: 'Prompt for this request' })
    expect(prompt).toHaveValue(preview.prompt)
    fireEvent.change(prompt, { target: { value: '  Keep the drawing.\nRemove text only.  ' } })
    await user.click(screen.getByRole('button', { name: 'Send to API' }))

    expect(invoke).toHaveBeenCalledExactlyOnceWith('resolve_image_edit_preview', {
      id: 'preview-1',
      prompt: '  Keep the drawing.\nRemove text only.  ',
    })
    expect(screen.getByRole('button', { name: 'Send to API' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Send to API' }))
    expect(invoke).toHaveBeenCalledTimes(1)
    expect(useKoharuStore.getState().imageEditPreviews).toEqual([preview])
  })

  it('shows Replicate instead of falling through to another provider name', () => {
    useKoharuStore.setState({
      imageEditPreviews: [
        {
          ...preview,
          provider: 'replicate',
          model: 'black-forest-labs/flux-2-dev',
        },
      ],
    })

    render(<ImageEditPreviewDialog />)

    expect(screen.getByText('Replicate')).toBeInTheDocument()
    expect(screen.queryByText('OpenRouter')).not.toBeInTheDocument()
  })

  it.each(['button', 'escape', 'close'] as const)(
    'cancels the job through %s without sending a prompt',
    async (action) => {
      const user = userEvent.setup()
      render(<ImageEditPreviewDialog />)

      if (action === 'escape') await user.keyboard('{Escape}')
      else
        await user.click(
          screen.getByRole('button', { name: action === 'close' ? 'Close' : 'Cancel job' }),
        )

      await waitFor(() =>
        expect(invoke).toHaveBeenCalledExactlyOnceWith('resolve_image_edit_preview', {
          id: 'preview-1',
          prompt: null,
        }),
      )
    },
  )

  it('restores the original prompt and replaces the draft when the native queue advances or clears', async () => {
    const user = userEvent.setup()
    render(<ImageEditPreviewDialog />)
    const prompt = screen.getByRole('textbox', { name: 'Prompt for this request' })
    fireEvent.change(prompt, { target: { value: 'First draft' } })
    await user.click(screen.getByRole('button', { name: 'Reset prompt' }))
    expect(prompt).toHaveValue(preview.prompt)
    fireEvent.change(prompt, { target: { value: 'Discard this draft' } })

    act(() =>
      useKoharuStore.setState({
        imageEditPreviews: [
          {
            ...preview,
            id: 'preview-2',
            page_label: 'Page 2',
            prompt: 'Restore this area.',
            restoration: true,
          },
        ],
      }),
    )
    expect(screen.getByRole('textbox', { name: 'Prompt for this request' })).toHaveValue(
      'Restore this area.',
    )
    expect(screen.getByText('Page 2')).toBeInTheDocument()
    expect(screen.getByText('Restore selection')).toBeInTheDocument()
    expect(invoke).not.toHaveBeenCalled()

    act(() => useKoharuStore.setState({ imageEditPreviews: [] }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('keeps the pending preview and draft after an IPC failure so the request can be retried', async () => {
    const user = userEvent.setup()
    invoke.mockRejectedValueOnce(new Error('Preview connection failed'))
    render(<ImageEditPreviewDialog />)
    fireEvent.change(screen.getByRole('textbox', { name: 'Prompt for this request' }), {
      target: { value: 'Keep this edit' },
    })
    await user.click(screen.getByRole('button', { name: 'Send to API' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Preview connection failed')
    expect(screen.getByRole('textbox', { name: 'Prompt for this request' })).toHaveValue(
      'Keep this edit',
    )
    expect(useKoharuStore.getState().imageEditPreviews).toEqual([preview])
    await user.click(screen.getByRole('button', { name: 'Send to API' }))
    expect(invoke).toHaveBeenLastCalledWith('resolve_image_edit_preview', {
      id: 'preview-1',
      prompt: 'Keep this edit',
    })
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})
