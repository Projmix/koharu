import { fireEvent, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { bindHistoryShortcuts, enqueueHistory, registerPendingEdit } from '@/lib/history'
import { commands } from '@koharu/bridge/protocol'

describe('project history shortcuts', () => {
  it('serializes repeated global undo/redo commands', async () => {
    const first = deferred<null>()
    const undo = vi
      .spyOn(commands, 'undo')
      .mockImplementationOnce(() => first.promise)
      .mockResolvedValue(null)
    const redo = vi.spyOn(commands, 'redo').mockResolvedValue(null)
    const unbind = bindHistoryShortcuts(true)

    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    await waitFor(() => expect(undo).toHaveBeenCalledOnce())
    expect(redo).not.toHaveBeenCalled()

    first.resolve(null)
    await waitFor(() => expect(undo).toHaveBeenCalledTimes(2))

    fireEvent.keyDown(window, { key: 'y', ctrlKey: true })
    fireEvent.keyDown(window, { key: 'z', ctrlKey: true, shiftKey: true })
    await waitFor(() => expect(redo).toHaveBeenCalledTimes(2))
    unbind()
  })

  it('leaves native text-field undo untouched', async () => {
    const undo = vi.spyOn(commands, 'undo').mockResolvedValue(null)
    const unbind = bindHistoryShortcuts(true)
    const input = document.body.appendChild(document.createElement('input'))
    const event = new KeyboardEvent('keydown', { key: 'z', ctrlKey: true, bubbles: true })
    input.dispatchEvent(event)
    await Promise.resolve()
    expect(undo).not.toHaveBeenCalled()
    expect(event.defaultPrevented).toBe(false)
    input.remove()
    unbind()
  })

  it('uses physical Z/Y keys so shortcuts work with a non-Latin keyboard layout', async () => {
    const undo = vi.spyOn(commands, 'undo').mockResolvedValue(null)
    const redo = vi.spyOn(commands, 'redo').mockResolvedValue(null)
    const unbind = bindHistoryShortcuts(true)

    fireEvent.keyDown(window, { key: 'я', code: 'KeyZ', ctrlKey: true })
    fireEvent.keyDown(window, { key: 'н', code: 'KeyY', ctrlKey: true })

    await waitFor(() => {
      expect(undo).toHaveBeenCalledOnce()
      expect(redo).toHaveBeenCalledOnce()
    })
    unbind()
  })

  it('does not bind history when no project is open', async () => {
    const undo = vi.spyOn(commands, 'undo').mockResolvedValue(null)
    const unbind = bindHistoryShortcuts(false)
    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    await Promise.resolve()
    expect(undo).not.toHaveBeenCalled()
    unbind()
  })

  it('queues direct menu-style actions through the same history queue', async () => {
    const first = deferred<null>()
    const undo = vi
      .spyOn(commands, 'undo')
      .mockImplementationOnce(() => first.promise)
      .mockResolvedValue(null)
    const firstAction = enqueueHistory('undo')
    const secondAction = enqueueHistory('undo')
    await waitFor(() => expect(undo).toHaveBeenCalledOnce())
    first.resolve(null)
    await Promise.all([firstAction, secondAction])
    expect(undo).toHaveBeenCalledTimes(2)
  })

  it('flushes a pending text edit before changing the project revision', async () => {
    let finishEdit!: () => void
    const edit = new Promise<void>((resolve) => {
      finishEdit = resolve
    })
    const unregister = registerPendingEdit(() => edit)
    const undo = vi.spyOn(commands, 'undo').mockResolvedValue(null)

    const action = enqueueHistory('undo')
    await Promise.resolve()
    expect(undo).not.toHaveBeenCalled()

    finishEdit()
    await action
    expect(undo).toHaveBeenCalledOnce()
    unregister()
  })
})

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}
