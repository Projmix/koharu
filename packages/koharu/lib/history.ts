'use client'

import { commands } from '@koharu/bridge/protocol'

import { call } from './backend'
import { pageKey, pagesKey, projectKey, refresh } from './queries'

export type HistoryAction = 'undo' | 'redo'

type PendingEdit = () => void | Promise<void>

// Text controls debounce their durable command. Keep their flush callbacks in
// one place so project history never runs ahead of an edit that is still being
// committed after focus moves to another layer.
const pendingEdits = new Set<PendingEdit>()

export function registerPendingEdit(flush: PendingEdit): () => void {
  pendingEdits.add(flush)
  return () => pendingEdits.delete(flush)
}

export async function flushPendingEdits(): Promise<void> {
  await Promise.all([...pendingEdits].map((flush) => flush()))
}

export function historyShortcutKey(event: KeyboardEvent): 'z' | 'y' | null {
  if (event.code === 'KeyZ') return 'z'
  if (event.code === 'KeyY') return 'y'
  const key = event.key.toLowerCase()
  return key === 'z' || key === 'y' ? key : null
}

// Native commands are asynchronous.  Keep one process-wide queue so keyboard
// repeats and menu clicks cannot observe the same backend revision and race
// each other.  The backend still owns the actual undo/redo stacks.
let queue: Promise<void> = Promise.resolve()

export function enqueueHistory(action: HistoryAction): Promise<void> {
  const pending = queue.then(async () => {
    await flushPendingEdits()
    await call(action === 'undo' ? commands.undo : commands.redo)
    await refresh(projectKey, pagesKey, pageKey)
  })
  queue = pending.then(
    () => undefined,
    () => undefined,
  )
  return pending
}

export function isEditableTarget(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    (target instanceof HTMLElement && target.isContentEditable)
  )
}

/** Bind project history shortcuts for the lifetime of an application shell. */
export function bindHistoryShortcuts(projectOpen: boolean): () => void {
  const onKeyDown = (event: KeyboardEvent) => {
    const command = event.ctrlKey || event.metaKey
    const key = historyShortcutKey(event)
    if (!command || key === null || isEditableTarget(event.target)) return
    if (!projectOpen || event.defaultPrevented) return
    event.preventDefault()
    void enqueueHistory(key === 'y' || event.shiftKey ? 'redo' : 'undo').catch(() => undefined)
  }
  window.addEventListener('keydown', onKeyDown, true)
  return () => window.removeEventListener('keydown', onKeyDown, true)
}
