'use client'

import { call } from '@/lib/backend'
import { pageKey, pagesKey, projectKey, queryClient, refresh } from '@/lib/queries'
import { commands, type Page } from '@koharu/bridge/protocol'

/** Mark the active page's cleanup as done, or clear the mark. */
export function toggleCleanupDone(): void {
  const page = queryClient.getQueryData<Page | null>(pageKey)
  if (!page) return
  void call(commands.setCleanupDone, page.id, !page.cleanup_done)
    .then(() => refresh(projectKey, pagesKey, pageKey))
    .catch(() => undefined)
}
