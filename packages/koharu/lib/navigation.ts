'use client'

import { call } from '@/lib/backend'
import { pageKey, pagesKey, preparedPageKey, projectKey, queryClient } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import { showCanvasPage } from '@koharu/bridge/canvas'
import {
  commands,
  type CanvasPagePreparation,
  type Page,
  type PageSummary,
  type ProjectInfo,
} from '@koharu/bridge/protocol'

let request = 0
/** The page the newest switch asked for, while native code has not confirmed it. */
let requested: string | null = null

/**
 * Switch to the page `offset` places from the newest requested one, so
 * repeated presses step on before the previous switch is confirmed.
 */
export function stepPage(offset: number): void {
  const pages = queryClient.getQueryData<PageSummary[]>(pagesKey) ?? []
  const from = requested ?? queryClient.getQueryData<Page | null>(pageKey)?.id
  const target = pages[pages.findIndex((page) => page.id === from) + offset]
  if (from && target) switchPage(target.id, [target.id])
}

/** Abandon a page switch still waiting for native confirmation. */
export function abandonPageSwitch(): void {
  request += 1
  requested = null
}

/**
 * Activate `page` with the rail selection and layer selection given. A prepared
 * page shows at once from the optimistic cache; the native selection then
 * confirms it, and a failed confirmation restores the previous page. A newer
 * switch makes an older one obsolete.
 */
export function switchPage(page: string, pages: string[], layers: string[] = []): void {
  const previousProject = queryClient.getQueryData<ProjectInfo | null>(projectKey)
  const previousPage = queryClient.getQueryData<Page | null>(pageKey)
  const prepared = queryClient.getQueryData<CanvasPagePreparation>(preparedPageKey(page))
  const activated = showCanvasPage(page, previousProject?.revision ?? null)
  const current = ++request
  requested = page
  const synchronize = () => {
    if (request !== current) return
    if (activated && previousProject && prepared?.revision === previousProject.revision) {
      queryClient.setQueryData(projectKey, { ...previousProject, active_page: page })
      queryClient.setQueryData(pageKey, prepared.page)
    }
    const { selectPages, selectLayers } = useKoharuStore.getState()
    selectPages(pages)
    selectLayers(layers)
    void call(commands.selectPage, page)
      .then((selection) => {
        if (request !== current) return
        requested = null
        queryClient.setQueryData(projectKey, selection.project)
        queryClient.setQueryData(pageKey, selection.page)
      })
      .catch(() => {
        if (request !== current) return
        requested = null
        if (queryClient.getQueryData<ProjectInfo | null>(projectKey)?.active_page === page) {
          queryClient.setQueryData(projectKey, previousProject)
          queryClient.setQueryData(pageKey, previousPage)
        }
      })
  }
  if (activated) {
    requestAnimationFrame(() => window.setTimeout(synchronize, 0))
  } else {
    synchronize()
  }
}
