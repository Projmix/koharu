'use client'

import {
  CheckCheck,
  ChevronLeft,
  ChevronRight,
  Columns2,
  PanelLeft,
  PanelRight,
  Redo2,
  RefreshCw,
  Undo2,
} from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { InferenceControl } from '@/components/editor/InferenceControl'
import { call } from '@/lib/backend'
import { toggleCleanupDone } from '@/lib/cleanup'
import { enqueueHistory } from '@/lib/history'
import { stepPage } from '@/lib/navigation'
import { usePage, usePages, useProject } from '@/lib/queries'
import { useKoharuStore, type PipelineScope } from '@/lib/store'
import { commands, type Scope, type Stage } from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'

export function CanvasCommandBar() {
  const { t } = useTranslation()
  const project = useProject().data
  const page = usePage().data
  const selectedPages = useKoharuStore((state) => state.selectedPages)
  const jobs = useKoharuStore((state) => state.jobs)
  const running = Object.values(jobs).find((job) => job.state === 'running')
  const compareOriginal = useKoharuStore((state) => state.compareOriginal)
  const setCompareOriginal = useKoharuStore((state) => state.setCompareOriginal)
  const panels = useKoharuStore((state) => state.panels)
  const cleanupShortcut = useKoharuStore((state) => state.shortcuts.cleanup_done)
  const setPanelOpen = useKoharuStore((state) => state.setPanelOpen)
  const pages = usePages().data ?? []
  const index = page ? pages.findIndex((summary) => summary.id === page.id) : -1

  const run = (selection: PipelineScope, stages: Stage[]) => {
    if (!page) return
    const scope: Scope =
      selection === 'project'
        ? { scope: 'project' }
        : selection === 'selected-pages'
          ? { scope: 'pages', value: selectedPages }
          : { scope: 'pages', value: [page.id] }
    void call(commands.process, scope, { operation: 'stages', stages }).catch(() => undefined)
  }

  return (
    <header className='flex h-10 shrink-0 items-center gap-2 border-b border-border/80 bg-[var(--surface-toolbar)] px-2.5'>
      <div className='flex shrink-0 items-center gap-0.5'>
        <Button
          type='button'
          variant='ghost'
          size='icon-sm'
          aria-label={t('menu.pagesPanel')}
          aria-pressed={panels.pages}
          title={t('menu.pagesPanel')}
          onClick={() => setPanelOpen('pages', !panels.pages)}
        >
          <PanelLeft className='size-3.5' />
        </Button>
        <Button
          type='button'
          variant={compareOriginal ? 'secondary' : 'ghost'}
          size='icon-sm'
          aria-label={t('canvas.compareOriginal')}
          aria-pressed={compareOriginal}
          title={t('canvas.compareOriginal')}
          disabled={!page}
          onClick={() => setCompareOriginal(!compareOriginal)}
        >
          <Columns2 className='size-3.5' />
        </Button>
        {compareOriginal && (
          <>
            <Button
              type='button'
              variant='ghost'
              size='icon-sm'
              aria-label={t('canvas.previousPage')}
              title={`${t('canvas.previousPage')} (←)`}
              disabled={index <= 0}
              onClick={() => stepPage(-1)}
            >
              <ChevronLeft className='size-3.5' />
            </Button>
            <span className='text-[11px] text-muted-foreground tabular-nums'>
              {index + 1} / {pages.length}
            </span>
            <Button
              type='button'
              variant='ghost'
              size='icon-sm'
              aria-label={t('canvas.nextPage')}
              title={`${t('canvas.nextPage')} (→)`}
              disabled={index < 0 || index >= pages.length - 1}
              onClick={() => stepPage(1)}
            >
              <ChevronRight className='size-3.5' />
            </Button>
            <Button
              type='button'
              variant={page?.cleanup_done ? 'secondary' : 'ghost'}
              size='icon-sm'
              aria-pressed={Boolean(page?.cleanup_done)}
              aria-label={t('canvas.cleanupDone')}
              title={`${t('canvas.cleanupDone')} (Alt+${cleanupShortcut.toUpperCase()}): ${t('canvas.cleanupDoneDescription')}`}
              disabled={!page}
              onClick={toggleCleanupDone}
            >
              <CheckCheck className='size-3.5' />
            </Button>
            <Button
              type='button'
              variant='ghost'
              size='icon-sm'
              aria-label={t('canvas.regenerateCleanup')}
              title={`${t('canvas.regenerateCleanup')}: ${t('canvas.regenerateCleanupDescription')}`}
              disabled={!page || Boolean(running)}
              onClick={() => run('page', ['inpainting'])}
            >
              <RefreshCw className='size-3.5' />
            </Button>
          </>
        )}
      </div>
      <div className='min-w-0 flex-1' />
      <div className='flex shrink-0 items-center gap-0.5' aria-label={t('menu.edit')}>
        <Button
          type='button'
          variant='ghost'
          size='icon-sm'
          aria-label={t('menu.undo')}
          title={`${t('menu.undo')} (Ctrl+Z)`}
          disabled={!project?.can_undo}
          onClick={() => void enqueueHistory('undo').catch(() => undefined)}
        >
          <Undo2 className='size-3.5' />
        </Button>
        <Button
          type='button'
          variant='ghost'
          size='icon-sm'
          aria-label={t('menu.redo')}
          title={`${t('menu.redo')} (Ctrl+Y)`}
          disabled={!project?.can_redo}
          onClick={() => void enqueueHistory('redo').catch(() => undefined)}
        >
          <Redo2 className='size-3.5' />
        </Button>
      </div>
      <InferenceControl disabled={!page || Boolean(running)} onRun={run} />
      <Button
        type='button'
        variant='ghost'
        size='icon-sm'
        aria-label={t('menu.inspectorPanel')}
        aria-pressed={panels.inspector}
        title={t('menu.inspectorPanel')}
        onClick={() => setPanelOpen('inspector', !panels.inspector)}
      >
        <PanelRight className='size-3.5' />
      </Button>
    </header>
  )
}
