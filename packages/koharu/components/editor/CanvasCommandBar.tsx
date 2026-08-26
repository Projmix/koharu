'use client'

import { Redo2, Undo2 } from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { InferenceControl } from '@/components/editor/InferenceControl'
import { call } from '@/lib/backend'
import { enqueueHistory } from '@/lib/history'
import { usePage, useProject } from '@/lib/queries'
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
    </header>
  )
}
