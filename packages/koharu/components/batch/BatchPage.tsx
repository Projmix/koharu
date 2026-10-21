'use client'

import { ArrowLeft, LoaderCircle, Play, RefreshCw, Square } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { call } from '@/lib/backend'
import { pipelineStages, useKoharuStore } from '@/lib/store'
import { commands, type BatchProject, type Stage } from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import { Checkbox } from '@koharu/ui/components/checkbox'
import { ScrollArea } from '@koharu/ui/components/scroll-area'
import { cn } from '@koharu/ui/lib/utils'

const byName = (left: BatchProject, right: BatchProject) =>
  left.name.localeCompare(right.name, undefined, { numeric: true })

/** Stages a project cannot run: a required stage that never ran and is not selected. */
function missingStages(project: BatchProject, stages: Stage[]): Stage[] {
  const missing = new Set<Stage>()
  for (const stage of stages) {
    const progress = project.stages.find((entry) => entry.stage === stage)
    for (const required of progress?.requires ?? []) {
      const done = project.stages.find((entry) => entry.stage === required)?.done ?? 0
      if (done === 0 && !stages.includes(required)) missing.add(required)
    }
  }
  return pipelineStages.filter((stage) => missing.has(stage))
}

export function BatchPage() {
  const { t } = useTranslation()
  const setOpen = useKoharuStore((state) => state.setBatchOpen)
  const jobs = useKoharuStore((state) => state.jobs)
  const [projects, setProjects] = useState<BatchProject[] | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [stages, setStages] = useState<Stage[]>(['detection', 'inpainting'])
  const [starting, setStarting] = useState(false)

  const running = projects?.some((project) => project.run === 'queued' || project.run === 'running')
  const runningJob = Object.values(jobs).find((job) => job.state === 'running')

  const reload = useCallback(async () => {
    setProjects((await call(commands.getBatchProjects)).sort(byName))
  }, [])

  useEffect(() => {
    void reload().catch(() => undefined)
  }, [reload])

  useEffect(() => {
    if (!running) return
    const interval = window.setInterval(() => void reload().catch(() => undefined), 2000)
    return () => window.clearInterval(interval)
  }, [reload, running])

  const runnable = useMemo(
    () =>
      (projects ?? []).filter(
        (project) => selected.has(project.name) && missingStages(project, stages).length === 0,
      ),
    [projects, selected, stages],
  )

  const toggleStage = (stage: Stage) =>
    setStages((current) =>
      current.includes(stage)
        ? current.filter((value) => value !== stage)
        : pipelineStages.filter((value) => value === stage || current.includes(value)),
    )

  const toggleProject = (name: string, checked: boolean) =>
    setSelected((current) => {
      const next = new Set(current)
      if (checked) next.add(name)
      else next.delete(name)
      return next
    })

  const allSelected = !!projects?.length && projects.every((project) => selected.has(project.name))

  const start = async () => {
    if (runnable.length === 0 || stages.length === 0) return
    setStarting(true)
    try {
      await call(
        commands.runBatch,
        runnable.map((project) => project.name),
        stages,
      )
      await reload()
    } finally {
      setStarting(false)
    }
  }

  return (
    <main className='flex min-h-0 flex-1 flex-col overflow-hidden bg-[var(--surface-canvas)]'>
      <header className='flex h-14 shrink-0 items-center gap-3 border-b border-border/80 px-6'>
        <Button
          type='button'
          variant='ghost'
          size='icon-sm'
          className='size-8 text-muted-foreground hover:text-foreground'
          aria-label={t('common.back')}
          onClick={() => setOpen(false)}
        >
          <ArrowLeft className='size-4' />
        </Button>
        <div className='min-w-0'>
          <h1 className='text-[13px] font-semibold tracking-[-0.02em]'>{t('batch.title')}</h1>
          <p className='truncate text-[10px] text-muted-foreground'>{t('batch.description')}</p>
        </div>
        <Button
          type='button'
          variant='ghost'
          size='icon-sm'
          className='ml-auto size-8 text-muted-foreground hover:text-foreground'
          aria-label={t('batch.refresh')}
          onClick={() => void reload().catch(() => undefined)}
        >
          <RefreshCw className='size-3.5' />
        </Button>
      </header>

      <div className='flex shrink-0 flex-wrap items-center gap-2 border-b border-border/70 px-6 py-3'>
        <span className='mr-1 text-[11px] text-muted-foreground'>{t('batch.stages')}</span>
        {pipelineStages.map((stage) => (
          <button
            key={stage}
            type='button'
            aria-pressed={stages.includes(stage)}
            onClick={() => toggleStage(stage)}
            disabled={running}
            className={cn(
              'h-7 rounded-md border px-2.5 text-[11px] transition-colors disabled:opacity-50',
              stages.includes(stage)
                ? 'border-primary bg-primary text-primary-foreground'
                : 'border-border bg-background text-foreground hover:bg-foreground/[0.05]',
            )}
          >
            {t(`phaseShort.${stage}`)}
          </button>
        ))}
        <div className='ml-auto flex items-center gap-2'>
          {running ? (
            <Button
              type='button'
              size='sm'
              variant='outline'
              className='h-8 gap-1.5 text-[11px]'
              disabled={!runningJob}
              onClick={() =>
                runningJob && void call(commands.stopJob, runningJob.id).catch(() => undefined)
              }
            >
              <Square className='size-3' />
              {t('batch.stop')}
            </Button>
          ) : (
            <Button
              type='button'
              size='sm'
              className='h-8 gap-1.5 text-[11px]'
              disabled={starting || runnable.length === 0 || stages.length === 0}
              onClick={() => void start().catch(() => undefined)}
            >
              {starting ? (
                <LoaderCircle className='size-3.5 animate-spin' />
              ) : (
                <Play className='size-3.5' />
              )}
              {t('batch.run', { count: runnable.length })}
            </Button>
          )}
        </div>
      </div>

      <ScrollArea className='min-h-0 flex-1'>
        {projects === null ? (
          <p className='p-6 text-[11px] text-muted-foreground'>{t('start.loadingProjects')}</p>
        ) : projects.length === 0 ? (
          <p className='p-6 text-[11px] text-muted-foreground'>{t('start.emptyTitle')}</p>
        ) : (
          <table className='w-full border-collapse text-[11px]'>
            <thead className='sticky top-0 bg-[var(--surface-canvas)] text-left text-muted-foreground'>
              <tr className='border-b border-border/70'>
                <th className='w-10 py-2 pl-6'>
                  <Checkbox
                    checked={allSelected}
                    aria-label={t('batch.selectAll')}
                    disabled={running}
                    onCheckedChange={(checked) =>
                      setSelected(new Set(checked ? projects.map((project) => project.name) : []))
                    }
                  />
                </th>
                <th className='py-2 pr-4 pl-3 font-medium'>{t('start.projectName')}</th>
                {pipelineStages.map((stage) => (
                  <th key={stage} className='px-2 py-2 text-center font-medium'>
                    {t(`phaseShort.${stage}`)}
                  </th>
                ))}
                <th className='py-2 pr-6 pl-2 font-medium'>{t('batch.state')}</th>
              </tr>
            </thead>
            <tbody>
              {projects.map((project) => {
                const missing = selected.has(project.name) ? missingStages(project, stages) : []
                const note =
                  missing.length > 0
                    ? t('batch.missing', {
                        stages: missing.map((stage) => t(`phaseShort.${stage}`)).join(', '),
                      })
                    : project.error
                return (
                  <tr key={project.name} className='border-b border-border/50 align-top'>
                    <td className='py-2.5 pl-6'>
                      <Checkbox
                        checked={selected.has(project.name)}
                        aria-label={project.name}
                        disabled={running}
                        onCheckedChange={(checked) => toggleProject(project.name, checked)}
                      />
                    </td>
                    <td className='max-w-[320px] py-2.5 pr-4 pl-3'>
                      <span className='block truncate font-medium'>{project.name}</span>
                      {note && (
                        <span
                          className={cn(
                            'mt-0.5 block text-[10px] leading-4 break-words',
                            missing.length > 0 || project.run === 'failed'
                              ? 'text-destructive'
                              : 'text-muted-foreground',
                          )}
                        >
                          {note}
                        </span>
                      )}
                    </td>
                    {pipelineStages.map((stage) => {
                      const progress = project.stages.find((entry) => entry.stage === stage)
                      const done = progress?.done ?? 0
                      const total = progress?.total ?? 0
                      return (
                        <td
                          key={stage}
                          className={cn(
                            'px-2 py-2.5 text-center tabular-nums',
                            total > 0 && done === total
                              ? 'text-emerald-600 dark:text-emerald-400'
                              : done > 0
                                ? 'text-amber-600 dark:text-amber-400'
                                : 'text-muted-foreground',
                          )}
                        >
                          {total > 0 ? `${done}/${total}` : '–'}
                        </td>
                      )
                    })}
                    <td className='py-2.5 pr-6 pl-2 whitespace-nowrap text-muted-foreground'>
                      {project.run ? t(`batch.runState.${project.run}`) : ''}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
        <p className='px-6 py-4 text-[10px] leading-4 text-muted-foreground'>{t('batch.hint')}</p>
      </ScrollArea>
    </main>
  )
}
