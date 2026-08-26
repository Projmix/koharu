'use client'

import { History, LoaderCircle, Plus, RotateCcw, Trash2 } from 'lucide-react'
import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { useTranslation } from 'react-i18next'

import { call } from '@/lib/backend'
import { pageKey, pagesKey, projectKey, refresh } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import { commands, type ProjectVersion } from '@koharu/bridge/protocol'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
} from '@koharu/ui/components/alert-dialog'
import { Button } from '@koharu/ui/components/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@koharu/ui/components/dialog'
import { Input } from '@koharu/ui/components/input'
import { ScrollArea } from '@koharu/ui/components/scroll-area'

type PendingAction = { kind: 'restore' | 'delete'; version: ProjectVersion }

export function ProjectVersionsDialog({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const { t, i18n } = useTranslation()
  const [versions, setVersions] = useState<ProjectVersion[]>([])
  const [name, setName] = useState('')
  const [busy, setBusy] = useState<string | null>(null)
  const [pending, setPending] = useState<PendingAction | null>(null)
  const selectPages = useKoharuStore((state) => state.selectPages)
  const selectLayers = useKoharuStore((state) => state.selectLayers)

  const reload = useCallback(async () => {
    setBusy('list')
    try {
      setVersions(await call(commands.listProjectVersions))
    } finally {
      setBusy(null)
    }
  }, [])

  useEffect(() => {
    if (!open) return
    void reload().catch(() => undefined)
  }, [open, reload])

  const save = async (event: FormEvent) => {
    event.preventDefault()
    const versionName = name.trim()
    if (!versionName || busy) return
    setBusy('save')
    try {
      const version = await call(commands.saveProjectVersion, versionName)
      setVersions((current) => [version, ...current])
      setName('')
    } finally {
      setBusy(null)
    }
  }

  const confirm = async () => {
    if (!pending || busy) return
    const { kind, version } = pending
    setBusy(`${kind}:${version.id}`)
    try {
      if (kind === 'delete') {
        await call(commands.deleteProjectVersion, version.id)
        setVersions((current) => current.filter((item) => item.id !== version.id))
      } else {
        await call(commands.restoreProjectVersion, version.id)
        selectPages([])
        selectLayers([])
        await refresh(projectKey, pagesKey, pageKey)
      }
      setPending(null)
    } finally {
      setBusy(null)
    }
  }

  const formatDate = (timestamp: number) =>
    new Intl.DateTimeFormat(i18n.language, {
      dateStyle: 'medium',
      timeStyle: 'short',
    }).format(new Date(timestamp))

  return (
    <AlertDialog
      open={pending !== null}
      onOpenChange={(next) => {
        if (!next && busy === null) setPending(null)
      }}
    >
      <Dialog
        open={open}
        onOpenChange={(next) => {
          if (busy === null) onOpenChange(next)
        }}
      >
        <DialogContent className='max-w-lg gap-4 p-5'>
          <DialogHeader className='gap-1'>
            <DialogTitle>{t('versions.title')}</DialogTitle>
            <DialogDescription>{t('versions.description')}</DialogDescription>
          </DialogHeader>

          <form className='flex gap-2' onSubmit={save}>
            <Input
              value={name}
              maxLength={120}
              autoComplete='off'
              disabled={busy !== null}
              placeholder={t('versions.namePlaceholder')}
              aria-label={t('versions.nameLabel')}
              onChange={(event) => setName(event.target.value)}
            />
            <Button type='submit' size='sm' disabled={!name.trim() || busy !== null}>
              {busy === 'save' ? (
                <LoaderCircle className='animate-spin' />
              ) : (
                <Plus aria-hidden='true' />
              )}
              {t('versions.save')}
            </Button>
          </form>

          <ScrollArea className='h-72 rounded-lg border border-border/75'>
            {busy === 'list' && versions.length === 0 ? (
              <div className='grid h-72 place-items-center text-xs text-muted-foreground'>
                <LoaderCircle className='size-4 animate-spin' aria-label={t('common.loading')} />
              </div>
            ) : versions.length === 0 ? (
              <div className='grid h-72 place-items-center px-8 text-center'>
                <div>
                  <History className='mx-auto size-6 text-muted-foreground' />
                  <p className='mt-2 text-sm font-medium'>{t('versions.emptyTitle')}</p>
                  <p className='mt-1 text-xs leading-5 text-muted-foreground'>
                    {t('versions.emptyDescription')}
                  </p>
                </div>
              </div>
            ) : (
              <ul className='divide-y divide-border/65'>
                {versions.map((version) => {
                  const rowBusy = busy?.endsWith(version.id) ?? false
                  return (
                    <li key={version.id} className='flex items-center gap-3 px-3 py-2.5'>
                      <div className='min-w-0 flex-1'>
                        <p className='truncate text-xs font-medium'>{version.name}</p>
                        <p className='mt-0.5 text-[10px] text-muted-foreground'>
                          {formatDate(version.created_at_ms)} ·{' '}
                          {t('versions.revision', {
                            revision: version.revision,
                          })}
                        </p>
                      </div>
                      {rowBusy ? (
                        <LoaderCircle className='mr-2 size-4 animate-spin text-muted-foreground' />
                      ) : (
                        <div className='flex shrink-0 gap-1'>
                          <Button
                            type='button'
                            variant='ghost'
                            size='icon-sm'
                            disabled={busy !== null}
                            aria-label={t('versions.restoreLabel', { name: version.name })}
                            title={t('versions.restore')}
                            onClick={() => setPending({ kind: 'restore', version })}
                          >
                            <RotateCcw />
                          </Button>
                          <Button
                            type='button'
                            variant='ghost'
                            size='icon-sm'
                            className='text-muted-foreground hover:text-destructive'
                            disabled={busy !== null}
                            aria-label={t('versions.deleteLabel', { name: version.name })}
                            title={t('versions.delete')}
                            onClick={() => setPending({ kind: 'delete', version })}
                          >
                            <Trash2 />
                          </Button>
                        </div>
                      )}
                    </li>
                  )
                })}
              </ul>
            )}
          </ScrollArea>
        </DialogContent>
      </Dialog>

      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogMedia
            className={
              pending?.kind === 'delete'
                ? 'bg-destructive/10 text-destructive'
                : 'bg-primary/10 text-primary'
            }
          >
            {pending?.kind === 'delete' ? <Trash2 /> : <RotateCcw />}
          </AlertDialogMedia>
          <AlertDialogTitle>
            {pending?.kind === 'delete' ? t('versions.deleteTitle') : t('versions.restoreTitle')}
          </AlertDialogTitle>
          <AlertDialogDescription>
            {pending?.kind === 'delete'
              ? t('versions.deleteDescription', { name: pending.version.name })
              : t('versions.restoreDescription', { name: pending?.version.name })}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={busy !== null}>{t('common.cancel')}</AlertDialogCancel>
          <AlertDialogAction
            variant={pending?.kind === 'delete' ? 'destructive' : 'default'}
            disabled={busy !== null}
            aria-busy={busy !== null}
            onClick={() => void confirm().catch(() => undefined)}
          >
            {pending?.kind === 'delete' ? t('versions.delete') : t('versions.restore')}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
