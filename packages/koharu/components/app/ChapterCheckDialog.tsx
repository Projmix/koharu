'use client'

import { LoaderCircle } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { call } from '@/lib/backend'
import { switchPage } from '@/lib/navigation'
import { usePages } from '@/lib/queries'
import {
  commands,
  type ChapterIssue,
  type ExportFormat,
  type TextFitWarningKind,
  type TranslationIssue,
} from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@koharu/ui/components/dialog'
import { ScrollArea } from '@koharu/ui/components/scroll-area'

const translationIssueLabels = {
  empty: 'chapterCheck.emptyTranslation',
  japanese_text: 'chapterCheck.japaneseText',
} as const satisfies Record<TranslationIssue, string>

const fitIssueLabels = {
  overflow: 'layers.textOverflow',
  below_readable_size: 'layers.textBelowReadableSize',
  low_contrast: 'layers.textLowContrast',
} as const satisfies Record<TextFitWarningKind, string>

/**
 * The final chapter check that runs when an export format is chosen. A clean
 * chapter exports at once; otherwise the list shows what still needs a look and
 * export stays one click away.
 */
export function ChapterCheckDialog({
  format,
  onClose,
  onExport,
}: {
  format: ExportFormat | null
  onClose: () => void
  onExport: (format: ExportFormat) => void
}) {
  const { t } = useTranslation()
  const pages = usePages(format !== null).data ?? []
  const [issues, setIssues] = useState<ChapterIssue[] | null>(null)
  const handlers = useRef({ onClose, onExport })
  handlers.current = { onClose, onExport }

  useEffect(() => {
    if (format === null) return
    let current = true
    setIssues(null)
    call(commands.checkChapter)
      .then((found) => {
        if (!current) return
        if (found.length === 0) {
          handlers.current.onClose()
          handlers.current.onExport(format)
        } else {
          setIssues(found)
        }
      })
      .catch(() => {
        if (current) handlers.current.onClose()
      })
    return () => {
      current = false
    }
  }, [format])

  const labels = new Map(pages.map((page) => [page.id, page.label]))

  return (
    <Dialog open={format !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className='gap-4 p-5 sm:max-w-lg'>
        <DialogHeader className='gap-1'>
          <DialogTitle>{t('chapterCheck.title')}</DialogTitle>
          <DialogDescription>
            {issues === null
              ? t('chapterCheck.checking')
              : t('chapterCheck.issueCount', { count: issues.length })}
          </DialogDescription>
        </DialogHeader>
        <ScrollArea className='h-72 min-w-0 rounded-lg border border-border/75'>
          {issues === null ? (
            <div className='grid h-72 place-items-center'>
              <LoaderCircle
                className='size-4 animate-spin text-muted-foreground'
                aria-label={t('chapterCheck.checking')}
              />
            </div>
          ) : (
            <ul className='divide-y divide-border/65'>
              {issues.map((issue) => (
                <li key={issue.element}>
                  <button
                    type='button'
                    className='flex w-full items-start gap-3 px-3 py-2 text-left hover:bg-primary/10'
                    onClick={() => {
                      switchPage(issue.page, [issue.page], [issue.element])
                      onClose()
                    }}
                  >
                    <span className='w-14 shrink-0 truncate pt-px text-[10px] text-muted-foreground'>
                      {labels.get(issue.page) ?? ''}
                    </span>
                    <span className='min-w-0 flex-1'>
                      <span className='line-clamp-2 text-xs break-words'>{issue.text}</span>
                      <span className='mt-0.5 block text-[10px] text-destructive'>
                        {[
                          issue.translation && t(translationIssueLabels[issue.translation]),
                          issue.fit && t(fitIssueLabels[issue.fit]),
                          issue.ocr_unverified && t('layers.ocrUnverified'),
                        ]
                          .filter(Boolean)
                          .join(' · ')}
                      </span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </ScrollArea>
        <DialogFooter className='-mx-5 -mb-5'>
          <Button variant='outline' size='sm' onClick={onClose}>
            {t('common.cancel')}
          </Button>
          <Button
            size='sm'
            disabled={format === null || issues === null}
            onClick={() => {
              if (format === null) return
              onClose()
              onExport(format)
            }}
          >
            {t('chapterCheck.exportAnyway')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
