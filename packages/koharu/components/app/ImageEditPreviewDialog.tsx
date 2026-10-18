'use client'

import { X } from 'lucide-react'
import { useId, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { inpaintingProviderName } from '@/lib/inpaintingProviders'
import { useKoharuStore } from '@/lib/store'
import { commands, type ImageEditPreview } from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@koharu/ui/components/dialog'
import { Textarea } from '@koharu/ui/components/textarea'

export function ImageEditPreviewDialog() {
  const preview = useKoharuStore((state) => state.imageEditPreviews[0])
  return preview ? <PreviewEditor key={preview.id} preview={preview} /> : null
}

function PreviewEditor({ preview }: { preview: ImageEditPreview }) {
  const { t } = useTranslation()
  const promptId = useId()
  const [prompt, setPrompt] = useState(preview.prompt)
  const [actualSize, setActualSize] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const submitting = useRef(false)

  const resolve = async (prompt: string | null) => {
    if (submitting.current) return
    submitting.current = true
    setBusy(true)
    setError(null)
    try {
      await commands.resolveImageEditPreview(preview.id, prompt)
      // The native queue owns removal and advancement, including stop/project changes.
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error))
      submitting.current = false
      setBusy(false)
    }
  }

  return (
    <Dialog
      open
      disablePointerDismissal
      onOpenChange={(open, event) => {
        if (!open) {
          event.cancel()
          void resolve(null)
        }
      }}
    >
      <DialogContent
        showCloseButton={false}
        className='flex max-h-[calc(100dvh-2rem)] w-[calc(100vw-2rem)] flex-col sm:max-w-6xl'
      >
        <DialogHeader className='shrink-0 pr-8'>
          <DialogTitle>{t('imageEditPreview.title')}</DialogTitle>
          <DialogDescription>{t('imageEditPreview.description')}</DialogDescription>
        </DialogHeader>
        <DialogClose
          disabled={busy}
          render={<Button variant='ghost' size='icon-sm' className='absolute top-2 right-2' />}
          aria-label={t('imageEditPreview.close')}
        >
          <X />
        </DialogClose>
        <div className='grid min-h-0 gap-4 overflow-y-auto lg:grid-cols-[minmax(0,1.4fr)_minmax(18rem,1fr)]'>
          <div className='grid min-w-0 content-start gap-2'>
            <div className='flex items-center justify-between gap-2 text-xs'>
              <span className='text-muted-foreground'>
                {preview.width} × {preview.height} px
              </span>
              <Button
                variant='outline'
                size='sm'
                aria-pressed={actualSize}
                onClick={() => setActualSize((value) => !value)}
              >
                {actualSize ? t('imageEditPreview.fitImage') : t('imageEditPreview.actualSize')}
              </Button>
            </div>
            <div className='h-[min(55vh,40rem)] overflow-auto rounded-lg border border-border bg-muted/40'>
              <img
                src={preview.image_url}
                alt={t('imageEditPreview.image')}
                width={preview.width}
                height={preview.height}
                draggable={false}
                className={actualSize ? 'max-w-none' : 'h-full w-full object-contain'}
              />
            </div>
            <p className='text-xs leading-5 text-muted-foreground'>
              {t(
                preview.restoration
                  ? 'imageEditPreview.restorationLayers'
                  : 'imageEditPreview.cleanupLayers',
              )}
            </p>
          </div>
          <div className='flex min-w-0 flex-col gap-3'>
            <div className='grid gap-1 text-xs'>
              <span className='font-medium'>{preview.page_label}</span>
              <span className='text-muted-foreground'>
                {t(
                  preview.restoration ? 'imageEditPreview.restoration' : 'imageEditPreview.cleanup',
                )}
              </span>
              <div className='flex flex-wrap gap-x-2 text-muted-foreground'>
                <span>{inpaintingProviderName(preview.provider)}</span>
                <span className='break-all'>{preview.model}</span>
              </div>
            </div>
            <label htmlFor={promptId} className='text-xs font-medium'>
              {t('imageEditPreview.prompt')}
            </label>
            <Textarea
              id={promptId}
              value={prompt}
              disabled={busy}
              onChange={(event) => setPrompt(event.currentTarget.value)}
              className='field-sizing-fixed min-h-48 flex-1 resize-y overflow-y-auto text-xs leading-5'
            />
            <Button
              variant='ghost'
              size='sm'
              className='self-start'
              disabled={busy || prompt === preview.prompt}
              onClick={() => setPrompt(preview.prompt)}
            >
              {t('imageEditPreview.resetPrompt')}
            </Button>
          </div>
        </div>
        {error && (
          <p role='alert' className='text-sm text-destructive'>
            {t('imageEditPreview.failed', { error })}
          </p>
        )}
        <DialogFooter className='shrink-0'>
          <Button variant='outline' disabled={busy} onClick={() => void resolve(null)}>
            {t('imageEditPreview.cancel')}
          </Button>
          <Button
            disabled={busy || !prompt.trim() || prompt.includes('\0')}
            aria-busy={busy}
            onClick={() => void resolve(prompt)}
          >
            {t('imageEditPreview.send')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
