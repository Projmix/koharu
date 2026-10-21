'use client'

import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { call } from '@/lib/backend'
import { useKoharuStore } from '@/lib/store'
import { commands, type Page } from '@koharu/bridge/protocol'

/**
 * The page's imported source, read-only, beside the editable canvas. It
 * follows the canvas camera, so both panes show the same part of the page;
 * the pane has the canvas surface's size for that reason.
 */
export function OriginalView({ page }: { page: Page }) {
  const { t } = useTranslation()
  const camera = useKoharuStore((state) => state.camera)
  const [source, setSource] = useState<string | null>(null)

  const bytes = useQuery({
    queryKey: ['page-source', page.id],
    queryFn: async () => new Uint8Array(await call(commands.getPageSource, page.id)),
    staleTime: Number.POSITIVE_INFINITY,
  }).data

  useEffect(() => {
    if (!bytes) return
    const url = URL.createObjectURL(new Blob([bytes.buffer]))
    setSource(url)
    return () => {
      URL.revokeObjectURL(url)
      setSource(null)
    }
  }, [bytes])

  const dpr = window.devicePixelRatio
  const scale = camera.zoom / dpr
  return (
    <section
      aria-label={t('canvas.original')}
      className='relative min-h-0 min-w-0 flex-1 overflow-hidden border-r border-border/80 bg-[var(--surface-canvas)]'
    >
      {source ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={source}
          alt={t('canvas.original')}
          draggable={false}
          className='pointer-events-none absolute max-w-none select-none'
          style={{
            left: camera.translation[0] / dpr,
            top: camera.translation[1] / dpr,
            width: page.size.width * scale,
            height: page.size.height * scale,
          }}
        />
      ) : (
        <span className='absolute inset-0 grid place-items-center text-xs text-muted-foreground'>
          {t('common.loading')}
        </span>
      )}
      <span className='pointer-events-none absolute top-2 right-2 rounded bg-black/60 px-1.5 py-0.5 text-[11px] text-white'>
        {t('canvas.original')}
      </span>
    </section>
  )
}
