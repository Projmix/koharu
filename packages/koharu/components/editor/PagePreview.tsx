'use client'

import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'

import { call } from '@/lib/backend'
import { commands, type PageSummary } from '@koharu/bridge/protocol'

/**
 * The page rail thumbnail of a requested page that was not prepared yet,
 * fitted to the canvas, so a page switch shows the new page at once. The
 * prepared page replaces it as soon as its frame is active; the canvas itself
 * still keeps the previous frame until then.
 */
export function PagePreview({ page }: { page: PageSummary | undefined }) {
  const [source, setSource] = useState<string | null>(null)
  const thumbnail = useQuery({
    queryKey: ['thumbnail', page?.source_asset],
    queryFn: async () => new Uint8Array(await call(commands.getThumbnail, page!.id)),
    enabled: Boolean(page?.source_asset),
    staleTime: Number.POSITIVE_INFINITY,
  }).data

  useEffect(() => {
    if (!thumbnail) return
    const url = URL.createObjectURL(new Blob([thumbnail.buffer], { type: 'image/webp' }))
    setSource(url)
    return () => {
      URL.revokeObjectURL(url)
      setSource(null)
    }
  }, [thumbnail])

  return (
    <div className='pointer-events-none absolute inset-0 z-10 bg-[var(--surface-canvas)] p-4'>
      {source && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={source} alt='' draggable={false} className='size-full object-contain' />
      )}
    </div>
  )
}
