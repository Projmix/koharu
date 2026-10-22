'use client'

import { useThumbnail } from '@/lib/queries'
import { type PageSummary } from '@koharu/bridge/protocol'

/**
 * The page rail thumbnail of a requested page that was not prepared yet,
 * fitted to the canvas, so a page switch shows the new page at once. The
 * prepared page replaces it as soon as its frame is active; the canvas itself
 * still keeps the previous frame until then.
 */
export function PagePreview({ page }: { page: PageSummary | undefined }) {
  const source = useThumbnail(page?.id, page?.source_asset ?? undefined)

  return (
    <div
      data-testid='page-preview'
      className='pointer-events-none absolute inset-0 z-10 bg-[var(--surface-canvas)] p-4'
    >
      {source && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={source} alt='' draggable={false} className='size-full object-contain' />
      )}
    </div>
  )
}
