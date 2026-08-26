'use client'

import { Trash2 } from 'lucide-react'
import { useTranslation } from 'react-i18next'

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

export function DeleteAllPagesDialog({
  open,
  count,
  busy,
  onOpenChange,
  onConfirm,
}: {
  open: boolean
  count: number
  busy: boolean
  onOpenChange: (open: boolean) => void
  onConfirm: () => void
}) {
  const { t } = useTranslation()

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogMedia className='bg-destructive/10 text-destructive'>
            <Trash2 className='size-5' />
          </AlertDialogMedia>
          <AlertDialogTitle>{t('navigator.deleteAllTitle')}</AlertDialogTitle>
          <AlertDialogDescription>
            {t('navigator.deleteAllDescription', { count })}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={busy}>{t('common.cancel')}</AlertDialogCancel>
          <AlertDialogAction
            variant='destructive'
            disabled={busy}
            aria-busy={busy}
            onClick={onConfirm}
          >
            {busy ? t('navigator.deletingPages') : t('navigator.deleteAllAction')}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
