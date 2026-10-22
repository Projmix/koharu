'use client'

import { ChevronDown, FolderOpen, LoaderCircle } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'

import { call } from '@/lib/backend'
import { abandonPageSwitch } from '@/lib/navigation'
import { pageKey, pagesKey, projectKey, refresh, useProject } from '@/lib/queries'
import { commands, type ProjectSummary } from '@koharu/bridge/protocol'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@koharu/ui/components/dropdown-menu'
import { cn } from '@koharu/ui/lib/utils'

/** The open project's name; opens another project without going back to the start view. */
export function ProjectSwitcher() {
  const { t } = useTranslation()
  const project = useProject().data
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [opening, setOpening] = useState<string | null>(null)

  const open = async (name: string) => {
    if (opening || name === project?.name) return
    setOpening(name)
    try {
      abandonPageSwitch()
      await call(commands.openProject, name)
      await refresh(projectKey, pagesKey, pageKey)
    } finally {
      setOpening(null)
    }
  }

  return (
    <DropdownMenu
      onOpenChange={(isOpen) => {
        if (isOpen) {
          void call(commands.listProjects)
            .then(setProjects)
            .catch(() => undefined)
        }
      }}
    >
      <DropdownMenuTrigger
        aria-label={t('navigator.switchProject')}
        title={project?.name}
        className='flex min-w-0 items-center gap-1 rounded-md px-1 py-0.5 text-[11px] font-semibold outline-none hover:bg-primary/10 focus-visible:ring-2 focus-visible:ring-ring/30'
      >
        {opening ? (
          <LoaderCircle className='size-3 shrink-0 animate-spin' />
        ) : (
          <FolderOpen className='size-3 shrink-0 text-primary' />
        )}
        <span className='min-w-0 truncate'>{project?.name ?? t('navigator.pages')}</span>
        <ChevronDown className='size-3 shrink-0 text-muted-foreground' />
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align='start'
        className='max-h-[70vh] w-80 overflow-y-auto border border-border/50 p-0.5 shadow-sm ring-0'
      >
        {projects.map(({ name }) => (
          <DropdownMenuItem
            key={name}
            title={name}
            className={cn(
              'min-h-7 px-1.5 py-0.5 text-[11px]',
              name === project?.name && 'font-semibold text-primary',
            )}
            onClick={() => void open(name).catch(() => undefined)}
          >
            <span className='min-w-0 truncate'>{name}</span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
