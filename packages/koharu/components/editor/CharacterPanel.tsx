'use client'

import { FolderInput, Plus, Trash2, Users } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { call } from '@/lib/backend'
import { pageKey, projectKey, refresh, useCharacters, useProject } from '@/lib/queries'
import {
  type Character,
  type CharacterInput,
  commands,
  type Gender,
  type ProjectSummary,
} from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@koharu/ui/components/dropdown-menu'
import { Input } from '@koharu/ui/components/input'
import { ScrollArea } from '@koharu/ui/components/scroll-area'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'
import { toast } from '@koharu/ui/components/toast'
import { Tooltip, TooltipContent, TooltipTrigger } from '@koharu/ui/components/tooltip'

const genders: Gender[] = ['male', 'female', 'unknown']

const toInput = (character: Character): CharacterInput => ({
  id: character.id,
  name: character.name,
  gender: character.gender,
  notes: character.notes,
})

/**
 * The project's character list. Every committed change saves the whole list as
 * the user's; removing a character also clears it as a speaker natively.
 */
export function CharacterPanel() {
  const { t } = useTranslation()
  const characters = useCharacters().data ?? []
  const [draft, setDraft] = useState('')

  const save = (next: CharacterInput[]) =>
    call(commands.setCharacters, next)
      .then(() => refresh(projectKey, pageKey))
      .catch(() => undefined)
  const update = (index: number, change: Partial<CharacterInput>) =>
    save(
      characters.map((character, position) =>
        position === index ? { ...toInput(character), ...change } : toInput(character),
      ),
    )
  const add = () => {
    const name = draft.trim()
    if (!name || characters.some((character) => character.name === name)) return
    setDraft('')
    void save([...characters.map(toInput), { id: null, name, gender: 'unknown', notes: '' }])
  }

  return (
    <ScrollArea className='h-full'>
      <div className='grid gap-2 p-2.5'>
        <div className='flex items-center gap-1.5 text-[11px] font-medium'>
          <Users className='size-3.5 text-muted-foreground' />
          {t('characters.title')}
        </div>
        <p className='text-[10px] leading-4 text-muted-foreground'>{t('characters.description')}</p>
        {characters.length === 0 && (
          <p className='rounded-md bg-foreground/[0.04] px-2 py-1.5 text-[11px] text-muted-foreground'>
            {t('characters.empty')}
          </p>
        )}
        {characters.map((character, index) => (
          <div
            key={character.id}
            data-testid={`character-${character.id}`}
            className='grid min-w-0 gap-1 rounded-md border border-border/70 p-1.5'
          >
            <div className='flex min-w-0 items-center gap-1'>
              <CommitInput
                aria-label={t('characters.name')}
                className='h-6 min-w-0 flex-1 text-[11px]'
                value={character.name}
                onCommit={(name) =>
                  name.trim() &&
                  !characters.some(
                    (other, position) => position !== index && other.name === name.trim(),
                  ) &&
                  update(index, { name })
                }
              />
              <Select
                value={character.gender}
                items={Object.fromEntries(
                  genders.map((gender) => [gender, t(`characters.genders.${gender}`)]),
                )}
                onValueChange={(gender) => gender && update(index, { gender: gender as Gender })}
              >
                <SelectTrigger
                  aria-label={t('characters.gender')}
                  className='h-6 w-24 shrink-0 text-[11px]'
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {genders.map((gender) => (
                    <SelectItem key={gender} value={gender}>
                      {t(`characters.genders.${gender}`)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Tooltip>
                <TooltipTrigger
                  render={
                    <Button
                      type='button'
                      variant='ghost'
                      size='icon-xs'
                      aria-label={t('characters.remove', { name: character.name })}
                      className='size-6 shrink-0 text-muted-foreground hover:text-foreground'
                      onClick={() =>
                        save(characters.filter((_, position) => position !== index).map(toInput))
                      }
                    />
                  }
                >
                  <Trash2 className='size-3' />
                </TooltipTrigger>
                <TooltipContent side='left'>
                  {t('characters.remove', { name: character.name })}
                </TooltipContent>
              </Tooltip>
            </div>
            <CommitInput
              aria-label={t('characters.notes')}
              placeholder={t('characters.notesPlaceholder')}
              className='h-6 min-w-0 text-[11px]'
              value={character.notes}
              onCommit={(notes) => update(index, { notes })}
            />
          </div>
        ))}
        <div className='flex min-w-0 items-center gap-1'>
          <Input
            aria-label={t('characters.newName')}
            placeholder={t('characters.newName')}
            className='h-6 min-w-0 flex-1 text-[11px]'
            value={draft}
            onChange={(event) => setDraft(event.currentTarget.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') add()
            }}
          />
          <Button
            type='button'
            size='xs'
            variant='secondary'
            className='h-6 shrink-0 gap-1 text-[11px]'
            disabled={!draft.trim()}
            onClick={add}
          >
            <Plus className='size-3' />
            {t('characters.add')}
          </Button>
        </div>
        <ImportCharactersMenu />
      </div>
    </ScrollArea>
  )
}

/**
 * Imports the character list of another project, such as the previous
 * chapter. Native code reads that project without changing it and adds the
 * characters this project does not list yet as one undo step.
 */
function ImportCharactersMenu() {
  const { t } = useTranslation()
  const current = useProject().data?.name
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null)
  const sources = (projects ?? []).filter((project) => project.name !== current)

  const load = (open: boolean) => {
    if (!open) return
    call(commands.listProjects)
      .then(setProjects)
      .catch(() => setProjects([]))
  }
  const importFrom = (source: string) =>
    call(commands.importCharacters, source)
      .then((count) => {
        toast.add({ type: 'success', title: t('characters.imported', { count }) })
        return refresh(projectKey, pageKey)
      })
      .catch(() => undefined)

  return (
    <DropdownMenu onOpenChange={load}>
      <DropdownMenuTrigger
        render={
          <Button
            type='button'
            size='xs'
            variant='ghost'
            className='h-6 justify-start gap-1 text-[11px] text-muted-foreground'
          />
        }
      >
        <FolderInput className='size-3' />
        {t('characters.import')}
      </DropdownMenuTrigger>
      <DropdownMenuContent align='start' sideOffset={4} className='max-h-64 w-56'>
        {projects !== null && sources.length === 0 && (
          <DropdownMenuItem disabled>{t('characters.importNone')}</DropdownMenuItem>
        )}
        {sources.map((project) => (
          <DropdownMenuItem key={project.name} onClick={() => void importFrom(project.name)}>
            <span className='truncate'>{project.name}</span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/** A text input that commits on Enter or blur and reverts on Escape. */
function CommitInput({
  value,
  onCommit,
  ...props
}: Omit<React.ComponentProps<typeof Input>, 'value' | 'onChange'> & {
  value: string
  onCommit: (value: string) => unknown
}) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  const commit = () => {
    if (draft !== value) onCommit(draft)
  }
  return (
    <Input
      {...props}
      value={draft}
      onChange={(event) => setDraft(event.currentTarget.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === 'Enter') event.currentTarget.blur()
        if (event.key === 'Escape') {
          setDraft(value)
          event.currentTarget.blur()
        }
      }}
    />
  )
}
