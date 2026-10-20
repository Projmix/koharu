import { QueryClientProvider } from '@tanstack/react-query'
import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { CharacterPanel } from '@/components/editor/CharacterPanel'
import { projectKey, queryClient, refresh } from '@/lib/queries'
import { commands } from '@koharu/bridge/protocol'
import { TooltipProvider } from '@koharu/ui/components/tooltip'

describe('character panel', () => {
  it('imports characters from another project, never the open one', async () => {
    vi.spyOn(commands, 'getProject').mockResolvedValue({
      name: 'Chapter 2',
      revision: 1,
      active_page: null,
      can_undo: false,
      can_redo: false,
    })
    vi.spyOn(commands, 'getCharacters').mockResolvedValue([])
    vi.spyOn(commands, 'listProjects').mockResolvedValue([
      { name: 'Chapter 2' },
      { name: 'Chapter 1' },
    ])
    const importCharacters = vi.spyOn(commands, 'importCharacters').mockResolvedValue(2)
    const user = userEvent.setup()
    render(
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          <CharacterPanel />
        </TooltipProvider>
      </QueryClientProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Import from project…' }))
    const source = await screen.findByRole('menuitem', { name: 'Chapter 1' })
    expect(screen.queryByRole('menuitem', { name: 'Chapter 2' })).not.toBeInTheDocument()
    await user.click(source)

    await waitFor(() => expect(importCharacters).toHaveBeenCalledWith('Chapter 1'))
    expect(importCharacters).toHaveBeenCalledTimes(1)
  })

  it('does not show another project cast at the same revision', async () => {
    const project = (name: string) => ({
      name,
      revision: 7,
      active_page: null,
      can_undo: false,
      can_redo: false,
    })
    const getProject = vi.spyOn(commands, 'getProject').mockResolvedValue(project('Chapter 1'))
    const getCharacters = vi
      .spyOn(commands, 'getCharacters')
      .mockResolvedValue([{ id: 'izuku', name: 'Izuku', gender: 'male', notes: '' }])
    render(
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          <CharacterPanel />
        </TooltipProvider>
      </QueryClientProvider>,
    )
    expect(await screen.findByDisplayValue('Izuku')).toBeInTheDocument()

    getProject.mockResolvedValue(project('Chapter 2'))
    getCharacters.mockResolvedValue([])
    await act(() => refresh(projectKey))

    await waitFor(() => expect(screen.queryByDisplayValue('Izuku')).not.toBeInTheDocument())
  })
})
