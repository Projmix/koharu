'use client'

import { useEffect } from 'react'

import { ColorSamplingProvider } from '@/components/controls/ColorSampling'
import { ActivityCenter } from '@/components/editor/ActivityCenter'
import { CanvasWorkspace } from '@/components/editor/CanvasWorkspace'
import { PageRail } from '@/components/editor/PageRail'
import { RightSidebar } from '@/components/editor/RightSidebar'
import { useKoharuStore, type EditorPanel } from '@/lib/store'
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
  usePanelRef,
} from '@koharu/ui/components/resizable'

/**
 * Keeps a collapsible side panel in step with its shown/hidden state: the
 * View menu and the canvas bar change the state, a drag to the edge changes
 * the panel.
 */
function usePanelToggle(panel: EditorPanel) {
  const ref = usePanelRef()
  const open = useKoharuStore((state) => state.panels[panel])
  useEffect(() => {
    const handle = ref.current
    if (!handle || handle.isCollapsed() === !open) return
    if (open) handle.expand()
    else handle.collapse()
  }, [open, ref])
  const onResize = () => {
    const handle = ref.current
    if (!handle) return
    const { panels, setPanelOpen } = useKoharuStore.getState()
    if (panels[panel] === handle.isCollapsed()) setPanelOpen(panel, !handle.isCollapsed())
  }
  return { panelRef: ref, onResize }
}

export function Editor() {
  const pages = usePanelToggle('pages')
  const inspector = usePanelToggle('inspector')
  return (
    <ColorSamplingProvider>
      <div className='relative min-h-0 flex-1 bg-transparent'>
        <ActivityCenter />
        <ResizablePanelGroup id='editor' orientation='horizontal' className='h-full min-h-0'>
          <ResizablePanel
            id='pages'
            {...pages}
            defaultSize='18%'
            minSize='14%'
            maxSize='22%'
            collapsible
            collapsedSize='0%'
            className='min-h-0 overflow-hidden'
          >
            <PageRail />
          </ResizablePanel>
          <ResizableHandle className='w-0 bg-transparent' />
          <ResizablePanel
            id='canvas'
            defaultSize='58%'
            minSize='50%'
            className='workspace-corner-mask relative z-10 min-h-0 rounded-tl-2xl bg-transparent shadow-[var(--shadow-content)]'
          >
            <CanvasWorkspace />
          </ResizablePanel>
          <ResizableHandle className='w-0 bg-transparent' />
          <ResizablePanel
            id='inspector'
            {...inspector}
            defaultSize='24%'
            minSize='20%'
            maxSize='27%'
            collapsible
            collapsedSize='0%'
            className='min-h-0 overflow-hidden border-l border-border/40 bg-[var(--surface-panel)]'
          >
            <RightSidebar />
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </ColorSamplingProvider>
  )
}
