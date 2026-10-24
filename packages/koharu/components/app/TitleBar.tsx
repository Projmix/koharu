'use client'

import { openUrl } from '@tauri-apps/plugin-opener'
import { FilePlus2, FolderOpen, History, LoaderCircle, Settings, Trash2 } from 'lucide-react'
import Image from 'next/image'
import { useState, type ComponentProps } from 'react'
import { useTranslation } from 'react-i18next'

import { AboutDialog } from '@/components/app/AboutDialog'
import { ChapterCheckDialog } from '@/components/app/ChapterCheckDialog'
import { DeleteAllPagesDialog } from '@/components/app/DeleteAllPagesDialog'
import { ProjectVersionsDialog } from '@/components/app/ProjectVersionsDialog'
import { useMacOS, WindowControls } from '@/components/app/WindowChrome'
import { call } from '@/lib/backend'
import { selectableLayer } from '@/lib/geometry'
import { enqueueHistory } from '@/lib/history'
import {
  pageKey,
  pagesKey,
  projectKey,
  refresh,
  useCommand,
  useImportPages,
  usePage,
  usePages,
  useProject,
} from '@/lib/queries'
import {
  duplicableSelection,
  duplicateLayer,
  mergeableSelection,
  mergeLayers,
} from '@/lib/rasterLayers'
import { useKoharuStore } from '@/lib/store'
import {
  commands,
  type ExportFormat,
  type Operation,
  type Scope,
  type Stage,
} from '@koharu/bridge/protocol'
import {
  Menubar,
  MenubarCheckboxItem,
  MenubarContent as UiMenubarContent,
  MenubarItem as UiMenubarItem,
  MenubarMenu,
  MenubarSeparator as UiMenubarSeparator,
  MenubarShortcut as UiMenubarShortcut,
  MenubarSub,
  MenubarSubContent,
  MenubarSubTrigger,
  MenubarTrigger as UiMenubarTrigger,
} from '@koharu/ui/components/menubar'
import { cn } from '@koharu/ui/lib/utils'

export function TitleBar() {
  const { t } = useTranslation()
  const [aboutOpen, setAboutOpen] = useState(false)
  const [clearProjectOpen, setClearProjectOpen] = useState(false)
  const [versionsOpen, setVersionsOpen] = useState(false)
  const [clearingProject, setClearingProject] = useState(false)
  const [checkedFormat, setCheckedFormat] = useState<ExportFormat | null>(null)
  const macOS = useMacOS()
  const project = useProject().data
  const pagesQuery = usePages(Boolean(project))
  const pageQuery = usePage(Boolean(project))
  const pages = project ? (pagesQuery.data ?? []) : []
  const page = project ? (pageQuery.data ?? null) : null
  const selectedPages = useKoharuStore((state) => state.selectedPages)
  const selectedLayers = useKoharuStore((state) => state.selectedLayers)
  const selectPages = useKoharuStore((state) => state.selectPages)
  const selectLayers = useKoharuStore((state) => state.selectLayers)
  const setSettingsOpen = useKoharuStore((state) => state.setSettingsOpen)
  const setBatchOpen = useKoharuStore((state) => state.setBatchOpen)
  const requestCanvasFit = useKoharuStore((state) => state.requestCanvasFit)
  const panels = useKoharuStore((state) => state.panels)
  const setPanelOpen = useKoharuStore((state) => state.setPanelOpen)
  const { importPages, importing } = useImportPages()
  const { run: exportProject, busy: exporting } = useCommand(
    ['export-project'],
    commands.export,
    'menu.export',
  )

  const run = (scope: Scope, operation: Operation = { operation: 'full' }) =>
    void call(commands.process, scope, operation).catch(() => undefined)

  const closeProject = () => void call(commands.closeProject).catch(() => undefined)

  // A promise chain rather than try/finally, which React Compiler cannot compile.
  const clearProject = async () => {
    if (clearingProject || pages.length === 0) return
    setClearingProject(true)
    await call(
      commands.deletePages,
      pages.map((page) => page.id),
    )
      .then(() => {
        selectPages([])
        selectLayers([])
        return refresh(projectKey, pagesKey, pageKey)
      })
      .then(() => setClearProjectOpen(false))
      .finally(() => setClearingProject(false))
  }

  return (
    <>
      <header
        data-tauri-drag-region='deep'
        className='relative flex h-10 shrink-0 items-center bg-[var(--surface-titlebar)] text-[12px]'
      >
        {macOS ? (
          <div className='w-[84px] shrink-0' />
        ) : (
          <div className='relative z-10 flex h-full w-10 shrink-0 items-center justify-center rounded-br-lg'>
            <Image
              className='pointer-events-none'
              src='/icon.png'
              alt='Koharu'
              width={17}
              height={17}
              draggable={false}
              priority
            />
          </div>
        )}
        <Menubar className='relative z-10 h-full shrink-0 gap-0 border-0 bg-transparent p-0 shadow-none'>
          <MenubarMenu>
            <MenubarTrigger>{t('menu.file')}</MenubarTrigger>
            <MenubarContent>
              <MenubarSub>
                <MenubarSubTrigger
                  disabled={!project || importing}
                  aria-busy={importing}
                  className='min-h-8 gap-1.5 px-2 py-1 text-xs'
                >
                  {importing && <LoaderCircle className='animate-spin' aria-hidden='true' />}
                  {importing ? t('navigator.importing') : t('menu.import')}
                </MenubarSubTrigger>
                <MenubarSubContent className='min-w-40 p-1'>
                  <MenubarItem disabled={importing} onClick={() => importPages('files')}>
                    <FilePlus2 />
                    {t('navigator.importFiles')}
                  </MenubarItem>
                  <MenubarItem disabled={importing} onClick={() => importPages('folder')}>
                    <FolderOpen />
                    {t('navigator.importFolder')}
                  </MenubarItem>
                </MenubarSubContent>
              </MenubarSub>
              <MenubarSub>
                <MenubarSubTrigger
                  disabled={!project || pages.length === 0 || exporting}
                  aria-busy={exporting}
                  className='min-h-8 gap-1.5 px-2 py-1 text-xs'
                >
                  {exporting && <LoaderCircle className='animate-spin' aria-hidden='true' />}
                  {t('menu.export')}
                </MenubarSubTrigger>
                <MenubarSubContent className='min-w-40 p-1'>
                  {(['png', 'psd', 'cbz'] as const).map((format) => (
                    <MenubarItem
                      key={format}
                      disabled={exporting}
                      onClick={() => setCheckedFormat(format)}
                    >
                      {format.toUpperCase()}…
                    </MenubarItem>
                  ))}
                </MenubarSubContent>
              </MenubarSub>
              <MenubarSeparator />
              <MenubarItem disabled={!project || importing} onClick={() => setVersionsOpen(true)}>
                <History />
                {t('menu.projectVersions')}
              </MenubarItem>
              <MenubarItem
                disabled={!project || pages.length === 0 || importing}
                variant='destructive'
                onClick={() => setClearProjectOpen(true)}
              >
                <Trash2 />
                {t('menu.clearProject')}
              </MenubarItem>
              <MenubarItem disabled={!project} onClick={closeProject}>
                {t('menu.closeProject')}
              </MenubarItem>
              <MenubarSeparator />
              <MenubarItem onClick={() => setSettingsOpen(true)}>
                <Settings />
                {t('menu.settings')}
              </MenubarItem>
            </MenubarContent>
          </MenubarMenu>

          <MenubarMenu>
            <MenubarTrigger>{t('menu.edit')}</MenubarTrigger>
            <MenubarContent>
              <MenubarItem
                disabled={!project?.can_undo}
                onClick={() => void enqueueHistory('undo').catch(() => undefined)}
              >
                {t('menu.undo')}
                <MenubarShortcut>Ctrl+Z</MenubarShortcut>
              </MenubarItem>
              <MenubarItem
                disabled={!project?.can_redo}
                onClick={() => void enqueueHistory('redo').catch(() => undefined)}
              >
                {t('menu.redo')}
                <MenubarShortcut>Ctrl+Y</MenubarShortcut>
              </MenubarItem>
              <MenubarSeparator />
              <MenubarItem
                disabled={!page}
                onClick={() =>
                  selectLayers(page?.layers.filter(selectableLayer).map((layer) => layer.id) ?? [])
                }
              >
                {t('menu.selectAllLayers')}
                <MenubarShortcut>Ctrl+A</MenubarShortcut>
              </MenubarItem>
              <MenubarItem
                disabled={selectedLayers.length === 0}
                variant='destructive'
                onClick={() =>
                  void call(commands.deleteLayers, selectedLayers)
                    .then(() => refresh(projectKey, pagesKey, pageKey))
                    .catch(() => undefined)
                }
              >
                {t('menu.delete')}
                <MenubarShortcut>Del</MenubarShortcut>
              </MenubarItem>
              <MenubarSeparator />
              <MenubarItem
                disabled={duplicableSelection(page, selectedLayers) === null}
                onClick={() => {
                  const layer = duplicableSelection(page, selectedLayers)
                  if (layer) duplicateLayer(layer)
                }}
              >
                {t('menu.duplicateLayer')}
                <MenubarShortcut>Ctrl+J</MenubarShortcut>
              </MenubarItem>
              <MenubarItem
                disabled={mergeableSelection(page, selectedLayers) === null}
                onClick={() => {
                  const layers = mergeableSelection(page, selectedLayers)
                  if (layers) mergeLayers(layers)
                }}
              >
                {selectedLayers.length > 1 ? t('menu.mergeLayers') : t('menu.mergeDown')}
                <MenubarShortcut>Ctrl+E</MenubarShortcut>
              </MenubarItem>
            </MenubarContent>
          </MenubarMenu>

          <MenubarMenu>
            <MenubarTrigger>{t('menu.process')}</MenubarTrigger>
            <MenubarContent>
              <MenubarItem
                disabled={!project || pages.length === 0}
                onClick={() => run({ scope: 'project' })}
              >
                {t('menu.processProject')}
              </MenubarItem>
              <MenubarItem
                disabled={selectedPages.length === 0}
                onClick={() => run({ scope: 'pages', value: selectedPages })}
              >
                {t('menu.processPages')}
              </MenubarItem>
              <MenubarItem
                disabled={selectedLayers.length === 0}
                onClick={() =>
                  run(
                    { scope: 'entities', value: selectedLayers },
                    { operation: 'stages', stages: ['ocr', 'translation'] },
                  )
                }
              >
                {t('menu.processLayers')}
              </MenubarItem>
              <MenubarSeparator />
              {(
                [
                  'enhancement',
                  'detection',
                  'ocr',
                  'markup',
                  'translation',
                  'inpainting',
                ] as Stage[]
              ).map((stage) => (
                <MenubarItem
                  key={stage}
                  disabled={!project || pages.length === 0}
                  onClick={() => run({ scope: 'project' }, { operation: 'through', stage })}
                >
                  {t('menu.runPhase', {
                    phase: t(`phase.${stage}`),
                  })}
                </MenubarItem>
              ))}
              <MenubarSeparator />
              <MenubarItem onClick={() => setBatchOpen(true)}>{t('menu.batch')}</MenubarItem>
            </MenubarContent>
          </MenubarMenu>

          <MenubarMenu>
            <MenubarTrigger>{t('menu.view')}</MenubarTrigger>
            <MenubarContent>
              <MenubarItem disabled={!page} onClick={requestCanvasFit}>
                {t('menu.fit')}
                <MenubarShortcut>Alt+F</MenubarShortcut>
              </MenubarItem>
              <MenubarCheckboxItem
                checked={panels.pages}
                onCheckedChange={(open) => setPanelOpen('pages', open)}
              >
                {t('menu.pagesPanel')}
              </MenubarCheckboxItem>
              <MenubarCheckboxItem
                checked={panels.inspector}
                onCheckedChange={(open) => setPanelOpen('inspector', open)}
              >
                {t('menu.inspectorPanel')}
              </MenubarCheckboxItem>
            </MenubarContent>
          </MenubarMenu>

          <MenubarMenu>
            <MenubarTrigger>{t('menu.help')}</MenubarTrigger>
            <MenubarContent>
              <MenubarItem
                onClick={() => void openUrl('https://discord.gg/mHvHkxGnUY').catch(() => undefined)}
              >
                {t('menu.discord')}
              </MenubarItem>
              <MenubarItem
                onClick={() =>
                  void openUrl('https://github.com/koharu-rs/koharu').catch(() => undefined)
                }
              >
                {t('menu.github')}
              </MenubarItem>
              <MenubarSeparator />
              <MenubarItem onClick={() => setAboutOpen(true)}>{t('menu.about')}</MenubarItem>
            </MenubarContent>
          </MenubarMenu>
        </Menubar>

        <div className='min-w-0 flex-1' />
        <div className='pointer-events-none absolute inset-y-0 left-1/2 flex max-w-[40vw] min-w-16 -translate-x-1/2 items-center justify-center px-3 text-[11px] text-muted-foreground select-none'>
          {project ? (
            <span className='truncate'>
              <span className='font-medium text-foreground'>{project.name}</span>
              {page && (
                <>
                  <span className='mx-2'>/</span>
                  <span>{page.label}</span>
                </>
              )}
            </span>
          ) : (
            <span>Koharu</span>
          )}
        </div>

        {!macOS && <WindowControls />}
      </header>
      <AboutDialog open={aboutOpen} onOpenChange={setAboutOpen} />
      <ChapterCheckDialog
        format={checkedFormat}
        onClose={() => setCheckedFormat(null)}
        onExport={exportProject}
      />
      <ProjectVersionsDialog open={versionsOpen} onOpenChange={setVersionsOpen} />
      <DeleteAllPagesDialog
        open={clearProjectOpen}
        count={pages.length}
        busy={clearingProject}
        onOpenChange={(open) => {
          if (!clearingProject) setClearProjectOpen(open)
        }}
        onConfirm={() => void clearProject().catch(() => undefined)}
      />
    </>
  )
}

function MenubarTrigger({ className, ...props }: ComponentProps<typeof UiMenubarTrigger>) {
  return (
    <UiMenubarTrigger
      className={cn(
        'h-6 px-1.5 text-[11px] text-muted-foreground transition-colors hover:bg-primary/10 hover:text-muted-foreground aria-expanded:bg-primary/10 aria-expanded:text-muted-foreground',
        className,
      )}
      {...props}
    />
  )
}

function MenubarContent({ className, ...props }: ComponentProps<typeof UiMenubarContent>) {
  return <UiMenubarContent className={cn('min-w-44 p-1', className)} {...props} />
}

function MenubarItem({ className, ...props }: ComponentProps<typeof UiMenubarItem>) {
  return (
    <UiMenubarItem
      className={cn(
        "min-h-8 gap-1.5 px-2 py-1 text-xs [&_svg:not([class*='size-'])]:size-3.5",
        className,
      )}
      {...props}
    />
  )
}

function MenubarShortcut({ className, ...props }: ComponentProps<typeof UiMenubarShortcut>) {
  return <UiMenubarShortcut className={cn('text-[11px]', className)} {...props} />
}

function MenubarSeparator({ className, ...props }: ComponentProps<typeof UiMenubarSeparator>) {
  return <UiMenubarSeparator className={cn('my-0.5', className)} {...props} />
}
