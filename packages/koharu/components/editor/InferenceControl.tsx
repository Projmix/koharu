'use client'

import {
  Bot,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Download,
  Languages,
  Maximize2,
  Minimize2,
  Paintbrush,
  PanelsTopLeft,
  Play,
  ScanSearch,
  ScanText,
  Settings,
  Sparkles,
  Square,
  Type,
  Upload,
} from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'

import { FontPicker } from '@/components/controls/FontPicker'
import { ImageModelPicker } from '@/components/controls/ImageModelPicker'
import { ModelPicker } from '@/components/controls/ModelPicker'
import { OutputPicker, type OutputDraft } from '@/components/controls/OutputPicker'
import {
  defaultModel,
  localOcrModels,
  localOcrNames,
  modelNames,
  modelOptions,
  replaceStage,
  selectApiInpainting,
  stageModel,
  type ModelName,
  type ModelStage,
  type PipelineModel,
} from '@/components/preferences/models'
import {
  call,
  refreshInpaintingModels,
  refreshTranslationModels,
  savePreferences,
} from '@/lib/backend'
import { useFonts } from '@/lib/queries'
import { pipelineStages, receivePreferences, useKoharuStore, type PipelineScope } from '@/lib/store'
import {
  modelKey,
  modelSelection,
  providerName,
  translationProfileSelection,
} from '@/lib/translation'
import {
  commands,
  type FontFamily,
  type InpaintingConfig,
  type InpaintingModelChoice,
  type Model,
  type ModelSelection,
  type OcrConfig,
  type PipelineConfig,
  type ProviderPreference,
  type Stage,
  type TypesettingConfig,
} from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import { Popover, PopoverContent, PopoverTrigger } from '@koharu/ui/components/popover'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'
import { cn } from '@koharu/ui/lib/utils'

type SelectorView = 'root' | 'model' | 'scope' | 'stages' | 'output'
type SelectorMode = 'compact' | 'expanded'

export function InferenceControl({
  onRun,
  disabled,
}: {
  onRun: (scope: PipelineScope, stages: Stage[]) => void
  disabled: boolean
}) {
  const { t } = useTranslation()
  const scope = useKoharuStore((state) => state.processingScope)
  const stages = useKoharuStore((state) => state.processingStages)
  const setScope = useKoharuStore((state) => state.setProcessingScope)
  const setStages = useKoharuStore((state) => state.setProcessingStages)
  const jobs = useKoharuStore((state) => state.jobs)
  const selectedPages = useKoharuStore((state) => state.selectedPages)
  const running = Object.values(jobs).find((job) => job.state === 'running') ?? null
  const unavailable =
    stages.length === 0 || (scope === 'selected-pages' && selectedPages.length === 0)

  const stop = () => {
    if (!running) return
    void call(commands.stopJob, running.id).catch(() => undefined)
  }

  return (
    <div className='flex w-full max-w-full min-w-0 items-center justify-end gap-1'>
      <RuntimeSelector
        scope={scope}
        stages={stages}
        selectionCount={selectedPages.length}
        running={Boolean(running)}
        onScopeChange={setScope}
        onStagesChange={setStages}
      />

      {scope === 'project' && (
        <>
          <Button
            type='button'
            variant='ghost'
            size='icon-sm'
            className='shrink-0 rounded-lg'
            disabled={disabled || Boolean(running)}
            aria-label={t('inference.exportChapter')}
            title={t('inference.exportChapter')}
            onClick={() => void call(commands.exportChapterTranslation).catch(() => undefined)}
          >
            <Download className='size-3.5' />
          </Button>
          <Button
            type='button'
            variant='ghost'
            size='icon-sm'
            className='shrink-0 rounded-lg'
            disabled={disabled || Boolean(running)}
            aria-label={t('inference.importChapter')}
            title={t('inference.importChapter')}
            onClick={() => void call(commands.importChapterTranslation).catch(() => undefined)}
          >
            <Upload className='size-3.5' />
          </Button>
        </>
      )}

      <Button
        type='button'
        size='sm'
        className={cn(
          'shrink-0 rounded-lg px-2.5 text-[11px] disabled:bg-muted disabled:text-muted-foreground',
          !running && 'bg-primary/80 hover:bg-primary/90',
        )}
        disabled={(disabled || unavailable) && !running}
        aria-label={running ? t('inference.stopProcessing') : t('inference.runProcessing')}
        onClick={running ? stop : () => onRun(scope, stages)}
      >
        {running ? <Square className='size-3 fill-current' /> : <Play className='size-3' />}
        <span>{running ? t('inference.stop') : t('inference.run')}</span>
      </Button>
    </div>
  )
}

function RuntimeSelector({
  scope,
  stages,
  selectionCount,
  running,
  onScopeChange,
  onStagesChange,
}: {
  scope: PipelineScope
  stages: Stage[]
  selectionCount: number
  running: boolean
  onScopeChange: (scope: PipelineScope) => void
  onStagesChange: (stages: Stage[]) => void
}) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [view, setView] = useState<SelectorView>('root')
  const [mode, setMode] = useState<SelectorMode>('expanded')
  const [loadingModels, setLoadingModels] = useState(false)
  const [savingModel, setSavingModel] = useState<string | null>(null)
  const [savingOutput, setSavingOutput] = useState(false)
  const [savingStage, setSavingStage] = useState<string | null>(null)
  const preferences = useKoharuStore((state) => state.preferences)
  const translationModels = useKoharuStore((state) => state.translationModels)
  const inpaintingModels = useKoharuStore((state) => state.inpaintingModels)
  const setSettingsOpen = useKoharuStore((state) => state.setSettingsOpen)
  const fonts = useFonts(mode === 'expanded').data ?? []
  const pipeline = preferences?.pipeline ?? null
  const translation = preferences?.pipeline.translation ?? null
  const profileName = scope === 'project' ? 'chapter' : 'page'
  const profile = translation?.[profileName] ?? null
  const model = profile?.model ?? null
  const providers = preferences?.providers.entries ?? []
  const languages = preferences?.languages ?? []
  const typesetting = preferences?.typesetting ?? null
  const choices = availableModels(model, translationModels, providers)
  const modelLabel =
    choices.find((choice) => model && modelKey(choice) === modelKey(model))?.name ??
    (model ? (model.model ?? t('inference.providerDefault')) : t('inference.noModel'))
  const outputLabel =
    languages.find((language) => language.tag === translation?.target_language)?.name ??
    translation?.target_language ??
    t('inference.notSet')
  const saving = savingModel !== null || savingOutput || savingStage !== null

  const savePipeline = (next: PipelineConfig, key: string) => {
    if (!preferences || saving) return
    setSavingStage(key)
    void savePreferences(next, preferences.providers, preferences.typesetting)
      .then((saved) => receivePreferences(saved))
      .catch(() => undefined)
      .finally(() => setSavingStage(null))
  }

  const chooseStageModel = (stage: ModelStage, name: ModelName) => {
    if (!pipeline) return
    const selected = stageModel(pipeline, stage)
    if (selected.model === name) return
    const saved = pipeline.processor?.[name as keyof typeof pipeline.processor]
    const next = saved ? ({ model: name, ...saved } as PipelineModel) : defaultModel(name)
    savePipeline(replaceStage(pipeline, stage, next), stage)
  }

  const saveOcr = (ocr: OcrConfig) => {
    if (!pipeline) return
    savePipeline({ ...pipeline, ocr }, 'ocr')
  }

  const saveInpainting = (inpainting: InpaintingConfig) => {
    if (!pipeline) return
    savePipeline({ ...pipeline, inpainting }, 'inpainting')
  }

  const chooseInpaintingApi = (provider: InpaintingConfig['api']['provider'], apiModel: string) => {
    if (!pipeline) return
    savePipeline(selectApiInpainting(pipeline, provider, apiModel), 'inpainting')
  }

  const saveTypesetting = (next: TypesettingConfig) => {
    if (!preferences || saving) return
    setSavingStage('typesetting')
    void savePreferences(preferences.pipeline, preferences.providers, next)
      .then((saved) => receivePreferences(saved))
      .catch(() => undefined)
      .finally(() => setSavingStage(null))
  }

  const handleOpenChange = (next: boolean) => {
    setOpen(next)
    setView('root')
    if (!next) return
    setLoadingModels(true)
    void Promise.all([refreshTranslationModels(true), refreshInpaintingModels(true)])
      .catch(() => undefined)
      .finally(() => setLoadingModels(false))
  }

  const chooseModel = (next: Model) => {
    if (!preferences || saving) return
    const nextProfile = translationProfileSelection(
      preferences.pipeline.translation[profileName],
      next,
    )
    if (
      model &&
      modelKey(model) === modelKey(next) &&
      JSON.stringify(preferences.pipeline.translation[profileName]) === JSON.stringify(nextProfile)
    ) {
      setView('root')
      return
    }

    const key = modelKey(next)
    setSavingModel(key)
    const pipeline = {
      ...preferences.pipeline,
      translation: {
        ...preferences.pipeline.translation,
        [profileName]: nextProfile,
      },
    }
    void savePreferences(pipeline, preferences.providers, preferences.typesetting)
      .then((saved) => {
        receivePreferences(saved)
      })
      .catch(() => undefined)
      .finally(() => setSavingModel(null))
  }

  const chooseScope = (next: PipelineScope) => {
    onScopeChange(next)
    setView('root')
  }

  const saveOutput = (draft: OutputDraft) => {
    if (!preferences || !translation || saving) return
    setSavingOutput(true)
    const pipeline = {
      ...preferences.pipeline,
      translation: {
        ...translation,
        target_language: draft.targetLanguage,
        [profileName]: {
          ...translation[profileName],
          instructions: draft.instructions || null,
        },
      },
    }
    void savePreferences(pipeline, preferences.providers, preferences.typesetting)
      .then((saved) => {
        receivePreferences(saved)
      })
      .catch(() => undefined)
      .finally(() => setSavingOutput(false))
  }

  const toggleStage = (stage: Stage) => {
    if (stages.includes(stage)) {
      onStagesChange(stages.filter((candidate) => candidate !== stage))
      return
    }
    onStagesChange(
      pipelineStages.filter((candidate) => candidate === stage || stages.includes(candidate)),
    )
  }

  if (mode === 'expanded') {
    return (
      <ExpandedRuntimeSelector
        scope={scope}
        stages={stages}
        selectionCount={selectionCount}
        running={running}
        pipeline={pipeline}
        translation={translation}
        profileName={profileName}
        model={model}
        modelLabel={modelLabel}
        choices={choices}
        providers={providers}
        translationModels={translationModels}
        inpaintingModels={inpaintingModels}
        typesetting={typesetting}
        fonts={fonts}
        loadingModels={loadingModels}
        savingModel={savingModel}
        saving={saving}
        outputLabel={outputLabel}
        onScopeChange={onScopeChange}
        onStagesChange={onStagesChange}
        onToggleMode={() => setMode('compact')}
        onChooseModel={chooseModel}
        onChooseStageModel={chooseStageModel}
        onSaveOcr={saveOcr}
        onSaveInpainting={saveInpainting}
        onSelectInpaintingApi={chooseInpaintingApi}
        onSaveTypesetting={saveTypesetting}
        onRefreshModels={() => {
          setLoadingModels(true)
          void Promise.all([refreshTranslationModels(true), refreshInpaintingModels(true)])
            .catch(() => undefined)
            .finally(() => setLoadingModels(false))
        }}
      />
    )
  }

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger
        type='button'
        aria-label={t('inference.selector')}
        className={cn(
          'flex h-7 max-w-52 items-center gap-1.5 rounded-lg bg-foreground/[0.05] px-2 text-[10px] text-muted-foreground transition-colors outline-none hover:bg-primary/10 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/25 data-open:bg-primary/10 data-open:text-foreground',
          running && 'text-foreground',
        )}
      >
        {running ? (
          <Sparkles className='size-3 shrink-0 text-primary' />
        ) : (
          <Bot className='size-3 shrink-0 text-primary' />
        )}
        <span className='min-w-0 truncate'>{modelLabel}</span>
        <ChevronDown className='size-3 shrink-0' />
      </PopoverTrigger>

      <PopoverContent
        align='start'
        sideOffset={4}
        className='w-64 min-w-0 gap-0 overflow-hidden rounded-xl border border-border/50 p-1 shadow-sm ring-0'
      >
        {view === 'root' && (
          <div className='grid gap-0.5' aria-label={t('inference.shortcuts')}>
            <SelectorRow
              label={t('inference.model')}
              value={modelLabel}
              onClick={() => setView('model')}
            />
            <SelectorRow
              label={t('inference.scope')}
              value={t(`inference.scopeShort.${scope}`)}
              onClick={() => setView('scope')}
            />
            <SelectorRow
              label={t('inference.stages')}
              value={
                stages.length === 1
                  ? t(`phase.${stages[0]}`)
                  : t('inference.stageCount', { count: stages.length })
              }
              onClick={() => setView('stages')}
            />
            <SelectorRow
              label={t('inference.output')}
              value={outputLabel}
              onClick={() => setView('output')}
            />
            <div className='my-1 border-t border-border/70' />
            <Button
              type='button'
              variant='ghost'
              size='sm'
              className='h-8 justify-start gap-2 rounded-lg px-2 text-[11px] font-normal text-muted-foreground hover:bg-primary/10 hover:text-foreground'
              onClick={() => {
                setOpen(false)
                setSettingsOpen(true)
              }}
            >
              <Settings className='size-3.5' /> {t('menu.settings')}
            </Button>
            <Button
              type='button'
              variant='ghost'
              size='sm'
              className='h-8 justify-start gap-2 rounded-lg px-2 text-[11px] font-normal text-muted-foreground hover:bg-primary/10 hover:text-foreground'
              onClick={() => {
                setOpen(false)
                setMode('expanded')
              }}
            >
              <Maximize2 className='size-3.5' /> {t('inference.expandedMode')}
            </Button>
          </div>
        )}

        {view === 'model' && (
          <ModelPicker
            value={model}
            models={choices}
            providers={providers}
            loading={loadingModels}
            disabled={running}
            busyModel={savingModel}
            onBack={() => setView('root')}
            onSelect={chooseModel}
          />
        )}

        {view === 'scope' && (
          <SelectorPanel title={t('inference.scope')} onBack={() => setView('root')}>
            <SelectorOption
              value='page'
              label={t('inference.currentPage')}
              detail={t('inference.currentPageDescription')}
              selected={scope === 'page'}
              onSelect={chooseScope}
            />
            <SelectorOption
              value='selected-pages'
              label={t('inference.selectedPages')}
              detail={
                selectionCount
                  ? t('inference.selectedCount', { count: selectionCount })
                  : t('inference.selectPagesFirst')
              }
              selected={scope === 'selected-pages'}
              disabled={selectionCount === 0}
              onSelect={chooseScope}
            />
            <SelectorOption
              value='project'
              label={t('inference.entireProject')}
              detail={t('inference.entireProjectDescription')}
              selected={scope === 'project'}
              onSelect={chooseScope}
            />
          </SelectorPanel>
        )}

        {view === 'stages' && (
          <SelectorPanel title={t('inference.pipelineStages')} onBack={() => setView('root')}>
            <SelectorOption
              value='detection'
              label={t('phase.detection')}
              detail={t('phaseDescription.detection')}
              selected={stages.includes('detection')}
              onSelect={toggleStage}
            />
            <SelectorOption
              value='ocr'
              label={t('phase.ocr')}
              detail={t('phaseDescription.ocr')}
              selected={stages.includes('ocr')}
              onSelect={toggleStage}
            />
            <SelectorOption
              value='translation'
              label={t('phase.translation')}
              detail={t('phaseDescription.translation')}
              selected={stages.includes('translation')}
              onSelect={toggleStage}
            />
            <SelectorOption
              value='inpainting'
              label={t('phase.inpainting')}
              detail={t('phaseDescription.inpainting')}
              selected={stages.includes('inpainting')}
              onSelect={toggleStage}
            />
          </SelectorPanel>
        )}

        {view === 'output' && translation && profile && (
          <OutputPicker
            targetLanguage={translation.target_language}
            instructions={profile.instructions}
            languages={languages}
            disabled={running}
            saving={savingOutput}
            onBack={() => setView('root')}
            onChange={saveOutput}
          />
        )}
      </PopoverContent>
    </Popover>
  )
}

function ExpandedRuntimeSelector({
  scope,
  stages,
  selectionCount,
  running,
  pipeline,
  translation,
  profileName,
  model,
  modelLabel,
  choices,
  providers,
  translationModels,
  inpaintingModels,
  typesetting,
  fonts,
  loadingModels,
  savingModel,
  saving,
  outputLabel,
  onScopeChange,
  onStagesChange,
  onToggleMode,
  onChooseModel,
  onChooseStageModel,
  onSaveOcr,
  onSaveInpainting,
  onSelectInpaintingApi,
  onSaveTypesetting,
  onRefreshModels,
}: {
  scope: PipelineScope
  stages: Stage[]
  selectionCount: number
  running: boolean
  pipeline: PipelineConfig | null
  translation: PipelineConfig['translation'] | null
  profileName: 'page' | 'chapter'
  model: ModelSelection | null
  modelLabel: string
  choices: Model[]
  providers: ProviderPreference[]
  translationModels: Model[]
  inpaintingModels: InpaintingModelChoice[]
  typesetting: TypesettingConfig | null
  fonts: FontFamily[]
  loadingModels: boolean
  savingModel: string | null
  saving: boolean
  outputLabel: string
  onScopeChange: (scope: PipelineScope) => void
  onStagesChange: (stages: Stage[]) => void
  onToggleMode: () => void
  onChooseModel: (model: Model) => void
  onChooseStageModel: (stage: ModelStage, model: ModelName) => void
  onSaveOcr: (ocr: OcrConfig) => void
  onSaveInpainting: (inpainting: InpaintingConfig) => void
  onSelectInpaintingApi: (provider: InpaintingConfig['api']['provider'], model: string) => void
  onSaveTypesetting: (typesetting: TypesettingConfig) => void
  onRefreshModels: () => void
}) {
  const { t } = useTranslation()
  const detection = pipeline ? stageModel(pipeline, 'detection') : null
  const inpaintingConfig = pipeline?.inpainting ?? null
  const ocr = pipeline?.ocr ?? null
  const firstFont = typesetting?.font_families?.[0] ?? ''
  const typingLabel = firstFont
    ? `${t('inference.automatic')} · ${firstFont}`
    : t('inference.automatic')
  const ocrLabel = ocr
    ? ocr.method === 'local'
      ? localOcrNames[ocr.local_model.model]
      : apiModelLabel(ocr.api.model, translationModels, providers, t('inference.providerDefault'))
    : t('inference.notSet')
  const inpaintingLabel = inpaintingConfig
    ? inpaintingConfig.method === 'local'
      ? modelNames[inpaintingConfig.local_model.model]
      : (inpaintingModels.find(
          (model) =>
            model.provider === inpaintingConfig.api.provider &&
            model.model === inpaintingConfig.api.model,
        )?.name ?? inpaintingConfig.api.model)
    : t('inference.notSet')

  const toggleStage = (stage: Stage) => {
    if (stages.includes(stage)) {
      onStagesChange(stages.filter((candidate) => candidate !== stage))
      return
    }
    onStagesChange(
      pipelineStages.filter((candidate) => candidate === stage || stages.includes(candidate)),
    )
  }

  return (
    <div
      className='flex max-w-full min-w-0 flex-1 items-center gap-1'
      data-testid='inference-expanded'
      data-mode='expanded'
    >
      <Button
        type='button'
        variant='ghost'
        size='icon-sm'
        aria-label={t('inference.compactMode')}
        title={t('inference.compactMode')}
        className='size-7 shrink-0 rounded-lg text-muted-foreground hover:bg-primary/10 hover:text-foreground'
        onClick={onToggleMode}
      >
        <Minimize2 className='size-3.5' />
      </Button>

      <div className='max-w-full min-w-0 flex-1 overflow-hidden'>
        <div className='grid w-full min-w-0 grid-cols-6 items-center gap-1 pr-1'>
          <ExpandedStageChip
            label={t('inference.scope')}
            shortLabel={t('inference.scope')}
            value={t(`inference.scopeShort.${scope}`)}
            ariaLabel={`${t('inference.scope')}: ${t(`inference.scopeShort.${scope}`)}`}
            icon={<PanelsTopLeft className='size-3.5 shrink-0 text-primary' />}
          >
            <ScopePicker scope={scope} selectionCount={selectionCount} onSelect={onScopeChange} />
          </ExpandedStageChip>

          <ExpandedStageChip
            stage='detection'
            label={t('phase.detection')}
            shortLabel={t('phaseShort.detection')}
            value={detection ? modelNames[detection.model] : t('inference.notSet')}
            active={stages.includes('detection')}
            disabled={running}
            onToggle={() => toggleStage('detection')}
            icon={<ScanSearch className='size-3.5 shrink-0 text-primary' />}
          >
            <PipelineStagePicker
              stage='detection'
              model={detection}
              disabled={running || saving}
              onSelect={(modelName) => onChooseStageModel('detection', modelName)}
            />
          </ExpandedStageChip>

          <ExpandedStageChip
            stage='ocr'
            label={t('phase.ocr')}
            shortLabel={t('phaseShort.ocr')}
            value={ocrLabel}
            active={stages.includes('ocr')}
            disabled={running}
            onToggle={() => toggleStage('ocr')}
            onOpenChange={(open) => open && onRefreshModels()}
            icon={<ScanText className='size-3.5 shrink-0 text-primary' />}
          >
            {ocr && (
              <OcrStagePicker
                value={ocr}
                modelChoices={translationModels}
                providers={providers}
                disabled={running || saving}
                onChange={onSaveOcr}
              />
            )}
          </ExpandedStageChip>

          <ExpandedStageChip
            stage='translation'
            label={t('phase.translation')}
            shortLabel={t('phaseShort.translation')}
            value={`${modelLabel} · ${outputLabel}`}
            active={stages.includes('translation')}
            disabled={running}
            onToggle={() => toggleStage('translation')}
            onOpenChange={(open) => open && onRefreshModels()}
            icon={<Languages className='size-3.5 shrink-0 text-primary' />}
          >
            <TranslationStagePicker
              profileName={profileName}
              model={model}
              modelLabel={modelLabel}
              models={choices}
              providers={providers}
              loading={loadingModels}
              disabled={running || saving}
              busyModel={savingModel}
              onSelect={onChooseModel}
              targetLanguage={translation?.target_language ?? null}
            />
          </ExpandedStageChip>

          <ExpandedStageChip
            stage='inpainting'
            label={t('phase.inpainting')}
            shortLabel={t('phaseShort.inpainting')}
            value={inpaintingLabel}
            active={stages.includes('inpainting')}
            disabled={running}
            onToggle={() => toggleStage('inpainting')}
            onOpenChange={(open) => open && onRefreshModels()}
            icon={<Paintbrush className='size-3.5 shrink-0 text-primary' />}
          >
            {inpaintingConfig && (
              <InpaintingStagePicker
                value={inpaintingConfig}
                models={inpaintingModels}
                disabled={running || saving}
                onChange={onSaveInpainting}
                onSelectLocal={(modelName) => onChooseStageModel('inpainting', modelName)}
                onSelectApi={(provider, model) => {
                  onSelectInpaintingApi(provider, model)
                }}
              />
            )}
          </ExpandedStageChip>

          {/* Typesetting is renderer configuration, not a scheduler stage. */}
          <ExpandedStageChip
            label={t('inference.typing')}
            shortLabel={t('inference.typing')}
            value={typingLabel}
            ariaLabel={`${t('inference.typing')}: ${typingLabel}`}
            icon={<Type className='size-3.5 shrink-0 text-primary' />}
          >
            {typesetting && (
              <TypingStagePicker
                value={typesetting}
                fonts={fonts}
                disabled={saving}
                onChange={onSaveTypesetting}
              />
            )}
          </ExpandedStageChip>
        </div>
      </div>
    </div>
  )
}

function ExpandedStageChip({
  stage,
  label,
  shortLabel,
  value,
  active,
  disabled = false,
  ariaLabel,
  onToggle,
  onOpenChange,
  icon,
  children,
}: {
  stage?: Stage
  label: string
  shortLabel: string
  value: string
  active?: boolean
  disabled?: boolean
  ariaLabel?: string
  onToggle?: () => void
  onOpenChange?: (open: boolean) => void
  icon?: ReactNode
  children: ReactNode
}) {
  return (
    <div
      className={cn(
        'flex h-7 min-w-0 flex-1 items-center overflow-hidden rounded-lg border border-border/70 bg-foreground/[0.04] text-[10px] transition-colors',
        active === false && 'opacity-55',
      )}
      data-stage={stage}
    >
      {onToggle && (
        <Button
          type='button'
          variant='ghost'
          size='icon-xs'
          aria-label={label}
          aria-pressed={active}
          disabled={disabled}
          className='size-7 shrink-0 rounded-none border-r border-border/60 px-1 text-muted-foreground hover:bg-primary/10 hover:text-foreground'
          onClick={onToggle}
        >
          {active ? (
            <Check className='size-3 text-primary' />
          ) : (
            <span className='size-2 rounded-full border border-current' />
          )}
        </Button>
      )}
      <Popover onOpenChange={onOpenChange}>
        <PopoverTrigger
          type='button'
          aria-label={ariaLabel ?? `${label}: ${value}`}
          title={ariaLabel ?? `${label}: ${value}`}
          className='flex h-7 min-w-0 flex-1 items-center gap-1 px-1.5 text-left transition-colors outline-none hover:bg-primary/10 focus-visible:ring-2 focus-visible:ring-ring/30'
        >
          {icon}
          <span
            data-slot='expanded-stage-label'
            title={label}
            className='min-w-0 flex-1 truncate font-medium'
          >
            {shortLabel}
          </span>
          <span
            data-slot='expanded-stage-value'
            title={value}
            className='sr-only max-w-20 min-w-0 flex-1 truncate text-muted-foreground'
          >
            {value}
          </span>
          <ChevronDown className='size-3 shrink-0 text-muted-foreground' />
        </PopoverTrigger>
        <PopoverContent
          align='start'
          sideOffset={4}
          className='w-72 min-w-0 gap-0 overflow-hidden rounded-xl border border-border/50 p-1 shadow-sm ring-0'
        >
          {children}
        </PopoverContent>
      </Popover>
    </div>
  )
}

function ScopePicker({
  scope,
  selectionCount,
  onSelect,
}: {
  scope: PipelineScope
  selectionCount: number
  onSelect: (scope: PipelineScope) => void
}) {
  const { t } = useTranslation()
  return (
    <div className='grid gap-0.5' aria-label={t('inference.scope')}>
      <span className='px-2 py-1 text-[11px] font-medium'>{t('inference.scope')}</span>
      <SelectorOption
        value='page'
        label={t('inference.currentPage')}
        detail={t('inference.currentPageDescription')}
        selected={scope === 'page'}
        onSelect={onSelect}
      />
      <SelectorOption
        value='selected-pages'
        label={t('inference.selectedPages')}
        detail={
          selectionCount
            ? t('inference.selectedCount', { count: selectionCount })
            : t('inference.selectPagesFirst')
        }
        selected={scope === 'selected-pages'}
        disabled={selectionCount === 0}
        onSelect={onSelect}
      />
      <SelectorOption
        value='project'
        label={t('inference.entireProject')}
        detail={t('inference.entireProjectDescription')}
        selected={scope === 'project'}
        onSelect={onSelect}
      />
    </div>
  )
}

function PipelineStagePicker({
  stage,
  model,
  disabled,
  onSelect,
}: {
  stage: ModelStage
  model: PipelineModel | null
  disabled: boolean
  onSelect: (model: ModelName) => void
}) {
  const { t } = useTranslation()
  if (!model) return null
  return (
    <div className='grid gap-2 p-1'>
      <span className='px-1 text-[11px] font-medium'>{t(`phase.${stage}`)}</span>
      <Select
        value={model.model}
        items={Object.fromEntries(modelOptions[stage].map((name) => [name, modelNames[name]]))}
        onValueChange={(name) => name && onSelect(name as ModelName)}
      >
        <SelectTrigger
          aria-label={t('settings.pipeline.modelLabel', { stage: t(`phase.${stage}`) })}
          className='h-8 w-full text-[11px]'
          disabled={disabled}
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {modelOptions[stage].map((name) => (
            <SelectItem key={name} value={name}>
              {modelNames[name]}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

function OcrStagePicker({
  value,
  modelChoices,
  providers,
  disabled,
  onChange,
}: {
  value: OcrConfig
  modelChoices: Model[]
  providers: ProviderPreference[]
  disabled: boolean
  onChange: (value: OcrConfig) => void
}) {
  const { t } = useTranslation()
  const [modelOpen, setModelOpen] = useState(false)
  const apiModels = modelChoices.filter(
    (model) =>
      model.vision &&
      (model.provider === 'openrouter' ||
        model.provider === 'openai' ||
        model.provider === 'openai-compatible'),
  )
  const selected =
    apiModels.find((candidate) => modelKey(candidate) === modelKey(value.api.model)) ?? null
  const current = selected ?? {
    ...value.api.model,
    model: value.api.model.model ?? null,
    name: value.api.model.model ?? providerName(providers, value.api.model.provider),
    quantizations: [],
    vision: value.api.model.vision ?? true,
    reasoning: value.api.model.reasoning ?? false,
    reasoning_required: value.api.model.reasoning_required ?? false,
  }
  const choices = selected ? apiModels : [current, ...apiModels]

  return (
    <div className='grid gap-2 p-1'>
      <span className='px-1 text-[11px] font-medium'>{t('phase.ocr')}</span>
      <Select
        value={value.method}
        onValueChange={(method) =>
          method && onChange({ ...value, method: method as OcrConfig['method'] })
        }
      >
        <SelectTrigger
          aria-label={t('settings.pipeline.ocr.method')}
          className='h-8 w-full text-[11px]'
          disabled={disabled}
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value='local'>{t('settings.pipeline.ocr.local')}</SelectItem>
          <SelectItem value='api'>{t('settings.pipeline.ocr.api')}</SelectItem>
        </SelectContent>
      </Select>
      {value.method === 'local' ? (
        <Select
          value={value.local_model.model}
          onValueChange={(model) =>
            model && onChange({ ...value, local_model: { model } as OcrConfig['local_model'] })
          }
        >
          <SelectTrigger
            aria-label={t('settings.pipeline.ocr.localModel')}
            className='h-8 w-full text-[11px]'
            disabled={disabled}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {localOcrModels.map((model) => (
              <SelectItem key={model} value={model}>
                {localOcrNames[model]}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : (
        <Popover open={modelOpen} onOpenChange={setModelOpen}>
          <PopoverTrigger
            type='button'
            aria-label={t('settings.pipeline.ocr.apiModel')}
            disabled={disabled}
            className='flex h-8 min-w-0 items-center justify-between gap-2 rounded-lg border border-input px-2.5 text-[11px] outline-none hover:bg-foreground/[0.03] focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50'
          >
            <span className='min-w-0 truncate'>{current.name}</span>
            <ChevronDown className='size-3.5 shrink-0 text-muted-foreground' />
          </PopoverTrigger>
          <PopoverContent align='start' sideOffset={4} className='w-64 gap-0 p-1'>
            <ModelPicker
              value={value.api.model}
              models={choices}
              providers={providers}
              disabled={disabled}
              onBack={() => setModelOpen(false)}
              onSelect={(model) => {
                onChange({ ...value, api: { ...value.api, model: modelSelection(model) } })
                setModelOpen(false)
              }}
            />
          </PopoverContent>
        </Popover>
      )}
    </div>
  )
}

function InpaintingStagePicker({
  value,
  models,
  disabled,
  onChange,
  onSelectLocal,
  onSelectApi,
}: {
  value: InpaintingConfig
  models: InpaintingModelChoice[]
  disabled: boolean
  onChange: (value: InpaintingConfig) => void
  onSelectLocal: (model: ModelName) => void
  onSelectApi: (provider: InpaintingConfig['api']['provider'], model: string) => void
}) {
  const { t } = useTranslation()
  const [modelOpen, setModelOpen] = useState(false)
  const apiModels = models.filter((model) => model.provider === value.api.provider)
  const selected = apiModels.find((model) => model.model === value.api.model) ?? null
  const current: InpaintingModelChoice = selected ?? {
    provider: value.api.provider,
    model: value.api.model,
    name: value.api.model,
  }
  const choices = selected ? apiModels : [current, ...apiModels]

  return (
    <div className='grid gap-2 p-1'>
      <span className='px-1 text-[11px] font-medium'>{t('phase.inpainting')}</span>
      <Select
        value={value.method}
        onValueChange={(method) =>
          method && onChange({ ...value, method: method as InpaintingConfig['method'] })
        }
      >
        <SelectTrigger
          aria-label={t('settings.pipeline.inpainting.method')}
          className='h-8 w-full text-[11px]'
          disabled={disabled}
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value='local'>{t('settings.pipeline.ocr.local')}</SelectItem>
          <SelectItem value='api'>{t('settings.pipeline.ocr.api')}</SelectItem>
        </SelectContent>
      </Select>
      {value.method === 'local' ? (
        <Select
          value={value.local_model.model}
          onValueChange={(model) => model && onSelectLocal(model as ModelName)}
        >
          <SelectTrigger
            aria-label={t('settings.pipeline.inpainting.localModel')}
            className='h-8 w-full text-[11px]'
            disabled={disabled}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {modelOptions.inpainting.map((model) => (
              <SelectItem key={model} value={model}>
                {modelNames[model]}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : (
        <>
          <Select
            value={value.api.provider}
            onValueChange={(provider) => {
              if (!provider) return
              const next = models.find((model) => model.provider === provider)
              if (next) onSelectApi(next.provider, next.model)
            }}
          >
            <SelectTrigger
              aria-label={t('settings.pipeline.inpainting.provider')}
              className='h-8 w-full text-[11px]'
              disabled={disabled}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value='fal'>Fal.ai</SelectItem>
              <SelectItem value='openrouter'>OpenRouter</SelectItem>
            </SelectContent>
          </Select>
          <Popover open={modelOpen} onOpenChange={setModelOpen}>
            <PopoverTrigger
              type='button'
              aria-label={t('settings.pipeline.inpainting.apiModel')}
              disabled={disabled}
              className='flex h-8 min-w-0 items-center justify-between gap-2 rounded-lg border border-input px-2.5 text-[11px] outline-none hover:bg-foreground/[0.03] focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50'
            >
              <span className='min-w-0 truncate'>{current.name}</span>
              <ChevronDown className='size-3.5 shrink-0 text-muted-foreground' />
            </PopoverTrigger>
            <PopoverContent align='start' sideOffset={4} className='w-64 gap-0 p-1'>
              <ImageModelPicker
                value={value.api}
                models={choices}
                disabled={disabled}
                onSelect={(model) => {
                  onSelectApi(model.provider, model.model)
                  setModelOpen(false)
                }}
              />
            </PopoverContent>
          </Popover>
        </>
      )}
    </div>
  )
}

function TranslationStagePicker({
  profileName,
  model,
  modelLabel,
  models,
  providers,
  loading,
  disabled,
  busyModel,
  onSelect,
  targetLanguage,
}: {
  profileName: 'page' | 'chapter'
  model: ModelSelection | null
  modelLabel: string
  models: Model[]
  providers: ProviderPreference[]
  loading: boolean
  disabled: boolean
  busyModel: string | null
  onSelect: (model: Model) => void
  targetLanguage: string | null
}) {
  const { t } = useTranslation()
  return (
    <div className='grid gap-2 p-1'>
      <div className='px-1'>
        <span className='block text-[11px] font-medium'>{t('phase.translation')}</span>
        <span
          data-slot='current-translation-model'
          className='block truncate text-[10px] font-medium text-foreground'
          title={modelLabel}
        >
          {modelLabel}
        </span>
        <span className='block text-[9px] text-muted-foreground'>
          {profileName === 'chapter' ? t('inference.entireProject') : t('inference.pageProfile')}
          {targetLanguage ? ` · ${targetLanguage}` : ''}
        </span>
      </div>
      <ModelPicker
        value={model}
        models={models}
        providers={providers}
        loading={loading}
        disabled={disabled}
        busyModel={busyModel}
        onSelect={onSelect}
      />
    </div>
  )
}

function TypingStagePicker({
  value,
  fonts,
  disabled,
  onChange,
}: {
  value: TypesettingConfig
  fonts: FontFamily[]
  disabled: boolean
  onChange: (value: TypesettingConfig) => void
}) {
  const { t } = useTranslation()
  const families = value.font_families ?? []
  const current = families[0] ?? ''
  return (
    <div className='grid gap-2 p-1'>
      <div className='px-1'>
        <span className='block text-[11px] font-medium'>{t('inference.typing')}</span>
        <span className='block text-[9px] text-muted-foreground'>
          {t('settings.typesetting.fontFallbackDescription')}
        </span>
      </div>
      <FontPicker
        value={current}
        families={fonts}
        size='sm'
        disabled={disabled || fonts.length === 0}
        ariaLabel={t('settings.typesetting.familyLabel', { index: 1 })}
        placeholder={t('settings.typesetting.addFamily')}
        onChange={(family) =>
          onChange({
            ...value,
            font_families: [family, ...families.slice(1)],
          })
        }
      />
    </div>
  )
}

function apiModelLabel(
  selected: ModelSelection,
  models: Model[],
  providers: ProviderPreference[],
  fallback: string,
): string {
  return (
    models.find((model) => modelKey(model) === modelKey(selected))?.name ??
    selected.model ??
    providerName(providers, selected.provider) ??
    fallback
  )
}

function SelectorRow({
  label,
  value,
  onClick,
}: {
  label: string
  value: string
  onClick: () => void
}) {
  return (
    <Button
      type='button'
      variant='ghost'
      size='sm'
      aria-label={`${label} ${value}`}
      className='h-8 min-w-0 justify-start gap-3 overflow-hidden rounded-lg px-2 text-[11px] font-normal hover:bg-primary/10'
      onClick={onClick}
    >
      <span className='shrink-0'>{label}</span>
      <span className='ml-auto min-w-0 flex-1 truncate text-right text-muted-foreground'>
        {value}
      </span>
      <ChevronRight className='size-3.5 shrink-0 text-muted-foreground' />
    </Button>
  )
}

function SelectorPanel({
  title,
  onBack,
  children,
}: {
  title: string
  onBack: () => void
  children: ReactNode
}) {
  const { t } = useTranslation()
  return (
    <div className='min-w-0 overflow-hidden'>
      <div className='mb-1 flex h-7 items-center border-b border-border/60 px-0.5 pb-1'>
        <Button
          type='button'
          variant='ghost'
          size='icon-xs'
          aria-label={t('common.back')}
          className='rounded-md text-muted-foreground hover:bg-primary/10 hover:text-foreground'
          onClick={onBack}
        >
          <ChevronLeft className='size-3.5' />
        </Button>
        <span className='ml-1 text-[11px] font-medium'>{title}</span>
      </div>
      <div className='grid min-w-0 gap-0.5 overflow-hidden'>{children}</div>
    </div>
  )
}

function SelectorOption<Value extends string>({
  value,
  label,
  detail,
  selected,
  disabled = false,
  onSelect,
}: {
  value: Value
  label: string
  detail: string
  selected: boolean
  disabled?: boolean
  onSelect: (value: Value) => void
}) {
  return (
    <Button
      type='button'
      variant='ghost'
      aria-pressed={selected}
      disabled={disabled}
      className='h-auto min-h-9 justify-start gap-2 rounded-lg px-2 py-1 text-left font-normal hover:bg-primary/10'
      onClick={() => onSelect(value)}
    >
      <span className='min-w-0 flex-1'>
        <span className='block text-[11px]'>{label}</span>
        <span className='block text-[9px] text-muted-foreground'>{detail}</span>
      </span>
      {selected && <Check className='size-3.5 shrink-0 text-primary' />}
    </Button>
  )
}

function availableModels(
  selected: ModelSelection | null,
  models: Model[],
  providers: ProviderPreference[],
): Model[] {
  if (!selected || models.some((model) => modelKey(model) === modelKey(selected))) return models
  return [
    {
      provider: selected.provider,
      model: selected.model ?? null,
      name: selected.model ?? providerName(providers, selected.provider),
      quantizations: [],
      vision: selected.vision ?? false,
      reasoning: selected.reasoning ?? false,
      reasoning_required: selected.reasoning_required ?? false,
    },
    ...models,
  ]
}
