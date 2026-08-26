'use client'

import { ChevronDown, Eraser, FileText, Search } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'

import { ImageModelPicker } from '@/components/controls/ImageModelPicker'
import { ModelPicker } from '@/components/controls/ModelPicker'
import { GenerationPreferences } from '@/components/preferences/GenerationPreferences'
import {
  defaultModel,
  localOcrModels,
  localOcrNames,
  modelNames,
  modelOptions,
  replaceApiInpainting,
  replaceManualInpainting,
  replaceStage,
  selectApiInpainting,
  stageModel,
  type ModelName,
  type ModelStage,
  type PipelineModel,
} from '@/components/preferences/models'
import {
  NumberField,
  PreferencePage,
  PreferenceRow,
  PreferenceSection,
  TextField,
} from '@/components/preferences/PreferenceFields'
import {
  identifyInpaintingPrompt,
  inpaintingPromptForPreset,
  inpaintingPromptPresets,
} from '@/lib/inpaintingPrompts'
import { modelKey, modelSelection, providerName } from '@/lib/translation'
import type {
  InpaintingModelChoice,
  LocalInpaintingModel,
  Model,
  OcrModel,
  PipelineConfig,
  ProviderPreference,
} from '@koharu/bridge/protocol'
import { Badge } from '@koharu/ui/components/badge'
import { Popover, PopoverContent, PopoverTrigger } from '@koharu/ui/components/popover'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'
import { Textarea } from '@koharu/ui/components/textarea'

const stages = [['detection', Search]] as const satisfies ReadonlyArray<
  readonly [ModelStage, typeof Search]
>

export function PipelinePreferences({
  value,
  modelChoices,
  inpaintingModels,
  providers,
  onChange,
}: {
  value: PipelineConfig
  modelChoices: Model[]
  inpaintingModels: InpaintingModelChoice[]
  providers: ProviderPreference[]
  onChange: (value: PipelineConfig) => void
}) {
  const { t } = useTranslation()
  return (
    <PreferencePage
      title={t('settings.pipeline.title')}
      description={t('settings.pipeline.description')}
    >
      <PreferenceSection title={t('settings.pipeline.processing')}>
        {stages.map(([stage, Icon]) => {
          const model = stageModel(value, stage)
          const title = t(`settings.pipeline.stages.${stage}.title`)
          return (
            <PreferenceRow
              key={stage}
              title={title}
              description={t(`settings.pipeline.stages.${stage}.description`)}
              align='start'
            >
              <div className='grid gap-3'>
                <div className='flex items-center gap-2'>
                  <Icon className='size-3.5 shrink-0 text-muted-foreground' />
                  <Select
                    value={model.model}
                    items={Object.fromEntries(
                      modelOptions[stage].map((name) => [name, modelNames[name]]),
                    )}
                    onValueChange={(name) => {
                      if (name) {
                        const saved = value.processor?.[name as keyof typeof value.processor]
                        const next = saved
                          ? ({ model: name, ...saved } as PipelineModel)
                          : defaultModel(name as ModelName)
                        onChange(replaceStage(value, stage, next))
                      }
                    }}
                  >
                    <SelectTrigger
                      aria-label={t('settings.pipeline.modelLabel', { stage: title })}
                      className='h-8 min-w-0 flex-1 text-[11px]'
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
                <ModelOptions
                  model={model}
                  onChange={(next) => onChange(replaceStage(value, stage, next))}
                />
              </div>
            </PreferenceRow>
          )
        })}
      </PreferenceSection>
      <OcrPreferences
        value={value}
        modelChoices={modelChoices}
        providers={providers}
        onChange={onChange}
      />
      <InpaintingPreferences value={value} models={inpaintingModels} onChange={onChange} />
    </PreferencePage>
  )
}

function InpaintingPreferences({
  value,
  models,
  onChange,
}: {
  value: PipelineConfig
  models: InpaintingModelChoice[]
  onChange: (value: PipelineConfig) => void
}) {
  const { t } = useTranslation()
  const [modelOpen, setModelOpen] = useState(false)
  const inpainting = value.inpainting
  const local = stageModel(value, 'inpainting')
  const manualProfile =
    value.processor?.[inpainting.manual_model.model as keyof typeof value.processor]
  const manual = (
    manualProfile
      ? { model: inpainting.manual_model.model, ...manualProfile }
      : inpainting.manual_model
  ) as PipelineModel
  const apiModels = models.filter((model) => model.provider === inpainting.api.provider)
  const selected = apiModels.find((model) => model.model === inpainting.api.model) ?? null
  const current: InpaintingModelChoice = selected ?? {
    provider: inpainting.api.provider,
    model: inpainting.api.model,
    name: inpainting.api.model,
  }
  const choices = selected ? apiModels : [current, ...apiModels]
  const promptPreset = identifyInpaintingPrompt(inpainting.api.prompt)
  const updateMethod = (method: typeof inpainting.method) =>
    onChange({ ...value, inpainting: { ...inpainting, method } })

  return (
    <>
      <PreferenceSection
        title={t('settings.pipeline.stages.inpainting.title')}
        description={t('settings.pipeline.stages.inpainting.description')}
      >
        <PreferenceRow title={t('settings.pipeline.inpainting.method')} align='start'>
          <div className='grid gap-3'>
            <div className='flex items-center gap-2'>
              <Eraser className='size-3.5 shrink-0 text-muted-foreground' />
              <Select
                value={inpainting.method}
                onValueChange={(method) =>
                  method && updateMethod(method as typeof inpainting.method)
                }
              >
                <SelectTrigger
                  aria-label={t('settings.pipeline.inpainting.method')}
                  className='h-8 min-w-0 flex-1 text-[11px]'
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value='local'>{t('settings.pipeline.ocr.local')}</SelectItem>
                  <SelectItem value='api'>{t('settings.pipeline.ocr.api')}</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {inpainting.method === 'local' ? (
              <>
                <Select
                  value={local.model}
                  items={Object.fromEntries(
                    modelOptions.inpainting.map((name) => [name, modelNames[name]]),
                  )}
                  onValueChange={(name) => {
                    if (!name) return
                    const saved = value.processor?.[name as keyof typeof value.processor]
                    const next = saved
                      ? ({ model: name, ...saved } as PipelineModel)
                      : defaultModel(name as ModelName)
                    onChange(replaceStage(value, 'inpainting', next))
                  }}
                >
                  <SelectTrigger
                    aria-label={t('settings.pipeline.inpainting.localModel')}
                    className='h-8 text-[11px]'
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {modelOptions.inpainting.map((name) => (
                      <SelectItem key={name} value={name}>
                        {modelNames[name]}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <ModelOptions
                  model={local}
                  onChange={(next) => onChange(replaceStage(value, 'inpainting', next))}
                />
              </>
            ) : (
              <>
                <Select
                  value={inpainting.api.provider}
                  onValueChange={(provider) => {
                    if (!provider) return
                    const next = models.find((model) => model.provider === provider)
                    if (next) onChange(selectApiInpainting(value, next.provider, next.model))
                  }}
                >
                  <SelectTrigger
                    aria-label={t('settings.pipeline.inpainting.provider')}
                    className='h-8 text-[11px]'
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
                    className='flex h-9 w-full min-w-0 items-center justify-between gap-2 rounded-lg border border-input bg-transparent px-2.5 text-[11px] transition-colors outline-none hover:bg-foreground/[0.03] focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50'
                  >
                    <span className='min-w-0 truncate'>{current.name}</span>
                    <ChevronDown className='size-3.5 shrink-0 text-muted-foreground' />
                  </PopoverTrigger>
                  <PopoverContent
                    align='start'
                    sideOffset={4}
                    className='w-(--anchor-width) min-w-64 gap-0 overflow-hidden rounded-xl border border-border/50 p-1 shadow-sm ring-0'
                  >
                    <ImageModelPicker
                      value={inpainting.api}
                      models={choices}
                      onSelect={(model) => {
                        onChange(selectApiInpainting(value, model.provider, model.model))
                        setModelOpen(false)
                      }}
                    />
                  </PopoverContent>
                </Popover>
              </>
            )}
          </div>
        </PreferenceRow>
      </PreferenceSection>
      <PreferenceSection title={t('tools.remove')}>
        <PreferenceRow title={t('settings.pipeline.inpainting.localModel')} align='start'>
          <div className='grid gap-3'>
            <Select
              value={manual.model}
              items={Object.fromEntries(
                modelOptions.inpainting.map((name) => [name, modelNames[name]]),
              )}
              onValueChange={(name) => {
                if (!name) return
                const saved = value.processor?.[name as keyof typeof value.processor]
                const next = saved
                  ? ({ model: name, ...saved } as PipelineModel)
                  : defaultModel(name as ModelName)
                onChange(replaceManualInpainting(value, next as LocalInpaintingModel))
              }}
            >
              <SelectTrigger
                aria-label={`${t('tools.remove')}: ${t('settings.pipeline.inpainting.localModel')}`}
                className='h-8 text-[11px]'
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {modelOptions.inpainting.map((name) => (
                  <SelectItem key={name} value={name}>
                    {modelNames[name]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <ModelOptions
              model={manual}
              onChange={(next) =>
                onChange(replaceManualInpainting(value, next as LocalInpaintingModel))
              }
            />
          </div>
        </PreferenceRow>
      </PreferenceSection>
      {inpainting.method === 'api' && (
        <PreferenceSection title={t('settings.pipeline.inpainting.options')}>
          <PreferenceRow title={t('settings.pipeline.options.prompt')} align='start'>
            <div className='grid gap-3'>
              <label className='grid gap-1 text-[10px] text-muted-foreground'>
                {t('settings.pipeline.inpainting.promptPreset')}
                <Select
                  value={promptPreset}
                  onValueChange={(preset) => {
                    if (!preset) return
                    const prompt = inpaintingPromptForPreset(preset)
                    if (!prompt) return
                    onChange(
                      replaceApiInpainting(value, {
                        ...inpainting.api,
                        prompt,
                      }),
                    )
                  }}
                >
                  <SelectTrigger
                    aria-label={t('settings.pipeline.inpainting.promptPreset')}
                    className='h-8 text-[11px]'
                  >
                    <SelectValue>
                      {t(`settings.pipeline.inpainting.promptPresets.${promptPreset}`)}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {inpaintingPromptPresets.map((preset) => (
                      <SelectItem key={preset.id} value={preset.id}>
                        {t(`settings.pipeline.inpainting.promptPresets.${preset.id}`)}
                      </SelectItem>
                    ))}
                    {promptPreset === 'custom' && (
                      <SelectItem value='custom'>
                        {t('settings.pipeline.inpainting.promptPresets.custom')}
                      </SelectItem>
                    )}
                  </SelectContent>
                </Select>
              </label>
              <Textarea
                aria-label={t('settings.pipeline.options.prompt')}
                value={inpainting.api.prompt}
                className='field-sizing-fixed max-h-80 min-h-40 resize-y overflow-y-auto text-[12px] leading-5'
                onChange={(event) =>
                  onChange(
                    replaceApiInpainting(value, {
                      ...inpainting.api,
                      prompt: event.currentTarget.value,
                    }),
                  )
                }
              />
              <label className='grid gap-1 text-[10px] text-muted-foreground'>
                {t('settings.pipeline.options.applyMode')}
                <Select
                  value={inpainting.api.apply_mode}
                  onValueChange={(apply_mode) =>
                    apply_mode &&
                    onChange(
                      replaceApiInpainting(value, {
                        ...inpainting.api,
                        apply_mode: apply_mode as typeof inpainting.api.apply_mode,
                      }),
                    )
                  }
                >
                  <SelectTrigger className='h-8 text-[11px]'>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value='full-page'>
                      {t('settings.pipeline.options.fullPage')}
                    </SelectItem>
                    <SelectItem value='mask'>{t('settings.pipeline.options.mask')}</SelectItem>
                  </SelectContent>
                </Select>
              </label>
            </div>
          </PreferenceRow>
        </PreferenceSection>
      )}
    </>
  )
}

function OcrPreferences({
  value,
  modelChoices,
  providers,
  onChange,
}: {
  value: PipelineConfig
  modelChoices: Model[]
  providers: ProviderPreference[]
  onChange: (value: PipelineConfig) => void
}) {
  const { t } = useTranslation()
  const [modelOpen, setModelOpen] = useState(false)
  const ocr = value.ocr
  const apiModels = modelChoices.filter(
    (model) =>
      model.vision &&
      (model.provider === 'openrouter' ||
        model.provider === 'openai' ||
        model.provider === 'openai-compatible'),
  )
  const selected =
    apiModels.find((candidate) => modelKey(candidate) === modelKey(ocr.api.model)) ?? null
  const current: Model = selected ?? {
    ...ocr.api.model,
    model: ocr.api.model.model ?? null,
    name: ocr.api.model.model ?? providerName(providers, ocr.api.model.provider),
    quantizations: [],
    vision: ocr.api.model.vision ?? true,
    reasoning: ocr.api.model.reasoning ?? false,
    reasoning_required: ocr.api.model.reasoning_required ?? false,
  }
  const choices = selected ? apiModels : [current, ...apiModels]
  const updateOcr = (next: PipelineConfig['ocr']) => onChange({ ...value, ocr: next })

  return (
    <>
      <PreferenceSection
        title={t('settings.pipeline.stages.ocr.title')}
        description={t('settings.pipeline.stages.ocr.description')}
      >
        <PreferenceRow title={t('settings.pipeline.ocr.method')} align='start'>
          <div className='grid gap-3'>
            <div className='flex items-center gap-2'>
              <FileText className='size-3.5 shrink-0 text-muted-foreground' />
              <Select
                value={ocr.method}
                onValueChange={(method) =>
                  method && updateOcr({ ...ocr, method: method as typeof ocr.method })
                }
              >
                <SelectTrigger
                  aria-label={t('settings.pipeline.ocr.method')}
                  className='h-8 min-w-0 flex-1 text-[11px]'
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value='local'>{t('settings.pipeline.ocr.local')}</SelectItem>
                  <SelectItem value='api'>{t('settings.pipeline.ocr.api')}</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {ocr.method === 'local' ? (
              <Select
                value={ocr.local_model.model}
                items={localOcrNames}
                onValueChange={(model) =>
                  model &&
                  updateOcr({
                    ...ocr,
                    local_model: { model } as OcrModel,
                  })
                }
              >
                <SelectTrigger
                  aria-label={t('settings.pipeline.ocr.localModel')}
                  className='h-8 text-[11px]'
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
                  className='flex h-9 w-full min-w-0 items-center justify-between gap-2 rounded-lg border border-input bg-transparent px-2.5 text-[11px] transition-colors outline-none hover:bg-foreground/[0.03] focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50'
                >
                  <span className='flex min-w-0 flex-1 items-center gap-2 text-left'>
                    <Badge
                      variant='outline'
                      className='shrink-0 px-1.5 py-0 text-[9px] font-medium'
                    >
                      {providerName(providers, current.provider)}
                    </Badge>
                    <span className='truncate'>{current.name}</span>
                  </span>
                  <ChevronDown className='size-3.5 shrink-0 text-muted-foreground' />
                </PopoverTrigger>
                <PopoverContent
                  align='start'
                  sideOffset={4}
                  className='w-(--anchor-width) min-w-64 gap-0 overflow-hidden rounded-xl border border-border/50 p-1 shadow-sm ring-0'
                >
                  <ModelPicker
                    value={ocr.api.model}
                    models={choices}
                    providers={providers}
                    onBack={() => setModelOpen(false)}
                    onSelect={(model) => {
                      updateOcr({
                        ...ocr,
                        api: { ...ocr.api, model: modelSelection(model) },
                      })
                      setModelOpen(false)
                    }}
                  />
                </PopoverContent>
              </Popover>
            )}
          </div>
        </PreferenceRow>
      </PreferenceSection>
      {ocr.method === 'api' && (
        <>
          <GenerationPreferences
            value={ocr.api.generation}
            showVision={false}
            onChange={(generation) => updateOcr({ ...ocr, api: { ...ocr.api, generation } })}
          />
          <PreferenceSection title={t('settings.pipeline.ocr.instructions')}>
            <PreferenceRow
              title={t('settings.pipeline.ocr.instructions')}
              description={t('settings.pipeline.ocr.instructionsDescription')}
              align='start'
            >
              <Textarea
                aria-label={t('settings.pipeline.ocr.instructions')}
                value={ocr.api.instructions ?? ''}
                className='field-sizing-fixed min-h-24 resize-y overflow-y-auto text-[12px] leading-5'
                placeholder={t('settings.pipeline.ocr.instructionsPlaceholder')}
                onChange={(event) =>
                  updateOcr({
                    ...ocr,
                    api: { ...ocr.api, instructions: event.currentTarget.value || null },
                  })
                }
              />
            </PreferenceRow>
          </PreferenceSection>
        </>
      )}
    </>
  )
}

function ModelOptions({
  model,
  onChange,
}: {
  model: PipelineModel
  onChange: (model: PipelineModel) => void
}) {
  const { t } = useTranslation()
  switch (model.model) {
    case 'koharu-layout-rfdetr-seg-2xl':
      return (
        <div className='grid grid-cols-3 gap-2'>
          <NumberField
            label={t('settings.pipeline.options.textThreshold')}
            value={model.text_threshold ?? null}
            min={0}
            max={1}
            step={0.05}
            onChange={(text_threshold) => onChange({ ...model, text_threshold })}
          />
          <NumberField
            label={t('settings.pipeline.options.bubbleThreshold')}
            value={model.bubble_threshold ?? null}
            min={0}
            max={1}
            step={0.05}
            onChange={(bubble_threshold) => onChange({ ...model, bubble_threshold })}
          />
          <NumberField
            label={t('settings.pipeline.options.panelThreshold')}
            value={model.panel_threshold ?? null}
            min={0}
            max={1}
            step={0.05}
            onChange={(panel_threshold) => onChange({ ...model, panel_threshold })}
          />
        </div>
      )
    case 'flux2-klein':
      return (
        <TextField
          label={t('settings.pipeline.options.prompt')}
          value={model.prompt ?? 'Remove the text and reconstruct the background.'}
          onChange={(prompt) => onChange({ ...model, prompt })}
        />
      )
    case 'rorem-mixed':
      return (
        <div className='grid grid-cols-2 gap-2'>
          <TextField
            label={t('settings.pipeline.options.prompt')}
            value={model.prompt ?? ''}
            onChange={(prompt) => onChange({ ...model, prompt })}
          />
          <TextField
            label={t('settings.pipeline.options.negativePrompt')}
            value={model.negative_prompt ?? ''}
            onChange={(negative_prompt) => onChange({ ...model, negative_prompt })}
          />
        </div>
      )
    case 'lama':
    case 'aot-inpainting':
      return null
  }
}
