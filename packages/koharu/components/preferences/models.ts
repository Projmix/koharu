import type {
  ApiInpaintingConfig,
  DetectionModel,
  InpaintingProvider,
  LocalInpaintingModel,
  OcrModel,
  PipelineConfig,
  Stage,
} from '@koharu/bridge/protocol'

export type PipelineModel = DetectionModel | LocalInpaintingModel
export type ModelStage = Exclude<Stage, 'ocr' | 'translation'>
export type ModelName = PipelineModel['model']

export const localOcrModels = ['paddleocr-vl-1.6', 'manga-ocr', 'baberu-ocr'] as const
export const localOcrNames: Record<OcrModel['model'], string> = {
  'paddleocr-vl-1.6': 'PaddleOCR-VL 1.6',
  'manga-ocr': 'Manga OCR',
  'baberu-ocr': 'Baberu OCR',
}

export const modelOptions = {
  detection: ['koharu-layout-rfdetr-seg-2xl'],
  inpainting: ['lama', 'aot-inpainting', 'flux2-klein', 'rorem-mixed'],
} satisfies Record<ModelStage, ModelName[]>

export const modelNames: Record<ModelName, string> = {
  'koharu-layout-rfdetr-seg-2xl': 'Koharu Layout RF-DETR Seg 2XL',
  lama: 'LaMa',
  'aot-inpainting': 'AOT Inpainting',
  'flux2-klein': 'FLUX.2 Klein',
  'rorem-mixed': 'RORem Mixed',
}

export function defaultModel(model: ModelName): PipelineModel {
  switch (model) {
    case 'koharu-layout-rfdetr-seg-2xl':
      return { model, text_threshold: null, bubble_threshold: null, panel_threshold: null }
    case 'lama':
    case 'aot-inpainting':
      return { model }
    case 'flux2-klein':
      return { model, prompt: 'Remove the text and reconstruct the background.' }
    case 'rorem-mixed':
      return { model }
  }
}

export function stageModel(config: PipelineConfig, stage: ModelStage): PipelineModel {
  const selected = stage === 'detection' ? config.detection : config.inpainting.local_model
  if (selected) {
    const profile = config.processor?.[selected.model as keyof typeof config.processor]
    return profile ? ({ model: selected.model, ...profile } as PipelineModel) : selected
  }
  return defaultModel(modelOptions[stage][0]!)
}

export function replaceStage(
  config: PipelineConfig,
  stage: ModelStage,
  model: PipelineModel,
): PipelineConfig {
  switch (stage) {
    case 'detection':
      return {
        ...config,
        detection: model as DetectionModel,
        processor: {
          ...config.processor,
          'koharu-layout-rfdetr-seg-2xl':
            model.model === 'koharu-layout-rfdetr-seg-2xl'
              ? {
                  text_threshold: model.text_threshold ?? null,
                  bubble_threshold: model.bubble_threshold ?? null,
                  panel_threshold: model.panel_threshold ?? null,
                }
              : (config.processor?.['koharu-layout-rfdetr-seg-2xl'] ?? null),
        },
      }
    case 'inpainting':
      return replaceLocalInpainting(config, 'local_model', model as LocalInpaintingModel)
  }
}

export function replaceManualInpainting(
  config: PipelineConfig,
  model: LocalInpaintingModel,
): PipelineConfig {
  return replaceLocalInpainting(config, 'manual_model', model)
}

function replaceLocalInpainting(
  config: PipelineConfig,
  field: 'local_model' | 'manual_model',
  model: LocalInpaintingModel,
): PipelineConfig {
  return {
    ...config,
    inpainting: { ...config.inpainting, [field]: model },
    processor: {
      ...config.processor,
      ...(model.model === 'flux2-klein'
        ? { 'flux2-klein': { prompt: model.prompt ?? undefined } }
        : {}),
      ...(model.model === 'rorem-mixed'
        ? {
            'rorem-mixed': {
              prompt: model.prompt ?? undefined,
              negative_prompt: model.negative_prompt ?? undefined,
            },
          }
        : {}),
    },
  }
}

export const pinnedFalModels = [
  'microsoft/mai-image-2.5/edit',
  'microsoft/mai-image-2.5-pro/edit',
] as const

export function selectApiInpainting(
  config: PipelineConfig,
  provider: InpaintingProvider,
  model: string,
): PipelineConfig {
  const saved =
    provider === 'fal' && pinnedFalModels.includes(model as (typeof pinnedFalModels)[number])
      ? config.processor?.[model as keyof typeof config.processor]
      : null
  const api: ApiInpaintingConfig = {
    ...config.inpainting.api,
    provider,
    model,
    ...saved,
  }
  return replaceApiInpainting(config, api)
}

export function replaceApiInpainting(
  config: PipelineConfig,
  api: ApiInpaintingConfig,
): PipelineConfig {
  return {
    ...config,
    inpainting: { ...config.inpainting, api },
    processor: {
      ...config.processor,
      ...(api.provider === 'fal' &&
      pinnedFalModels.includes(api.model as (typeof pinnedFalModels)[number])
        ? {
            [api.model]: {
              prompt: api.prompt,
              apply_mode: api.apply_mode,
            },
          }
        : {}),
    },
  }
}
