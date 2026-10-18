import type {
  LanguageChoice,
  Model,
  ModelSelection,
  Provider,
  ProviderPreference,
  TranslationConfig,
  TranslationProfile,
} from '@koharu/bridge/protocol'

export function providerName(entries: ProviderPreference[], provider: Provider): string {
  return entries.find((entry) => entry.config.provider === provider)?.name ?? provider
}

export function modelKey(model: Model | ModelSelection): string {
  return `${model.provider}:${model.model ?? ''}`
}

export function modelSelection(model: Model, quantization: string | null): ModelSelection {
  return {
    provider: model.provider,
    model: model.model,
    quantization,
    vision: model.vision,
    reasoning: model.reasoning,
    reasoning_required: model.reasoning_required,
  }
}

export function translationProfileSelection(
  profile: TranslationProfile,
  model: ModelSelection,
): TranslationProfile {
  return {
    ...profile,
    model,
    generation: model.reasoning_required
      ? { ...profile.generation, reasoning: true }
      : profile.generation,
  }
}

export function reconcileTranslationProfiles(
  translation: TranslationConfig,
  models: readonly Model[],
): TranslationConfig {
  let next = translation
  for (const profileName of ['page', 'chapter'] as const) {
    const profile = translation[profileName]
    const model = models.find((candidate) => modelKey(candidate) === modelKey(profile.model))
    if (!model) continue
    const reconciled = translationProfileSelection(
      profile,
      modelSelection(model, profile.model.quantization ?? null),
    )
    if (JSON.stringify(reconciled) === JSON.stringify(profile)) continue
    if (next === translation) next = { ...translation }
    next[profileName] = reconciled
  }
  return next
}

export function orderedLanguageChoices(
  languages: readonly LanguageChoice[],
): Array<{ tag: string; name: string }> {
  return languages
    .map((language) => ({ tag: language.tag, name: language.name }))
    .sort((left, right) =>
      left.name.localeCompare(right.name, undefined, { numeric: true, sensitivity: 'base' }),
    )
}
