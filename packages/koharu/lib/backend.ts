'use client'

import { commands } from '@koharu/bridge/protocol'
import type {
  PipelineConfig,
  Preferences,
  ProviderPreferences,
  TypesettingConfig,
} from '@koharu/bridge/protocol'

import {
  receiveError,
  receiveInpaintingModels,
  receivePreferences,
  receiveTranslationModels,
} from './store'

type Command<Args extends unknown[], Result> = (...args: Args) => Promise<Result>

let translationModelsRequest: Promise<void> | null = null
let translationModelsGeneration = 0
let inpaintingModelsRequest: Promise<void> | null = null
let inpaintingModelsGeneration = 0
let preferencesWriteGeneration = 0
let preferencesWriteQueue: Promise<void> = Promise.resolve()

export async function call<Args extends unknown[], Result>(
  command: Command<Args, Result>,
  ...args: Args
): Promise<Result> {
  try {
    return await command(...args)
  } catch (error) {
    throw report(error)
  }
}

export function dispatch<Args extends unknown[], Result>(
  command: Command<Args, Result>,
  ...args: Args
): void {
  void call(command, ...args).catch(() => undefined)
}

export async function refreshPreferences(): Promise<void> {
  const generation = preferencesWriteGeneration
  await preferencesWriteQueue
  const preferences = await call(commands.getPreferences)
  if (generation === preferencesWriteGeneration) receivePreferences(preferences)
}

export function savePreferences(
  pipeline: PipelineConfig,
  providers: ProviderPreferences,
  typesetting: TypesettingConfig,
): Promise<Preferences> {
  preferencesWriteGeneration += 1
  const pending = preferencesWriteQueue
    .catch(() => undefined)
    .then(() => call(commands.savePreferences, pipeline, providers, typesetting))
  preferencesWriteQueue = pending.then(
    () => undefined,
    () => undefined,
  )
  return pending
}

export function refreshTranslationModels(force = false): Promise<void> {
  if (!force && translationModelsRequest) return translationModelsRequest

  const generation = ++translationModelsGeneration
  const request = call(commands.getTranslationModels)
    .then((models) => {
      if (generation === translationModelsGeneration) receiveTranslationModels(models)
    })
    .finally(() => {
      if (translationModelsRequest === request) translationModelsRequest = null
    })
  translationModelsRequest = request
  return request
}

export function refreshInpaintingModels(force = false): Promise<void> {
  if (!force && inpaintingModelsRequest) return inpaintingModelsRequest

  const generation = ++inpaintingModelsGeneration
  const request = call(commands.getInpaintingModels)
    .then((models) => {
      if (generation === inpaintingModelsGeneration) receiveInpaintingModels(models)
    })
    .finally(() => {
      if (inpaintingModelsRequest === request) inpaintingModelsRequest = null
    })
  inpaintingModelsRequest = request
  return request
}

function report(error: unknown): Error {
  const message =
    error instanceof Error
      ? error.message
      : typeof error === 'string'
        ? error
        : 'The native application returned an unknown error.'
  receiveError(message)
  return error instanceof Error ? error : new Error(message)
}
