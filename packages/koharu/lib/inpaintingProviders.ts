import type { InpaintingProvider } from '@koharu/bridge/protocol'

export const inpaintingProviders = [
  'fal',
  'openrouter',
  'replicate',
] as const satisfies readonly InpaintingProvider[]

const names: Record<InpaintingProvider, string> = {
  fal: 'Fal.ai',
  openrouter: 'OpenRouter',
  replicate: 'Replicate',
}

export function inpaintingProviderName(provider: InpaintingProvider): string {
  return names[provider]
}
