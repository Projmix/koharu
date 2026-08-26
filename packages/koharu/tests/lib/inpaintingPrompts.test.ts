import { describe, expect, it } from 'vitest'

import {
  identifyInpaintingPrompt,
  inpaintingPromptForPreset,
  inpaintingPromptPresets,
} from '@/lib/inpaintingPrompts'

describe('inpainting prompt presets', () => {
  it('identifies every preset and treats edited prompts as custom', () => {
    for (const preset of inpaintingPromptPresets) {
      expect(identifyInpaintingPrompt(preset.prompt)).toBe(preset.id)
      expect(inpaintingPromptForPreset(preset.id)).toBe(preset.prompt)
    }
    expect(identifyInpaintingPrompt('My own restoration instructions')).toBe('custom')
  })

  it('keeps the text-only cleanup preset free of watermark instructions', () => {
    const prompt = inpaintingPromptForPreset('cleanup-text')!

    expect(prompt.toLowerCase()).not.toContain('watermark')
    expect(prompt).toContain('sound-effect lettering')
    expect(prompt).toContain('Preserve every speech bubble')
  })

  it('asks the image editor to replace lettering with Russian text', () => {
    const prompt = inpaintingPromptForPreset('translate-russian')!

    expect(prompt).toContain('Russian translation')
    expect(prompt).toContain('stylistically faithful')
    expect(prompt).toContain('Do not leave the original-language text behind')
  })
})
