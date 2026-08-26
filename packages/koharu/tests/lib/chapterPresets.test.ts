import { describe, expect, it } from 'vitest'

import { chapterPromptPresets, findChapterPromptPreset } from '@/lib/chapterPresets'

describe('chapter prompt presets', () => {
  it('provides the BNHA Bakugo/Midoriya preset with unsmoothed voice rules', () => {
    const preset = findChapterPromptPreset('bnha_bakugo_midoriya_ru')

    expect(preset).toBeDefined()
    expect(preset?.instructions).toContain('Не смягчай ругань')
    expect(preset?.instructions).toContain('Бакуго')
    expect(preset?.instructions).toContain('Мидория')
    expect(preset?.instructions).toContain('причуда')
    expect(preset?.instructions).toContain('SFX')
  })

  it('keeps presets domain-focused and uniquely addressable', () => {
    expect(chapterPromptPresets.length).toBeGreaterThanOrEqual(3)
    expect(new Set(chapterPromptPresets.map((preset) => preset.id)).size).toBe(
      chapterPromptPresets.length,
    )
    expect(chapterPromptPresets.every((preset) => preset.instructions.trim().length > 200)).toBe(
      true,
    )
  })
})
