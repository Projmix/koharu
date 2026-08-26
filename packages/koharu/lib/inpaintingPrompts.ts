export const inpaintingPromptPresets = [
  {
    id: 'cleanup-all',
    prompt:
      'Remove all visible text from the image. Do not leave any text behind. This includes dialogue, captions, labels, credits, signatures, logos, watermarks, re-upload notices, and especially stylized onomatopoeia and sound-effect lettering, even when it is rotated, distorted, hand-drawn, outlined, partly occluded, or outside speech bubbles. Preserve every speech bubble and caption box exactly as drawn: do not remove, redraw, reshape, move, resize, recolor, or change their outlines, tails, fills, or textures. Only after preserving those containers, reconstruct the artwork and background hidden behind the removed text. Do not alter characters, panel borders, or any non-text artwork.',
  },
  {
    id: 'cleanup-text',
    prompt:
      'Remove all visible dialogue, captions, labels, and especially stylized onomatopoeia and sound-effect lettering, even when it is rotated, distorted, hand-drawn, outlined, partly occluded, or outside speech bubbles. Do not leave any of that text behind. Preserve every speech bubble and caption box exactly as drawn: do not remove, redraw, reshape, move, resize, recolor, or change their outlines, tails, fills, or textures. Only after preserving those containers, reconstruct the artwork and background hidden behind the removed text. Do not alter characters, panel borders, or any non-text artwork.',
  },
  {
    id: 'translate-russian',
    prompt:
      "Replace every visible piece of text in the image with a natural, stylistically faithful Russian translation. Translate dialogue, captions, labels, and stylized onomatopoeia and sound-effect lettering. Match the original lettering's placement, scale, orientation, color, outline, emphasis, and visual style while keeping it legible in Russian. Keep each translation in the same speech bubble, caption box, or artwork position as its source. Preserve every speech bubble and caption box exactly as drawn, including their outlines, tails, fills, and textures. Do not alter characters, artwork, panel borders, composition, or anything except the lettering being replaced. Do not leave the original-language text behind.",
  },
] as const

export type InpaintingPromptPresetId = (typeof inpaintingPromptPresets)[number]['id']
export type InpaintingPromptSelection = InpaintingPromptPresetId | 'custom'

export function identifyInpaintingPrompt(prompt: string): InpaintingPromptSelection {
  return inpaintingPromptPresets.find((preset) => preset.prompt === prompt)?.id ?? 'custom'
}

export function inpaintingPromptForPreset(id: string): string | null {
  return inpaintingPromptPresets.find((preset) => preset.id === id)?.prompt ?? null
}
