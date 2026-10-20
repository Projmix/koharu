# Pipeline improvement plan

A per-stage review on 2026-10-20 after measuring every stage on the owner's
test chapter. Cleaning is reviewed separately later. The owner-facing
version is [improvement-plan.ru.md](improvement-plan.ru.md); measured
results are in [results.ru.md](results.ru.md), models in
[models.md](models.md). Items are ordered by value for the goal of a
chapter that needs no manual edits. Each lands as its own change.

## Detection

1. **Missed text.** Text the detector misses is never read or translated
   (p22 lower-right panel, ヌチュッ/ヌチ/ニチュ). Scene markup already sends
   the whole page to a vision model; ask it to also list visible text
   outside the numbered boxes, flag those places for review, and turn a
   flag into a region on request.
2. **Vertical-text boxes cross panel borders.** The fit area of free text
   follows the Japanese vertical box, which can reach into the next panel
   (p7 «ДА БЛЯТЬ»). Clip the free-text fit area to its panel.
3. Remaining box errors: outlined sound effects missed, false sound
   effects on artwork, merged or split boxes. Measured 2026-10-20: all 140
   balloon blocks have a matching fresh detection box, but 22 of the 78
   blocks outside balloons have none (missed, merged or split): the
   yellow lines on the black p4, SFX written as spaced letters, several
   outlined SFX. They also cause 7 of the 9 blocks the owner outlined and
   automation did not, since nothing is sampled without a box. Classify
   the 22 and fix the most common kind.

Checked and rejected: choosing an outline from background busyness. The
ring standard deviation around non-balloon blocks does not separate the
owner's outlined blocks from the rest.

## OCR

1. **Measure the cross-check mode on the chapter.** The test copy was read
   before the mode existed, so its effect on translation is unmeasured.
2. **Hayai repetition loops** («かかかか…»). **Done:** a run of one to three
   characters repeated 8 times is never an agreement and goes to the API.
3. **Local confidence:** today it is only agreement (1 or 0), so two
   readers agreeing on the same mistake pass silently. The local decoders
   can expose token probabilities without an extra pass; send
   low-confidence agreements to the API too.
4. **Flag disputed text** (API failed, or disagreed with both local
   readings) in the layer list, like fit warnings.

## Scene markup

1. **Dense pages** (p19) fail the output schema: retry such a page in
   halves of its boxes.
2. **Unstable 18+ refusals:** retry a refused page with the next
   configured model instead of leaving it unmarked.
3. **Soft vs hard sound effects** (くにっ/ぬちゅ vs ドン) so typesetting can
   use a second sound-effect font.

## Translation

1. **Characters across chapters:** import the character list and forms
   of address from a previous project.
2. **"Shorten" for fit warnings:** retranslate one flagged region with a
   length limit, leaving the chapter untouched.
3. **Sound-effect glossary** in the translation request (one of the two
   remaining errors was くにっ rendered as «Чмок»).
4. The other translation errors come from OCR, so OCR item 1 is the main
   lever.

## Typesetting

Current match with the owner's hand typesetting on 218 blocks: size
within 20% for 91%, line count 72%, text colour 95%, outline 91%, all four
58%.

1. **Contrast against the cleaned page:** when cleaning paints a dark
   balloon white, white text vanishes (p22 «Чмок»). Flag regions whose
   text colour has no contrast with the cleaned image; the ⇄ button fixes
   one in a click.
2. **Six small balloons** below 0.8x the owner's size, where the owner
   lets text run past the balloon edge. Consider allowing a small overrun
   in narrow balloons when the size would otherwise drop sharply.
3. **Second sound-effect font** once markup gives soft vs hard.
4. **Final chapter check before export:** empty translations, Japanese
   left in Russian text, fit warnings, no-contrast text, missed text. All
   computed locally; export is not blocked.

## Cross-cutting

1. **Merge `claude/project-thread-lbw4bo` into `main`.** On 2026-10-20 it
   is more than 85 commits ahead; everything done since 2026-10-18 lives
   only there.
2. **One evaluation command** for the chapter comparison that produced
   these numbers (pages stay out of the repository). It is a new command,
   so it needs an issue first.
