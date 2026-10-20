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
   **Done:** the fit area stops at the detected panel holding at least
   half of it. On the chapter's 78 free-text blocks at automatic size,
   text crossing its panel by more than 2 px fell from 12 to 5, with no
   overflow and the median size against the owner's unchanged (1.00). The
   five left are rotated regions, which keep their own area.
3. Remaining box errors. Of the 78 owner blocks outside balloons, 25 have
   no matching fresh detection box (all 140 balloon blocks match). Sorted
   by hand on 2026-10-20: 12 are found, but the owner moved or split the
   block away from the lettering; 4 (p4) have no text in the source image;
   9 are real misses, all scored by the model just below its thresholds
   (text 0.25, sound effects 0.2): small hand-drawn sound effects on
   screentone (ハァ, ぐ, ふきふき, グサッ; 0.14 to 0.23), white letters in
   black shapes on p22 (0.11 to 0.17), an outlined ポ (0.09) and a sign
   (事務所, 0.23). Lowering both thresholds to 0.2/0.14 finds about 10 of
   them but adds about 14 boxes on hair, speed lines and screentone.
   **Done:** a text candidate down to 0.1 is kept when it is light
   lettering inside a closed dark balloon. On the chapter this added six
   boxes, all real text (p22 ヌチュッ, ヌチ, a whole チュコ, the cover
   lines), and none on artwork. Open: the hand-drawn sound effects.

Checked and rejected: choosing an outline from background busyness. The
ring standard deviation around non-balloon blocks does not separate the
owner's outlined blocks from the rest.

## OCR

1. **Measure the cross-check mode on the chapter.** The test copy was read
   before the mode existed, so its effect on translation is unmeasured.
   **Done:** the owner's chapter was re-read (23 pages, 218 regions,
   $0.30). Of 100 real differences from the old text, the new reading is
   right in 73, the old in 3, both are equal in 8, and in 16 the OCR
   region does not cover the whole text. Two of the three regressions are
   disagreements whose API request failed, so Hayai's draft stayed
   (item 4).
2. **Hayai repetition loops** («かかかか…»). **Done:** a run of one to three
   characters repeated 8 times is never an agreement and goes to the API.
3. **Local confidence:** today it is only agreement (1 or 0), so two
   readers agreeing on the same mistake pass silently. The local decoders
   can expose token probabilities without an extra pass; send
   low-confidence agreements to the API too. **Measured and rejected:**
   2 of 164 agreements are wrong on the 240 crops and none on the chapter
   (regions that cover the text). No threshold improves the result: a
   Manga OCR minimum below 0.5 sends 4.2 points more regions to the API
   and raises handwritten CER from 2.37% to 2.56%. The confident mistake
   (コネが read as これが, every token above 0.97) is out of reach. No
   code landed.
4. **Flag disputed text** (API failed, or disagreed with both local
   readings) in the layer list, like fit warnings. Two of the chapter's
   three regressions are this case: the readers disagreed, the API
   request failed and Hayai's draft stayed silently.

## Scene markup

1. **Dense pages** (p19) fail the output schema. **Done:** a page with at
   least 8 texts and an invalid answer is retried in two halves; the second
   half sees the first half's answers as earlier context.
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
   **Tried 2026-10-20 and reverted:** one general prompt line (render a
   mimetic word by what it depicts, never a stock kiss sound) fixed くに
   and ヌチ but left クチ as «Чмок» and turned two «Вздрог» into the noun
   «Дрожь». Half the lines change between unchanged runs, so the effect is
   not separable from noise. Open: a short glossary with examples
   (クチ/ヌチ → «Хлюп», くに → «Мнёт») checked over several runs.
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
   **Done:** text whose fill and outline both stay below a 1.5 contrast
   ratio with the image layers under it gets the fit warning in the layer
   list and on the canvas. On the automatic chapter it flags 9 of 227
   blocks: the three p22 «Чмок» (1.04), four p4 notes set black on the
   black page, and black «Ух» (p7) and «А» (p24) on dark art. The owner's
   own chapter flags only the last two.
2. **Six small balloons** below 0.8x the owner's size, where the owner
   lets text run past the balloon edge. Consider allowing a small overrun
   in narrow balloons when the size would otherwise drop sharply.
3. **Second sound-effect font** once markup gives soft vs hard.
4. **Final chapter check before export:** empty translations, Japanese
   left in Russian text, fit warnings, no-contrast text, missed text. All
   computed locally; export is not blocked.
   **Done:** choosing an export format lists blocks with an empty
   translation, Japanese left in the translation, or a fit warning
   (including low contrast). A row opens its page and layer; a clean
   chapter exports at once. Open: missed text, which depends on detection
   item 1.

## Cross-cutting

1. **Merge `claude/project-thread-lbw4bo` into `main`.** On 2026-10-20 it
   is 102 commits ahead; everything done since 2026-10-18 lives
   only there.
2. **One evaluation command** for the chapter comparison that produced
   these numbers (pages stay out of the repository). It is a new command,
   so it needs an issue first.
