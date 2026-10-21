# Model registry

Every detector, OCR and cleaning model the fork uses, ships unused, has
measured, or has decided against. The point is to never evaluate the same
model twice: before trying a model, look it up here; after trying it, record
the result. Last updated 2026-10-19.

## Rules

- **Status** is one of:
  - `in use`: selectable in the pipeline today.
  - `shipped, unused`: ported in `koharu-ml` (pinned revision), not called by
    the pipeline.
  - `candidate`: worth measuring; not ported.
  - `rejected`: do not revisit unless the condition in its row changes.
- A model moves to `rejected` only with a reason: a measurement on the
  evaluation set (see [results.ru.md](results.ru.md#как-проверяли)), a
  license that blocks it, or a duplicate of something already in use. Record
  the date and the evidence. "Looked worse once by eye" is not a reason.
- Self-reported numbers from a model card are leads, not evidence. Numbers
  from different benchmarks are never compared with each other.
- Porting a candidate is a model integration: compare on identical inputs and
  open an issue first (`AGENTS.md`).
- Licenses: Koharu is MIT/Apache. Weights under a non-commercial license
  (CC BY-NC*) or code under AGPL can be used only while the fork stays
  non-commercial and the model is downloaded, not bundled; record that in the
  row. Manga109-trained models inherit Manga109's academic-use terms.

## Detection

| Model | Status | Source | Notes |
| --- | --- | --- | --- |
| Koharu Layout RF-DETR Seg 2XL (1152) | in use (only detector) | `mayocream/koharu-layout-rfdetr-seg-2xl-1152` @ `aed55fd` | Text, bubble, panel and onomatopoeia boxes with masks. Global thresholds per label. |
| comic-layout-yolo26s | shipped, unused | `mayocream/comic-layout-yolo26s` @ `90f556d` | Smaller layout detector; candidate fallback for weak hardware. |
| comic-text-and-bubble-detector | shipped, unused | `ogkalu/comic-text-and-bubble-detector` @ `16e8a62` | Text and bubble boxes. |
| comic-text-detector | shipped, unused | `mayocream/comic-text-detector` @ `15ade02` | Classic manga-image-translator detector; the prior project's best local stack started with it. |
| manga-text-segmentation-2025 | shipped, unused | `mayocream/manga-text-segmentation-2025` @ `efd866e` | Pixel text mask. Candidate for tighter cleaning masks (plan item 12). |
| speech bubble YOLO11n / YOLOv8m seg | shipped, unused | `mayocream/manga109-segmentation-bubble`, `mayocream/speech-bubble-segmentation` | Bubble masks only. |
| comic onomatopoeia detector | shipped, unused | `koharu-ml/src/comic_onomatopoeia` | SFX detection and recognition; candidate if SFX recall is poor. |
| PP-DocLayoutV3, PP-OCRv6 det | shipped, unused | PaddlePaddle | Document layout and line detection; not manga-specific. |
| Manga109 panel/balloon/text YOLO26 seg | candidate (low) | [`ShadowB/Manga109-panel-balloon-text-yolov26-segmentation`](https://huggingface.co/ShadowB/Manga109-panel-balloon-text-yolov26-segmentation) | 11.4M params, MIT repo, Manga109 data. Self-reported mask mAP50 0.970 on its own split; no comparison with other detectors. Only worth it as a fast detector. |
| manga-page-element-segmentation (YOLO11m) | rejected 2026-10-18 | [`anonimkaq4/manga-page-element-segmentation`](https://huggingface.co/anonimkaq4/manga-page-element-segmentation) | Classes are character, frame and speech bubble; no text class, so it cannot replace text detection, and bubbles are already covered. Revisit only if character masks become useful. |

## OCR

Local, per region crop.

Measured 2026-10-18 on 240 crops of JMangaBench_Read @ `b3e5765` (180
printed, 60 handwritten, seeded sample; pages newer than Manga109). Local
models ran on CPU only (i5-1235U), debug build. API models ran through
Koharu's own OpenRouter OCR path at 4 requests in flight, before `ffab8a4`
changed the OCR prompt. CER is after the benchmark's RIN normalization;
exact line is printed / handwritten.

| Model | CER printed | CER handwritten | Exact line | Time per crop | $ per 1000 crops |
| --- | --- | --- | --- | --- | --- |
| PaddleOCR-VL-For-Manga (local) | 0.68% | 9.47% | 94 / 58% | 17.7 s | 0 |
| Manga OCR (local) | 1.61% | 14.4% | 87 / 50% | 0.4 s | 0 |
| Hayai OCR v2.1 (local) | 2.00% | 4.73% | 89 / 73% | 0.6 s | 0 |
| PaddleOCR-VL 1.6 GGUF (local, default) | 2.48% | 12.4% | 87 / 53% | 11.7 s | 0 |
| Hayai v2.1 + Gemini 3.7 Flash (`local-api`) | 2.87% | 2.76% | 94 / 87% | 4.7 s | ~2.0 |
| Hayai + Manga OCR, Gemini 3.7 Flash on disagreement (`local-api`, `16aaea7`) | 0.29% | 2.37% | 97 / 90% | ~1 s + API on 32% | ~0.74 |
| Hayai + Gemini 3.7 Flash on every crop, prompt after `693f400` | 0.24% | 1.97% | 97 / 92% | 4.7 s | ~2.05 |
| Gemini 3.7 Flash | 2.73% | 5.13% | 94 / 83% | 3.6 s | 1.8 |
| Gemini 3.5 Flash | 2.29% | 5.33% | 92 / 82% | 3.1 s | 4.3 |
| Claude Haiku 5.5 | 2.29% | 13.4% | 91 / 53% | 1.2 s | 0.04 |
| Claude Sonnet 5.5 | 3.31% | 6.51% | 88 / 68% | 1.9 s | 0.8 |
| GPT-6.1 Sol | 3.89% | 8.48% | 81 / 70% | 6–8 s | 2.0 |
| Qwen3.7-plus | 3.70% | 12.0% | | | 0.10 |
| Qwen3-VL-235B | 4.24% | 9.47% | | | 0.06 |
| Gemini 3.5 Flash-Lite | 5.50% | 7.10% | | 0.9 s | 0.40 |
| Gemini 3.1 Flash-Lite | 6.91% | 6.11% | | | 0.33 |

The hybrid fixed 25 crops of Hayai's draft and broke 8. Keeping the draft when the
API request fails (`693f400`) brings the hybrid to 1.22% / 2.76%, and dropping
furigana brings it to 0.63% / 2.76% (both computed offline from the saved
answers). A Gemini 3.5 Flash arbiter that picks between the draft and the API
reading on the 35 crops where they disagree was rejected: 30/35 correct
picks, but 0.68% / 2.56% after the furigana fix, which is noise. It fails on the
same small kana, punctuation and stray strokes as the reader. The best possible
pick between the two readings would reach 0.49% / 2.17%. The Claude rows and
GPT-6 Luna ran before `e9930e2`. Since then, OpenRouter OCR, like OpenRouter
translation, requires native json_schema output, and OpenRouter has no
endpoint for those models that accepts it (404 on every crop). The Gemini
3.5/3.7 Flash rows ran after it; before it, they failed on every crop
because the JSON came back fenced in Markdown.

| Model | Status | Source | Notes |
| --- | --- | --- | --- |
| PaddleOCR-VL 1.6 (GGUF) | in use (default) | `PaddlePaddle/PaddleOCR-VL-1.6-GGUF` @ `511b096` | General document VLM, multilingual. No manga-specific evidence. |
| Hayai OCR v2.1 | in use | `JustANormalTinkerer/hayai-ocr-v2` @ `4a4ce47` | Japanese. Recommended local OCR in `docs/research/2026-09-12-japanese-ocr.md`. |
| Baberu OCR | in use | `genshiai-daichi/baberu-ocr` @ `d9cc131` | Japanese. |
| Manga OCR | in use | `mayocream/manga-ocr` @ `4380edb` | Japanese baseline; known to invent text on empty crops. |
| PP-OCRv6 rec | shipped, unused | PaddlePaddle | Line recognizer; not manga-specific. |
| PaddleOCR-VL-For-Manga | candidate (high, measured: best printed CER) | `jzhang533/PaddleOCR-VL-For-Manga` ([demo](https://huggingface.co/spaces/jzhang533/paddleocr-vl-for-manga-demo)) | Best CER in [JMangaBench_Read](https://github.com/muscgab/JMangaBench_Read) (2.13% vs Hayai v2.1 2.67%, Manga OCR 5.01%), but about 2.5x slower per crop. Fine-tuned from PaddleOCR-VL (the original 0.9B, not 1.6) and run with the plain `OCR:` prompt in its demo. The pipeline uses a GGUF of 1.6, so trying it needs either the unquantized `paddle_ocr_vl` path with these weights or a GGUF conversion, then a comparison on identical crops. Measured above: its weights load unchanged through the unquantized PaddleOCR-VL 1.6 path (same tensor names, shapes and tokenizer; only the image pixel limits differ). A quantized GGUF is untested. |
| PaddleOCR-VL-1.6 Manga fine-tune | candidate (medium) | [`gigass03/PaddleOCR-VL-1.6-Manga-Finetuned`](https://mixpeek.com/model/gigass03/PaddleOCR-VL-1.6-Manga-Finetuned) | Apache-2.0, Manga109-s and synthetic data, same base as the default. No published metrics. |
| Hayai OCR v2.5 Nova | candidate (low) | JMangaBench_Read | CER 2.63–3.08% depending on input size versus 2.67% for the v2.1 already in use; no clear gain. |
| manga-ocr-nar-small / -distil | candidate (blocked by license) | [`muscgab/manga-ocr-nar`](https://github.com/muscgab/manga-ocr-nar) | Non-autoregressive, 54M/30M params, CER 1.73% / 2.32% on JMangaBench_Read v1.3, very fast on CPU (ONNX int8). Same author as the benchmark. Weights CC BY-NC-SA 4.0: usable only while the fork stays non-commercial. |
| GLM-OCR (0.9B) | candidate (low) | [ranking](https://ofox.ai/blog/best-ai-model-for-ocr-2026/) | Tops a document benchmark (OmniDocBench); no Japanese or manga evidence. |
| EasyOCR, PaddleOCR-JA (classic) | measured earlier | prior project, 5 pages / 43 blocks | CER 0.93 and 0.95 on the owner's set, but Manga OCR scored 1.01 on the same set, so that run does not separate the models. Not ported; re-measure only on the new evaluation set. |

API, through `ocr.method = api` or `local-api`.

| Model | Status | Source | Notes |
| --- | --- | --- | --- |
| Qwen-VL-OCR 2025-11-20 | in use (API default) | DashScope adapter `remote/qwen_ocr.rs` | Owner's choice. $0.07 / $0.16 per million tokens. `multi_lan` covers Japanese. |
| Any vision model on OpenAI, OpenAI-compatible or OpenRouter | in use | `provider.rs:212` | Gemini and Claude are reachable through OpenRouter. |
| Gemini Flash-Lite (2.5 tried earlier; 3.1 / 3.5 now) | candidate (high) | `docs/research/2026-09-12-japanese-ocr.md` | The prior project's best corrector was Gemini 2.5 Flash Lite with crops and a local draft; no CER table survived. Measure 3.1 / 3.5 Flash-Lite through OpenRouter. |
| Qwen3.8-27B, Qwen3.5-397B, Qwen3-VL 235B / 32B | candidate (medium) | `docs/research/api-ocr-models-2026-08.md` | General OCR evidence only. |
| Mistral OCR 3 / 4.1 | candidate (medium) | `docs/research/2026-09-12-japanese-ocr.md` | Whole-page API, $2–4 per 1000 pages; 4.1 returns boxes and confidence. Most useful for the missed-text check (plan item 9), not per crop. |
| GLM-5.3 Flash, GPT-6 Luna, GPT-5.4 mini, Ling-3.0 Flash VL (OpenRouter) | rejected 2026-10-18 | 240-crop run above | CER 10.7–16.2% printed and 14–44% handwritten, worse than every local model. GLM returned no content on 24/240 crops. |
| Qwen3.8 Flash (OpenRouter) | rejected 2026-10-18 | 240-crop run above | 67% CER. It ignores the `{"text"}` shape on 137/240 crops, and at 4 requests in flight it hit rate limits on 184/240. |
| DeepSeek-OCR 2 | rejected 2026-08-24 | `docs/research/api-ocr-models-2026-08.md` | No hosted OpenAI-compatible endpoint; self-hosting needs CUDA/vLLM. Revisit if a hosted API appears. |

## Cleaning

Local.

| Model | Status | Source | Notes |
| --- | --- | --- | --- |
| LaMa (manga weights) | in use (default) | `mayocream/lama-manga` @ `f91c85b` | Tiled at 512 px; flat regions are filled without inference. |
| AOT inpainting | in use | `mayocream/aot-inpainting` @ `cffe234` | |
| FLUX.2 Klein 4B (GGUF) | in use | `unsloth/FLUX.2-klein-4B-GGUF` @ `0084d1d` | Prompted diffusion; slow without a strong GPU. |
| RORem mixed (SDXL inpainting) | in use | `mayocream/RORem-mixed-GGUF` @ `62c75b3` | Object-removal diffusion. |
| FLUX 3 Image Klein | candidate (medium) | [`black-forest-labs/flux-3-image-klein`](https://huggingface.co/unsloth/flux-3-image-klein) (2026-10-13), [GGUF](https://huggingface.co/unsloth/flux-3-image-klein-GGUF) 2.6–11 GB | 5.6B distilled (16 steps) plus a Qwen3-VL-4B text encoder; needs a prompt upsampler. Removal by bounding box, not a hard mask. 25–31 GB peak GPU memory at 1024²; impractical on the owner's CPU-only PC for full pages, crops untested. FLUX Kommunity License (non-commercial; commercial output under $5M revenue). The diffusers path has no NSFW classifier. See [cleaning-models.ru.md](cleaning-models.ru.md). |
| Qwen-Image 2.1 / 2.1 Turbo | candidate (medium) | [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) (2026-09-14), Turbo 2026-10-09 (8 steps), [GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF) 2.5–14 GB | 7B. The only recent model whose card lists local edits by a separate mask, matching the pipeline. Qwen Research License (non-commercial, download only). No official API. Same CPU caveat as FLUX 3 Klein. |
| manga-cleaner | rejected 2026-10-18 | [`kzome/manga-cleaner`](https://huggingface.co/kzome/manga-cleaner) | comic-text-detector plus big-LaMa with median-colour fill: the same pieces the fork already has. AGPL-3.0. No quality metrics. |

API (Fal, Replicate, OpenRouter image models).

| Model | Status | Notes |
| --- | --- | --- |
| Microsoft MAI Image 2.5 Edit / Pro Edit (Fal) | in use (default) | Default cleanup and restoration model. Full-page by default (plan item 6). |
| FLUX Kontext dev / pro / max, FLUX.2 dev / pro / flex / max / Klein 4B / 9B (Fal, Replicate) | in use | Prompted edits. |
| Qwen Image Edit 2511, Seedream 4.5 Edit (Fal) | in use | |
| Any OpenRouter image model | in use | Free-form model id. |
| FLUX 3 Image (BFL, Replicate, Fal, OpenRouter) | candidate (high) | Released 2026-10-01. Owner's manual check 2026-10-20: gentlest on the art, but leaves small handwritten text, and BFL blocked page 23 as sexual content. ~$0.048 per 1K image (third-party figure). `safety_tolerance` 0–4 on Replicate and BFL; whether 4 passes explicit input pages is untested. Needs a Replicate catalog entry; reachable now through OpenRouter without moderation control. |
| FLUX 3 Image Klein API (BFL, Runware) | candidate (medium) | ~$0.009 per 1K image at BFL (third-party figure). Not wired. |
| Qwen Image Edit 2511 on Replicate | candidate (high) | $0.03 per image; `disable_safety_checker` disables the output check. Not in the Replicate catalog yet. First pick for adult pages. |
| MAI Image 2.6 / 2.6 Flash (OpenRouter) | candidate (high) | P1 check 2026-10-20 through OpenRouter `/api/v1/images` (chat completions returns 404 for image output): on par with Krea 3, clean on 3 of 5 pages including small handwriting; left a large SFX on page 6, turned a dark balloon white on page 23. Adult page 19 timed out upstream twice (524); page 24 was refused by Azure moderation once and passed on retry. About $0.03 per page, 1024 px output. Flash untested. Run 2 (2026-10-20, prompt P1b): with balloons pre-cleaned locally it cleaned all 6 pages, kept the dark balloon dark and the speed lines intact; best candidate for pre-cleaned pages. Page 19 without pre-cleaning failed twice upstream (504, 524). |
| GPT-Image-2.5, Nano Banana 2.1 (OpenRouter) | candidate (quality ceiling only) | Top of Image Edit Arena; provider policies forbid sexual content, so expected to fail adult pages. Test on ordinary pages only. |
| Krea 3 | candidate (high) | Best overall in the owner's manual P1 check on 6 pages (2026-10-20): clean on 4 pages; on page 23 it erased a dark balloon and left text in another; it leaves white outline shapes where outlined SFX were. Run 2 (2026-10-20, OpenRouter `krea/k3-edit`, 2K, about $0.08 per page, prompt P1b): with balloons pre-cleaned it redraws the art under outlined SFX on page 19 and stays closest to the source screentone, but on page 7 it redrew a character over a vertical SFX. |
| Ideogram 4.5 | candidate (medium) | Same check: best at removing outlined SFX and rebuilding the art, but missed a large SFX (page 6) and a large caption (page 18), and erased a balloon on page 7. |
| Grok Imagine Image 2.0 | candidate (medium) | Owner's manual P1 check 2026-10-20: the only model that removed every letter on the dense adult page 19, leaving empty white SFX outline shapes. It redraws the rest of the page (balloons and panel backgrounds vanish on page 6, lines shift on page 18), so it is usable only with mask compositing. Run 2 (2026-10-20, 2K, about $0.08 per page, prompt P1b): with balloons pre-cleaned it redraws the art under page 19 SFX; without pre-cleaning it still leaves white shapes and drops screentone and speed lines; on page 7 it redrew a character over a vertical SFX. |
| Qwen Image 2.1 Pro / Qwen Image 3 Pro (OpenRouter) | rejected | Run 3 (2026-10-20) tested 3 Pro instead of 2.1 Pro on pre-cleaned pages: left all text on page 19, invented new text on page 6, painted a white band over an SFX on page 7; about $0.08 and 3.5 min per image. |
| Seedream 5.0 Flash / Pro (OpenRouter) | rejected | Run 3 (2026-10-20): moderation blocked 4-5 of 6 non-explicit pages without a reason while passing the adult page 19; Pro silently replaced the page 19 scene with a different character. |
| Muse Image (Meta) | candidate (medium) | Run 3 (2026-10-20, OpenRouter `meta/muse-image`, 1600 px, about $0.015): cleaned all 6 pre-cleaned pages including adult page 19, but erased balloons and drew hair and faces in their place on page 18, so mask compositing only. |
| HY-Image 3.5 preview (Tencent, OpenRouter `tencent/hy-image-v3.5-preview`) | candidate (high) | Run 3 (2026-10-20): best of all runs on pre-cleaned pages: 6 of 6 including adult page 19, removed every remaining SFX and redrew art under outlines, kept the dark balloon and speed lines, did not redraw characters; 2K, about $0.016 per image. Preview model, may change. |
| Recraft V4.1 Flash / V4 Styles | rejected | Generation only: Flash takes no image input, Styles is a style model, so neither can edit a page. |
| Seedream 5.0 Pro Edit | candidate (low) | Official API refuses even mild content and checks edits harder than generation (third-party tests); only resellers claim otherwise. |

Candidates, prompts and the test plan (including cleaning with balloons already emptied versus not) are in [cleaning-models.ru.md](cleaning-models.ru.md), checked 2026-10-19. No cloud cleaning model has been measured on the owner's pages. When the
evaluation set exists, score residual ink inside the mask, damage outside it,
policy rejections, cost and time per page, and move losers to `rejected`.

## Scan restoration

Design and measurements in [scan-restoration.md](scan-restoration.md);
sources checked 2026-10-21 on OpenModelDB and GitHub.

| Model | Status | Source | Notes |
| --- | --- | --- | --- |
| 1x-MangaJPEG HQPlus / HQ / MQ / LQ | in use (scan enhancement) | [OpenModelDB](https://openmodeldb.info/models/1x-MangaJPEGHQ), pinned by SHA-256 | Old-arch ESRGAN, 32 features, 12 blocks, one grey channel. Loaded natively (`koharu-ml::manga_jpeg`, legacy `.pth` reader in `koharu-torch`). Chosen one step lighter than the published ranges; LQ adds halos (measurements in scan-restoration.md). CC BY-NC-SA 4.0, accepted by the owner for this fork; downloaded, not bundled. |
| MangaJaNai V1 (2x / 4x, per input height) | candidate (later, upscaling) | [the-database/MangaJaNai](https://github.com/the-database/MangaJaNai) | B&W upscaler that also removes JPEG noise and restores dots. Changes page size; license not stated in the repo. |
| IllustrationJaNai V2 DeJPEG FDAT (1x) | candidate (low) | OpenModelDB | Colour pages: removes JPEG artifacts and keeps halftone. FDAT is not a quick port. CC BY-NC-SA 4.0. |
| 1x eula digimanga bw | rejected 2026-10-21 | OpenModelDB | Its 1x companion deliberately smooths halftone (it was made to build a dataset). |
| Waifu2x, FBCNN, SCUNet | rejected 2026-10-21 | | Trained on anime or photos; flatten screentone. FBCNN's JPEG-quality prediction idea is replaced by reading the DQT table. |

## Translation models (JA→RU chapter)

Measured 2026-10-18 on 6 pages of the owner's chapter, 99 segments. Four of the pages are adult content. The run used Koharu's chapter path (adaptive_v1, OpenRouter, strict output check) and was blind: the references for those pages were stripped. Scores are chrF against the owner's own translation, overall / adult pages / ordinary pages.

| Model | chrF | Cost for 6 pages | Notes |
| --- | --- | --- | --- |
| Gemini 3.7 Flash | 44.2 / 43.0 / 45.5 | $0.105 | Default for chapter translation. Won the owner's blind check (9 of 13 decided lines). |
| Claude Sonnet 5.5 | 35.8 on adult pages only | $0.162 for 4 pages | Keeps male gender for the narrator. |
| DeepSeek V4.1 Flash | 36.2 / 36.0 / 36.3 | <$0.01 | Gives the male narrator female gender. |
| Qwen3.7-plus | 30.3 / 25.6 / 35.1 | $0.034 | Same gender errors. |
| Grok 4.3 | 27.1 / 25.2 / 28.9 | $0.067 | |

None of the models refused, censored, or softened explicit lines. Sending page images with page-mode translation broke Koharu's strict source-echo check: Gemini "corrected" the source text from the image, and Qwen returned invalid JSON. Images therefore go only to scene markup. Gemini 3.5 Flash, Kimi K3 and GPT-6.1 Sol were not run because of the budget.

A second Gemini 3.7 Flash run on 5 of those pages (88 segments, 2026-10-19) added scene markup and the prompt fixes `d3a0910` (bare sound effects) and `b867a7b` (rough speech stays rough). It scored chrF 44.3 overall and 46.0 on the lines the owner edited (39.0 on the first markup run). A line-by-line check against the Japanese found 86 of 88 lines with the right meaning, and 77 with no remarks. One of the two errors came from OCR.
