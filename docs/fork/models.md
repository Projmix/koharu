# Model registry

Every detector, OCR and cleaning model the fork uses, ships unused, has
measured, or has decided against. The point is to never evaluate the same
model twice: before trying a model, look it up here; after trying it, record
the result. Last updated 2026-10-18.

## Rules

- **Status** is one of:
  - `in use`: selectable in the pipeline today.
  - `shipped, unused`: ported in `koharu-ml` (pinned revision), not called by
    the pipeline.
  - `candidate`: worth measuring; not ported.
  - `rejected`: do not revisit unless the condition in its row changes.
- A model moves to `rejected` only with a reason: a measurement on the
  evaluation set (see [improvement-plan.md](improvement-plan.md), item 2), a
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

| Model | Status | Source | Notes |
| --- | --- | --- | --- |
| PaddleOCR-VL 1.6 (GGUF) | in use (default) | `PaddlePaddle/PaddleOCR-VL-1.6-GGUF` @ `511b096` | General document VLM, multilingual. No manga-specific evidence. |
| Hayai OCR v2.1 | in use | `JustANormalTinkerer/hayai-ocr-v2` @ `4a4ce47` | Japanese. Recommended local OCR in `docs/research/2026-09-12-japanese-ocr.md`. |
| Baberu OCR | in use | `genshiai-daichi/baberu-ocr` @ `d9cc131` | Japanese. |
| Manga OCR | in use | `mayocream/manga-ocr` @ `4380edb` | Japanese baseline; known to invent text on empty crops. |
| PP-OCRv6 rec | shipped, unused | PaddlePaddle | Line recognizer; not manga-specific. |
| PaddleOCR-VL-For-Manga | candidate (high) | `jzhang533/PaddleOCR-VL-For-Manga` ([demo](https://huggingface.co/spaces/jzhang533/paddleocr-vl-for-manga-demo)) | Best CER in [JMangaBench_Read](https://github.com/muscgab/JMangaBench_Read) (2.13% vs Hayai v2.1 2.67%, Manga OCR 5.01%), but about 2.5x slower per crop. Fine-tuned from PaddleOCR-VL (the original 0.9B, not 1.6) and run with the plain `OCR:` prompt in its demo. The pipeline uses a GGUF of 1.6, so trying it needs either the unquantized `paddle_ocr_vl` path with these weights or a GGUF conversion, then a comparison on identical crops. Hugging Face is not reachable from the cloud sessions, so this runs on a desktop. |
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
| DeepSeek-OCR 2 | rejected 2026-08-24 | `docs/research/api-ocr-models-2026-08.md` | No hosted OpenAI-compatible endpoint; self-hosting needs CUDA/vLLM. Revisit if a hosted API appears. |

## Cleaning

Local.

| Model | Status | Source | Notes |
| --- | --- | --- | --- |
| LaMa (manga weights) | in use (default) | `mayocream/lama-manga` @ `f91c85b` | Tiled at 512 px; flat regions are filled without inference. |
| AOT inpainting | in use | `mayocream/aot-inpainting` @ `cffe234` | |
| FLUX.2 Klein 4B (GGUF) | in use | `unsloth/FLUX.2-klein-4B-GGUF` @ `0084d1d` | Prompted diffusion; slow without a strong GPU. |
| RORem mixed (SDXL inpainting) | in use | `mayocream/RORem-mixed-GGUF` @ `62c75b3` | Object-removal diffusion. |
| manga-cleaner | rejected 2026-10-18 | [`kzome/manga-cleaner`](https://huggingface.co/kzome/manga-cleaner) | comic-text-detector plus big-LaMa with median-colour fill: the same pieces the fork already has. AGPL-3.0. No quality metrics. |

API (Fal, Replicate, OpenRouter image models).

| Model | Status | Notes |
| --- | --- | --- |
| Microsoft MAI Image 2.5 Edit / Pro Edit (Fal) | in use (default) | Default cleanup and restoration model. Full-page by default (plan item 6). |
| FLUX Kontext dev / pro / max, FLUX.2 dev / pro / flex / max / Klein 4B / 9B (Fal, Replicate) | in use | Prompted edits. |
| Qwen Image Edit 2511, Seedream 4.5 Edit (Fal) | in use | |
| Any OpenRouter image model | in use | Free-form model id. |

No cloud cleaning model has been measured on the owner's pages. When the
evaluation set exists, score residual ink inside the mask, damage outside it,
policy rejections, cost and time per page, and move losers to `rejected`.
