# API vision models for manga OCR (2026-08-24)

## Scope and evidence

This note is a shortlist for Japanese, Chinese, and Korean manga OCR through
OpenRouter or an OpenAI-compatible endpoint. It uses only first-party model
cards/reports and OpenRouter's live model API. It does **not** claim manga OCR
quality: the sources below evaluate general OCR/documents, not vertical manga
text, furigana, stylized sound effects, or bubble reading order. Those cases
need a Koharu-owned test set before choosing a default.

OpenRouter availability and parameter support are a point-in-time observation
from its [`GET /api/v1/models`](https://openrouter.ai/api/v1/models) response on
2026-08-24. `structured_outputs` below means that OpenRouter advertises that
request parameter; it is transport/schema support, not proof that every OCR
character will be correct.

## Shortlist

| Priority | Model ID | Why it is a credible candidate | Important caveat |
| --- | --- | --- | --- |
| 1 | `qwen/qwen3.8-27b` | Current open-weight Qwen VLM. Its official card says it natively accepts images/videos and documents; the official repository documents OpenAI-compatible serving. OpenRouter currently advertises image input, `structured_outputs`, and `response_format`. | The official 3.8 card publishes no OCR-specific or manga-specific result. Treat it as the first model to benchmark, not a proven winner. |
| 2 | `qwen/qwen3.5-397b-a17b` | Official card identifies it as a native VLM, supports 201 languages/dialects, reports OCRBench 93.1 and CC-OCR 82.0, and documents OpenAI-compatible serving. OpenRouter advertises image input and structured output. | Very large model; API cost/latency may be worse than 27B. Vendor benchmark is general OCR and does not isolate Japanese vertical manga text. |
| 3 | `qwen/qwen3-vl-235b-a22b-instruct` | Qwen's official report/card explicitly focuses on OCR and document parsing. The report says 39 OCR languages were evaluated, with over 70% accuracy for 32; its published language figure includes Japanese and Korean (and Chinese is part of the primary corpus). OpenRouter advertises image input and structured output. | Older than Qwen3.5/3.8. The multilingual set is in-house, and the report still does not test manga/furigana/SFX. |
| 4 | `qwen/qwen3-vl-32b-instruct` | Same Qwen3-VL OCR-oriented family in a smaller API option. The official report says the 32B variants were competitive across OCR parsing/document benchmarks. OpenRouter advertises image input and structured output. | Lower-capacity fallback; no evidence that 32B preserves tiny/stylized manga characters as well as 235B. |

### Practical recommendation

Start the Koharu selector with `qwen/qwen3.8-27b` as the current balanced
candidate, but benchmark it against `qwen/qwen3.5-397b-a17b` and
`qwen/qwen3-vl-235b-a22b-instruct` on the same crops. Keep
`qwen/qwen3-vl-32b-instruct` as a cheaper baseline. Do not label any model as
“best manga OCR” until measured on representative pages.

For every model, send the project source language explicitly, use deterministic
generation where the provider permits it, require a JSON Schema response, and
validate it locally. Structured output prevents malformed envelopes; it cannot
prevent invented or omitted text.

## Models deliberately not recommended yet

- **DeepSeek-OCR 2** is OCR-specialized and officially supports “Free OCR” and
  document-to-Markdown prompts, but its official setup is CUDA/vLLM or custom
  Transformers inference. It was not present in the OpenRouter catalog checked
  above, and the official project does not document a ready hosted
  OpenAI-compatible endpoint. It is therefore not a drop-in API choice for this
  CPU-only laptop.
- General current VLMs from OpenAI, Google, Anthropic, GLM, and DeepSeek may be
  worth an empirical comparison, and several are currently vision-capable on
  OpenRouter. They are omitted from this short list because the primary sources
  inspected here did not provide manga-specific evidence, while Qwen publishes
  explicit OCR/document evidence. Availability alone is not OCR evidence.

## What the benchmark must cover

Use held-out pages for all three source languages and score exact Unicode text,
not only visually similar output. Split results at least by horizontal versus
vertical writing, furigana/ruby, tiny text, low contrast, handwritten/stylized
SFX, text crossing bubble art, and reading order. Also measure missing and
hallucinated regions and JSON-schema failure rate. A whole-page test and the
current detector-crop test answer different questions and should both be kept.

## Primary sources

1. [Qwen3.8-27B official model card](https://huggingface.co/Qwen/Qwen3.8-27B)
   — native vision-language model, image/video and document understanding.
2. [Qwen3.8 official repository](https://github.com/QwenLM/Qwen3.8)
   — supported weights and OpenAI-compatible serving examples.
3. [Qwen3.5-397B-A17B official model card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
   — native VLM, language coverage, OCRBench/CC-OCR results, and
   OpenAI-compatible serving.
4. [Qwen3-VL official model card/repository](https://github.com/QwenLM/Qwen3-VL)
   — expanded multilingual OCR and document parsing.
5. [Qwen3-VL technical report](https://arxiv.org/abs/2511.21631)
   — OCR training/evaluation, 39-language experiment, and results for the 235B
   and 32B families.
6. [OpenRouter live model catalog API](https://openrouter.ai/api/v1/models)
   — current IDs, image modality, and advertised structured-output parameters.
7. [DeepSeek-OCR 2 official repository](https://github.com/deepseek-ai/DeepSeek-OCR-2)
   and [model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) — supported
   OCR prompts and local CUDA/vLLM/Transformers setup.
