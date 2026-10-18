# API редактирования манги с отключаемой проверкой результата

Проверено по первичным страницам провайдеров 13 сентября 2026 года. Платные запросы и локальные модели не запускались.

## Кандидаты

| API | Цена | Управление проверкой |
| --- | ---: | --- |
| Replicate `qwen/qwen-image-edit-2511` | **$0.03 за выходное изображение** | В схеме есть `disable_safety_checker: true`: параметр отключает safety checker **сгенерированного изображения**. [Модель и цена](https://replicate.com/qwen/qwen-image-edit-2511), [схема API](https://replicate.com/qwen/qwen-image-edit-2511/api/schema). |
| Replicate `black-forest-labs/flux-kontext-dev` | **$0.025 за выходное изображение** | В схеме есть `disable_safety_checker: true`, описанный как отключение NSFW checker. [Модель, цена и параметр](https://replicate.com/black-forest-labs/flux-kontext-dev), [схема API](https://replicate.com/black-forest-labs/flux-kontext-dev/api/schema). |
| Segmind `flux-kontext-dev` | Не фиксировалась | API явно описывает `disable_safety_checker` как отключение NSFW-фильтра. [Документация API](https://www.segmind.com/models/flux-kontext-dev/api). |
| fal `fal-ai/qwen-image-edit-2511` | Не фиксировалась | Есть `enable_safety_checker: false`, но отключение требует авторизации аккаунта; неавторизованные запросы всегда проверяются. [Документация API](https://fal.ai/models/fal-ai/qwen-image-edit-2511/api). |
| fal `fal-ai/flux-kontext/dev` | $0.025 за мегапиксель, с округлением вверх | `enable_safety_checker: false` также требует разрешения для аккаунта; без него проверка остаётся, небезопасные результаты возвращаются чёрными. [Схема API](https://fal.ai/models/fal-ai/flux-kontext/dev/api), [цена](https://fal.ai/models/fal-ai/flux-kontext/dev). |
| fal `fal-ai/flux-2/klein/9b/edit` | $0.011 за мегапиксель входа и выхода; пример провайдера — $0.022 за 1 MP входа + 1 MP выхода | `enable_safety_checker: false` также требует разрешения для аккаунта. [Схема API](https://fal.ai/models/fal-ai/flux-2/klein/9b/edit/api), [правила расчёта цены](https://fal.ai/models/fal-ai/flux-2/klein/9b/edit). |
| fal `fal-ai/flux-pro/kontext` | Не фиксировалась | `safety_tolerance` регулирует строгость проверки результата; наличие параметра не означает отключения всех фильтров. [Схема API](https://fal.ai/models/fal-ai/flux-pro/kontext/api). |

Replicate отдельно объясняет, что встроенная проверка иногда даёт ложные срабатывания, и в таких случаях её разрешено отключить при запуске через API. [Политика Replicate](https://replicate.com/docs/topics/predictions/safety-checking/).

## Граница гарантии

У Qwen на Replicate параметр явно относится к проверке результата модели. Документация остальных перечисленных параметров не доказывает отсутствие отдельной проверки входного изображения, промпта, аккаунта или общей платформенной модерации. Ошибка MAI2.6 `Input content violated safety policies` сообщает об отказе по входному содержимому. Перечисленные варианты требуют смены модели или провайдера; это не параметры отключения фильтра MAI2.6.

Первый кандидат для сравнения — Replicate Qwen Image Edit 2511: он принимает несколько исходных изображений и позиционируется для точечных правок, включая текст. [Описание модели](https://replicate.com/qwen/qwen-image-edit-2511). Практическое качество удаления японского текста с пользовательской картинки ещё не проверено; цена означает один полученный результат, а не гарантированно удачную правку.

## Что подтверждают отзывы

В проверенных обсуждениях не найден воспроизводимый тест, где одинаковые изображение и промпт отклонены fal и приняты Replicate. Наличие переключателя в API не доказывает, что Replicate систематически менее строгий.

- В августе 2024 года пользователь FLUX Pro сообщил о ложных NSFW-срабатываниях на Replicate, которых пока не встречал на fal. Одинаковые входные данные и seed не зафиксированы. Это отдельное наблюдение в обратную сторону, не современный сравнительный тест. [Обсуждение логотипов](https://www.reddit.com/r/FluxAI/comments/1eosnqt/flux_pro_for_logo_development/).
- 12 августа 2024 года автор обёртки для API fal и Replicate написал, что в ней checker отключался для FLUX dev/schnell, а у pro использовалась максимальная tolerance. Сообщение относится к старым версиям FLUX и не подтверждает текущее поведение Kontext, Klein или Qwen. [Сообщение автора интеграции](https://www.reddit.com/r/StableDiffusion/comments/1ei0tgf/how_to_use_the_flux_api_in_your_local_workflows/).

Для проверки на существующем fal-аккаунте подходят именно модели редактирования Kontext dev или FLUX.2 Klein 9B Edit, но их текущие схемы требуют разрешения аккаунту на отключение checker. Успех на пользовательском фрагменте пока не проверен.
