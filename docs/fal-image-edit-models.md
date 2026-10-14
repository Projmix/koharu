# Модели редактирования FAL

Каталог используется в автоматическом клининге, настройках восстановления и инструментах редактора. Источник каталога и параметров запросов — `FalModel` в `crates/koharu-pipeline/src/stages/fal.rs`; конфигурация проверяет выбранный идентификатор через тот же каталог.

Схемы проверены 13 сентября 2026 года:

| Модель | Документация endpoint |
| --- | --- |
| Microsoft MAI Image 2.5 Edit | [microsoft/mai-image-2.5/edit](https://fal.ai/models/microsoft/mai-image-2.5/edit/api) |
| Microsoft MAI Image 2.5 Pro Edit | [microsoft/mai-image-2.5-pro/edit](https://fal.ai/models/microsoft/mai-image-2.5-pro/edit/api) |
| FLUX.1 Kontext Dev | [fal-ai/flux-kontext/dev](https://fal.ai/models/fal-ai/flux-kontext/dev/api) |
| FLUX.1 Kontext Pro | [fal-ai/flux-pro/kontext](https://fal.ai/models/fal-ai/flux-pro/kontext/api) |
| FLUX.1 Kontext Max | [fal-ai/flux-pro/kontext/max](https://fal.ai/models/fal-ai/flux-pro/kontext/max/api) |
| FLUX.2 Dev Edit | [fal-ai/flux-2/edit](https://fal.ai/models/fal-ai/flux-2/edit/api) |
| FLUX.2 Pro Edit | [fal-ai/flux-2-pro/edit](https://fal.ai/models/fal-ai/flux-2-pro/edit/api) |
| FLUX.2 Flex Edit | [fal-ai/flux-2-flex/edit](https://fal.ai/models/fal-ai/flux-2-flex/edit/api) |
| FLUX.2 Klein 4B Edit | [fal-ai/flux-2/klein/4b/edit](https://fal.ai/models/fal-ai/flux-2/klein/4b/edit/api) |
| FLUX.2 Klein 9B Edit | [fal-ai/flux-2/klein/9b/edit](https://fal.ai/models/fal-ai/flux-2/klein/9b/edit/api) |
| Qwen Image Edit 2511 | [fal-ai/qwen-image-edit-2511](https://fal.ai/models/fal-ai/qwen-image-edit-2511/api) |
| ByteDance Seedream 4.5 Edit | [fal-ai/bytedance/seedream/v4.5/edit](https://fal.ai/models/fal-ai/bytedance/seedream/v4.5/edit/api) |

MAI Pro и Kontext принимают одно изображение в `image_url`; остальные — массив `image_urls`. Все запросы передают один исходный PNG как Data URI и возвращают один результат через очередь FAL. FLUX.2 Pro/Flex не принимают `num_images`. Seedream не принимает `output_format`: его ответ декодируется и при применении сохраняется в PNG; явно выбраны одна картинка и `auto_2K`.

Параметр MAI `aspect_ratio: auto` не отправляется другим моделям. Kontext Dev использует `resolution_mode: match_input`; остальные модели редактирования используют свои настройки размера по умолчанию. Полученный результат приводится к размеру отправленного участка перед применением, как и у существующих MAI-моделей. Побочные изменения рисунка зависят от модели и требуют проверки на примерах.

Koharu не меняет политику модерации моделей. Если результат содержит `has_nsfw_concepts: [true]`, приложение сообщает об отказе и не применяет чёрную картинку-заглушку к странице. Проверки запросов выполняются на локальном HTTP-сервере; качество моделей и фактическая модерация платными запросами не проверялись.

## Сетевые сбои

Отправка изображения создаёт платное задание. Обработчик FAL в Koharu не повторяет POST при ошибке: если соединение оборвалось или подтверждение не удалось прочитать, проверьте задание в разделе Requests на Fal.ai перед повторным запуском. Штатные перенаправления и безопасные повторы HTTP-клиента при подтверждённом отказе протокола принять запрос остаются без изменений; повтора всей генерации на уровне конвейера нет.

Чтение статуса, получение результата и скачивание изображения повторяются при сетевых обрывах, таймаутах и HTTP 408/502/503/504: до четырёх попыток одного URL с паузами 1, 2 и 4 секунды. Повтор охватывает тело ответа, поэтому разрыв посреди скачивания также обрабатывается. На отдельный запрос установлен лимит 120 секунд; ожидание генерации в очереди этим лимитом не ограничено. «Стоп» прерывает чтение и паузу перед повтором.

Ошибки авторизации, модерации, некорректный JSON и неподходящие изображения не повторяются. Если скачать созданное изображение не удалось после всех попыток, ошибка содержит ссылку на готовый файл. Ошибки чтения очереди содержат URL существующего задания. Это позволяет найти уже оплаченный результат без новой генерации.
