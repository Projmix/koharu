# Replicate: редактирование через FLUX

В настройках провайдеров появился отдельный ключ Replicate. Он сохраняется через системное хранилище секретов Koharu под именем `replicate`, не включается в конфигурацию проекта и не возвращается интерфейсу при чтении настроек. Интерфейс показывает только признак сохранённого ключа.

Для автоматического клининга выберите **Конвейер → Удаление текста → API → Replicate → FLUX.2 Dev Edit**. Для инструмента восстановления выберите тот же провайдер и модель в разделе **Восстановление**. Выбор клининга и восстановления независим. Существующие настройки других провайдеров автоматически не переключаются.

Подключены 11 официальных моделей редактирования. Во всех идентификаторах владелец — `black-forest-labs`; отдельного суффикса `/edit` нет.

| Модели (схемы API) | Поле изображения | Настройки |
| --- | --- | --- |
| [FLUX.2 Dev](https://replicate.com/black-forest-labs/flux-2-dev/api/schema) | `input_images` | `go_fast: false`, `disable_safety_checker: true` |
| [FLUX.2 Pro](https://replicate.com/black-forest-labs/flux-2-pro/api/schema), [Max](https://replicate.com/black-forest-labs/flux-2-max/api/schema), [Flex](https://replicate.com/black-forest-labs/flux-2-flex/api/schema) | `input_images` | `resolution: match_input_image`, `safety_tolerance: 5`; у Flex также `prompt_upsampling: false` |
| [Klein 4B](https://replicate.com/black-forest-labs/flux-2-klein-4b/api/schema), [9B](https://replicate.com/black-forest-labs/flux-2-klein-9b/api/schema), [4B Base](https://replicate.com/black-forest-labs/flux-2-klein-4b-base/api/schema), [9B Base](https://replicate.com/black-forest-labs/flux-2-klein-9b-base/api/schema) | `images` | `go_fast: false`, `disable_safety_checker: true`; разрешение — ближайший не меньший доступный размер от 0.25 до 4 MP, с ограничением 4 MP |
| [Kontext Dev](https://replicate.com/black-forest-labs/flux-kontext-dev/api/schema) | `input_image` | `go_fast: false`, `disable_safety_checker: true` |
| [Kontext Pro](https://replicate.com/black-forest-labs/flux-kontext-pro/api/schema), [Max](https://replicate.com/black-forest-labs/flux-kontext-max/api/schema) | `input_image` | `safety_tolerance: 2` — допустимый максимум с изображением; `prompt_upsampling: false` |

Это модели изменения изображения по инструкции. FLUX Fill, Depth, Canny, Redux и варианты LoRA относятся к другим сценариям и в этот список не включены.

Общие параметры:

- одно подготовленное изображение в поле своей модели, соответствующее окну предпросмотра;
- `prompt`: промпт из настроек или его утверждённая в предпросмотре редакция, без дописывания инструкций;
- `aspect_ratio: match_input_image` и `output_format: png`.

Схемы проверены 13.09.2026. Ответ Klein — массив из одного URL, у остальных моделей — один URL. Несоответствие схеме или несколько изображений приводят к ошибке до применения результата. Отключение проверки результата используется только там, где оно поддерживается; ограничения Pro/Max/Flex и общие правила Replicate продолжают действовать. Replicate отдельно описывает [отключение встроенного фильтра через API при ложных срабатываниях](https://replicate.com/docs/topics/predictions/safety-checking/).

Небольшой PNG до 256 KiB передаётся как Data URI. Более крупный PNG загружается через файловый API Replicate (`POST /v1/files`, multipart-поле `content`), после чего в запрос включается возвращённый `urls.get`. Для файлов используется механизм провайдера, описанный в [документации](https://replicate.com/docs/topics/predictions/input-files) и [официальном клиенте](https://github.com/replicate/replicate-javascript/blob/main/lib/files.js). Koharu не уменьшает исходное изображение перед отправкой; готовый ответ приводится к размеру отправленного участка перед применением.

Создание prediction выполняется асинхронно и не повторяется обработчиком при ошибке. Статусы `starting`/`processing` опрашиваются до `succeeded`; `failed` и `canceled` возвращают ошибку. «Стоп» прерывает локальное ожидание и отправляет отмену существующему prediction. После неоднозначного сбоя создания проверьте историю Replicate перед ручным повтором.

Чтение prediction и загрузка результата повторяются до четырёх раз при временных сетевых ошибках, HTTP 408/429/500/502/503/504, включая прерванное чтение тела. На запрос установлен лимит 120 секунд. Ключ не отправляется на адрес выходного изображения. При окончательном сбое скачивания сообщение содержит URL уже созданного результата. Размер входного и выходного PNG/файла ограничен 64 MiB.

Интеграция проверяется локальным HTTP-сервером и тестами предпросмотра; платный вызов и сравнение качества с FAL остаются для пользовательской проверки после ввода ключа.
