# Модели клининга: что тестировать

Собрано 19 октября 2026 года. Платных запусков не было: это список кандидатов,
фактов о провайдерах и план проверки, чтобы потом протестировать по делу.
Результаты измерений записываются в [models.md](models.md), раздел Cleaning.

Как читать пометки:

- **проверено** — взято с первичной страницы (карточка модели на Hugging Face,
  схема API провайдера) или из прошлого опыта владельца;
- **по сторонним источникам** — блоги, агрегаторы, перепродавцы API. Цифры
  там часто расходятся, их надо подтвердить первым же запросом;
- **не проверено** — гипотеза, которую закрывает тест.

## Что уже известно по нашему опыту

- **MAI Image 2.5 / 2.5 Pro Edit** (Fal) хорошо слушаются инструкции и не
  портят рисунок, но на страницах 18+ отказывают. 2.6 через OpenRouter даёт
  ошибку `Input content violated safety policies`: это отказ по входной
  картинке, параметром его не отключить
  ([прошлое исследование](../research/2026-09-13-image-edit-safety-checkers.md)).
- **Fal:** фильтр 18+ отключается параметром `enable_safety_checker: false`
  только для аккаунтов, которым Fal это разрешил. Без разрешения проверка
  остаётся, а запрещённый результат приходит чёрной картинкой. На главе
  владельца откровенные страницы через Fal не проходят.
- **Replicate:** у части моделей есть `disable_safety_checker`, в Koharu он
  выставлен в коде без переключателя. Политика Replicate прямо разрешает
  отключать проверку через API
  ([документация](https://replicate.com/docs/topics/predictions/safety-checking/)).
  Страницы 18+ через Replicate ещё не проверялись.
- В последних правках пузыри с однотонной заливкой после любого облачного
  клинера закрашиваются своим цветом, а маска текста не выходит за контур
  пузыря. Значит, внутри обычных пузырей облако уже почти ничего не решает:
  его работа — звуки и текст поверх рисунка.

## Первая ручная проверка (20.10.2026)

Владелец прогнал запрос P1 через сайты моделей на страницах 6, 7, 18, 19
(18+), 23 и 24. Результаты сравнивались с исходником по карте разницы и
вблизи. Каждая модель запускалась один раз на странице.

| Модель | Итог | Проблемы |
| --- | --- | --- |
| Krea 3 | лучший в целом | 23: стёр тёмный пузырь и оставил текст в белом; 19: от звуков остались белые обводки |
| FLUX 3 Image | бережнее всех к рисунку | оставляет мелкий рукописный текст (7, 24); на 19 белые обводки звуков; 23 заблокирована BFL («Sexual Content») |
| Ideogram 4.5 | лучше всех убирает звуки вместе с обводкой (19, 23) | пропустил «ポッ» (6) и крупную надпись (18); на 7 стёр пузырь |
| FLUX 3 Image Klein | звуки на 19 убраны вместе с обводкой | стирает пузыри целиком (7, 24) и перерисовывает фон; 23 заблокирована BFL |
| Krea 3 Beta | 6 чисто | на 19 остались звуки |
| Grok Imagine 2 | лучший на плотной странице 19: убрал все буквы, даже мелкие (остались белые обводки звуков) | перерисовывает остальную страницу: пропадают пузыри и фоны панелей (6), сдвигаются линии (18). Годится только с вклейкой по маске |

Выводы:

- Страница 19 (18+) прошла у всех. BFL заблокировал 23 и, видимо, судил по
  тексту, а не по рисунку. Значит, если пузыри очищены заранее, облаку
  почти нечего блокировать.
- Звуки с белой обводкой: часть моделей стирает только буквы и оставляет
  белое пятно. В запрос нужно добавить, что буква стирается вместе с
  обводкой, а под ней дорисовывается рисунок.
- Стёртые пузыри не страшны при вклейке по маске текста. Опасен оставшийся
  текст.
- Модели возвращают меньшее разрешение: FLUX и Grok дают 1024 px вместо
  1280.

## Кандидаты

Цена за страницу дана для размера около 1 Мп (страница владельца ~1280 px
по высоте). Колонка «в Koharu» говорит, можно ли запустить модель без правки
кода.

### Облачные модели

| Модель | Вышла | Где есть | Цена за страницу | 18+ | В Koharu |
| --- | --- | --- | --- | --- | --- |
| **FLUX 3 Image** (BFL) | 1–2 окт 2026 | BFL API, Replicate `black-forest-labs/flux-3-image`, Fal, OpenRouter | $0,048 (1K), $0,10 (2K); у BFL отказ модерации не тарифицируется — по сторонним источникам | `safety_tolerance` 0–4, по умолчанию 2 (страница Replicate). Пропустит ли 4 откровенную входную картинку — не проверено | через OpenRouter (без `safety_tolerance`); для Replicate нужен id модели в каталоге |
| **FLUX 3 Image Klein** (BFL, API) | 13 окт 2026 | BFL API, Runware | ~$0,009 (1K) у BFL, от $0,003 за 512 px у Runware — по сторонним источникам | у BFL та же модерация; у Runware только флаг «NSFW обнаружен» в ответе, политику надо уточнить | нет |
| **Qwen Image Edit 2511** | дек 2025 | Replicate, Fal | $0,03 на Replicate | Replicate: `disable_safety_checker: true` отключает проверку **результата** (схема API) | да, Fal; Replicate — не в каталоге |
| **MAI Image 2.6 / 2.6 Flash** (Microsoft) | авг–сен 2026 | OpenRouter, Microsoft Foundry | 2.6: $8 / $38 за 1M токенов картинки на вход / выход, ~$0,04 за кадр; Flash: $2,5 / $19 | отказывает по входу (наш опыт) | да, OpenRouter |
| **Grok Imagine Image 2.0** (xAI) | 2026 | xAI API, OpenRouter, Replicate, WaveSpeed | $0,04 + $0,01 за входную картинку (xAI и Replicate) — по сторонним источникам | неизвестно; в приложении у xAI есть «spicy», про API данных нет | через OpenRouter |
| **GPT-Image-2.5** Flare / Sunburst (OpenAI) | 8 сен 2026 | OpenAI API, OpenRouter | по токенам: ~$0,013 medium, ~$0,053 high за 1024² — по сторонним источникам | у OpenAI сексуальный контент запрещён при любой настройке модерации — ожидаем отказ | через OpenRouter |
| **Nano Banana 2.1** (Google) | 2026 | Gemini API, OpenRouter | $0,034 (1K), $0,017 в batch — по сторонним источникам | два слоя: входной фильтр можно ослабить `BLOCK_NONE`, проверку результата (IMAGE_SAFETY) нельзя — ожидаем отказ | через OpenRouter |
| **Seedream 5.0 Pro Edit** (ByteDance) | 2026 | BytePlus, перепродавцы | ~$0,035 у перепродавцов | официальный API отказывает, правки строже генерации; перепродавцы обещают «uncensored» — не проверено | нет (в каталоге Fal только 4.5) |

Лидеры [Image Edit Arena](https://arena.ai/leaderboard/image-edit) на 6 октября:
GPT-Image-2.5 Sunburst (1524), GPT-Image-2.5 Flare (1481), GPT-Image-2
(1462), Grok Imagine Image 2.0 (1439), Nano Banana 2.1 (1428). MAI-Image-2.6
Preview был первым на
[лидерборде редактирования Artificial Analysis](https://x.com/ArtificialAnlys/status/2092284543705456991).
Арена оценивает фото и общие правки, не чистку манги, поэтому это только
ориентир.

### Локальные модели (только за последние два месяца)

| Модель | Вышла | Размер | Лицензия | На CPU владельца |
| --- | --- | --- | --- | --- |
| **FLUX 3 Image Klein** | 13 окт 2026 | 5,6B трансформер + текстовый кодер Qwen3-VL-4B; GGUF от 2,6 до 11 ГБ ([unsloth](https://huggingface.co/unsloth/flux-3-image-klein-GGUF)) | FLUX Kommunity: некоммерческое использование, коммерческий вывод при выручке < $5M | непрактично для целой страницы, см. ниже |
| **Qwen-Image 2.1** / **2.1 Turbo** | 14 сен / 9 окт 2026 | 7B; GGUF от 2,5 до 14 ГБ ([unsloth](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF)); Turbo — 8 шагов | Qwen Research License: только некоммерческое | то же |

**FLUX 3 Image Klein** ([карточка](https://huggingface.co/unsloth/flux-3-image-klein),
[код](https://github.com/black-forest-labs/flux-3-image-klein)) — проверено
по карточке модели:

- Дистиллированная версия делает 16 шагов, правки с одной или несколькими
  (до 10) входными картинками.
- Удаление объекта задаётся рамкой в запросе (`src_bbox` и `tgt_bbox: null`),
  но рамка «не жёсткая маска» и правка может задеть область вне неё. Для нас
  это не страшно: результат вклеивается по нашей маске.
- Короткий запрос как есть даёт «общую» картинку и может перекадрировать
  сцену. Модели нужен полный разбор кадра от «улучшителя запроса» (отдельная
  9B-модель или любой LLM с их системным запросом). Через API BFL это,
  видимо, делается на сервере — не проверено.
- Пиковая память на GPU при 1024×1024: 25–31 ГБ. Квантованные версии
  заявлены для видеокарт с 8 ГБ.
- Эталонный CLI всегда грузит NSFW-классификатор (Falconsai); в примере для
  diffusers его нет. Значит, при своём запуске (дома или на арендованном GPU)
  фильтра 18+ нет.

**Qwen-Image 2.1** ([карточка](https://huggingface.co/Qwen/Qwen-Image-2.1)) —
проверено по карточке модели:

- Генерация и правка в одной модели, до 10 входных картинок.
- **Локальная правка по отдельной маске**, кругу или пометке кистью — ровно
  то, что даёт наш конвейер. Из всех свежих моделей только у неё маска есть
  прямо в описании.
- Есть «uncensored» сборки сообщества
  ([пример](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized))
  и текстовый кодер без отказов. Базовые веса и так не фильтруют вывод.
- Официального API нет. Цены $0,02 за 1K и $0,04 за 2K у kie.ai и reAPI —
  по сторонним источникам; у Fal и Replicate есть только Qwen Image Edit
  2511 и более ранние.

**CPU-only Windows (i5-1235U).** Ни одну из двух моделей на CPU не мерили.
Оценка: FLUX.2 Klein 4B, который уже есть в Koharu, на этом ПК медленный;
5,6–7B при 8–16 шагах на целой странице займёт порядка десятков минут.
Реалистичный вариант — только вырезки вокруг звуков (512 px) и Qwen-Image
2.1 Turbo с 8 шагами. Иначе локальные веса имеют смысл только на
арендованном GPU, где нет фильтра провайдера. Перед любым портом нужен
замер на одной вырезке.

Старые локальные модели (LaMa, AOT, FLUX.2 Klein 4B, RORem) владелец уже
перепробовал, они остаются в реестре как есть.

## Что пишут в сообществе

Сравнений моделей именно на чистке манги почти нет: найденное — карточки
моделей, обзоры перепродавцов и issues соседних проектов.

- [MangaTranslator](https://github.com/meangrinch/MangaTranslator) чистит
  пузыри и текст на фоне через FLUX.2 Klein или FLUX.1 Kontext, с OpenCV
  как запасным вариантом. Это самый близкий к нам проект, использующий
  генеративную правку для клининга.
- [Qwen-Image #250](https://github.com/QwenLM/Qwen-Image/issues/250): при
  замене текста в ёнкоме через чат Qwen около четверти знаков вышли
  испорченными, а подписи «Рассказчик:» модель приняла за текст для вставки.
  Вывод для нас: просить только стирание, не перерисовку текста.
- [manga-image-translator #933](https://github.com/zyddnys/manga-image-translator/issues/933):
  классический клинер иногда возвращает картинку без изменений. Нужна
  проверка остатков текста после клининга.
- LoRA для удаления объектов:
  [Qwen-Image-Edit-2511 Object Remover](https://huggingface.co/prithivMLmods/Qwen-Image-Edit-2511-Object-Remover),
  [FLUX.2 Klein 4B object remove](https://huggingface.co/fal/flux-2-klein-4B-object-remove-lora)
  (обучена на 100 парах). Готовой LoRA для текста в манге не нашлось.
- Seedream 5: по отзывам тестеров, официальный API отказывает даже на лёгком
  контенте, который 4.5 пропускал, а правки проверяются строже генерации
  ([разбор](https://www.atlascloud.ai/blog/guides/does-seedream-5-pro-allow-nsfw)).
  Перепродавцы ([SpicyAPI](https://spicyapi.ai/blog/is-seedream-uncensored))
  пишут, что у них проходит всё, но это их же реклама.
- Nano Banana: даже со всеми порогами `BLOCK_NONE` готовая картинка
  проверяется ещё раз и запрос падает с IMAGE_SAFETY
  ([разбор ошибок](https://help.apiyi.com/en/nano-banana-pro-image-safety-error-analysis-fix-guide-en.html)).
- По FLUX 3 Image пока нет независимых проверок того, насколько он
  сохраняет нетронутые области; обещание есть только в анонсе
  ([AI Frontier Post](https://aifrontierpost.com/articles/flux-3-image-black-forest-labs-launch/)).

## Запросы для проверки

Сейчас по умолчанию стоит длинный запрос из `DEFAULT_CLEANUP_PROMPT`
(`crates/koharu-pipeline/src/stages/fal.rs`). Варианты для сравнения:

**P0, текущий.** Без изменений, как база.

**P1, короткий и запретительный** (для MAI, Qwen, Grok, GPT, Nano Banana —
моделей, которые слушают инструкции):

```text
Erase all lettering from this manga page: dialogue, captions, and sound effects, including outlined, rotated, or hand-drawn ones. Fill each erased area with the artwork, screentone, or flat color that surrounds it. Do not add any new text, symbols, or speech bubbles. Do not change anything else: keep every line, character, panel border, screentone, and bubble outline pixel-identical. Output the same page at the same size and framing.
```

**P2, только звуки поверх рисунка** (для страницы, где пузыри уже очищены
локально):

```text
This manga page still has sound-effect lettering drawn over the artwork. Remove only that lettering and redraw the artwork, lines, and screentone behind it so it continues naturally. The speech bubbles are already empty: leave them empty and do not put any text, marks, or symbols inside them. Do not change anything else on the page.
```

**P3, по пометке цветом.** Перед отправкой обвести маску текста ярко-пурпурным
контуром (#FF00FF, 3 px), запрос:

```text
Text inside the magenta outlines must be erased. Remove the magenta outlines and the lettering inside them, and reconstruct the artwork underneath. Leave everything outside the outlines unchanged.
```

Пометка помогает модели найти мелкие звуки, но если модель оставит следы
пурпура, вклейка по маске их всё равно уберёт; проверить, не тянет ли
пурпурный оттенок в соседние пиксели.

**P4, для FLUX 3 Image / Klein.** Модели нужен полный разбор кадра. Вместо
одной фразы — описание страницы плюс задача:

```text
The same black-and-white manga page, unchanged in layout, framing, panel borders, characters, line art, and screentone, with all Japanese lettering removed: speech bubbles are empty white shapes with their original outlines and tails, and areas that had sound-effect lettering show the continued artwork and screentone behind them. No text anywhere on the page.
```

и вариант с рамками удаления (`src_bbox` → `tgt_bbox: null`) по рамкам
детекции. Сравнить с подачей того же запроса через «улучшитель» BFL.

**P5, Qwen-Image 2.1 с маской.** Маска из конвейера отдельной картинкой и
запрос `Remove the text in the masked area and restore the background.`
Только при локальном или арендованном запуске.

## План тестов

### Набор страниц

- Глава владельца, страницы 2–27 (обложки 1 и 28 не берём), на копии
  проекта.
- Отдельно помечаются: страницы 18+ (в том числе 19–22), страницы с крупными
  звуками поверх рисунка, тёмные пузыри и текст с обводкой.
- Эталон: 11 страниц, которые владелец чистил кистью сам, и 14 страниц,
  очищенных через Fal.

### Метрики (считаются скриптом, бесплатно)

1. **Остатки текста:** доля тёмных пикселей внутри маски текста после
   клининга относительно её окружения. Порог настроить на ручной чистке
   владельца.
2. **Порча вне маски** (для режима «вся страница»): разница с оригиналом вне
   расширенной маски. Отдельно — пиксели контура пузыря (та же метрика, что
   уже дала 4282 → 1739 на главе).
3. **Новый текст:** OCR (Hayai) по пустым пузырям и вычищенным областям
   после клининга. Любые найденные знаки — ошибка.
4. **Отказы:** доля страниц с отказом модерации, отдельно на 18+ и на
   обычных страницах; чёрная картинка от Fal тоже считается отказом.
5. **Цена и время** на страницу по фактическим счетам провайдера.
6. **Слепой выбор владельца** на 8–10 спорных страницах: две версии рядом,
   без названия модели.

### Режимы подачи

- **Вся страница, результат целиком** (сейчас по умолчанию у Fal).
- **Вся страница, вклейка по маске** (`apply_mode = mask`): рисунок вне
  текста не может «поплыть».
- **Вырезки вокруг звуков** (512–1024 px). Гипотеза: на вырезке меньше
  откровенного контекста, поэтому меньше отказов, и это дешевле.

### Сравнение «пузыри уже пустые» и «нет»

Это отдельный эксперимент на каждой модели-финалисте:

- **A:** в облако уходит исходная страница с текстом в пузырях и звуками,
  запрос P0 или P1.
- **B:** сначала локальный клининг пузырей (однотонная заливка своим цветом
  и LaMa по маске внутри пузыря), потом в облако уходит страница, где
  остались только звуки, запрос P2.
- Одинаковые страницы, модель, seed (если есть) и режим вклейки.
- Сравниваем остатки звуков, порчу рисунка вне маски, новый текст в пустых
  пузырях (гипотеза: модель может «дописать» пустой пузырь), отказы и цену.
- Решение: если B не хуже по звукам и не дописывает пузыри, делаем B
  режимом по умолчанию — он дешевле в токенах запроса и сохраняет пузыри
  точно.

### Порядок и бюджет

Сначала пилот на 8 страницах (4 обычных, страницы 19–22 из 18+), два
варианта подачи (A и B):

| Шаг | Модели | Вызовов | Примерно |
| --- | --- | --- | --- |
| 1 | Qwen Image Edit 2511 (Replicate, проверка выключена), FLUX 3 Image (Replicate, `safety_tolerance` 4), Grok Imagine Image 2.0 | 3 × 8 × 2 = 48 | $1,5–2 |
| 2 | FLUX 3 Image Klein (API), MAI Image 2.6 Flash, Nano Banana 2.1, GPT-Image-2.5 medium — последние три только как потолок качества на обычных страницах | ~40 | $1–1,5 |
| 3 | Финалисты на всех 26 страницах, A и B, 2 запроса | ~200 | $5–8 |

Итого около $10 на всё. Каждый шаг начинается только после разрешения на
бюджет.

Для шага 1 нужны небольшие правки кода: id `black-forest-labs/flux-3-image`
и `qwen/qwen-image-edit-2511` в каталоге Replicate и передача
`safety_tolerance`. FLUX 3 Image, Grok, MAI, Nano Banana и GPT-Image уже
можно гонять через OpenRouter по свободному id модели, но без управления
модерацией.

### Что можно сделать до доступа к тестам

- Скрипт метрик (остатки, порча, новый текст) на уже очищенных страницах:
  он же сразу оценит текущий Fal на 14 страницах владельца.
- Явный переключатель фильтра 18+ у Replicate и разное сообщение для отказа
  модерации и сетевой ошибки (план из [results.ru.md](results.ru.md)).
- Запрос в поддержку Fal на разрешение `enable_safety_checker: false` для
  аккаунта владельца: тогда Qwen 2511, FLUX.2 Klein 9B и Kontext на Fal
  тоже станут пригодны для 18+.

## Источники

- FLUX 3 Image: [анонс (Dataist)](https://dataist.ai/news/black-forest-labs-releases-flux-3-image-with-staged-editing/),
  [Replicate](https://replicate.com/black-forest-labs/flux-3-image),
  [документация BFL](https://docs.bfl.ai/flux_3/flux3_image_overview),
  [цены BFL](https://docs.bfl.ai/quick_start/pricing),
  [OpenRouter](https://openrouter.ai/black-forest-labs/flux-3-image).
- FLUX 3 Image Klein: [блог BFL](https://bfl.ai/blog/flux-3-image-klein),
  [карточка (зеркало без гейта)](https://huggingface.co/unsloth/flux-3-image-klein),
  [GGUF](https://huggingface.co/unsloth/flux-3-image-klein-GGUF),
  [VentureBeat о лицензии](https://venturebeat.com/technology/black-forest-labs-opens-flux-3-image-klein-weights-with-free-commercial-use-limited-to-smaller-businesses),
  [цены Klein у BFL (Digital Applied)](https://www.digitalapplied.com/blog/flux-3-image-klein-open-weights-8gb-gpu).
- Qwen-Image 2.1: [карточка](https://huggingface.co/Qwen/Qwen-Image-2.1),
  [Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo),
  [MarkTechPost](https://www.marktechpost.com/2026/09/21/alibaba-qwen-releases-qwen-image-2-1/),
  [API и лицензия (eesel)](https://www.eesel.ai/blog/qwen-image-2-1),
  [kie.ai](https://kie.ai/qwen-image-2.1).
- MAI Image 2.6: [OpenRouter](https://openrouter.ai/microsoft/mai-image-2.6),
  [Flash](https://openrouter.ai/microsoft/mai-image-2.6-flash),
  [Windows Report](https://windowsreport.com/microsoft-releases-mai-image-2-6-surges-to-no-2-on-arena/).
- Grok Imagine Image 2.0: [OpenRouter](https://openrouter.ai/x-ai/grok-imagine-image-2.0),
  [WaveSpeed](https://wavespeed.ai/blog/ai-models/grok-imagine-image-2-0-api/).
- GPT-Image-2.5: [eesel о ценах](https://www.eesel.ai/blog/chatgpt-images-2-5-pricing),
  [Atlas Cloud](https://www.atlascloud.ai/blog/tips/gpt-image-2.5-api-cost).
- Nano Banana 2.1: [Apidog](https://apidog.com/blog/nano-banana-2-1/),
  [сравнение с GPT Image 2.5](https://dev.to/akartit/nano-banana-21-vs-gpt-image-25-8-prompts-real-cost-per-image-87i).
- Seedream 5: [Atlas Cloud](https://www.atlascloud.ai/blog/guides/does-seedream-5-pro-allow-nsfw),
  [Wiro](https://wiro.ai/blog/best-nsfw-ai-api-2026/).
- Fal и Replicate, фильтры: [прошлое исследование](../research/2026-09-13-image-edit-safety-checkers.md),
  [Fal Qwen 2511 API](https://fal.ai/models/fal-ai/qwen-image-edit-2511/api),
  [Replicate Qwen 2511](https://replicate.com/qwen/qwen-image-edit-2511).
- Хостинг Klein: [Runware](https://runware.ai/models),
  [DeepInfra](https://deepinfra.com/flux).
