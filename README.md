# ДЗ 3 · Классификация тональности отзывов

RuReviews (отзывы на женскую одежду и аксессуары), три класса: negative, neutral, positive.

Автор: Артем Шарипов.

ЛЛМ использовалась для создания данного README.md файла (ну не руками же этот файл писать лол)

Решение — в ноутбуке `hw3_Artem_Sharipov.ipynb`. Он проверяет данные, обучает TF-IDF с логистической регрессией и подбирает порог ручной проверки, дообучает `cointegrated/rubert-tiny2`, сравнивает промпты для `Qwen/Qwen2.5-0.5B-Instruct`, оценивает всё на test, повторно загружает ruBERT с замером времени и проверяет генератор на prompt injection. Ответы на контрольные вопросы — в `answers.md`.

## Результаты

| Модель | Данные | macro-F1 |
|---|---|---|
| TF-IDF, словные 1–2-граммы | validation, 8 578 | 0,727 |
| TF-IDF, символьные 2–5-граммы — **выбран** | validation, 8 578 | 0,738 |
| ruBERT, `checkpoint-750` — **выбран** | validation, 8 578 | 0,740 |
| Qwen zero-shot — **выбран** / few-shot | validation, 30 | 0,581 / 0,581 |
| TF-IDF char / ruBERT | test, 8 578 | 0,734 / 0,736 |
| TF-IDF char / ruBERT / Qwen zero-shot | test, 150 | 0,736 / 0,730 / 0,584 |

- Ручная проверка: отзыв уходит на проверку при P(negative) ≥ 0,55 по модели TF-IDF. На validation precision 0,817 и recall 0,579, на test — 0,799 и 0,581.
- ruBERT против TF-IDF на test: разница macro-F1 +0,0018 (95% ДИ [−0,0064; 0,0105]), F1 negative +0,0126 ([0,0017; 0,0234]).
- Все 222 ответа генератора валидны. Инструкция внутри отзыва меняет метку на positive в 2 парах из 3 при обоих способах передачи текста.
- ruBERT, загруженный в новом процессе, даёт те же вероятности (разница 0). Прямой проход на CPU занимает 0,8 мс на отзыв при батче 16.
- Для первого запуска рекомендую символьный TF-IDF с порогом 0,55; ruBERT — следующий кандидат.

## Запуск

Окружение: macOS 26.6 (arm64), Apple M5 Pro, только CPU. Python 3.12.14, torch 2.15.0, transformers 4.57.1, datasets 4.2.0, scikit-learn 1.7.2.

```bash
git clone https://github.com/meowser52/ml_Avito_practice_course.git && cd ml_Avito_practice_course
uv venv --seed --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt ipykernel nbconvert
jupyter nbconvert --to notebook --execute --inplace hw3_Artem_Sharipov.ipynb
```

При первом запуске `prepare_data.py` скачивает RuReviews, а ноутбук — модели с Hugging Face, всего около 1,1 ГБ. Полный прогон идёт около 6 минут, из них около 3,5 минуты — обучение ruBERT. Логи обучения открываются командой `tensorboard --logdir results/tensorboard`.

## Файлы

- `hw3_Artem_Sharipov.ipynb` — решение;
- `answers.md` — ответы на контрольные вопросы;
- `reload_and_latency.py` — повторная загрузка и замер времени в отдельном процессе (файл создаёт ноутбук);
- `results/`:
  - `summary.csv`, `metrics.json` — метрики;
  - `selection.json` — выбор, зафиксированный до оценки на test;
  - `predictions_validation.csv` — предсказания обоих вариантов TF-IDF и ruBERT по ID отзывов, вероятности символьного TF-IDF и ruBERT;
  - `predictions_test.csv` — предсказания и вероятности выбранного TF-IDF и ruBERT по ID отзывов, флаг ручной проверки `manual_review`;
  - `threshold_validation.csv` — перебор порогов;
  - `generator_validation.jsonl` (60), `generator_test.jsonl` (150), `injection.jsonl` (12) — исходные ответы генератора;
  - `tensorboard/` — логи ruBERT: loss, validation macro-F1 и примеры ошибок во вкладке Text;
  - `reload_reference.json`, `reload_check.json`, `latency.json`, `latency.csv` — повторная загрузка и замер времени;
- `requirements.txt`, `prepare_data.py`, `split_ids.json` — зависимости, скачивание данных и фиксированное разбиение.

Веса моделей (`artifacts/`) и данные (`data/`) в репозиторий не входят: их создаёт запуск ноутбука.

## Примечания

- Модели закреплены по ревизиям на Hugging Face: `rubert-tiny2` — `e8ed3b0`, `Qwen2.5-0.5B-Instruct` — `7ae5576`. Повторные полные прогоны дают одинаковые метрики, предсказания и ответы генератора, различаются только замеры времени.
- Метки RuReviews выставлены по оценкам покупателей, поэтому часть ошибок — шум разметки. Например, `ru-21250` размечен как negative, хотя текст — «Юбка супер, оооочень довольна…».
