[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Предсказание мошенничества в блокчейне Ethereum

**Студент:** Саликова Мария  
**Группа:** БИВ235

---

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Быстрый старт](#быстрый-старт)
4. [Запуск через Docker](#запуск-через-docker)
5. [Данные](#данные)
6. [Результаты](#результаты)
7. [Линтинг](#линтинг)
8. [Отчёт](#отчёт)

---

## Описание задачи

**Задача:** Бинарная классификация — определить, является ли адрес Ethereum мошенническим (фрод / не-фрод), на основе транзакционной активности в блокчейне.

**Датасет:** [Ethereum Fraud Detection Dataset](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset) (~9 000 адресов, ~50 признаков)

**Целевая метрика:** F1-score (класс 1 — фрод)

---

## Структура репозитория

```
.
├── data
│   ├── processed/              # Очищенные и обработанные данные (train/val/test.csv)
│   └── raw/                    # Исходные файлы (transaction_dataset.csv)
├── models/                     # Сохранённые модели (.pkl)
├── notebooks/
│   ├── 01_eda.ipynb            # EDA: распределения, boxplot, корреляции, зависимость от таргета
│   ├── 02_baseline.ipynb       # Baseline-модели (LR + RF, без feature engineering)
│   └── 03_experiments.ipynb    # Эксперименты: 5+ моделей, ансамбли, PCA, таблица, выводы
├── presentation/               # Презентация для защиты
├── report/
│   ├── images/                 # Графики и визуализации
│   └── report.md               # Финальный отчёт
├── src/
│   ├── preprocessing.py        # Очистка данных, feature engineering, сплит
│   └── modeling.py             # Обучение, оценка, сохранение моделей
├── tests/
│   └── test.py                 # Unit-тесты пайплайна
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml              # Конфигурация ruff (линтер)
├── requirements.txt
└── README.md
```

---

## Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Предобработать данные (нужен data/raw/transaction_dataset.csv)
cd src
python preprocessing.py

# 5. Запустить тесты
pytest tests/

# 6. Открыть ноутбуки
jupyter lab
```

---

## Запуск через Docker

```bash
# Собрать образ и запустить Jupyter Lab
docker-compose up --build

# Jupyter Lab будет доступен по адресу:
# http://localhost:8888

# Только предобработка данных
docker-compose --profile preprocess run preprocess
```

---

## Данные

- `data/raw/transaction_dataset.csv` — исходный датасет с Kaggle
- `data/processed/train.csv` — обучающая выборка (70%)
- `data/processed/val.csv` — валидационная выборка (15%)
- `data/processed/test.csv` — тестовая выборка (15%, не использовалась при подборе параметров)

**Пайплайн предобработки:**
1. Нормализация названий колонок
2. Удаление дубликатов (`drop_duplicates`)
3. Удаление нерелевантных колонок (Index, Address, токены с >50% None)
4. Заполнение пропусков нулями (ERC20-транзакции без данных)
5. Удаление константных признаков
6. Feature engineering: `ratio_sent_received`, `log_total_ether_*`
7. Стратифицированный сплит 70/15/15
8. StandardScaler, обученный только на train

---

## Результаты

| Модель | F1 (val) | ROC-AUC | Примечание |
|--------|----------|---------|------------|
| LR Baseline (no FE) | ~0.65 | ~0.87 | Нижняя граница |
| KNN | ~0.80 | ~0.92 | GridSearch k |
| Decision Tree | ~0.82 | ~0.93 | GridSearch depth |
| Random Forest | ~0.90 | ~0.99 | RandomizedSearch |
| Gradient Boosting | ~0.88 | ~0.98 | RandomizedSearch |
| XGBoost | ~0.91 | ~0.99 | RandomizedSearch |
| LightGBM | ~0.92 | ~0.99 | RandomizedSearch |
| VotingClassifier | ~0.90 | ~0.99 | soft voting |
| StackingClassifier | ~0.92 | ~0.99 | meta=LR |

*Точные значения — в `report/experiment_results.csv` и `notebooks/03_experiments.ipynb`*

---

## Линтинг

Проект использует **ruff** для линтинга и форматирования:

```bash
# Проверка кода
ruff check src/

# Автоматическое исправление
ruff check src/ --fix

# Форматирование
ruff format src/
```

Конфигурация: `pyproject.toml`

---

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
