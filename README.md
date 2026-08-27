# Titanic — Survival Prediction (Classic ML)

Решение задачи бинарной классификации (соревнование Kaggle Titanic): предсказать, выжил ли пассажир, по демографическим и билетным признакам.

## Результаты

Лучшая модель по итогам стратифицированной 5-fold кросс-валидации — **CatBoost** (см. `notebook/drafts&eda.ipynb`, раздел "Бустинги"). Полная таблица метрик по всем обученным моделям сохраняется в `model_info.csv` при запуске `main.py`.

Итоговый график сравнения моделей по CV score строится автоматически в конце пайплайна.

## Структура репозитория

```
├── data/                   # train.csv, test.csv, gender_submission.csv
├── notebook/
│   └── drafts&eda.ipynb    # EDA, feature engineering, подбор моделей и гиперпараметров
├── src/
│   ├── config/             # global_config.yaml (общие параметры), param_config.yaml (гиперпараметры моделей)
│   ├── models/             # реестр моделей (models.py) и менеджер выбора модели (ModelsManager.py)
│   ├── training/           # PipelineRunner (оркестрация), Validater (k-fold валидация)
│   ├── utils/               # вспомогательные функции: сплиты, заполнение пропусков, метрики
│   └── DataManager.py      # загрузка и препроцессинг данных
├── dnn/                    # PyTorch-реализация: Dataset, DNNClassifier (MLP), Trainer
└── main.py                 # точка входа
```

## Конфиг
Гиперпараметры всех моделей вынесены в `src/config/param_config.yaml`, общие настройки пайплайна (использовать все модели или только лучшую, версия набора фичей, random state и т.д.) — в `src/config/global_config.yaml`.

## Запуск

```bash
pip install -r requirements.txt   
python main.py
```



## Ноутбук

`notebook/drafts&eda.ipynb` содержит полный цикл экспериментов: EDA с текстовыми выводами по каждому графику, проверку гипотез по признакам (`Cabin`, `FamilySize`), подбор гиперпараметров для каждой модели через `GridSearchCV`, обучение и сравнение бустингов, обучение DNN с кросс-валидацией.
