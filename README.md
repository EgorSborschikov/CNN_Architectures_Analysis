# Сравнение архитектур для распознавания лиц в SKUD

Проект сравнительного анализа трех архитектур нейронных сетей для задачи верификации лиц в системах контроля доступа.

## Архитектуры
1. **Классическая CNN** - Многоклассовая классификация
2. **Сиамская сеть** - Бинарная классификация пар
3. **Триплетная сеть** - Метрическое обучение

## Структура проекта
```shell
FaceRecognitionModels/
├── data/ # Датасеты
├── src/ # Исходный код
├── experiments/ # Результаты
├── models/ # Сохраненные модели
└── configs/ # Конфигурации
```


## Установка
```bash
git clone <repository-url>
cd FaceRecognitionModels
python -m venv venv
source venv/bin/activate  # или venv\Scripts\activate на Windows
pip install -r requirements.txt
```

## Использование
```shell
# Запуск всех экспериментов
python main.py

# Только CNN
python main.py --model cnn --dataset lfw

# С определенным датасетом
python main.py --dataset celeba
```

## Результаты
Результаты сохраняются в ./experiments/:
- [x] Графики обучения
- [x] Таблицы сравнения
- [x] Матрицы ошибок
- [x] Сохраненные модели