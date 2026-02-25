# AI SQL Query Agent

Интеллектуальный агент для преобразования запросов на естественном языке в SQL-запросы и их выполнения в PostgreSQL.

**Текущая версия:** 1.0.0  
**Статус:** Активная разработка  
**Последнее обновление:** 25.02.2026

## Описание

Этот проект представляет собой нейросетевой агент, который:
- Преобразует запросы на естественном языке в SQL-запросы
- Выполняет SQL-запросы к PostgreSQL базе данных
- Обеспечивает защиту от SQL-инъекций
- Поддерживает обучение на пользовательских данных

## Архитектура

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   NLProcessor   │    │   Seq2SeqModel  │    │ DatabaseConnector│
│                 │    │                 │    │                 │
│ • Токенизация   │───▶│ • Энкодер       │───▶│ • Подключение к │
│ • Предобработка │    │ • Декодер       │    │   PostgreSQL    │
│ • Векторизация  │    │ • Обучение      │    │ • Выполнение    │
└─────────────────┘    └─────────────────┘    │   запросов      │
                                                └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐              ▼
│   QueryBuilder  │    │  ResponseParser │    ┌─────────────────┐
│                 │    │                 │    │     Agent       │
│ • Постобработка │◀───│ • Форматирование│◀───│ • Оркестрация   │
│ • Валидация     │    │ • Вывод         │    │ • Логирование   │
│ • Оптимизация   │    │ • Ошибки        │    │ • Конфигурация  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Возможности

- 🤖 **Нейросетевое преобразование** NL → SQL с использованием Seq2Seq архитектуры
- 🔄 **Обучение на пользовательских данных** с поддержкой дообучения
- 🛡️ **Защита от SQL-инъекций** через валидацию и санитизацию запросов
- 📊 **Множество форматов вывода** (TABLE, JSON, CSV, PLAIN)
- 📝 **Подробное логирование** всех операций
- ⚙️ **Гибкая конфигурация** через конфигурационные файлы
- 🎯 **Поддержка нескольких моделей** (bookings, multidomain)
- 📈 **Прогресс-бар обучения** с мониторингом метрик

## Требования

### Системные требования
- **Операционная система**: Linux (Ubuntu 20.04+, Debian 10+, Arch Linux)
- **Процессор**: x86_64 с поддержкой AVX2 (рекомендуется)
- **Оперативная память**: 8 ГБ (минимум), 16 ГБ (рекомендуется)
- **Место на диске**: 2 ГБ (для моделей и данных)

### Программные зависимости
- **C++17** или выше
- **CMake 3.15+**
- **PostgreSQL 12+**
- **libpqxx** (C++ PostgreSQL client library)
- **nlohmann/json** (JSON library)
- **PyTorch C++** (libtorch) - для нейросетевых операций

### Рекомендуемые зависимости
- **CUDA 11.7+** (для ускорения обучения на GPU)
- **cuDNN 8.0+** (для оптимизации нейросетей)

## Установка зависимостей

### Ubuntu/Debian/Astra linux
```bash
sudo apt-get install libpqxx-dev nlohmann-json3-dev postgresql-server-dev-all
```

### Arch linux
```bash
sudo pacman -S libpqxx nlohmann-json postgresql-libs
```

## Сборка

```bash
cmake -S /home/andrew/Projects/Ai_c-_bot_psql -B /home/andrew/Projects/Ai_c-_bot_psql/build
cmake --build /home/andrew/Projects/Ai_c-_bot_psql/build -j
```

## Запуск нейросети (обучение/инференс)

- Инференс на предобученной модели (без обучения):
```bash
/home/andrew/Projects/Ai_c-_bot_psql/build/bin/train_model 0 0.001 --resume
```

- Дообучение с чекпойнта (пример: 20 эпох):
```bash
/home/andrew/Projects/Ai_c-_bot_psql/build/bin/train_model 20 0.001 --resume
```

- Обучение с нуля (пример: 50 эпох):
```bash
/home/andrew/Projects/Ai_c-_bot_psql/build/bin/train_model 50 0.001
```

- Альтернативный тренер из `training_data/` (аналогичный интерфейс):
```bash
/home/andrew/Projects/Ai_c-_bot_psql/build/bin/train_model_data 0 0.001 --resume
```

## Настройка базы данных

Перед запуском основного приложения необходимо настроить базу данных PostgreSQL.

### Автоматическая настройка (рекомендуется)

```bash
./setup_database.sh
```

Скрипт автоматически:
- Создаст пользователя `ai_user` с паролем `123`
- Создаст базу данных `ai_db`
- Инициализирует схему и заполнит тестовыми данными

**Требуются права sudo** для создания пользователя и БД.

### Ручная настройка

Если нет прав sudo, выполните от имени пользователя postgres:

```bash
# 1. Создать пользователя
sudo -u postgres psql -c "CREATE USER ai_user WITH PASSWORD '123';"

# 2. Создать базу данных
sudo -u postgres psql -c "CREATE DATABASE ai_db OWNER ai_user;"

# 3. Инициализировать схему
PGPASSWORD=123 psql -h localhost -U ai_user -d ai_db -f init_database.sql
```

### Проверка подключения

```bash
PGPASSWORD=123 psql -h localhost -U ai_user -d ai_db -c "SELECT COUNT(*) FROM users;"
```

Должно вернуть количество пользователей (8 по умолчанию).

## Запуск основного приложения

```bash
/home/andrew/Projects/Ai_c-_bot_psql/build/bin/AIQueryAgent
```

После запуска вы сможете:
- Использовать команду `query <текст>` для преобразования NL в SQL и выполнения
- Использовать команду `sql <запрос>` для прямого выполнения SQL
- Просматривать таблицы командой `tables`

**Примеры:**
```
> query show all users
> query count all products
> query get 5 cheapest products
> sql SELECT * FROM users LIMIT 3
```

## Быстрый тест окружения PyTorch C++

```bash
/home/andrew/Projects/Ai_c-_bot_psql/build/bin/test_model
```

## Использование обученной модели (инференс)

После обучения модель сохраняется в `models/seq2seq_model_*.pt`. Для использования последней обученной версии:

### Через train_model (инференс без обучения)

```bash
# 0 эпох = только инференс, без обучения
./build/bin/train_model 0 0.001 --resume
```

После загрузки модель покажет тестовые примеры.

### Через основное приложение AIQueryAgent

```bash
./build/bin/AIQueryAgent
```

Затем в интерактивном режиме:
```
> query Show all users
> query Count all products
> query Get 5 cheapest products
```

**Важно:** Убедитесь, что в `src/config/default.conf` указан правильный путь к модели:
```
model_path=models/seq2seq_model
```

### Проверка наличия обученной модели

Убедитесь, что файлы модели существуют:

```bash
ls -lh models/seq2seq_model_*.pt models/seq2seq_model_*.txt
```

Должны быть:
- `models/seq2seq_model_encoder.pt` - веса энкодера
- `models/seq2seq_model_decoder.pt` - веса декодера  
- `models/seq2seq_model_nl_vocab.txt` - словарь NL
- `models/seq2seq_model_sql_vocab.txt` - словарь SQL

## Подробные инструкции по обучению нейросети

### Подготовка датасета

Датасет уже содержит **10000 примеров** SQL-запросов в файле `training_data/nl_to_sql_train.json`.

Если нужно расширить датасет дополнительно, используйте скрипт:

```bash
cd training_data
python3 expand_dataset.py
```

### Обучение модели

#### 1. Обучение с нуля (рекомендуется для первого запуска)

Обучает модель на полном датасете из 10000 примеров:

```bash
# 50 эпох, learning rate 0.001
./build/bin/train_model 50 0.001

# Или с альтернативным тренером
./build/bin/train_model_data 50 0.001
```

**Параметры:**
- Первый аргумент: количество эпох (рекомендуется 50-100 для первого обучения)
- Второй аргумент: learning rate (рекомендуется 0.001)

**Время обучения:** ~30-40 минут на CPU для 50 эпох

#### 2. Дообучение существующей модели

Продолжает обучение с сохраненного чекпойнта:

```bash
# 20 дополнительных эпох
./build/bin/train_model 20 0.001 --resume

# Или с альтернативным тренером
./build/bin/train_model_data 20 0.001 --resume
```

**Флаг `--resume`:** загружает модель из `models/seq2seq_model_encoder.pt` и `models/seq2seq_model_decoder.pt`

#### 3. Инференс (тестирование без обучения)

Проверяет работу обученной модели на тестовых примерах:

```bash
# 0 эпох = только инференс
./build/bin/train_model 0 0.001 --resume
```

### Сохранение модели

После обучения модель автоматически сохраняется в:
- `models/seq2seq_model_encoder.pt` - веса энкодера
- `models/seq2seq_model_decoder.pt` - веса декодера
- `models/seq2seq_model_nl_vocab.txt` - словарь естественного языка
- `models/seq2seq_model_sql_vocab.txt` - словарь SQL

### Мониторинг обучения

Во время обучения вы увидите:
- Прогресс-бар для каждой эпохи
- Средний loss после каждой эпохи
- Примеры предсказаний каждые 5 эпох
- Время выполнения каждой эпохи

**Пример вывода:**
```
Epoch 1/50 [==================================================] 100% (35s) Loss: 2.3456
Epoch 2/50 [==================================================] 100% (33s) Loss: 1.8923
...
Example predictions after epoch 5:
  NL: Count all products
  Pred SQL: select count( *) from products
  True SQL: SELECT COUNT(*) FROM products
```

### Рекомендации по обучению

1. **Первое обучение:** 50-100 эпох с learning rate 0.001
2. **Дообучение:** 10-20 эпох с тем же learning rate
3. **Fine-tuning:** 5-10 эпох с меньшим learning rate (0.0001)
4. **Мониторинг:** Следите за loss - он должен уменьшаться. Если loss перестал падать, можно остановить обучение.

### Примеры команд для разных сценариев

```bash
# Быстрый тест (инференс)
./build/bin/train_model 0 0.001 --resume

# Короткое обучение (10 эпох)
./build/bin/train_model 10 0.001

# Полное обучение (100 эпох)
./build/bin/train_model 100 0.001

# Дообучение с меньшим learning rate
./build/bin/train_model 20 0.0001 --resume
```

## Структура проекта

```
src/
├── config/           # Конфигурационные файлы и классы
│   ├── Config.cpp    # Реализация конфигурации
│   ├── Config.h      # Заголовочный файл конфигурации
│   ├── default.conf  # Конфигурация по умолчанию
│   └── 1M_model.conf # Конфигурация для 1M модели
├── core/             # Основные компоненты системы
│   ├── Agent.cpp     # Главный агент системы
│   ├── Agent.h       # Заголовочный файл агента
│   ├── DatabaseConnector.cpp  # Подключение к БД
│   ├── DatabaseConnector.h
│   ├── QueryBuilder.cpp       # Построение SQL-запросов
│   ├── QueryBuilder.h
│   ├── ResponseParser.cpp     # Парсинг ответов
│   └── ResponseParser.h
├── ml/               # Машинное обучение
│   ├── ModelTrainer.cpp       # Тренировка моделей
│   ├── ModelTrainer.h
│   ├── Seq2SeqModel.cpp       # Реализация Seq2Seq модели
│   ├── Seq2SeqModel.h
│   ├── Vocabulary.cpp         # Работа со словарями
│   └── Vocabulary.h
├── nlprocessor/      # Обработка естественного языка
│   ├── NLProcessor.cpp
│   └── NLProcessor.h
└── utils/            # Вспомогательные утилиты
    ├── Logger.cpp    # Система логирования
    ├── Logger.h
    └── Utilities.h   # Различные утилиты

training_data/        # Обучающие данные
├── nl_to_sql_train.json      # Основной датасет
├── bookings_nl_to_sql_from_db.json  # Датасет для bookings
├── bookings_nl_to_sql_1m_from_db.json # Большой датасет
├── generate_bookings_dataset_from_db.cpp  # Генератор датасета
└── queries.json              # Примеры запросов

models/               # Обученные модели
├── seq2seq_bookings_tiny_*.pt    # Маленькая bookings модель
├── seq2seq_bookings_tiny_*.txt   # Словари для bookings
├── seq2seq_multidomain_1m_*.pt   # Большая multidomain модель
└── seq2seq_multidomain_1m_*.txt  # Словари для multidomain
```

## Предобученные модели

Проект поставляется с двумя предобученными моделями:

### 1. Bookings Tiny Model
- **Назначение**: Базовая модель для тестирования
- **Размер**: Маленькая (до 1000 примеров)
- **Файлы**:
  - `models/seq2seq_bookings_tiny_encoder.pt`
  - `models/seq2seq_bookings_tiny_decoder.pt`
  - `models/seq2seq_bookings_tiny_nl_vocab.txt`
  - `models/seq2seq_bookings_tiny_sql_vocab.txt`

### 2. Multidomain 1M Model
- **Назначение**: Производственная модель с большим датасетом
- **Размер**: Большая (1 миллион примеров)
- **Файлы**:
  - `models/seq2seq_multidomain_1m_encoder.pt`
  - `models/seq2seq_multidomain_1m_decoder.pt`
  - `models/seq2seq_multidomain_1m_nl_vocab.txt`
  - `models/seq2seq_multidomain_1m_sql_vocab.txt`

### Использование моделей

Для использования конкретной модели обновите конфигурацию:

```bash
# Для bookings модели
cp src/config/1M_model.conf src/config/default.conf
# Затем вручную измените пути в default.conf на bookings_tiny

# Для multidomain модели
cp src/config/1M_model.conf src/config/default.conf
```

## Конфигурация

Проект поддерживает гибкую конфигурацию через файлы `.conf`. Основные параметры:

```ini
# Подключение к базе данных
db_host=localhost
db_port=5432
db_name=ai_db
db_user=ai_user
db_password=123

# Пути к моделям
model_path=models/seq2seq_model
encoder_path=models/seq2seq_model_encoder.pt
decoder_path=models/seq2seq_model_decoder.pt
nl_vocab_path=models/seq2seq_model_nl_vocab.txt
sql_vocab_path=models/seq2seq_model_sql_vocab.txt

# Параметры модели
hidden_size=256
embedding_size=128
dropout=0.1
max_length=50

# Параметры обучения
batch_size=32
learning_rate=0.001
num_epochs=50
```

## Разработка

### Добавление новых моделей

1. Создайте новую конфигурацию в `src/config/`
2. Обновите `src/config/Config.h` для поддержки новой модели
3. Добавьте соответствующие файлы модели в `models/`

### Расширение функциональности

Проект спроектирован с учетом расширяемости:

- **NLProcessor**: Для улучшения обработки естественного языка
- **QueryBuilder**: Для добавления новых типов SQL-запросов
- **ResponseParser**: Для поддержки новых форматов вывода
- **DatabaseConnector**: Для поддержки других СУБД

### Тестирование

```bash
# Тест PyTorch C++
./build/bin/test_model

# Тестирование модели
./build/bin/train_model 0 0.001 --resume

# Запуск основного приложения
./build/bin/AIQueryAgent
```

## Производительность

### Характеристики производительности

| Компонент | CPU (Intel i7-12700K) | GPU (RTX 3080) | Память |
|-----------|------------------------|----------------|---------|
| Обучение (50 эпох) | 30-40 мин | 3-5 мин | 8-16 ГБ |
| Инференс (1 запрос) | 100-500 мс | 10-50 мс | 512 МБ-2 ГБ |
| Загрузка модели | 5-10 с | 2-5 с | 1-3 ГБ |

### Оптимизация

- **GPU ускорение**: Используйте CUDA для ускорения обучения в 6-10 раз
- **Batch processing**: Для массовой обработки используйте batch_size > 1
- **Модельная оптимизация**: Quantization и pruning для уменьшения размера модели
- **Кэширование**: Кэшируйте часто используемые запросы для ускорения ответов

## Лицензия

Этот проект находится под лицензией MIT. Подробности смотрите в файле LICENSE.

## Контакты

- Разработчик: VelesZero
- Репозиторий: https://github.com/VelesZero/Ai_c-_bot_psql.git
- Версия: 1.0.0
- Дата релиза: 25.02.2026
