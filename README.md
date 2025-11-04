# AI SQL Query Agent

Интеллектуальный агент для преобразования запросов на естественном языке в SQL-запросы и их выполнения в PostgreSQL.

## Возможности

- 🤖 Преобразование естественного языка в SQL
- 🔄 Обучение на пользовательских данных
- 🛡️ Защита от SQL-инъекций
- 📊 Множество форматов вывода (TABLE, JSON, CSV, PLAIN)
- 📝 Подробное логирование
- ⚙️ Гибкая конфигурация

## Требования

- C++17 или выше
- CMake 3.15+
- PostgreSQL 12+
- libpqxx
- nlohmann/json

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

## Запуск основного приложения

```bash
/home/andrew/Projects/Ai_c-_bot_psql/build/bin/AIQueryAgent
```

## Быстрый тест окружения PyTorch C++

```bash
/home/andrew/Projects/Ai_c-_bot_psql/build/bin/test_model
```