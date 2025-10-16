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

### Ubuntu/Debian
```bash
sudo apt-get install libpqxx-dev nlohmann-json3-dev postgresql-server-dev-all