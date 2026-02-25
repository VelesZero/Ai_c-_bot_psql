1M датасет “для обычного пользователя” — как сделать
Я обновил генератор generate_bookings_dataset_from_db так, чтобы он:

Генерировал RU+EN формулировки (синонимы: “покажи/выведи/дай”, “сколько/количество/посчитай”, “find/contains”, и т.д.)
Делал разные намерения:
показать всё / лимит N
посчитать строки
выбрать колонку
WHERE ... IS NOT NULL
WHERE text = 'реальное_значение'
WHERE text ILIKE '%подстрока%'
WHERE num > значение
сортировка ORDER BY ... ASC/DESC
AVG(...)
простая агрегация GROUP BY ... COUNT(*)
Берёт реальные значения из базы (сэмплы), чтобы фильтры выглядели “живыми”
Пишет JSON потоком на диск, без хранения 1M в памяти
Команда генерации 1,000,000 примеров
bash
./build/bin/generate_bookings_dataset_from_db \
  --host localhost --port 5432 --db ai_db --user ai_user --password 123 \
  --schema bookings \
  --out training_data/bookings_nl_to_sql_1m_from_db.json \
  --examples 1000000 \
  --seed 42
Результат:

training_data/bookings_nl_to_sql_1m_from_db.json (формат { "examples": [ ... ] })
Если прав на bookings не хватает — генератор/агент будет ругаться permission denied. Тогда нужно выдать права (скажи — дам команды под твою роль в Postgres).

Обучение на этом датасете
Вариант 1 (рекомендую старт): tiny-модель (быстрее, меньше RAM/VRAM)
bash
./build/bin/train_model \
  --epochs 20 \
  --lr 0.001 \
  --cpu \
  --data training_data/bookings_nl_to_sql_1m_from_db.json \
  --out models/seq2seq_bookings_1m_from_db_tiny \
  --emb 64 --hid 128 \
  --batch 64
Вариант 2: большая модель (лучше качество, но тяжелее)
bash
./build/bin/train_model \
  --epochs 20 \
  --lr 0.001 \
  --cuda \
  --data training_data/bookings_nl_to_sql_1m_from_db.json \
  --out models/seq2seq_bookings_1m_from_db_large \
  --emb 256 --hid 512 \
  --batch 64
Важно: теперь train_model нормально понимает --epochs и --lr (раньше он падал).

Подключить обученную модель в агенте
Для tiny (пример):

ini
model_path=models/seq2seq_bookings_1m_from_db_tiny
model_emb_dim=64
model_hid_dim=128
Потом:

bash
cmake --build build
./build/bin/AIQueryAgent src/config/default.conf
Ограничения (честно)
Сейчас датасет без JOIN-ов (только single-table + агрегаты). Это уже сильно повышает “юзер-френдли” (простые вопросы), но для запросов типа “рейсы по билету/пассажиру/аэропорту” лучше добавить JOIN-шаблоны по foreign keys. Если скажешь “добавь join-ы тоже” — расширю генератор так, чтобы он делал и многотабличные запросы с реальными значениями.

Статус
Готово: генератор 1M с разнообразным RU+EN NL и реальными значениями из БД; обучение теперь запускается флагами.
Нужно от тебя: хочешь ли ты добавить JOIN-ы (да/нет). Если да — сделаю и ты перегенеришь ещё раз 1M (или 1.2M) уже с multi-table примерами.
