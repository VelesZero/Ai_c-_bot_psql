import argparse
import json
import random
import sys
from datetime import date, timedelta


def _choice(rng, items):
    return items[rng.randrange(len(items))]


def _rand_date(rng, start=date(2018, 1, 1), end=date(2026, 1, 1)):
    delta = (end - start).days
    return start + timedelta(days=rng.randrange(max(1, delta)))


def _rand_ts(rng):
    d = _rand_date(rng)
    hh = rng.randrange(0, 24)
    mm = rng.randrange(0, 60)
    ss = rng.randrange(0, 60)
    return f"{d.isoformat()} {hh:02d}:{mm:02d}:{ss:02d}"


def _write_example(fp, first, nl, sql):
    obj = {"nl": nl, "sql": sql}
    s = json.dumps(obj, ensure_ascii=False)
    if not first:
        fp.write(",\n")
    fp.write("    " + s)


def _mk_ecommerce(rng):
    prod_cols = ["id", "name", "category", "price", "stock", "created_at"]
    ord_cols = ["id", "user_id", "status", "total", "created_at"]
    status = _choice(rng, ["paid", "pending", "shipped", "cancelled", "refunded"])
    cat = _choice(rng, ["electronics", "books", "home", "clothing", "sports", "toys"])
    price = rng.randrange(10, 5000)
    n = rng.randrange(1, 50)
    d1 = _rand_date(rng).isoformat()
    d2 = _rand_date(rng).isoformat()
    if d2 < d1:
        d1, d2 = d2, d1

    variants = []

    variants.append((
        _choice(rng, [
            f"покажи товары из категории {cat}",
            f"show products in category {cat}",
            f"list all products where category is {cat}",
        ]),
        f"SELECT id, name, category, price, stock FROM ecommerce.products WHERE category = '{cat}' ORDER BY price ASC",
    ))

    variants.append((
        _choice(rng, [
            f"найди {n} самых дешёвых товаров", 
            f"get {n} cheapest products", 
            f"show {n} lowest priced products",
        ]),
        f"SELECT id, name, price FROM ecommerce.products ORDER BY price ASC LIMIT {n}",
    ))

    variants.append((
        _choice(rng, [
            f"сколько заказов со статусом {status}",
            f"count orders with status {status}",
            f"how many orders are {status}",
        ]),
        f"SELECT COUNT(*) FROM ecommerce.orders WHERE status = '{status}'",
    ))

    variants.append((
        _choice(rng, [
            f"сумма продаж по дням между {d1} и {d2}",
            f"daily sales total between {d1} and {d2}",
            f"group orders by day between {d1} and {d2}",
        ]),
        "SELECT date_trunc('day', created_at) AS day, SUM(total) AS revenue "
        "FROM ecommerce.orders "
        f"WHERE created_at::date BETWEEN '{d1}' AND '{d2}' "
        "GROUP BY day ORDER BY day ASC",
    ))

    variants.append((
        _choice(rng, [
            f"покажи товары дороже {price}",
            f"show products with price greater than {price}",
            f"list products where price > {price}",
        ]),
        f"SELECT id, name, price FROM ecommerce.products WHERE price > {price} ORDER BY price DESC",
    ))

    return _choice(rng, variants)


def _mk_finance(rng):
    currency = _choice(rng, ["USD", "EUR", "RUB", "GBP", "JPY"])
    kind = _choice(rng, ["income", "expense", "transfer"])
    amount = rng.randrange(10, 100000)
    n = rng.randrange(5, 100)
    d = _rand_date(rng).isoformat()

    variants = []

    variants.append((
        _choice(rng, [
            f"покажи транзакции в валюте {currency}",
            f"show transactions in {currency}",
            f"list all transactions where currency = {currency}",
        ]),
        f"SELECT id, account_id, amount, currency, kind, created_at FROM finance.transactions WHERE currency = '{currency}' ORDER BY created_at DESC",
    ))

    variants.append((
        _choice(rng, [
            f"сумма расходов по категориям", 
            "expense total by category", 
            "group expenses by category",
        ]),
        "SELECT category, SUM(amount) AS total_expense "
        "FROM finance.transactions "
        "WHERE kind = 'expense' "
        "GROUP BY category ORDER BY total_expense DESC",
    ))

    variants.append((
        _choice(rng, [
            f"посчитай баланс счёта 1", 
            "calculate account 1 balance", 
            "current balance for account 1",
        ]),
        "SELECT COALESCE(SUM(CASE WHEN kind = 'income' THEN amount WHEN kind = 'expense' THEN -amount ELSE 0 END), 0) AS balance "
        "FROM finance.transactions WHERE account_id = 1",
    ))

    variants.append((
        _choice(rng, [
            f"покажи последние {n} транзакций", 
            f"show last {n} transactions", 
            f"get {n} most recent transactions",
        ]),
        f"SELECT id, amount, currency, kind, created_at FROM finance.transactions ORDER BY created_at DESC LIMIT {n}",
    ))

    variants.append((
        _choice(rng, [
            f"найди транзакции больше {amount}",
            f"find transactions greater than {amount}",
            f"transactions with amount > {amount}",
        ]),
        f"SELECT id, account_id, amount, currency, kind FROM finance.transactions WHERE amount > {amount} ORDER BY amount DESC",
    ))

    variants.append((
        _choice(rng, [
            f"сколько транзакций типа {kind} на дату {d}",
            f"count {kind} transactions on {d}",
            f"how many {kind} operations happened on {d}",
        ]),
        f"SELECT COUNT(*) FROM finance.transactions WHERE kind = '{kind}' AND created_at::date = '{d}'",
    ))

    return _choice(rng, variants)


def _mk_hr(rng):
    dept = _choice(rng, ["engineering", "sales", "hr", "support", "marketing", "finance"])
    title = _choice(rng, ["engineer", "manager", "analyst", "specialist", "director"])
    salary = rng.randrange(500, 20000) * 10
    n = rng.randrange(3, 50)

    variants = []

    variants.append((
        _choice(rng, [
            f"покажи сотрудников из отдела {dept}",
            f"show employees in {dept} department",
            f"list staff where department = {dept}",
        ]),
        f"SELECT id, full_name, department, title, salary FROM hr.employees WHERE department = '{dept}' ORDER BY full_name ASC",
    ))

    variants.append((
        _choice(rng, [
            f"сколько сотрудников с должностью {title}",
            f"count employees with title {title}",
            f"how many people are {title}",
        ]),
        f"SELECT COUNT(*) FROM hr.employees WHERE title = '{title}'",
    ))

    variants.append((
        _choice(rng, [
            f"топ {n} сотрудников по зарплате", 
            f"top {n} employees by salary", 
            f"get {n} highest paid employees",
        ]),
        f"SELECT id, full_name, salary FROM hr.employees ORDER BY salary DESC LIMIT {n}",
    ))

    variants.append((
        _choice(rng, [
            f"средняя зарплата по отделам", 
            "average salary per department", 
            "group employees by department and average salary",
        ]),
        "SELECT department, AVG(salary) AS avg_salary FROM hr.employees GROUP BY department ORDER BY avg_salary DESC",
    ))

    variants.append((
        _choice(rng, [
            f"покажи сотрудников с зарплатой больше {salary}",
            f"employees with salary greater than {salary}",
            f"list employees where salary > {salary}",
        ]),
        f"SELECT id, full_name, salary FROM hr.employees WHERE salary > {salary} ORDER BY salary DESC",
    ))

    return _choice(rng, variants)


def _mk_healthcare(rng):
    speciality = _choice(rng, ["cardiology", "neurology", "therapy", "pediatrics", "dermatology"])
    city = _choice(rng, ["moscow", "spb", "kazan", "novosibirsk", "sochi"])
    n = rng.randrange(1, 30)
    d = _rand_date(rng).isoformat()

    variants = []

    variants.append((
        _choice(rng, [
            f"покажи врачей по специализации {speciality}",
            f"show doctors with specialty {speciality}",
            f"list doctors where specialty is {speciality}",
        ]),
        f"SELECT id, full_name, specialty, clinic_id FROM healthcare.doctors WHERE specialty = '{speciality}' ORDER BY full_name ASC",
    ))

    variants.append((
        _choice(rng, [
            f"сколько приёмов на дату {d}",
            f"count appointments on {d}",
            f"how many visits happened on {d}",
        ]),
        f"SELECT COUNT(*) FROM healthcare.appointments WHERE scheduled_at::date = '{d}'",
    ))

    variants.append((
        _choice(rng, [
            f"покажи последние {n} диагнозов", 
            f"show last {n} diagnoses", 
            f"get {n} most recent diagnosis records",
        ]),
        f"SELECT id, patient_id, diagnosis, created_at FROM healthcare.diagnoses ORDER BY created_at DESC LIMIT {n}",
    ))

    variants.append((
        _choice(rng, [
            f"клиники в городе {city}",
            f"clinics in {city}",
            f"list clinics where city = {city}",
        ]),
        f"SELECT id, name, city FROM healthcare.clinics WHERE city = '{city}' ORDER BY name ASC",
    ))

    variants.append((
        _choice(rng, [
            f"покажи приёмы за последние 7 дней", 
            "appointments in last 7 days", 
            "visits from the last week",
        ]),
        "SELECT id, patient_id, doctor_id, scheduled_at FROM healthcare.appointments "
        "WHERE scheduled_at >= now() - interval '7 days' ORDER BY scheduled_at DESC",
    ))

    return _choice(rng, variants)


def _mk_education(rng):
    course = _choice(rng, ["math", "physics", "history", "programming", "biology"])
    n = rng.randrange(5, 100)
    min_score = rng.randrange(50, 95)

    variants = []

    variants.append((
        _choice(rng, [
            f"покажи студентов на курсе {course}",
            f"show students enrolled in {course}",
            f"list enrollments for course {course}",
        ]),
        "SELECT s.id, s.full_name, c.name AS course "
        "FROM education.enrollments e "
        "JOIN education.students s ON s.id = e.student_id "
        "JOIN education.courses c ON c.id = e.course_id "
        f"WHERE c.name = '{course}' ORDER BY s.full_name ASC",
    ))

    variants.append((
        _choice(rng, [
            f"топ {n} студентов по среднему баллу", 
            f"top {n} students by average grade", 
            f"get {n} best students by avg score",
        ]),
        "SELECT s.id, s.full_name, AVG(g.score) AS avg_score "
        "FROM education.grades g "
        "JOIN education.students s ON s.id = g.student_id "
        "GROUP BY s.id, s.full_name ORDER BY avg_score DESC LIMIT " + str(n),
    ))

    variants.append((
        _choice(rng, [
            f"покажи оценки выше {min_score}",
            f"show grades above {min_score}",
            f"list grade records where score > {min_score}",
        ]),
        f"SELECT student_id, course_id, score FROM education.grades WHERE score > {min_score} ORDER BY score DESC",
    ))

    variants.append((
        _choice(rng, [
            "средний балл по курсам",
            "average grade per course",
            "group grades by course and average score",
        ]),
        "SELECT c.name AS course, AVG(g.score) AS avg_score "
        "FROM education.grades g JOIN education.courses c ON c.id = g.course_id "
        "GROUP BY c.name ORDER BY avg_score DESC",
    ))

    return _choice(rng, variants)


def _mk_logistics(rng):
    status = _choice(rng, ["created", "in_transit", "delivered", "delayed", "returned"])
    city = _choice(rng, ["moscow", "spb", "kazan", "ekb", "sochi"])
    n = rng.randrange(10, 200)

    variants = []

    variants.append((
        _choice(rng, [
            f"покажи посылки со статусом {status}",
            f"show shipments with status {status}",
            f"list parcels where status = {status}",
        ]),
        f"SELECT id, tracking_no, status, updated_at FROM logistics.shipments WHERE status = '{status}' ORDER BY updated_at DESC",
    ))

    variants.append((
        _choice(rng, [
            f"посылки в город {city}",
            f"shipments to {city}",
            f"list deliveries where destination_city = {city}",
        ]),
        f"SELECT id, tracking_no, destination_city, status FROM logistics.shipments WHERE destination_city = '{city}' ORDER BY id DESC",
    ))

    variants.append((
        _choice(rng, [
            f"покажи {n} последних событий трекинга", 
            f"show last {n} tracking events", 
            f"get {n} most recent tracking updates",
        ]),
        f"SELECT id, shipment_id, event, created_at FROM logistics.tracking_events ORDER BY created_at DESC LIMIT {n}",
    ))

    variants.append((
        _choice(rng, [
            "сколько доставок по статусам",
            "count shipments by status",
            "group shipments by status and count",
        ]),
        "SELECT status, COUNT(*) FROM logistics.shipments GROUP BY status ORDER BY COUNT(*) DESC",
    ))

    return _choice(rng, variants)


def _mk_social(rng):
    tag = _choice(rng, ["ai", "music", "sports", "travel", "news", "food"])
    n = rng.randrange(3, 100)
    min_likes = rng.randrange(0, 2000)

    variants = []

    variants.append((
        _choice(rng, [
            f"покажи посты с тегом {tag}",
            f"show posts with tag {tag}",
            f"list posts where tag is {tag}",
        ]),
        "SELECT p.id, p.user_id, p.body, p.created_at "
        "FROM social.posts p JOIN social.post_tags t ON t.post_id = p.id "
        f"WHERE t.tag = '{tag}' ORDER BY p.created_at DESC",
    ))

    variants.append((
        _choice(rng, [
            f"покажи {n} самых популярных постов", 
            f"show {n} most popular posts", 
            f"top {n} posts by likes",
        ]),
        f"SELECT id, user_id, likes_count FROM social.posts ORDER BY likes_count DESC LIMIT {n}",
    ))

    variants.append((
        _choice(rng, [
            f"посты с лайками больше {min_likes}",
            f"posts with likes greater than {min_likes}",
            f"list posts where likes_count > {min_likes}",
        ]),
        f"SELECT id, user_id, likes_count FROM social.posts WHERE likes_count > {min_likes} ORDER BY likes_count DESC",
    ))

    variants.append((
        _choice(rng, [
            "сколько комментариев по постам",
            "count comments per post",
            "group comments by post_id and count",
        ]),
        "SELECT post_id, COUNT(*) AS comments FROM social.comments GROUP BY post_id ORDER BY comments DESC",
    ))

    return _choice(rng, variants)


def _mk_iot(rng):
    device_type = _choice(rng, ["thermostat", "meter", "sensor", "camera", "router"])
    metric = _choice(rng, ["temperature", "humidity", "voltage", "power", "signal"])
    threshold = rng.randrange(1, 200)

    variants = []

    variants.append((
        _choice(rng, [
            f"покажи устройства типа {device_type}",
            f"show devices of type {device_type}",
            f"list devices where device_type = {device_type}",
        ]),
        f"SELECT id, name, device_type, last_seen FROM iot.devices WHERE device_type = '{device_type}' ORDER BY last_seen DESC",
    ))

    variants.append((
        _choice(rng, [
            f"среднее значение {metric} по часам за сутки",
            f"hourly average {metric} for last day",
            f"group {metric} readings by hour in last 24 hours",
        ]),
        "SELECT date_trunc('hour', ts) AS hour, AVG(value) AS avg_value "
        "FROM iot.telemetry "
        f"WHERE metric = '{metric}' AND ts >= now() - interval '24 hours' "
        "GROUP BY hour ORDER BY hour ASC",
    ))

    variants.append((
        _choice(rng, [
            f"покажи показания {metric} выше {threshold}",
            f"show {metric} readings above {threshold}",
            f"telemetry where metric {metric} has value > {threshold}",
        ]),
        f"SELECT device_id, ts, value FROM iot.telemetry WHERE metric = '{metric}' AND value > {threshold} ORDER BY ts DESC LIMIT 200",
    ))

    variants.append((
        _choice(rng, [
            "какие устройства не выходили на связь 1 день",
            "devices not seen for 1 day",
            "list devices where last_seen older than 1 day",
        ]),
        "SELECT id, name, last_seen FROM iot.devices WHERE last_seen < now() - interval '1 day' ORDER BY last_seen ASC",
    ))

    return _choice(rng, variants)


def _mk_analytics(rng):
    page = _choice(rng, ["/", "/pricing", "/docs", "/blog", "/login", "/signup"])
    n = rng.randrange(10, 500)
    d1 = _rand_date(rng).isoformat()
    d2 = _rand_date(rng).isoformat()
    if d2 < d1:
        d1, d2 = d2, d1

    variants = []

    variants.append((
        _choice(rng, [
            f"посчитай визиты на страницу {page}",
            f"count visits for page {page}",
            f"how many pageviews for {page}",
        ]),
        f"SELECT COUNT(*) FROM analytics.pageviews WHERE path = '{page}'",
    ))

    variants.append((
        _choice(rng, [
            f"покажи {n} последних визитов", 
            f"show last {n} visits", 
            f"get {n} most recent pageviews",
        ]),
        f"SELECT id, user_id, path, ts FROM analytics.pageviews ORDER BY ts DESC LIMIT {n}",
    ))

    variants.append((
        _choice(rng, [
            f"уникальные пользователи по дням между {d1} и {d2}",
            f"daily unique users between {d1} and {d2}",
            f"group unique visitors by day between {d1} and {d2}",
        ]),
        "SELECT date_trunc('day', ts) AS day, COUNT(DISTINCT user_id) AS dau "
        "FROM analytics.pageviews "
        f"WHERE ts::date BETWEEN '{d1}' AND '{d2}' "
        "GROUP BY day ORDER BY day ASC",
    ))

    variants.append((
        _choice(rng, [
            "топ страниц по просмотрам",
            "top pages by views",
            "group pageviews by path and count",
        ]),
        "SELECT path, COUNT(*) AS views FROM analytics.pageviews GROUP BY path ORDER BY views DESC LIMIT 50",
    ))

    return _choice(rng, variants)


def _mk_crud(rng):
    email = _choice(rng, ["alice@example.com", "bob@example.com", "carol@example.com", "dan@example.com"])
    new_city = _choice(rng, ["moscow", "spb", "kazan", "sochi", "ekb"])
    uid = rng.randrange(1, 50000)

    variants = []

    variants.append((
        _choice(rng, [
            f"создай пользователя с email {email}",
            f"insert user with email {email}",
            f"add a new user, email {email}",
        ]),
        f"INSERT INTO public.users (email) VALUES ('{email}') RETURNING id",
    ))

    variants.append((
        _choice(rng, [
            f"обнови город пользователя {uid} на {new_city}",
            f"update user {uid} city to {new_city}",
            f"set city to {new_city} for user id {uid}",
        ]),
        f"UPDATE public.users SET city = '{new_city}' WHERE id = {uid} RETURNING id",
    ))

    variants.append((
        _choice(rng, [
            f"удали пользователя {uid}",
            f"delete user {uid}",
            f"remove user with id {uid}",
        ]),
        f"DELETE FROM public.users WHERE id = {uid} RETURNING id",
    ))

    return _choice(rng, variants)


def generate(out_path: str, examples: int, seed: int, mix_crud: float):
    rng = random.Random(seed)

    makers = [
        _mk_ecommerce,
        _mk_finance,
        _mk_hr,
        _mk_healthcare,
        _mk_education,
        _mk_logistics,
        _mk_social,
        _mk_iot,
        _mk_analytics,
    ]

    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("{\n  \"examples\": [\n")
        first = True
        for i in range(examples):
            if rng.random() < mix_crud:
                nl, sql = _mk_crud(rng)
            else:
                maker = _choice(rng, makers)
                nl, sql = maker(rng)

            _write_example(fp, first, nl, sql)
            first = False

            if (i + 1) % 100000 == 0:
                fp.flush()
                print(f"generated {i + 1}/{examples}", file=sys.stderr)

        fp.write("\n  ]\n}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training_data/multidomain_nl_to_sql_1m.json")
    ap.add_argument("--examples", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mix-crud", type=float, default=0.10)
    args = ap.parse_args()

    generate(args.out, args.examples, args.seed, args.mix_crud)


if __name__ == "__main__":
    main()
