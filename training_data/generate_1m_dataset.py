#!/usr/bin/env python3
"""
Generate 1M NL→SQL training pairs from the bookings database.
Uses real data values to create diverse, realistic examples.

Usage:
    python3 generate_1m_dataset.py [--out PATH] [--count N] [--seed N]
"""

import json
import os
import random
import sys
import argparse
from itertools import product

import psycopg2
from psycopg2.extras import RealDictCursor


def connect_db():
    return psycopg2.connect(
        host=os.getenv("AI_DB_HOST", "localhost"),
        port=int(os.getenv("AI_DB_PORT", "5432")),
        dbname=os.getenv("AI_DB_NAME", "ai_db"),
        user=os.getenv("AI_DB_USER", "ai_user"),
        password=os.getenv("AI_DB_PASSWORD", ""),
        options="-c search_path=bookings,public",
    )


def sample_values(cur, query, limit=500):
    """Fetch sample values from DB."""
    cur.execute(query)
    return [r[0] for r in cur.fetchall() if r[0] is not None][:limit]


def sql_lit(val):
    """SQL string literal."""
    return "'" + str(val).replace("'", "''") + "'"


class DatasetGenerator:
    def __init__(self, cur, rng):
        self.cur = cur
        self.rng = rng
        self._load_samples()

    def _load_samples(self):
        """Pre-load sample values from DB for realistic examples."""
        c = self.cur
        print("Loading sample values from database...")

        # Airports
        self.cities = sample_values(c, "SELECT DISTINCT city FROM airports ORDER BY 1 LIMIT 500")
        self.airport_codes = sample_values(c, "SELECT airport_code FROM airports ORDER BY 1 LIMIT 500")
        self.airport_names = sample_values(c, "SELECT airport_name FROM airports ORDER BY 1 LIMIT 500")
        self.countries = sample_values(c, "SELECT DISTINCT country FROM airports ORDER BY 1 LIMIT 200")
        self.timezones = sample_values(c, "SELECT DISTINCT timezone FROM airports ORDER BY 1 LIMIT 100")

        # Airplanes
        self.airplane_codes = sample_values(c, "SELECT airplane_code FROM airplanes")
        self.airplane_models = sample_values(c, "SELECT model FROM airplanes")
        self.airplane_ranges = sample_values(c, "SELECT DISTINCT range FROM airplanes")
        self.airplane_speeds = sample_values(c, "SELECT DISTINCT speed FROM airplanes")

        # Flights
        self.flight_statuses = sample_values(c, "SELECT DISTINCT status FROM flights")
        self.route_nos = sample_values(c, "SELECT DISTINCT route_no FROM routes ORDER BY 1 LIMIT 300")
        self.dep_airports = sample_values(c, "SELECT DISTINCT departure_airport FROM routes")
        self.arr_airports = sample_values(c, "SELECT DISTINCT arrival_airport FROM routes")

        # Tickets / passengers
        self.passenger_names = sample_values(c, "SELECT DISTINCT passenger_name FROM tickets ORDER BY 1 LIMIT 500")
        self.fare_conditions = sample_values(c, "SELECT DISTINCT fare_conditions FROM seats")
        self.seat_nos = sample_values(c, "SELECT DISTINCT seat_no FROM seats ORDER BY 1 LIMIT 200")

        # Bookings
        self.book_refs = sample_values(c, "SELECT book_ref FROM bookings ORDER BY 1 LIMIT 300")
        c.execute("SELECT min(total_amount), max(total_amount), avg(total_amount)::int FROM bookings")
        row = c.fetchone()
        self.amount_min, self.amount_max, self.amount_avg = float(row[0]), float(row[1]), int(row[2])

        # Prices
        self.prices = sample_values(c, "SELECT DISTINCT price FROM segments ORDER BY 1 LIMIT 200")

        print(f"  Cities: {len(self.cities)}, Airports: {len(self.airport_codes)}, "
              f"Passengers: {len(self.passenger_names)}, Routes: {len(self.route_nos)}")

    def pick(self, lst):
        return self.rng.choice(lst) if lst else "unknown"

    def rnd_int(self, lo, hi):
        return self.rng.randint(lo, hi)

    def rnd_amount(self):
        return self.rng.randint(1000, int(self.amount_max))

    def rnd_limit(self):
        return self.rng.choice([1, 3, 5, 10, 15, 20, 25, 50, 100])

    # ──────── NL verb variants ────────

    # English
    EN_SHOW = ["show", "list", "display", "get", "fetch", "find", "return", "give me", "print"]
    EN_COUNT = ["count", "how many", "number of", "total number of", "total"]
    EN_ALL = ["all", "every", "the", "all the"]
    EN_FROM = ["from", "in"]
    EN_WHERE = ["where", "with", "having", "whose", "that have"]
    EN_SORT = ["sort by", "order by", "sorted by", "ordered by", "arrange by"]
    EN_TOP = ["top", "first", "best"]
    EN_CHEAP = ["cheapest", "least expensive", "lowest price", "most affordable"]
    EN_EXPENSIVE = ["most expensive", "highest price", "priciest", "costliest"]
    EN_AVG = ["average", "mean", "avg"]
    EN_BETWEEN = ["between", "from {lo} to {hi}", "in range {lo} to {hi}"]

    # Russian
    RU_SHOW = ["покажи", "выведи", "найди", "дай", "получи", "отобрази", "верни", "пожалуйста покажи"]
    RU_COUNT = ["сколько", "количество", "посчитай", "подсчитай", "какое количество"]
    RU_ALL = ["все", "всё", "все имеющиеся"]
    RU_FROM = ["из", "в", "по"]
    RU_WHERE = ["где", "с", "у которых", "которые имеют"]
    RU_SORT = ["отсортируй по", "упорядочь по", "сортировка по"]
    RU_ROWS = ["строк", "записей", "результатов"]
    RU_TOP = ["первые", "топ", "лучшие"]
    RU_CHEAP = ["самые дешёвые", "наименее дорогие"]
    RU_EXPENSIVE = ["самые дорогие", "наиболее дорогие"]
    RU_AVG = ["среднее", "средняя", "средний"]

    def en(self, lst):
        return self.rng.choice(lst)

    def lang_show(self, ru):
        return self.en(self.RU_SHOW if ru else self.EN_SHOW)

    def lang_count(self, ru):
        return self.en(self.RU_COUNT if ru else self.EN_COUNT)

    def lang_all(self, ru):
        return self.en(self.RU_ALL if ru else self.EN_ALL)

    def lang_from(self, ru):
        return self.en(self.RU_FROM if ru else self.EN_FROM)

    def lang_where(self, ru):
        return self.en(self.RU_WHERE if ru else self.EN_WHERE)

    # ──────── Query generators ────────

    def gen_airports_select_all(self, ru):
        v = self.lang_show(ru)
        a = self.lang_all(ru)
        return (f"{v} {a} airports" if not ru else f"{v} {a} аэропорты",
                "select * from bookings.airports_data")

    def gen_airports_by_city(self, ru):
        city = self.pick(self.cities)
        v = self.lang_show(ru)
        w = self.lang_where(ru)
        return (f"{v} airports {w} city = {city}" if not ru
                else f"{v} аэропорты {w} город = {city}",
                f"select * from bookings.airports_data where city = {sql_lit(city)}")

    def gen_airports_by_country(self, ru):
        country = self.pick(self.countries)
        v = self.lang_show(ru)
        return (f"{v} airports in {country}" if not ru
                else f"{v} аэропорты в {country}",
                f"select * from bookings.airports_data where country = {sql_lit(country)}")

    def gen_airports_count(self, ru):
        c = self.lang_count(ru)
        return (f"{c} airports" if not ru else f"{c} аэропортов",
                "select count(*) from bookings.airports_data")

    def gen_airports_count_by_country(self, ru):
        country = self.pick(self.countries)
        c = self.lang_count(ru)
        return (f"{c} airports in {country}" if not ru
                else f"{c} аэропортов в {country}",
                f"select count(*) from bookings.airports_data where country = {sql_lit(country)}")

    def gen_airports_by_timezone(self, ru):
        tz = self.pick(self.timezones)
        v = self.lang_show(ru)
        return (f"{v} airports in timezone {tz}" if not ru
                else f"{v} аэропорты в часовом поясе {tz}",
                f"select * from bookings.airports_data where timezone = {sql_lit(tz)}")

    def gen_airports_group_country(self, ru):
        return ("count airports by country" if not ru
                else "количество аэропортов по странам",
                "select country, count(*) from bookings.airports_data group by country order by count desc limit 50")

    def gen_airports_limit(self, ru):
        n = self.rnd_limit()
        v = self.lang_show(ru)
        return (f"{v} {n} airports" if not ru else f"{v} {n} аэропортов",
                f"select * from bookings.airports_data limit {n}")

    # ── Airplanes ──

    def gen_airplanes_all(self, ru):
        v = self.lang_show(ru)
        a = self.lang_all(ru)
        return (f"{v} {a} airplanes" if not ru else f"{v} {a} самолёты",
                "select * from bookings.airplanes_data")

    def gen_airplanes_by_range(self, ru):
        r = self.rng.choice([3000, 5000, 7000, 10000, 12000])
        v = self.lang_show(ru)
        return (f"{v} airplanes with range greater than {r}" if not ru
                else f"{v} самолёты с дальностью больше {r}",
                f"select * from bookings.airplanes_data where range > {r}")

    def gen_airplanes_by_speed(self, ru):
        s = self.rng.choice([800, 850, 900, 950])
        v = self.lang_show(ru)
        return (f"{v} airplanes faster than {s} km/h" if not ru
                else f"{v} самолёты быстрее {s} км/ч",
                f"select * from bookings.airplanes_data where speed > {s}")

    def gen_airplanes_by_code(self, ru):
        code = self.pick(self.airplane_codes)
        v = self.lang_show(ru)
        return (f"{v} airplane {code}" if not ru else f"{v} самолёт {code}",
                f"select * from bookings.airplanes_data where airplane_code = {sql_lit(code)}")

    def gen_airplanes_sort_range(self, ru):
        desc = self.rng.choice([True, False])
        d = "desc" if desc else "asc"
        w = "по убыванию" if desc else "по возрастанию"
        return (f"sort airplanes by range {d}" if not ru
                else f"отсортируй самолёты по дальности {w}",
                f"select * from bookings.airplanes_data order by range {d}")

    def gen_airplanes_count(self, ru):
        c = self.lang_count(ru)
        return (f"{c} airplanes" if not ru else f"{c} самолётов",
                "select count(*) from bookings.airplanes_data")

    def gen_airplanes_avg_range(self, ru):
        a = self.en(self.RU_AVG if ru else self.EN_AVG)
        return (f"{a} range of airplanes" if not ru
                else f"{a} дальность самолётов",
                "select avg(range) from bookings.airplanes_data")

    # ── Flights ──

    def gen_flights_all(self, ru):
        v = self.lang_show(ru)
        n = self.rnd_limit()
        return (f"{v} {n} flights" if not ru else f"{v} {n} рейсов",
                f"select * from bookings.flights limit {n}")

    def gen_flights_by_status(self, ru):
        status = self.pick(self.flight_statuses)
        v = self.lang_show(ru)
        return (f"{v} flights with status {status}" if not ru
                else f"{v} рейсы со статусом {status}",
                f"select * from bookings.flights where status = {sql_lit(status)}")

    def gen_flights_count_by_status(self, ru):
        status = self.pick(self.flight_statuses)
        c = self.lang_count(ru)
        return (f"{c} flights with status {status}" if not ru
                else f"{c} рейсов со статусом {status}",
                f"select count(*) from bookings.flights where status = {sql_lit(status)}")

    def gen_flights_by_route(self, ru):
        route = self.pick(self.route_nos)
        v = self.lang_show(ru)
        return (f"{v} flights on route {route}" if not ru
                else f"{v} рейсы по маршруту {route}",
                f"select * from bookings.flights where route_no = {sql_lit(route)}")

    def gen_flights_count(self, ru):
        c = self.lang_count(ru)
        return (f"{c} flights" if not ru else f"{c} рейсов",
                "select count(*) from bookings.flights")

    def gen_flights_group_status(self, ru):
        return ("count flights by status" if not ru
                else "количество рейсов по статусам",
                "select status, count(*) from bookings.flights group by status order by count desc")

    def gen_flights_departed(self, ru):
        v = self.lang_show(ru)
        return (f"{v} departed flights" if not ru
                else f"{v} вылетевшие рейсы",
                "select * from bookings.flights where actual_departure is not null limit 50")

    def gen_flights_delayed(self, ru):
        v = self.lang_show(ru)
        return (f"{v} delayed flights" if not ru
                else f"{v} задержанные рейсы",
                "select * from bookings.flights where status = 'Delayed' limit 50")

    def gen_flights_cancelled(self, ru):
        v = self.lang_show(ru)
        return (f"{v} cancelled flights" if not ru
                else f"{v} отменённые рейсы",
                "select * from bookings.flights where status = 'Cancelled' limit 50")

    # ── Routes ──

    def gen_routes_all(self, ru):
        v = self.lang_show(ru)
        a = self.lang_all(ru)
        return (f"{v} {a} routes" if not ru else f"{v} {a} маршруты",
                "select * from bookings.routes")

    def gen_routes_by_dep(self, ru):
        dep = self.pick(self.dep_airports)
        v = self.lang_show(ru)
        return (f"{v} routes from {dep}" if not ru
                else f"{v} маршруты из {dep}",
                f"select * from bookings.routes where departure_airport = {sql_lit(dep)}")

    def gen_routes_by_arr(self, ru):
        arr = self.pick(self.arr_airports)
        v = self.lang_show(ru)
        return (f"{v} routes to {arr}" if not ru
                else f"{v} маршруты в {arr}",
                f"select * from bookings.routes where arrival_airport = {sql_lit(arr)}")

    def gen_routes_by_airplane(self, ru):
        code = self.pick(self.airplane_codes)
        v = self.lang_show(ru)
        return (f"{v} routes for airplane {code}" if not ru
                else f"{v} маршруты для самолёта {code}",
                f"select * from bookings.routes where airplane_code = {sql_lit(code)}")

    def gen_routes_count(self, ru):
        c = self.lang_count(ru)
        return (f"{c} routes" if not ru else f"{c} маршрутов",
                "select count(*) from bookings.routes")

    def gen_routes_dep_arr(self, ru):
        dep = self.pick(self.dep_airports)
        arr = self.pick(self.arr_airports)
        v = self.lang_show(ru)
        return (f"{v} route from {dep} to {arr}" if not ru
                else f"{v} маршрут из {dep} в {arr}",
                f"select * from bookings.routes where departure_airport = {sql_lit(dep)} and arrival_airport = {sql_lit(arr)}")

    # ── Bookings ──

    def gen_bookings_all(self, ru):
        v = self.lang_show(ru)
        n = self.rnd_limit()
        return (f"{v} {n} bookings" if not ru else f"{v} {n} бронирований",
                f"select * from bookings.bookings limit {n}")

    def gen_bookings_by_ref(self, ru):
        ref = self.pick(self.book_refs)
        v = self.lang_show(ru)
        return (f"{v} booking {ref}" if not ru else f"{v} бронирование {ref}",
                f"select * from bookings.bookings where book_ref = {sql_lit(ref)}")

    def gen_bookings_amount_gt(self, ru):
        amount = self.rnd_amount()
        v = self.lang_show(ru)
        return (f"{v} bookings with total amount greater than {amount}" if not ru
                else f"{v} бронирования с суммой больше {amount}",
                f"select * from bookings.bookings where total_amount > {amount} limit 50")

    def gen_bookings_amount_lt(self, ru):
        amount = self.rng.randint(1000, 10000)
        v = self.lang_show(ru)
        return (f"{v} bookings with total amount less than {amount}" if not ru
                else f"{v} бронирования с суммой меньше {amount}",
                f"select * from bookings.bookings where total_amount < {amount} limit 50")

    def gen_bookings_amount_between(self, ru):
        lo = self.rng.randint(1000, 50000)
        hi = lo + self.rng.randint(5000, 100000)
        v = self.lang_show(ru)
        return (f"{v} bookings with amount between {lo} and {hi}" if not ru
                else f"{v} бронирования с суммой от {lo} до {hi}",
                f"select * from bookings.bookings where total_amount between {lo} and {hi} limit 50")

    def gen_bookings_count(self, ru):
        c = self.lang_count(ru)
        return (f"{c} bookings" if not ru else f"{c} бронирований",
                "select count(*) from bookings.bookings")

    def gen_bookings_avg_amount(self, ru):
        a = self.en(self.RU_AVG if ru else self.EN_AVG)
        return (f"{a} booking amount" if not ru
                else f"{a} сумма бронирований",
                "select avg(total_amount) from bookings.bookings")

    def gen_bookings_max_amount(self, ru):
        return ("maximum booking amount" if not ru
                else "максимальная сумма бронирования",
                "select max(total_amount) from bookings.bookings")

    def gen_bookings_min_amount(self, ru):
        return ("minimum booking amount" if not ru
                else "минимальная сумма бронирования",
                "select min(total_amount) from bookings.bookings")

    def gen_bookings_sort_amount(self, ru):
        desc = self.rng.choice([True, False])
        d = "desc" if desc else "asc"
        n = self.rnd_limit()
        return (f"top {n} bookings by amount {'descending' if desc else 'ascending'}" if not ru
                else f"топ {n} бронирований по сумме {'по убыванию' if desc else 'по возрастанию'}",
                f"select * from bookings.bookings order by total_amount {d} limit {n}")

    # ── Tickets ──

    def gen_tickets_all(self, ru):
        v = self.lang_show(ru)
        n = self.rnd_limit()
        return (f"{v} {n} tickets" if not ru else f"{v} {n} билетов",
                f"select * from bookings.tickets limit {n}")

    def gen_tickets_by_name(self, ru):
        name = self.pick(self.passenger_names)
        v = self.lang_show(ru)
        return (f"{v} tickets for {name}" if not ru
                else f"{v} билеты пассажира {name}",
                f"select * from bookings.tickets where passenger_name = {sql_lit(name)}")

    def gen_tickets_by_name_like(self, ru):
        name = self.pick(self.passenger_names)
        first = name.split()[0] if " " in name else name
        v = self.lang_show(ru)
        return (f"{v} tickets for passengers named {first}" if not ru
                else f"{v} билеты пассажиров с именем {first}",
                f"select * from bookings.tickets where passenger_name ilike {sql_lit(first + '%')} limit 50")

    def gen_tickets_by_booking(self, ru):
        ref = self.pick(self.book_refs)
        v = self.lang_show(ru)
        return (f"{v} tickets for booking {ref}" if not ru
                else f"{v} билеты бронирования {ref}",
                f"select * from bookings.tickets where book_ref = {sql_lit(ref)}")

    def gen_tickets_count(self, ru):
        c = self.lang_count(ru)
        return (f"{c} tickets" if not ru else f"{c} билетов",
                "select count(*) from bookings.tickets")

    def gen_tickets_count_by_booking(self, ru):
        ref = self.pick(self.book_refs)
        c = self.lang_count(ru)
        return (f"{c} tickets in booking {ref}" if not ru
                else f"{c} билетов в бронировании {ref}",
                f"select count(*) from bookings.tickets where book_ref = {sql_lit(ref)}")

    def gen_tickets_outbound(self, ru):
        v = self.lang_show(ru)
        n = self.rnd_limit()
        return (f"{v} {n} outbound tickets" if not ru
                else f"{v} {n} билетов туда",
                f"select * from bookings.tickets where outbound = true limit {n}")

    def gen_tickets_return(self, ru):
        v = self.lang_show(ru)
        n = self.rnd_limit()
        return (f"{v} {n} return tickets" if not ru
                else f"{v} {n} обратных билетов",
                f"select * from bookings.tickets where outbound = false limit {n}")

    # ── Segments ──

    def gen_segments_all(self, ru):
        v = self.lang_show(ru)
        n = self.rnd_limit()
        return (f"{v} {n} ticket segments" if not ru else f"{v} {n} сегментов",
                f"select * from bookings.segments limit {n}")

    def gen_segments_by_fare(self, ru):
        fare = self.pick(self.fare_conditions)
        v = self.lang_show(ru)
        n = self.rnd_limit()
        return (f"{v} {n} {fare.lower()} class segments" if not ru
                else f"{v} {n} сегментов класса {fare}",
                f"select * from bookings.segments where fare_conditions = {sql_lit(fare)} limit {n}")

    def gen_segments_price_gt(self, ru):
        price = self.rng.choice([5000, 10000, 20000, 50000, 100000])
        v = self.lang_show(ru)
        return (f"{v} segments with price above {price}" if not ru
                else f"{v} сегменты с ценой выше {price}",
                f"select * from bookings.segments where price > {price} limit 50")

    def gen_segments_avg_price(self, ru):
        a = self.en(self.RU_AVG if ru else self.EN_AVG)
        return (f"{a} segment price" if not ru
                else f"{a} цена сегмента",
                "select avg(price) from bookings.segments")

    def gen_segments_avg_price_by_fare(self, ru):
        return ("average price by fare class" if not ru
                else "средняя цена по классам обслуживания",
                "select fare_conditions, avg(price) from bookings.segments group by fare_conditions order by avg desc")

    def gen_segments_count(self, ru):
        c = self.lang_count(ru)
        return (f"{c} segments" if not ru else f"{c} сегментов",
                "select count(*) from bookings.segments")

    def gen_segments_count_by_fare(self, ru):
        return ("count segments by fare class" if not ru
                else "количество сегментов по классам",
                "select fare_conditions, count(*) from bookings.segments group by fare_conditions order by count desc")

    def gen_segments_max_price(self, ru):
        return ("most expensive segment" if not ru
                else "самый дорогой сегмент",
                "select * from bookings.segments order by price desc limit 1")

    # ── Seats ──

    def gen_seats_all(self, ru):
        v = self.lang_show(ru)
        a = self.lang_all(ru)
        return (f"{v} {a} seats" if not ru else f"{v} {a} места",
                "select * from bookings.seats")

    def gen_seats_by_airplane(self, ru):
        code = self.pick(self.airplane_codes)
        v = self.lang_show(ru)
        return (f"{v} seats on airplane {code}" if not ru
                else f"{v} места в самолёте {code}",
                f"select * from bookings.seats where airplane_code = {sql_lit(code)}")

    def gen_seats_by_fare(self, ru):
        fare = self.pick(self.fare_conditions)
        v = self.lang_show(ru)
        return (f"{v} {fare.lower()} seats" if not ru
                else f"{v} места класса {fare}",
                f"select * from bookings.seats where fare_conditions = {sql_lit(fare)}")

    def gen_seats_count_by_airplane(self, ru):
        code = self.pick(self.airplane_codes)
        c = self.lang_count(ru)
        return (f"{c} seats on airplane {code}" if not ru
                else f"{c} мест в самолёте {code}",
                f"select count(*) from bookings.seats where airplane_code = {sql_lit(code)}")

    def gen_seats_group_fare(self, ru):
        code = self.pick(self.airplane_codes)
        return (f"seats by fare class on airplane {code}" if not ru
                else f"места по классам в самолёте {code}",
                f"select fare_conditions, count(*) from bookings.seats where airplane_code = {sql_lit(code)} group by fare_conditions")

    def gen_seats_count(self, ru):
        c = self.lang_count(ru)
        return (f"{c} seats" if not ru else f"{c} мест",
                "select count(*) from bookings.seats")

    # ── Boarding passes ──

    def gen_boarding_all(self, ru):
        v = self.lang_show(ru)
        n = self.rnd_limit()
        return (f"{v} {n} boarding passes" if not ru else f"{v} {n} посадочных талонов",
                f"select * from bookings.boarding_passes limit {n}")

    def gen_boarding_by_flight(self, ru):
        fid = self.rng.randint(1, 37000)
        v = self.lang_show(ru)
        return (f"{v} boarding passes for flight {fid}" if not ru
                else f"{v} посадочные талоны рейса {fid}",
                f"select * from bookings.boarding_passes where flight_id = {fid} limit 50")

    def gen_boarding_by_seat(self, ru):
        seat = self.pick(self.seat_nos)
        v = self.lang_show(ru)
        return (f"{v} boarding passes for seat {seat}" if not ru
                else f"{v} посадочные талоны на место {seat}",
                f"select * from bookings.boarding_passes where seat_no = {sql_lit(seat)} limit 50")

    def gen_boarding_count(self, ru):
        c = self.lang_count(ru)
        return (f"{c} boarding passes" if not ru else f"{c} посадочных талонов",
                "select count(*) from bookings.boarding_passes")

    # ── JOIN queries ──

    def gen_join_tickets_segments(self, ru):
        name = self.pick(self.passenger_names)
        v = self.lang_show(ru)
        return (f"{v} segments for passenger {name}" if not ru
                else f"{v} сегменты пассажира {name}",
                f"select t.passenger_name, s.fare_conditions, s.price "
                f"from bookings.tickets t join bookings.segments s on t.ticket_no = s.ticket_no "
                f"where t.passenger_name = {sql_lit(name)} limit 50")

    def gen_join_tickets_bookings(self, ru):
        ref = self.pick(self.book_refs)
        v = self.lang_show(ru)
        return (f"{v} tickets and booking info for {ref}" if not ru
                else f"{v} билеты и бронирование {ref}",
                f"select t.ticket_no, t.passenger_name, b.total_amount "
                f"from bookings.tickets t join bookings.bookings b on t.book_ref = b.book_ref "
                f"where b.book_ref = {sql_lit(ref)}")

    def gen_join_flights_routes(self, ru):
        dep = self.pick(self.dep_airports)
        v = self.lang_show(ru)
        return (f"{v} flights departing from {dep}" if not ru
                else f"{v} рейсы из аэропорта {dep}",
                f"select f.flight_id, f.status, r.departure_airport, r.arrival_airport "
                f"from bookings.flights f join bookings.routes r on f.route_no = r.route_no "
                f"where r.departure_airport = {sql_lit(dep)} limit 50")

    def gen_join_boarding_tickets(self, ru):
        name = self.pick(self.passenger_names)
        v = self.lang_show(ru)
        return (f"{v} boarding passes for {name}" if not ru
                else f"{v} посадочные талоны для {name}",
                f"select bp.seat_no, bp.boarding_no, t.passenger_name "
                f"from bookings.boarding_passes bp "
                f"join bookings.tickets t on bp.ticket_no = t.ticket_no "
                f"where t.passenger_name = {sql_lit(name)} limit 50")

    def gen_join_routes_airplanes(self, ru):
        v = self.lang_show(ru)
        n = self.rnd_limit()
        return (f"{v} {n} routes with airplane models" if not ru
                else f"{v} {n} маршрутов с моделями самолётов",
                f"select r.route_no, r.departure_airport, r.arrival_airport, a.model "
                f"from bookings.routes r join bookings.airplanes_data a on r.airplane_code = a.airplane_code "
                f"limit {n}")

    def gen_join_seats_airplanes(self, ru):
        code = self.pick(self.airplane_codes)
        return (f"seats and airplane info for {code}" if not ru
                else f"места и информация о самолёте {code}",
                f"select s.seat_no, s.fare_conditions, a.model, a.range "
                f"from bookings.seats s join bookings.airplanes_data a on s.airplane_code = a.airplane_code "
                f"where s.airplane_code = {sql_lit(code)}")

    # ── Aggregate queries ──

    def gen_agg_total_revenue(self, ru):
        return ("total revenue from all bookings" if not ru
                else "общая выручка от бронирований",
                "select sum(total_amount) from bookings.bookings")

    def gen_agg_passengers_per_booking(self, ru):
        return ("average number of tickets per booking" if not ru
                else "среднее количество билетов на бронирование",
                "select avg(cnt) from (select book_ref, count(*) as cnt from bookings.tickets group by book_ref) sub")

    def gen_agg_top_routes(self, ru):
        n = self.rng.choice([5, 10, 20])
        return (f"top {n} most popular routes" if not ru
                else f"топ {n} популярных маршрутов",
                f"select route_no, count(*) as flight_count from bookings.flights group by route_no order by flight_count desc limit {n}")

    def gen_agg_revenue_by_fare(self, ru):
        return ("revenue by fare class" if not ru
                else "выручка по классам обслуживания",
                "select fare_conditions, sum(price) as revenue from bookings.segments group by fare_conditions order by revenue desc")

    def gen_agg_flights_per_route(self, ru):
        return ("flights per route" if not ru
                else "количество рейсов по маршрутам",
                "select route_no, count(*) from bookings.flights group by route_no order by count desc limit 50")

    def gen_agg_avg_price_by_fare(self, ru):
        return ("average ticket price by fare class" if not ru
                else "средняя цена билета по классам",
                "select fare_conditions, avg(price) as avg_price from bookings.segments group by fare_conditions order by avg_price desc")

    def gen_agg_busiest_airports(self, ru):
        n = self.rng.choice([5, 10, 20])
        return (f"top {n} busiest departure airports" if not ru
                else f"топ {n} аэропортов по вылетам",
                f"select departure_airport, count(*) as cnt from bookings.routes group by departure_airport order by cnt desc limit {n}")

    # ── Subqueries ──

    def gen_sub_expensive_bookings(self, ru):
        v = self.lang_show(ru)
        return (f"{v} bookings above average amount" if not ru
                else f"{v} бронирования с суммой выше средней",
                "select * from bookings.bookings where total_amount > (select avg(total_amount) from bookings.bookings) limit 50")

    def gen_sub_passengers_multiple_tickets(self, ru):
        v = self.lang_show(ru)
        return (f"{v} passengers with more than 2 tickets" if not ru
                else f"{v} пассажиров с более чем 2 билетами",
                "select passenger_name, count(*) as cnt from bookings.tickets group by passenger_name having count(*) > 2 order by cnt desc limit 50")

    # ── Timetable (view) ──

    def gen_timetable_all(self, ru):
        v = self.lang_show(ru)
        n = self.rnd_limit()
        return (f"{v} {n} entries from timetable" if not ru
                else f"{v} {n} записей из расписания",
                f"select * from bookings.timetable limit {n}")

    def gen_timetable_by_dep(self, ru):
        dep = self.pick(self.dep_airports)
        v = self.lang_show(ru)
        return (f"{v} timetable for departures from {dep}" if not ru
                else f"{v} расписание вылетов из {dep}",
                f"select * from bookings.timetable where departure_airport = {sql_lit(dep)} limit 50")

    def gen_timetable_by_status(self, ru):
        status = self.pick(self.flight_statuses)
        v = self.lang_show(ru)
        return (f"{v} timetable entries with status {status}" if not ru
                else f"{v} расписание рейсов со статусом {status}",
                f"select * from bookings.timetable where status = {sql_lit(status)} limit 50")

    # ──────── Master generator ────────

    def all_generators(self):
        """Return list of all generator methods."""
        return [
            # Airports (9)
            self.gen_airports_select_all, self.gen_airports_by_city,
            self.gen_airports_by_country, self.gen_airports_count,
            self.gen_airports_count_by_country, self.gen_airports_by_timezone,
            self.gen_airports_group_country, self.gen_airports_limit,
            # Airplanes (7)
            self.gen_airplanes_all, self.gen_airplanes_by_range,
            self.gen_airplanes_by_speed, self.gen_airplanes_by_code,
            self.gen_airplanes_sort_range, self.gen_airplanes_count,
            self.gen_airplanes_avg_range,
            # Flights (9)
            self.gen_flights_all, self.gen_flights_by_status,
            self.gen_flights_count_by_status, self.gen_flights_by_route,
            self.gen_flights_count, self.gen_flights_group_status,
            self.gen_flights_departed, self.gen_flights_delayed,
            self.gen_flights_cancelled,
            # Routes (6)
            self.gen_routes_all, self.gen_routes_by_dep, self.gen_routes_by_arr,
            self.gen_routes_by_airplane, self.gen_routes_count,
            self.gen_routes_dep_arr,
            # Bookings (9)
            self.gen_bookings_all, self.gen_bookings_by_ref,
            self.gen_bookings_amount_gt, self.gen_bookings_amount_lt,
            self.gen_bookings_amount_between, self.gen_bookings_count,
            self.gen_bookings_avg_amount, self.gen_bookings_max_amount,
            self.gen_bookings_min_amount, self.gen_bookings_sort_amount,
            # Tickets (8)
            self.gen_tickets_all, self.gen_tickets_by_name,
            self.gen_tickets_by_name_like, self.gen_tickets_by_booking,
            self.gen_tickets_count, self.gen_tickets_count_by_booking,
            self.gen_tickets_outbound, self.gen_tickets_return,
            # Segments (8)
            self.gen_segments_all, self.gen_segments_by_fare,
            self.gen_segments_price_gt, self.gen_segments_avg_price,
            self.gen_segments_avg_price_by_fare, self.gen_segments_count,
            self.gen_segments_count_by_fare, self.gen_segments_max_price,
            # Seats (6)
            self.gen_seats_all, self.gen_seats_by_airplane,
            self.gen_seats_by_fare, self.gen_seats_count_by_airplane,
            self.gen_seats_group_fare, self.gen_seats_count,
            # Boarding passes (4)
            self.gen_boarding_all, self.gen_boarding_by_flight,
            self.gen_boarding_by_seat, self.gen_boarding_count,
            # JOINs (6)
            self.gen_join_tickets_segments, self.gen_join_tickets_bookings,
            self.gen_join_flights_routes, self.gen_join_boarding_tickets,
            self.gen_join_routes_airplanes, self.gen_join_seats_airplanes,
            # Aggregates (7)
            self.gen_agg_total_revenue, self.gen_agg_passengers_per_booking,
            self.gen_agg_top_routes, self.gen_agg_revenue_by_fare,
            self.gen_agg_flights_per_route, self.gen_agg_avg_price_by_fare,
            self.gen_agg_busiest_airports,
            # Subqueries (2)
            self.gen_sub_expensive_bookings, self.gen_sub_passengers_multiple_tickets,
            # Timetable (3)
            self.gen_timetable_all, self.gen_timetable_by_dep,
            self.gen_timetable_by_status,
        ]

    def generate(self, count):
        """Generate `count` NL→SQL pairs."""
        generators = self.all_generators()
        examples = []
        seen = set()

        print(f"Generating {count:,} examples using {len(generators)} query templates...")
        print(f"Each template in 2 languages → ~{2 * len(generators)} base patterns")

        report_interval = max(1, count // 20)

        while len(examples) < count:
            gen = self.rng.choice(generators)
            ru = self.rng.choice([True, False])

            try:
                nl, sql = gen(ru)
            except Exception:
                continue

            # Deduplicate by NL query
            key = nl.strip().lower()
            if key in seen:
                continue
            seen.add(key)

            examples.append({"nl": nl, "sql": sql})

            if len(examples) % report_interval == 0:
                pct = len(examples) * 100 // count
                print(f"  {len(examples):>10,} / {count:,} ({pct}%)")

        return examples


def main():
    parser = argparse.ArgumentParser(description="Generate NL→SQL training dataset")
    parser.add_argument("--out", default="training_data/bookings_1m_dataset.json",
                        help="Output file path")
    parser.add_argument("--count", type=int, default=1_000_000,
                        help="Number of examples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("Connecting to database...")
    conn = connect_db()
    cur = conn.cursor()

    gen = DatasetGenerator(cur, rng)
    examples = gen.generate(args.count)

    cur.close()
    conn.close()

    print(f"\nWriting {len(examples):,} examples to {args.out}...")

    # Stream write to avoid loading all JSON in memory
    with open(args.out, "w", encoding="utf-8") as f:
        f.write('{\n  "examples": [\n')
        for i, ex in enumerate(examples):
            if i > 0:
                f.write(",\n")
            f.write("    " + json.dumps(ex, ensure_ascii=False))
        f.write("\n  ]\n}\n")

    print(f"Done! Dataset saved to: {args.out}")
    print(f"Total examples: {len(examples):,}")


if __name__ == "__main__":
    main()
