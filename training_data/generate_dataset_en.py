#!/usr/bin/env python3
"""
Generate English-only NL→SQL training dataset with natural conversational phrasing.

Usage:
    python3 generate_dataset_en.py [--out PATH] [--count N] [--seed N]
"""

import json
import os
import random
import argparse

import psycopg2

# ── Vocabulary ──────────────────────────────────────────────────────────────

SHOW = [
    "show", "show me", "list", "display", "get", "get me", "fetch", "find",
    "return", "give me", "print", "pull up", "let me see", "retrieve", "view",
]

COUNT_PH = [
    "count", "how many", "number of", "total number of", "total count of",
    "count of", "count all",
]

ALL_W = ["all", "every", "the", "all the", "all of the"]

SORT_PH = ["sort by", "order by", "sorted by", "ordered by", "arrange by"]

AVG_W = ["average", "mean", "avg"]

# Conversational wrappers
# Prefixes that require a verb to follow (show, list, get, find, etc.)
VERB_PRE = [
    "", "", "", "",
    "please ", "pls ", "plz ",
    "can you ", "could you ", "would you ",
    "hey ", "hey can you ",
    "can you please ", "pls can you ",
    "I want to ", "I need to ",
    "I'd like to ",
]

# Prefixes safe for any NL (including questions and noun phrases)
SAFE_PRE = [
    "", "", "", "",
    "please ", "pls ", "plz ",
    "hey ",
    "I want to see ", "I need to see ",
    "I'd like to see ",
]

# Words that indicate NL starts with a verb
VERB_STARTS = frozenset({
    "show", "list", "display", "get", "fetch", "find", "return",
    "give", "print", "pull", "let", "retrieve", "view", "sort",
    "count", "search", "describe", "rank",
})

CONV_SUF = [
    "", "", "", "", "",                   # 56% no suffix
    " please", " pls",
    " from the database", " from my db", " from the db",
    " in my database", " in the db",
]

# ── Table name aliases ──────────────────────────────────────────────────────

TBL_NAMES = {
    "airports": [
        "airports", "airport data", "airport info", "the airports",
        "airport list", "airport information", "airports table",
    ],
    "airplanes": [
        "airplanes", "planes", "aircraft", "airplane data",
        "the airplanes", "airplane info", "airplanes table",
    ],
    "flights": [
        "flights", "flight data", "flight info", "the flights",
        "flights table", "flight records",
    ],
    "routes": [
        "routes", "route data", "the routes", "routes table",
        "flight routes", "route info",
    ],
    "bookings": [
        "bookings", "reservations", "booking data", "the bookings",
        "bookings table", "booking records",
    ],
    "tickets": [
        "tickets", "ticket data", "the tickets", "tickets table",
        "ticket records", "ticket info",
    ],
    "segments": [
        "segments", "ticket segments", "segment data", "the segments",
        "segments table",
    ],
    "seats": [
        "seats", "seat data", "the seats", "seats table",
        "seat info", "seat map",
    ],
    "boarding": [
        "boarding passes", "boarding pass data", "the boarding passes",
        "boarding passes table",
    ],
    "timetable": [
        "timetable", "the timetable", "flight schedule",
        "schedule", "timetable data",
    ],
}

TBL_SINGULAR = {
    "airports": "airport",
    "airplanes": "airplane",
    "flights": "flight",
    "routes": "route",
    "bookings": "booking",
    "tickets": "ticket",
    "segments": "segment",
    "seats": "seat",
    "boarding": "boarding pass",
    "timetable": "timetable entry",
}

TBL_SQL = {
    "airports": "bookings.airports_data",
    "airplanes": "bookings.airplanes_data",
    "flights": "bookings.flights",
    "routes": "bookings.routes",
    "bookings": "bookings.bookings",
    "tickets": "bookings.tickets",
    "segments": "bookings.segments",
    "seats": "bookings.seats",
    "boarding": "bookings.boarding_passes",
    "timetable": "bookings.timetable",
}

TBL_RAW = {
    "airports": "airports_data",
    "airplanes": "airplanes_data",
    "flights": "flights",
    "routes": "routes",
    "bookings": "bookings",
    "tickets": "tickets",
    "segments": "segments",
    "seats": "seats",
    "boarding": "boarding_passes",
    "timetable": "timetable",
}

ALL_TABLE_KEYS = [
    "airports", "airplanes", "flights", "routes", "bookings",
    "tickets", "segments", "seats", "boarding",
]


# ── Utilities ───────────────────────────────────────────────────────────────

def sql_lit(val):
    return "'" + str(val).replace("'", "''") + "'"


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
    cur.execute(query)
    return [r[0] for r in cur.fetchall() if r[0] is not None][:limit]


# ── Generator ───────────────────────────────────────────────────────────────

class DatasetGenerator:
    def __init__(self, cur, rng):
        self.cur = cur
        self.rng = rng
        self._load_samples()

    # ── DB sampling ──

    def _load_samples(self):
        c = self.cur
        print("Loading sample values from database...")

        self.cities = sample_values(c, "SELECT DISTINCT city FROM airports ORDER BY 1 LIMIT 500")
        self.airport_codes = sample_values(c, "SELECT airport_code FROM airports ORDER BY 1 LIMIT 500")
        self.airport_names = sample_values(c, "SELECT airport_name FROM airports ORDER BY 1 LIMIT 500")
        self.countries = sample_values(c, "SELECT DISTINCT country FROM airports ORDER BY 1 LIMIT 200")
        self.timezones = sample_values(c, "SELECT DISTINCT timezone FROM airports ORDER BY 1 LIMIT 100")

        self.airplane_codes = sample_values(c, "SELECT airplane_code FROM airplanes")
        self.airplane_models = sample_values(c, "SELECT model FROM airplanes")

        self.flight_statuses = sample_values(c, "SELECT DISTINCT status FROM flights")
        self.route_nos = sample_values(c, "SELECT DISTINCT route_no FROM routes ORDER BY 1 LIMIT 300")
        self.dep_airports = sample_values(c, "SELECT DISTINCT departure_airport FROM routes")
        self.arr_airports = sample_values(c, "SELECT DISTINCT arrival_airport FROM routes")

        self.passenger_names = sample_values(c, "SELECT DISTINCT passenger_name FROM tickets ORDER BY 1 LIMIT 500")
        self.fare_conditions = sample_values(c, "SELECT DISTINCT fare_conditions FROM seats")
        self.seat_nos = sample_values(c, "SELECT DISTINCT seat_no FROM seats ORDER BY 1 LIMIT 200")

        self.book_refs = sample_values(c, "SELECT book_ref FROM bookings ORDER BY 1 LIMIT 300")
        c.execute("SELECT min(total_amount), max(total_amount), avg(total_amount)::int FROM bookings")
        row = c.fetchone()
        self.amount_min, self.amount_max, self.amount_avg = float(row[0]), float(row[1]), int(row[2])

        self.prices = sample_values(c, "SELECT DISTINCT price FROM segments ORDER BY 1 LIMIT 200")

        print(f"  Cities: {len(self.cities)}, Airports: {len(self.airport_codes)}, "
              f"Passengers: {len(self.passenger_names)}, Routes: {len(self.route_nos)}")

    # ── Helpers ──

    def pick(self, lst):
        return self.rng.choice(lst) if lst else "unknown"

    def show(self):
        return self.rng.choice(SHOW)

    def tbl(self, key):
        return self.pick(TBL_NAMES[key])

    def sing(self, key):
        return TBL_SINGULAR[key]

    def sql(self, key):
        return TBL_SQL[key]

    def raw(self, key):
        return TBL_RAW[key]

    def rnd_limit(self):
        return self.rng.choice([1, 2, 3, 5, 10, 15, 20, 25, 50, 100])

    def rnd_amount(self):
        return self.rng.randint(1000, int(self.amount_max))

    def rand_tbl(self):
        return self.rng.choice(ALL_TABLE_KEYS)

    def wrap(self, nl):
        first_word = nl.split()[0].lower() if nl else ""
        if first_word in VERB_STARTS:
            pre = self.rng.choice(VERB_PRE)
        else:
            pre = self.rng.choice(SAFE_PRE)
        suf = self.rng.choice(CONV_SUF)
        result = (pre + nl + suf).strip()
        return result

    # ════════════════════════════════════════════════════════════════════════
    #  GENERIC GENERATORS (work for any table)
    # ════════════════════════════════════════════════════════════════════════

    def gen_select_all(self):
        k = self.rand_tbl()
        name = self.tbl(k)
        nl = self.pick([
            f"{self.show()} {self.pick(ALL_W)} {name}",
            f"{self.show()} {name}",
            f"{self.show()} everything from {name}",
            f"{self.show()} everything in {name}",
            f"what {name} are there",
            f"what {name} do we have",
            f"what's in the {name}",
            f"{self.show()} the {name}",
            f"{name}",
            f"I want to see {name}",
            f"what {name} exist",
            f"list of {name}",
            f"{self.show()} all {name} data",
            f"everything from {name}",
        ])
        return self.wrap(nl), f"select * from {self.sql(k)}"

    def gen_first_row(self):
        k = self.rand_tbl()
        name = self.tbl(k)
        s = self.sing(k)
        nl = self.pick([
            f"{self.show()} first row from {name}",
            f"first line in {name}",
            f"first entry from {name}",
            f"first record in {name}",
            f"{self.show()} one row from {name}",
            f"one {s} from the database",
            f"{self.show()} a single {s}",
            f"first row of {name}",
            f"give me just one row from {name}",
            f"{self.show()} first line of {name}",
            f"the first {s}",
            f"1 row from {name}",
            f"first row in {name}",
            f"first line from {name}",
            f"{self.show()} first {s}",
            f"first {s}",
            f"just one {s}",
            f"any one {s}",
            f"single row from {name}",
        ])
        return self.wrap(nl), f"select * from {self.sql(k)} limit 1"

    def gen_select_limit(self):
        k = self.rand_tbl()
        name = self.tbl(k)
        n = self.rnd_limit()
        nl = self.pick([
            f"{self.show()} {n} {name}",
            f"{self.show()} first {n} rows from {name}",
            f"first {n} {name}",
            f"top {n} {name}",
            f"{self.show()} {n} rows from {name}",
            f"give me {n} {name}",
            f"just {n} {name}",
            f"{self.show()} {n} entries from {name}",
            f"{n} {name}",
            f"{self.show()} {n} records from {name}",
            f"first {n} lines from {name}",
            f"first {n} rows of {name}",
            f"{self.show()} first {n} {name}",
            f"get {n} rows from {name}",
        ])
        return self.wrap(nl), f"select * from {self.sql(k)} limit {n}"

    def gen_count(self):
        k = self.rand_tbl()
        name = self.tbl(k)
        nl = self.pick([
            f"{self.pick(COUNT_PH)} {name}",
            f"how many {name} are there",
            f"how many {name} do we have",
            f"what's the total count of {name}",
            f"total {name}",
            f"count {self.pick(ALL_W)} {name}",
            f"how many {name} exist",
            f"how many rows in {name}",
            f"how many records in {name}",
            f"how many {name} in the database",
            f"{self.show()} count of {name}",
            f"number of {name} in the db",
        ])
        return self.wrap(nl), f"select count(*) from {self.sql(k)}"

    def gen_preview(self):
        k = self.rand_tbl()
        name = self.tbl(k)
        n = self.rng.choice([3, 5, 10])
        nl = self.pick([
            f"preview {name}",
            f"sample from {name}",
            f"quick look at {name}",
            f"what does {name} look like",
            f"peek at {name}",
            f"a few rows from {name}",
            f"some {name}",
            f"show me some {name}",
            f"a sample of {name}",
            f"give me a preview of {name}",
        ])
        return self.wrap(nl), f"select * from {self.sql(k)} limit {n}"

    def gen_list_tables(self):
        nl = self.pick([
            "what tables are in my database",
            "show me all tables",
            "list all tables",
            "what tables do I have",
            "show database tables",
            "what tables exist",
            "show me my tables",
            "list tables in my db",
            "what tables are available",
            "show all tables in the database",
            "list tables",
            "tables in my database",
            "show tables",
            "database tables",
            "what tables are there",
            "all tables",
        ])
        return (self.wrap(nl),
                "select table_name from information_schema.tables "
                "where table_schema = 'bookings' and table_type = 'BASE TABLE' "
                "order by table_name")

    def gen_describe_table(self):
        k = self.rand_tbl()
        name = self.tbl(k)
        raw = self.raw(k)
        nl = self.pick([
            f"what columns does {name} have",
            f"describe {name}",
            f"what fields are in {name}",
            f"show columns of {name}",
            f"what's the structure of {name}",
            f"schema of {name}",
            f"{name} columns",
            f"what data is in {name}",
            f"columns in {name}",
            f"describe the {name} table",
            f"what info does {name} contain",
            f"fields of {name}",
        ])
        return (self.wrap(nl),
                f"select column_name, data_type from information_schema.columns "
                f"where table_schema = 'bookings' and table_name = '{raw}' "
                f"order by ordinal_position")

    # ════════════════════════════════════════════════════════════════════════
    #  AIRPORTS
    # ════════════════════════════════════════════════════════════════════════

    def gen_airports_by_city(self):
        city = self.pick(self.cities)
        name = self.tbl("airports")
        nl = self.pick([
            f"{self.show()} {name} in {city}",
            f"{self.show()} {name} where city is {city}",
            f"what {name} are in {city}",
            f"which {name} are in {city}",
            f"{name} in {city}",
            f"find {name} in city {city}",
            f"{self.show()} {name} from {city}",
            f"{self.show()} {name} located in {city}",
            f"airports in {city}",
        ])
        return self.wrap(nl), f"select * from bookings.airports_data where city = {sql_lit(city)}"

    def gen_airports_by_country(self):
        country = self.pick(self.countries)
        name = self.tbl("airports")
        nl = self.pick([
            f"{self.show()} {name} in {country}",
            f"what {name} are in {country}",
            f"{name} in {country}",
            f"which {name} are located in {country}",
            f"airports in {country}",
            f"find {name} in {country}",
            f"{self.show()} all {name} in {country}",
        ])
        return self.wrap(nl), f"select * from bookings.airports_data where country = {sql_lit(country)}"

    def gen_airports_count(self):
        nl = self.pick([
            f"{self.pick(COUNT_PH)} airports",
            "how many airports are there",
            "how many airports do we have",
            "total airports",
            "how many airports exist",
            "airport count",
        ])
        return self.wrap(nl), "select count(*) from bookings.airports_data"

    def gen_airports_count_by_country(self):
        country = self.pick(self.countries)
        nl = self.pick([
            f"{self.pick(COUNT_PH)} airports in {country}",
            f"how many airports are in {country}",
            f"how many airports does {country} have",
            f"airports in {country} count",
        ])
        return self.wrap(nl), f"select count(*) from bookings.airports_data where country = {sql_lit(country)}"

    def gen_airports_by_timezone(self):
        tz = self.pick(self.timezones)
        name = self.tbl("airports")
        nl = self.pick([
            f"{self.show()} {name} in timezone {tz}",
            f"{name} in timezone {tz}",
            f"airports with timezone {tz}",
            f"which airports are in timezone {tz}",
        ])
        return self.wrap(nl), f"select * from bookings.airports_data where timezone = {sql_lit(tz)}"

    def gen_airports_group_country(self):
        nl = self.pick([
            "count airports by country",
            "airports per country",
            "how many airports in each country",
            "group airports by country",
            "airport count by country",
            "number of airports per country",
        ])
        return self.wrap(nl), "select country, count(*) from bookings.airports_data group by country order by count desc limit 50"

    def gen_airports_by_code(self):
        code = self.pick(self.airport_codes)
        nl = self.pick([
            f"{self.show()} airport {code}",
            f"airport with code {code}",
            f"find airport {code}",
            f"info about airport {code}",
            f"airport {code}",
            f"what is airport {code}",
            f"details for airport {code}",
        ])
        return self.wrap(nl), f"select * from bookings.airports_data where airport_code = {sql_lit(code)}"

    def gen_airports_search_name(self):
        name_part = self.pick(self.airport_names)
        if len(name_part) > 8:
            name_part = name_part[:8]
        nl = self.pick([
            f"find airports with name containing {name_part}",
            f"search airports for {name_part}",
            f"airports named like {name_part}",
            f"{self.show()} airports matching {name_part}",
        ])
        return self.wrap(nl), f"select * from bookings.airports_data where airport_name ilike {sql_lit('%' + name_part + '%')}"

    def gen_airports_limit(self):
        n = self.rnd_limit()
        name = self.tbl("airports")
        nl = self.pick([
            f"{self.show()} {n} {name}",
            f"first {n} airports",
            f"{n} airports",
            f"give me {n} airports",
            f"top {n} airports",
        ])
        return self.wrap(nl), f"select * from bookings.airports_data limit {n}"

    # ════════════════════════════════════════════════════════════════════════
    #  AIRPLANES
    # ════════════════════════════════════════════════════════════════════════

    def gen_airplanes_all(self):
        name = self.tbl("airplanes")
        nl = self.pick([
            f"{self.show()} {self.pick(ALL_W)} {name}",
            f"{self.show()} {name}",
            f"what {name} do we have",
            f"list of {name}",
            f"all available {name}",
            f"what {name} are there",
        ])
        return self.wrap(nl), "select * from bookings.airplanes_data"

    def gen_airplanes_by_range(self):
        r = self.rng.choice([3000, 5000, 7000, 10000, 12000])
        name = self.tbl("airplanes")
        nl = self.pick([
            f"{self.show()} {name} with range greater than {r}",
            f"{name} with range over {r}",
            f"{name} that can fly more than {r} km",
            f"which {name} have range above {r}",
            f"long range {name} over {r}",
            f"{self.show()} {name} where range is more than {r}",
        ])
        return self.wrap(nl), f"select * from bookings.airplanes_data where range > {r}"

    def gen_airplanes_by_speed(self):
        s = self.rng.choice([800, 850, 900, 950])
        name = self.tbl("airplanes")
        nl = self.pick([
            f"{self.show()} {name} faster than {s} km/h",
            f"{name} with speed above {s}",
            f"fast {name} over {s} km/h",
            f"which {name} are faster than {s}",
            f"{self.show()} {name} with speed greater than {s}",
        ])
        return self.wrap(nl), f"select * from bookings.airplanes_data where speed > {s}"

    def gen_airplanes_by_code(self):
        code = self.pick(self.airplane_codes)
        nl = self.pick([
            f"{self.show()} airplane {code}",
            f"info about plane {code}",
            f"airplane with code {code}",
            f"find airplane {code}",
            f"airplane {code}",
            f"details for aircraft {code}",
        ])
        return self.wrap(nl), f"select * from bookings.airplanes_data where airplane_code = {sql_lit(code)}"

    def gen_airplanes_sort_range(self):
        desc = self.rng.choice([True, False])
        d = "desc" if desc else "asc"
        w = "descending" if desc else "ascending"
        nl = self.pick([
            f"sort airplanes by range {w}",
            f"airplanes sorted by range {d}",
            f"airplanes ordered by range {w}",
            f"{self.show()} airplanes by range {w}",
            f"rank airplanes by range {w}",
        ])
        return self.wrap(nl), f"select * from bookings.airplanes_data order by range {d}"

    def gen_airplanes_count(self):
        nl = self.pick([
            f"{self.pick(COUNT_PH)} airplanes",
            "how many planes are there",
            "how many aircraft do we have",
            "total airplanes",
            "airplane count",
        ])
        return self.wrap(nl), "select count(*) from bookings.airplanes_data"

    def gen_airplanes_avg_range(self):
        a = self.pick(AVG_W)
        nl = self.pick([
            f"{a} range of airplanes",
            f"{a} airplane range",
            f"what is the {a} range of our planes",
            f"{a} flight range",
        ])
        return self.wrap(nl), "select avg(range) from bookings.airplanes_data"

    def gen_airplanes_long_range(self):
        name = self.tbl("airplanes")
        nl = self.pick([
            f"long range {name}",
            f"{self.show()} long range planes",
            f"planes that can fly far",
            f"long haul aircraft",
            f"which planes have the longest range",
        ])
        return self.wrap(nl), "select * from bookings.airplanes_data where range > 10000"

    # ════════════════════════════════════════════════════════════════════════
    #  FLIGHTS
    # ════════════════════════════════════════════════════════════════════════

    def gen_flights_all(self):
        name = self.tbl("flights")
        n = self.rnd_limit()
        nl = self.pick([
            f"{self.show()} {n} {name}",
            f"first {n} flights",
            f"{n} flights",
            f"{self.show()} some flights",
            f"{self.show()} recent flights",
        ])
        return self.wrap(nl), f"select * from bookings.flights limit {n}"

    def gen_flights_by_status(self):
        status = self.pick(self.flight_statuses)
        name = self.tbl("flights")
        nl = self.pick([
            f"{self.show()} {name} with status {status}",
            f"{status.lower()} flights",
            f"{self.show()} {status.lower()} flights",
            f"flights that are {status.lower()}",
            f"which flights are {status.lower()}",
            f"find {status.lower()} flights",
            f"{name} where status is {status}",
        ])
        return self.wrap(nl), f"select * from bookings.flights where status = {sql_lit(status)}"

    def gen_flights_count_by_status(self):
        status = self.pick(self.flight_statuses)
        nl = self.pick([
            f"{self.pick(COUNT_PH)} flights with status {status}",
            f"how many {status.lower()} flights",
            f"how many flights are {status.lower()}",
            f"count {status.lower()} flights",
        ])
        return self.wrap(nl), f"select count(*) from bookings.flights where status = {sql_lit(status)}"

    def gen_flights_by_route(self):
        route = self.pick(self.route_nos)
        nl = self.pick([
            f"{self.show()} flights on route {route}",
            f"flights for route {route}",
            f"flights on route number {route}",
            f"find flights for route {route}",
            f"{self.show()} flights where route is {route}",
        ])
        return self.wrap(nl), f"select * from bookings.flights where route_no = {sql_lit(route)}"

    def gen_flights_count(self):
        nl = self.pick([
            f"{self.pick(COUNT_PH)} flights",
            "how many flights are there",
            "total flights",
            "total number of flights",
            "how many flights do we have",
        ])
        return self.wrap(nl), "select count(*) from bookings.flights"

    def gen_flights_group_status(self):
        nl = self.pick([
            "count flights by status",
            "flights per status",
            "how many flights for each status",
            "group flights by status",
            "flight count by status",
            "flight status breakdown",
            "breakdown of flights by status",
        ])
        return self.wrap(nl), "select status, count(*) from bookings.flights group by status order by count desc"

    def gen_flights_departed(self):
        nl = self.pick([
            f"{self.show()} departed flights",
            "flights that have departed",
            "which flights already departed",
            f"{self.show()} flights that left",
            "departed flights",
        ])
        return self.wrap(nl), "select * from bookings.flights where actual_departure is not null limit 50"

    def gen_flights_delayed(self):
        nl = self.pick([
            f"{self.show()} delayed flights",
            "delayed flights",
            "which flights are delayed",
            "flights that are late",
            "find delayed flights",
            "any delayed flights",
            f"{self.show()} late flights",
        ])
        return self.wrap(nl), "select * from bookings.flights where status = 'Delayed' limit 50"

    def gen_flights_cancelled(self):
        nl = self.pick([
            f"{self.show()} cancelled flights",
            "cancelled flights",
            "which flights are cancelled",
            "find cancelled flights",
            "any cancelled flights",
            f"{self.show()} canceled flights",
        ])
        return self.wrap(nl), "select * from bookings.flights where status = 'Cancelled' limit 50"

    # ════════════════════════════════════════════════════════════════════════
    #  ROUTES
    # ════════════════════════════════════════════════════════════════════════

    def gen_routes_all(self):
        name = self.tbl("routes")
        nl = self.pick([
            f"{self.show()} {self.pick(ALL_W)} {name}",
            f"{self.show()} {name}",
            f"what {name} are available",
            f"list of {name}",
            f"all {name}",
        ])
        return self.wrap(nl), "select * from bookings.routes"

    def gen_routes_by_dep(self):
        dep = self.pick(self.dep_airports)
        name = self.tbl("routes")
        nl = self.pick([
            f"{self.show()} {name} from {dep}",
            f"routes departing from {dep}",
            f"flights from {dep}",
            f"departures from {dep}",
            f"where can I fly from {dep}",
            f"which routes go from {dep}",
            f"{name} from {dep}",
        ])
        return self.wrap(nl), f"select * from bookings.routes where departure_airport = {sql_lit(dep)}"

    def gen_routes_by_arr(self):
        arr = self.pick(self.arr_airports)
        name = self.tbl("routes")
        nl = self.pick([
            f"{self.show()} {name} to {arr}",
            f"routes arriving at {arr}",
            f"flights to {arr}",
            f"routes going to {arr}",
            f"which routes go to {arr}",
            f"{name} to {arr}",
        ])
        return self.wrap(nl), f"select * from bookings.routes where arrival_airport = {sql_lit(arr)}"

    def gen_routes_by_airplane(self):
        code = self.pick(self.airplane_codes)
        nl = self.pick([
            f"{self.show()} routes for airplane {code}",
            f"routes using airplane {code}",
            f"routes with plane {code}",
            f"which routes use aircraft {code}",
            f"routes for plane {code}",
        ])
        return self.wrap(nl), f"select * from bookings.routes where airplane_code = {sql_lit(code)}"

    def gen_routes_count(self):
        nl = self.pick([
            f"{self.pick(COUNT_PH)} routes",
            "how many routes are there",
            "total routes",
            "how many routes do we have",
        ])
        return self.wrap(nl), "select count(*) from bookings.routes"

    def gen_routes_dep_arr(self):
        dep = self.pick(self.dep_airports)
        arr = self.pick(self.arr_airports)
        nl = self.pick([
            f"{self.show()} route from {dep} to {arr}",
            f"routes from {dep} to {arr}",
            f"flights from {dep} to {arr}",
            f"is there a route from {dep} to {arr}",
            f"find route from {dep} to {arr}",
            f"how to get from {dep} to {arr}",
        ])
        return self.wrap(nl), (
            f"select * from bookings.routes "
            f"where departure_airport = {sql_lit(dep)} and arrival_airport = {sql_lit(arr)}")

    # ════════════════════════════════════════════════════════════════════════
    #  BOOKINGS
    # ════════════════════════════════════════════════════════════════════════

    def gen_bookings_all(self):
        name = self.tbl("bookings")
        n = self.rnd_limit()
        nl = self.pick([
            f"{self.show()} {n} {name}",
            f"first {n} bookings",
            f"{n} bookings",
            f"latest {n} bookings",
            f"{self.show()} some bookings",
        ])
        return self.wrap(nl), f"select * from bookings.bookings limit {n}"

    def gen_bookings_by_ref(self):
        ref = self.pick(self.book_refs)
        nl = self.pick([
            f"{self.show()} booking {ref}",
            f"find booking {ref}",
            f"booking with reference {ref}",
            f"look up booking {ref}",
            f"booking {ref}",
            f"details for booking {ref}",
            f"info about booking {ref}",
        ])
        return self.wrap(nl), f"select * from bookings.bookings where book_ref = {sql_lit(ref)}"

    def gen_bookings_amount_gt(self):
        amount = self.rnd_amount()
        name = self.tbl("bookings")
        nl = self.pick([
            f"{self.show()} {name} with amount greater than {amount}",
            f"{name} over {amount}",
            f"{name} above {amount}",
            f"bookings more than {amount}",
            f"expensive bookings over {amount}",
            f"{self.show()} bookings where total amount exceeds {amount}",
            f"bookings with total above {amount}",
        ])
        return self.wrap(nl), f"select * from bookings.bookings where total_amount > {amount} limit 50"

    def gen_bookings_amount_lt(self):
        amount = self.rng.randint(1000, 10000)
        nl = self.pick([
            f"{self.show()} bookings with amount less than {amount}",
            f"bookings under {amount}",
            f"bookings below {amount}",
            f"cheap bookings under {amount}",
            f"bookings cheaper than {amount}",
        ])
        return self.wrap(nl), f"select * from bookings.bookings where total_amount < {amount} limit 50"

    def gen_bookings_amount_between(self):
        lo = self.rng.randint(1000, 50000)
        hi = lo + self.rng.randint(5000, 100000)
        nl = self.pick([
            f"{self.show()} bookings with amount between {lo} and {hi}",
            f"bookings from {lo} to {hi}",
            f"bookings in range {lo} to {hi}",
            f"bookings costing between {lo} and {hi}",
        ])
        return self.wrap(nl), f"select * from bookings.bookings where total_amount between {lo} and {hi} limit 50"

    def gen_bookings_count(self):
        nl = self.pick([
            f"{self.pick(COUNT_PH)} bookings",
            "how many bookings are there",
            "total bookings",
            "total number of reservations",
        ])
        return self.wrap(nl), "select count(*) from bookings.bookings"

    def gen_bookings_avg_amount(self):
        a = self.pick(AVG_W)
        nl = self.pick([
            f"{a} booking amount",
            f"what is the {a} booking amount",
            f"{a} total amount of bookings",
            f"{a} reservation cost",
        ])
        return self.wrap(nl), "select avg(total_amount) from bookings.bookings"

    def gen_bookings_max_amount(self):
        nl = self.pick([
            "maximum booking amount",
            "largest booking",
            "most expensive booking",
            "biggest booking amount",
            "what is the highest booking amount",
            "max booking amount",
        ])
        return self.wrap(nl), "select max(total_amount) from bookings.bookings"

    def gen_bookings_min_amount(self):
        nl = self.pick([
            "minimum booking amount",
            "smallest booking",
            "cheapest booking",
            "lowest booking amount",
            "what is the lowest booking amount",
            "min booking amount",
        ])
        return self.wrap(nl), "select min(total_amount) from bookings.bookings"

    def gen_bookings_sort_amount(self):
        desc = self.rng.choice([True, False])
        d = "desc" if desc else "asc"
        n = self.rnd_limit()
        nl = self.pick([
            f"top {n} bookings by amount {'descending' if desc else 'ascending'}",
            f"{'most expensive' if desc else 'cheapest'} {n} bookings",
            f"bookings sorted by amount {d}",
            f"bookings ordered by total {'highest first' if desc else 'lowest first'}",
            f"{self.show()} top {n} {'most expensive' if desc else 'cheapest'} bookings",
        ])
        return self.wrap(nl), f"select * from bookings.bookings order by total_amount {d} limit {n}"

    # ════════════════════════════════════════════════════════════════════════
    #  TICKETS
    # ════════════════════════════════════════════════════════════════════════

    def gen_tickets_all(self):
        name = self.tbl("tickets")
        n = self.rnd_limit()
        nl = self.pick([
            f"{self.show()} {n} {name}",
            f"first {n} tickets",
            f"{n} tickets",
            f"{self.show()} some tickets",
        ])
        return self.wrap(nl), f"select * from bookings.tickets limit {n}"

    def gen_tickets_by_name(self):
        pname = self.pick(self.passenger_names)
        nl = self.pick([
            f"{self.show()} tickets for {pname}",
            f"tickets for passenger {pname}",
            f"find tickets for {pname}",
            f"{pname}'s tickets",
            f"{self.show()} tickets where passenger is {pname}",
            f"tickets belonging to {pname}",
            f"look up tickets for {pname}",
        ])
        return self.wrap(nl), f"select * from bookings.tickets where passenger_name = {sql_lit(pname)}"

    def gen_tickets_by_name_like(self):
        pname = self.pick(self.passenger_names)
        first = pname.split()[0] if " " in pname else pname
        nl = self.pick([
            f"{self.show()} tickets for passengers named {first}",
            f"tickets where passenger name starts with {first}",
            f"find passengers named {first}",
            f"search tickets for {first}",
            f"tickets matching name {first}",
        ])
        return self.wrap(nl), f"select * from bookings.tickets where passenger_name ilike {sql_lit(first + '%')} limit 50"

    def gen_tickets_by_booking(self):
        ref = self.pick(self.book_refs)
        nl = self.pick([
            f"{self.show()} tickets for booking {ref}",
            f"tickets in booking {ref}",
            f"tickets linked to booking {ref}",
            f"what tickets are in booking {ref}",
            f"tickets with book ref {ref}",
        ])
        return self.wrap(nl), f"select * from bookings.tickets where book_ref = {sql_lit(ref)}"

    def gen_tickets_count(self):
        nl = self.pick([
            f"{self.pick(COUNT_PH)} tickets",
            "how many tickets are there",
            "total tickets",
            "how many tickets sold",
        ])
        return self.wrap(nl), "select count(*) from bookings.tickets"

    def gen_tickets_count_by_booking(self):
        ref = self.pick(self.book_refs)
        nl = self.pick([
            f"{self.pick(COUNT_PH)} tickets in booking {ref}",
            f"how many tickets are in booking {ref}",
            f"ticket count for booking {ref}",
        ])
        return self.wrap(nl), f"select count(*) from bookings.tickets where book_ref = {sql_lit(ref)}"

    def gen_tickets_outbound(self):
        n = self.rnd_limit()
        nl = self.pick([
            f"{self.show()} {n} outbound tickets",
            f"outbound tickets",
            f"one way tickets",
            f"{self.show()} outgoing tickets",
        ])
        return self.wrap(nl), f"select * from bookings.tickets where outbound = true limit {n}"

    def gen_tickets_return(self):
        n = self.rnd_limit()
        nl = self.pick([
            f"{self.show()} {n} return tickets",
            f"return tickets",
            f"inbound tickets",
            f"{self.show()} return trip tickets",
        ])
        return self.wrap(nl), f"select * from bookings.tickets where outbound = false limit {n}"

    # ════════════════════════════════════════════════════════════════════════
    #  SEGMENTS
    # ════════════════════════════════════════════════════════════════════════

    def gen_segments_all(self):
        name = self.tbl("segments")
        n = self.rnd_limit()
        nl = self.pick([
            f"{self.show()} {n} {name}",
            f"first {n} segments",
            f"{n} ticket segments",
        ])
        return self.wrap(nl), f"select * from bookings.segments limit {n}"

    def gen_segments_by_fare(self):
        fare = self.pick(self.fare_conditions)
        n = self.rnd_limit()
        nl = self.pick([
            f"{self.show()} {n} {fare.lower()} class segments",
            f"{fare.lower()} class tickets",
            f"segments with fare {fare}",
            f"{self.show()} {fare.lower()} segments",
            f"tickets in {fare.lower()} class",
        ])
        return self.wrap(nl), f"select * from bookings.segments where fare_conditions = {sql_lit(fare)} limit {n}"

    def gen_segments_price_gt(self):
        price = self.rng.choice([5000, 10000, 20000, 50000, 100000])
        nl = self.pick([
            f"{self.show()} segments with price above {price}",
            f"segments costing more than {price}",
            f"expensive segments over {price}",
            f"tickets priced above {price}",
            f"segments where price exceeds {price}",
        ])
        return self.wrap(nl), f"select * from bookings.segments where price > {price} limit 50"

    def gen_segments_avg_price(self):
        a = self.pick(AVG_W)
        nl = self.pick([
            f"{a} segment price",
            f"{a} ticket price",
            f"what is the {a} segment price",
            f"{a} price of segments",
        ])
        return self.wrap(nl), "select avg(price) from bookings.segments"

    def gen_segments_avg_by_fare(self):
        nl = self.pick([
            "average price by fare class",
            "avg price per class",
            "average ticket price by class",
            "what is the average price for each fare class",
            "price breakdown by class",
        ])
        return self.wrap(nl), "select fare_conditions, avg(price) from bookings.segments group by fare_conditions order by avg desc"

    def gen_segments_count(self):
        nl = self.pick([
            f"{self.pick(COUNT_PH)} segments",
            "how many segments are there",
            "total segments",
        ])
        return self.wrap(nl), "select count(*) from bookings.segments"

    def gen_segments_count_by_fare(self):
        nl = self.pick([
            "count segments by fare class",
            "segments per class",
            "how many segments in each class",
            "segment count by fare class",
        ])
        return self.wrap(nl), "select fare_conditions, count(*) from bookings.segments group by fare_conditions order by count desc"

    def gen_segments_most_expensive(self):
        nl = self.pick([
            "most expensive segment",
            "highest priced ticket",
            "most expensive ticket segment",
            "priciest segment",
            "segment with highest price",
        ])
        return self.wrap(nl), "select * from bookings.segments order by price desc limit 1"

    # ════════════════════════════════════════════════════════════════════════
    #  SEATS
    # ════════════════════════════════════════════════════════════════════════

    def gen_seats_all(self):
        name = self.tbl("seats")
        nl = self.pick([
            f"{self.show()} {self.pick(ALL_W)} {name}",
            f"{self.show()} {name}",
            f"all available seats",
            f"list of seats",
        ])
        return self.wrap(nl), "select * from bookings.seats"

    def gen_seats_by_airplane(self):
        code = self.pick(self.airplane_codes)
        nl = self.pick([
            f"{self.show()} seats on airplane {code}",
            f"seats in plane {code}",
            f"seat map for airplane {code}",
            f"what seats does airplane {code} have",
            f"seats for aircraft {code}",
        ])
        return self.wrap(nl), f"select * from bookings.seats where airplane_code = {sql_lit(code)}"

    def gen_seats_by_fare(self):
        fare = self.pick(self.fare_conditions)
        nl = self.pick([
            f"{self.show()} {fare.lower()} seats",
            f"{fare.lower()} class seats",
            f"seats with fare class {fare}",
            f"all {fare.lower()} seats",
        ])
        return self.wrap(nl), f"select * from bookings.seats where fare_conditions = {sql_lit(fare)}"

    def gen_seats_count_by_airplane(self):
        code = self.pick(self.airplane_codes)
        nl = self.pick([
            f"{self.pick(COUNT_PH)} seats on airplane {code}",
            f"how many seats does airplane {code} have",
            f"seat count for plane {code}",
            f"total seats in aircraft {code}",
        ])
        return self.wrap(nl), f"select count(*) from bookings.seats where airplane_code = {sql_lit(code)}"

    def gen_seats_group_fare(self):
        code = self.pick(self.airplane_codes)
        nl = self.pick([
            f"seats by fare class on airplane {code}",
            f"seat class breakdown for plane {code}",
            f"how many seats per class on airplane {code}",
            f"seat distribution by class for airplane {code}",
        ])
        return self.wrap(nl), (
            f"select fare_conditions, count(*) from bookings.seats "
            f"where airplane_code = {sql_lit(code)} group by fare_conditions")

    def gen_seats_count(self):
        nl = self.pick([
            f"{self.pick(COUNT_PH)} seats",
            "how many seats are there",
            "total seats",
            "total number of seats",
        ])
        return self.wrap(nl), "select count(*) from bookings.seats"

    # ════════════════════════════════════════════════════════════════════════
    #  BOARDING PASSES
    # ════════════════════════════════════════════════════════════════════════

    def gen_boarding_all(self):
        name = self.tbl("boarding")
        n = self.rnd_limit()
        nl = self.pick([
            f"{self.show()} {n} {name}",
            f"first {n} boarding passes",
            f"{n} boarding passes",
            f"{self.show()} some boarding passes",
        ])
        return self.wrap(nl), f"select * from bookings.boarding_passes limit {n}"

    def gen_boarding_by_flight(self):
        fid = self.rng.randint(1, 37000)
        nl = self.pick([
            f"{self.show()} boarding passes for flight {fid}",
            f"boarding passes on flight {fid}",
            f"who is boarding flight {fid}",
            f"passengers on flight {fid}",
            f"boarding list for flight {fid}",
        ])
        return self.wrap(nl), f"select * from bookings.boarding_passes where flight_id = {fid} limit 50"

    def gen_boarding_by_seat(self):
        seat = self.pick(self.seat_nos)
        nl = self.pick([
            f"{self.show()} boarding passes for seat {seat}",
            f"who sat in seat {seat}",
            f"boarding passes with seat {seat}",
            f"find boarding pass for seat {seat}",
        ])
        return self.wrap(nl), f"select * from bookings.boarding_passes where seat_no = {sql_lit(seat)} limit 50"

    def gen_boarding_count(self):
        nl = self.pick([
            f"{self.pick(COUNT_PH)} boarding passes",
            "how many boarding passes are there",
            "total boarding passes",
        ])
        return self.wrap(nl), "select count(*) from bookings.boarding_passes"

    # ════════════════════════════════════════════════════════════════════════
    #  JOIN QUERIES
    # ════════════════════════════════════════════════════════════════════════

    def gen_join_tickets_segments(self):
        name = self.pick(self.passenger_names)
        nl = self.pick([
            f"{self.show()} segments for passenger {name}",
            f"ticket details for {name}",
            f"what did {name} pay for tickets",
            f"fare info for {name}",
            f"tickets and prices for {name}",
        ])
        return self.wrap(nl), (
            f"select t.passenger_name, s.fare_conditions, s.price "
            f"from bookings.tickets t join bookings.segments s on t.ticket_no = s.ticket_no "
            f"where t.passenger_name = {sql_lit(name)} limit 50")

    def gen_join_tickets_bookings(self):
        ref = self.pick(self.book_refs)
        nl = self.pick([
            f"{self.show()} tickets and booking info for {ref}",
            f"booking {ref} with ticket details",
            f"full booking info for {ref}",
            f"passengers and amount for booking {ref}",
        ])
        return self.wrap(nl), (
            f"select t.ticket_no, t.passenger_name, b.total_amount "
            f"from bookings.tickets t join bookings.bookings b on t.book_ref = b.book_ref "
            f"where b.book_ref = {sql_lit(ref)}")

    def gen_join_flights_routes(self):
        dep = self.pick(self.dep_airports)
        nl = self.pick([
            f"{self.show()} flights departing from {dep} with route info",
            f"flights from {dep} with route details",
            f"flight and route info for departures from {dep}",
            f"all flights from airport {dep}",
        ])
        return self.wrap(nl), (
            f"select f.flight_id, f.status, r.departure_airport, r.arrival_airport "
            f"from bookings.flights f join bookings.routes r on f.route_no = r.route_no "
            f"where r.departure_airport = {sql_lit(dep)} limit 50")

    def gen_join_boarding_tickets(self):
        name = self.pick(self.passenger_names)
        nl = self.pick([
            f"{self.show()} boarding passes for {name}",
            f"boarding info for passenger {name}",
            f"where does {name} sit",
            f"seat assignment for {name}",
        ])
        return self.wrap(nl), (
            f"select bp.seat_no, bp.boarding_no, t.passenger_name "
            f"from bookings.boarding_passes bp "
            f"join bookings.tickets t on bp.ticket_no = t.ticket_no "
            f"where t.passenger_name = {sql_lit(name)} limit 50")

    def gen_join_routes_airplanes(self):
        n = self.rnd_limit()
        nl = self.pick([
            f"{self.show()} {n} routes with airplane models",
            f"routes and their airplanes",
            f"which planes fly which routes",
            f"{self.show()} {n} routes with plane info",
        ])
        return self.wrap(nl), (
            f"select r.route_no, r.departure_airport, r.arrival_airport, a.model "
            f"from bookings.routes r join bookings.airplanes_data a on r.airplane_code = a.airplane_code "
            f"limit {n}")

    def gen_join_seats_airplanes(self):
        code = self.pick(self.airplane_codes)
        nl = self.pick([
            f"seats and airplane info for {code}",
            f"seat layout with plane details for {code}",
            f"seats and model info for aircraft {code}",
        ])
        return self.wrap(nl), (
            f"select s.seat_no, s.fare_conditions, a.model, a.range "
            f"from bookings.seats s join bookings.airplanes_data a on s.airplane_code = a.airplane_code "
            f"where s.airplane_code = {sql_lit(code)}")

    # ════════════════════════════════════════════════════════════════════════
    #  AGGREGATE QUERIES
    # ════════════════════════════════════════════════════════════════════════

    def gen_agg_total_revenue(self):
        nl = self.pick([
            "total revenue from all bookings",
            "total booking revenue",
            "sum of all booking amounts",
            "how much revenue from bookings",
            "total money from bookings",
            "overall revenue",
        ])
        return self.wrap(nl), "select sum(total_amount) from bookings.bookings"

    def gen_agg_avg_tickets_per_booking(self):
        nl = self.pick([
            "average number of tickets per booking",
            "avg tickets per booking",
            "how many tickets per booking on average",
            "average tickets per reservation",
        ])
        return self.wrap(nl), (
            "select avg(cnt) from "
            "(select book_ref, count(*) as cnt from bookings.tickets group by book_ref) sub")

    def gen_agg_top_routes(self):
        n = self.rng.choice([5, 10, 20])
        nl = self.pick([
            f"top {n} most popular routes",
            f"{n} busiest routes",
            f"most popular {n} routes",
            f"top {n} routes by number of flights",
            f"which are the {n} most used routes",
        ])
        return self.wrap(nl), (
            f"select route_no, count(*) as flight_count from bookings.flights "
            f"group by route_no order by flight_count desc limit {n}")

    def gen_agg_revenue_by_fare(self):
        nl = self.pick([
            "revenue by fare class",
            "total revenue per class",
            "how much money from each fare class",
            "revenue breakdown by class",
            "income by fare class",
        ])
        return self.wrap(nl), (
            "select fare_conditions, sum(price) as revenue "
            "from bookings.segments group by fare_conditions order by revenue desc")

    def gen_agg_flights_per_route(self):
        nl = self.pick([
            "flights per route",
            "how many flights on each route",
            "flight count by route",
            "number of flights per route",
        ])
        return self.wrap(nl), (
            "select route_no, count(*) from bookings.flights "
            "group by route_no order by count desc limit 50")

    def gen_agg_avg_price_by_fare(self):
        nl = self.pick([
            "average ticket price by fare class",
            "avg price per class",
            "what is the average price for each class",
            "average segment price by fare",
        ])
        return self.wrap(nl), (
            "select fare_conditions, avg(price) as avg_price "
            "from bookings.segments group by fare_conditions order by avg_price desc")

    def gen_agg_busiest_airports(self):
        n = self.rng.choice([5, 10, 20])
        nl = self.pick([
            f"top {n} busiest departure airports",
            f"{n} most active airports",
            f"busiest {n} airports",
            f"top {n} airports by number of routes",
            f"which airports have the most departures",
        ])
        return self.wrap(nl), (
            f"select departure_airport, count(*) as cnt from bookings.routes "
            f"group by departure_airport order by cnt desc limit {n}")

    # ════════════════════════════════════════════════════════════════════════
    #  SUBQUERIES
    # ════════════════════════════════════════════════════════════════════════

    def gen_sub_expensive_bookings(self):
        nl = self.pick([
            f"{self.show()} bookings above average amount",
            "bookings that cost more than average",
            "expensive bookings above avg",
            "bookings above the average price",
            "which bookings are above average",
        ])
        return self.wrap(nl), (
            "select * from bookings.bookings "
            "where total_amount > (select avg(total_amount) from bookings.bookings) limit 50")

    def gen_sub_passengers_multiple_tickets(self):
        nl = self.pick([
            f"{self.show()} passengers with more than 2 tickets",
            "passengers who bought multiple tickets",
            "frequent flyers with many tickets",
            "passengers with at least 3 tickets",
            "who has the most tickets",
        ])
        return self.wrap(nl), (
            "select passenger_name, count(*) as cnt from bookings.tickets "
            "group by passenger_name having count(*) > 2 order by cnt desc limit 50")

    # ════════════════════════════════════════════════════════════════════════
    #  TIMETABLE (view)
    # ════════════════════════════════════════════════════════════════════════

    def gen_timetable_all(self):
        name = self.tbl("timetable")
        n = self.rnd_limit()
        nl = self.pick([
            f"{self.show()} {n} entries from {name}",
            f"first {n} rows of {name}",
            f"{self.show()} {name}",
            f"{n} entries from {name}",
            f"{self.show()} the {name}",
        ])
        return self.wrap(nl), f"select * from bookings.timetable limit {n}"

    def gen_timetable_by_dep(self):
        dep = self.pick(self.dep_airports)
        name = self.tbl("timetable")
        nl = self.pick([
            f"{self.show()} {name} for departures from {dep}",
            f"schedule from {dep}",
            f"departure schedule for {dep}",
            f"flights schedule from {dep}",
        ])
        return self.wrap(nl), (
            f"select * from bookings.timetable "
            f"where departure_airport = {sql_lit(dep)} limit 50")

    def gen_timetable_by_status(self):
        status = self.pick(self.flight_statuses)
        name = self.tbl("timetable")
        nl = self.pick([
            f"{self.show()} {name} entries with status {status}",
            f"{status.lower()} entries in {name}",
            f"schedule entries that are {status.lower()}",
            f"{name} where status is {status}",
        ])
        return self.wrap(nl), (
            f"select * from bookings.timetable "
            f"where status = {sql_lit(status)} limit 50")

    # ════════════════════════════════════════════════════════════════════════
    #  MASTER
    # ════════════════════════════════════════════════════════════════════════

    def all_generators(self):
        return [
            # Generic (weighted: select_all and first_row appear more often)
            self.gen_select_all, self.gen_select_all,
            self.gen_select_all, self.gen_select_all,
            self.gen_first_row, self.gen_first_row,
            self.gen_first_row, self.gen_first_row,
            self.gen_select_limit, self.gen_select_limit,
            self.gen_select_limit,
            self.gen_count, self.gen_count,
            self.gen_preview, self.gen_preview,
            self.gen_list_tables, self.gen_list_tables,
            self.gen_describe_table, self.gen_describe_table,
            # Airports
            self.gen_airports_by_city, self.gen_airports_by_country,
            self.gen_airports_count, self.gen_airports_count_by_country,
            self.gen_airports_by_timezone, self.gen_airports_group_country,
            self.gen_airports_by_code, self.gen_airports_search_name,
            self.gen_airports_limit,
            # Airplanes
            self.gen_airplanes_all, self.gen_airplanes_by_range,
            self.gen_airplanes_by_speed, self.gen_airplanes_by_code,
            self.gen_airplanes_sort_range, self.gen_airplanes_count,
            self.gen_airplanes_avg_range, self.gen_airplanes_long_range,
            # Flights
            self.gen_flights_all, self.gen_flights_by_status,
            self.gen_flights_count_by_status, self.gen_flights_by_route,
            self.gen_flights_count, self.gen_flights_group_status,
            self.gen_flights_departed, self.gen_flights_delayed,
            self.gen_flights_cancelled,
            # Routes
            self.gen_routes_all, self.gen_routes_by_dep,
            self.gen_routes_by_arr, self.gen_routes_by_airplane,
            self.gen_routes_count, self.gen_routes_dep_arr,
            # Bookings
            self.gen_bookings_all, self.gen_bookings_by_ref,
            self.gen_bookings_amount_gt, self.gen_bookings_amount_lt,
            self.gen_bookings_amount_between, self.gen_bookings_count,
            self.gen_bookings_avg_amount, self.gen_bookings_max_amount,
            self.gen_bookings_min_amount, self.gen_bookings_sort_amount,
            # Tickets
            self.gen_tickets_all, self.gen_tickets_by_name,
            self.gen_tickets_by_name_like, self.gen_tickets_by_booking,
            self.gen_tickets_count, self.gen_tickets_count_by_booking,
            self.gen_tickets_outbound, self.gen_tickets_return,
            # Segments
            self.gen_segments_all, self.gen_segments_by_fare,
            self.gen_segments_price_gt, self.gen_segments_avg_price,
            self.gen_segments_avg_by_fare, self.gen_segments_count,
            self.gen_segments_count_by_fare, self.gen_segments_most_expensive,
            # Seats
            self.gen_seats_all, self.gen_seats_by_airplane,
            self.gen_seats_by_fare, self.gen_seats_count_by_airplane,
            self.gen_seats_group_fare, self.gen_seats_count,
            # Boarding passes
            self.gen_boarding_all, self.gen_boarding_by_flight,
            self.gen_boarding_by_seat, self.gen_boarding_count,
            # JOINs
            self.gen_join_tickets_segments, self.gen_join_tickets_bookings,
            self.gen_join_flights_routes, self.gen_join_boarding_tickets,
            self.gen_join_routes_airplanes, self.gen_join_seats_airplanes,
            # Aggregates
            self.gen_agg_total_revenue, self.gen_agg_avg_tickets_per_booking,
            self.gen_agg_top_routes, self.gen_agg_revenue_by_fare,
            self.gen_agg_flights_per_route, self.gen_agg_avg_price_by_fare,
            self.gen_agg_busiest_airports,
            # Subqueries
            self.gen_sub_expensive_bookings, self.gen_sub_passengers_multiple_tickets,
            # Timetable
            self.gen_timetable_all, self.gen_timetable_by_dep,
            self.gen_timetable_by_status,
        ]

    def generate(self, count):
        generators = self.all_generators()
        examples = []
        seen = set()

        print(f"Generating {count:,} examples using {len(generators)} generators "
              f"(English only, conversational style)...")

        report_interval = max(1, count // 20)

        while len(examples) < count:
            gen = self.rng.choice(generators)

            try:
                nl, sql = gen()
            except Exception:
                continue

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
    parser = argparse.ArgumentParser(description="Generate English NL→SQL training dataset")
    parser.add_argument("--out", default="training_data/dataset_200k.json",
                        help="Output file path")
    parser.add_argument("--count", type=int, default=200_000,
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
