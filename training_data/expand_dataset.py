#!/usr/bin/env python3
"""
Скрипт для расширения датасета NL->SQL до 10000 примеров
"""
import json
import random
from typing import List, Dict

# Шаблоны для генерации разнообразных запросов
TEMPLATES = {
    "select_all": [
        ("Show all {table}", "SELECT * FROM {table}"),
        ("List all {table}", "SELECT * FROM {table}"),
        ("Get all {table}", "SELECT * FROM {table}"),
        ("Display all {table}", "SELECT * FROM {table}"),
        ("Fetch all {table}", "SELECT * FROM {table}"),
    ],
    "select_columns": [
        ("Show {cols} from {table}", "SELECT {cols} FROM {table}"),
        ("Get {cols} from {table}", "SELECT {cols} FROM {table}"),
        ("Display {cols} from {table}", "SELECT {cols} FROM {table}"),
        ("List {cols} from {table}", "SELECT {cols} FROM {table}"),
    ],
    "select_where": [
        ("{table} where {field} = '{value}'", "SELECT * FROM {table} WHERE {field} = '{value}'"),
        ("{table} from {city}", "SELECT * FROM {table} WHERE city = '{city}'"),
        ("{table} in category {cat}", "SELECT * FROM {table} WHERE category = '{cat}'"),
        ("{table} more expensive than {price}", "SELECT * FROM {table} WHERE price > {price}"),
        ("{table} cheaper than {price}", "SELECT * FROM {table} WHERE price < {price}"),
        ("{table} more than {price}", "SELECT * FROM {table} WHERE price > {price}"),
        ("{table} less than {price}", "SELECT * FROM {table} WHERE price < {price}"),
        ("{table} older than {age}", "SELECT * FROM {table} WHERE age > {age}"),
        ("{table} younger than {age}", "SELECT * FROM {table} WHERE age < {age}"),
        ("{table} named {name}", "SELECT * FROM {table} WHERE name = '{name}'"),
        ("{table} with email containing {domain}", "SELECT * FROM {table} WHERE email LIKE '%{domain}%'"),
        ("{table} with name containing {term}", "SELECT * FROM {table} WHERE name LIKE '%{term}%'"),
    ],
    "select_count": [
        ("Count all {table}", "SELECT COUNT(*) FROM {table}"),
        ("Total number of {table}", "SELECT COUNT(*) FROM {table}"),
        ("How many {table}", "SELECT COUNT(*) FROM {table}"),
        ("Number of {table}", "SELECT COUNT(*) FROM {table}"),
    ],
    "select_aggregate": [
        ("Max {field} from {table}", "SELECT MAX({field}) FROM {table}"),
        ("Min {field} from {table}", "SELECT MIN({field}) FROM {table}"),
        ("Avg {field} from {table}", "SELECT AVG({field}) FROM {table}"),
        ("Sum {field} from {table}", "SELECT SUM({field}) FROM {table}"),
        ("Max {field}", "SELECT MAX({field}) FROM {table}"),
        ("Min {field}", "SELECT MIN({field}) FROM {table}"),
        ("Avg {field}", "SELECT AVG({field}) FROM {table}"),
    ],
    "select_order_limit": [
        ("Show {n} most expensive {table}", "SELECT * FROM {table} ORDER BY price DESC LIMIT {n}"),
        ("Show {n} cheapest {table}", "SELECT * FROM {table} ORDER BY price ASC LIMIT {n}"),
        ("Get top {n} {table} by price DESC", "SELECT * FROM {table} ORDER BY price DESC LIMIT {n}"),
        ("Get top {n} {table} by price ASC", "SELECT * FROM {table} ORDER BY price ASC LIMIT {n}"),
        ("Show top {n} {table}", "SELECT * FROM {table} ORDER BY price DESC LIMIT {n}"),
        ("{table} sorted by {field} descending", "SELECT * FROM {table} ORDER BY {field} DESC"),
        ("{table} sorted by {field} ascending", "SELECT * FROM {table} ORDER BY {field} ASC"),
        ("{table} sorted by {field}", "SELECT * FROM {table} ORDER BY {field}"),
    ],
    "select_group_by": [
        ("{table} grouped by {field}", "SELECT {field}, COUNT(*) FROM {table} GROUP BY {field}"),
        ("{field} count from {table}", "SELECT {field}, COUNT(*) FROM {table} GROUP BY {field}"),
    ],
    "select_join": [
        ("Join {table1} and {table2}", "SELECT {table1}.name, {table2}.total FROM {table1} LEFT JOIN {table2} ON {table1}.id = {table2}.{table1}_id"),
        ("{table1} with {table2}", "SELECT {table1}.name, {table2}.total FROM {table1} LEFT JOIN {table2} ON {table1}.id = {table2}.{table1}_id"),
    ],
    "insert": [
        ("Insert new {table}", "INSERT INTO {table} (name, email) VALUES ('{name}', '{email}')"),
        ("Add new {table}", "INSERT INTO {table} (name, email) VALUES ('{name}', '{email}')"),
        ("Create new {table}", "INSERT INTO {table} (name, email) VALUES ('{name}', '{email}')"),
    ],
    "update": [
        ("Update {table} {field}", "UPDATE {table} SET {field} = {value} WHERE id = {id}"),
        ("Set {field} to {value} in {table}", "UPDATE {table} SET {field} = {value} WHERE id = {id}"),
    ],
    "delete": [
        ("Delete old {table}", "DELETE FROM {table} WHERE created_at < '{date}'"),
        ("Remove old {table}", "DELETE FROM {table} WHERE created_at < '{date}'"),
    ],
}

# Данные для подстановки
CITIES = ["Chicago", "New York", "Los Angeles", "Denver", "Miami", "San Francisco", 
          "Boston", "Seattle", "Portland", "Austin", "Dallas", "Houston", "Phoenix",
          "Atlanta", "Detroit", "Philadelphia", "Washington", "Nashville", "Orlando"]

CATEGORIES = ["Sports", "Toys", "Garden", "Grocery", "Automotive", "Electronics", 
              "Clothing", "Books", "Home", "Kitchen", "Health", "Beauty", "Tools"]

NAMES = ["Bob", "Alice", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", 
         "Ivan", "Judy", "Kevin", "Laura", "Mike", "Nancy", "Oscar", "Patricia",
         "Quinn", "Rachel", "Steve", "Tina", "Victor", "Wendy", "Xavier", "Yara", "Zoe"]

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "example.com", "mail.com"]

PRODUCT_TERMS = ["phone", "laptop", "tablet", "watch", "camera", "headphone", "speaker"]

def generate_examples(target_count: int = 10000) -> List[Dict[str, str]]:
    """Генерирует примеры до указанного количества"""
    examples = []
    
    # Загружаем существующий датасет
    try:
        with open("nl_to_sql_train.json", "r") as f:
            data = json.load(f)
            examples = data.get("examples", [])
            print(f"Loaded {len(examples)} existing examples")
    except FileNotFoundError:
        print("No existing dataset found, starting from scratch")
    
    # Генерируем новые примеры
    while len(examples) < target_count:
        template_type = random.choice(list(TEMPLATES.keys()))
        template_nl, template_sql = random.choice(TEMPLATES[template_type])
        
        try:
            if template_type == "select_all":
                table = random.choice(["users", "products", "orders", "categories"])
                nl = template_nl.format(table=table)
                sql = template_sql.format(table=table)
                
            elif template_type == "select_columns":
                table = random.choice(["users", "products", "orders"])
                if table == "users":
                    cols = random.choice(["name, email", "name", "email", "name, age", "id, name"])
                elif table == "products":
                    cols = random.choice(["name, price", "name", "price", "name, category", "id, name"])
                else:
                    cols = random.choice(["id, total", "id", "total", "id, created_at"])
                nl = template_nl.format(cols=cols, table=table)
                sql = template_sql.format(cols=cols, table=table)
                
            elif template_type == "select_where":
                table = random.choice(["users", "products"])
                if "city" in template_nl:
                    city = random.choice(CITIES)
                    nl = template_nl.format(table=table, city=city)
                    sql = template_sql.format(table=table, city=city)
                elif "category" in template_nl:
                    cat = random.choice(CATEGORIES)
                    nl = template_nl.format(table=table, cat=cat)
                    sql = template_sql.format(table=table, cat=cat)
                elif "price" in template_nl:
                    price = random.randint(50, 1000)
                    nl = template_nl.format(table=table, price=price)
                    sql = template_sql.format(table=table, price=price)
                elif "age" in template_nl:
                    age = random.randint(18, 80)
                    nl = template_nl.format(table=table, age=age)
                    sql = template_sql.format(table=table, age=age)
                elif "name" in template_nl and "named" in template_nl:
                    name = random.choice(NAMES)
                    nl = template_nl.format(table=table, name=name)
                    sql = template_sql.format(table=table, name=name)
                elif "email" in template_nl:
                    domain = random.choice(EMAIL_DOMAINS)
                    nl = template_nl.format(table=table, domain=domain)
                    sql = template_sql.format(table=table, domain=domain)
                elif "name containing" in template_nl:
                    term = random.choice(PRODUCT_TERMS)
                    nl = template_nl.format(table=table, term=term)
                    sql = template_sql.format(table=table, term=term)
                else:
                    field = random.choice(["id", "name", "status"])
                    value = random.choice(["active", "pending", "completed"])
                    nl = template_nl.format(table=table, field=field, value=value)
                    sql = template_sql.format(table=table, field=field, value=value)
                    
            elif template_type == "select_count":
                table = random.choice(["users", "products", "orders", "categories"])
                nl = template_nl.format(table=table)
                sql = template_sql.format(table=table)
                
            elif template_type == "select_aggregate":
                table = random.choice(["users", "products", "orders"])
                if table == "products":
                    field = "price"
                elif table == "orders":
                    field = "total"
                else:
                    field = "age"
                nl = template_nl.format(field=field, table=table)
                sql = template_sql.format(field=field, table=table)
                
            elif template_type == "select_order_limit":
                table = random.choice(["users", "products", "orders"])
                n = random.randint(1, 50)
                if "field" in template_nl:
                    field = random.choice(["price", "name", "created_at", "age"])
                    nl = template_nl.format(table=table, field=field)
                    sql = template_sql.format(table=table, field=field)
                else:
                    nl = template_nl.format(table=table, n=n)
                    sql = template_sql.format(table=table, n=n)
                    
            elif template_type == "select_group_by":
                table = random.choice(["users", "products"])
                field = random.choice(["city", "category"]) if table == "users" else "category"
                nl = template_nl.format(table=table, field=field)
                sql = template_sql.format(table=table, field=field)
                
            elif template_type == "select_join":
                table1 = "users"
                table2 = "orders"
                nl = template_nl.format(table1=table1, table2=table2)
                sql = template_sql.format(table1=table1, table2=table2)
                
            elif template_type == "insert":
                table = "users"
                name = random.choice(NAMES)
                email = f"{name.lower()}@{random.choice(EMAIL_DOMAINS)}"
                nl = template_nl.format(table=table)
                sql = template_sql.format(table=table, name=name, email=email)
                
            elif template_type == "update":
                table = "products"
                field = "price"
                value = random.randint(50, 500)
                id_val = random.randint(1, 200)
                nl = template_nl.format(table=table, field=field)
                sql = template_sql.format(table=table, field=field, value=value, id=id_val)
                
            elif template_type == "delete":
                table = "orders"
                year = random.randint(2018, 2023)
                date = f"{year}-01-01"
                nl = template_nl.format(table=table)
                sql = template_sql.format(table=table, date=date)
            
            # Проверяем уникальность
            if {"nl": nl, "sql": sql} not in examples:
                examples.append({"nl": nl, "sql": sql})
                
        except Exception as e:
            print(f"Error generating example: {e}")
            continue
        
        if len(examples) % 1000 == 0:
            print(f"Generated {len(examples)} examples...")
    
    return examples[:target_count]

def main():
    print("=== Расширение датасета NL->SQL до 10000 примеров ===")
    
    examples = generate_examples(10000)
    
    output = {"examples": examples}
    
    with open("nl_to_sql_train.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Датасет расширен до {len(examples)} примеров")
    print(f"✓ Сохранено в nl_to_sql_train.json")

if __name__ == "__main__":
    main()

