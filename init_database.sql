-- ============================================
-- Скрипт инициализации БД для AI SQL Agent
-- ============================================

-- Удаление таблиц если существуют (для чистой установки)
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS categories CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- Таблица пользователей
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    phone VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Таблица категорий товаров
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    parent_id INTEGER REFERENCES categories(id)
);

-- Таблица товаров
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    category_id INTEGER REFERENCES categories(id),
    stock_quantity INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_available BOOLEAN DEFAULT true
);

-- Таблица клиентов
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    phone VARCHAR(20),
    address TEXT,
    city VARCHAR(100),
    country VARCHAR(100),
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица заказов
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id) ON DELETE CASCADE,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10, 2) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    shipping_address TEXT,
    payment_method VARCHAR(50)
);

-- Таблица позиций заказа
CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    subtotal DECIMAL(10, 2) NOT NULL
);

-- Индексы для оптимизации запросов
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_price ON products(price);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);

-- ============================================
-- Заполнение тестовыми данными
-- ============================================

-- Пользователи
INSERT INTO users (name, email, phone, is_active) VALUES
    ('John Doe', 'john.doe@example.com', '+1-555-0101', true),
    ('Jane Smith', 'jane.smith@example.com', '+1-555-0102', true),
    ('Bob Johnson', 'bob.johnson@example.com', '+1-555-0103', true),
    ('Alice Williams', 'alice.williams@example.com', '+1-555-0104', true),
    ('Charlie Brown', 'charlie.brown@example.com', '+1-555-0105', false),
    ('Diana Prince', 'diana.prince@example.com', '+1-555-0106', true),
    ('Edward Norton', 'edward.norton@example.com', '+1-555-0107', true),
    ('Fiona Green', 'fiona.green@example.com', '+1-555-0108', true);

-- Категории
INSERT INTO categories (name, description, parent_id) VALUES
    ('Electronics', 'Electronic devices and accessories', NULL),
    ('Clothing', 'Apparel and fashion items', NULL),
    ('Books', 'Physical and digital books', NULL),
    ('Home & Garden', 'Home improvement and garden supplies', NULL),
    ('Smartphones', 'Mobile phones and accessories', 1),
    ('Laptops', 'Portable computers', 1),
    ('Men''s Clothing', 'Clothing for men', 2),
    ('Women''s Clothing', 'Clothing for women', 2);

-- Товары
INSERT INTO products (name, description, price, category_id, stock_quantity, is_available) VALUES
    ('iPhone 15 Pro', 'Latest Apple smartphone with A17 chip', 999.99, 5, 50, true),
    ('Samsung Galaxy S24', 'Flagship Android smartphone', 899.99, 5, 45, true),
    ('MacBook Pro 16"', 'Professional laptop with M3 chip', 2499.99, 6, 20, true),
    ('Dell XPS 15', 'High-performance Windows laptop', 1799.99, 6, 30, true),
    ('Sony WH-1000XM5', 'Noise-cancelling headphones', 399.99, 1, 100, true),
    ('Men''s Leather Jacket', 'Genuine leather jacket', 199.99, 7, 25, true),
    ('Women''s Summer Dress', 'Floral print summer dress', 79.99, 8, 60, true),
    ('The Great Gatsby', 'Classic American novel', 14.99, 3, 200, true),
    ('Garden Tool Set', 'Complete 10-piece garden tool set', 89.99, 4, 40, true),
    ('Smart Watch', 'Fitness tracking smartwatch', 249.99, 5, 75, true),
    ('Wireless Mouse', 'Ergonomic wireless mouse', 29.99, 1, 150, true),
    ('USB-C Cable', 'Fast charging USB-C cable', 12.99, 1, 500, true),
    ('Men''s Jeans', 'Classic blue denim jeans', 59.99, 7, 80, true),
    ('Cookbook', 'Italian cuisine recipes', 24.99, 3, 120, true),
    ('LED Desk Lamp', 'Adjustable LED desk lamp', 45.99, 4, 65, true);

-- Клиенты
INSERT INTO customers (first_name, last_name, email, phone, address, city, country) VALUES
    ('Michael', 'Scott', 'michael.scott@example.com', '+1-555-1001', '123 Main St', 'New York', 'USA'),
    ('Pam', 'Beesly', 'pam.beesly@example.com', '+1-555-1002', '456 Oak Ave', 'Los Angeles', 'USA'),
    ('Jim', 'Halpert', 'jim.halpert@example.com', '+1-555-1003', '789 Pine Rd', 'Chicago', 'USA'),
    ('Dwight', 'Schrute', 'dwight.schrute@example.com', '+1-555-1004', '321 Elm St', 'Houston', 'USA'),
    ('Angela', 'Martin', 'angela.martin@example.com', '+1-555-1005', '654 Maple Dr', 'Phoenix', 'USA'),
    ('Kevin', 'Malone', 'kevin.malone@example.com', '+1-555-1006', '987 Cedar Ln', 'Philadelphia', 'USA'),
    ('Oscar', 'Martinez', 'oscar.martinez@example.com', '+1-555-1007', '147 Birch Ct', 'San Antonio', 'USA'),
    ('Ryan', 'Howard', 'ryan.howard@example.com', '+1-555-1008', '258 Spruce Way', 'San Diego', 'USA');

-- Заказы
INSERT INTO orders (customer_id, order_date, total_amount, status, shipping_address, payment_method) VALUES
    (1, '2024-01-15 10:30:00', 1049.98, 'delivered', '123 Main St, New York, USA', 'credit_card'),
    (2, '2024-01-16 14:20:00', 899.99, 'shipped', '456 Oak Ave, Los Angeles, USA', 'paypal'),
    (3, '2024-01-17 09:15:00', 2499.99, 'processing', '789 Pine Rd, Chicago, USA', 'credit_card'),
    (1, '2024-01-18 16:45:00', 429.98, 'delivered', '123 Main St, New York, USA', 'debit_card'),
    (4, '2024-01-19 11:30:00', 139.98, 'pending', '321 Elm St, Houston, USA', 'credit_card'),
    (5, '2024-01-20 13:00:00', 79.99, 'shipped', '654 Maple Dr, Phoenix, USA', 'paypal'),
    (6, '2024-01-21 15:30:00', 89.99, 'delivered', '987 Cedar Ln, Philadelphia, USA', 'credit_card'),
    (7, '2024-01-22 10:00:00', 1829.98, 'processing', '147 Birch Ct, San Antonio, USA', 'credit_card'),
    (8, '2024-01-23 12:30:00', 324.97, 'pending', '258 Spruce Way, San Diego, USA', 'debit_card'),
    (2, '2024-01-24 14:00:00', 59.99, 'shipped', '456 Oak Ave, Los Angeles, USA', 'paypal');

-- Позиции заказов
INSERT INTO order_items (order_id, product_id, quantity, unit_price, subtotal) VALUES
    -- Заказ 1
    (1, 1, 1, 999.99, 999.99),
    (1, 11, 1, 29.99, 29.99),
    (1, 12, 1, 12.99, 12.99),
    -- Заказ 2
    (2, 2, 1, 899.99, 899.99),
    -- Заказ 3
    (3, 3, 1, 2499.99, 2499.99),
    -- Заказ 4
    (4, 5, 1, 399.99, 399.99),
    (4, 11, 1, 29.99, 29.99),
    -- Заказ 5
    (5, 7, 1, 79.99, 79.99),
    (5, 14, 2, 24.99, 49.99),
    (5, 12, 1, 12.99, 12.99),
    -- Заказ 6
    (6, 7, 1, 79.99, 79.99),
    -- Заказ 7
    (7, 9, 1, 89.99, 89.99),
    -- Заказ 8
    (8, 4, 1, 1799.99, 1799.99),
    (8, 11, 1, 29.99, 29.99),
    -- Заказ 9
    (9, 10, 1, 249.99, 249.99),
    (9, 15, 1, 45.99, 45.99),
    (9, 11, 1, 29.99, 29.99),
    -- Заказ 10
    (10, 13, 1, 59.99, 59.99);

-- Создание представлений для удобства
CREATE VIEW order_summary AS
SELECT 
    o.id as order_id,
    c.first_name || ' ' || c.last_name as customer_name,
    o.order_date,
    o.total_amount,
    o.status,
    COUNT(oi.id) as items_count
FROM orders o
JOIN customers c ON o.customer_id = c.id
LEFT JOIN order_items oi ON o.id = oi.order_id
GROUP BY o.id, c.first_name, c.last_name, o.order_date, o.total_amount, o.status;

CREATE VIEW product_inventory AS
SELECT 
    p.id,
    p.name,
    p.price,
    p.stock_quantity,
    c.name as category,
    p.is_available
FROM products p
LEFT JOIN categories c ON p.category_id = c.id;

-- Вывод статистики
SELECT 'Database initialized successfully!' as status;
SELECT 'Total users: ' || COUNT(*) as info FROM users;
SELECT 'Total products: ' || COUNT(*) as info FROM products;
SELECT 'Total customers: ' || COUNT(*) as info FROM customers;
SELECT 'Total orders: ' || COUNT(*) as info FROM orders;
