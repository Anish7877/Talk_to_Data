-- Database Setup Script
-- Run this in PostgreSQL to set up the read-only role

-- Create the read-only role
DO $$
BEGIN
    CREATE ROLE ai_readonly LOGIN;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER ROLE ai_readonly WITH PASSWORD '1234';

-- Grant connect on database
GRANT CONNECT ON DATABASE postgres TO ai_readonly;

-- Grant select on all tables (run for each schema)
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ai_readonly;

-- Grant usage on sequences
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO ai_readonly;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO ai_readonly;

-- Revoke dangerous privileges (defense in depth)
REVOKE INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public FROM ai_readonly;
REVOKE TRUNCATE ON ALL TABLES IN SCHEMA public FROM ai_readonly;

-- Create sample tables for testing
CREATE TABLE IF NOT EXISTS customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    region VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    order_date DATE NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    price DECIMAL(10, 2) NOT NULL,
    inventory_count INTEGER DEFAULT 0
);

-- Grant access to new tables
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ai_readonly;

-- Insert sample data
INSERT INTO customers (name, email, region) VALUES
    ('John Doe', 'john@example.com', 'North America'),
    ('Jane Smith', 'jane@example.com', 'Europe'),
    ('Bob Wilson', 'bob@example.com', 'Asia Pacific')
ON CONFLICT (email) DO NOTHING;

INSERT INTO products (name, category, price, inventory_count) VALUES
    ('Widget A', 'Electronics', 29.99, 100),
    ('Widget B', 'Electronics', 49.99, 50),
    ('Gadget X', 'Tools', 99.99, 25)
ON CONFLICT DO NOTHING;

INSERT INTO orders (customer_id, order_date, total_amount, status)
SELECT
    c.customer_id,
    CURRENT_DATE - (random() * 30)::INTEGER,
    (random() * 500 + 50)::DECIMAL(10,2),
    CASE WHEN random() > 0.2 THEN 'completed' ELSE 'pending' END
FROM customers c
CROSS JOIN generate_series(1, 3);
