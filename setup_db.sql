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
