-- CCEA Cloud Database Initialization Script
--
-- This script initializes the PostgreSQL database for CCEA Cloud Control Plane.
-- It creates the necessary schemas, extensions, and initial configuration.
--
-- Design Doc Reference: Phase 6 - Cloud Control Plane data model
-- Phase 9: Enterprise on-prem deployment

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- Create schemas for logical separation
CREATE SCHEMA IF NOT EXISTS ccea_core;
CREATE SCHEMA IF NOT EXISTS ccea_telemetry;
CREATE SCHEMA IF NOT EXISTS ccea_audit;

-- Set default schema
SET search_path TO ccea_core, public;

-- Create application role (for RLS)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'ccea_app') THEN
        CREATE ROLE ccea_app NOLOGIN;
    END IF;
END
$$;

-- Grant usage on schemas
GRANT USAGE ON SCHEMA ccea_core TO ccea_app;
GRANT USAGE ON SCHEMA ccea_telemetry TO ccea_app;
GRANT USAGE ON SCHEMA ccea_audit TO ccea_app;

-- Function to set current workspace for RLS
CREATE OR REPLACE FUNCTION set_current_workspace(workspace_uuid UUID)
RETURNS VOID AS $$
BEGIN
    PERFORM set_config('app.current_workspace_id', workspace_uuid::text, false);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get current workspace
CREATE OR REPLACE FUNCTION get_current_workspace()
RETURNS UUID AS $$
BEGIN
    RETURN current_setting('app.current_workspace_id', true)::UUID;
EXCEPTION
    WHEN OTHERS THEN
        RETURN NULL;
END;
$$ LANGUAGE plpgsql STABLE;

-- Audit trigger function
CREATE OR REPLACE FUNCTION ccea_audit.log_change()
RETURNS TRIGGER AS $$
DECLARE
    audit_row ccea_audit.change_log;
    old_data JSONB;
    new_data JSONB;
BEGIN
    audit_row.table_name = TG_TABLE_NAME::TEXT;
    audit_row.action = TG_OP;
    audit_row.changed_at = CURRENT_TIMESTAMP;
    audit_row.changed_by = current_setting('app.current_user_id', true);
    audit_row.workspace_id = get_current_workspace();

    IF TG_OP = 'DELETE' THEN
        audit_row.row_id = OLD.id;
        old_data = to_jsonb(OLD);
        -- Redact sensitive fields
        old_data = old_data - 'password_hash' - 'mfa_secret' - 'token_hash';
        audit_row.old_data = old_data;
    ELSIF TG_OP = 'UPDATE' THEN
        audit_row.row_id = NEW.id;
        old_data = to_jsonb(OLD);
        new_data = to_jsonb(NEW);
        -- Redact sensitive fields
        old_data = old_data - 'password_hash' - 'mfa_secret' - 'token_hash';
        new_data = new_data - 'password_hash' - 'mfa_secret' - 'token_hash';
        audit_row.old_data = old_data;
        audit_row.new_data = new_data;
    ELSIF TG_OP = 'INSERT' THEN
        audit_row.row_id = NEW.id;
        new_data = to_jsonb(NEW);
        -- Redact sensitive fields
        new_data = new_data - 'password_hash' - 'mfa_secret' - 'token_hash';
        audit_row.new_data = new_data;
    END IF;

    INSERT INTO ccea_audit.change_log VALUES (audit_row.*);

    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create audit log table
CREATE TABLE IF NOT EXISTS ccea_audit.change_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name TEXT NOT NULL,
    row_id UUID,
    action TEXT NOT NULL,
    old_data JSONB,
    new_data JSONB,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    changed_by TEXT,
    workspace_id UUID
);

CREATE INDEX IF NOT EXISTS idx_change_log_table ON ccea_audit.change_log(table_name);
CREATE INDEX IF NOT EXISTS idx_change_log_changed_at ON ccea_audit.change_log(changed_at);
CREATE INDEX IF NOT EXISTS idx_change_log_workspace ON ccea_audit.change_log(workspace_id);

-- Grant permissions on audit schema
GRANT SELECT, INSERT ON ccea_audit.change_log TO ccea_app;

-- Partitioned telemetry table for high-volume events
CREATE TABLE IF NOT EXISTS ccea_telemetry.events (
    id UUID NOT NULL DEFAULT uuid_generate_v4(),
    workspace_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    run_id UUID,
    event_type TEXT NOT NULL,
    event_timestamp TIMESTAMPTZ NOT NULL,
    telemetry_level TEXT NOT NULL DEFAULT 'aggregated',
    payload JSONB NOT NULL,
    redaction_applied BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, event_timestamp)
) PARTITION BY RANGE (event_timestamp);

-- Create monthly partitions for telemetry (12 months ahead)
DO $$
DECLARE
    start_date DATE := DATE_TRUNC('month', CURRENT_DATE);
    end_date DATE;
    partition_name TEXT;
BEGIN
    FOR i IN 0..12 LOOP
        end_date := start_date + INTERVAL '1 month';
        partition_name := 'events_' || TO_CHAR(start_date, 'YYYY_MM');

        IF NOT EXISTS (
            SELECT FROM pg_tables
            WHERE schemaname = 'ccea_telemetry'
            AND tablename = partition_name
        ) THEN
            EXECUTE format(
                'CREATE TABLE ccea_telemetry.%I PARTITION OF ccea_telemetry.events
                FOR VALUES FROM (%L) TO (%L)',
                partition_name,
                start_date,
                end_date
            );
        END IF;

        start_date := end_date;
    END LOOP;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_telemetry_agent_time
    ON ccea_telemetry.events(agent_id, event_timestamp);
CREATE INDEX IF NOT EXISTS idx_telemetry_workspace_time
    ON ccea_telemetry.events(workspace_id, event_timestamp);
CREATE INDEX IF NOT EXISTS idx_telemetry_type
    ON ccea_telemetry.events(event_type);

-- Grant permissions on telemetry schema
GRANT SELECT, INSERT ON ccea_telemetry.events TO ccea_app;

-- Create default retention policy
INSERT INTO ccea_core.data_retention_policies (workspace_id, data_type, retention_days, auto_purge_enabled)
SELECT
    '00000000-0000-0000-0000-000000000000'::UUID,
    data_type,
    retention_days,
    TRUE
FROM (VALUES
    ('telemetry_aggregated', 90),
    ('telemetry_detailed', 30),
    ('telemetry_raw', 7),
    ('audit_logs', 365),
    ('commands', 90),
    ('approvals', 365),
    ('alerts', 180)
) AS defaults(data_type, retention_days)
ON CONFLICT DO NOTHING;

-- Vacuum settings for telemetry tables
ALTER TABLE ccea_telemetry.events SET (
    autovacuum_vacuum_scale_factor = 0.0,
    autovacuum_vacuum_threshold = 5000,
    autovacuum_analyze_scale_factor = 0.0,
    autovacuum_analyze_threshold = 5000
);

-- Output initialization status
DO $$
BEGIN
    RAISE NOTICE 'CCEA Database initialized successfully';
    RAISE NOTICE 'Extensions: uuid-ossp, pgcrypto, btree_gist';
    RAISE NOTICE 'Schemas: ccea_core, ccea_telemetry, ccea_audit';
    RAISE NOTICE 'Telemetry partitions created for next 12 months';
END;
$$;
