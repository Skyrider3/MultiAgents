-- Initialize MultiAgents Database Schema

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS agents;
CREATE SCHEMA IF NOT EXISTS knowledge;
CREATE SCHEMA IF NOT EXISTS insights;

-- Create papers table
CREATE TABLE IF NOT EXISTS knowledge.papers (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    arxiv_id VARCHAR(50) UNIQUE,
    title TEXT NOT NULL,
    abstract TEXT,
    authors JSONB,
    categories TEXT[],
    pdf_url TEXT,
    published_date TIMESTAMP,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create agent_sessions table
CREATE TABLE IF NOT EXISTS agents.sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_name VARCHAR(255),
    agent_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    metadata JSONB
);

-- Create agent_interactions table
CREATE TABLE IF NOT EXISTS agents.interactions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES agents.sessions(id),
    from_agent VARCHAR(50),
    to_agent VARCHAR(50),
    message_type VARCHAR(50),
    content TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create insights table
CREATE TABLE IF NOT EXISTS insights.discoveries (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES agents.sessions(id),
    discovery_type VARCHAR(100),
    title TEXT NOT NULL,
    description TEXT,
    confidence_score DECIMAL(3,2),
    supporting_papers UUID[],
    reasoning_trace JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_papers_arxiv_id ON knowledge.papers(arxiv_id);
CREATE INDEX idx_papers_published ON knowledge.papers(published_date);
CREATE INDEX idx_sessions_status ON agents.sessions(status);
CREATE INDEX idx_interactions_session ON agents.interactions(session_id);
CREATE INDEX idx_discoveries_confidence ON insights.discoveries(confidence_score);

-- Create GIN indexes for JSONB columns
CREATE INDEX idx_papers_metadata ON knowledge.papers USING GIN (metadata);
CREATE INDEX idx_papers_authors ON knowledge.papers USING GIN (authors);

-- Add update trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_papers_updated_at BEFORE UPDATE
    ON knowledge.papers FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();