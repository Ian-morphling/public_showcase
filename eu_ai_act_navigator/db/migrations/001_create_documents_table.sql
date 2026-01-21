-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table for EU AI Act Navigator
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Legal structure metadata
    section_type TEXT NOT NULL,       -- Article | Recital | Annex
    section_title TEXT NOT NULL,      -- e.g. "Article 6"
    content TEXT NOT NULL,            -- Chunked legal text
    url TEXT NOT NULL,                -- Source URL

    -- Vector embedding (MiniLM = 384 dimensions)
    embedding VECTOR(384) NOT NULL,

    -- Operational metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vector similarity index (HNSW)
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
ON documents
USING hnsw (embedding vector_cosine_ops);

-- Optional filter index for agentic routing
CREATE INDEX IF NOT EXISTS documents_section_type_idx
ON documents (section_type);

-- Optional lexical search index (debug / fallback / hybrid)
CREATE INDEX IF NOT EXISTS documents_content_gin_idx
ON documents
USING GIN (to_tsvector('english', content));

-- RPC: semantic similarity search
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding VECTOR(384),
  match_count INT DEFAULT 5
)
RETURNS TABLE (
  id UUID,
  section_type TEXT,
  section_title TEXT,
  url TEXT,
  content TEXT,
  similarity FLOAT
)
LANGUAGE sql
STABLE
AS $$
  SELECT
    d.id,
    d.section_type,
    d.section_title,
    d.url,
    d.content,
    1 - (d.embedding <=> query_embedding) AS similarity
  FROM documents d
  WHERE d.embedding IS NOT NULL
  ORDER BY d.embedding <=> query_embedding
  LIMIT match_count;
$$;
