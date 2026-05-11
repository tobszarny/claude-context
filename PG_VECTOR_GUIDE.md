# PGVector Setup Guide

This guide covers using **PostgreSQL + pgvector** as the vector database backend for Claude Context, as an alternative to Zilliz Cloud or Local Milvus.

## Prerequisites

- **PostgreSQL 12+** with the [pgvector](https://github.com/pgvector/pgvector) extension installed
- An **embedding provider** API key (OpenAI, VoyageAI, Gemini, or Ollama)
- **Node.js >= 20.0.0**

## Step 1: Set Up PostgreSQL with pgvector

### Install pgvector

Follow the [pgvector installation guide](https://github.com/pgvector/pgvector#installation) for your platform. On most systems:

**macOS (Homebrew):**
```bash
brew install pgvector
```

**Docker:**
```bash
docker run -d \
  --name pgvector \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=claude_context \
  -p 5432:5432 \
  pgvector/pgvector:pg17
```

**Ubuntu/Debian:**
```bash
sudo apt install postgresql-17-pgvector
```

### Create a Database

```bash
psql -U postgres -c "CREATE DATABASE claude_context;"
```

The pgvector extension is created automatically by Claude Context on first use.

## Step 2: Configure Your MCP Client

Set two environment variables to switch from Milvus to PGVector:

| Variable | Value |
|----------|-------|
| `VECTOR_DB_PROVIDER` | `pgvector` |
| `PG_CONNECTION_STRING` | `postgresql://user:password@host:5432/dbname` |

### Claude Code

**From npm (published):**
```bash
claude mcp add claude-context \
  -e OPENAI_API_KEY=sk-your-openai-api-key \
  -e VECTOR_DB_PROVIDER=pgvector \
  -e PG_CONNECTION_STRING=postgresql://postgres:yourpassword@localhost:5432/claude_context \
  -- npx @zilliz/claude-context-mcp@latest
```

**From local build:**
```bash
claude mcp add claude-context \
  -e OPENAI_API_KEY=sk-your-openai-api-key \
  -e VECTOR_DB_PROVIDER=pgvector \
  -e PG_CONNECTION_STRING=postgresql://postgres:yourpassword@localhost:5432/claude_context \
  -- node /path/to/claude-context/packages/mcp/dist/index.js
```

**From local source (auto-rebuild on changes):**
```bash
claude mcp add claude-context \
  -e OPENAI_API_KEY=sk-your-openai-api-key \
  -e VECTOR_DB_PROVIDER=pgvector \
  -e PG_CONNECTION_STRING=postgresql://postgres:yourpassword@localhost:5432/claude_context \
  -- npx tsx /path/to/claude-context/packages/mcp/src/index.ts
```

### Cursor / Windsurf / VS Code

**From npm (published):**
```json
{
  "mcpServers": {
    "claude-context": {
      "command": "npx",
      "args": ["-y", "@zilliz/claude-context-mcp@latest"],
      "env": {
        "OPENAI_API_KEY": "sk-your-openai-api-key",
        "VECTOR_DB_PROVIDER": "pgvector",
        "PG_CONNECTION_STRING": "postgresql://postgres:yourpassword@localhost:5432/claude_context"
      }
    }
  }
}
```

**From local build:**
```json
{
  "mcpServers": {
    "claude-context": {
      "command": "node",
      "args": ["/path/to/claude-context/packages/mcp/dist/index.js"],
      "env": {
        "OPENAI_API_KEY": "sk-your-openai-api-key",
        "VECTOR_DB_PROVIDER": "pgvector",
        "PG_CONNECTION_STRING": "postgresql://postgres:yourpassword@localhost:5432/claude_context"
      }
    }
  }
}
```

### Global Configuration (~/.context/.env)

```bash
mkdir -p ~/.context
cat > ~/.context/.env << 'EOF'
EMBEDDING_PROVIDER=OpenAI
OPENAI_API_KEY=sk-your-openai-api-key
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_DB_PROVIDER=pgvector
PG_CONNECTION_STRING=postgresql://postgres:yourpassword@localhost:5432/claude_context
EOF
```

### Other MCP Clients

The same environment variables work with any MCP client. Just set `VECTOR_DB_PROVIDER=pgvector` and `PG_CONNECTION_STRING` alongside your embedding provider configuration.

## Step 3: Use Claude Context

1. **Open your AI coding assistant** in your project directory
2. **Index your codebase:**
   ```
   Index this codebase
   ```
3. **Search your code:**
   ```
   Find functions that handle user authentication
   ```

## Connection String Format

The `PG_CONNECTION_STRING` follows the standard PostgreSQL URI format:

```
postgresql://[user]:[password]@[host]:[port]/[dbname]
```

Examples:

| Setup | Connection String |
|-------|-------------------|
| Local PostgreSQL | `postgresql://postgres:password@localhost:5432/claude_context` |
| Docker (default) | `postgresql://postgres:password@localhost:5432/claude_context` |
| Remote server | `postgresql://user:pass@db.example.com:5432/claude_context` |
| With SSL | `postgresql://user:pass@host:5432/db?sslmode=require` |

## How Data Is Stored

Claude Context creates one PostgreSQL table per indexed codebase. Tables are named using the same convention as Milvus collections: `code_chunks_<override>_<pathHash>` or `hybrid_code_chunks_<override>_<pathHash>`.

Each table contains:

| Column | Type | Description |
|--------|------|-------------|
| `id` | VARCHAR(512) | Unique chunk ID (primary key) |
| `vector` | VECTOR(dim) | Dense embedding vector |
| `content` | TEXT | Source code chunk text |
| `relativePath` | VARCHAR(1024) | File path relative to codebase root |
| `startLine` | BIGINT | Start line number |
| `endLine` | BIGINT | End line number |
| `fileExtension` | VARCHAR(32) | File extension (e.g., `.ts`) |
| `metadata` | JSONB | Additional metadata |
| `content_tsvector` | TSVECTOR | Auto-generated full-text search vector |

## Hybrid Search

PGVector supports hybrid search (dense vector + full-text) using PostgreSQL's built-in `tsvector` full-text search combined with `pgvector` cosine similarity. Results are fused using Reciprocal Rank Fusion (RRF), matching the behavior of Milvus hybrid search.

Hybrid search is enabled by default (`HYBRID_MODE=true`). No additional configuration is needed.

## Limitations

- **MCP server only** — the VS Code and Chrome extensions continue using Milvus REST API
- **No migration tool** — to move data between Milvus and PGVector, re-index your codebase
- **IVFFlat index** — uses approximate nearest neighbor search. For datasets under 100K vectors, consider switching to exact search by recreating the index
