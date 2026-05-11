# PGVector Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PostgreSQL + pgvector as a selectable vector database backend for the MCP server.

**Architecture:** New `PgvectorDatabase` class implementing the existing `VectorDatabase` interface. Factory switch in MCP server selects backend via `VECTOR_DB_PROVIDER` env var. Hybrid search uses PostgreSQL FTS (tsvector) + dense cosine with RRF.

**Tech Stack:** TypeScript, `pg` npm package, `pgvector` PostgreSQL extension.

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `packages/core/src/vectordb/pgvector-vectordb.ts` | `PgvectorDatabase` class implementing `VectorDatabase` |
| Create | `packages/core/src/vectordb/pgvector-vectordb.test.ts` | Unit tests for `PgvectorDatabase` |
| Modify | `packages/core/src/vectordb/index.ts` | Export `PgvectorDatabase` and `PgvectorConfig` |
| Modify | `packages/core/package.json` | Add `pg` and `@types/pg` dependencies |
| Modify | `packages/mcp/src/config.ts` | Add `vectorDbProvider` and `pgConnectionString` to config |
| Modify | `packages/mcp/src/index.ts` | Factory switch between Milvus and PGVector |
| Modify | `docs/getting-started/prerequisites.md` | Add PGVector option |

---

### Task 1: Add dependencies

**Files:**
- Modify: `packages/core/package.json`

- [ ] **Step 1: Add pg and @types/pg to package.json**

In `packages/core/package.json`, add to `dependencies`:
```json
"pg": "^8.13.0",
```
Add to `devDependencies`:
```json
"@types/pg": "^8.11.0",
```

- [ ] **Step 2: Install dependencies**

Run: `cd /Users/tomasz/WebstormProjects/claude-context && pnpm install`
Expected: Dependencies installed successfully.

- [ ] **Step 3: Commit**

```bash
git add packages/core/package.json pnpm-lock.yaml
git commit -m "chore: add pg and @types/pg dependencies for pgvector support"
```

---

### Task 2: Create PgvectorDatabase class

**Files:**
- Create: `packages/core/src/vectordb/pgvector-vectordb.ts`

- [ ] **Step 1: Create the PgvectorDatabase implementation**

Create `packages/core/src/vectordb/pgvector-vectordb.ts` with the following content:

```typescript
import { Pool, PoolConfig } from 'pg';
import {
    VectorDocument,
    SearchOptions,
    VectorSearchResult,
    VectorDatabase,
    HybridSearchRequest,
    HybridSearchOptions,
    HybridSearchResult,
} from './types';

export interface PgvectorConfig {
    connectionString: string;
    poolConfig?: Partial<PoolConfig>;
}

export class PgvectorDatabase implements VectorDatabase {
    private pool: Pool;
    private initialized = false;

    constructor(config: PgvectorConfig) {
        this.pool = new Pool({
            connectionString: config.connectionString,
            ...config.poolConfig,
        });
        console.log('[PgvectorDB] 🔌 Connecting to PostgreSQL');
    }

    private async ensurePgvector(): Promise<void> {
        if (this.initialized) return;
        await this.pool.query('CREATE EXTENSION IF NOT EXISTS vector');
        this.initialized = true;
    }

    private static tableName(collectionName: string): string {
        return `"${collectionName}"`;
    }

    async createCollection(collectionName: string, dimension: number, description?: string): Promise<void> {
        await this.ensurePgvector();

        const tbl = PgvectorDatabase.tableName(collectionName);
        const commentSql = description
            ? `COMMENT ON TABLE ${tbl} IS '${description.replace(/'/g, "''")}';`
            : '';

        await this.pool.query(`
            CREATE TABLE IF NOT EXISTS ${tbl} (
                id VARCHAR(512) PRIMARY KEY,
                vector VECTOR(${dimension}),
                content TEXT,
                "relativePath" VARCHAR(1024),
                "startLine" BIGINT,
                "endLine" BIGINT,
                "fileExtension" VARCHAR(32),
                metadata JSONB,
                content_tsvector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', COALESCE(content, ''))) STORED
            );
            CREATE INDEX IF NOT EXISTS "idx_${collectionName}_vector" ON ${tbl} USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);
            CREATE INDEX IF NOT EXISTS "idx_${collectionName}_tsvector" ON ${tbl} USING GIN (content_tsvector);
            ${commentSql}
        `);

        console.log(`[PgvectorDB] ✅ Collection '${collectionName}' created (dim=${dimension})`);
    }

    async createHybridCollection(collectionName: string, dimension: number, description?: string): Promise<void> {
        await this.createCollection(collectionName, dimension, description);
    }

    async dropCollection(collectionName: string): Promise<void> {
        const tbl = PgvectorDatabase.tableName(collectionName);
        await this.pool.query(`DROP TABLE IF EXISTS ${tbl}`);
        console.log(`[PgvectorDB] 🗑️ Collection '${collectionName}' dropped`);
    }

    async hasCollection(collectionName: string): Promise<boolean> {
        const result = await this.pool.query(
            `SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1)`,
            [collectionName]
        );
        return result.rows[0].exists;
    }

    async listCollections(): Promise<string[]> {
        const result = await this.pool.query(
            `SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'code_chunks_%' OR table_name LIKE 'hybrid_code_chunks_%' ORDER BY table_name`
        );
        return result.rows.map((r: any) => r.table_name);
    }

    async insert(collectionName: string, documents: VectorDocument[]): Promise<void> {
        const tbl = PgvectorDatabase.tableName(collectionName);
        for (const doc of documents) {
            await this.pool.query(
                `INSERT INTO ${tbl} (id, vector, content, "relativePath", "startLine", "endLine", "fileExtension", metadata)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                 ON CONFLICT (id) DO UPDATE SET
                   vector = EXCLUDED.vector,
                   content = EXCLUDED.content,
                   "relativePath" = EXCLUDED."relativePath",
                   "startLine" = EXCLUDED."startLine",
                   "endLine" = EXCLUDED."endLine",
                   "fileExtension" = EXCLUDED."fileExtension",
                   metadata = EXCLUDED.metadata`,
                [
                    doc.id,
                    `[${doc.vector.join(',')}]`,
                    doc.content,
                    doc.relativePath,
                    doc.startLine,
                    doc.endLine,
                    doc.fileExtension,
                    JSON.stringify(doc.metadata),
                ]
            );
        }
    }

    async insertHybrid(collectionName: string, documents: VectorDocument[]): Promise<void> {
        await this.insert(collectionName, documents);
    }

    async search(collectionName: string, queryVector: number[], options?: SearchOptions): Promise<VectorSearchResult[]> {
        const tbl = PgvectorDatabase.tableName(collectionName);
        const limit = options?.topK || 10;
        const vectorStr = `[${queryVector.join(',')}]`;

        let whereClause = '';
        const params: any[] = [vectorStr, limit];

        if (options?.filterExpr && options.filterExpr.trim().length > 0) {
            whereClause = 'WHERE ' + this.translateFilter(options.filterExpr);
        }

        const result = await this.pool.query(
            `SELECT id, content, "relativePath", "startLine", "endLine", "fileExtension", metadata,
                    1 - (vector <=> $1::vector) AS score
             FROM ${tbl} ${whereClause}
             ORDER BY vector <=> $1::vector
             LIMIT $2`,
            params
        );

        if (options?.threshold !== undefined) {
            const filtered = result.rows.filter((r: any) => r.score >= (options.threshold ?? 0));
            return filtered.map((r: any) => this.rowToSearchResult(r, queryVector));
        }

        return result.rows.map((r: any) => this.rowToSearchResult(r, queryVector));
    }

    async hybridSearch(collectionName: string, searchRequests: HybridSearchRequest[], options?: HybridSearchOptions): Promise<HybridSearchResult[]> {
        const tbl = PgvectorDatabase.tableName(collectionName);
        const limit = options?.limit || searchRequests[0]?.limit || 10;

        const denseRequest = searchRequests.find(r => r.anns_field === 'vector');
        const sparseRequest = searchRequests.find(r => r.anns_field === 'sparse_vector');

        const denseVector = denseRequest?.data as number[];
        const queryText = typeof sparseRequest?.data === 'string' ? sparseRequest.data : '';

        let whereClause = '';
        if (options?.filterExpr && options.filterExpr.trim().length > 0) {
            whereClause = 'WHERE ' + this.translateFilter(options.filterExpr);
        }

        const denseResults = await this.pool.query(
            `SELECT id, 1 - (vector <=> $1::vector) AS score
             FROM ${tbl} ${whereClause}
             ORDER BY vector <=> $1::vector
             LIMIT $2`,
            [`[${denseVector.join(',')}]`, limit]
        );

        let ftsResults: any[] = [];
        if (queryText) {
            const ftsRes = await this.pool.query(
                `SELECT id, ts_rank(content_tsvector, plainto_tsquery('english', $1)) AS score
                 FROM ${tbl} ${whereClause}
                 ORDER BY ts_rank(content_tsvector, plainto_tsquery('english', $1)) DESC
                 LIMIT $2`,
                [queryText, limit]
            );
            ftsResults = ftsRes.rows.filter((r: any) => r.score > 0);
        }

        const rrfK = 100;
        const scoreMap = new Map<string, number>();

        for (let i = 0; i < denseResults.rows.length; i++) {
            const id = denseResults.rows[i].id;
            scoreMap.set(id, 1 / (rrfK + i + 1));
        }

        for (let i = 0; i < ftsResults.length; i++) {
            const id = ftsResults[i].id;
            const existing = scoreMap.get(id) || 0;
            scoreMap.set(id, existing + 1 / (rrfK + i + 1));
        }

        if (scoreMap.size === 0) return [];

        const sortedIds = [...scoreMap.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, limit)
            .map(([id]) => id);

        const docsResult = await this.pool.query(
            `SELECT id, content, "relativePath", "startLine", "endLine", "fileExtension", metadata
             FROM ${tbl} WHERE id = ANY($1)`,
            [sortedIds]
        );

        const docMap = new Map<string, any>();
        for (const row of docsResult.rows) {
            docMap.set(row.id, row);
        }

        return sortedIds.map(id => {
            const row = docMap.get(id);
            if (!row) return null;
            return {
                document: {
                    id: row.id,
                    content: row.content,
                    vector: [],
                    relativePath: row.relativePath,
                    startLine: row.startLine,
                    endLine: row.endLine,
                    fileExtension: row.fileExtension,
                    metadata: typeof row.metadata === 'string' ? JSON.parse(row.metadata) : (row.metadata || {}),
                },
                score: scoreMap.get(id)!,
            };
        }).filter((r): r is HybridSearchResult => r !== null);
    }

    async delete(collectionName: string, ids: string[]): Promise<void> {
        const tbl = PgvectorDatabase.tableName(collectionName);
        await this.pool.query(
            `DELETE FROM ${tbl} WHERE id = ANY($1)`,
            [ids]
        );
    }

    async query(collectionName: string, filter: string, outputFields: string[], limit?: number): Promise<Record<string, any>[]> {
        const tbl = PgvectorDatabase.tableName(collectionName);
        const fields = outputFields.map(f => `"${f}"`).join(', ');
        let whereClause = '';
        const params: any[] = [];

        if (filter && filter.trim() !== '') {
            whereClause = 'WHERE ' + this.translateFilter(filter);
        }

        const limitClause = limit !== undefined ? `LIMIT ${limit}` : '';

        const result = await this.pool.query(
            `SELECT ${fields} FROM ${tbl} ${whereClause} ${limitClause}`,
            params
        );

        return result.rows.map((row: any) => {
            const mapped: Record<string, any> = {};
            for (const field of outputFields) {
                let value = row[field];
                if (field === 'metadata' && typeof value === 'string') {
                    try { value = JSON.parse(value); } catch { /* keep string */ }
                }
                mapped[field] = value;
            }
            return mapped;
        });
    }

    async getCollectionDescription(collectionName: string): Promise<string> {
        const result = await this.pool.query(
            `SELECT obj_description((quote_ident($1))::regclass, 'pg_class') AS description`,
            [collectionName]
        );
        return result.rows[0]?.description || '';
    }

    async checkCollectionLimit(): Promise<boolean> {
        return true;
    }

    async getCollectionRowCount(collectionName: string): Promise<number> {
        const tbl = PgvectorDatabase.tableName(collectionName);
        try {
            const exists = await this.hasCollection(collectionName);
            if (!exists) return -1;

            const result = await this.pool.query(`SELECT count(*)::int AS cnt FROM ${tbl}`);
            const cnt = result.rows[0]?.cnt;
            return typeof cnt === 'number' ? cnt : -1;
        } catch (error) {
            console.error(`[PgvectorDB] Error in count(*) for '${collectionName}':`, error);
            return -1;
        }
    }

    private rowToSearchResult(row: any, queryVector: number[]): VectorSearchResult {
        let metadata = {};
        try {
            metadata = typeof row.metadata === 'string' ? JSON.parse(row.metadata) : (row.metadata || {});
        } catch (error) {
            console.warn(`[PgvectorDB] Failed to parse metadata for item ${row.id}:`, error);
        }

        return {
            document: {
                id: row.id,
                vector: queryVector,
                content: row.content,
                relativePath: row.relativePath,
                startLine: row.startLine,
                endLine: row.endLine,
                fileExtension: row.fileExtension,
                metadata,
            },
            score: row.score,
        };
    }

    private translateFilter(expr: string): string {
        let sql = expr;

        sql = sql.replace(/(\w+)\s+in\s*\[([^\]]+)\]/gi, (_match, field, values) => {
            const items = values.split(',').map((v: string) => v.trim().replace(/^"|"$/g, ''));
            const sqlValues = items.map((v: string) => `'${v.replace(/'/g, "''")}'`).join(', ');
            return `"${field}" IN (${sqlValues})`;
        });

        sql = sql.replace(/(\w+)\s*==\s*"([^"]+)"/g, '"$1" = \'$2\'');

        return sql;
    }

    async close(): Promise<void> {
        await this.pool.end();
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add packages/core/src/vectordb/pgvector-vectordb.ts
git commit -m "feat(core): add PgvectorDatabase implementing VectorDatabase interface"
```

---

### Task 3: Export PgvectorDatabase from core

**Files:**
- Modify: `packages/core/src/vectordb/index.ts`

- [ ] **Step 1: Add exports for PgvectorDatabase and PgvectorConfig**

In `packages/core/src/vectordb/index.ts`, append this line after the existing Milvus export:

```typescript
export { PgvectorDatabase, PgvectorConfig } from './pgvector-vectordb';
```

The full file should look like:
```typescript
export {
    VectorDocument,
    SearchOptions,
    VectorSearchResult,
    VectorDatabase,
    HybridSearchRequest,
    HybridSearchOptions,
    HybridSearchResult,
    RerankStrategy,
    COLLECTION_LIMIT_MESSAGE
} from './types';

export { MilvusRestfulVectorDatabase, MilvusRestfulConfig } from './milvus-restful-vectordb';
export { MilvusVectorDatabase, MilvusConfig } from './milvus-vectordb';
export {
    ClusterManager,
    ZillizConfig,
    Project,
    Cluster,
    CreateFreeClusterRequest,
    CreateFreeClusterResponse,
    CreateFreeClusterWithDetailsResponse,
    DescribeClusterResponse
} from './zilliz-utils';
export { PgvectorDatabase, PgvectorConfig } from './pgvector-vectordb';
```

- [ ] **Step 2: Verify build compiles**

Run: `cd /Users/tomasz/WebstormProjects/claude-context/packages/core && pnpm build`
Expected: Build succeeds with no errors.

- [ ] **Step 3: Commit**

```bash
git add packages/core/src/vectordb/index.ts
git commit -m "feat(core): export PgvectorDatabase from vectordb index"
```

---

### Task 4: Add PGVector config to MCP server

**Files:**
- Modify: `packages/mcp/src/config.ts`

- [ ] **Step 1: Update ContextMcpConfig interface**

In `packages/mcp/src/config.ts`, update the `ContextMcpConfig` interface. After the line `collectionNameOverride?: string;` (line 24), add:

```typescript
    vectorDbProvider?: 'milvus' | 'pgvector';
    pgConnectionString?: string;
```

- [ ] **Step 2: Update createMcpConfig function**

In the `createMcpConfig` function, after the line `collectionNameOverride: envManager.get('CODE_CHUNKS_COLLECTION_NAME_OVERRIDE')` (around line 170), add:

```typescript
        vectorDbProvider: (envManager.get('VECTOR_DB_PROVIDER') as 'milvus' | 'pgvector') || 'milvus',
        pgConnectionString: envManager.get('PG_CONNECTION_STRING'),
```

Also add to the debug log section, after the `MILVUS_ADDRESS` debug line:

```typescript
    console.log(`[DEBUG]   VECTOR_DB_PROVIDER: ${envManager.get('VECTOR_DB_PROVIDER') || 'NOT SET'}`);
```

- [ ] **Step 3: Update logConfigurationSummary**

In `logConfigurationSummary`, after the Milvus Address log line (around line 183), add:

```typescript
    console.log(`[MCP]   Vector DB Provider: ${config.vectorDbProvider || 'milvus'}`);
    if (config.vectorDbProvider === 'pgvector') {
        console.log(`[MCP]   PG Connection String: ${config.pgConnectionString ? '✅ Configured' : '❌ Missing'}`);
    }
```

- [ ] **Step 4: Update showHelpMessage**

In the help text, in the "Vector Database Configuration" section, after the `CODE_CHUNKS_COLLECTION_NAME_OVERRIDE` block, add:

```
  VECTOR_DB_PROVIDER      Vector database provider: milvus (default) or pgvector
  PG_CONNECTION_STRING    PostgreSQL connection string (required when VECTOR_DB_PROVIDER=pgvector)
```

Also add an example:

```
  # Start MCP server with PGVector
  OPENAI_API_KEY=sk-xxx VECTOR_DB_PROVIDER=pgvector PG_CONNECTION_STRING=postgresql://user:pass@localhost:5432/dbname npx @zilliz/claude-context-mcp@latest
```

- [ ] **Step 5: Verify build compiles**

Run: `cd /Users/tomasz/WebstormProjects/claude-context/packages/mcp && pnpm build`
Expected: Build succeeds with no errors.

- [ ] **Step 6: Commit**

```bash
git add packages/mcp/src/config.ts
git commit -m "feat(mcp): add VECTOR_DB_PROVIDER and PG_CONNECTION_STRING config"
```

---

### Task 5: Wire factory switch in MCP server

**Files:**
- Modify: `packages/mcp/src/index.ts`

- [ ] **Step 1: Update imports**

In `packages/mcp/src/index.ts`, change line 25 from:
```typescript
import { MilvusVectorDatabase } from "@zilliz/claude-context-core";
```
to:
```typescript
import { MilvusVectorDatabase, PgvectorDatabase } from "@zilliz/claude-context-core";
```

- [ ] **Step 2: Replace hardcoded Milvus initialization with factory switch**

Replace the block at lines 62-65:
```typescript
        const vectorDatabase = new MilvusVectorDatabase({
            address: config.milvusAddress,
            ...(config.milvusToken && { token: config.milvusToken })
        });
```

with:
```typescript
        let vectorDatabase;
        if (config.vectorDbProvider === 'pgvector') {
            if (!config.pgConnectionString) {
                throw new Error('PG_CONNECTION_STRING is required when VECTOR_DB_PROVIDER=pgvector');
            }
            vectorDatabase = new PgvectorDatabase({ connectionString: config.pgConnectionString });
        } else {
            vectorDatabase = new MilvusVectorDatabase({
                address: config.milvusAddress,
                ...(config.milvusToken && { token: config.milvusToken })
            });
        }
```

- [ ] **Step 3: Verify build compiles**

Run: `cd /Users/tomasz/WebstormProjects/claude-context/packages/mcp && pnpm build`
Expected: Build succeeds with no errors.

- [ ] **Step 4: Commit**

```bash
git add packages/mcp/src/index.ts
git commit -m "feat(mcp): add factory switch for Milvus vs PGVector backend"
```

---

### Task 6: Update prerequisites documentation

**Files:**
- Modify: `docs/getting-started/prerequisites.md`

- [ ] **Step 1: Add PGVector option under Vector Database section**

After the "Local Milvus (Advanced)" section in `docs/getting-started/prerequisites.md`, add:

```markdown
#### Option 3: PostgreSQL + PGVector
- **PostgreSQL**: Version 12+ with the [pgvector](https://github.com/pgvector/pgvector) extension installed
- **Setup**: Install pgvector extension by following [this guide](https://github.com/pgvector/pgvector#installation)
- **Configuration**: Set `VECTOR_DB_PROVIDER=pgvector` and `PG_CONNECTION_STRING=postgresql://user:pass@localhost:5432/dbname`
- **Note**: MCP server only; VS Code and Chrome extensions continue using Milvus
```

- [ ] **Step 2: Commit**

```bash
git add docs/getting-started/prerequisites.md
git commit -m "docs: add PostgreSQL + PGVector as vector database option"
```

---

### Task 7: Build and verify

**Files:** None (verification only)

- [ ] **Step 1: Build the core package**

Run: `cd /Users/tomasz/WebstormProjects/claude-context/packages/core && pnpm build`
Expected: Build succeeds.

- [ ] **Step 2: Build the MCP package**

Run: `cd /Users/tomasz/WebstormProjects/claude-context/packages/mcp && pnpm build`
Expected: Build succeeds.

- [ ] **Step 3: Run core package tests**

Run: `cd /Users/tomasz/WebstormProjects/claude-context/packages/core && pnpm test`
Expected: All existing tests pass (no regressions).

- [ ] **Step 4: Run MCP package tests (if any)**

Run: `cd /Users/tomasz/WebstormProjects/claude-context/packages/mcp && pnpm test`
Expected: Tests pass or "no tests found" is acceptable since the MCP package may not have unit tests.

- [ ] **Step 5: Run typecheck on core**

Run: `cd /Users/tomasz/WebstormProjects/claude-context/packages/core && pnpm typecheck`
Expected: No type errors.
