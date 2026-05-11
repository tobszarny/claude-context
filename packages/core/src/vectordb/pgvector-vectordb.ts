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
    private pgvectorEnsured: boolean = false;

    constructor(config: PgvectorConfig) {
        this.pool = new Pool({
            connectionString: config.connectionString,
            ...config.poolConfig,
        });
    }

    private async ensurePgvector(): Promise<void> {
        if (this.pgvectorEnsured) return;
        await this.pool.query('CREATE EXTENSION IF NOT EXISTS vector');
        this.pgvectorEnsured = true;
    }

    private translateFilter(filterExpr: string): string {
        let sql = filterExpr;
        sql = sql.replace(/(\w+)\s+in\s+\[([^\]]+)\]/g, (_match, field, values) => {
            const items = values.split(',').map((v: string) => v.trim().replace(/^"(.*)"$/, "'$1'"));
            return `"${field}" IN (${items.join(', ')})`;
        });
        sql = sql.replace(/(\w+)\s*==\s*"([^"]*)"/g, '"$1" = \'$2\'');
        return sql;
    }

    async createCollection(collectionName: string, dimension: number, description?: string): Promise<void> {
        await this.ensurePgvector();
        console.log(`[PgvectorDB] Creating collection '${collectionName}' with dimension ${dimension}`);

        await this.pool.query(`
            CREATE TABLE IF NOT EXISTS "${collectionName}" (
                id VARCHAR(512) PRIMARY KEY,
                vector VECTOR(${dimension}),
                content TEXT,
                "relativePath" VARCHAR(1024),
                "startLine" BIGINT,
                "endLine" BIGINT,
                "fileExtension" VARCHAR(32),
                metadata JSONB,
                content_tsvector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', COALESCE(content, ''))) STORED
            )
        `);

        await this.pool.query(`
            CREATE INDEX IF NOT EXISTS "idx_${collectionName}_vector"
            ON "${collectionName}" USING ivfflat (vector vector_cosine_ops) WITH (lists = 100)
        `);

        await this.pool.query(`
            CREATE INDEX IF NOT EXISTS "idx_${collectionName}_tsvector"
            ON "${collectionName}" USING GIN (content_tsvector)
        `);

        if (description) {
            await this.pool.query(`COMMENT ON TABLE "${collectionName}" IS $1`, [description]);
        }

        console.log(`[PgvectorDB] ✅ Collection '${collectionName}' created successfully`);
    }

    async createHybridCollection(collectionName: string, dimension: number, description?: string): Promise<void> {
        await this.createCollection(collectionName, dimension, description);
    }

    async dropCollection(collectionName: string): Promise<void> {
        console.log(`[PgvectorDB] Dropping collection '${collectionName}'`);
        await this.pool.query(`DROP TABLE IF EXISTS "${collectionName}"`);
    }

    async hasCollection(collectionName: string): Promise<boolean> {
        const result = await this.pool.query(
            `SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = $1
            )`,
            [collectionName]
        );
        return result.rows[0].exists;
    }

    async listCollections(): Promise<string[]> {
        const result = await this.pool.query(
            `SELECT table_name FROM information_schema.tables
             WHERE table_schema = 'public'
             AND (table_name LIKE 'code_chunks_%' OR table_name LIKE 'hybrid_code_chunks_%')`
        );
        return result.rows.map((row: any) => row.table_name);
    }

    async insert(collectionName: string, documents: VectorDocument[]): Promise<void> {
        console.log(`[PgvectorDB] Inserting ${documents.length} documents into '${collectionName}'`);

        for (const doc of documents) {
            const vectorStr = `[${doc.vector.join(',')}]`;
            await this.pool.query(
                `INSERT INTO "${collectionName}" (id, vector, content, "relativePath", "startLine", "endLine", "fileExtension", metadata)
                 VALUES ($1, $2::vector, $3, $4, $5, $6, $7, $8)
                 ON CONFLICT (id) DO UPDATE SET
                     vector = EXCLUDED.vector,
                     content = EXCLUDED.content,
                     "relativePath" = EXCLUDED."relativePath",
                     "startLine" = EXCLUDED."startLine",
                     "endLine" = EXCLUDED."endLine",
                     "fileExtension" = EXCLUDED."fileExtension",
                     metadata = EXCLUDED.metadata`,
                [doc.id, vectorStr, doc.content, doc.relativePath, doc.startLine, doc.endLine, doc.fileExtension, JSON.stringify(doc.metadata)]
            );
        }
    }

    async insertHybrid(collectionName: string, documents: VectorDocument[]): Promise<void> {
        await this.insert(collectionName, documents);
    }

    async search(collectionName: string, queryVector: number[], options?: SearchOptions): Promise<VectorSearchResult[]> {
        const vectorStr = `[${queryVector.join(',')}]`;
        const limit = options?.topK || 10;

        let whereClause = '';
        const params: any[] = [vectorStr, limit];
        let paramIdx = 2;

        if (options?.filterExpr && options.filterExpr.trim().length > 0) {
            whereClause = `WHERE ${this.translateFilter(options.filterExpr)}`;
        }

        const query = `
            SELECT id, vector, content, "relativePath", "startLine", "endLine", "fileExtension", metadata,
                   1 - (vector <=> $1::vector) AS score
            FROM "${collectionName}"
            ${whereClause}
            ORDER BY vector <=> $1::vector
            LIMIT $${paramIdx + 1}
        `;

        const result = await this.pool.query(query, params);

        return result.rows
            .map((row: any) => {
                const score = row.score;
                if (options?.threshold !== undefined && score < options.threshold) {
                    return null;
                }

                let metadata = {};
                try {
                    metadata = typeof row.metadata === 'string' ? JSON.parse(row.metadata) : (row.metadata || {});
                } catch (error) {
                    console.error(`[PgvectorDB] Failed to parse metadata for item ${row.id}:`, error);
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
                    score,
                };
            })
            .filter((r: VectorSearchResult | null): r is VectorSearchResult => r !== null);
    }

    async hybridSearch(collectionName: string, searchRequests: HybridSearchRequest[], options?: HybridSearchOptions): Promise<HybridSearchResult[]> {
        const denseRequest = searchRequests[0];
        const sparseRequest = searchRequests[1];

        const limit = options?.limit || denseRequest?.limit || 10;
        const k = options?.rerank?.params?.k || 100;

        let filterClause = '';
        if (options?.filterExpr && options.filterExpr.trim().length > 0) {
            filterClause = `WHERE ${this.translateFilter(options.filterExpr)}`;
        }

        const denseVectorStr = Array.isArray(denseRequest.data) ? `[${denseRequest.data.join(',')}]` : '';
        const denseLimit = denseRequest.limit || limit;

        const densePromise = this.pool.query(
            `SELECT id, 1 - (vector <=> $1::vector) AS score
             FROM "${collectionName}"
             ${filterClause}
             ORDER BY vector <=> $1::vector
             LIMIT $2`,
            [denseVectorStr, denseLimit]
        ).catch((error: any) => {
            console.error(`[PgvectorDB] Dense search failed:`, error);
            return { rows: [] };
        });

        let ftsPromise: Promise<any>;
        if (sparseRequest && typeof sparseRequest.data === 'string' && sparseRequest.data.trim().length > 0) {
            const ftsLimit = sparseRequest.limit || limit;
            ftsPromise = this.pool.query(
                `SELECT id, ts_rank(content_tsvector, plainto_tsquery('english', $1)) AS score
                 FROM "${collectionName}"
                 WHERE content_tsvector @@ plainto_tsquery('english', $1)
                 ORDER BY ts_rank(content_tsvector, plainto_tsquery('english', $1)) DESC
                 LIMIT $2`,
                [sparseRequest.data, ftsLimit]
            ).catch((error: any) => {
                console.error(`[PgvectorDB] FTS search failed:`, error);
                return { rows: [] };
            });
        } else {
            ftsPromise = Promise.resolve({ rows: [] });
        }

        const [denseResults, ftsResults] = await Promise.all([densePromise, ftsPromise]);

        const rrfScores: Map<string, number> = new Map();

        denseResults.rows.forEach((row: any, index: number) => {
            const rrfScore = 1 / (k + index + 1);
            rrfScores.set(row.id, (rrfScores.get(row.id) || 0) + rrfScore);
        });

        ftsResults.rows.forEach((row: any, index: number) => {
            const rrfScore = 1 / (k + index + 1);
            rrfScores.set(row.id, (rrfScores.get(row.id) || 0) + rrfScore);
        });

        const sortedIds = [...rrfScores.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, limit);

        if (sortedIds.length === 0) return [];

        const ids = sortedIds.map(([id]) => id);
        const docResults = await this.pool.query(
            `SELECT id, vector, content, "relativePath", "startLine", "endLine", "fileExtension", metadata
             FROM "${collectionName}"
             WHERE id = ANY($1)`,
            [ids]
        );

        const docMap = new Map<string, any>();
        for (const row of docResults.rows) {
            docMap.set(row.id, row);
        }

        const results: HybridSearchResult[] = [];
        for (const [id, score] of sortedIds) {
            const row = docMap.get(id);
            if (!row) continue;

            let metadata = {};
            try {
                metadata = typeof row.metadata === 'string' ? JSON.parse(row.metadata) : (row.metadata || {});
            } catch (error) {
                console.error(`[PgvectorDB] Failed to parse metadata for item ${row.id}:`, error);
            }

            results.push({
                document: {
                    id: row.id,
                    vector: [],
                    content: row.content,
                    relativePath: row.relativePath,
                    startLine: row.startLine,
                    endLine: row.endLine,
                    fileExtension: row.fileExtension,
                    metadata,
                },
                score,
            });
        }
        return results;
    }

    async delete(collectionName: string, ids: string[]): Promise<void> {
        await this.pool.query(
            `DELETE FROM "${collectionName}" WHERE id = ANY($1)`,
            [ids]
        );
    }

    async query(collectionName: string, filter: string, outputFields: string[], limit?: number): Promise<Record<string, any>[]> {
        let whereClause = '';
        if (filter && filter.trim().length > 0) {
            whereClause = `WHERE ${this.translateFilter(filter)}`;
        }

        const fields = outputFields.map(f => {
            if (f === 'metadata') return `metadata::text AS metadata`;
            if (f === 'count(*)') return 'count(*)::int AS "count(*)"';
            return `"${f}"`;
        }).join(', ');

        const limitClause = limit !== undefined ? `LIMIT ${limit}` : '';

        const result = await this.pool.query(
            `SELECT ${fields} FROM "${collectionName}" ${whereClause} ${limitClause}`
        );

        return result.rows.map((row: any) => {
            const mapped: Record<string, any> = {};
            for (const [key, value] of Object.entries(row)) {
                if (key === 'metadata' && typeof value === 'string') {
                    try {
                        mapped[key] = JSON.parse(value);
                    } catch {
                        mapped[key] = value;
                    }
                } else {
                    mapped[key] = value;
                }
            }
            return mapped;
        });
    }

    async getCollectionDescription(collectionName: string): Promise<string> {
        const result = await this.pool.query(
            `SELECT obj_description((quote_ident('public') || '.' || quote_ident($1))::regclass, 'pg_class') AS description`,
            [collectionName]
        );
        return result.rows[0]?.description || '';
    }

    async checkCollectionLimit(): Promise<boolean> {
        return true;
    }

    async getCollectionRowCount(collectionName: string): Promise<number> {
        try {
            const exists = await this.hasCollection(collectionName);
            if (!exists) return -1;

            const result = await this.pool.query(
                `SELECT count(*)::int AS cnt FROM "${collectionName}"`
            );
            return result.rows[0]?.cnt ?? -1;
        } catch (error) {
            console.error(`[PgvectorDB] Error getting row count for '${collectionName}':`, error);
            return -1;
        }
    }

    async close(): Promise<void> {
        await this.pool.end();
    }
}
