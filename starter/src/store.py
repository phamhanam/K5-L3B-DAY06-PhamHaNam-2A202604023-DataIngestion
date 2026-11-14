"""Store and search vectors using PostgreSQL + pgvector."""
import json
from typing import Any
import psycopg2
from psycopg2.extras import Json, execute_values

from .models import Chunk

ALLOWED_FILTER_KEYS = {"chapter", "section", "year", "category", "source", "document_id"}


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("database_url must not be empty")
        if dimensions <= 0:
            raise ValueError(f"dimensions must be > 0, got {dimensions}")
        self.database_url = database_url
        self.dimensions = dimensions

    def _connect(self):
        return psycopg2.connect(self.database_url)

    def create_schema(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    strategy TEXT NOT NULL,
                    id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, id)
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks (strategy);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON chunks USING GIN (metadata);"
            )
            conn.commit()

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Length mismatch: {len(chunks)} chunks vs {len(vectors)} vectors"
            )
        if not chunks:
            return

        for idx, vec in enumerate(vectors):
            if len(vec) != self.dimensions:
                raise ValueError(
                    f"Vector {idx} has {len(vec)} dimensions; expected {self.dimensions}"
                )

        rows = []
        ids_by_doc: dict[str, list[str]] = {}

        for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            chunk_id = getattr(chunk, "chunk_id", None) or (
                f"{chunk.document_id}_p{chunk.page}_{idx}"
            )
            ids_by_doc.setdefault(chunk.document_id, []).append(chunk_id)
            rows.append(
                (
                    strategy,
                    chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page,
                    chunk.text,
                    Json(chunk.metadata or {}),
                    _vec_literal(vec),
                )
            )

        with self._connect() as conn, conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO chunks
                    (strategy, id, document_id, source, page, text, metadata, embedding)
                VALUES %s
                ON CONFLICT (strategy, id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    source = EXCLUDED.source,
                    page = EXCLUDED.page,
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding;
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s, %s, %s::vector)",
            )

            # Remove obsolete chunks when source docs are re-ingested
            for doc_id, active_ids in ids_by_doc.items():
                cur.execute(
                    """
                    DELETE FROM chunks
                    WHERE strategy = %s
                      AND document_id = %s
                      AND NOT (id = ANY(%s));
                    """,
                    (strategy, doc_id, active_ids),
                )
            conn.commit()

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query vector has {len(query_vector)} dimensions; expected {self.dimensions}"
            )

        qvec = _vec_literal(query_vector)
        where_parts = ["strategy = %s"]
        params: list[Any] = [qvec, strategy]

        if filters:
            invalid_keys = set(filters.keys()) - ALLOWED_FILTER_KEYS
            if invalid_keys:
                raise ValueError(f"Unsupported filter keys: {sorted(invalid_keys)}")
            where_parts.append("metadata @> %s::jsonb")
            params.append(json.dumps(filters))

        params.extend([qvec, top_k])
        where_sql = " AND ".join(where_parts)

        sql = f"""
            SELECT
                id,
                text,
                1 - (embedding <=> %s::vector) AS score,
                metadata,
                document_id,
                source,
                page
            FROM chunks
            WHERE {where_sql}
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s;
        """

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return [
                {
                    "chunk_id": row[0],
                    "text": row[1],
                    "score": round(float(row[2]), 4),
                    "metadata": row[3],
                    "document_id": row[4],
                    "source": row[5],
                    "page": row[6],
                }
                for row in cur.fetchall()
            ]
        