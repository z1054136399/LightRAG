#!/usr/bin/env python3
"""
Migrate LightRAG data from JSON file storage to PostgreSQL.

Reads all workspace directories under the configured working directory and
inserts their contents into the corresponding PostgreSQL tables.  Re-running
is safe: every INSERT uses ON CONFLICT DO NOTHING, so existing rows are never
overwritten.

Usage
-----
    python scripts/migrate_json_to_pg.py [--working-dir PATH]
                                         [--vdb-table-suffix SUFFIX]

Arguments
---------
--working-dir   Root of the LightRAG JSON storage tree.  The script looks
                for data files both directly in this directory (workspace="")
                and in every non-hidden, non-underscore subdirectory.
                Default: data/rag_storage

--vdb-table-suffix
                Model-name suffix appended by PGVectorStorage when a model
                name is configured (e.g. "text_embedding_3_large").  When
                given, vectors are inserted into tables like
                LIGHTRAG_VDB_ENTITY_text_embedding_3_large instead of the
                base table.  Leave empty (default) to use base table names.

Prerequisites
-------------
1.  PostgreSQL tables must already exist.  The easiest way to create them is
    to start LightRAG once with PG storage configured — it creates all tables
    on first run.  Alternatively, set --create-tables to let this script
    create them (requires VECTOR dimension to match your embedding model).

2.  The following packages must be installed:
        pip install asyncpg pgvector python-dotenv networkx

3.  A .env file with PostgreSQL credentials must be present in the current
    working directory (or the variables must be set in the environment):
        POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER,
        POSTGRES_PASSWORD, POSTGRES_DATABASE

What gets migrated
------------------
KV stores:
    kv_store_full_docs.json       → LIGHTRAG_DOC_FULL
    kv_store_text_chunks.json     → LIGHTRAG_DOC_CHUNKS
    kv_store_llm_response_cache.json → LIGHTRAG_LLM_CACHE
    kv_store_full_entities.json   → LIGHTRAG_FULL_ENTITIES
    kv_store_full_relations.json  → LIGHTRAG_FULL_RELATIONS
    kv_store_entity_chunks.json   → LIGHTRAG_ENTITY_CHUNKS
    kv_store_relation_chunks.json → LIGHTRAG_RELATION_CHUNKS

Doc-status:
    kv_store_doc_status.json      → LIGHTRAG_DOC_STATUS

Graph:
    graph_chunk_entity_relation.graphml → lightrag_graph_nodes / lightrag_graph_edges
                                          (PGTableGraphStorage plain SQL tables, no AGE)

Vector stores (vectors decoded from NanoVectorDB base64+zlib+float16 format):
    vdb_entities.json             → LIGHTRAG_VDB_ENTITY[_suffix]
    vdb_relationships.json        → LIGHTRAG_VDB_RELATION[_suffix]
    vdb_chunks.json               → LIGHTRAG_VDB_CHUNKS[_suffix]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports — fail with a clear message if missing
# ---------------------------------------------------------------------------
try:
    import asyncpg  # type: ignore
except ImportError:
    sys.exit("asyncpg is required.  Run: pip install asyncpg")

try:
    from pgvector.asyncpg import register_vector  # type: ignore
except ImportError:
    sys.exit("pgvector is required.  Run: pip install pgvector")

try:
    import networkx as nx  # type: ignore
except ImportError:
    sys.exit("networkx is required.  Run: pip install networkx")

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    sys.exit("python-dotenv is required.  Run: pip install python-dotenv")

# Load .env from the current directory (env vars already set in the OS take
# precedence because override=False).
load_dotenv(dotenv_path=".env", override=False)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRAPH_NAMESPACE = "chunk_entity_relation"
BATCH_SIZE = 200  # records per executemany call


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def decode_nano_vector(encoded: str) -> np.ndarray:
    """Decode a NanoVectorDB compressed vector → float32 numpy array.

    NanoVectorDB stores vectors as base64( zlib( float16_bytes ) ).
    """
    raw = base64.b64decode(encoded)
    decompressed = zlib.decompress(raw)
    arr_f16 = np.frombuffer(decompressed, dtype=np.float16)
    return arr_f16.astype(np.float32)


def parse_timestamp(ts: Any) -> datetime | None:
    """Parse a timestamp from various formats to a naive UTC datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None)
    try:
        dt = datetime.fromisoformat(str(ts))
        return dt.replace(tzinfo=None)
    except Exception:
        return None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def load_json_file(path: Path) -> dict | None:
    """Read and parse a JSON file, returning None on failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"    ERROR reading {path}: {exc}")
        return None


def chunks(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------


def get_pg_config() -> dict[str, Any]:
    """Read PG connection parameters from environment / .env."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    user = os.environ.get("POSTGRES_USER")
    password = os.environ.get("POSTGRES_PASSWORD")
    database = os.environ.get("POSTGRES_DATABASE")
    if not all([user, password, database]):
        sys.exit(
            "Missing required environment variables: "
            "POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DATABASE"
        )
    return {"host": host, "port": port, "user": user, "password": password, "database": database}


async def create_pool(config: dict[str, Any]) -> asyncpg.Pool:
    """Create an asyncpg connection pool with the pgvector codec registered."""
    pool = await asyncpg.create_pool(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        database=config["database"],
        min_size=1,
        max_size=10,
        init=register_vector,
    )
    return pool


# ---------------------------------------------------------------------------
# KV store migration functions
# ---------------------------------------------------------------------------


async def migrate_full_docs(
    pool: asyncpg.Pool, workspace: str, data: dict, verbose: bool
) -> int:
    """kv_store_full_docs.json  →  LIGHTRAG_DOC_FULL."""
    sql = """
        INSERT INTO LIGHTRAG_DOC_FULL
            (id, workspace, doc_name, content, sidecar_location,
             parse_format, content_hash, process_options, chunk_options, parse_engine)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10)
        ON CONFLICT (workspace, id) DO NOTHING
    """
    rows = [
        (
            doc_id,
            workspace,
            v.get("file_path", ""),           # doc_name
            v.get("content", ""),
            v.get("sidecar_location"),
            v.get("parse_format", "raw") or "raw",
            v.get("content_hash"),
            v.get("process_options"),
            json.dumps(v.get("chunk_options") or {}),
            v.get("parse_engine"),
        )
        for doc_id, v in data.items()
    ]
    return await _executemany_batched(pool, sql, rows, verbose)


async def migrate_text_chunks(
    pool: asyncpg.Pool, workspace: str, data: dict, verbose: bool
) -> int:
    """kv_store_text_chunks.json  →  LIGHTRAG_DOC_CHUNKS."""
    now = _now_utc()
    sql = """
        INSERT INTO LIGHTRAG_DOC_CHUNKS
            (workspace, id, tokens, chunk_order_index, full_doc_id, content,
             file_path, llm_cache_list, heading, sidecar, create_time, update_time)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10::jsonb, $11, $12)
        ON CONFLICT (workspace, id) DO NOTHING
    """
    rows = [
        (
            workspace,
            chunk_id,
            int(v.get("tokens", 0)),
            int(v.get("chunk_order_index", 0)),
            v.get("full_doc_id", ""),
            v.get("content", ""),
            v.get("file_path"),
            json.dumps(v.get("llm_cache_list") or []),
            json.dumps(v.get("heading") or {}),
            json.dumps(v.get("sidecar") or {}),
            now,
            now,
        )
        for chunk_id, v in data.items()
    ]
    return await _executemany_batched(pool, sql, rows, verbose)


async def migrate_llm_cache(
    pool: asyncpg.Pool, workspace: str, data: dict, verbose: bool
) -> int:
    """kv_store_llm_response_cache.json  →  LIGHTRAG_LLM_CACHE."""
    sql = """
        INSERT INTO LIGHTRAG_LLM_CACHE
            (workspace, id, original_prompt, return_value, chunk_id, cache_type, queryparam)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
        ON CONFLICT (workspace, id) DO NOTHING
    """
    rows = [
        (
            workspace,
            cache_id,
            v.get("original_prompt"),
            v.get("return"),             # JSON key is "return" (not "return_value")
            v.get("chunk_id"),
            v.get("cache_type", "extract") or "extract",
            json.dumps(v.get("queryparam")) if v.get("queryparam") else None,
        )
        for cache_id, v in data.items()
    ]
    return await _executemany_batched(pool, sql, rows, verbose)


async def migrate_full_entities(
    pool: asyncpg.Pool, workspace: str, data: dict, verbose: bool
) -> int:
    """kv_store_full_entities.json  →  LIGHTRAG_FULL_ENTITIES."""
    now = _now_utc()
    sql = """
        INSERT INTO LIGHTRAG_FULL_ENTITIES
            (workspace, id, entity_names, count, create_time, update_time)
        VALUES ($1, $2, $3::jsonb, $4, $5, $6)
        ON CONFLICT (workspace, id) DO NOTHING
    """
    rows = [
        (
            workspace,
            doc_id,
            json.dumps(v.get("entity_names") or []),
            int(v.get("count", 0)),
            now,
            now,
        )
        for doc_id, v in data.items()
    ]
    return await _executemany_batched(pool, sql, rows, verbose)


async def migrate_full_relations(
    pool: asyncpg.Pool, workspace: str, data: dict, verbose: bool
) -> int:
    """kv_store_full_relations.json  →  LIGHTRAG_FULL_RELATIONS."""
    now = _now_utc()
    sql = """
        INSERT INTO LIGHTRAG_FULL_RELATIONS
            (workspace, id, relation_pairs, count, create_time, update_time)
        VALUES ($1, $2, $3::jsonb, $4, $5, $6)
        ON CONFLICT (workspace, id) DO NOTHING
    """
    rows = [
        (
            workspace,
            doc_id,
            json.dumps(v.get("relation_pairs") or []),
            int(v.get("count", 0)),
            now,
            now,
        )
        for doc_id, v in data.items()
    ]
    return await _executemany_batched(pool, sql, rows, verbose)


async def migrate_entity_chunks(
    pool: asyncpg.Pool, workspace: str, data: dict, verbose: bool
) -> int:
    """kv_store_entity_chunks.json  →  LIGHTRAG_ENTITY_CHUNKS."""
    now = _now_utc()
    sql = """
        INSERT INTO LIGHTRAG_ENTITY_CHUNKS
            (workspace, id, chunk_ids, count, create_time, update_time)
        VALUES ($1, $2, $3::jsonb, $4, $5, $6)
        ON CONFLICT (workspace, id) DO NOTHING
    """
    rows = [
        (
            workspace,
            entity_id,
            json.dumps(v.get("chunk_ids") or []),
            int(v.get("count", 0)),
            now,
            now,
        )
        for entity_id, v in data.items()
    ]
    return await _executemany_batched(pool, sql, rows, verbose)


async def migrate_relation_chunks(
    pool: asyncpg.Pool, workspace: str, data: dict, verbose: bool
) -> int:
    """kv_store_relation_chunks.json  →  LIGHTRAG_RELATION_CHUNKS."""
    now = _now_utc()
    sql = """
        INSERT INTO LIGHTRAG_RELATION_CHUNKS
            (workspace, id, chunk_ids, count, create_time, update_time)
        VALUES ($1, $2, $3::jsonb, $4, $5, $6)
        ON CONFLICT (workspace, id) DO NOTHING
    """
    rows = [
        (
            workspace,
            rel_id,
            json.dumps(v.get("chunk_ids") or []),
            int(v.get("count", 0)),
            now,
            now,
        )
        for rel_id, v in data.items()
    ]
    return await _executemany_batched(pool, sql, rows, verbose)


async def migrate_doc_status(
    pool: asyncpg.Pool, workspace: str, data: dict, verbose: bool
) -> int:
    """kv_store_doc_status.json  →  LIGHTRAG_DOC_STATUS."""
    sql = """
        INSERT INTO LIGHTRAG_DOC_STATUS
            (workspace, id, content_summary, content_length, chunks_count, status,
             file_path, chunks_list, track_id, metadata, error_msg, content_hash,
             created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10::jsonb, $11, $12, $13, $14)
        ON CONFLICT (workspace, id) DO NOTHING
    """
    rows = []
    for doc_id, v in data.items():
        if not isinstance(v, dict):
            continue
        created_at = parse_timestamp(v.get("created_at")) or _now_utc()
        updated_at = parse_timestamp(v.get("updated_at")) or created_at
        rows.append((
            workspace,
            doc_id,
            v.get("content_summary"),
            int(v.get("content_length", 0)),
            int(v.get("chunks_count", -1)),
            v.get("status", "unknown"),
            v.get("file_path"),
            json.dumps(v.get("chunks_list") or []),
            v.get("track_id"),
            json.dumps(v.get("metadata") or {}),
            v.get("error_msg"),
            v.get("content_hash"),
            created_at,
            updated_at,
        ))
    return await _executemany_batched(pool, sql, rows, verbose)


# ---------------------------------------------------------------------------
# Vector store migration
# ---------------------------------------------------------------------------


def _parse_vdb_entries(
    vdb_data: dict,
) -> list[tuple[dict, np.ndarray | None]]:
    """Return list of (record, vector) pairs from a NanoVectorDB JSON.

    NanoVectorDB stores the embedding matrix in one of two formats:

    * **String** (current format): the ``matrix`` top-level field is a
      base64-encoded blob of raw float32 bytes with no zlib compression.
      Shape is ``(n_records, embedding_dim)``.  This is the common case for
      NanoVectorDB ≥ 0.2.

    * **List-of-lists** (older format): each element is a plain Python list
      of floats, one per record.

    The per-record ``vector`` field (base64 + zlib + float16) is used as a
    fallback when the matrix row is unavailable or cannot be decoded.

    The returned vector is always a 1-D ``np.float32`` array.
    """
    entries = vdb_data.get("data", [])
    matrix_raw = vdb_data.get("matrix")

    # ------------------------------------------------------------------
    # Decode the matrix into a usable form.
    # matrix_raw may be:
    #   str  → base64(float32_bytes), no zlib; reshape to (n, dim)
    #   list → list-of-lists of floats (older NanoVectorDB format)
    #   None/missing → no matrix; rely entirely on per-record vectors
    # ------------------------------------------------------------------
    decoded_matrix: np.ndarray | list | None = None

    if isinstance(matrix_raw, str) and matrix_raw:
        try:
            raw = base64.b64decode(matrix_raw)
            dim = vdb_data.get("embedding_dim")
            n = len(entries)
            if dim and n and len(raw) == n * int(dim) * 4:
                decoded_matrix = np.frombuffer(raw, dtype=np.float32).reshape(n, int(dim))
            # If size does not match float32 layout, fall through to per-record vectors.
        except Exception:
            pass
    elif isinstance(matrix_raw, list):
        decoded_matrix = matrix_raw

    result = []
    for i, entry in enumerate(entries):
        vec: np.ndarray | None = None

        # Primary: row from decoded matrix.
        if decoded_matrix is not None:
            try:
                if isinstance(decoded_matrix, np.ndarray):
                    # decoded_matrix[i] is already a 1-D float32 row.
                    vec = decoded_matrix[i].copy()
                elif i < len(decoded_matrix) and decoded_matrix[i]:
                    # Older list-of-lists format.
                    candidate = np.array(decoded_matrix[i], dtype=np.float32)
                    if candidate.ndim == 1:
                        vec = candidate
            except Exception:
                pass

        # Fallback: decode from compressed per-record field.
        if vec is None:
            encoded = entry.get("vector")
            if encoded:
                try:
                    vec = decode_nano_vector(encoded)
                except Exception:
                    pass

        # Safety net: ensure the vector is exactly 1-D float32 before
        # handing it to asyncpg (a 0-D scalar would cause DataError).
        if vec is not None:
            if vec.ndim != 1:
                vec = vec.ravel().astype(np.float32)
            elif vec.dtype != np.float32:
                vec = vec.astype(np.float32)

        result.append((entry, vec))
    return result


async def migrate_vdb_entities(
    pool: asyncpg.Pool,
    workspace: str,
    vdb_data: dict,
    table: str,
    verbose: bool,
) -> int:
    """vdb_entities.json  →  LIGHTRAG_VDB_ENTITY[_suffix]."""
    now = _now_utc()
    sql = f"""
        INSERT INTO {table}
            (workspace, id, entity_name, content, content_vector,
             chunk_ids, file_path, create_time, update_time)
        VALUES ($1, $2, $3, $4, $5, $6::varchar[], $7, $8, $9)
        ON CONFLICT (workspace, id) DO NOTHING
    """
    rows = []
    for entry, vec in _parse_vdb_entries(vdb_data):
        if vec is None:
            continue
        src = entry.get("source_id", "") or ""
        chunk_ids = [c.strip() for c in src.split("<SEP>") if c.strip()]
        rows.append((
            workspace,
            entry["__id__"],
            entry.get("entity_name", ""),
            entry.get("content", ""),
            vec,
            chunk_ids,
            entry.get("file_path"),
            now,
            now,
        ))
    return await _executemany_batched(pool, sql, rows, verbose)


async def migrate_vdb_relationships(
    pool: asyncpg.Pool,
    workspace: str,
    vdb_data: dict,
    table: str,
    verbose: bool,
) -> int:
    """vdb_relationships.json  →  LIGHTRAG_VDB_RELATION[_suffix]."""
    now = _now_utc()
    sql = f"""
        INSERT INTO {table}
            (workspace, id, source_id, target_id, content, content_vector,
             chunk_ids, file_path, create_time, update_time)
        VALUES ($1, $2, $3, $4, $5, $6, $7::varchar[], $8, $9, $10)
        ON CONFLICT (workspace, id) DO NOTHING
    """
    rows = []
    for entry, vec in _parse_vdb_entries(vdb_data):
        if vec is None:
            continue
        src_field = entry.get("source_id", "") or ""
        chunk_ids = [c.strip() for c in src_field.split("<SEP>") if c.strip()]
        rows.append((
            workspace,
            entry["__id__"],
            entry.get("src_id", ""),
            entry.get("tgt_id", ""),
            entry.get("content", ""),
            vec,
            chunk_ids,
            entry.get("file_path"),
            now,
            now,
        ))
    return await _executemany_batched(pool, sql, rows, verbose)


async def migrate_vdb_chunks(
    pool: asyncpg.Pool,
    workspace: str,
    vdb_data: dict,
    table: str,
    verbose: bool,
) -> int:
    """vdb_chunks.json  →  LIGHTRAG_VDB_CHUNKS[_suffix]."""
    now = _now_utc()
    sql = f"""
        INSERT INTO {table}
            (workspace, id, full_doc_id, chunk_order_index, tokens, content,
             content_vector, file_path, create_time, update_time)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (workspace, id) DO NOTHING
    """
    rows = []
    for entry, vec in _parse_vdb_entries(vdb_data):
        if vec is None:
            continue
        rows.append((
            workspace,
            entry["__id__"],
            entry.get("full_doc_id", ""),
            int(entry.get("chunk_order_index", 0)),
            int(entry.get("tokens", 0)),
            entry.get("content", ""),
            vec,
            entry.get("file_path"),
            now,
            now,
        ))
    return await _executemany_batched(pool, sql, rows, verbose)


# ---------------------------------------------------------------------------
# Graph migration
# ---------------------------------------------------------------------------


async def migrate_graph(
    pool: asyncpg.Pool,
    workspace: str,
    graphml_path: Path,
    verbose: bool,
) -> tuple[int, int]:
    """Parse graphml and insert nodes + edges into PGTableGraphStorage tables.

    Uses lightrag_graph_nodes / lightrag_graph_edges (plain SQL tables, no AGE
    extension required).  Edges are stored in canonical order:
        src_id = min(a, b), tgt_id = max(a, b)
    matching PGTableGraphStorage's undirected-edge contract so the migrated data
    is immediately compatible with the live storage backend.

    Returns (node_count, edge_count) for the rows submitted to the DB.
    """
    G = nx.read_graphml(str(graphml_path))
    namespace = GRAPH_NAMESPACE

    # --- Nodes ---
    # entity_id must be present in properties (PGTableGraphStorage requirement).
    node_sql = """
        INSERT INTO lightrag_graph_nodes (workspace, namespace, id, properties, updated_at)
        VALUES ($1, $2, $3, $4::jsonb, now())
        ON CONFLICT (workspace, namespace, id) DO NOTHING
    """
    node_rows = []
    for node_id, node_data in G.nodes(data=True):
        props = dict(node_data)
        props["entity_id"] = node_id  # canonical; force even if already present
        node_rows.append((
            workspace,
            namespace,
            node_id,
            json.dumps(props, ensure_ascii=False),
        ))

    # --- Edges ---
    # Normalise to canonical order and deduplicate reversed duplicates so a
    # single row lands in the table regardless of GraphML edge direction.
    edge_sql = """
        INSERT INTO lightrag_graph_edges (workspace, namespace, src_id, tgt_id, properties, updated_at)
        VALUES ($1, $2, $3, $4, $5::jsonb, now())
        ON CONFLICT (workspace, namespace, src_id, tgt_id) DO NOTHING
    """
    edge_rows = []
    seen_pairs: set[tuple[str, str]] = set()
    for a, b, edge_data in G.edges(data=True):
        src, tgt = min(a, b), max(a, b)
        if (src, tgt) in seen_pairs:
            continue
        seen_pairs.add((src, tgt))
        props = dict(edge_data) if edge_data else {}
        edge_rows.append((
            workspace,
            namespace,
            src,
            tgt,
            json.dumps(props, ensure_ascii=False),
        ))

    # Nodes must be inserted before edges because of the FK constraints
    # (fk_lightrag_graph_edges_src / _tgt) created by PGTableGraphStorage DDL.
    node_count = await _executemany_batched(pool, node_sql, node_rows, verbose)
    edge_count = await _executemany_batched(pool, edge_sql, edge_rows, verbose)

    return node_count, edge_count


# ---------------------------------------------------------------------------
# Batched executemany helper
# ---------------------------------------------------------------------------


async def _executemany_batched(
    pool: asyncpg.Pool,
    sql: str,
    rows: list[tuple],
    verbose: bool,
) -> int:
    """Run executemany in BATCH_SIZE chunks; return total rows submitted."""
    if not rows:
        return 0
    total = 0
    for batch in chunks(rows, BATCH_SIZE):
        async with pool.acquire() as conn:
            await conn.executemany(sql, batch)
        total += len(batch)
        if verbose and len(rows) > BATCH_SIZE:
            print(f"      ... {total}/{len(rows)}", end="\r")
    if verbose and len(rows) > BATCH_SIZE:
        print()
    return total


# ---------------------------------------------------------------------------
# Per-workspace orchestration
# ---------------------------------------------------------------------------

# Maps JSON filename → (label, async migrate function)
_KV_TASKS = [
    ("kv_store_full_docs.json", "full_docs", migrate_full_docs),
    ("kv_store_text_chunks.json", "text_chunks", migrate_text_chunks),
    ("kv_store_llm_response_cache.json", "llm_response_cache", migrate_llm_cache),
    ("kv_store_full_entities.json", "full_entities", migrate_full_entities),
    ("kv_store_full_relations.json", "full_relations", migrate_full_relations),
    ("kv_store_entity_chunks.json", "entity_chunks", migrate_entity_chunks),
    ("kv_store_relation_chunks.json", "relation_chunks", migrate_relation_chunks),
    ("kv_store_doc_status.json", "doc_status", migrate_doc_status),
]


async def migrate_workspace(
    pool: asyncpg.Pool,
    workspace_dir: Path,
    workspace: str,
    vdb_suffix: str,
    verbose: bool,
) -> None:
    """Migrate all data from a single workspace directory."""
    ws_label = workspace if workspace else "(root)"
    print(f"\n{'='*64}")
    print(f"Workspace : {ws_label}")
    print(f"Directory : {workspace_dir}")
    print(f"{'='*64}")

    # --- KV stores + doc status ---
    for filename, label, fn in _KV_TASKS:
        path = workspace_dir / filename
        if not path.exists():
            print(f"  [skip] {label}")
            continue
        data = load_json_file(path)
        if data is None:
            continue
        count = await fn(pool, workspace, data, verbose)
        print(f"  [ok]   {label}: {count} record(s)")

    # --- Vector stores ---
    vdb_tasks = [
        ("vdb_entities.json", "vdb_entities", migrate_vdb_entities,
         f"LIGHTRAG_VDB_ENTITY{'_' + vdb_suffix if vdb_suffix else ''}"),
        ("vdb_relationships.json", "vdb_relationships", migrate_vdb_relationships,
         f"LIGHTRAG_VDB_RELATION{'_' + vdb_suffix if vdb_suffix else ''}"),
        ("vdb_chunks.json", "vdb_chunks", migrate_vdb_chunks,
         f"LIGHTRAG_VDB_CHUNKS{'_' + vdb_suffix if vdb_suffix else ''}"),
    ]

    for filename, label, fn, table in vdb_tasks:
        path = workspace_dir / filename
        if not path.exists():
            print(f"  [skip] {label}")
            continue
        data = load_json_file(path)
        if data is None:
            continue
        dim = data.get("embedding_dim", "?")
        count = await fn(pool, workspace, data, table, verbose)
        print(f"  [ok]   {label}: {count} record(s)  (dim={dim}, table={table})")

    # --- Graph ---
    graphml_path = workspace_dir / "graph_chunk_entity_relation.graphml"
    if not graphml_path.exists():
        print(f"  [skip] graph")
    else:
        print(f"  [...]  graph  (namespace: {GRAPH_NAMESPACE!r})")
        node_count, edge_count = await migrate_graph(pool, workspace, graphml_path, verbose)
        print(f"  [ok]   graph: {node_count} node(s), {edge_count} edge(s)")


# ---------------------------------------------------------------------------
# Workspace discovery
# ---------------------------------------------------------------------------


def discover_workspaces(working_dir: Path) -> list[tuple[Path, str]]:
    """Return list of (directory, workspace_id) pairs to migrate.

    Includes the root directory (workspace="") if it contains any data files,
    and every non-hidden, non-underscore-prefixed subdirectory.
    """
    data_files = {
        "kv_store_full_docs.json",
        "kv_store_doc_status.json",
        "kv_store_text_chunks.json",
        "graph_chunk_entity_relation.graphml",
        "vdb_entities.json",
    }

    result: list[tuple[Path, str]] = []

    # Root-level data
    if any((working_dir / f).exists() for f in data_files):
        result.append((working_dir, ""))

    # Subdirectory workspaces
    for subdir in sorted(working_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        if name.startswith(".") or name.startswith("_"):
            continue
        if any((subdir / f).exists() for f in data_files):
            result.append((subdir, name))

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate LightRAG JSON storage to PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--working-dir",
        default="data/rag_storage",
        metavar="PATH",
        help="LightRAG JSON storage root (default: data/rag_storage)",
    )
    parser.add_argument(
        "--vdb-table-suffix",
        default="",
        metavar="SUFFIX",
        dest="vdb_suffix",
        help=(
            "Model-name suffix for vector tables "
            "(e.g. 'text_embedding_3_large').  "
            "Leave empty to use base table names."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print extra progress information (batch progress, per-record errors)",
    )
    args = parser.parse_args()

    working_dir = Path(args.working_dir)
    if not working_dir.exists():
        sys.exit(f"Error: working directory not found: {working_dir}")

    config = get_pg_config()
    print(
        f"Connecting to PostgreSQL  "
        f"{config['user']}@{config['host']}:{config['port']}/{config['database']}"
    )

    pool = await create_pool(config)
    print("Connected.\n")

    try:
        workspaces = discover_workspaces(working_dir)
        if not workspaces:
            print(f"No workspace data found under {working_dir}")
            return

        print(f"Found {len(workspaces)} workspace(s) to migrate:")
        for _dir, ws in workspaces:
            print(f"  {ws!r:40s}  ({_dir})")

        for ws_dir, ws_id in workspaces:
            await migrate_workspace(pool, ws_dir, ws_id, args.vdb_suffix, args.verbose)

        print(f"\n{'='*64}")
        print("Migration complete.")
        print(
            "NOTE: If PGVectorStorage is configured with a model-name suffix, "
            "LightRAG will automatically migrate vectors from the base tables "
            "to the suffixed tables on first startup."
        )
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
