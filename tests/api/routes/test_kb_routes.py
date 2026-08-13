"""HTTP-level tests for /api/kbs CRUD, backed by a real KBRegistry
(offline LightRAG, dummy llm/embedding funcs) so validation and error
mapping are exercised end to end."""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# lightrag.api.kb_registry (and lightrag.api.routers.kb_routes, which imports
# lightrag.api.routers.document_routes) transitively import
# lightrag.api.auth -> lightrag.api.config, whose module-level global_args
# triggers argparse's parse_args() against sys.argv on first access. Under
# pytest, sys.argv holds pytest's own CLI args, which argparse doesn't
# recognize and aborts on. Temporarily clear sys.argv for the import,
# matching the pattern in tests/api/test_kb_registry.py and
# tests/api/routes/test_document_routes_chunking.py.
#
# lightrag.api.utils_api additionally computes its module-level
# `whitelist_patterns` / `auth_configured` once, at first import, from
# WHITELIST_PATHS. The documented default ("/health,/api/*") deliberately
# keeps the whole /api/* namespace open for Ollama-compatible endpoints
# (see env.example) -- but our router also lives under /api/*, so without
# pinning WHITELIST_PATHS before this first import, test_requires_auth below
# would see /api/kbs treated as whitelisted and never reach the auth check.
# Force a narrower default for this module's (first) import of utils_api so
# the auth dependency is actually exercised, matching the explicit
# WHITELIST_PATHS override pattern in tests/api/routes/test_login_route.py.
os.environ.setdefault("WHITELIST_PATHS", "/health")
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_kb_registry = importlib.import_module("lightrag.api.kb_registry")
_kb_routes = importlib.import_module("lightrag.api.routers.kb_routes")
sys.argv = _original_argv

KBRegistry = _kb_registry.KBRegistry
create_kb_routes = _kb_routes.create_kb_routes

from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline

# X-API-Key alone authenticates (matches env.example's curl usage). An
# "Authorization: Bearer <api-key>" header would instead be parsed as a JWT
# by the oauth2 half of the combined auth dependency, fail to decode, and
# short-circuit to 401 before the X-API-Key check ever runs -- so it is
# deliberately omitted here.
_HEADERS = {"X-API-Key": "test-key"}


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


@pytest.fixture
async def client(tmp_path):
    def _build_rag_kwargs() -> dict:
        return dict(
            working_dir=str(tmp_path / "wd"),
            llm_model_func=_dummy_llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=8, max_token_size=8192, func=_dummy_embedding
            ),
            max_parallel_insert=1,
        )

    registry = KBRegistry(
        working_dir=str(tmp_path / "wd"),
        input_dir=str(tmp_path / "inputs"),
        api_key="test-key",
        top_k=10,
        build_rag_kwargs=_build_rag_kwargs,
    )
    app = FastAPI()
    await registry.initialize(app)
    app.include_router(create_kb_routes(registry, api_key="test-key"))
    return TestClient(app)


@pytest.mark.asyncio
async def test_list_empty(client):
    resp = client.get("/api/kbs", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_create_returns_id_and_appears_in_list(client):
    resp = client.post("/api/kbs", headers=_HEADERS, json={"name": "Docs"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Docs"
    assert body["id"]

    listed = client.get("/api/kbs", headers=_HEADERS).json()
    assert [kb["id"] for kb in listed] == [body["id"]]


@pytest.mark.asyncio
async def test_create_duplicate_name_returns_409(client):
    client.post("/api/kbs", headers=_HEADERS, json={"name": "Dup"})
    resp = client.post("/api/kbs", headers=_HEADERS, json={"name": "dup"})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_create_with_default_chunking(client):
    resp = client.post(
        "/api/kbs",
        headers=_HEADERS,
        json={
            "name": "Chunked",
            "default_chunking": {
                "strategy": "recursive_character",
                "params": {"chunk_token_size": 800},
            },
        },
    )
    assert resp.status_code == 200
    assert resp.json()["default_chunking"]["strategy"] == "recursive_character"


@pytest.mark.asyncio
async def test_create_invalid_chunking_returns_422(client):
    resp = client.post(
        "/api/kbs",
        headers=_HEADERS,
        json={
            "name": "Bad",
            "default_chunking": {"strategy": "fixed_token", "params": {"chunk_token_size": 0}},
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_get_unknown_kb_returns_404(client):
    resp = client.get("/api/kbs/does-not-exist", headers=_HEADERS)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_patch_renames_kb(client):
    created = client.post("/api/kbs", headers=_HEADERS, json={"name": "Old"}).json()
    resp = client.patch(f"/api/kbs/{created['id']}", headers=_HEADERS, json={"name": "New"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "New"


@pytest.mark.asyncio
async def test_delete_removes_kb(client):
    created = client.post("/api/kbs", headers=_HEADERS, json={"name": "ToDelete"}).json()
    resp = client.delete(f"/api/kbs/{created['id']}", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json() == {"status": "success"}

    assert client.get(f"/api/kbs/{created['id']}", headers=_HEADERS).status_code == 404


@pytest.mark.asyncio
async def test_requires_auth(client):
    # Registry/router are configured with an API key and no AUTH_ACCOUNTS
    # (API-key-only mode). get_combined_auth_dependency's real behavior for
    # a request with no credentials in that mode is 403 ("API Key
    # required") -- 401 is reserved for password-auth mode with a missing
    # token. See lightrag/api/utils_api.py's combined_dependency.
    resp = client.get("/api/kbs")
    assert resp.status_code == 403
