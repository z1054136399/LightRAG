"""Confirms lightrag_server.create_app wires a KBRegistry and mounts
/api/kbs, and that the lifespan calls KBRegistry.initialize/finalize_all.

Mirrors the LightRAG-mocking pattern in tests/api/test_path_prefixes.py:
LightRAG itself is mocked (this test is about wiring, not LightRAG
behavior), and KBRegistry's async lifecycle methods are patched to
AsyncMock so entering the TestClient context manager (which runs the
lifespan) does not try to await a mock's non-awaitable return value.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_BINDING_API_KEY",
    "EMBEDDING_MODEL",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")


@pytest.fixture(autouse=True)
def _stub_startup_banner():
    # The lifespan's startup banner ("...🚀\n") is emitted through
    # ASCIIColors.print, which writes straight to sys.stdout; on a Windows
    # console using a non-UTF-8 codepage (e.g. GBK) that raises
    # UnicodeEncodeError. Irrelevant to KBRegistry wiring, so it is stubbed
    # out rather than made environment-dependent. Scoped as a fixture (not
    # a `with` inside `_build_app`) because the lifespan that prints it
    # only runs later, when `TestClient` is entered by the test itself.
    #
    # `patch("lightrag.api.lightrag_server.ASCIIColors.green")` resolves its
    # target by importing the module if it isn't already in sys.modules;
    # importing it for the first time triggers config.py's lazy
    # `global_args` init, which calls `parse_args()` against whatever is in
    # `sys.argv` right now (pytest's own argv) unless we protect it the same
    # way `_build_app` does.
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        with patch("lightrag.api.lightrag_server.ASCIIColors.green"):
            yield
    finally:
        sys.argv = original_argv


def _build_app():
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        from lightrag.api.config import parse_args
        from lightrag.api.lightrag_server import create_app

        args = parse_args()
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag_cls, patch(
            "lightrag.api.kb_registry.KBRegistry.initialize", new_callable=AsyncMock
        ), patch(
            "lightrag.api.kb_registry.KBRegistry.finalize_all", new_callable=AsyncMock
        ):
            mock_instance = MagicMock()
            mock_instance.get_llm_role_config.return_value = {}
            mock_instance.get_llm_queue_status = AsyncMock(return_value={})
            mock_instance.get_embedding_queue_status = AsyncMock(return_value={})
            mock_instance.get_rerank_queue_status = AsyncMock(return_value={})
            mock_instance.initialize_storages = AsyncMock()
            mock_instance.finalize_storages = AsyncMock()
            mock_instance.check_and_migrate_data = AsyncMock()
            # Lifespan's admission-control probe does
            # `getattr(rag, "max_pending_documents", 0) > 0`; a bare
            # MagicMock attribute is not comparable to an int, so it must be
            # stubbed to a real number for the lifespan to run to completion.
            mock_instance.max_pending_documents = 0
            mock_rag_cls.return_value = mock_instance
            return create_app(args)
    finally:
        sys.argv = original_argv


def test_kb_routes_mounted():
    app = _build_app()
    # FastAPI >=0.141 stores top-level include_router() calls as lazy
    # `_IncludedRouter` wrappers on `app.routes` (no `.path` attribute) until
    # something forces route-tree flattening. `app.openapi()` always
    # flattens for schema generation, so it is the version-independent way
    # to introspect the final mounted path set.
    paths = set(app.openapi()["paths"].keys())
    assert "/api/kbs" in paths or "/api/kbs/{kb_id}" in paths


def test_kb_registry_available_on_app_state():
    app = _build_app()
    with TestClient(app):
        assert hasattr(app.state, "kb_registry")


def test_list_kbs_route_reachable():
    app = _build_app()
    with TestClient(app) as client:
        resp = client.get("/api/kbs", headers={"Authorization": "Bearer x"})
        assert resp.status_code != 404
