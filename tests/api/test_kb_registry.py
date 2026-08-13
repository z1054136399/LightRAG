"""KBRegistry: metadata CRUD + per-KB LightRAG lifecycle.

Uses real LightRAG instances (dummy llm/embedding funcs, JSON storages) and a
real FastAPI app with parameterized ``/api/kbs/{kb_id}`` routes so route
behaviour is exercised end to end.  The KB-scoped routes are now mounted ONCE
with a ``{kb_id}`` path parameter instead of being dynamically added/removed
per KB, so tests create a test app with parameterized routes via
``_make_test_app``.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# lightrag.api.kb_registry transitively imports lightrag.api.routers.document_routes,
# which imports lightrag.api.auth -> lightrag.api.config, whose module-level
# global_args triggers argparse's parse_args() against sys.argv on first access.
# Under pytest, sys.argv holds pytest's own CLI args (e.g. this file's path,
# "-v"), which argparse doesn't recognize and aborts on. Temporarily clear
# sys.argv for the import, matching the pattern in
# tests/api/routes/test_document_routes_chunking.py.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_kb_registry = importlib.import_module("lightrag.api.kb_registry")
_doc_routes_mod = importlib.import_module("lightrag.api.routers.document_routes")
_graph_routes_mod = importlib.import_module("lightrag.api.routers.graph_routes")
_query_routes_mod = importlib.import_module("lightrag.api.routers.query_routes")
sys.argv = _original_argv

KBDuplicateNameError = _kb_registry.KBDuplicateNameError
KBNotFoundError = _kb_registry.KBNotFoundError
KBRegistry = _kb_registry.KBRegistry
create_document_routes = _doc_routes_mod.create_document_routes
create_graph_routes = _graph_routes_mod.create_graph_routes
create_query_routes = _query_routes_mod.create_query_routes

from lightrag.kg.shared_storage import finalize_share_data
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _isolated_shared_storage():
    """Give every test a fresh in-process shared-storage namespace map."""
    finalize_share_data()
    yield
    finalize_share_data()


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


def _build_rag_kwargs_factory(tmp_path):
    def _build() -> dict:
        return dict(
            working_dir=str(tmp_path / "wd"),
            llm_model_func=_dummy_llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=8, max_token_size=8192, func=_dummy_embedding
            ),
            max_parallel_insert=1,
        )

    return _build


def _make_registry(tmp_path) -> KBRegistry:
    return KBRegistry(
        working_dir=str(tmp_path / "wd"),
        input_dir=str(tmp_path / "inputs"),
        api_key=None,
        top_k=10,
        build_rag_kwargs=_build_rag_kwargs_factory(tmp_path),
    )


_TEST_API_KEY = "test-api-key"
_TEST_AUTH = {"X-API-Key": _TEST_API_KEY}


def _make_test_app(registry: KBRegistry, input_dir: str) -> FastAPI:
    """Create a FastAPI app with parameterized KB-scoped routes mounted."""
    app = FastAPI()
    _kb_prefix = "/api/kbs/{kb_id}"
    app.include_router(
        create_document_routes(api_key=_TEST_API_KEY, kb_registry=registry, input_dir=input_dir),
        prefix=_kb_prefix,
    )
    app.include_router(
        create_graph_routes(api_key=_TEST_API_KEY, kb_registry=registry),
        prefix=_kb_prefix,
    )
    app.include_router(
        create_query_routes(api_key=_TEST_API_KEY, kb_registry=registry),
        prefix=_kb_prefix,
    )
    return app


@pytest.mark.asyncio
async def test_create_persists_meta_and_routes_accessible(tmp_path):
    registry = _make_registry(tmp_path)
    await registry.initialize()

    meta = await registry.create(name="My KB", description="test kb")

    assert meta.name == "My KB"
    assert meta.description == "test kb"
    assert meta.id  # non-empty uuid hex
    assert meta.default_chunking is None

    app = _make_test_app(registry, str(tmp_path / "inputs"))
    client = TestClient(app)
    # Parameterized route mounted: a GET to the KB-scoped documents list must not 404.
    response = client.get(f"/api/kbs/{meta.id}/documents", headers=_TEST_AUTH)
    assert response.status_code != 404


@pytest.mark.asyncio
async def test_create_rejects_duplicate_name_case_insensitive(tmp_path):
    registry = _make_registry(tmp_path)
    await registry.initialize()

    await registry.create(name="Sales Docs")
    with pytest.raises(KBDuplicateNameError):
        await registry.create(name="sales docs")


@pytest.mark.asyncio
async def test_list_returns_all_created_kbs(tmp_path):
    registry = _make_registry(tmp_path)
    await registry.initialize()

    a = await registry.create(name="A")
    b = await registry.create(name="B")

    metas = await registry.list()
    assert {m.id for m in metas} == {a.id, b.id}


@pytest.mark.asyncio
async def test_get_meta_unknown_id_raises(tmp_path):
    registry = _make_registry(tmp_path)
    await registry.initialize()

    with pytest.raises(KBNotFoundError):
        await registry.get_meta("does-not-exist")


@pytest.mark.asyncio
async def test_get_rag_returns_initialized_instance(tmp_path):
    registry = _make_registry(tmp_path)
    await registry.initialize()

    meta = await registry.create(name="Rag KB")
    rag = await registry.get_rag(meta.id)
    assert rag.workspace == meta.id


@pytest.mark.asyncio
async def test_update_renames_and_rejects_duplicate(tmp_path):
    registry = _make_registry(tmp_path)
    await registry.initialize()

    a = await registry.create(name="Original")
    b = await registry.create(name="Other")

    updated = await registry.update(a.id, name="Renamed")
    assert updated.name == "Renamed"

    with pytest.raises(KBDuplicateNameError):
        await registry.update(a.id, name="Other")

    with pytest.raises(KBNotFoundError):
        await registry.update("nope", name="X")


@pytest.mark.asyncio
async def test_delete_makes_kb_routes_return_404(tmp_path):
    """After deleting a KB, its routes return 404 (registry returns KBNotFoundError)."""
    registry = _make_registry(tmp_path)
    await registry.initialize()

    meta = await registry.create(name="Temp KB")
    kb_dir = tmp_path / "wd" / meta.id
    assert kb_dir.exists()

    app = _make_test_app(registry, str(tmp_path / "inputs"))
    client = TestClient(app)
    assert client.get(f"/api/kbs/{meta.id}/documents", headers=_TEST_AUTH).status_code != 404

    await registry.delete(meta.id)

    # After deletion the registry raises KBNotFoundError → route handler returns 404.
    assert client.get(f"/api/kbs/{meta.id}/documents", headers=_TEST_AUTH).status_code == 404
    assert not kb_dir.exists()
    with pytest.raises(KBNotFoundError):
        await registry.get_meta(meta.id)


@pytest.mark.asyncio
async def test_delete_unknown_id_raises(tmp_path):
    registry = _make_registry(tmp_path)
    await registry.initialize()

    with pytest.raises(KBNotFoundError):
        await registry.delete("nope")


@pytest.mark.asyncio
async def test_delete_one_kb_leaves_sibling_kb_routes_intact(tmp_path):
    registry = _make_registry(tmp_path)
    await registry.initialize()

    keep = await registry.create(name="Keep")
    doomed = await registry.create(name="Doomed")

    app = _make_test_app(registry, str(tmp_path / "inputs"))
    client = TestClient(app)

    await registry.delete(doomed.id)

    assert client.get(f"/api/kbs/{keep.id}/documents", headers=_TEST_AUTH).status_code != 404
    assert client.get(f"/api/kbs/{doomed.id}/documents", headers=_TEST_AUTH).status_code == 404


@pytest.mark.asyncio
async def test_initialize_reloads_persisted_kb_metadata(tmp_path):
    """After a restart, persisted KB metadata is loaded and get_rag() works."""
    build_kwargs = _build_rag_kwargs_factory(tmp_path)

    first_registry = KBRegistry(
        working_dir=str(tmp_path / "wd"),
        input_dir=str(tmp_path / "inputs"),
        api_key=None,
        top_k=10,
        build_rag_kwargs=build_kwargs,
    )
    await first_registry.initialize()
    meta = await first_registry.create(name="Survives Restart")
    await first_registry.finalize_all()

    second_registry = KBRegistry(
        working_dir=str(tmp_path / "wd"),
        input_dir=str(tmp_path / "inputs"),
        api_key=None,
        top_k=10,
        build_rag_kwargs=build_kwargs,
    )
    await second_registry.initialize()

    # Metadata is available immediately after initialize()
    reloaded = await second_registry.get_meta(meta.id)
    assert reloaded.name == "Survives Restart"

    # get_rag() triggers lazy init
    rag = await second_registry.get_rag(meta.id)
    assert rag.workspace == meta.id

    # Routes work via parameterized app
    app = _make_test_app(second_registry, str(tmp_path / "inputs"))
    client = TestClient(app)
    assert client.get(f"/api/kbs/{meta.id}/documents", headers=_TEST_AUTH).status_code != 404


@pytest.mark.asyncio
async def test_initialize_storage_lazy_not_eager(tmp_path):
    """After initialize(), metadata is loaded but LightRAG storages are not yet open."""
    build_kwargs = _build_rag_kwargs_factory(tmp_path)

    # Create a KB and persist it.
    first_registry = _make_registry(tmp_path)
    await first_registry.initialize()
    meta = await first_registry.create(name="Lazy KB")
    await first_registry.finalize_all()

    # Second registry: simulate a server restart.
    second_registry = KBRegistry(
        working_dir=str(tmp_path / "wd"),
        input_dir=str(tmp_path / "inputs"),
        api_key=None,
        top_k=10,
        build_rag_kwargs=build_kwargs,
    )
    await second_registry.initialize()

    # After initialize(), metadata is loaded but rag storage is NOT yet open.
    assert meta.id in second_registry._metas
    assert meta.id not in second_registry._rags

    # After get_rag(), the rag is cached and initialized.
    await second_registry.get_rag(meta.id)
    assert meta.id in second_registry._rags


@pytest.mark.asyncio
async def test_delete_never_initialized_kb(tmp_path):
    """Deleting a KB that was loaded from metadata but never accessed works."""
    build_kwargs = _build_rag_kwargs_factory(tmp_path)

    # Create a KB.
    first_registry = _make_registry(tmp_path)
    await first_registry.initialize()
    meta = await first_registry.create(name="Never Used")
    await first_registry.finalize_all()

    # Second registry: load from metadata, never call get_rag.
    second_registry = KBRegistry(
        working_dir=str(tmp_path / "wd"),
        input_dir=str(tmp_path / "inputs"),
        api_key=None,
        top_k=10,
        build_rag_kwargs=build_kwargs,
    )
    await second_registry.initialize()

    assert meta.id in second_registry._metas
    assert meta.id not in second_registry._rags  # never initialized

    # delete() must succeed without calling finalize_storages (rag was never created)
    await second_registry.delete(meta.id)

    assert meta.id not in second_registry._metas
    with pytest.raises(KBNotFoundError):
        await second_registry.get_meta(meta.id)
