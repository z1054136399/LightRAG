"""Registry of isolated knowledge bases (KBs), each backed by its own
LightRAG workspace.

A KB's ``id`` (a UUID) is the LightRAG ``workspace`` value and is immutable;
``name`` is a mutable display label. Metadata for every KB is persisted in a
single JSON blob in a reserved ``__meta__`` workspace so restarts can reload
every previously created KB.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from lightrag import LightRAG
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import initialize_share_data

_REGISTRY_NAMESPACE = "kb_registry"
_REGISTRY_WORKSPACE = "__meta__"
_REGISTRY_KEY = "index"


class KBNotFoundError(Exception):
    """Raised when a KB id has no matching registry entry."""

    def __init__(self, kb_id: str):
        super().__init__(f"Knowledge base '{kb_id}' not found")
        self.kb_id = kb_id


class KBDuplicateNameError(Exception):
    """Raised when creating/renaming a KB to a name already in use (case-insensitive)."""

    def __init__(self, name: str):
        super().__init__(f"A knowledge base named '{name}' already exists")
        self.name = name


@dataclass
class KBMeta:
    id: str
    name: str
    description: str = ""
    created_at: str = ""
    default_chunking: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "default_chunking": self.default_chunking,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KBMeta":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            default_chunking=data.get("default_chunking"),
        )


class KBRegistry:
    def __init__(
        self,
        *,
        working_dir: str,
        input_dir: str,
        api_key: Optional[str],
        top_k: int,
        build_rag_kwargs: Callable[[], dict[str, Any]],
    ):
        self._working_dir = working_dir
        self._input_dir = input_dir
        self._api_key = api_key
        self._top_k = top_k
        self._build_rag_kwargs = build_rag_kwargs
        self._metas: dict[str, KBMeta] = {}
        self._rags: dict[str, LightRAG] = {}
        self._lock = asyncio.Lock()
        self._meta_store: Optional[JsonKVStorage] = None

    async def initialize(self) -> None:
        """Load persisted KB metadata only. Call once from the server lifespan
        startup, before the app accepts requests.

        LightRAG instances and storage backends are initialized lazily on the
        first :meth:`get_rag` call for each KB. This keeps server startup time
        proportional to the *number* of KBs (a simple metadata read) rather
        than to their data volume.
        """
        # LightRAG.__post_init__ normally does this on first instantiation,
        # but the registry's meta store (a bare JsonKVStorage, not a full
        # LightRAG instance) is constructed here before any per-KB LightRAG
        # exists, so it must ensure shared storage itself. Idempotent: a
        # later call in the same process is a no-op.
        initialize_share_data()
        self._meta_store = JsonKVStorage(
            namespace=_REGISTRY_NAMESPACE,
            workspace=_REGISTRY_WORKSPACE,
            global_config={"working_dir": self._working_dir},
            embedding_func=None,
        )
        await self._meta_store.initialize()
        record = await self._meta_store.get_by_id(_REGISTRY_KEY)
        kbs = (record or {}).get("kbs", {})
        for kb_id, data in kbs.items():
            self._metas[kb_id] = KBMeta.from_dict(data)
        # No LightRAG instantiation, no initialize_storages here.
        # All of that is deferred to _ensure_initialized() on first get_rag().

    async def finalize_all(self) -> None:
        """Finalize every KB's storages. Call once from server shutdown."""
        for rag in self._rags.values():
            await rag.finalize_storages()

    async def _persist(self) -> None:
        await self._meta_store.upsert(
            {
                _REGISTRY_KEY: {
                    "kbs": {kb_id: meta.to_dict() for kb_id, meta in self._metas.items()}
                }
            }
        )
        await self._meta_store.index_done_callback()

    async def list(self) -> list[KBMeta]:
        return list(self._metas.values())

    async def get_meta(self, kb_id: str) -> KBMeta:
        try:
            return self._metas[kb_id]
        except KeyError:
            raise KBNotFoundError(kb_id) from None

    async def _ensure_initialized(self, kb_id: str) -> LightRAG:
        """Initialize storages for *kb_id* on first access.

        Idempotent and concurrency-safe: a double-check lock ensures that
        even if two coroutines race on the same *kb_id*, only one performs
        the (I/O-heavy) initialization; the second sees the cached result.
        """
        if kb_id in self._rags:  # fast path — no lock needed for a read
            return self._rags[kb_id]
        async with self._lock:
            if kb_id in self._rags:  # double-check after acquiring the lock
                return self._rags[kb_id]
            if kb_id not in self._metas:
                raise KBNotFoundError(kb_id)
            rag = LightRAG(workspace=kb_id, **self._build_rag_kwargs())
            await rag.initialize_storages()
            self._rags[kb_id] = rag
            return rag

    async def get_rag(self, kb_id: str) -> LightRAG:
        """Return the initialized LightRAG for *kb_id*, triggering lazy init on first call.

        Raises :class:`KBNotFoundError` for unknown *kb_id* values.
        """
        if kb_id not in self._metas:
            raise KBNotFoundError(kb_id)
        return await self._ensure_initialized(kb_id)

    def get_rag_if_ready(self, kb_id: str) -> Optional[LightRAG]:
        """Return the cached LightRAG for *kb_id* if already initialized, else ``None``.

        Never triggers lazy initialization. Used by :class:`AdmissionMiddleware`
        whose ``rag_getter`` is synchronous. A ``None`` return degrades
        gracefully to the route's own reservation.
        """
        return self._rags.get(kb_id)

    def _assert_name_available(self, name: str, *, exclude_kb_id: Optional[str] = None) -> None:
        normalized = name.strip().lower()
        for other_id, other in self._metas.items():
            if other_id == exclude_kb_id:
                continue
            if other.name.strip().lower() == normalized:
                raise KBDuplicateNameError(name)

    async def create(
        self,
        *,
        name: str,
        description: str = "",
        default_chunking: Optional[dict[str, Any]] = None,
    ) -> KBMeta:
        async with self._lock:
            self._assert_name_available(name)
            kb_id = uuid4().hex
            meta = KBMeta(
                id=kb_id,
                name=name,
                description=description,
                created_at=datetime.now(timezone.utc).isoformat(),
                default_chunking=default_chunking,
            )
            rag = LightRAG(workspace=kb_id, **self._build_rag_kwargs())
            await rag.initialize_storages()
            self._rags[kb_id] = rag
            self._metas[kb_id] = meta
            await self._persist()
            return meta

    async def update(
        self,
        kb_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        default_chunking: Optional[dict[str, Any]] = None,
        clear_default_chunking: bool = False,
    ) -> KBMeta:
        async with self._lock:
            meta = self._metas.get(kb_id)
            if meta is None:
                raise KBNotFoundError(kb_id)
            if name is not None:
                self._assert_name_available(name, exclude_kb_id=kb_id)
                meta.name = name
            if description is not None:
                meta.description = description
            if clear_default_chunking:
                meta.default_chunking = None
            elif default_chunking is not None:
                meta.default_chunking = default_chunking
            await self._persist()
            return meta

    async def delete(self, kb_id: str) -> None:
        async with self._lock:
            if kb_id not in self._metas:
                raise KBNotFoundError(kb_id)
            rag = self._rags.pop(kb_id, None)
            if rag is not None:
                await rag.finalize_storages()
            del self._metas[kb_id]
            await self._persist()
            kb_dir = Path(self._working_dir) / kb_id
            if kb_dir.exists():
                shutil.rmtree(kb_dir, ignore_errors=True)
