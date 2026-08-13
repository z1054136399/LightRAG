"""KB management endpoints: create, list, get, rename/update, delete.

Mounted once at server startup under the router's own ``/api/kbs`` prefix
(no per-KB scoping — this router manages the registry itself, not any one
KB's documents/query/graph, which are mounted separately by
``KBRegistry._mount``).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from lightrag.api.kb_registry import KBDuplicateNameError, KBNotFoundError, KBRegistry
from lightrag.api.routers.document_routes import TextChunkingConfig
from lightrag.api.routers.query_routes import QueryRequest, ReferenceItem
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.base import QueryParam
from lightrag.multimodal_utils import (
    doc_stem_from_filepath,
    inject_image_urls,
    is_multimodal_enabled,
    multimodal_answer_hint,
    resolve_public_base,
)
from lightrag.prompt import PROMPTS
from lightrag.utils import logger


class KBMetaResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    default_chunking: Optional[dict] = None


class CreateKBRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=2000)
    default_chunking: Optional[TextChunkingConfig] = None


class UpdateKBRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    description: Optional[str] = Field(default=None, max_length=2000)
    default_chunking: Optional[TextChunkingConfig] = None
    clear_default_chunking: bool = False


def _to_response(meta) -> KBMetaResponse:
    return KBMetaResponse(
        id=meta.id,
        name=meta.name,
        description=meta.description,
        created_at=meta.created_at,
        default_chunking=meta.default_chunking,
    )


class MultiKBQueryRequest(QueryRequest):
    kbs: List[str] = Field(
        default_factory=list,
        min_length=1,
        description="List of KB IDs to query in parallel.",
    )


class KBQueryResult(BaseModel):
    kb_id: str
    kb_name: str
    response: str
    references: List[ReferenceItem] = Field(default_factory=list)


class MultiKBQueryResponse(BaseModel):
    responses: List[KBQueryResult]
    all_references: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Deduplicated union of references from all KBs.",
    )


def create_kb_routes(registry: KBRegistry, api_key: Optional[str] = None) -> APIRouter:
    router = APIRouter(prefix="/api/kbs", tags=["knowledge-bases"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        "/query",
        response_model=MultiKBQueryResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def multi_kb_query(request: MultiKBQueryRequest):
        """Query multiple knowledge bases in parallel and return per-KB responses."""
        if not request.kbs:
            raise HTTPException(status_code=422, detail="'kbs' must contain at least one KB ID")

        # Validate all KB IDs exist before fanning out
        unknown = []
        for kb_id in request.kbs:
            if kb_id not in registry._metas:
                unknown.append(kb_id)
        if unknown:
            raise HTTPException(status_code=404, detail=f"Unknown KB IDs: {unknown}")

        param = request.to_query_params(False)
        param.stream = False

        async def _query_one(kb_id: str) -> KBQueryResult:
            try:
                rag = await registry.get_rag(kb_id)
                meta = await registry.get_meta(kb_id)
                result = await rag.aquery_llm(request.query, param=param)
                llm_response = result.get("llm_response", {})
                data = result.get("data", {})
                response_content = llm_response.get("content", "") or "No relevant context found."
                refs = [
                    ReferenceItem(
                        reference_id=r.get("reference_id", ""),
                        file_path=r.get("file_path", ""),
                    )
                    for r in data.get("references", [])
                ]
                return KBQueryResult(
                    kb_id=kb_id,
                    kb_name=meta.name,
                    response=response_content,
                    references=refs,
                )
            except Exception as e:
                logger.error(f"multi_kb_query: error querying KB {kb_id}: {e}", exc_info=True)
                return KBQueryResult(
                    kb_id=kb_id,
                    kb_name=registry._metas.get(kb_id, type("", (), {"name": kb_id})()).name,
                    response=f"Error: {e}",
                    references=[],
                )

        kb_results = await asyncio.gather(*[_query_one(kb_id) for kb_id in request.kbs])

        # Merge references: deduplicate by file_path across all KBs, re-number IDs
        seen_paths: set[str] = set()
        all_refs: list[dict] = []
        for kr in kb_results:
            for ref in kr.references:
                if ref.file_path not in seen_paths:
                    seen_paths.add(ref.file_path)
                    all_refs.append({"reference_id": str(len(all_refs) + 1), "file_path": ref.file_path})

        return MultiKBQueryResponse(responses=list(kb_results), all_references=all_refs)

    @router.post(
        "/query/stream",
        dependencies=[Depends(combined_auth)],
    )
    async def multi_kb_query_stream(request: MultiKBQueryRequest, http_req: Request):
        """Fan out context retrieval across KBs, merge, then stream a single LLM response."""
        if not request.kbs:
            raise HTTPException(status_code=422, detail="'kbs' must contain at least one KB ID")

        unknown = [kb_id for kb_id in request.kbs if kb_id not in registry._metas]
        if unknown:
            raise HTTPException(status_code=404, detail=f"Unknown KB IDs: {unknown}")

        # Build retrieval param (no LLM, no streaming)
        context_param = request.to_query_params(False)
        context_param.stream = False
        # bypass mode has no retrieval — fall back to mix for context gathering
        if context_param.mode == "bypass":
            context_param.mode = "mix"

        multimodal = is_multimodal_enabled()
        base_url = resolve_public_base(http_req) if multimodal else ""

        async def _get_kb_context(
            kb_id: str,
        ) -> tuple[str, str, list[dict], list[dict], list[dict]]:
            try:
                rag = await registry.get_rag(kb_id)
                meta = await registry.get_meta(kb_id)
                result = await rag.aquery_data(request.query, context_param)
                data = result.get("data", {})
                return (
                    kb_id,
                    meta.name,
                    data.get("chunks", []),
                    data.get("entities", []),
                    data.get("relationships", []),
                )
            except Exception as e:
                logger.error(
                    f"multi_kb_query_stream: context retrieval failed for KB {kb_id}: {e}",
                    exc_info=True,
                )
                kb_name = registry._metas[kb_id].name if kb_id in registry._metas else kb_id
                return kb_id, kb_name, [], [], []

        kb_results = await asyncio.gather(*[_get_kb_context(kb_id) for kb_id in request.kbs])

        # Merge context from all KBs: entities + relations + chunks per KB,
        # using the same kg_query_context format as the single-KB pipeline.
        context_parts: list[str] = []
        all_refs: list[dict] = []
        seen_paths: set[str] = set()

        for kb_id, kb_name, chunks, entities, relationships in kb_results:
            if not chunks and not entities and not relationships:
                continue

            # Scope multimodal URLs to this KB's route: /api/kbs/{kb_id}/multimodal/...
            kb_public_base = f"{base_url}/api/kbs/{kb_id}" if base_url else f"/api/kbs/{kb_id}"

            # Render entity and relation context (same JSON-lines format as single-KB)
            entities_str = (
                "\n".join(json.dumps(e, ensure_ascii=False) for e in entities)
                if entities
                else ""
            )
            relations_str = (
                "\n".join(json.dumps(r, ensure_ascii=False) for r in relationships)
                if relationships
                else ""
            )

            # Render chunk context; collect references
            chunk_text_parts: list[str] = []
            for chunk in chunks:
                content = chunk.get("content", "")
                if multimodal and content:
                    content = inject_image_urls(
                        content,
                        doc_stem_from_filepath(chunk.get("file_path", "")),
                        kb_public_base,
                    )
                if content:
                    chunk_text_parts.append(content)
                fp = chunk.get("file_path", "")
                if fp and fp not in seen_paths:
                    seen_paths.add(fp)
                    all_refs.append({"reference_id": str(len(all_refs) + 1), "file_path": fp})

            text_chunks_str = "\n\n".join(chunk_text_parts)

            kb_context = PROMPTS["kg_query_context"].format(
                entities_str=entities_str,
                relations_str=relations_str,
                text_chunks_str=text_chunks_str,
                reference_list_str="",
            )
            context_parts.append(f"## Knowledge Base: {kb_name}\n\n{kb_context}")

        merged_context = (
            "\n\n---\n\n".join(context_parts)
            if context_parts
            else "No relevant context found in the selected knowledge bases."
        )

        system_prompt = PROMPTS["rag_response"].format(
            response_type=request.response_type or "Multiple Paragraphs",
            user_prompt=request.user_prompt or "",
            context_data=merged_context,
        ) + multimodal_answer_hint(merged_context)

        # Single streaming LLM call via bypass mode (skips per-KB retrieval)
        first_rag = await registry.get_rag(request.kbs[0])
        bypass_param = QueryParam(
            mode="bypass",
            stream=True,
            conversation_history=request.conversation_history or [],
        )
        try:
            llm_result = await first_rag.aquery_llm(request.query, bypass_param, system_prompt=system_prompt)
        except Exception as e:
            logger.error(f"multi_kb_query_stream: LLM call failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        start_time = time.perf_counter()
        include_refs = request.include_references is not False
        include_progress = bool(request.include_progress)

        async def _generate():
            llm_response = llm_result.get("llm_response", {})

            if include_refs:
                yield f"{json.dumps({'references': all_refs})}\n"

            if llm_response.get("is_streaming"):
                response_stream = llm_response.get("response_iterator")
                if response_stream:
                    try:
                        async for chunk in response_stream:
                            if chunk:
                                yield f"{json.dumps({'response': chunk})}\n"
                    except Exception as e:
                        logger.error(f"multi_kb_query_stream: streaming error: {e}")
                        yield f"{json.dumps({'error': str(e)})}\n"
            else:
                content = llm_response.get("content") or "No relevant context found."
                yield f"{json.dumps({'response': content})}\n"

            if include_progress:
                token_tracker = llm_result.get("token_tracker")
                metadata: dict = {"response_time": round(time.perf_counter() - start_time, 3)}
                if token_tracker is not None and token_tracker.call_count > 0:
                    usage = token_tracker.get_usage()
                    metadata["input_tokens"] = usage["prompt_tokens"]
                    metadata["output_tokens"] = usage["completion_tokens"]
                yield f"{json.dumps(metadata)}\n"

        return StreamingResponse(_generate(), media_type="application/x-ndjson")

    @router.get(
        "", response_model=list[KBMetaResponse], dependencies=[Depends(combined_auth)]
    )
    async def list_kbs():
        return [_to_response(m) for m in await registry.list()]

    @router.post(
        "", response_model=KBMetaResponse, dependencies=[Depends(combined_auth)]
    )
    async def create_kb(request: CreateKBRequest):
        try:
            meta = await registry.create(
                name=request.name,
                description=request.description,
                default_chunking=(
                    request.default_chunking.model_dump()
                    if request.default_chunking
                    else None
                ),
            )
        except KBDuplicateNameError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return _to_response(meta)

    @router.get(
        "/{kb_id}",
        response_model=KBMetaResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_kb(kb_id: str):
        try:
            meta = await registry.get_meta(kb_id)
        except KBNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return _to_response(meta)

    @router.patch(
        "/{kb_id}",
        response_model=KBMetaResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def update_kb(kb_id: str, request: UpdateKBRequest):
        try:
            meta = await registry.update(
                kb_id,
                name=request.name,
                description=request.description,
                default_chunking=(
                    request.default_chunking.model_dump()
                    if request.default_chunking
                    else None
                ),
                clear_default_chunking=request.clear_default_chunking,
            )
        except KBNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except KBDuplicateNameError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return _to_response(meta)

    @router.delete("/{kb_id}", dependencies=[Depends(combined_auth)])
    async def delete_kb(kb_id: str):
        try:
            await registry.delete(kb_id)
        except KBNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return {"status": "success"}

    return router
