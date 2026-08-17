"""Qdrant vector store.

Two changes of substance versus the old Chroma setup:

1.  **One collection, filtered by ``doc_id``** — instead of a collection per
    PDF named after a sanitised filename. That old scheme meant two uploads
    whose names sanitised to the same string silently merged, and a question
    could only ever be answered from one document.
2.  **Named dense + sparse vectors on every point**, so Qdrant can run BM25
    and vector search in one round trip and fuse them server-side with RRF.
    The sparse vector carries the IDF modifier, which lets Qdrant compute
    corpus-wide IDF rather than fastembed guessing it per batch.
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from rag.config import Settings, get_settings
from rag.logging import get_logger
from rag.schemas import Chunk, SearchResult

log = get_logger(__name__)

DENSE = "dense"
SPARSE = "sparse"


class CollectionDimensionMismatchError(RuntimeError):
    """Raised when the live collection was built with a different embedding model."""


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._collection = settings.qdrant_collection
        self._client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=(
                settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            ),
            timeout=30,
        )

    @property
    def client(self) -> AsyncQdrantClient:
        return self._client

    async def close(self) -> None:
        await self._client.close()

    # --- schema ----------------------------------------------------------

    async def ensure_collection(self, dim: int) -> None:
        """Create the collection and its payload index if absent; verify dim if present."""
        if await self._client.collection_exists(self._collection):
            info = await self._client.get_collection(self._collection)
            existing = self._dense_size(info)
            if existing is not None and existing != dim:
                raise CollectionDimensionMismatchError(
                    f"Collection {self._collection!r} stores {existing}-dim vectors but the "
                    f"configured embedding model produces {dim}-dim. Either restore the "
                    f"previous DENSE_MODEL or drop the collection and re-ingest."
                )
            return

        log.info("creating_collection", collection=self._collection, dim=dim)
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config={DENSE: qm.VectorParams(size=dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )
        # Without this index, every filtered search is a full scan.
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="doc_id",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )

    @staticmethod
    def _dense_size(info: qm.CollectionInfo) -> int | None:
        params = info.config.params.vectors
        if isinstance(params, dict):
            named = params.get(DENSE)
            return named.size if named is not None else None
        return params.size if params is not None else None

    async def health(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception as exc:
            log.warning("qdrant_unreachable", error=str(exc))
            return False

    # --- writes ----------------------------------------------------------

    async def upsert(
        self,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[qm.SparseVector],
    ) -> None:
        if not chunks:
            return
        if not len(chunks) == len(dense) == len(sparse):
            raise ValueError("chunks, dense and sparse must be the same length")

        points = [
            qm.PointStruct(
                id=chunk.id,
                vector={DENSE: dense_vec, SPARSE: sparse_vec},
                payload=chunk.to_payload(),
            )
            for chunk, dense_vec, sparse_vec in zip(chunks, dense, sparse, strict=True)
        ]
        await self._client.upsert(collection_name=self._collection, points=points, wait=True)
        log.debug("upserted_chunks", count=len(points), doc_id=chunks[0].doc_id)

    async def delete_document(self, doc_id: str) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(filter=self._doc_filter([doc_id])),
            wait=True,
        )
        log.info("deleted_document_vectors", doc_id=doc_id)

    async def count(self, doc_id: str | None = None) -> int:
        result = await self._client.count(
            collection_name=self._collection,
            count_filter=self._doc_filter([doc_id]) if doc_id else None,
            exact=True,
        )
        return result.count

    # --- reads -----------------------------------------------------------

    @staticmethod
    def _doc_filter(doc_ids: list[str]) -> qm.Filter:
        return qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids))])

    async def hybrid_search(
        self,
        dense_query: list[float],
        sparse_query: qm.SparseVector,
        limit: int,
        doc_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        """Dense + BM25 search fused with reciprocal rank fusion, in one round trip."""
        flt = self._doc_filter(doc_ids) if doc_ids else None
        # Over-fetch per branch so fusion has room to promote passages that only
        # one of the two retrievers found.
        branch_limit = limit * 2

        response = await self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                qm.Prefetch(query=dense_query, using=DENSE, limit=branch_limit, filter=flt),
                qm.Prefetch(query=sparse_query, using=SPARSE, limit=branch_limit, filter=flt),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for point in response.points:
            if not point.payload:
                continue
            results.append(
                SearchResult(
                    chunk=Chunk(id=str(point.id), **point.payload),
                    score=point.score,
                )
            )
        return results


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(get_settings())
    return _store


async def reset_store() -> None:
    """Drop the cached client. Used by tests and by the worker on shutdown."""
    global _store
    if _store is not None:
        await _store.close()
        _store = None
