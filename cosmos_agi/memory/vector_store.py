"""Vector memory store backed by ChromaDB."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from cosmos_agi.config.settings import MemoryConfig

logger = logging.getLogger(__name__)


class VectorMemory:
    """Persistent vector memory using ChromaDB."""

    def __init__(self, config: MemoryConfig | None = None):
        if config is None:
            from cosmos_agi.config.settings import settings
            config = settings.memory

        self._config = config
        self._client = chromadb.Client(ChromaSettings(
            anonymized_telemetry=False,
            persist_directory=config.persist_directory,
        ))
        self._collection = self._client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorMemory initialized: collection=%s, count=%d",
            config.collection_name,
            self._collection.count(),
        )

    def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Store a text in the vector memory. Returns the document ID."""
        doc_id = doc_id or str(uuid.uuid4())
        self._collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id],
        )
        logger.debug("Stored memory: id=%s, len=%d", doc_id, len(text))
        return doc_id

    def query(
        self,
        text: str,
        n_results: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query similar memories. Returns list of {id, text, metadata, distance}."""
        n = n_results or self._config.max_results
        params: dict[str, Any] = {
            "query_texts": [text],
            "n_results": min(n, self._collection.count() or 1),
        }
        if where:
            params["where"] = where

        if self._collection.count() == 0:
            return []

        results = self._collection.query(**params)

        memories = []
        for i in range(len(results["ids"][0])):
            memories.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })
        return memories

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        """Delete all memories."""
        ids = self._collection.get()["ids"]
        if ids:
            self._collection.delete(ids=ids)
        logger.info("Cleared all memories")
