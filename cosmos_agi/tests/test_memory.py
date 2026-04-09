"""Tests for vector memory store."""

from cosmos_agi.config.settings import MemoryConfig
from cosmos_agi.memory.vector_store import VectorMemory


def test_vector_memory_add_and_query():
    config = MemoryConfig(
        persist_directory="/tmp/cosmos_agi_test_memory",
        collection_name="test_collection",
    )
    mem = VectorMemory(config)
    mem.clear()

    mem.add("The sky is blue", metadata={"source": "fact"})
    mem.add("Python is a programming language", metadata={"source": "fact"})
    mem.add("Water boils at 100 degrees Celsius", metadata={"source": "fact"})

    assert mem.count() == 3

    results = mem.query("What color is the sky?", n_results=1)
    assert len(results) == 1
    assert "sky" in results[0]["text"].lower()

    mem.clear()
    assert mem.count() == 0
