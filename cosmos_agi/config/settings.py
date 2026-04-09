"""Global configuration for CosmosAGI."""

from enum import Enum
from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    LITELLM = "litellm"


class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.LITELLM
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: str | None = None
    base_url: str | None = None


class MemoryConfig(BaseModel):
    vector_store: str = "chroma"
    collection_name: str = "cosmos_agi_memory"
    persist_directory: str = "./data/memory"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_results: int = 10


class SafetyConfig(BaseModel):
    enable_human_in_the_loop: bool = True
    max_iterations: int = 20
    max_tool_calls_per_step: int = 5
    blocked_actions: list[str] = Field(default_factory=lambda: [
        "rm -rf /",
        "sudo rm",
        "DROP TABLE",
        "FORMAT",
    ])


class Settings(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    log_level: str = "INFO"
    log_file: str = "./data/logs/cosmos_agi.log"


settings = Settings()
