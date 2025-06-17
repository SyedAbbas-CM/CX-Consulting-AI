from pydantic import BaseModel, Field


class LlmBackendUpdateRequest(BaseModel):
    backend: str = Field(
        ..., description="Target LLM backend e.g. llama.cpp, azure, ollama"
    )
    # Optional suggested model id or path (backend-specific)
    model_id: str | None = Field(
        None, description="Optional model identifier for the backend"
    )
    model_path: str | None = Field(
        None, description="Optional local model path if backend is llama.cpp"
    )
