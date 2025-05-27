from typing import Optional

from pydantic import BaseModel


class ModelActionRequest(BaseModel):
    model_id: str
    force_download: bool = False


class LlmConfigResponse(BaseModel):
    backend: str
    model_id: Optional[str]  # May not be set if only path is used
    model_path: Optional[str]
    max_model_len: int
    gpu_count: int
