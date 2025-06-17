from pydantic import BaseModel


class VectorDbActionRequest(BaseModel):
    db_id: str
    force_download: bool = False
