from typing import Optional
from pydantic import BaseModel

class ScamRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: Optional[str] = None
    turn: Optional[int] = None
