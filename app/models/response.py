from typing import Optional, List
from pydantic import BaseModel

class ExtractedIntelligence(BaseModel):
    upi_id: Optional[str]
    bank_account: Optional[str]
    phishing_links: List[str]

class ScamResponse(BaseModel):
    is_scam: bool
    confidence: float
    conversation_turns: int
    extracted_intelligence: ExtractedIntelligence
