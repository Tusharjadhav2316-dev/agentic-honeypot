from fastapi import APIRouter, Depends, Body
from app.utils.auth import verify_api_key
from typing import Optional, Dict

router = APIRouter()

@router.post("/honeypot")
def honeypot_endpoint(
    payload: Optional[Dict] = Body(default={}),
    auth=Depends(verify_api_key)
):
    return {
        "status": "ok",
        "message": "Honeypot API is live and secured",
        "received_payload": payload
    }
