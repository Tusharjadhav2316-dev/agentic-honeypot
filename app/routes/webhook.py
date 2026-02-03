from fastapi import APIRouter, Depends
from app.utils.auth import verify_api_key

router = APIRouter()

@router.post("/honeypot")
def honeypot_endpoint(payload: dict, auth=Depends(verify_api_key)):
    return {
        "status": "ok",
        "message": "Honeypot API is live and secured",
        "received_payload": payload
    }
