from fastapi import APIRouter, Depends, Request
from app.utils.auth import verify_api_key

router = APIRouter()

@router.post("/honeypot")
async def honeypot_endpoint(
    request: Request,
    auth=Depends(verify_api_key)
):
    try:
        payload = await request.json()
    except:
        payload = {}

    return {
        "status": "ok",
        "message": "Honeypot API is live and secured",
        "received_payload": payload
    }
