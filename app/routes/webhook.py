from fastapi import APIRouter, Depends, Response
from app.utils.auth import verify_api_key

router = APIRouter()

@router.api_route(
    "/honeypot",
    methods=["GET", "POST", "HEAD", "OPTIONS"],
)
async def honeypot_endpoint(
    auth=Depends(verify_api_key)
):
    return {
        "status": "ok",
        "message": "Honeypot API is live and secured"
    }
