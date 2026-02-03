from fastapi import APIRouter, Depends
from app.utils.auth import verify_api_key

router = APIRouter()

@router.api_route(
    "/honeypot",
    methods=["GET", "POST", "HEAD", "OPTIONS"],
)
async def honeypot_endpoint(auth=Depends(verify_api_key)):
    return {
        "is_scam": False,
        "status": "ok",
        "data": {}
    }
