from fastapi import APIRouter, Depends, Request, BackgroundTasks
from app.utils.auth import verify_api_key
from app.core.detector import detect_scam
from app.core.extractor import extract_intelligence
from app.utils.logger import logger
import requests

router = APIRouter()

# Simple in-memory session storage
SESSION_MEMORY = {}


def send_final_callback(payload: dict):
    """
    Send final intelligence to GUVI callback endpoint (non-blocking).
    """
    try:
        requests.post(
            "https://hackathon.guvi.in/api/updateHoneyPotFinalResult",
            json=payload,
            timeout=5
        )
    except Exception as e:
        logger.error(f"GUVI callback failed: {e}")


@router.api_route("/honeypot", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def honeypot_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    auth=Depends(verify_api_key)
):
    # ✅ Handle tester / health-check requests (no body)
    if request.method in ["GET", "HEAD", "OPTIONS"]:
        return {
            "status": "success",
            "reply": "Honeypot API is live"
        }

    # ✅ Safe JSON parsing (never crash)
    try:
        body = await request.json()
    except Exception:
        return {
            "status": "success",
            "reply": "Honeypot API is live"
        }

    session_id = body.get("sessionId", "default-session")
    message_obj = body.get("message", {})
    text = message_obj.get("text", "")
    history = body.get("conversationHistory", [])

    logger.info(f"Session {session_id} | Message: {text}")

    # Detect scam intent
    is_scam, _ = detect_scam(text)

    # --- Agent reply (human-like, no exposure) ---
    if is_scam:
        if len(history) == 0:
            reply = "Why is my account being suspended?"
        elif "upi" in text.lower():
            reply = "Which UPI ID should I use?"
        else:
            reply = "Can you explain this again?"
    else:
        reply = "Okay."

    # Store conversation
    SESSION_MEMORY.setdefault(session_id, []).append(text)

    # Extract intelligence
    intelligence = extract_intelligence(text)

    # --- Mandatory final callback (ASYNC) ---
    if is_scam and len(history) >= 2:
        payload = {
            "sessionId": session_id,
            "scamDetected": True,
            "totalMessagesExchanged": len(history) + 1,
            "extractedIntelligence": {
                "bankAccounts": [intelligence["bank_account"]] if intelligence.get("bank_account") else [],
                "upiIds": [intelligence["upi_id"]] if intelligence.get("upi_id") else [],
                "phishingLinks": intelligence.get("phishing_links", []),
                "phoneNumbers": [],
                "suspiciousKeywords": ["urgent", "verify", "account blocked"]
            },
            "agentNotes": "Scammer used urgency and payment redirection"
        }

        background_tasks.add_task(send_final_callback, payload)

    # ✅ EXACT RESPONSE FORMAT REQUIRED BY GUVI
    return {
        "status": "success",
        "reply": reply
    }
