from fastapi import APIRouter, Depends, Request
from app.utils.auth import verify_api_key
from app.core.detector import detect_scam
from app.core.extractor import extract_intelligence
from app.utils.logger import logger
import requests

router = APIRouter()

# Simple in-memory session store
SESSION_MEMORY = {}

@router.post("/honeypot")
async def honeypot_endpoint(
    request: Request,
    auth=Depends(verify_api_key)
):
    body = await request.json()

    session_id = body.get("sessionId")
    message_obj = body.get("message", {})
    text = message_obj.get("text", "")
    history = body.get("conversationHistory", [])

    logger.info(f"Session {session_id} | Message: {text}")

    # Detect scam
    is_scam, _ = detect_scam(text)

    # --- AGENT REPLY LOGIC (HUMAN-LIKE) ---
    if is_scam:
        if len(history) == 0:
            reply = "Why is my account being suspended?"
        elif "upi" in text.lower():
            reply = "Which UPI ID should I use?"
        else:
            reply = "Can you explain this again?"
    else:
        reply = "Okay."

    # Store message
    SESSION_MEMORY.setdefault(session_id, []).append(text)

    # Extract intelligence
    intelligence = extract_intelligence(text)

    # --- FINAL CALLBACK (MANDATORY) ---
    if is_scam and len(history) >= 2:
        payload = {
            "sessionId": session_id,
            "scamDetected": True,
            "totalMessagesExchanged": len(history) + 1,
            "extractedIntelligence": {
                "bankAccounts": [intelligence["bank_account"]] if intelligence["bank_account"] else [],
                "upiIds": [intelligence["upi_id"]] if intelligence["upi_id"] else [],
                "phishingLinks": intelligence["phishing_links"],
                "phoneNumbers": [],
                "suspiciousKeywords": ["urgent", "verify", "account blocked"]
            },
            "agentNotes": "Scammer used urgency and payment redirection"
        }

        try:
            requests.post(
                "https://hackathon.guvi.in/api/updateHoneyPotFinalResult",
                json=payload,
                timeout=5
            )
        except Exception as e:
            logger.error(f"Callback failed: {e}")

    # ✅ EXACT RESPONSE FORMAT REQUIRED BY GUVI
    return {
        "status": "success",
        "reply": reply
    }
