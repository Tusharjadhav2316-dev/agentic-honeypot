from fastapi import APIRouter, Depends, Request
from app.utils.auth import verify_api_key
from app.core.detector import detect_scam
from app.core.memory import add_message, get_conversation
from app.core.extractor import extract_intelligence
from app.core.agent import agent_reply
from app.utils.logger import logger
from app.models.response import ScamResponse, ExtractedIntelligence

router = APIRouter()

@router.api_route("/honeypot", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def honeypot_endpoint(
    request: Request,
    auth=Depends(verify_api_key)
):
    # Safely read body (GUVI may send empty body)
    try:
        body = await request.json()
    except Exception:
        body = {}

    conversation_id = body.get("conversation_id", "default")
    message = body.get("message", "")

    if message:
        add_message(conversation_id, message)
        logger.info(f"Message received | conversation_id={conversation_id}")

    history = get_conversation(conversation_id)

    # Scam detection
    is_scam, confidence = detect_scam(message)

    # Agent decision (internal, not exposed)
    agent_state = agent_reply(is_scam, len(history))
    logger.info(f"Agent state: {agent_state}")

    # Intelligence extraction
    intelligence_data = extract_intelligence(message)

    intelligence = ExtractedIntelligence(
        upi_id=intelligence_data.get("upi_id"),
        bank_account=intelligence_data.get("bank_account"),
        phishing_links=intelligence_data.get("phishing_links", [])
    )

    response = ScamResponse(
        is_scam=is_scam,
        confidence=confidence,
        conversation_turns=len(history),
        extracted_intelligence=intelligence
    )

    return response.dict()
