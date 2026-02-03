# Simple scam detection using keywords

SCAM_KEYWORDS = [
    "win", "won", "prize", "urgent", "click",
    "reward", "upi", "bank", "lottery", "offer"
]

def detect_scam(message: str):
    message_lower = message.lower()
    score = 0

    for word in SCAM_KEYWORDS:
        if word in message_lower:
            score += 1

    is_scam = score >= 2
    confidence = min(score / len(SCAM_KEYWORDS), 1.0)

    return is_scam, round(confidence, 2)
