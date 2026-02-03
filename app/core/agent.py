# Agent logic for honeypot engagement
# For final submission, this is lightweight and deterministic

def agent_reply(is_scam: bool, conversation_turns: int):
    """
    Generate a simple agent response strategy.
    This is NOT returned to scammer directly,
    but shows agentic decision-making.
    """

    if not is_scam:
        return "Normal conversation"

    # Scam detected → agent continues engagement
    if conversation_turns < 2:
        return "Ask for more details politely"
    elif conversation_turns < 4:
        return "Show interest but confusion"
    else:
        return "Delay and collect info"
