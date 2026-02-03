# Simple in-memory conversation store

conversation_memory = {}

def add_message(conversation_id: str, message: str):
    if conversation_id not in conversation_memory:
        conversation_memory[conversation_id] = []
    conversation_memory[conversation_id].append(message)

def get_conversation(conversation_id: str):
    return conversation_memory.get(conversation_id, [])
