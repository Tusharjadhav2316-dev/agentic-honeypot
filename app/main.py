from fastapi import FastAPI
from app.routes.webhook import router

app = FastAPI(
    title="Agentic Honeypot API",
    version="1.0"
)

app.include_router(router)
