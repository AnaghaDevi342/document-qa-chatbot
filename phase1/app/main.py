from fastapi import FastAPI
from .routes import router

app = FastAPI(
    title="Document Q&A Chatbot",
    description="A chatbot that answers questions based on uploaded documents.",
    version="1.0.0",
    )

app.include_router(router)