# app/models.py
# Shared Pydantic models for the Humsafar API.

from pydantic import BaseModel
from typing import List


class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []
    site_name: str = ""
    site_id: str = ""


class ChatResponse(BaseModel):
    reply: str
    suggest_video: bool = False  # True when reply is long → frontend shows "Watch as Video"