import logging
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models   import ChatRequest, ChatResponse
from app.services import call_openrouter
from app.routers  import voice as voice_router
from app.routers  import video_proxy as video_proxy_router

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%H:%M:%S",
)

app = FastAPI(title="Humsafar API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(voice_router.router)
app.include_router(video_proxy_router.router)

# Replies longer than this word count get flagged so the frontend
# can offer the "Watch as Video" button.
VIDEO_THRESHOLD_WORDS = 60


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    system_prompt = f"""
    You are HUMSAFAR, an intelligent heritage guide.
    User is currently at: {req.site_name}
    Instructions:
    - If question is about the monument, answer in detail.
    - Suggest important areas to explore.
    - If question is general, answer normally.
    - Keep responses engaging and clear.
    - Do not hallucinate unknown facts.
    """
    messages  = [{"role": "system", "content": system_prompt}]
    messages += [{"role": m.role, "content": m.content} for m in req.history]
    messages.append({"role": "user", "content": req.message})

    reply = await call_openrouter(messages)

    # Tell the frontend whether this reply is long enough to offer a video
    word_count    = len(reply.split())
    suggest_video = word_count >= VIDEO_THRESHOLD_WORDS

    return ChatResponse(reply=reply, suggest_video=suggest_video)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "humsafar-backend"}