"""API routers."""

from fastapi import APIRouter

from rag.api.routes import audio, chat, documents, health, search

router = APIRouter(prefix="/api")
router.include_router(health.router)
router.include_router(documents.router)
router.include_router(chat.router)
router.include_router(search.router)
router.include_router(audio.router)

__all__ = ["router"]
