"""INU AI chat endpoints (screen 7).

Conversations persist so the history sidebar survives a reload and a deployed
instance can show a returning visitor what they asked last time. Attachments
are passed to the model in-request rather than stored: an image only needs to
exist for the turn that asks about it, and keeping uploads out of the database
avoids a storage problem this project does not otherwise have.
"""

from __future__ import annotations

import base64

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select

from quantedge.api.deps import require_api_key
from quantedge.config import settings
from quantedge.db.models import ChatMessage, Conversation
from quantedge.db.session import session_scope
from quantedge.inu import tools
from quantedge.inu.agent import chat as run_chat
from quantedge.inu.models import ALL_MODELS
from quantedge.logging_config import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/inu", tags=["inu"])

#: Beyond this an image is likely a photo rather than a chart, and the free
#: vision models start refusing or truncating.
MAX_IMAGE_BYTES = 4 * 1024 * 1024
#: Documents are read as text and prepended to the question, so the ceiling is
#: about context rather than storage.
MAX_DOC_BYTES = 512 * 1024


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: int | None = None


def _title_from(message: str) -> str:
    """A thread title the sidebar can show, taken from the opening question."""
    cleaned = " ".join(message.split())
    return cleaned[:80] + ("…" if len(cleaned) > 80 else "")


def _history(conversation_id: int) -> list[dict]:
    """Recent turns, oldest first, trimmed to what the context can hold."""
    with session_scope() as s:
        rows = (
            s.scalars(
                select(ChatMessage)
                .where(ChatMessage.conversation_id == conversation_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(settings.inu_history_turns)
            )
            .all()
        )
    return [{"role": r.role, "content": r.content} for r in reversed(rows)]


def _persist(
    conversation_id: int | None,
    message: str,
    turn,
    attachment: str | None = None,
) -> int:
    """Store the exchange, creating the conversation on the first message."""
    with session_scope() as s:
        if conversation_id is None:
            convo = Conversation(title=_title_from(message))
            s.add(convo)
            s.flush()
            conversation_id = convo.id
        else:
            convo = s.get(Conversation, conversation_id)
            if convo is None:
                raise HTTPException(404, f"Conversation {conversation_id} not found.")
            # Touch it so the sidebar orders by genuine recency.
            convo.updated_at = func.now()

        s.add(
            ChatMessage(
                conversation_id=conversation_id,
                role="user",
                content=message,
                attachment_name=attachment,
            )
        )
        s.add(
            ChatMessage(
                conversation_id=conversation_id,
                role="assistant",
                content=turn.content,
                model=turn.model,
                provider=turn.provider,
                tools_used={"tools": turn.tools_used},
                latency_ms=turn.latency_ms,
            )
        )
    return conversation_id


@router.get("/status", dependencies=[Depends(require_api_key)])
def status() -> dict:
    """What INU can do, and on which free models."""
    return {
        "models": [
            {
                "id": m.id,
                "provider": m.provider,
                "context": m.context,
                "tpm": m.tpm,
                "rpd": m.rpd,
                "notes": m.notes,
                "good_for": [t.value for t in m.good_for],
            }
            for m in ALL_MODELS
        ],
        "tools": [
            {
                "name": name,
                "description": schema["function"]["description"],
            }
            for name, (_, schema) in tools.TOOLS.items()
        ],
        "capabilities": {
            "web_search": "groq/compound runs search inside the model",
            "vision": "open-weight Gemma 4 and Nemotron VL read images",
            "documents": "text and markdown are read into the question",
        },
        "note": (
            "Every model here is free and open-weight. Numbers in answers come "
            "from platform tools, not from model memory."
        ),
    }


@router.post("/chat", dependencies=[Depends(require_api_key)])
def send(req: ChatRequest) -> dict:
    """Ask INU a question."""
    history = _history(req.conversation_id) if req.conversation_id else []
    turn = run_chat(req.message, history)
    conversation_id = _persist(req.conversation_id, req.message, turn)

    return {
        "conversation_id": conversation_id,
        "content": turn.content,
        "model": turn.model,
        "provider": turn.provider,
        "tools_used": turn.tools_used,
        "latency_ms": turn.latency_ms,
        "fell_back": turn.fell_back,
    }


@router.post("/chat/upload", dependencies=[Depends(require_api_key)])
async def send_with_file(
    message: str = Form(default=""),
    conversation_id: int | None = Form(default=None),
    file: UploadFile = File(...),  # noqa: B008 - required FastAPI idiom
) -> dict:
    """Ask INU about an uploaded image or document.

    Images become a data URL the vision models read directly. Text documents
    are decoded and prepended to the question, which is why the size ceiling
    differs between the two.
    """
    raw = await file.read()
    content_type = (file.content_type or "").lower()
    is_image = content_type.startswith("image/")

    limit = MAX_IMAGE_BYTES if is_image else MAX_DOC_BYTES
    if len(raw) > limit:
        raise HTTPException(
            413,
            f"{'Image' if is_image else 'Document'} is {len(raw) // 1024}KB; "
            f"the limit is {limit // 1024}KB.",
        )

    image_data_url = None
    prompt = message

    if is_image:
        encoded = base64.b64encode(raw).decode()
        image_data_url = f"data:{content_type};base64,{encoded}"
        prompt = message or "What do you make of this?"
    else:
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            raise HTTPException(
                415,
                "That file isn't readable as text. Images, .txt, .md, .csv and "
                ".json work.",
            ) from None
        prompt = (
            f"Document `{file.filename}`:\n\n{text}\n\n"
            f"{message or 'Summarise this and tell me what stands out.'}"
        )

    history = _history(conversation_id) if conversation_id else []
    turn = run_chat(prompt, history, image_data_url=image_data_url)
    convo_id = _persist(
        conversation_id,
        message or f"[{file.filename}]",
        turn,
        attachment=file.filename,
    )

    return {
        "conversation_id": convo_id,
        "content": turn.content,
        "model": turn.model,
        "provider": turn.provider,
        "tools_used": turn.tools_used,
        "latency_ms": turn.latency_ms,
        "fell_back": turn.fell_back,
        "attachment": file.filename,
    }


@router.get("/conversations", dependencies=[Depends(require_api_key)])
def conversations(limit: int = Query(default=50, le=200)) -> dict:
    """Thread list for the history sidebar, most recent first."""
    with session_scope() as s:
        rows = s.execute(
            select(
                Conversation.id,
                Conversation.title,
                Conversation.created_at,
                Conversation.updated_at,
                func.count(ChatMessage.id).label("n_messages"),
            )
            .outerjoin(ChatMessage, ChatMessage.conversation_id == Conversation.id)
            .group_by(Conversation.id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
        ).all()

    return {
        "conversations": [
            {
                "id": r.id,
                "title": r.title,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                "n_messages": r.n_messages,
            }
            for r in rows
        ]
    }


@router.get("/conversations/{conversation_id}", dependencies=[Depends(require_api_key)])
def conversation(conversation_id: int) -> dict:
    """Every turn in one thread, for replaying it in the UI."""
    with session_scope() as s:
        convo = s.get(Conversation, conversation_id)
        if convo is None:
            raise HTTPException(404, f"Conversation {conversation_id} not found.")
        rows = s.scalars(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation_id)
            .order_by(ChatMessage.created_at)
        ).all()

        return {
            "id": convo.id,
            "title": convo.title,
            "messages": [
                {
                    "id": r.id,
                    "role": r.role,
                    "content": r.content,
                    "model": r.model,
                    "provider": r.provider,
                    "tools_used": (r.tools_used or {}).get("tools", []),
                    "latency_ms": r.latency_ms,
                    "attachment_name": r.attachment_name,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ],
        }


@router.delete("/conversations/{conversation_id}", dependencies=[Depends(require_api_key)])
def remove(conversation_id: int) -> dict:
    with session_scope() as s:
        if s.get(Conversation, conversation_id) is None:
            raise HTTPException(404, f"Conversation {conversation_id} not found.")
        s.execute(
            delete(ChatMessage).where(ChatMessage.conversation_id == conversation_id)
        )
        s.execute(delete(Conversation).where(Conversation.id == conversation_id))
    return {"deleted": conversation_id}
