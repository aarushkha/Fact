"""HTTP API: JSON and server-sent-event endpoints, plus the static test page."""

from __future__ import annotations

import json
from datetime import datetime
from typing import AsyncIterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.api.security import authenticate, guard_check
from app.config import ROOT_DIR
from app.models.schemas import CheckInput, CheckResponse
from app.pipeline.orchestrator import Pipeline

router = APIRouter()
INDEX_HTML = ROOT_DIR / "web" / "index.html"


def _pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


async def _check_input(
    request: Request, text: str | None, post_date: str | None, image: UploadFile | None
) -> CheckInput:
    parsed_date = None
    if post_date and post_date.strip():
        try:
            parsed_date = datetime.fromisoformat(post_date.strip())
        except ValueError:
            raise HTTPException(422, "post_date must be ISO 8601, e.g. 2026-10-01 or 2026-10-01T14:30")
    data = None
    mime = None
    if image is not None and image.filename:
        mime = image.content_type or ""
        if not mime.startswith("image/"):
            raise HTTPException(415, "Screenshot must be an image.")
        limit = request.app.state.settings.max_upload_mb * 1024 * 1024
        data = await image.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, "Screenshot is too large.")
    if not data and not (text and text.strip()):
        raise HTTPException(422, "Provide a screenshot or some text.")
    return CheckInput(text=text, image=data or None, image_mime=mime, post_date=parsed_date)


@router.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@router.get("/api/health")
async def health(request: Request) -> dict:
    p = _pipeline(request)
    return {
        "ok": True,
        "mock_mode": request.app.state.settings.mock_mode,
        "auth_required": bool(request.app.state.api_keys),
        "model_versions": p.a.model_versions(),
    }


@router.post("/api/check", response_model=CheckResponse, dependencies=[Depends(guard_check)])
async def check(
    request: Request,
    text: str | None = Form(None),
    post_date: str | None = Form(None),
    image: UploadFile | None = File(None),
) -> CheckResponse:
    inp = await _check_input(request, text, post_date, image)
    try:
        return await _pipeline(request).run(inp)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))


@router.post("/api/check/stream", dependencies=[Depends(guard_check)])
async def check_stream(
    request: Request,
    text: str | None = Form(None),
    post_date: str | None = Form(None),
    image: UploadFile | None = File(None),
) -> StreamingResponse:
    inp = await _check_input(request, text, post_date, image)

    async def events() -> AsyncIterator[str]:
        async for ev in _pipeline(request).stream(inp):
            yield f"event: {ev.name}\ndata: {json.dumps(ev.data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@router.get("/api/checks/{check_id}", response_model=CheckResponse, dependencies=[Depends(authenticate)])
async def get_check(request: Request, check_id: str) -> CheckResponse:
    found = await _pipeline(request).store.get_check(check_id)
    if found is None:
        raise HTTPException(404, "Unknown check_id")
    return found
