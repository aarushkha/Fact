"""The settings page (/settings) and its API. Guarded by `require_admin`; secrets are never returned."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.api.security import require_admin
from app.config import ROOT_DIR
from app.runtime_settings import SettingsInvalid, check_readiness, describe

router = APIRouter()


class SettingsUpdate(BaseModel):
    values: dict[str, bool | int | float | str] = Field(default_factory=dict)  # field -> new value
    clear: list[str] = Field(default_factory=list)  # fields to revert to the environment / default value


def _state(request: Request) -> dict:
    rt = request.app.state.runtime
    s = request.app.state.settings
    if s.admin_token:
        auth = "admin_token"
    elif request.app.state.api_keys:
        auth = "api_key"
    else:
        auth = "open"
    ready = check_readiness(s)
    return {
        "mock_mode": s.mock_mode,
        "auth": auth,  # who may open this page: the admin token, any API key, or anyone
        "persistence": request.app.state.overrides_kind,  # where changes survive a restart: database | file | memory
        "readiness": {"errors": ready.errors, "warnings": ready.warnings},
        "groups": describe(rt.base, s, rt.overrides),
    }


@router.get("/settings", include_in_schema=False)
async def settings_page() -> FileResponse:
    return FileResponse(ROOT_DIR / "web" / "settings.html")


@router.get("/api/settings", dependencies=[Depends(require_admin)])
async def get_settings_state(request: Request) -> dict:
    return _state(request)


@router.put("/api/settings", dependencies=[Depends(require_admin)])
async def update_settings(request: Request, body: SettingsUpdate) -> dict:
    try:
        await request.app.state.runtime.update(body.values, body.clear)
    except SettingsInvalid as exc:
        raise HTTPException(422, {"message": str(exc), "errors": exc.errors})
    return _state(request)
