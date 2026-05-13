"""
app.py — FastAPI backend for the Neural Machine Translator.

Endpoints:
    POST /api/translate       Translate text between English and French.
    GET  /api/status          List which models are available.
    GET  /api/health          Liveness probe.

Static frontend is served from ../frontend/ at the root path.

Run:
    uvicorn backend.app:app --reload
"""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from backend.translator import TranslationEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
FRONTEND_DIR = ROOT_DIR / "frontend"

# ── Engine (singleton) ────────────────────────────────────────────────────────
engine = TranslationEngine(MODELS_DIR)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Neural Machine Translator",
    description="English ↔ French translation powered by GRU seq2seq and Transformer models.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class TranslateRequest(BaseModel):
    text: str
    source_lang: str  # "en" or "fr"
    target_lang: str  # "en" or "fr"
    model: str = "transformer"  # "rnn" or "transformer"

    @field_validator("source_lang", "target_lang")
    @classmethod
    def validate_lang(cls, v: str) -> str:
        if v not in ("en", "fr"):
            raise ValueError("lang must be 'en' or 'fr'")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in ("rnn", "transformer"):
            raise ValueError("model must be 'rnn' or 'transformer'")
        return v

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        if len(v) > 500:
            raise ValueError("text must be ≤ 500 characters")
        return v


class TranslateResponse(BaseModel):
    translation: str
    model: str
    direction: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/status")
async def status():
    return JSONResponse(engine.status())


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    if req.source_lang == req.target_lang:
        return TranslateResponse(
            translation=req.text,
            model=req.model,
            direction=f"{req.source_lang}→{req.target_lang}",
        )

    direction = f"{req.source_lang}_{req.target_lang}"
    if not engine.model_exists(direction, req.model):  # type: ignore[arg-type]
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model '{req.model}' for direction '{direction}' is not available. "
                "Please run `python -m backend.train` to train the models."
            ),
        )

    try:
        translation = engine.translate(
            text=req.text,
            source_lang=req.source_lang,  # type: ignore[arg-type]
            target_lang=req.target_lang,  # type: ignore[arg-type]
            model_type=req.model,  # type: ignore[arg-type]
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Translation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Translation failed.") from exc

    return TranslateResponse(
        translation=translation,
        model=req.model,
        direction=f"{req.source_lang}→{req.target_lang}",
    )


# ── Serve frontend ────────────────────────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
