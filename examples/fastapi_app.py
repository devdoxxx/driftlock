"""
What:     FastAPI integration (middleware tagging, optimization, cache) plus the
          single-page mission dashboard + JSON data API served at /.
Requires: pip install "driftlock[fastapi]". OPENAI_API_KEY only for the /chat
          and /summarise routes; the dashboard itself reads existing telemetry.
Run:      uvicorn examples.fastapi_app:app --reload   # then open http://localhost:8000

Tip: run `python examples/agent_demo.py "any topic"` first (mock mode, no key)
so the dashboard has mission data to show.
"""

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

import driftlock
from driftlock import CacheConfig, DriftlockClient, DriftlockConfig, OptimizationConfig

# ---------------------------------------------------------------------------
# Initialise the client once at startup.
#
# api_key falls back to a placeholder so the app can be imported in test/dev
# without a real key (no network call happens at construction). DRIFTLOCK_DB_PATH
# points the dashboard at an existing telemetry database.
# ---------------------------------------------------------------------------
client = DriftlockClient(
    api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
    config=DriftlockConfig(
        log_json=True,
        db_path=os.environ.get("DRIFTLOCK_DB_PATH", "driftlock.sqlite"),
        prompt_token_warning_threshold=3000,
        cost_warning_threshold=0.05,
    ),
    optimization=OptimizationConfig(
        max_prompt_tokens=3000,
        keep_last_n_messages=10,
        default_max_output_tokens=512,
        max_cost_per_request_usd=0.10,
        budget_exceeded_action="fallback",
        fallback_model="gpt-4o-mini",
    ),
    cache=CacheConfig(
        ttl_seconds=600,    # cache responses for 10 minutes
        max_entries=500,
    ),
)

app = FastAPI(title="Driftlock Example", version="0.1.0")


# ---------------------------------------------------------------------------
# Middleware: inject request_id and route into every DriftlockClient call
# made within this request, with zero changes to the route handlers.
# ---------------------------------------------------------------------------
class DriftlockTagMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        with driftlock.tag(
            request_id=request_id,
            route=request.url.path,
            method=request.method,
        ):
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(DriftlockTagMiddleware)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-4o-mini"


class ChatResponse(BaseModel):
    reply: str
    model: str
    cache_hit: bool = False


class SummaryRequest(BaseModel):
    text: str
    model: str = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Simple chat endpoint.
    Tags endpoint='chat' per-call; request_id/route come from middleware.
    """
    try:
        response = client.chat.completions.create(
            model=req.model,
            messages=[{"role": "user", "content": req.message}],
            temperature=0.0,           # deterministic → cache-friendly
            _dl_endpoint="chat",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    recent = client.recent_calls(limit=1)
    was_cached = recent[0]["cache_hit"] if recent else False

    return ChatResponse(
        reply=response.choices[0].message.content,
        model=response.model,
        cache_hit=was_cached,
    )


@app.post("/summarise", response_model=ChatResponse)
async def summarise(req: SummaryRequest):
    """Summarise arbitrary text. Long inputs are trimmed by the optimizer."""
    try:
        response = client.chat.completions.create(
            model=req.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise summariser. Return a 2-3 sentence summary.",
                },
                {"role": "user", "content": req.text},
            ],
            temperature=0.0,
            _dl_endpoint="summarise",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return ChatResponse(
        reply=response.choices[0].message.content,
        model=response.model,
    )


@app.get("/metrics")
async def metrics():
    """Aggregated usage + savings stats from local SQLite."""
    return {
        "all_time": client.stats(),
        "by_endpoint": {
            "chat": client.stats(endpoint="chat"),
            "summarise": client.stats(endpoint="summarise"),
        },
        "cache": client.cache_stats(),
    }


@app.get("/metrics/recent")
async def recent_calls(limit: int = 10):
    """Return the N most recent tracked calls (includes cache hits)."""
    return client.recent_calls(limit=limit)


# ---------------------------------------------------------------------------
# Mission dashboard data API (Phase 2).
#
# These are the data routes a future web dashboard will be built on. No auth.
# ---------------------------------------------------------------------------
@app.get("/missions")
async def list_missions(limit: int = 20, offset: int = 0):
    """Recent missions, paginated (id, spend, calls, status, interventions)."""
    rows = client.missions(limit=limit + offset)
    return {
        "missions": rows[offset : offset + limit],
        "limit": limit,
        "offset": offset,
        "count": len(rows),
    }


@app.get("/missions/{mission_id}")
async def mission_detail(mission_id: str):
    """Full mission stats: spend, direct/nested split, model mix, interventions."""
    summary = client.resume_mission(mission_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"mission '{mission_id}' not found")
    return client.mission_stats(mission_id)


@app.get("/missions/{mission_id}/calls")
async def mission_call_graph(mission_id: str):
    """Parent/child call graph for a mission."""
    summary = client.resume_mission(mission_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"mission '{mission_id}' not found")
    stats = client.mission_stats(mission_id)
    return {"mission_id": mission_id, "call_graph": stats["call_graph"]}


@app.get("/metrics/summary")
async def metrics_summary():
    """Total spend, calls, and mission counts for today and this month."""
    return client._storage.metrics_summary()


@app.get("/metrics/burn-rate")
async def burn_rate(hours: int = 24):
    """Hourly spend + call count over the last N hours."""
    return {"hours": hours, "buckets": client._storage.hourly_burn_rate(hours=hours)}


@app.get("/metrics/top-endpoints")
async def top_endpoints(limit: int = 5):
    """Top endpoints by spend: calls, total/avg cost, avg latency."""
    return {"endpoints": client._storage.top_endpoints(limit=limit)}


# ---------------------------------------------------------------------------
# Dashboard — single-page frontend over the routes above. Vanilla HTML/CSS/JS,
# no build step; just open http://localhost:8000.
# ---------------------------------------------------------------------------
_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
async def dashboard():
    return FileResponse(_STATIC_DIR / "dashboard.html", media_type="text/html")
