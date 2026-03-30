"""
OpenEnv HTTP server for the Customer Support Routing environment.

Endpoints:
  POST /reset          → Observation
  POST /step           → StepResult
  GET  /state          → state dict
  GET  /tasks          → list of available tasks
  GET  /health         → health check
  GET  /               → API info
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import CustomerSupportRoutingEnv
from env.models import Action, Observation, StepResult
from tasks.tasks import TASKS
from typing import Optional


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Support Routing — OpenEnv",
    description=(
        "An OpenEnv-compliant environment where AI agents learn to route, "
        "prioritise, and escalate customer support tickets."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single session)
_env: Optional[CustomerSupportRoutingEnv] = None


def _get_env() -> CustomerSupportRoutingEnv:
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    return _env


# ---------------------------------------------------------------------------
# Request / Response helpers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task1_basic_routing"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Action


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "Customer Support Routing — OpenEnv",
        "version": "1.0.0",
        "tasks": list(TASKS.keys()),
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health"],
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    return {
        task_id: {
            "id": meta["id"],
            "name": meta["name"],
            "difficulty": meta["difficulty"],
            "description": meta["description"],
            "num_tickets": meta["num_tickets"],
        }
        for task_id, meta in TASKS.items()
    }


@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = None) -> Observation:
    global _env

    # Default values if validator sends empty POST body
    task_id = req.task_id if req else "task1_basic_routing"
    seed = req.seed if req else None

    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'")

    _env = CustomerSupportRoutingEnv(task_id=task_id, seed=seed)
    obs = _env.reset(seed=seed)
    return obs


@app.post("/step", response_model=StepResult)
def step(req: StepRequest) -> StepResult:
    env = _get_env()
    try:
        result = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/state")
def state() -> Dict[str, Any]:
    env = _get_env()
    return env.state()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
