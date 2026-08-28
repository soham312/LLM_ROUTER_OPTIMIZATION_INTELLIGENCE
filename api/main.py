"""
STAGE 10a: Deployment-grade API.

Exposes the OptimizationRouter over HTTP via FastAPI, gated by API-key
authentication and per-key rate limiting, so the routing intelligence built
in Phase 2-4 can be called as a real service instead of only from Python.

Run with: uvicorn api.main:app --reload
Docs at:  http://127.0.0.1:8000/docs
"""

from fastapi import Depends, FastAPI

from api.dependencies import get_router
from api.rate_limiter import enforce_rate_limit
from api.schemas import RouteRequest, RouteResponse
from router.router_core import OptimizationRouter

app = FastAPI(
    title="LLM Inference Router API",
    description="Routes queries to the cheapest/fastest model that still clears the quality bar.",
    version="1.0.0",
)


@app.get("/health")
def health_check() -> dict:
    """Unauthenticated liveness probe - standard for deployment health checks."""
    return {"status": "ok"}


@app.post("/v1/route", response_model=RouteResponse)
def route_query(
    request: RouteRequest,
    api_key: str = Depends(enforce_rate_limit),
    router: OptimizationRouter = Depends(get_router),
) -> RouteResponse:
    """
    Routes a query through the bandit, executes it, scores it with the
    judge, and returns the result. Requires a valid X-API-Key header and is
    subject to per-key rate limiting (see api/rate_limiter.py).
    """
    history = (
        [message.model_dump() for message in request.conversation_history]
        if request.conversation_history
        else None
    )
    result = router.route_and_execute(request.query, conversation_history=history)
    response = result["response"]

    return RouteResponse(
        model_used=result["model_used"],
        response_text=response.response_text,
        judge_score=result["judge_score"],
        final_reward=result["final_reward"],
        escalated=result["escalated"],
        escalation_reason=result["escalation_reason"],
        cost=response.simulated_cost,
        latency_ms=response.latency_ms,
    )
