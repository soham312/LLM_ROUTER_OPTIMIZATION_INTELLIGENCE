from typing import List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class RouteRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's query to route")
    conversation_history: Optional[List[Message]] = Field(
        default=None, description="Prior turns, oldest first, for multi-turn context"
    )


class RouteResponse(BaseModel):
    model_used: str
    response_text: str
    judge_score: float
    final_reward: float
    escalated: bool
    escalation_reason: Optional[str]
    cost: float
    latency_ms: float
