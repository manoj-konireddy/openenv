"""
Typed Pydantic models for the Customer Support Routing environment.
Implements OpenEnv spec: Observation, Action, Reward.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Department(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    SHIPPING = "shipping"
    RETURNS = "returns"
    GENERAL = "general"
    ESCALATION = "escalation"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANGRY = "angry"


# ---------------------------------------------------------------------------
# Ticket model (part of observation)
# ---------------------------------------------------------------------------

class SupportTicket(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_tier: str = Field(
        description="Customer tier: free | pro | enterprise"
    )
    sentiment: SentimentLabel
    previous_contacts: int = Field(
        description="How many times this customer has contacted support before"
    )
    wait_time_minutes: int = Field(
        description="How long the customer has been waiting"
    )
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    What the agent sees at each step.
    Contains the current ticket queue and queue-level statistics.
    """
    tickets: List[SupportTicket] = Field(
        description="Pending tickets the agent must route"
    )
    queue_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate queue info: length, avg wait, dept load"
    )
    step_number: int = Field(default=0)
    task_id: str = Field(default="")
    instructions: str = Field(
        default="",
        description="Natural-language task instructions shown to the agent"
    )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class TicketRouting(BaseModel):
    ticket_id: str
    department: Department
    priority: Priority
    reason: str = Field(
        description="One-sentence justification for this routing decision"
    )


class Action(BaseModel):
    """
    Agent's action: route one or more tickets per step.
    """
    routings: List[TicketRouting] = Field(
        description="List of routing decisions, one per ticket"
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    value: float = Field(
        description="Step reward in [-1.0, 1.0]"
    )
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component reward breakdown for debugging"
    )
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
