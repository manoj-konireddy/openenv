"""
Three graded tasks for the Customer Support Routing environment.

Task 1 (Easy)   — Basic Department Routing
Task 2 (Medium) — Priority-Aware Routing with SLA constraints
Task 3 (Hard)   — Full Triage: Department + Priority + Escalation logic

Each task exposes:
  - description: str
  - generate_episode(seed) → (tickets, ground_truth)
  - grade(routings, ground_truth) → float in [0.0, 1.0]
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from env.models import (
    Action, Department, Priority, SentimentLabel,
    SupportTicket, TicketRouting,
)
from data.ticket_generator import generate_tickets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dept_from_metadata(ticket: SupportTicket) -> Department:
    """Ground-truth department stored in ticket metadata."""
    return Department(ticket.metadata["dept_hint"])


def _expected_priority(ticket: SupportTicket) -> Priority:
    """
    Deterministic priority rule used as ground truth:

    URGENT  → enterprise tier  AND (angry sentiment OR wait > 240 min)
    HIGH    → enterprise tier  OR  angry sentiment  OR  wait > 120 min
    MEDIUM  → pro tier         OR  negative sentiment OR wait > 30 min
    LOW     → everything else
    """
    is_enterprise = ticket.customer_tier == "enterprise"
    is_pro = ticket.customer_tier == "pro"
    is_angry = ticket.sentiment == SentimentLabel.ANGRY
    is_negative = ticket.sentiment == SentimentLabel.NEGATIVE
    long_wait = ticket.wait_time_minutes > 240
    medium_wait = ticket.wait_time_minutes > 120
    short_wait = ticket.wait_time_minutes > 30
    repeated = ticket.previous_contacts >= 3

    if is_enterprise and (is_angry or long_wait):
        return Priority.URGENT
    if is_enterprise or is_angry or medium_wait:
        return Priority.HIGH
    if is_pro or is_negative or short_wait or repeated:
        return Priority.MEDIUM
    return Priority.LOW


def _should_escalate(ticket: SupportTicket) -> bool:
    """
    Escalation rule:
    Escalate if previous_contacts >= 5  OR
               (enterprise tier AND angry AND wait > 180 min)
    """
    if ticket.previous_contacts >= 5:
        return True
    if (
        ticket.customer_tier == "enterprise"
        and ticket.sentiment == SentimentLabel.ANGRY
        and ticket.wait_time_minutes > 180
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Ground-truth record
# ---------------------------------------------------------------------------

@dataclass
class GroundTruth:
    ticket_id: str
    expected_department: Department
    expected_priority: Priority
    should_escalate: bool


# ---------------------------------------------------------------------------
# Task 1 — Basic Department Routing (Easy)
# ---------------------------------------------------------------------------

TASK1_DESCRIPTION = """
**Task 1 — Basic Department Routing (Easy)**

You are given a queue of 5 customer support tickets. Your job is to route
each ticket to the correct department:

  • billing      — payment, invoices, charges, subscriptions
  • technical    — bugs, API issues, login problems, performance
  • shipping     — delivery, tracking, lost/damaged packages
  • returns      — refunds, exchanges, return labels
  • general      — account help, feature requests, general questions

For this task, set every ticket's priority to "medium".

Score: (# correctly routed departments) / (total tickets)
"""

def task1_episode(seed: int = 42) -> Tuple[List[SupportTicket], List[GroundTruth]]:
    tickets = generate_tickets(5, seed=seed)
    gt = [
        GroundTruth(
            ticket_id=t.ticket_id,
            expected_department=_dept_from_metadata(t),
            expected_priority=Priority.MEDIUM,  # not graded in task 1
            should_escalate=False,
        )
        for t in tickets
    ]
    return tickets, gt


def task1_grade(routings: List[TicketRouting], ground_truth: List[GroundTruth]) -> float:
    gt_map = {g.ticket_id: g for g in ground_truth}
    correct = 0
    for r in routings:
        gt = gt_map.get(r.ticket_id)
        if gt and r.department == gt.expected_department:
            correct += 1
    return correct / max(len(ground_truth), 1)


# ---------------------------------------------------------------------------
# Task 2 — Priority-Aware Routing (Medium)
# ---------------------------------------------------------------------------

TASK2_DESCRIPTION = """
**Task 2 — Priority-Aware Routing (Medium)**

You are given a queue of 8 customer support tickets. Route each ticket to the
correct department AND assign the correct priority using these SLA rules:

  URGENT  → Enterprise customer who is angry OR has waited more than 4 hours
  HIGH    → Enterprise customer, OR angry sentiment, OR wait > 2 hours
  MEDIUM  → Pro customer, OR negative sentiment, OR wait > 30 min, OR 3+ previous contacts
  LOW     → All others

Score: 0.5 × (dept accuracy) + 0.5 × (priority accuracy)
"""

def task2_episode(seed: int = 99) -> Tuple[List[SupportTicket], List[GroundTruth]]:
    tickets = generate_tickets(8, seed=seed)
    gt = [
        GroundTruth(
            ticket_id=t.ticket_id,
            expected_department=_dept_from_metadata(t),
            expected_priority=_expected_priority(t),
            should_escalate=False,
        )
        for t in tickets
    ]
    return tickets, gt


def task2_grade(routings: List[TicketRouting], ground_truth: List[GroundTruth]) -> float:
    gt_map = {g.ticket_id: g for g in ground_truth}
    dept_correct = 0
    priority_correct = 0
    n = len(ground_truth)
    for r in routings:
        gt = gt_map.get(r.ticket_id)
        if not gt:
            continue
        if r.department == gt.expected_department:
            dept_correct += 1
        if r.priority == gt.expected_priority:
            priority_correct += 1
    dept_score = dept_correct / max(n, 1)
    priority_score = priority_correct / max(n, 1)
    return 0.5 * dept_score + 0.5 * priority_score


# ---------------------------------------------------------------------------
# Task 3 — Full Triage with Escalation (Hard)
# ---------------------------------------------------------------------------

TASK3_DESCRIPTION = """
**Task 3 — Full Triage with Escalation (Hard)**

You are given a queue of 12 customer support tickets. For each ticket you must:

  1. Route to the correct department.
  2. Assign the correct priority (same SLA rules as Task 2).
  3. Escalate tickets that meet escalation criteria — use department "escalation"
     when a ticket should be escalated.

Escalation criteria (either condition triggers it):
  • Customer has contacted support 5 or more previous times
  • Enterprise customer who is angry AND has waited more than 3 hours

Score breakdown:
  • 40% — department accuracy (escalated tickets counted as correct if escalation was right)
  • 30% — priority accuracy
  • 30% — escalation accuracy (precision + recall of escalation decisions)
"""

def task3_episode(seed: int = 7) -> Tuple[List[SupportTicket], List[GroundTruth]]:
    # Force a mix that guarantees some escalation-worthy tickets
    from data.ticket_generator import generate_tickets as _gen
    tickets = _gen(12, seed=seed)

    # Manually inject two guaranteed escalation candidates
    tickets[2].previous_contacts = 6   # repeat contact → escalate
    tickets[2].metadata["dept_hint"] = tickets[2].metadata["dept_hint"]

    tickets[7].customer_tier = "enterprise"
    tickets[7].sentiment = SentimentLabel.ANGRY
    tickets[7].wait_time_minutes = 210  # > 180 min → escalate

    gt = []
    for t in tickets:
        escalate = _should_escalate(t)
        gt.append(GroundTruth(
            ticket_id=t.ticket_id,
            expected_department=Department.ESCALATION if escalate else _dept_from_metadata(t),
            expected_priority=_expected_priority(t),
            should_escalate=escalate,
        ))
    return tickets, gt


def task3_grade(routings: List[TicketRouting], ground_truth: List[GroundTruth]) -> float:
    gt_map = {g.ticket_id: g for g in ground_truth}
    n = len(ground_truth)
    dept_correct = 0
    priority_correct = 0

    # Escalation: treat as binary classification
    true_pos = false_pos = false_neg = 0

    for r in routings:
        gt = gt_map.get(r.ticket_id)
        if not gt:
            continue

        predicted_escalate = (r.department == Department.ESCALATION)
        expected_escalate = gt.should_escalate

        # Dept: if should escalate, agent must use Department.ESCALATION
        if r.department == gt.expected_department:
            dept_correct += 1

        if r.priority == gt.expected_priority:
            priority_correct += 1

        # Escalation confusion matrix
        if predicted_escalate and expected_escalate:
            true_pos += 1
        elif predicted_escalate and not expected_escalate:
            false_pos += 1
        elif not predicted_escalate and expected_escalate:
            false_neg += 1

    dept_score = dept_correct / max(n, 1)
    priority_score = priority_correct / max(n, 1)

    # F1 for escalation
    precision = true_pos / max(true_pos + false_pos, 1)
    recall = true_pos / max(true_pos + false_neg, 1)
    if precision + recall > 0:
        escalation_f1 = 2 * precision * recall / (precision + recall)
    else:
        escalation_f1 = 0.0

    final = 0.4 * dept_score + 0.3 * priority_score + 0.3 * escalation_f1
    return round(final, 4)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASKS = {
    "task1_basic_routing": {
        "id": "task1_basic_routing",
        "name": "Basic Department Routing",
        "difficulty": "easy",
        "description": TASK1_DESCRIPTION,
        "episode_fn": task1_episode,
        "grade_fn": task1_grade,
        "default_seed": 42,
        "num_tickets": 5,
    },
    "task2_priority_routing": {
        "id": "task2_priority_routing",
        "name": "Priority-Aware Routing",
        "difficulty": "medium",
        "description": TASK2_DESCRIPTION,
        "episode_fn": task2_episode,
        "grade_fn": task2_grade,
        "default_seed": 99,
        "num_tickets": 8,
    },
    "task3_full_triage": {
        "id": "task3_full_triage",
        "name": "Full Triage with Escalation",
        "difficulty": "hard",
        "description": TASK3_DESCRIPTION,
        "episode_fn": task3_episode,
        "grade_fn": task3_grade,
        "default_seed": 7,
        "num_tickets": 12,
    },
}
