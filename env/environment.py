"""
CustomerSupportRoutingEnv — OpenEnv-compliant environment.

Implements:
  reset()  → Observation
  step()   → StepResult
  state()  → dict
  close()  → None
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

from env.models import (
    Action, Department, Observation, Priority, Reward,
    StepResult, SupportTicket, TicketRouting,
)
from tasks.tasks import TASKS


# ---------------------------------------------------------------------------
# Reward shaping constants
# ---------------------------------------------------------------------------

DEPT_CORRECT_REWARD = 0.20       # per-ticket reward for correct department
PRIORITY_CORRECT_REWARD = 0.10   # per-ticket reward for correct priority
ESCALATION_CORRECT_REWARD = 0.15 # per-ticket reward for correct escalation
REASON_BONUS = 0.02              # bonus if agent provides a non-empty reason
LOOP_PENALTY = -0.10             # penalty for routing same ticket twice
INVALID_ACTION_PENALTY = -0.05   # penalty for referencing unknown ticket_id
MAX_STEPS = 3                    # agent gets ≤3 steps per episode (re-routes allowed)


def _queue_stats(tickets: List[SupportTicket]) -> Dict[str, Any]:
    if not tickets:
        return {"length": 0, "avg_wait_min": 0, "dept_load": {}}
    avg_wait = sum(t.wait_time_minutes for t in tickets) / len(tickets)
    dept_load: Dict[str, int] = {}
    for t in tickets:
        dept_load[t.metadata.get("dept_hint", "general")] = (
            dept_load.get(t.metadata.get("dept_hint", "general"), 0) + 1
        )
    return {
        "length": len(tickets),
        "avg_wait_min": round(avg_wait, 1),
        "dept_load": dept_load,
    }


class CustomerSupportRoutingEnv:
    """
    OpenEnv-compliant Customer Support Routing environment.

    Usage:
        env = CustomerSupportRoutingEnv(task_id="task1_basic_routing")
        obs = env.reset()
        result = env.step(action)
        state = env.state()
        env.close()
    """

    def __init__(self, task_id: str = "task1_basic_routing", seed: Optional[int] = None):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASKS)}")
        self.task_id = task_id
        self._task = TASKS[task_id]
        self._seed = seed if seed is not None else self._task["default_seed"]

        # Episode state
        self._tickets: List[SupportTicket] = []
        self._ground_truth = []
        self._routings_so_far: Dict[str, TicketRouting] = {}
        self._step_count: int = 0
        self._done: bool = False
        self._episode_score: float = 0.0
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset the environment, return initial observation."""
        if seed is not None:
            self._seed = seed

        tickets, gt = self._task["episode_fn"](seed=self._seed)
        self._tickets = tickets
        self._ground_truth = gt
        self._routings_so_far = {}
        self._step_count = 0
        self._done = False
        self._episode_score = 0.0
        self._start_time = time.time()

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """
        Apply the agent's routing decisions. Returns (observation, reward, done, info).

        Reward shaping:
          +0.20 per ticket with correct department
          +0.10 per ticket with correct priority
          +0.15 per ticket with correct escalation decision
          +0.02 bonus for non-empty reason strings
          -0.10 penalty for routing a ticket already routed this episode
          -0.05 penalty for unknown ticket IDs
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._step_count += 1
        gt_map = {g.ticket_id: g for g in self._ground_truth}
        reward_components: Dict[str, float] = {
            "dept_reward": 0.0,
            "priority_reward": 0.0,
            "escalation_reward": 0.0,
            "reason_bonus": 0.0,
            "loop_penalty": 0.0,
            "invalid_penalty": 0.0,
        }

        for routing in action.routings:
            # Penalty: unknown ticket
            if routing.ticket_id not in gt_map:
                reward_components["invalid_penalty"] += INVALID_ACTION_PENALTY
                continue

            # Penalty: routing already-routed ticket (loop detection)
            if routing.ticket_id in self._routings_so_far:
                reward_components["loop_penalty"] += LOOP_PENALTY

            self._routings_so_far[routing.ticket_id] = routing
            gt = gt_map[routing.ticket_id]

            # Department correctness
            if routing.department == gt.expected_department:
                reward_components["dept_reward"] += DEPT_CORRECT_REWARD

            # Priority correctness (not graded in task1 but still rewarded)
            if routing.priority == gt.expected_priority:
                reward_components["priority_reward"] += PRIORITY_CORRECT_REWARD

            # Escalation correctness
            predicted_esc = routing.department == Department.ESCALATION
            if predicted_esc == gt.should_escalate:
                reward_components["escalation_reward"] += ESCALATION_CORRECT_REWARD

            # Reason bonus
            if routing.reason and len(routing.reason.strip()) > 5:
                reward_components["reason_bonus"] += REASON_BONUS

        step_reward = sum(reward_components.values())

        # Normalise reward to [-1, 1]
        n = max(len(self._ground_truth), 1)
        max_possible = n * (DEPT_CORRECT_REWARD + PRIORITY_CORRECT_REWARD +
                            ESCALATION_CORRECT_REWARD + REASON_BONUS)
        if max_possible > 0:
            step_reward_norm = max(-1.0, min(1.0, step_reward / max_possible))
        else:
            step_reward_norm = 0.0

        # Episode done: all tickets routed OR max steps reached
        all_routed = len(self._routings_so_far) >= len(self._ground_truth)
        done = all_routed or self._step_count >= MAX_STEPS
        self._done = done

        if done:
            # Final task score via task-specific grader
            final_routings = list(self._routings_so_far.values())
            self._episode_score = self._task["grade_fn"](final_routings, self._ground_truth)

        obs = self._build_observation()
        info = {
            "step_reward_raw": step_reward,
            "reward_breakdown": reward_components,
            "routed_so_far": len(self._routings_so_far),
            "total_tickets": len(self._ground_truth),
            "episode_score": self._episode_score if done else None,
            "elapsed_seconds": round(time.time() - self._start_time, 2),
        }

        return StepResult(
            observation=obs,
            reward=round(step_reward_norm, 4),
            done=done,
            info=info,
        )

    def state(self) -> Dict[str, Any]:
        """Return the full internal state snapshot."""
        return {
            "task_id": self.task_id,
            "seed": self._seed,
            "step_count": self._step_count,
            "done": self._done,
            "episode_score": self._episode_score,
            "tickets": [t.model_dump() for t in self._tickets],
            "routings_so_far": {
                k: v.model_dump() for k, v in self._routings_so_far.items()
            },
            "ground_truth": [
                {
                    "ticket_id": g.ticket_id,
                    "expected_department": g.expected_department.value,
                    "expected_priority": g.expected_priority.value,
                    "should_escalate": g.should_escalate,
                }
                for g in self._ground_truth
            ],
        }

    def close(self) -> None:
        """Clean up resources."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        """Build the observation visible to the agent."""
        # Pending = tickets not yet routed
        routed_ids = set(self._routings_so_far.keys())
        pending = [t for t in self._tickets if t.ticket_id not in routed_ids]

        # Strip internal metadata before showing to agent
        visible_tickets = []
        for t in pending:
            d = t.model_dump()
            d["metadata"] = {}  # hide dept_hint ground truth from agent
            visible_tickets.append(SupportTicket(**d))

        return Observation(
            tickets=visible_tickets,
            queue_stats=_queue_stats(pending),
            step_number=self._step_count,
            task_id=self.task_id,
            instructions=self._task["description"],
        )
