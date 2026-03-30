#!/usr/bin/env python3
"""
validate.py — Pre-submission OpenEnv validator for Customer Support Routing.

Checks:
  [1] openenv.yaml is present and valid
  [2] All required source files exist
  [3] Dockerfile is present
  [4] inference.py is present at root
  [5] Task registry has 3+ tasks
  [6] Each task has episode_fn and grade_fn
  [7] Graders produce scores in [0.0, 1.0]
  [8] reset() returns clean Observation
  [9] step() returns StepResult with reward in [-1, 1]
  [10] state() returns a dict
  [11] Done flag triggers after max steps
  [12] Metadata (dept_hint) is hidden from agent

Run:
  python validate.py
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback

# Ensure project root on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

results = []


def check(name: str, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True, None))
    except Exception as e:
        msg = str(e)
        print(f"  {FAIL}  {name}")
        print(f"       → {msg}")
        results.append((name, False, msg))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def c1_openenv_yaml():
    import yaml
    path = os.path.join(ROOT, "openenv.yaml")
    assert os.path.exists(path), "openenv.yaml not found"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    required_keys = ["name", "version", "tasks", "observation_space", "action_space", "reward"]
    for k in required_keys:
        assert k in cfg, f"openenv.yaml missing key: '{k}'"
    assert len(cfg["tasks"]) >= 3, "openenv.yaml must define at least 3 tasks"


def c2_required_files():
    required = [
        "openenv.yaml",
        "Dockerfile",
        "inference.py",
        "requirements.txt",
        "server.py",
        os.path.join("env", "models.py"),
        os.path.join("env", "environment.py"),
        os.path.join("tasks", "tasks.py"),
        os.path.join("data", "ticket_generator.py"),
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(ROOT, f))]
    assert not missing, f"Missing files: {missing}"


def c3_dockerfile():
    path = os.path.join(ROOT, "Dockerfile")
    assert os.path.exists(path), "Dockerfile not found"
    content = open(path).read()
    assert "FROM" in content, "Dockerfile missing FROM instruction"
    assert "7860" in content, "Dockerfile should expose port 7860 (HF Spaces)"
    assert "uvicorn" in content.lower() or "CMD" in content, "Dockerfile should have a CMD"


def c4_inference_script():
    path = os.path.join(ROOT, "inference.py")
    assert os.path.exists(path), "inference.py not found at project root"
    content = open(path).read()
    assert "OPENAI_API_KEY" in content, "inference.py must read OPENAI_API_KEY"
    assert "API_BASE_URL" in content, "inference.py must read API_BASE_URL"
    assert "MODEL_NAME" in content, "inference.py must read MODEL_NAME"
    assert "HF_TOKEN" in content, "inference.py must read HF_TOKEN"


def c5_task_registry():
    from tasks.tasks import TASKS
    assert len(TASKS) >= 3, f"Need ≥3 tasks, found {len(TASKS)}"


def c6_task_keys():
    from tasks.tasks import TASKS
    required = {"id", "name", "difficulty", "description", "episode_fn", "grade_fn", "num_tickets"}
    for task_id, task in TASKS.items():
        missing = required - set(task.keys())
        assert not missing, f"Task '{task_id}' missing keys: {missing}"
    difficulties = {t["difficulty"] for t in TASKS.values()}
    assert "easy" in difficulties, "No easy task found"
    assert "medium" in difficulties, "No medium task found"
    assert "hard" in difficulties, "No hard task found"


def c7_grader_scores():
    from tasks.tasks import TASKS
    from env.models import Department, Priority, TicketRouting
    for task_id, task in TASKS.items():
        _, gt = task["episode_fn"]()
        routings = [
            TicketRouting(
                ticket_id=g.ticket_id,
                department=Department.GENERAL,
                priority=Priority.MEDIUM,
                reason="validator test",
            )
            for g in gt
        ]
        score = task["grade_fn"](routings, gt)
        assert isinstance(score, float), f"{task_id}: grader must return float"
        assert 0.0 <= score <= 1.0, f"{task_id}: score {score} out of [0,1]"
        # Perfect score
        perfect = [
            TicketRouting(
                ticket_id=g.ticket_id,
                department=g.expected_department,
                priority=g.expected_priority,
                reason="perfect",
            )
            for g in gt
        ]
        perfect_score = task["grade_fn"](perfect, gt)
        assert perfect_score == 1.0, f"{task_id}: perfect routing should score 1.0, got {perfect_score}"


def c8_reset():
    from env.environment import CustomerSupportRoutingEnv
    from env.models import Observation
    for task_id in ["task1_basic_routing", "task2_priority_routing", "task3_full_triage"]:
        env = CustomerSupportRoutingEnv(task_id)
        obs = env.reset()
        assert isinstance(obs, Observation), f"{task_id}: reset() must return Observation"
        assert len(obs.tickets) > 0, f"{task_id}: reset() must return tickets"
        assert obs.step_number == 0, f"{task_id}: step_number must be 0 after reset"
        # Second reset should give same tickets (deterministic)
        obs2 = env.reset()
        ids1 = [t.ticket_id for t in obs.tickets]
        ids2 = [t.ticket_id for t in obs2.tickets]
        assert ids1 == ids2, f"{task_id}: reset() must be deterministic"


def c9_step():
    from env.environment import CustomerSupportRoutingEnv
    from env.models import Action, Department, Priority, StepResult, TicketRouting
    env = CustomerSupportRoutingEnv("task1_basic_routing")
    obs = env.reset()
    routings = [
        TicketRouting(
            ticket_id=t.ticket_id,
            department=Department.BILLING,
            priority=Priority.MEDIUM,
            reason="test",
        )
        for t in obs.tickets
    ]
    result = env.step(Action(routings=routings))
    assert isinstance(result, StepResult), "step() must return StepResult"
    assert -1.0 <= result.reward <= 1.0, f"Reward {result.reward} out of [-1, 1]"
    assert isinstance(result.done, bool), "done must be bool"
    assert isinstance(result.info, dict), "info must be dict"


def c10_state():
    from env.environment import CustomerSupportRoutingEnv
    env = CustomerSupportRoutingEnv("task1_basic_routing")
    env.reset()
    s = env.state()
    assert isinstance(s, dict), "state() must return dict"
    for key in ["task_id", "step_count", "done", "tickets", "routings_so_far"]:
        assert key in s, f"state() missing key '{key}'"


def c11_done_flag():
    from env.environment import CustomerSupportRoutingEnv, MAX_STEPS
    from env.models import Action, Department, Priority, TicketRouting
    env = CustomerSupportRoutingEnv("task1_basic_routing")
    obs = env.reset()
    # Take MAX_STEPS steps
    for _ in range(MAX_STEPS):
        action = Action(routings=[
            TicketRouting(
                ticket_id=t.ticket_id,
                department=Department.GENERAL,
                priority=Priority.LOW,
                reason="test",
            )
            for t in obs.tickets
        ])
        result = env.step(action)
        if result.done:
            break
    assert result.done, "Episode must be done after all tickets routed or max steps reached"


def c12_metadata_hidden():
    from env.environment import CustomerSupportRoutingEnv
    env = CustomerSupportRoutingEnv("task1_basic_routing")
    obs = env.reset()
    for ticket in obs.tickets:
        assert "dept_hint" not in ticket.metadata, \
            f"dept_hint ground truth must be hidden from agent observation (ticket {ticket.ticket_id})"


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  OpenEnv Pre-Submission Validator")
    print("  Customer Support Routing Environment")
    print("=" * 60 + "\n")

    checks = [
        ("[1] openenv.yaml valid",          c1_openenv_yaml),
        ("[2] Required files present",       c2_required_files),
        ("[3] Dockerfile valid",             c3_dockerfile),
        ("[4] inference.py spec compliant",  c4_inference_script),
        ("[5] 3+ tasks registered",          c5_task_registry),
        ("[6] Task keys & difficulty range", c6_task_keys),
        ("[7] Grader scores in [0.0, 1.0]", c7_grader_scores),
        ("[8] reset() deterministic",        c8_reset),
        ("[9] step() reward in [-1, 1]",     c9_step),
        ("[10] state() returns valid dict",  c10_state),
        ("[11] Done flag triggers correctly",c11_done_flag),
        ("[12] Metadata hidden from agent",  c12_metadata_hidden),
    ]

    for name, fn in checks:
        check(name, fn)

    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    failed = total - passed

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED ← fix before submitting")
    else:
        print("  🎉  All checks passed — ready to submit!")
    print("=" * 60 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
