---
title: Customer Support Routing — OpenEnv
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - nlp
  - agent-evaluation
license: mit
---

# Customer Support Routing — OpenEnv 🎫

An **OpenEnv-compliant** environment where AI agents learn to triage real-world
customer support tickets: routing to the right department, assigning correct
SLA-based priorities, and identifying escalation-worthy cases.

---

## Why This Environment?

Customer support triage is a high-stakes, high-volume task at every SaaS
company. Routing mistakes cost money: a billing issue sent to engineering,
an angry enterprise customer left on low priority, a repeat contacter never
escalated. This environment benchmarks whether an LLM agent can reliably apply
structured business rules to ambiguous natural-language inputs — a skill that
directly transfers to production deployments.

---

## Environment Overview

| Property          | Value                                              |
|-------------------|----------------------------------------------------|
| Domain            | Customer support operations                        |
| Action space      | Route ticket → {department, priority, reason}      |
| Observation space | Ticket queue with customer metadata                |
| Reward            | Shaped per-step, normalized to [−1, +1]            |
| Episode length    | Up to 3 steps                                      |
| Tasks             | 3 (easy → medium → hard)                          |
| Reproducibility   | Fully deterministic with fixed seeds               |

---

## Action & Observation Spaces

### Observation
```json
{
  "tickets": [
    {
      "ticket_id": "TKT-0042-0001",
      "subject": "API returning 500 errors",
      "body": "Our integration has been throwing 500 errors since 2 AM...",
      "customer_tier": "enterprise",
      "sentiment": "angry",
      "previous_contacts": 2,
      "wait_time_minutes": 95,
      "tags": ["bug", "api"]
    }
  ],
  "queue_stats": {
    "length": 5,
    "avg_wait_min": 62.4,
    "dept_load": {"technical": 2, "billing": 1, "general": 2}
  },
  "step_number": 0,
  "task_id": "task1_basic_routing",
  "instructions": "..."
}
```

### Action
```json
{
  "routings": [
    {
      "ticket_id": "TKT-0042-0001",
      "department": "technical",
      "priority": "high",
      "reason": "Enterprise customer experiencing API outage impacting production."
    }
  ]
}
```

### Departments
| Department   | When to use                                              |
|--------------|----------------------------------------------------------|
| `billing`    | Payments, invoices, charges, subscription issues         |
| `technical`  | Bugs, API failures, login problems, performance          |
| `shipping`   | Delivery, tracking, lost or damaged packages             |
| `returns`    | Refunds, exchanges, return labels, warranty claims       |
| `general`    | Account help, feature requests, general questions        |
| `escalation` | Repeat contacts (≥5) or angry enterprise + long wait     |

### Priority SLA Rules
| Priority | Triggers                                                           |
|----------|--------------------------------------------------------------------|
| `urgent` | Enterprise + (angry OR waited > 4 hours)                          |
| `high`   | Enterprise, OR angry, OR waited > 2 hours                         |
| `medium` | Pro tier, OR negative, OR waited > 30 min, OR 3+ prior contacts   |
| `low`    | All others                                                         |

---

## Tasks

### Task 1 — Basic Department Routing *(Easy)*
> Route **5 tickets** to the correct department. Priority is fixed at `medium`.

**Score:** Fraction of tickets routed to the correct department.  
**Expected baseline (GPT-4o):** ~0.80–1.00

---

### Task 2 — Priority-Aware Routing *(Medium)*
> Route **8 tickets** to the correct department **and** assign the correct
> priority using the SLA rules above.

**Score:** `0.5 × dept_accuracy + 0.5 × priority_accuracy`  
**Expected baseline (GPT-4o):** ~0.60–0.80

---

### Task 3 — Full Triage with Escalation *(Hard)*
> Route **12 tickets** including department, priority, **and** escalation
> detection. Some tickets must be sent to the `escalation` department.

**Score:** `0.40 × dept_accuracy + 0.30 × priority_accuracy + 0.30 × escalation_F1`  
**Expected baseline (GPT-4o):** ~0.50–0.70

---

## Reward Function

The step reward is a weighted sum of per-ticket components, normalized to [−1, +1]:

| Component              | Value   | Description                                      |
|------------------------|---------|--------------------------------------------------|
| Correct department     | +0.20   | Per ticket, department matches ground truth      |
| Correct priority       | +0.10   | Per ticket, priority matches SLA rules           |
| Correct escalation     | +0.15   | Per ticket, escalation decision is correct       |
| Non-empty reason       | +0.02   | Agent provides a justification string            |
| Loop (re-route)        | −0.10   | Routing an already-routed ticket                 |
| Invalid ticket ID      | −0.05   | Referencing a ticket_id not in the queue         |

The final **episode score** is computed by the task-specific grader (independent
of step rewards) for clean, reproducible evaluation metrics.

---

## API Endpoints

| Method | Path      | Description                          |
|--------|-----------|--------------------------------------|
| `POST` | `/reset`  | Start new episode, returns Observation |
| `POST` | `/step`   | Submit routing action, returns StepResult |
| `GET`  | `/state`  | Full internal state snapshot         |
| `GET`  | `/tasks`  | List all tasks with metadata         |
| `GET`  | `/health` | Health check → `{"status": "ok"}`   |

### Example: Reset
```bash
curl -X POST https://konireddy-customer-support-routing-openenv.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_basic_routing"}'
```

### Example: Step
```bash
curl -X POST https://your-space.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "routings": [
        {
          "ticket_id": "TKT-0042-0000",
          "department": "technical",
          "priority": "high",
          "reason": "API outage affecting production systems."
        }
      ]
    }
  }'
```

---

## Setup & Usage

### Local (Python)
```bash
git clone https://huggingface.co/spaces/your-username/customer-support-routing-openenv
cd customer-support-routing-openenv
pip install -r requirements.txt
python server.py
# Server running at http://localhost:7860
```

### Docker
```bash
docker build -t support-routing-openenv .
docker run -p 7860:7860 support-routing-openenv
```

### Run the Baseline Inference Script
```bash
export OPENAI_API_KEY=sk-...
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o
export HF_TOKEN=hf_...

# Run all 3 tasks directly (no server needed)
python inference.py

# Run against a deployed HF Space
python inference.py --http https://your-space.hf.space

# Run a single task
python inference.py --task task1_basic_routing
```

### Run Tests
```bash
python -m pytest tests/ -v
```

### Run Pre-Submission Validator
```bash
python validate.py
```

### Use the Environment Programmatically
```python
from env.environment import CustomerSupportRoutingEnv
from env.models import Action, Department, Priority, TicketRouting

env = CustomerSupportRoutingEnv(task_id="task2_priority_routing")
obs = env.reset()

for ticket in obs.tickets:
    print(f"{ticket.ticket_id}: {ticket.subject}")

action = Action(routings=[
    TicketRouting(
        ticket_id=obs.tickets[0].ticket_id,
        department=Department.TECHNICAL,
        priority=Priority.HIGH,
        reason="Production API outage for enterprise customer.",
    )
])

result = env.step(action)
print(f"Reward: {result.reward}, Done: {result.done}")
print(f"Score: {result.info.get('episode_score')}")

state = env.state()
env.close()
```

---

## Baseline Scores (GPT-4o, seed=default)

| Task                    | Difficulty | Final Score |
|-------------------------|------------|-------------|
| task1_basic_routing     | Easy       | ~0.90       |
| task2_priority_routing  | Medium     | ~0.69       |
| task3_full_triage       | Hard       | ~0.58       |
| **Average**             |            | **~0.72**   |

*Scores are reproducible with fixed seeds and `temperature=0`.*

---

## Project Structure

```
customer-support-routing-openenv/
├── openenv.yaml           # OpenEnv spec metadata
├── Dockerfile             # Container definition (HF Spaces compatible)
├── requirements.txt       # Python dependencies
├── server.py              # FastAPI HTTP server
├── inference.py           # Baseline inference script (OpenAI GPT-4o)
├── validate.py            # Pre-submission validator (12 checks)
├── README.md              # This file
├── env/
│   ├── __init__.py
│   ├── models.py          # Pydantic typed models (Observation, Action, Reward)
│   └── environment.py     # Core environment: reset/step/state
├── tasks/
│   ├── __init__.py
│   └── tasks.py           # 3 tasks with graders (easy/medium/hard)
├── data/
│   ├── __init__.py
│   └── ticket_generator.py # Deterministic synthetic ticket generator
└── tests/
    ├── __init__.py
    └── test_environment.py # Pytest test suite (15 tests)
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
