"""
inference.py — Baseline inference script for Customer Support Routing OpenEnv.

Runs GPT-4o against all 3 tasks and reports per-task scores.

Required environment variables:
  OPENAI_API_KEY  — OpenAI API key
  API_BASE_URL    — API base URL (default: https://api.openai.com/v1)
  MODEL_NAME      — Model identifier (default: gpt-4o)
  HF_TOKEN        — Hugging Face token (required for HF Space deployment)

Usage:
  python inference.py
  python inference.py --task task1_basic_routing
  python inference.py --local   # run against local FastAPI server
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

MAX_STEPS    = 3
TEMPERATURE  = 0.0   # deterministic for reproducibility

SYSTEM_PROMPT = """You are an expert customer support operations manager.
Your job is to triage support tickets: route each ticket to the correct
department, assign the right priority, and flag tickets that need escalation.

Departments:
  billing    — payment issues, invoices, charges, subscription problems
  technical  — bugs, API issues, login failures, performance problems
  shipping   — delivery, tracking, lost or damaged packages
  returns    — refunds, exchanges, return labels, warranty claims
  general    — account questions, feature requests, general help
  escalation — use ONLY when escalation criteria are explicitly met

Priority levels:
  urgent  — enterprise customer who is angry OR has waited over 4 hours
  high    — enterprise customer, OR angry sentiment, OR wait > 2 hours
  medium  — pro customer, OR negative sentiment, OR wait > 30 min, OR 3+ contacts
  low     — everything else

Escalation criteria (use department="escalation"):
  • Customer has contacted support 5 or more times previously
  • Enterprise customer who is angry AND has waited more than 3 hours

Always respond with valid JSON in the exact format shown below.
Do not include any text outside the JSON block.

Response format:
{
  "routings": [
    {
      "ticket_id": "TKT-XXXX-XXXX",
      "department": "<department>",
      "priority": "<priority>",
      "reason": "<one sentence justification>"
    }
  ]
}"""


FALLBACK_ACTION = json.dumps({
    "routings": []
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_user_message(observation: Dict[str, Any]) -> str:
    """Convert an observation dict into a clear prompt for the model."""
    lines = [
        f"# Task: {observation.get('task_id', '')}",
        "",
        observation.get("instructions", ""),
        "",
        f"## Ticket Queue ({observation['queue_stats'].get('length', 0)} pending)",
        f"Average wait: {observation['queue_stats'].get('avg_wait_min', 0)} minutes",
        "",
    ]
    for ticket in observation.get("tickets", []):
        lines += [
            f"### {ticket['ticket_id']}",
            f"Subject   : {ticket['subject']}",
            f"Body      : {ticket['body']}",
            f"Tier      : {ticket['customer_tier']}",
            f"Sentiment : {ticket['sentiment']}",
            f"Wait (min): {ticket['wait_time_minutes']}",
            f"Prev contacts: {ticket['previous_contacts']}",
            f"Tags      : {', '.join(ticket.get('tags', []))}",
            "",
        ]
    lines.append("Route ALL tickets listed above. Include every ticket_id in your response.")
    return "\n".join(lines)


def parse_action(response_text: str) -> Dict[str, Any]:
    """Extract JSON action from model response, with fallback."""
    # Try to extract JSON block
    text = response_text.strip()
    # Handle markdown code fences
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        inner = text[start:end].lstrip("`").lstrip("json").strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass
    # Try raw parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback
    print("  [WARN] Could not parse model response, using fallback action.")
    return json.loads(FALLBACK_ACTION)


# ---------------------------------------------------------------------------
# Direct-mode environment (no HTTP)
# ---------------------------------------------------------------------------

def run_task_direct(
    client: OpenAI,
    task_id: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one task episode using the environment Python API directly."""
    # Local import so inference.py works standalone from root
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from env.environment import CustomerSupportRoutingEnv
    from env.models import Action

    env = CustomerSupportRoutingEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    obs_dict = obs.model_dump()

    episode_reward = 0.0
    final_score = 0.0

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"Tickets: {len(obs_dict['tickets'])}")
    print(f"{'='*60}")

    for step_num in range(1, MAX_STEPS + 1):
        if not obs_dict.get("tickets"):
            print(f"  Step {step_num}: No pending tickets. Done.")
            break

        user_message = build_user_message(obs_dict)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        print(f"\n  Step {step_num}: Calling {MODEL_NAME}...")
        t0 = time.time()
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=1500,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [ERROR] API call failed: {exc}")
            response_text = FALLBACK_ACTION

        elapsed = round(time.time() - t0, 2)
        print(f"  Response received in {elapsed}s")

        action_dict = parse_action(response_text)
        print(f"  Routing {len(action_dict.get('routings', []))} tickets")

        # Print routing decisions
        for r in action_dict.get("routings", []):
            print(f"    {r.get('ticket_id')} → {r.get('department')} [{r.get('priority')}]  | {r.get('reason', '')[:60]}")

        # Convert dict to Action model
        try:
            action = Action(**action_dict)
        except Exception as e:
            print(f"  [WARN] Invalid action format: {e}")
            action = Action(routings=[])

        result = env.step(action)
        obs_dict = result.observation.model_dump()
        episode_reward += result.reward

        print(f"  Step reward: {result.reward:+.4f} | Done: {result.done}")
        print(f"  Reward breakdown: {result.info.get('reward_breakdown', {})}")

        if result.done:
            final_score = result.info.get("episode_score", 0.0)
            print(f"\n  ✅ Episode complete — Final task score: {final_score:.4f}")
            break

    env.close()
    return {
        "task_id": task_id,
        "episode_reward": round(episode_reward, 4),
        "final_score": final_score,
    }


# ---------------------------------------------------------------------------
# HTTP-mode (against HF Space or local server)
# ---------------------------------------------------------------------------

def run_task_http(
    client: OpenAI,
    task_id: str,
    base_url: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one task episode via HTTP against a deployed OpenEnv server."""
    import requests

    reset_url = f"{base_url}/reset"
    step_url  = f"{base_url}/step"

    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {"task_id": task_id}
    if seed is not None:
        payload["seed"] = seed

    resp = requests.post(reset_url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    obs_dict = resp.json()

    episode_reward = 0.0
    final_score = 0.0

    print(f"\n{'='*60}")
    print(f"Task (HTTP): {task_id} @ {base_url}")
    print(f"Tickets: {len(obs_dict.get('tickets', []))}")
    print(f"{'='*60}")

    for step_num in range(1, MAX_STEPS + 1):
        if not obs_dict.get("tickets"):
            break

        user_message = build_user_message(obs_dict)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        print(f"\n  Step {step_num}: Calling {MODEL_NAME}...")
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=1500,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [ERROR] API call failed: {exc}")
            response_text = FALLBACK_ACTION

        action_dict = parse_action(response_text)
        step_payload = {"action": action_dict}

        resp = requests.post(step_url, json=step_payload, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        obs_dict = result["observation"]
        episode_reward += result["reward"]
        print(f"  Step {step_num} reward: {result['reward']:+.4f} | Done: {result['done']}")

        if result["done"]:
            final_score = result["info"].get("episode_score", 0.0)
            print(f"\n  ✅ Episode complete — Final task score: {final_score:.4f}")
            break

    return {
        "task_id": task_id,
        "episode_reward": round(episode_reward, 4),
        "final_score": final_score,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline inference for Customer Support Routing OpenEnv")
    parser.add_argument("--task", default=None, help="Run only a specific task ID")
    parser.add_argument("--http", default=None, metavar="URL",
                        help="Run against HTTP server at this base URL (e.g. http://localhost:7860)")
    parser.add_argument("--seed", type=int, default=None, help="Override episode seed")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=API_BASE_URL,
    )

    task_ids = (
        [args.task]
        if args.task
        else ["task1_basic_routing", "task2_priority_routing", "task3_full_triage"]
    )

    results = []
    total_start = time.time()

    for task_id in task_ids:
        try:
            if args.http:
                result = run_task_http(client, task_id, args.http, seed=args.seed)
            else:
                result = run_task_direct(client, task_id, seed=args.seed)
            results.append(result)
        except Exception as exc:
            print(f"\n[ERROR] Task {task_id} failed: {exc}")
            results.append({"task_id": task_id, "episode_reward": 0.0, "final_score": 0.0})

    total_elapsed = round(time.time() - total_start, 1)

    # ---------------------------------------------------------------------------
    # Summary report
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"Model : {MODEL_NAME}")
    print(f"API   : {API_BASE_URL}")
    print(f"Time  : {total_elapsed}s")
    print(f"{'='*60}")
    print(f"{'Task':<35} {'Score':>8} {'Ep.Reward':>12}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['task_id']:<35} {r['final_score']:>8.4f} {r['episode_reward']:>12.4f}")
    print(f"{'-'*60}")
    if results:
        avg_score = sum(r["final_score"] for r in results) / len(results)
        print(f"{'AVERAGE':<35} {avg_score:>8.4f}")
    print(f"{'='*60}\n")

    # Write JSON results for CI / automated evaluation
    with open("baseline_results.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "api_base": API_BASE_URL,
            "elapsed_seconds": total_elapsed,
            "results": results,
            "average_score": avg_score if results else 0.0,
        }, f, indent=2)
    print("Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
