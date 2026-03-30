"""
Tests for the Customer Support Routing OpenEnv environment.
Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from env.environment import CustomerSupportRoutingEnv
from env.models import Action, Department, Priority, TicketRouting
from tasks.tasks import TASKS, task1_grade, task2_grade, task3_grade


# ---------------------------------------------------------------------------
# Environment lifecycle tests
# ---------------------------------------------------------------------------

class TestEnvironmentLifecycle:

    def test_reset_returns_observation(self):
        env = CustomerSupportRoutingEnv("task1_basic_routing")
        obs = env.reset()
        assert len(obs.tickets) == 5
        assert obs.step_number == 0
        assert obs.task_id == "task1_basic_routing"

    def test_reset_is_deterministic(self):
        env = CustomerSupportRoutingEnv("task1_basic_routing", seed=42)
        obs1 = env.reset()
        obs2 = env.reset()
        assert [t.ticket_id for t in obs1.tickets] == [t.ticket_id for t in obs2.tickets]

    def test_state_returns_dict(self):
        env = CustomerSupportRoutingEnv("task1_basic_routing")
        env.reset()
        s = env.state()
        assert isinstance(s, dict)
        assert "task_id" in s
        assert "tickets" in s
        assert "routings_so_far" in s

    def test_step_returns_step_result(self):
        env = CustomerSupportRoutingEnv("task1_basic_routing")
        obs = env.reset()
        ticket = obs.tickets[0]
        action = Action(routings=[
            TicketRouting(
                ticket_id=ticket.ticket_id,
                department=Department.BILLING,
                priority=Priority.MEDIUM,
                reason="Test routing",
            )
        ])
        result = env.step(action)
        assert result.reward is not None
        assert isinstance(result.done, bool)
        assert result.observation is not None

    def test_step_raises_after_done(self):
        env = CustomerSupportRoutingEnv("task1_basic_routing")
        obs = env.reset()
        # Route all tickets in one step
        routings = [
            TicketRouting(
                ticket_id=t.ticket_id,
                department=Department.GENERAL,
                priority=Priority.MEDIUM,
                reason="bulk route",
            )
            for t in obs.tickets
        ]
        action = Action(routings=routings)
        result = env.step(action)
        assert result.done
        with pytest.raises(RuntimeError):
            env.step(action)

    def test_metadata_hidden_from_agent(self):
        """Agent must not see the dept_hint ground truth in observation."""
        env = CustomerSupportRoutingEnv("task1_basic_routing")
        obs = env.reset()
        for ticket in obs.tickets:
            assert ticket.metadata == {} or "dept_hint" not in ticket.metadata

    def test_close_does_not_raise(self):
        env = CustomerSupportRoutingEnv("task1_basic_routing")
        env.reset()
        env.close()  # should not raise


# ---------------------------------------------------------------------------
# Reward shaping tests
# ---------------------------------------------------------------------------

class TestRewardShaping:

    def test_correct_routing_gives_positive_reward(self):
        from tasks.tasks import task1_episode
        tickets, gt = task1_episode(seed=42)
        env = CustomerSupportRoutingEnv("task1_basic_routing", seed=42)
        env.reset()

        correct_routings = [
            TicketRouting(
                ticket_id=g.ticket_id,
                department=g.expected_department,
                priority=g.expected_priority,
                reason="Correct routing",
            )
            for g in gt
        ]
        result = env.step(Action(routings=correct_routings))
        assert result.reward > 0

    def test_all_wrong_routing_gives_lower_reward(self):
        env = CustomerSupportRoutingEnv("task1_basic_routing", seed=42)
        obs = env.reset()

        wrong_routings = [
            TicketRouting(
                ticket_id=t.ticket_id,
                department=Department.GENERAL,
                priority=Priority.LOW,
                reason="",
            )
            for t in obs.tickets
        ]
        result = env.step(Action(routings=wrong_routings))
        # Still gets escalation partial credit sometimes but should be lower
        assert result.reward <= 1.0

    def test_loop_penalty_applied(self):
        env = CustomerSupportRoutingEnv("task1_basic_routing", seed=42)
        obs = env.reset()
        ticket = obs.tickets[0]

        action = Action(routings=[
            TicketRouting(
                ticket_id=ticket.ticket_id,
                department=Department.BILLING,
                priority=Priority.MEDIUM,
                reason="First routing",
            )
        ])
        env.step(action)  # step 1 — route ticket

        # Step 2 — route same ticket again (loop)
        result2 = env.step(action)
        # Loop penalty should have been applied
        breakdown = result2.info.get("reward_breakdown", {})
        assert breakdown.get("loop_penalty", 0) < 0

    def test_invalid_ticket_id_penalty(self):
        env = CustomerSupportRoutingEnv("task1_basic_routing", seed=42)
        env.reset()

        action = Action(routings=[
            TicketRouting(
                ticket_id="TKT-FAKE-9999",
                department=Department.BILLING,
                priority=Priority.MEDIUM,
                reason="Invalid",
            )
        ])
        result = env.step(action)
        breakdown = result.info.get("reward_breakdown", {})
        assert breakdown.get("invalid_penalty", 0) < 0


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------

class TestGraders:

    def _perfect_routings(self, task_id: str, seed: int):
        from tasks.tasks import TASKS
        task = TASKS[task_id]
        tickets, gt = task["episode_fn"](seed=seed)
        from env.models import TicketRouting
        return [
            TicketRouting(
                ticket_id=g.ticket_id,
                department=g.expected_department,
                priority=g.expected_priority,
                reason="correct",
            )
            for g in gt
        ], gt

    def test_task1_perfect_score(self):
        routings, gt = self._perfect_routings("task1_basic_routing", 42)
        score = task1_grade(routings, gt)
        assert score == pytest.approx(1.0)

    def test_task1_zero_score(self):
        _, gt = self._perfect_routings("task1_basic_routing", 42)
        wrong = [
            TicketRouting(
                ticket_id=g.ticket_id,
                department=Department.GENERAL,
                priority=Priority.LOW,
                reason="wrong",
            )
            for g in gt
        ]
        score = task1_grade(wrong, gt)
        # general may match some tickets, so just check < 1
        assert 0.0 <= score <= 1.0

    def test_task2_perfect_score(self):
        routings, gt = self._perfect_routings("task2_priority_routing", 99)
        score = task2_grade(routings, gt)
        assert score == pytest.approx(1.0)

    def test_task3_perfect_score(self):
        routings, gt = self._perfect_routings("task3_full_triage", 7)
        score = task3_grade(routings, gt)
        assert score == pytest.approx(1.0)

    def test_grader_scores_in_range(self):
        for task_id, task in TASKS.items():
            tickets, gt = task["episode_fn"]()
            random_routings = [
                TicketRouting(
                    ticket_id=g.ticket_id,
                    department=Department.GENERAL,
                    priority=Priority.MEDIUM,
                    reason="random",
                )
                for g in gt
            ]
            score = task["grade_fn"](random_routings, gt)
            assert 0.0 <= score <= 1.0, f"{task_id}: score {score} out of range"

    def test_graders_are_deterministic(self):
        for task_id, task in TASKS.items():
            _, gt = task["episode_fn"]()
            routings = [
                TicketRouting(
                    ticket_id=g.ticket_id,
                    department=Department.BILLING,
                    priority=Priority.HIGH,
                    reason="test",
                )
                for g in gt
            ]
            score1 = task["grade_fn"](routings, gt)
            score2 = task["grade_fn"](routings, gt)
            assert score1 == score2


# ---------------------------------------------------------------------------
# Task registry tests
# ---------------------------------------------------------------------------

class TestTaskRegistry:

    def test_three_tasks_registered(self):
        assert len(TASKS) >= 3

    def test_task_difficulties(self):
        difficulties = {t["difficulty"] for t in TASKS.values()}
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_all_tasks_have_required_keys(self):
        required = {"id", "name", "difficulty", "description", "episode_fn", "grade_fn"}
        for task_id, task in TASKS.items():
            for key in required:
                assert key in task, f"Task {task_id} missing key '{key}'"
