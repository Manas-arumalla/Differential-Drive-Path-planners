"""Tests for reciprocal-RVO multi-robot swarm coordination."""
import pytest

from navstack.controllers.swarm import simulate_circle_swap


@pytest.mark.parametrize("n", [4, 6, 8])
def test_circle_swap_is_collision_free_and_reaches_goals(n):
    res = simulate_circle_swap(n_robots=n, steps=600)
    assert res["all_reached"], f"{n} robots did not all reach their goals"
    assert res["min_clearance"] > 0.0, f"{n} robots collided (clearance {res['min_clearance']})"
