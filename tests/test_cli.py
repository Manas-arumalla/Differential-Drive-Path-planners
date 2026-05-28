"""Tests for the navstack command-line interface."""
import numpy as np

from navstack.__main__ import main, _xy


def test_version_runs(capsys):
    assert main(["version"]) == 0
    assert "navstack" in capsys.readouterr().out


def test_plan_finds_path(capsys):
    rc = main(["plan", "--algo", "A*", "--start", "3,3", "--goal", "45,45"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "length=" in out and "waypoints=" in out


def test_plan_reports_failure_for_blocked_goal():
    # (15, 35) is inside a circular obstacle in the demo environment
    assert main(["plan", "--algo", "A*", "--goal", "15,35"]) == 1


def test_xy_parser():
    assert _xy("3.5,4") == (3.5, 4.0)
