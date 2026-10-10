"""Tests for the web frontend's API and game-data conversions."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.web.app import app
from src.web.game_data import machine_speed, recycler_speed

client = TestClient(app)

LEGENDARY = 4
LEGENDARY_Q3 = {"level": 3, "quality": LEGENDARY}


def u238_request(**overrides):
    """The README's legendary U-238 example, in game terms."""
    req = {
        "input_rates": [8, 0, 0, 0, 0],
        "machine": "assembling-machine-3",
        "machine_quality": LEGENDARY,
        "tiers": [{"count": n, "prod_modules": 0, "quality_modules": 4} for n in [40, 4, 2, 1]]
        + [{"count": 1, "prod_modules": 4, "quality_modules": 0}],
        "prod_module": LEGENDARY_Q3,
        "quality_module": LEGENDARY_Q3,
        "recycler": {"quality": LEGENDARY, "count": 8, "quality_module": LEGENDARY_Q3},
        "recipe_time": 10,
        "ingredient_quality_to_keep": LEGENDARY,
        "product_quality_to_keep": None,
    }
    return req | overrides


def test_recycler_speed_matches_library_defaults():
    """Recycler speeds line up with the library defaults and demos."""
    assert recycler_speed(0) == pytest.approx(0.4)
    assert recycler_speed(LEGENDARY) == pytest.approx(1.0)


def test_machine_speed_floor():
    """Module penalties can't slow a machine below 20%."""
    assert machine_speed(1.0, 0, 0.95) == pytest.approx(0.2)


def test_index_and_game_data_served():
    """The page and game data are served."""
    assert client.get("/").status_code == 200
    data = client.get("/api/game-data").json()
    assert data["machines"]["assembling-machine-3"]["slots"] == 4


def test_recycler_loop_matches_tungsten_example():
    """The tungsten preset reproduces the README example."""
    res = client.post(
        "/api/recycler-loop",
        json={
            "input_rates": [45, 15, 0, 0, 0],
            "recycler": {"quality": LEGENDARY, "count": 5, "quality_module": LEGENDARY_Q3},
            "recipe_time": 1,
            "quality_to_keep": LEGENDARY,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["production_rate"][LEGENDARY] * 3600 == pytest.approx(176.8, abs=0.1)
    assert np.allclose(data["flows"], [55.42, 22.28, 1.91, 0.32, 0.05], atol=0.01)
    assert data["kept"] == [False, False, False, False, True]


def test_assembler_loop_matches_u238_example():
    """The U-238 preset reproduces the README example."""
    res = client.post("/api/assembler-loop", json=u238_request())
    assert res.status_code == 200
    data = res.json()
    assert np.allclose(data["ingredients"]["flows"], [9.32, 0.91, 0.32, 0.09, 0.03], atol=0.01)
    assert np.allclose(data["ingredients"]["max_flow"][:4], [10, 1, 0.5, 0.25])
    assert data["recycler"]["flow"] == pytest.approx(10.63, abs=0.01)
    assert data["recycler"]["max_flow"] == pytest.approx(12.8)


def test_assembler_loop_rejects_too_many_modules():
    """Module counts beyond the machine's slots are rejected."""
    req = u238_request()
    req["tiers"][0]["quality_modules"] = 5
    assert client.post("/api/assembler-loop", json=req).status_code == 422


def test_separate_recycler_modules_change_result():
    """Weaker recycler modules reduce legendary output."""
    weaker = u238_request(recycler={"quality": LEGENDARY, "count": 8, "quality_module": {"level": 1, "quality": 0}})
    base = client.post("/api/assembler-loop", json=u238_request()).json()
    weak = client.post("/api/assembler-loop", json=weaker).json()
    assert weak["ingredients"]["flows"][LEGENDARY] < base["ingredients"]["flows"][LEGENDARY]
