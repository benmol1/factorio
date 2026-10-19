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
        "tiers": [
            {"count": n, "machine_quality": LEGENDARY, "prod_modules": 0, "quality_modules": 4} for n in [40, 4, 2, 1]
        ]
        + [{"count": 1, "machine_quality": LEGENDARY, "prod_modules": 4, "quality_modules": 0}],
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


def test_recipes_served():
    """Recipe data comes with the game data."""
    data = client.get("/api/game-data").json()
    toolbelt = data["recipes"]["toolbelt-equipment"]
    assert toolbelt["time"] == 10
    assert toolbelt["ingredients"] == {"advanced-circuit": 3, "carbon-fiber": 10}
    assert data["recipes"]["carbon-fiber"]["machines"] == ["biochamber"]
    assert all(m in data["machines"] for r in data["recipes"].values() for m in r["machines"])


def test_recipe_ratio():
    """The loop's ratio is products per unit of the upcycled ingredient."""
    from src.web.recipes import RECIPES

    assert RECIPES["toolbelt-equipment"].ratio("carbon-fiber") == pytest.approx(0.1)
    assert RECIPES["carbon-fiber"].ratio("carbon") == pytest.approx(1)


def test_assembler_loop_rejects_wrong_machine_for_recipe():
    """A recipe can only be crafted in its own machines."""
    assert client.post("/api/assembler-loop", json=u238_request(recipe="carbon-fiber")).status_code == 422
    ok = u238_request(recipe="toolbelt-equipment", recipe_ratio=0.1)
    assert client.post("/api/assembler-loop", json=ok).status_code == 200


def test_solve_builds_have_no_bottlenecks():
    """Simulating the solver's build shows every machine at or below capacity."""
    req = u238_request()
    solved = client.post("/api/solve", json=req).json()

    req["tiers"] = [
        t | {"count": max(s["count"], 1), "machine_quality": s["machine_quality"]}
        for t, s in zip(req["tiers"], solved["tiers"], strict=True)
    ]
    req["recycler"] |= {"count": solved["recycler"]["count"], "quality": solved["recycler"]["quality"]}
    result = client.post("/api/assembler-loop", json=req).json()

    ingredients = result["ingredients"]
    for flow, max_flow, kept in zip(ingredients["flows"], ingredients["max_flow"], ingredients["kept"], strict=True):
        assert kept or flow <= max_flow + 1e-9
    assert result["recycler"]["flow"] <= result["recycler"]["max_flow"] + 1e-9


def test_solve_ignores_the_machines_in_the_request():
    """The solver picks counts and qualities itself, whatever machines the request describes."""
    base = client.post("/api/solve", json=u238_request()).json()
    other = u238_request()
    other["tiers"] = [t | {"count": 1, "machine_quality": 0} for t in other["tiers"]]
    other["recycler"] |= {"count": 1, "quality": 0}
    assert client.post("/api/solve", json=other).json() == base


def test_recipes_grouped():
    """Every recipe belongs to one of the in-game tabs the picker lists."""
    data = client.get("/api/game-data").json()
    assert data["groups"] == ["Production", "Logistics", "Intermediate", "Combat", "Fluids"]
    assert all(r["group"] in data["groups"] for r in data["recipes"].values())


def test_pipe_chain_matches_solver_and_rejects_bad_modules():
    """The endpoint agrees with solve_chain and enforces the 4 module slots."""
    from src.quality.pipe_chain import solve_chain

    res = client.post("/api/pipe-chain", json={}).json()
    assert res["legendary_plates"] == pytest.approx(solve_chain()["legendary_plates"])
    assert client.post("/api/pipe-chain", json={"pipe_foundry": {"prod": 3, "quality": 3}}).status_code == 422


def test_recycler_solve_leaves_no_bottleneck_and_is_minimal():
    """The solved recycler setup is within capacity, and one fewer recycler wouldn't be."""
    body = {
        "input_rates": [45, 15, 0, 0, 0],
        "recycler": {"quality": 0, "count": 1, "quality_module": LEGENDARY_Q3},
        "recipe_time": 1,
    }
    solved = client.post("/api/recycler-solve", json=body).json()["recycler"]

    def load(count, quality):
        req = {**body, "recycler": {**body["recycler"], "count": count, "quality": quality}}
        return client.post("/api/recycler-loop", json=req).json()["recycler_load"]

    assert load(solved["count"], solved["quality"]) <= 1 + 1e-9
    assert solved["count"] == 1 or load(solved["count"] - 1, solved["quality"]) > 1
