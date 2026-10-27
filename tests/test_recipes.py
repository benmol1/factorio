"""Tests for the recipe data and the recycling derived from it."""

import numpy as np
import pytest

from src.quality.assembler_recycler_loop import assembler_recycler_loop, get_recycler_parameters
from src.quality.pipe_chain import ChainConfig, solve_chain
from src.web.recipes import ITEMS, RECIPES, RECYCLING, recycling


def test_every_item_has_at_most_one_recycling():
    """Alternative recipes mustn't define a second recycling for the same item."""
    producers = {}
    for key, recipe in RECIPES.items():
        if recipe.recyclable:
            for item in recipe.products:
                if not ITEMS[item].fluid:
                    producers.setdefault(item, []).append(key)
    assert all(len(keys) == 1 for keys in producers.values())
    assert set(producers) == set(RECYCLING)


def test_casting_recipes_are_not_recyclable():
    """Pipe to ground is recycled via the assembler recipe, not the foundry one."""
    assert not RECIPES["casting-pipe-to-ground"].recyclable
    assert recycling("pipe-to-ground").recipe == "pipe-to-ground"
    # The casting recipe's molten iron never comes back, while the assembler recipe returns plates too
    assert recycling("pipe-to-ground").outputs == {"iron-plate": pytest.approx(0.625), "pipe": pytest.approx(1.25)}


def test_fluids_have_no_recycling_and_are_never_returned():
    """Recycling skips fluid products and fluid ingredients."""
    assert "molten-iron" not in RECYCLING
    assert recycling("battery").outputs == {"iron-plate": 0.25, "copper-plate": 0.25}


def test_recycle_time_defaults_to_a_sixteenth_per_craft():
    """Time is per craft, which consumes as many items as the recipe makes; the loops want it per item."""
    ptg = recycling("pipe-to-ground")
    assert ptg.time == pytest.approx(0.5 / 16)
    assert ptg.items_per_craft == 2
    assert ptg.time_per_item == pytest.approx(0.5 / 32)


def test_recycle_ratio_defaults_to_reversing_the_recipe():
    """Without `recycle_ratio` recycling is the assembly recipe in reverse, as before."""
    assert get_recycler_parameters(recipe_ratio=0.2) == get_recycler_parameters(recipe_ratio=0.2, recycle_ratio=1.25)
    assert get_recycler_parameters(recipe_ratio=0.2, recycle_ratio=0)[0][1] == 0


def test_loop_recycle_ratio_override():
    """A recycler that returns nothing of the loop ingredient lets nothing back in."""
    kwargs = {
        "input_vector": np.array([1.0] + [0.0] * 9),
        "assembler_modules_config": (0, 4),
        "product_quality_to_keep": 5,
        "ingredient_quality_to_keep": None,
        "base_prod_bonus": 1.5,
        "recipe_ratio": 0.2,
    }
    closed = assembler_recycler_loop(**kwargs)[0]
    open_loop = assembler_recycler_loop(**kwargs, recycle_ratio=0)[0]
    assert closed[9] > open_loop[9]
    assert open_loop[:5].sum() == pytest.approx(1.0)


def test_pipe_chain_balances():
    """Every pipe to ground made is recycled, and molten iron use matches what the lava foundries make."""
    cfg = ChainConfig(lava_per_second=100)
    res = solve_chain(cfg)
    ptg_made = res["pipes"].sum() / 10 * 2 * 1.5  # 10 pipes -> 2 pipe to ground, +50% foundry productivity
    assert res["pipe_to_ground"].sum() == pytest.approx(ptg_made)
    # Plates returned are 0.625 per pipe to ground recycled
    assert res["plates"].sum() == pytest.approx(0.625 * res["pipe_to_ground"].sum())
    assert res["legendary_plates"] > 0
    # Scales linearly with lava
    assert solve_chain(ChainConfig(lava_per_second=200))["legendary_plates"] == pytest.approx(
        2 * res["legendary_plates"]
    )


def test_pipe_chain_assembling_low_plates_gives_more_legendary():
    """Re-assembling normal plates too feeds more pipes back in."""
    assert (
        solve_chain(ChainConfig(min_plate_tier=0))["legendary_plates"]
        > solve_chain(ChainConfig(min_plate_tier=1))["legendary_plates"]
    )


def test_pipe_chain_legendary_machines_shrink_loads_not_flows():
    """Machine quality is speed only: flows are unchanged and each machine count drops by the 2.5x speed bonus."""
    normal = solve_chain(ChainConfig())
    legendary = solve_chain(ChainConfig(machine_quality=4, recycler_quality=4))
    assert legendary["legendary_plates"] == pytest.approx(normal["legendary_plates"])
    for name, load in normal["loads"].items():
        assert legendary["loads"][name] == pytest.approx(load / 2.5)
