"""Tests for the machine count solver."""

import numpy as np
import pytest

from src.quality import NUM_TIERS
from src.quality.assembler_recycler_loop import assembler_recycler_loop
from src.quality.solver import solve_machine_counts

# Uranium ammo loop: legendary ingredients are kept and every product is recycled
U238_LOOP = {
    "input_vector": np.array([8.0] + [0.0] * (2 * NUM_TIERS - 1)),
    "assembler_modules_config": [(0, 4)] * (NUM_TIERS - 1) + [(4, 0)],
    "product_quality_to_keep": None,
    "ingredient_quality_to_keep": NUM_TIERS,
    "base_prod_bonus": 1.0,
    "recipe_time": 10,
    "prod_module_bonus": 0.0,
    "speed_assemblers": [1.25] * NUM_TIERS,
    "speed_recycler": 0.5,
}


def loads(counts_assemblers: list[int], counts_recyclers: int, **loop):
    """Assembler and recycler load (flow / capacity) for the given machine counts."""
    flows, _, _, max_flow = assembler_recycler_loop(
        **loop, num_assemblers=counts_assemblers, num_recyclers=counts_recyclers
    )
    kept = loop["ingredient_quality_to_keep"]
    assembled = NUM_TIERS if kept is None else kept - 1
    assembler_loads = [flows[i] / max_flow[i] for i in range(assembled)]
    return assembler_loads, flows[NUM_TIERS:].sum() / max_flow[NUM_TIERS]


class TestSolveMachineCounts:
    """Tests for solve_machine_counts."""

    def test_no_bottlenecks(self):
        """Every machine stays at or below 100% load with the solved counts."""
        result = solve_machine_counts(**U238_LOOP)
        # Counts are padded to 1 for tiers that need none, which the simulation needs to avoid dividing by zero
        counts = [max(n, 1) for n in result.num_assemblers]
        assembler_loads, recycler_load = loads(counts, result.num_recyclers, **U238_LOOP)
        assert all(load <= 1 + 1e-9 for load in assembler_loads)
        assert recycler_load <= 1 + 1e-9

    def test_one_fewer_machine_is_a_bottleneck(self):
        """Removing any single machine from a stage overloads it, so the counts are minimal."""
        result = solve_machine_counts(**U238_LOOP)
        counts = [max(n, 1) for n in result.num_assemblers]

        for tier, n in enumerate(counts):
            if n > 1:
                fewer = counts.copy()
                fewer[tier] -= 1
                assembler_loads, _ = loads(fewer, result.num_recyclers, **U238_LOOP)
                assert assembler_loads[tier] > 1
        if result.num_recyclers > 1:
            _, recycler_load = loads(counts, result.num_recyclers - 1, **U238_LOOP)
            assert recycler_load > 1

    def test_kept_tier_needs_no_assemblers(self):
        """The kept legendary tier is extracted, not assembled."""
        result = solve_machine_counts(**U238_LOOP)
        assert result.num_assemblers[4] == 0
        assert result.assembler_flows[4] == 0

    def test_faster_machines_need_fewer(self):
        """Doubling every machine's speed roughly halves the machine counts."""
        slow = solve_machine_counts(**U238_LOOP)
        fast = solve_machine_counts(**{**U238_LOOP, "speed_assemblers": [2.5] * NUM_TIERS, "speed_recycler": 1.0})
        assert fast.num_recyclers <= slow.num_recyclers
        assert sum(fast.num_assemblers) < sum(slow.num_assemblers)

    def test_flow_scales_linearly_with_input(self):
        """Ten times the input needs about ten times the machines."""
        base = solve_machine_counts(**U238_LOOP)
        scaled = solve_machine_counts(**{**U238_LOOP, "input_vector": U238_LOOP["input_vector"] * 10})
        assert scaled.num_recyclers <= 10 * base.num_recyclers
        assert scaled.num_recyclers > 9 * (base.num_recyclers - 1)

    def test_no_input_needs_no_machines(self):
        """Nothing flowing means nothing to build."""
        result = solve_machine_counts(**{**U238_LOOP, "input_vector": np.zeros(2 * NUM_TIERS)})
        assert result.num_assemblers == [0] * NUM_TIERS
        assert result.num_recyclers == 0

    def test_keeping_everything_needs_no_assemblers(self):
        """With every ingredient tier kept, nothing is assembled."""
        loop = {**U238_LOOP, "assembler_modules_config": [(0, 0)] * NUM_TIERS, "ingredient_quality_to_keep": 1}
        result = solve_machine_counts(**loop)
        assert result.num_assemblers == [0] * NUM_TIERS

    @pytest.mark.parametrize("keep", [None, 1, 3])
    def test_other_keep_settings_are_feasible(self, keep):
        """The solved counts hold for different ingredient keep settings."""
        loop = {**U238_LOOP, "ingredient_quality_to_keep": keep}
        result = solve_machine_counts(**loop)
        counts = [max(n, 1) for n in result.num_assemblers]
        assembler_loads, recycler_load = loads(counts, max(result.num_recyclers, 1), **loop)
        assert all(load <= 1 + 1e-9 for load in assembler_loads)
        assert recycler_load <= 1 + 1e-9
