from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.quality.assembler_recycler_loop import assembler_recycler_loop
from src.quality.utils import BEST_PROD_MODULE, BEST_QUAL_MODULE, NUM_TIERS

# Flows are floats, so a machine that is loaded to exactly 100% shouldn't round up to another machine
TOLERANCE = 1e-9


@dataclass(frozen=True)
class QualitySolution:
    """Cheapest machine quality and count for each stage, within a machine limit."""

    assembler_qualities: list[int]
    num_assemblers: list[int]
    recycler_quality: int
    num_recyclers: int
    # False if a stage needs more machines than the limit even at the best quality (it is then built at the best)
    fits: bool


@dataclass(frozen=True)
class SolverResult:
    """The smallest machine counts that keep every machine at or below its capacity."""

    num_assemblers: list[int]
    num_recyclers: int
    # Flow through each stage, in items per second. Kept tiers don't go through the machines at all
    assembler_flows: list[float]
    recycler_flow: float


def solve_machine_counts(
    input_vector: np.ndarray | float,
    assembler_modules_config: tuple[float, float] | list[tuple[float, float]],
    product_quality_to_keep: int | None = NUM_TIERS,
    ingredient_quality_to_keep: int | None = NUM_TIERS,
    base_prod_bonus: float = 0,
    recipe_ratio: float = 1,
    recipe_time: float = 1,
    prod_module_bonus: float = BEST_PROD_MODULE,
    qual_module_bonus: float = BEST_QUAL_MODULE,
    speed_assemblers: list | None = None,
    speed_recycler: float = 0.5,
    recycle_time: float | None = None,
    recycler_qual_module_bonus: float | None = None,
) -> SolverResult:
    """Find the fewest assemblers (per ingredient tier) and recyclers that leave nothing bottlenecked.

    The flow through the loop depends on the input, modules and what's kept, but not on how many machines there are.
    Machine counts only set the capacity, so each stage needs `ceil(flow / capacity of one machine)` machines.

    The machine quality is whatever the speeds describe: pass the speeds of the lowest quality machines you have
    (Normal by default) to get the cheapest build at that quality. The "target output" is chosen the same way as in
    the simulation, with the `*_quality_to_keep` arguments.

    Args:
        input_vector: The ingredients and items intake of the system.
        assembler_modules_config: Module configuration (prod, qual) per tier or single tuple.
        product_quality_to_keep: Min quality tier to extract products. None = recycle all.
        ingredient_quality_to_keep: Min quality tier to extract ingredients. None = assemble all.
        base_prod_bonus: Base productivity from machine + technologies.
        recipe_ratio: Ratio of products to ingredients in the recipe.
        recipe_time: Base recipe crafting time in seconds.
        prod_module_bonus: Productivity bonus per productivity module.
        qual_module_bonus: Quality chance bonus per quality module.
        speed_assemblers: Assembler speed per ingredient tier. Defaults to 1 (Normal quality, no speed modules).
        speed_recycler: Speed of the recyclers.
        recycle_time: Base time in seconds of the product's recycling recipe. Defaults to `recipe_time / 16`.
        recycler_qual_module_bonus: Quality chance bonus per quality module in the recyclers. Defaults to
            `qual_module_bonus`.

    Returns:
        A SolverResult. Tiers with no flow or whose quality is kept need zero assemblers.
    """
    if speed_assemblers is None:
        speed_assemblers = [1] * NUM_TIERS

    # One machine per stage makes the "max flow" vector the capacity of a single machine
    flows, _, _, single_machine_max_flow = assembler_recycler_loop(
        input_vector=input_vector,
        assembler_modules_config=assembler_modules_config,
        product_quality_to_keep=product_quality_to_keep,
        ingredient_quality_to_keep=ingredient_quality_to_keep,
        base_prod_bonus=base_prod_bonus,
        recipe_ratio=recipe_ratio,
        recipe_time=recipe_time,
        prod_module_bonus=prod_module_bonus,
        qual_module_bonus=qual_module_bonus,
        speed_assemblers=speed_assemblers,
        speed_recycler=speed_recycler,
        num_assemblers=[1] * NUM_TIERS,
        num_recyclers=1,
        recycle_time=recycle_time,
        recycler_qual_module_bonus=recycler_qual_module_bonus,
    )

    # Kept ingredient tiers are taken out of the loop rather than assembled
    first_kept_ingredient = NUM_TIERS if ingredient_quality_to_keep is None else ingredient_quality_to_keep - 1
    assembler_flows = [0.0 if i >= first_kept_ingredient else float(flows[i]) for i in range(NUM_TIERS)]
    num_assemblers = [_machines_needed(flow, single_machine_max_flow[i]) for i, flow in enumerate(assembler_flows)]

    # Kept products leave the loop; everything else goes through the recyclers
    recycled = NUM_TIERS if product_quality_to_keep is None else product_quality_to_keep - 1
    recycler_flow = float(flows[NUM_TIERS : NUM_TIERS + recycled].sum())
    num_recyclers = _machines_needed(recycler_flow, single_machine_max_flow[NUM_TIERS])

    return SolverResult(num_assemblers, num_recyclers, assembler_flows, recycler_flow)


def _machines_needed(flow: float, single_machine_capacity: float) -> int:
    """Whole machines needed to carry a flow, given what one machine can handle."""
    return max(0, math.ceil(flow / single_machine_capacity - TOLERANCE))


def solve_machine_quality(
    assembler_speeds_by_quality: list[list[float]],
    recycler_speeds_by_quality: list[float],
    max_assemblers: int,
    max_recyclers: int,
    **loop,
) -> QualitySolution:
    """Pick the lowest machine quality per stage that fits within a machine limit, with the fewest machines.

    Each stage is solved independently, since the flows don't depend on the machines. For each stage the qualities
    are tried from lowest to highest, and the first whose solved count is within the limit wins. If even the best
    quality needs more machines than allowed, the stage is built at the best quality and `fits` is False.

    Args:
        assembler_speeds_by_quality: Assembler speed per ingredient tier for each machine quality, indexed
            `[quality][tier]`. Speeds already include any module penalties.
        recycler_speeds_by_quality: Recycler speed for each machine quality.
        max_assemblers: Most assemblers allowed for any one ingredient tier.
        max_recyclers: Most recyclers allowed.
        **loop: The loop arguments of `solve_machine_counts`, other than the speeds.

    Returns:
        A QualitySolution.
    """
    solutions = [
        solve_machine_counts(
            **loop, speed_assemblers=assembler_speeds_by_quality[q], speed_recycler=recycler_speeds_by_quality[q]
        )
        for q in range(len(recycler_speeds_by_quality))
    ]
    best = len(solutions) - 1

    def cheapest(counts: list[int], limit: int) -> tuple[int, bool]:
        """Lowest quality whose count is within the limit, or the best quality if none is."""
        for q, n in enumerate(counts):
            if n <= limit:
                return q, True
        return best, False

    qualities, counts, fits = [], [], []
    for tier in range(NUM_TIERS):
        q, ok = cheapest([sol.num_assemblers[tier] for sol in solutions], max_assemblers)
        qualities.append(q)
        counts.append(solutions[q].num_assemblers[tier])
        fits.append(ok)

    recycler_quality, ok = cheapest([sol.num_recyclers for sol in solutions], max_recyclers)
    fits.append(ok)

    return QualitySolution(qualities, counts, recycler_quality, solutions[recycler_quality].num_recyclers, all(fits))
