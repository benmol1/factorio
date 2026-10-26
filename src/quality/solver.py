from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from src.quality.assembler_recycler_loop import (
    assembler_recycler_loop,
    create_transition_matrix,
    get_assembler_parameters,
    get_recycler_parameters,
)
from src.quality.pure_recycler_loop import recycler_loop
from src.quality.utils import BEST_PROD_MODULE, BEST_QUAL_MODULE, NUM_TIERS, create_production_matrix

# Flows are floats, so a machine that is loaded to exactly 100% shouldn't round up to another machine
TOLERANCE = 1e-9


@dataclass(frozen=True)
class QualitySolution:
    """Fewest machines for each stage, at the lowest machine quality that achieves that count."""

    assembler_qualities: list[int]
    num_assemblers: list[int]
    recycler_quality: int
    num_recyclers: int


@dataclass(frozen=True)
class ModuleOptimum:
    """The module split that maximises the kept output of an assembler-recycler loop."""

    # (productivity, quality) modules per ingredient tier. None for tiers that aren't assembled (kept ingredients)
    modules: list[tuple[int, int] | None]
    # Kept output in items per second (see `optimise_modules` for which items count)
    output: float


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
    recycle_ratio: float | None = None,
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
        recycle_ratio: Loop ingredients returned per product recycled. Defaults to `0.25 / recipe_ratio`.

    Returns:
        A SolverResult. Tiers with no flow or whose quality is kept need zero assemblers.
    """
    if speed_assemblers is None:
        speed_assemblers = [1] * NUM_TIERS

    # One machine per stage makes the "max flow" vector the capacity of a single machine
    flows, _, single_machine_max_flow = assembler_recycler_loop(
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
        recycle_ratio=recycle_ratio,
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


def solve_recycler_count(speed_recycler: float, **loop) -> int:
    """Fewest recyclers that carry a pure recycler loop, at least 1.

    Args:
        speed_recycler: Speed of one recycler.
        **loop: The arguments of `recycler_loop`, other than the speed and recycler count.
    """
    # With one recycler, the total crafting time is the fraction of a recycler the loop needs
    _, _, load = recycler_loop(**loop, speed_recycler=speed_recycler, num_recyclers=1)
    return max(1, math.ceil(load - TOLERANCE))


def solve_recycler_quality(speed_by_quality: list[float], **loop) -> tuple[int, int]:
    """Fewest recyclers for a pure recycler loop, then the lowest recycler quality that still needs no more.

    Returns:
        (recycler quality, recycler count).
    """
    counts = [solve_recycler_count(speed, **loop) for speed in speed_by_quality]
    quality = counts.index(counts[-1])
    return quality, counts[quality]


def solve_machine_quality(
    assembler_speeds_by_quality: list[list[float]],
    recycler_speeds_by_quality: list[float],
    **loop,
) -> QualitySolution:
    """Pick the fewest machines per stage, then the lowest machine quality that still needs no more of them.

    Each stage is solved independently, since the flows don't depend on the machines. The footprint comes first:
    a stage's machine count is the fewest it can have, which is what the best quality needs. Then the quality is
    lowered for as long as the count doesn't grow, so the machines are as loaded as possible while staying at or below
    100%.

    Args:
        assembler_speeds_by_quality: Assembler speed per ingredient tier for each machine quality, indexed
            `[quality][tier]`. Speeds already include any module penalties.
        recycler_speeds_by_quality: Recycler speed for each machine quality.
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

    def lowest_quality(counts: list[int]) -> int:
        """Lowest quality that needs as few machines as the best quality does."""
        return counts.index(counts[best])

    qualities = [lowest_quality([sol.num_assemblers[tier] for sol in solutions]) for tier in range(NUM_TIERS)]
    counts = [solutions[q].num_assemblers[tier] for tier, q in enumerate(qualities)]
    recycler_quality = lowest_quality([sol.num_recyclers for sol in solutions])

    return QualitySolution(qualities, counts, recycler_quality, solutions[recycler_quality].num_recyclers)


def optimise_modules(
    input_vector: np.ndarray,
    slots: int,
    product_quality_to_keep: int | None = NUM_TIERS,
    ingredient_quality_to_keep: int | None = NUM_TIERS,
    base_prod_bonus: float = 0,
    recipe_ratio: float = 1,
    prod_module_bonus: float = BEST_PROD_MODULE,
    qual_module_bonus: float = BEST_QUAL_MODULE,
    recycler_qual_module_bonus: float | None = None,
    recycle_ratio: float | None = None,
) -> ModuleOptimum:
    """Find the productivity/quality module split in each assembler tier that maximises the kept output.

    For a fixed input rate, the output is a function of the module split alone, so this tries every split. Module
    speed penalties and machine counts only set capacity, not flow, so they don't matter here.

    Every slot is filled: an empty slot is never better than a productivity module, so a split is just the number of
    quality modules `q` in 0..slots, with `slots - q` productivity modules. Each assembled tier picks its own split
    (the best split differs by tier), which is `(slots + 1) ** assembled tiers` loops. The loop is linear in the input,
    so the winner doesn't depend on the input rate.

    The output is the flow of the kept products if `product_quality_to_keep` is set, otherwise of the kept ingredients.
    Kept tiers are summed (e.g. keeping Epic and up counts both). If both are kept, only the products are optimised.

    Args:
        input_vector: The ingredients and items intake of the system.
        slots: Module slots in the assemblers.
        product_quality_to_keep: Min quality tier to extract products. None = recycle all.
        ingredient_quality_to_keep: Min quality tier to extract ingredients. None = assemble all.
        base_prod_bonus: Base productivity from machine + technologies.
        recipe_ratio: Ratio of products to ingredients in the recipe.
        prod_module_bonus: Productivity bonus per productivity module.
        qual_module_bonus: Quality chance bonus per quality module.
        recycler_qual_module_bonus: Quality chance bonus per quality module in the recyclers. Defaults to
            `qual_module_bonus`.
        recycle_ratio: Loop ingredients returned per product recycled. Defaults to `0.25 / recipe_ratio`.

    Returns:
        A ModuleOptimum.

    Raises:
        ValueError: If nothing is kept, so there is no output to maximise.
    """
    if product_quality_to_keep is None and ingredient_quality_to_keep is None:
        raise ValueError("Keep some products or ingredients, or there is no output to maximise.")

    ingredient_keep = NUM_TIERS + 1 if ingredient_quality_to_keep is None else ingredient_quality_to_keep
    assembled = min(ingredient_keep - 1, NUM_TIERS)

    # Each tier's row of the assembler matrix depends only on its own split, so build them once per split
    rows_by_split = []
    for quality_modules in range(slots + 1):
        params = get_assembler_parameters(
            [(slots - quality_modules, quality_modules)] * NUM_TIERS,
            ingredient_keep,
            base_prod_bonus,
            recipe_ratio,
            prod_module_bonus,
            qual_module_bonus,
        )
        rows_by_split.append(create_production_matrix(params))

    recycler_parameters = get_recycler_parameters(
        NUM_TIERS + 1 if product_quality_to_keep is None else product_quality_to_keep,
        recipe_ratio,
        qual_module_bonus if recycler_qual_module_bonus is None else recycler_qual_module_bonus,
        recycle_ratio,
    )
    recycler_matrix = create_production_matrix(recycler_parameters)

    if product_quality_to_keep is not None:
        output_slice = slice(NUM_TIERS + product_quality_to_keep - 1, 2 * NUM_TIERS)
    else:
        output_slice = slice(ingredient_keep - 1, NUM_TIERS)

    best_output = -math.inf
    best_splits: tuple[int, ...] = ()
    identity = np.eye(2 * NUM_TIERS)
    for splits in itertools.product(range(slots + 1), repeat=assembled):
        assembler_matrix = np.zeros((NUM_TIERS, NUM_TIERS))
        for tier, quality_modules in enumerate(splits):
            assembler_matrix[tier] = rows_by_split[quality_modules][tier]
        transition = create_transition_matrix(assembler_matrix, recycler_matrix)

        # Same steady state as `x_0 (I - T)^-1`, without forming the inverse
        try:
            flows = np.linalg.solve((identity - transition).T, input_vector)
        except np.linalg.LinAlgError:
            continue
        # A loop that more than replaces itself has no steady state
        if not np.all(np.isfinite(flows)) or flows.min() < -TOLERANCE:
            continue

        output = float(flows[output_slice].sum())
        # Strict improvement only, so ties go to the earlier split, i.e. fewer quality modules
        if not best_splits or output > best_output + TOLERANCE * max(1.0, abs(best_output)):
            best_output, best_splits = output, splits

    if not best_splits and assembled:
        raise ValueError("No module split gives a loop with a steady state.")

    modules = [(slots - q, q) for q in best_splits] + [None] * (NUM_TIERS - assembled)
    return ModuleOptimum(modules, best_output)
