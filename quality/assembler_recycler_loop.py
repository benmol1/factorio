from __future__ import annotations

import itertools
from enum import Enum

import numpy as np
import pandas as pd
from tqdm import tqdm

from quality import BEST_PROD_MODULE, BEST_QUAL_MODULE, NUM_TIERS, create_production_matrix


def create_transition_matrix(assembler_matrix: np.ndarray, recycler_matrix: np.ndarray) -> np.ndarray:
    """Create a combined transition matrix from assembler and recycler matrices.

    The transition matrix has the recycler production matrix in the lower left
    and assembler production matrix in the upper right.

    Args:
        assembler_matrix: Assembler production matrix (NxN).
        recycler_matrix: Recycler production matrix (NxN).

    Returns:
        Combined 2Nx2N transition matrix for the AR loop simulation.
    """
    res = np.zeros((NUM_TIERS * 2, NUM_TIERS * 2))

    for i in range(NUM_TIERS):
        for j in range(NUM_TIERS):
            res[i + NUM_TIERS][j] = recycler_matrix[i, j]
            res[i][j + NUM_TIERS] = assembler_matrix[i, j]

    return res


def create_crafting_time_vector(
    speed_assembler: list, num_assemblers: list, speed_recycler: float, num_recyclers: int, recipe_time: float
) -> np.ndarray:
    """Create a vector of crafting times per tier for assemblers and recyclers.

    Args:
        speed_assembler: List of assembler speeds per quality tier.
        num_assemblers: List of assembler counts per quality tier.
        speed_recycler: Speed of the recyclers.
        num_recyclers: Number of recyclers.
        recipe_time: Base recipe crafting time in seconds.

    Returns:
        Vector of effective crafting times for each tier (assemblers + recyclers).
    """
    speed_assembler = np.array(speed_assembler)

    res = list(recipe_time / speed_assembler) + [recipe_time / (16 * speed_recycler)] * NUM_TIERS
    num_entities_list = num_assemblers + ([num_recyclers] * NUM_TIERS)
    num_entities = np.array(num_entities_list)

    res = res / num_entities

    return np.array(res)


def create_max_flow_vector(crafting_time_vector: np.ndarray, recipe_ratio: float) -> np.ndarray:
    """Calculate the maximum throughput for each tier based on crafting times.

    Args:
        crafting_time_vector: Vector of crafting times per tier.
        recipe_ratio: Ratio of products to ingredients in the recipe.

    Returns:
        Vector of maximum flow rates (items/second) per tier.
    """
    mf_vector = np.zeros_like(crafting_time_vector)
    mf_vector = 1 / crafting_time_vector
    mf_vector[:NUM_TIERS] = mf_vector[:NUM_TIERS] / recipe_ratio

    return mf_vector


def compute_crafting_time(
    input_flows: np.ndarray, prev_crafting_time: list, recipe_ratio: float, ct_vector: np.ndarray
) -> list:
    """Compute the crafting time for one iteration of the AR loop.

    Args:
        input_flows: Flow rates entering each tier this iteration.
        prev_crafting_time: Previous iteration's crafting time result.
        recipe_ratio: Ratio of products to ingredients in the recipe.
        ct_vector: Crafting time vector from create_crafting_time_vector.

    Returns:
        List containing [total_crafting_time, bottleneck_tier_index].
    """
    ct = 0

    # The assemblers are in parallel, so only the max crafting time is relevant
    assembler_crafting_times = np.multiply(input_flows[:NUM_TIERS] * recipe_ratio, ct_vector[:NUM_TIERS])
    max_assembler_crafting_time = np.max(assembler_crafting_times)
    max_assembler_crafting_time_tier = np.argmax(assembler_crafting_times) + 1
    ct += max_assembler_crafting_time

    # The recyclers are in series, so the total crafting time is relevant
    recycler_crafting_time = np.dot(input_flows[NUM_TIERS:], ct_vector[NUM_TIERS:])
    ct += recycler_crafting_time

    if max_assembler_crafting_time > recycler_crafting_time:
        max_time_index = max_assembler_crafting_time_tier
    else:
        max_time_index = NUM_TIERS + 1

    return [ct, max_time_index]


def get_assembler_parameters(
    assembler_modules_config: tuple[float, float] | list[tuple[float, float]],
    quality_to_keep: int = NUM_TIERS,
    base_prod_bonus: float = 0,
    recipe_ratio: float = 1,
    prod_module_bonus: float = BEST_PROD_MODULE,
    qual_module_bonus: float = BEST_QUAL_MODULE,
) -> list[tuple[float, float]]:
    """Convert module configuration to quality/production parameters per tier.

    Args:
        assembler_modules_config: Tuple of (productivity_modules, quality_modules) or list per tier.
        quality_to_keep: Minimum quality tier to keep (don't assemble). Defaults to legendary.
        base_prod_bonus: Base productivity bonus from machine + technologies.
        recipe_ratio: Ratio of products to ingredients in the recipe.
        prod_module_bonus: Productivity bonus per module.
        qual_module_bonus: Quality chance bonus per module.

    Returns:
        List of (quality_chance, production_ratio) tuples per tier.
    """
    production_rows = quality_to_keep - 1

    res = [(0, 0)] * NUM_TIERS

    for i, (nP, nQ) in enumerate(assembler_modules_config):
        if i == production_rows:
            break

        # Assembler stats
        production_ratio = min(base_prod_bonus + nP * prod_module_bonus, 4) * recipe_ratio
        quality_chance = nQ * qual_module_bonus

        res[i] = (quality_chance, production_ratio)

    return res


def get_recycler_parameters(
    quality_to_keep: int = NUM_TIERS,
    recipe_ratio: float = 1,
    qual_module_bonus: float = BEST_QUAL_MODULE,
) -> list[tuple[float, float]]:
    """Generate recycler parameters for the transition matrix.

    Args:
        quality_to_keep: Minimum quality tier to keep (don't recycle). Defaults to legendary.
        recipe_ratio: Ratio of products to ingredients in the recipe.
        qual_module_bonus: Quality chance bonus per module (recyclers use 4 quality modules).

    Returns:
        List of (quality_chance, production_ratio) tuples per tier.
    """
    recycling_rows = quality_to_keep - 1
    saving_rows = NUM_TIERS - recycling_rows

    # Recycler stats
    production_ratio = 0.25 / recipe_ratio
    quality_chance = 4 * qual_module_bonus

    return [(quality_chance, production_ratio)] * recycling_rows + [(0, 0)] * saving_rows


def assembler_recycler_loop(
    input_vector: np.array | float,
    assembler_modules_config: tuple[float, float] | list[tuple[float, float]],
    product_quality_to_keep: int | None = NUM_TIERS,
    ingredient_quality_to_keep: int | None = NUM_TIERS,
    base_prod_bonus: float = 0,
    recipe_ratio: float = 1,
    recipe_time: float = 1,
    prod_module_bonus: float = BEST_PROD_MODULE,
    qual_module_bonus: float = BEST_QUAL_MODULE,
    speed_assemblers: list = [1] * NUM_TIERS,
    speed_recycler: float = 0.5,
    num_assemblers: list = [1] * NUM_TIERS,
    num_recyclers: int = 1,
    verbose: bool = False,
) -> np.array:
    """Simulate an assembler-recycler loop to calculate quality tier flows.

    Returns a vector with values for each quality level:
    - If the quality is kept: the production rate of ingredients/items at that tier.
    - If the quality is recycled: the internal flow rate at that tier.

    The first five values represent ingredients, the last five represent products.

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
        speed_assemblers: List of assembler speeds per quality tier.
        speed_recycler: Speed of the recyclers.
        num_assemblers: List of assembler counts per quality tier.
        num_recyclers: Number of recyclers.
        verbose: If True, print debug information.

    Returns:
        Tuple of (flow_vector, transition_matrix, total_crafting_time, max_flow_vector).
    """
    if isinstance(assembler_modules_config, tuple):
        assembler_modules_config = [assembler_modules_config] * NUM_TIERS
    elif isinstance(assembler_modules_config, list):
        assert len(assembler_modules_config) == NUM_TIERS

    # Parameters for the production matrices
    assembler_parameters = get_assembler_parameters(
        assembler_modules_config,
        ingredient_quality_to_keep if ingredient_quality_to_keep is not None else NUM_TIERS + 1,
        base_prod_bonus,
        recipe_ratio,
        prod_module_bonus,
        qual_module_bonus,
    )
    recycler_parameters = get_recycler_parameters(
        product_quality_to_keep if product_quality_to_keep is not None else NUM_TIERS + 1,
        recipe_ratio,
        qual_module_bonus,
    )
    # Create the transition matrix
    transition_matrix = create_transition_matrix(
        assembler_matrix=create_production_matrix(assembler_parameters),
        recycler_matrix=create_production_matrix(recycler_parameters),
    )

    crafting_time_vector = create_crafting_time_vector(
        speed_assemblers, num_assemblers, speed_recycler, num_recyclers, recipe_time
    )

    max_flow_vector = create_max_flow_vector(crafting_time_vector, recipe_ratio)

    if verbose:
        print("\n## Transition matrix:\n", transition_matrix)
        print("\n## Crafting time vector:\n", crafting_time_vector)
        print("\n## Max flow vector:\n", max_flow_vector)

    # Handle the case where input_vector has just been given as a single scalar value
    if type(input_vector) in (float, int):
        input_vector = np.array([input_vector] + [0] * (NUM_TIERS * 2 - 1))

    # Initialise loop variable and output arrays
    ii = 0
    result_flows = [input_vector]
    crafting_time = [[0.0, 0]]
    total_crafting_time = 0.0

    while True:
        ii += 1
        ct_this_iteration = compute_crafting_time(
            result_flows[-1], crafting_time[-1], recipe_ratio, crafting_time_vector
        )

        total_crafting_time += ct_this_iteration[0]
        crafting_time.append(ct_this_iteration)
        result_flows.append(result_flows[-1] @ transition_matrix)

        if sum(abs(result_flows[-2] - result_flows[-1])) < 1e-2:
            # There's nothing left in the system
            break

    # Create the output dataframe
    col_headers = ["I1", "I2", "I3", "I4", "I5", "P1", "P2", "P3", "P4", "P5"]
    output_df = pd.DataFrame(data=result_flows, columns=col_headers)
    crafting_time_df = pd.DataFrame(data=crafting_time, columns=["Crafting time", "Max time index"])
    output_df = output_df.join(crafting_time_df)

    if verbose:
        print("\n## Iterations:")
        print(output_df.head(10))

    return sum(result_flows), transition_matrix, total_crafting_time, max_flow_vector


class SystemOutput(Enum):
    """Specifies whether to optimize for ingredient or item output."""

    INGREDIENTS = 0
    ITEMS = 1


class ModuleStrategy(Enum):
    """Module configuration strategy for efficiency calculations."""

    FULL_QUALITY = 0
    FULL_PRODUCTIVITY = 1
    OPTIMIZE = 2


def get_all_configs(module_slots: int):
    """Generate all possible configurations for an assembler with `n` module slots."""
    module_variations_for_assembler = []

    for p in range(module_slots + 1):
        q = module_slots - p
        module_variations_for_assembler.append((p, q))

    res = list(itertools.product(*[module_variations_for_assembler] * NUM_TIERS))

    for i in range(len(res)):
        res[i] = list(res[i])

    return res


def assembler_recycler_efficiency(
    module_slots: int,
    base_productivity: float,
    system_output: SystemOutput,
    module_strategy: ModuleStrategy,
    prod_mod_bonus: float = BEST_PROD_MODULE,
    qual_mod_bonus: float = BEST_QUAL_MODULE,
    target_tier: int = NUM_TIERS,
) -> float:
    """Returns the efficiency of the setup with the given parameters (%)."""
    assert module_slots >= 0 and base_productivity >= 0

    if system_output == SystemOutput.ITEMS:
        keep_items = target_tier
        keep_ingredients = None
        result_index = NUM_TIERS + (target_tier - 1)
    else:  # system_output == SystemOutput.INGREDIENTS:
        keep_items = None
        keep_ingredients = target_tier
        result_index = target_tier - 1

    if module_strategy != ModuleStrategy.OPTIMIZE:
        config = (module_slots, 0) if module_strategy == ModuleStrategy.FULL_PRODUCTIVITY else (0, module_slots)
        output = assembler_recycler_loop(
            100,
            config,
            keep_items,
            keep_ingredients,
            base_productivity,
            prod_module_bonus=prod_mod_bonus,
            qual_module_bonus=qual_mod_bonus,
        )
        return output[result_index]
    else:
        best_efficiency = 0
        all_configs = get_all_configs(module_slots)

        for config in tqdm(list(all_configs)):
            if config[NUM_TIERS - 1] != (module_slots, 0):
                # Makes no sense to put quality modules on legendary item crafter
                continue

            output = assembler_recycler_loop(
                100,
                config,
                keep_items,
                keep_ingredients,
                base_productivity,
                prod_module_bonus=prod_mod_bonus,
                qual_module_bonus=qual_mod_bonus,
            )
            efficiency = float(output[0][result_index])

            if best_efficiency < efficiency:
                best_efficiency = efficiency
                print(f"{config}: {efficiency:.2f}")

        return best_efficiency


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=1000)
    pd.set_option("display.max_columns", 14)
    pd.set_option("display.max_rows", 10)
    pd.set_option("colheader_justify", "right")
    pd.options.display.float_format = "{:.2f}".format

    n_slots = 4
    base_prod = 1.0
    full_qual_config = [(0, n_slots)] * (NUM_TIERS - 1) + [(n_slots, 0)]
    full_prod_config = [(n_slots, 0)] * NUM_TIERS
    optimal_leg_config = [(n_slots - 1, 1)] * (NUM_TIERS - 1) + [(n_slots, 0)]

    # AR loop for producing legendary u-238 via uranium ammo
    input_vector = np.array([8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    results = assembler_recycler_loop(
        input_vector=input_vector,
        assembler_modules_config=full_qual_config,
        product_quality_to_keep=None,
        ingredient_quality_to_keep=NUM_TIERS,
        base_prod_bonus=base_prod,
        recipe_ratio=1,  # NB: ratio of products:ingredients in the recipe
        prod_module_bonus=0,
        qual_module_bonus=BEST_QUAL_MODULE,
        speed_assemblers=[2.5, 2.5, 2.5, 2.5, 1],  # Legendary assemblers
        speed_recycler=1,  # legendary recyclers
        recipe_time=10,
        num_assemblers=[40, 4, 2, 1, 1],
        num_recyclers=8,
        verbose=True,
    )

    flows = results[0]
    transition_matrix = results[1]
    total_crafting_time = results[2]
    max_flow_vector = results[3]

    flows_with_summed_recyclers = np.concatenate((flows[:NUM_TIERS], np.array([sum(flows[NUM_TIERS:])])))

    print("## Flow per second:")
    print(flows_with_summed_recyclers)
    print("## Max flow per second:")
    print(max_flow_vector[: NUM_TIERS + 1])

    flow_ratio = flows_with_summed_recyclers / max_flow_vector[: NUM_TIERS + 1]

    print("\n## Flows over max:")
    print(flow_ratio)

    if np.any(flow_ratio > 1):
        print("The following tiers need more machines:")
        for ii in range(NUM_TIERS + 1):
            if flow_ratio[ii] > 1:
                print(ii + 1)

    np.set_printoptions(precision=2, suppress=True, linewidth=1000)

    print("\n## Flow per minute (to compare with the production statistics panel):")
    print(flows * 60)

    # print("## Legendary production rate per hour: %.1f" % (flows[9] * 3600))

    # eff = assembler_recycler_efficiency(
    #     n_slots,
    #     base_prod,
    #     system_output=SystemOutput.ITEMS,
    #     module_strategy=ModuleStrategy.OPTIMIZE,
    #     target_tier=NUM_TIERS,  # targeting legendary products
    #     prod_mod_bonus=BEST_PROD_MODULE,
    #     qual_mod_bonus=BEST_QUAL_MODULE,
    # )
