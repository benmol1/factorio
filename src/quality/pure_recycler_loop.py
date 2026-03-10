from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.quality.quality import NUM_TIERS, create_production_matrix


@lru_cache
def recycler_matrix(quality_chance: float, quality_to_keep: int = 5, is_asteroid_crusher: bool = False) -> np.ndarray:
    """Create a recycler transition matrix for the given quality chance.

    Args:
        quality_chance: Quality chance of the recyclers as a decimal.
        quality_to_keep: Minimum quality tier to extract (1-5). Defaults to 5 (legendary).
        is_asteroid_crusher: If True, uses 0.8 production ratio instead of 0.25.

    Returns:
        NxN production matrix for the recycler.
    """
    assert quality_chance > 0
    assert isinstance(quality_to_keep, int) and 1 <= quality_to_keep <= 5

    # Set the production ratio for this loop
    production_ratio = 0.8 if is_asteroid_crusher else 0.25

    recycling_rows = quality_to_keep - 1
    saving_rows = NUM_TIERS - recycling_rows

    return create_production_matrix([(quality_chance, production_ratio)] * recycling_rows + [(0, 0)] * saving_rows)


def recycler_loop(
    input_vector: np.array | float,
    quality_chance: float,
    quality_to_keep: int = 5,
    speed_recycler: float = 0.4,
    num_recyclers: np.array | int = 1,
    recipe_time: float = 1.0,
    recipe_ratio: float = 1.0,
    is_asteroid_crusher: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulate a pure recycler loop to calculate quality tier flows.

    Returns flow values per quality tier:
    - Kept tiers: production rate of items at that quality level.
    - Recycled tiers: internal flow rate in the system.

    Args:
        input_vector: Flow rate of items entering the system, or scalar for Q1 only.
        quality_chance: Quality chance of the recycler loop as a decimal.
        quality_to_keep: Minimum quality tier to extract. Defaults to 5 (legendary).
        speed_recycler: Speed of the recyclers.
        num_recyclers: Number of recyclers (int or array per tier).
        recipe_time: Base recipe crafting time in seconds.
        recipe_ratio: Ratio of products to ingredients.
        is_asteroid_crusher: If True, uses asteroid crusher parameters.
        verbose: If True, print debug information.

    Returns:
        Tuple of (flow_vector, transition_matrix, total_crafting_time).
    """
    if isinstance(input_vector, (float, int)):
        input_vector = np.array([input_vector, 0, 0, 0, 0])

    if isinstance(num_recyclers, int):
        num_recyclers = np.array([num_recyclers] * NUM_TIERS)

    transition_matrix = recycler_matrix(quality_chance, quality_to_keep, is_asteroid_crusher)
    crafting_time_vector = create_crafting_time_vector(speed_recycler, num_recyclers, recipe_time, is_asteroid_crusher)

    if verbose:
        print("## Transition matrix:\n", transition_matrix)
        print("## Crafting time vector:\n", crafting_time_vector)

    # Initialise loop variable and output arrays
    ii = 0
    result_flows = [input_vector]
    crafting_time = [[0.0, 0, False]]
    total_crafting_time = 0

    while True:
        ii += 1
        ct_this_iteration = compute_crafting_time(
            result_flows[-1], crafting_time[-1], recipe_ratio, crafting_time_vector
        )

        total_crafting_time += ct_this_iteration[0]
        crafting_time.append(ct_this_iteration)
        result_flows.append(result_flows[-1] @ transition_matrix)

        if sum(abs(result_flows[-2] - result_flows[-1])) < 1e-6:
            # There's nothing left in the system
            break

    # Create the output dataframe
    col_headers = ["P1", "P2", "P3", "P4", "P5"]
    output_df = pd.DataFrame(data=result_flows, columns=col_headers)
    crafting_time_df = pd.DataFrame(data=crafting_time, columns=["Crafting time", "Max time index", "Bottleneck"])
    output_df = output_df.join(crafting_time_df)

    if verbose:
        print("## Iterations:")
        print(output_df.head(10))

        if sum(output_df["Bottleneck"] > 0):
            print(output_df[output_df["Bottleneck"]])
            output_df.to_csv("output_df.csv")

    return sum(result_flows), transition_matrix, total_crafting_time


def create_crafting_time_vector(
    speed_recycler: float = 0.4,
    num_recyclers: np.ndarray | None = None,
    recipe_time: float = 1,
    is_asteroid_crusher: bool = False,
) -> np.ndarray:
    """Create a vector of crafting times per tier for recyclers.

    Args:
        speed_recycler: Speed of the recyclers (0.4 default for normal + quality modules).
        num_recyclers: Number of recyclers per tier. Defaults to 1 per tier.
        recipe_time: Base recipe crafting time in seconds.
        is_asteroid_crusher: If True, removes the 16x recycling speed bonus.

    Returns:
        Vector of effective crafting times for each tier.
    """
    if num_recyclers is None:
        num_recyclers = np.array([1.0] * NUM_TIERS)
    res = [recipe_time / (16 * speed_recycler)] * NUM_TIERS
    res = np.array(res) / num_recyclers

    # If this is an asteroid crusher, re-multiply the crafting time by 16 because that discount
    # only applies to pure recycling
    if is_asteroid_crusher:
        res *= 16

    return np.array(res)


def compute_crafting_time(
    input_flows: np.ndarray, prev_crafting_time: list, recipe_ratio: float, ct_vector: np.ndarray
) -> list:
    """Compute the crafting time for one iteration of the recycler loop.

    Args:
        input_flows: Flow rates entering each tier this iteration.
        prev_crafting_time: Previous iteration's crafting time result.
        recipe_ratio: Ratio of products to ingredients in the recipe.
        ct_vector: Crafting time vector from create_crafting_time_vector.

    Returns:
        List containing [total_crafting_time, max_time_index, is_bottleneck].
    """
    ct = 0

    # The recyclers are in series, so the total crafting time is relevant
    recycler_crafting_time = np.dot(input_flows, ct_vector)
    ct += recycler_crafting_time

    # This is a pure recycler loop, so the recyclers must always be the most time-consuming component of each iteration
    max_time_index = NUM_TIERS + 1

    # Compare the crafting time of this iteration to the previous one; if the crafting time here is greater then
    # report the bottleneck
    bottleneck = False
    if (prev_crafting_time[0] > 0) and (ct > 1) and (ct - prev_crafting_time[0] > 0.1):
        bottleneck = True

    return [ct, max_time_index, bottleneck]


def get_production_rate(
    input_vector: np.ndarray,
    output_flows: np.ndarray,
    transition_matrix: np.ndarray,
    is_asteroid_crusher: bool = False,
) -> np.ndarray:
    """Compute the production rate at each quality level.

    The game's production tracker only captures item-created and item-destroyed events.
    When an item is recycled into itself, this doesn't register. This function computes
    actual production rates comparable to the game's production statistics panel.

    Args:
        input_vector: Initial input flow rates per tier.
        output_flows: Total output flow rates from the simulation.
        transition_matrix: The transition matrix used in the simulation.
        is_asteroid_crusher: If True, applies asteroid crusher production counting.

    Returns:
        Production rate per quality tier.
    """
    # In order to provide the input_vector, it must first be produced somewhere
    production_rate = input_vector

    # For typical recycling loops, items are only produced by upcycling from lower tiers.
    # For asteroid crushers, equal-tier production events happen frequently.
    # ~50% of asteroid reprocessing crafts return an equiv-tier product (the other 50%
    # return the same item, so there's no item-created event).
    # We apply a 0.5 scalar for on-diagonal transitions for asteroid crushers.
    for ii in range(0, NUM_TIERS):
        prod_rate = 0
        for jj in range(ii + 1):
            if jj == ii:
                scalar = 0
                if is_asteroid_crusher:
                    scalar = 0.5
            else:
                scalar = 1
            prod_rate += output_flows[jj] * transition_matrix[jj][ii] * scalar
        production_rate[ii] += prod_rate

    return production_rate


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=1000)
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.max_rows", 20)
    pd.set_option("colheader_justify", "right")
    pd.options.display.float_format = "{:.2f}".format

    # Recycler loop for tungsten ore
    input_vector = np.array([45.0, 15.0, 0.0, 0.0, 0.0])
    q = 4 * 0.062
    results = recycler_loop(
        input_vector=input_vector,
        quality_chance=q,
        recipe_time=1,
        num_recyclers=5,
        speed_recycler=1,  # legendary recyclers
        verbose=True,
    )

    # Recycler loop for biter eggs
    # input_vector = np.array([32.0, 0.0, 0.0, 0.0, 0.0])
    # q = 4 * 0.062
    # results = recycler_loop(
    #     input_vector=input_vector,
    #     quality_chance=q,
    #     recipe_time=10,
    #     num_recyclers=28,
    #     speed_recycler=1,  # legendary recyclers
    #     verbose=True,
    # )

    flows = results[0]
    transition_matrix = results[1]
    total_crafting_time = results[2]

    print("## Flow per second:")
    print(flows)

    print("## Flow per minute:")
    print(flows * 60)

    production_rate = get_production_rate(input_vector, flows, transition_matrix, is_asteroid_crusher=False)

    print("## Production rates per minute:")
    print(production_rate * 60)
    print(f"## Legendary production rate per hour: {production_rate[4] * 3600:.1f}")

    print(f"## Total crafting time: {total_crafting_time:.2f} seconds")
