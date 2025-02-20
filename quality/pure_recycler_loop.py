from quality import create_production_matrix
import numpy as np
from functools import lru_cache
from typing import Union
import matplotlib.pyplot as plt
import pandas as pd


@lru_cache()
def recycler_matrix(quality_chance: float, quality_to_keep: int = 5, production_ratio: float = 0.25) -> np.ndarray:
    """Returns a matrix of a recycler with quality chance `quality_chance`
    that saves any item of quality level `quality_to_keep` or above.

    Args:
        quality_chance (float): Quality chance of the recyclers (in %).
        quality_to_keep (int): Minimum quality level of the items to be removed from the system
            (By default only removes legendaries).
        production_ratio (float): Productivity ratio of the recyclers (0.25 by default)

    Returns:
        np.ndarray: Standard production matrix.
    """
    assert quality_chance > 0
    assert type(quality_to_keep) == int and 1 <= quality_to_keep <= 5
    assert production_ratio >= 0

    recycling_rows = quality_to_keep - 1
    saving_rows = 5 - recycling_rows

    return create_production_matrix([(quality_chance, production_ratio)] * recycling_rows + [(0, 0)] * saving_rows)


def recycler_loop(
    input_vector: Union[np.array, float],
    quality_chance: float,
    quality_to_keep: int = 5,
    production_ratio: float = 0.25,
    verbose: bool = False,
) -> np.ndarray:
    """Returns a vector with values for each quality level that mean different things,
    depending on whether that quality is kept or recycled:
        - If the quality is kept: the value is the production rate of items of that quality level.
        - If the quality is recycled: the value is the internal flow rate of items of that quality level in the system.

    Args:
        input_vector (np.array): The flow rate of items going into the system. If a single value is passed,
            it is assumed to be the input rate of Q1 items going into the system.
        quality_chance (float): Quality chance of the recycler loop (in %).
        quality_to_keep (int): Minimum quality level of the items to be removed from the system
            (By default only removes legendaries).

    Returns:
        np.ndarray: Vector with values for each quality level.
    """
    if type(input_vector) in (float, int):
        input_vector = np.array([input_vector, 0, 0, 0, 0])

    result_flows = [input_vector]
    while True:
        result_flows.append(result_flows[-1] @ recycler_matrix(quality_chance, quality_to_keep, production_ratio))

        if sum(result_flows[-2] - result_flows[-1]) < 1e-10:
            # There's nothing left in the system
            break

    # Create the output dataframe
    col_headers = ["P1", "P2", "P3", "P4", "P5"]
    output_df = pd.DataFrame(data=result_flows, columns=col_headers)

    if verbose:
        print("## Iterations:")
        print(output_df)

    return sum(result_flows)


def bmatrix(a):  # https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError("bmatrix can at most display two dimensions")
    lines = str(a).replace("[", "").replace("]", "").splitlines()
    rv = [r"\begin{bmatrix}"]
    rv += ["  " + " & ".join(l.split()) + r"\\" for l in lines]
    rv += [r"\end{bmatrix}"]
    return "\n".join(rv)


def normal_to_legendary_ratio():
    indices = list(range(1, 25)) + [24.8]
    ratios = [float(1 / recycler_loop(1, i)[4]) for i in indices]

    print(f"{indices[9:]=}")
    print(f"{ratios[9:]=}")


def efficiency_data():
    indices = list(range(1, 25)) + [24.8]

    uncommon = [float(recycler_loop(100, i, 2)[1]) for i in indices]
    rare = [float(recycler_loop(100, i, 3)[2]) for i in indices]
    epic = [float(recycler_loop(100, i, 4)[3]) for i in indices]
    legendary = [float(recycler_loop(100, i, 5)[4]) for i in indices]

    print(f"{uncommon=}")
    print(f"{rare=}")
    print(f"{epic=}")
    print(f"{legendary=}")


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=1000)
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.max_rows", 20)
    pd.set_option("colheader_justify", "right")
    pd.options.display.float_format = "{:.1f}".format

    q = 2 * 0.047
    p = 0.8

    # crusher loop with epic quality modules
    print(recycler_matrix(quality_chance=q, production_ratio=p))

    # production flows with 550 tier-1 inputs per second
    print(recycler_loop(input_vector=100, quality_chance=q, production_ratio=p, verbose=True))

    # internal flows relative to the epic flows
    output = recycler_loop(1, q, production_ratio=0.8) / recycler_loop(1, q, production_ratio=0.8)[3]
    print(output)

    # quality_range = np.arange(0.01, 0.13, 0.01)
    # efficiency_output = [1 / recycler_loop(1, q, production_ratio=0.8)[4] for q in quality_range]

    # # Define two ranges for x and y
    # x_values = quality_range
    # y_values = efficiency_output

    # # Create scatter plot
    # plt.plot(x_values, y_values, color="blue")
    # plt.yscale("log")

    # # Labels and title
    # plt.xlabel("Quality chance")
    # plt.ylabel("Efficiency")
    # plt.title("Efficiency of crusher loop, by quality chance")
    # plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # # Show legend
    # plt.legend()

    # # Display plot
    # plt.show()
