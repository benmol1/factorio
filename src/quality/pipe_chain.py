"""Worked example: a multi-stage chain that upcycles lava into legendary iron plates via pipes and pipe to ground.

    lava --foundry--> molten iron --foundry (quality)--> pipes
    pipes + molten iron --foundry (casting)--> pipe to ground --recycler (quality)--> pipes + iron plates
    iron plates (tier >= `min_plate_tier`) --assembler (quality)--> pipes --> back into the pipe to ground foundries

Legendary iron plates leave the system. Everything else circulates, including legendary pipe to ground, which is
recycled for its legendary plates. Legendary pipes skip the pipe to ground foundries and are recycled directly.

This isn't a single assembler-recycler loop: the recycler returns two items (pipes and plates), the casting recipe
isn't the recipe the recycler reverses, and the plates need their own assembly step to get back into the loop. So the
chain is solved as one linear system over (pipes, pipe to ground, plates) x five quality tiers, built from the same
production matrices as the loops. The flows are the steady state, found by solving x = x0 + x T rather than iterating.

Run as `python -m src.quality.pipe_chain` from the repo root.
"""

from dataclasses import dataclass

import numpy as np

from src.quality.utils import BEST_PROD_MODULE, BEST_QUAL_MODULE, NUM_TIERS, create_production_matrix
from src.web.game_data import (
    MACHINES,
    PROD_MODULE_SPEED_PENALTY,
    QUALITY_MODULE_SPEED_PENALTY,
    machine_speed,
    recycler_speed,
)
from src.web.recipes import RECIPES, recycling

PIPE, PTG, PLATE = 0, NUM_TIERS, 2 * NUM_TIERS  # Offsets of each item's tiers in the state vector
LEGENDARY = NUM_TIERS - 1


@dataclass(frozen=True)
class ChainConfig:
    """Modules (productivity, quality) per machine and the policy for which plates are re-assembled."""

    lava_per_second: float = 100.0
    lava_foundry_prod_modules: int = 4
    pipe_foundry: tuple[int, int] = (0, 4)
    ptg_foundry: tuple[int, int] = (0, 4)
    pipe_assembler: tuple[int, int] = (0, 4)
    recycler_quality_modules: int = 4
    # Plates below this tier are surplus: they leave the chain rather than being assembled into pipes
    min_plate_tier: int = 1
    research_prod_bonus: float = 0.0
    machine_quality: int = 0  # Quality tier of the foundries and assemblers (speed only; slots don't change)
    recycler_quality: int = 0


def _prod_ratio(machine: str, prod_modules: int, research: float, recipe_ratio: float) -> float:
    """Products per ingredient unit for a machine with productivity modules, including the 4x productivity cap."""
    return min(1 + MACHINES[machine].prod_bonus + research + prod_modules * BEST_PROD_MODULE, 4) * recipe_ratio


def _machine_speed(machine: str, modules: tuple[int, int], quality: int = 0) -> float:
    """Machine speed after the penalty of (prod, quality) modules; productivity modules are level 3."""
    penalty = modules[0] * PROD_MODULE_SPEED_PENALTY[3] + modules[1] * QUALITY_MODULE_SPEED_PENALTY
    return machine_speed(MACHINES[machine].speed, quality, penalty)


def _transition(cfg: ChainConfig) -> tuple[np.ndarray, np.ndarray]:
    """The 15x15 transition matrix of the loop, and the pipe foundry's tier split per molten iron/s into it."""
    casting_pipe = RECIPES["casting-pipe"]
    casting_ptg = RECIPES["casting-pipe-to-ground"]
    assembling_pipe = RECIPES["pipe"]
    ptg_recycling = recycling("pipe-to-ground")
    pipe_recycling = recycling("pipe")
    recycler_qual = cfg.recycler_quality_modules * BEST_QUAL_MODULE

    # Each block maps the arrivals of one item to the arrivals of another, per tier (rows in, columns out)
    def block(quality_chance: float, ratio: float, rows_in_use: range) -> np.ndarray:
        params = [(quality_chance, ratio if i in rows_in_use else 0.0) for i in range(NUM_TIERS)]
        return create_production_matrix(params)

    pipes_to_ptg = block(
        cfg.ptg_foundry[1] * BEST_QUAL_MODULE,
        _prod_ratio("foundry", cfg.ptg_foundry[0], cfg.research_prod_bonus, casting_ptg.ratio("pipe")),
        range(LEGENDARY),  # Legendary pipes skip the foundry and go straight to a recycler
    )
    pipes_to_plates = block(recycler_qual, pipe_recycling.outputs["iron-plate"], range(LEGENDARY, NUM_TIERS))
    # The recycler returns a different amount of each item, but rolls quality the same way for both
    ptg_to_pipes = block(recycler_qual, ptg_recycling.outputs["pipe"], range(NUM_TIERS))
    ptg_to_plates = block(recycler_qual, ptg_recycling.outputs["iron-plate"], range(NUM_TIERS))
    plates_to_pipes = block(
        cfg.pipe_assembler[1] * BEST_QUAL_MODULE,
        _prod_ratio("assembling-machine-3", cfg.pipe_assembler[0], 0.0, assembling_pipe.ratio("iron-plate")),
        range(cfg.min_plate_tier, LEGENDARY),  # Legendary plates are the product, so they aren't assembled
    )

    transition = np.zeros((3 * NUM_TIERS, 3 * NUM_TIERS))
    transition[PIPE : PIPE + NUM_TIERS, PTG : PTG + NUM_TIERS] = pipes_to_ptg
    transition[PIPE : PIPE + NUM_TIERS, PLATE : PLATE + NUM_TIERS] = pipes_to_plates
    transition[PTG : PTG + NUM_TIERS, PIPE : PIPE + NUM_TIERS] = ptg_to_pipes
    transition[PTG : PTG + NUM_TIERS, PLATE : PLATE + NUM_TIERS] = ptg_to_plates
    transition[PLATE : PLATE + NUM_TIERS, PIPE : PIPE + NUM_TIERS] = plates_to_pipes

    # The pipe foundry turns molten iron (which has no quality, so only "tier 0") into pipes
    pipe_foundry_row = block(
        cfg.pipe_foundry[1] * BEST_QUAL_MODULE,
        _prod_ratio("foundry", cfg.pipe_foundry[0], cfg.research_prod_bonus, casting_pipe.ratio("molten-iron")),
        range(1),
    )[0]
    return transition, pipe_foundry_row


def pipe_foundry_output(molten_per_second: float, cfg: ChainConfig | None = None) -> np.ndarray:
    """Pipes/s per tier made by the pipe foundries from a molten iron flow."""
    return molten_per_second * _transition(cfg or ChainConfig())[1]


def solve_input(pipes_in: np.ndarray, cfg: ChainConfig | None = None) -> dict:
    """Steady state of the loop for fresh pipes (items/s per tier, normal first) arriving at the ptg foundries.

    Returns:
        The total flows (items/s per tier) of pipes into the foundries, pipe to ground recycled and plates out of the
        recyclers, plus the legendary and surplus plates, the molten iron the foundries eat and machine loads.
    """
    cfg = cfg or ChainConfig()
    casting_ptg = RECIPES["casting-pipe-to-ground"]
    assembling_pipe = RECIPES["pipe"]
    ptg_recycling = recycling("pipe-to-ground")
    pipe_recycling = recycling("pipe")
    transition, _ = _transition(cfg)

    x0 = np.zeros(3 * NUM_TIERS)
    x0[PIPE : PIPE + NUM_TIERS] = pipes_in
    flows = x0 @ np.linalg.inv(np.eye(3 * NUM_TIERS) - transition)
    pipes, ptg, plates = (flows[o : o + NUM_TIERS] for o in (PIPE, PTG, PLATE))

    mq = cfg.machine_quality
    pipes_per_craft = casting_ptg.ingredients["pipe"]
    cast_pipes = pipes[:LEGENDARY].sum()  # Legendary pipes are recycled, not cast
    assembled_plates = np.where(
        (np.arange(NUM_TIERS) >= cfg.min_plate_tier) & (np.arange(NUM_TIERS) < LEGENDARY), plates, 0
    )
    loads = {
        "pipe-to-ground foundries": np.where(np.arange(NUM_TIERS) < LEGENDARY, pipes, 0)
        / pipes_per_craft
        * casting_ptg.time
        / _machine_speed("foundry", cfg.ptg_foundry, mq),
        "recyclers": (ptg.sum() * ptg_recycling.time_per_item + pipes[LEGENDARY] * pipe_recycling.time_per_item)
        / recycler_speed(cfg.recycler_quality, cfg.recycler_quality_modules),
        "pipe assemblers": assembled_plates
        * assembling_pipe.time
        / _machine_speed("assembling-machine-3", cfg.pipe_assembler, mq),
    }
    return {
        "pipes": pipes,
        "pipe_to_ground": ptg,
        "plates": plates,
        "legendary_plates": plates[LEGENDARY],
        "surplus_plates": plates[: cfg.min_plate_tier].sum(),
        "molten_iron": cast_pipes / pipes_per_craft * casting_ptg.ingredients["molten-iron"],
        "loads": loads,
    }


def solve_chain(cfg: ChainConfig | None = None) -> dict:
    """Solve the steady state of the chain from lava.

    Returns:
        A dict with the flows (items/s per tier) of every item, the legendary plate output and machine loads.
    """
    cfg = cfg or ChainConfig()
    casting_pipe = RECIPES["casting-pipe"]
    casting_ptg = RECIPES["casting-pipe-to-ground"]
    assembling_pipe = RECIPES["pipe"]
    lava = RECIPES["molten-iron-from-lava"]
    ptg_recycling = recycling("pipe-to-ground")
    pipe_recycling = recycling("pipe")
    transition, pipe_foundry_row = _transition(cfg)

    # Solve for 1 molten iron/s into the pipe foundry
    x0 = np.zeros(3 * NUM_TIERS)
    x0[PIPE : PIPE + NUM_TIERS] = pipe_foundry_row
    unit = x0 @ np.linalg.inv(np.eye(3 * NUM_TIERS) - transition)

    # The pipe to ground foundries also eat molten iron, in proportion to the pipes they cast from
    pipes_per_craft = casting_ptg.ingredients["pipe"]
    molten_per_pipe = casting_ptg.ingredients["molten-iron"] / pipes_per_craft
    molten_per_unit = 1 + molten_per_pipe * unit[PIPE:LEGENDARY].sum()

    # Scale so the molten iron demand is exactly what the lava foundries make
    lava_crafts = cfg.lava_per_second / lava.ingredients["lava"]
    molten_made = (
        lava_crafts
        * lava.products["molten-iron"]
        * _prod_ratio("foundry", cfg.lava_foundry_prod_modules, cfg.research_prod_bonus, 1)
    )
    scale = molten_made / molten_per_unit
    flows = unit * scale

    pipes_in = flows[PIPE : PIPE + NUM_TIERS]
    plates_in = flows[PLATE : PLATE + NUM_TIERS]
    ptg_in = flows[PTG : PTG + NUM_TIERS]
    assembled = plates_in[cfg.min_plate_tier : LEGENDARY].sum()
    mq = cfg.machine_quality
    ptg_speed = _machine_speed("foundry", cfg.ptg_foundry, mq)
    loads = {
        "lava foundries": lava_crafts * lava.time / _machine_speed("foundry", (cfg.lava_foundry_prod_modules, 0), mq),
        "pipe foundries": scale
        / casting_pipe.ingredients["molten-iron"]
        * casting_pipe.time
        / _machine_speed("foundry", cfg.pipe_foundry, mq),
        "pipe-to-ground foundries": pipes_in[:LEGENDARY].sum() / pipes_per_craft * casting_ptg.time / ptg_speed,
        "recyclers": (ptg_in.sum() * ptg_recycling.time_per_item + pipes_in[LEGENDARY] * pipe_recycling.time_per_item)
        / recycler_speed(cfg.recycler_quality, cfg.recycler_quality_modules),
        "pipe assemblers": assembled
        * assembling_pipe.time
        / _machine_speed("assembling-machine-3", cfg.pipe_assembler, mq),
    }
    return {
        "pipes": pipes_in,
        "pipe_to_ground": ptg_in,
        "plates": plates_in,
        "legendary_plates": plates_in[LEGENDARY],
        "surplus_plates": plates_in[: cfg.min_plate_tier].sum(),
        "stone": lava_crafts * lava.products["stone"],
        "loads": {k: float(v) for k, v in loads.items()},
        "molten_split": (scale / molten_made, 1 - scale / molten_made),
    }


def report(cfg: ChainConfig) -> None:
    """Print the flows and machine loads of a configuration."""
    res = solve_chain(cfg)
    print(
        f"lava {cfg.lava_per_second:g}/s | pipe foundry (P,Q)={cfg.pipe_foundry} ptg foundry={cfg.ptg_foundry} "
        f"pipe assembler={cfg.pipe_assembler} min plate tier={cfg.min_plate_tier}"
    )
    print("  pipes into ptg foundries /s  ", np.round(res["pipes"], 3))
    print("  pipe to ground recycled /s   ", np.round(res["pipe_to_ground"], 3))
    print("  plates out of recyclers /s   ", np.round(res["plates"], 3))
    print(
        f"  LEGENDARY PLATES: {res['legendary_plates']:.3f}/s = {res['legendary_plates'] * 60:.1f}/min; "
        f"surplus lower-tier plates {res['surplus_plates']:.3f}/s; stone {res['stone']:.2f}/s"
    )
    print("  machine-equivalents at 100% load:", {k: round(v, 2) for k, v in res["loads"].items()})


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    cfg = ChainConfig(machine_quality=LEGENDARY, recycler_quality=LEGENDARY)
    # Example input: 5 legendary pipe foundries eating 400 molten iron/s
    res = solve_input(pipe_foundry_output(800, cfg), cfg)
    print("pipes into ptg foundries /m ", np.round(res["pipes"] * 60, 2))
    print("pipe to ground recycled /m  ", np.round(res["pipe_to_ground"] * 60, 2))
    print("plates out of recyclers /m  ", np.round(res["plates"] * 60, 2))
    print(f"LEGENDARY PLATES: {res['legendary_plates']:.3f}/s; surplus {res['surplus_plates']:.3f}/s")
    print(f"molten iron into ptg foundries: {res['molten_iron']:.0f}/s")
    print("machine-equivalents at 100% load (by tier, normal first):")
    for name, load in res["loads"].items():
        print(f"  {name:25}", np.round(load, 2), f"total {np.sum(load):.2f}")
