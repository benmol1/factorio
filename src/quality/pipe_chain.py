"""Worked example: a multi-stage chain that upcycles lava into legendary iron plates via pipes and pipe to ground.

    lava --foundry--> molten iron --foundry (quality)--> pipes
    pipes + molten iron --foundry (casting)--> pipe to ground --recycler (quality)--> pipes + iron plates
    iron plates (tier >= `min_plate_tier`) --assembler (quality)--> pipes --> back into the pipe to ground foundries

Legendary iron plates leave the system. Everything else circulates, including legendary pipe to ground, which is
recycled for its legendary plates.

This isn't a single assembler-recycler loop: the recycler returns two items (pipes and plates), the casting recipe
isn't the recipe the recycler reverses, and the plates need their own assembly step to get back into the loop. So the
chain is solved as one linear system over (pipes, pipe to ground, plates) x five quality tiers, built from the same
production matrices as the loops. The flows are the steady state, found by solving x = x0 + x T rather than iterating.

Run as `python -m src.quality.pipe_chain` from the repo root.
"""

from dataclasses import dataclass, replace
from itertools import product

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


def _prod_ratio(machine: str, prod_modules: int, research: float, recipe_ratio: float) -> float:
    """Products per ingredient unit for a machine with productivity modules, including the 4x productivity cap."""
    return min(1 + MACHINES[machine].prod_bonus + research + prod_modules * BEST_PROD_MODULE, 4) * recipe_ratio


def _machine_speed(machine: str, modules: tuple[int, int]) -> float:
    """Normal quality machine speed after the penalty of (prod, quality) modules; productivity modules are level 3."""
    penalty = modules[0] * PROD_MODULE_SPEED_PENALTY[3] + modules[1] * QUALITY_MODULE_SPEED_PENALTY
    return machine_speed(MACHINES[machine].speed, 0, penalty)


def solve_chain(cfg: ChainConfig | None = None) -> dict:
    """Solve the steady state of the chain.

    Returns:
        A dict with the flows (items/s per tier) of every item, the legendary plate output and machine loads.
    """
    cfg = cfg or ChainConfig()
    casting_pipe = RECIPES["casting-pipe"]
    casting_ptg = RECIPES["casting-pipe-to-ground"]
    assembling_pipe = RECIPES["pipe"]
    lava = RECIPES["molten-iron-from-lava"]
    ptg_recycling = recycling("pipe-to-ground")
    recycler_qual = cfg.recycler_quality_modules * BEST_QUAL_MODULE

    # Each block maps the arrivals of one item to the arrivals of another, per tier (rows in, columns out)
    def block(quality_chance: float, ratio: float, rows_in_use: range) -> np.ndarray:
        params = [(quality_chance, ratio if i in rows_in_use else 0.0) for i in range(NUM_TIERS)]
        return create_production_matrix(params)

    pipes_to_ptg = block(
        cfg.ptg_foundry[1] * BEST_QUAL_MODULE,
        _prod_ratio("foundry", cfg.ptg_foundry[0], cfg.research_prod_bonus, casting_ptg.ratio("pipe")),
        range(NUM_TIERS),
    )
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
    transition[PTG : PTG + NUM_TIERS, PIPE : PIPE + NUM_TIERS] = ptg_to_pipes
    transition[PTG : PTG + NUM_TIERS, PLATE : PLATE + NUM_TIERS] = ptg_to_plates
    transition[PLATE : PLATE + NUM_TIERS, PIPE : PIPE + NUM_TIERS] = plates_to_pipes

    # Stage 1 turns molten iron (which has no quality, so only "tier 0") into pipes. Solve for 1 molten iron/s into it
    pipe_foundry_row = block(
        cfg.pipe_foundry[1] * BEST_QUAL_MODULE,
        _prod_ratio("foundry", cfg.pipe_foundry[0], cfg.research_prod_bonus, casting_pipe.ratio("molten-iron")),
        range(1),
    )[0]
    x0 = np.zeros(3 * NUM_TIERS)
    x0[PIPE : PIPE + NUM_TIERS] = pipe_foundry_row
    unit = x0 @ np.linalg.inv(np.eye(3 * NUM_TIERS) - transition)

    # The pipe to ground foundries also eat molten iron, in proportion to the pipes they cast from
    pipes_per_craft = casting_ptg.ingredients["pipe"]
    molten_per_pipe = casting_ptg.ingredients["molten-iron"] / pipes_per_craft
    molten_per_unit = 1 + molten_per_pipe * unit[PIPE : PIPE + NUM_TIERS].sum()

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
    ptg_speed = _machine_speed("foundry", cfg.ptg_foundry)
    loads = {
        "lava foundries": lava_crafts * lava.time / _machine_speed("foundry", (cfg.lava_foundry_prod_modules, 0)),
        "pipe foundries": scale
        / casting_pipe.ingredients["molten-iron"]
        * casting_pipe.time
        / _machine_speed("foundry", cfg.pipe_foundry),
        "pipe-to-ground foundries": pipes_in.sum() / pipes_per_craft * casting_ptg.time / ptg_speed,
        "recyclers": ptg_in.sum() * ptg_recycling.time_per_item / recycler_speed(0, cfg.recycler_quality_modules),
        "pipe assemblers": assembled
        * assembling_pipe.time
        / _machine_speed("assembling-machine-3", cfg.pipe_assembler),
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
    print("  machine-equivalents at 100% load:", {k: round(v, 1) for k, v in res["loads"].items()})


MODULE_SPLITS = [(p, q) for p in range(5) for q in range(5 - p)]  # (prod, quality) with at most 4 modules


@dataclass(frozen=True)
class Optimum:
    """The best configuration found for a machine budget, scaled so the budget is exactly used."""

    cfg: ChainConfig  # lava_per_second is the rate that exactly uses the budget
    legendary_per_second: float
    loads: dict[str, float]  # Machines of each kind used at that lava rate


def optimise(crafting_machines: float, recyclers: float) -> Optimum:
    """Find the module setup and lava rate that maximise legendary plates/s within a machine budget.

    Lava is free, so the only limits are the total crafting machines (lava + pipe + pipe to ground foundries and plate
    assemblers, pooled) and the recyclers. Flows and loads are linear in the lava rate, so each setup is solved once
    and scaled until the tighter of the two budgets binds. Machine counts are fractional (no rounding up).
    """
    best: Optimum | None = None
    for pipe_foundry in MODULE_SPLITS:
        for ptg_foundry in MODULE_SPLITS:
            for pipe_assembler in MODULE_SPLITS:
                for lava_prod, recycler_qual, min_tier in product(range(5), range(5), range(LEGENDARY)):
                    cfg = ChainConfig(
                        lava_foundry_prod_modules=lava_prod,
                        pipe_foundry=pipe_foundry,
                        ptg_foundry=ptg_foundry,
                        pipe_assembler=pipe_assembler,
                        recycler_quality_modules=recycler_qual,
                        min_plate_tier=min_tier,
                    )
                    try:
                        res = solve_chain(cfg)
                    except np.linalg.LinAlgError:  # Loop gain of exactly 1: no finite steady state
                        continue
                    if min(res["pipes"].min(), res["plates"].min()) < 0:  # Loop gain above 1: no steady state
                        continue
                    loads = res["loads"]
                    crafting = sum(v for k, v in loads.items() if k != "recyclers")
                    factor = min(crafting_machines / crafting, recyclers / loads["recyclers"])
                    legendary = res["legendary_plates"] * factor
                    if best is None or legendary > best.legendary_per_second:
                        scaled = replace(cfg, lava_per_second=cfg.lava_per_second * factor)
                        best = Optimum(scaled, legendary, {k: v * factor for k, v in loads.items()})
    assert best is not None
    return best


def report_optimum(crafting_machines: float, recyclers: float) -> None:
    """Print the best setup for a budget."""
    opt = optimise(crafting_machines, recyclers)
    c = opt.cfg
    print(f"budget: {crafting_machines:g} crafting machines, {recyclers:g} recyclers")
    print(
        f"  LEGENDARY PLATES: {opt.legendary_per_second:.4f}/s = {opt.legendary_per_second * 60:.1f}/min "
        f"(lava {c.lava_per_second:.1f}/s)"
    )
    print(
        f"  (P,Q): lava foundry P={c.lava_foundry_prod_modules} pipe foundry={c.pipe_foundry} "
        f"ptg foundry={c.ptg_foundry} pipe assembler={c.pipe_assembler} recycler Q={c.recycler_quality_modules} "
        f"min plate tier={c.min_plate_tier}"
    )
    print("  machines used:", {k: round(v, 2) for k, v in opt.loads.items()})


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    report(ChainConfig())
    report(ChainConfig(min_plate_tier=0))
    report(ChainConfig(ptg_foundry=(4, 0)))
    report(ChainConfig(ptg_foundry=(2, 2), pipe_foundry=(0, 4)))
    for budget in [(10, 2), (20, 4), (40, 4)]:
        report_optimum(*budget)
