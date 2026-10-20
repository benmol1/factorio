"""FastAPI frontend for the quality loop simulations.

Run with `uv run uvicorn src.web.app:app --reload` and open http://127.0.0.1:8000.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from src.quality.assembler_recycler_loop import assembler_recycler_loop
from src.quality.pure_recycler_loop import get_production_rate, recycler_loop
from src.quality.solver import solve_machine_quality
from src.quality.utils import NUM_TIERS
from src.web.game_data import (
    MACHINES,
    PROD_MODULE_BONUS,
    PROD_MODULE_SPEED_PENALTY,
    QUALITY_MODULE_BONUS,
    QUALITY_MODULE_SPEED_PENALTY,
    RECYCLER_SLOTS,
    game_data,
    machine_speed,
    recycler_speed,
)
from src.web.recipes import RECIPES

STATIC_DIR = Path(__file__).parent / "static"

Tier = Annotated[int, Field(ge=0, le=NUM_TIERS - 1)]
TierVector = Annotated[list[Annotated[float, Field(ge=0)]], Field(min_length=NUM_TIERS, max_length=NUM_TIERS)]


class ModuleChoice(BaseModel):
    """A module type (level 1-3) and its quality tier."""

    level: Annotated[int, Field(ge=1, le=3)] = 3
    quality: Tier = 0


class RecyclerSetup(BaseModel):
    """Recyclers filled with quality modules."""

    quality: Tier = 0
    count: Annotated[int, Field(ge=1)] = 1
    quality_module: ModuleChoice = ModuleChoice()


class RecyclerLoopRequest(BaseModel):
    """Inputs for the pure recycler loop."""

    input_rates: TierVector
    recycler: RecyclerSetup
    recipe_time: Annotated[float, Field(gt=0)] = 1.0
    recycle_time: Annotated[float, Field(gt=0)] | None = None
    quality_to_keep: Tier = NUM_TIERS - 1


class TierSetup(BaseModel):
    """Machines and modules used to craft ingredients of one quality tier."""

    count: Annotated[int, Field(ge=1)] = 1
    machine_quality: Tier = 0
    prod_modules: Annotated[int, Field(ge=0)] = 0
    quality_modules: Annotated[int, Field(ge=0)] = 0


class AssemblerLoopRequest(BaseModel):
    """Inputs for the assembler → recycler loop."""

    input_rates: TierVector
    machine: str = "assembling-machine-3"
    # Optional, only used to check the machine can craft it; times and ratios are passed explicitly
    recipe: str | None = None
    research_prod_bonus: Annotated[float, Field(ge=0)] = 0.0
    tiers: Annotated[list[TierSetup], Field(min_length=NUM_TIERS, max_length=NUM_TIERS)]
    prod_module: ModuleChoice = ModuleChoice()
    quality_module: ModuleChoice = ModuleChoice()
    recycler: RecyclerSetup
    recipe_time: Annotated[float, Field(gt=0)] = 1.0
    recipe_ratio: Annotated[float, Field(gt=0)] = 1.0
    recycle_time: Annotated[float, Field(gt=0)] | None = None
    # None means never extract (assemble/recycle every tier)
    ingredient_quality_to_keep: Tier | None = NUM_TIERS - 1
    product_quality_to_keep: Tier | None = None

    @model_validator(mode="after")
    def check_slots(self):
        """Reject unknown machines and module counts that don't fit in the machine."""
        if self.machine not in MACHINES:
            raise ValueError(f"Unknown machine '{self.machine}'")
        if self.recipe is not None:
            if self.recipe not in RECIPES:
                raise ValueError(f"Unknown recipe '{self.recipe}'")
            if self.machine not in RECIPES[self.recipe].machines:
                raise ValueError(f"{MACHINES[self.machine].name} can't craft {RECIPES[self.recipe].name}")
        slots = MACHINES[self.machine].slots
        for i, tier in enumerate(self.tiers):
            if tier.prod_modules + tier.quality_modules > slots:
                raise ValueError(f"Tier {i + 1} has more modules than the machine's {slots} slots")
        return self


app = FastAPI(title="Factorio Quality Loop Simulator")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/game-data")
def get_game_data() -> dict:
    """Machine and module stats for the frontend's dropdowns."""
    return game_data()


def _recycler_stats(recycler: RecyclerSetup) -> tuple[float, float]:
    """Quality chance and speed of a recycler setup."""
    qual_bonus = QUALITY_MODULE_BONUS[recycler.quality_module.level][recycler.quality_module.quality]
    return RECYCLER_SLOTS * qual_bonus, recycler_speed(recycler.quality)


@app.post("/api/recycler-loop")
def run_recycler_loop(req: RecyclerLoopRequest) -> dict:
    """Simulate a pure recycler loop."""
    quality_chance, speed = _recycler_stats(req.recycler)
    input_vector = np.array(req.input_rates, dtype=float)

    flows, transition_matrix, total_crafting_time = recycler_loop(
        input_vector=input_vector,
        quality_chance=quality_chance,
        quality_to_keep=req.quality_to_keep + 1,
        speed_recycler=speed,
        num_recyclers=req.recycler.count,
        recipe_time=req.recipe_time,
        recycle_time=req.recycle_time,
    )
    production_rate = get_production_rate(input_vector, flows, transition_matrix)

    return {
        "derived": {"quality_chance": quality_chance, "recycler_speed": speed},
        "flows": flows.tolist(),
        "production_rate": production_rate.tolist(),
        "kept": [i >= req.quality_to_keep for i in range(NUM_TIERS)],
        # Crafting time is in recycler-seconds per second, i.e. the fraction of recycler capacity in use
        "recycler_load": total_crafting_time,
    }


def _loop_kwargs(req: AssemblerLoopRequest) -> dict:
    """The loop arguments shared by simulating a build and solving for one, which exclude the machines themselves."""
    machine = MACHINES[req.machine]
    ingredient_keep = None if req.ingredient_quality_to_keep is None else req.ingredient_quality_to_keep + 1
    product_keep = None if req.product_quality_to_keep is None else req.product_quality_to_keep + 1
    recycler_quality_chance, _ = _recycler_stats(req.recycler)
    return {
        "input_vector": np.array(req.input_rates + [0.0] * NUM_TIERS, dtype=float),
        "assembler_modules_config": [(t.prod_modules, t.quality_modules) for t in req.tiers],
        "product_quality_to_keep": product_keep,
        "ingredient_quality_to_keep": ingredient_keep,
        # The library's base bonus is the production ratio before modules, so it includes the base 1×
        "base_prod_bonus": 1 + machine.prod_bonus + req.research_prod_bonus,
        "recipe_ratio": req.recipe_ratio,
        "recipe_time": req.recipe_time,
        "prod_module_bonus": PROD_MODULE_BONUS[req.prod_module.level][req.prod_module.quality],
        "qual_module_bonus": QUALITY_MODULE_BONUS[req.quality_module.level][req.quality_module.quality],
        "recycle_time": req.recycle_time,
        "recycler_qual_module_bonus": recycler_quality_chance / RECYCLER_SLOTS,
    }


def _assembler_speed(req: AssemblerLoopRequest, tier: TierSetup, machine_quality: int) -> float:
    """Crafting speed of one assembler of a tier at a given machine quality, after its module penalties."""
    penalty = (
        tier.prod_modules * PROD_MODULE_SPEED_PENALTY[req.prod_module.level]
        + tier.quality_modules * QUALITY_MODULE_SPEED_PENALTY
    )
    return machine_speed(MACHINES[req.machine].speed, machine_quality, penalty)


@app.post("/api/solve")
def solve(req: AssemblerLoopRequest) -> dict:
    """Find the fewest machines for each stage that leave no bottlenecks, at the lowest machine quality that allows it.

    The machine counts and qualities in the request's `tiers` and `recycler` are ignored.
    """
    qualities = range(NUM_TIERS)
    solution = solve_machine_quality(
        assembler_speeds_by_quality=[[_assembler_speed(req, t, q) for t in req.tiers] for q in qualities],
        recycler_speeds_by_quality=[recycler_speed(q) for q in qualities],
        **_loop_kwargs(req),
    )
    return {
        "tiers": [
            {"machine_quality": q, "count": n}
            for q, n in zip(solution.assembler_qualities, solution.num_assemblers, strict=True)
        ],
        "recycler": {"quality": solution.recycler_quality, "count": solution.num_recyclers},
    }


@app.post("/api/assembler-loop")
def run_assembler_loop(req: AssemblerLoopRequest) -> dict:
    """Simulate an assembler → recycler loop."""
    machine = MACHINES[req.machine]
    loop = _loop_kwargs(req)
    prod_bonus = loop["prod_module_bonus"]
    qual_bonus = loop["qual_module_bonus"]

    speeds = [_assembler_speed(req, t, t.machine_quality) for t in req.tiers]
    recycler_quality_chance, rec_speed = _recycler_stats(req.recycler)
    ingredient_keep = loop["ingredient_quality_to_keep"]
    product_keep = loop["product_quality_to_keep"]

    flows, _, max_flow = assembler_recycler_loop(
        **loop,
        speed_assemblers=speeds,
        speed_recycler=rec_speed,
        num_assemblers=[t.count for t in req.tiers],
        num_recyclers=req.recycler.count,
    )

    ingredient_flows = flows[:NUM_TIERS]
    product_flows = flows[NUM_TIERS:]
    ingredient_max = max_flow[:NUM_TIERS]
    # Kept products leave the loop; everything else goes through the recyclers
    recycled = NUM_TIERS if product_keep is None else product_keep - 1
    recycler_flow = float(product_flows[:recycled].sum())

    return {
        "derived": {
            "assembler_speeds": speeds,
            "assembler_quality_chance": [t.quality_modules * qual_bonus for t in req.tiers],
            "assembler_prod_bonus": [
                min(machine.prod_bonus + req.research_prod_bonus + t.prod_modules * prod_bonus, 3) for t in req.tiers
            ],
            "recycler_quality_chance": recycler_quality_chance,
            "recycler_speed": rec_speed,
        },
        "ingredients": {
            "flows": ingredient_flows.tolist(),
            "max_flow": ingredient_max.tolist(),
            "kept": [ingredient_keep is not None and i >= ingredient_keep - 1 for i in range(NUM_TIERS)],
        },
        "products": {
            "flows": product_flows.tolist(),
            "kept": [product_keep is not None and i >= product_keep - 1 for i in range(NUM_TIERS)],
        },
        "recycler": {"flow": recycler_flow, "max_flow": float(max_flow[NUM_TIERS])},
    }
