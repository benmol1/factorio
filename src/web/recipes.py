"""Factorio: Space Age recipe data, taken from the wiki (https://wiki.factorio.com)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    """An item's display name and icon."""

    name: str
    icon: str


@dataclass(frozen=True)
class Recipe:
    """A recipe's inputs, outputs and the machines (keys of MACHINES) that can craft it."""

    name: str
    time: float
    ingredients: dict[str, float]
    products: dict[str, float]
    machines: tuple[str, ...]

    def ratio(self, ingredient: str) -> float:
        """Products per unit of one ingredient, as used by the assembler loop."""
        return sum(self.products.values()) / self.ingredients[ingredient]


ITEMS = {
    "advanced-circuit": Item("Advanced circuit", "advanced_circuit"),
    "carbon": Item("Carbon", "carbon"),
    "carbon-fiber": Item("Carbon fiber", "carbon_fiber"),
    "toolbelt-equipment": Item("Toolbelt equipment", "toolbelt_equipment"),
    "yumako-mash": Item("Yumako mash", "yumako_mash"),
}

ASSEMBLERS = ("assembling-machine-2", "assembling-machine-3")

RECIPES = {
    "carbon-fiber": Recipe(
        "Carbon fiber",
        time=5,
        ingredients={"carbon": 1, "yumako-mash": 10},
        products={"carbon-fiber": 1},
        machines=("biochamber",),
    ),
    "toolbelt-equipment": Recipe(
        "Toolbelt equipment",
        time=10,
        ingredients={"advanced-circuit": 3, "carbon-fiber": 10},
        products={"toolbelt-equipment": 1},
        machines=ASSEMBLERS,
    ),
}


def recipe_data() -> dict:
    """Serialisable recipes and items, for the frontend's recipe picker."""
    return {
        "items": {key: {"name": i.name, "icon": i.icon} for key, i in ITEMS.items()},
        "recipes": {
            key: {
                "name": r.name,
                "time": r.time,
                "ingredients": r.ingredients,
                "products": r.products,
                "machines": list(r.machines),
                "icon": ITEMS[next(iter(r.products))].icon,
            }
            for key, r in RECIPES.items()
        },
    }
