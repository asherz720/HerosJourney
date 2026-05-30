"""Action registry: add ActionDef + _handle_<name>() in AdventureEnv to extend."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionDef:
    name: str
    generates_rule: bool
    dep_display: str = ""      # f-string: {arg}, {loc}, {cost} available
    is_acquisition: bool = False  # True for actions that place items in inventory (buy, get)


# ------------------------------------------------------------------
# Registry – single source of truth for every action in the world
# ------------------------------------------------------------------
ACTION_REGISTRY: dict[str, ActionDef] = {

    # ---- atomic ----
    "go": ActionDef(
        name="go",
        generates_rule=False,
        dep_display="",          # shown as "must be at {loc}" in parent rule
    ),

    # ---- acquire ----
    "get": ActionDef(
        name="get",
        generates_rule=False,
        dep_display="need {arg} (at {loc}, free)",
        is_acquisition=True,
    ),
    "buy": ActionDef(
        name="buy",
        generates_rule=False,
        dep_display="need {arg} (at {loc}, costs {cost} currency)",
        is_acquisition=True,
    ),

    # ---- process / ritual-style (no sub-rules exposed) ----
    "perform": ActionDef(
        name="perform",
        generates_rule=False,
        dep_display="perform {arg}",
    ),
    "drink": ActionDef(
        name="drink",
        generates_rule=False,
        dep_display="drink {arg}",
    ),

    # ---- root goals ----
    "defeat": ActionDef(
        name="defeat",
        generates_rule=True,
        dep_display="{name} {arg} (at {loc})",
    ),
    "rescue": ActionDef(
        name="rescue",
        generates_rule=True,
        dep_display="{name} {arg} (at {loc})",
    ),
}


def get_action(name: str) -> ActionDef:
    """Return ActionDef for *name*, raising KeyError with a helpful message."""
    if name not in ACTION_REGISTRY:
        raise KeyError(
            f"Unknown action '{name}'. "
            f"Register it in world_info/actions.py before using it."
        )
    return ACTION_REGISTRY[name]


def valid_action_names() -> list[str]:
    return list(ACTION_REGISTRY.keys())
