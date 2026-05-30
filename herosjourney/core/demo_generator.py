"""Generate Demo objects by running source-split tasks through AdventureEnv."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from herosjourney.core.generator import GeneratedTask
from herosjourney.env.env import AdventureEnv
from herosjourney.world_info.actions import ACTION_REGISTRY


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Demo:
    """Complete demonstration episode: rules text, solution trace, and metadata."""
    entity_name: str
    goal:        str
    rules_text:  str
    trace:       List[Tuple[str, str]] = field(default_factory=list)
    metadata:    Dict[str, Any]        = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format(self, show_world: bool = False) -> str:
        """Render as an RPG-style episode string; prepends world listing when show_world=True."""
        lines: List[str] = []

        if show_world and "world_listing" in self.metadata:
            lines.append(self.metadata["world_listing"])
            lines.append("")

        lines.append(self.rules_text)
        lines.append("")

        for action_str, obs_msg in self.trace:
            lines.append(f"> {action_str}")
            lines.append(f"  {obs_msg}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "goal":        self.goal,
            "rules_text":  self.rules_text,
            "trace":       self.trace,
            "metadata":    self.metadata,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Demo:
        return Demo(
            entity_name=d["entity_name"],
            goal=d["goal"],
            rules_text=d["rules_text"],
            trace=[tuple(step) for step in d["trace"]],
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# World listing helpers
# ---------------------------------------------------------------------------

def _rpg_world_line(entity_name: str, attr_names: List[str], attr_values: List[str],
                    location: Optional[str]) -> str:
    """Build one world listing line: [ Entity ]  attr1: val  |  @ location"""
    parts: List[str] = [f"{n}: {v}" for n, v in zip(attr_names, attr_values) if n and v]
    if location:
        parts.append(f"@ {location}")
    inner = "  |  ".join(parts)
    return f"[ {entity_name} ]  {inner}" if inner else f"[ {entity_name} ]"


def build_world_listing(tasks: List["GeneratedTask"]) -> str:
    """Build the === World === block listing all entities and items across the given tasks."""
    # --- Entities ---
    entity_lines: List[str] = []
    for task in tasks:
        # Find the node carrying entity attribute information.
        # Scan all nodes for the first one with "attribute_names" in properties;
        # fall back to the root node for backward compatibility.
        entity_node = None
        for node in task.tree.nodes.values():
            if "attribute_names" in node.properties:
                entity_node = node
                break
        if entity_node is None:
            entity_node = task.tree.nodes[task.tree.root_id]

        props  = entity_node.properties
        entity = entity_node.argument

        attr_names  = props.get("attribute_names")  or task.metadata.get("attribute_names",  [])
        attr_values = props.get("attribute_values") or task.metadata.get("attribute_values", [])
        location    = props.get("location") or task.metadata.get("entity_location")

        entity_lines.append(_rpg_world_line(entity, attr_names, attr_values, location))

    # --- Items: collect unique acquisition-node items from all task trees ---
    seen_items: Dict[str, Dict] = {}  # name -> {location, cost}
    for task in tasks:
        for node in task.tree.nodes.values():
            action = node.meta.get("incoming_edge")
            action_def = ACTION_REGISTRY.get(action) if action else None
            if action_def and action_def.is_acquisition:
                name = node.argument
                if name not in seen_items:
                    seen_items[name] = {
                        "location": node.properties.get("location", ""),
                        "cost":     node.properties.get("cost"),
                    }

    item_lines: List[str] = []
    for name, info in sorted(seen_items.items()):
        parts: List[str] = []
        if info["location"]:
            parts.append(f"@ {info['location']}")
        if info["cost"] is not None:
            parts.append(f"cost: {info['cost']}")
        inner = "  |  ".join(parts)
        item_lines.append(f"[ {name} ]  {inner}" if inner else f"[ {name} ]")

    lines = ["=== World ===", ""]
    lines.append("[Entities]")
    lines.extend(entity_lines)
    if item_lines:
        lines.append("")
        lines.append("[Items]")
        lines.extend(item_lines)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_demos(
    source_tasks: List[GeneratedTask],
    initial_currency: int = 500,
    initial_location: str = "GameStart",
    seed: int = 0,
) -> List[Demo]:
    """Run each task's optimal solution through a fresh env and return one Demo per task."""
    demos: List[Demo] = []

    for task in source_tasks:
        env = AdventureEnv(
            [(task.tree, task.tree.root_id)],
            initial_currency=initial_currency,
            initial_location=initial_location,
        )

        root_node = task.tree.nodes[task.tree.root_id]
        root_action = root_node.meta.get("incoming_edge", "")
        goal = f"{root_action} {root_node.argument}"

        initial_obs = env.reset(
            tree_index=0,
            initial_currency=initial_currency,
            initial_location=initial_location,
            seed=seed,
            rules_to_skip=task.rules_to_skip,
            task_label="Episode",
        )

        solution = task.tree.get_solution()
        trace: List[Tuple[str, str]] = []

        for step_str in solution:
            parts  = step_str.split(None, 1)
            action = parts[0]
            arg    = parts[1] if len(parts) > 1 else ""
            full_action, obs, _ = env.step(action, arg)
            trace.append((full_action, obs.message))

        demos.append(Demo(
            entity_name=root_node.argument,
            goal=goal,
            rules_text=initial_obs,
            trace=trace,
            metadata={**task.metadata, "rules_to_skip": task.rules_to_skip},
        ))

    return demos


# ---------------------------------------------------------------------------
# Mixed demos (property + distractor interleaved)
# ---------------------------------------------------------------------------

def generate_mixed_demos(
    property_tasks: List[GeneratedTask],
    distractor_tasks: List[GeneratedTask],
    gen_tasks: Optional[List[GeneratedTask]] = None,
    initial_currency: int = 500,
    initial_location: str = "GameStart",
    seed: int = 0,
) -> List[Demo]:
    """Interleave property and distractor demos; tag each with is_distractor; attach world listing."""
    import random as _random

    property_demos   = generate_demos(property_tasks,   initial_currency, initial_location, seed)
    distractor_demos = generate_demos(distractor_tasks, initial_currency, initial_location, seed)

    for d in property_demos:
        d.metadata["is_distractor"] = False
    for d in distractor_demos:
        d.metadata["is_distractor"] = True

    # Build a world listing covering all entities and items in this variant.
    # Including gen_tasks ensures every possible item name is visible even when
    # some items only appear in gen combinations (never bought in source demos).
    world_tasks = property_tasks + (gen_tasks or []) + distractor_tasks
    world_listing = build_world_listing(world_tasks)
    all_demos = property_demos + distractor_demos
    for d in all_demos:
        d.metadata["world_listing"] = world_listing

    _random.Random(seed).shuffle(all_demos)
    return all_demos


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def save_demos(demos: List[Demo], path: str) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w") as f:
        json.dump([d.to_dict() for d in demos], f, indent=2)
