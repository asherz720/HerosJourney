"""
Demo generation for the adventure-story generalization benchmark.

A demo is a complete episode record run against a source-split task:
  - The initial rules text (as shown to the agent, with hidden steps skipped)
  - The full solution trace (action → observation message)

Demos are used as in-context examples.  The trace intentionally reveals the
hidden steps (ritual, correct blade) so the agent can induce both:
  (1) the attribute→item mapping  (which blade to buy for a given entity)
  (2) the ordering constraint     (ritual must precede forge purchase)

Usage
-----
    from tree_management.registry import get_task
    tasks = get_task("additive").gen_fn(elements_path, split="source")
    demos = generate_demos(tasks, initial_currency=500)
    for demo in demos:
        print(demo.format())
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tree_management.generator import GeneratedTask
from env.env import AdventureEnv
from world_info.actions import ACTION_REGISTRY


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Demo:
    """
    One complete demonstration episode.

    Fields
    ------
    entity_name  : name of the goal entity (semantic or nonce)
    goal         : goal string, e.g. "defeat Gareth_Stonehall"
    rules_text   : complete initial observation from env.reset (header + cards)
    trace        : list of (action_string, observation_message) pairs
    metadata     : arbitrary extra info (task metadata, use_nonce, etc.)
    """
    entity_name: str
    goal:        str
    rules_text:  str
    trace:       List[Tuple[str, str]] = field(default_factory=list)
    metadata:    Dict[str, Any]        = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format(self, show_world: bool = False) -> str:
        """
        Render demo as a human-readable RPG-style episode string.

        Parameters
        ----------
        show_world : prepend the world listing (stored in metadata["world_listing"])
        """
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
    """
    Build a single RPG world listing line for an entity.

    Format:  [ Entity ]  attr1: val  |  attr2: val  |  @ location
    """
    parts: List[str] = [f"{n}: {v}" for n, v in zip(attr_names, attr_values) if n and v]
    if location:
        parts.append(f"@ {location}")
    inner = "  |  ".join(parts)
    return f"[ {entity_name} ]  {inner}" if inner else f"[ {entity_name} ]"


def build_world_listing(tasks: List["GeneratedTask"]) -> str:
    """
    Build a world listing block that shows all entities and all items,
    their attributes/properties, and their locations.

    The listing intentionally does NOT reveal which item each entity needs —
    that causal mapping is what the agent must infer from the rules and demos.

    Pass all tasks for this variant (source + gen + distractor) so that every
    entity and every possible item name is visible to the model.

    Entity nodes are identified by scanning all tree nodes for those that carry
    an "attribute_names" property (set by the process via a properties declaration).
    This makes the listing independent of which node is the root — the entity
    node does not have to be the root/defeat node.

    Items are collected from any node whose action has is_acquisition=True in
    ACTION_REGISTRY (covers "buy", "get", and any future acquisition actions).
    """
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
    """
    Generate one Demo per task by running each task's optimal solution through
    a fresh AdventureEnv.

    Parameters
    ----------
    source_tasks     : list of GeneratedTask (typically the source split)
    initial_currency : starting currency for each episode
    initial_location : starting location for each episode
    seed             : random seed passed to env.reset (affects rule shuffle)

    Returns
    -------
    List[Demo], one per task, in the same order as source_tasks.
    """
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
    """
    Generate demos from property source tasks interleaved with distractor tasks.

    The resulting list contains one Demo per task (property + distractor),
    shuffled so distractors are not all clustered at the end.  Each Demo
    carries a ``metadata["is_distractor"]`` flag so callers can audit the mix.

    Parameters
    ----------
    property_tasks   : source-split property tasks (the signal)
    distractor_tasks : distractor tasks sampled from the distractor pool (the noise)
    gen_tasks        : gen-split tasks for this variant.  Not used for demo
                       generation, but included in the world listing so that
                       gen entities and their possible items are visible to the
                       model (gen-task items may not appear in source demos).
    seed             : controls the shuffle order

    Returns
    -------
    Shuffled list of Demo objects.
    """
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
