"""
Adventure Story Environment (New Tree Structure).

Tracks agent state (location, inventory, currency), validates actions,
and presents rules generated from goal trees.
"""

import random
from typing import Dict, Set, Optional, Tuple, List
from dataclasses import dataclass

from herosjourney.core.goal_tree import GoalTree
from herosjourney.world_info.actions import ACTION_REGISTRY, valid_action_names


# ============================================================================
# Utility Functions
# ============================================================================

def get_entity_location(entity_name: str, tree: GoalTree) -> Optional[str]:
    """Get location of an entity from tree nodes."""
    for node in tree.nodes.values():
        if node.argument == entity_name and "location" in node.properties:
            return node.properties["location"]
    return None


def get_object_cost(entity_name: str, tree: GoalTree) -> int:
    """Get cost of an object from tree nodes."""
    for node in tree.nodes.values():
        if node.argument == entity_name and "cost" in node.properties:
            return node.properties["cost"]
    return 0


def get_object_cost_from_all_trees(entity_name: str, trees: Dict[int, Tuple[GoalTree, str]]) -> int:
    """Get cost of an object by searching across all trees."""
    for tree, _ in trees.values():
        for node in tree.nodes.values():
            if node.argument == entity_name and "cost" in node.properties:
                return node.properties["cost"]
    return 0


def get_entity_location_from_all_trees(entity_name: str, trees: Dict[int, Tuple[GoalTree, str]]) -> Optional[str]:
    """Get location of an entity by searching across all trees."""
    for tree, _ in trees.values():
        for node in tree.nodes.values():
            if node.argument == entity_name and "location" in node.properties:
                return node.properties["location"]
    return None


def get_item_instance_kind_from_all_trees(entity_name: str, trees: Dict[int, Tuple[GoalTree, str]]) -> Optional[str]:
    """Get instance_kind of an item by searching across all trees (for search items)."""
    for tree, _ in trees.values():
        for node in tree.nodes.values():
            if node.argument == entity_name and "instance_kind" in node.properties:
                return node.properties["instance_kind"]
    return None


def is_search_node(entity_name: str, trees: Dict[int, Tuple[GoalTree, str]]) -> bool:
    """Check if an item is a search node (has is_search_node=True in meta)."""
    for tree, _ in trees.values():
        for node in tree.nodes.values():
            if node.argument == entity_name:
                return node.meta.get("is_search_node", False)
    return False


def check_inventory_has_instance_kind(inventory: Dict[str, int], instance_kind: str, trees: Dict[int, Tuple[GoalTree, str]]) -> Optional[str]:
    """Check if inventory already has an item of the given instance_kind. Returns the item name if found, None otherwise."""
    for item_name in inventory:
        if inventory[item_name] > 0:  # Only check items that are actually in inventory
            item_kind = get_item_instance_kind_from_all_trees(item_name, trees)
            if item_kind == instance_kind:
                return item_name
    return None


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Observation:
    """Observation returned after an action."""
    success: bool
    message: str
    done: bool = False


# ============================================================================
# Adventure Environment
# ============================================================================

class AdventureEnv:
    """Adventure story environment - tracks state and validates/executes actions."""
    
    def __init__(
        self,
        trees: List[Tuple[GoalTree, str]],  # List of (tree, root_id) tuples
        initial_currency: int = 1000000,
        initial_location: str = "GameStart",
    ):
        self.tree_index_map = {i: (tree, root_id) for i, (tree, root_id) in enumerate(trees)}
        
        # State variables
        self.current_tree: Optional[GoalTree] = None
        self.current_root_id: Optional[str] = None
        self.main_goal: Optional[Tuple[str, str]] = None
        self.inventory: Dict[str, int] = {}
        self.current_location: str = initial_location
        self.currency: int = initial_currency
        self.defeated_enemies: Set[str] = set()
        self.rescued_npcs: Set[str] = set()
        self.visited_locations: Set[str] = {initial_location}
        # Maps (action, argument) → completion sequence number (monotonically increasing).
        # Dict membership check works identically to the old Set — all "in" tests still hold.
        # The sequence numbers are used to verify ordering_constraints on the GoalTree.
        self.completed_actions: Dict[Tuple[str, str], int] = {}
        self._step_counter: int = 0
        # Per-(action, argument) execution count.  Used by _is_requirement_satisfied to
        # enforce the exact number of repeated actions required by the current tree
        # (e.g., proc_add tasks require perform × N before buy).
        self.action_counts: Dict[Tuple[str, str], int] = {}
        self.done: bool = False
        self.steps: int = 0
        self.action_history: list = []
        self.num_search_node_actions: int = 0  # Count of actions with search node arguments
        self.rng = random.Random()
        # In __init__:
        self.action_target_index: Dict[Tuple[str, str], List[Tuple[GoalTree, str]]] = {}
        # Format: (action, target_lowercase) -> [(tree, node_id), ...]

        # Build index from all trees
        for tree, _ in self.tree_index_map.values():
            for node_id, node in tree.nodes.items():
                if node.meta and node.meta.get("incoming_edge"):
                    action = node.meta.get("incoming_edge")
                    target = node.argument.lower()
                    key = (action, target)
                    
                    # Append to list (multiple trees can have same action+target)
                    if key not in self.action_target_index:
                        self.action_target_index[key] = []
                    self.action_target_index[key].append((tree, node_id))
    
    # ========================================================================
    # Public Interface
    # ========================================================================
    
    def reset(
        self,
        tree_index: int,
        initial_currency: int = 100,
        initial_location: str = "GameStart",
        seed: Optional[int] = None,
        rules_to_skip: List[str] = [],
        task_label: str = "Your task",
    ) -> str:
        """Reset state and return formatted initial observation (header + RPG cards)."""
        # Reset state
        self.inventory = {}
        self.current_location = initial_location
        self.currency = initial_currency
        self.defeated_enemies = set()
        self.rescued_npcs = set()
        self.visited_locations = {initial_location}
        self.done = False
        self.steps = 0
        self.rules_to_skip = rules_to_skip
        self.action_history = []
        self.completed_actions = {}
        self._step_counter = 0
        self.action_counts = {}
        self.num_search_node_actions = 0
        self.rng.seed(seed if seed is not None else 0)

        # Load the tree
        self.current_tree, self.current_root_id = self.tree_index_map[tree_index]
        root_node = self.current_tree.nodes[self.current_root_id]
        root_action = root_node.meta.get("incoming_edge")
        root_argument = root_node.argument
        self.main_goal = (root_action, root_argument)

        # Build RPG cards for all non-hidden nodes that carry displayable properties.
        # Root node first, then remaining nodes in deterministic order.
        skip_set = set(rules_to_skip)

        def _card(node) -> Optional[str]:
            if node.type == "agent" or node.argument in skip_set:
                return None
            p = node.properties
            parts: List[str] = []
            for _n, _v in zip(p.get("attribute_names", []), p.get("attribute_values", [])):
                if _n and _v:
                    parts.append(f"{_n}: {_v}")
            if p.get("location"):
                parts.append(f"@ {p['location']}")
            if p.get("cost") is not None:
                parts.append(f"cost: {p['cost']}")
            if not parts:
                return None
            return f"[ {node.argument} ]  {'  |  '.join(parts)}"

        card_lines: List[str] = []
        root_card = _card(root_node)
        if root_card:
            card_lines.append(root_card)
        for node_id in sorted(self.current_tree.nodes):
            node = self.current_tree.nodes[node_id]
            if node_id == self.current_root_id or node.type == "agent":
                continue
            c = _card(node)
            if c and c not in card_lines:
                card_lines.append(c)

        header = f"=== {task_label}: {root_action} {root_argument} ==="
        if card_lines:
            return header + "\n" + "\n".join(card_lines)
        return header
    
    def step(self, action: str, argument: str) -> Tuple[Optional[str], Observation, Dict[str, float]]:
        """Execute action; return (full_action_str, Observation, {node_id: completion_ratio})."""
        if self.done:
            return None, Observation(False, "Episode already finished", done=True), {}
        
        # Normalize inputs
        action = action.strip().lower()
        argument = argument.strip()

        # Model sometimes puts "action argument" in the action field and leaves
        # argument empty (e.g. action="buy iron_sword", argument="").
        # Detect this and split so the step is not wasted.
        registry_actions_set = set(valid_action_names()) | {"check_inventory", "check_location"}
        if action not in registry_actions_set and " " in action:
            first, rest = action.split(" ", 1)
            if first in registry_actions_set:
                # Prepend any existing argument so nothing is lost
                argument = (rest + " " + argument).strip() if argument else rest
                action = first

        full_action = f"{action} {argument}"

        self.steps += 1
        self.action_history.append(full_action)

        # Track if argument is a search node
        if argument and is_search_node(argument, self.tree_index_map):
            self.num_search_node_actions += 1

        try:
            # Handle special actions without arguments
            if action == "check_inventory":
                completion_map = self._compute_completion_map()
                return "check inventory", Observation(True, self.check_inventory()), completion_map
            elif action == "check_location":
                completion_map = self._compute_completion_map()
                return "check location", Observation(True, self.check_location()), completion_map

            # Validate action type against registry
            registry_actions = valid_action_names()
            all_valid = registry_actions + ["check_inventory", "check_location"]
            if action not in all_valid:
                completion_map = self._compute_completion_map()
                return full_action, Observation(False, f"Invalid action '{action}'. Valid actions: {', '.join(all_valid)}"), completion_map

            # Route to handler — dispatch through registry so new actions only
            # need a handler method added here, nothing else to change.
            handler = getattr(self, f"_handle_{action}", None)
            if handler is None:
                completion_map = self._compute_completion_map()
                return full_action, Observation(False, f"No handler for action '{action}'. Add _handle_{action}() to AdventureEnv."), completion_map
            obs = handler(argument)
            
            # Track successful actions with a monotonic sequence number.
            # All actions always update their timestamp so the model can recover
            # from wrong-order attempts by re-executing the out-of-order steps.
            # perform/drink also increment their execution count (for proc_add
            # count enforcement); other actions are logically single-execution
            # but their timestamp must move so ordering constraints can be
            # re-satisfied (e.g. model does buy before perform, then re-does buy).
            if obs.success:
                key = (action, argument)
                self.completed_actions[key] = self._step_counter
                self._step_counter += 1
                if action in ("perform", "drink"):
                    self.action_counts[key] = self.action_counts.get(key, 0) + 1
            
            # Check if main goal achieved
            if obs.success and (action, argument) == self.main_goal:
                obs.done = True
                obs.message += " 🎉 MAIN GOAL ACHIEVED! Episode complete!"
                self.done = True
            
            # Compute completion map
            completion_map = self._compute_completion_map()
            
            return full_action, obs, completion_map
            
        except Exception as e:
            completion_map = self._compute_completion_map()
            return full_action, Observation(False, f"Error executing action: {e}"), completion_map
    
    def get_observation(self) -> str:
        """Get current state observation."""
        obs = []
        obs.append(f"Location: {self.current_location}")
        obs.append(f"Currency: {self.currency}")
        obs.append(f"Inventory: {dict(self.inventory) if self.inventory else 'empty'}")
        obs.append(f"Defeated: {self.defeated_enemies if self.defeated_enemies else 'none'}")
        obs.append(f"Rescued: {self.rescued_npcs if self.rescued_npcs else 'none'}")
        obs.append(f"Steps: {self.steps}")
        return "\n".join(obs)
    
    def check_inventory(self) -> str:
        """Return inventory contents."""
        if not self.inventory:
            return "Inventory is empty"
        return "Inventory: " + ", ".join(f"{item} x{count}" for item, count in self.inventory.items())
    
    def check_location(self) -> str:
        """Return current location info."""
        return f"Current location: {self.current_location}"
    
    def get_solution(self, tree_index: int) -> List[str]:
        """
        Get a reference solution for a tree by traversing it in execution order.
        Delegates to GoalTree.get_solution(root_id).
        """
        tree, root_id = self.tree_index_map[tree_index]
        return tree.get_solution(root_id)
    
    def get_max_attempts(self, tree_index: int) -> int:
        """
        Get the maximum number of attempts for a tree.
        Search nodes require 2x attempts (get/buy + parent action attempt).
        """
        tree, root_id = self.tree_index_map[tree_index]
        tree_length = len(tree.get_solution(root_id))
        nums_of_tries = sum(len(v.properties.get("props_options", [])) for v in tree.nodes.values() if v.meta.get("is_search_node"))

        return (tree_length + (nums_of_tries * 2)) + 10
        

    def get_solution_cost(self, tree_index: int) -> int:
        """
        Calculate the cost of the solution for a tree.
        Delegates to GoalTree.get_solution_cost(root_id).
        """
        tree, root_id = self.tree_index_map[tree_index]
        return tree.get_solution_cost(root_id)
    
    # ========================================================================
    # Rule Generation
    # ========================================================================
    
    def _generate_rules_for_tree(self, tree: GoalTree, root_id: str, rules_to_skip: List[str] = []) -> List[str]:
        """
        Generate agent-facing rules from a tree by traversing dependencies.
        
        For each non-leaf action node, generate a rule like:
        "defeat dragon requires: need sword (at market, costs 50), must be at volcano"
        """
        rules = []
        visited = set()
        
        def traverse(node_id: str):
            if node_id in visited or node_id == tree.agent_node_id:
                return
            visited.add(node_id)
            
            node = tree.nodes[node_id]
            action = node.meta.get("incoming_edge")
            if not action:
                return

            # Only generate a rule header for actions marked generates_rule=True
            action_def = ACTION_REGISTRY.get(action)
            if action_def is None or not action_def.generates_rule:
                return

            rule_parts = [f"{action} {node.argument} requires:"]
            requirements = []

            for edge_idx in tree.in_edges.get(node_id, []):
                edge = tree.edges[edge_idx]
                dep_node = tree.nodes[edge.src]
                dep_action = dep_node.meta.get("incoming_edge")

                if dep_node.type == "agent":
                    continue
                if dep_node.argument in rules_to_skip:
                    continue

                dep_def = ACTION_REGISTRY.get(dep_action)

                if dep_action == "go":
                    # "go" always rendered as a location constraint on the parent
                    if "location" in node.properties and node.properties["location"]:
                        requirements.append(f"must be at {node.properties['location']}")

                elif dep_action in ("get", "buy"):
                    if dep_node.meta.get("is_search_node"):
                        tested_props = dep_node.properties.get("props_options", [])
                        location = dep_node.properties.get("location", "unknown")
                        cost = dep_node.properties.get("cost", 0)
                        instance_kind = dep_node.properties.get("instance_kind", "")
                        options_str = ", ".join([f"{tested}_{instance_kind}" for tested in tested_props])
                        cost_str = f"costs {cost} currency" if cost > 0 else "free"
                        requirements.append(
                            f"need {instance_kind} (at {location}, {cost_str}) but there are many options "
                            f"provided: {options_str}, you may figure out which one to use by trying them"
                        )
                    elif dep_def and dep_def.dep_display:
                        cost = dep_node.properties.get("cost", 0)
                        loc  = dep_node.properties.get("location", "unknown")
                        requirements.append(
                            dep_def.dep_display.format(arg=dep_node.argument, loc=loc, cost=cost, name=dep_action)
                        )

                elif dep_def and dep_def.dep_display:
                    # Generic: use dep_display template from the registry
                    loc  = dep_node.properties.get("location", "unknown")
                    cost = dep_node.properties.get("cost", 0)
                    requirements.append(
                        dep_def.dep_display.format(arg=dep_node.argument, loc=loc, cost=cost, name=dep_action)
                    )

                traverse(edge.src)

            if requirements:
                numbered_reqs = [f"({i+1}) {req}" for i, req in enumerate(requirements)]
                rule_parts.append(", ".join(numbered_reqs))
                rules.append(" ".join(rule_parts))
        
        traverse(root_id)
        # Use sorted order so rule list is deterministic before shuffle (set order varies by run).
        return sorted(set(rules))

    def _compute_completion_map(self) -> Dict[str, float]:
        """
        Compute completion ratio for each node in the tree.
        
        Returns:
            Dictionary mapping node_id to completion_ratio (0.0 to 1.0)
        """
        completion_map = {}
        
        for node_id, node in self.current_tree.nodes.items():
            # Skip agent node
            if node.type == "agent":
                continue
            
            # Get incoming edges (dependencies) excluding agent
            in_edge_indices = self.current_tree.in_edges.get(node_id, [])
            dependencies = []
            
            for edge_idx in in_edge_indices:
                edge = self.current_tree.edges[edge_idx]
                dep_node = self.current_tree.nodes[edge.src]
                
                # Skip agent dependencies
                if dep_node.type == "agent":
                    continue
                    
                dependencies.append((dep_node.meta.get("incoming_edge"), dep_node.argument))
            
            # Calculate completion ratio
            if len(dependencies) == 0:
                # Leaf nodes with no dependencies (or only agent dependency)
                # are complete if their action is executed
                node_action = node.meta.get("incoming_edge")
                node_argument = node.argument
                completion_map[node_id] = 1.0 if (node_action, node_argument) in self.completed_actions else 0.0
            else:
                # Non-leaf nodes: ratio of completed dependencies
                completed_deps = sum(1 for dep in dependencies if dep in self.completed_actions)
                completion_map[node_id] = completed_deps / len(dependencies)
        
        return completion_map
    
    # ========================================================================
    # Action Handlers
    # ========================================================================
    
    def _handle_go(self, location: str) -> Observation:
        """Handle 'go' action."""
        if location == self.current_location:
            return Observation(True, f"Already at {location}")
        
        self.current_location = location
        self.visited_locations.add(location)
        return Observation(True, f"Traveled to {location}. You are now at {location}.")
    
    def _apply_search_node_replacement(self, item: str) -> Optional["Observation"]:
        """If item is a search node and inventory already has an item of the same kind,
        remove the old one, add the new one, and return an Observation. Otherwise None."""
        if not is_search_node(item, self.tree_index_map):
            return None
        item_kind = get_item_instance_kind_from_all_trees(item, self.tree_index_map)
        if not item_kind:
            return None
        existing_item = check_inventory_has_instance_kind(self.inventory, item_kind, self.tree_index_map)
        if not existing_item:
            return None
        self.inventory[existing_item] = 0
        self.inventory[item] = self.inventory.get(item, 0) + 1
        return Observation(True, f"You already have {existing_item}, and now you replaced it with {item}.")

    def _handle_get(self, item: str) -> Observation:
        """Handle 'get' action."""
        # Check cost across all trees (not just current tree) for consistency with rules
        cost = get_object_cost_from_all_trees(item, self.tree_index_map)
        if cost > 0:
            return Observation(False, f"Cannot get {item} directly. It costs {cost} currency.")
        
        # Check location across all trees
        item_location = get_entity_location_from_all_trees(item, self.tree_index_map)
        if item_location and item_location != self.current_location:
            return Observation(False, f"Cannot get {item}. It's at {item_location}, but you're at {self.current_location}")
        
        replacement = self._apply_search_node_replacement(item)
        if replacement is not None:
            return replacement

        self.inventory[item] = self.inventory.get(item, 0) + 1
        return Observation(True, f"Picked up {item}, You now have {item}.")
    
    def _handle_buy(self, item: str) -> Observation:
        """Handle 'buy' action."""
        # Check location across all trees (gen + source demo trees)
        item_location = get_entity_location_from_all_trees(item, self.tree_index_map)
        if item_location and item_location != self.current_location:
            return Observation(False, f"Cannot buy {item} here. Go to {item_location} first.")

        # Check cost across all trees
        cost = get_object_cost_from_all_trees(item, self.tree_index_map)
        if cost == 0:
            return Observation(False, f"{item} is not for sale.")
        if self.currency < cost:
            return Observation(False, f"Not enough currency. {item} costs {cost}, but you only have {self.currency}")
        
        replacement = self._apply_search_node_replacement(item)
        if replacement is not None:
            self.currency -= cost
            return replacement

        self.currency -= cost
        self.inventory[item] = self.inventory.get(item, 0) + 1
        return Observation(True, f"Bought {item} for {cost} currency (remaining: {self.currency}). You now have {item}.")
    
    def _handle_defeat(self, enemy: str) -> Observation:
        """Handle 'defeat' action."""
        if enemy in self.defeated_enemies:
            return Observation(True, f"{enemy} already defeated")
        
        # Check location across all trees for consistency with rules
        enemy_location = get_entity_location_from_all_trees(enemy, self.tree_index_map)
        if enemy_location and enemy_location != self.current_location:
            return Observation(False, f"Cannot defeat {enemy}. Requirements not met.")
        
        # Check tree requirements (includes tool/item requirements)
        if not self._check_tree_requirements("defeat", enemy):
            # Check if failure is due to wrong search node item
            kind = self._get_failed_search_node_kind("defeat", enemy)
            if kind:
                return Observation(False, f"Cannot defeat {enemy} yet. Wrong {kind}.")
            return Observation(False, f"Cannot defeat {enemy} yet. Requirements not met.")
        
        # Consume items used for this action
        self._consume_items_for_action("defeat", enemy)
        
        self.defeated_enemies.add(enemy)
        return Observation(True, f"Defeated {enemy}!")
    
    def _handle_rescue(self, npc: str) -> Observation:
        """Handle 'rescue' action."""
        if npc in self.rescued_npcs:
            return Observation(True, f"{npc} already rescued")
        
        # Check location across all trees for consistency with rules
        npc_location = get_entity_location_from_all_trees(npc, self.tree_index_map)
        if npc_location and npc_location != self.current_location:
            return Observation(False, f"Cannot rescue {npc}. Requirements not met.")
        
        # Check tree requirements
        if not self._check_tree_requirements("rescue", npc):
            # Check if failure is due to wrong search node item
            kind = self._get_failed_search_node_kind("rescue", npc)
            if kind:
                return Observation(False, f"Cannot rescue {npc} yet. Wrong {kind}.")
            return Observation(False, f"Cannot rescue {npc} yet. Requirements not met.")
        
        # Consume items used for this action
        self._consume_items_for_action("rescue", npc)
        
        self.rescued_npcs.add(npc)
        return Observation(True, f"Rescued {npc}!")
    
    def _handle_perform(self, ritual: str) -> Observation:
        """Handle 'perform' action (e.g. perform moonlit_prayer). No location dependency.
        Repeatable: calling perform multiple times is allowed and each call is counted."""
        return Observation(True, f"Performed {ritual}.")

    def _handle_drink(self, potion: str) -> Observation:
        """Handle 'drink' action (e.g. drink strength_brew). No location dependency.
        Repeatable: calling drink multiple times is allowed and each call is counted."""
        return Observation(True, f"Drank {potion}.")

    # ========================================================================
    # Internal Helpers
    # ========================================================================
    
    def _validate_target(self, action: str, target: str) -> bool:
        """Validate that target exists in any tree with the specified action."""
        return (action, target.lower()) in self.action_target_index
    
    def _trees_to_check_for(self, action: str, target: str) -> List[Tuple["GoalTree", str]]:
        """Return (tree, node_id) pairs for the given action/target, current tree first."""
        key = (action, target.lower())
        if key not in self.action_target_index:
            return []
        ordered = []
        for tree, node_id in self.action_target_index[key]:
            if tree == self.current_tree:
                ordered.insert(0, (tree, node_id))
            else:
                ordered.append((tree, node_id))
        return ordered

    def _get_failed_search_node_kind(self, action: str, target: str) -> Optional[str]:
        """Get the 'kind' property of a failed search node dependency, if any."""
        trees_to_check = self._trees_to_check_for(action, target)
        if not trees_to_check:
            return None

        for tree, target_node_id in trees_to_check:
            for edge_idx in tree.in_edges.get(target_node_id, []):
                edge = tree.edges[edge_idx]
                dep_node = tree.nodes[edge.src]
                
                if dep_node.type == "agent":
                    continue
                
                # Check if this is a search node that's not satisfied
                if dep_node.meta.get("is_search_node"):
                    dep_action = dep_node.meta.get("incoming_edge")
                    dep_target = dep_node.argument
                    
                    if not self._is_requirement_satisfied(dep_action, dep_target):
                        # This search node failed, return its kind
                        return dep_node.properties.get("instance_kind", dep_node.properties.get("kind", "item"))
        
        return None
    
    def _check_tree_requirements(self, action: str, target: str) -> bool:
        """
        Check if all tree requirements for an action are satisfied.

        Two-phase check:
          1. All direct prerequisite nodes of the target node are satisfied (state check).
          2. All ordering_constraints on the tree are respected (temporal check).
             A constraint ((before_action, before_arg), (after_action, after_arg)) fails
             only when BOTH steps have been completed but before was done after after.
             This lets the agent execute steps in any order but blocks the terminal goal
             if the required temporal ordering was violated.
        """
        trees_to_check = self._trees_to_check_for(action, target)
        if not trees_to_check:
            raise ValueError(f"No requirements found for action {action} and target {target}")

        for tree, target_node_id in trees_to_check:
            # Phase 1: prerequisite satisfaction
            all_satisfied = True
            for edge_idx in tree.in_edges.get(target_node_id, []):
                edge = tree.edges[edge_idx]
                dep_node = tree.nodes[edge.src]

                if dep_node.type == "agent":
                    continue

                dep_action = dep_node.meta.get("incoming_edge")
                dep_target = dep_node.argument

                if not self._is_requirement_satisfied(dep_action, dep_target):
                    all_satisfied = False
                    break

            if not all_satisfied:
                continue

            # Phase 2: ordering constraints
            ordering_ok = True
            for (before_action, before_arg), (after_action, after_arg) in getattr(tree, "ordering_constraints", []):
                before_ts = self.completed_actions.get((before_action, before_arg))
                after_ts  = self.completed_actions.get((after_action, after_arg))
                # Only enforce when both steps have been completed
                if before_ts is not None and after_ts is not None:
                    if before_ts > after_ts:
                        ordering_ok = False
                        break

            if ordering_ok:
                return True

        return False
    
    def _is_requirement_satisfied(self, action: str, target: str) -> bool:
        """Check if a requirement is satisfied."""
        if action == "defeat":
            return target in self.defeated_enemies
        elif action == "rescue":
            return target in self.rescued_npcs
        elif action in ("get", "buy"):
            return target in self.inventory and self.inventory[target] > 0
        elif action == "go":
            return self.current_location == target
        elif action == "perform":
            key = ("perform", target)
            required = self._count_required_actions("perform", target)
            return self.action_counts.get(key, 0) >= required
        elif action == "drink":
            key = ("drink", target)
            required = self._count_required_actions("drink", target)
            return self.action_counts.get(key, 0) >= required
        return False

    def _count_required_actions(self, action: str, target: str) -> int:
        """Count how many times (action, target) appears as a node in the current tree.
        Used to enforce repeated-action requirements (e.g. perform × N for proc_add)."""
        if self.current_tree is None:
            return 1
        count = sum(
            1 for node in self.current_tree.nodes.values()
            if node.meta.get("incoming_edge") == action and node.argument == target
        )
        return max(count, 1)
    
    def _consume_items_for_action(self, action: str, target: str):
        """
        Consume items (from get/buy actions) that were used as dependencies for this action.
        This ensures items are single-use and prevents agents from reusing the same item
        for multiple actions.
        """
        key = (action, target.lower())
        if key not in self.action_target_index:
            return
        
        # Priority: consume from current tree first, then others
        trees_to_check = []
        
        # Add current tree occurrences first
        for tree, node_id in self.action_target_index[key]:
            if tree == self.current_tree:
                trees_to_check.insert(0, (tree, node_id))
            else:
                trees_to_check.append((tree, node_id))
        
        # Find the first tree where all requirements are satisfied and consume its items
        for tree, target_node_id in trees_to_check:
            items_to_consume = []
            all_satisfied = True
            
            for edge_idx in tree.in_edges.get(target_node_id, []):
                edge = tree.edges[edge_idx]
                dep_node = tree.nodes[edge.src]
                
                if dep_node.type == "agent":
                    continue
                
                dep_action = dep_node.meta.get("incoming_edge")
                dep_target = dep_node.argument
                
                # Only track items (get/buy actions) for consumption
                if dep_action in ["get", "buy"]:
                    items_to_consume.append(dep_target)
                
                # Check if this requirement is satisfied
                if not self._is_requirement_satisfied(dep_action, dep_target):
                    all_satisfied = False
                    break
            
            # If all requirements satisfied for this tree, consume the items
            if all_satisfied:
                for item in items_to_consume:
                    if item in self.inventory and self.inventory[item] > 0:
                        self.inventory[item] -= 1
                        if self.inventory[item] == 0:
                            del self.inventory[item]
                return  # Only consume from the first matching tree
