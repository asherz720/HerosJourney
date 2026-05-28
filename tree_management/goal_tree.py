"""
This goal tree is used to create and parse goal tree.
"""
from __future__ import annotations

import json
from typing import List, Dict, Optional, Any, Literal, Tuple
from dataclasses import dataclass, field
import uuid
from collections import defaultdict, deque

NodeType = Literal["root", "leaf", "agent"]

# GoalTree Class
@dataclass
class Node: # each node represents an argument
    id: str
    type: NodeType
    argument: str
    properties: Dict[str,Any] = field(default_factory=dict) # properties of the arguments (location, kind, cost, etc)
    meta: Dict[str, Any] = field(default_factory=dict) #incoming_edge

@dataclass
class Edge:
    src: str
    dst: str
    action: str
    meta: Dict[str, Any] = field(default_factory=dict) # execution order, action class-

class GoalTree:
    """Manages goal recipes and generates optimal action sequences."""
    
    def __init__(self, root_id:Optional[str] = None):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.out_edges: Dict[str, List[int]] = defaultdict(list) # node_id -> edge_indices
        self.in_edges: Dict[str, List[int]] = defaultdict(list) # node_id -> edge_indices
        self.root_id: Optional[str] = root_id 
        self.agent_node_id: Optional[str] = None
        # Temporal ordering constraints between prerequisites, checked at the terminal goal.
        # Each entry ((before_action, before_arg), (after_action, after_arg)) asserts that
        # before_action before_arg must have completed before after_action after_arg for the
        # root goal to succeed. Populated from process "ordering_constraints" at build time.
        self.ordering_constraints: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
        
        # Create agent node automatically - every tree needs one
        self._create_agent_node()
    

    def _new_id(self) -> str:
        return uuid.uuid4().hex[:10] # every node needs a unique id; also for edge reference; increase the truncation length if you intend to generate millions of nodes

    def _add_node(self, type: str, argument: str, *, properties: Optional[Dict[str, Any]] = None, 
                  node_id: Optional[str]=None, meta: Optional[Dict[str, Any]] = None) -> str:
        nid = node_id or self._new_id()
        if nid in self.nodes:
            raise ValueError(f"Node with id {nid} already exists")
        
        node = Node(id=nid, type=type, argument=argument, 
                   properties=properties or {}, meta=meta or {})
        self.nodes[nid] = node

        if self.root_id is None and type == "root":
            self.root_id = nid
        
        return nid
    
    def _add_edge(self, src: str, dst: str, action: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
        if src not in self.nodes or dst not in self.nodes:
            raise ValueError(f"Edge {src} -> {dst} invalid: node does not exist")

        edge = Edge(src=src, dst=dst, action=action, meta=meta or {})
        self.edges.append(edge)
        self.out_edges[src].append(len(self.edges) - 1)
        self.in_edges[dst].append(len(self.edges) - 1)
    
    def _create_agent_node(self) -> None:
        """Create the agent node. Called automatically during tree initialization."""
        self.agent_node_id = self._add_node(
            type="agent", argument='agent', properties={},
            node_id='agent_node', meta={})
    
    def validate(self, *, require_dag: bool = True) -> None:
        if self.root_id is None: 
            raise ValueError("Root node is not set")
        if self.root_id not in self.nodes:
            raise ValueError(f"Root node {self.root_id} does not exist")
        for e in self.edges:
            if e.src not in self.nodes or e.dst not in self.nodes:
                raise ValueError(f"Edge {e.src} -> {e.dst} invalid: node does not exist")
        if require_dag and not self.is_dag():
            raise ValueError("The graph is not a DAG")

    def topo_sort(self) -> List[str]:
        indeg = {nid: 0 for nid in self.nodes}
        for e in self.edges:
            indeg[e.dst] += 1
        
        q = deque(nid for nid in self.nodes if indeg[nid] == 0)
        order = []
        while q:
            nid = q.popleft()
            order.append(nid)
            for ei in self.out_edges.get(nid, []):
                indeg[self.edges[ei].dst] -= 1
                if indeg[self.edges[ei].dst] == 0:
                    q.append(self.edges[ei].dst)
        
        if len(order) != len(self.nodes):
            raise ValueError("The graph is not a DAG")
        return order
    
    def is_dag(self) -> bool:
        try:
            self.topo_sort()
            return True
        except ValueError:
            return False
    
    def depth_map(self) -> Dict[str, int]:
        """Compute depth of each node (distance to furthest descendant)."""
        order = self.topo_sort()
        depth = {nid: 0 for nid in self.nodes}
        for nid in reversed(order):
            child_depths = [depth[self.edges[ei].dst] for ei in self.out_edges.get(nid, [])]
            depth[nid] = max(child_depths) + 1 if child_depths else 0
        return depth
    def max_depth(self, exclude_agent: bool = True) -> int:
        """
        Calculate maximum depth of the tree. Also exclude the agent as a level of the tree.
        """
        d = self.depth_map()
        if exclude_agent and self.agent_node_id:
            d = {k: v for k, v in d.items() if k != self.agent_node_id}
        return max(d.values()) if d else 0

    def get_solution(self, root_id: Optional[str] = None) -> List[str]:
        """
        Generate reference solution by traversing the tree in execution order.

        execution_order on edges uses the convention: lower value = executed first.
        Steps with the same execution_order are parallel; their relative order in the
        trace is determined by their position in the process steps list (stable sort).
        root_id defaults to self.root_id.
        """
        rid = root_id if root_id is not None else self.root_id
        if rid is None:
            return []
        actions: List[str] = []
        visited: set = set()

        def collect_actions(node_id: str) -> None:
            if node_id in visited or node_id == self.agent_node_id:
                return
            visited.add(node_id)
            node = self.nodes[node_id]
            action = node.meta.get("incoming_edge")
            dependencies = []
            for edge_idx in self.in_edges.get(node_id, []):
                edge = self.edges[edge_idx]
                if self.nodes[edge.src].type != "agent":
                    execution_order = edge.meta.get("execution_order", 0)
                    dependencies.append((execution_order, edge.src))
            # Lower execution_order = earlier in the trace; stable sort preserves list-position
            # tie-breaking for parallel steps (same execution_order value).
            dependencies.sort(key=lambda x: x[0])
            for _, dep_id in dependencies:
                collect_actions(dep_id)
            if action and node.type != "agent":
                actions.append(f"{action} {node.argument}")

        collect_actions(rid)
        return actions

    def get_solution_cost(self, root_id: Optional[str] = None) -> int:
        """Total currency cost of the solution (sum of buy costs)."""
        solution = self.get_solution(root_id)
        cost = 0
        for action in solution:
            parts = action.split()
            if len(parts) >= 2 and parts[0] == "buy":
                item = " ".join(parts[1:])
                for node in self.nodes.values():
                    if node.argument == item and "cost" in node.properties:
                        cost += node.properties["cost"]
                        break
        return cost

    # Serialize and parse 
    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_id": self.root_id,
            "agent_node_id": self.agent_node_id,
            "nodes": [self.nodes[n].__dict__ for n in self.nodes],
            "edges": [self.edges[i].__dict__ for i in range(len(self.edges))],
            "ordering_constraints": [
                [list(before), list(after)] for before, after in self.ordering_constraints
            ],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GoalTree":
        gt = GoalTree(root_id=None)
        gt.root_id = d.get("root_id")
        gt.agent_node_id = 'agent_node'
        
        # Clear the nodes dict and rebuild from saved data
        # (agent node was already created in __init__, will be overwritten)
        gt.nodes.clear()
        gt.edges.clear()
        gt.out_edges.clear()
        gt.in_edges.clear()
        
        for nd in d["nodes"]:
            node = Node(**nd)
            gt.nodes[node.id] = node
        for ed in d["edges"]:
            edge = Edge(**ed)
            gt.edges.append(edge)
            gt.out_edges[edge.src].append(len(gt.edges)-1)
            gt.in_edges[edge.dst].append(len(gt.edges)-1)
        gt.ordering_constraints = [
            (tuple(before), tuple(after))
            for before, after in d.get("ordering_constraints", [])
        ]
        return gt
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    @staticmethod
    def from_json(s: str) -> "GoalTree":
        return GoalTree.from_dict(json.loads(s))

