import numpy as np
from matplotlib.pylab import rand
import networkx as nx
from typing import Dict, List, Tuple
import util


class CEOption(util.Action):
    """Represents a leaf node (Reuse, Recycle) with Fuzzy Logic."""

    def __init__(
        self,
        name: str,
        base_value: float,
        # fuzziness_k: float, # Slope of the sigmoid (original version)
        # threshold: float, # Center of the sigmoid (original version)
        fuzzy_params: Tuple[float, float, float, float],
        cost: float = 0.0,
        time: float = 0.0,
        resources: list = None,
        prerequisites: list = None,
    ) -> None:
        super().__init__(name, cost, time, resources, prerequisites=prerequisites)
        self.base_value = base_value

        # self.k = fuzziness_k  # Slope of the sigmoid (original version)
        # self.threshold = threshold  # Center of the sigmoid (original version)

        # fuzzy_params is a tuple of 4 points (a, b, c, d) defining the trapezoid
        # recovers a crisp value == operation if all points are the same
        self.fuzzy_params = fuzzy_params

    ### Sigmoid memebershiip function (original version)
    # def get_fuzzy_membership(self, observed_health: float) -> float:
    #     """Returns degree of membership [0, 1]."""
    #     # Avoid overflow in exp
    #     try:
    #         val = math.exp(-self.k * (observed_health - self.threshold))
    #     except OverflowError:
    #         val = float("inf")
    #     return 1.0 / (1.0 + val)

    def get_fuzzy_membership(self, observed_health: float) -> float:
        """Returns degree of membership [0, 1] based on a trapezoidal function."""
        a, b, c, d = self.fuzzy_params

        # 1. Complete rejection (Outside the trapezoid)
        if observed_health < a or observed_health > d:
            return 0.0

        # 2. Ramp up (Partial acceptance)
        if a <= observed_health < b:
            return (observed_health - a) / (b - a) if b > a else 1.0

        # 3. Core Safe Zone (Perfect condition, full reward)
        if b <= observed_health <= c:
            return 1.0

        # 4. Ramp down (Risk penalty as health degrades)
        if c < observed_health <= d:
            return (d - observed_health) / (d - c) if d > c else 1.0

        return 0.0

    def get_expected_reward(self, observed_health: float) -> float:
        """Returns scalar reward: (Membership * Value) - Cost."""
        mu = self.get_fuzzy_membership(observed_health)
        return (
            self.base_value * mu
        ) - self.cost  # deterministic reward based on fuzzy membership
        # return (
        #     (self.base_value + (50 * np.random.rand()) - 25) * mu
        # ) - self.cost  # stochastic reward with noise, scaled by fuzzy membership


class DisassemblyEdge(util.Action):
    """An edge representing a disassembly action with cost."""

    def __init__(self, action_name: str, cost: float = 0.0, time: float = 0.0) -> None:
        super().__init__(action_name, cost, time)

        # FIX: Only append history brackets if history actually exists
        hist_str = ",".join(sorted(list(history)))
        # Action -> List of resulting Nodes (Hyper-edges for disassembly)
        self.children: List[Tuple[util.Action, List["Node"]]] = []
        self.ce_options: List[CEOption] = []


class Node:
    """A node in the State-Augmented Graph encoding physical product + history."""

    def __init__(self, product: util.Product, history: frozenset) -> None:
        self.product = product
        self.history = history

        # State ID is Physical ID + Sorted History for uniqueness
        hist_str = ",".join(sorted(list(history)))
        if hist_str:
            self.id = f"{product.name}_[{hist_str}]"
        else:
            self.id = product.name

        # Action -> List of resulting Nodes (Hyper-edges for disassembly)
        self.children: List[Tuple[util.Action, List["Node"]]] = []
        self.ce_options: List[CEOption] = []

    def __repr__(self):
        return f"Node({self.id})"


class StateAugmentedDisassemblyGraph:
    """Builds the Unrolled Decision Graph using NetworkX."""

    def __init__(self, root_product: util.Product) -> None:
        self.state_map: Dict[Tuple[str, frozenset], Node] = (
            {}
        )  # memoisation of (product, history) to Node
        self.graph = nx.DiGraph()  # empty graph to be populated
        self.root_key = (root_product.name, frozenset())
        self._build_tree(root_product, frozenset())

    def _build_tree(self, product: util.Product, current_history: frozenset) -> Node:
        state_key = (product.name, current_history)
        # check if we have called this state before and return the existing node to avoid redundant computations
        if state_key in self.state_map:
            return self.state_map[state_key]

        node = Node(product, current_history)
        self.state_map[state_key] = node
        self.graph.add_node(node.id)

        # 1. Attach valid Terminal CE Options based on History
        for opt in product.ce_options:
            if set(opt.prerequisites).issubset(current_history):
                node.ce_options.append(opt)
                # Add terminal node for visual completeness in NetworkX
                term_id = f"{opt.name}_{node.id}"
                self.graph.add_node(term_id)
                self.graph.add_edge(node.id, term_id, action=opt.name)

        # 2. Attach valid Actions based on History
        for action in product.actions:
            # Action hasn't been done yet AND prerequisites are met
            if action.name not in current_history and set(
                action.prerequisites
            ).issubset(current_history):

                next_history = current_history.union({action.name})

                if action.is_disassembly:
                    # Splits product -> children inherit the history + the disassembly action itself
                    child_nodes = []
                    for comp in product.components:
                        # FIX: Pass the inherited history down, do NOT reset to frozenset()
                        child_node = self._build_tree(comp, next_history)
                        child_nodes.append(child_node)
                        self.graph.add_edge(node.id, child_node.id, action=action.name)
                    node.children.append((action, child_nodes))
                else:
                    # Prep Action -> same physical product, new history
                    child_node = self._build_tree(product, next_history)
                    node.children.append((action, [child_node]))
                    self.graph.add_edge(node.id, child_node.id, action=action.name)

        return node


class Solver:
    def solve(self, start_node: Node, observations: dict) -> dict:
        memo = {}
        policy = {}

        def get_value(node: Node) -> float:
            if node.id in memo:
                return memo[node.id]

            # Physical component name is used to fetch sensor observations
            obs = observations.get(node.product.name, 0.5)

            # 1. Terminal CE Options
            ce_rewards = [
                (opt.get_expected_reward(obs), opt.name) for opt in node.ce_options
            ]
            best_ce_val, best_ce_name = (
                max(ce_rewards, key=lambda x: x[0])
                if ce_rewards
                else (-float("inf"), "None")
            )

            # 2. Process Actions (Prep or Disassembly)
            action_rewards = []
            for action, child_nodes in node.children:
                # Sum values of all resulting states (handles hyper-edges naturally)
                net_val = -action.cost + sum(get_value(child) for child in child_nodes)
                action_rewards.append((net_val, action.name))

            best_act_val, best_act_name = (
                max(action_rewards, key=lambda x: x[0])
                if action_rewards
                else (-float("inf"), "None")
            )

            # 3. Decision
            if best_ce_val >= best_act_val:
                memo[node.id] = best_ce_val
                policy[node.id] = f"CE: {best_ce_name}"
            else:
                memo[node.id] = best_act_val
                policy[node.id] = f"Action: {best_act_name}"

            return memo[node.id]

        get_value(start_node)
        return policy


if __name__ == "__main__":
    pass
