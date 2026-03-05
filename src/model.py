import math
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
        return (self.base_value * mu) - self.cost


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


class StateAugmentedGraph:
    """Builds the Unrolled Decision Graph using NetworkX."""

    def __init__(self, root_product: util.Product) -> None:
        self.graph = nx.MultiDiGraph()
        self.product_map: Dict[Tuple[str, frozenset], Node] = {}
        self.root = self._build_tree(root_product, frozenset())

    def _build_tree(self, product: util.Product, current_history: frozenset) -> Node:
        state_key = (product.name, current_history)
        if state_key in self.product_map:
            return self.product_map[state_key]

        node = Node(product, current_history)
        self.product_map[state_key] = node
        self.graph.add_node(node.id)

        # 1. Attach valid Terminal CE Options based on History
        for opt in product.ce_options:
            if set(opt.prerequisites).issubset(current_history):
                node.ce_options.append(opt)
                # Add terminal node for visual completeness in NetworkX
                term_id = f"TERMINAL_{opt.name}_{node.id}"
                self.graph.add_node(term_id)
                self.graph.add_edge(node.id, term_id, action=opt.name)

        # 2. Attach valid Actions based on History
        for action in product.actions:
            # Action hasn't been done yet AND prerequisites are met
            if action.name not in current_history and set(
                action.prerequisites
            ).issubset(current_history):

                if action.is_disassembly:
                    # Splits product -> children inherit the history + the disassembly action itself
                    child_nodes = []
                    next_history = frozenset(current_history.union({action.name}))

                    for comp in product.components:
                        # FIX: Pass the inherited history down, do NOT reset to frozenset()
                        child_node = self._build_tree(comp, next_history)
                        child_nodes.append(child_node)
                        self.graph.add_edge(node.id, child_node.id, action=action.name)
                    node.children.append((action, child_nodes))
                else:
                    # Prep Action -> same physical product, new history
                    next_history = current_history.union({action.name})
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
    print("=" * 60)
    print("TEST 2: State-Augmented Graph & Fuzzy DP Solver")
    print("-" * 60)
    print("SCENARIO:")
    print("  A Module containing 1 Cell. Extracting the cell costs £20.")
    print("  Module CE Option: Recycle (Base Value £15, flat fuzzy acceptance).")
    print("  Cell CE Option: Reuse (Base Value £50, strict fuzzy threshold at 0.8).")
    print("\nINPUT OBSERVATIONS:")
    print("  The Module looks degraded (Health = 0.50).")
    print("  The internal Cell looks excellent (Health = 0.95).")
    print("\nEXPECTED OUTCOME:")
    print("  Recycling the module yields £15.")
    print("  Disassembling costs £20, but unlocks £50 (Net £30).")
    print(
        "  The solver should choose 'Disassemble' for the Module, and 'Reuse' for the Cell."
    )
    print("-" * 60)

    # 1. Setup Physical Hierarchy
    cell_hs = util.HealthState(0.95, {})
    cell = util.Product(cell_hs, name="Cell_1")
    cell.ce_options = [
        CEOption("Reuse_Cell", base_value=50, fuzziness_k=10, threshold=0.8),
    ]

    mod_hs = util.HealthState(0.5, {})
    module = util.Product(mod_hs, name="Module_1")
    module.components = [cell]  # Link hierarchy
    module.actions = [util.Action("Extract_Cell_1", cost=20.0, time=5.0)]
    module.ce_options = [
        CEOption("Recycle_Module", base_value=15, fuzziness_k=0.1, threshold=0.0)
    ]

    # 2. Build Decision Graph & Solve
    sag = StateAugmentedGraph(module)
    observations = {"Module_1": 0.5, "Cell_1": 0.95}

    solver = Solver()
    policy = solver.solve(sag.root, observations)

    # 3. Output
    print("SOLVER POLICY OUTPUT:")
    for node_id, decision in policy.items():
        print(f"  > Node [{node_id}]: {decision}")

    print("=" * 60 + "\n")


# import math
# from util import Action, Product, HealthState


# class CEOption(Action):
#     """Represents a leaf node (Reuse, Recycle, etc.) with Fuzzy Logic."""

#     def __init__(
#         self,
#         name,
#         base_value,
#         fuzziness_k,
#         threshold,
#         cost=0.0,
#         time=0.0,
#         resources: list = [],
#     ) -> None:
#         super().__init__(name, cost, time, resources)
#         self.base_value = base_value
#         self.k = fuzziness_k  # Slope of the sigmoid
#         self.threshold = threshold  # Center of the sigmoid

#     def get_fuzzy_membership(self, observed_health: float) -> float:
#         """
#         The 'Fuzzy' part. Returns degree of membership [0, 1].
#         Sigmoid function: 1 / (1 + e^(-k * (x - x0)))
#         """
#         return 1.0 / (1.0 + math.exp(-self.k * (observed_health - self.threshold)))

#     def get_expected_reward(self, observed_health: float) -> float:
#         """The 'Interface' to the MDP. Returns a scalar reward."""
#         mu = self.get_fuzzy_membership(observed_health=observed_health)
#         # Hybrid logic: Fuzzy membership acts as a probability/discount on value
#         return self.base_value * mu


# class DissassemblyEdge(Action):
#     """An edge representing a disassembly action with cost."""

#     def __init__(self, action_name, cost=0.0, time=0.0, resources: list = []) -> None:
#         super().__init__(action_name, cost, time, resources)
#         self.required = []  # List of required edges/actions to perform this action


# class Node:
#     """A node in the State-Augmented Graph."""

#     def __init__(self, product: Product, is_terminal=False) -> None:
#         self.product = product
#         self.id = product.name
#         self.children = []  # List of (DisassemblyEdge, Node) tuples
#         self.ce_options = []  # List of CEOption objects


# class StateAugmentedGraph:
#     """The State-Augmented Graph."""

#     def __init__(self, root: Node) -> None:
#         self.root = root
#         self.productNodeList = []  # All nodes for reference
#         self.disassemblyEdges = []  # All edges for reference
#         self.ceOptions = []  # All CE options for reference


# class Solver:
#     """The Exact DP Solver."""

#     def solve(self, start_node: Node, observations: dict) -> dict:
#         """
#         Returns a map of {NodeID: OptimalAction}
#         Uses Backward Induction (Recursion with Memoization).
#         """
#         memo = {}

#         def get_value(node) -> float:
#             if node.id in memo:
#                 return memo[node.id]

#             # 1. Evaluate Terminal CE Options (Fuzzy Step)
#             # We use the noisy observation here!
#             obs = observations.get(node.component_id, 0.5)
#             ce_values = [opt.get_expected_reward(obs) for opt in node.ce_options]
#             best_ce_val = max(ce_values) if ce_values else -float("inf")

#             # 2. Evaluate Disassembly Steps (MDP Step)
#             disassembly_vals = []
#             for edge, child_node in node.children:
#                 # Bellman: -Cost + Value(Next)
#                 val = -edge.cost + get_value(node=child_node)
#                 disassembly_vals.append(val)

#             best_disassembly_val = (
#                 max(disassembly_vals) if disassembly_vals else -float(x="inf")
#             )

#             # 3. Maximize
#             memo[node.id] = max(best_ce_val, best_disassembly_val)
#             return memo[node.id]

#         get_value(node=start_node)
#         return memo  # Contains the policy (values for every node)
