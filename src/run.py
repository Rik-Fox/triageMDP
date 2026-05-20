import os
import pandas as pd
import json
from sim import Simulation
import matplotlib.pyplot as plt
import numpy as np
from util import Product
from model import StateAugmentedDisassemblyGraph, Solver, Node


def run_noise_sensitivity_experiment(
    sim: Simulation,
    noise_levels=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    episodes_per_level=500,
):
    """Runs a sweep over different sensor noise levels and plots the result."""
    print(
        f"Starting Noise Sensitivity Monte Carlo Experiment for {sim.base_product.name}..."
    )

    results = {"avg": [], "std": []}

    fig, ax = plt.subplots(figsize=(10, 6))

    for noise in noise_levels:
        print(f"  Running {episodes_per_level} episodes at noise std {noise:.2f}...")
        metrics = sim.run_experiment(n_episodes=episodes_per_level, noise_level=noise)
        results["avg"].append(metrics["average_profit"])
        results["std"].append(metrics["std_profit"])
        # print(f"Avg Profit for noise {noise}: £{metrics['average_profit']:.2f}")
        # print(f"Std Profit for noise {noise}: {metrics['std_profit']:.2f}")

        # Ensure your Simulation class returns the raw array of all episode profits
        raw_profits = metrics["raw_profits"]

        # Plot individual data points with high transparency to show density
        ax.scatter(
            [noise] * episodes_per_level,
            raw_profits,
            color="b",
            alpha=0.01,  # extremely low alpha due to 5000 points
            s=10,  # small point size
            edgecolors="none",
            zorder=1,
        )

    avg_profits = np.array(results["avg"])
    std_profits = np.array(results["std"])

    ax.plot(
        noise_levels,
        avg_profits,
        marker="o",
        linestyle="-",
        color="b",
        linewidth=2,
        label="Average Profit",
    )

    # Real Confidence Interval / Standard Deviation shading
    ax.fill_between(
        noise_levels,
        avg_profits - std_profits,
        avg_profits + std_profits,
        color="b",
        alpha=0.15,
        label="±1 Standard Deviation",
    )

    ax.set_title(
        f"Impact of Diagnostic Sensor Noise on Triage Profitability for {sim.base_product.name}",
        fontsize=14,
    )
    ax.set_xlabel("Sensor Noise (Standard Deviation)", fontsize=12)
    ax.set_ylabel("Average Realized Profit (£)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right")

    # Add an annotation explaining the shaded region
    # ax.annotate(
    #     "Variance (shaded region) widens\nas noise unpredictability increases.",
    #     xy=(noise_levels[-2], avg_profits[-2] + std_profits[-2]),
    #     xytext=(noise_levels[-3], avg_profits[-3] + std_profits[-3] + 40),
    #     arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
    # )
    ax.set_ylim(0, 1000)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def run_variance_comparison_experiment(
    sim: Simulation,
    episodes=500,
    noise_levels=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
):
    """Compares different sub-assembly variance profiles sharing the SAME pack health."""
    print(
        f"Starting Sub-Assembly Variance Impact Experiment for {sim.base_product.name}..."
    )

    modes = ["uniform", "moderate", "extreme"]

    results_map = {mode: {"avg": [], "std": []} for mode in modes}

    for mode in modes:
        print(f"  Evaluating Profile: {mode}")
        for noise in noise_levels:
            metrics = sim.run_experiment(
                n_episodes=episodes, noise_level=noise, state_mode=mode
            )
            results_map[mode]["avg"].append(metrics["average_profit"])
            results_map[mode]["std"].append(metrics["std_profit"])

    # Plotting the comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["b", "g", "r"]
    markers = ["o", "s", "^"]

    for (name, metrics), c, m in zip(results_map.items(), colors, markers):
        ax.plot(
            noise_levels, metrics["avg"], marker=m, color=c, linewidth=2, label=name
        )
        # ax.fill_between(
        #     noise_levels,
        #     np.array(metrics["avg"]) - np.array(metrics["std"]),
        #     np.array(metrics["avg"]) + np.array(metrics["std"]),
        #     color=c,
        #     alpha=0.15,
        #     label=f"{name} ± STD",
        # )

    ax.set_title(
        f"Impact of Sub-Assembly Variance on Realized Profit for {sim.base_product.name}\n(All profiles have identical Pack Health = 0.80)",
        fontsize=14,
    )
    ax.set_xlabel("Sensor Noise (Standard Deviation)", fontsize=12)
    ax.set_ylabel("Average Realized Profit (£)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(title="Module Health Distribution")

    # ax.annotate(
    #     "High variance unlocks highly profitable\nReuse options for specific modules.",
    #     xy=(0.0, results_map["Extreme [1.0, 1.0, 0.4]"][0]),
    #     xytext=(0.02, results_map["Extreme [1.0, 1.0, 0.4]"][0] - 30),
    #     arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
    # )
    ax.set_ylim(0, 1000)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    import math


def calculate_econ_score(
    observed_health: float,
    membership_val: float,
    phi_k: float,
    # beta: float,
    omega: float,
    alpha: float,
) -> float:
    """
    Calculates the Economic Triage Score (T_Econ,k) for a given CE pathway.
    """

    if type(omega) == list:
        omega = omega[0]
    if type(phi_k) == list:
        phi_k = phi_k[0]

    # original value of the product (alpha) is adjusted by the observed health
    virgin_value = alpha * max(observed_health, 0.01)

    beta = 0.5 * virgin_value
    # Calculate expected resale value after value addition/recovery
    profitability = phi_k * beta

    # Core equation
    t_econ = membership_val * np.sqrt((profitability - omega) / virgin_value)

    if t_econ > 1 or t_econ < 0:
        print(f"Warning: T_Econ,k value {t_econ} is out of bounds. Clipping to [0, 1].")
        t_econ = np.clip(t_econ, 0, 1)

    return t_econ


def calculate_eco_score(
    membership_val: float, Phi_k: float, E_k: float, E_v: float
) -> float:
    """
    Calculates the Environmental Triage Score (T_Eco,k) for a given CE pathway.
    """
    if type(Phi_k) == list:
        Phi_k = Phi_k[0]
    if type(E_k) == list:
        E_k = E_k[0]
    if type(E_v) == list:
        E_v = E_v[0]

    # Prevent division by zero if virgin emissions are equal to process emissions
    denominator = max(E_v - E_k, 0.01)

    penalty_ratio = E_k / denominator

    # If the process emissions are extremely high, the penalty ratio will drive
    # the term (1 - penalty) negative. Floor it at 0.
    emission_benefit = max(0.0, 1.0 - penalty_ratio)

    # Core equation
    t_eco = membership_val * Phi_k * emission_benefit

    # Ensure the score is strictly bounded between [0, 1]
    return min(1.0, float(t_eco))


def comp_value_map(product_name):

    if product_name == "Display":
        f = 0.3  # component_scaling_factor
    elif product_name == "Chassis":
        f = 0.06
    elif product_name == "Supply":
        f = 0.05
    elif product_name == "Battery":
        f = 0.06
    elif product_name == "Motherboard":
        f = 0.47
    elif product_name == "Storage":
        f = 0.24
    elif product_name == "PCB":
        f = 0.1
    elif product_name == "RAM":
        f = 0.05
    elif product_name == "GPU":
        f = 0.05
    elif product_name == "CPU":
        f = 0.03
    elif product_name == "Keyboard":
        f = 0.03
    else:  # full laptop
        f = 1.0

    return f


def get_maximal_pathway(
    policy: dict,
    sadg: StateAugmentedDisassemblyGraph,
    state: tuple,
    params=None,
) -> str | list:
    """
    Recursively traverses the policy using the graph edges, handling both
    linear actions and disassembly branching.
    """
    if params is None:
        params = {
            "optimal_policy": [],
            "cost": [0.0],  # Start with a single float in a list to allow mutation
            "obs": [],
            "E_v": [],
            "phi_k": [],
            "Phi": [],
            "E_k": [],
            "value_weight": [],
        }
    product, history = state

    # 1. Format the native tuple into the sorted policy string key
    if not history:
        policy_key = product
    else:
        # Sort the frozenset alphabetically to guarantee string consistency
        sorted_history = sorted(list(history))
        policy_key = f"{product}_[{','.join(sorted_history)}]"
    # Note: If your policy solver stringified the keys, you must either convert
    # current_key to that string format here, or update your solver to retain the raw tuples.
    if policy_key not in policy:
        return "No policy defined for this state"

    decision = policy[policy_key]

    if decision.startswith("CE:"):
        params["optimal_policy"].append(decision)
        # virgin emissions for a laptop (kg CO2e)
        E_v = 331.0 + np.random.randn() * (331.0 * 0.2)
        # Extract base product name
        product_name = product.split("_")[-1]

        f = comp_value_map(product_name)

        if "CE: Reuse" in decision:
            phi_k = 0.7 + np.random.randn() * (1.0 * 0.2)
            E_k = 0.0  #   for process emissions for reuse pathway
            Phi = 1.0
        elif "CE: Repair" in decision or "CE: Refurbish" in decision:
            phi_k = 1.0 + np.random.randn() * (1.0 * 0.2)
            E_k = (
                21.0 + np.random.randn() * (21.0 * 0.2)
            ) * f  #   for process emissions for repair pathway
            Phi = 0.8
        elif "CE: Recycle" in decision:
            phi_k = 0.5 + np.random.randn() * (1.0 * 0.2)
            E_k = 2.0 + np.random.randn() * (2.0 * 0.2)  # not by f
            #   for environmental impact factor
            Phi = 0.5

        # cost = 50.0 + np.random.randn() * (50.0 * 0.2)  # CE processing cost

        # params["cost"].append(cost)
        params["cost"][0] += 50.0 + np.random.randn() * (50.0 * 0.2)
        params["E_v"].append(E_v)
        params["phi_k"].append(phi_k)
        params["Phi"].append(Phi)
        params["E_k"].append(E_k)
        params["value_weight"].append(f)

    else:

        # 2. Parse the core action name (stripping out "Action: " if present)
        action_name = decision.split(": ")[-1] if ": " in decision else decision

        # 3. Use the graph to find the exact next state node(s)
        node = sadg.state_map.get(state)

        # Match the chosen action to the graph's generated edges to find the children
        next_nodes = [
            child
            for edge_action, child in node.children
            if edge_action.name == action_name
        ]

        # 4. Handle Linear vs Branching Paths
        if action_name != "Extract_Components":
            # Linear progression (e.g., Diagnostics, Wiping Data)

            next_state = (product, node.history.union({action_name}))
            params["cost"][0] += [
                a for a in node.product.actions if a.name == action_name
            ][0].cost
            params["optimal_policy"].append(action_name)
            get_maximal_pathway(
                policy=policy,
                sadg=sadg,
                state=next_state,
                params=params,
            )

        else:
            # Disassembly creates multiple branches. Recursively map all child components.
            # branched = True
            params["cost"][0] += [
                a for a in node.product.actions if a.name == action_name
            ][0].cost
            params["optimal_policy"].append(action_name)
            components = []
            for child in next_nodes[0]:
                child_state = (
                    child.product.name,
                    frozenset(node.history.union({action_name})),
                )
                child_params = {
                    "optimal_policy": [],
                    "cost": [0.0],
                    "obs": [],
                    "E_v": [],
                    "phi_k": [],
                    "Phi": [],
                    "E_k": [],
                    "value_weight": [],
                }

                components.append(
                    [
                        child.product.name,
                        get_maximal_pathway(
                            policy=policy,
                            sadg=sadg,
                            state=child_state,
                            params=child_params,
                        ),
                    ]
                )

            for comp in components:
                params["optimal_policy"].append(comp[1]["optimal_policy"])
                params["cost"].append(comp[1]["cost"])
                params["obs"].append(comp[1]["obs"])
                params["E_v"].append(comp[1]["E_v"])
                params["phi_k"].append(comp[1]["phi_k"])
                params["Phi"].append(comp[1]["Phi"])
                params["E_k"].append(comp[1]["E_k"])
                params["value_weight"].append(comp[1]["value_weight"])

    return params


def calculate_econ_eco_scores(policy_params, health, node, alpha):
    # observed health of the laptop OR components parts depending on policy action requirements
    ce_opt = next(
        opt
        for opt in node.ce_options
        if opt.name == policy_params["optimal_policy"][-1].split(" ")[-1]
    )

    mu = ce_opt.get_fuzzy_membership(health)

    phi_k = policy_params["phi_k"]
    E_k = policy_params["E_k"]
    E_v = policy_params["E_v"]
    Phi = policy_params["Phi"]
    cost = policy_params["cost"]

    t_econ = calculate_econ_score(
        observed_health=health,
        membership_val=mu,  #   for fuzzy membership value
        phi_k=phi_k,  #   for profitability factor
        omega=cost,  #   for processing cost
        alpha=alpha,
    )

    t_eco = calculate_eco_score(
        membership_val=mu,  #   for fuzzy membership value
        Phi_k=Phi,  #   for environmental impact factor
        E_k=E_k,  #   for process emissions
        E_v=E_v,  #   for virgin emissions
    )

    return t_econ, t_eco


def calculate_mdp_reward(
    policy,
    sadg,
    obs,
    alpha,
):

    opt_pol_params = get_maximal_pathway(
        policy=policy,
        sadg=sadg,
        state=sadg.root_key,
    )

    product = sadg.root_key[0]

    # if CE here then a laptop strat was taken
    if "CE: " in opt_pol_params["optimal_policy"][-1]:
        health = obs[product]

        state = (product, frozenset(opt_pol_params["optimal_policy"][:-1]))

        node = sadg.state_map.get(state)

        t_econ, t_eco = calculate_econ_eco_scores(
            policy_params=opt_pol_params, health=health, node=node, alpha=alpha
        )

    # if no CE then it was a component reuse strategy and we need to sum up the parameters of the components
    else:
        t_econs = []
        t_ecos = []
        values = []
        base_path = opt_pol_params["optimal_policy"][:-5]

        comp_pol = opt_pol_params["optimal_policy"][-5:]
        # first entry of comp is motherboard if this doesnt end in CE then it was further broken down
        # and we need to sum up the parameters of the motherboard components
        for comp in comp_pol:
            if "CE: " in comp[-1]:
                comp_name = comp[-1].split("_")[-1]
                comp_product = product + "_" + comp_name
                health = obs[comp_product]

                state = (
                    comp_product,
                    frozenset(base_path).union(comp[:-1]),
                )

                node = sadg.state_map.get(state)

                policy = base_path + comp
                comp_params = {
                    "optimal_policy": policy,
                    "cost": opt_pol_params["cost"].pop(0),
                    "E_v": opt_pol_params["E_v"].pop(0),
                    "obs": opt_pol_params["obs"].pop(0),
                    "phi_k": opt_pol_params["phi_k"].pop(0),
                    "Phi": opt_pol_params["Phi"].pop(0),
                    "E_k": opt_pol_params["E_k"].pop(0),
                    "value_weight": opt_pol_params["value_weight"].pop(0),
                }

                t_econ, t_eco = calculate_econ_eco_scores(
                    policy_params=comp_params, health=health, node=node, alpha=alpha
                )

                t_econs.append(t_econ)
                t_ecos.append(t_eco)
                values.append(node.ce_options[0].get_expected_reward(health))

                # phi_k = np.sum(opt_pol_params["phi_k"])
                # E_k = np.sum(opt_pol_params["E_k"])
                # E_v = np.sum(opt_pol_params["E_v"])
                # Phi = np.sum(opt_pol_params["Phi"])
                # cost = np.sum(opt_pol_params["cost"])

            else:

                # sum over motherboard elements in first entry then add to sum of other elements
                phi_k = np.sum(opt_pol_params["phi_k"][0]) + np.sum(
                    opt_pol_params["phi_k"][1:]
                )
                E_k = np.sum(opt_pol_params["E_k"][0]) + np.sum(
                    opt_pol_params["E_k"][1:]
                )
                E_v = np.sum(opt_pol_params["E_v"][0]) + np.sum(
                    opt_pol_params["E_v"][1:]
                )
                Phi = np.sum(opt_pol_params["Phi"][0]) + np.sum(
                    opt_pol_params["Phi"][1:]
                )
                cost = np.sum(opt_pol_params["cost"][0]) + np.sum(
                    opt_pol_params["cost"][1:]
                )

    mdp_reward = np.sqrt((t_econ**2) + (t_eco**2))

    return t_econ, t_eco, mdp_reward


def run_mass_laptop_triage_experiment(sim, n_episodes=10000, noise_level=0.2):
    """
    Runs a Monte Carlo simulation for laptop ITAD triage.
    Generates instances, applies observation noise, varies market value, and solves the MDP.
    """
    print(
        f"Starting Mass Triage Experiment: {n_episodes} episodes, Noise Level: {noise_level}"
    )

    # Baseline market value defined in paper (£1000)
    BASE_ALPHA = 1000.0

    results = []

    for i in range(n_episodes):
        laptop_name = f"{i}"
        laptop = Laptop(name=laptop_name)
        alpha = BASE_ALPHA * np.random.uniform(0.8, 1.2)
        true_state = laptop.generate_true_state(mode="random")

        obs = {}
        for comp_name, true_health in true_state.items():
            noisy_health = np.random.normal(loc=true_health, scale=noise_level)
            obs[comp_name] = float(np.clip(noisy_health, 0.01, 1.0))

        sadg = StateAugmentedDisassemblyGraph(laptop)

        policy = sim.solver.solve(sadg.state_map[sadg.root_key], obs)

        t_econ, t_eco, mdp_reward = calculate_mdp_reward(policy, sadg, obs, alpha)

        # 7. Record the Episode
        record = {
            "episode_id": i,
            "laptop_name": laptop_name,
            "alpha_value": round(alpha, 2),
            "true_state": json.dumps(true_state),  # Serialize dict to string for CSV
            "observation": json.dumps(obs),
            "policy": json.dumps(policy),
            "econ_reward": t_econ,
            "eco_reward": t_eco,
            "mdp_reward": mdp_reward,
        }
        results.append(record)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{n_episodes} laptops...")

    df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)
    filename = f"results/mass_triage_experiment_n{n_episodes}_noise{noise_level}.csv"

    df.to_csv(filename, index=False)

    return df


if __name__ == "__main__":

    from battery import Battery
    from laptop import Laptop

    # Product = Battery(name="Test_Battery_Pack")
    Product = Laptop(name="Test_Laptop")

    sim = Simulation(product=Product)

    # sim.inspect_dag()

    # sim.inspect_graph()

    # sim.run_single_trace(noise_level=0.1)

    sim.plot_fuzzy_reward_curves()

    # run_noise_sensitivity_experiment(sim)

    # run_variance_comparison_experiment(sim)

    run_mass_laptop_triage_experiment(sim, n_episodes=10000, noise_level=0.2)

    # 7. Final Blocking call
    print("All tests complete. Please close the plot windows to exit.")
    plt.show()
