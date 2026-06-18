# triageMDP
using  dynamic programming to solve contructed MDPs that have fuzzy rewards due to a triage process that is inherently fuzzy.

## Project Structure

## install and run


### Source Files

-   **`run.py`**: The main entry point for running simulations and experiments. It contains functions to run different scenarios, such as noise sensitivity analysis and variance comparison experiments for different products.

-   **`sim.py`**: Defines the `Simulation` class, which is the core of the simulation framework. It is responsible for creating the State-Augmented Graph (SAG), running experiments, and plotting the results. It is designed to be product-agnostic, meaning it can simulate any product that inherits from the `Product` class.

-   **`model.py`**: This file contains the core logic for the Markov Decision Process (MDP) model. It defines the `StateAugmentedGraph` class, which builds the decision graph, and the `Solver` class, which implements the dynamic programming algorithm to find the optimal policy. It also defines the `CEOption` class, which represents a Circular Economy option with fuzzy logic.

-   **`util.py`**: A collection of utility classes and functions used throughout the project. It defines the base `Product` class, which is an abstract class for all products, as well as the `HealthState`, `Sensor`, and `Action` classes. It also contains helper functions for plotting graphs.

-   **`battery.py`**: Defines the `Battery` product and its components.

    -   **Physical Structure**: A `Battery` is composed of `Module`s, and each `Module` is composed of `Cell`s.
        -   `Battery` -> `Module` -> `Cell`

    -   **CE Options**:
        -   **Battery**: `Reuse_EV_Pack`, `Repurpose_Storage`, `Recycle_Pack_Direct`
        -   **Module**: `Reuse_Module`, `Recycle_Module_Direct`
        -   **Cell**: `Reuse_Cell`, `Recycle_Materials`

    -   **Processing Edges**:
        -   **Battery**: `High_Voltage_Isolation`, `Remove_Top_Cover`, `Remove_Thermal_Shield`, `Extract_Modules` (disassembly)
        -   **Module**: `Bench_Diagnostic`, `Extract_Cells` (disassembly)

-   **`laptop.py`**: Defines the `Laptop` product and its components.

    -   **Physical Structure**:
        -   `Laptop` -> `Motherboard`, `Screen`, `Keyboard`, `Touchpad`, `Casing`
        -   `Motherboard` -> `CPU`, `GPU`, `Memory`, `DataStorage`

    -   **CE Options**:
        -   **Laptop**: `Reuse`, `Refurbish`, `Repair`, `Recycle`
        -   **Motherboard**: `Reuse`, `Refurbish`, `Repair`, `Recycle`
        -   **CPU**: `Reuse`, `Repair`, `Recycle`
        -   **GPU**: `Reuse`, `Repair`, `Recycle`
        -   **Memory**: `Reuse`,`Repair`, `Recycle`
        -   **DataStorage**: `Reuse`, `Repair`, `Recycle`
        -   **Screen**: `Reuse`, `Repair`, `Recycle`
        -   **Keyboard**: `Reuse`, `Repair`, `Recycle`
        -   **Touchpad**: `Reuse`, `Repair`, `Recycle`
        -   **Casing**: `Reuse`, `Repair`, `Recycle`

        here refurbish represents a full repair and/or replacement of constituant components

    -   **Processing Edges (Actions & Disassembly)**:
        -   **Laptop**:
            -   `Visual_Inspection` -> `Power_On_Test`
            -   In-Situ Diagnostics (run on device): `CPU`, `GPU`, `Memory`, `DataStorage`, `Screen`, `Keyboard`, `Touchpad`
            -   `Open_Casing` (Disassembly)
        -   **Motherboard**:
            -   `Full_System_Diagnostic`
            -   `Bench_Diagnostic_Motherboard`
            -   `Extract_Components` (Disassembly)
        -   **CPU**: `Bench_Diagnostic_CPU`
        -   **GPU**: `Bench_Diagnostic_GPU`
        -   **DataStorage**: `Wipe_DataStorage`

    
