# triageMDP

Using dynamic programming to solve constructed MDPs with fuzzy rewards, where the
fuzziness reflects a triage process that is inherently uncertain.

## Install and run

```
python -m venv .venv

powershell -> .\.venv\Scripts\Activate.ps1
bash -> source .venv/bin/activate

pip install -r requirements.txt
python ./src/run.py
```

The analysis notebook additionally requires Jupyter (`pip install jupyter`).

## Reproducibility

The entry point seeds NumPy (`SEED = 42`). Noise levels and the number of episodes
are set in the `if __name__ == "__main__"` block of `src/run.py`; changing the seed
generates a different reproducible sequence. Existing matching results CSVs in
`results/` are overwritten.

### Source Files

- **`run.py`**: The main entry point. Runs the mass laptop triage Monte Carlo
  experiment and computes the per-component economic and ecological rewards for
  each optimal policy.
- **`sim.py`**: Defines the `Simulation` class, which builds the State-Augmented
  Disassembly Graph (SADG) and its solver, and provides graph and fuzzy-reward
  figure helpers. It is product-agnostic: any product inheriting from `Product`
  can be simulated.
- **`model.py`**: Core MDP logic. Defines `StateAugmentedDisassemblyGraph`, which
  builds the decision graph; `Solver`, which implements the dynamic programming
  algorithm; and `CEOption`, which represents a Circular Economy option with
  fuzzy logic.
- **`util.py`**: Utility classes used throughout: the abstract base `Product`
  class, `HealthState`, and `Action`, plus helper functions for drawing graphs.
- **`laptop.py`**: Defines the `Laptop` product and its components.

### Laptop model

- **Physical Structure**:
    - `Laptop` -> `Motherboard`, `Display`, `Keyboard`, `Battery`, `Chassis`
    - `Motherboard` -> `CPU`, `GPU`, `Memory`, `DataStorage`

    `CPU`, `GPU`, `Memory`, `DataStorage` are not reachable with the current structure and parameters but are included for completeness

- **CE Options**:
    - **Laptop**: `Reuse`, `Refurbish`, `Repair`, `Recycle`
    - **Motherboard**: `Reuse`, `Refurbish`, `Repair`, `Recycle`
    - **CPU**, **GPU**, **Memory**, **DataStorage**, **Display**, **Keyboard**,
      **Battery**, **Chassis**: `Reuse`, `Repair`, `Recycle`

    Here refurbish represents a full repair and/or replacement of constituent
    components.

- **Processing Edges (Actions & Disassembly)**:
    - **Laptop**:
        - `Visual_Inspection` -> `Power_On_Test`
        - In-Situ Diagnostics (run on device): `Display`, `Keyboard`, `Battery`,
          `Motherboard`
        - `Open_Chassis`, then `Wipe_DataStorage` (required before reuse,
          refurbish or repair of the storage)
        - `Extract_Components` (Disassembly)
    - **Motherboard**:
        - `Full_System_Diagnostic`
        - `Bench_Diagnostic_Motherboard`
        - In-Situ Diagnostics: `CPU`, `GPU`, `Memory`, `DataStorage`
        - `Extract_Components` (Disassembly)
    - **Bench Diagnostics**: `CPU`, `GPU`, `Memory`, `DataStorage`, `Display`,
      `Keyboard`, `Battery`, `Chassis`