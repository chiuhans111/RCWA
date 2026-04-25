# RCWA Test Suite

This directory contains a comprehensive, tiered testing framework for the RCWA project. The suite is designed to ensure mathematical correctness, API stability, and physical accuracy.

## Architecture

The tests are organized into three logical tiers:

1. **Tier 1: Unit Tests (`tests/test_unit/`)**
    * Verifies mathematical primitives and scattering matrix algebra.
2. **Tier 2: Integration Tests (`tests/test_integration/`)**
    * Verifies the internal data flow and field reconstruction continuity.
3. **Tier 3: System Tests (`tests/test_system/`)**
    * Verifies physical ground truths (ARC, DBR, Brewster) and legacy application benchmarks (Angular/Thickness Sweeps and convergence studies).

## Running the Tests

### Full Suite

To run all 20+ tests, execute the following from the project root:

```powershell
python -m unittest discover tests
```

### Specific Tiers

You can run a specific tier by pointing to its directory:

```powershell
python -m unittest discover tests/test_unit
python -m unittest discover tests/test_integration
python -m unittest discover tests/test_system
```

## Automatic API Coverage Audit

A custom, dynamic audit script is provided to verify that all critical mathematical and physical paths in the `rcwa` package are explicitly exercised by the suite.

Unlike standard "line coverage," which just checks if a line was executed, this script performs **Logical API Discovery**:

1. **Dynamic Discovery**: It uses Python reflection to find every public function and class in the `rcwa` package.
2. **Explicit Verification**: It checks if each public member is actually referenced by the test suite by name.
3. **Gap Analysis**: It identifies modules with low coverage and lists exactly which functions are "Missing" from the suite.

```powershell
python tests/audit_coverage.py
```

This is your **Safety Dashboard** to ensure the library's public API remains robust as features are added.

## Robust Golden Tests

For complex numerical simulations, we use file-based "Golden Baselines" stored in `tests/goldens/*.json`. This allows for explicit tracking of numerical changes through Git.

### Updating Goldens

If you intentionally change the simulation math and need to update the baselines:

```powershell
# Run the benchmark runner with the --update flag
python tests/run_benchmarks.py --update
```

Once updated, review the numerical changes using `git diff tests/goldens/` before committing.

## Debugging and Output

### Standard Coverage Tools

1. **Install**: `pip install coverage`
2. **Run**: `coverage run -m unittest discover tests`
3. **Report**: `coverage report` or `coverage html`

### Verbose Output

To see exactly which tests are running and their individual status:

```powershell
python -m unittest discover -v tests
```

### Running a Single Test

To isolate a specific test file or even a specific method:

```powershell
# Run one file
python tests/unit/test_fourier_logic.py

# Run one specific test method
python -m unittest tests.unit.test_fourier_logic.TestFourierLogic.test_convolution_identity
```

### Interactive Debugging

You can insert `breakpoint()` anywhere in the code or tests to drop into an interactive debugger (PDB) during the test run:

```python
def test_something(self):
    result = complex_math()
    breakpoint()  # Execution will pause here
    self.assertEqual(result, expected)
```

## Dynamic Mutation Testing (`tests/dynamic_mutation_engine.py`)

To ensure the test suite is truly sensitive to "bugs," we use an automated **Mutation Engine**. This tool purposefully corrupts the mathematical logic of the `rcwa/` library and verifies that at least one test in the suite detects the regression.

### Safety & Bulletproofing

The engine is designed with "Safe-Construction" principles to ensure your source code is never left in a corrupted state:

* **Self-Healing**: On startup, it automatically detects and restores any `.original` files left by a previous crash.
* **Signal Handling**: Catches `Ctrl+C` and automatically restores source code before exiting.
* **Concurrency Lock**: Uses a `mutation.lock` file to prevent multiple instances from running at once.

### Running a Stress Test

To run a randomized sensitivity sweep (default 10 random mutations):

```powershell
python tests/dynamic_mutation_engine.py
```

### Detection Performance

The current suite achieves a **90% - 100% Sensitivity Score** across the core math kernels:

* **S-Matrix Algebra**: 100% detection.
* **Wavevector Logic**: 100% detection.
* **Intensity Conversion**: 100% detection.
* **FFT Phase Mapping**: 100% detection.

## Physical Scenarios (Golden Tests)

The system tests act as a "Physical Gallery" for the library:

* **Brewster Window**: Confirms that p-polarized reflection vanishes at the Brewster angle.
* **ARC**: Confirms that a $\lambda/4$ coating cancels reflection.
* **DBR**: Confirms the formation of high-reflectivity bands in periodic stacks.
* **Benchmarks**: Pins the current diffraction efficiencies for research-grade gratings.
* **Angular/Thickness/Harmonic Sweeps**: Reproduces all legacy application behaviors.
