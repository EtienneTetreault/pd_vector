# pd-vector

A pandas extension package providing a native 3-D vector dtype and Series accessor. Store, manipulate, and compute with 3-D vectors directly inside pandas DataFrames — with full support for missing values, arithmetic operators, and vectorised maths.

---

## Table of Contents

- [For Users](#for-users)
  - [What is VectorDtype?](#what-is-vectordtype)
  - [What is VectAccessor?](#what-is-vectaccessor)
  - [Quick Start](#quick-start)
  - [Arithmetic Operations](#arithmetic-operations)
  - [VectAccessor Reference](#vectaccessor-reference)
- [For Developers](#for-developers)
  - [Environment Setup](#environment-setup)
  - [Running the Demo](#running-the-demo)
  - [Building the Package](#building-the-package)

---

## For Users

### What is VectorDtype?

`VectorDtype` is a **pandas ExtensionDtype** — a custom column type that teaches pandas how to store and operate on 3-D vectors natively.

If you have used pandas before, you are already familiar with this concept: `DatetimeTZDtype` and `Categoricals` are built-in extension type from pandas. Once a column has "dedicated" dtype, pandas knows how to parse, display, and do the appropriate arithmetic on it automatically.

`VectorDtype` works the same way, but for 3-D float64 vectors. Each element of a `vector` Series is a NumPy array of shape `(3,)`. Missing vectors are represented as `[NaN, NaN, NaN]` rows, so `Series.isna()` and `pd.concat()` work exactly as you would expect.

### What is VectAccessor?

`VectAccessor` is a **pandas Series accessor** registered under the `.vect` namespace. It exposes vector-specific operations directly on any Series whose dtype is `vector`, without you having to unpack the underlying data manually.

Again, there is a familiar pandas analogue: the `.str` accessor on `object` or `StringDtype` Series gives you string methods like `.str.upper()`, `.str.split()`, and `.str.contains()`. You do not need to loop over the Series yourself — pandas dispatches the operation row-by-row for you.

`.vect` is the same idea for vectors: `.vect.norm()`, `.vect.dot()`, `.vect.cross()`, and so on — all fully vectorised under the hood using NumPy.

---

### Quick Start

```python
import pandas as pd
import pd_vector                        # registers 'vector' dtype + 'vect' accessor/methods. The variable pd_vector is NEVER USED
from pd_vector import VectorArray       # FACULTATIVE! Only if you want to build VectorArray BEFORE converting to a pd.Series


# --- Build vector Series ---
# Method 1 (recommended) : create a pd.Series with list of list, THEN convert to 'vector' dtype
velocities = pd.Series([
        [1.0,  0.0, 0.0],
        [0.0,  2.0, 0.0],
        [0.0,  0.0, 3.0],
        [1.0,  1.0, 1.0],
        None,              # missing value
    ], name="velocity")
velocities = velocities.astype("vector")  # 'astype' conversion using Dtype name. No need to import VectorArray

# Method 2 (need import VectorArray) : create a VectorArray object first, then pass it to pd.Series()
forces = pd.Series(
    VectorArray._from_sequence([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [2.0, -1.0, 0.5],
        [1.0,  0.0, 0.0],
    ]), name="force")

print(velocities.dtype)   # vector
print(velocities.isna())  # [False, False, False, False, True]

# --- Store vector columns in a DataFrame ---
df = pd.DataFrame({
    "id":       range(5),
    "velocity": velocities,
    "force":    forces,
})

df["resultant"] = df["velocity"] + df["force"]
df["speed"]     = df["velocity"].vect.norm()
df["power"]     = df["velocity"].vect.dot(df["force"])

print(df)
print(df.dtypes)
```

---

### Arithmetic Operations

All standard arithmetic is supported directly on `vector` Series, with no accessor required.

| Operation | Syntax | Description |
|---|---|---|
| Vector addition | `s1 + s2` | Element-wise addition of two vector Series |
| In-place addition | `s1 += s2` | In-place element-wise addition |
| Scalar multiply | `s * 3.0` | Multiply every vector by a scalar |
| Reflected scalar | `0.5 * s` | Scalar on the left-hand side |
| Per-row scalar | `s * pd.Series([...])` | Multiply each vector by its own scalar |

```python
# Vector addition
resultant = velocities + forces

# Scalar multiplication (uniform)
doubled = velocities * 2.0

# Reflected scalar multiplication
halved = 0.5 * forces

# Per-row scalar multiplication (e.g. F = m * a)
mass = pd.Series([1.0, 2.0, 0.5, 3.0, 1.0])
weighted_forces = forces * mass
```

> **Note:** Multiplying two `vector` Series together directly is intentionally not supported, since the result would be ambiguous (dot product vs cross product vs element-wise). Use `.vect.dot()` or `.vect.cross()` instead.

---

### VectAccessor Reference

Access all methods via `series.vect.*` on any Series with dtype `vector`.

#### Access vector's components (x,y,z)

| Property/Method | Return type | Description |
|-----------------|---|---|
| `.vect.x`       | `Series[float64]` | X component of every vector |
| `.vect.y`       | `Series[float64]` | Y component of every vector |
| `.vect.z`       | `Series[float64]` | Z component of every vector |
| `.vect.components()` | `DataFrame` | All three components as a three-column DataFrame |

```python
velocities.vect.x   # → float64 Series of x components
velocities.vect.y   # → float64 Series of y components
velocities.vect.z   # → float64 Series of z components
velocities.vect.components()   # → DataFrame with 3 columns
```
#### Vector mathematics

| Method | Return type | Description |
|---|---|---|
| `.vect.norm()` | `Series[float64]` | L2 norm (magnitude) of every vector |
| `.vect.normalized()` | `Series[vector]` | Unit vectors; zero vectors become NaN |
| `.vect.dot(other)` | `Series[float64]` | Element-wise dot product with `other` |
| `.vect.cross(other)` | `Series[vector]` | Element-wise cross product with `other` |

```python
# Magnitude of each velocity vector
speeds = velocities.vect.norm()

# Unit vectors (direction only)
directions = velocities.vect.normalized()

# Power = velocity · force  (dot product)
power = velocities.vect.dot(forces)

# Torque-like result = velocity × force  (cross product)
torque = velocities.vect.cross(forces)
```

`other` can be either a `vector` Series or a bare `VectorArray` of the same length.

---

## For Developers

### Environment Setup

[uv](https://docs.astral.sh/uv/) is the recommended tool for environment and dependency management. Install it once, then from the project root:

```bash
# Create a virtual environment and install all dependencies
# (including dev extras: pytest, pytest-cov)
uv sync
```

`uv sync` reads `pyproject.toml`, resolves all dependencies, writes a `uv.lock` lockfile for reproducibility, and creates a `.venv` in the project root. The package is installed in editable mode automatically.

To add a new runtime dependency:

```bash
uv add <package>
```

To add a new development-only dependency:

```bash
uv add --dev <package>
```

---

### Running the Demo

```bash
uv run demo.py
```

`uv run` uses the local `.venv` automatically — no need to activate it first.

---

### Building the Package

```bash
# Build both the wheel (.whl) and source distribution (.tar.gz) into dist/
uv build

# Build only the wheel (faster)
uv build --wheel

# Build only the source distribution
uv build --sdist
```

The output will appear in the `dist/` folder:

```
dist/
├── pd_vector-0.1.0-py3-none-any.whl
└── pd_vector-0.1.0.tar.gz
```

To publish to PyPI once the package is ready:

```bash
uv publish
```

### Running the Tests

```bash
# Run the full test suite
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with coverage report
uv run pytest --cov=pd_vector --cov-report=term-missing
```
