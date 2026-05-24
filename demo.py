"""
demo.py
=======
Demonstrates the pd_vector package.

Run from the directory that contains the pd_vector/ folder:
    python demo.py
"""

import pandas as pd

# A single import registers:
#   - the "vector_array_dtype" dtype with pandas
#   - the .vect accessor on every pd.Series
import pd_vector
from pd_vector import VectorArray  # Facultative! Only if you want to build VectorArray BEFORE converting the Dataframe

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

SEP = "-" * 52

def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ---------------------------------------------------------------------------
# 1. Build two vector_array_dtype Series
# ---------------------------------------------------------------------------

section("1. Constructing vector Series")
# Method 1 (recommended) : create a pd.Series with list of list, THEN convert to 'vector' dtype
velocities = pd.Series([
        [1.0,  0.0, 0.0],
        [0.0,  2.0, 0.0],
        [0.0,  0.0, 3.0],
        [1.0,  1.0, 1.0],
        None,              # missing value
    ], name="velocity")
velocities = velocities.astype("vector")  # 'astype' conversion using Dtypes name. No need to import VectorArray

# Method 2 (need import VectorArray) : create a VectorArray object first, then pass it to pd.Series()
forces = pd.Series(
    VectorArray._from_sequence([
        [0.0,  1.0, 0.0],
        [0.0,  0.0, 1.0],
        [1.0,  0.0, 0.0],
        [2.0, -1.0, 0.5],
        [1.0,  0.0, 0.0],
    ]), name="force")

print("velocities:\n", velocities)
print("\nforces:\n", forces)
print("\ndtype  :", velocities.dtype)
print("isna   :", velocities.isna().tolist())


# ---------------------------------------------------------------------------
# 2. Arithmetic operators
# ---------------------------------------------------------------------------

section("2. Vector addition  (velocities + forces)")
print(velocities + forces)

section("3. Scalar multiplication  (velocities * 3)")
print(velocities * 3)

section("4. Reflected scalar multiplication  (0.5 * forces)")
print(0.5 * forces)

section("5. Per-row scalar multiplication")
mass = pd.Series([1.0, 2.0, 0.5, 3.0, 1.0], name="mass")
print("mass:\n", mass)
print("\nforces * mass:\n", forces * mass)

section("6. In-place addition  (v += f)")
v_copy = velocities.copy()
v_copy += forces
print(v_copy)


# ---------------------------------------------------------------------------
# 3. .vect accessor — component properties
# ---------------------------------------------------------------------------

section("7. Component extraction via .vect")
print(".vect.x :\n", velocities.vect.x)
print("\n.vect.y :\n", velocities.vect.y)
print("\n.vect.z :\n", velocities.vect.z)

section("8. .vect.components() -> DataFrame")
print(velocities.vect.components())


# ---------------------------------------------------------------------------
# 4. .vect accessor — vector maths
# ---------------------------------------------------------------------------

section("9. .vect.norm()")
print(velocities.vect.norm())

section("10. .vect.normalized()")
print(velocities.vect.normalized())

section("11. .vect.dot(forces)")
print(velocities.vect.dot(forces))

section("12. .vect.cross(forces)")
print(velocities.vect.cross(forces))


# ---------------------------------------------------------------------------
# 5. DataFrame integration
# ---------------------------------------------------------------------------

section("13. Storing vector columns in a DataFrame")

df = pd.DataFrame({
    "id":       range(5),
    "velocity": velocities,
    "force":    forces,
})
print(df)
print("\ndtypes:\n", df.dtypes)

# Computed columns live alongside the originals
df["resultant"] = df["velocity"] + df["force"]
df["speed"]     = df["velocity"].vect.norm()
df["power"]     = df["velocity"].vect.dot(df["force"])

section("14. Enriched DataFrame (resultant, speed, power)")
print(df)
