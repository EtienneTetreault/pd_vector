"""
tests/test_vector_accessor.py
==============================
Unit tests for the VectAccessor (.vect) Series accessor.

Covers:
  - Accessor registration and dtype guard
  - Component properties: .x, .y, .z
  - .components() -> DataFrame
  - .norm()
  - .normalized()
  - .dot()
  - .cross()
  - Index and name preservation throughout
  - Error paths: wrong dtype input, wrong-type 'other'
"""

import numpy as np
import pandas as pd
import pytest

import pd_vector
from pd_vector import VectorArray, VectorDtype


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def velocities():
    """Named Series of 4 velocity vectors (no missing values)."""
    return pd.Series(
        VectorArray._from_sequence([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ]),
        name="velocity",
    )


@pytest.fixture
def forces():
    """Named Series of 4 force vectors (no missing values)."""
    return pd.Series(
        VectorArray._from_sequence([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [2.0, -1.0, 0.5],
        ]),
        name="force",
    )


@pytest.fixture
def velocities_with_nan():
    """Series with one NaN row to test NA propagation."""
    return pd.Series(
        VectorArray._from_sequence([
            [1.0, 0.0, 0.0],
            None,
            [0.0, 0.0, 1.0],
        ]),
        name="velocity",
    )


@pytest.fixture
def custom_index_series():
    """Series with a non-default index to verify index preservation."""
    return pd.Series(
        VectorArray._from_sequence([
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 5.0],
        ]),
        index=["a", "b"],
        name="vec",
    )


# ---------------------------------------------------------------------------
# 1. Accessor registration & dtype guard
# ---------------------------------------------------------------------------

class TestAccessorGuard:
    def test_accessor_available_on_vector_series(self, velocities):
        """Accessing .vect on a vector Series should not raise."""
        _ = velocities.vect

    def test_accessor_raises_on_float_series(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(AttributeError, match="vect"):
            _ = s.vect

    def test_accessor_raises_on_int_series(self):
        s = pd.Series([1, 2, 3])
        with pytest.raises(AttributeError):
            _ = s.vect

    def test_accessor_raises_on_string_series(self):
        s = pd.Series(["a", "b", "c"])
        with pytest.raises(AttributeError):
            _ = s.vect


# ---------------------------------------------------------------------------
# 2. Component properties
# ---------------------------------------------------------------------------

class TestComponentProperties:
    def test_x_values(self, velocities):
        result = velocities.vect.x
        np.testing.assert_array_equal(result.values, [1.0, 0.0, 0.0, 1.0])

    def test_y_values(self, velocities):
        result = velocities.vect.y
        np.testing.assert_array_equal(result.values, [0.0, 2.0, 0.0, 1.0])

    def test_z_values(self, velocities):
        result = velocities.vect.z
        np.testing.assert_array_equal(result.values, [0.0, 0.0, 3.0, 1.0])

    def test_x_name(self, velocities):
        assert velocities.vect.x.name == "velocity.x"

    def test_y_name(self, velocities):
        assert velocities.vect.y.name == "velocity.y"

    def test_z_name(self, velocities):
        assert velocities.vect.z.name == "velocity.z"

    def test_x_name_unnamed_series(self):
        s = pd.Series(VectorArray._from_sequence([[1.0, 2.0, 3.0]]))
        assert s.vect.x.name == "x"

    def test_component_index_preserved(self, custom_index_series):
        assert list(custom_index_series.vect.x.index) == ["a", "b"]

    def test_x_is_float64(self, velocities):
        assert velocities.vect.x.dtype == np.float64

    def test_x_nan_propagation(self, velocities_with_nan):
        x = velocities_with_nan.vect.x
        assert np.isnan(x.iloc[1])


# ---------------------------------------------------------------------------
# 3. .components()
# ---------------------------------------------------------------------------

class TestComponents:
    def test_returns_dataframe(self, velocities):
        df = velocities.vect.components()
        assert isinstance(df, pd.DataFrame)

    def test_column_names_with_prefix(self, velocities):
        df = velocities.vect.components()
        assert list(df.columns) == ["velocity.x", "velocity.y", "velocity.z"]

    def test_column_names_no_prefix(self):
        s = pd.Series(VectorArray._from_sequence([[1.0, 2.0, 3.0]]))
        df = s.vect.components()
        assert list(df.columns) == ["x", "y", "z"]

    def test_shape(self, velocities):
        df = velocities.vect.components()
        assert df.shape == (4, 3)

    def test_index_preserved(self, custom_index_series):
        df = custom_index_series.vect.components()
        assert list(df.index) == ["a", "b"]

    def test_values_match_components(self, velocities):
        df = velocities.vect.components()
        np.testing.assert_array_equal(df["velocity.x"].values, velocities.vect.x.values)
        np.testing.assert_array_equal(df["velocity.y"].values, velocities.vect.y.values)
        np.testing.assert_array_equal(df["velocity.z"].values, velocities.vect.z.values)


# ---------------------------------------------------------------------------
# 4. .norm()
# ---------------------------------------------------------------------------

class TestNorm:
    def test_norm_unit_x_vector(self, velocities):
        norms = velocities.vect.norm()
        np.testing.assert_almost_equal(norms.iloc[0], 1.0)

    def test_norm_known_value(self):
        s = pd.Series(VectorArray._from_sequence([[3.0, 4.0, 0.0]]), name="v")
        np.testing.assert_almost_equal(s.vect.norm().iloc[0], 5.0)

    def test_norm_returns_float64_series(self, velocities):
        result = velocities.vect.norm()
        assert isinstance(result, pd.Series)
        assert result.dtype == np.float64

    def test_norm_name(self, velocities):
        assert velocities.vect.norm().name == "velocity.norm"

    def test_norm_index_preserved(self, custom_index_series):
        assert list(custom_index_series.vect.norm().index) == ["a", "b"]

    def test_norm_nan_propagation(self, velocities_with_nan):
        norms = velocities_with_nan.vect.norm()
        assert np.isnan(norms.iloc[1])


# ---------------------------------------------------------------------------
# 5. .normalized()
# ---------------------------------------------------------------------------

class TestNormalized:
    def test_normalized_is_vector_series(self, velocities):
        result = velocities.vect.normalized()
        assert isinstance(result.dtype, VectorDtype)

    def test_normalized_unit_length(self, velocities):
        result = velocities.vect.normalized()
        norms = result.vect.norm()
        np.testing.assert_array_almost_equal(norms.values, [1.0, 1.0, 1.0, 1.0])

    def test_normalized_known_vector(self):
        s = pd.Series(VectorArray._from_sequence([[3.0, 4.0, 0.0]]))
        result = s.vect.normalized()
        np.testing.assert_array_almost_equal(result.array._data[0], [0.6, 0.8, 0.0])

    def test_normalized_zero_vector_is_nan(self):
        s = pd.Series(VectorArray._from_sequence([[0.0, 0.0, 0.0]]))
        result = s.vect.normalized()
        assert np.all(np.isnan(result.array._data[0]))

    def test_normalized_name(self, velocities):
        assert velocities.vect.normalized().name == "velocity.normalized"

    def test_normalized_index_preserved(self, custom_index_series):
        assert list(custom_index_series.vect.normalized().index) == ["a", "b"]


# ---------------------------------------------------------------------------
# 6. .dot()
# ---------------------------------------------------------------------------

class TestDot:
    def test_dot_returns_float64_series(self, velocities, forces):
        result = velocities.vect.dot(forces)
        assert isinstance(result, pd.Series)
        assert result.dtype == np.float64

    def test_dot_known_values(self, velocities, forces):
        # [1,0,0]·[0,1,0]=0, [0,2,0]·[0,0,1]=0, [0,0,3]·[1,0,0]=0
        result = velocities.vect.dot(forces)
        np.testing.assert_array_almost_equal(result.values[:3], [0.0, 0.0, 0.0])

    def test_dot_self_equals_squared_norm(self, velocities):
        result = velocities.vect.dot(velocities)
        expected = velocities.vect.norm() ** 2
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_dot_name(self, velocities, forces):
        assert velocities.vect.dot(forces).name == "dot"

    def test_dot_index_preserved(self, custom_index_series):
        result = custom_index_series.vect.dot(custom_index_series)
        assert list(result.index) == ["a", "b"]

    def test_dot_accepts_vector_array(self, velocities, forces):
        result = velocities.vect.dot(forces.array)
        assert isinstance(result, pd.Series)

    def test_dot_wrong_dtype_raises(self, velocities):
        float_series = pd.Series([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(TypeError):
            velocities.vect.dot(float_series)

    def test_dot_wrong_type_raises(self, velocities):
        with pytest.raises(TypeError):
            velocities.vect.dot([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# 7. .cross()
# ---------------------------------------------------------------------------

class TestCross:
    def test_cross_returns_vector_series(self, velocities, forces):
        result = velocities.vect.cross(forces)
        assert isinstance(result.dtype, VectorDtype)

    def test_cross_known_values(self):
        # [1,0,0] x [0,1,0] = [0,0,1]
        a = pd.Series(VectorArray._from_sequence([[1.0, 0.0, 0.0]]))
        b = pd.Series(VectorArray._from_sequence([[0.0, 1.0, 0.0]]))
        result = a.vect.cross(b)
        np.testing.assert_array_almost_equal(result.array._data[0], [0.0, 0.0, 1.0])

    def test_cross_anti_commutative(self, velocities, forces):
        ab = velocities.vect.cross(forces)
        ba = forces.vect.cross(velocities)
        np.testing.assert_array_almost_equal(ab.array._data, -ba.array._data)

    def test_cross_name(self, velocities, forces):
        assert velocities.vect.cross(forces).name == "cross"

    def test_cross_index_preserved(self, custom_index_series):
        result = custom_index_series.vect.cross(custom_index_series)
        assert list(result.index) == ["a", "b"]

    def test_cross_accepts_vector_array(self, velocities, forces):
        result = velocities.vect.cross(forces.array)
        assert isinstance(result.dtype, VectorDtype)

    def test_cross_wrong_dtype_raises(self, velocities):
        float_series = pd.Series([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(TypeError):
            velocities.vect.cross(float_series)

    def test_cross_wrong_type_raises(self, velocities):
        with pytest.raises(TypeError):
            velocities.vect.cross([1.0, 2.0, 3.0])