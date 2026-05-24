"""
tests/test_vector_array.py
==========================
Unit tests for VectorArray and VectorDtype.

Covers:
  - Construction: __init__, _from_sequence, _from_sequence_of_strings
  - ExtensionArray protocol: __len__, __getitem__, __setitem__, isna, copy, take, _concat_same_type
  - Arithmetic: __add__, __radd__, __iadd__, __mul__, __rmul__, __imul__
  - Arithmetic error paths: type mismatches, shape mismatches
  - Vector maths: norm, dot, cross, normalized
  - Pandas internals: _values_for_factorize, _values_for_argsort
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
def simple_array():
    """A plain 3-row VectorArray with no missing values."""
    return VectorArray._from_sequence([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])


@pytest.fixture
def array_with_nan():
    """A 3-row VectorArray whose second row is NaN (missing)."""
    return VectorArray._from_sequence([
        [1.0, 2.0, 3.0],
        None,
        [4.0, 5.0, 6.0],
    ])


@pytest.fixture
def other_array():
    """A second 3-row VectorArray for binary-op tests."""
    return VectorArray._from_sequence([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ])


# ---------------------------------------------------------------------------
# 1. dtype
# ---------------------------------------------------------------------------

class TestVectorDtype:
    def test_name(self):
        assert VectorDtype.name == "vector"

    def test_construct_from_string_valid(self):
        dtype = VectorDtype.construct_from_string("vector")
        assert isinstance(dtype, VectorDtype)

    def test_construct_from_string_invalid(self):
        with pytest.raises(TypeError):
            VectorDtype.construct_from_string("not_a_vector")

    def test_construct_array_type(self):
        assert VectorDtype.construct_array_type() is VectorArray

    def test_repr(self):
        assert repr(VectorDtype()) == "vector"

    def test_pandas_recognises_dtype_string(self):
        """pandas.array() should resolve the 'vector' string to VectorArray."""
        arr = pd.array([[1, 2, 3], [4, 5, 6]], dtype="vector")
        assert isinstance(arr, VectorArray)


# ---------------------------------------------------------------------------
# 2. Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_init_valid(self):
        data = np.ones((4, 3), dtype=np.float64)
        arr = VectorArray(data)
        assert len(arr) == 4

    def test_init_wrong_type_raises(self):
        with pytest.raises(TypeError):
            VectorArray([[1, 2, 3]])

    def test_init_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            VectorArray(np.ones((3, 4)))

    def test_from_sequence_basic(self, simple_array):
        assert len(simple_array) == 3
        assert simple_array._data.dtype == np.float64

    def test_from_sequence_none_becomes_nan(self, array_with_nan):
        assert np.all(np.isnan(array_with_nan._data[1]))

    def test_from_sequence_wrong_length_raises(self):
        with pytest.raises(ValueError):
            VectorArray._from_sequence([[1, 2]])  # length 2, not 3

    def test_from_sequence_of_strings(self):
        arr = VectorArray._from_sequence_of_strings(["[1.0, 2.0, 3.0]", "[4.0, 5.0, 6.0]"])
        assert arr._data.shape == (2, 3)
        np.testing.assert_array_equal(arr._data[0], [1.0, 2.0, 3.0])

    def test_from_sequence_of_strings_invalid_raises(self):
        with pytest.raises(ValueError):
            VectorArray._from_sequence_of_strings(["[1.0, 2.0]"])  # only 2 numbers


# ---------------------------------------------------------------------------
# 3. ExtensionArray protocol
# ---------------------------------------------------------------------------

class TestExtensionArrayProtocol:
    def test_len(self, simple_array):
        assert len(simple_array) == 3

    def test_dtype(self, simple_array):
        assert isinstance(simple_array.dtype, VectorDtype)

    def test_getitem_scalar(self, simple_array):
        row = simple_array[0]
        assert isinstance(row, np.ndarray)
        np.testing.assert_array_equal(row, [1.0, 0.0, 0.0])

    def test_getitem_slice(self, simple_array):
        sliced = simple_array[1:]
        assert isinstance(sliced, VectorArray)
        assert len(sliced) == 2

    def test_setitem_vector(self, simple_array):
        simple_array[0] = np.array([9.0, 8.0, 7.0])
        np.testing.assert_array_equal(simple_array._data[0], [9.0, 8.0, 7.0])

    def test_setitem_none_becomes_nan(self, simple_array):
        simple_array[0] = None
        assert np.all(np.isnan(simple_array._data[0]))

    def test_isna_no_missing(self, simple_array):
        assert not any(simple_array.isna())

    def test_isna_with_missing(self, array_with_nan):
        mask = array_with_nan.isna()
        assert mask.tolist() == [False, True, False]

    def test_copy_is_independent(self, simple_array):
        copy = simple_array.copy()
        copy._data[0, 0] = 999.0
        assert simple_array._data[0, 0] != 999.0

    def test_take_basic(self, simple_array):
        taken = simple_array.take([2, 0])
        np.testing.assert_array_equal(taken._data[0], simple_array._data[2])
        np.testing.assert_array_equal(taken._data[1], simple_array._data[0])

    def test_take_with_fill(self, simple_array):
        taken = simple_array.take([0, -1], allow_fill=True)
        assert np.all(np.isnan(taken._data[1]))

    def test_concat_same_type(self, simple_array, other_array):
        combined = VectorArray._concat_same_type([simple_array, other_array])
        assert len(combined) == 6

    def test_eq(self, simple_array):
        same = simple_array.copy()
        result = simple_array == same
        assert all(result)

    def test_repr(self, simple_array):
        r = repr(simple_array)
        assert "VectorArray" in r


# ---------------------------------------------------------------------------
# 4. Arithmetic operators
# ---------------------------------------------------------------------------

class TestArithmetic:
    def test_add_two_arrays(self, simple_array, other_array):
        result = simple_array + other_array
        expected = simple_array._data + other_array._data
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_radd(self, simple_array, other_array):
        result = other_array.__radd__(simple_array)
        expected = simple_array._data + other_array._data
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_iadd(self, simple_array, other_array):
        original_data = simple_array._data.copy()
        result = simple_array.__iadd__(other_array)
        np.testing.assert_array_almost_equal(result._data, original_data + other_array._data)

    def test_mul_scalar_int(self, simple_array):
        result = simple_array * 3
        np.testing.assert_array_almost_equal(result._data, simple_array._data * 3)

    def test_mul_scalar_float(self, simple_array):
        result = simple_array * 0.5
        np.testing.assert_array_almost_equal(result._data, simple_array._data * 0.5)

    def test_rmul_scalar(self, simple_array):
        result = 2 * simple_array
        np.testing.assert_array_almost_equal(result._data, simple_array._data * 2)

    def test_mul_per_row_series(self, simple_array):
        scalars = pd.Series([1.0, 2.0, 3.0])
        result = simple_array * scalars
        for i in range(3):
            np.testing.assert_array_almost_equal(result._data[i], simple_array._data[i] * scalars[i])

    def test_mul_per_row_array(self, simple_array):
        scalars = np.array([1.0, 2.0, 3.0])
        result = simple_array * scalars
        for i in range(3):
            np.testing.assert_array_almost_equal(result._data[i], simple_array._data[i] * scalars[i])

    def test_add_wrong_dtype_raises(self, simple_array):
        float_series = pd.Series([1.0, 2.0, 3.0])
        result = simple_array.__add__(float_series)
        # should return NotImplemented for unsupported type
        assert result is NotImplemented

    def test_mul_vector_by_vector_raises(self, simple_array, other_array):
        """Multiplying two VectorArrays directly should raise TypeError."""
        va_series = pd.Series(other_array)
        result = simple_array.__mul__(va_series)
        assert result is NotImplemented

    def test_mul_wrong_length_raises(self, simple_array):
        with pytest.raises(TypeError):
            simple_array * np.array([1.0, 2.0])  # length 2, array length 3


# ---------------------------------------------------------------------------
# 5. Vector maths
# ---------------------------------------------------------------------------

class TestVectorMaths:
    def test_norm_unit_vectors(self, simple_array):
        norms = simple_array.norm()
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0, 1.0])

    def test_norm_known_value(self):
        arr = VectorArray._from_sequence([[3.0, 4.0, 0.0]])
        np.testing.assert_almost_equal(arr.norm()[0], 5.0)

    def test_dot_orthogonal(self, simple_array, other_array):
        # [1,0,0]·[1,1,0] = 1,  [0,1,0]·[0,1,1] = 1,  [0,0,1]·[1,0,1] = 1
        result = simple_array.dot(other_array)
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0])

    def test_dot_self(self, simple_array):
        """Dot product of a vector with itself equals its squared norm."""
        result = simple_array.dot(simple_array)
        expected = np.sum(simple_array._data ** 2, axis=1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_dot_wrong_length_raises(self, simple_array):
        short = VectorArray._from_sequence([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            simple_array.dot(short)

    def test_cross_known_values(self):
        a = VectorArray._from_sequence([[1.0, 0.0, 0.0]])
        b = VectorArray._from_sequence([[0.0, 1.0, 0.0]])
        result = a.cross(b)
        np.testing.assert_array_almost_equal(result._data[0], [0.0, 0.0, 1.0])

    def test_cross_anti_commutative(self, simple_array, other_array):
        ab = simple_array.cross(other_array)
        ba = other_array.cross(simple_array)
        np.testing.assert_array_almost_equal(ab._data, -ba._data)

    def test_cross_wrong_length_raises(self, simple_array):
        short = VectorArray._from_sequence([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            simple_array.cross(short)

    def test_normalized_unit_vectors_unchanged(self, simple_array):
        result = simple_array.normalized()
        np.testing.assert_array_almost_equal(result._data, simple_array._data)

    def test_normalized_known_vector(self):
        arr = VectorArray._from_sequence([[3.0, 4.0, 0.0]])
        result = arr.normalized()
        np.testing.assert_array_almost_equal(result._data[0], [0.6, 0.8, 0.0])

    def test_normalized_zero_vector_is_nan(self):
        arr = VectorArray._from_sequence([[0.0, 0.0, 0.0]])
        result = arr.normalized()
        assert np.all(np.isnan(result._data[0]))


# ---------------------------------------------------------------------------
# 6. Pandas internals
# ---------------------------------------------------------------------------

class TestPandasInternals:
    def test_values_for_argsort(self, simple_array):
        """_values_for_argsort should return L2 norms (all 1.0 for unit vectors)."""
        norms = simple_array._values_for_argsort()
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0, 1.0])

    def test_values_for_factorize_no_nan(self, simple_array):
        keys, na = simple_array._values_for_factorize()
        assert all(isinstance(k, tuple) for k in keys)

    def test_values_for_factorize_with_nan(self, array_with_nan):
        keys, na = array_with_nan._values_for_factorize()
        assert keys[1] == (np.nan, np.nan, np.nan)

    def test_series_isna(self, array_with_nan):
        s = pd.Series(array_with_nan)
        assert s.isna().tolist() == [False, True, False]

    def test_series_dtype(self, simple_array):
        s = pd.Series(simple_array)
        assert isinstance(s.dtype, VectorDtype)

    def test_series_concat(self, simple_array, other_array):
        s1 = pd.Series(simple_array)
        s2 = pd.Series(other_array)
        combined = pd.concat([s1, s2], ignore_index=True)
        assert len(combined) == 6
        assert isinstance(combined.dtype, VectorDtype)