from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.extensions import (
    ExtensionDtype,
    ExtensionArray,
    register_extension_dtype,
)
from pandas.core.arrays.base import ExtensionArray
from typing import Any, Sequence
import re


# ---------------------------------------------------------------------------
# Dtype
# ---------------------------------------------------------------------------

@register_extension_dtype
class VectorDtype(ExtensionDtype):
    """Pandas ExtensionDtype for a 3D vector (numpy array of shape (3,))."""

    name = "vector"
    type = np.ndarray          # the scalar "type" exposed to pandas
    kind = "O"                 # treated as object-like by pandas internals
    na_value = np.array([np.nan, np.nan, np.nan])

    @classmethod
    def construct_array_type(cls) -> type[VectorArray]:
        return VectorArray

    @classmethod
    def construct_from_string(cls, string: str) -> "VectorDtype":
        if string == cls.name:
            return cls()
        raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    def __repr__(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Array
# ---------------------------------------------------------------------------

class VectorArray(ExtensionArray):
    """
    Pandas ExtensionArray whose elements are 3-element numpy float64 vectors.

    Internal storage: a 2-D numpy array of shape (N, 3).
    Missing values are represented as rows of NaN.
    """

    dtype = VectorDtype()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __init__(self, data: np.ndarray, copy: bool = False) -> None:
        """
        Parameters
        ----------
        data : np.ndarray, shape (N, 3), dtype float64
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be a numpy ndarray, got {type(data)}")
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(
                f"data must have shape (N, 3), got {data.shape}"
            )
        self._data = data.astype(np.float64, copy=copy)

    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[Any],
        *,
        dtype=None,
        copy: bool = False,
    ) -> "VectorArray":
        """Build from an iterable of (array-like of length 3 | None)."""
        rows = []
        for s in scalars:
            if s is None or (isinstance(s, float) and np.isnan(s)):
                rows.append([np.nan, np.nan, np.nan])
            else:
                arr = np.asarray(s, dtype=np.float64)
                if arr.shape != (3,):
                    raise ValueError(
                        f"Each scalar must be array-like of length 3, got shape {arr.shape}"
                    )
                rows.append(arr)
        return cls(np.array(rows, dtype=np.float64), copy=False)

    @classmethod
    def _from_sequence_of_strings(
        cls,
        strings: Sequence[str],
        *,
        dtype=None,
        copy: bool = False,
    ) -> "VectorArray":
        """Parse strings like '[1.0, 2.0, 3.0]'."""
        scalars = []
        for s in strings:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            if len(nums) != 3:
                raise ValueError(f"Cannot parse '{s}' as a 3D vector")
            scalars.append([float(n) for n in nums])
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values, original) -> "VectorArray":
        return cls._from_sequence(values)

    # ------------------------------------------------------------------
    # Required ExtensionArray interface
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        result = self._data[key]
        # Scalar access -> return a plain 1-D numpy array
        if result.ndim == 1:
            return result
        return type(self)(result)

    def __setitem__(self, key, value) -> None:
        if isinstance(value, VectorArray):
            self._data[key] = value._data
        elif value is None or (isinstance(value, float) and np.isnan(value)):
            self._data[key] = np.nan
        else:
            value = np.asarray(value, dtype=np.float64)
            if value.shape == (3,):
                self._data[key] = value
            else:
                self._data[key] = value  # let numpy raise if incompatible

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> np.ndarray:
        if isinstance(other, VectorArray):
            return np.all(self._data == other._data, axis=1)
        return NotImplemented

    def isna(self) -> np.ndarray:
        """Return boolean mask; a row is NA when any component is NaN."""
        return np.any(np.isnan(self._data), axis=1)

    def take(
        self,
        indices: Sequence[int],
        *,
        allow_fill: bool = False,
        fill_value=None,
    ) -> "VectorArray":
        from pandas.api.extensions import take

        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = take(
            self._data,
            indices,
            allow_fill=allow_fill,
            fill_value=fill_value if not isinstance(fill_value, np.ndarray)
                        else fill_value,
        )
        return type(self)(np.atleast_2d(result))

    def copy(self) -> "VectorArray":
        return type(self)(self._data.copy())

    @classmethod
    def _concat_same_type(
        cls, to_concat: Sequence["VectorArray"]
    ) -> "VectorArray":
        return cls(np.concatenate([arr._data for arr in to_concat], axis=0))

    # ------------------------------------------------------------------
    # Pandas internals / serialisation
    # ------------------------------------------------------------------

    def _values_for_factorize(self):
        """Represent each vector as a Python tuple for hashing."""
        na = (np.nan, np.nan, np.nan)
        keys = np.empty(len(self), dtype=object)
        for i, row in enumerate(self._data):
            keys[i] = na if np.any(np.isnan(row)) else tuple(row)
        return keys, na

    def _values_for_argsort(self) -> np.ndarray:
        """Sort by vector magnitude (L2 norm)."""
        return np.linalg.norm(self._data, axis=1)

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def _coerce_operand(self, other) -> np.ndarray:
        """
        Return the (N, 3) data array from *other*, accepting:
        - VectorArray
        - pandas Series[vector_array_dtype]
        - array-like of shape (N, 3) or (3,)  <- broadcast a single vector
        """
        if isinstance(other, VectorArray):
            return other._data
        # duck-type a pandas Series
        if hasattr(other, "dtype") and hasattr(other, "array"):
            if not isinstance(other.dtype, VectorDtype):
                raise TypeError(
                    f"Unsupported operand dtype for +: '{other.dtype}'. "
                    "Expected an vector_array_dtype Series."
                )
            return other.array._data
        # fall back: try to coerce a plain array-like
        arr = np.asarray(other, dtype=np.float64)
        if arr.shape == (3,):
            return arr          # numpy will broadcast (N, 3) + (3,)
        if arr.shape == (len(self), 3):
            return arr
        raise TypeError(
            f"Cannot add operand of shape {arr.shape} to an VectorArray "
            f"of length {len(self)}."
        )

    def __add__(self, other) -> "VectorArray":
        """Element-wise vector addition:  a + b"""
        try:
            return type(self)(self._data + self._coerce_operand(other))
        except TypeError:
            return NotImplemented

    def __radd__(self, other) -> "VectorArray":
        """Support  other + vector_array_dtype_array  (commutative for vector addition)."""
        return self.__add__(other)

    def __iadd__(self, other) -> "VectorArray":
        """In-place addition:  a += b  (returns a new array; pandas EAs are value-typed)."""
        return self.__add__(other)

    def _coerce_scalar(self, other) -> np.ndarray:
        """
        Validate and return a scalar multiplier as a broadcastable float64 value.

        Accepted forms
        --------------
        - A Python int or float             -> broadcast the same scalar to every row
        - A pandas Series[numeric]          -> one scalar per row (length must match)
        - A 1-D array-like of length N      -> one scalar per row
        - A 1-D array-like of length 1      -> broadcast to every row
        """
        # Plain Python / numpy scalar
        if np.ndim(other) == 0:
            return np.float64(other)

        # pandas Series with a numeric dtype
        if hasattr(other, "dtype") and hasattr(other, "to_numpy"):
            if isinstance(other.dtype, VectorDtype):
                raise TypeError(
                    "Cannot multiply two vector_array_dtype Series together. "
                    "Use .vect.dot() for element-wise dot product."
                )
            arr = other.to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(other, dtype=np.float64)

        if arr.ndim != 1:
            raise TypeError(
                f"Scalar multiplier must be 0-D or 1-D, got shape {arr.shape}."
            )
        if arr.shape[0] == 1:
            return arr          # numpy will broadcast (1,) -> (N,)
        if arr.shape[0] == len(self):
            return arr[:, np.newaxis]   # reshape to (N, 1) for (N, 3) broadcast
        raise TypeError(
            f"Length of scalar array ({arr.shape[0]}) does not match "
            f"VectorArray length ({len(self)})."
        )

    def __mul__(self, other) -> "VectorArray":
        """Scalar multiplication:  vec_array * scalar  (or  * per-row scalars)."""
        try:
            return type(self)(self._data * self._coerce_scalar(other))
        except TypeError:
            return NotImplemented

    def __rmul__(self, other) -> "VectorArray":
        """Support  scalar * vec_array."""
        return self.__mul__(other)

    def __imul__(self, other) -> "VectorArray":
        """In-place scalar multiplication:  a *= scalar."""
        return self.__mul__(other)

    # ------------------------------------------------------------------
    # Nice-to-haves
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"VectorArray(\n{self._data!r}\n, dtype={self.dtype!r})"
        )

    # Vectorised helpers ----------------------------------------------------

    def norm(self) -> np.ndarray:
        """Return the L2 norm of every vector as a float64 ndarray."""
        return np.linalg.norm(self._data, axis=1)

    def dot(self, other: "VectorArray") -> np.ndarray:
        """Element-wise dot product with another VectorArray."""
        if not isinstance(other, VectorArray) or len(other) != len(self):
            raise ValueError("other must be an VectorArray of the same length")
        return np.einsum("ij,ij->i", self._data, other._data)

    def cross(self, other: "VectorArray") -> "VectorArray":
        """Element-wise cross product with another VectorArray."""
        if not isinstance(other, VectorArray) or len(other) != len(self):
            raise ValueError("other must be an VectorArray of the same length")
        return type(self)(np.cross(self._data, other._data))

    def normalized(self) -> "VectorArray":
        """Return unit vectors (rows with zero norm become NaN)."""
        norms = self.norm()[:, np.newaxis]
        with np.errstate(invalid="ignore"):
            result = self._data / norms
        return type(self)(result)
