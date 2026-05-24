from __future__ import annotations

import numpy as np
import pandas as pd

from .vector_array_dtype import VectorArray, VectorDtype


# ---------------------------------------------------------------------------
# Accessor
# ---------------------------------------------------------------------------

@pd.api.extensions.register_series_accessor("vect")
class VectAccessor:
    """
    Series accessor for ``vector_array_dtype`` dtype.

    Usage
    -----
    >>> s.vect.norm()
    >>> s.vect.dot(other)
    >>> s.vect.cross(other)
    >>> s.vect.normalized()
    >>> s.vect.x  /  s.vect.y  /  s.vect.z
    >>> s.vect.components()
    """

    def __init__(self, series: pd.Series) -> None:
        if not isinstance(series.dtype, VectorDtype):
            raise AttributeError(
                "The 'vect' accessor is only valid for Series with dtype "
                f"'vector_array_dtype', but got '{series.dtype}'."
            )
        self._series = series
        self._array: VectorArray = series.array  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wrap_array(self, result: VectorArray, name: str | None = None) -> pd.Series:
        """Wrap an VectorArray back into a Series, preserving the index."""
        return pd.Series(result, index=self._series.index, name=name)

    def _coerce_other(self, other: pd.Series | VectorArray) -> VectorArray:
        """Accept either a same-dtype Series or a bare VectorArray."""
        if isinstance(other, pd.Series):
            if not isinstance(other.dtype, VectorDtype):
                raise TypeError(
                    f"'other' must be an vector_array_dtype Series, got dtype '{other.dtype}'."
                )
            return other.array  # type: ignore[return-value]
        if isinstance(other, VectorArray):
            return other
        raise TypeError(
            f"'other' must be an vector_array_dtype Series or VectorArray, got {type(other)}."
        )

    # ------------------------------------------------------------------
    # Scalar-component properties
    # ------------------------------------------------------------------

    @property
    def x(self) -> pd.Series:
        """X component of every vector as a float64 Series."""
        return pd.Series(
            self._array._data[:, 0],
            index=self._series.index,
            name=(self._series.name and f"{self._series.name}.x") or "x",
        )

    @property
    def y(self) -> pd.Series:
        """Y component of every vector as a float64 Series."""
        return pd.Series(
            self._array._data[:, 1],
            index=self._series.index,
            name=(self._series.name and f"{self._series.name}.y") or "y",
        )

    @property
    def z(self) -> pd.Series:
        """Z component of every vector as a float64 Series."""
        return pd.Series(
            self._array._data[:, 2],
            index=self._series.index,
            name=(self._series.name and f"{self._series.name}.z") or "z",
        )

    # ------------------------------------------------------------------
    # Vectorised helpers (delegate to VectorArray)
    # ------------------------------------------------------------------

    def norm(self) -> pd.Series:
        """
        L2 norm (magnitude) of every vector.

        Returns
        -------
        pd.Series[float64]
        """
        return pd.Series(
            self._array.norm(),
            index=self._series.index,
            name=(self._series.name and f"{self._series.name}.norm") or "norm",
        )

    def dot(self, other: pd.Series | VectorArray) -> pd.Series:
        """
        Element-wise dot product with *other*.

        Parameters
        ----------
        other : vector_array_dtype Series or VectorArray of the same length.

        Returns
        -------
        pd.Series[float64]
        """
        other_arr = self._coerce_other(other)
        return pd.Series(
            self._array.dot(other_arr),
            index=self._series.index,
            name="dot",
        )

    def cross(self, other: pd.Series | VectorArray) -> pd.Series:
        """
        Element-wise cross product with *other*.

        Parameters
        ----------
        other : vector_array_dtype Series or VectorArray of the same length.

        Returns
        -------
        pd.Series[vector_array_dtype]
        """
        other_arr = self._coerce_other(other)
        return self._wrap_array(self._array.cross(other_arr), name="cross")

    def normalized(self) -> pd.Series:
        """
        Unit vectors (rows with zero magnitude become NaN vectors).

        Returns
        -------
        pd.Series[vector_array_dtype]
        """
        return self._wrap_array(
            self._array.normalized(),
            name=(self._series.name and f"{self._series.name}.normalized") or "normalized",
        )

    def components(self) -> pd.DataFrame:
        """
        Decompose every vector into a three-column DataFrame (x, y, z).

        Returns
        -------
        pd.DataFrame with columns ['x', 'y', 'z'] and the same index as the Series.
        """
        prefix = f"{self._series.name}." if self._series.name else ""
        return pd.DataFrame(
            self._array._data,
            index=self._series.index,
            columns=[f"{prefix}x", f"{prefix}y", f"{prefix}z"],
        )
