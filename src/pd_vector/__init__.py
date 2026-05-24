"""
pd_vector
=========
A pandas extension package providing a native 3-D vector dtype.

Public API
----------
Dtype / Array (core building blocks)
    VectorDtype   – the pandas ExtensionDtype ("vector_array_dtype")
    VectorArray   – the pandas ExtensionArray backed by an (N, 3) float64 ndarray

Series accessor
    VectAccessor     – registered as ``Series.vect``; exposes vector maths directly
                       on any Series whose dtype is "vector_array_dtype"

Typical usage
-------------
    import pandas as pd
    import pd_vector                          # registration happens on import
    from pd_vector import VectorArray

    s = pd.Series(VectorArray._from_sequence([[1, 0, 0], [0, 1, 0]]))
    s.vect.norm()                             # -> float64 Series of magnitudes
    s + s                                     # -> vector_array_dtype Series (vector addition)
    s * 2                                     # -> vector_array_dtype Series (scalar multiply)
"""

# -- Core dtype & array -------------------------------------------------------
# Importing this module triggers @register_extension_dtype, which makes pandas
# recognise the "vector_array_dtype" dtype string everywhere (pd.array(), astype(), …).
from .vector_array_dtype import VectorDtype, VectorArray

# -- Accessor -----------------------------------------------------------------
# Importing this module triggers @register_series_accessor("vect"), which
# attaches the .vect namespace to every Series at runtime.
from .vector_accessor import VectAccessor

# -- Public re-exports --------------------------------------------------------
__all__ = [
    "VectorDtype",
    "VectorArray",
    "VectAccessor",
]

__version__ = "0.1.0"
