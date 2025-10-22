"""
Author: Yoshiki Cook
Date: 2025-10-20
"""

from typing import Optional, Union, Self, Any, Dict, Tuple
import numpy as np
from metadata import Metadata
from measurement import Measurement
import weakref

class Spectrum:
    """
    Represents a single Raman spectrum and its associated metadata.

    Parameters
    ----------
        
    Attributes
    ----------

    Methods
    ----------

    """

    def __init__(self, wavenumbers: np.ndarray, intensities: np.ndarray, metadata: Optional[Union[Dict[str, Any], Metadata]] = None, parent: Optional[Measurement] = None) -> None:
        w: np.ndarray = np.asarray(wavenumbers, dtype=float)
        i: np.ndarray = np.asarray(intensities, dtype=float)

        if w.ndim > 1:
            raise ValueError(
                f"Invalid shape: wavenumbers must be a 1D array-like. "
                f"Got ndim={w.ndim}"
            )

        if i.ndim > 1:
            raise ValueError(
                f"Invalid shape: intensities must be a 1D array-like. "
                f"Got ndim={i.ndim}"
            )

        if len(w) != len(i):
            raise ValueError(
                f"Invalid shape: wavenumbers and intensities must have the same shape. "
                f"Got len(wavenumbers)={len(w)} and len(intensities)={len(i)}."
            )
        
        self._wavenumbers: np.ndarray = w
        self._intensities: np.ndarray = i
        self._sort()
        self.metadata: Metadata = Metadata.as_metadata(metadata) or None
        self._parent = weakref.ref(parent) if parent is not None else None


    def __repr__(self) -> str:
        return f"Spectrum(wavenumbers=array(shape={self._wavenumbers.shape}), intensities=array(shape={self._intensities.shape}), metadata={self.metadata})"
    
    def __len__(self) -> int:
        """Return the number of spectral points in the spectrum"""
        return len(self.wavenumbers)
    
    def __eq__(self, other: Any) -> bool:
        """Check equality between two Spectrum instances."""
        if not isinstance(other, Spectrum):
            return False
        
        w_equal: bool = np.array_equal(self.wavenumbers, other.wavenumbers)
        i_equal: bool = np.array_equal(self.intensities, other.intensities)
        m_equal: bool = self.metadata == other.metadata
        return w_equal and i_equal and m_equal
    
    def __add__(self, other: Self) -> Self:
        """Add two Spectrum instances point-wise."""
        if not isinstance(other, Spectrum):
            raise TypeError(
                f"Invalid type: can only add Spectrum instances. "
                f"Got type='{type(other)}'."
            )
        
        if not np.array_equal(self.wavenumbers, other.wavenumbers):
            raise ValueError(
                f"Invalid wavenumbers: cannot add spectra with different wavenumbers. "
                f"Got self.wavenumbers={self.wavenumbers} and other.wavenumbers={other.wavenumbers}."
            )
        
        new_intensities: np.ndarray = self.intensities + other.intensities
        new_metadata: Metadata = Metadata.merge(self.metadata.copy(), other.metadata.copy())
        return Spectrum(self.wavenumbers.copy(), new_intensities, metadata=new_metadata, parent=None)

    def __sub__(self, other: Self) -> Self:
        """Subtract two Spectrum instances point-wise."""
        if not isinstance(other, Spectrum):
            raise TypeError(
                f"Invalid type: can only subtract Spectrum instances. "
                f"Got type='{type(other)}'."
            )
        
        if not np.array_equal(self.wavenumbers, other.wavenumbers):
            raise ValueError(
                f"Invalid wavenumbers: cannot subtract spectra with different wavenumbers. "
                f"Got self.wavenumbers={self.wavenumbers} and other.wavenumbers={other.wavenumbers}."
            )
        
        new_intensities: np.ndarray = self.intensities - other.intensities
        new_metadata: Metadata = Metadata.merge(self.metadata.copy(), other.metadata.copy())
        return Spectrum(self.wavenumbers.copy(), new_intensities, metadata=new_metadata, parent=None)

    @property
    def wavenumbers(self) -> np.ndarray:
        return self._wavenumbers
    
    @property
    def intensities(self) -> np.ndarray:
        return self._intensities
    
    @property
    def metadata(self) -> Metadata:
        return self._metadata
    
    @metadata.setter
    def metadata(self, metadata: Union[Dict[str, Any], Metadata]) -> None:
        self._metadata = Metadata.as_metadata(metadata)

    @property
    def index(self) -> Optional[int]:
        """Return the index of the spectrum within its parent Measurement."""
        if self._parent is None:
            return None
        
        for i, s in enumerate(self._parent.spectra):
            if s is self:
                return i
        return None
    
    @property
    def position(self) -> Optional[Tuple[float, ...]]:
        """Return the positions of the spectrum within its parent Measurement."""
        if self._parent is None or self._parent.positions is None:
            return None
        
        if self.index is None:
            return None
        return self._parent.positions[self.index]
        
    @property
    def n_points(self) -> int:
        """Return the number of spectral points in the spectrum."""
        return len(self._wavenumbers)

    @property
    def resolution(self) -> int:
        """Return the spectral resolution of the spectrum."""
        return abs(np.diff(self._wavenumbers)).max()
    
    @property
    def wavenumber_range(self) -> Tuple[int, int]:
        """Return the wavenumber range of the spectrum as (min, max)."""
        return self._wavenumbers.min(), self._wavenumbers.max()

    def _sort(self, reverse=False) -> None:
        """Sort the spectrum by wavenumber in ascending or descending order."""
        sorted_idx: np.ndarray = self._wavenumbers.argsort()

        if reverse:
            sorted_idx = sorted_idx[::-1]

        self._wavenumbers = self._wavenumbers[sorted_idx]
        self._intensities = self._intensities[sorted_idx]

    def copy(self):
        """Return a deep copy of the spectrum."""
        return Spectrum(self.wavenumbers.copy(), self.intensities.copy(), metadata=self.metadata.copy(), parent=None)