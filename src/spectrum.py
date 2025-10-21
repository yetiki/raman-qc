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
        """Initialize a single spectral measurement."""
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
        """Return the index of this Spectrum within its parent Measurement."""
        if self._parent is None:
            return None
        
        for i, s in enumerate(self._parent.spectra):
            if s is self:
                return i
        return None
    
    @property
    def position(self) -> Optional[Tuple[float, ...]]:
        if self._parent is None or self._parent.positions is None:
            return None
        
        if self.index is None:
            return None
        return self._parent.positions[self.index]
        
    @property
    def n_points(self) -> int:
        return len(self._wavenumbers)

    @property
    def resolution(self) -> int:
        return abs(np.diff(self._wavenumbers)).max()
    
    @property
    def wavenumber_range(self) -> Tuple[int, int]:
        return self._wavenumbers.min(), self._wavenumbers.max()

    def _sort(self, reverse=False) -> None:
        sorted_idx: np.ndarray = self._wavenumbers.argsort()

        if reverse:
            sorted_idx = sorted_idx[::-1]

        self._wavenumbers = self._wavenumbers[sorted_idx]
        self._intensities = self._intensities[sorted_idx]

    def copy(self):
        """Return a deep copy of the spectrum."""
        return Spectrum(self.wavenumbers.copy(), self.intensities.copy(), metadata=self.metadata.copy(), parent=None)