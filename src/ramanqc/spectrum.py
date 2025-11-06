"""
Author: Yoshiki Cook
Date: 2025-10-20

Updated: 2025-11-03
"""
from __future__ import annotations
from typing import Optional, Union, Self, Any, Dict, Tuple, List
from typing import TYPE_CHECKING 
if TYPE_CHECKING:
    from ramanqc.measurement import Measurement  # type hints only, avoid circular referencing

import numpy as np
import weakref
from ramanqc.metadata import Metadata

class Spectrum:
    """
    Represents a single Raman spectrum and its associated metadata.

    _parent : weakref.ReferenceType["Measurement"]
        Weak reference to the parent Measurement. 
        Assigned internally by Measurement during registration.

    Parameters
    ----------
        
    Attributes
    ----------

    Methods
    ----------

    """

    def __init__(
            self,
            wavenumbers: np.ndarray,
            intensities: np.ndarray,
            metadata: Optional[Union[Dict[str, Any], Metadata]] = None,
            parent: Optional['Measurement'] = None
        ) -> None:
        self._wavenumbers: np.ndarray = wavenumbers
        self._intensities: np.ndarray = intensities
        self._metadata: Metadata = Metadata.as_metadata(metadata)
        self._parent = weakref.ref(parent) if parent else None

        # ensure wavenumbers and intensities are of the correct dtype and length
        self._validate_wavenumbers_and_intensities()

        # sort wavenumbers and intensities in order of ascending wavenumber
        self._sort()
        
    def __repr__(self) -> str:
        """Return an unambiguous string representation of the spectrum."""
        return f"Spectrum(wavenumbers=array(shape={self._wavenumbers.shape}), intensities=array(shape={self._intensities.shape}), metadata={self._metadata}, parent={self._parent})"

    def __str__(self) -> str:
        """Return a human-readable summary of the spectrum."""
        description: List[str] = []
        description.append(f"{'Number of points':>24s}:\t{self._wavenumbers.shape[0]}")
        description.append(f"{'Wavenumber range':>24s}:\t{self.wavenumber_range[0]} - {self.wavenumber_range[1]} cm⁻¹")
        description.append(f"{'Resolution':>24s}:\t{self.resolution:.2f} cm⁻¹")
        description.append(f"{'Metadata entries':>24s}:\t{len(self._metadata) if self._metadata else 0}")
        return "\n".join(description)

    def __len__(self) -> int:
        """Return the number of (wavenumber, intensity) points in the spectrum."""
        return len(self._wavenumbers)
    
    def __eq__(self, other: Any) -> bool:
        """Check wavenumber and intensity equality between two Spectrum instances."""
        if not isinstance(other, Spectrum):
            return False
        w_equal: bool = np.array_equal(self._wavenumbers, other.wavenumbers)
        i_equal: bool = np.array_equal(self._intensities, other.intensities)
        return w_equal and i_equal
    
    def __add__(self, other: Self) -> Self:
        """Pointwise addition of two Spectrum instances."""
        if not isinstance(other, Spectrum):
            raise TypeError(
                f"Invalid type: can only add Spectrum instances. "
                f"Got type='Spectrum' and type='{type(other)}'."
            )
        
        if not np.array_equal(self._wavenumbers, other.wavenumbers):
            raise ValueError(
                f"Invalid wavenumbers: cannot add spectra with different wavenumbers. "
                f"Got wavenumbers={self._wavenumbers} and wavenumbers={other.wavenumbers}."
            )
        
        new_intensities: np.ndarray = self._intensities + other.intensities
        new_metadata: Metadata = Metadata.merge(self._metadata.copy(), other.metadata.copy())
        return Spectrum(self._wavenumbers.copy(), new_intensities, metadata=new_metadata, parent=None)

    def __sub__(self, other: Self) -> Self:
        """Pointwise subtraction of two Spectrum instances."""
        if not isinstance(other, Spectrum):
            raise TypeError(
                f"Invalid type: can only subtract Spectrum instances. "
                f"Got type='{type(other)}'."
            )
        
        if not np.array_equal(self._wavenumbers, other.wavenumbers):
            raise ValueError(
                f"Invalid wavenumbers: cannot subtract spectra with different wavenumbers. "
                f"Got self.wavenumbers={self._wavenumbers} and other.wavenumbers={other.wavenumbers}."
            )
        
        new_intensities: np.ndarray = self._intensities - other.intensities
        new_metadata: Metadata = Metadata.merge(self._metadata.copy(), other.metadata.copy())
        return Spectrum(self._wavenumbers.copy(), new_intensities, metadata=new_metadata, parent=None)

    def _validate_wavenumbers_and_intensities(self) -> None:
        """Ensure wavenumbers and intensities are of the correct dtype and length."""
        try:
            self._wavenumbers = np.asarray(self._wavenumbers, dtype=float)
        except (ValueError, TypeError):
            raise TypeError(
                f"Invalid wavenumbers: wavenumbers must be convertible to a numpy array of floats. "
                f"Got type='{type(self._wavenumbers)}'."
            )
        try:
            self._intensities = np.asarray(self._intensities, dtype=float)
        except (ValueError, TypeError):
            raise TypeError(
                f"Invalid intensities: intensities must be convertible to a numpy array of floats. "
                f"Got type='{type(self._intensities)}'."
            )

        if self._wavenumbers.ndim > 1:
            raise ValueError(
                f"Invalid wavenumbers: wavenumbers must be a 1D array of floats. "
                f"Got ndim={self._wavenumbers.ndim}"
            )

        if self._intensities.ndim > 1:
            raise ValueError(
                f"Invalid intensities: intensities must be a 1D array of floats. "
                f"Got ndim={self._intensities.ndim}"
            )

        if len(self._wavenumbers) != len(self._intensities):
            raise ValueError(
                f"Invalid shape: wavenumbers and intensities must have the same shape. "
                f"Got len(wavenumbers)={len(self._wavenumbers)} and len(intensities)={len(self._intensities)}."
            )

    def _sort(self, reverse=False) -> None:
        """Sort the wavenumbers and intesities by wavenumber in ascending or descending order."""
        sorted_idx: np.ndarray = self._wavenumbers.argsort()

        if reverse:
            sorted_idx = sorted_idx[::-1]

        self._wavenumbers = self._wavenumbers[sorted_idx]
        self._intensities = self._intensities[sorted_idx]

    def _set_parent(self, parent: 'Measurement') -> None:
        """Set weak reference to parent Measurment (for internal use only)."""
        self._parent = weakref.ref(parent)

    @property
    def wavenumbers(self) -> np.ndarray:
        """Return the wavenumber axis of the spectrum."""
        return self._wavenumbers.copy()
    
    @property
    def intensities(self) -> np.ndarray:
        """Return the intensity axis of the spectrum."""
        return self._intensities.copy()
    
    @property
    def metadata(self) -> Metadata:
        """Return the metadata of the spectrum."""
        return self._metadata.copy() if self._metadata else Metadata()
    
    @metadata.setter
    def metadata(self, metadata: Union[Dict[str, Any], Metadata]) -> None:
        """Set the metadata of the spectrum."""
        self._metadata = Metadata.as_metadata(metadata)

    @property
    def parent(self) -> Optional['Measurement']:
        """Return the parent Measurement."""
        return self._parent() if self._parent else None
    
    @property
    def spectral_index(self) -> Optional[int]:   
        """Return the spectral index of the spectrum within its parent Measurement."""
        parent: Measurement = self.parent
        if not parent:
            return None
        return parent._spectral_index_of(self)

    @property
    def grid_index(self) -> Optional[np.ndarray]:
        """Return the spatial grid index (i, j, k, ...) of the spectrum within its parent Measurement."""
        parent: Measurement = self.parent
        if not parent or parent.grid_indices is None:
            return None
        return parent.grid_indices[self.spectral_index]
    
    @property
    def position(self) -> Optional[np.ndarray]:
        """Return the spatial position (x, y, z, ...) of the spectrum within its parent Measurement."""
        parent: Measurement = self.parent
        if not parent or parent.positions is None:
            return None
        return parent.positions[self.spectral_index]
        
    @property
    def n_points(self) -> int:
        """Return the number of (wavenumber, intensity) points in the spectrum."""
        return len(self._wavenumbers)

    @property
    def resolution(self) -> int:
        """Return the average spectral resolution of the spectrum."""
        return abs(np.diff(self._wavenumbers)).mean()
    
    @property
    def wavenumber_range(self) -> Tuple[int, int]:
        """Return the wavenumber range of the spectrum."""
        return self._wavenumbers.min(), self._wavenumbers.max()
    
    def copy(self):
        """Return a deep copy of the spectrum."""
        return Spectrum(self.wavenumbers.copy(), self.intensities.copy(), metadata=self.metadata.copy(), parent=None)