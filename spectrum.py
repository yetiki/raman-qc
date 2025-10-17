from typing import Optional, Union, Self, Any, Dict, Tuple
import numpy as np
from metadata import Metadata

class Spectrum:
    """
    Represents a single Raman spectrum and its associated metadata.

    - Validates shapes on construction and when attributes are updated.
    - Uses private attributes to avoid recursive property access.
    - Converts inputs to numpy arrays.

    Attributes
    ----------
    wavenumbers : np.ndarray
        1D Raman shift axis array.
    intensities : np.ndarray
        1D measured intensity array corresponding to each wavenumber. Shape must be the same as wavenumbers.
    metadata : dict, or Metadata, optional
        Optional metadata such as sample ID, instrument info, or acquisition parameters.
    """

    def __init__(self, wavenumbers: np.ndarray, intensities: np.ndarray, metadata: Optional[Union[Dict[str, Any], Metadata]] = None) -> None:
        self.update(wavenumbers, intensities)
        self.metadata: Metadata = Metadata.as_metadata(metadata) or None

    def __repr__(self) -> str:
        return f"Spectrum(wavenumbers=array(shape={self.wavenumbers.shape}), intensities=array(shape={self.intensities.shape}), metadata={self.metadata})"
    
    def __len__(self) -> int:
        return len(self.wavenumbers)
    
    def __add__(self, other: Self) -> Self:
        if not isinstance(other, Spectrum):
            raise TypeError(
                f"Invalid type: objects must both be of type Spectrum. "
                f"Got type={type(other).__name__}"
            )
        if not (self.wavenumbers == other.wavenumbers).all():
            raise ValueError(
                f"Invalid wavenumbers: Spectrum objects must have identical wavenumbers. "
                f"Got wavenumbers={self.wavenumbers} and wavenumbers={other.wavenumbers}"
            )
        
        i: np.ndarray = self._intensities + other.intensities
        m: Metadata = self._metadata.merge(other.metadata)

        return Spectrum(self.wavenumbers, i, metadata=m)
    
    @property
    def metadata(self) -> Metadata:
        return self._metadata
    
    @metadata.setter
    def metadata(self, metadata: Union[Dict[str, Any], Metadata]) -> None:
        self._metadata = Metadata.as_metadata(metadata)

    @property
    def wavenumbers(self) -> np.ndarray:
        return self._wavenumbers
    
    @wavenumbers.setter
    def wavenumbers(self, wavenumbers: np.ndarray) -> None:
        w: np.ndarray = np.asarray(wavenumbers)

        if w.ndim != 1:
            raise ValueError(
                f"Invalid shape: wavenumbers must be 1-dimensional. "
                f"Got wavenumbers.ndim={w.ndim}."
            )

        if hasattr(self, "intensities") and self.intensities is not None and w.shape != self.intensities.shape:
            raise ValueError(
                f"Invalid shape: wavenumbers and intensities must have the same shape. "
                f"Got wavenumbers.shape={w.shape} and self.intensities.shape={self.intensities.shape}."
            )
        self._wavenumbers = w

    @property
    def intensities(self) -> np.ndarray:
        return self._intensities
    
    @intensities.setter
    def intensities(self, intensities: np.ndarray) -> None:
        i: np.ndarray = np.asarray(intensities)

        if i.ndim != 1:
            raise ValueError(
                f"Invalid shape: intensities must be 1-dimensional. "
                f"Got intensities.ndim={i.ndim}. "
                f"Use Measurement() for multiple spectra."
            )

        if hasattr(self, "wavenumbers") and self.wavenumbers is not None and i.shape != self.wavenumbers.shape:
            raise ValueError(
                f"Invalid shape: wavenumbers and intensities must have the same shape. "
                f"Got self.wavenumbers.shape={self.wavenumbers.shape} and intensities.shape={i.shape}."
            )
        self._intensities = i
    
    @property
    def n_points(self) -> int:
        return len(self.wavenumbers)

    @property
    def resolution(self) -> int:
        return abs(np.diff(self.wavenumbers)).max()
    
    @property
    def wavenumber_range(self) -> Tuple[int, int]:
        return self.wavenumbers.min(), self.wavenumbers.max()

    def update(self, wavenumbers: np.ndarray, intensities: np.ndarray) -> None:
        w: np.ndarray = np.asarray(wavenumbers)
        i: np.ndarray = np.asarray(intensities)

        if w.shape != i.shape:
            raise ValueError(
                f"Invalid shape: wavenumbers and intensities must have the same shape. "
                f"Got wavenumbers.shape={w.shape} and intensities.shape={i.shape}."
            )
        
        self.wavenumbers, self.intensities = w, i
        self.sort()

    def sort(self, reverse=False) -> None:
        sorted_idx: np.ndarray = self.wavenumbers.argsort()

        if reverse:
            sorted_idx = sorted_idx[::-1]

        self.wavenumbers = self.wavenumbers[sorted_idx]
        self.intensities = self.intensities[sorted_idx]