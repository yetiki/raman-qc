"""
Author: Yoshiki Cook
Date: 2025-10-20

Updated: 2025-10-22
"""

from typing import Optional, Union, Self, Any,List, Dict, Tuple, Sequence
import numpy as np
from ramanqc.metadata import Metadata
from ramanqc.spectrum import Spectrum
from ramanqc.spatial_profile import SpatialProfile

class Measurement():
    """
    Represents a single Raman spectral measurement.

    Parameters
    ----------
        
    Attributes
    ----------

    Methods
    ---------
    """
    def __init__(
            self,
            spectra: List[Spectrum],
            grid_indices: Optional[Union[np.ndarray, Sequence[Sequence[int]]]] = None,
            positions: Optional[Union[np.ndarray, Sequence[Sequence[int]]]] = None,
            metadata: Optional[Union[Metadata, Dict[str, Any]]] = None,
            percolate_metadata: bool = True) -> None:
        for i, item in enumerate(spectra):
            if not isinstance(item, Spectrum):
                raise TypeError(
                    f"Invalid item type: Each item in spectra must be a Spectrum. "
                    f"Got type='{type(item).__name__}' with value={item} at index={i}."
                )

        self._spectra: List[Spectrum] = spectra
        self._profile: SpatialProfile = SpatialProfile(grid_indices=grid_indices, positions=positions)

        if self._profile.is_structured() and len(self._spectra) != self._profile.n_points:
            raise ValueError(
                f"Invalid lengths: Number of spectra must match the number of spatial points. "
                f"Got n_spectra={len(spectra)} and n_spatial_points={self._profile.n_points}."
            )
        
        # sort spectra according to spatial profile sorting order
        if self._profile.sort_order is not None:
            self._spectra = [self._spectra[i] for i in self._profile.sort_order]

        self._metadata: Metadata = Metadata.as_metadata(metadata)
    
    def __repr__(self) -> str:
        pass

    def __len__(self) -> int:
        """Return the number of spectra in the measurement."""
        pass

    @property
    def wavenumbers(self) -> np.ndarray:
        pass

    @property
    def intensities(self) -> np.ndarray:
        pass

    @property
    def metadata(self) -> Metadata:
        pass

    @property
    def resolution(self) -> float:
        pass

    @property
    def wavenumber_range(self) -> Tuple[float, float]:
        pass

    @property
    def n_spectra(self) -> int:
        pass
  
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the spatial grid if structured."""
        pass

    @property
    def positions(self) -> List[Tuple[float, ...]]:
        pass

    @property
    def indexes(self) -> List[Tuple[int, ...]]:
        pass

    @property
    def profile_type(self) -> str:
        pass

    @property
    def ndim(self):
        """Return the number of spatial dimensions."""
        pass

    @property
    def is_structured(self):
        """Return True if the measurement data form a regular grid."""
        pass
    
    @classmethod
    def from_array(self, wavenumbers: np.ndarray, intensities: np.ndarray, positions: np.ndarray = None, metadata: Union[Metadata, Dict[str, Any]] = None, percolate_metadata: bool = True) -> None:
        pass

    def _percolate_spectrum_metadata(self) -> None:
        pass

    def is_structured(self) -> bool:
        pass

    def get_spectrum(self, index):
        """Return the spectrum at the specified index."""
        pass

    def get_position(self, index):
        """Return the spatial position corresponding to a given spectrum index."""
        pass