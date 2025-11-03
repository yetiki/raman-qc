"""
Author: Yoshiki Cook
Date: 2025-10-20

Updated: 2025-11-03
"""

from typing import Optional, Union, Any, List, Dict, Tuple, Sequence
import numpy as np
import weakref
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
            percolate_metadata: bool = True
        ) -> None:
        self._spectra: List[Spectrum] = spectra
        self._profile: SpatialProfile = SpatialProfile(grid_indices=grid_indices, positions=positions)
        self._metadata: Metadata = Metadata.as_metadata(metadata)
        
        # ensure each spectrum in spectra have the same dtype, length, and wavenumbers
        self._validate_spectra()

        # ensure the number of spectra match the number of spatial points
        self._validate_spectra_and_profile()
        
        # sort spectra according to the sorting order specified by the spatial profile
        self._sort_spectra()

        # register the parent reference in each spectrum
        self._register_spectra()

        # optionally propagate the metadata to each spectrum
        if percolate_metadata:
            self._percolate_metadata_to_spectra()

        self._wavenumbers: np.ndarray = self._spectra[0].wavenumbers
        self._intensities: np.ndarray = np.array([s.intensities for s in self._spectra])
        self._resolution: float = self._spectra[0].resolution
        self._wavenumber_range: Tuple[float, float] = self._spectra[0].wavenumber_range
        self._n_spectra: int = len(self._spectra)
        self._n_points: int = self._spectra[0].n_points
        self._profile_type: str = self._profile.profile_type
        self._shape: Optional[Tuple[int, ...]] = self._profile.shape

    def _validate_spectra(self, tolerance: float = 1e-6) -> None:
        """Ensure each spectrum in spectra have the same dtype, length, and wavenumbers."""
        # ensure all items are of type 'Spectrum'
        self._validate_spectrum_dtypes()

        # ensure all spectra have the same length
        self._validate_spectrum_lengths()

        # ensure all spectra have an identical wavenumber axis within a tolerance
        self._validate_wavenumbers(tolerance=tolerance)         
        
    def _validate_spectrum_dtypes(self) -> None:
        """Ensure all items are of type 'Spectrum'."""
        for i, item in enumerate(self._spectra):
            if not isinstance(item, Spectrum):
                raise TypeError(
                    f"Invalid type: Each item in spectra must be of type 'Spectrum'. "
                    f"Got type='{type(item).__name__}' at index={i} with value={item}."
                )

    def _validate_spectrum_lengths(self) -> None:
        """Ensure all spectra have the same length."""
        ref_spectrum_length: Spectrum = len(self._spectra[0])
        for i, spectrum in enumerate(self._spectra):
            if ref_spectrum_length != len(spectrum):
                raise ValueError(
                    f"Invalid spectra: each spectrum in spectra must be of the same length. "
                    f"Got len(spectrum)={ref_spectrum_length} at index=0, and len(spectrum)={len(spectrum)} at index={i}."
                )

    def _validate_wavenumbers(self, tolerance: float = 1e-6) -> None:
        """Ensure all spectra have identical wavenumber axis within a tolerance."""
        ref_wavenumbers: np.ndarray = self._spectra[0].wavenumbers
        for i, spectrum in enumerate(self._spectra):
            if not np.allclose(ref_wavenumbers, spectrum.wavenumbers, atol=tolerance):
                raise ValueError(
                    f"Invalid wavenumbers: wavenumbers must be identical across all spectra. "
                    f"Got exceeded tolerance={tolerance} for spectrum at index={i} with respect to spectrum at index=0."
                )
            
    def _validate_spectra_and_profile(self) -> None:
        """Ensure the number of spectra match the number of spatial points."""
        if self._profile.is_structured() and len(self._spectra) != self._profile.n_points:
            raise ValueError(
                f"Invalid lengths: Number of spectra must match the number of spatial points. "
                f"Got n_spectra={len(self._spectra)} and n_spatial_points={self._profile.n_points}."
            )
        
    def _sort_spectra(self) -> None:
        """Sort spectra according to sorting order specified by the spatial profile."""
        if self._profile.sort_order is not None:
            self._spectra = [self._spectra[i] for i in self._profile.sort_order]

    def _register_spectra(self) -> None:
        """Register the current Measurement as the parent reference in each spectrum."""
        for s in self._spectra:
            s._parent = weakref.ref(self)

    def _percolate_metadata_to_spectra(self) -> None:
        """Propagate the measurement metadata to each spectrum."""
        for s in self._spectra:
            s._metadata.set_default('measurement_metadata', self._metadata)
            
    def _index_of(self, spectrum: Spectrum) -> int:
        """Return the index of the specified spectrum in the measurement."""
        for i, s in enumerate(self._spectra):
            if s is spectrum:
                return i
        raise ValueError(
            f"Spectrum not found: the specified spectrum is not part of this measurement. "
            f"Got spectrum={spectrum}."
        )
        
    def __repr__(self) -> str:
        """Return an unambiguous string representation of the Measurement."""
        return f"Measurement(n_spectra={self.n_spectra}, profile_type={self.profile_type})"
    
    def __str__(self) -> str:
        """Return a human-readable summary of the Measurement."""
        description: List[str] = []
        description.append(f"{'Number of points':>24s}:\t{self._n_points}")
        description.append(f"{'Profile type':>24s}:\t{self._profile_type}")
        description.append(f"{'Shape':>24s}:\t{self._shape}")
        description.append(f"{'Metadata entries':>24s}:\t{len(self._metadata) if self._metadata else 0}")
        return "\n".join(description)

    def __len__(self) -> int:
        """Return the number of spectra in the measurement."""
        return self.n_spectra

    @property
    def wavenumbers(self) -> np.ndarray:
        """Return the wavenumber axis of the measurement."""
        return self._wavenumbers

    @property
    def intensities(self) -> np.ndarray:
        """Return the intensities array of the measurement."""
        return self._intensities

    @property
    def metadata(self) -> Metadata:
        """Return the metadata of the measurement."""
        return self._metadata
    
    @property
    def n_spectra(self) -> int:
        """Return the number of spectra in the measurement."""
        return self._n_spectra
    
    @property
    def n_points(self) -> int:
        """Return the number of (wavenumber, intensity) points in each spectrum."""
        return self._n_points

    @property
    def resolution(self) -> float:
        """Return the spectral resolution of the measurement."""
        return self._resolution

    @property
    def wavenumber_range(self) -> Tuple[float, float]:
        """Return the wavenumber range of the measurement."""
        return self._wavenumber_range

    @property
    def grid_indices(self) -> Optional[np.ndarray]:
        """Return the spatial grid indices (i, j, k, ...) of the measurement."""
        return self._profile.grid_indices

    @property
    def positions(self) -> Optional[np.ndarray]:
        """Return the spatial positions (x, y, z, ...) of the measurement."""
        return self._profile.positions

    @property
    def profile_type(self) -> str:
        """Return the type of spatial profile."""
        return self._profile_type
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the spatial grid if structured."""
        return self._shape

    @property
    def ndim(self):
        """Return the number of spatial dimensions."""
        return self._profile.ndim

    @property
    def is_structured(self):
        """Return True if the measurement data form a regular grid."""
        return self._profile.is_structured
    
    @classmethod
    def from_array(
        self,
        wavenumbers: np.ndarray,
        intensities: np.ndarray,
        positions: np.ndarray = None,
        metadata: Union[Metadata, Dict[str, Any]] = None,
        percolate_metadata: bool = True
    ) -> None:
        """Create a Measurement instance from wavenumber and intensity arrays."""
        pass

    def get_spectrum(self, index: int, default: Spectrum = None) -> Spectrum:
        """Return the spectrum at the specified index."""
        return self._spectra[index] if 0 <= index < len(self._spectra) else default
    
    def get_grid_index(self, index: int) -> Optional[np.ndarray]:
        """Return the spatial grid index corresponding to a given spectrum index."""
        return self._profile.grid_indices[index] if 0 <= index < len(self._profile.grid_indices) else None

    def get_position(self, index: int) -> Optional[np.ndarray]:
        """Return the spatial position corresponding to a given spectrum index."""
        return self._profile.positions[index] if 0 <= index < len(self._profile.positions) else None