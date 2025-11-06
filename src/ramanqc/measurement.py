"""
Author: Yoshiki Cook
Date: 2025-10-20

Updated: 2025-11-03
"""
from __future__ import annotations
from typing import Optional, Union, Any, List, Dict, Tuple, Sequence

import numpy as np
from ramanqc.metadata import Metadata
from ramanqc.spectrum import Spectrum
from ramanqc.spatial_profile import SpatialProfile

class Measurement():
    """
    Represents a single Raman spectral measurement.



    During initialization, this class assigns itself as the parent 
    of each Spectrum via its internal _parent attribute.

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
        self._metadata: Metadata = Metadata.as_metadata(metadata)
        
        # ensure each spectrum in spectra have the same dtype, length, and wavenumbers
        self._validate_spectra()

        # infer measurement profile type to be a 'point' spectrum
        if len(self._spectra) == 1 and grid_indices is None and positions is None:
            grid_indices: np.ndarray = np.zeros(shape=(1, 1))

        self._profile: SpatialProfile = SpatialProfile(grid_indices=grid_indices, positions=positions)

        # ensure the number of spectra match the number of spatial points
        self._validate_spectra_and_profile()
        
        # sort spectra according to the sorting order specified by the spatial profile
        self._sort_spectra()

        # register the current measurement as the parent reference in each spectrum
        self._register_spectra()

        # optionally propagate the metadata to each spectrum
        if percolate_metadata:
            self._percolate_metadata_to_spectra()

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
        if self._profile.is_structured and len(self._spectra) != self._profile.n_points:
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
            s._set_parent(self)

    def _percolate_metadata_to_spectra(self) -> None:
        """Propagate the measurement metadata to each spectrum."""
        for s in self._spectra:
            s._metadata.update(self._metadata)
            
    def _spectral_index_of(self, spectrum: Spectrum) -> int:
        """Return the spectral index of the specified spectrum in the measurement."""
        for i, s in enumerate(self._spectra):
            if s is spectrum:
                return i
        raise ValueError(
            f"Spectrum not found: the specified spectrum is not part of this measurement. "
            f"Got spectrum={spectrum}."
        )
        
    def __repr__(self) -> str:
        """Return an unambiguous string representation of the measurement."""
        return f"Measurement(profile_type='{self.profile_type}', n_spectra={self.n_spectra}, metadata={self._metadata})"
    
    def __str__(self) -> str:
        """Return a human-readable summary of the measurement."""
        description: List[str] = []
        description.append(f"{'Number of spectra':>24s}:\t{self.n_spectra}")
        description.append(f"{'Number of points':>24s}:\t{self.n_points}")
        description.append(f"{'Profile type':>24s}:\t{self.profile_type}")
        description.append(f"{'Shape':>24s}:\t{self.shape}")
        description.append(f"{'Metadata entries':>24s}:\t{len(self._metadata) if self._metadata else 0}")
        return "\n".join(description)

    def __len__(self) -> int:
        """Return the number of spectra in the measurement."""
        return self.n_spectra
    
    @property
    def spectra(self) -> List[Spectrum]:
        "Return each spectrum of the measurement."
        return self._spectra.copy()

    @property
    def wavenumbers(self) -> np.ndarray:
        """Return the wavenumber axis of the measurement."""
        return self._spectra[0].wavenumbers

    @property
    def intensities(self) -> np.ndarray:
        """Return the intensities array of the measurement."""
        return np.array([s.intensities for s in self._spectra])

    @property
    def metadata(self) -> Metadata:
        """Return the metadata of the measurement."""
        return self._metadata
    
    @property
    def n_spectra(self) -> int:
        """Return the number of spectra in the measurement."""
        return len(self._spectra)
    
    @property
    def n_points(self) -> int:
        """Return the number of (wavenumber, intensity) points in each spectrum."""
        return self._spectra[0].n_points

    @property
    def resolution(self) -> float:
        """Return the spectral resolution of each spectrum in the measurement."""
        return self._spectra[0].resolution

    @property
    def wavenumber_range(self) -> Tuple[float, float]:
        """Return the wavenumber range of the measurement."""
        return self._spectra[0].wavenumber_range

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
        return self._profile.profile_type
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the spatial grid if structured."""
        return self._profile.shape

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
        cls,
        wavenumbers: np.ndarray,
        intensities: np.ndarray,
        positions: np.ndarray = None,
        metadata: Union[Metadata, Dict[str, Any]] = None,
        percolate_metadata: bool = True,
        infer_grid_indices: bool = False,
    ) -> None:
        """Create a Measurement instance from wavenumber and intensity arrays."""
        # validate wavenumbers and intensities 
        try:
            wavenumbers: np.ndarray = np.asarray(wavenumbers, dtype=float)
        except (ValueError, TypeError):
            raise TypeError(
                f"Invalid wavenumbers: wavenumbers must be convertible to a numpy array of floats. "
                f"Got type='{type(wavenumbers)}'."
            )
        try:
            intensities: np.ndarray = np.asarray(intensities, dtype=float)
        except (ValueError, TypeError):
            raise TypeError(
                f"Invalid intensities: intensities must be convertible to a numpy array of floats. "
                f"Got type='{type(intensities)}'."
            )
        if intensities.shape[-1] != wavenumbers.shape[0]:
            raise ValueError(
                f"Invalid wavenumbers and intensities: Last dimension of intensities must match length of wavenumbers. "
                f"Got intensities.shape={intensities.shape} and len(wavenumbers)={wavenumbers.shape[0]}."
            )
        
        # infer grid_indices
        shape: Tuple[int] = intensities.shape[:-1]
        n_points: int = intensities.shape[-1]
        n_spectra: int = sum(shape)
        ndim: int = len(shape)

        if infer_grid_indices and n_spectra > 1:
            grid_indices: Optional[np.ndarray] = np.stack(np.meshgrid(
                *[np.arange(n) for n in shape],
                indexing='ij'
                ), axis=-1).reshape(-1, ndim)
            flattened_intensities: np.ndarray = intensities.reshape(-1, n_points)
        else:
            grid_indices: Optional[np.ndarray] = None
            flattened_intensities: np.ndarray = intensities.reshape(-1, n_points)

        # construct Spectrum objects
        spectra: List[Spectrum] = []
        for i in flattened_intensities:
            s: Spectrum = Spectrum(
                wavenumbers=wavenumbers,
                intensities=i,
                metadata=None,
                parent=None, # temporary, set below
            )
            spectra.append(s)
        
        self: Measurement = cls(
            spectra=spectra,
            grid_indices=grid_indices,
            positions=positions,
            metadata=metadata,
            percolate_metadata=percolate_metadata,
        )
        return self