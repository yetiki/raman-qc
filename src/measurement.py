from typing import Optional, Union, Self, Any,List, Dict, Tuple
import numpy as np
from metadata import Metadata
from spectrum import Spectrum
from profile import Profile

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
    def __init__(self, spectra: List[Spectrum], positions: List[Tuple[Union[float, int], ...]], indexes: List[Tuple[int, ...]], metadata: Union[Metadata, Dict[str, Any]], percolate_metadata: bool = True) -> None:
        self.spectra: np.ndarray = spectra
    
    def __repr__(self) -> str:
        pass

    def __len__(self) -> int:
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
    def n_points(self) -> int:
        pass

    @property
    def n_spectra(self) -> int:
        pass
    
    @property
    def shape(self) -> Tuple[int, ...]:
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
    
    @classmethod
    def from_array(self, wavenumbers: np.ndarray, intensities: np.ndarray, positions: np.ndarray = None, metadata: Union[Metadata, Dict[str, Any]] = None, percolate_metadata: bool = True) -> None:
        pass

    def _percolate_spectrum_metadata(self) -> None:
        pass