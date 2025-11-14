from rapidqc.core.containers.spectrum import Spectrum
from rapidqc.core.containers.measurement import Measurement
from typing import Callable, Union, List, Any, Dict
import numpy as np
import copy

class Operator:
    def __init__(self,
                 method: Callable,
                 **kwargs
        ) -> None:
        self.method: Callable = method
        self.kwargs: Dict[str, Any] = kwargs

    def __repr__(self):
        return f"{self.__class__.__name__}(kwargs={self.kwargs})"
    
    def __call__(self,
                 wavenumbers: np.ndarray,
                 intensities: np.ndarray,
                 *args,
                 **kwargs
        ):
        return self.method(wavenumbers, intensities, *args, **kwargs)
    
    def _process_spectrum(self, spectrum: Spectrum) -> Spectrum:
        """Return a deep-copied Spectrum with transformed wavenumbers and intensities."""
        return copy.deepcopy(spectrum)
    
    def _process_measurement(self, measurement: Measurement) -> Measurement:
        """Process all spectra in the measurement."""
        processed_measurement: Measurement = copy.deepcopy(measurement)
        processed_measurement._spectra = [self._process_spectrum(s) for s in measurement.spectra]
        return processed_measurement
    
    def apply(self, spectral_obj: Union[Spectrum, Measurement]) -> Union[Spectrum, Measurement]:
        """Apply preprocessing operator to a Spectrum, or Measurement."""
        if isinstance(spectral_obj, Spectrum):
            return self._process_spectrum(spectral_obj)
        elif isinstance(spectral_obj, Measurement):
            return self._process_measurement(spectral_obj)
        else:
            raise TypeError(
                f"Invalid type: spectral_obj must be of type: Spectrum or Measurement. "
                f"Got type={type(spectral_obj)}."
            )
