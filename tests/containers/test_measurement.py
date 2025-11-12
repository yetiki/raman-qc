"""
Author: Yoshiki Cook
Date: 2025-10-22

Updated: 2025-11-12
"""

import numpy as np
import pytest
from rapidqc.core.containers.measurement import Measurement
from rapidqc.core.containers.spectrum import Spectrum
from rapidqc.core.containers.metadata import Metadata

def test_valid_init():
    # with metadata
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    metadata: Metadata = Metadata({'sample_id': 1})
    _ = Measurement(
        spectra=[spectrum for _ in range(10)],
        metadata=metadata
    )

    # without grid indices or positions
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    _ = Measurement(spectra=[spectrum for _ in range(10)])

    # with grid indices and without positions
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    _ = Measurement(
        spectra=[spectrum for _ in range(10)],
        grid_indices=np.array([[i, j] for i in range(2) for j in range(5)]),
        )

    # with positions and without grid indices
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    _ = Measurement(
        spectra=[spectrum for _ in range(10)],
        positions=np.array([[i*0.25, j*0.5] for i in range(2) for j in range(5)]),
        )

    # with grid indices and positions
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    _ = Measurement(
        spectra=[spectrum for _ in range(10)],
        grid_indices=np.array([[i, j] for i in range(2) for j in range(5)]),
        positions=np.array([[i*0.25, j*0.5] for i in range(2) for j in range(5)]),
        )

def test_invalid_init():
    # empty
    with pytest.raises(TypeError):
        _ = Measurement(spectra=None)

    # inconsistent spectra and positions lengths
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    with pytest.raises(ValueError):
        _ = Measurement(
            spectra=[spectrum for _ in range(12)],
            positions=np.array([[i*0.25, j*0.5] for i in range(2) for j in range(5)]),
            )

    # inconsistent spectra and grid indices lengths
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    with pytest.raises(ValueError):
        _ = Measurement(
            spectra=[spectrum for _ in range(12)],
            grid_indices=np.array([[i, j] for i in range(2) for j in range(5)]),
            )

    # invalid spectra type
    with pytest.raises(TypeError):
        _ = Measurement(spectra='invalid_type')

    # inconsistent lengths across spectra
    spectrum_1: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    spectrum_2: Spectrum = Spectrum(wavenumbers=range(600, 1802), intensities=[0]*1202)
    with pytest.raises(ValueError):
        _ = Measurement(spectra=[spectrum_1, spectrum_2])

    # inconsistent wavenumber axes across spectra
    spectrum_1: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    spectrum_2: Spectrum = Spectrum(wavenumbers=range(700, 1901), intensities=[0]*1201)
    with pytest.raises(ValueError):
        _ = Measurement(spectra=[spectrum_1, spectrum_2])

def test_sort_spectra():
    spectrum_1: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[1]*1201)
    spectrum_2: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[2]*1201)
    spectrum_3: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[3]*1201)

    measurement: Measurement = Measurement(
        spectra=[spectrum_1, spectrum_2, spectrum_3],
        grid_indices=[[3], [1], [2]],
    )
    assert measurement.spectra == [spectrum_2, spectrum_3, spectrum_1]

def test_percolate_metadata_to_spectra():
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201, metadata={'sample_id': 1})
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(10)],
        metadata={'excitation_wavelength': '532 nm'},
        percolate_metadata=True,
    )
    assert measurement.spectra[0].metadata == Metadata({'sample_id': 1, 'excitation_wavelength': '532 nm'})

def test_spectral_index_of():
    spectrum_1: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[1]*1201)
    spectrum_2: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[2]*1201)
    spectrum_3: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[3]*1201)

    measurement: Measurement = Measurement(
        spectra=[spectrum_1, spectrum_2, spectrum_3],
    )
    assert measurement._spectral_index_of(spectrum_1) == 0
    assert measurement._spectral_index_of(spectrum_2) == 1
    assert measurement._spectral_index_of(spectrum_3) == 2

def test_wavenumbers():
    spectrum_1: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[1]*1201)
    spectrum_2: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[2]*1201)
    spectrum_3: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[3]*1201)

    measurement: Measurement = Measurement(
        spectra=[spectrum_1, spectrum_2, spectrum_3],
    )
    assert np.array_equal(measurement.wavenumbers, np.arange(600, 1801))

def test_intensities():
    spectrum_1: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[1]*1201)
    spectrum_2: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[2]*1201)
    spectrum_3: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[3]*1201)

    measurement: Measurement = Measurement(
        spectra=[spectrum_1, spectrum_2, spectrum_3],
    )
    assert np.array_equal(measurement.intensities, np.asarray([[i]*1201 for i in range(1, 4)]))

def test_metadata():
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    metadata: Metadata = Metadata({'sample_id': 1})
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(10)],
        metadata=metadata
    )
    assert measurement.metadata == Metadata({'sample_id': 1})

def test_n_spectra():
    spectrum_1: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[1]*1201)
    spectrum_2: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[2]*1201)
    spectrum_3: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[3]*1201)

    measurement: Measurement = Measurement(
        spectra=[spectrum_1, spectrum_2, spectrum_3],
    )
    assert measurement.n_spectra == 3

def test_n_points():
    spectrum_1: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[1]*1201)
    spectrum_2: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[2]*1201)
    spectrum_3: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[3]*1201)

    measurement: Measurement = Measurement(
        spectra=[spectrum_1, spectrum_2, spectrum_3],
    )
    assert measurement.n_points == 1201

def test_resolution():
    spectrum_1: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[1]*1201)
    spectrum_2: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[2]*1201)
    spectrum_3: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[3]*1201)

    measurement: Measurement = Measurement(
        spectra=[spectrum_1, spectrum_2, spectrum_3],
    )
    assert measurement.resolution == 1

def test_wavenumber_range():
    spectrum_1: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[1]*1201)
    spectrum_2: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[2]*1201)
    spectrum_3: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[3]*1201)

    measurement: Measurement = Measurement(
        spectra=[spectrum_1, spectrum_2, spectrum_3],
    )
    assert measurement.wavenumber_range == (600, 1800)

def test_grid_indices():
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(10)],
        grid_indices=np.array([[i, j] for i in range(2) for j in range(5)]),
        )
    assert np.array_equal(measurement.grid_indices, np.array([[i, j] for i in range(2) for j in range(5)]))
    
def test_positions():
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(10)],
        positions=np.array([[i*0.25, j*0.5] for i in range(2) for j in range(5)]),
        )
    assert np.array_equal(measurement.positions, np.array([[i*0.25, j*0.5] for i in range(2) for j in range(5)]))
    
def test_profile_type():
    # unstructured
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(10)],
        )
    assert measurement.profile_type == 'unstructured'

    # point
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(spectra=[spectrum])
    assert measurement.profile_type == 'point'

    # line
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(10)],
        grid_indices=np.array([[i] for i in range(10)]),
        )
    assert measurement.profile_type == 'line'

    # map
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(10)],
        grid_indices=np.array([[i, j] for i in range(2) for j in range(5)]),
        )
    assert measurement.profile_type == 'map'
    
    # volume
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(12)],
        grid_indices=np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(3)]),
        )
    assert measurement.profile_type == 'volume'

def test_shape():
    # unstructured
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(10)],
        )
    assert measurement.shape is None

    # point
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(spectra=[spectrum])
    assert measurement.shape == (1,)

    # line
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(10)],
        grid_indices=np.array([[i] for i in range(10)]),
        )
    assert measurement.shape == (10,)

    # map
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(10)],
        grid_indices=np.array([[i, j] for i in range(2) for j in range(5)]),
        )
    assert measurement.shape == (2, 5)
    
    # volume
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(12)],
        grid_indices=np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(3)]),
        )
    assert measurement.shape == (2, 2, 3)

def test_ndim():
    # unstructured
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(10)],
        )
    assert measurement.ndim is None

    # point
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(spectra=[spectrum])
    assert measurement.ndim == 1

    # line
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(10)],
        grid_indices=np.array([[i] for i in range(10)]),
        )
    assert measurement.ndim == 1

    # map
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(10)],
        grid_indices=np.array([[i, j] for i in range(2) for j in range(5)]),
        )
    assert measurement.ndim == 2
    
    # volume
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(12)],
        grid_indices=np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(3)]),
        )
    assert measurement.ndim == 3

def test_is_structured():
    # unstructured
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(10)],
        )
    assert not measurement.is_structured

    # point
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(spectra=[spectrum])
    assert measurement.is_structured

    # line
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(10)],
        grid_indices=np.array([[i] for i in range(10)]),
        )
    assert measurement.is_structured

    # map
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(10)],
        grid_indices=np.array([[i, j] for i in range(2) for j in range(5)]),
        )
    assert measurement.is_structured
    
    # volume
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(12)],
        grid_indices=np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(3)]),
        )
    assert measurement.is_structured

def test_from_array():
    # unstructured
    wavenumbers: np.ndarray = np.arange(600, 1801)
    intensities: np.ndarray = [[0]*1201 for _ in range(10)]
    measurement: Measurement = Measurement.from_array(
        wavenumbers=wavenumbers,
        intensities=intensities,
    )
    assert measurement.profile_type == 'unstructured'

    # point
    wavenumbers: np.ndarray = np.arange(600, 1801)
    intensities: np.ndarray = [0]*1201
    measurement: Measurement = Measurement.from_array(
        wavenumbers=wavenumbers,
        intensities=intensities,
    )
    assert measurement.profile_type == 'point'
    assert measurement.shape == (1,)
    assert np.array_equal(measurement.grid_indices, np.array([[0]]))

    # line
    wavenumbers: np.ndarray = np.arange(600, 1801)
    intensities: np.ndarray = [[0]*1201 for _ in range(10)]
    measurement: Measurement = Measurement.from_array(
        wavenumbers=wavenumbers,
        intensities=intensities,
        infer_grid_indices=True,
    )
    assert measurement.profile_type == 'line'
    assert measurement.shape == (10,)
    assert np.array_equal(measurement.grid_indices, np.array([[i] for i in range(10)]))

    # map
    wavenumbers: np.ndarray = np.arange(600, 1801)
    intensities: np.ndarray = np.asarray([[[0]*1201 for _ in range(5)] for _ in range(2)])
    measurement: Measurement = Measurement.from_array(
        wavenumbers=wavenumbers,
        intensities=intensities,
        infer_grid_indices=True,
    )
    assert measurement.profile_type == 'map'
    assert measurement.shape == (2, 5)
    assert np.array_equal(measurement.grid_indices, np.array([[i, j] for i in range(2) for j in range(5)]))

    # volume
    wavenumbers: np.ndarray = np.arange(600, 1801)
    intensities: np.ndarray = np.asarray([[[[0]*1201 for _ in range(3)] for _ in range(2)] for _ in range(2)])
    measurement: Measurement = Measurement.from_array(
        wavenumbers=wavenumbers,
        intensities=intensities,
        infer_grid_indices=True,
    )
    assert measurement.profile_type == 'volume'
    assert measurement.shape == (2, 2, 3)
    assert np.array_equal(measurement.grid_indices, np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(3)]))

def test_spectrum_parent():
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    measurement: Measurement = Measurement(
        spectra=[spectrum for _ in range(10)],
        grid_indices=np.array([[i] for i in range(10)]),
        )
    s: Spectrum = measurement.spectra[0]
    assert s.parent == measurement
    
def test_spectrum_spectral_index():
    spectrum_1: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[1]*1201)
    spectrum_2: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[2]*1201)
    spectrum_3: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[3]*1201)

    measurement: Measurement = Measurement(
        spectra=[spectrum_1, spectrum_2, spectrum_3],
        grid_indices=np.array([[i] for i in range(3)]),
        )
    s: Spectrum = measurement.spectra[1]
    assert s.spectral_index == 1

def test_spectrum_grid_index():
    spectrum_1: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[1]*1201)
    spectrum_2: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[2]*1201)
    spectrum_3: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[3]*1201)

    measurement: Measurement = Measurement(
        spectra=[spectrum_1, spectrum_2, spectrum_3],
        grid_indices=np.array([[i] for i in range(3)]),
        )
    s: Spectrum = measurement.spectra[1]
    assert s.grid_index == [1]

def test_spectrum_position():
    spectrum_1: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[1]*1201)
    spectrum_2: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[2]*1201)
    spectrum_3: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[3]*1201)

    measurement: Measurement = Measurement(
        spectra=[spectrum_1, spectrum_2, spectrum_3],
        positions=np.array([[i*0.5] for i in range(3)]),
        )
    s: Spectrum = measurement.spectra[1]
    assert np.array_equal(s.position, [0.5])