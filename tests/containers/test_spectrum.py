"""
Author: Yoshiki Cook
Date: 2025-10-22

Updated: 2025-11-12
"""

import pytest
from typing import Optional, Union, Self, Any, Dict, Tuple
import numpy as np
from rapidqc.core.containers.spectrum import Spectrum
from rapidqc.core.containers.metadata import Metadata

def test_valid_init():
    # valid wavenumbers and intensities without metadata
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)

    # valid wavenumbers and intensities with metadata
    metadata: Metadata = Metadata({'sample_id': 'S1', 'excitation_wavelength': '532 nm'})
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201, metadata=metadata)

def test_invalid_init():
    # invalid wavenumbers type
    with pytest.raises(TypeError):
        _ = Spectrum(wavenumbers='invalid_type', intensities=[0]*1201)

    # invalid intensities type
    with pytest.raises(TypeError):
        _ = Spectrum(wavenumbers=range(600, 1801), intensities='invalid_type')

    # invalid wavenumbers and intensities length
    with pytest.raises(ValueError):
        _ = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1200)

    # invalid metadata type
    with pytest.raises(TypeError):
        _ = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201, metadata='invalid_type')

def test_eq_():
    # equal wavenumbers and intensities
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    other: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    assert spectrum == other

    # equal wavenumbers, different intensities
    other: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[1]*1201)
    assert spectrum != other

    # different wavenumbers, equal intensities
    other: Spectrum = Spectrum(wavenumbers=range(601, 1802), intensities=[0]*1201)
    assert spectrum != other

    # equal wavenumbers and intensities different metadata
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201, metadata={'sample_id': 0})
    other: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201, metadata={'sample_id': 1})
    assert spectrum == other

def test_add_():
    # equal wavenumbers
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[1]*1201)
    other: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[2]*1201)
    sum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[3]*1201)
    assert spectrum + other == sum

    # different wavenumbers
    other: Spectrum = Spectrum(wavenumbers=range(601, 1802), intensities=[0]*1201)
    with pytest.raises(ValueError):
        _ = spectrum + other

def test_sub_():
    # equal wavenumbers
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[3]*1201)
    other: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[2]*1201)
    sum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[1]*1201)
    assert spectrum - other == sum

    # different wavenumbers
    other: Spectrum = Spectrum(wavenumbers=range(601, 1802), intensities=[0]*1201)
    with pytest.raises(ValueError):
        _ = spectrum - other

def test_sort():
    spectrum: Spectrum = Spectrum(wavenumbers=np.random.permutation(np.arange(600, 1801)), intensities=[0]*1201)
    assert np.array_equal(spectrum.wavenumbers, np.arange(600, 1801))

def test_wavenumbers():
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    assert np.array_equal(spectrum.wavenumbers, np.arange(600, 1801))


def test_intensities():
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    assert np.array_equal(spectrum.intensities, [0]*1201)

def test_metadata():
    # with metadata
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201, metadata={'key': 'value'})
    assert spectrum.metadata == Metadata({'key': 'value'})

    # without metadata
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    assert spectrum.metadata == Metadata({})


def test_n_points():
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    assert spectrum.n_points == 1201

def test_resolution():
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    assert spectrum.resolution == 1.0

def test_wavenumber_range():
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    assert spectrum.wavenumber_range == (600, 1800)

def test_copy():
    spectrum: Spectrum = Spectrum(wavenumbers=range(600, 1801), intensities=[0]*1201)
    copy: Spectrum = spectrum.copy()
    assert copy == spectrum
    assert copy is not spectrum