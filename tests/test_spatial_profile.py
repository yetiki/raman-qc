"""
Author: Yoshiki Cook
Date: 2025-10-22
"""

import pytest
from ramanqc.spatial_profile import SpatialProfile
from typing import Optional, Literal, Union, Self, Any,List, Dict, Tuple


def test_init():
    # valid initialization
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (0, 1), (1, 0), (1, 1)])
    assert profile.grid_indices == [(0, 0), (0, 1), (1, 0), (1, 1)]

    # inconsistent dimensions
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[(0, 0), (1,)])

    # non-unique grid_indices
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[(0, 0), (0, 1), (1, 0), (0, 0)])

    # empty grid_indices
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[])

    # too many dimensions
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[(0, 0, 0, 0), (1, 1, 1, 1)])
    
def test_init_with_positions():
    # valid initialization
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (0, 1), (1, 0), (1, 1)],
                                             positions=[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
    assert profile.grid_indices == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert profile.positions == [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]

    # inconsistent position dimensions
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[(0, 0), (1, 1)], positions=[(0.0,), (1.0, 1.0)])

    # non-unique positions
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[(0, 0), (1, 1)], positions=[(0.0, 0.0), (0.0, 0.0)])

    # mismatched lengths with grid_indices
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[(0, 0), (1, 1)], positions=[(0.0, 0.0)])

    # too many dimensions
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[(0, 0), (1, 1)], positions=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])

def test_shape():
    # Single spectrum profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0)]) 
    assert profile.shape == (1,)

    # Line profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0,), (1,), (2,), (3,)])
    assert profile.shape == (4,)

    # Map profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (0, 1), (0, 2),
                                                      (1, 0), (1, 1), (1, 2)])
    assert profile.shape == (2, 3)

    # Volume profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0, 0), (0, 0, 1), (0, 0, 2),
                                                      (0, 1, 0), (0, 1, 1), (0, 1, 2),
                                                      (1, 0, 0), (1, 0, 1), (1, 0, 2),
                                                      (1, 1, 0), (1, 1, 1), (1, 1, 2),])
    assert profile.shape == (2, 2, 3)

    # Unstructured profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (2, 3), (5, 1)])
    assert profile.shape is None

def test_profile_type():
    # Single spectrum profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0)]) 
    assert profile.profile_type == 'single'

    # Line profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (1, 0), (2, 0), (3, 0)])
    assert profile.profile_type == 'line'

    # Map profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (0, 1), (0, 2),
                                                      (1, 0), (1, 1), (1, 2)])
    assert profile.profile_type == 'map'

    # Volume profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0, 0), (0, 0, 1), (0, 0, 2),
                                                      (0, 1, 0), (0, 1, 1), (0, 1, 2),
                                                      (1, 0, 0), (1, 0, 1), (1, 0, 2),
                                                      (1, 1, 0), (1, 1, 1), (1, 1, 2),])
    assert profile.profile_type == 'volume'

    # Unstructured profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (2, 3), (5, 1)])
    assert profile.profile_type == 'unstructured'

def test_n_grid_indices():
    # Single spectrum profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0)]) 
    assert profile.n_points == 1

    # Line profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (1, 0), (2, 0), (3, 0)])
    assert profile.n_points == 4

    # Map profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (0, 1), (0, 2),
                                                      (1, 0), (1, 1), (1, 2)])
    assert profile.n_points == 6


    # Volume profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0, 0), (0, 0, 1), (0, 0, 2),
                                                      (0, 1, 0), (0, 1, 1), (0, 1, 2),
                                                      (1, 0, 0), (1, 0, 1), (1, 0, 2),
                                                      (1, 1, 0), (1, 1, 1), (1, 1, 2),])
    assert profile.n_points == 12

    # Unstructured profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (2, 3), (5, 1)])
    assert profile.n_points == 3

def test_ndim():
    # Single spectrum profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0)]) 
    assert profile.ndim == 2

    # Line profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0,), (1,), (2,), (3,)])
    assert profile.ndim == 1

    # Map profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (0, 1), (0, 2),
                                                      (1, 0), (1, 1), (1, 2)])
    assert profile.ndim == 2

    # Volume profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0, 0), (0, 0, 1), (0, 0, 2),
                                                      (0, 1, 0), (0, 1, 1), (0, 1, 2),
                                                      (1, 0, 0), (1, 0, 1), (1, 0, 2),
                                                      (1, 1, 0), (1, 1, 1), (1, 1, 2),])
    assert profile.ndim == 3

    # Unstructured profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (2, 3), (5, 1)])
    assert profile.ndim == 2

def test_consecutive_grid_indices():
    # Single spectrum profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0)]) 
    assert profile._check_consecutive_grid_indices() is True

    # Line profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (1, 0), (2, 0), (3, 0)])
    assert profile._check_consecutive_grid_indices() is True

    # Map profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (0, 1), (0, 2),
                                                      (1, 0), (1, 1), (1, 2)])
    assert profile._check_consecutive_grid_indices() is True

    # Volume profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0, 0), (0, 0, 1), (0, 0, 2),
                                                      (0, 1, 0), (0, 1, 1), (0, 1, 2),
                                                      (1, 0, 0), (1, 0, 1), (1, 0, 2),
                                                      (1, 1, 0), (1, 1, 1), (1, 1, 2),])
    assert profile._check_consecutive_grid_indices() is True

    # Unstructured profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (2, 3), (5, 1)])
    assert profile._check_consecutive_grid_indices() is False

def test_is_structured():
    # Single spectrum profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0)]) 
    assert profile._is_structured() is True

    # Line profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (1, 0), (2, 0), (3, 0)])
    assert profile._is_structured() is True

    # Map profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (0, 1), (0, 2),
                                                      (1, 0), (1, 1), (1, 2)])
    assert profile._is_structured() is True

    # Volume profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0, 0), (0, 0, 1), (0, 0, 2),
                                                      (0, 1, 0), (0, 1, 1), (0, 1, 2),
                                                      (1, 0, 0), (1, 0, 1), (1, 0, 2),
                                                      (1, 1, 0), (1, 1, 1), (1, 1, 2),])
    assert profile._is_structured() is True

    # Unstructured profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (2, 3), (5, 1)])
    assert profile._is_structured() is False

def test_get_neighbours_by_index():
    # Single spectrum profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0)])
    with pytest.raises(ValueError):
        _ = profile._get_neighbours_by_grid_index((0, 0))

    # Line profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0), (1, 0), (2, 0), (3, 0)])
    neighbours: List[Tuple[int, ...]] = profile._get_neighbours_by_grid_index((1, 0), mode='2-connectivity')
    assert set(neighbours) == {(0, 0), (2, 0)}

    with pytest.raises(ValueError):
        _ = profile._get_neighbours_by_grid_index((1, 0), mode='4-connectivity')

    # Map profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[(0, 0, 0), (0, 0, 1), (0, 0, 2),
                                                      (0, 1, 0), (0, 1, 1), (0, 1, 2),
                                                      (1, 0, 0), (1, 0, 1), (1, 0, 2),
                                                      (1, 1, 0), (1, 1, 1), (1, 1, 2),])
    neighbours: List[Tuple[int, ...]] = profile._get_neighbours_by_grid_index((0, 1, 1), mode='4-connectivity')
    assert set(neighbours) == {(0, 0, 1), (0, 2, 1), (1, 1, 1), (0, 1, 0), (0, 1, 2)}

    # Volume profile

    # Unstructured profile
    pass

def test_get_neighbours_by_positions():
    # Single spectrum profile

    # Line profile

    # Map profile

    # Volume profile

    # Unstructured profile

    # No positions
    pass

def test_get_neighbours():
    # Single spectrum profile

    # Line profile

    # Map profile

    # Volume profile

    # Unstructured profile
    pass