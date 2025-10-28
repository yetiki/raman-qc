"""
Author: Yoshiki Cook
Date: 2025-10-22

Updated: 2025-10-24
"""

import pytest
import numpy as np
from ramanqc.spatial_profile import SpatialProfile

def test_valid_init_with_grid_indices():
    # point
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0]])
    assert np.all(profile.grid_indices == np.array([[0, 0]]))

    # line
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0], [0, 1]])
    assert np.all(profile.grid_indices == np.array([[0, 0], [0, 1]]))

    # map
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0], [0, 1],
                                                           [1, 0], [1, 1]])
    assert np.all(profile.grid_indices == np.array([[0, 0], [0, 1],
                                                    [1, 0], [1, 1]]))

    # volume
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                           [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                           [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                                           [1, 1, 0], [1, 1, 1], [1, 1, 2]])
    assert np.all(profile.grid_indices == np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                    [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                    [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                                    [1, 1, 0], [1, 1, 1], [1, 1, 2]]))
    
    # non-ascending indices
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 2], [0, 0], [0, 1]])
    assert np.all(profile.grid_indices == np.array([[0, 0], [0, 1], [0, 2]]))

def test_invalid_init_with_grid_indices():
    # invalid data type
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[['foo']])

    # inconsistent dimensions
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[[0, 0], [1]])

    # non-unique grid_indices
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[[0, 0], [0, 1], [1, 0], [0, 0]])

    # too many dimensions
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[[0, 0, 0, 0], [1, 1, 1, 1]])
    
    # non-2D array (after conversion)
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[[[0, 0], [0, 1]], [[1, 0], [1, 1]]])

def test_invalid_init_with_missing_indices():
    # no indices
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[])

    # line
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[[0, 0], [0, 2]])

    # map
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[[0, 0], [0, 1],
                                         [1, 0]])
        
    # volume
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                         [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                         [0, 1, 0], [0, 1, 1]])
    
def test_valid_init_with_positions():
    # point
    profile: SpatialProfile = SpatialProfile(positions=[[0, 0]])
    assert np.all(profile.positions == np.array([[0, 0]]))

    # line
    profile: SpatialProfile = SpatialProfile(positions=[[0.0, 0.0], [0.0, 1.0]])
    assert np.all(profile.positions == np.array([[0.0, 0.0], [0.0, 1.0]]))

    # map
    profile: SpatialProfile = SpatialProfile(positions=[[0.0, 0.0], [0.0, 1.0],
                                                        [1.0, 0.0], [1.0, 1.0]])
    assert np.all(profile.positions == np.array([[0.0, 0.0], [0.0, 1.0],
                                                 [1.0, 0.0], [1.0, 1.0]]))

    # volume
    profile: SpatialProfile = SpatialProfile(positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                        [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                        [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                                        [1, 1, 0], [1, 1, 1], [1, 1, 2]])
    assert np.all(profile.positions == np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                    [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                    [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                                    [1, 1, 0], [1, 1, 1], [1, 1, 2]]))
    
    # non-ascending positions
    profile: SpatialProfile = SpatialProfile(positions=[[0, 2], [0, 0], [0, 1]])
    assert np.all(profile.positions == np.array([[0, 0], [0, 1], [0, 2]]))
    
def test_invalid_init_with_positions():
    # invalid data type
    with pytest.raises(ValueError):
        _ = SpatialProfile(positions=[['foo']])

    # inconsistent dimensions 
    with pytest.raises(ValueError):
        _ = SpatialProfile(positions=[[0.0], [0.0, 1.0]])

    # non-unique positions
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[[0, 0], [0, 1]], positions=[[0.0, 0.0], [0.0, 0.0]])

    # empty positions
    with pytest.raises(ValueError):
        _ = SpatialProfile(positions=[])

    # too many dimensions
    with pytest.raises(ValueError):
        _ = SpatialProfile(positions=[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    # non-2D array (after conversion)
    with pytest.raises(ValueError):
        _ = SpatialProfile(positions=[[[0, 0], [0, 1]], [[1, 0], [1, 1]]])

def test_invalid_init_with_missing_positions():
    # no positions
    with pytest.raises(ValueError):
        _ = SpatialProfile(positions=[])

    # line
    with pytest.raises(ValueError):
        _ = SpatialProfile(positions=[[0.0, 1.5], [0.0, 3.0], [0.0, 6.0],]) 

    # map
    with pytest.raises(ValueError):
        _ = SpatialProfile(positions=[[12, 32], [12, 36],
                                      [24, 36]])
        
    # volume
    with pytest.raises(ValueError):
        _ = SpatialProfile(positions=[[89, 56, 34], [89, 56, 40], [89, 56, 46],
                                      [89, 78, 34], [89, 78, 40], [89, 78, 46],
                                      [91, 56, 34], [91, 56, 40]]) 

def test_valid_init_with_grid_indices_and_positions():
    # single
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0]],
                                             positions=[[1.2, 3.4, 5.6]])
    assert np.all(profile.grid_indices == np.array([[0, 0]]))
    assert np.all(profile.positions == np.array([[1.2, 3.4, 5.6]]))

    # line
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0], [1], [2]],
                                             positions=[[1.2, 3.4, 5.6], [7.8, 3.4, 5.6], [14.2, 3.4, 5.6]])
    assert np.all(profile.grid_indices == np.array([[0], [1], [2]]))
    assert np.all(profile.positions == np.array([[1.2, 3.4, 5.6], [7.8, 3.4, 5.6], [14.2, 3.4, 5.6]]))

    # map
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0], [0, 1],
                                                           [1, 0], [1, 1]],
                                             positions=[[0.0, 0.0], [0.0, 1.0],
                                                        [1.0, 0.0], [1.0, 1.0]])
    assert np.all(profile.grid_indices == np.array([[0, 0], [0, 1],
                                                    [1, 0], [1, 1]]))
    assert np.all(profile.positions == np.array([[0.0, 0.0], [0.0, 1.0],
                                                 [1.0, 0.0], [1.0, 1.0]]))

    # volume
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                           [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                           [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                                           [1, 1, 0], [1, 1, 1], [1, 1, 2]],
                                             positions=np.array([[0.1, 5.2, 3.8], [0.1, 5.2, 7.0], [0.1, 5.2, 10.2],
                                                                 [0.1, 6.4, 3.8], [0.1, 6.4, 7.0], [0.1, 6.4, 10.2],
                                                                 [1.6, 5.2, 3.8], [1.6, 5.2, 7.0], [1.6, 5.2, 10.2],
                                                                 [1.6, 6.4, 3.8], [1.6, 6.4, 7.0], [1.6, 6.4, 10.2]]))
    
    assert np.all(profile.grid_indices == np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                    [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                    [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                                    [1, 1, 0], [1, 1, 1], [1, 1, 2]]))
    assert np.all(profile.positions == np.array([[0.1, 5.2, 3.8], [0.1, 5.2, 7.0], [0.1, 5.2, 10.2],
                                                 [0.1, 6.4, 3.8], [0.1, 6.4, 7.0], [0.1, 6.4, 10.2],
                                                 [1.6, 5.2, 3.8], [1.6, 5.2, 7.0], [1.6, 5.2, 10.2],
                                                 [1.6, 6.4, 3.8], [1.6, 6.4, 7.0], [1.6, 6.4, 10.2]]))

def test_invalid_init_with_grid_indices_and_positions():
    # mismatched shape
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[[0, 0], [0, 1]], positions=[[0.0, 1.0]])

    # mismatched length
    with pytest.raises(ValueError):
        _ = SpatialProfile(grid_indices=[[0, 0], [0, 1]], positions=[[0.0], [0.0, 1.0], [0.0, 2.0]])

def test_infer_indices_from_positions():
    # point
    profile: SpatialProfile = SpatialProfile(positions = np.array([[1, 2]]))
    assert np.all(profile.grid_indices == np.array([[0, 0]]))

    # line
    profile: SpatialProfile = SpatialProfile(positions = np.array([[1.2, 3.4, 5.6], [1.2, 3.4, 6.2], [1.2, 3.4, 6.8]]))
    assert np.all(profile.grid_indices == np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]]))

    # map
    profile: SpatialProfile = SpatialProfile(positions = np.array([[5, 8], [5, 10], [5, 12],
                                                                   [6, 8], [6, 10], [6, 12],
                                                                   [7, 8], [7, 10], [7, 12],
                                                                   ]))
    assert np.all(profile.grid_indices == np.array([[0, 0], [0, 1], [0, 2],
                                                    [1, 0], [1, 1], [1, 2],
                                                    [2, 0], [2, 1], [2, 2],
                                                    ]))
    
    # volume
    profile: SpatialProfile = SpatialProfile(positions=[[98, 56, 12], [98, 56, 34], [98, 56, 56],
                                                        [98, 78, 12], [98, 78, 34], [98, 78, 56],
                                                        [100, 56, 12], [100, 56, 34], [100, 56, 56],
                                                        [100, 78, 12], [100, 78, 34], [100, 78, 56]])
    assert np.all(profile.grid_indices == np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                    [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                    [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                                    [1, 1, 0], [1, 1, 1], [1, 1, 2]]))


def test_shape():
    # point
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0]]) 
    assert profile.shape == (1,)

    # line
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0], [1], [2], [3]])
    assert profile.shape == (4,)

    # map
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0], [0, 1], [0, 2],
                                                           [1, 0], [1, 1], [1, 2]])
    assert profile.shape == (2, 3)

    # volume
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                           [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                           [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                                           [1, 1, 0], [1, 1, 1], [1, 1, 2],])
    assert profile.shape == (2, 2, 3)

def test_profile_type():
    # point
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0]]) 
    assert profile.profile_type == 'point'

    # line
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0], [1, 0], [2, 0], [3, 0]])
    assert profile.profile_type == 'line'

    # map
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0], [0, 1], [0, 2],
                                                           [1, 0], [1, 1], [1, 2]])
    assert profile.profile_type == 'map'

    # volume
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                           [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                           [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                                           [1, 1, 0], [1, 1, 1], [1, 1, 2],])
    assert profile.profile_type == 'volume'

def test_n_points():
    # single
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0]]) 
    assert profile.n_points == 1

    # line
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0], [1, 0], [2, 0], [3, 0]])
    assert profile.n_points == 4

    # map
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0], [0, 1], [0, 2],
                                                           [1, 0], [1, 1], [1, 2]])
    assert profile.n_points == 6


    # volume
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                           [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                           [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                                           [1, 1, 0], [1, 1, 1], [1, 1, 2],])
    assert profile.n_points == 12

def test_ndim_with_grid_indices():
    # single
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0, 0]]) 
    assert profile.ndim == 1

    # line
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0], [1], [2], [3]])
    assert profile.ndim == 1

    # map
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0], [0, 1], [0, 2],
                                                           [1, 0], [1, 1], [1, 2]])
    assert profile.ndim == 2

    # volume
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                           [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                           [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                                           [1, 1, 0], [1, 1, 1], [1, 1, 2],])
    assert profile.ndim == 3

def test_ndim_with_positions():
    # single
    profile: SpatialProfile = SpatialProfile(positions=[[0, 0, 0]]) 
    assert profile.ndim == 1

    # line
    profile: SpatialProfile = SpatialProfile(positions=[[0], [1], [2], [3]])
    assert profile.ndim == 1

    # map
    profile: SpatialProfile = SpatialProfile(positions=[[0, 0], [0, 1], [0, 2],
                                                           [1, 0], [1, 1], [1, 2]])
    assert profile.ndim == 2

    # volume
    profile: SpatialProfile = SpatialProfile(positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                           [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                           [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                                           [1, 1, 0], [1, 1, 1], [1, 1, 2],])
    assert profile.ndim == 3

def test_is_structured_with_grid_indices():
    # single
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0, 0]]) 
    assert profile.is_structured

    # line
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0], [1], [2], [3]])
    assert profile.is_structured

    # map
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0], [0, 1], [0, 2],
                                                           [1, 0], [1, 1], [1, 2]])
    assert profile.is_structured
    
    # volume
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                           [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                           [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                                           [1, 1, 0], [1, 1, 1], [1, 1, 2],])
    assert profile.is_structured

def test_is_structured_with_positions():
    # single
    profile: SpatialProfile = SpatialProfile(positions=[[0, 0, 0]]) 
    assert profile.is_structured

    # line
    profile: SpatialProfile = SpatialProfile(positions=[[0], [1], [2], [3]])
    assert profile.is_structured

    # map
    profile: SpatialProfile = SpatialProfile(positions=[[0, 0], [0, 1], [0, 2],
                                                           [1, 0], [1, 1], [1, 2]])
    assert profile.is_structured
    
    # volume
    profile: SpatialProfile = SpatialProfile(positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                           [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                           [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                                           [1, 1, 0], [1, 1, 1], [1, 1, 2],])
    assert profile.is_structured

def test_has_grid_indices():
    # no grid_indices
    profile: SpatialProfile = SpatialProfile(positions=[[0, 0, 0]]) 
    assert profile.has_grid_indices

    # grid_indices
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0, 0]]) 
    assert profile.has_grid_indices

def test_has_positions():
    # no positions
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0, 0]]) 
    assert profile.has_positions is False

    # positions
    profile: SpatialProfile = SpatialProfile(positions=[[0, 0, 0]]) 
    assert profile.has_positions

def test_bounds():
    # no positions
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0, 0]]) 
    assert profile.bounds is None

    # single
    profile: SpatialProfile = SpatialProfile(positions=[[0, 0, 0]]) 
    assert np.all(profile.bounds[0] == np.array([0, 0, 0]))
    assert np.all(profile.bounds[1] == np.array([0, 0, 0]))

    # line
    profile: SpatialProfile = SpatialProfile(positions=[[0], [1], [2], [3]])
    assert np.all(profile.bounds[0] == np.array([0]))
    assert np.all(profile.bounds[1] == np.array([3]))

    # map
    profile: SpatialProfile = SpatialProfile(positions=[[0, 0], [0, 1], [0, 2],
                                                           [1, 0], [1, 1], [1, 2]])
    assert np.all(profile.bounds[0] == np.array([0, 0]))
    assert np.all(profile.bounds[1] == np.array([1, 2]))
    
    # volume
    profile: SpatialProfile = SpatialProfile(positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                           [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                           [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                                           [1, 1, 0], [1, 1, 1], [1, 1, 2],])
    assert np.all(profile.bounds[0] == np.array([0, 0, 0]))
    assert np.all(profile.bounds[1] == np.array([1, 1, 2]))

def test_get_grid_index():
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0, 0]]) 
    assert np.all(profile.get_grid_index(0) == np.array([0, 0, 0]))

    # index out of bounds
    assert profile.get_grid_index(1) is None
    assert profile.get_grid_index(1, default='default') == 'default'

def test_get_position():
    profile: SpatialProfile = SpatialProfile(positions=[[0, 0, 0]]) 
    assert np.all(profile.get_position(0) == np.array([0, 0, 0]))

    # index out of bounds
    assert profile.get_position(1) is None
    assert profile.get_position(1, default='default') == 'default'

    # no positions
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0, 0]]) 
    assert profile.get_position(0) is None
    assert profile.get_position(0, default='default') == 'default'

def test_get_neighbours_by_grid_index():
    # 1d profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0], [1], [2], [3]])
    assert np.all(profile.get_neighbours(1, mode='2-connectivity') == np.array([[0], [2]]))

    # 2d profile
    profile: SpatialProfile = SpatialProfile(grid_indices=[[0, 0], [0, 1], [0, 2],
                                                           [1, 0], [1, 1], [1, 2]])
    assert np.all(profile.get_neighbours(1, mode='4-connectivity') == np.array([[0, 0], [0, 2], [1, 1]]))
    assert np.all(profile.get_neighbours(1, mode='8-connectivity') == np.array([[0, 0], [0, 2], [1, 0], [1, 1], [1, 2]]))

    # 3d profile
    profile: SpatialProfile = SpatialProfile(positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                        [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                        [0, 2, 0], [0, 2, 1], [0, 2, 2],
                                                        [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                                        [1, 1, 0], [1, 1, 1], [1, 1, 2],
                                                        [1, 2, 0], [1, 2, 1], [1, 2, 2],
                                                        [2, 0, 0], [2, 0, 1], [2, 0, 2],
                                                        [2, 1, 0], [2, 1, 1], [2, 1, 2],
                                                        [2, 2, 0], [2, 2, 1], [2, 2, 2],
                                                        ])
    assert np.all(profile.get_neighbours(0, mode='6-connectivity') == np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))
    assert np.all(profile.get_neighbours(10, mode='6-connectivity') == np.array([[0, 0, 1], [1, 0, 0], [1, 0, 2], [1, 1, 1], [2, 0, 1]]))
    assert np.all(profile.get_neighbours(10, mode='26-connectivity') == np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                                                  [1, 0, 0], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2],
                                                                                  [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2]]))

def test_get_neighbours_by_positions():
    # single

    # line

    # map

    # volume

    # no positions
    pass

def test_valid_get_neighbours():
    # single

    # line

    # map

    # volume

    pass

def test_invalid_get_neighbours():
    # single

    # line

    # map

    # volume

    pass