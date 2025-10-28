"""
Author: Yoshiki Cook
Date: 2025-10-20

Updated: 2025-10-24
"""

from typing import Optional, Literal, Union, Self, Any, List, Dict, Tuple, Sequence, Set
import numpy as np

from typing import List, Tuple, Optional, Literal, Union

class SpatialProfile:
    """
    Represents the spatial organisation of spectra within a measurement.
    
    The SpatialProfile class stores and manages the spatial structure
    (grid indices and/or physical positions) associated with each spectrum.
    It can infer shape and dimensionality, validate consistency between
    indices and positions, and provide neighbour relationships.

    Assumes the provided spatial grid represents a complete, valid structure
    with no missing values.

    Parameters
    ----------
        
    Attributes
    ----------

    Methods
    ----------
    """

    def __init__(
        self,
        grid_indices: Optional[Union[np.ndarray, Sequence[Sequence[int]]]] = None,
        positions: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
    ) -> None:
        
        # ensure arrays 2D, homogeneous, and at most 3 spatial dimensions
        if grid_indices is not None:
            self._grid_indices: np.ndarray = np.asarray(grid_indices, dtype=int)
            self._validate_array(self._grid_indices, name='grid_indices')
        else:
            self._grid_indices: None = None
        
        if positions is not None:
            self._positions: np.ndarray = np.asarray(positions, dtype=float)
            self._validate_array(self._positions, name='positions')
        else:
            self._positions: None = None

        # ensure at least one of grid_indices or positions is defined
        self._validate_inputs()

        # ensure lengths match if both grid_indices and positions are defined
        self._validate_lengths()

        # ensure grid_indices and positions are sorted in ascending order
        self._normalise_grid_order()

        # ensure no missing values
        self._validate_n_missing_points()

        # ensure grid_indices and positions structures match
        if self._grid_indices is not None and self._positions is not None:
            self._validate_positions_against_indices()

        # infer grid_indices from positions if not defined
        if self._grid_indices is None:
            inferred_indices: np.ndarray = self._infer_indices_from_positions()

            if inferred_indices is not None:
                self._grid_indices: np.ndarray = inferred_indices

        # infer shape and ndim
        if self._grid_indices is not None:
            self._shape: Tuple[int, ...] = self._infer_shape_from_indices()
            self._ndim: int = len(self.shape)
        elif self._positions is not None:
            self._shape: Tuple[int, ...] = None
            self._ndim: int = len(self._positions[0])
        else:
            self._ndim: None = None
            self._shape: None = None

        # infer profile type from grid indices and shape
        self._profile_type: str = self._infer_profile_type()
        self._n_points: int = self._infer_n_points() 

    def _validate_array(self, arr: np.ndarray, name: str) -> None:
        """Ensure array is 2D, homogeneous, and at most 3 spatial dimensions."""
        if arr.ndim != 2:
            raise ValueError(
                f"Invalid {name} shape: {name} must be a 2D array of at most 3 spatial dimensions. "
                f"Got {arr.ndim}D array after conversion."
            )
        if arr.shape[1] == 0 or arr.shape[1] > 3:
            raise ValueError(
                f"Invalid number of spatial dimensions: {name} must have between 1 and 3 spatial dimensions. "
                f"Got {arr.shape[1]} spatial dimensions."
            )
        if len(np.unique(arr, axis=0)) != len(arr):
            raise ValueError(
                f"Invalid {name} values: each item in {name} must be unique. "
                f"Got {len(arr) - len(np.unique(arr, axis=0))} duplicates in {name}."
            )

    def _validate_inputs(self) -> None:
        """Ensure that at least one of grid_indices or positions is defined."""
        if self._grid_indices is None and self._positions is None:
            raise ValueError(
                f"Invalid value: either grid_indices or positions must be provided. "
                f"Got grid_indices=None and positions=None."
            )
           
    def _validate_lengths(self) -> None:
        """Ensure grid_indices and positions (if both defined) have the same length."""
        if self._grid_indices is not None and self._positions is not None:
            if len(self._grid_indices) != len(self._positions):
                raise ValueError(
                    f"Invalid length: grid_indices and positions must have equal length. "
                    f"Got len(grid_indices)={len(self._grid_indices)} and len(positions)={len(self._positions)}"
                )
            
    def _normalise_grid_order(self) -> None:
        """Ensure grid_indices and positions are sorted in ascending order."""
        if self._grid_indices is None and self._positions is None:
            return None

        if self._grid_indices is None:
            # sort based on lexicographic order of indices
            sort_order: np.ndarray = np.lexsort(self._positions.T[::-1])
            self._positions: np.ndarray = self._positions[sort_order]
            return None
        else:
            # sort based on lexicographic order of indices
            sort_order: np.ndarray = np.lexsort(self._grid_indices.T[::-1])
            self._grid_indices: np.ndarray = self._grid_indices[sort_order]

            if self._positions is not None:
                self._positions: np.ndarray = self._positions[sort_order]
        return None

    def _validate_n_missing_points(self, tolerance: float = 1e-6) -> None:
        """Ensure there are no missing grid indices or positions."""
        if self._grid_indices is not None:
            if len(self._grid_indices) == 1:
                return None

            grid: np.ndarray = self._grid_indices
            ndim: int = grid.shape[1]

            # build the full set of expected coordinates for a complete grid
            expected_coords: Set = set(
                tuple(idx) for idx in np.ndindex(
                    *[grid[:, d].max() - grid[:, d].min() + 1 for d in range(ndim)]
                )
            )
            # shift actual coordinates to start from zero for comparison
            shifted_actual: Set = set(
                tuple(idx - grid[:, d].min() for d, idx in enumerate(point))
                for point in grid
            )

            missing: Set = expected_coords - shifted_actual
            if missing:
                raise ValueError(
                    f"Invalid grid_indices values: grid_indices must be from a complete uniform grid. "
                    f"Got {len(missing)} missing indices from expected uniform grid."
                )

        elif self._positions is not None:
            if len(self._positions) <= 2:
                return None

            ndim: int = self._positions.shape[1]
            mins: np.ndarray = self._positions.min(axis=0)
            maxs: np.ndarray = self._positions.max(axis=0)

            # unique coordinates and counts
            unique_per_dim: List = [np.unique(self._positions[:, d]) for d in range(ndim)]
            counts: np.ndarray = np.array([len(u) for u in unique_per_dim])
            units: np.ndarray = np.empty_like(maxs, dtype=float)
            for d in range(ndim):
                if counts[d] <= 1 or np.isclose(maxs[d], mins[d]):
                    units[d] = 1.0
                else:
                    units[d] = (maxs[d] - mins[d]) / (counts[d] - 1)

            # expected regular grid positions
            mesh: np.ndarray = np.meshgrid(*[np.arange(counts[d]) for d in range(ndim)], indexing="ij")
            grid_indices: np.ndarray = np.stack([m.flatten() for m in mesh], axis=1)
            expected_positions: np.ndarray = mins + grid_indices * units

            # check that all expected positions exist in actual positions
            matched: List = []
            for epos in expected_positions:
                diffs = np.linalg.norm(self._positions - epos, axis=1)
                matched.append(np.any(diffs < tolerance))

            if not all(matched):
                missing_count: int = len(expected_positions) - sum(matched)
                raise ValueError(
                    f"Invalid positions: positions must be a complete uniform grid. "
                    f"Got {missing_count} missing positions from expected uniform grid."
                )

        else:
            # grid_indices is None and positions is None
            return None

    def _validate_positions_against_indices(self) -> None:
        """Ensure positions and grid_indices correspond to the same structure."""
        pass
        # TODO
        
        if self._grid_indices is None or self._positions is None:
            return None
        
        self._validate_lengths()

        lex_sorted_indices: np.ndarray = self._grid_indices[np.lexsort(self._grid_indices.T)]
        lex_sorted_positions: np.ndarray = self._positions[np.lexsort(self._positions.T)]

        unique_indices: np.ndarray
        unique_indices_map: np.ndarray
        unique_indices, unique_indices_map = np.unique(lex_sorted_indices, axis=0, return_inverse=True)
        
        unique_positions: np.ndarray
        unique_positions_map: np.ndarray
        unique_positions, unique_positions_map = np.unique(lex_sorted_positions, axis=0, return_inverse=True)

        if len(unique_indices) != len(unique_positions):
            raise ValueError(
                f"Invalid values: grid_indices and positions must have the same number of unique values. "
                f"Got {len(unique_indices)} unique values in grid_indices. "
                f"Got {len(unique_positions)} unique values in positions. "
            )
        
        if not np.array_equal(unique_indices_map, unique_positions_map):
            raise ValueError(
                f"Invalid values: grid_indices and positions must describe the same structure. "
            )

    def _infer_indices_from_positions(self, tolerance: float = 1e-6) -> np.ndarray:
        """Infer grid indices from ordered and regular spatial positions."""
        if self._positions is None:
            raise ValueError("Cannot infer grid indices — positions are not defined.")      

        ndim: int = self._positions.shape[1]

        if len(self._positions) == 1:
            return np.zeros(shape=(1, ndim))

        # compute range and unique counts per dimension
        mins: np.ndarray = self._positions.min(axis=0)
        maxs: np.ndarray = self._positions.max(axis=0)
        unique_vals: np.ndarray = [np.unique(self._positions[:, d]) for d in range(ndim)]
        counts: np.ndarray = np.array([len(u) for u in unique_vals])

        # compute unit spacing per dimension
        units: np.ndarray = np.empty_like(maxs, dtype=float)
        for d in range(ndim):
            if counts[d] <= 1 or np.isclose(maxs[d], mins[d]):
                # arbitrary unit size if no variation in dimension d
                units[d] = 1.0
            else:
                units[d] = (maxs[d] - mins[d]) / (counts[d] - 1)

        # offset and normalise positions
        norm_pos: np.ndarray = (self._positions - mins) / units

        # Round to nearest integer index
        indices: np.ndarray = np.round(norm_pos).astype(int)

        # Verify indices reconstruct positions within tolerance
        reconstructed: np.ndarray = mins + indices * units
        if not np.allclose(self._positions, reconstructed, atol=tolerance):
            raise ValueError(
                "Invalid positions values: positions must be consistent with a regular grid. " 
                f"Cannot infer grid_indices reliably."
            )

        return indices
    
    def _infer_shape_from_indices(self) -> Sequence[int]:
        """Infer the overall grid shape from grid indices."""
        if self._grid_indices is None:
            return None
        
        n_grid_indices: int = len(self._grid_indices)
        ndim: int = self._grid_indices.shape[1]

        # Count unique coordinate values along each axis
        unique_counts: List[int] = [len(np.unique(self._grid_indices[:, i])) for i in range(ndim)]

        # Expected total points if it's a full grid
        expected_total: int = int(np.prod(unique_counts))

        # If it's a perfect grid return all axis n_pointss, otherwise if consecutive treat as a line (list of points), else None
        consecutive: bool = self._check_consecutive_grid_indices()

        if n_grid_indices == 1:
            shape: Tuple[int, ...] = (1,)
        elif n_grid_indices == expected_total:
            shape: Tuple[int, ...] = tuple(unique_counts)
        elif consecutive:
            shape: Tuple[int, ...] = (n_grid_indices,)
        else:
            shape: None = None
        return shape
    
    def _check_consecutive_grid_indices(self) -> bool:
        """Check whether grid indices are consecutive (for structured profiles)."""
        if self._grid_indices is None:
            return False
        
        if len(self._grid_indices) == 1:
            return True
        
        ndim: int = self._grid_indices.shape[1]
        expected: List = [set(range(self._grid_indices[:, d].min(), self._grid_indices[:, d].max() + 1))
                          for d in range(ndim)]
        actual: List = [set(self._grid_indices[:, d])
                        for d in range(ndim)]
        return all(e.issubset(a) for e, a in zip(expected, actual))

    def _infer_profile_type(self) -> Literal['point', 'line', 'map', 'volume', 'unstructured']:
        """Infer the spatial profile type based on dimensionality and structure."""
        if self._grid_indices is None or self._shape is None or not self._check_consecutive_grid_indices():
            return 'unstructured'

        if len(self._grid_indices) == 1:
            return 'point'

        true_ndim: int = sum(n_idx != 1 for n_idx in self._shape)
        if true_ndim == 1:
            return 'line'
        elif true_ndim == 2:
            return 'map'
        elif true_ndim == 3:
            return 'volume'
        else:
            return 'unstructured'
        
    def _infer_n_points(self) -> int:
        """Infer the number of spatial points from grid_indices and positions."""
        if self._grid_indices is None and self._positions is None:
            return None
        
        if self._grid_indices is None:
            return len(self._positions)
        
        if self._positions is None:
            return len(self._grid_indices)
        
        self._validate_lengths()
        return len(self._grid_indices)

    def __repr__(self) -> str:
        if self._grid_indices is None:
            if self._positions is None:
                return f"SpatialProfile(grid_indices=None, positions=None)"
            else:
                return f"SpatialProfile(grid_indices=None, positions=(shape={self._shape}, ndim={self._ndim}))"
        else:
            if self._positions is None:
                return f"SpatialProfile(grid_indices=(shape={self._shape}, ndim={self._ndim}), positions=None)"
            else:
                return f"SpatialProfile(grid_indices=(shape={self._shape}, ndim={self._grid_indices.shape[1]}), positions=(shape={self._shape}, ndim={self._positions.shape[1]}))"

    def __len__(self) -> int:
        """Return the number of spectra (i.e. number of grid points or positions)."""
        return self._n_points

    @property
    def ndim(self) -> int:
        """Return the number of spatial dimensions (e.g. 1, 2, or 3)."""
        return self._ndim

    @property
    def shape(self) -> Tuple[int]:
        """Return the inferred grid shape if structured, otherwise None."""
        return self._shape

    @property
    def n_points(self) -> int:
        """Return the number of spectra (i.e. number of grid points or positions)."""
        return self._n_points

    @property
    def profile_type(self) -> str:
        """Return the profile type ('point', 'line', 'map', 'volume', or 'unstructured')."""
        return self._profile_type

    @property
    def is_structured(self) -> bool:
        """Return True if indices form a regular structured grid."""
        return self._shape is not None

    @property
    def has_positions(self) -> bool:
        """Return True if physical spatial positions are defined."""
        return self._positions is not None

    @property
    def has_grid_indices(self) -> bool:
        """Return True if grid indices are defined."""
        return self._grid_indices is not None

    @property
    def grid_indices(self) -> np.ndarray:
        """Return the grid indices associated with the profile."""
        if self._grid_indices is None:
            return None
        return self._grid_indices.copy()

    @property
    def positions(self) -> np.ndarray:
        """Return the physical positions associated with the profile."""
        if self._positions is None:
            return None
        return self._positions.copy()

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the minimum and maximum coordinate bounds if positions are defined."""
        if self._positions is None:
            return None
        return np.min(self._positions, axis=0), np.max(self._positions, axis=0)
    
    def get_grid_index(self, index: int, default: Any = None) -> np.ndarray:
        """Return the grid index for the given spectrum index."""
        if self._grid_indices is None or index not in range(self._n_points):
            return default
        return self._grid_indices[index]
    
    def get_position(self, index: int, default: Any = None) -> np.ndarray:
        """Return the position for the given spectrum index."""
        if self._positions is None or index not in range(self._n_points):
            return default
        return self._positions[index]
    
    def get_neighbours(
            self,
            index: int,
            use_positions: bool = False,
            mode: Literal['grid', 'full'] = None,
            k: int = None,
    ) -> List[int]:
        """Return indices of neighbouring spectra based on grid or spatial proximity."""
        if self._n_points == 1:
            return None
        
        if index < 0 or index >= self._n_points - 1:
            raise IndexError(
                f"Invalid index: index must be within the range of profile indices. "
                f"Got index={index}, valid range is [0, {self._n_points - 1}]."
            )
        
        if use_positions and mode is not None:
            raise ValueError(
                f"Invalid mode: mode only used when use_positions is False. "
                f"Got use_positions={use_positions} and mode={mode}"
            )
        
        if not use_positions and k is not None:
            raise ValueError(
                f"Invalid k: k only used when use_positions is True. "
                f"Got use_positions={use_positions} and k={k}"
            )
                       
        if use_positions:
            position: np.ndarray = self._positions[index]
            k: int = k or 4
            return self._get_neighbours_by_position(position, k=k)
        else:
            grid_index: np.ndarray = self._grid_indices[index]
            mode: str = mode or 'grid'
            return self._get_neighbours_by_grid_index(grid_index, mode)
        
    def _get_neighbours_by_grid_index(
            self,
            grid_index: np.ndarray,
            mode: Literal['grid', 'full'] = 'grid'
    ) -> np.ndarray:
        """Neighbour detection for regular grid indices."""
        VALID_MODES: List[str] = ['grid', 'full']
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode value: mode must be in {VALID_MODES}. "
                f"Got mode={mode}"
            )
        
        if grid_index not in self._grid_indices:
            raise ValueError(
                f"Invalid grid_index: grid_index must be in grid_indices. "
                f"Got grid_index={grid_index} and grid_indices={self._grid_indices}."
            )
        
        if self._n_points == 1:
            return None
        
        deltas: np.ndarray = self._connectivity_deltas(mode)
        potential: List[np.ndarray] = [grid_index + d for d in deltas]

        # map existing grid indices to their spectral indices for fast lookup
        grid_to_idx: Dict[Tuple[int], int] = {tuple(gi): i for i, gi in enumerate(self._grid_indices)}

        # collect spectral indices of existing neighbours
        neighbour_indices: List[Tuple[int]] = [grid_to_idx[tuple(p)] for p in potential if tuple(p) in grid_to_idx]

        return np.array(sorted(neighbour_indices), dtype=int)
    
    def _get_neighbours_by_position(
        self,
        position: np.ndarray,
        k: int = 4,
        ) -> np.ndarray:
        """Neighbour detection using k-nearest neighbours and Euclidean distance between spatial positions."""
        if self._positions is None:
            raise ValueError(
                f"Invalid use: positions must be defined to get neighbours by position. "
                f"Got positions=None."
            )
        if self._n_points == 1:
            return None
        
        if not (1 <= k <= self._n_points):
            raise ValueError(
                f"Invalid k: k must be between 1 and number of points. "
                f"Got k={k}, and n_points={self._n_points}."
            )
        
        # Compute Euclidean distances
        distances: np.ndarray = np.linalg.norm(self._positions - position, axis=1)

        # Get indices of k nearest neighbours (excluding the point itself)
        sorted_indices: np.ndarray = np.argsort(distances)
        neighbour_indices: np.ndarray = sorted_indices[1 : k + 1]  # exclude self (distance=0)
        return np.sort(neighbour_indices)
        
    def _connectivity_deltas(
            self,
            mode: Literal['grid', 'full']
        ) -> np.ndarray:
        """Return offset patterns for supported connectivity modes."""
        VALID_MODES: List[str] = ['grid', 'full']
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode value: mode must be in {VALID_MODES}. "
                f"Got mode={mode}"
            )
        
        # 1d profile
        if self.ndim == 1:
            return np.array([[-1], [1]])

        # 2d profile
        if self.ndim == 2:
            if mode == 'full':
                return np.array([
                    [-1, -1], [-1, 0], [-1, 1],
                    [0, -1],            [0, 1],
                    [1, -1],  [1, 0],   [1, 1],
                ])
            if mode == 'grid':
                return np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

        # 3d profile
        if self.ndim == 3:
            if mode == 'full':
                return np.array([
                    [x, y, z]
                    for x in [-1, 0, 1]
                    for y in [-1, 0, 1]
                    for z in [-1, 0, 1]
                    if not (x == y == z == 0)
                ])
            if mode == 'grid':
                return np.array([
                    [-1, 0, 0], [1, 0, 0],
                    [0, -1, 0], [0, 1, 0],
                    [0, 0, -1], [0, 0, 1],
                ])

    def summary(self) -> str:
        """Return a human-readable summary of the spatial profile."""
        pass