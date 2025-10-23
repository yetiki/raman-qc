"""
Author: Yoshiki Cook
Date: 2025-10-20

Updated: 2025-10-23
"""

from typing import Optional, Literal, Union, Self, Any, List, Dict, Tuple, Sequence
import numpy as np

from typing import List, Tuple, Optional, Literal, Union

class SpatialProfile:
    """
    Represents the spatial organisation of spectra within a measurement.
    
    The SpatialProfile class stores and manages the spatial structure
    (grid indices and/or physical positions) associated with each spectrum.
    It can infer shape and dimensionality, validate consistency between
    indices and positions, and provide neighbour relationships.

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
            self._positions: np.ndarray = np.asarray(positions, dtype=int)
            self._validate_array(self._positions, name='positions')
        else:
            self._positions: None = None

        # ensure at least one of grid_indices or positions is defined
        self._validate_inputs()

        # ensure lengths match if both grid_indices and positions are defined
        self._validate_lengths()

        # infer grid_indices from positions if not defined
        if self._grid_indices is None:
            inferred_indices: np.ndarray = self._infer_indices_from_positions()

            if inferred_indices is not None:
                self._grid_indices: np.ndarray = inferred_indices

        # ensure grid_indices structure and positions structure match
        if self._grid_indices is not None and self._positions is not None:
            self._validate_positions_against_indices()

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
        if self._grid_indices is None:
            self._profile_type: str = 'unstructured'
        else:
            self._profile_type: str = self._infer_profile_type()

        self._size: int = sum(self._shape)

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

    def _infer_indices_from_positions(self, tolerance: float = 1e-6) -> np.ndarray:
        """Infer integer grid indices from approximately regular positions."""
        # Round positions to nearest tolerance
        rounded: np.ndarray = np.round(self._positions / tolerance).astype(int)

        # Reconstruct relative integer grid coordinates
        inverse: np.ndarray
        _, inverse = np.unique(rounded, axis=0, return_inverse=True)
        indices: np.ndarray = inverse.reshape(-1, 1) if rounded.ndim == 1 else rounded
        return indices
    
    def _validate_positions_against_indices(self) -> None:
        """Ensure positions and grid_indices correspond to the same structure."""
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
    
    def _infer_shape_from_indices(self) -> Union[Tuple[int, ...], None]:
        """Infer the overall grid shape from grid indices."""
        if self._grid_indices is None:
            return None
        
        n_grid_indices: int = len(self._grid_indices)
        ndim: int = self._grid_indices.shape[1]

        # Count unique coordinate values along each axis
        unique_counts: List[int] = [len(np.unique(self._grid_indices[:, i])) for i in range(ndim)]

        # Expected total points if it's a full grid
        expected_total: int = int(np.prod(unique_counts))

        # If it's a perfect grid return all axis sizes, otherwise if consecutive treat as a line (list of points), else None
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

    def __repr__(self) -> str:
        if self._grid_indices is None:
            if self._positions is None:
                return f"SpatialProfile(grid_indices=None, positions=None)"
            else:
                return f"SpatialProfile(grid_indices=None, positions=(shape={...}, ndim={...}))"
        else:
            if self._positions is None:
                return f"SpatialProfile(grid_indices=(shape={...}, ndim={...}), positions=None)"
            else:
                return f"SpatialProfile(grid_indices=(shape={...}, ndim={...}), positions=(shape={...}, ndim={...}))"

    def __len__(self) -> int:
        """Return the number of spectra (i.e. number of grid points or positions)."""
        return self._size

    @property
    def ndim(self) -> int:
        """Return the number of spatial dimensions (e.g. 1, 2, or 3)."""
        return self._ndim

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Return the inferred grid shape if structured, otherwise None."""
        return self._shape

    @property
    def size(self) -> int:
        """Return the number of spectra (i.e. number of grid points or positions)."""
        return self._size

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
    def missing_indices(self) -> Optional[List[Tuple[int, ...]]]:
        """Return a list of missing or non-consecutive grid indices, if any."""
        pass

    @property
    def bounds(self) -> Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        """Return the minimum and maximum coordinate bounds if positions are defined."""
        if self._positions is None:
            return None
        return np.min(self._positions, axis=0), np.max(self._positions, axis=0)

    def is_structured(self) -> bool:
        """Check if the profile is structured (regular grid)."""
        return self._shape is not None

    def get_neighbours(
        self,
        index: int,
        mode: Literal['2-connectivity', '4-connectivity', '6-connectivity', '8-connectivity', '26-connectivity', 'kNN'] = '4-connectivity',
        use_positions: bool = False,
        k: int = 4,
    ) -> List[Tuple[int, ...]]:
        """Return neighbouring spectra indices based on grid or spatial proximity."""
        if self._size == 1:
            raise ValueError(
                f"Invalid profile_type:Neighbour lookup not applicable for single-point profiles."
                f"Got profile_type='{self._profile_type}'."
                )
        
        if index < 0 or index >= len(self._grid_indices):
            raise IndexError(
                f"Invalid index: index must be within the range of profile indices. "
                f"Got index={index}, valid range is [0, {self._size - 1}]."
            )
        
        if use_positions:
            position: Tuple[float, ...] = self._positions[index]
            return self._get_neighbours_by_position(position, mode, k=k)
        else:
            grid_index: Tuple[int, ...] = self._grid_indices[index]
            return self._get_neighbours_by_grid_index(grid_index, mode)
        
    def _get_neighbours_by_grid_index(
        self,
        grid_index: Tuple[int, ...],
        mode: Literal['2-connectivity', '4-connectivity', '6-connectivity', '8-connectivity', '26-connectivity'],
    ) -> List[Tuple[int, ...]]:
        """Neighbour detection for regular grid indices."""
        pass

    def _get_neighbours_by_position(
        self,
        position: Tuple[float, ...],
        mode: Literal['2-connectivity', '4-connectivity', '6-connectivity', '8-connectivity', '26-connectivity', 'kNN'],
        k: int = 4,
    ) -> List[Tuple[int, ...]]:
        """Neighbour detection using Euclidean distance between spatial positions."""
        pass

    def summary(self) -> str:
        """Return a human-readable summary of the spatial profile."""
        pass


