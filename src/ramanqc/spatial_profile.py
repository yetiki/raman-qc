"""
Author: Yoshiki Cook
Date: 2025-10-20

Updated: 2025-10-22
"""

from typing import Optional, Literal, Union, Self, Any,List, Dict, Tuple
import numpy as np

class SpatialProfile():
    """
    Represents the spatial structure of a single Raman spectral measurement.

    The SpatialProfile class defines how individual spectra are spatially arranged,
    based on logical grid indices and, optionally, physical positions (e.g., in µm).
    The profile shape and type is inferred automatically, using the provided grid indices
    and provides spatial utility methods such as neighbour lookup.
    
    Parameters
    ----------
        
    Attributes
    ----------

    Methods
    ----------
    """
    def __init__(
            self,
            grid_indices: List[Tuple[int, ...]],
            positions: Optional[List[Tuple[float, ...]]] = None,
            validate_positions: bool = True) -> None:
        
        self._grid_indices: List[Tuple[int, ...]] = self._set_grid_indices(grid_indices)
        self._shape: Tuple[int, ...] = self._infer_shape(grid_indices)
        self._profile_type: str = self._infer_profile_type()
        self._positions: List[Tuple[float, ...]] = self._set_positions(positions, validate_positions)

    def __repr__(self) -> str:
        if self._positions is not None:
            return f"Profile(grid_indices=(shape{self._shape}, ndim={self.ndim}), positions=(shape{self._shape}, ndim={self.ndim}))"
        else:
            return f"Profile(grid_indices=(shape{self._shape}, ndim={self.ndim}), positions=None)"
    
    def __len__(self) -> int:
        return len(self._grid_indices)

    @property
    def grid_indices(self) -> List[Tuple[int, ...]]:
        return self._grid_indices

    @property
    def positions(self) -> List[Tuple[float, ...]]:
        return self._positions

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the spatial grid if structured."""
        return self._shape

    @property
    def profile_type(self) -> Literal['point', 'line', 'map', 'volume', 'unstructured']:
        """Return the inferred profile type: 'point', 'line', 'map', 'volume', or 'unstructured'."""
        return self._profile_type

    @property
    def n_points(self) -> int:
        """Return the number of grid / position points in the profile."""
        return sum(self.shape)
    
    @property
    def ndim(self) -> int:
        """Return the number of spatial dimensions of each point in the profile."""
        grid_idc = np.asarray(self.grid_indices, dtype=int)
        if grid_idc.size == 0:
            return 0
        return grid_idc.shape[1]

    def _set_grid_indices(self, grid_indices: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
        """Validate and normalize the provided grid indices."""
        grid_idc: np.ndarray = np.asarray(grid_indices, dtype=int)

        if grid_idc.ndim != 2:
            raise ValueError(
                f"Invalid grid_indices shape: grid_indices must be a sequence of fixed-length index tuples (2D after conversion). "
                f"Got {grid_idc.ndim}D array after conversion."
            )

        if grid_idc.shape[1] > 3:
            raise ValueError(
                f"Invalid grid_indices shape: each tuple in grid_indices must not exceed three spatial coordinates. "
                f"Got ndim={grid_idc.shape[1]}."
            )
        
        if len(np.unique(grid_idc, axis=0)) != len(grid_idc):
            raise ValueError(
                f"Invalid grid_indices: each index in grid_indices must be unique. "
                f"Got {len(grid_idc) - len(np.unique(grid_idc, axis=0))} duplicate grid_indices."
            )

        # normalize to list of tuples of ints
        return [tuple(map(int, i)) for i in grid_idc.tolist()]

    def _set_positions(self, positions: List[Tuple[float, ...]], validate: bool) -> List[Tuple[float]]:
        """Validate and normalize the provided positions."""
        if positions is None:
            return positions
        
        pos: np.ndarray = np.asarray(positions, dtype=float)
        grid_idc: np.ndarray = np.asarray(self._grid_indices, dtype=float)

        if pos.ndim != 2:
            raise ValueError(
                f"Invalid positions shape: positions must be a sequence of of fixed-length index tuples (2D after conversion). "
                f"Got {pos.ndim}D array after conversion."
            )

        if pos.shape[1] > 3:
            raise ValueError(
                f"Invalid positions shape: each tuple in positions must not exceed three spatial coordinates. "
                f"Got ndim={pos.shape[1]}."
            )
        
        if pos.shape[1] != grid_idc.shape[1]:
            raise ValueError(
                f"Invalid positions shape: positions must have same number of spatial coordinates as grid_indices. "
                f"Got ndim={pos.shape[1]} for positions, and ndim={grid_idc.shape[1]} for grid_indices."
            )

        if len(pos) != len(grid_idc):
            raise ValueError(
                f"Invalid positions length: positions must have same length as grid_indices. "
                f"Got len(positions)={len(pos)}, len(grid_indices)={len(grid_idc)}."
            )
        
        if len(np.unique(pos, axis=0)) != len(pos):
            raise ValueError(
                f"Invalid positions: each position in positions must be unique. "
                f"Got {len(pos) - len(np.unique(pos, axis=0))} duplicate positions."
            )

        if not validate:
            # normalize to list of tuples of floats
            return [tuple(map(float, p)) for p in pos.tolist()]

        pos_shape: Tuple[int, ...] = self._infer_shape(positions)

        if pos_shape != self._shape:
            raise ValueError(
                f"Invalid shape: inferred positions shape does not match inferred grid_indices shape. "
                f"Got shape={pos_shape} from positions, and shape={self._shape} from grid_indices."
            )

        # normalize to list of tuples of floats
        return [tuple(map(float, p)) for p in pos.tolist()]

    def _infer_shape(self, locations: List[Tuple[Union[int, float], ...]]) -> Union[Tuple[int, ...], None]:
        """Infer the spatial shape from the provided locations (i.e. grid_indices or positions)."""
        lcn: np.ndarray = np.asarray(locations, dtype=float)
        if lcn.size == 0:
            return (0,)

        n_lcn: int = len(lcn)
        ndim: int = lcn.shape[1]

        if n_lcn == 1:
            return (1,)

        # Count unique coordinate values along each axis
        unique_counts: List[int] = [len(np.unique(lcn[:, i])) for i in range(ndim)]

        # Expected total points if it's a full grid
        expected_total: int = int(np.prod(unique_counts))

        # If it's a perfect grid return all axis sizes, otherwise if consecutive treat as a line (list of points), else None
        consecutive: bool = self._check_consecutive_grid_indices()
        if n_lcn == expected_total:
            shape: Tuple[int, ...] = tuple(unique_counts)
        elif consecutive:
            shape: Tuple[int, ...] = (n_lcn,)
        else:
            shape: None = None
        return shape

    def _infer_profile_type(self) -> str:
        """Infer the profile type from the grid_indices."""
        if self.n_points == 1:
            return 'point'
        
        consecutive: bool = self._check_consecutive_grid_indices()
        if not consecutive:
            return 'unstructured'
        
        true_ndim: int = sum(s != 1 for s in self.shape)

        if true_ndim == 1:
            return 'line'
        elif true_ndim == 2:
            return 'map'
        elif true_ndim == 3:
            return 'volume'
        else:
            return 'unstructured'

    def _check_consecutive_grid_indices(self) -> bool:
        """Check whether integer grid_indices form a consecutive grid (no missing coordinates)."""
        grid_idc: np.ndarray = np.asarray(self._grid_indices, dtype=int)
        if grid_idc.size == 0:
            return True

        ndim: int = grid_idc.shape[1]

        # For each axis compute min..max range and build expected grid
        ranges: List = [np.arange(grid_idc[:, i].min(), grid_idc[:, i].max() + 1) for i in range(ndim)]
        expected: np.ndarray = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(ndim, -1).T
        return set(map(tuple, expected)) == set(map(tuple, grid_idc))

    def is_structured(self) -> bool:
        """Check if the profile is structured (regular grid)."""
        return self._shape is not None
    
    # def get_spec_index_by_position(self, position: Tuple[float, ...]) -> Tuple[int, ...]:
    #     """Return the index of a spectrum in measurement.spectra for the given spatial position."""
    #     if self._positions is None:
    #         raise ValueError(
    #             f"Invalid operation: Positions must be defined to get index by position. "
    #             f"Got positions=None."
    #         )
        
    #     for i, pos in enumerate(self._positions):
    #         if all(np.isclose(np.array(pos), np.array(position))):
    #             return i
    #     raise ValueError(
    #         f"Position not found: No spectrum found at the specified position. "
    #         f"Got position={position}."
    #     )
    
    # def get_spec_index_by_grid_index(self, grid_index: Tuple[int, ...]) -> Tuple[int, ...]:
    #     """Return the index of a spectrum in measurement.spectra for the given grid index."""
    #     for i, _g_idx in enumerate(self._grid_indices):
    #         if _g_idx == grid_index:
    #             return i
    #     raise ValueError(
    #         f"Grid index not found: No spectrum found at the specified grid index. "
    #         f"Got index={grid_index}."
    #     )
    
    # def get_grid_index(self, position: Tuple[float, ...]) -> Tuple[int, ...]:
    #     """Return the grid index nearest to the given spatial position."""
    #     if self._positions is None:
    #         raise ValueError(
    #             f"Invalid operation: Positions must be defined to get grid index by position. "
    #             f"Got positions=None."
    #         )
    #     for i, p in enumerate(np.asarray(self._positions, dtype=float)):
    #         if all(np.isclose(p, position)):
    #             return self._grid_indices[i]
            
    #     raise ValueError(
    #         f"Position not found: No spectrum found at the specified position. "
    #         f"Got position={position}."
    #     )
    
    # def get_position(self, grid_index: Tuple[int, ...]) -> Tuple[float, ...]:
    #     """Return the spatial position for the given grid index."""
    #     for i, g_idx in enumerate(self._grid_indices):
    #         if g_idx == grid_index:
    #             return self._positions[i]
            
    #     raise ValueError(
    #         f"Grid index not found: No spectrum found at the specified grid index. "
    #         f"Got index={grid_index}."
    #     )

    def get_neighbours(
        self,
        spec_index: int,
        mode: Literal[
            '2-connectivity', '4-connectivity', '6-connectivity',
            '8-connectivity', '26-connectivity', 'kNN'
        ] = '4-connectivity',
        use_positions: bool = False,
        k: int = 4
    ) -> List[Tuple[int, ...]]:
        """
        Return neighbouring spectra indices based on grid or spatial proximity.

        Parameters
        ----------
        spec_index : int
            The index of the spectrum within the profile to find neighbours for.

        mode : {'2-connectivity', '4-connectivity', '6-connectivity', '8-connectivity', '26-connectivity', 'kNN'}, default='4-connectivity'
            Connectivity mode for neighbour lookup:
            - '2-connectivity' : For 1D line profiles (left/right)
            - '4-connectivity' : Up/down/left/right (2D)
            - '8-connectivity' : Includes diagonals (2D)
            - '6-connectivity' : Faces only (3D)
            - '26-connectivity': Full 3×3×3 cube excluding self (3D)
            - 'kNN'            : k nearest neighbours (requires positions)

        use_positions : bool, default=False
            Whether to compute neighbours based on spatial coordinates
            rather than grid indices.

        k : int, default=4
            Number of neighbours to return for kNN mode.

        Returns
        -------
        List[Tuple[int, ...]]
            List of neighbouring indices within the profile.

        Raises
        ------
        ValueError
            If mode is invalid for the profile dimensionality or structure.
        """
        if self._profile_type == 'single':
            raise ValueError(
                f"Invalid profile_type:Neighbour lookup not applicable for single-point profiles."
                f"Got profile_type='{self._profile_type}'.")
        
        if spec_index < 0 or spec_index >= len(self._grid_indices):
            raise IndexError(
                f"Invalid spec_index: spec_index must be within the range of profile indices. "
                f"Got spec_index={spec_index}, valid range is [0, {len(self._grid_indices) - 1}]."
            )

        if use_positions:
            position: Tuple[float, ...] = self._positions[spec_index]
            return self._get_neighbours_by_position(position, mode, k=k)
        else:
            grid_index: Tuple[int, ...] = self._grid_indices[spec_index]
            return self._get_neighbours_by_grid_index(grid_index, mode)

    def _get_neighbours_by_grid_index(self, grid_index: Tuple[int, ...], mode: Literal['2-connectivity', '4-connectivity', '6-connectivity', '8-connectivity', '26-connectivity', 'kNN']) -> List[Tuple[int, ...]]:
        """Detect neighbouring indices for a regular grid."""
        grid_idc: np.ndarray = np.asarray(grid_index, dtype=int)

        if len(grid_idc) != self.ndim:
            raise ValueError(
                f"Index dimensionality {len(grid_idc)} does not match profile ({self.ndim})."
            )
        if self.profile_type == 'unstructured':
            raise ValueError(
                f"Invalid mode: connectivity-based neighbour lookup by index requires a structured profile. "
                f"Use 'kNN' with positions instead for unstructured profiles. "
                f"Got mode={mode} for profile_type={self.profile_type}."
            )

        # Validate mode for dimensionality
        valid_modes = {
            1: ('2-connectivity',),
            2: ('4-connectivity', '8-connectivity'),
            3: ('6-connectivity', '26-connectivity'),
        }
        if mode not in valid_modes.get(self.ndim, ()):
            raise ValueError(
                f"Invalid mode: mode must be valid for profile dimensionality. "
                f"Use {valid_modes.get(self.ndim, [])} for {self.ndim}D profiles. "
                f"Got mode='{mode}' for ndim={self.ndim}."
                )

        deltas: np.ndarray = self._connectivity_deltas(mode)
        potential: List[Tuple[int, ...]] = [tuple(grid_idc + d) for d in deltas]
        existing: set = set(map(tuple, self.grid_indices))
        return [p for p in potential if p in existing]

    def _get_neighbours_by_position(self, position: Tuple[float, ...], mode: Literal['2-connectivity', '4-connectivity', '6-connectivity', '8-connectivity', '26-connectivity', 'kNN'], k: int = 4) -> List[Tuple[int, ...]]:
        """Neighbour detection using Euclidean distance between spatial positions."""
        if self.positions is None:
            raise ValueError("Positions must be defined for position-based neighbour lookup.")

        pos: np.ndarray = np.array(position, dtype=float)

        if mode == 'kNN':
            dists = np.linalg.norm(self.positions - pos, axis=1)
            nearest_grid_idc = np.argsort(dists)
            nearest_grid_idc = nearest_grid_idc[1:k+1]  # exclude self
            return [tuple(self.grid_indices[i]) for i in nearest_grid_idc]

        # All other connectivity modes require a structured grid
        if self.profile_type == 'unstructured':
            raise ValueError(f"Mode '{mode}' invalid for unstructured profiles (use 'kNN').")

        # Validate mode for dimensionality
        valid_modes = {
            1: ('2-connectivity',),
            2: ('4-connectivity', '8-connectivity'),
            3: ('6-connectivity', '26-connectivity'),
        }
        if mode not in valid_modes.get(self.ndim, ()):
            raise ValueError(f"Mode '{mode}' invalid for {self.ndim}D profile.")

        # Find nearest grid index and delegate to index-based lookup
        nearest_index = self.get_index(pos)
        return self._get_neighbours_by_grid_index(nearest_index, mode)

    def _connectivity_deltas(self, mode: Literal['2-connectivity', '4-connectivity', '6-connectivity', '8-connectivity', '26-connectivity']) -> np.ndarray:
        """Return offset patterns for supported connectivity modes."""
        if self.ndim == 1:
            if mode != '2-connectivity':
                raise ValueError(f"Mode '{mode}' invalid for 1D profile.")
            return np.array([(-1,), (1,)])

        if self.ndim == 2:
            if mode == '8-connectivity':
                return np.array([
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1),            (0, 1),
                    (1, -1),  (1, 0),   (1, 1),
                ])
            if mode == '4-connectivity':
                return np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])

        if self.ndim == 3:
            if mode == '26-connectivity':
                return np.array([
                    (x, y, z)
                    for x in (-1, 0, 1)
                    for y in (-1, 0, 1)
                    for z in (-1, 0, 1)
                    if not (x == y == z == 0)
                ])
            if mode == '6-connectivity':
                return np.array([
                    (-1, 0, 0), (1, 0, 0),
                    (0, -1, 0), (0, 1, 0),
                    (0, 0, -1), (0, 0, 1),
                ])

        raise ValueError(f"Unsupported connectivity mode '{mode}' for ndim={self.ndim}.")