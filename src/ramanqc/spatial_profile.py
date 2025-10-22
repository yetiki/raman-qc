"""
Author: Yoshiki Cook
Date: 2025-10-20
"""

from typing import Optional, Literal, Union, Self, Any,List, Dict, Tuple
import numpy as np

class SpatialProfile():
    """
    Represents the spatial structure of a Raman spectral measurement.

    The SpatialProfile class defines how individual spectra are spatially arranged,
    based on logical indexes (e.g., grid positions) and, optionally, physical
    positions (e.g., in µm). The profile shape and type is inferred automatically,
    using the provided indexes and provides spatial utility methods such as neighbour lookup.
    
    Parameters
    ----------
        
    Attributes
    ----------

    Methods
    ----------
    """
    def __init__(self, indexes: List[Tuple[int, ...]], positions: Optional[List[Tuple[float, ...]]] = None, validate_positions: bool = True) -> None:
        self._indexes: List[Tuple[int, ...]] = self._set_indexes(indexes)
        self._shape: Tuple[int, ...] = self._infer_shape(indexes)
        self._profile_type: str = self._infer_profile_type_from_indexes()
        self._positions: List[Tuple[float, ...]] = self._set_positions(positions, validate_positions)

    def __repr__(self) -> str:
        if self._positions is not None:
            return f"Profile(indexes=(shape{self._shape}, ndim={self.ndim}), positions=(shape{self._shape}, ndim={self.ndim}))"
        else:
            return f"Profile(indexes=(shape{self._shape}, ndim={self.ndim}), positions=None)"
    
    def __len__(self) -> int:
        return self.n_indexes

    @property
    def indexes(self) -> List[Tuple[int, ...]]:
        return self._indexes

    @property
    def positions(self) -> List[Tuple[float, ...]]:
        return self._positions

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the spatial grid if structured."""
        return self._shape

    @property
    def profile_type(self) -> str:
        """Return the inferred profile type: 'point', 'line', 'map', 'volume', or 'unstructured'."""
        return self._profile_type

    @property
    def n_indexes(self) -> int:
        """Return the number of spatial indexes in the profile."""
        return len(self._indexes)

    @property
    def ndim(self) -> int:
        """Return the number of spatial dimensions in the profile."""
        idx = np.asarray(self.indexes, dtype=int)
        if idx.size == 0:
            return 0
        return idx.shape[1]

    def _set_indexes(self, indexes: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
        idx: np.ndarray = np.asarray(indexes, dtype=int)

        if idx.ndim != 2:
            raise ValueError(
                f"Invalid indexes shape: indexes must be a sequence of fixed-length index tuples (2D after conversion). "
                f"Got ndim={idx.ndim}."
            )

        if idx.shape[1] > 3:
            raise ValueError(
                f"Invalid indexes shape: each tuple in indexes must not exceed three spatial coordinates. "
                f"Got n_coordinates={idx.shape[1]}."
            )

        # normalize to list of tuples of ints
        return [tuple(map(int, i)) for i in idx.tolist()]

    def _set_positions(self, positions: List[Tuple[float, ...]], validate: bool) -> List[Tuple[float]]:
        if positions is None:
            return positions
        
        pos: np.ndarray = np.asarray(positions, dtype=float)
        idx: np.ndarray = np.asarray(self._indexes, dtype=float)

        if pos.ndim != 2:
            raise ValueError(
                f"Invalid positions shape: positions must be a sequence of spatial coordinate tuples. "
                f"Got ndim={pos.ndim}."
            )

        if pos.shape[1] > 3:
            raise ValueError(
                f"Invalid positions shape: each tuple in positions must not exceed three spatial coordinates. "
                f"Got n_coordinates={pos.shape[1]}."
            )

        if len(pos) != len(idx):
            raise ValueError(
                f"Invalid positions length: positions must have same length as indexes. "
                f"Got len(positions)={len(pos)}, len(indexes)={len(idx)}."
            )

        if not validate:
            # normalize to list of tuples of floats
            return [tuple(map(float, p)) for p in pos.tolist()]

        pos_shape: Tuple[int, ...] = self._infer_shape(positions)

        if pos_shape != self._shape:
            raise ValueError(
                f"Invalid shape: inferred positions shape does not match inferred indexes shape. "
                f"Got shape={pos_shape} from positions, and shape={self._shape} from indexes."
            )

        # normalize to list of tuples of floats
        return [tuple(map(float, p)) for p in pos.tolist()]

    def _infer_shape(self, locations: List[Tuple[Union[int, float], ...]]) -> Tuple[int, ...]:
        """Infer the spatial shape from the provided locations (indexes or positions)."""
        lcn: np.ndarray = np.asarray(locations, dtype=float)
        if lcn.size == 0:
            return (0,)

        n_lcn: int = len(lcn)
        ndim: int = lcn.shape[1]

        # Count unique coordinate values along each axis
        unique_counts: List[int] = [len(np.unique(lcn[:, i])) for i in range(ndim)]

        # Expected total points if it's a full grid
        expected_total: int = int(np.prod(unique_counts))

        # If it's a perfect grid return all axis sizes, otherwise treat as a line (list of points)
        if n_lcn == expected_total:
            shape: Tuple[int, ...] = tuple(unique_counts)
        else:
            shape: Tuple[int, ...] = (n_lcn,)
        return shape

    def _infer_profile_type_from_indexes(self) -> str:
        """Infer the profile type from the indexes."""
        consecutive: bool = self._check_consecutive_indexes()

        if not consecutive:
            return 'unstructured'

        if self.ndim == 1:
            return 'line' if self.shape[0] > 1 else 'point'
        elif self.ndim == 2:
            return 'map'
        elif self.ndim == 3:
            return 'volume'
        else:
            return 'unstructured'

    def _check_consecutive_indexes(self) -> bool:
        """Check whether integer indexes form a consecutive grid (no missing coordinates)."""
        idx: np.ndarray = np.asarray(self._indexes, dtype=int)
        if idx.size == 0:
            return True

        ndim: int = idx.shape[1]

        # For each axis compute min..max range and build expected grid
        ranges: List = [np.arange(idx[:, i].min(), idx[:, i].max() + 1) for i in range(ndim)]
        expected: np.ndarray = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(ndim, -1).T
        return set(map(tuple, expected)) == set(map(tuple, idx))

    def get_neighbours(
        self,
        location: Tuple[Union[int, float], ...],
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
        location : Tuple[int or float, ...]
            The target location. If `use_positions=False`, interpreted as a
            grid index (e.g., (1, 1)); if True, as a spatial coordinate
            (e.g., (10.2, 5.7)).

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
        if use_positions:
            return self._get_neighbours_by_position(location, mode, k=k)
        else:
            return self._get_neighbours_by_index(location, mode)

    def _get_neighbours_by_index(self, index: Tuple[int, ...], mode: Literal['2-connectivity', '4-connectivity', '6-connectivity', '8-connectivity', '26-connectivity', 'kNN']) -> List[Tuple[int, ...]]:
        """Detect neighbouring indices for regular grid indices."""
        idx: np.ndarray = np.asarray(index, dtype=int)

        if len(idx) != self.ndim:
            raise ValueError(
                f"Index dimensionality {len(idx)} does not match profile ({self.ndim})."
            )
        if self.profile_type == "unstructured":
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
            raise ValueError(f"Mode '{mode}' invalid for {self.ndim}D profile.")

        deltas = self._connectivity_deltas(mode)
        potential = [tuple(idx + d) for d in deltas]
        existing = set(map(tuple, self.indexes))
        return [p for p in potential if p in existing]

    def _get_neighbours_by_position(self, position: Tuple[float, ...], mode: Literal['2-connectivity', '4-connectivity', '6-connectivity', '8-connectivity', '26-connectivity', 'kNN'], k: int = 4) -> List[Tuple[int, ...]]:
        """Neighbour detection using Euclidean distance between spatial positions."""
        if self.positions is None:
            raise ValueError("Positions must be defined for position-based neighbour lookup.")

        pos: np.ndarray = np.array(position, dtype=float)

        if mode == 'kNN':
            dists = np.linalg.norm(self.positions - pos, axis=1)
            nearest_idx = np.argsort(dists)
            nearest_idx = nearest_idx[1:k+1]  # exclude self
            return [tuple(self.indexes[i]) for i in nearest_idx]

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
        return self._get_neighbours_by_index(nearest_index, mode)

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