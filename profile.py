from typing import Optional, Union, Literal, List, Set, Tuple
import numpy as np

class Profile:
    _VALID_TYPES: Set[str] = {'point', 'line', 'map', 'volume', 'unstructured'}

    def __init__(self, positions: List[Tuple[Union[int, float], ...]], shape: Optional[Tuple[int, ...]] = None, profile_type: Optional[str] = None, infer_shape: bool = False, infer_profile_type: bool = False) -> None:
        self._positions: List[Tuple[Union[int, float], ...]] = self._set_positions(positions)
        self._shape: Tuple[int, ...] = self._set_shape(shape, infer_shape)
        self._profile_type: str = self._set_profile_type(profile_type, infer_profile_type=infer_profile_type)

    def __repr__(self) -> str:
        return f"Profile(positions=(n_positions={self.n_positions}), shape={self.shape}, profile_type='{self.profile_type}')"

    def __len__(self) -> int:
        return self.n_positions

    @property
    def positions(self) -> List[Tuple[Union[int, float], ...]]:
        return self._positions

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape
       
    @property
    def ndim(self) -> int:
        p: np.ndarray = np.asarray(self._positions)
        return p.shape[1] if p.ndim == 2 else 1

    @property
    def profile_type(self) -> str:
        return self._profile_type

    @property
    def n_positions(self) -> int:
        return len(self._positions)

    def _set_positions(self, positions: List[Tuple[Union[int, float], ...]]) -> List[Tuple[Union[int, float], ...]]:
        try:
            p: np.ndarray = np.asarray(positions).astype(float)
        except ValueError:
            raise ValueError(
                f"Invalid format: positions must be an array-like of tuples of int or float, with uniform tuple lengths. "
                f"Got positions={positions}."
            )
        return positions

    def _set_shape(self, shape: Tuple[int, ...], infer_shape: bool) -> Tuple[int, ...]:
        if infer_shape:
            return self._infer_shape()

        if shape is None:
            return shape       
        try:
            n_spectra: int = sum(shape)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid value: shape must be a tuple of int. A trailing comma is required for a single-element tuple. "
                f"Got shape={shape}."
            )

        if not self.n_positions == n_spectra:
            raise ValueError(
                f"Invalid value: shape must contain the same number of spectra as number of positions in positions. "
                f"Got n_spectra={n_spectra} from shape={shape}, and n_positions={self.n_positions} from positions."
            )
        return shape

    def _infer_shape(self) -> Tuple[int, ...]:
        ...

    def _set_profile_type(self, profile_type: str, infer_profile_type: bool) -> Union[None, str]:
        if profile_type is not None and profile_type not in self._VALID_TYPES:
            raise ValueError(
                f"Invalid value: profile_type must be in {self._VALID_TYPES}. "
                f"Got profile_type='{profile_type}'."
            )
        return self._infer_profile_type() if infer_profile_type else profile_type

    def _infer_profile_type(self) -> str:
        if self.n_positions == 1:
            return 'single'
        ...


import numpy as np
from typing import Optional, Union, Literal, List, Set, Tuple

def infer_profile_type(positions: List[Tuple[int, ...]]) -> str:
    n_positions: int = len(positions)
    pos: np.ndarray = np.asarray(positions)

    if n_positions < 2:
        return 'single'

    ndim: int = pos.shape[1]

    shifted_pos: np.ndarray = pos - pos[0]
    rank: int = np.linalg.matrix_rank(shifted_pos)

    # 1D, 2D, 3D geometry based on rank
    if rank == 1:
        return 'line'

    elif rank == 2 and ndim >= 2:
        # Check if grid-like: all unique x and y combinations appear
        xs: np.ndarray = np.unique(pos[:, 0])
        ys: np.ndarray = np.unique(pos[:, 1])
        grid_positions: Set[Tuple[int]] = {(x,y) for x in xs for y in ys}
        actual_positions: Set[Tuple[int]] = {tuple(p[:2]) for p in pos}

        if grid_positions == actual_positions:
            return 'map'
        else:
            return 'unstructured'

    elif rank == 3 and ndim == 3:
        # Check if full 3D grid
        xs: np.ndarray  = np.unique(pos[:, 0])
        ys: np.ndarray  = np.unique(pos[:, 1])
        zs: np.ndarray  = np.unique(pos[:, 2])
        grid_positions: Set[Tuple[int]] = {(x,y,z) for x in xs for y in ys for z in zs}
        actual_positions: Set[Tuple[int]] = {tuple(p) for p in pos}

        if grid_positions == actual_positions:
            return 'volume'
        else:
            return 'unstructured'
    else:
        return 'unstructured'


def infer_shape(positions: List[Tuple[int, ...]]) -> Tuple[int, ...]:
    n_positions: int = len(positions)
    pos: np.ndarray = np.asarray(positions)
    ndim: int = pos.shape[1]

    # Count unique coordinate values along each axis
    unique_counts: List[int] = [len(np.unique(pos[:, i])) for i in range(ndim)]

    # Identify non-constant axes
    non_constant_counts: List[int] = [c for c in unique_counts if c > 1]

    # Expected total points if it's a perfect grid
    expected_total: int = np.prod(unique_counts)

    # Check if all combinations are present (i.e. a full grid)
    if n_positions == expected_total:
        shape: Tuple[int, ...] = tuple(non_constant_counts) if non_constant_counts else (n_positions,)
    else:
        # Not a perfect grid, treat as a line
        shape: Tuple[int, ...] = (n_positions,)

    return shape


# positions: List[Tuple[int, ...]] = [(0,)]
# positions: List[Tuple[int, ...]] = [(0, 0)]
# positions: List[Tuple[int, ...]] = [(0, 0, 0)]
positions: List[Tuple[int, ...]] = [(0, 0), (0, 1)]
positions: List[Tuple[int, ...]] = [(0, 0), (0, 1), (0, 2)]
positions: List[Tuple[int, ...]] = [(0, 0), (0, 1), (0, 10)]
# positions: List[Tuple[int, ...]] = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (0, 2, 0), (1, 2, 0), (2, 2, 0),]
# positions: List[Tuple[int, ...]] = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (0, 2, 0), (1, 2, 0), (2, 2, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1), (0, 1, 1), (1, 1, 1), (2, 1, 1), (0, 2, 1), (1, 2, 1), (2, 2, 1),]

print(f"{positions=}")
print(f"{infer_profile_type(positions)=}")
print(f"{infer_shape(positions)=}")
































# class Profile:
#     _VALID_TYPES: Set[str] = {'point', 'line', 'map', 'volume', 'unstructured'}

#     def __init__(self, positions: List[Tuple[Union[int, float], ...]], shape: Optional[Tuple[int, ...]] = None, profile_type: Optional[str] = None, infer_shape: bool = False, infer_profile_type: bool = False) -> None:
#         self._positions: List[Tuple[Union[int, float], ...]] = self._set_positions(positions)
#         self._shape: Tuple[int, ...] = self._set_shape(shape, infer_shape)
#         self._profile_type: str = self._set_profile_type(profile_type, infer_profile_type=infer_profile_type)

#     def __repr__(self) -> str:
#         return f"Profile(positions=(n_positions={self.n_positions}), shape={self.shape}, profile_type='{self.profile_type}')"

#     def __len__(self) -> int:
#         return self.n_positions

#     @property
#     def positions(self) -> List[Tuple[Union[int, float], ...]]:
#         return self._positions

#     @property
#     def shape(self) -> Tuple[int, ...]:
#         return self._shape
       
#     @property
#     def ndim(self) -> int:
#         p: np.ndarray = np.asarray(self._positions)
#         return p.shape[1] if p.ndim == 2 else 1

#     @property
#     def profile_type(self) -> str:
#         return self._profile_type

#     @property
#     def n_positions(self) -> int:
#         return len(self._positions)

#     def _set_positions(self, positions: List[Tuple[Union[int, float], ...]]) -> List[Tuple[Union[int, float], ...]]:
#         try:
#             p: np.ndarray = np.asarray(positions).astype(float)
#         except ValueError:
#             raise ValueError(
#                 f"Invalid format: positions must be an array-like of tuples of int or float, with uniform tuple lengths. "
#                 f"Got positions={positions}."
#             )
#         return positions

#     def _set_shape(self, shape: Tuple[int, ...], infer_shape: bool) -> Tuple[int, ...]:
#         if infer_shape:
#             return self._infer_shape()

#         if shape is None:
#             return shape       
#         try:
#             n_spectra: int = sum(shape)
#         except (TypeError, ValueError):
#             raise ValueError(
#                 f"Invalid value: shape must be a tuple of int. A trailing comma is required for a single-element tuple. "
#                 f"Got shape={shape}."
#             )

#         if not self.n_positions == n_spectra:
#             raise ValueError(
#                 f"Invalid value: shape must contain the same number of spectra as number of positions in positions. "
#                 f"Got n_spectra={n_spectra} from shape={shape}, and n_positions={self.n_positions} from positions."
#             )
#         return shape

#     def _infer_shape(self) -> Tuple[int, ...]:
#         if self.n_positions == 1:
#             return tuple((1,))

#         p: np.ndarray = np.asarray(self.positions)

#         # Compute unique values per axis to estimate grid shape
#         unique_counts: List[int] = [len(np.unique(p[:, i])) for i in range(p.shape[1])]
        
#         return tuple(unique_counts)

#     def _set_profile_type(self, profile_type: str, infer_profile_type: bool) -> Union[None, str]:
#         if profile_type is not None and profile_type not in self._VALID_TYPES:
#             raise ValueError(
#                 f"Invalid value: profile_type must be in {self._VALID_TYPES}. "
#                 f"Got profile_type='{profile_type}'."
#             )
#         return self._infer_profile_type() if infer_profile_type else profile_type

#     def _infer_profile_type(self) -> str:
#         if self.n_positions == 1:
#             return 'single'
#         elif self.shape is None:
#             return 'unstructured'
        
#         shape_ndim: int = len(self.shape)

#         if shape_ndim == 1:
#             return 'line'
#         elif shape_ndim == 2:
#             return 'map'
#         elif shape_ndim == 3:
#             return 'volume'













































































# class Profile:
#     _VALID_TYPES: Set[str] = {'point', 'line', 'map', 'volume', 'unstructured'}

#     def __init__(self, positions: List[Tuple[Union[int, float], ...]], shape: Optional[Tuple[int, ...]] = None, profile_type: Optional[str] = None) -> None:
#         self.positions: List[Tuple[Union[int, float], ...]] = positions
#         self.shape: Tuple[int, ...] = shape
#         self.profile_type: str = profile_type

#     def __repr__(self) -> str:
#         return f"Profile(positions=(n_positions={self.n_positions}), shape={self.shape}, profile_type='{self.profile_type}')"

#     def __len__(self) -> int:
#         return self.n_positions

#     @property
#     def positions(self) -> List[Tuple[Union[int, float], ...]]:
#         return self._positions

#     @positions.setter
#     def positions(self, positions: List[Tuple[Union[int, float], ...]]) -> None:
#         try:
#             p: np.ndarray = np.asarray(positions).astype(float)
#         except ValueError:
#             raise ValueError(
#                 f"Invalid format: positions must be an array-like of tuples of int or float, with uniform tuple lengths. "
#                 f"Got positions={positions}."
#             )
#         self._positions: List[Tuple[Union[int, float], ...]] = positions

#     @property
#     def shape(self) -> Tuple[int, ...]:
#         return self._shape

#     @shape.setter
#     def shape(self, shape: Tuple[int, ...]) -> None:
#         try:
#             n_spectra: int = sum(shape)
#         except (TypeError, ValueError):
#             raise ValueError(
#                 f"Invalid value: shape must be a tuple of int. A trailing comma is required for a single-element tuple. "
#                 f"Got shape={shape}."
#             )

#         if shape is not None and not self.n_positions == n_spectra:
#             raise ValueError(
#                 f"Invalid value: shape must contain the same number of spectra as number of positions in positions. "
#                 f"Got n_spectra={n_spectra} from shape, and n_positions={self.n_positions} from positions."
#             )
#         self._shape: Tuple[int, ...] = shape

#     @property
#     def ndim(self) -> int:
#         p: np.ndarray = np.asarray(self.positions)
#         return p.shape[1] if p.ndim == 2 else 1

#     @property
#     def profile_type(self) -> str:
#         return self._profile_type

#     @profile_type.setter
#     def profile_type(self, profile_type: str) -> None:
#         if profile_type is not None and profile_type not in self._VALID_TYPES:
#             raise ValueError(
#                 f"Invalid value: profile_type must be in {self._VALID_TYPES}. "
#                 f"Got profile_type='{profile_type}'."
#             )
#         self._profile_type: str = profile_type or self._infer_profile_type()

#     @property
#     def n_positions(self) -> int:
#         return len(self.positions)

#     def _infer_profile_type(self) -> str:
#         if self.n_positions == 1:
#             return 'single'
#         elif self.shape is None:
#             return 'unstructured'
        
#         shape_ndim: int = len(self.shape)

#         if shape_ndim == 1:
#             return 'line'
#         elif shape_ndim == 2:
#             return 'map'
#         elif shape_ndim == 3:
#             return 'volume'

#     def _validate(self) -> None:
#         ...












































# class Profile:
#     """
#     Represents the spatial organisation of spectra within a single Raman Measurement.
#     Can describe 1D line scans, 2D lines/depth scans, 3D volume scans, or an aribitrary collection of points.
    
#     Attributes
#     ----------
#     type : {'point', 'line', 'map', 'volume'}, optional
#         The type of measurement profile.
#     positions : array-like, shape (n_spectra, n_dims), optional
#         Spatial coordinates for each spectrum (e.g. [x, y] positions).
#     shape : tuple[int], optional
#         Shape of the measurement grid (e.g. (rows, cols) for a map).

#     Notes
#     -----
#     - `positions` defines the coordinates of each spectrum.
#     - `shape` defines the grid arrangement, if applicable.
#     - Both must be consistent in length (len(positions) == np.prod(shape)).
#     """

#     VALID_TYPES: Set[str] = {'point', 'line', 'map', 'volume'}

#     def __init__(self, profile_type: Optional[str] = None, positions: Optional[np.ndarray] = None, shape: Optional[Tuple[int, ...]] = None) -> None:
#         ...
#         # TODO: Select appropriate profile constructor for measurement class use

#         # if profile_type is not None and profile_type not in self.VALID_TYPES:
#         #     raise ValueError(
#         #         f"Invalid profile_type: profile_type must be in {self.VALID_TYPES}. "
#         #         f"Got profile_type={profile_type}."
#         #     )
#         # self.profile_type: str = profile_type

#         # self._positions: np.ndarray = positions
#         # self._shape: Tuple[int, ...] = shape #TODO: check n_spectra in shape matches n_spectr in positions

#     def __repr__(self) -> str:
#         return f"Profile(type={self.profile_type}, positions={None if self._positions is None else len(self._positions)}, shape={self._shape})"

#     def __len__(self) -> int:
#         return 0 if self._positions is None else len(self.positions)
    
#     @property
#     def positions(self) -> np.ndarray:
#         return self._positions
    
#     @property
#     def shape(self) -> np.shape:
#         return self._shape
    
#     @property
#     def type(self) -> str:
#         return self.profile_type

# ---------------------------------------------------------------------------
# Data representations



# class Profile:
#     """
#     Represents the spatial organisation of a measurement, describing how individual
#     spectra are positioned relative to each other.

#     Parameters
#     ----------
#     positions : list of tuple of int or float
#         Spatial coordinates or grid indices of each spectrum.
#     shape : tuple of int, optional
#         Shape of the spatial grid (e.g. (rows, cols) for a 2D map).
#         If not provided, inferred automatically.
#     profile_type : {'point', 'line', 'map', 'volume', 'unstructured'}, optional
#         Type of spatial profile. If not provided, inferred automatically.
#     validate : bool, default=True
#         Whether to validate consistency between `positions`, `shape`, and `profile_type`.

#     Attributes
#     ----------
#     positions : np.ndarray
#         Array of spatial positions with shape (n_spectra, n_dims).
#     shape : tuple of int
#         Spatial grid shape, inferred from unique coordinate values if not provided.
#     profile_type : str
#         Spatial profile type, inferred from dimensionality.
#     ndim : int
#         Number of spatial dimensions.
#     """

#     _VALID_TYPES: Set[str] = {'point', 'line', 'map', 'volume', 'unstructured'}

#     def __init__(self, positions: List[Tuple[Union[int, float], ...]], shape: Optional[Tuple[int, ...]] = None, profile_type: Optional[Literal['point', 'line', 'map', 'volume', 'unstructured']] = None, validate: bool = True,) -> None:
#         self.positions: List[Tuple[Union[int, float], ...]] = positions

#         # Infer or set shape/type
#         self.shape: Tuple[int, ...] = shape or self._infer_shape()
#         self.profile_type: str = profile_type or self._infer_profile_type()

#         if validate:
#             self._validate()

#     def __repr__(self):
#         pass

#     @property
#     def positions(self) -> List[Tuple[Union[int, float], ...]]:
#         return self._positions

#     @positions.setter
#     def postions(self, positions: List[Tuple[Union[int, float], ...]]) -> None:
#         positions: List[Tuple[Union[int, float], ...]] = positions

#     @property
#     def shape(self) -> Tuple[int, ...]:
#         return self._shape

#     @property
#     def profile_type(self) -> str:
#         return self._profile_type

#     @property
#     def ndim(self) -> int:
#         p: np.ndarray = np.asarray(self.positions)
#         return p.shape[1] if p.ndim == 2 else 1

#     def _infer_shape(self) -> Tuple[int, ...]:
#         """
#         Infer grid shape from positions.

#         Returns
#         -------
#         tuple of int
#             Estimated shape of the spatial grid.
#         """
#         ...

#     def _infer_profile_type(self) -> str:
#         """
#         Infer the type of spatial profile ('point', 'line', 'map', 'volume', 'unstructured').
#         """
#         ...

#     def _validate(self) -> None:
#         """Ensure positions, shape, and type are consistent."""
#         ...

#     def get_neighbours(self, location, mode: Literal["4-connectivity", "8-connectivity", "6-connectivity", "kNN"] = "4-connectivity", use_indices: bool = True, k: int = 4,) -> List[Tuple[Union[int, float], ...]]:
#         """
#         Return neighbouring spectra indices based on grid or spatial proximity.

#         Parameters
#         ----------
#         location : tuple of int or float
#             Grid index (if `use_indices=True`) or spatial position (if False).
#         mode : {'4-connectivity', '8-connectivity', '6-connectivity', 'kNN'}, default='4-connectivity'
#             Connectivity pattern for neighbour determination.
#         use_indices : bool, default=True
#             Whether to interpret `location` as grid indices instead of spatial coordinates.
#         k : int, default=4
#             Number of neighbours for 'kNN' mode.

#         Returns
#         -------
#         list of tuple of int
#             Indices (i, j[, k]) of neighbouring spectra.

#         Notes
#         -----
#         - For index-based queries, the grid shape must be defined.
#         - For position-based queries, physical coordinates are used to find the
#           nearest positions by Euclidean distance.
#         """
#         ...

#     def _get_neighbours_by_index(self, index: Tuple[int, ...], mode: str,) -> List[Tuple[int, ...]]:
#         """Neighbour detection for regular grid indices."""
#         ...

#     def _get_neighbours_by_position(self, position: Tuple[float, ...], mode: str, k: int = 4,) -> List[Tuple[Union[int, float], ...]]:
#         """Neighbour detection using Euclidean distance between spatial positions."""
#         ...

#     def _index_from_flat(self, flat_index: int) -> Tuple[int, ...]:
#         """Convert flat index to grid coordinates based on shape."""
#         ...
