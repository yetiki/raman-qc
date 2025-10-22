"""
Author: Yoshiki Cook
Date: 2025-10-20
"""

from typing import Optional, Union, Self, Any, Dict, List, Tuple

class Metadata:
    """
    Represents arbitrary metadata associated with a spectroscopic object
    (i.e. Spectrum, Measurement, or Dataset).
    Provides both dictionary-style and attribute-style access to metadata.

    Dictionary access (metadata['key']) works for all valid string keys.
    Attribute access (metadata.key) only works for keys that are valid Python identifiers
    (no spaces, special characters, or keys starting with digits).

    Examples
    --------
    >>> metadata = Metadata({'valid_key': 1, 'invalid_attribute_key_%': 2})
    >>> metadata.valid_key  # attribute-style access works: 1
    >>> metadata['valid_key']  # dict-style access works: 1
    >>> metadata['invalid_attribute_key_%']  # dict-style access works: 2
    # metadata.invalid_attribute_key_%  # attribute-style access won't work - SyntaxError: invalid syntax
    >>> metadata.nonexistent_key  # Raises AttributeError - key not found

    Parameters
    ----------
    data : Optional[Dict[str, Any]]
        A dictionary containing metadata fields and values.
        
    Attributes
    ----------
    _data : Dict[str, Any]
        A dictionary containing metadata fields and values.
    """
    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        self._data: Dict[str, Any] = data or {}

    def __getitem__(self, key) -> Any:
        if key in self._data:
            return self._data.get(key)
        raise KeyError(
            f"Invalid field: metadata field must be in {list(self._data.keys())}. ",
            f"Got field='{key}'."
        )

    def __setitem__(self, key, value) -> None:
        self._data[key] = value

    def __getattr__(self, key) -> Any:
        if key in self._data:
            return self._data[key]
        raise AttributeError(
            f"Invalid field: metadata field must be in {list(self._data.keys())}. ",
            f"Got field='{key}'."
            )

    def __setattr__(self, key, value) -> None:
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._data[key] = value
    
    def __repr__(self) -> str:
        return f"Metadata(data={self._data})"
    
    @classmethod
    def as_metadata(cls, other: Union[Dict[str, Any], Self]) -> Self:
        """Convert a dictionary to a Metadata instance if needed."""
        if not isinstance(other, cls):
            return cls(other)
        return other
    
    @property
    def keys(self) -> List[str]:
        """Return the list of metadata keys."""
        return list(self._data.keys())
    
    @property
    def values(self) -> List[Any]:
        """Return the list of metadata values."""
        return list(self._data.values())
    
    @property
    def items(self) -> List[Tuple[str, Any]]:
        """Return the list of metadata items as (key, value) tuples."""
        return list(self._data.items())

    def to_dict(self) -> Dict[str, Any]:
        """Return the metadata as a dictionary."""
        return dict(self._data)

    def update(self, other: Dict[str, Any], overwrite=True) -> None:
        """Update metadata with another dictionary."""
        for k, v in other.items():
            if overwrite or k not in self._data:
                self._data[k] = v
    
    def merge(self, other: Self) -> Self:
        """Return a new Metadata instance by merging with another Metadata instance."""
        merged: Dict[str, Any] = self._data.copy()
        merged.update(other.to_dict())
        return Metadata(merged)
    
    def copy(self) -> Self:
        """Return a deep copy of this metadata."""
        return Metadata(self._data.copy())