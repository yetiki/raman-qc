"""
Author: Yoshiki Cook
Date: 2025-10-20
"""

from typing import Optional, Union, Self, Any, Dict

class Metadata:
    """
    Represents arbitrary metadata associates with a spectroscopic object
    (i.e. Spectrum, Measurement, or Dataset).
    Provides dict + attribute access.

    Parameters
    ----------
        
    Attributes
    ----------

    Methods
    ----------

    """
    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        self._data: Dict[str, Any] = data or {}

    def __getitem__(self, key) -> Any:
        return self._data.get(key, None)

    def __setitem__(self, key, value) -> None:
        self._data[key] = value

    def __getattr__(self, key) -> Any:
        if key in self._data:
            return self._data[key]
        raise AttributeError(
            f"Invalid field: metadata field must be in {list(self._data.keys())}. ",
            f"Got field='{key}'.")

    def __setattr__(self, key, value) -> None:
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._data[key] = value
    
    def __repr__(self) -> str:
        return f"Metadata(data={self._data})"
    
    @classmethod
    def as_metadata(cls, other: Union[Dict[str, Any], Self]) -> Self:
        if not isinstance(other, cls):
            return cls(other)
        return other

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def update(self, other: Dict[str, Any], overwrite=True) -> None:
        for k, v in other.items():
            if overwrite or k not in self._data:
                self._data[k] = v
    
    def merge(self, other: Self) -> Self:
        merged: Dict[str, Any] = self._data.copy()
        merged.update(other.to_dict())
        return Metadata(merged)