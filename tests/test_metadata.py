"""
Author: Yoshiki Cook
Date: 2025-10-22
"""

import pytest
from typing import Any, Dict
from ramanqc.metadata import Metadata

def test_dict_style_access():
    """Test accessing metadata values using dict-style access."""
    data : Dict[str, Any] = {
        'str_value': 'str',
        'int_value': 1,
        'float_value': 0.1,
        'invalid attribute key': 0,
    }
    metadata: Metadata = Metadata(data)

    assert metadata['str_value'] == 'str'
    assert metadata['int_value'] == 1
    assert metadata['float_value'] == 0.1
    assert metadata['invalid attribute key'] == 0

    with pytest.raises(KeyError):
        _ = metadata['nonexistent_key']

def test_attribute_style_access():
    """Test accessing metadata values using attribute-style access."""
    data : Dict[str, Any] = {
        'str_value': 'str',
        'int_value': 1,
        'float_value': 0.1,
    }
    metadata: Metadata = Metadata(data)

    assert metadata.str_value == 'str'
    assert metadata.int_value == 1
    assert metadata.float_value == 0.1
    
    with pytest.raises(AttributeError):
        _ = metadata.nonexistent_key

def test_set_item():
    """Test setting metadata values using dict-style access."""
    metadata: Metadata = Metadata()
    metadata['new_key'] = 'new_value'
    assert metadata['new_key'] == 'new_value'

def test_get():
    """Test accessing metadata values without KeyError."""
    data : Dict[str, Any] = {
        'str_value': 'str',
        'int_value': 1,
        'float_value': 0.1,
        'invalid attribute key': 0,
    }
    metadata: Metadata = Metadata(data)

    assert metadata.get('str_value') == 'str'
    assert metadata.get('int_value') == 1
    assert metadata.get('float_value') == 0.1
    assert metadata.get('invalid attribute key') == 0

    assert metadata.get('nonexistent_key') == None
    assert metadata.get('nonexistent_key', 'default_value') == 'default_value'

def test_to_dict():
    """Test converting Metadata to a dictionary."""
    metadata: Metadata = Metadata({'key': 'value'})
    assert metadata.to_dict() == {'key': 'value'}

def test_as_metadata():
    """Test creating Metadata from various inputs."""
    # from dict
    metadata: Metadata = Metadata.as_metadata({'key': 'value'})
    assert metadata['key'] == 'value'

    # from Metadata
    metadata: Metadata = Metadata.as_metadata(Metadata({'key': 'value'}))
    assert metadata['key'] == 'value'

    # from None
    metadata: Metadata = Metadata.as_metadata(None)
    assert isinstance(metadata, Metadata)

def test_n_keys():
    metadata: Metadata = Metadata({'key1': 'value1', 'key2': 'value2'})
    assert metadata.n_keys == 2

def test_update():
    metadata: Metadata = Metadata({'key': 'value'})
    metadata.update({'key': 'new_value'})
    assert metadata['key'] == 'new_value'

def test_merge():
    metadata1: Metadata = Metadata({'key1': 'value1'})
    metadata2: Metadata = Metadata({'key2': 'value2'})
    merged: Metadata = metadata1.merge(metadata2)
    assert merged['key1'] == 'value1'
    assert merged['key2'] == 'value2'

def test_copy():
    metadata: Metadata = Metadata({'key': 'value'})
    copied: Metadata = metadata.copy()
    assert copied['key'] == 'value'

def test_set_default():
    metadata: Metadata = Metadata({'key': 'value'})
    metadata.set_default('key', 'default_value')
    metadata.set_default('nonexistent_key', 'default_value')
    assert metadata.key == 'value'
    assert metadata.nonexistent_key == 'default_value'