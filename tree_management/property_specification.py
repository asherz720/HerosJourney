"""
tree_management/property_specification.py  —  DEPRECATED shim.

This module has been renamed to tree_management.elements.
All public names are re-exported here for backward compatibility.
Import from tree_management.elements going forward.
"""
from tree_management.elements import (  # noqa: F401
    fill_elements,
    load_lexicons,
    save_variant,
    generate_entity_repeat,
    _load_lexicon,
    _resolve_pool,
    LEXICON_DIR,
)
