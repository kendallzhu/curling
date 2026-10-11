"""Compatibility import for the root-level sequential training module."""

import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("sequential_training")
_sys.modules[__name__] = _implementation
