"""Compatibility import for the root-level self-play training module."""

import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("self_play_training")
_sys.modules[__name__] = _implementation
