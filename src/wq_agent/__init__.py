"""Compatibility shim for the former ``wq_agent`` package name.

New integrations should import :mod:`alphagen_agent` instead.
"""

from alphagen_agent import __version__
from alphagen_agent import __path__ as __path__

__all__ = ["__version__"]
