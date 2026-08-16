"""Compatibility bridge to the shared project utilities.

Research modules historically imported ``src.utils`` through a sibling
relative import. This bridge preserves those imports after consolidation.
"""

from ..utils import *  # noqa: F401,F403

