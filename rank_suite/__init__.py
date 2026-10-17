"""Feasible Rank Window experiment suite.

LoRA rank x catastrophic-forgetting experiments on decoder-only sub-3B
language models over sequential GLUE classification tasks. Fixed models
from methodology v1.1; explicit protocol-v2 amendments live under docs/.
"""

from . import config

__version__ = config.SUITE_VERSION
