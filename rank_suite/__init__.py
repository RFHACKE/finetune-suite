"""Feasible Rank Window experiment suite.

LoRA rank x catastrophic-forgetting experiments on 1B–3B-class decoder-only
language models over sequential GLUE classification tasks. Fixed models
from methodology v1.1; explicit protocol-v2/v3 amendments live under docs/.
"""

from . import config

__version__ = config.SUITE_VERSION
