# Registries for dispatching rules and metaheuristics
# Import modules to auto-register functions
from . import dispatching_rules
from . import metaheuristics_impl

from .dispatching_registry import register_dr, get_dr, has_dr, DR_REGISTRY
from .mh_registry import register_mh, get_mh, has_mh, MH_REGISTRY

__all__ = [
    'register_dr', 'get_dr', 'has_dr', 'DR_REGISTRY',
    'register_mh', 'get_mh', 'has_mh', 'MH_REGISTRY'
]
