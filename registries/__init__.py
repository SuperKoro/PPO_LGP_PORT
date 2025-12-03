# Registries for dispatching rules and metaheuristics
# Note: Import dispatching_rules and metaheuristics_impl modules to auto-register

from registries.dispatching_registry import register_dr, get_dr, has_dr, DR_REGISTRY
from registries.mh_registry import register_mh, get_mh, has_mh, MH_REGISTRY

# Auto-register by importing the implementation modules
import registries.dispatching_rules
import registries.metaheuristics_impl

__all__ = [
    'register_dr', 'get_dr', 'has_dr', 'DR_REGISTRY',
    'register_mh', 'get_mh', 'has_mh', 'MH_REGISTRY'
]
