from typing import Optional
from nodetracker.node.core.solver.core import ODESolver, EulerMethod, RK4
from nodetracker.node.core.solver.global_state import EulerMethodWithGlobalState, RK4WithGlobalState

def ode_solver_factory(name: Optional[str] = None, params: Optional[dict] = None) -> ODESolver:
    """
    Factory for ODESolvers. Supports:
    - EulerMethod
    - RK4
    If parameters and name are not set then returns DefaultODESolver
    Args:
        name: Solver name.
        params: ODESolver parameters
    Returns:
        ODESolver
    """
    if name is None:
        return RK4(  # Default ODESolver
            max_step_size=0.05
        )
    else:
        if params is None:
            raise TypeError('Parameters are not set!')

    catalog = {
        'euler': EulerMethod,
        'rk4': RK4,
        'euler_global': EulerMethodWithGlobalState,
        'rk4_global': RK4WithGlobalState
    }

    if name not in catalog:
        raise ValueError(f'Unknown solver name "{name}". Valid options: {list(catalog.keys())}.')

    return catalog[name](**params)
