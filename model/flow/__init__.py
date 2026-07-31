from .flow_matching import FlowMatching
from .scheduler import NoiseScheduler


def create_flow(
        ode_method,
        ode_step_size
):
    obj = FlowMatching(
        NoiseScheduler(), ode_method, ode_step_size
    )

    return obj
