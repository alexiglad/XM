import torch
from typing import Union, Optional
from functools import partial
from flow_matching.path import CondOTProbPath


class NoiseScheduler:
    def __init__(self):
        self.scheduler = CondOTProbPath()

    def __getattr__(self, name):
        # Forward anything not explicitly implemented to the underlying scheduler
        return getattr(self.scheduler, name)

    def sample(self, t, samples, noise):
        """
        Unified sample interface.
        Returns x_t and u_t (and rho if needed for non-flow_matching).
        """

        path_sample = self.scheduler.sample(t=t, x_0=noise, x_1=samples)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t

        return x_t, u_t

    def get_velocity_function(self, model):
        """
        For flow_matching: return the model itself (velocity model).
        For noise-schedule training: return scheduler-provided velocity fn.
        """
        return model
    
        