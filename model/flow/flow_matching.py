import torch
from model.flow.scheduler import NoiseScheduler
import torch.nn.functional as F
from flow_matching.solver.ode_solver import ODESolver

class FlowMatching:
    def __init__(
            self, noise_scheduler: NoiseScheduler,
            ode_method, ode_step_size
    ):
        self.noise_scheduler = noise_scheduler
        self.ode_method = ode_method
        self.ode_step_size = ode_step_size

    def generate_noisy_samples(self, x_start, t, noise):
        x_t, u_t = self.noise_scheduler.sample(
                t=t,
                samples=x_start,
                noise=noise,
        )

        return x_t, u_t

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, reduce_loss=True):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)
        x_t, u_t = self.generate_noisy_samples(x_start, t, noise)
        out = model(x_t, t, **model_kwargs)
        diff = out - u_t
        per_sample = diff.reshape(diff.shape[0], -1).pow(2).mean(dim=1)
        if reduce_loss:
            loss = per_sample.mean()
        else:
            loss = per_sample
        return {'loss': loss}
    
    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        timestep_start=0.0     
    ):
        device = noise.device
        vf = self.noise_scheduler.get_velocity_function(model)
        solver = ODESolver(velocity_model=vf)
        time_grid = torch.tensor([timestep_start, 1.0], device=device)
         
        synthetic_samples = solver.sample(
            time_grid=time_grid,
            x_init=noise,
            method=self.ode_method,
            return_intermediates=False,
            atol=1e-5,
            rtol=1e-5,
            step_size=self.ode_step_size,
            y=model_kwargs.get('y', None),
            # learning=model_kwargs.get('learning', False) #NOTE this goes unused and caused errors so removed
        )
        
        return synthetic_samples



    def solve(self, x, t, labels, **kwargs):
        device = x.device
        norm_t = self.norm_reverse(t)
        
        model = kwargs['model']
        
        vf = self.noise_scheduler.get_velocity_function(model)
        solver = ODESolver(velocity_model=vf)
        time_grid = torch.tensor([norm_t, 1.0], device=device)

        synthetic_samples = solver.sample(
            time_grid=time_grid,
            x_init=x,
            method=self.ode_method,
            return_intermediates=False,
            atol=1e-5,
            rtol=1e-5,
            step_size=self.ode_step_size,
        )

        return synthetic_samples