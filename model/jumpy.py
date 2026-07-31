import torch
from torch import nn

from model.model_utils import init_weights, TimestepEmbedder

class JumpyGenerativeModel(nn.Module): # nn.Module so time_embed (when used) is registered on the wrapper: trained, moved to device, and checkpointed
    """
    Jumpy: a simple generative modeling approach where discrete jumps are made based on
    the number of modes in the distribution, enabling a simple form of "chunking/chopping" the 
    modes and interpolation between completely end-to-end generative modeling and discrete 
    time flow matching.

    Training: sample k ~ Uniform({0,...,K-1}) per element, compute interpolated input x_k
    and train to predict the clean data x_1 directly (x-prediction, https://arxiv.org/abs/2511.13720)
    or the velocity (data - noise), depending on pred_type.

    Inference: run num_inference_jumps steps (can differ from training K, effectively
    scaling the "jump size").

    With num_jumps=1 equivalent to standard one-shot direct/vel regressive prediction.
    """

    def __init__(self, num_jumps: int = 1, pred_type: str = "direct", num_inference_jumps: int = None, time_conditioning: bool = False, time_embed_weight_init : str = "xavier", time_embed_dim: int = None, time_embed_cont: bool = False, t_eps: float = 5e-2, loss_reweighting: bool = False):
        super().__init__()
        assert num_jumps >= 1
        assert pred_type in ("direct", "velocity")
        assert not loss_reweighting or pred_type == "direct", "loss_reweighting requires pred_type=direct"
        self.num_jumps = num_jumps
        self.pred_type = pred_type
        self.t_eps = t_eps # lower clamp on (1 - t), as in JiT
        self.loss_reweighting = loss_reweighting
        self.num_inference_jumps = num_inference_jumps if num_inference_jumps else num_jumps # None/0 default to num jumps
        self.time_conditioning = time_conditioning
        self.time_embed_cont = time_embed_cont
        self.time_embed = None

        if self.time_conditioning:
            assert num_jumps > 1, "jumpy_time_conditioning requires jumpy_n_jumps > 1"

            if time_embed_dim is not None: # None when the model has its own time embedder (e.g. DiT)
                if time_embed_cont:
                    self.time_embed = TimestepEmbedder(time_embed_dim)
                else:
                    self.time_embed = nn.Embedding(self.num_jumps, time_embed_dim)
                    init_weights(self.time_embed, time_embed_weight_init)

    def sample_timestep(self, batch_size: int, device) -> torch.Tensor:
        """Sample timestep k ~ Uniform({0,...,K-1}) per element. Returns (B,) int64 tensor."""
        return torch.randint(0, self.num_jumps, (batch_size,), device=device) # (B,)

    def k_to_time(self, k: torch.Tensor) -> torch.Tensor:
        """Convert integer jump index k to time value for model conditioning.
        Continuous: k/K float in [0,1). Discrete: k as int64."""
        if self.time_embed_cont:
            return k.float() / self.num_jumps
        return k.long()

    def interpolate(self, noise: torch.Tensor, data: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Linearly interpolates between noise and data to get x_k
        x_k = (1 - k/K)*noise + (k/K)*data. k is a (B,) int/float tensor.
        """
        t = k.float().reshape(-1, *([1] * (noise.dim() - 1))) / self.num_jumps
        return (1.0 - t) * noise + t * data

    def get_input_and_target(self, noise: torch.Tensor, data: torch.Tensor, k: torch.Tensor):
        """
        Returns (x_k, target) for training.
        For velocity: target is data - noise (O(1), doing this and not the raw difference avoids tiny targets when num jumps is large).
        For direct: target is the clean data x_1 (x-prediction).
        """
        x_k = self.interpolate(noise, data, k)
        if self.pred_type == "velocity":
            return x_k, data - noise
        else: # direct
            return x_k, data

    def apply_loss_weight(self, loss: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Optionally weight per-sample losses by 1/(1-t)^2, equivalent to JiT's v-space loss."""
        if not self.loss_reweighting:
            return loss
        t = k.float() / self.num_jumps
        return loss / (1.0 - t).clamp_min(self.t_eps) ** 2

    def apply_prediction(self, x: torch.Tensor, pred: torch.Tensor, scale: float = 1.0, t: float = 0.0) -> torch.Tensor:
        """Map raw model output to x_{k+1} based on pred_type, optionally scaling the step size.
        t is the current time in [0,1); for direct the velocity is recovered as (pred - x)/(1 - t)."""
        if self.pred_type == "velocity":
            return x + (scale / self.num_jumps) * pred
        else: # direct
            return x + (scale / self.num_jumps) * (pred - x) / max(1.0 - t, self.t_eps)

    def inference(self, noise: torch.Tensor, model_fn, post_step_fn=None) -> torch.Tensor:
        """
        Run num_inference_jumps steps from noise to output.

        Args:
            noise:        (B, ...) starting noise.
            model_fn:     callable(x) or callable(x, t) when time_conditioning=True,
                          where t is a (B,) tensor (int64 discrete or float continuous).
            post_step_fn: optional callable(x) -> x after each step
                          (e.g. restore conditioning frames in world models).
        """
        x = noise.clone()
        scale = self.num_jumps / self.num_inference_jumps  # scale step size when inference steps != training steps
        for i in range(self.num_inference_jumps):
            if self.time_conditioning:
                k_i = i * self.num_jumps // self.num_inference_jumps
                t = torch.full((x.shape[0],), k_i, device=x.device, dtype=torch.long)
                t = self.k_to_time(t)
                pred = model_fn(x, t)
            else:
                pred = model_fn(x)
            
            x = self.apply_prediction(x, pred, scale, t=i / self.num_inference_jumps)
            if post_step_fn is not None:
                x = post_step_fn(x)
        return x
