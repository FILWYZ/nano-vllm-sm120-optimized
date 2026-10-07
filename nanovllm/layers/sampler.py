import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self.noise_buffers = {}

    @torch.compile
    def _greedy(self, logits: torch.Tensor):
        return logits.argmax(dim=-1)

    @torch.compile
    def _stochastic(self, logits: torch.Tensor, temperatures: torch.Tensor,
                    noise: torch.Tensor, has_greedy: bool):
        original_logits = logits
        temperatures = temperatures.clamp_min(1e-10)
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(
            noise.exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        if has_greedy:
            greedy_tokens = original_logits.argmax(dim=-1)
            sample_tokens = torch.where(
                temperatures <= 1e-10, greedy_tokens, sample_tokens
            )
        return sample_tokens

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor,
                all_greedy: bool = False, has_greedy: bool = False):
        if all_greedy:
            return self._greedy(logits)
        key = (logits.device, tuple(logits.shape))
        noise = self.noise_buffers.get(key)
        if noise is None:
            noise = torch.empty(
                logits.shape, device=logits.device, dtype=torch.float32
            )
            self.noise_buffers[key] = noise
        return self._stochastic(logits, temperatures, noise, has_greedy)
