"""Adam with one learning rate per parameter group and resizable state.

Densification adds and removes Gaussians during the optimisation. The moment estimates of
the Gaussians that survive must survive too, which is easier to see in thirty explicit lines
than through the internals of ``torch.optim``. The update is the one of Kingma and Ba, "Adam:
A Method for Stochastic Optimization", ICLR 2015 (algorithm 1); a test checks it against
``torch.optim.Adam`` step by step.
"""

from __future__ import annotations

import torch

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
#: Kerbl et al. use a much smaller epsilon than the PyTorch default (1e-8), because the
#: gradients of a single Gaussian in a mean-reduced image loss are tiny.
ADAM_EPS = 1e-15


class Adam:
    """Adam over a dictionary of leaf tensors that all share their first dimension."""

    def __init__(self, params: dict[str, torch.Tensor]) -> None:
        self.first = {name: torch.zeros_like(p) for name, p in params.items()}
        self.second = {name: torch.zeros_like(p) for name, p in params.items()}
        self.steps = 0

    @torch.no_grad()
    def step(self, params: dict[str, torch.Tensor], learning_rates: dict[str, float]) -> None:
        """Apply one update in place from ``p.grad`` and clear the gradients."""
        self.steps += 1
        correction1 = 1.0 - ADAM_BETA1**self.steps
        correction2 = 1.0 - ADAM_BETA2**self.steps
        for name, p in params.items():
            grad = p.grad
            m = self.first[name].mul_(ADAM_BETA1).add_(grad, alpha=1.0 - ADAM_BETA1)
            v = self.second[name].mul_(ADAM_BETA2).addcmul_(grad, grad, value=1.0 - ADAM_BETA2)
            update = (m / correction1) / ((v / correction2).sqrt() + ADAM_EPS)
            p.sub_(learning_rates[name] * update)
            p.grad = None

    def resize(self, keep: torch.Tensor, added: int) -> None:
        """Keep the moments of the rows selected by ``keep`` and append ``added`` zero rows."""
        for moments in (self.first, self.second):
            for name, value in moments.items():
                fresh = value.new_zeros((added, *value.shape[1:]))
                moments[name] = torch.cat([value[keep], fresh])
