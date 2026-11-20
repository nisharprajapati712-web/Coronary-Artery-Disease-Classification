"""
Sharpness-Aware Minimization (SAM) optimizer.

Implements the two-step gradient update from the paper:
  "Sharpness-Aware Minimization for Efficiently Improving Generalization"
  (Foret et al., 2021)

SAM optimizes towards flat minima of the loss landscape:
  1. Compute gradient w.r.t. current params.
  2. Take an ascent step within the neighbourhood radius rho to find
     the worst-case perturbation (epsilon).
  3. Compute gradient at the perturbed point and apply the true update
     using the base optimizer (e.g., Adam).

Used with rho = 0.5 as specified in the paper Table 1.
"""

from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization wrapper.

    Args:
        params:            model parameters (iterable)
        base_optimizer:    base optimizer class (e.g., torch.optim.Adam)
        rho:               neighbourhood radius rho (paper: 0.5)
        adaptive:          use adaptive SAM (ASAM) if True
        **base_optimizer_kwargs: kwargs for the base optimizer
    """

    def __init__(
        self,
        params,
        base_optimizer: type[torch.optim.Optimizer],
        rho: float = 0.5,
        adaptive: bool = False,
        **base_optimizer_kwargs,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **base_optimizer_kwargs)
        super(SAM, self).__init__(params, defaults)

        # Create the base optimizer over the same param groups
        self.base_optimizer = base_optimizer(self.param_groups, **base_optimizer_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """
        First step of SAM: perform the ascent (perturbation) step.
        Computes epsilon = rho * grad / ||grad|| and adds it to params.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (
                    (torch.abs(p) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p.device)
                )
                p.add_(e_w)  # climb to the neighbourhood maximum
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Second step of SAM: restore params and apply the base optimizer update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # restore original params

        self.base_optimizer.step()  # apply the true update

        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        """
        SAM step.  Caller is expected to call first_step(), then manually
        compute the loss/gradients at the perturbed parameters using a
        closure, then call second_step().

        For compatibility, `step` here delegates to the base optimizer but
        logs a warning; use first_step/second_step explicitly.
        """
        raise NotImplementedError(
            "Use first_step() and second_step() explicitly with a closure:\n"
            "  opt.first_step(zero_grad=True)\n"
            "  loss = closure()\n"
            "  loss.backward()\n"
            "  opt.second_step(zero_grad=True)"
        )

    def _grad_norm(self) -> torch.Tensor:
        """Compute the global gradient norm across all parameters."""
        device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    (
                        (torch.abs(p) if group["adaptive"] else 1.0) * p.grad
                    ).norm()
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm.to(device)


def build_sam_optimizer(model, rho: float = 0.5, lr: float = 0.001, weight_decay: float = 0.0001):
    """
    Build a SAM optimizer using Adam as the base optimizer (paper Table 1).

    Returns:
        sam:   SAM wrapper instance
        base_optimizer: the inner Adam optimizer (for param_groups access)
    """
    base_optimizer = torch.optim.Adam
    sam = SAM(
        model.parameters(),
        base_optimizer,
        rho=rho,
        lr=lr,
        weight_decay=weight_decay,
    )
    return sam, base_optimizer


if __name__ == "__main__":
    import torch.nn as nn

    net = nn.Linear(10, 2)
    sam, base = build_sam_optimizer(net, rho=0.5, lr=0.001)
    print("SAM optimizer built:", sam)
    print("param groups:", len(sam.param_groups))