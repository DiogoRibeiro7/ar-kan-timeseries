from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class UnivariateRBFSpline(nn.Module):
    """Learnable univariate function via Gaussian RBF "spline":
      ϕ(x) = [exp(-β_m (x - c_m)^2)]_{m=1..n_basis}

    Centers and log-bandwidths are learnable; the linear mixing occurs in the
    following KAN layer.
    """

    def __init__(self, n_basis: int = 8, init_span: tuple[float, float] = (-2.5, 2.5)) -> None:
        super().__init__()
        if n_basis < 2:
            raise ValueError("n_basis must be >= 2.")
        self.n = int(n_basis)
        low, high = map(float, init_span)
        if not (high > low):
            raise ValueError("init_span must have high > low.")
        centers = torch.linspace(low, high, steps=self.n)
        self.centers = nn.Parameter(centers)  # (n,)
        self.log_band = nn.Parameter(torch.zeros(self.n))  # (n,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parameters
        ----------
        x : torch.Tensor
            Shape (...,)

        Returns:
        -------
        torch.Tensor
            Shape (..., n_basis)
        """
        if x.dtype not in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            x = x.to(torch.get_default_dtype())
        x_exp = x.unsqueeze(-1)  # (..., 1)
        beta = torch.exp(self.log_band)  # (n,)
        phi = torch.exp(-beta * (x_exp - self.centers) ** 2)  # (..., n)
        return phi


class UnivariateBSpline(nn.Module):
    """Cubic B-spline basis with open uniform knots (degree = 3).
    We keep knots fixed (non-trainable) for stability; learning happens in the
    linear mixing of the KAN layer.
    """

    def __init__(
        self,
        n_basis: int = 8,
        init_span: tuple[float, float] = (-2.5, 2.5),
        degree: int = 3,
    ) -> None:
        super().__init__()
        if degree != 3:
            raise ValueError("Only cubic (degree=3) is supported for now.")
        if n_basis < degree + 1:
            raise ValueError("n_basis must be >= degree + 1 for B-splines.")
        self.n_basis = int(n_basis)
        self.degree = int(degree)
        low, high = map(float, init_span)
        if not (high > low):
            raise ValueError("init_span must have high > low.")

        # Build an open uniform knot vector on [low, high]
        # Internal distinct knots for domain partitions:
        internal = torch.linspace(low, high, steps=self.n_basis - self.degree + 1).to(torch.float32)
        # Open uniform: repeat boundaries by 'degree' on each side
        self.register_buffer(
            "knots_full",
            torch.cat(
                [internal[:1].repeat(self.degree), internal, internal[-1:].repeat(self.degree)]
            ),
        )

    def _cox_de_boor(self, x: torch.Tensor, k: int, d: int, knots: torch.Tensor) -> torch.Tensor:
        """Evaluate B-spline basis N_{k,d}(x) via Cox–de Boor recursion."""
        if d == 0:
            left = knots[k]
            right = knots[k + 1]
            # include right end only for the last interval
            return ((x >= left) & (x < right)).to(x.dtype) | (
                (x == right) & (k + 1 == len(knots) - 1)
            ).to(x.dtype)

        denom1 = knots[k + d] - knots[k]
        denom2 = knots[k + d + 1] - knots[k + 1]

        term1 = 0.0
        term2 = 0.0
        if denom1 > 0:
            term1 = ((x - knots[k]) / denom1) * self._cox_de_boor(x, k, d - 1, knots)
        if denom2 > 0:
            term2 = ((knots[k + d + 1] - x) / denom2) * self._cox_de_boor(x, k + 1, d - 1, knots)
        return term1 + term2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parameters
        ----------
        x : torch.Tensor
            Shape (...,)

        Returns:
        -------
        torch.Tensor
            Shape (..., n_basis)
        """
        if x.dtype not in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            x = x.to(torch.get_default_dtype())
        # Evaluate on flattened view, then reshape back
        flat = x.reshape(-1)
        feats = []
        for k in range(self.n_basis):
            nk = self._cox_de_boor(flat, k, self.degree, self.knots_full)
            feats.append(nk.unsqueeze(-1))
        Phi = torch.cat(feats, dim=-1)  # (N, n_basis)
        out_shape = list(x.shape) + [self.n_basis]
        return Phi.reshape(out_shape)
