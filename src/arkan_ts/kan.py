from __future__ import annotations

from typing import List
import torch
from torch import nn
from .basis import UnivariateRBFSpline, UnivariateBSpline


class KANBlock(nn.Module):
    """
    KAN-like layer: apply a learnable univariate basis per input dimension,
    then linearly mix all features to the output dimension.

    Parameters
    ----------
    in_dim : int
        Number of input dimensions.
    out_dim : int
        Number of outputs.
    n_basis : int
        Basis functions per input channel.
    basis : str
        "rbf" (Gaussian bumps) or "bspline" (cubic B-splines).
    """

    def __init__(self, in_dim: int, out_dim: int, n_basis: int = 8, basis: str = "rbf") -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0 or n_basis <= 1:
            raise ValueError("Invalid KANBlock dimensions.")
        if basis not in ("rbf", "bspline"):
            raise ValueError('basis must be "rbf" or "bspline".')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_basis = n_basis
        self.basis = basis

        Spline = UnivariateRBFSpline if basis == "rbf" else UnivariateBSpline
        self.splines = nn.ModuleList([Spline(n_basis=n_basis) for _ in range(in_dim)])
        self.mix = nn.Linear(in_dim * n_basis, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, in_dim).

        Returns
        -------
        torch.Tensor
            Shape (B, out_dim).
        """
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected (B,{self.in_dim}), got {tuple(x.shape)}")
        feats: List[torch.Tensor] = []
        for i in range(self.in_dim):
            phi_i = self.splines[i](x[:, i])  # (B, n_basis)
            feats.append(phi_i)
        H = torch.cat(feats, dim=1)           # (B, in_dim * n_basis)
        return self.mix(H)
