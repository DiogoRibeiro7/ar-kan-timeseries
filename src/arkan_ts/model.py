from __future__ import annotations

import torch
from torch import nn
from .kan import KANBlock


class ARKAN(nn.Module):
    """
    Full AR-KAN: AR memory (external) feeds a static KAN-style nonlinear map.
    For 1-step-ahead forecasting, out_dim=1.
    """

    def __init__(self, p: int, hidden: int = 50, n_basis: int = 8, basis: str = "rbf") -> None:
        super().__init__()
        if p <= 0:
            raise ValueError("p must be positive.")
        if hidden <= 0:
            raise ValueError("hidden must be positive.")
        if n_basis <= 1:
            raise ValueError("n_basis must be >= 2.")

        self.kan1 = KANBlock(in_dim=p, out_dim=hidden, n_basis=n_basis, basis=basis)
        self.act = nn.SiLU()
        self.kan2 = KANBlock(in_dim=hidden, out_dim=1, n_basis=max(4, n_basis // 2), basis=basis)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : torch.Tensor
            AR-weighted lags (B, p)

        Returns
        -------
        torch.Tensor
            Predicted next value (B,)
        """
        h = self.act(self.kan1(z))
        y = self.kan2(h)
        return y.squeeze(-1)
