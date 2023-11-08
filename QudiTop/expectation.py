"""Calculate expectation of given hamiltonian.
Refer: https://gitee.com/forcekeng/quditop
"""
import torch
import torch.nn as nn
from .gates import GateBase
from typing import List, Iterable


class Expectation(nn.Module):
    """
    ham = [(1.0, Z(1,2,3).on(1)), (2.0, X(1, 0, 3).on(2))]
    """

    def __init__(self, ham: List):
        super().__init__()
        self.ham = ham

    def forward(self, qs):
        """Forward function.
        """
        res = []
        for coef, gates in self.ham:
            expect = self._get_item_expectation(qs, gates)
            res.append(coef * expect)
        return torch.stack(res)

    def _get_item_expectation(self, qs, gates):
        """Calculate expectation based on given gates."""
        qs2 = qs
        if isinstance(gates, GateBase):
            qs2 = gates(qs)
        elif isinstance(gates, Iterable):
            for gate in gates:
                qs2 = gate(qs2)
        else:
            raise ValueError(f"Gates should be iterable or a single QuditGateBase, but got "
                             f"{type(gates)}.")
        return (qs[0] * qs2[0]).sum() + (qs[1] * qs2[1]).sum()


__all__ = ['Expectation']