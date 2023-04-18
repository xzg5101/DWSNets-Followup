import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

# GPT optimized
class BaseLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_shape: Optional[Tuple] = None,
        out_shape: Optional[Tuple] = None,
        bias: bool = True,
        reduction: str = "mean",
        n_fc_layers: int = 1,
        num_heads: int = 8,
        set_layer: str = "ds",
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.bias = bias
        self.reduction = reduction
        self.n_fc_layers = n_fc_layers
        self.num_heads = num_heads

        assert set_layer in ["ds", "sab"], "Invalid set_layer"
        assert reduction in ["mean", "sum", "attn", "max"], "Invalid reduction"

        self.mlp = self._get_mlp(in_features, out_features, n_fc_layers, bias)

    def _get_mlp(self, in_features, out_features, n_fc_layers, bias=False):
        layers = [nn.Linear(in_features, out_features, bias=bias)]
        for _ in range(n_fc_layers - 1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(out_features, out_features, bias=bias))
        return nn.ModuleList(layers)

    def _reduction(self, x: torch.tensor, dim=1, keepdim=False):
        if self.reduction == "mean":
            return x.mean(dim=dim, keepdim=keepdim)
        elif self.reduction == "sum":
            return x.sum(dim=dim, keepdim=keepdim)
        elif self.reduction == "attn":
            raise NotImplementedError("Attention reduction not implemented")
        else:  # self.reduction == "max":
            return torch.max(x, dim=dim, keepdim=keepdim).values


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V, bias=False)
        self.fc_k = nn.Linear(dim_K, dim_V, bias=False)
        self.fc_v = nn.Linear(dim_K, dim_V, bias=False)
        self.fc_o = nn.Linear(dim_V, dim_V, bias=False)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        # Multi-head split
        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
        K = K.view(K.size(0), K.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
        V = V.view(V.size(0), V.size(1), self.num_heads, -1).permute(0, 2, 1, 3)

        A = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.dim_V), dim=-1)
        O = A @ V

        # Concatenate multi-head outputs
        O = O.permute(0, 2, 1, 3).contiguous().view(Q.size(0), -1, self.dim_V)

        if hasattr(self, "ln0"):
            O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        if hasattr(self, "ln1"):
            O = self.ln1(O)
        return O


class SAB(BaseLayer):
    def __init__(self, in_features, out_features, num_heads=8, ln=False):
        super().__init__(in_features, out_features)
        self.mab = MAB(in_features, in_features, out_features, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class SetLayer(BaseLayer):
    """
    from https://github.com/manzilzaheer/DeepSets/tree/master/PointClouds
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        reduction: str = "mean",
        n_fc_layers: int = 1,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
        )
        self.Gamma = self._get_mlp(in_features, out_features, bias=self.bias)
        self.Lambda = self._get_mlp(in_features, out_features, bias=False)
        self.reduction = reduction
        if self.reduction == "attn":
            self.attn = Attn(dim=in_features)

    def forward(self, x):
        # set dim is 1
        if self.reduction == "mean":
            xm = x.mean(1, keepdim=True)
        elif self.reduction == "sum":
            xm = x.sum(1, keepdim=True)
        elif self.reduction == "attn":
            xm = self.attn(x.transpose(-1, -2), keepdim=True).transpose(-1, -2)
        else:
            xm, _ = torch.max(x, dim=1, keepdim=True)

        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm
        return x


class GeneralSetLayer(BaseLayer):
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        reduction: str = "mean",
        n_fc_layers: int = 1,
        num_heads=8,
        set_layer="ds",
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
        )
        self.set_layer = dict(
            ds=SetLayer(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                reduction=reduction,
                n_fc_layers=n_fc_layers,
            ),
            sab=SAB(
                in_features=in_features, out_features=out_features, num_heads=num_heads
            ),
        )[set_layer]

    def forward(self, x):
        return self.set_layer(x)


class Attn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.query = nn.Parameter(
            torch.ones(
                dim,
            )
        )

    def forward(self, x, keepdim=False):
        # Note: reduction is applied to last dim. For example for (bs, d, d') we compute d' attn weights
        # by multiplying over d.
        attn = (x.transpose(-1, -2) * self.query).sum(-1)
        attn = F.softmax(attn, dim=-1)
        # todo: change to attn.unsqueeze(-2) ?
        if x.ndim == 3:
            attn = attn.unsqueeze(1)
        elif x.ndim == 4:
            attn = attn.unsqueeze(2)

        output = (x * attn).sum(-1, keepdim=keepdim)

        return output
