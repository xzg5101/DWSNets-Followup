from typing import Optional, Tuple

import torch
from torch.nn import ModuleDict

from nn.layers.base import BaseLayer, GeneralSetLayer


from typing import Tuple

class SelfToSelfLayer(BaseLayer):
    """Mapping bi -> bi"""

    def __init__(
        self,
        in_features,
        out_features,
        in_shape,
        out_shape,
        bias: bool = True,
        reduction: str = "max",
        n_fc_layers: int = 1,
        num_heads: int = 8,
        set_layer: str = "sab",
        is_output_layer=False,
    ):
        """

        :param in_features: input feature dim
        :param out_features:
        :param in_shape:
        :param out_shape:
        :param bias:
        :param reduction:
        :param n_fc_layers:
        :param num_heads:
        :param set_layer:
        :param is_output_layer: indicates that the bias is that of the last layer.
        """
        super().__init__(
            in_features,
            out_features,
            in_shape=in_shape,
            out_shape=out_shape,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
        )
        self.is_output_layer = is_output_layer
        if is_output_layer:
            # i=L-1
            assert in_shape == out_shape
            self.layer = self._get_mlp(
                in_features=in_shape[0] * in_features,
                out_features=in_shape[0] * out_features,
                bias=bias,
            )
        else:
            self.layer = GeneralSetLayer(
                in_features=in_features,
                out_features=out_features,
                reduction=reduction,
                bias=bias,
                n_fc_layers=n_fc_layers,
                num_heads=num_heads,
                set_layer=set_layer,
            )

    def forward(self, x):
        # (bs, d{i+1}, in_features)
        if self.is_output_layer:
            # (bs, d{i+1} * out_features)
            x = self.layer(x.flatten(start_dim=1))
            # (bs, d{i+1}, out_features)
            x = x.reshape(x.shape[0], self.out_shape[0], self.out_features)
        else:
            # (bs, d{i+1}, out_features)
            x = self.layer(x)
        return x

class SelfToOtherLayer(BaseLayer):
    """Mapping bi -> bj"""

    def __init__(
        self,
        in_features,
        out_features,
        in_shape,
        out_shape,
        bias: bool = True,
        reduction: str = "max",
        n_fc_layers: int = 1,
        first_dim_is_output=False,
        last_dim_is_output=False,
    ):
        """

        :param in_features: input feature dim
        :param out_features:
        :param in_shape:
        :param out_shape:
        :param bias:
        :param reduction:
        :param n_fc_layers:
        """
        super().__init__(
            in_features,
            out_features,
            in_shape=in_shape,
            out_shape=out_shape,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
        )

        assert not (first_dim_is_output and last_dim_is_output)
        self.first_dim_is_output = first_dim_is_output
        self.last_dim_is_output = last_dim_is_output

        if self.first_dim_is_output:
            # b{L-1} -> bj
            self.layer = self._get_mlp(
                in_features=in_features * in_shape[0],  # in_features * dL
                out_features=out_features,
                bias=bias,
            )
        elif self.last_dim_is_output:
            # bi -> b{L-1}
            self.layer = self._get_mlp(
                in_features=in_features,
                out_features=out_features * out_shape[0],  # out_features * dL
                bias=bias,
            )
        else:
            # i,j != L-1
            self.layer = self._get_mlp(
                in_features=in_features, out_features=out_features, bias=bias
            )

    def forward(self, x):
        if self.first_dim_is_output:
            # b{L-1} -> bj
            # (bs, dL, in_features)
            # (bs, dL * in_features)
            x = x.flatten(start_dim=1)
            # (bs, out_features)
            x = self.layer(x)
            # (bs, b{j+1}, out_features)
            x = x.unsqueeze(1).repeat(1, self.out_shape[0], 1)

        elif self.last_dim_is_output:
            # bi -> b{L-1}
            # (bs, d{i+1}, in_features)
            # (bs, in_features)
            x = self._reduction(x, dim=1)
            # (bs, dL * out_features)
            x = self.layer(x)
            # (bs, dL, out_features)
            x = x.reshape(x.shape[0], self.out_shape[0], self.out_features)
        else:
            # i,j != L-1
            # (bs, d{i+1}, in_features)
            # (bs, in_features)
            x = self._reduction(x, dim=1)
            # (bs, out_features)
            x = self.layer(x)
            # (bs, b{j+1}, out_features)
            x = x.unsqueeze(1).repeat(1, self.out_shape[0], 1)

        return x

class BiasToBiasBlock(BaseLayer):
    def __init__(
        self,
        in_features,
        out_features,
        shapes,
        bias: bool = True,
        reduction: str = "max",
        n_fc_layers: int = 1,
        num_heads: int = 8,
        set_layer: str = "sab",
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
        assert all([len(shape) == 1 for shape in shapes])

        self.shapes = shapes
        self.n_layers = len(shapes)

        self.layer = SelfToOtherLayer(
            in_features=in_features,
            out_features=out_features,
            in_shape=shapes[-1],
            out_shape=shapes[-1],
            reduction=reduction,
            bias=bias,
            n_fc_layers=n_fc_layers,
            first_dim_is_output=True,
            last_dim_is_output=True,
        )

    def forward(self, x: Tuple[torch.tensor]):
        inputs = torch.stack(x, dim=1)  # stack the inputs along the second dimension
        for i in range(self.n_layers):
            layer_output = self.layer(inputs[:, i, :, :])
            if i < self.n_layers - 1:
                inputs = torch.bmm(layer_output, inputs)
