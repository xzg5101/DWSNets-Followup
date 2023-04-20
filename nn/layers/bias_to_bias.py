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
            self.layer = self._get_mlp(
                in_features=in_features * in_shape[0],
                out_features=out_features,
                bias=bias,
            )
        elif self.last_dim_is_output:
            self.layer = self._get_mlp(
                in_features=in_features,
                out_features=out_features * out_shape[0],
                bias=bias,
            )
        else:
            self.layer = self._get_mlp(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
            )

    def forward(self, x):
        x = self._reduction(x, dim=1) if not self.first_dim_is_output else x.flatten(start_dim=1)
        x = self.layer(x)

        if not self.last_dim_is_output:
            x = x.unsqueeze(1).expand(-1, self.out_shape[0], -1)

        if self.last_dim_is_output:
            x = x.reshape(x.shape[0], self.out_shape[0], self.out_features)

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

        self.layers = ModuleDict()
        # construct layers:
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if i == j:
                    self.layers[f"{i}_{j}"] = SelfToSelfLayer(
                        in_features=in_features,
                        out_features=out_features,
                        in_shape=shapes[i],
                        out_shape=shapes[j],
                        reduction=reduction,
                        bias=bias,
                        num_heads=num_heads,
                        set_layer=set_layer,
                        n_fc_layers=n_fc_layers,
                        is_output_layer=(
                            j == self.n_layers - 1
                        ),  # todo: make sure this condition is correct
                    )
                else:
                    self.layers[f"{i}_{j}"] = SelfToOtherLayer(
                        in_features=in_features,
                        out_features=out_features,
                        in_shape=shapes[i],
                        out_shape=shapes[j],
                        reduction=reduction,
                        bias=bias,
                        n_fc_layers=n_fc_layers,
                        first_dim_is_output=(
                            i == self.n_layers - 1
                        ),  # todo: make sure this condition is correct
                        last_dim_is_output=(
                            j == self.n_layers - 1
                        ),  # todo: make sure this condition is correct
                    )

    def forward(self, x: Tuple[torch.tensor]):
        out_biases = [
            0.0,
        ] * len(x)
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                out_biases[j] = out_biases[j] + self.layers[f"{i}_{j}"](x[i])

        return tuple(out_biases)
