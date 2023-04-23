from typing import Tuple

import torch
from torch import nn

from nn.layers import BN, DownSampleDWSLayer, Dropout, DWSLayer, InvariantLayer, ReLU


class MLPModel(nn.Module):
    def __init__(self, in_dim=2208, hidden_dim=256, n_hidden=2, bn=False, init_scale=1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for i in range(n_hidden):
            if i < n_hidden - 1:
                if not bn:
                    layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
                else:
                    layers.extend(
                        [
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.BatchNorm1d(hidden_dim),
                            nn.ReLU(),
                        ]
                    )
            else:
                layers.append(nn.Linear(hidden_dim, in_dim))
        # # todo: this model have one extra layer compare with the other alternatives for model-to-model
        # layers.append(nn.Linear(hidden_dim, in_dim))
        self.seq = nn.Sequential(*layers)

        self._init_model_params(init_scale)

    def _init_model_params(self, scale):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                out_c, in_c = m.weight.shape
                g = (2 * in_c / out_c) ** 0.5
                # nn.init.xavier_normal_(m.weight, gain=g)
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight)
                m.weight.data = m.weight.data * g * scale
                if m.bias is not None:
                    # m.bias.data.fill_(0.0)
                    m.bias.data.uniform_(-1e-4, 1e-4)

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        weight, bias = x
        bs = weight[0].shape[0]
        weight_shape, bias_shape = [w[0, :].shape for w in weight], [
            b[0, :].shape for b in bias
        ]
        all_weights = weight + bias
        weight = torch.cat([w.flatten(start_dim=1) for w in all_weights], dim=-1)
        weights_and_biases = self.seq(weight)
        n_weights = sum([w.numel() for w in weight_shape])
        weights = weights_and_biases[:, :n_weights]
        biases = weights_and_biases[:, n_weights:]
        weight, bias = [], []
        w_index = 0
        for s in weight_shape:
            weight.append(weights[:, w_index : w_index + s.numel()].reshape(bs, *s))
            w_index += s.numel()
        w_index = 0
        for s in bias_shape:
            bias.append(biases[:, w_index : w_index + s.numel()].reshape(bs, *s))
            w_index += s.numel()
        return tuple(weight), tuple(bias)


class MLPModelForClassification(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, n_hidden=2, n_classes=10, bn=False):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden):
            if not bn:
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            else:
                layers.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                    ]
                )

        layers.append(nn.Linear(hidden_dim, n_classes))
        self.seq = nn.Sequential(*layers)

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        weight, bias = x
        all_weights = weight + bias
        weight = torch.cat([w.flatten(start_dim=1) for w in all_weights], dim=-1)
        return self.seq(weight)


from torch.nn import ReLU, Dropout
import torch.nn.functional as F

class TupleReLU(nn.Module):
    def __init__(self):
        super(TupleReLU, self).__init__()

    def forward(self, x: Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]):
        return tuple(torch.relu(t) for t in x[0]), tuple(torch.relu(t) for t in x[1])

class TupleDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(TupleDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]):
        return tuple(F.dropout(t, self.p, self.training, self.inplace) for t in x[0]), tuple(F.dropout(t, self.p, self.training, self.inplace) for t in x[1])

class DWSModel(nn.Module):
    def __init__(
        self,
        weight_shapes: Tuple[Tuple[int, int], ...],
        bias_shapes: Tuple[Tuple[int], ...],
        input_features: int,
        hidden_dim: int,
        n_hidden: int = 2,
        output_features: int = None,
        reduction: str = "max",
        bias: bool = True,
        n_fc_layers: int = 1,
        num_heads: int = 8,
        set_layer: str = "sab",
        input_dim_downsample: int = None,
        dropout_rate: float = 0.0,
        add_skip: bool = False,
        add_layer_skip: bool = False,
        init_scale: float = 1e-4,
        init_off_diag_scale_penalty: float = 1.0,
        bn: bool = False,
    ):
        super().__init__()
        assert (
            len(weight_shapes) > 2
        ), "The current implementation only supports input networks with M > 2 layers."

        self.input_features = input_features
        self.input_dim_downsample = input_dim_downsample
        output_features = hidden_dim if output_features is None else output_features

        self.add_skip = add_skip
        if self.add_skip:
            self.skip = nn.Linear(input_features, output_features, bias=bias)
            with torch.no_grad():
                torch.nn.init.constant_(self.skip.weight, 1.0 / self.skip.weight.numel())
                torch.nn.init.constant_(self.skip.bias, 0.0)

        layers = []

        if input_dim_downsample is None:
            layers.append(DWSLayer(
                weight_shapes=weight_shapes,
                bias_shapes=bias_shapes,
                in_features=input_features,
                out_features=hidden_dim,
                reduction=reduction,
                bias=bias,
                n_fc_layers=n_fc_layers,
                num_heads=num_heads,
                set_layer=set_layer,
                add_skip=add_layer_skip,
                init_scale=init_scale,
                init_off_diag_scale_penalty=init_off_diag_scale_penalty,
            ))
        else:
            layers.append(DownSampleDWSLayer(
                weight_shapes=weight_shapes,
                bias_shapes=bias_shapes,
                in_features=input_features,
                out_features=hidden_dim,
                reduction=reduction,
                bias=bias,
                n_fc_layers=n_fc_layers,
                num_heads=num_heads,
                set_layer=set_layer,
                downsample_dim=input_dim_downsample,
                add_skip=add_layer_skip,
                init_scale=init_scale,
                init_off_diag_scale_penalty=init_off_diag_scale_penalty,
            ))

        for i in range(n_hidden):
            if bn:
                layers.append(BN(hidden_dim, len(weight_shapes), len(bias_shapes)))

            next_num_heads = 1 if i != (n_hidden - 1) else num_heads
            next_out_features = output_features if i != (n_hidden - 1) else hidden_dim

            if input_dim_downsample is None:
                layers.extend([
                    TupleReLU(),
                    TupleDropout(dropout_rate),
                    DWSLayer(
                        weight_shapes=weight_shapes,
                        bias_shapes=bias_shapes,
                        in_features=hidden_dim,
                        out_features=next_out_features,
                        reduction=reduction,
                        bias=bias,
                        n_fc_layers=n_fc_layers,
                        num_heads=next_num_heads,
                        set_layer=set_layer,
                        add_skip=add_layer_skip,
                        init_scale=init_scale,
                        init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                    ),
                ])
            else:
                layers.extend([
                    TupleReLU(),
                    TupleDropout(dropout_rate),
                    DownSampleDWSLayer(
                        weight_shapes=weight_shapes,
                        bias_shapes=bias_shapes,
                        in_features=hidden_dim,
                        out_features=next_out_features,
                        reduction=reduction,
                        bias=bias,
                        n_fc_layers=n_fc_layers,
                        num_heads=next_num_heads,
                        set_layer=set_layer,
                        downsample_dim=input_dim_downsample,
                        add_skip=add_layer_skip,
                        init_scale=init_scale,
                        init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                    ),
                ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        if isinstance(x, tuple):
            x = x[0]
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        out = self.clf(x)
        return out

from torch.nn import Dropout, ReLU
from typing import Tuple, Optional, Union, Dict
class DWSModelForClassification(nn.Module):
    def __init__(
        self,
        weight_shapes: Tuple[Tuple[int, int], ...],
        bias_shapes: Tuple[Tuple[int], ...],
        input_features: int,
        hidden_dim: int,
        dws_model_params: Optional[Dict[str, Union[int, float, bool, str]]] = None,
        invariant_layer_params: Optional[Dict[str, Union[int, float, str]]] = None,
    ):
        super().__init__()

        if dws_model_params is None:
            dws_model_params = {
                'n_hidden': 2,
                'reduction': 'max',
                'bias': True,
                'n_fc_layers': 1,
                'num_heads': 8,
                'set_layer': 'sab',
                'dropout_rate': 0.0,
                'input_dim_downsample': None,
                'init_scale': 1.0,
                'init_off_diag_scale_penalty': 1.0,
                'bn': False,
                'add_skip': False,
                'add_layer_skip': False,
                'equiv_out_features': None,
            }

        if invariant_layer_params is None:
            invariant_layer_params = {
                'n_classes': 10,
                'reduction': 'max',
                'n_out_fc': 1,
            }

        self.layers = DWSModel(
            weight_shapes=weight_shapes,
            bias_shapes=bias_shapes,
            input_features=input_features,
            hidden_dim=hidden_dim,
            **dws_model_params,
        )
        self.dropout = Dropout(dws_model_params['dropout_rate'])
        self.relu = ReLU()
        self.clf = InvariantLayer(
            weight_shapes=weight_shapes,
            bias_shapes=bias_shapes,
            in_features=hidden_dim if dws_model_params['equiv_out_features'] is None else dws_model_params['equiv_out_features'],
            **invariant_layer_params,
        )

    def forward(
        self, x: Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]], return_equiv: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.layers(x)
        out = self.clf(self.dropout(self.relu(x)))
        if return_equiv:
            return out, x
        else:
            return out


if __name__ == "__main__":
    weights = (
        torch.randn(4, 784, 128, 1),
        torch.randn(4, 128, 128, 1),
        torch.randn(4, 128, 10, 1),
    )
    biases = (torch.randn(4, 128, 1), torch.randn(4, 128, 1), torch.randn(4, 10, 1))
    in_dim = sum([w[0, :].numel() for w in weights]) + sum(
        [w[0, :].numel() for w in biases]
    )
    weight_shapes = tuple(w.shape[1:3] for w in weights)
    bias_shapes = tuple(b.shape[1:2] for b in biases)
    n_params = sum([i.numel() for i in weight_shapes + bias_shapes])
