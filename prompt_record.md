# Prompting Record

### Attempt on the forward function of the DWSNet Layer

This is the first query.

```
You are a machine learning expert specialized in pytorch and neural networks. Help me optimize this python code:

def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        weights, biases = x
        new_weights_from_weights = self.weight_to_weight(weights)
        new_weights_from_biases = self.bias_to_weight(biases)

        new_biases_from_biases = self.bias_to_bias(biases)
        new_biases_from_weights = self.weight_to_bias(weights)

        # add and normalize by the number of matrices
        new_weights = tuple(
            (w0 + w1) / self.n_matrices
            for w0, w1 in zip(new_weights_from_weights, new_weights_from_biases)
        )
        new_biases = tuple(
            (b0 + b1) / self.n_matrices
            for b0, b1 in zip(new_biases_from_biases, new_biases_from_weights)
        )

        if self.add_skip:
            skip_out = tuple(self.skip(w) for w in x[0]), tuple(
                self.skip(b) for b in x[1]
            )
            new_weights = tuple(ws + w for w, ws in zip(new_weights, skip_out[0]))
            new_biases = tuple(bs + b for b, bs in zip(new_biases, skip_out[1]))

        return new_weights, new_biases
```

ChatGPT returned a code that raised an error. Second querey:

```
Running your code give me this error:
RuntimeError: stack expects each tensor to be equal size, but got [1024, 1, 32, 16] at entry 0 and [1024, 32, 32, 16] at entry 1

Fix it please.
```

ChatGPT's response:

```
I apologize for the oversight. It seems that the tensors within the tuples have different shapes, so torch.stack cannot be used directly. In this case, we can use list comprehensions instead of tuple comprehensions to store the intermediate results, and then convert them back to tuples at the end.

Here's the modified code:

import torch
from typing import Tuple

def forward(self, x: Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]):
    weights, biases = x
    new_weights_from_weights = self.weight_to_weight(weights)
    new_weights_from_biases = self.bias_to_weight(biases)

    new_biases_from_biases = self.bias_to_bias(biases)
    new_biases_from_weights = self.weight_to_bias(weights)

    # add and normalize by the number of matrices
    new_weights = [
        (w0 + w1) / self.n_matrices
        for w0, w1 in zip(new_weights_from_weights, new_weights_from_biases)
    ]
    new_biases = [
        (b0 + b1) / self.n_matrices
        for b0, b1 in zip(new_biases_from_biases, new_biases_from_weights)
    ]

    if self.add_skip:
        skip_out_weights = [self.skip(w) for w in weights]
        skip_out_biases = [self.skip(b) for b in biases]

        new_weights = [w + ws for w, ws in zip(new_weights, skip_out_weights)]
        new_biases = [b + bs for b, bs in zip(new_biases, skip_out_biases)]

    # Convert back to tuple of tensors
    new_weights = tuple(new_weights)
    new_biases = tuple(new_biases)

    return new_weights, new_biases

This modified code should now work correctly without throwing an error. The change made was to replace the tuple comprehensions with list comprehensions for creating new_weights and new_biases, which should not cause any issues with different tensor sizes.
```

And this piece of code does run without error. However essentially it gave a piece of equivalent code.
