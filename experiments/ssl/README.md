# Self-supervised learning for dense representation

The INR data is available [here](https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0). Please download the data and place it in, e.g. `dataset/ssl-inrs` folder.

Next, create the data split using:

```shell
python generate_data_splits.py --data-path dataset/ssl-inrs --save-path dataset
```

This will create a json file `dataset/ssl_splits.json`.

Next, compute the dataset (INRs) statistics using `compute_statistics.py`:

```shell
python compute_statistics.py --data-path dataset/ssl_splits.json
```

This will create `dataset/statistics.pth` object.

Now, to run the experiment:

```shell
python trainer.py --data-path dataset/ssl_splits.json --statistics-path dataset/statistics.pth --model dwsnet
```

To enable [wandb](https://wandb.ai/site) experiment tracking:

```shell
python trainer.py --data-path dataset/ssl_splits.json --statistics-path dataset/statistics.pth --model dwsnet --wandb --wandb-project dws-nets --wandb-entity <your-entity>
```

To notify gpt changes:

```shell
python trainer.py --data-path dataset/ssl_splits.json --statistics-path dataset/statistics.pth --model dwsnet --wandb --wandb-project weight-space --gpt true --gptchagne ssltrainer
```
