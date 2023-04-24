import logging
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from experiments.ssl.data.inr_dataset import SineINR2CoefDataset
from experiments.ssl.loss import info_nce_loss
from experiments.utils import (
    common_parser,
    count_parameters,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)
from nn.models import DWSModelForClassification, MLPModelForClassification

set_logger()


@torch.no_grad()
def evaluate(model, projection, loader):
    model.eval()
    loss = 0.0
    correct = 0.0
    total = 0.0
    all_features = []
    all_labels = []
    
    def concatenate_tensors(tensor1, tensor2):
        return torch.cat([tensor1, tensor2])

    for batch in loader:
        batch = batch.to(device)
        inputs = (
            tuple(concatenate_tensors(w, aug_w) for w, aug_w in zip(batch.weights, batch.aug_weights)),
            tuple(concatenate_tensors(b, aug_b) for b, aug_b in zip(batch.biases, batch.aug_biases)),
        )
        features = model(inputs)
        zs = projection(features)
        logits, labels = info_nce_loss(zs, args.temperature)
        loss += F.cross_entropy(logits, labels, reduction="sum").item()
        total += len(labels)
        real_bs = batch.weights[0].shape[0]
        pred = logits.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        all_features.append(features[:real_bs, :].cpu().numpy())
        all_labels.extend(batch.label.cpu().numpy())

    model.train()
    avg_loss = loss / total
    avg_acc = correct / total

    return dict(
        avg_loss=avg_loss,
        avg_acc=avg_acc,
        features=np.concatenate(all_features),
        labels=np.array(all_labels),
    )

def main(
    path,
    epochs: int,
    lr: float,
    batch_size: int,
    device,
    eval_every: int,
):
    # load dataset
    train_set = SineINR2CoefDataset(
        path=path,
        split="train",
        normalize=args.normalize,
        augmentation=args.augmentation,
        permutation=args.permutation,
        statistics_path=args.statistics_path,
    )
    val_set = SineINR2CoefDataset(
        path=path,
        split="val",
        normalize=args.normalize,
        statistics_path=args.statistics_path,
    )
    test_set = SineINR2CoefDataset(
        path=path,
        split="test",
        normalize=args.normalize,
        statistics_path=args.statistics_path,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logging.info(
        f"train size {len(train_set)}, "
        f"val size {len(val_set)}, "
        f"test size {len(test_set)}"
    )

    point = train_set[0]
    weight_shapes = tuple(w.shape[:2] for w in point.weights)
    bias_shapes = tuple(b.shape[:1] for b in point.biases)

    logging.info(f"weight shapes: {weight_shapes}, bias shapes: {bias_shapes}")

    # Create a dictionary to hold model parameters
    model_params = {
        'hidden_dim': args.dim_hidden,
        'n_hidden': args.n_hidden,
        'bn': args.add_bn,
        'n_classes': args.embedding_dim
    } 

    # Add specific parameters for DWSModelForClassification
    dws_params = {
        'weight_shapes': weight_shapes,
        'bias_shapes': bias_shapes,
        'input_features': 1,
        'reduction': args.reduction,
        'n_fc_layers': args.n_fc_layers,
        'set_layer': args.set_layer,
        'n_out_fc': args.n_out_fc,
        'dropout_rate': args.do_rate,
        **model_params
    }

    # Add specific parameters for MLPModelForClassification
    mlp_params = {
        'in_dim': sum(w.numel() for w in weight_shapes + bias_shapes),
        **model_params
    }

    # Create model based on args.model
    model = {
        "dwsnet": DWSModelForClassification(**dws_params).to(device),
        "mlp": MLPModelForClassification(**mlp_params).to(device),
    }[args.model]


    projection = nn.Sequential(
        nn.Linear(args.embedding_dim, args.embedding_dim),
        nn.ReLU(),
        nn.Linear(args.embedding_dim, args.embedding_dim),
    ).to(device)

    logging.info(f"number of parameters: {count_parameters(model)}")

    optimizers = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW
    }

    optimizer_params = {
        "params": list(model.parameters()) + list(projection.parameters()),
        "lr": lr,
        "weight_decay": 5e-4
    }

    if args.optim == "sgd":
        optimizer_params["momentum"] = 0.9
    elif args.optim == "adamw":
        optimizer_params["amsgrad"] = True

    optimizer = optimizers[args.optim](**optimizer_params)

    def log_results_to_wandb(args, epoch, train_loss, val_loss_dict, test_loss_dict, best_val_results, best_test_results):
        log = {
            "train/loss": train_loss,
            "val/loss": val_loss_dict["avg_loss"],
            "val/acc": val_loss_dict["avg_acc"],
            "val/best_loss": best_val_results["avg_loss"],
            "val/best_acc": best_val_results["avg_acc"],
            "test/loss": test_loss_dict["avg_loss"],
            "test/acc": test_loss_dict["avg_acc"],
            "test/best_loss": best_test_results["avg_loss"],
            "test/best_acc": best_test_results["avg_acc"],
            "epoch": epoch,
        }
        wandb.log(log)

    def train_epoch(model, projection, train_loader, criterion, optimizer, device):
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            batch = batch.to(device)
            inputs = (
                tuple(torch.cat([w, aug_w]) for w, aug_w in zip(batch.weights, batch.aug_weights)),
                tuple(torch.cat([b, aug_b]) for b, aug_b in zip(batch.biases, batch.aug_biases)),
            )
            features = model(inputs)
            zs = projection(features)
            logits, labels = info_nce_loss(zs, args.temperature)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        return loss.item()


    epoch_iter = trange(epochs)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = 1e6
    best_test_results, best_val_results = None, None
    test_acc, test_loss = -1.0, -1.0

    for epoch in epoch_iter:
        train_loss = train_epoch(model, projection, train_loader, criterion, optimizer, device)
        epoch_iter.set_description(
            f"[{epoch}], train loss: {train_loss:.3f}, test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}"
        )

        if (epoch + 1) % eval_every == 0:
            val_loss_dict = evaluate(model, projection, val_loader)
            test_loss_dict = evaluate(model, projection, test_loader)
            val_loss = val_loss_dict["avg_loss"]
            val_acc = val_loss_dict["avg_acc"]
            test_loss = test_loss_dict["avg_loss"]
            test_acc = test_loss_dict["avg_acc"]

            best_val_criteria = val_loss <= best_val_loss

            if best_val_criteria:
                best_val_loss = val_loss
                best_test_results = test_loss_dict
                best_val_results = val_loss_dict

            # Log the results with wandb
            if args.wandb:
                log_results_to_wandb(args, epoch, train_loss, val_loss_dict, test_loss_dict,
                                     best_val_results, best_test_results)


if __name__ == "__main__":
    parser = ArgumentParser("SSL trainer", parents=[common_parser])
    parser.set_defaults(
        data_path="dataset/ssl_splits.json",
        lr=5e-3,
        n_epochs=500,
        batch_size=512,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dwsnet",
        choices=["dwsnet", "mlp"],
        help="model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="temperature",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="embedding dimension",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw",
        choices=["adam", "sgd", "adamw"],
        help="optimizer",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="num workers")
    parser.add_argument(
        "--reduction",
        type=str,
        default="max",
        choices=["mean", "sum", "max"],
        help="reduction strategy",
    )
    parser.add_argument(
        "--dim-hidden",
        type=int,
        default=16,
        help="dim hidden layers",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=4,
        help="num hidden layers",
    )
    parser.add_argument(
        "--n-fc-layers",
        type=int,
        default=1,
        help="num linear layers at each ff block",
    )
    parser.add_argument(
        "--n-out-fc",
        type=int,
        default=1,
        help="num linear layers at final layer (invariant block)",
    )
    parser.add_argument(
        "--set-layer",
        type=str,
        default="sab",
        choices=["sab", "ds"],
        help="set layer",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=8,
        help="number of attention heads",
    )
    parser.add_argument(
        "--statistics-path",
        type=str,
        default="dataset/statistics.pth",
        help="path to dataset statistics",
    )
    parser.add_argument("--eval-every", type=int, default=1, help="eval every")
    parser.add_argument(
        "--augmentation", type=str2bool, default=True, help="use augmentation"
    )
    parser.add_argument(
        "--permutation", type=str2bool, default=False, help="use permutations"
    )
    parser.add_argument(
        "--normalize", type=str2bool, default=True, help="normalize data"
    )

    parser.add_argument("--do-rate", type=float, default=0.0, help="dropout rate")
    parser.add_argument(
        "--add-bn", type=str2bool, default=True, help="add batch norm layers"
    )
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)
    # wandb
    if args.wandb:
        name = (
            f"model_embedding_{args.model}_lr_{args.lr}_hid_dim_{args.dim_hidden}_reduction_{args.reduction}"
            f"_bs_{args.batch_size}_seed_{args.seed}"
        )
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=name,
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(args)

    device = get_device(gpus=args.gpu)

    main(
        path=args.data_path,
        epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        device=device,
    )
