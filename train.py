from whale_data_loader import get_data_loader
from dataset_loader import DatasetLoader
from models import ModelLin
import torch
import os
import pandas as pd
from utils import print_bar


def main():
    data_loader = get_data_loader()
    model = ModelLin(40, output_size=7)
    fit(model, get_params(), data_loader)


def get_output(
    feats: torch.Tensor, model: torch.nn.Module
) -> torch.Tensor:
    """Get output of a model, given the model, and an input"""
    feats_b = feats.unsqueeze(0)
    output  = model(feats_b)
    return output


def calc_loss(
    output: torch.Tensor, target: torch.Tensor, criterion: torch.nn
) -> torch.Tensor:
    """Calculate the loss"""
    target_b = target.unsqueeze(0)
    loss = criterion(output, target_b)
    return loss


def fit_epoch(feats: torch.Tensor, target: torch.Tensor, model: torch.nn.Module
, criterion: torch.nn, optimizer: torch.optim.Optimizer) -> torch.Tensor:
    """Fits an epoch and returns the loss of the epoch"""
    optimizer.zero_grad()
    output = get_output(feats, model)
    loss = calc_loss(output, target, criterion)
    loss.backward()
    optimizer.step()
    return loss 


def fit(model: torch.nn.Module, params: {str: str}, data_loader: DatasetLoader) -> None:
    """Train a model, given parameters, a data loader, and the data keys for training and development"""
    if not os.path.exists(params["save_dir"]):
        os.makedirs(params["save_dir"])
    if not os.path.exists(params["save_model_dir"]):
        os.makedirs(params["save_model_dir"])
    torch.manual_seed(params["seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()
    train_losses = []
    dev_losses = []
    for epoch in range(params["max_epoch"]):
        print_bar(epoch + 1, params["max_epoch"], prefix="Training model:", length=40)
        train_loss = 0
        model.train()
        for key in data_loader.train_keys:
            feats, target = data_loader[key]
            loss = fit_epoch(feats, target, model, criterion, optimizer)
            train_loss += loss.item()
        train_loss /= len(data_loader.train_keys)
        train_losses.append(round(train_loss, 3))
        save_path = os.path.join(params["save_model_dir"], f"{epoch}.pth")
        torch.save(model, save_path)

        dev_loss = 0
        model.eval()
        for key in data_loader.dev_keys:
            feats, target = data_loader[key]
            output = get_output(feats, model)
            loss = calc_loss(output, target, criterion)
            dev_loss += loss.item()
        dev_loss /= len(data_loader.dev_keys)
        dev_loss = round(dev_loss, 3)
        dev_losses.append(dev_loss)
        save_path = os.path.join(params["save_dir"], "log.csv")
        save_training_log(epoch, train_losses, dev_losses, save_path)

        if dev_loss == min(dev_losses): 
            save_path = os.path.join(params["save_model_dir"], f"best_jit.pth")
            save_jit_model(feats, model, save_path)

def save_jit_model(input_fake, model, save_path):
    """Save the just in time (jit) version of the model.
    It requires an input to trace the path.
    """
    with torch.no_grad(): # the tracing with jit is needed to save a model without the code
        traced_cell = torch.jit.trace(model, input_fake)
    torch.jit.save(traced_cell, save_path)

def save_training_log(epoch: int, train_losses: [float], dev_losses: [float], save_path) -> None:
    """Save training log in a given path"""
    df = pd.DataFrame(
            {
                "epoch": [e for e in range(epoch + 1)],
                "train_loss": train_losses,
                "dev_loss": dev_losses,
            }
        )
    df.to_csv(save_path, index=False)

def get_params() -> {str: str}:
    """Function to get hyperparameters in the form of a dictionary."""
    params = {
        "seed": 0,
        "device": "cpu",
        "learning_rate": 0.0001,
        "save_dir": "results",
        "save_model_dir": "results/models",
        "max_epoch": 5,
    }
    return params


if __name__ == "__main__":
    main()
