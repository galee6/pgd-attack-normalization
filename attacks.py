"""PGD Attacks on models"""
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import Optional

# Torch imports
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Optuna
import optuna


# Model
from models import LightNet

# Random seed
SEED = 1443

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Constant (hyper-) parameters
epsilon = 25/255.0


# Load the test dataset
fTransform = transforms.Compose([
    transforms.ToTensor(), # Transform from [0,255] uint8 to [0,1] float
])

FASHION_test = datasets.FashionMNIST('.', download=True, train=False, transform=fTransform)


# Data loader
def get_data_loader():
    return DataLoader(FASHION_test, batch_size=1, shuffle=True)


# Create model
def load_mdl(ptf: str, op: str, obj, dropout: Optional[float] = None, group1: Optional[float] = None, group2: Optional[float] = None):
    mdl = LightNet(op, obj, 0.01, dropout, group1, group2)
    mdl.load_state_dict(torch.load(ptf))
    mdl = mdl.model.to(device)
    return mdl


# pgd attack
def pgd_test(mdl_name, model, loss, testloader, steps=1, alpha=1e-2):
    attacks = 0
    success = 0

    model = model.eval()

    for (x,y) in tqdm(testloader, desc="attack", leave=False):
        x = x.to(device)
        y = y.to(device)

        delta = torch.zeros_like(x, requires_grad=True)

        # Initial prediction
        y_p = model((x+delta))
        pred = y_p.argmax(1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum()

        # no need to attack if already incorrect
        if not correct:
            continue

        for i in range(steps):
            # Calculate loss
            l = loss(y_p, y)
            l.backward()

            # Calculate gradient
            grad = delta.grad.data

            delta.data = (delta + alpha*grad.sign()).clamp(-epsilon, epsilon)

            delta.grad.zero_()

            y_p = model((x+delta).clamp(0, 1))


        # Adversarial example
        pred = y_p.argmax(1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum()

        # Get result
        if not correct:
            success += 1

        attacks += 1

    acc = attacks/len(testloader)
    asr = success/attacks

    print(f"==={mdl_name}===")
    print(f"Accuracy: {acc:.3f}")
    print(f"Attack Success Rate ({steps}-step, alpha={alpha}): {success/attacks:.3f}")

    return asr


def objective(trial: optuna.trial.Trial, ptf, op, obj, params) -> float:
    # Optimize 
    steps = trial.suggest_int("steps", 2, 10, 2)  
    alpha = trial.suggest_int("alpha", 0, 2)

    loader = get_data_loader()
    model = load_mdl(ptf, op, obj, **params)
    loss = nn.CrossEntropyLoss().to(device)

    asr = pgd_test(ptf, model, loss, loader, steps, (2**alpha) * 1e-2)

    return asr



def attack_driver(ptf, op, obj, params):

    search_space = {
        "steps": [i*2 for i in range(1,6)],
        "alpha": [i for i in range(3)]
    }

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(lambda trial: objective(trial, ptf, op, obj, params), n_trials=1)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    operations = {
        "Baseline": nn.Identity,
        "Dropout": nn.Dropout,
        "BatchNorm": nn.BatchNorm2d,
        "LayerNorm": nn.LayerNorm,
        "InstanceNorm": nn.InstanceNorm2d,
        "GroupNorm": nn.GroupNorm
    }

    params = {
        "Baseline": {
            # "batch_size": 60,
            # "lr": 0.0017676870045475008,
        },

        "Dropout": {
            # "batch_size": 80,
            # "lr": 0.0024674588637756756,
            "dropout": 0.0028855256688067876,
        },
        "BatchNorm": {
            # "batch_size": 50,
            # "lr": 0.004123213018442183,
        },
        "LayerNorm": {
            # "batch_size": 60,
            # "lr": 0.0028386924443544624,
        },
        "InstanceNorm": {
            # "batch_size": 50,
            # "lr": 0.0024875881303965163,
        },
        "GroupNorm": {
            # "batch_size": 60,
            # "lr": 0.003574710344602632,
            "group1": 2,
            "group2": 1
        }
    }

    for op in operations:
        attack_driver(f"{op}_model_final.pt", op, operations[op], params[op])



