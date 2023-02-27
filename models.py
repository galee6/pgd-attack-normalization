"""Models"""

# !pip install pytorch-lightning
# !pip install optuna

# General imports
from typing import List, Optional
import random
import numpy as np

# Optuna
import optuna

# Pytorch lightning
import pytorch_lightning as pl

# torch
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Random Seed
SEED = 1443

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# Stable (hyper-)parameters
CLASSES = 10
LOSS = nn.CrossEntropyLoss()
convs = [
            {
                "in_channels": 1,
                "out_channels": 6,
                "kernel_size": 5
            },
            {
                "in_channels": 6,
                "out_channels": 16,
                "kernel_size": 5
            }
]

fcs = [
        (16*4*4, 120),
        (120, 84),
        (84, CLASSES)
]

# Utils
def _create_convs(convs, op, obj, dropout: float, group1: int, group2: int):
    layers = []
    
    if op == "Baseline":
        adds = [
            obj(),
            obj()
        ]

    elif op == "Dropout":
        adds = [
            obj(p=dropout),
            obj(p=dropout)
        ]
    
    elif op == "LayerNorm":
        adds = [
            obj([6,12,12]),
            obj([16,4,4])
        ]
    
    elif op == "InstanceNorm" or op == "BatchNorm":
        adds = [
            obj(6),
            obj(16)
        ]

    elif op == "GroupNorm":
        adds = [
            obj(group1, 6),
            obj(group2, 16)
        ] 
    
    else:
        print("Error")
        exit()


    for i, conf in enumerate(convs):
        block = [
            nn.Conv2d(**conf),
            nn.MaxPool2d(kernel_size=2),
            adds[i],
            nn.ReLU()
        ]
    
        layers.extend(block)
    
    return layers


def _create_fcs(fcs):
    layers = []
    for conf in fcs:
        block = [
            nn.Linear(*conf),
            nn.ReLU()
        ]
    
        layers.extend(block)
        
        # Drop last activation
        del layers[-1]
    
    return layers

# LeNet for MNIST
class LeNet(nn.Module):
    def __init__(self, op, obj, dropout: float, group1: int, group2: int) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(*(_create_convs(convs, op, obj, dropout, group1, group2)))
        self.fc_layers = nn.Sequential(*(_create_fcs(fcs)))
        

    def forward(self, data: torch.Tensor) -> torch.Tensor:  # x = [batch size, 1, 28, 28]

        conv = self.conv_layers(data)

        flattened = conv.view(conv.shape[0], -1)  # x = [batch size, 16*4*4 = 256]

        logits = self.fc_layers(flattened)

        return logits


# Lightning
class LightNet(pl.LightningModule):
    def __init__(self, op, obj, lr: float, dropout: float, group1: int, group2: int) -> None:
        super().__init__()
        self.model = LeNet(op, obj, dropout, group1, group2)
        self.lr = lr

        # print(self.model)

    def forward(self, data: torch.Tensor) -> torch.Tensor():
        return self.model(data)
    

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data)
        return LOSS(output, target)
    

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        data, target = batch
        output = self(data)
        pred = output.argmax(1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum()
        accuracy = correct.float() / target.shape[0]

        self.log("val_acc", accuracy)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters(), lr=self.lr)

# FashionMNIST Transform
fTransform = transforms.Compose([
    transforms.ToTensor(), # Transform from [0,255] uint8 to [0,1] float
])

# FashionMNIST
class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
    
    def setup(self, stage: Optional[str] = None) -> None:
        mnist_full = datasets.FashionMNIST(
            ".", train=True, download=True, transform=fTransform
        )

        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        self.mnist_test = datasets.FashionMNIST(
            ".", train=False, download=True, transform=fTransform
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )

# Hyperparameter tuning
def objective(trial: optuna.trial.Trial, op: str, obj) -> float:
    # Optimize 
    batch_size = trial.suggest_int("batch_size", 10, 100, 10)  # between 10 and 100, step 10
    lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0, 1)
    group1 = trial.suggest_int("group1", 2, 3)
    group2 = trial.suggest_int("group2", 1, 3)

    model = LightNet(op, obj, lr, dropout, group1, 2**group2)
    datamodule = FashionMNISTDataModule(batch_size)

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=10,
        accelerator="auto",
        devices=1
    )

    hyperparameters = dict(batch_size=batch_size, lr=lr, dropout=dropout, group1=group1, group2=group2)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    torch.save(model.state_dict(), f"{op}_model{trial.number}.pt")

    return trainer.callback_metrics["val_acc"].item()


# Optimize driver function
def optimize(op: str, obj):
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(lambda trial: objective(trial, op, obj), n_trials=20)

    log_file = open("model_outputs.txt", "a")

    log_file.write(f"==={op}===\n")

    log_file.write(f"Number of finished trials: {len(study.trials)}\n")

    log_file.write("Best trial:\n")
    trial = study.best_trial

    log_file.write(f"  Value: {trial.value}")

    log_file.write("  Params: \n")
    for key, value in trial.params.items():
        log_file.write(f"    {key}: {value}\n")
    
    log_file.close()


# Loop for all
if __name__ == "__main__":
    operations = {
        "Baseline": nn.Identity,
        "Dropout": nn.Dropout,
        "BatchNorm": nn.BatchNorm2d,
        "LayerNorm": nn.LayerNorm,
        "InstanceNorm": nn.InstanceNorm2d,
        "GroupNorm": nn.GroupNorm
    }

    for op, obj in operations.items():
        optimize(op, obj)
    

