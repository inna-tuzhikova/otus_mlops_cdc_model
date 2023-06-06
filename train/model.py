import torch.nn as nn
from torchvision import models
import torch.optim as optim

from train import config


def prepare():
    model = prepare_model()
    criterion = get_criterion()
    optimizer = get_optimizer(model)
    return model, criterion, optimizer


def prepare_model():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, 2)
    return model


def get_criterion():
    return nn.CrossEntropyLoss()


def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=config.LR)
