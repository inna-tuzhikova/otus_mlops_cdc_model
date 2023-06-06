import torch
import torch.nn.functional as F
import torch.nn as nn


class Predictor(nn.Module):
    """Helper class to encapsulate image class prediction"""

    def __init__(self, model, transform):
        super().__init__()
        self._model = model.eval()
        self._transform = transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self._transform(x)
            y = self._model(x)
            probs = F.softmax(y, dim=1)
            return probs
