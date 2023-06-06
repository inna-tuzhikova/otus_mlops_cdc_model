from pathlib import Path
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from train import config
from train.model import prepare
from train.dataset import get_data_loaders


def main():
    parser = ArgumentParser('Starts Cat and Dog Classifier training')
    parser.add_argument(
        'dataset_path',
        type=Path,
        help='Path to dataset with categories image folders'
    )
    args = parser.parse_args()
    assert args.dataset_path.is_dir(), f'DS not found: {args.dataset_path}'

    train(args.dataset_path)


def train(dataset_path: Path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, criterion, optimizer = prepare()
    model.to(device)
    train_loader, test_loader = get_data_loaders(dataset_path)
    for epoch in range(config.N_EPOCHS):
        losses = []
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, targets) in loop:
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = model(data)

            loss = criterion(scores, targets)
            optimizer.zero_grad()
            losses.append(loss)
            loss.backward()
            optimizer.step()
            # _, preds = torch.max(scores, 1)
            # current_loss += loss.item() * data.size(0)
            # current_corrects += (preds == targets).sum().item()
            # accuracy = int(current_corrects / len(train_loader.dataset) * 100)
            loop.set_description(
                f'Epoch {epoch + 1}/{config.N_EPOCHS} '
                f'process: {int((batch_idx / len(train_loader)) * 100)}'
            )
            loop.set_postfix(loss=loss.data.item())
        torch.save(
            dict(
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict()
            ),
            f'checkpoint_epoch_{epoch}.pt'
        )
    test(model, criterion, test_loader, device)


def test(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            _, predictions = torch.max(output, 1)
            correct += (predictions == y).sum().item()
            test_loss = criterion(output, y)

    test_loss /= len(test_loader.dataset)
    accuracy = int(correct / len(test_loader.dataset) * 100)
    print(
        f'Average Loss: {test_loss}\n'
        f'Accuracy: {accuracy}%, ({correct} / {len(test_loader.dataset)})'
    )


if __name__ == '__main__':
    main()
