from pathlib import Path
from argparse import ArgumentParser
import os

import torch
from tqdm import tqdm
import mlflow.pytorch
import mlflow
from dotenv import load_dotenv

from train import config
from train.model import prepare
from train.dataset import get_data_loaders
from inference.run import prepare_executable_script

load_dotenv()
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
mlflow.set_experiment('cdc_baseline')


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

    mlflow.log_params(dict(
        N_EPOCHS=config.N_EPOCHS,
        LR=config.LR,
        BATCH_SIZE=config.BATCH_SIZE,
        TEST_SIZE=config.TEST_SIZE,
        RANDOM_STATE=config.RANDOM_STATE,
    ))

    for epoch in range(config.N_EPOCHS):
        model.train()

        loop = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f'Epoch {epoch + 1}/{config.N_EPOCHS}'
        )

        current_loss = 0
        current_correct = 0

        for data, targets in loop:
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = model(data)

            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.data.item())

            _, predictions = torch.max(scores, 1)
            current_loss += loss.item() * data.size(0)
            current_correct += (predictions == targets).sum().item()

        train_loss = current_loss / len(train_loader.dataset)
        train_accuracy = current_correct / len(train_loader.dataset)

        test_loss, test_accuracy = test(model, criterion, test_loader, device)
        print(
            f'TOTAL. '
            f'Train: loss {train_loss:0.4f}, acc {train_accuracy:0.4f}. '
            f'Test: loss {test_loss:0.4f}, acc {test_accuracy:0.4f}'
        )
        mlflow.log_metrics(
            metrics=dict(
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                test_loss=test_loss,
                test_accuracy=test_accuracy,
            ),
            step=epoch
        )

    output_path = Path('scripted_model.pt')
    # To save scripted model as a model artifact
    inner_artifact_path = 'model_meta'

    prepare_executable_script(model, output_path)
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=inner_artifact_path
    )
    mlflow.log_artifact(str(output_path), inner_artifact_path)


def test(model, criterion, test_loader, device):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            _, predictions = torch.max(output, 1)
            correct += (predictions == y).sum().item()
            loss += criterion(output, y).item() * x.size(0)

    loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return loss, accuracy


if __name__ == '__main__':
    main()
