from pathlib import Path
from argparse import ArgumentParser

import torch

from train.model import prepare_model
from train.dataset import get_inference_transform
from inference.predictor import Predictor


def main():
    parser = ArgumentParser('Converts model to executable script')
    parser.add_argument(
        'model_path',
        type=Path,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        'output_path',
        type=Path,
        help='Path to save executable script'
    )
    args = parser.parse_args()
    assert args.model_path.is_file(), f'Model not found: {args.model_path}'
    prepare_model(args.model_path, args.output_path)


def prepare_model(model_path: Path, output_path: Path):
    device = torch.device('cpu')
    model = prepare_model()
    transform = get_inference_transform()

    model.load_state_dict(torch.load(model_path))
    model.to(device)

    predictor = Predictor(model, transform).to(device)
    scripted_predictor = torch.jit.script(predictor).to(device)
    scripted_predictor.save(str(output_path))


if __name__ == '__main__':
    main()
