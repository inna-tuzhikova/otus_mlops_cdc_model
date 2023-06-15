from pathlib import Path
from argparse import ArgumentParser

import torch

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
    prepare_executable_script(args.model_path, args.output_path)


def prepare_executable_script(model, output_path: Path) -> None:
    device = torch.device('cpu')
    model.to(device)

    transform = get_inference_transform()

    predictor = Predictor(model, transform).to(device)
    scripted_predictor = torch.jit.script(predictor).to(device)
    scripted_predictor.save(str(output_path))


if __name__ == '__main__':
    main()
