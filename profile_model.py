import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from fvcore.nn import FlopCountAnalysis

from models.main_model import ChangeDetection


def main():
    parser = argparse.ArgumentParser(description="Profile FSG-Net parameters and FLOPs.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model_args = SimpleNamespace(
        dual_label=False,
        backbone="resnet18",
        neck="fpn+aspp+fuse+drop",
        head="fcn",
        input_size=max(args.height, args.width),
        pretrain="",
    )
    device = torch.device(args.device)
    model = ChangeDetection(model_args).eval().to(device)
    image_a = torch.randn(1, 3, args.height, args.width, device=device)
    image_b = torch.randn(1, 3, args.height, args.width, device=device)
    inputs = (image_a, image_b)

    flop_error = None
    unsupported = {}
    try:
        analysis = FlopCountAnalysis(model, inputs)
        flops = analysis.total()
        unsupported = dict(Counter(analysis.unsupported_ops()))
    except Exception as exc:
        flops = None
        flop_error = f"{type(exc).__name__}: {exc}"

    params = sum(p.numel() for p in model.parameters())
    result = {
        "model": "FSG-Net-ResNet18",
        "input": "two RGB images, batch=1",
        "image_size": [args.height, args.width],
        "params_m": params / 1_000_000,
        "parameters_total": params,
        "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "flops_g": flops / 1_000_000_000 if flops is not None else None,
        "fvcore_flops_per_pair": flops,
        "fvcore_convention": "one multiply-add is counted as one FLOP",
        "unsupported_ops": unsupported,
        "fvcore_error": flop_error,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2)


if __name__ == "__main__":
    main()
