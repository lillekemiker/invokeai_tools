import argparse

import torch
from safetensors.torch import save_file


def get_args():
    parser = argparse.ArgumentParser(
        description="Script for converting .pt format TI models to .safetensors"
    )
    parser.add_argument("--pt-file", required=True)
    parser.add_argument("--save-to", required=True)
    return parser.parse_args()


def main():
    args = get_args()
    pt_dict = torch.load(args.pt_file, map_location="cpu")
    safe_dict = {"emb_params": pt_dict["string_to_param"]["*"]}
    save_file(safe_dict, args.safe_to)


if __name__ == "__main__":
    main()
