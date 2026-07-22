"""Provide CLI entrypoints for project demos."""

import logging
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from demos.basic_hand_draw import basic_hand_draw_demo
from demos.label_dataset import label_dataset_demo
from demos.make_dataset import make_dataset_demo

logging.basicConfig(level=logging.INFO)

OPERATIONS = [
    {
        "name": "Basic Hand Draw Demo",
        "handler": basic_hand_draw_demo,
        "help": "Launch a basic hand landmark drawing demo",
        "slug": "base_draw"
    },
    {
        "name": "Make Dataset Demo",
        "handler": make_dataset_demo,
        "help": "Create the dataset for RPS",
        "slug": "make_dataset"
    },
    {
        "name": "Label Dataset Demo",
        "handler": label_dataset_demo,
        "help": "Label the dataset for RPS",
        "slug": "label_dataset"
    }
]

cli = ArgumentParser(
    description=f'Choose operation to perform from the following\n{"".join([f"{idx+1}. {op["help"]}\n" for idx, op in enumerate(
        OPERATIONS)])}',
    formatter_class=RawDescriptionHelpFormatter
)
cli.add_argument(
    "--operation",
    required=True,
    choices=[op["slug"] for op in OPERATIONS],
    help=f"Choose operation to perform"
)

if __name__ == "__main__":
    args = cli.parse_args()
    operation = args.operation
    logging.info(f"Captured operation : {operation}")
    handler = None

    for op in OPERATIONS:
        if op["slug"] == operation:
            logging.info(f"Launching {op["name"]}...")
            handler = op["handler"]
            break

    if handler is not None:
        handler()
