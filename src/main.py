import argparse
import sys
from pathlib import Path

import torch
from models.transformers_model import bert_base
from torch import nn, optim

save_path = str(Path().resolve()) + "/savedmodels"

from predict_model import predict_model
from train_model import train_model


class TrainOREvaluate(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        train_model()

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        # parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        # args = parser.parse_args(sys.argv[2:])
        # print(args)

        predict_model()

    def visualize(self):
        pass


if __name__ == "__main__":
    TrainOREvaluate()
