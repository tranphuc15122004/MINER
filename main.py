import argparse
import sys

import arguments
from src import utils
from src.trainer import Trainer


def _train(args):
    trainer = Trainer(args)
    trainer.train()


def _eval(args):
    trainer = Trainer(args)
    trainer.eval()


def _submission(args):
    trainer = Trainer(args)
    trainer.submission_generator('rank')


def main():
    parser = argparse.ArgumentParser(description='Arguments for Miner model', fromfile_prefix_chars='@',
                                     allow_abbrev=False)
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args
    subparsers = parser.add_subparsers(dest='mode', help='Mode of the process: train, eval or submission')

    train_parser = subparsers.add_parser('train', help='Training phase')
    arguments.add_train_arguments(train_parser)
    eval_parser = subparsers.add_parser('eval', help='Evaluation phase')
    arguments.add_eval_arguments(eval_parser)
    submission_parser = subparsers.add_parser('submission', help='Submission generation phase')
    arguments.add_submission_arguments(submission_parser)

    args = parser.parse_args()
    if args.mode == 'train':
        _train(args)
    elif args.mode == 'eval':
        _eval(args)
    elif args.mode == 'submission':
        _submission(args)


if __name__ == '__main__':
    main()
