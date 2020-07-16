from configargparse import ArgumentParser
import argparse

def get_parser():
    parser = ArgumentParser(description='Road Segmentation CILlitbang',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add('-c', '--config', is_config_file=True, help='config file path')

    parser.add('--epochs', type=int, required=False, default=50,
               help='number of training epochs')
    parser.add('--steps', type=int, required=False, default=100,
               help='number of steps per epoch')

    parser.add('--predict-best', type=bool, default=True,
               help='predict with best model weights')
    parser.add('--train-model', type=bool, default=True,
               help='perform training on the model')
    parser.add('--comb-pred', type=bool, default=True,
               help='combine different rotated and flipped predictions into one')

    parser.add('--train-path', type=str, default='../data/training/',
               help='path containing training images & groundtruth')
    parser.add('--val-path', type=str, default='../data/validation/',
               help='path containing validation images & groundtruth')
    parser.add('--test-path', type=str, default='../data/test/',
               help='path containing test images to predict')
    parser.add('--model-path', type=str, default='../tmp/model.h5',
               help='path where the current best model weights will be stored')

    return parser
