from configargparse import ArgumentParser
import argparse

def get_parser():
    parser = ArgumentParser(description='Road Segmentation CILlitbang',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add('-c', '--config', is_config_file=True, help='config file path')

    parser.add('--epochs', type=int, default=50,
               help='number of training epochs')
    parser.add('--steps', type=int, default=100,
               help='number of steps per epoch')
    parser.add('--val-steps', type=int, default=10,
               help='number of validation steps after each epoch')
    parser.add('--batch-size', type=int, default=4,
               help='batch size for training')
    parser.add('--seed', type=int, default=1,
               help='random seed for training (necessary for reproducibility)')

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

    parser.add('--rotation-range', type=int, default=360,
               help='rotation range to use for augmentation dictionary')
    parser.add('--width-shift-range', type=float, default=0.05,
               help='range for width shift in augmentation dictionary')
    parser.add('--height-shift-range', type=float, default=0.05,
               help='range for height shift in augmentation dictionary')
    parser.add('--shear-range', type=float, default=0.05,
               help='shear range for augmentation dictionary')
    parser.add('--zoom-range', type=float, default=0.05,
               help='zoom range for augmentation dictionary')
    parser.add('--horizontal-flip', type=bool, default=True,
               help='lets ImageDataGenerator flip images horizontally')
    parser.add('--vertical-flip', type=bool, default=True,
               help='lets ImageDataGenerator flip images vertically')
    parser.add('--fill-mode', type=str, default='reflect',
               help='fill mode when shifting images')

    parser.add('--model', choices=['unet', 'unet_dilated1', 'unet_dilated2'], default='unet_dilated2',
               help='which model to use for training')
    parser.add('--adam-lr', type=float, default=1e-4,
               help='learning rate of adam to use during training')

    return parser
