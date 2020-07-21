import argparser
import shutil
import os

from submission.mask_to_submission import masks_to_submission


def log_submission(submission_identifier, args):
    # Initializing out directory and submission output path
    out_dir = os.path.join('..', 'out')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Initializing submission path where submission will be logged
    submission_path = os.path.join(out_dir, submission_identifier)
    if not os.path.exists(submission_path):
        os.mkdir(submission_path)
    print('Logging submission: ' + submission_path)

    # Copying model weights
    shutil.copy2(args.model_path, os.path.join(submission_path, args.model + '.h5'))

    # Copying results directories
    shutil.copytree(os.path.join(args.test_path, 'results', 'discrete'), os.path.join(submission_path, 'discrete'))
    shutil.copytree(os.path.join(args.test_path, 'results', 'continuous'), os.path.join(submission_path, 'continuous'))
    #shutil.copytree(os.path.join(args.test_path, 'results', 'post_processing'), os.path.join(submission_path, 'post_processing'))

    # Copying tensorboard log files
    if args.train_model:
        shutil.copytree(os.path.join('..', 'logs', 'fit', submission_identifier), os.path.join(submission_path, 'tensorboard'))

    # Saving argument config file
    argparser.write_config_file(args, path=os.path.join(submission_path, 'config.conf'))

    # Masking result to kaggle submission format and saving file it as csv file
    result_path = os.path.join(submission_path, 'discrete')
    submission_filename = os.path.join(submission_path, 'submission_thresh{}.csv'.format(args.sub_thresh))
    image_filenames = []
    for file in os.listdir(result_path):
        image_filename = os.path.join(result_path, file)
        #print(image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames, foreground_threshold=args.sub_thresh)
