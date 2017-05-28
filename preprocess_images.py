"""Preprocesses images. Preprocessing images before training the CNN will reduce training time."""
import argparse
import os

from PIL import Image

from utils import load_image, preprocess


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        i   - Required  : current i (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == '__main__':
    print('\nPreprocessing images...\n')
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data img_dir', dest='data_dir', type=str, default='data')
    args = parser.parse_args()

    # print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    img_dir = os.path.join(args.data_dir, 'IMG')

    # count total number of images
    img_count = sum(os.path.isfile(os.path.join(img_dir, f)) for f in os.listdir(img_dir))

    i = 1
    for filename in os.listdir(img_dir):
        img = load_image(img_dir, filename)
        img = preprocess(img)
        img = Image.fromarray(img)
        img.save(os.path.join(img_dir, filename))
        printProgressBar(iteration=i, total=img_count, decimals=1, length=20, suffix='\t' + filename)
        i += 1
