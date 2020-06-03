import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PNG Noise Generator for MuJoCo height fields')
    parser.add_argument('--filename', '-f', action='store', default='noise',
                        help='Name of file output. '
                             'File will be saved as a PNG file outside of the folder this is located in'
                             '(usage: -f <filename>)')
    parser.add_argument('--dimension', '-d', type=int, nargs='+', default=(32, 32),
                        help='Size of the 2D array (usage: -d <dim1> <dim2>)')
    parser.add_argument('--granularity', '-g', type=int, default=100,
                        help='How fine or course the noise is. '
                             'The larger the number, the finer the noise (usage: -g <int>)')
    parser.add_argument('--start_size', '-s', type=int, default=2,
                        help='The middle of the map will be always flat for starting.'
                             'Choose how big this block size will be (usage: -s <int>)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set seed for reproducible maps (usage: --seed <int>)')

    args = parser.parse_args()

    if args.seed:
        np.random.seed(args.seed)

    midpoint = (int(args.dimension[0] / 2), int(args.dimension[1] / 2))

    # build noisy array
    terrain = np.random.randint(args.granularity, size=args.dimension)

    terrain[midpoint[0] - args.start_size:midpoint[0] + args.start_size,
    midpoint[1] - args.start_size:midpoint[1] + args.start_size] = 0

    # save as png file
    plt.imsave('../{}.png'.format(args.filename), terrain, cmap='gray')
