import os
import numpy as np
import matplotlib.pyplot as plt

# We're going to make a single 'local' image:
assert os.path.isdir('./../images')
if not os.path.isdir('./../images/figure_3'):
    os.mkdir('./../images/figure_3')
# ...and a bunch of images (for an interactive figure) in a separate
# directory:
if not os.path.isdir('./../../big_images'):
    os.mkdir('./../../big_images/')
if not os.path.isdir('./../../big_images/figure_3'):
    os.mkdir('./../../big_images/figure_3')
x = np.linspace(0, 10, 1000)
for period in np.arange(1, 10):
    fig = plt.figure(figsize=(7, 3))
    plt.plot(x, np.sin(2 * np.pi * x / period))
    plt.xlim(x.min(), x.max())
    plt.ylim(-1, 1)
    plt.grid()
    if period == 4:
        # Save one local placeholder for the interactive figure
        plt.savefig('./../images/figure_3/period_%06i.svg'%period)
    # Save the rest of the interactive figure in a separate directory:
    plt.savefig('./../../big_images/figure_3/period_%06i.svg'%period)
