import os
import numpy as np
import matplotlib.pyplot as plt

assert os.path.isdir('./../images')
if not os.path.isdir('./../images/figure_2'):
    os.mkdir('./../images/figure_2')
x = np.linspace(0, 10, 1000)
for period in np.arange(1, 10):
    fig = plt.figure(figsize=(7, 3))
    plt.plot(x, np.sin(2 * np.pi * x / period))
    plt.xlim(x.min(), x.max())
    plt.ylim(-1, 1)
    plt.grid()
    plt.savefig('./../images/figure_2/period_%06i.svg'%period)
