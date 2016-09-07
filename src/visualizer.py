import matplotlib.pyplot as plt
import numpy as np


def visualize_gaussians(x, y):
    minus = np.where(y == -1)
    plus = np.where(y == 1)
    plt.scatter(x[plus,0], x[plus,1],color='red')
    plt.scatter(x[minus,0], x[minus,1], color='blue')
    plt.show()


def visualize_mnist(x, y):
    for i in range(0, y.shape[0]):
        # The first column is the label
        label = y[i]
        # The rest of columns are pixels
        pixels = x[i]

        # Make those columns into a array of 8-bits pixels
        # This array will be of 1D with length 784
        # The pixel intensity values are integers from 0 to 255


        # Reshape the array into 28 x 28 array (2-dimensional array)
        pixels = pixels.reshape((28, 28))

        # Plot
        plt.title('Label is {label}'.format(label=label))
        plt.imshow(pixels, cmap='gray')
        plt.show()