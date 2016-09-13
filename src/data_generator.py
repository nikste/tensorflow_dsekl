import numpy as np

import scipy as sp

import datetime
import tensorflow as tf
from sklearn import datasets
from sklearn.datasets import fetch_mldata
from sklearn.datasets import make_gaussian_quantiles
from sklearn.preprocessing import StandardScaler
from tensorflow.examples.tutorials.mnist import input_data
from numpy.random import multivariate_normal as mvn


def scale_data(x, x_test, with_mean=True, with_std=True):
    sc = StandardScaler(with_mean=with_mean, with_std=with_std).fit(x)
    x = sc.transform(x)
    x_test = sc.transform(x_test)
    return x, x_test


def subsample_data(x, y, x_test, y_test, scale_down_fac):
    x = x[0:int(x.shape[0]):scale_down_fac]
    y = y[0:int(y.shape[0]):scale_down_fac]
    x_test = x_test[0:int(x_test.shape[0]):scale_down_fac]
    y_test = y_test[0:int(y_test.shape[0]):scale_down_fac]
    return x, y, x_test, y_test


def convert_to_1_hot(in_data, max_val):
    a = np.array(in_data)
    out_data = np.zeros((len(in_data),max_val))
    out_data[np.arange(len(in_data)), a] = 1
    return out_data


def get_diabetes(train_test_ratio):
    diabetes = datasets.load_diabetes()
    x = diabetes.data
    y = diabetes.target

    y = np.reshape(y,(y.shape[0],1))
    cutoff = int(x.shape[0] * train_test_ratio)

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    x_train = x[0:cutoff, :]
    x_test = x[cutoff:, :]
    y_train = y[:cutoff]
    y_test = y[cutoff:]
    return x_train, y_train, x_test, y_test


def get_boston(train_test_ratio):
    boston = datasets.load_boston()
    x = boston.data
    y = boston.target

    y = np.reshape(y,(y.shape[0],1))
    cutoff = int(x.shape[0] * train_test_ratio)

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    x_train = x[0:cutoff, :]
    x_test = x[cutoff:, :]
    y_train = y[:cutoff]
    y_test = y[cutoff:]
    return x_train, y_train, x_test, y_test


def create_data_home():
    import os
    data_sets_dir = "./datasets"
    if not os.path.isdir(data_sets_dir):
        os.mkdir(data_sets_dir)
    return data_sets_dir


def get_covertype(train_test_ratio, binary=False):

    if binary:
        data_sets_dir = create_data_home()
        print "reading covertype data .. \n", datetime.datetime.now()
        dd = fetch_mldata('covtype.binary', data_home=data_sets_dir)
        print "read covertype data", datetime.datetime.now()
        print "densifying .. \n", datetime.datetime.now()
        xtotal = dd.data.todense()
        ytotal = dd.target
        print "densified"

        xtotal = xtotal
        ytotal = sp.sign(ytotal - 1.5)

        cutoff = int(xtotal.shape[0] * train_test_ratio)

        x = xtotal.astype(np.float32)
        y = ytotal.astype(np.float32)

        x_train = x[0:cutoff, :]
        x_test = x[cutoff:, :]
        y_train = np.expand_dims(y[0:cutoff], 1)
        y_test = np.expand_dims(y[cutoff:], 1)

    else:
        covertype = datasets.fetch_covtype()
        x = covertype.data

        y = convert_to_1_hot(covertype.target - 1, 7)
        cutoff = int(x.shape[0] * train_test_ratio)

        x = x.astype(np.float32)
        y = y.astype(np.float32)

        x_train = x[0:cutoff, :]
        x_test = x[cutoff:, :]
        y_train = y[0:cutoff]
        y_test = y[cutoff:]
    return x_train, y_train, x_test, y_test

def get_iris(train_test_ratio):
    iris = datasets.load_iris()
    x = iris.data
    y = convert_to_1_hot(iris.target, 3)

    cutoff = int(x.shape[0] * train_test_ratio)

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    x_train = x[0:cutoff, :]
    x_test = x[cutoff:, :]
    y_train = y[:cutoff]
    y_test = y[cutoff:]
    return x_train, y_train, x_test, y_test


def get_mnist_full(binary=False,train_test_ratio=1.-1/7.):
    custom_data_home = create_data_home()
    mnist = fetch_mldata('MNIST original', data_home=custom_data_home)

    x = mnist.data
    y = mnist.target

    if binary:
        mask = (y == 6) | (y == 8)
        y = y[mask]
        # svm only needs 2 labels
        y -= 7
        y = np.expand_dims(y, axis=1)
        x = x[mask]
    else:
        y = convert_to_1_hot(mnist.target, 10)

    cutoff = int(x.shape[0] * train_test_ratio)

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    rnd = np.random.permutation(y.shape[0])
    x = x[rnd]
    y = y[rnd]

    x = x / 255.
    x_train = x[0:cutoff, :]
    x_test = x[cutoff:, :]
    y_train = y[0:cutoff]
    y_test = y[cutoff:]
    return x_train, y_train, x_test, y_test


def get_mnist(train_test_ratio, binary=False):
    # mnist = tf.examples.tutorials.mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)
    digits = datasets.load_digits()
    x = digits.data

    if binary:
        y = digits.target
        mask = (y == 6) | (y == 8)
        y = y[mask]
        # svm only needs 2 labels
        y -= 7
        y = np.expand_dims(y, axis=1)
        x = x[mask]
        # y = convert_to_1_hot(y, 2)
    else:
        y = convert_to_1_hot(digits.target, 10)
    cutoff = int(x.shape[0] * train_test_ratio)

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    x_train = x[0:cutoff, :] / 16.
    x_test = x[cutoff:, :] / 16.
    y_train = y[0:cutoff]
    y_test = y[cutoff:]
    return x_train, y_train, x_test, y_test


def generate_noisy_polinomial_data(mu, sigma, n_samples, f0, f1, f2, low=-100., high=100. ):
    if mu == 0. and sigma == 0.:
        noise = np.zeros(n_samples)
    else:
        noise = np.random.normal(mu, sigma, n_samples)

    x = (high - low) * np.random.random_sample(n_samples) + low

    y = f0 * x**2 + f1 * x**1 + f2 + noise
    x = np.expand_dims(x, 1)
    y = np.expand_dims(y, 1)

    return x, y


def generate_noisy_linear_data(mu,sigma,n_samples, w, b, low=-100.,high=100.):
    if mu == 0. and sigma == 0.:
        noise = np.zeros(n_samples)
    else:
        noise = np.random.normal(mu, sigma, n_samples)
    x = (high - low) * np.random.random_sample(n_samples) + low

    y = x * w + b + noise

    x = np.expand_dims(x,1)
    y = np.expand_dims(y,1)

    return x, y


def make_data_xor(N=80,noise=.25):
    # generates some toy data
    mu = sp.array([[-1,1],[1,1]]).T
    C = sp.eye(2)*noise
    X = sp.hstack((mvn(mu[:,0],C,N/4).T,mvn(-mu[:,0],C,N/4).T, mvn(mu[:,1],C,N/4).T,mvn(-mu[:,1],C,N/4).T))
    Y = sp.hstack((sp.ones((1,N/2.)),-sp.ones((1,N/2.))))
    randidx = sp.random.permutation(N)
    Y = Y[0,randidx]
    X = X[:,randidx]
    return X, Y


def get_gaussian_xor(n_samples, noise=.25):
    x, y = make_data_xor(N=n_samples, noise=noise)
    x = x.T
    y = np.expand_dims(y.T, 1)
    x_test, y_test = make_data_xor(N=n_samples, noise=noise)
    x_test = x_test.T
    y_test = np.expand_dims(y_test.T, 1)

    return x, y, x_test, y_test


def get_gaussian(n_samples):
    cov_x = .1
    cov_y = .1

    m_x_1 = 1.
    m_y_1 = 1.
    m_x_2 = -1.
    m_y_2 = -1.
    x,y = make_gaussian_quantiles(mean=[m_x_1,m_y_1], cov=[cov_x, cov_y], n_samples=100, n_features=2, n_classes=1, shuffle=True,
                                  random_state=None)
    x2,y2 = make_gaussian_quantiles(mean=[m_x_2, m_y_2], cov=[cov_x, cov_y], n_samples=100, n_features=2, n_classes=1, shuffle=True,
                                  random_state=None)

    x = np.concatenate((x, x2), axis=0)
    y = np.concatenate(((y - 1), y2), axis=0)
    x_test, y_test = make_gaussian_quantiles(mean=[m_x_1,m_y_1], cov=[cov_x, cov_y], n_samples=100, n_features=2, n_classes=1, shuffle=True,
                                  random_state=None)
    x_test2, y_test2 = make_gaussian_quantiles(mean=[m_x_2, m_y_2], cov=[cov_x, cov_y], n_samples=100, n_features=2, n_classes=1, shuffle=True,
                                  random_state=None)
    x_test = np.concatenate((x_test, x_test2))
    y_test = np.concatenate(((y_test - 1), y_test2))

    y = np.expand_dims(np.sign(y + .5), 1)
    y_test = np.expand_dims(np.sign(y_test + .5), 1)

    return x, y, x_test, y_test


if __name__ == '__main__':
    mnist = get_mnist()