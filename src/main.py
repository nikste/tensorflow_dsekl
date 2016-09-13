from data_generator import get_mnist, get_gaussian, get_gaussian_xor, get_covertype, scale_data, get_mnist_full, \
    subsample_data

from src.models.tf_svm import train_svm
from src.visualizer import visualize_gaussians, visualize_mnist

import collections, numpy
def reference_classifier(x, y):
    unique, counts = numpy.unique(y, return_counts=True)
    print counts
    print unique
    negs = counts[0]
    poss = counts[1]
    if poss > negs:
        return 1
    else:
        return -1

def prepare_covertype(x, y, x_test, y_test, scale=True):
    n_half = int(1 * x.shape[0])
    print "using only:", n_half, "samples"
    x = x[:n_half, :]
    y = y[:n_half, :]
    n_half = int(1 * x_test.shape[0])
    x_test = x_test[:n_half, :]
    y_test = y_test[:n_half, :]
    if scale:
        print "scaling"
        x, y, x_test, y_test = scale_data(x, y, x_test, y_test)
        print "scaled"
    return x, y, x_test, y_test




dataset = "mnist"
scaling = False
subsample = 1
# get and prepare data
if dataset == "mnist":
    x, y, x_test, y_test = get_mnist_full(binary=True)
elif dataset == "mnist_sklearn":
    x, y, x_test, y_test = get_mnist(0.9, binary=True)
elif dataset == "covertype":
    x, y, x_test, y_test = get_covertype(0.9, binary=True)

if scaling:
    x, x_test = scale_data(x, x_test, with_mean=True, with_std=True)
if subsample > 1:
    x, y, x_test, y_test = subsample_data(x, y, x_test, y_test, scale_down_fac=10)


# reference classifier classifies as most common label in the training set
stupidclassifier = reference_classifier(x, y)


# parameters
nIter = 20000
C_scaled = 1
gamma = 1.
n_pred = 10000
n_exp = 10000
# negative values will be discounted, positive remain the same
learning_rate_start = -1.
# trains on all datapoints in a structured manner
ordered_batches = False
# samples with replacement
with_replacement = False
log_prefix = "/dataset_" + dataset + "_scaling_" + str(scaling) +        \
             "_subsample_" + str(subsample) + "_nIter_" + str(nIter) +  \
             "_C_" + str(C_scaled) + "_gamma_" + str(gamma) +           \
             "_n_pred_" + str(n_pred) + "_n_exp_" + str(n_exp) +        \
             "_lr_" + str(learning_rate_start) +                        \
             "_ordered_" + str(ordered_batches) + "_replacement_" + str(with_replacement)
train_svm(x, y, x_test, y_test, nIter=nIter, C=C_scaled, gamma=gamma, n_pred=n_pred, n_exp=n_exp, learning_rate_start=learning_rate_start, ordered_batches=ordered_batches,
                                    with_replacement=with_replacement, most_common_label=stupidclassifier, log_prefix=log_prefix)




