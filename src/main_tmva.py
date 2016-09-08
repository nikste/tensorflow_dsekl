import numpy as np

import datetime
import tensorflow as tf



# import ROOT
#
# # Generates two 2D normal distributions in two TTrees.
# signal = ROOT.TNtuple("ntuple", "ntuple", "x:y:signal")
# background = ROOT.TNtuple("ntuple", "ntuple", "x:y:signal")
# for i in range(200):
#     signal.Fill(ROOT.gRandom.Gaus(1, 1), ROOT.gRandom.Gaus(1, 1), 1)
#     background.Fill(ROOT.gRandom.Gaus(-1, 1), ROOT.gRandom.Gaus(-1, 1), -1)
#
# # Draws the distribution.
# gcSaver = []
# gcSaver.append(ROOT.TCanvas())
# histo = ROOT.TH2F("histo", "", 1, -5, 5, 1, -5, 5)
# histo.Draw()
# signal.SetMarkerColor(ROOT.kRed)
# signal.Draw("y:x", "signal > 0", "same")
# background.SetMarkerColor(ROOT.kBlue)
# background.Draw("y:x", "signal < 0", "same")
#
# # Reads the TTrees into our data structure using {-1, 1} labels instead of
# # a one-hot matrix.
# binary = tmva.preprocessing.ttree.ttrees_to_internal(
#     [signal, background], ["x", "y"], binary=True)
#

#

#
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

from data_generator import get_mnist, get_gaussian, get_gaussian_xor, get_covertype, scale_data, get_mnist_full

from src.models.tf_svm_new import train_svm
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
# x, y, x_test, y_test = get_mnist_full(binary=True)
# x, y, x_test, y_test = get_mnist(0.9, binary=True)

# x2, y2, x_test2, y_test2 = get_mnist(0.9, binary=True)
x, y, x_test, y_test = get_covertype(0.9, binary=True)

x, x_test = scale_data(x, x_test, with_mean=True, with_std=True)
print "loaded", x.shape[0], "datapoints"
print "scaling data", datetime.datetime.now()
# x, y, x_test, y_test = scale_data(x, y, x_test, y_test, with_mean=True, with_std=True)
print "scaled data", datetime.datetime.now()
# x, y, x_test, y_test = prepare_covertype(x, y, x_test, y_test, scale=False)
# x, y, x_test, y_test = get_gaussian(1000)
# x, y, x_test, y_test = get_gaussian_xor(1000, noise=.3)
# visualize_gaussians(x,y)
# visualize_gaussians(x_test,y_test)
# visualize_mnist(x, y)

# half training set
# x = np.concatenate((x, x, x, x, x, x, x, x, x))
# x = np.concatenate((x, x))
# x = np.concatenate((x, x))
# y = np.concatenate((y, y, y, y, y, y, y, y, y))
# y = np.concatenate((y, y))
# y = np.concatenate((y, y))

# batch_size = 10000
# n_pred = 100
# n_exp = 100
# covertype

# x = x[:int(x.shape[0] * 0.1)]
# y = y[:int(x.shape[0] * 0.1)]
# x_test = x_test[:int(x_test.shape[0] * 0.1)]
# y_test = y_test[:int(y_test.shape[0] * 0.1)]
scale_down_fac = 10
x = x[0:int(x.shape[0]):scale_down_fac]
y = y[0:int(y.shape[0]):scale_down_fac]
x_test = x_test[0:int(x_test.shape[0]):scale_down_fac]
y_test = y_test[0:int(y_test.shape[0]):scale_down_fac]
print "x.shape[0]", x.shape[0]
print "y.shape[0]", y.shape[0]
print "x_test.shape[0]", x_test.shape[0]
print "y_test.shape[0]", y_test.shape[0]

# visualize_mnist(x, y)
# visualize_mnist(x_test, y_test)
# x = x[0:int(x.shape[0])]
# y = y[0:int(x.shape[0])]
# x_test = x_test[0:int(x_test.shape[0])]
# y_test = y_test[0:int(y_test.shape[0])]
# n_pred = 10000
# n_exp = 10000
# gamma = 0.01
# C = .1
n_pred = 10000#y.shape[0] #y.shape[0]#15000#100#
print "n_pred", n_pred
n_exp = 10000#y.shape[0] #y.shape[0]#15000#100#
print "n_exp", n_exp
stupidclassifier = reference_classifier(x, y)
# n_pred = y.shape[0]
# n_exp = y.shape[0]
# C_range = np.logspace(-2, 3, 6)
# print "c_range", C_range
# gamma_range = np.logspace(-2, 3, 6)
# print "gamma_range", gamma_range
# for C_scaled in C_range:
#     for gamma in gamma_range:
#         print C_scaled, gamma
#         train_svm(x, y, x_test, y_test, C=C_scaled, gamma=gamma, n_pred=n_pred, n_exp=n_exp, learning_rate_start=1., specified_test_set=None,
#                                             with_replacement=False, most_common_label=stupidclassifier, log_prefix="/mnist_l2_" + str(gamma) + "_" + str(C_scaled))


gamma = 1.
C_scaled = 1
# gamma = 1.
# C_scaled = 1.
train_svm(x, y, x_test, y_test, nIter=20000, C=C_scaled, gamma=gamma, n_pred=n_pred, n_exp=n_exp, learning_rate_start=-1., ordered_batches=False,
                                    with_replacement=False, most_common_label=stupidclassifier, log_prefix="/covertype_l2_10000_sorted_part10")


# svc = SVC(C=0.1, gamma=0.01)
# svc.fit(x, y)
# yy_test = svc.predict(x_test)
#
# c = 0
# correct = 0
# for xx,yy in zip(y_test, yy_test):
#     c += 1
#     if xx == yy:
#         correct += 1
#
# print "total:", correct, c, correct / float(c)
# C_range = np.logspace(-2, 10, 13)
# print "c_range", C_range
# gamma_range = np.logspace(-9, 3, 13)
# print "gamma_range", gamma_range
# param_grid = dict(gamma=gamma_range, C=C_range)
# cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
# grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
# grid.fit(x, y)
# print "grid",grid
#
# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))



