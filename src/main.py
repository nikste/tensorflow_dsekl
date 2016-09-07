from src.data_generator import get_covertype, get_mnist, get_iris, get_diabetes


# get data
dataset = 'covertype'

train_test_ratio = 0.5
if dataset == 'mnist':
    x, y, x_test, y_test = get_mnist(train_test_ratio)
elif dataset == 'iris':
    x, y, x_test, y_test = get_iris(train_test_ratio)
elif dataset == 'diabetes':
    x, y, x_test, y_test = get_diabetes(train_test_ratio)
elif dataset == 'covertype':
    x, y, x_test, y_test = get_covertype(train_test_ratio)
# start training


# evaluate