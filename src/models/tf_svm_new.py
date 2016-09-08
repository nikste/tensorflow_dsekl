import datetime
import tensorflow as tf
import numpy as np

def gaussian_kernel(tensor_a, tensor_b, gamma, debug_print=False):
    """Returns the Gaussian kernel matrix of two matrices of vectors
    element-wise."""
    # bin formula
    a_inputs = tf.shape(tensor_a)[0]
    b_inputs = tf.shape(tensor_b)[0]
    #G = sp.outer(X1.multiply(X1).sum(axis=0), sp.ones(X2.shape[1]))
    G = tf.transpose(tf.tile(tf.expand_dims(tf.reduce_sum(tf.mul(tensor_a, tensor_a), reduction_indices=1), dim=0), tf.pack([b_inputs, 1])))
    H = tf.transpose(tf.tile(tf.expand_dims(tf.reduce_sum(tf.mul(tensor_b, tensor_b), reduction_indices=1), dim=0), tf.pack([a_inputs, 1])))
    # K = sp.exp(-(G + H.T - 2.*(X1.T.dot(X2)))/(2.*sigma**2))
    g_t_h = G + tf.transpose(H)
    sq_diff = g_t_h - 2. * tf.matmul(tensor_a, tensor_b, transpose_b=True)
    kernel = tf.exp(-(sq_diff / (2. * tf.cast(tf.pow(gamma, 2), tf.float32))))
    return kernel


def chose_indices(i, n, with_replacement, ordered_batches, maxval):
    if n > maxval:
        n = maxval
    num_batches = int(maxval / n)
    not_divisible = 0
    if maxval / float(n) - num_batches > 0:
        not_divisible = 1

    if ordered_batches:
        # chose bucket of indices:
        ind = i % (num_batches + not_divisible)
        # if last index, add the rest
        if ind == num_batches:
            rnd = range(ind * n, maxval)
        else:
            rnd = range(ind * n, (ind + 1) * n)
    elif with_replacement:
        rnd = np.random.randint(low=0, high=maxval, size=n)
    else:
        rnd = np.random.choice(maxval, n, replace=False)
    # so random, yo
    return rnd


def train_svm(x, y, x_test, y_test, C=1, gamma=0.001, nIter=100, kernel_type="gaussian", learning_rate_start=1., n_pred=100, n_exp=100, with_replacement=False, ordered_batches=False, most_common_label=None, log_prefix='/data'):
    assert (x.shape[0] == y.shape[0])
    assert (x_test.shape[0] == y_test.shape[0])

    n_inputs = x.shape[0]
    n_features = x.shape[1]
    learning_rate = tf.placeholder(tf.float32, shape=[])

    with tf.name_scope('input'):
        input_x_1 = tf.placeholder(tf.float32, [None, n_features], name="input_x1")
        input_x_2 = tf.placeholder(tf.float32, [None, n_features], name="input_x2")

        y_ = tf.placeholder(tf.float32, [None, 1], name="ground_truth")

        pred_coef = tf.placeholder(tf.int32, [None], name="predicate")
        exp_coef = tf.placeholder(tf.int32, [None], name="xpansion")

    with tf.name_scope('alphas_and_bias'):
        alphas = tf.Variable(tf.truncated_normal([n_inputs, 1], mean=0.5, stddev=.1) , name="alphas")
        alphas_exp = alphas

    with tf.name_scope('gathered_inputs_alphas_gt'):
        alphas_gathered_exp = tf.gather(alphas_exp, exp_coef)
        y_gathered_pred = tf.gather(y_, pred_coef)

        input_x_gathered_pred = tf.gather(input_x_1, pred_coef)
        input_x_gathered_exp = tf.gather(input_x_2, exp_coef)

    with tf.name_scope('kernel_map'):
        if kernel_type == "gaussian":
            kernel = gaussian_kernel(input_x_gathered_pred, input_x_gathered_exp, gamma)
        else:
            print "only gaussian kernel support so far."
            return

    with tf.name_scope('prediction'):
        # prediction of the machine:
        yhat = tf.matmul(kernel, alphas_gathered_exp) #+ bias

    with tf.name_scope('loss'):
        # accuracy
        yhat_sign = tf.sign(yhat)
        prediction_is_correct = tf.equal(yhat_sign, y_gathered_pred)
        accuracy = tf.reduce_mean(tf.cast(prediction_is_correct, tf.float32))

        # preparation of hinge loss
        yhat_gt = tf.mul(y_gathered_pred, yhat, name='yhat_gt')
        pred_error = 1. - yhat_gt

        # hinge loss computation
        hinge_max = tf.maximum(0., pred_error)
        hinge_loss = tf.reduce_sum(hinge_max, name='hinge_loss')

        # squared error regularization loss for alphas
        regularization_loss_l2 = tf.reduce_sum(tf.square(alphas_gathered_exp), name='regularization_sq_loss_l2')

        # l1 regularization loss for alphas
        regularization_loss_l1 = tf.reduce_sum(tf.abs(alphas_gathered_exp), name='regularization_loss_l1')

        # only negative is loss
        regularization_loss_l1_only_neg = tf.reduce_sum(tf.square(tf.maximum(0., -1. * alphas_gathered_exp)))

        hinge_loss_part = C * hinge_loss
        regularization_part = regularization_loss_l2
        loss = tf.add(regularization_part, hinge_loss_part, name='loss')#regularization_loss_l2 + regularization_loss_kl + C * hinge_loss

    with tf.name_scope('summaries'):
        acc_summary = tf.scalar_summary('accuracy_summary', accuracy)
        loss_summary = tf.scalar_summary('loss_summary', loss)
        regloss_summary = tf.scalar_summary('reg_loss_summary', regularization_part)
        hingeloss_summary = tf.scalar_summary('hinge_loss', hinge_loss_part)

    with tf.name_scope('zero_elements_counter'):
        # sparsity counter:
        elements_equal_to_value = tf.equal(alphas_exp, 0.)
        as_ints = tf.cast(elements_equal_to_value, tf.int32)
        zero_count = tf.reduce_sum(as_ints)

    with tf.name_scope('optimizer'):
        # Create an optimizer.
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        grads_and_vars = opt.compute_gradients(loss, [alphas])
        alpha_grad = grads_and_vars[0][0]
        train_step = opt.apply_gradients(grads_and_vars)

    # initialize variables
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # logs
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('./logs' + log_prefix + '/train',
                                          sess.graph)
    test_writer = tf.train.SummaryWriter('./logs' + log_prefix + '/test')


    # for ordered batches
    max_i_exp = compute_num_batches(y.shape[0], n_exp)
    max_i_pred = compute_num_batches(y.shape[0], n_pred)

    print "max_i_pred", max_i_pred
    print "max_i_exp", max_i_exp
    # for tracking difference in parameters
    alphas_old = sess.run(alphas)

    # training
    for i in range(1, nIter):
        # TODO: iterate through whole training set separately for expansion and predicate coefficients?
        samples_seen = i * n_pred
        if learning_rate_start < 0:
            lr_discounted = - learning_rate_start / float(i)
        else:
            lr_discounted = learning_rate_start
        print "lr_discounte", lr_discounted

        for i_pred in range(0, max_i_pred):
            for i_exp in range(0, max_i_exp):
                rnd_pred = chose_indices(i, n_pred, with_replacement, ordered_batches, y.shape[0])
                rnd_exp = chose_indices(i, n_exp, with_replacement, ordered_batches, y.shape[0])
                _, accuracy__, summary__ = sess.run([train_step, accuracy, merged], feed_dict={
                    input_x_1: x,
                    input_x_2: x,
                    y_: y,
                    pred_coef: rnd_pred,
                    exp_coef: rnd_exp,
                    learning_rate: lr_discounted})

                max_batches = max_i_exp * max_i_pred
                num_batches = (max_batches * i) + (i_pred * max_i_exp) + i_exp
                train_writer.add_summary(summary__, num_batches)
                print i_pred, i_exp, accuracy__

        # evaluation
        rnd_pred = chose_indices(i, n_pred, with_replacement, ordered_batches, y_test.shape[0])
        rnd_exp = chose_indices(i, n_exp, with_replacement, ordered_batches, y.shape[0])
        t_start = datetime.datetime.now()
        accuracy_test, summary_test = test(sess, with_replacement, rnd_pred, rnd_exp, lr_discounted, x, y, x_test, y_test, input_x_1, input_x_2, y_, pred_coef, exp_coef, accuracy, learning_rate, merged)
        test_writer.add_summary(summary_test, num_batches)

        #print_alpha_histogram(sess, alphas_exp)

        #zero_count__ = sess.run([zero_count])
        #print "zero_count", zero_count__

        # regularization_loss_l2__, hinge_loss_part__, loss__ = sess.run([regularization_loss_l2, hinge_loss_part, loss], feed_dict={
        # input_x_1: x,
        # input_x_2: x,
        # y_: y,
        # pred_coef: rnd_pred,
        # exp_coef: rnd_exp,
        # learning_rate: lr_discounted})
        # print "losses", regularization_loss_l2__, hinge_loss_part__, loss__

        #print "accuracy_test", accuracy_test, lr_discounted
        t_start = datetime.datetime.now()
        print "testing full", t_start
        accuracy_test = test_with_full_model(sess, with_replacement, n_pred, n_exp, lr_discounted, x, y, x_test, y_test, input_x_1,
                         input_x_2, y_, pred_coef, exp_coef, accuracy, learning_rate, merged)

        print "full accuracy on test", accuracy_test, "took", datetime.datetime.now() - t_start



        alphas__ = sess.run(alphas)
        print "alpha delta", np.sum(np.abs(alphas__ - alphas_old))
        alphas_old = alphas__

def compute_num_batches(nmax, n):
    if n > nmax:
        n = nmax
    n_batches = int(nmax / n)
    n_batch_rest = 0
    if nmax / float(n) - n_batches > 0:
        n_batch_rest = 1

    return n_batches + n_batch_rest

def test_with_full_model(sess, with_replacement, n_pred, n_exp, lr_discounted, x, y, x_test, y_test, input_x_1, input_x_2, y_, pred_coef, exp_coef, accuracy, learning_rate, merged):

    max_i_test = compute_num_batches(y_test.shape[0], n_pred)
    max_i_train = compute_num_batches(y.shape[0], n_exp)

    # create weighted sum of accuracies
    accuracy_test = 0.
    for i_test in range(0, max_i_test):
        rnd_pred = chose_indices(i_test, n_pred, with_replacement, ordered_batches=True, maxval=y_test.shape[0])
        # all datapoints in model
        accuracy_pred_batch = 0.
        for i_train in range(0, max_i_train):
            rnd_exp = chose_indices(i_train, n_exp, with_replacement, ordered_batches=True, maxval=y.shape[0])
            accuracy_test_one_train_batch, _ = test(sess, with_replacement, rnd_pred, rnd_exp, lr_discounted, x, y, x_test, y_test, input_x_1, input_x_2,
                 y_, pred_coef, exp_coef, accuracy, learning_rate, merged)
            accuracy_pred_batch += len(rnd_exp) * accuracy_test_one_train_batch
        accuracy_test += accuracy_pred_batch / float(y.shape[0]) * len(rnd_pred)

    accuracy_test = accuracy_test / float(y_test.shape[0])

    return accuracy_test


def test(sess, with_replacement, rnd_pred, rnd_exp, lr_discounted, x, y, x_test, y_test, input_x_1, input_x_2, y_, pred_coef, exp_coef, accuracy, learning_rate, merged):
    accuracy_test, summary__ = sess.run([accuracy, merged], feed_dict={
        input_x_1: x_test,
        input_x_2: x,
        y_: y_test,
        pred_coef: rnd_pred,
        exp_coef: rnd_exp,
        learning_rate: lr_discounted
    })
    return accuracy_test, summary__


def print_alpha_histogram(sess, alphas):
    alphas__ = sess.run([alphas], feed_dict={})
    print "min alphas:", np.min(alphas__)
    print "max alphas:", np.max(alphas__)
    bins = range((np.floor(np.min(alphas__)) * 10).astype(int), (np.ceil(np.max(alphas__)) * 10 + 1).astype(int), 1)
    bins = [el / 10. for el in bins]
    hist = np.histogram(alphas__, bins=bins)
    print "histogram alphas"
    for hx, hy in zip(hist[0], hist[1]):
        print hy, "\t", hx