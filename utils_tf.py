import tensorflow as tf
import keras
import numpy as np
import math
import matplotlib.pyplot as plt


def initialize_gaussianunif_weights(d, N1, N2):
    initt = np.random.randn(d, N1)
    # initt = np.random.uniform(-1, 1, size=(d, N))
    # initt = np.random.chisquare(5, size=(d, N))
    # initt = np.random.laplace(1, 2, size=(d, N))

    initt = initt - np.expand_dims(initt.mean(axis=1), axis=1)
    initt = initt / np.linalg.norm(initt, axis=0)
    initt = tf.constant_initializer(initt)

    initt_b = np.random.uniform(0, 2, size=(N1, N2))/math.sqrt(N1)
    initt_b = tf.constant_initializer(initt_b)

    return initt, initt_b


def make_2layer_net(d, N1, N2, kernel_init=None, last_layer_initt=None):
    if (kernel_init == None):
        kernel_init, last_layer_initt = initialize_gaussianunif_weights(d, N1, N2)

    model = tf.keras.models.Sequential([
      tf.keras.layers.Input(d),
      tf.keras.layers.Dense(N1, activation='relu', name='dense1', use_bias=False, kernel_initializer=kernel_init),
      tf.keras.layers.Dense(N2, use_bias=False, kernel_initializer=last_layer_initt, name='out')
    ])
    return model

def initialize_weights(N1, N2):
    initt = np.random.randn(N1, N2)
    # initt = np.random.uniform(-1, 1, size=(d, N))
    # initt = np.random.chisquare(5, size=(d, N))
    # initt = np.random.laplace(1, 2, size=(d, N))

    initt = initt - np.expand_dims(initt.mean(axis=1), axis=1)
    initt = initt / np.linalg.norm(initt, axis=0)
    initt = tf.constant_initializer(initt)
    return initt

def neural_net(in_dim, Ns, out_dim, activation=tf.keras.activations.relu):
    kernel_inits = []
    model = [tf.keras.layers.Input(in_dim, name='input'),
            tf.keras.layers.Dense(Ns[0], activation=tf.keras.activations.relu,
                                use_bias=False,
                                kernel_initializer=initialize_weights(in_dim, Ns[0])), ]

    for i in range(len(Ns)-1):
        kernel_init=initialize_weights(Ns[i], Ns[i+1])
        model.append(tf.keras.layers.Dense(Ns[i+1], activation=tf.keras.activations.relu,
                    use_bias=False, kernel_initializer=kernel_init),)


    last_layer_initt = np.random.uniform(0, 2, size=(Ns[-1], out_dim))/math.sqrt(Ns[-1])
    last_layer_initt = tf.constant_initializer(last_layer_initt)
    model.append(tf.keras.layers.Dense(out_dim, use_bias=False,
                kernel_initializer=last_layer_initt, name='out'))

    model = tf.keras.models.Sequential(model)
    return model

def train_teacher_student_gaussian_input(student_model, teacher_model,
                    num_iter, bsize, optimizer=tf.keras.optimizers.SGD(),
                    loss=tf.keras.losses.MeanSquaredError()):
    losses = []
    d = teacher_model.get_weights()[0].shape[0]
    for step in range(num_iter):

        x = np.random.randn(bsize, d)
        y = teacher_model(x, training=False)

        with tf.GradientTape() as tape:
            y_hat = student_model(x, training=True)
            loss_value = loss(y, y_hat)

        losses.append(loss_value)

        grads = tape.gradient(loss_value, student_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, student_model.trainable_weights))

        # Log every 200 batches
        if step % 200 == 0:
            print(
                "Training loss at step %d: %.4f"
                % (step, float(loss_value))
            )
    return(student_model)

def kernel_of_nn(model, layer=0, size=5):
    Ws = model.get_weights()[layer]
    Ws = Ws / np.linalg.norm(Ws, axis=0)
    kernel = Ws.T.dot(Ws)
    return kernel[:size, :size]

def kernel_dist_from_etf(model, layer=0):
    size = model.get_weights()[layer].shape[1]
    kernel_model = kernel_of_nn(model, layer, size)
    a = -1/(size-1)
    kernel_etf = a * np.ones((size, size)) + np.eye(size) * (1 - a)
    return np.sum((kernel_etf - kernel_model)**2)/(size * (size-1))



from scipy.stats import ortho_group  # Requires version 0.18 of scipy

def make_simplex_2layer_nn(M, d):

  m = ortho_group.rvs(dim=d)

  simplex_vecs = m[:M]
  simplex_vecs -= np.mean(simplex_vecs, axis=0)
  simplex_vecs /= np.linalg.norm(simplex_vecs[0])

  # The simplex neural network
  init_simplex = tf.constant_initializer(simplex_vecs.T)

  model = tf.keras.models.Sequential([
    tf.keras.layers.Input(d),
    tf.keras.layers.Dense(M, activation='relu', name='dense1', trainable=True, use_bias=False,
                          kernel_initializer=init_simplex, kernel_constraint=tf.keras.constraints.UnitNorm(
    axis=1)),
    tf.keras.layers.Dense(1, use_bias=False, name='out')
  ])

  return model


def mse_on_Gaussian_input(model1, model2, bsize):
    d = teacher_model.get_weights()[0].shape[0]
    mse = tf.keras.losses.MeanSquaredError()
    x = np.random.randn(bsize, d)
    y1 = model1(x, training=False)
    y1 = model2(x, training=False)

    return (mse(y1, y2))





################################################################################
# functions for mnist experiments

def load_mnist_train_test(whiten=False, ev_odd=False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return (x_train, y_train, x_test, y_test)

def load_mnist_evod(whiten=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    y_train = (y_train%2==0)
    y_test = (y_test%2==0)
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    N_train = x_train.shape[0]
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
    X = np.concatenate([x_train, x_test], axis=0)

    # whitenning
    X -= np.mean(X, axis = 0) # zero-center the data (important)
    cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix

    U,S,V = np.linalg.svd(cov)
    Xrot = np.dot(X, U) # decorrelate the data
    Xwhite = Xrot / np.sqrt(S + 1e-5)

    x_train, x_test = Xwhite[:N_train], Xwhite[N_train:]

    return (x_train, y_train, x_test, y_test)




    ############################################################################
    # Rattana's kmeans and subsampling algorthims
from sklearn.cluster import KMeans

def group(M, weight, mode):
    if mode == 'heuristic':
        #randomly assign group
        N = len(weight)
        w_center_idcs = np.random.choice(N, replace=False, size= M)
        w_centers = np.array([w/norm(w) for w in weight[w_center_idcs]])

        w_lists = [[] for _ in range(M)]
        label_index = []
        for w_old in weight:
            ips = w_centers@(w_old/norm(w_old))
            c_ind = np.argmax(ips)
            w_lists[c_ind].append(w_old)
            label_index.append(c_ind)

        return w_lists, w_centers, np.array(label_index)

    if mode == 'kmean':
        # k mean on normalised weight
        normalised_weight = (weight/np.linalg.norm(weight, axis = 1).reshape(-1,1))
        kmeans = KMeans(n_clusters = M).fit(normalised_weight)
        w_lists = []
        w_centers = []
        for i in range(M):
            w_lists.append(weight[kmeans.labels_ == i])
            w_centers.append(np.mean(normalised_weight[kmeans.labels_ == i], axis = 0))

        label_index = kmeans.labels_
        w_centers = np.array(w_centers)
        return w_lists, w_centers, label_index

def get_merge_coefficient(next_weight, new_weight, label_index, old_weight):
    M = int(new_weight.shape[0])
    N = int(next_weight.shape[1])

    old_norm = np.linalg.norm(old_weight, axis = 1).reshape(-1)
    coeff_matrix = np.zeros((N,M))
    for i in range(M):
        group_norm = np.linalg.norm(new_weight[i])
        coeff_matrix[:, i] = (old_norm/group_norm)*(label_index == i)


    new_next_weight = (next_weight@coeff_matrix)

    return new_next_weight
