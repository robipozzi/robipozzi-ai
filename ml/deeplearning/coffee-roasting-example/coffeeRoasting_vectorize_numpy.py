import numpy as np
import matplotlib.pyplot as plt
import loadCoffeeData
import tensorflow as tf
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# *****************************
# ***** Functions - START *****
# *****************************
# ===== Sigmoid Function
def sigmoid(z):
    """
    Compute the sigmoid of z
    Args:
        z (ndarray): A scalar, numpy array of any size.
    Returns:
        g (ndarray): sigmoid(z), with the same shape as z  
    """
    g = 1/(1+np.exp(-z))
    return g

# ===== Neural network Layer computation using vectorization and matrices
def my_dense(a_in, W, b):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
    Returns
      a_out (ndarray (j,))  : j units|
    """
    z = np.matmul(a_in, W) + b 
    a_out = sigmoid(z)
    return a_out

def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x,  W1, b1)
    a2 = my_dense(a1, W2, b2)
    return(a2)

def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
    return(p)
# ***************************
# ***** Functions - END *****
# ***************************

# ###### Program execution
if __name__ == '__main__':
    print("########################################################################")
    print("########## Coffee Roasting neural network example using Numpy ##########")
    print("########################################################################")
    print("")

    print("========== Initializing training set ...")
    (X,Y) = loadCoffeeData.load_coffee_data()
    # data is stored in numpy array/matrix
    print("***** Printing training set features vector ...")
    print(f"X Shape: {X.shape}, X Type:{type(X)})")
    print(X)
    print("")
    print("***** Printing training set output result vector ...")
    print(f"Y Shape: {Y.shape}, y Type:{type(Y)})")
    print(Y)
    print("")

    print("========== Normalize data ...")
    # Fitting the weights to the data (back-propagation) will proceed more quickly if the data is normalized. 
    # This is the same procedure where features in the data are each normalized to have a similar range. 
    # The procedure below uses a Keras normalization layer. 
    # It has the following steps:
    #   - create a "Normalization Layer". Note, as applied here, this is not a layer in your model.
    #   - 'adapt' the data. This learns the mean and variance of the data set and saves the values internally.
    #   - normalize the data.
    # It is important to apply normalization to any future data that utilizes the learned model. 
    print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
    print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
    norm_l = tf.keras.layers.Normalization(axis=-1)
    norm_l.adapt(X)  # learns mean, variance
    Xn = norm_l(X)
    print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
    print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")
    print("")
    
    print("========== Set weights and bias ...")
    W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
    b1_tmp = np.array( [-9.82, -9.28,  0.96] )
    W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
    b2_tmp = np.array( [15.41] )
    print("W1_tmp:\n", W1_tmp, "\nb1_tmp:", b1_tmp)
    print("W2_tmp:\n", W2_tmp, "\nb2_tmp:", b2_tmp)

    a0 = W1_tmp.shape[0]
    units_1 = W1_tmp.shape[1]
    a1 = W2_tmp.shape[0]
    units_2 = W2_tmp.shape[1]
    print(f"n° of input features: {a0}")
    print(f"n° of units (layer 1): {units_1}")
    print(f"n° of layer 1 output features: {a1}")
    print(f"n° of units (layer 2): {units_2}")
    print("")

    print("========== Run predictions using numpy")
    X_tst = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
    X_tstn = norm_l(X_tst)  # remember to normalize
    predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

    yhat = np.zeros_like(predictions)
    for i in range(len(predictions)):
        if predictions[i] >= 0.5:
            yhat[i] = 1
        else:
            yhat[i] = 0
    print(f"decisions = \n{yhat}")

    yhat = (predictions >= 0.5).astype(int)
    print(f"decisions = \n{yhat}")