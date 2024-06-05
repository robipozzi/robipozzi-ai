import numpy as np
import matplotlib.pyplot as plt
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

# Logistic regression model function (using vectorization)
def predict(X, w, b): 
    """
    single predict using logistic regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    print(f"f_wb: {f_wb}, f_wb Type:{type(f_wb)})")
    # Compute prediction (0 or 1) in case f_wb is an array
    if type(f_wb) == np.ndarray:
        print("*** ARRAY !!!")
        size = f_wb.size
        p = np.zeros(size)
        for k in range(size):
            #p[k] = int(f_wb[k] >= 0.5)
            p[k] = f_wb[k]
        return p
    return p
# ***************************
# ***** Functions - END *****
# ***************************

# ###### Program execution
if __name__ == '__main__':
    print("#######################################################################")
    print("########## Deep Learning and Neural networks with Tensorflow ##########")
    print("#######################################################################")
    print("")

    print("########## Neuron without activation - Regression/Linear Model ##########")
    print("")

    print("========== Initializing training set ...")
    X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
    Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)
    print(f"X_train =\n {X_train}")
    print(f"Y_train =\n {Y_train}")

    fig, ax = plt.subplots(1,1)
    ax.scatter(X_train, Y_train, marker='x', c='r', label="Data Points")
    ax.legend( fontsize='xx-large')
    ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
    ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
    plt.show()
    print("------------------------------------------------")
    print("")

    print("========== Linear regression ...")
    # We can define a layer with one neuron or unit and compare it to the familiar linear regression function.
    linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear', )
    # Let's examine the weights.
    # There are no weights as the weights are not yet instantiated. 
    linear_layer.get_weights()
    # Let's try the model on one example in X_train. 
    # This will trigger the instantiation of the weights. Note, the input to the layer must be 2-D, so we'll reshape it.
    print("***** Run the model on one example in the training set ...")
    print(f"X_train[0] = {X_train[0]} --> Y_train[0] = {Y_train[0]}")
    # The result is a tensor (another name for an array) with a shape of (1,1) or one entry.
    print("***** This will trigger the instantiation of the weights")
    a1 = linear_layer(X_train[0].reshape(1,1))
    print(a1)
    # Now let's look at the weights and bias.
    print("***** Weights are randomly initialized to small numbers and the bias defaults to being initialized to zero")
    w, b = linear_layer.get_weights()
    print(f"w = {w}, b={b}")
    print("***** Let's set weights to known values instead")
    set_w = np.array([[200]])
    set_b = np.array([100])

    # set_weights takes a list of numpy arrays
    linear_layer.set_weights([set_w, set_b])
    print(linear_layer.get_weights())

    # Let's compare equation (1) to the layer output.
    print("***** Run Linear regression model using numpy")
    alin = np.dot(set_w, X_train[0].reshape(1,1)) + set_b
    print(f"Input = {X_train[0]} --> Output = {alin}")
    print("***** Run Neural network using Tensorflow")
    a1 = linear_layer(X_train[0].reshape(1,1))
    print(f"Input = {X_train[0]} --> Output = {a1}")
    print(a1)

    print("***** Run Predictions using both Neural network and Linear regression model")
    prediction_np = np.dot( X_train, set_w) + set_b
    prediction_tf = linear_layer(X_train)
    print(f"prediction using Numpy linear regression = {prediction_np}")
    print(f"prediction using Tensorflow model = {prediction_tf}")
    print("------------------------------------------------")
    print("")
    
    print("########## Neuron with Sigmoid activation ##########")
    print("")

    print("========== Initializing training set ...")
    X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
    Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
    print(f"X_train =\n {X_train}")
    print(f"Y_train =\n {Y_train}")

    pos = Y_train == 1
    neg = Y_train == 0

    fig,ax = plt.subplots(1,1,figsize=(4,3))
    ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c = 'red', label="y=1")
    ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", c = 'blue', facecolors='none', lw=3)

    ax.set_ylim(-0.08,1.1)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xlabel('x', fontsize=12)
    ax.set_title('one variable plot')
    ax.legend(fontsize=12)
    plt.show()
    print("------------------------------------------------")
    print("")

    print("========== Logistic neuron ...")
    # We can implement a 'logistic neuron' by adding a sigmoid activation.
    # This section will create a Tensorflow Model that contains our logistic layer to demonstrate an alternate method of creating models. 
    # Tensorflow is most often used to create multi-layer models. 
    # The Sequential model is a convenient means of constructing these models.
    print("***** Instantiate model using Tensorflow")
    keras = tf.keras
    model = keras.Sequential(
        [
            tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')
        ]
    )
    model.summary()

    print("***** Run the model ...")
    print("***** This will trigger the instantiation of the weights")
    logistic_layer = model.get_layer('L1')
    w,b = logistic_layer.get_weights()
    print(f"w = {w} - b = {b}")
    print(f"w.shape = {w.shape} - b.shape = {b.shape}")

    print("***** Let's set weights to known values")
    set_w = np.array([[2]])
    set_b = np.array([-4.5])
    print(f"set_w = {set_w} - set_b = {set_b}")
    # set_weights takes a list of numpy arrays
    logistic_layer.set_weights([set_w, set_b])
    print(logistic_layer.get_weights())

    print("***** Run Predictions using both Neural network and Logistic regression model")
    #alog = sigmoidnp(np.dot(set_w,X_train[0].reshape(1,1)) + set_b)
    alog = predict(X_train[0].reshape(1,1), set_w, set_b)
    a1 = model.predict(X_train[0].reshape(1,1))
    print(f"prediction using Numpy logistic regression = {alog}")
    print(f"prediction using Tensorflow model = {a1}")
    print("------------------------------------------------")
    print("")