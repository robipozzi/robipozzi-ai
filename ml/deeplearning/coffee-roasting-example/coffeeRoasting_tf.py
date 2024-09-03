import numpy as np
import matplotlib.pyplot as plt
import loadCoffeeData
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from lab_utils_common import dlc
#from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# ###### Program execution
if __name__ == '__main__':
    print("#############################################################################")
    print("########## Coffee Roasting neural network example using Tensorflow ##########")
    print("#############################################################################")
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

    Xt = np.tile(Xn,(1000,1))
    Yt= np.tile(Y,(1000,1))   
    print(Xt.shape, Yt.shape)
    print("")

    print("========== Istantiate Model ...")
    tf.random.set_seed(1234)  # applied to achieve consistent results
    model = Sequential(
        [
            tf.keras.Input(shape=(2,)),
            Dense(3, activation='sigmoid', name = 'layer1'),
            Dense(1, activation='sigmoid', name = 'layer2')
        ]
    )
    model.summary()
    print("")

    L1_num_params = 2 * 3 + 3   # W1 parameters  + b1 parameters
    L2_num_params = 3 * 1 + 1   # W2 parameters  + b2 parameters
    print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params  )

    print("========== Let's examine the weights and biases Tensorflow has instantiated.")
    # The weights 𝑊 should be of size (number of features in input, number of units in the layer) 
    # The bias 𝑏 size should match the number of units in the layer:
    # In the first layer with 3 units, we expect W to have a size of (2,3) and 𝑏 should have 3 elements.
    # In the second layer with 1 unit, we expect W to have a size of (3,1) and 𝑏 should have 1 element.
    W1, b1 = model.get_layer("layer1").get_weights()
    W2, b2 = model.get_layer("layer2").get_weights()
    print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
    print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

    print("========== Compile the model")
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    )

    print("========== Fit the model")
    model.fit(
        Xt,Yt,            
        epochs=10,
    )

    print("========== Let's examine the updated weights and biases after model fitting")
    W1, b1 = model.get_layer("layer1").get_weights()
    W2, b2 = model.get_layer("layer2").get_weights()
    print("W1 (after model fitting):\n", W1, "\nb1 (after model fitting):", b1)
    print("W2 (after model fitting):\n", W2, "\nb2 (after model fitting):", b2)
    print("")

    # Set weights from a previous run. 
    W1 = np.array([
        [-8.94,  0.29, 12.89],
        [-0.17, -7.34, 10.79]] )
    b1 = np.array([-9.87, -9.28,  1.01])
    W2 = np.array([
        [-31.38],
        [-27.86],
        [-32.79]])
    b2 = np.array([15.54])

    # Replace the weights from your trained model with the values above.
    model.get_layer("layer1").set_weights([W1,b1])
    model.get_layer("layer2").set_weights([W2,b2])

    # Check if the weights are successfully replaced
    W1, b1 = model.get_layer("layer1").get_weights()
    W2, b2 = model.get_layer("layer2").get_weights()
    print("W1 (after replacement):\n", W1, "\nb1 (after replacement):", b1)
    print("W2 (after replacement):\n", W2, "\nb2 (after replacement):", b2)

    print("========== Run predictions using fit model")
    X_test = np.array([
            [200,13.9],  # positive example
            [200,17]])   # negative example
    X_testn = norm_l(X_test)
    predictions = model.predict(X_testn)
    print("predictions = \n", predictions)
    print("")

    yhat = np.zeros_like(predictions)
    for i in range(len(predictions)):
        if predictions[i] >= 0.5:
            yhat[i] = 1
        else:
            yhat[i] = 0
    print(f"decisions = \n{yhat}")

    yhat = (predictions >= 0.5).astype(int)
    print(f"decisions = \n{yhat}")