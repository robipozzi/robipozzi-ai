import numpy as np
import matplotlib.pyplot as plt
#matplotlib widget
from sklearn.datasets import make_blobs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
np.set_printoptions(precision=2)
#from lab_utils_multiclass_TF import *
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


# *****************************
# ***** Functions - START *****
# *****************************

# ***************************
# ***** Functions - END *****
# ***************************

# ###### Program execution
if __name__ == '__main__':
    print("################################################################")
    print("########## Multi-class classification with Tensorflow ##########")
    print("################################################################")
    print("")

    print("========== Use Scikit-Learn make_blobs function to make a training data set with 4 categories ...")
    # make 4-class dataset for classification
    classes = 4
    m = 100
    centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
    std = 1.0
    X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=std,random_state=30)
    print(f"X_train = {X_train}")
    print(f"y_train = {y_train}")
    print("")

    #plt_mc(X_train,y_train,classes, centers, std=std)

    # show classes in data set
    print(f"unique classes {np.unique(y_train)}")
    # show how classes are represented
    print(f"class representation {y_train[:10]}")
    # show shapes of our dataset
    print(f"shape of X_train: {X_train.shape}, shape of y_train: {y_train.shape}")
    print("")

    print("########## Neural Network ##########")
    print("")

    print("***** Instantiate model using Tensorflow")
    tf.random.set_seed(1234)  # applied to achieve consistent results
    model = Sequential(
        [
            Dense(2, activation = 'relu',   name = "L1"),
            Dense(4, activation = 'linear', name = "L2")
        ]
    )
    print("")

    print("***** Compile the model using SparseCategoricalCrossentropy loss function ...")
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.01),
    )
    print("")

    print("***** Train the model ...")
    model.fit(
        X_train,y_train,
        epochs=200
    )
    print("")

    print("***** Gather the trained parameters from the first layer...")
    l1 = model.get_layer("L1")
    W1,b1 = l1.get_weights()
    print(f"W1 =\n {W1}")
    print(f"b1 = {b1}")
    print("")

    print("***** Gather the trained parameters from the first layer...")
    l2 = model.get_layer("L2")
    W2, b2 = l2.get_weights()
    print(f"W2 =\n {W2}")
    print(f"b2 = {b2}")
    print("")

    # create the 'new features', the training examples after L1 transformation
    Xl2 = np.maximum(0, np.dot(X_train,W1) + b1)