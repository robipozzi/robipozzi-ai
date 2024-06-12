import tensorflow as tf

# ###### Program execution
if __name__ == '__main__':
    print("TensorFlow version:", tf.__version__)
    print(" ")
    print("===== Load MNIST dataset")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print(f"x_train Shape: {x_train.shape}, x_train Type:{type(x_train)})")
    print(" ")

    print("===== Build a Machine Learning model")
    print("***** Build a Sequential model")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    
    print("***** Returning a vector of logits or log-odds scores, one for each class.")
    predictions = model(x_train[:1]).numpy()
    print(predictions)
    
    print("***** The softmax function converts logits to probabilities for each class: ")
    softmax = tf.nn.softmax(predictions).numpy()
    print(softmax)
    
    print("***** Define a Loss function")
    # The loss function takes a vector of ground truth values and a vector of logits and returns a scalar loss for each example. 
    # This loss is equal to the negative log probability of the true class: The loss is zero if the model is sure of the correct class.
    # This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.math.log(1/10) ~= 2.3.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = loss_fn(y_train[:1], predictions).numpy()
    print(f"----- Loss for untrained model = {loss}")

    print("***** Compile the model before start training")
    # Before you start training, configure and compile the model using Keras Model.compile. 
    # Set the optimizer class to adam, set the loss to the loss_fn function you defined earlier, and specify a metric to be evaluated for the model 
    # by setting the metrics parameter to accuracy.
    model.compile(  optimizer='adam',
                    loss=loss_fn,
                    metrics=['accuracy'])
    
    print("***** Train and evaluate the model before start training")
    print("----- Fit the model ...")
    model.fit(x_train, y_train, epochs=5)
    print("----- Evaluate the model ...")
    model.evaluate(x_test,  y_test, verbose=2)
    
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    probability_model(x_test[:5])