# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

import numpy as np
# Standard scientific Python imports
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

# ###### Program execution
if __name__ == '__main__':
    print("############################################")
    print("##### SCIKIT-LEARN - Digit recognition #####")
    print("############################################")
    print(" ")

    print("===== Loading Digits dataset =====")
    digits = datasets.load_digits()
    print(f"--> The Digits dataset is a dictionary-like object, printing it out\n {digits} \n")
    data = digits.data
    target = digits.target
    n_samples = len(digits.images)
    print(f"Digits dictionary data (input features) has {n_samples} n° of samples, printing ... \n {data} \n")
    print(f"Printing digits dictionary target data (output) \n {target} \n")
    #print(digits.images[1])
    print(" ")

    print("To apply a classifier on this data, we need to flatten the images, turning each 2-D array of grayscale values from shape (8, 8) into shape (64,).")
    print("Subsequently, the entire dataset will be of shape (n_samples, n_features), where n_samples is the number of images and n_features is the total number of pixels in each image.")
    print("-----------------------------------------------------------------------------------")
    print("# flatten the images")
    print(f"n_samples = {n_samples} \n")
    data = digits.images.reshape((n_samples, -1))
    print("-----------------------------------------------------------------------------------")
    print("# Create a classifier: a support vector classifier")
    clf = svm.SVC(gamma=0.001)
    print("-----------------------------------------------------------------------------------")
    print(" ")
    print("We can then split the data into train and test subsets and fit a support vector classifier on the train samples.") 
    print("The fitted classifier can subsequently be used to predict the value of the digit for the samples in the test subset.")
    print("-----------------------------------------------------------------------------------")
    print("# Split data into 50% train and 50% test subsets")
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )
    print("-----------------------------------------------------------------------------------")
    print("# Learn the digits on the train subset")
    clf.fit(X_train, y_train)
    print("-----------------------------------------------------------------------------------")
    print("# Predict the value of the digit on the test subset")
    predicted = clf.predict(X_test)

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    plt.show()


    print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()