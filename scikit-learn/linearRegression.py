# Code source: Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# ###### Program execution
if __name__ == '__main__':
    print("############################################")
    print("##### SCIKIT-LEARN - Linear regression #####")
    print("############################################")
    print(" ")

    print("# Loading diabetes dataset ...")
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]
    print("# Diabetes dataset loaded")
    print(" ")

    print("# Split the data into training/testing sets")
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    print("# Split the targets into training/testing sets")
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]
    print(" ")

    print("# Creating linear regression object ...")
    regr = linear_model.LinearRegression()
    print("# Linear regression object created")
    print(" ")

    print("# Train the model using the training sets")
    regr.fit(diabetes_X_train, diabetes_y_train)
    # The coefficients
    print("Model Coefficients: ", regr.coef_)
    print("Model Intercept: ", regr.intercept_)
    print(" ")

    print("# Make predictions using the training set")
    diabetes_y_train_pred = regr.predict(diabetes_X_train)
    # The mean squared error
    print("Mean squared error on training set: %.2f" % mean_squared_error(diabetes_y_train, diabetes_y_train_pred))
    print(" ")

    print("# Make predictions using the testing set")
    diabetes_y_test_pred = regr.predict(diabetes_X_test)
    # The mean squared error
    print("Mean squared error on testing set: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_test_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_test_pred))
    print(" ")

    print("===== Plotting =====")
    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
    plt.plot(diabetes_X_test, diabetes_y_test_pred, color="blue", linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()
    print(" ")