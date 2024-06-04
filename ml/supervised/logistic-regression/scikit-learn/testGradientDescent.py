import numpy as np
from sklearn.linear_model import LogisticRegression

# *****************************
# ***** Functions - START *****
# *****************************

# ***************************
# ***** Functions - END *****
# ***************************

# ###### Program execution
if __name__ == '__main__':
    print("############################################################")
    print("########## Logistic regression using SCIKIT-LEARN ##########")
    print("############################################################")
    print("")

    print("========== Initializing training set ...")
    X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1])

    # data is stored in numpy array/matrix
    print("***** Printing training set features vector ...")
    print(f"X Shape: {X.shape}, X Type:{type(X)})")
    print(X)
    print("")
    print("***** Printing training set output result vector ...")
    print(f"y Shape: {y.shape}, y Type:{type(y)})")
    print(y)
    print("--------------------------------------------------------------")
    print("")

    print("========== Computing Gradient for logistic regression using SCIKIT-LEARN ...")
    print("***** Fitting model on training set ...")
    lr_model = LogisticRegression()
    lr_model.fit(X, y)
    
    print("***** Make prediction on training set ...")
    y_pred = lr_model.predict(X)
    print("Prediction on training set:", y_pred)

    print("***** Calculate accuracy on training set ...")
    print("Accuracy on training set:", lr_model.score(X, y))
    print("--------------------------------------------------------------")
    print("")

    


    