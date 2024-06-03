import numpy as np
#%matplotlib widget
import matplotlib.pyplot as plt
#from lab_utils_common import  plot_data, sigmoid, dlc
#plt.style.use('./deeplearning.mplstyle')
  
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

def compute_cost_logistic(X, y, w, b):
    """
    Computes cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        # Sigmoid function is g = 1/(1+np.exp(-z_i))
        #f_wb_i = 1/(1+np.exp(-z_i))
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost = cost / m
    return cost
# ***************************
# ***** Functions - END *****
# ***************************

# ###### Program execution
if __name__ == '__main__':
    print("========== Initializing training set ...")
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
    y_train = np.array([0, 0, 0, 1, 1, 1])                                           #(m,)
    print("--------------------------------------------------------------")
    print("")

    # data is stored in numpy array/matrix
    print("========== Printing training set ...")
    print("***** Features vector ...")
    print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
    print(X_train)
    print("")
    print("***** Output result vector ...")
    print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
    print(y_train)
    print("--------------------------------------------------------------")
    print("")

    fig,ax = plt.subplots(1,1,figsize=(4,4))
    #plot_data(X_train, y_train, ax)
    # Set both axes to be from 0-4
    ax.axis([0, 4, 0, 3.5])
    ax.set_ylabel('$x_1$', fontsize=12)
    ax.set_xlabel('$x_0$', fontsize=12)
    plt.show()

    # Compute and display cost using our pre-chosen optimal parameters.
    print("========== Computing cost with initial model weights b and w ...")
    w_array1 = np.array([1,1])
    b_1 = -3
    w_array2 = np.array([1,1])
    b_2 = -4
    print("Cost for b = -3 : ", compute_cost_logistic(X_train, y_train, w_array1, b_1))
    print("Cost for b = -4 : ", compute_cost_logistic(X_train, y_train, w_array2, b_2))
    print("--------------------------------------------------------------")
    print("")