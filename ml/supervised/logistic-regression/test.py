import copy, math
import numpy as np
import matplotlib.pyplot as plt

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

# Logistic regression model function
def predict(X, w, b): 
    """
    single predict using logistic regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar / array):  prediction
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
            p[k] = int(f_wb[k] >= 0.5)
        return p
    # Compute prediction (0 or 1) in case f_wb is a number
    p = int(f_wb >= 0.5)
    return p

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
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost = cost / m
    return cost

def compute_gradient_logistic(X, y, w, b): 
    """
    Computes the gradient for logistic regression 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                           #(n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar
        
    return dj_db, dj_dw 

def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    """
    Performs batch gradient descent
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic(X, y, w, b) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history         #return final w,b and J history for graphing
# ***************************
# ***** Functions - END *****
# ***************************

# ###### Program execution
if __name__ == '__main__':
    print("#########################################")
    print("########## Logistic regression ##########")
    print("#########################################")
    print("")

    print("========== Initializing training set ...")
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    # data is stored in numpy array/matrix
    print("***** Printing training set features vector ...")
    print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
    print(X_train)
    print("")
    print("***** Printing training set output result vector ...")
    print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
    print(y_train)
    print("--------------------------------------------------------------")
    print("")

    print("========== Computing prediction ...")
    w_tmp = np.array([2.,3.])
    b_tmp = 1.
    print("***** Initalizing and printing model parameters ...")
    print(f"w_tmp: {w_tmp}, w_tmp Type:{type(w_tmp)})")
    print(f"b_tmp: {b_tmp}, b_tmp Type:{type(b_tmp)})")
    print("***** Run prediction on training set...")
    print(f"X_train Shape: {X_train.shape}")
    y = predict(X_train, w_tmp, b_tmp)
    print(f"y Shape: {y.shape}")
    print(y)
    print("***** Run prediction on new input data...")
    X = np.array([0.5, 1.5])
    print(f"X Shape: {X.shape}")
    y = predict(X, w_tmp, b_tmp)
    print(f"y: {y}")
    print("--------------------------------------------------------------")
    print("")