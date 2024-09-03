import copy, math
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# *****************************
# ***** Functions - START *****
# *****************************
# Multiple linear regression model function
def predict(x, w, b): 
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b     
    return p

# Cost function
def compute_cost(X, y, w, b): 
    """
    compute cost
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
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost

# Compute Gradient function
def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

# Gradient descent
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    print("##### Function gradient_descent - START #####")
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    print(f"w: {w}")
    print(f"b: {b}")
    
    for i in range(num_iters):      
        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    print("##### Function gradient_descent - END #####")
    return w, b, J_history #return final w,b and J history for graphing
# ***************************
# ***** Functions - END *****
# ***************************

# ###### Program execution
if __name__ == '__main__':
    print("========== Initializing training set ...")
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])
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

    # 𝐰 is a vector with 𝑛 elements.
    #   Each element contains the parameter associated with one feature.
    #   in our dataset, n is 4.
    #   notionally, we draw this as a column vector
    # 
    # 𝑏 is a scalar parameter.
    print("========== Initializing model weights b and w ...")
    b_init = 785.1811367994083
    w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
    print(f"w_init shape: {w_init.shape}")
    print(f"w_init: {w_init}")
    print(f"b_init type: {type(b_init)}")
    print(f"b_init: {b_init}")
    print("--------------------------------------------------------------")
    print("")

    # Compute and display cost using our pre-chosen optimal parameters.
    print("========== Computing cost with initial model weights b and w ...")
    cost = compute_cost(X_train, y_train, w_init, b_init)
    print(f'Cost at optimal w : {cost}')
    print("--------------------------------------------------------------")
    print("")

    # Compute and display gradient 
    print("========== Computing gradient using initial model weights b and w ...")
    tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
    print(f'dj_db at initial w,b: {tmp_dj_db}')
    print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')
    print("--------------------------------------------------------------")
    print("")

    # Gradient descent method 
    print("========== Using gradient descent method to compute optimal model weights b and w ...")
    # initialize parameters
    print("***** Initializing model weights b and w ...")
    initial_w = np.zeros_like(w_init)
    print(f"***** initial_w = {initial_w}")
    initial_b = 0.
    print(f"***** initial_b = {initial_b}")
    # some gradient descent settings
    iterations = 1000
    alpha = 5.0e-7
    print(f"***** Setting interations to = {iterations}")
    print(f"***** Setting learning rate alpha to = {alpha}")
    # run gradient descent
    print("***** Running Gradient descent ...")
    w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
    m,_ = X_train.shape
    for i in range(m):
        print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

    # plot cost versus iteration  
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(J_hist)
    ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
    ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
    ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
    ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
    plt.show()