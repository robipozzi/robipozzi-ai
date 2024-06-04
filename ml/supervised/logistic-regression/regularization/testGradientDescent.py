import copy, math
import numpy as np
#%matplotlib widget
import matplotlib.pyplot as plt
#from lab_utils_common import  dlc, plot_data, plt_tumor_data, sigmoid, compute_cost_logistic
#from plt_quad_logistic import plt_quad_logistic, plt_prob
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
            p[k] = int(f_wb[k] >= 0.5)
        return p
    # Compute prediction (0 or 1) in case f_wb is a number
    p = int(f_wb >= 0.5)
    return p

# Logistic regression model function (no vectorization)
def predictNoVectorization(X, w, b): 
    """
    single predict using logistic regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    # number of training examples
    m, n = X.shape
    print(f"m: {m}")
    print(f"n: {n}")
    p = np.zeros(m)

    for i in range(m):   
        # Calculate f_wb (exactly how you did it in the compute_cost function above)
        z_wb = 0
        # Loop over each feature
        for j in range(n): 
            # Add the corresponding term to z_wb
            z_wb_ij = X[i, j] * w[j]
            z_wb += z_wb_ij
            
        # Add bias term 
        z_wb += b
        # Calculate the prediction from the model
        f_wb = sigmoid(z_wb)
        # Apply the threshold
        p[i] = f_wb >= 0.5

    return p

# Logistic Regression Cost function
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

# Logistic Regression Cost function with regularization
def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """

    m,n  = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b                                      #(n,)(n,)=scalar, see np.dot
        f_wb_i = sigmoid(z_i)                                          #scalar
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)      #scalar
             
    cost = cost/m                                                      #scalar

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar
    
    total_cost = cost + reg_cost                                       #scalar
    return total_cost                                                  #scalar

# Compute Gradient function with Regularization
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

def compute_gradient_logistic_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                            #(n,)
    dj_db = 0.0                                       #scalar

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw  

# Gradient descent
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

    fig,ax = plt.subplots(1,1,figsize=(4,4))
    #plot_data(X_train, y_train, ax)
    ax.axis([0, 4, 0, 3.5])
    ax.set_ylabel('$x_1$', fontsize=12)
    ax.set_xlabel('$x_0$', fontsize=12)
    #plt.show()

    print("--------------------------------------------------------------")
    print("")

    print("========== Computing Gradient for logistic regression ...")
    w_tmp = np.array([2.,3.])
    b_tmp = 1.
    print("***** Initalizing and printing model parameters ...")
    print(f"w_tmp: {w_tmp}, w_tmp Type:{type(w_tmp)})")
    print(f"b_tmp: {b_tmp}, b_tmp Type:{type(b_tmp)})")

    dj_db_tmp, dj_dw_tmp = compute_gradient_logistic(X_train, y_train, w_tmp, b_tmp)
    print(f"dj_db: {dj_db_tmp}" )
    print(f"dj_dw: {dj_dw_tmp.tolist()}" )
    print("--------------------------------------------------------------")
    print("")

    print("========== Computing Gradient Descent for logistic regression ...")
    w_tmp  = np.zeros_like(X_train[0])
    b_tmp  = 0.
    alph = 0.1
    iters = 10000

    w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
    print(f"\nupdated parameters: w:{w_out}, b:{b_out}")
    
    print("--------------------------------------------------------------")
    print("")

    print("########## USING REGULARIZATION ##########")

    # Compute and display cost using regularization
    print("========== Computing cost with regularization ...")
    np.random.seed(1)
    X_tmp = np.random.rand(5,6)
    y_tmp = np.array([0,1,0,1,0])
    w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
    b_tmp = 0.5
    lambda_tmp = 0.7
    print(f"X_tmp shape: {X_tmp.shape}")
    print(f"X_tmp: {X_tmp}")
    print(f"y_tmp: {y_tmp}")
    print(f"w_tmp: {w_tmp}")
    print(f"b_tmp: {b_tmp}")
    print(f"lambda_tmp: {lambda_tmp}")
    cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

    print("Regularized cost:", cost_tmp)
    print("--------------------------------------------------------------")
    print("")

    # Compute Gradient with regularization
    print("========== Computing gradient with regularization ...")
    np.random.seed(1)
    X_tmp = np.random.rand(5,3)
    y_tmp = np.array([0,1,0,1,0])
    w_tmp = np.random.rand(X_tmp.shape[1])
    b_tmp = 0.5
    lambda_tmp = 0.7
    print(f"X_tmp shape: {X_tmp.shape}")
    print(f"X_tmp: {X_tmp}")
    print(f"y_tmp: {y_tmp}")
    print(f"w_tmp: {w_tmp}")
    print(f"b_tmp: {b_tmp}")
    print(f"lambda_tmp: {lambda_tmp}")

    dj_db_tmp, dj_dw_tmp =  compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
    print(f"dj_db: {dj_db_tmp}", )
    print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )
    

    print("--------------------------------------------------------------")
    print("")