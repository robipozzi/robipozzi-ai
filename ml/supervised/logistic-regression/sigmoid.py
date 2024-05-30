import numpy as np
#%matplotlib widget
import matplotlib.pyplot as plt
#from plt_one_addpt_onclick import plt_one_addpt_onclick
#from lab_utils_common import draw_vthresh
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
# ***************************
# ***** Functions - END *****
# ***************************

# ###### Program execution
if __name__ == '__main__':
    # Input is an array. 
    input_array = np.array([1,2,3])
    exp_array = np.exp(input_array)

    print("Input to exp:", input_array)
    print("Output of exp:", exp_array)

    # Input is a single number
    input_val = 1  
    exp_val = np.exp(input_val)

    print("Input to exp:", input_val)
    print("Output of exp:", exp_val)

    print("===== Generate an array of evenly spaced values between -10 and 10 ...")
    # Generate an array of evenly spaced values between -10 and 10
    z_tmp = np.arange(-10,11)

    print("===== Apply sigmoid function to values generated before ...")
    # Use the function implemented above to get the sigmoid values
    y = sigmoid(z_tmp)

    print("===== Print Input values along with Output from sigmoid function ...")
    # Code for pretty printing the two arrays next to each other
    np.set_printoptions(precision=3) 
    print("Input (z), Output (sigmoid(z))")
    print(np.c_[z_tmp, y])

    print("===== Plot Input values vs Output from sigmoid function ...")
    # Plot z vs sigmoid(z)
    fig,ax = plt.subplots(1,1,figsize=(5,3))
    ax.plot(z_tmp, y, c="b")
    ax.set_title("Sigmoid function")
    ax.set_ylabel('sigmoid(z)')
    ax.set_xlabel('z')
    plt.show()
    #draw_vthresh(ax,0)

    print("===== Apply Logistic regression ...")
    print("Initializing training set ...")
    x_train = np.array([0., 1, 2, 3, 4, 5])
    y_train = np.array([0,  0, 0, 1, 1, 1])
    print(f"Features X:\n{x_train}")
    print(f"Output Y:\n{y_train}")
    print("Initializing model parameters ...")
    w_in = np.zeros((1))
    b_in = 0
    print(f"w_in:\n{w_in}")
    print(f"b_in:\n{b_in}")
    
    plt.close('all') 
    #addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=True)