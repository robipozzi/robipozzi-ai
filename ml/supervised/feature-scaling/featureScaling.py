import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import  load_house_data, run_gradient_descent 
from lab_utils_multi import  norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc
np.set_printoptions(precision=2)
#plt.style.use('./deeplearning.mplstyle')

def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma
    return (X_norm, mu, sigma)
 
    #check our work
    #from sklearn.preprocessing import scale
    #scale(X_orig, axis=0, with_mean=True, with_std=True, copy=True)

# ###### Program execution
if __name__ == '__main__':
    # load the dataset
    print("========== Load training set ...")
    X_train, y_train = load_house_data()
    X_features = ['size(sqft)','bedrooms','floors','age']
    print("--------------------------------------------------------------")
    print("")

    print("========== Show the dataset ...")
    fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:,i],y_train)
        ax[i].set_xlabel(X_features[i])
    ax[0].set_ylabel("Price (1000's)")
    plt.show()
    print("--------------------------------------------------------------")
    print("")

    print("========== Find appropriate learning rate to make gradient descent converge ...")
    #set alpha to 9.9e-7
    alpha = 9.9e-7
    print(f"Set Learning rate alpha to = {alpha}")
    _, _, hist = run_gradient_descent(X_train, y_train, 10, alpha = alpha)
    plot_cost_i_w(X_train, y_train, hist)

    #set alpha to 9e-7
    alpha = 9e-7
    print(f"Set Learning rate alpha to a smaller value = {alpha}")
    _,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = alpha)
    plot_cost_i_w(X_train, y_train, hist)

    #set alpha to 1e-7
    alpha = 1e-7
    print(f"Set Learning rate alpha to an even smaller value = {alpha}")
    _,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = alpha)
    print("--------------------------------------------------------------")
    print("")

    print("========== Feature scaling using Z-score normalization ...")
    mu     = np.mean(X_train,axis=0)
    print(f"Calculate feature mean value = {mu}")
    sigma  = np.std(X_train,axis=0) 
    print(f"Calculate standard deviation = {sigma}")
    X_mean = (X_train - mu)
    X_norm = (X_train - mu)/sigma      

    fig,ax=plt.subplots(1, 3, figsize=(12, 3))
    ax[0].scatter(X_train[:,0], X_train[:,3])
    ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
    ax[0].set_title("unnormalized")
    ax[0].axis('equal')

    ax[1].scatter(X_mean[:,0], X_mean[:,3])
    ax[1].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
    ax[1].set_title(r"X - $\mu$")
    ax[1].axis('equal')

    ax[2].scatter(X_norm[:,0], X_norm[:,3])
    ax[2].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
    ax[2].set_title(r"Z-score normalized")
    ax[2].axis('equal')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle("distribution of features before, during, after normalization")
    plt.show()
    print("--------------------------------------------------------------")
    print("")

    print("========== Normalize the original features with Z-score normalization ...")
    # normalize the original features
    X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
    print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
    print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
    print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

    print("Plot feature before and after Z-score normalization ...")
    fig,ax=plt.subplots(1, 4, figsize=(12, 3))
    for i in range(len(ax)):
        norm_plot(ax[i],X_train[:,i],)
        ax[i].set_xlabel(X_features[i])
    ax[0].set_ylabel("count");
    fig.suptitle("distribution of features before normalization")
    plt.show()
    fig,ax=plt.subplots(1,4,figsize=(12,3))
    for i in range(len(ax)):
        norm_plot(ax[i],X_norm[:,i],)
        ax[i].set_xlabel(X_features[i])
    ax[0].set_ylabel("count"); 
    fig.suptitle("distribution of features after normalization")

    plt.show()

    print("--------------------------------------------------------------")
    print("")

    print("========== Running Gradient descent with normalized features ...")
    w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )
    print("--------------------------------------------------------------")
    print("")

    print("========== Verifying model predictions on training sets using normalized features ...")
    #predict target using normalized features
    m = X_norm.shape[0]
    yp = np.zeros(m)
    for i in range(m):
        yp[i] = np.dot(X_norm[i], w_norm) + b_norm

        # plot predictions and targets versus original features    
    fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:,i],y_train, label = 'target')
        ax[i].set_xlabel(X_features[i])
        ax[i].scatter(X_train[:,i],yp,color=dlc["dlorange"], label = 'predict')
    ax[0].set_ylabel("Price"); ax[0].legend();
    fig.suptitle("target versus prediction using z-score normalized model")
    plt.show()
    print("--------------------------------------------------------------")
    print("")

    print("========== Predictions using normalized features ...")
    # First, normalize out example.
    x_house = np.array([1200, 3, 1, 40])
    x_house_norm = (x_house - X_mu) / X_sigma
    print(x_house_norm)
    x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
    print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")
    plt_equal_scale(X_train, X_norm, y_train)
    print("--------------------------------------------------------------")
    print("")