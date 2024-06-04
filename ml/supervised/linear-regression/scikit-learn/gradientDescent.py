import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=2)

# *****************************
# ***** Functions - START *****
# *****************************
# ===== Function to calculate the cost
def load_house_data():
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])
    return X_train, y_train
# ***************************
# ***** Functions - END *****
# ***************************

# ###### Program execution
if __name__ == '__main__':
    print("###################################################################")
    print("########## Multiple Linear regression using SCIKIT-LEARN ##########")
    print("###################################################################")
    print("")
    print("========== Loading training set ...")
    X_train, y_train = load_house_data()
    X_features = ['size(sqft)','bedrooms','floors','age']
    print("Printing training set ...")
    print(f"Features X:\n{X_train}")
    print(f"Output Y:\n{y_train}")
    print("--------------------------------------------------------------")
    print("")

    print("========== Feature scaling using Z-score normalization ...")
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_train)
    print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
    print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")
    print("--------------------------------------------------------------")
    print("")

    print("========== Fit linear regression model ...")
    iterations = 1300
    print(f"Fitting linear regression model using {iterations} iterations ...")
    sgdr = SGDRegressor(max_iter=iterations)
    sgdr.fit(X_norm, y_train)
    print(sgdr)
    print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

    b_norm = sgdr.intercept_
    w_norm = sgdr.coef_
    print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
    #print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")
    print("--------------------------------------------------------------")
    print("")

    print("========== Test prediction on training set ...")
    # make a prediction using sgdr.predict()
    y_pred_sgd = sgdr.predict(X_norm)
    # make a prediction using w,b. 
    y_pred = np.dot(X_norm, w_norm) + b_norm  
    print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

    print(f"Prediction on training set (using sgdr.predict()):\n{y_pred_sgd[:4]}" )
    print(f"Prediction on training set (using np.dot()):\n{y_pred[:4]}" )
    print(f"Target values \n{y_train[:4]}")
    print("--------------------------------------------------------------")
    print("")

    # plot predictions and targets vs original features    
    fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:,i],y_train, label = 'target')
        ax[i].set_xlabel(X_features[i])
        #ax[i].scatter(X_train[:,i],y_pred,color=dlc["dlorange"], label = 'predict')
        ax[i].scatter(X_train[:,i],y_pred, label = 'predict')
    ax[0].set_ylabel("Price"); ax[0].legend();
    fig.suptitle("target versus prediction using z-score normalized model")
    plt.show()