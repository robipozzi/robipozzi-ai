import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_validate

# ###### Program execution
if __name__ == '__main__':
    print("########################################")
    print("##### SCIKIT-LEARN getting started #####")
    print("########################################")
    print(" ")

    print("===== Random Forest example =====")
    clf = RandomForestClassifier(random_state=0)
    X = [[ 1,  2,  3],  # 2 samples, 3 features
        [11, 12, 13]]
    y = [0, 1]  # classes of each sample
    print("Training model with:")
    print(f"--> Input Features = \n {X}")
    print(f"--> Output = \n {y}")
    clf.fit(X, y)
    print(" ")

    print("Predict on Training Set ...")
    output = clf.predict(X)  # predict classes of the training data
    print(f"Output = \n {output}")
    print("Predict on New unseen data Set ...")
    newX = [[4, 5, 6], [14, 15, 16]]
    print(f"Input Features = \n {newX}")
    clf.predict(newX)  # predict classes of new data
    print(f"Output = \n {output}")
    print(" ")

    print("===== Linear Regression example =====")
    ln = LinearRegression()
    print("Training model with:")
    print(f"--> Input Features = \n {X}")
    print(f"--> Output = \n {y}")
    ln.fit(X, y)
    modelParams = ln.coef_
    intercept = ln.intercept_
    print(f"Model Params = \n {modelParams}")
    print(f"Intercept = \n {intercept}")
    print(" ")

    print("===== Transformers and pre-processors =====")
    from sklearn.preprocessing import StandardScaler
    X = [[0, 15],
        [1, -10]]
    # scale data according to computed scaling values
    output = StandardScaler().fit(X).transform(X)
    print(f"Input Features = \n {X}")
    print(f"Output = \n {output}")
    print(" ")

    print("===== Pipelines: chaining pre-processors and estimators =====")
    print("# create a pipeline object, chaining StandardScaler and LogisticRegression")
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression()
    )
    print("# load the iris dataset and split it into train and test sets")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print("# fit the whole pipeline")
    pipe.fit(X_train, y_train)
    print("# we can now use it like any other estimator")
    accuracyScore = accuracy_score(pipe.predict(X_test), y_test)
    print(f"Accuracy score is {accuracyScore}")
    print(" ")

    print("===== Model evaluation =====")
    X, y = make_regression(n_samples=1000, random_state=0)
    lr = LinearRegression()
    result = cross_validate(lr, X, y)  # defaults to 5-fold CV
    testScore = result['test_score']  # r_squared score is high because dataset is easy
    print(f"Test score on cross validate is {testScore}")
    print(" ")