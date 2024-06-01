# Classification and Logistic Regression
- [Model Function](#model-function)
- [Loss and Cost Functions](#loss-and-cost-functions)
- [Gradient Descent](#gradient-descent)

## Model Function
The predictions of a classification model must be between 0 and 1 since the output variable y is either 0 or 1.

This can be accomplished by using a "sigmoid function" which outputs input values to values between 0 and 1.

![](img/Sigmoid.png)

In the case of logistic regression, z (the input to the sigmoid function), is the output of a linear regression model.

A logistic regression model applies the sigmoid to the familiar linear regression model as shown below:

![](img/LogisticRegression.png)

## Loss and Cost Functions
Logistic Regression uses a loss function more suited to the task of categorization where the target is 0 or 1 rather than any number.

![](img/LossFunction.png)

The defining feature of this loss function is the fact that it uses two separate curves.

One for the case when the target is zero or (𝑦=0) and another for when the target is one (𝑦=1).

Combined, these curves provide the behavior useful for a loss function, namely, being zero when the prediction matches the target and rapidly increasing in value as the prediction differs from the target.

The loss function can be simplified as follows:

![](img/SimplifiedLossFunction.png)

## Gradient Descent
The gradient descent algorithm utilizes the gradient calculation:

![](img/GradientDescent1.png)

Each iteration performs simultaneous updates on 𝑤𝑗 for all 𝑗

![](img/GradientDescent2.png)

where

![](img/GradientDescent3.png)

In the end the formula for Gradient Descent is exactly the same for both Logistic and Linear regression, what changes is the model function.

![](img/GradientDescent4.png)

***************************************************************************************************************************************************
***** Credit to Andrew Ng (definitions and formulas are taken from his course **Supervised Machine Learning: Regression and Classification**) *****
***************************************************************************************************************************************************