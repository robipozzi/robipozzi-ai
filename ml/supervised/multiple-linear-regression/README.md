# Multiple variable Linear Regression
- [Model Function](#model-function)
- [Cost Function](#cost-function)
- [Gradient Descent](#gradient-descent)
- [Regularization](#regularization)

## Model Function
[TODO]

## Cost Function
[TODO]

## Gradient Descent
[TODO]

## Regularization
Regularization serves the purpose to limit model parameters wj in order to avoid or at least minimize overfitting.

The equation for the cost function regularized linear regression is:

![](img/RegularizedLinearRegressionCostFunction.png)

where 

![](img/LinearRegressionModelFunction.png)

The difference is the regularization term

![](img/RegularizationTerm.png)

Including this term encourages gradient descent to minimize the size of the parameters. Note, in this example, the parameter 𝑏 is not regularized. This is standard practice.

Here is how Linear regression cost function and gradient descent equations gets modified when regularization is added.

![](img/RegularizedLinearRegression.png)


***************************************************************************************************************************************************
***** Credit to Andrew Ng (definitions and formulas are taken from his course **Supervised Machine Learning: Regression and Classification**) *****
***************************************************************************************************************************************************