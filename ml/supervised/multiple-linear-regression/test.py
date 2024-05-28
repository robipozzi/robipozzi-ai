# Assisted by WCA@IBM
# Latest GenAI contribution: ibm/granite-20b-code-instruct-v2

# In this function, 
# - X is the feature matrix, 
# - y is the target vector, 
# - theta is the parameter vector, 
# - alpha is the learning rate, 
# - num_iters is the number of iterations to run the algorithm. 
# The function updates the parameter vector using gradient descent and returns the updated vector along with a history of the cost function 
# over each iteration.
def gradient_descent(X, y, theta, alpha, num_iters):
    m, n = X.shape
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        h = hypothesis(X, theta)
        grad = (1 / m) * np.dot(X.T, (h - y))
        theta -= alpha * grad
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history

# Assisted by WCA@IBM
# Latest GenAI contribution: ibm/granite-20b-code-instruct-v2
def compute_cost(X, y, theta):
    m = len(y)
    J = (1 / (2 * m)) * np.sum(np.square(hypothesis(X, theta) - y))
    return J

# Assisted by WCA@IBM
# Latest GenAI contribution: ibm/granite-20b-code-instruct-v2
def hypothesis(X, theta):
    return np.dot(X, theta)

# Assisted by WCA@IBM
# Latest GenAI contribution: ibm/granite-20b-code-instruct-v2
theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)
