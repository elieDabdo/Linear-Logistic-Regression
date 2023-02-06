import numpy as np

#closed form linear regression
class LinearRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
    
    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]                         #add a dimension for the features
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])    #add bias by adding a constant feature of value 1
        self.w = np.linalg.inv(x.T @ x)@x.T@y
        return self
    
    def predict(self, x):                       #add a dimension for the features
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])
        yh = x@self.w                             #predict the y values
        return yh

#closed form with L2 regularization
class L2RegularizedLinearRegression:
    def __init__(self, add_bias=True, l2_reg=0):
        self.add_bias = add_bias
        #l2 reg decides strength (coefficient) of regularization
        self.l2_reg = l2_reg
        pass
            
    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]                         #add a dimension for the features
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])    #add bias by adding a constant feature of value 1
        #identity matrix might need to be size D
        self.w = np.linalg.inv(x.T @ x + self.l2_reg*np.identity(N))@x.T@y
        return self
    
    def predict(self, x):                       #add a dimension for the features
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])
        yh = x@self.w                             #predict the y values
        return yh

class GradientDescent:
    def __init__(self, learning_rate=.001, max_iters=1e4, epsilon=1e-8, momentum=0):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.momentum = momentum
        self.previousGrad = None
            
    def run(self, gradient_fn, x, y, w):
        grad = np.inf
        t = 1
        while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:
            grad = gradient_fn(x, y, w)               # compute the gradient with present weight
            if previousGrad == None: previousGrad = grad
            grad = grad*(1.0-self.momentum) + previousGrad*self.momentum
            previousGrad = grad
            w = w - self.learning_rate * grad         # weight update step
            t += 1
        return w

#gradient descent regression with options for any combinations of non-linear bases, l1, l2 regularization
class RegressionWithBasesAndRegularization:
    def __init__(self, add_bias=True, non_linear_base_fn = (lambda x: x), l1_lambda=0, l2_lambda=0):
        self.add_bias = add_bias
        self.non_linear_base_fn = non_linear_base_fn
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
            
    def fit(self, x, y, optimizer):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)])
        N,D = x.shape
        def gradient(x, y, w):                          # define the gradient function
            yh =  self.non_linear_base_fn(x @ w) 
            N, D = x.shape
            grad = .5*np.dot(yh - y, x)/N
            if self.add_bias:
                grad[1:] += self.l1_lambda * np.sign(w[1:])
                grad[1:] += self.l2_lambda * w[1:]
            else:
                grad += self.l1_lambda * np.sign(w)
                grad += self.l2_lambda * w
            return grad
        w0 = np.zeros(D)                                # initialize the weights to 0
        self.w = optimizer.run(gradient, x, y, w0)      # run the optimizer to get the optimal weights
        return self
    
    def predict(self, x):
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])
        yh = x@self.w
        return yh