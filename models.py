import numpy as np
import pandas as pd

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
    def __init__(self, learning_rate=.001, max_iters=1e4, epsilon=1e-8, momentum=0, batch_size=None):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.momentum = momentum
        self.previousGrad = None
        self.batch_size = batch_size

    def make_batches(self, x, y, sizeOfMiniBatch):
        if (sizeOfMiniBatch==None):
            return [(x,y)]
        if x.ndim == 1:
            x = x[:, None]                      #add a dimension for the features
        batches = []
        datax = pd.DataFrame(x)
        datay =pd.DataFrame(y)
        data = pd.concat([datax,datay],axis=1, join='inner')
        data = data.sample(frac=1).reset_index(drop=True)
        x = data.iloc[:,:9]
        y = data.iloc[:,9:]
        numberOfRowsData = x.shape[0]        #number of rows in our data
        
        i = 0
        for i in range(int(numberOfRowsData/sizeOfMiniBatch)):
            endOfBatch= (i+1)*sizeOfMiniBatch           
            if endOfBatch<numberOfRowsData: #if end of the batch is still within range allowed
                single_batch_x = x.iloc[i * sizeOfMiniBatch:endOfBatch, :] #slice into a batch
                single_batch_y = y.iloc[i * sizeOfMiniBatch:endOfBatch, :] #slice into a batch
                batches.append((single_batch_x, single_batch_y))
            else: #if end of batch not within range 
                single_batch_x = x.iloc[i * sizeOfMiniBatch:numberOfRowsData, :] #slice into a batch
                single_batch_y = y.iloc[i * sizeOfMiniBatch:numberOfRowsData, :] #slice into a batch
                batches.append((single_batch_x, single_batch_y))
        return batches
            
    def run(self, gradient_fn, x, y, w):

        batches = self.make_batches(x,y, self.batch_size)

        grad = np.inf
        t = 1
        i = 1
        while np.linalg.norm(grad) > self.epsilon and i < self.max_iters:
            if (t-1)>=len(batches):
                batches = self.make_batches(x,y, self.batch_size)
                t=1
            grad = gradient_fn(batches[t-1][0], batches[t-1][1], w)  # compute the gradient with present weight
            if self.previousGrad is None: self.previousGrad = grad
            grad = grad*(1.0-self.momentum) + self.previousGrad*self.momentum
            self.previousGrad = grad
            w = w - self.learning_rate * grad         # weight update step
            t += 1
            i+=1
        self.iterationsPerformed = i
        return w

#gradient descent regression with options for any combinations of non-linear bases, l1, l2 regularization
class RegressionWithBasesAndRegularization:
    def __init__(self, add_bias=True, non_linear_base_fn=(lambda x: x), l1_lambda=0, l2_lambda=0):
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
            yh =  self.non_linear_base_fn(x @ w.T) 
            N, D = x.shape
            yh = pd.DataFrame(yh)
            y = pd.DataFrame(y)
            yh = yh.rename(columns={0: 'Y1', 1: 'Y2'})
            y = y.rename(columns={6:"Y1", 7:"Y2"})
            grad = .5*np.dot((yh - y).transpose(), x)/N
            if self.add_bias:
                if len(y.columns) > 1:
                    grad[:,1:] += self.l1_lambda * np.sign(w[:,1:])
                    grad[:,1:] += self.l2_lambda * w[:,1:]
                else:
                    grad[1:] += self.l1_lambda * np.sign(w[1:])
                    grad[1:] += self.l2_lambda * w[1:]
            else:
                grad += self.l1_lambda * np.sign(w)
                grad += self.l2_lambda * w
            return grad
        w0 = np.zeros((len(pd.DataFrame(y).columns),D))                                # initialize the weights to 0
        self.w = optimizer.run(gradient, x, y, w0)      # run the optimizer to get the optimal weights
        return self
    
    def predict(self, x):
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)])
        yh = x@self.w.T
        return yh

    