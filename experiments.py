import numpy as np
from models import LinearRegression, GradientDescent, RegressionWithBasesAndRegularization, L2RegularizedLinearRegression
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from process_data import clean
import math

def mse(yh, y):
    mean_sqe = 0
    yh = pd.DataFrame(yh)
    y = pd.DataFrame(y)
    yh = yh.rename(columns={0: 'Y1', 1: 'Y2'})
    y = y.rename(columns={6:"Y1", 7:"Y2"})
    for i in range(len(y)):
        yh_i = yh.iloc[i]
        y_i = y.iloc[i]        
        mean_sqe += ((yh_i-y_i).mean() ** 2)
    return mean_sqe / len(y)

def f1_score(yh, y):
    yh = pd.DataFrame(yh)
    y = pd.DataFrame(y)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(y)):
        yh_i = float(yh.iloc[i])
        y_i = float(y.iloc[i])
        if (yh_i <= 0.5):
            yh_i = 0
        else : yh_i = 1 

        if ((yh_i==1) & (y_i==1)):
            TP+=1
        if ((yh_i==0) & (y_i==0)):
            TN+=1
        if ((yh_i==0) & (y_i==1)):
            FN+=1
        if ((yh_i==1) & (y_i==0)):
            FP+=1
    if TP == 0:
        precision = 0
        recall = 0
    else:
        precision = TP/ (TP + FP)
        recall = TP /(TP +FN)
    if precision == 0 and recall == 0:
        return 0
    else:
        return (2*precision*recall)/(precision+recall)

def logistic(x):
    x = pd.DataFrame(x)
    return x.applymap(lambda x : 1.0 / (1.0 + math.exp(-x)))

def polynomial1(x): #polynomial k=2
    x = pd.DataFrame(x)
    return x.applymap(lambda x : x ** 2)
    
def polynomial2(x): #polynomial k=3
    x = pd.DataFrame(x)
    return x.applymap(lambda x : x ** 3)
    
def polynomial3(x): #polynomial k=4
    x = pd.DataFrame(x)
    return x.applymap(lambda x : x ** 4)
    
def gaussian(x, k, s=1):
    return math.exp(-((x-k) ** 2) / (s ** 2))

def gaussian1(x): #gaussian k=-0.5
    x = pd.DataFrame(x)
    return x.applymap(lambda x : gaussian(x, k=-0.5))
    
def gaussian2(x): #gaussian k=0
    x = pd.DataFrame(x)
    return x.applymap(lambda x : gaussian(x, k=0))
    
def gaussian3(x): #gaussian k=0.5
    x = pd.DataFrame(x)
    return x.applymap(lambda x : gaussian(x, k=0.5))
    
def sigmoid(x, k, s=1):
    return 1.0 / (1 + math.exp(-(x-k)/s))

def sigmoid1(x): #sigmoid k=-0.5
    x = pd.DataFrame(x)
    return x.applymap(lambda x : sigmoid(x, k=-0.5))
    
def sigmoid2(x): #sigmoid k=0
    x = pd.DataFrame(x)
    return x.applymap(lambda x : sigmoid(x, k=0))
    
def sigmoid3(x): #sigmoid k=0.5
    x = pd.DataFrame(x)
    return x.applymap(lambda x : sigmoid(x, k=0.5))

regression_file = "raw_datasets/ENB2012_data.xlsx"
classification_file = "raw_datasets/Qualitative_Bankruptcy.data.txt"

#visualize dataset distributions
#clean(classification_file, 0.8, True)
#clean(regression_file, 0.8, True)
def runTrainSizeExperiment():

    model = LinearRegression()

    train_performance_to_train_size = {}
    test_performance_to_train_size = {}

    #experiments with different train/test splits
    for train_split in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        r_train, r_test = clean(regression_file, train_split)
        r_train_X = r_train.iloc[:,:8]
        r_train_Y = r_train.iloc[:,8:]
        r_test_X = r_test.iloc[:,:8]
        r_test_Y = r_test.iloc[:,8:]

        model.fit(r_train_X,r_train_Y)
        train_results = mse(model.predict(r_train_X), r_train_Y)
        test_results = mse(model.predict(r_test_X), r_test_Y)
        if train_split == 0.8:
            print("Linear Regression (analytical, {}% train) training error:".format(100*train_split), train_results)
            print("Linear Regression (analytical, {}% train) test error:".format(100*train_split), test_results)
            print("Linear Regression (analytical, {}% train) weights:\n".format(100*train_split), model.w)
            
        train_performance_to_train_size[train_split] = train_results
        test_performance_to_train_size[train_split] = test_results

    plt.plot(list(train_performance_to_train_size.keys()), list(train_performance_to_train_size.values()))
    plt.title("Analytical linear regression train set MSE as a function of training size")
    plt.show()
    plt.plot(list(test_performance_to_train_size.keys()), list(test_performance_to_train_size.values()))
    plt.title("Analytical linear regression test set MSE as a function of training size")
    plt.show()


    #training size experiments for logistic regression
    model = RegressionWithBasesAndRegularization(non_linear_base_fn=logistic)
    gradient = GradientDescent(batch_size=None) #none means full batch

    train_performance_to_train_size = {}
    test_performance_to_train_size = {}

    for train_split in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        c_train, c_test = clean(classification_file, train_split)
        c_train_X = c_train.iloc[:,:6]
        c_train_Y = c_train.iloc[:,6]
        c_test_X = c_test.iloc[:,:6]
        c_test_Y = c_test.iloc[:,6]

        model.fit(c_train_X, c_train_Y, gradient)
        train_results = f1_score(model.predict(c_train_X), c_train_Y)
        test_results = f1_score(model.predict(c_test_X), c_test_Y)
        if train_split == 0.8:
            print("Logistic Regression (full-batched, {}% train) training performance:".format(100*train_split), train_results)
            print("Logistic Regression (full-batched, {}% train) test performance:".format(100*train_split), test_results)
            print("Logistic Regression (full-batched, {}% train) weights:\n".format(100*train_split), model.w)
        
        train_performance_to_train_size[train_split] = train_results
        test_performance_to_train_size[train_split] = test_results

    plt.plot(list(train_performance_to_train_size.keys()), list(train_performance_to_train_size.values()))
    plt.title("Logistic regression train set F1 score as a function of training size")
    plt.show()
    plt.plot(list(test_performance_to_train_size.keys()), list(test_performance_to_train_size.values()))
    plt.title("Logistic regression test set F1 score as a function of training size")
    plt.show()

def runMiniBatchExperiment():
    #mini batch size experiments for logistic regression
    model = RegressionWithBasesAndRegularization(non_linear_base_fn=logistic)

    test_performance_to_batch_size = {}
    train_performance_to_batch_size = {}
    num_iterations_to_batch_size = {}

    for batch_size in [8, 16, 32, 64, 128, None]:
        gradient = GradientDescent(batch_size=batch_size, epsilon=1e-1)
        c_train, c_test = clean(classification_file, 0.8)
        c_train_X = c_train.iloc[:,:6]
        c_train_Y = c_train.iloc[:,6]
        c_test_X = c_test.iloc[:,:6]
        c_test_Y = c_test.iloc[:,6]

        model.fit(c_train_X, c_train_Y, gradient)
        train_results = f1_score(model.predict(c_train_X), c_train_Y)
        test_results = f1_score(model.predict(c_test_X), c_test_Y)
        
        if batch_size is None:
            test_performance_to_batch_size["full"] = test_results
            train_performance_to_batch_size["full"] = train_results
            num_iterations_to_batch_size["full"] = gradient.iterationsPerformed
        else:
            test_performance_to_batch_size[batch_size] = test_results
            train_performance_to_batch_size[batch_size] = train_results
            num_iterations_to_batch_size[batch_size] = gradient.iterationsPerformed

    plt.plot(list(train_performance_to_batch_size.keys()), list(train_performance_to_batch_size.values()))
    plt.title("Logistic regression train set F1 score as a function of mini-batch size")
    plt.show()
    plt.plot(list(test_performance_to_batch_size.keys()), list(test_performance_to_batch_size.values()))
    plt.title("Logistic regression test set F1 score as a function of mini-batch size")
    plt.show()
    plt.plot(list(num_iterations_to_batch_size.keys()), list(num_iterations_to_batch_size.values()))
    plt.title("Logistic regression convergence speed (# iterations) as a function of mini-batch size")
    plt.show()


    #mini batch size experiments for linear regression
    model = RegressionWithBasesAndRegularization()

    test_performance_to_batch_size = {}
    train_performance_to_batch_size = {}
    num_iterations_to_batch_size = {}

    for batch_size in [8, 16, 32, 64, 128, None]:
        gradient = GradientDescent(batch_size=batch_size, epsilon=1e0)
        r_train, r_test = clean(regression_file, 0.8)
        r_train_X = r_train.iloc[:,:8]
        r_train_Y = r_train.iloc[:,8:]
        r_test_X = r_test.iloc[:,:8]
        r_test_Y = r_test.iloc[:,8:]

        model.fit(r_train_X, r_train_Y, gradient)
        train_results = mse(model.predict(r_train_X), r_train_Y)
        test_results = mse(model.predict(r_test_X), r_test_Y)
        
        if batch_size is None:
            test_performance_to_batch_size["full"] = test_results
            train_performance_to_batch_size["full"] = train_results
            num_iterations_to_batch_size["full"] = gradient.iterationsPerformed
        else:
            test_performance_to_batch_size[batch_size] = test_results
            train_performance_to_batch_size[batch_size] = train_results
            num_iterations_to_batch_size[batch_size] = gradient.iterationsPerformed

    plt.plot(list(train_performance_to_batch_size.keys()), list(train_performance_to_batch_size.values()))
    plt.title("Gradient descent linear regression train set MSE as a function of mini-batch size")
    plt.show()
    plt.plot(list(test_performance_to_batch_size.keys()), list(test_performance_to_batch_size.values()))
    plt.title("Gradient descent linear regression test set MSE as a function of mini-batch size")
    plt.show()
    plt.plot(list(num_iterations_to_batch_size.keys()), list(num_iterations_to_batch_size.values()))
    plt.title("Gradient descent linear regression convergence speed (# iterations) as a function of mini-batch size")
    plt.show()

def runLearningRateExperiment():
    #learning rate experiments for linear regression
    model = RegressionWithBasesAndRegularization()

    test_performance_to_learning_rate = {}
    train_performance_to_learning_rate = {}

    for lr in [0.01, 0.1, 0.5, 1.0]:
        gradient = GradientDescent(learning_rate=lr, batch_size=None)
        r_train, r_test = clean(regression_file, 0.8)
        r_train_X = r_train.iloc[:,:8]
        r_train_Y = r_train.iloc[:,8:]
        r_test_X = r_test.iloc[:,:8]
        r_test_Y = r_test.iloc[:,8:]

        model.fit(r_train_X, r_train_Y, gradient)
        train_results = mse(model.predict(r_train_X), r_train_Y)
        test_results = mse(model.predict(r_test_X), r_test_Y)
        
        test_performance_to_learning_rate[lr] = test_results
        train_performance_to_learning_rate[lr] = train_results

    plt.plot(list(train_performance_to_learning_rate.keys()), list(train_performance_to_learning_rate.values()))
    plt.title("Gradient descent linear regression train set MSE as a function of learning rate")
    plt.show()
    plt.plot(list(test_performance_to_learning_rate.keys()), list(test_performance_to_learning_rate.values()))
    plt.title("Gradient descent linear regression test set MSE as a function of learning rate")
    plt.show()


    #learning rate experiments for logistic regression
    model = RegressionWithBasesAndRegularization(non_linear_base_fn=logistic)

    test_performance_to_learning_rate = {}
    train_performance_to_learning_rate = {}

    for lr in [0.01, 0.1, 0.5, 1.0]:
        gradient = GradientDescent(learning_rate=lr, batch_size=None)
        c_train, c_test = clean(classification_file, 0.8)
        c_train_X = c_train.iloc[:,:6]
        c_train_Y = c_train.iloc[:,6]
        c_test_X = c_test.iloc[:,:6]
        c_test_Y = c_test.iloc[:,6]

        model.fit(c_train_X, c_train_Y, gradient)
        train_results = f1_score(model.predict(c_train_X), c_train_Y)
        test_results = f1_score(model.predict(c_test_X), c_test_Y)
        
        test_performance_to_learning_rate[lr] = test_results
        train_performance_to_learning_rate[lr] = train_results

    plt.plot(list(train_performance_to_learning_rate.keys()), list(train_performance_to_learning_rate.values()))
    plt.title("Logistic regression train set F1 score as a function of learning rate")
    plt.show()
    plt.plot(list(test_performance_to_learning_rate.keys()), list(test_performance_to_learning_rate.values()))
    plt.title("Logistic regression test set F1 score as a function of learning rate")
    plt.show()

def runMomentumExperiment():
    #momentum experiments for linear regression
    model = RegressionWithBasesAndRegularization()

    test_performance_to_momentum = {}
    train_performance_to_momentum = {}

    for momentum in [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]:
        gradient = GradientDescent(momentum=momentum, batch_size=None)
        r_train, r_test = clean(regression_file, 0.8)
        r_train_X = r_train.iloc[:,:8]
        r_train_Y = r_train.iloc[:,8:]
        r_test_X = r_test.iloc[:,:8]
        r_test_Y = r_test.iloc[:,8:]

        model.fit(r_train_X, r_train_Y, gradient)
        train_results = mse(model.predict(r_train_X), r_train_Y)
        test_results = mse(model.predict(r_test_X), r_test_Y)

        print(train_results)
        print(test_results)
        
        test_performance_to_momentum[momentum] = test_results
        train_performance_to_momentum[momentum] = train_results

    # plt.plot(list(train_performance_to_momentum.keys()), list(train_performance_to_momentum.values()))
    # plt.title("Gradient descent linear regression train set MSE as a function of momentum")
    # plt.show()
    # plt.plot(list(test_performance_to_momentum.keys()), list(test_performance_to_momentum.values()))
    # plt.title("Gradient descent linear regression test set MSE as a function of momentum")
    # plt.show()

def runL2RegularizationExperiment():
    #L2 regularization experiments for linear regression

    test_performance_to_L2 = {}
    train_performance_to_L2 = {}

    for lambdaa in [0.1, 1.0, 10.0]:
        model = L2RegularizedLinearRegression(l2_lambda=lambdaa)
        r_train, r_test = clean(regression_file, 0.8)
        r_train_X = r_train.iloc[:,:8]
        r_train_Y = r_train.iloc[:,8:]
        r_test_X = r_test.iloc[:,:8]
        r_test_Y = r_test.iloc[:,8:]

        model.fit(r_train_X, r_train_Y)
        train_results = mse(model.predict(r_train_X), r_train_Y)
        test_results = mse(model.predict(r_test_X), r_test_Y)
        print("\nWeights:", model.w, "\nTrain error:", train_results, "\nTest error:", test_results)
        
        test_performance_to_L2[lambdaa] = test_results
        train_performance_to_L2[lambdaa] = train_results

    plt.plot(list(train_performance_to_L2.keys()), list(train_performance_to_L2.values()))
    plt.title("Analytical linear regression train set MSE as a function of L2 regularization strength")
    plt.show()
    plt.plot(list(test_performance_to_L2.keys()), list(test_performance_to_L2.values()))
    plt.title("Analytical linear regression test set MSE as a function of L2 regularization strength")
    plt.show()


def runL1RegularizationExperiment():
    #L1 regularization experiments for linear regression

    test_performance_to_L1 = {}
    train_performance_to_L1 = {}

    for lambdaa in [0.1, 1.0, 10.0]:
        model = RegressionWithBasesAndRegularization(non_linear_base_fn=logistic, l1_lambda=lambdaa)
        gradient = GradientDescent(batch_size=None)
        r_train, r_test = clean(regression_file, 0.5)
        r_train_X = r_train.iloc[:,:8]
        r_train_Y = r_train.iloc[:,8:]
        r_test_X = r_test.iloc[:,:8]
        r_test_Y = r_test.iloc[:,8:]

        model.fit(r_train_X, r_train_Y, gradient)
        train_results = mse(model.predict(r_train_X), r_train_Y)
        test_results = mse(model.predict(r_test_X), r_test_Y)
        print("\nWeights:", model.w, "\nTrain error:", train_results, "\nTest error:", test_results)
        
        test_performance_to_L1[lambdaa] = test_results
        train_performance_to_L1[lambdaa] = train_results

    plt.plot(list(train_performance_to_L1.keys()), list(train_performance_to_L1.values()))
    plt.title("Gradient descent linear regression train set MSE as a function of L1 regularization strength")
    plt.show()
    plt.plot(list(test_performance_to_L1.keys()), list(test_performance_to_L1.values()))
    plt.title("Gradient descent linear regression test set MSE as a function of L1 regularization strength")
    plt.show()

def runNonLinearBasesExperiment():
    #non-linear bases experiments for regression on categorical dataset

    test_performance_to_base = {}
    train_performance_to_base = {}

    for base in [polynomial1, polynomial2, polynomial3, gaussian1, gaussian2, gaussian3, sigmoid1, sigmoid2, sigmoid3]:
        model = RegressionWithBasesAndRegularization(non_linear_base_fn=base)
        gradient = GradientDescent(batch_size=None)
        c_train, c_test = clean(classification_file, 0.8)
        c_train_X = c_train.iloc[:,:6]
        c_train_Y = c_train.iloc[:,6]
        c_test_X = c_test.iloc[:,:6]
        c_test_Y = c_test.iloc[:,6]

        model.fit(c_train_X, c_train_Y, gradient)
        train_results = f1_score(model.predict(c_train_X), c_train_Y)
        test_results = f1_score(model.predict(c_test_X), c_test_Y)
        
        base_names = {polynomial1:"polynomial k=2", polynomial2:"polynomial k=3", polynomial3:"polynomial k=4", gaussian1:"gaussian k=-0.5", gaussian2:"gaussian k=0", gaussian3:"gaussian k=0.5", sigmoid1:"sigmoid k=-0.5", sigmoid2:"sigmoid k=0", sigmoid3:"sigmoid k=0.5"}
        test_performance_to_base[base_names[base]] = test_results
        train_performance_to_base[base_names[base]] = train_results

    plt.plot(list(train_performance_to_base.keys()), list(train_performance_to_base.values()))
    plt.title("Categorical train set F1 score as a function of non-linear base")
    plt.show()
    plt.plot(list(test_performance_to_base.keys()), list(test_performance_to_base.values()))
    plt.title("Categorical test set F1 score as a function of non-linear base")
    plt.show()


    #non-linear bases experiments for regression on regression dataset

    test_performance_to_base = {}
    train_performance_to_base = {}

    for base in [polynomial1, polynomial2, polynomial3, gaussian1, gaussian2, gaussian3, sigmoid1, sigmoid2, sigmoid3]:
        model = RegressionWithBasesAndRegularization(non_linear_base_fn=base)
        gradient = GradientDescent(batch_size=None)
        r_train, r_test = clean(regression_file, 0.8)
        r_train_X = r_train.iloc[:,:8]
        r_train_Y = r_train.iloc[:,8:]
        r_test_X = r_test.iloc[:,:8]
        r_test_Y = r_test.iloc[:,8:]

        model.fit(r_train_X, r_train_Y, gradient)
        train_results = mse(model.predict(r_train_X), r_train_Y)
        test_results = mse(model.predict(r_test_X), r_test_Y)
        
        base_names = {polynomial1:"polynomial k=2", polynomial2:"polynomial k=3", polynomial3:"polynomial k=4", gaussian1:"gaussian k=-0.5", gaussian2:"gaussian k=0", gaussian3:"gaussian k=0.5", sigmoid1:"sigmoid k=-0.5", sigmoid2:"sigmoid k=0", sigmoid3:"sigmoid k=0.5"}
        test_performance_to_base[base_names[base]] = test_results
        train_performance_to_base[base_names[base]] = train_results

    plt.plot(list(train_performance_to_base.keys()), list(train_performance_to_base.values()))
    plt.title("Regression train set MSE as a function of non-linear base")
    plt.show()
    plt.plot(list(test_performance_to_base.keys()), list(test_performance_to_base.values()))
    plt.title("Regression test set MSE as a function of non-linear base")
    plt.show()

#runLearningRateExperiment()
#runTrainSizeExperiment()

#runMiniBatchExperiment()
runMomentumExperiment()
#runNonLinearBasesExperiment()
#runL2RegularizationExperiment()
runL1RegularizationExperiment()
