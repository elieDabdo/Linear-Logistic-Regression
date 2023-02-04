import numpy as np
from models import LinearRegression, LogisticRegression, GradientDescent
import matplotlib.pyplot as plt
import pickle

def mse(yh, y):
    mean_sqe = [0,0]
    for i in range(len(y)):
        yh_i = float(yh.iloc[i][0])
        y_i = float(y.iloc[i][0])
        mean_sqe[0] += ((yh_i-y_i) ** 2) / len(y)
        yh_i = float(yh.iloc[i][1])
        y_i = float(y.iloc[i][1])
        mean_sqe[1] += ((yh_i-y_i) ** 2) / len(y)
    return mean_sqe 


r_train_file = "clean_data\ENB2012_data_train.sav"
r_test_file = "clean_data\ENB2012_data_test.sav"

with open(r_train_file, "rb") as f:
    r_train = pickle.load(f)

with open(r_test_file, "rb") as f:
    r_test = pickle.load(f)

r_train_X = r_train.iloc[:,:8]
r_train_Y = r_train.iloc[:,8:]
r_test_X = r_test.iloc[:,:8]
r_test_Y = r_test.iloc[:,8:]

model = LinearRegression()
model.fit(r_train_X,r_train_Y)

print(mse(model.predict(r_train_X), r_train_Y))
print(mse(model.predict(r_test_X), r_test_Y))
print(model.w)