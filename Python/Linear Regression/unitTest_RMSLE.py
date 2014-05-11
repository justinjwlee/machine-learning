import pandas as pd
import numpy as np
from math import sqrt, log

def evaluate1(y1, y2):
    y1_log = np.log(y1 + np.ones(y1.shape))
    y2_log = np.log(y2 + np.ones(y2.shape))
    return sqrt(np.mean(np.square(y1_log - y2_log)))

def evaluate2(y1, y2): 
    a, b = y1.tolist(), y2.tolist()
    score = 0.0
    nrows = len(a)
    if nrows != len(b): 
         raise Exception("Mismatched length")
    for i in xrange(nrows):
         score += (log(a[i] + 1.0) - log(b[i] + 1.0))**2
    return sqrt(score / nrows)

df1 = pd.read_excel("Sales.xlsx",0)   # read from excel file
#df = pd.read_csv("./Data/Actual Sales.csv")  # read from csv file
df1['Sales qty'] = df1['Sales qty'].clip(lower=0.0)   # clip minimum sales qty to zero
actual = df1['Sales qty'].values   # extract actual sales

df2 = pd.read_excel("predicted.xlsx",0)   # read from excel file
#df2 = pd.read_csv("./Data/Actual Sales.csv")  # read from csv file
df2['Sales qty'] = df2['predict qty'].clip(lower=0.0)   # clip minimum sales qty to zero
prediction = df2['predict qty'].values   # extract actual sales



#prediction = np.array([150.0 for i in xrange(len(df))])    # replace this line with the predictions from the models
print "RMSLE (evaluate1) = %f" % evaluate1(actual, prediction)
print "RMSLE (evaluate2) = %f" % evaluate2(actual, prediction)


