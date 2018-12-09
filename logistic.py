import pandas as pd
import numpy as np
import math

df=pd.read_csv('Social_Network_Ads.csv')

X=df.iloc[:,1:4].values
y=df.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,0]=le.fit_transform(X[:,0])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

d = np.ones((len(X_train), 1))
X_train=np.append(d, X_train, axis=1)
theta=np.array([1.0,1.0,1.0,1.0])
hypothesis=0
thetaTX=0
for k in range(1000):
    for i in range(len(X_train)):
        thetaTX=np.matmul(theta.transpose(),X_train[i])
        hypothesis=1/(1+math.exp(-1*thetaTX))
        for j in range(len(theta)):
            theta[j]=theta[j]-(((0.01/len(X_train))*((hypothesis-y_train[i])*X_train[i][j])))
        
print(theta)
y_pred=[]
for i in range(len(X_test)):
    thetaTX=np.matmul(theta.transpose(),X_train[i])
    hypothesis=1/(1+math.exp(-1*thetaTX))
    if hypothesis>=0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)


