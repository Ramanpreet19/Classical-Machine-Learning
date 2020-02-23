import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
import numpy as np

#e) Plot the data and show the class of the sample using different colors.
colnames = ['feature1', 'feature2', 'class']
data = pd.read_csv('./dataset3.txt', names=colnames, header=None)
fig = plt.figure(figsize=(10,8))
sns.scatterplot(x="feature1",  data= data, y="feature2", hue="class").set_title("Scatter plot for data distribution", fontsize=20)
#print(data.describe())

def sigmoid(x):
    '''
    function used to calculate the probabilities of class 
    '''
    return  1/(1 + np.exp(-(x)))

def y_predicted(input_data, theta):
    '''
    This function claculates the probablity of the class 
    '''
    #print(np.dot(input_data, theta))
    return  sigmoid(np.dot(input_data, theta))

def cost_function(theta, x, y, y_dash):
    '''
    This function computes the cost function y_predicted(x, theta)
    '''
    N = len(x)
    cost = -(1/N)*np.sum(y*np.log(y_dash) + (1-y)*np.log(1- y_dash))
    return cost

def gradient(theta, x, y, y_dash):
    '''
    the gradient of the cost function at the point theta
    '''
    N = x.shape[0]
    diff_prob = y_dash - y
    #print("x.shape; ", x.shape)
    delta_grad = (1/N)*(np.dot(diff_prob.T, x))
    #print("````",delta_grad)
    return delta_grad.T

def sgd_optimize(alpha, theta, n_iter, x, y):
    '''
    '''
    epocs = range(n_iter)
    costs = []
    #print("x, y shape:", x.shape, y.shape)
    X = x.copy()
    Y = y.copy()
    N = x.shape[0]
    for i in epocs:
        y_dash = y_predicted(x, theta)
        XY = np.hstack((X,Y))
        #np.random.shuffle(XY)
        x = XY[:, :-1]
        y = XY[:, -1].reshape(len(x), 1)
        #print("x, y shape:", x.shape, y.shape)
        for j in range (0, N):
            y_dash[j] = y_predicted(x[j], theta)
            x_ = x[j].reshape(1, x[j].shape[0])
            y_ = y[j].reshape(y[j].shape[0], -1)
            y_dash_ = y_dash[j].reshape(y_dash[j].shape[0], -1)
            cost = cost_function(theta, x_, y_, y_dash_)
            delta_theta = gradient(theta, x_, y_, y_dash_)
            theta = theta - alpha * delta_theta
        costs.append(cost)
            #new_cost = cost_function(theta, x, y)
            #print("Epoc = "+str(i)+"; Theta = "+str(theta)+"; Cost = "+str(cost))
    return costs, epocs, theta

def accuracy(y, y_dash):
    #print(y_dash)
    #print(max(y_dash), min(y_dash))
    y_dash = y_dash > 0.5
    y_dash = y_dash.astype(int)
    diff = abs(y - y_dash)
    predicted = np.sum(diff)
    #print(predicted)
    accuracy = (1 - predicted/len(y)) * 100
    print("Accuracy is : "+str(accuracy))

N = data.shape[0]
#print(N)
f0 = np.ones((N,1))
f1 = data["feature1"].to_numpy()
#print(type(f1), type(f0))
f1 = f1.reshape(f1.shape[0],-1)
f2 = data["feature2"].to_numpy()
f2 = f2.reshape(f2.shape[0],-1)
#print(f1.shape, f2.shape, f0.shape)

#regularize data:
f1 = (f1 - np.mean(f1)) / (np.max(f1) - np.min(f1))
f2 = (f2 - np.mean(f2)) / (np.max(f2) - np.min(f2))

feature = np.hstack((f0, f1, f2))
#print (feature.shape)
theta = np.random.random((3,1)) * 0.1
#theta = np.zeros((3,1))
#print(theta)
#print(theta.shape)
y = data["class"].to_numpy()
y = y.reshape(y.shape[0], -1)
#print (y.shape)

costs, epocs, theta = sgd_optimize(.001, theta, 1000, feature, y)
#print("cost.shape: ", len(costs))
#print("epocs.shape: ", len(epocs))
t = np.arange(0,100000)
#e = range(epocs)
#plt.subplot(221)
fig = plt.figure(figsize=(10,8))
plt.xlabel("Number of epoches",Fontsize=16)
plt.ylabel("Value of cost function", Fontsize=16)
plt.title("Value of cost function vs number of epoches", Fontsize=20)
plt.plot(epocs, costs)

y_dash = y_predicted(feature, theta)
accu = accuracy(y, y_dash)
plt.plot(epocs, costs)
#plt.show()

y_dash = y_predicted(feature, theta)
accu = accuracy(y, y_dash)

N = data.shape[0]
#print(N)
f0 = np.ones((N,1))
f1 = data["feature1"].to_numpy()
#print(type(f1), type(f0))
f1 = f1.reshape(f1.shape[0],-1)
f2 = data["feature2"].to_numpy()
f2 = f2.reshape(f2.shape[0],-1)
#print(f1.shape, f2.shape, f0.shape)
fig = plt.figure(figsize=(10,8))

sns.scatterplot(x="feature1",  data= data, y="feature2", hue="class")
#print(data.describe())
x_var =  np.arange(20,110)

x_coord = []
for i in x_var:
    x = (i - np.mean(f1)) / (np.max(f1) - np.min(f1))
    x_coord.append(x)

y_var =  -(theta[0] + theta[2]*x_coord)*(1/theta[1])
#f1 = (f1 - np.mean(f1)) / (np.max(f1) - np.min(f1))
#f2 = (f2 - np.mean(f2)) / (np.max(f2) - np.min(f2))

y_coord =[]
for i in y_var:
    y = i*(np.max(f2) - np.min(f2)) + np.mean(f2) 
    y_coord.append(y)

#plt.plot(x_var, y_var)

plt.plot(x_var, y_coord)
plt.title("Decision boundary plot", fontsize=20)
plt.show()


