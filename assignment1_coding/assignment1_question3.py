from mlxtend.data import loadlocal_mnist
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np

X, y = loadlocal_mnist(images_path='./train-images-idx3-ubyte', labels_path='./train-labels-idx1-ubyte')
#print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
#print('\n1st row', X[0])
#print('\n1st row of label', y[0])
#X = preprocessing.normalize(X)

######################################Part A#######################################################################
def PCA (input_data, d):
    #Compute Mean of every Dimension 
    #print('Dimensions: %s x %s' % (input_data.shape[0], input_data.shape[1]))
    x_mean = np.mean(input_data.T, axis=1)
    #print(x_mean.shape[0])
    x_bar = X - x_mean
    #print('Dimensions: %s x %s' % (x_bar.shape[0], x_bar.shape[1]))
    x_bar = preprocessing.normalize(x_bar)
    #Compute Covariance 
    x_cov = np.cov(x_bar.T)
    #print(x_cov.shape[0], x_cov.shape[1])

    #Compute Eigen vector and Eigen values  
    eigvals, eigvecs = np.linalg.eig(x_cov)
    
    # Make a list of (eigenvalue, eigenvector) tuples, Sort eigen values 
    eig_pairs = [(np.abs(eigvals[i]), eigvecs[:,i]) for i in range(len(eigvals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    eig_matrix = eig_pairs[0][1].reshape(784,1)
    for i in range(1,d):
        nth_eig_vec = eig_pairs[i][1].reshape(784,1)
        eig_matrix = np.concatenate((eig_matrix, nth_eig_vec), axis=1)

    #print(eig_matrix.shape[0], eig_matrix.shape[1])

    return x_bar, eig_matrix, eig_pairs

x_bar, x_eig_matrix, eig_pairs = PCA(X, 28)

transform_data = np.dot(x_eig_matrix.T, X.T)
#print(transform_data.shape[0], transform_data.shape[1])
reconst_data= np.dot(x_eig_matrix, transform_data)
#print(reconst_data.shape[0], reconst_data.shape[1])

######################################Part B#######################################################################
# proportion of variance is calculated as ratio of sum of selected eigen vectors to tolat sum of eigen vectors 
sorted_eigen_list1 = []
sorted_eigen_list1 = list(zip(*eig_pairs))[0]
sorted_eigen_list2 = list(zip(*eig_pairs))[0]
#print(type(sorted_eigen_list1))
sorted_eigen_list1 = list(sorted_eigen_list1)
#print(eig_pairs[0][1])
matrix_w = np.hstack((eig_pairs[0][1].reshape(784,1), eig_pairs[1][1].reshape(784,1)))
#print(matrix_w)
sum_eigen_val = 0

for i in range (0, len(sorted_eigen_list1)):
    sum_eigen_val  = sum_eigen_val + sorted_eigen_list1[i]
    
#print("Sum of all Eigen values ", sum_eigen_val)
threshold_val = (sum_eigen_val)*(.95)
#print("Threhold value", threshold_val)

sum = 0 
for i in range (0, len(sorted_eigen_list2)):
    sum = sum + sorted_eigen_list2[i]
    #print("value of sum", sum)
    if (sum >= threshold_val):
        print ("value of index for PoV is", i)
        break; 


######################################Part C#######################################################################

from sklearn.metrics import mean_squared_error

#d = [1, 2, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 500, 600, 700, 784]
d = [1, 2, 5, 8, 11, 14, 17, 20, 23, 27, 30, 33, 37, 45, 60, 80, 100, 150, 300, 500, 784]

mse  = []
for i in d:
    x_bar, x_eig_matrix, eig_pairs = PCA(X, i)
    transform_data = np.dot(x_eig_matrix.T, X.T)
    #print(transform_data.shape[0], transform_data.shape[1])
    reconst_data= np.dot(x_eig_matrix, transform_data)
    #print(reconst_data.shape[0], reconst_data.shape[1]) 

    mse.append(mean_squared_error(reconst_data,X.T))

#print(mse)

fig = plt.figure(figsize=(10,8))
d = [1, 2, 5, 8, 11, 14, 17, 20, 23, 27, 30, 33, 37, 45, 60, 80, 100, 150, 300, 500, 784]
ave_mse = []
for i in mse:
    ave_mse.append(i/60000)
#print(ave_mse)
#print(mse)
plt.plot(d, ave_mse)
plt.title("Average Mean Squared Error vs Dimensions", Fontsize=20)
plt.xlabel('Dimensions (d)', fontsize=16)
plt.ylabel('Average MSE', fontsize=16)
#plt.show()

#fig.savefig('q3_mse.jpg')


######################################Part D#######################################################################

d = [1, 10, 50, 250, 784]

reconst_data_8  = []
for i in d:
    x_bar, x_eig_matrix, eig_pairs = PCA(X, i)
    transform_data = np.dot(x_eig_matrix.T, X.T)
    #print(transform_data.shape[0], transform_data.shape[1])
    reconst_data= np.dot(x_eig_matrix, transform_data)
    reconst_data_8.append(reconst_data[:,6508])

fig = plt.figure(figsize=(12,10))
plt.subplot(2, 3, 1)
plt.imshow(reconst_data_8[0].reshape(28,28), cmap='gray')
plt.subplot(2, 3, 2)
plt.imshow(reconst_data_8[1].reshape(28,28), cmap='gray')
plt.subplot(2, 3, 3)
plt.imshow(reconst_data_8[2].reshape(28,28), cmap='gray')

plt.subplot(2, 3, 4)
plt.imshow(reconst_data_8[3].reshape(28,28), cmap='gray')

plt.subplot(2, 3, 5)
plt.imshow(reconst_data_8[4].reshape(28,28), cmap='gray')
#plt.show()

######################################Part E#######################################################################
sorted_eigen_list = list(zip(*eig_pairs))[0]
fig = plt.figure(figsize=(10,8))
plt.plot(np.arange(0,784), sorted_eigen_list)
plt.title("Eigen values vs Dimensions", Fontsize=20)
plt.xlabel('Dimensions (d)', fontsize=16)
plt.ylabel('Eigen Values', fontsize=16)
plt.show()

#fig.savefig('q3_eigen_vector.jpg')