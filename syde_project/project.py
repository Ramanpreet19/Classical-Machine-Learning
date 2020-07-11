import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from sklearn import  preprocessing
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from copy import  deepcopy
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.linear_model import LogisticRegression

df_X_test = pd.read_csv('./Dataset/test/X_test.txt', header=None, delim_whitespace=True)
df_X_test = df_X_test.values

df_y_test = pd.read_csv('./Dataset/test/y_test.txt', header=None, delim_whitespace=True)
df_y_test = df_y_test.values

df_X_train = pd.read_csv('./Dataset/train/X_train.txt', header=None, delim_whitespace=True)
df_X_train = df_X_train.values

df_y_train = pd.read_csv('./Dataset/train/y_train.txt', header=None, delim_whitespace=True)
df_y_train = df_y_train.values

# Standardizing the features
#df_X = StandardScaler().fit_transform(df_X)
# PCA for PCA you need to standardize your data 
pca  = PCA(n_components=5).fit(df_X_train)
principle_X_train = pca.transform(df_X_train)
principle_X_test  = pca.transform(df_X_test)

print("Shape of feature data after PCA")
print(principle_X_train.shape)
print(principle_X_test.shape)
#Percentage of variance explained by each of the selected components.
#print(pca.explained_variance_ratio_) 

#Total information contained in all the components
print("Explained Vaiance Raito", sum(pca.explained_variance_ratio_))
principle_X = np.vstack((principle_X_test, principle_X_train))
df_y        = np.vstack((df_y_test, df_y_train))

#Accuracy Prediction Using SVM 
C = [1]
for i in C : 
    clf = svm.SVC(kernel='linear', C=i)
    clf.fit(principle_X_train, df_y_train)
    df_Y_pred = clf.predict(principle_X_train)
    train_acc = accuracy_score(df_y_train, df_Y_pred)
    df_Y_pred = clf.predict(principle_X_test)
    test_acc = accuracy_score(df_y_test, df_Y_pred)

    print("training accuracy with svm", train_acc)
    print("testing accuracy with svm", test_acc)
    

scores = cross_val_score(clf, principle_X, df_y, cv=10)
#print("Cross Vlaidation score for linear SVM is", scores)
mean_accu = np.mean(scores)
print("Mean of cross validation score for SVM", mean_accu)
print("")

#Model training for KNN
clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(principle_X_train, df_y_train)
df_Y_pred = clf.predict(principle_X_train)
train_acc = accuracy_score(df_y_train, df_Y_pred)
df_Y_pred = clf.predict(principle_X_test)
test_acc = accuracy_score(df_y_test, df_Y_pred)

#print("Value of C ", i)
print("training accuracy with KNN", train_acc)
print("testing accuracy with KNN", test_acc)

scores = cross_val_score(clf, principle_X, df_y, cv=10)
#print("Cross Validation score for KNN is", scores)
mean_accu = np.mean(scores)
print("Mean of cross validation score for KNN is", mean_accu)
print("")

#Model training with logistic regression 
clf = LogisticRegression(random_state=0)
clf.fit(df_X_train, df_y_train)
df_Y_pred = clf.predict(df_X_train)
train_acc = accuracy_score(df_y_train, df_Y_pred)
df_Y_pred = clf.predict(df_X_test)
test_acc = accuracy_score(df_y_test, df_Y_pred)

print("training accuracy with LR", train_acc)
print("testing accuracy with LR", test_acc)


#clf = svm.SVC(kernel='rbf', C=1, degree=10)
scores = cross_val_score(clf, principle_X, df_y, cv=10)
#print("Cross Validation score for logistic reg is", scores)
mean_accu = np.mean(scores)
print("Mean of cross validation score is for LR", mean_accu)
print("")

'''
#Final Version
'''
import time
#Generating augmented features for the first layer 
N_FOLDS = 5

'''
Stacking function will add the column of probablitiy of each class in the data 
'''
def Stacking(model, n_fold, X, Y):
   folds=StratifiedKFold(n_splits=n_fold,shuffle=True, random_state=29)
   model_prob = np.zeros((len(X), 6))
   for train_idx, test_idx in folds.split(X,Y):
      x_train, x_test = X[train_idx,:],X[test_idx, :]
      y_train, y_test = Y[train_idx],Y[test_idx]

      model.fit(X=x_train,y=y_train)
      y_pred = model.predict_proba(x_test)
      #print (y_pred.shape)
      model_prob[test_idx, :] =  y_pred
   return model_prob

def CEL_train(X_train, Y_train, models_list, num_of_layers, N_FOLDS=5):
    '''
    returns:
    layers_fitted_models_list (2d list of fitted models) : for each layers stores 'n' fitted models
    '''
    X = X_train.copy()
    Y = Y_train.copy()
    layers_fitted_models_list = []
    for l in range(0, num_of_layers - 1):
        fitted_models_per_layer_list = []
        X_layer = X.copy()
        print('Layer: ', l)
        #print('X_layer.shape: ', X_layer.shape)
        for i in range(0, len(models_list)):
            clf = deepcopy(models_list[i])
            model_prob = Stacking(clf, N_FOLDS, X, Y)
            X_layer = np.hstack((X_layer, model_prob))
            fitted_models_per_layer_list.append(clf.fit(X, Y))
            #print('X..shape: ', X.shape)
        #Update the layer output that will be the input for next layer
        X = X_layer.copy()
        #print('Updated X.shape: ', X.shape)
        #print(fitted_models_per_layer_list[0].coef_.shape)
        layers_fitted_models_list.append(fitted_models_per_layer_list)
        #print(layers_fitted_models_list[0][0].coef_.shape)

    #stuff for last layer
    last_layer_fitted_models = []
    #X_layer = X.copy()
    #model_probs = []
    #print('!!!!', layers_fitted_models_list[0][0].coef_.shape)
    for i in range(0, len(models_list)):
        clf = deepcopy(models_list[i])       
        last_layer_fitted_models.append(clf.fit(X, Y))
    layers_fitted_models_list.append(last_layer_fitted_models)

    return layers_fitted_models_list

def CEL_predict(X_test, layers_fitted_models_list):
    
    for i in range(0, len(layers_fitted_models_list) - 1):
        #print('Layer: ', i)
        X_layer = X_test.copy()
        #print('X_layer.shape: ', X_layer.shape)
        fitted_models_per_layer = layers_fitted_models_list[i]
        for clf in fitted_models_per_layer:
            model_prob = clf.predict_proba(X_test)
            X_layer = np.hstack((X_layer, model_prob))
        X_test = X_layer.copy()
        #print('Updated X_test.shape: ', X_test.shape)
        
    #last layer output
    layer_prob = []
    X_layer = X_test.copy()
    final_prob = 0
    fitted_models_per_layer = layers_fitted_models_list[-1]
    for clf in fitted_models_per_layer:
        model_prob = clf.predict_proba(X_test)
        layer_prob.append(model_prob)
        #X_layer = np.hstack((X_layer, model_prob))
        final_prob += model_prob
    final_prob /= len(layer_prob)
    final_class_pred = np.argmax(final_prob, axis=1) + 1 
    return final_class_pred

clf_svm      = svm.SVC(kernel='linear', C=1, probability=True)
clf_knn      = KNeighborsClassifier(n_neighbors=7)
clf_rf       = RandomForestClassifier(max_depth=13, random_state=0, n_estimators=100)
clf_log_reg  = LogisticRegression(random_state=0)
clf_svm_rbf  = svm.SVC(kernel='rbf', C=1, degree=10, probability=True)
'''
models_list    = [clf_knn, clf_log_reg, clf_svm_rbf]
num_of_layers  = 2

layers_fitted_models_list = CEL_train(X_train=principle_X_train, Y_train=df_y_train, models_list=models_list, num_of_layers=num_of_layers, N_FOLDS=8)

y_pred = CEL_predict(principle_X_test, layers_fitted_models_list)
test_accu = accuracy_score(df_y_test, y_pred)

y_pred = CEL_predict(principle_X_train, layers_fitted_models_list)
train_accu = accuracy_score(df_y_train, y_pred)

print("Training accuracy of the model ", train_accu)
print("Testing accuracy of the model ", test_accu)
'''


def accuracy(Y_actual, Y_predicted):
    accu_arr = (Y_actual == Y_predicted)
    accu_arr = accu_arr.astype(int)
    accu = np.count_nonzero(accu_arr)/len(accu_arr)
    return accu

def crossValidate(df_X, df_Y):

    y_accuracy_list  = [] 
    max_accuracy = 0   
    accu = 0
    y_pred = 0
    y_actual = 0
 
    folds = StratifiedKFold(n_splits=10,shuffle=True, random_state=29)
    #rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=11)
    for train_index, test_index in folds.split(df_X, df_Y):
        y_predicted = []
        y_actual    = []
        y_accuracy  = []

        Y_train, Y_test = df_y[train_index, :], df_y[test_index, :]
        X_train, X_test   = df_X[train_index, :], df_X[test_index, :]

        clf_svm = svm.SVC(kernel='linear', C=1, probability=True)
        clf_knn = KNeighborsClassifier(n_neighbors=7)
        #clf_rf = RandomForestClassifier(max_depth=13, random_state=0, n_estimators=100)
        clf_log_reg  = LogisticRegression(random_state=0)

        #clf_svm_rbf  = svm.SVC(kernel='rbf', C=1, degree=10, probability=True)
        models_list    = [clf_log_reg, clf_knn, clf_svm]#, clf_svm_rbf]
        num_of_layers  = 2

        layers_fitted_models_list = CEL_train(X_train=X_train, Y_train=Y_train, models_list=models_list, num_of_layers=num_of_layers, N_FOLDS=5)

        y_pred = CEL_predict(X_test, layers_fitted_models_list)
        y_actual = Y_test
        accu = accuracy_score(Y_test, y_pred)

        #y_pred = CEL_predict(principle_X_train, layers_fitted_models_list)
        #train_accu = accuracy_score(df_y_train, y_pred)
        #weak_clf, clf_b_t, accu_list = ada_boost_m1_train(df, T=50)
        #Y_predicted = ada_boost_m1_predict(X_test, weak_clf, clf_b_t)
        #accu = accuracy(Y_test, y_pred)  
        y_accuracy_list.append(accu)
        print("Accuracy is ", accu)

    print(y_accuracy_list)
    Mean = np.mean(y_accuracy_list)
    Variance = np.var(y_accuracy_list)
    print("Mean: ", Mean)
    print("Variance", Variance)
    return y_pred, y_actual, accu

y_pred, y_actual, accu = crossValidate(principle_X, df_y)

#Plotting confusion matrix


codes = {1: 'WALKING',
        2: 'WALKING_UPSTAIRS',
        3: 'WALKING_DOWNSTAIRS',
        4: 'SITTING',
        5: 'STANDING',
        6: 'LAYING'
}

y_actual = np.squeeze(y_actual)
df_act = pd.Series(y_actual)
Y_actual = df_act.map(codes)

y_pred = np.squeeze(y_pred)
df_pred = pd.Series(y_pred)
Y_pred = df_pred.map(codes)
#print(Y_actual[:20])

def draw_confusion_matrix(Y_actual, Y_pred, class_names, title='', normalized=None):
  cm = confusion_matrix(Y_actual, Y_pred, labels=class_names, sample_weight=None, normalize=normalized)
  #print(cm)
  df_cm = pd.DataFrame(cm, class_names, class_names)
  print(title)
  plt.figure(figsize = (10,7))
  sns.heatmap(df_cm, cmap=sns.light_palette('green'), annot=True, annot_kws={"size": 16})
  plt.show()

title = 'Confusion Matrix'
class_names = np.array(['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING'])
#class_names = class_names.astype(str).tolist()
draw_confusion_matrix(Y_actual, Y_pred, class_names, title, normalized=None)
