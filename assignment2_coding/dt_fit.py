import numpy as np 
import math
import pandas as pd

def entropy_func(c, n):
    """
    The math formula
    """
    return -(c*1.0/n)*math.log(c*1.0/n, 2)

def entropy_cal(c1, c2):
    """
    Returns entropy of a group of data
    c1: count of one class
    c2: count of another class
    """
    if c1== 0 or c2 == 0:  # when there is only one class in the group, entropy is 0
        return 0
    return entropy_func(c1, c1+c2) + entropy_func(c2, c1+c2)

# get the entropy of one big circle showing above
def entropy_of_one_division(division): 
    """
    Returns entropy of a divided group of data
    Data may have multiple classes
    """
    s = 0
    n = len(division)
    classes = set(division)
    for c in classes:   # for each class, get entropy
        n_c = sum(division==c)
        e = n_c*1.0/n * entropy_cal(sum(division==c), sum(division!=c)) # weighted avg
        s += e
    return s, n

# The whole entropy of two big circles combined
def get_entropy(y_predict, y_real):
    """
    Returns entropy of a split
    y_predict is the split decision, True/Fasle, and y_true can be multi class
    """
    if len(y_predict) != len(y_real):
        print('They have to be the same length')
        return None
    n = len(y_real)
    s_true, n_true = entropy_of_one_division(y_real[y_predict]) # left hand side entropy
    s_false, n_false = entropy_of_one_division(y_real[~y_predict]) # right hand side entropy
    s = n_true*1.0/n * s_true + n_false*1.0/n * s_false # overall entropy, again weighted average
    return s


class DecisionTreeClassifier(object):
    def __init__(self, max_depth):
        self.depth = 0
        self.max_depth = max_depth


def find_best_split(self, col, y):
    """
    col: the column we split on
    y: target var
    """
    min_entropy = 10    
    n = len(y)
    for value in set(col):  # iterating through each value in the column
        y_predict = col < value  # separate y into 2 groups
        my_entropy = get_entropy(y_predict, y)  # get entropy of this split
        if my_entropy <= min_entropy:  # check if it's the best one so far
            min_entropy = my_entropy
            cutoff = value
    return min_entropy, cutoff

def find_best_split_of_all(self, x, y):
    """
    Find the best split from all features
    returns: the column to split on, the cutoff value, and the actual entropy
    """
    col = None
    min_entropy = 1
    cutoff = None
    for i, c in enumerate(x.T):  # iterating through each feature
        entropy, cur_cutoff = self.find_best_split(c, y)  # find the best split of that feature
        if entropy == 0:    # find the first perfect cutoff. Stop Iterating
            return i, cur_cutoff, entropy
        elif entropy <= min_entropy:  # check if it's best so far
            min_entropy = entropy
            col = i
            cutoff = cur_cutoff
    return col, cutoff, min_entropy

def fit(self, x, y, par_node={}, depth=0):
    """
    x: Feature set
    y: target variable
    par_node: will be the tree generated for this x and y. 
    depth: the depth of the current layer
    """
    if par_node is None:   # base case 1: tree stops at previous level
        return None
    elif len(y) == 0:   # base case 2: no data in this group
        return None
    elif self.all_same(y):   # base case 3: all y is the same in this group
        return {'val':y[0]}
    elif depth >= self.max_depth:   # base case 4: max depth reached 
        return None
    else:   # Recursively generate trees! 
        # find one split given an information gain 
        col, cutoff, entropy = self.find_best_split_of_all(x, y)   
        y_left = y[x[:, col] < cutoff]  # left hand side data
        y_right = y[x[:, col] >= cutoff]  # right hand side data
        par_node = {'col': iris.feature_names[col], 'index_col':col,
                    'cutoff':cutoff,
                   'val': np.round(np.mean(y))}  # save the information 
        # generate tree for the left hand side data
        par_node['left'] = self.fit(x[x[:, col] < cutoff], y_left, {}, depth+1)   
        # right hand side trees
        par_node['right'] = self.fit(x[x[:, col] >= cutoff], y_right, {}, depth+1)  
        self.depth += 1   # increase the depth since we call fit once
        self.trees = par_node  
        return par_node
    
def all_same(self, items):
    return all(x == items[0] for x in items)

class DecisionTreeClassifier(object):
    def __init__(self, max_depth):
        self.depth = 0
        self.max_depth = max_depth
    
    def fit(self, x, y, par_node={}, depth=0):
        if par_node is None: 
            return None
        elif len(y) == 0:
            return None
        elif self.all_same(y):
            return {'val':y[0]}
        elif depth >= self.max_depth:
            return None
        else: 
            col, cutoff, entropy = self.find_best_split_of_all(x, y)    # find one split given an information gain 
            y_left = y[x[:, col] < cutoff]
            y_right = y[x[:, col] >= cutoff]
            par_node = {'col': iris.feature_names[col], 'index_col':col,
                        'cutoff':cutoff,
                       'val': np.round(np.mean(y))}
            par_node['left'] = self.fit(x[x[:, col] < cutoff], y_left, {}, depth+1)
            par_node['right'] = self.fit(x[x[:, col] >= cutoff], y_right, {}, depth+1)
            self.depth += 1 
            self.trees = par_node
            return par_node
    
    def find_best_split_of_all(self, x, y):
        col = None
        min_entropy = 1
        cutoff = None
        for i, c in enumerate(x.T):
            entropy, cur_cutoff = self.find_best_split(c, y)
            if entropy == 0:    # find the first perfect cutoff. Stop Iterating
                return i, cur_cutoff, entropy
            elif entropy <= min_entropy:
                min_entropy = entropy
                col = i
                cutoff = cur_cutoff
        return col, cutoff, min_entropy
    
    def find_best_split(self, col, y):
        min_entropy = 10
        n = len(y)
        for value in set(col):
            y_predict = col < value
            my_entropy = get_entropy(y_predict, y)
            if my_entropy <= min_entropy:
                min_entropy = my_entropy
                cutoff = value
        return min_entropy, cutoff
    
    def all_same(self, items):
        return all(x == items[0] for x in items)
                                           
    def predict(self, x):
        tree = self.trees
        results = np.array([0]*len(x))
        for i, c in enumerate(x):
            results[i] = self._get_prediction(c)
        return results
    
    def _get_prediction(self, row):
        cur_layer = self.trees
        while cur_layer.get('cutoff'):
            if row[cur_layer['index_col']] < cur_layer['cutoff']:
                cur_layer = cur_layer['left']
            else:
                cur_layer = cur_layer['right']
        else:
            return cur_layer.get('val')


from sklearn.datasets import load_iris
from pprint import pprint

#iris = load_iris()
#x = iris.data
#y = iris.target

from sklearn.datasets import load_wine

iris = load_wine()
x = iris.data
y = iris.target

#data = pd.read_csv("./playtennis.csv")
#print(data.head())f
#x = data.iloc[:,1:5]
#print(attribute_values.head())
#y  = data.iloc[:,5]
#print(class_labels.head())

'''
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header = None)
#print(data.head())
data.columns = ['Label', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of ash ', 'Magnesium', 'Total phenols', 'Flavanoids' , 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
#print(data.head())

y  = data.iloc[:,0]
#print (y.head())
x = data.iloc[:,1:14]
#print (x.head())
'''

clf = DecisionTreeClassifier(max_depth=20)
m = clf.fit(x, y)

pprint(m)
