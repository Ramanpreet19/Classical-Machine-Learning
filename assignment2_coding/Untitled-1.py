import numpy as np
import math 
import operator

def CalEntropy(data, column_name):
    '''
    '''
    total_data_samples = len(data)
    #print("Total data samples", total_data_samples)

    col_feature_values = data[column_name]
    (vals, vals_count) = np.unique(col_feature_values, return_counts=True)
    #print("Values in the colum and there count", vals, vals_count)

    label_values = data.iloc[:,-1]
    class_labels  = np.unique(label_values)
    #print("Class Labels", class_labels)

    data_temp =  pd.concat([col_feature_values, label_values], axis=1)
    #print(data_temp.head())

    #class count for each feature variable
    #dictionary structure {'attribute value': [negetive sample, positive_sample]}
    label_count_dict = {}

    for x in vals:
        d2 = data_temp.loc[data_temp[column_name] == x]
        d2 = d2.iloc[:,-1]
        (class_vals, class_vals_count) = np.unique(d2, return_counts=True)
        #print(class_vals, class_vals_count)  
        label_count_dict[x] = class_vals_count
    #print(label_count_dict)

    #entropy list stores the enropy for each feature variable 
    entropy_list = []
    #calculate entropy as -plog(p)
    for key in label_count_dict:
        neg_count = label_count_dict[key][0]
        pos_count = label_count_dict[key][1]
        if ((pos_count != 0) or (neg_count != 0)):
            p1 = pos_count/(pos_count + neg_count)
            p2 = neg_count/(pos_count + neg_count)
            entropy = -((p1*math.log2(p1))+ (p2*math.log2(p2)))
            #print("Value of entropy is", entropy)
            entropy_list.append(entropy)
        else:
            entropy_list.append(0)
    #print("Entropy list", entropy_list)

    #Calculating the total entropy for the column 
    total_entropy = 0
    for x  in range(0, len(entropy_list)):
        total_entropy = total_entropy + (entropy_list[x]*(vals_count[x]/total_data_samples))
    #print(total_entropy)
    return total_entropy

def CalInformationGain(data, attributes):
    '''
    '''
    total_data_samples = len(data)
    #print("Total data samples", total_data_samples)  
    label_values = data.iloc[:,-1]
    
    class_labels, label_counts  = np.unique(label_values, return_counts=True)
    #print("Class Labels", class_labels)
    #print("Label Count", label_counts)

    raw_entropy = 0
    for x  in range(0, len(label_counts)):
        p = label_counts[x]/total_data_samples
        raw_entropy = raw_entropy -(p*math.log2(p))
    
    #information_gain_dict dictionary stores label of colum and IG corresponding to that 
    information_gain_dict = {}
    #attributes = data.iloc[:,0:9].columns.values
    for x in attributes:
        information_gain_dict[x] = raw_entropy - CalEntropy(data, x)

    print("Type of IG", type(information_gain_dict))
    return information_gain_dict

def select_best_feature(information_gain_dict):
    sorted_information_gain_dict = sorted(information_gain_dict.items(), key=operator.itemgetter(1), reverse=True)
    #print(type(sorted_information_gain_dict))
    max_ig_feature = next(iter(sorted_information_gain_dict))
    ret = max_ig_feature[0]
    print(ret)
    return ret 

class Node:
	def __init__(self, value=None, edge=None):
		self.value = value
		self.edge = edge #edge to this node (coming to this node)
		self.children = [] #list of Nodes

#id3 tree will return the root node 
def id3(data, attributes):
    root = Node()

    #check if all examples are positive or negative 
    class_labels = data.iloc[:,-1]
    labels, label_count = np.unique(class_labels, return_counts=True)
    #print(labels, label_count)

    for x in range(0,len(labels)):
        if label_count[x] == 0:
            root.value = labels[(x+1)%2]
            return root

    if attributes.size == 0:
        if(label_count[0]>=label_count[1]):
            root.value = labels[0]
        else:
            root.value = labels[1]
        return root
    else:
        information_gain_dict = CalInformationGain(data, attributes)
        A = select_best_feature(information_gain_dict)
        root.value = A
        root.children = []
        values_of_A = data[A]
        for e in values_of_A.unique():
            child_node = Node()
            child_node.edge = e
            data = data.loc[data[A] == e]
            if len(data) == 0:
                child_node.value = data['Class'].value_counts()[0]
            else:
                new_attributes = attributes[attributes != A]
                #TODO:
                child_node = id3()
            root.children.append(child_node)

    return root

#information_gain_dict = CalInformationGain(data)
#select_best_feature(information_gain_dict)
attributes = data.iloc[:,0:9].columns.values
#print(attributes)
#X = data.drop('Class', axis=1)
#y = data['Class']
id3(data, attributes)