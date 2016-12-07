#
# Tutorial from Google about Machine Learning.
#
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# Importing the library iris from sklearn
iris = load_iris()


#
#    VISUALIZATION OF THE DATA
#
# The dataset contains the values of the columns in 
# wikipedia and the metadata tells the name of the features,
# and the names of the different types of flowers.
print iris.feature_names
print iris.target_names

# We are printing the first value in the dataset
print iris.data[0]

# prints the lable of the type of flower for the first element
print iris.target[0]

#We are iterating over the data in the dataset
for i in range(len(iris.target)):
	print "Example %d: label %s, feature %s" % (i, iris.target[i], iris.data[i])



#
#	TRAINING THE DATA
#
# For this test we are going to remove the first element
# of each type of flower, so we can test after training the classifier
test_idx = [0,50,100]

train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing the data
# This testing data will only have the samples we removed.
test_target = iris.target[test_idx];
test_data = iris.data[test_idx];

clf = tree.DecisionTreeClassifier()
# We are training the classifier to recognize the different
# types of flowers. 
clf.fit(train_data, train_target)

# We will obtain the three types of flowers.
print test_target
# We let the classifier once it has been trained to 
# recognize the elements we removed from the dataset.
print clf.predict(test_data)


# We are going to visualize the tree viz code.
import pydotplus 
dot_data = tree.export_graphviz(clf, out_file="iris.pdf") 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("iris.pdf") 

