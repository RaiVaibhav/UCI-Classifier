# Run ``pip3 install -r requirements.txt`` to install all the packages

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn import metrics

'''
    
Apply IRIS dataset on SVM and Logistic Regression
classifier and calculate their accuracy

'''

# importing the Iris dataset with pandas
dataset = pd.read_excel('Iris.xls')
x = dataset.iloc[:, [0, 1, 2, 3]].values

my_data_set = dataset[['sepal length', 'sepal width','petal length', 'petal width','iris']]




train, test = train_test_split(my_data_set, test_size = 0.3)
train_X = train[['sepal length', 'sepal width','petal length', 'petal width']]
# taking the training data features
train_y=train.iris
# output of our training data
test_X= test[['sepal length', 'sepal width','petal length', 'petal width']] 
# taking test data features
test_y =test.iris   
#output value of test data

'''
    
Apply IRIS dataset on Logistic Regression classifier
and calculate the accuracy.

'''

model = LogisticRegression()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_y))


'''
    
Apply IRIS dataset on Logistic SVM classifier
and calculate the accuracy.

'''
model = svm.SVC() 
model.fit(train_X,train_y) 
prediction=model.predict(test_X)
print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y))



'''

Perform clustering using K-Means Algorithm

'''
# Finding the optimum number of clusters for k-means classification
# using "elbow method".
from sklearn.cluster import KMeans
Cost = []

# First running on random 10 cluster.
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    Cost.append(kmeans.inertia_)


# Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 10), Cost)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Cost')  # within cluster sum of squares
plt.show()

'''

After the 10 iteration we can observe form the plot that the we get our
"elbow joint" means we the value of "K" for clustering.

'''

# Applying kmeans to the dataset with number of cluster equal to 3 / Creating
# the kmeans classifier
kmeans = KMeans(n_clusters=3, init='k-means++',
                max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)


# Visualising the clusters on the basis of "Iris"/ Class
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1],
            s=100, c='red', label='Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1],
            s=100, c='blue', label='Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s=100, c='green', label='Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=100, c='yellow', label='Centroids')
plt.show()


'''

Visualising the clusters on the basis of features i.e.
Sepal Length, Sepal Width, Petal Length, petal Width.

'''
sns.pairplot(data=dataset, vars=('sepal length', 'sepal width',
                                 'petal length', 'petal width'), hue='iris')
plt.show()
