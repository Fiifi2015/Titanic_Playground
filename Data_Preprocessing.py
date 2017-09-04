from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
import pandas as pd
import pickle 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 


X_train_data = pickle.load(open( "Titanic_ML/Titanic_MLTitanic_train_data.pkl", "rb" ) )
#print(X_train_data.head())

#train = pd.read_csv('train.csv')

# insert code to get a list of categorical columns into a variable say categorical_columns
list_of_cat_features = ['Embarked', 'Cabin', 'Sex']
# insert code to take care of the missing values in the columns in whatever way you like to
print(X_train_data.isnull().sum())
X_train_data = X_train_data.fillna('missing')
print(X_train_data.isnull().sum())
# but is is important that missing values are replace

# pd.DataFrame(train_data.ix[:, train_data.columns != 'Survived'])



# Get the categorical values into a 2D numpy array
train_categorical_values = np.array(X_train_data[list_of_cat_features])
# OneHotEncoder will only work on integer categorical values, so if you have strings in your
# categorical columns, you need to use LabelEncoder to convert them first

# do the first column
enc_label = LabelEncoder()
enc_label.fit(train_categorical_values[:,0])
train_data = enc_label.transform(train_categorical_values[:,0])

# do the others
for i in range(1, train_categorical_values.shape[1]):
    enc_label = LabelEncoder()
    train_data = np.column_stack((train_data, enc_label.fit_transform(train_categorical_values[:,i])))

train_categorical_values = train_data.astype(float)

# if you have only integers then you can skip the above part from do the first column and uncomment the following line
# train_categorical_values = train_categorical_values.astype(float)

enc_onehot = OneHotEncoder()
train_cat_data = enc_onehot.fit_transform(train_categorical_values)

# play around and print enc.n_values_ features_indices_ to see how many unique values are there in each column

# create a list of columns to help create a DF from np array
# so say if you have col1 and col2 as the categorical columns with 2 and 3 unique values respectively. The following code
# will give you col1_0, col1_1, col2_1,col2_2,col2_3 as the columns

cols = [categorical_columns[i] + '_' + str(j) for i in range(0,len(categorical_columns)) for j in range(0,enc.n_values_[i]) ]
train_cat_data_df = pd.DataFrame(train_cat_data.toarray(),columns=cols)
# Fiifi is here 

model = KMeans(n_clusters=3)
results = model.fit(train_data)


fignum = 1
estimators = [('k_means_iris_8', KMeans(n_clusters=2),
              ('k_means_iris_3', KMeans(n_clusters=3)),
              ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1,
                                               init='random'))]
              
              
              
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
X = train_data
from mpl_toolkits.mplot3d import Axes3D
for name, est in estimators:    
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 0], X[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

# get this columns back into the data frame
train[cols] = train_cat_data_df[cols]

# append the target column. Obviously rename it to whatever is your target column
cols.append('target')
# So now you have a dataframe with only the categorical columns and the target. You can now do whatever you want to do with it :)
train_cat_df = train[cols]

 
# clustering dataset
# determine k using elbow method
 
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
 
 
x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])
 
plt.plot()
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()
 
# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']
 


# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(train_data) # before it was just X
    kmeanModel.fit(train_data)
    distortions.append(sum(np.min(cdist(train_data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / train_data.shape[0])
 
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()











