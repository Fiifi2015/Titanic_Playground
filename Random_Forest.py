import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.cluster import KMeans
import pickle


folder_path = '/Users/fiifiarthur/Documents/Data_Science/Python/Titanic_ML'

train_data = pd.read_csv(folder_path + '/Titanic_train.csv')
test_data = pd.read_csv(folder_path + '/Titanic_test.csv')



# if __name__ == '__main__':
#     print(test_data.head())
# else:
#     pass

# Split data into X & Y
list_of_features = train_data.columns.values.tolist()
print(list_of_features)

# Grab all the train data
X_train_data = pd.DataFrame(train_data.ix[:, train_data.columns != 'Survived'])
# Only selected the Survived column to use for your Y
Y_train_data = pd.DataFrame(train_data['Survived'])


#X_train_data.to_pickle(folder_path + 'Titanic_train_data.pkl')
#print('Completed picklising the data')
t#ennis = input('test for someting.....')


X_train, X_test, y_train, y_test = train_test_split(X_train_data, Y_train_data, test_size=0.40, random_state=0)
# Grab the testing data
#X_test_data = pd.DataFrame(test_data.ix[:, test_data.columns != 'Survived'])
#Y_test_data = pd.DataFrame(test_data['Survived'])


#['train_dataPassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']


#OneHotEncoder(n_values=’auto’, categorical_features=’all’, dtype=<class ‘numpy.float64’>, sparse=True, handle_unknown=’error’)[source]¶

def do_OneHotEnconding():
    X_train_data = X_train_data.fillna('missing')
    X_cat = X_train_data[['Embarked', 'Cabin', 'Sex']]
    #X_cat['Embarked'] = X_cat['Embarked'].replace(np.nan,'other')
    #X_cat = X_cat.as_matrix()
    list_of_features = ['Embarked', 'Cabin', 'Sex']
    #tennis = X_cat['Embarked'].unique()
    #print(tennis)
    onehotencoder = OneHotEncoder()
    labelencoder = LabelEncoder()
    
    
    for i in list_of_features:
        X= labelencoder.fit_transform(X_cat[i])
        X = onehotencoder.fit_transform(X).toarray()
        print(X[:20])

    #print(X_cat[:3,:3])
    #le = LabelEncoder()
    #X_new = le.fit_transform(X_cat)
    #print(X_new.shape)
    #labelencoder = LabelEncoder()
    #print()
    #X_cat[:, 3] = labelencoder.fit_transform(X_cat[:, 3])
    #print(X_cat[:20,:2])

    #onehotencoder = OneHotEncoder(categorical_features = [3])
    #X = onehotencoder.fit_transform(X_cat).toarray()
    #print(X.head(20))


    #X_cat = X_cat.as_matrix()
    #enc = OneHotEncoder()
    #new_X = enc.fit(X_cat)
    #new_X = pd.DataFrame(enc.transform(new_X))
    #print(new_x.head())

do_OneHotEnconding()




def plot_graph():
    estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
                     ('k_means_iris_3', KMeans(n_clusters=3)),
                     ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1,init='random'))]
    fignum = 1
    titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
    for name, est in estimators:
        fig = plt.figure(fignum, figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        est.fit(X_train)
        labels = est.labels_
        ax.scatter(X[:, 3], X[:, 0], X[:, 2],c=labels.astype(np.float), edgecolor='k')
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')
        ax.set_title(titles[fignum - 1])
        ax.dist = 12
        fignum = fignum + 1

#plot_graph()

def run_Kmeans(X_data):
    Kmeans(n_clusters=3).fit(X_data)

def train_Random_Forest_Model(X_data, Y_data):
    model = RandomForestClassifier()
    model.fit(X_data,Y_data)
    return model

def test_Random_Forest_Model(X_data, Y_data, model):
    tennis = model.score(X_data, Y_data)
    return tennis


#model = train_Random_Forest_Model(X_train, y_train)
#results = test_Random_Forest_Model(X_test, y_test, model)
#print(results)
