from model import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing

#pandas dataframes
train_set = pd.read_csv('dataset/train.csv')
test_set = pd.read_csv('dataset/test.csv')
df = train_set.append(test_set, ignore_index = True)
#df.info()

#print(df.head(10))

#print(df[['Pclass']].isnull().sum() * 100 / len(df))
#sns.countplot(x='Pclass', data=df, palette='hls', hue='Survived')
#plt.xticks(rotation=45)
#plt.show()

#print(df[['Age']].isnull().sum() * 100 / len(df))
#print(df[['Agegroup', 'Survived']].groupby(['Agegroup'], as_index=False).mean())

#sns.countplot(x='Agegroup', data=df, palette='hls', hue='Survived')
#plt.xticks(rotation=45)
#plt.show()

df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

normalized_titles = {
    "Capt":       0,
    "Col":        2,
    "Major":      2,
    "Jonkheer":   1,
    "Don":        1,
    "Sir" :       3,
    "Dr":         3,
    "Rev":        1,
    "Countess":   4,
    "Dona":       3,
    "Mme":        3,
    "Mlle":       3,
    "Ms":         3,
    "Mr" :        1,
    "Mrs" :       3,
    "Miss" :      3,
    "Master" :    3,
    "Lady" :      4
}

df.Title = df.Title.map(normalized_titles)
#print(df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
#print(df['Title'].value_counts())
df['Family'] = df['SibSp'] + df['Parch'] + 1
df['Ticket_Fq'] = df.groupby('Ticket')['Ticket'].transform('count')
#print(df[['Ticket_Fq', 'Survived']].groupby(['Ticket_Fq'], as_index=False).mean())

#print(df['Title'].value_counts())
#print(df['Fare'].value_counts())

#sns.countplot(x='Fare', data=df)
#plt.xticks(rotation=45)
#plt.show()


m = np.nanmedian(df.loc[ (df['Pclass'] == 3)& (df['Embarked'] == 'S'), 'Fare'].values)
df['Fare'].fillna(value = m , inplace=True)

df.Sex = df.Sex.map({"male": 0, "female":1})
df['Age'] = df.groupby(['Sex', 'Title', 'Pclass'])['Age'].apply(lambda k: k.fillna(k.median()))

pclass_dummies = pd.get_dummies(df.Pclass, prefix="Pclass")
embarked_dummies = pd.get_dummies(df.Embarked, prefix="Embarked")
df = pd.concat([df, pclass_dummies, embarked_dummies], axis=1)

df = df.drop(labels='Embarked', axis=1)
df = df.drop(labels='Cabin', axis=1)
df = df.drop(labels='Ticket', axis=1)
df = df.drop(labels='Name', axis=1)
df = df.drop(labels='Pclass', axis=1)

min_max_scaler = preprocessing.MinMaxScaler()

df['Age'] = min_max_scaler.fit_transform(np.array(df['Age']).reshape(-1,1)) 
df['Fare'] = min_max_scaler.fit_transform(np.array(df['Fare']).reshape(-1,1)) 
df['SibSp'] = min_max_scaler.fit_transform(np.array(df['SibSp']).reshape(-1,1)) 
df['Parch'] = min_max_scaler.fit_transform(np.array(df['Parch']).reshape(-1,1)) 
df['Family'] = min_max_scaler.fit_transform(np.array(df['Family']).reshape(-1,1)) 
df['Title'] = min_max_scaler.fit_transform(np.array(df['Title']).reshape(-1,1)) 
df['Ticket_Fq'] = min_max_scaler.fit_transform(np.array(df['Ticket_Fq']).reshape(-1,1)) 
#pd.set_option('display.max_columns', None)
#print(df.describe())




train = df[pd.notnull(df['Survived'])]
test = df[pd.isnull(df['Survived'])]



X = train.drop(['Survived', 'PassengerId'], axis = 1)
y = train['Survived']
X_test = test.drop(['Survived', 'PassengerId'], axis = 1)
y_test = test['Survived']

#---------------------------------------------------------

X_T = X.T
y_T = y.T.values.reshape(1,y.shape[0])
X_testT = X_test.T
y_testT = y_test.T.values.reshape(1,y_test.shape[0])

layers_sizes = [14, 29, 31, 37, 23, 19, 1] #prvi mora da odgovara broju kolona u X
parameters = L_layer_model(X_T, y_T, layers_sizes, "sigmoid", num_iterations = 2501)
pred_train = predict(X_T, y_T, parameters,"sigmoid")
generate_csv(X_testT, y_testT, parameters, test, "sigmoid")


