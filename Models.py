from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics

df = pd.read_csv('Train.csv')
# Data preparation
df = df.fillna(0)
df['Gender'] = df['Gender'].replace(['Male'],0)
df['Gender'] = df['Gender'].replace(['Female'],1)
df['Ever_Married'] = df['Ever_Married'].replace(['No'],0)
df['Ever_Married'] = df['Ever_Married'].replace(['Yes'],1)
df['Graduated'] = df['Graduated'].replace(['No'],0)
df['Graduated'] = df['Graduated'].replace(['Yes'],1)
df['Graduated'] = df['Graduated'].replace(['No'],0)
df['Profession'] = df['Profession'].replace(['Engineer'],1)
df['Profession'] = df['Profession'].replace(['Healthcare'],2)
df['Profession'] = df['Profession'].replace(['Executive'],3)
df['Profession'] = df['Profession'].replace(['Marketing'],4)
df['Profession'] = df['Profession'].replace(['Doctor'],5)
df['Profession'] = df['Profession'].replace(['Artist'],6)
df['Profession'] = df['Profession'].replace(['Lawyer'],7)
df['Profession'] = df['Profession'].replace(['Entertainment'],8)
df['Profession'] = df['Profession'].replace(['Homemaker'],9)
df['Spending_Score'] = df['Spending_Score'].replace(['Low'],1)
df['Spending_Score'] = df['Spending_Score'].replace(['Average'],2)
df['Spending_Score'] = df['Spending_Score'].replace(['High'],3)
df['Var_1'] = df['Var_1'].replace(['Cat_1'],1)
df['Var_1'] = df['Var_1'].replace(['Cat_2'],2)
df['Var_1'] = df['Var_1'].replace(['Cat_3'],3)
df['Var_1'] = df['Var_1'].replace(['Cat_4'],4)
df['Var_1'] = df['Var_1'].replace(['Cat_5'],5)
df['Var_1'] = df['Var_1'].replace(['Cat_6'],6)
df['Var_1'] = df['Var_1'].replace(['Cat_7'],7)
df['Segmentation'] = df['Segmentation'].replace(['A'],1)
df['Segmentation'] = df['Segmentation'].replace(['B'],2)
df['Segmentation'] = df['Segmentation'].replace(['C'],3)
df['Segmentation'] = df['Segmentation'].replace(['D'],4)

dataset = df.values
X = dataset[:,0:10]
Y = dataset[:,10]
X = np.asarray(X).astype('float32')
Y = np.asarray(Y).astype('float32')
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size=0.2)

# fit final model
# Gaussian NB
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
Y_train = Y_train.astype('int')
model.fit(X_train, Y_train)
ynew = model.predict(X_test)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = Y_train, cv = 10)
print("Gaussian NB Accuracy(cross validation):",accuracies.mean())
print("Gaussian NB Accuracy Score:", model.score(X_test,Y_test))


# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
Y_train = Y_train.astype('int')
model.fit(X_train, Y_train)
ynew = model.predict(X_test)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = Y_train, cv = 10)
print("LDA Accuracy(cross validation):",accuracies.mean())
print("LDA Accuracy Score:", model.score(X_test,Y_test))


# Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
Y_train = Y_train.astype('int')
model.fit(X_train, Y_train)
ynew = model.predict(X_test)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = Y_train, cv = 10)
print("Logistic Regression Accuracy(cross validation):",accuracies.mean())
print("Logistic Regression Accuracy Score:", model.score(X_test,Y_test))