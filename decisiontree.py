# Decision_Tree_Classifier
from sklearn.tree import DecisionTreeClassifier
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
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
# X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

# fit final model
model = DecisionTreeClassifier(criterion="entropy", max_depth=100000)
Y_train = Y_train.astype('int')
model.fit(X_train, Y_train)
y_sum = 0
for ind in range(len(Y_test)):
    y_sum += Y_test[ind]
y_mean = y_sum / len(Y_test)
ssr = 0
sst = 0
ynew = model.predict(X_test)

for i in range(len(X_test)):
    print("X= {}, True_Y= {} ,Predicted= {}".format(X_test[i], Y_test[i] ,ynew[i]))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = Y_train, cv = 3)
print("Accuracy:",accuracies.mean())
print("Std",accuracies.std())

print("Accuracy Score:", model.score(X_test,Y_test))

from matplotlib import pyplot as plt
from sklearn import tree

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(model,filled=True)
fig.savefig("decistion_tree.png")