import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import random

# Clustering on Train dataset
df = pd.read_csv('Train.csv')
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

for i, row in df.iterrows():
    sp_val = 0
    if row['Spending_Score'] == 'Low':
        sp_val = random.randrange(0, 34)
    elif row['Spending_Score'] == 'Average':
        sp_val = random.randrange(34, 67)
    elif row['Spending_Score'] == 'High':
        sp_val = random.randrange(67, 100)
    df.at[i,'Spending_Score'] = sp_val

print(df['Spending_Score'])
data = df[['Age','Spending_Score']]
scaler = MinMaxScaler()
data_scale = scaler.fit_transform(data)
k = 4
model = KMeans(n_clusters = k, random_state = 10)
model.fit(data_scale)
df['cluster'] = model.fit_predict(data_scale)
plt.figure(figsize = (8,8))
for i in range(k):
    plt.scatter(df.loc[df['cluster'] == i, 'Age'], df.loc[df['cluster']==i, 'Spending_Score'],label = 'cluster' + str(i))
plt.legend()
plt.title('K = %d results'%k, size = 15)
plt.xlabel('Age', size = 12)
plt.ylabel('Spending Score', size = 12)
plt.show()

# Clustering on Test dataset
df = pd.read_csv('Test.csv')
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

df['Var_1'] = df['Var_1'].replace(['Cat_1'],1)
df['Var_1'] = df['Var_1'].replace(['Cat_2'],2)
df['Var_1'] = df['Var_1'].replace(['Cat_3'],3)
df['Var_1'] = df['Var_1'].replace(['Cat_4'],4)
df['Var_1'] = df['Var_1'].replace(['Cat_5'],5)
df['Var_1'] = df['Var_1'].replace(['Cat_6'],6)
df['Var_1'] = df['Var_1'].replace(['Cat_7'],7)

for i, row in df.iterrows():
    sp_val = 0
    if row['Spending_Score'] == 'Low':
        sp_val = random.randrange(0, 34)
    elif row['Spending_Score'] == 'Average':
        sp_val = random.randrange(34, 67)
    elif row['Spending_Score'] == 'High':
        sp_val = random.randrange(67, 100)
    df.at[i,'Spending_Score'] = sp_val

print(df['Spending_Score'])
data = df[['Age','Spending_Score']]
scaler = MinMaxScaler()
data_scale = scaler.fit_transform(data)
k = 4
model = KMeans(n_clusters = k, random_state = 10)
model.fit(data_scale)
df['cluster'] = model.fit_predict(data_scale)

plt.figure(figsize = (8,8))
for i in range(k):
    plt.scatter(df.loc[df['cluster'] == i, 'Age'], df.loc[df['cluster']==i, 'Spending_Score'],label = 'cluster' + str(i))

plt.legend()
plt.title('K = %d results'%k, size = 15)
plt.xlabel('Age', size = 12)
plt.ylabel('Spending Score', size = 12)
plt.show()