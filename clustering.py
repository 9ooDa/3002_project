import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Train.csv')
df = df.fillna(0)

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

print(df['Spending_Score'])

data = df[['Profession','Spending_Score']]

# scaler = MinMaxScaler()
# data_scale = scaler.fit_transform(data)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scale = scaler.fit_transform(data)

# def z_score_normalize(lst):
#     normalized = []

#     for value in lst:
#         normalized_num = (value - np.mean(lst)) / np.std(lst)
#         normalized.append(normalized_num)

#     return normalized
# data_scale = z_score_normalize(data)

k = 3

model = KMeans(n_clusters = k, random_state = 10)

model.fit(data_scale)

df['cluster'] = model.fit_predict(data_scale)

plt.figure(figsize = (8,8))

for i in range(k):
    plt.scatter(df.loc[df['cluster'] == i, 'Profession'], df.loc[df['cluster']==i, 'Spending_Score'],label = 'cluster' + str(i))

plt.legend()
plt.title('K = %d results'%k, size = 15)
plt.xlabel('Profession', size = 12)
plt.ylabel('Spending_Score', size = 12)
plt.show()

# from yellowbrick.cluster import KElbowVisualizer

# model = KMeans()
# visualizer = KElbowVisualizer(model, k=(1,10))
# visualizer.fit(data_scale)