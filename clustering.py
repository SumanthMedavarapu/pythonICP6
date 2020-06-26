import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


#from sklearn import linear_model
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error

train = pd.read_csv('CC.csv')
print(train["CASH_ADVANCE_FREQUENCY"])

# Handling missing value
#data = train.select_dtypes(include=[np.number]).interpolate().dropna()
data = train.fillna(train.mean())
#data =train.apply(lambda x: x.fillna(x.mean()),axis=0)
print(data["CASH_ADVANCE_FREQUENCY"])
df = data.apply(LabelEncoder().fit_transform)
#df.fillna(df.mean())
#print(data)
Sum_of_squared_distances = []
K = range(1,11)
for k in K:
    km = KMeans(n_clusters=k,max_iter=300)
    kmeans = km.fit(df)
    Sum_of_squared_distances.append(kmeans.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
nclusters = 3
km1 = KMeans(n_clusters=nclusters)
km1.fit(df)

y_cluster_kmeans =km1.predict(df)
#print(y_cluster_kmeans)
from sklearn import metrics
score = metrics.silhouette_score(df,y_cluster_kmeans)
print("score before feature scaling"+str(score))
# Processing
scaler = StandardScaler()
scaler.fit(df)
scaled_data=scaler.transform(df)
print(scaled_data)

#X_scaled_array = scaler.transform(df)
X_scaled = pd.DataFrame(scaled_data, columns=df.columns)
#print(X_scaled)

#after feature scaling
nclusters = 3
km2 = KMeans(n_clusters=nclusters)
km2.fit(X_scaled)

y_cluster_kmeans1 =km2.predict(X_scaled)
#print(y_cluster_kmeans)

score = metrics.silhouette_score(X_scaled,y_cluster_kmeans1)
print("score after scaling" +str(score))

# PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler_pca = StandardScaler()
# Fit on training set only.
scaler_pca.fit(df)
x_scaler = scaler_pca.transform(df)
pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)
df2 = pd.DataFrame(data=x_pca)
finaldf1 = pd.concat([df2, df[['TENURE']]], axis=1)
print(finaldf1)

#After scaling+PCA+KMEANS
nclusters = 3
km3 = KMeans(n_clusters=nclusters)
km3.fit(finaldf1)

y_cluster_kmeans2 =km3.predict(finaldf1)
#print(y_cluster_kmeans)

score = metrics.silhouette_score(X_scaled, y_cluster_kmeans2)
print("score after scaling,pca,kmeans" +str(score))
colors=["red","green","blue"]
print(y_cluster_kmeans2)
for i in range(3):
    x_axis = finaldf1[y_cluster_kmeans2 == i].iloc[:,0]
    y_axis = finaldf1[y_cluster_kmeans2 == i].iloc[:,1]
    plt.scatter(x_axis,y_axis,color=colors[i])

plt.show()

"""from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler_pca = StandardScaler()
# Fit on training set only.
scaler_pca.fit(df)
x_scaler = scaler_pca.transform(df)
pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)
df2 = pd.DataFrame(data=x_pca)
finaldf = pd.concat([df2, df[['TENURE']]], axis=1)
print(finaldf)"""
