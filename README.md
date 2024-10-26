# EX 9 Implementation of K Means Clustering for Customer Segmentation
## DATE:
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation.
2. Choosing the Number of Clusters (K).
3. K-Means Algorithm Implementation.
4. Evaluate Clustering Results.
5. Deploy and Monitor.
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by:K.TAMIL MARUDHU 
RegisterNumber:2305001033  
*/
```
```
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from  sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data
x=data[['Annual Income (k$)', 'Spending Score (1-100)']]
plt.figure(figsize=(4, 4))
plt.scatter(data['Annual Income (k$)'],x['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
K=3
Kmeans=KMeans(n_clusters=K)
Kmeans.fit(x)
centroids=Kmeans.cluster_centers_
labels=Kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors=['r','g','b']
for i in range(k):
  cluster_points=x[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i], label=f'Cluster{i+1}')
  distances=euclidean_distances(cluster_points, [centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i], radius, color=colors[i], fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, color='k', label='Centroids')
plt.title('K-means Clustring')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

## Output:
![WhatsApp Image 2024-10-26 at 10 43 40_8f4e30cf](https://github.com/user-attachments/assets/1fe5387e-81e7-4fcf-956f-e5635702099a)
![WhatsApp Image 2024-10-26 at 10 44 09_669dee8d](https://github.com/user-attachments/assets/c884d378-98b3-4c44-b086-4d8f491febef)
![WhatsApp Image 2024-10-26 at 10 44 42_c337373a](https://github.com/user-attachments/assets/cd1c077b-09b0-4ae2-a168-9bb83cd17b44)
![WhatsApp Image 2024-10-26 at 10 45 16_5ac13f11](https://github.com/user-attachments/assets/00b5b941-a160-463a-b609-986cec37693f)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
