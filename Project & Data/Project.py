import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import  StandardScaler,OneHotEncoder
import warnings
import pickle

warnings.filterwarnings('ignore')

Data=pd.read_csv('D:\Documents\Python\VS code\Projects\Customer_Segment_RLProject\marketing_campaign.csv',sep='\t')

print('---Data Information---')

print(Data.head())
print(Data.info())
print(Data.shape)
print('All data points:',len(Data))
print(Data.isnull().sum())

Data['Income']=Data['Income'].fillna(Data['Income'].mean())
print(Data.isnull().any(axis=1))
print(Data.isnull().sum())

print('---Data Statistic Summary---')

Data=Data.rename(columns={"MntWines": "Wines","MntFruits":"Fruits",
"MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})

print(Data.describe())

Data['Kids']=Data['Kidhome']+Data['Teenhome']

Data['Age']=2026-Data['Year_Birth']

Data['Spent']=Data['Wines']+Data['Fruits']+Data['Meat']+Data['Fish']+Data['Sweets']+Data['Gold']


print('Older Custumer:',max(Data['Age']))
print('Younger Custumer:',min(Data['Age']))


print('---Feature Enginnering---')

Data=Data[(Data['Age']<60)]
Data=Data[(Data['Income']<600000)]
Data = Data.reset_index(drop=True)
print('All data point after Removing Outier Data:',len(Data))

Data=Data.rename(columns={"MntWines": "Wines","MntFruits":"Fruits",
"MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})

Drop_FE=['ID','Recency','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2',
'AcceptedCmp3','AcceptedCmp4','Z_CostContact',
'Z_Revenue','Year_Birth','Dt_Customer','Kidhome','Teenhome']

Data.drop(columns=Drop_FE,inplace=True)
print(list(Data.columns))
print(Data.head())


encodeer=OneHotEncoder(sparse_output=False)
encoded=encodeer.fit_transform(Data[['Marital_Status','Education']])
encoded_DF=pd.DataFrame(encoded,columns=encodeer.get_feature_names_out(['Marital_Status','Education']))

numerical=Data[['Age','Income','Spent','Kids']]


##Visualization
plt.figure(figsize=(12,8))
sns.histplot(x=Data['Age'],kde=True)
plt.savefig('Age---Count.png')
plt.title('Age---Count')
plt.show()

plt.figure(figsize=(12,8))
sns.histplot(x=Data['Spent'],kde=True)
plt.savefig('Count---Spent.png')
plt.title('Spent---Count')
plt.show()

plt.figure(figsize=(12,8))
sns.histplot(x=Data['Marital_Status'])
plt.savefig('Marital.png')
plt.title('Marital')
plt.show()

plt.figure(figsize=(12,8))
sns.histplot(x=Data['Education'])
plt.savefig('Education.png')
plt.title('Education')
plt.show()

plt.figure(figsize=(25,25))
sns.heatmap(Data.corr(numeric_only=True),cmap='coolwarm',annot=True)
plt.savefig('Corr of Features.png')
plt.title('Corr of Features')
plt.show()

##Split Data
x=pd.concat([numerical,encoded_DF],axis=1)
## Scalerized Data
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

## PCA for Visualization
pca_2d=PCA(n_components=2)
x_pca_2d=pca_2d.fit_transform(x_scaled)
print('Variance 2D:',pca_2d.explained_variance_)
print('Ratio 2D:',pca_2d.explained_variance_ratio_.sum())

##PCA for Kmean
pca=PCA(n_components=0.95)
x_pca=pca.fit_transform(x_scaled)
print('Variance:',pca.explained_variance_)
print('Ratio:',pca.explained_variance_ratio_)

## Kmean Model
kmean=KMeans(n_clusters=10,n_init=10,random_state=42)
k_label=kmean.fit_predict(x_scaled)
print('k_score:',silhouette_score(x_scaled,k_label))

## ElBow Method
k_params=[]

for k in range(2,11):
    for n in range(10,20):
        k_model=KMeans(
           n_clusters=k,
           n_init=n,
           random_state=42
        )
        label1=k_model.fit_predict(x_scaled)
        score1=silhouette_score(x_scaled,label1)
        k_params.append({
           'n_clusters':k,
           'n_init':n,
           'Score':score1
                        })
k_params=pd.DataFrame(k_params)
print(k_params.sort_values('Score',ascending=False))

## Elbow Method

inertia = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )
    
    kmeans.fit(x_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), inertia, marker='o')

plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.savefig('Elbow.png')
plt.show()

best_kmean=KMeans(n_clusters=9,n_init=10,random_state=42)
best_k_label=best_kmean.fit_predict(x_scaled)
print('best_k_score:',silhouette_score(x_scaled,best_k_label))
print(pd.DataFrame(best_k_label).value_counts())

k_DF=x.copy()
k_DF['Clusters']=best_k_label
print(k_DF.groupby('Clusters').mean())


plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=x_pca_2d[:, 0],
    y=x_pca_2d[:, 1],
    hue=best_k_label,
    palette="viridis"
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clustering")
plt.savefig('K Clusters.png')
plt.show()

loadings=pd.DataFrame(
    pca_2d.components_.T,columns=['PCA1','PCA2'],index=x.columns
)
print(loadings)


##DBscan model

dbscan=DBSCAN(eps=0.5,min_samples=5)
db_label=dbscan.fit_predict(x_scaled)
print('Score:',silhouette_score(x_scaled,db_label))
print('Clusters:',set(db_label))


param_grid = {
    'eps': [1.6,1.7,1.8,1.9,2,2.1,2.2,2.3],
    'min_samples': [3, 5, 8, 10]
}

db_params=[]

for eps in param_grid['eps']:
    for min_s in param_grid['min_samples']:
        db_model=DBSCAN(
            eps=eps,
            min_samples=min_s
        )
        label2=db_model.fit_predict(x_scaled)
        n_cluster=len(set(label2))-(1 if -1 in label2 else 0)
        noise=np.sum(label2==-1)
        score2=silhouette_score(x_scaled,label2)
        db_params.append({
            'N_clusters':n_cluster,
            'eps':eps,
            'Min_Samples':min_s,
            'Score':score2,
            'Noise':noise
        })        

db_params=pd.DataFrame(db_params)
print(db_params.sort_values('Score',ascending=False))


best_db=DBSCAN(eps=1.9,min_samples=3)
best_db_label=best_db.fit_predict(x_scaled)
print('best db score:',silhouette_score(x_scaled,best_db_label))

plt.figure(figsize=(12,8))
sns.scatterplot(
    x=x_pca_2d[:,0],
    y=x_pca_2d[:,1],
    hue=best_db_label,palette='viridis'
)
plt.title('DBscan Clusters')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.savefig('DBscan Clusters.png')
plt.show()

db_DF=x.copy()
db_DF['Clusters']=best_db_label
print(db_DF.groupby('Clusters').mean())


pickle.dump(best_k_label,open('K_model.pkl','wb'))
pickle.dump(best_db_label,open('DB_model.pkl','wb'))
pickle.dump(x.columns,open('Columns.pkl','wb'))
pickle.dump(scaler,open('Scaler.pkl','wb'))

print('All Files Saved')