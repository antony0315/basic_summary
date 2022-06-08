#%%PCA
from skleanr.decomposition import PCA
train_x,test_x=load_standarized_data()   #需先標準化
print(train_x.shape)
print(test_x.shape)
pca=PCA(n_component=5)
pca.fit(train_x)
train_x=pca.transform(train_x)
test_x=pca.transform(test_x)
print(train_x.shape)
print(test_x.shape)

#%%NMF  非負矩陣分解
from sklearn.decomposition import NMF
train_x,test_x=load_minmax_scaler_data()
model=NMF(n_components=5,init='random',random_state=71)
model.fit(tain_x)
train_x=model.transform(train_x)
test_x=model.transfrom(test_x)

#%%Latent_Dirichlet_Allocation(LDA)自然語言
train_x,test_x=load_minmax_scales_data()
from sklearn.decomposition import LatentDirichletAllocation
model=LatentDirichletAllocation(n_components=5,random_state=71)
model.fit(train_x)
train_x=model.transform(train_x)
test_x=model.transform(test_x)

#%%linear Discriminant Analysis(LDA)
train_x,test_x=load_standarized_data()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=1)
lda.fit(train_x,train_y)
train_x=lda.transfrom(train_x)
tes_x=lda.transform(test_x)

#%%t-SNE
train_x,test_x=load_standarized_data()
import bhtsne
data=pd.concat([train_x,test_x])
embedded=bhtsne.run_bh_tsne(data.astype(mp.float64),initial_dims=2)


#%%UMAP   較t-sne快
train_x,test_x=load_standarized_data
import umap
um=nmap.UMAP()
um.fit(train_x)
train_x=um.transform(train_x)
test_x=um.transform(test_x)



#%%cluster analysis群聚分析
train_x,test_x=load_standarized_data()
from sklearn.cluster import MiniBatchKMeans
kmeans=MiniBatchKMeans(n_clusters=10,random_state=71)
kmeans.fit(train_x)
kmeans.fit(train_x)
train_cluster=kmeans.predict(train_x)
test_clusters=kmeans.predict(test_x)
train_distances=kmeans.transform(train_x)
test_distances=kmeans.transform(test_X)












