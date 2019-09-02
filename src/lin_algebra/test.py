
from lin_algebra import *
import numpy as np
x=np.array([[-3,0,0],[0,1,0],[0,0,2]])
from sklearn.preprocessing import StandardScaler


from sklearn.decomposition import PCA

p = PCA(n_components=3)

#print(p.fit_transform(x))

x_std = StandardScaler().fit_transform(x);
p.fit_transform(x_std);

print(p.fit_transform(x_std))

print(pca(x_std,2))
