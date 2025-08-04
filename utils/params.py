###Taken from messy_examples.simulated_data.ipynb
import numpy as np

#simulate some data from a "cluster model"
nvox = 1024
nclus = 5
p = [0.1, 0.1, 0.2, 0.5]
p = np.append(p,1-np.sum(p))
clusters = np.random.choice(range(0,nclus),size=(nvox,),p=p)

#define the underlying tissue parameters for each cluster
D = [0.5,1,1.5,2,3]
K = [1,0.5,0.2,0.1,0.01]
#K = [0.1,0.05,0.2,0.1,0]

mu = np.stack((D,K))
var = np.diag([0.01,0.01])

params = np.zeros((nvox,2))

for vox in range(0,nvox):
    params[vox,:] = np.random.multivariate_normal(mu[:,clusters[vox]],var)

params[params < 0] = 0.01
