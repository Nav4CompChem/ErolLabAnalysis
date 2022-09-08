import MDAnalysis as mda
from MDAnalysis.tests.datafiles import (PSF, DCD, DCD2, GRO, XTC, PSF_NAMD_GBIS, DCD_NAMD_GBIS)
from MDAnalysis.analysis import encore
from MDAnalysis.analysis.encore.clustering import ClusteringMethod as clm

import numpy as np
import matplotlib.pyplot as plt

#matplotlib inline

u1 = mda.Universe("/home/nav/tcluster.gro", "/home/nav/tcluster.xtc")

labels = ['Multi']

print(len(u1.trajectory))

ces0, details0 = encore.ces([u1])

ces0

cluster_collection = encore.cluster([u1])
print(type(cluster_collection))
print('We have found {} clusters'.format(len(cluster_collection)))


first_cluster = cluster_collection
first_cluster

#first_cluster.elements

#print('The ID of this cluster is:', first_cluster.id)
#print('The centroid is', first_cluster.centroid)

fig0, ax0 = plt.subplots()
im0 = plt.imshow(ces0, vmax=np.log(2), vmin=0)
plt.title('Clustering ensemble similarity')
cbar0 = fig0.colorbar(im0)
cbar0.set_label('Jensen-Shannon divergence')
plt.show()


