from __future__ import print_function

import numpy
from sklearn.neighbors import NearestCentroid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import fcluster
import mdtraj as md
from scipy.spatial.distance import squareform
traj = md.load_xtc("/home/nav/tcluster.xtc", "/home/nav/tcluster.gro")
traj2 = traj
print(traj2)
distances = np.empty((traj2.n_frames, traj2.n_frames))
atom_indices = [a.index for a in traj.topology.atoms if a.element.symbol != 'H']

for i in range(traj2.n_frames):
    distances[i] = md.rmsd(traj2, traj2, i, atom_indices=atom_indices)
#for i in range(traj.n_frames):
#    distances[i] = md.rmsd(traj, traj, i)
print('Max pairwise rmsd: %f nm' % np.max(distances))
assert np.all(distances - distances.T < 1e-6)
reduced_distances = squareform(distances, checks=False)
linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method='average')
print(len(linkage))
#print(linkage)
np.savetxt("/home/nav/linkage.txt", linkage)
#clf = NearestCentroid()
#clf.fit(linkage)
#print(clf.centroids_)
flatclust = fcluster(linkage, t=0.40, criterion='distance')
print(flatclust)
nbOClust = len(numpy.unique(flatclust))
df = pd. DataFrame(flatclust, columns=['Cluster'])
i = 1
while i <= nbOClust:

    indexB = df.index[df['Cluster'] == i].array
    clustTraj = traj[indexB]

    atom_indices = [a.index for a in clustTraj.topology.atoms if a.element.symbol != 'H']
    distances = np.empty((clustTraj.n_frames, clustTraj.n_frames))
    for j in range(clustTraj.n_frames):
        distances[j] = md.rmsd(clustTraj, clustTraj, j, atom_indices=atom_indices)

    beta = 1
    index2 = np.exp(-beta * distances / distances.std()).sum(axis=1).argmax()
    print("centroid is {0}".format(indexB[index2]))
    #print(clustTraj)
    print("{0} {1}".format(i, len(indexB) ))
    md.Trajectory.save_gro(traj[indexB[index2]], "/home/nav/clustering/Cluster{0}frame{1}.gro".format(i, indexB[index2]))
    i += 1
#print(df.loc["Cluster":1])

plt.title('RMSD Average linkage hierarchical clustering')
_ = scipy.cluster.hierarchy.dendrogram(linkage, no_labels=False, count_sort='descendent')


plt.show()
