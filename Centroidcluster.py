

from __future__ import print_function
import mdtraj as md
import numpy as np

traj = md.load_xtc("/home/nav/tcluster.xtc", "/home/nav/tcluster.gro")
print(traj)

atom_indices = [a.index for a in traj.topology.atoms if a.element.symbol != 'H']
distances = np.empty((traj.n_frames, traj.n_frames))
for i in range(traj.n_frames):
    distances[i] = md.rmsd(traj, traj, i, atom_indices=atom_indices)


beta = 1
index = np.exp(-beta*distances / distances.std()).sum(axis=1).argmax()
print(index)

centroid = traj[index]
print(centroid)




