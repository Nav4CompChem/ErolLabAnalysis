import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import mdshare
import pyemma
from pyemma.util.contexts import settings
import mdtraj as md
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator
from matplotlib.cm import get_cmap
from pyemma.plots.markovtests import _add_ck_subplot

files = ["/media/nav/New Volume1/Peptide/NativePeptide//01/5_NPT_long/traj_comp_skip10_nowater.xtc", "/media/nav/New Volume1/Peptide/NativePeptide/02/5_NPT_long/traj_comp_skip10_nowater.xtc",
         "/media/nav/New Volume1/Peptide/NativePeptide/03/5_NPT_long/traj_comp_skip10_nowater.xtc", "/media/nav/New Volume1/Peptide/NativePeptide/04/5_NPT_long/traj_comp_skip10_nowater.xtc",
         "/media/nav/New Volume1/Peptide/NativePeptide/05/5_NPT_long/traj_comp_skip10_nowater.xtc"
        ]
pdb = "/media/nav/New Volume1/Peptide/NativePeptide/01/5_NPT_long/nowater.gro"
file = "/media/nav/New Volume1/Peptide/NativePeptide/NativePeptide/01/5_NPT_long/traj_comp_skip10_nowater.xtc"

sampleTraj = md.load_xtc(files[0], pdb, stride=2)
x = 0
atoms = sampleTraj.topology.atoms

atom_indices2 = [(a.index)+1 for a in sampleTraj.topology.atoms if a.element.symbol == 'N' ]
atom_indicesH = [ a.index for a in sampleTraj.topology.atoms if a.element.symbol == 'O' ]

print(atom_indices2)
print(atom_indicesH)

#pdb = mdshare.fetch('pentapeptide-impl-solv.pdb', working_directory='data')
#files = mdshare.fetch('pentapeptide-*-500ns-impl-solv.xtc', working_directory='data')
print("here")




torsions_feat = pyemma.coordinates.featurizer(pdb)
torsions_feat.add_backbone_torsions( cossin=True)
torsions_data = pyemma.coordinates.load(files, features=torsions_feat)
labels = ['backbone\ntorsions']



torsions_feat2 = pyemma.coordinates.featurizer(pdb)
torsions_feat2.add_backbone_torsions( selstr="not resname CYS", cossin=True)
torsions_feat2.add_chi1_torsions(selstr="resname CYS", cossin=True)
torsions_data2 = pyemma.coordinates.load(files, features=torsions_feat2)

#labels += ['minus \n 2 res']



torsions_feat3 = pyemma.coordinates.featurizer(pdb)
torsions_feat3.add_backbone_torsions( selstr="not resname CYS and not resid 2 and not resid 9 ", cossin=True)
torsions_data3 = pyemma.coordinates.load(files, features=torsions_feat3)
#labels += ['minus \n 4 res']



torsions_feat4 = pyemma.coordinates.featurizer(pdb)
torsions_feat4.add_backbone_torsions( selstr="not resname CYS and not resid 2 and not resid 9 and not resid 8 and not resid 3 and not resid 4 and not resid 7", cossin=True)
torsions_data4 = pyemma.coordinates.load(files, features=torsions_feat4)
#labels += ['minus \n 6 res']

torsions_feat5 = pyemma.coordinates.featurizer(pdb)
torsions_feat5.add_backbone_torsions( selstr="resname CYS", cossin=True)
torsions_data5 = pyemma.coordinates.load(files, features=torsions_feat5)
#labels += ['Just Cys']


print("selestr worked")

positions_feat = pyemma.coordinates.featurizer(pdb)
positions_feat.add_selection(positions_feat.select_Backbone())
positions_data = pyemma.coordinates.load(files, features=positions_feat)


#labels += ['backbone atom\npositions']

rmsd_feat = pyemma.coordinates.featurizer(pdb)
i = 0
k = 0
"""while i < 1000:
    rmsd_feat.add_minrmsd_to_ref(files[k], ref_frame=(i*50), atom_indices=positions_feat.select_Backbone())
    i+=1


rmsd_data = pyemma.coordinates.load(files, features=rmsd_feat)
labels += ['backbone\nrmsd']"""


distances_feat = pyemma.coordinates.featurizer(pdb)
distances_feat.add_distances(
    distances_feat.pairs(distances_feat.select_Backbone(), excluded_neighbors=2), periodic=False)
distances_data = pyemma.coordinates.load(files, features=distances_feat)
#labels += ['backbone atom\ndistances']

chi1_feat = pyemma.coordinates.featurizer(pdb)
chi1_feat.add_chi1_torsions(cossin=True)
chi1_data = pyemma.coordinates.load(files, features=torsions_feat)
#labels += ['Chi1\n torsions']


combined_torsions_feat = pyemma.coordinates.featurizer(pdb)
combined_torsions_feat.add_chi1_torsions(cossin=True)
combined_torsions_feat.add_backbone_torsions(cossin=True)
combined_torsions_data = pyemma.coordinates.load(files, features=combined_torsions_feat)
labels += ['Combined\n torsions']

print("Setting up contact features")
contacts_feat = pyemma.coordinates.featurizer(pdb)
contacts_feat.add_contacts(atom_indices2, atom_indicesH)
contacts_data = pyemma.coordinates.load(files, features=contacts_feat)
labels += ['Contacts']


contacts_torsions_feat = pyemma.coordinates.featurizer(pdb)
contacts_torsions_feat.add_chi1_torsions(cossin=True)
contacts_torsions_feat.add_backbone_torsions(cossin=True)
contacts_torsions_feat.add_contacts(atom_indices2, atom_indicesH)
contacts_torsions_data = pyemma.coordinates.load(files, features=contacts_torsions_feat )

#labels += ['Contacts and \n torsions']

print(contacts_feat.dimension())
print(torsions_feat5.dimension())

dim = 10
lags = [1, 2, 5, 10, 20]
dims = [i + 1 for i in range(10)]
n_clustercenters = [5, 10, 30, 75, 200, 450]
nstates = 5
nits = 15


def score_cv(data, dim, lag, number_of_splits=10, validation_fraction=0.5):
    """Compute a cross-validated VAMP2 score.

    We randomly split the list of independent trajectories into
    a training and a validation set, compute the VAMP2 score,
    and repeat this process several times.

    Parameters
    ----------
    data : list of numpy.ndarrays
        The input data.
    dim : int
        Number of processes to score; equivalent to the dimension
        after projecting the data with VAMP2.
    lag : int
        Lag time for the VAMP2 scoring.
    number_of_splits : int, optional, default=10
        How often do we repeat the splitting and score calculation.
    validation_fraction : int, optional, default=0.5
        Fraction of trajectories which should go into the validation
        set during a split.
    """
    # we temporarily suppress very short-lived progress bars
    with pyemma.util.contexts.settings(show_progress_bars=False):
        nval = int(len(data) * validation_fraction)
        scores = np.zeros(number_of_splits)
        for n in range(number_of_splits):
            ival = np.random.choice(len(data), size=nval, replace=False)
            vamp = pyemma.coordinates.vamp(
                [d for i, d in enumerate(data) if i not in ival], lag=lag, dim=dim)
            scores[n] = vamp.score([d for i, d in enumerate(data) if i in ival])
    return scores



def vampFeatureComparison():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    for ax, lag in zip(axes.flat, [5, 10, 20]):
        torsions_scores = score_cv(torsions_data, lag=lag, dim=dim)
        scores = [torsions_scores.mean()]
        errors = [torsions_scores.std()]
        positions_scores = score_cv(positions_data, lag=lag, dim=dim)
        scores += [positions_scores.mean()]
        errors += [positions_scores.std()]
        distances_scores = score_cv(distances_data, lag=lag, dim=dim)
        scores += [distances_scores.mean()]
        errors += [distances_scores.std()]
        ax.bar(labels, scores, yerr=errors, color=['C0', 'C1', 'C2'])
        ax.set_title(r'lag time $\tau$={:.1f}ns'.format(lag * 0.1))
        global vamp_bars_plot
        if lag == 5:
            # save for later
            vamp_bars_plot = dict(
                labels=labels, scores=scores, errors=errors, dim=dim, lag=lag)
    axes[0].set_ylabel('VAMP2 score')
    fig.tight_layout()



def dimensionVamp():
    fig, ax = plt.subplots()
    for i, lag in enumerate(lags):
        scores_ = np.array([score_cv(torsions_data, dim, lag)
                        for dim in dims])
        scores = np.mean(scores_, axis=1)
        errors = np.std(scores_, axis=1, ddof=1)
        color = 'C{}'.format(i)
        ax.fill_between(dims, scores - errors, scores + errors, alpha=0.3, facecolor=color)
        ax.plot(dims, scores, '--o', color=color, label='lag={:.1f}ns'.format(lag * 0.1))
    ax.legend()
    ax.set_xlabel('number of dimensions')
    ax.set_ylabel('VAMP2 score')
    fig.tight_layout()



def hMap():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pyemma.plots.plot_feature_histograms(
        tica_concatenated,
        ax=axes[0],
        ylog=True)
    pyemma.plots.plot_density(*tica_concatenated[:, :2].T, ax=axes[1], logscale=True)
    axes[1].set_xlabel('IC 1')
    axes[1].set_ylabel('IC 2')
    fig.tight_layout()

def stability():
    fig, axes = plt.subplots(4, 1, figsize=(12, 5), sharex=True)
    x = 0.1 * np.arange(tica_output[0].shape[0])
    for i, (ax, tic) in enumerate(zip(axes.flat, tica_output[0].T)):
        ax.plot(x, tic)
        ax.set_ylabel('IC {}'.format(i + 1))
    axes[-1].set_xlabel('time / ns')
    fig.tight_layout()

def clusterNum():


    scores = np.zeros((len(n_clustercenters), 5))
    for n, k in enumerate(n_clustercenters):
        for m in range(5):
            with pyemma.util.contexts.settings(show_progress_bars=False):
                _cl = pyemma.coordinates.cluster_kmeans(
                    tica_output, k=k, max_iter=50, stride=50)
                _msm = pyemma.msm.estimate_markov_model(_cl.dtrajs, 5)
                scores[n, m] = _msm.score_cv(
                    _cl.dtrajs, n=1, score_method='VAMP2', score_k=min(10, k))

    fig, ax = plt.subplots()
    lower, upper = pyemma.util.statistics.confidence_interval(scores.T.tolist(), conf=0.9)
    ax.fill_between(n_clustercenters, lower, upper, alpha=0.3)
    ax.plot(n_clustercenters, np.mean(scores, axis=1), '-o')
    ax.semilogx()
    ax.set_xlabel('number of cluster centers')
    ax.set_ylabel('VAMP-2 score')
    fig.tight_layout()





def density():
    fig, ax = plt.subplots(figsize=(4, 4))
    pyemma.plots.plot_density(
        *tica_concatenated[:, :2].T, ax=ax, cbar=False, alpha=0.3)
    ax.scatter(*cluster.clustercenters[:, :2].T, s=5, c='C1')
    ax.set_xlabel('IC 1')
    ax.set_ylabel('IC 2')
    fig.tight_layout()

def its_separation_err(ts, ts_err):
    """
    Error propagation from ITS standard deviation to timescale separation.
    """
    return ts[:-1] / ts[1:] * np.sqrt(
        (ts_err[:-1] / ts[:-1])**2 + (ts_err[1:] / ts[1:])**2)




def timeScales():
    timescales_mean = msm.sample_mean('timescales', k=nits)
    timescales_std = msm.sample_std('timescales', k=nits)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].errorbar(
        range(1, nits + 1),
        timescales_mean,
        yerr=timescales_std,
        fmt='.', markersize=10)
    axes[1].errorbar(
        range(1, nits),
        timescales_mean[:-1] / timescales_mean[1:],
        yerr=its_separation_err(
            timescales_mean,
            timescales_std),
        fmt='.',
        markersize=10,
        color='C0')

    for i, ax in enumerate(axes):
        ax.set_xticks(range(1, nits + 1))
        ax.grid(True, axis='x', linestyle=':')

    axes[0].axhline(msm.lag * 0.1, lw=1.5, color='k')
    axes[0].axhspan(0, msm.lag * 0.1, alpha=0.3, color='k')
    axes[0].set_xlabel('implied timescale index')
    axes[0].set_ylabel('implied timescales / ns')
    axes[1].set_xticks(range(1, nits))
    axes[1].set_xticklabels(
        ["{:d}/{:d}".format(k, k + 1) for k in range(1, nits)],
        rotation=45)
    axes[1].set_xlabel('implied timescale indices')
    axes[1].set_ylabel('timescale separation')
    fig.tight_layout()

def reweightedMap():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    pyemma.plots.plot_contour(
        *tica_concatenated[:, :2].T,
        msm.pi[dtrajs_concatenated],
        ax=axes[0],
        mask=True,
        cbar_label='stationary distribution')
    pyemma.plots.plot_free_energy(
        *tica_concatenated[:, :2].T,
        weights=np.concatenate(msm.trajectory_weights()),
        ax=axes[1],
        legacy=False)
    for ax in axes.flat:
        ax.set_xlabel('IC 1')
    axes[0].set_ylabel('IC 2')
    axes[0].set_title('Stationary distribution', fontweight='bold')
    axes[1].set_title('Reweighted free energy surface', fontweight='bold')
    fig.tight_layout()

def eigenMap():
    eigvec = msm.eigenvectors_right()
    print('The first eigenvector is one: {} (min={}, max={})'.format(
        np.allclose(eigvec[:, 0], 1, atol=1e-15), eigvec[:, 0].min(), eigvec[:, 0].max()))

    fig, axes = plt.subplots(1, 4, figsize=(15, 3), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        pyemma.plots.plot_contour(
            *tica_concatenated[:, :2].T,
            eigvec[dtrajs_concatenated, i + 1],
            ax=ax,
            cmap='PiYG',
            cbar_label='{}. right eigenvector'.format(i + 2),
            mask=True)
        ax.set_xlabel('IC 1')
    axes[0].set_ylabel('IC 2')
    fig.tight_layout()


vampFeatureComparison()
dimensionVamp()
plt.show()
tica = pyemma.coordinates.tica(torsions_data, lag=100)
tica_output = tica.get_output()
tica_concatenated = np.concatenate(tica_output)
cluster = pyemma.coordinates.cluster_kmeans(
    tica_output, k=30, max_iter=150, stride=10)
dtrajs_concatenated = np.concatenate(cluster.dtrajs)

hMap()
stability()
clusterNum()
density()
print("preits")
plt.show()

its = pyemma.msm.its(cluster.dtrajs, lags=150, nits=10, errors='bayes', n_jobs=1)
pyemma.plots.plot_implied_timescales(its, units='ns', dt=0.1);
msm = pyemma.msm.bayesian_markov_model(cluster.dtrajs, lag=100, dt_traj='0.1 ns')
print('fraction of states used = {:.2f}'.format(msm.active_state_fraction))
print('fraction of counts used = {:.2f}'.format(msm.active_count_fraction))
cktest = msm.cktest(nstates, mlags=6, n_jobs=1)
pyemma.plots.plot_cktest(cktest, dt=0.1, units='ns');


timeScales()
reweightedMap()
eigenMap()
print(vamp_bars_plot)
plt.show()

msm.pcca(nstates)

def membershipMap():
    fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        pyemma.plots.plot_contour(
            *tica_concatenated[:, :2].T,
            msm.metastable_distributions[i][dtrajs_concatenated],
            ax=ax,
            cmap='afmhot_r',
            mask=True,
            cbar_label='metastable distribution {}'.format(i + 1))
        ax.set_xlabel('IC 1')
    axes[0].set_ylabel('IC 2')
    fig.tight_layout()

def stateMap():
    metastable_traj = msm.metastable_assignments[dtrajs_concatenated]

    fig, ax = plt.subplots(figsize=(5, 4))
    _, _, misc = pyemma.plots.plot_state_map(
        *tica_concatenated[:, :2].T, metastable_traj, ax=ax)
    ax.set_xlabel('IC 1')
    ax.set_ylabel('IC 2')
    misc['cbar'].set_ticklabels([r'$\mathcal{S}_%d$' % (i + 1)
                                 for i in range(nstates)])
    fig.tight_layout()


membershipMap()
stateMap()
plt.show()

pcca_samples = msm.sample_by_distributions(msm.metastable_distributions, 10)
torsions_source = pyemma.coordinates.source(files, features=torsions_feat)
pyemma.coordinates.save_trajs(
    torsions_source,
    pcca_samples,
    outfiles=['./data/pcca{}_10samples.pdb'.format(n + 1)
              for n in range(msm.n_metastable)])

def visualize_metastable(samples, cmap, selection='not element H'):
    """ visualize metastable states
    Parameters
    ----------
    samples: list of mdtraj.Trajectory objects
        each element contains all samples for one metastable state.
    cmap: matplotlib.colors.ListedColormap
        color map used to visualize metastable states before.
    selection: str
        which part of the molecule to selection for visualization. For details have a look here:
        http://mdtraj.org/latest/examples/atom-selection.html#Atom-Selection-Language
    """
    import nglview
    from matplotlib.colors import to_hex

    widget = nglview.NGLWidget()
    widget.clear_representations()
    ref = samples[0]
    for i, s in enumerate(samples):
        s = s.superpose(ref, atom_indices=s.top.select('resid 2 3 and mass > 2'))
        s = s.atom_slice(s.top.select(selection))
        comp = widget.add_trajectory(s)
        comp.add_licorice()

    # this has to be done in a separate loop for whatever reason...
    x = np.linspace(0, 1, num=len(samples))
    for i, x_ in enumerate(x):
        c = to_hex(cmap(x_))
        widget.update_licorice(color=c, component=i, repr_index=i)
        widget.remove_cartoon(component=i)
    return widget

my_samples = [pyemma.coordinates.save_traj(files, idist, outfile=None, top=pdb)
              for idist in msm.sample_by_distributions(msm.metastable_distributions, 50)]

cmap = mpl.cm.get_cmap('viridis', nstates)
visualize_metastable(my_samples, cmap)

print('state\tπ\t\tG/kT')
for i, s in enumerate(msm.metastable_sets):
    p = msm.pi[s].sum()
    print('{}\t{:f}\t{:f}'.format(i + 1, p, -np.log(p)))

from itertools import product

mfpt = np.zeros((nstates, nstates))
for i, j in product(range(nstates), repeat=2):
    mfpt[i, j] = msm.mfpt(
        msm.metastable_sets[i],
        msm.metastable_sets[j])

from pandas import DataFrame
print('MFPT / ns:')
DataFrame(np.round(mfpt, decimals=2), index=range(1, nstates + 1), columns=range(1, nstates + 1))

A = msm.metastable_sets[0]
B = np.concatenate(msm.metastable_sets[1:])
print('MFPT 1 -> other: ({:6.1f} ± {:5.1f}) ns'.format(
    msm.sample_mean('mfpt', A, B), msm.sample_std('mfpt', A, B)))
print('MFPT other -> 1: ({:.1f} ± {:5.1f}) ns'.format(
    msm.sample_mean('mfpt', B, A), msm.sample_std('mfpt', B, A)))


start, final = 1, 3
A = msm.metastable_sets[start]
B = msm.metastable_sets[final]
flux = pyemma.msm.tpt(msm, A, B)

cg, cgflux = flux.coarse_grain(msm.metastable_sets)

fig, ax = plt.subplots(figsize=(5, 4))

pyemma.plots.plot_contour(
    *tica_concatenated[:, :2].T,
    flux.committor[dtrajs_concatenated],
    cmap='brg',
    ax=ax,
    mask=True,
    cbar_label=r'committor $\mathcal{S}_%d \to \mathcal{S}_%d$' % (
        start + 1, final + 1))
fig.tight_layout()

plt.show()
from mdtraj import shrake_rupley, compute_rg

#We compute a maximum likelihood MSM for comparison
mlmsm = pyemma.msm.estimate_markov_model(cluster.dtrajs, lag=100, dt_traj='0.1 ns')

from mdtraj import shrake_rupley, compute_rg

#We compute a maximum likelihood MSM for comparison
mlmsm = pyemma.msm.estimate_markov_model(cluster.dtrajs, lag=100, dt_traj='0.1 ns')

markov_samples = [smpl for smpl in msm.sample_by_state(20)]

reader = pyemma.coordinates.source(files, top=pdb)
samples = [pyemma.coordinates.save_traj(reader, smpl, outfile=None, top=pdb)
            for smpl in markov_samples]

# Compute solvent accessible surface area for all samples
markov_sasa_all = [shrake_rupley(sample, mode='residue')
                   for sample in samples]

# Compute radius of gyration for all samples
markov_rg_all = [compute_rg(sample) for sample in samples]

# Average over Markov states for both observables.
markov_average_trp_sasa = np.array(markov_sasa_all).mean(axis=1)[:, 0]
markov_average_rg = np.array(markov_rg_all).mean(axis=1)

print('The average radius of gyration of penta-peptide is'
      ' {:.3f} nm'.format(msm.expectation(markov_average_rg)))

print('The standard deviation of our prediction of the average radius of gyration'
      ' of pentapeptide is {:.9f} nm'.format(
          msm.sample_std('expectation', markov_average_rg)))
print('The {:d}% CI of our prediction of the average radius of gyration of'
      ' pentapeptide have the bounds ({:.5f}, {:.5f})'.format(
          int(msm.conf * 100), *msm.sample_conf('expectation', markov_average_rg)))

fig, ax = plt.subplots(figsize=(5, 4))
pyemma.plots.plot_contour(
    *tica_concatenated[:, :2].T,
    markov_average_trp_sasa[dtrajs_concatenated],
    ax=ax,
    mask=True,
    cbar_label=r'Trp-1 SASA / nm$^2$')
ax.set_xlabel('IC 1')
ax.set_ylabel('IC 2')
fig.tight_layout()

plt.show()
def trpAuto():
    eq_time_ml, eq_acf_ml = mlmsm.correlation(markov_average_trp_sasa, maxtime=150)

    eq_time_bayes, eq_acf_bayes = msm.sample_mean(
        'correlation',
        np.array(markov_average_trp_sasa),
        maxtime=150)

    eq_acf_bayes_ci_l, eq_acf_bayes_ci_u = msm.sample_conf(
        'correlation',
        np.array(markov_average_trp_sasa),
        maxtime=150)

    fig, ax = plt.subplots()
    ax.plot(eq_time_ml, eq_acf_ml, '-o', color='C1', label='ML MSM')
    ax.plot(
        eq_time_bayes,
        eq_acf_bayes,
        '--x',
        color='C0',
        label='Bayes sample mean')
    ax.fill_between(
        eq_time_bayes,
        eq_acf_bayes_ci_l[1],
        eq_acf_bayes_ci_u[1],
        facecolor='C0',
        alpha=0.3)
    ax.semilogx()

    ax.set_xlim((eq_time_ml[1], eq_time_ml[-1]))
    ax.set_xlabel(r'time / $\mathrm{ns}$')
    ax.set_ylabel(r'Trp-1 SASA ACF / $\mathrm{nm}^4$')

    ax.legend()
    fig.tight_layout()
    eq_time_ml, eq_relax_ml = mlmsm.relaxation(
        msm.metastable_distributions[0],
        markov_average_trp_sasa,
        maxtime=150)

    eq_time_bayes, eq_relax_bayes = msm.sample_mean(
        'relaxation',
        msm.metastable_distributions[0],
        np.array(markov_average_trp_sasa),
        maxtime=150)

    eq_relax_bayes_CI_l, eq_relax_bayes_CI_u = msm.sample_conf(
        'relaxation',
        msm.metastable_distributions[0],
        np.array(markov_average_trp_sasa),
        maxtime=150)

    fig, ax = plt.subplots()
    ax.plot(eq_time_ml, eq_relax_ml, '-o', color='C1', label='ML MSM')
    ax.plot(
        eq_time_bayes,
        eq_relax_bayes,
        '--x',
        color='C0',
        label='Bayes sample mean')
    ax.fill_between(
        eq_time_bayes,
        eq_relax_bayes_CI_l[1],
        eq_relax_bayes_CI_u[1],
        facecolor='C0',
        alpha=0.3)
    ax.semilogx()

    ax.set_xlim((eq_time_ml[1], eq_time_ml[-1]))
    ax.set_xlabel(r'time / $\mathrm{ns}$')
    ax.set_ylabel(r'Average Trp-1 SASA / $\mathrm{nm}^2$')

    ax.legend()
    fig.tight_layout()

trpAuto()
plt.show()

state2ensemble = np.abs(msm.expectation(markov_average_trp_sasa) -
                        msm.metastable_distributions.dot(np.array(markov_average_trp_sasa)))
DataFrame(np.round(state2ensemble, 3), index=range(1, nstates + 1), columns=[''])



mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 5
mpl.rcParams['xtick.labelsize'] = 5
mpl.rcParams['ytick.labelsize'] = 5
mpl.rcParams['xtick.minor.pad'] = 2
mpl.rcParams['xtick.major.pad'] = 3
mpl.rcParams['ytick.minor.pad'] = 2
mpl.rcParams['ytick.major.pad'] = 3
mpl.rcParams['axes.labelpad'] = 1
mpl.rcParams['lines.markersize'] = 4

def scoreFigure():
    fig = plt.figure(figsize=(3.47, 4.65))
    gw = int(np.floor(0.5 + 1000 * fig.get_figwidth()))
    gh = int(np.floor(0.5 + 1000 * fig.get_figheight()))
    gs = plt.GridSpec(gh, gw)
    gs.update(hspace=0.0, wspace=0.0, left=0.0, right=1.0, bottom=0.0, top=1.0)

    ax_box = fig.add_subplot(gs[:, :])
    ax_box.set_axis_off()
    ax_box.text(0.00, 0.95, '(a)', size=10, zorder=1)
    ax_box.text(0.00, 0.58, '(b)', size=10)
    ax_box.text(0.55, 0.58, '(c)', size=10)
    ax_box.text(0.00, 0.22, '(d)', size=10)

    ax_mol = fig.add_subplot(gs[:1600, -2820:-400])
    ax_mol.set_axis_off()
    ax_mol.imshow(plt.imread('notreduced.png'))

    ax_feat = fig.add_subplot(gs[2000:3150, 400:1800])
    ax_feat.bar(
        vamp_bars_plot['labels'],
        vamp_bars_plot['scores'],
        yerr=vamp_bars_plot['errors'],
        color=['C0', 'C1', 'C2'])
    ax_feat.set_ylabel('VAMP2 score')
    ax_feat.set_title(r'lag time $\tau$ = {:.1f} ns'.format(vamp_bars_plot['lag'] * 0.1))
    ax_feat.set_ylim(2.75, 3.65)
    ax_feat.tick_params(axis='x', labelrotation=20)

    ax_sample_free_energy = fig.add_subplot(gs[2000:3150, 2200:3350])
    _, _, misc = pyemma.plots.plot_free_energy(
        *tica_concatenated.T[:2],
        ax=ax_sample_free_energy,
        cax=fig.add_subplot(gs[1900:1950, 2200:3350]),
        cbar_orientation='horizontal',
        legacy=False)
    misc['cbar'].set_label('sample free energy / kT')
    misc['cbar'].set_ticks(np.arange(9))
    misc['cbar'].ax.xaxis.set_ticks_position('top')
    misc['cbar'].ax.xaxis.set_label_position('top')
    ax_sample_free_energy.set_xlabel('IC 1')
    ax_sample_free_energy.set_ylabel('IC 2')

    x = 0.1 * np.arange(tica_output[0].shape[0])
    ax_tic1 = fig.add_subplot(gs[3650:4000, 400:3350])
    ax_tic2 = fig.add_subplot(gs[4000:4350, 400:3350])

    ax_tic1.plot(x, tica_output[0][:, 0], linewidth=0.25)
    ax_tic2.plot(x, tica_output[0][:, 1], linewidth=0.25)
    ax_tic1.set_ylabel('IC 1')
    ax_tic2.set_ylabel('IC 2')
    ax_tic2.set_xlabel('time / ns')

    fig.savefig('data/figure_3.pdf', dpi=300)

scoreFigure()