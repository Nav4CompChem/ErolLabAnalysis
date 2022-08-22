import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from itertools import cycle, islice
my_colors = list(islice(cycle(['#BBCCEE', '#FFCCCC', '#CCEEFF', '#EEEEBB']), None, 4))

# colour pallette '#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499'
HDF01 = pd.read_csv('/home/nav/Documents/MultichainStats/dssp_fract.txtLH01', delimiter='\t')
HDF02 = pd.read_csv('/home/nav/Documents/MultichainStats/dssp_fract.txtLH02', delimiter='\t')
HDF03 = pd.read_csv('/home/nav/Documents/MultichainStats/dssp_fract.txtLH03', delimiter='\t')
HDF04 = pd.read_csv('/home/nav/Documents/MultichainStats/dssp_fract.txtLN01', delimiter='\t')
HDF05 = pd.read_csv('/home/nav/Documents/MultichainStats/dssp_fract.txtLN02', delimiter='\t')
HDF06 = pd.read_csv('/home/nav/Documents/MultichainStats/dssp_fract.txtLN03', delimiter='\t')
HDF07 = pd.read_csv("/home/nav/Documents/MultichainStats/dssp_fract.txtDH01", delimiter='\t')
HDF08 = pd.read_csv("/home/nav/Documents/MultichainStats/dssp_fract.txtDH02", delimiter='\t')
HDF09 = pd.read_csv("/home/nav/Documents/MultichainStats/dssp_fract.txtDH03", delimiter='\t')
HDF10 = pd.read_csv("/home/nav/Documents/MultichainStats/dssp_fract.txtDN01", delimiter='\t')
HDF11 = pd.read_csv("/home/nav/Documents/MultichainStats/dssp_fract.txtDN02", delimiter='\t')
HDF12 = pd.read_csv("/home/nav/Documents/MultichainStats/dssp_fract.txtDN03", delimiter='\t')

#RH = {"xticksH": ["10C", "9V", "8V", "7V", "6R", "5F", "4K", "3F", "2R", "1C"]}
#RN = {"xticksN": ["10C", "9I", "8V", "7I", "6R", "5F", "4K", "3F", "2R", "1C"]}

HDF01['Resnum'] += 1
HDF02['Resnum'] += 1
HDF03['Resnum'] += 1
HDF04['Resnum'] += 1
HDF05['Resnum'] += 1
HDF06['Resnum'] += 1
HDF07['Resnum'] += 1
HDF08['Resnum'] += 1
HDF09['Resnum'] += 1
HDF10['Resnum'] += 1
HDF11['Resnum'] += 1
HDF12['Resnum'] += 1

HDF_concat = pd.concat((HDF01, HDF02, HDF03))

b = HDF_concat.groupby(HDF_concat.index)
mean = b.mean()
All_err = b.sem()
All_err.loc[:, ['Resnum']] = HDF01.loc[:, ['Resnum']]
#xticksH = pd.DataFrame(data=RH)
# xticksH.insert(-1, "xticksH" , RH["xticks"], True)
#xticksN = pd.DataFrame(data=RN)
# xticksN.insert(-1, "xticksN" , RN["xticks"], True)

bmean = mean[["Beta", "Resnum"]]
berr = All_err["Beta"].values
bmean.insert(2, "yerr", berr, True)
hmean = mean[["Helix", "Resnum"]]
herr = All_err["Helix"].values
hmean.insert(2, "yerr", herr, True)
cmean = mean[["Coil", "Resnum"]]
cerr = All_err["Coil"].values
cmean.insert(2, "yerr", cerr, True)
btmean = mean[["BT", "Resnum"]]
bterr = All_err["BT"].values
btmean.insert(2, "yerr", bterr, True)
All_err = All_err.reindex(index=All_err.index[::-1])
All_err = pd.concat((All_err.loc[0:0, 'Coil': 'Helix'], All_err))

All_err.reset_index(drop=True, inplace=True)
All_err.drop("Resnum", 1, inplace=True)

ax = bmean.plot.bar(x='Resnum', y='Beta', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                    color='tab:gray', title='Beta Strand Propensity', yerr='yerr', ylim=(0, 1), capsize=5)
#ax.set_xticklabels(xticksH.xticksH)
ax.invert_xaxis()
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/LHydrophobicBetabar.png', bbox_inches='tight')

ax = hmean.plot.bar(x='Resnum', y='Helix', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                    color='tab:gray', title='Helix Propensity', yerr='yerr', ylim=(0, 1), capsize=5)
ax.invert_xaxis()
#ax.set_xticklabels(xticksH.xticksH)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/LHydrophobicHelixbar.png', bbox_inches='tight')

ax = cmean.plot.bar(x='Resnum', y='Coil', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                    color='tab:gray', title='Coil Propensity', yerr='yerr', ylim=(0, 1), capsize=5)
ax.invert_xaxis()
#ax.set_xticklabels(xticksH.xticksH)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/LHydrophobicCoilbar.png', bbox_inches='tight')

ax = btmean.plot.bar(x='Resnum', y='BT', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                     color='tab:gray', title='Bend+Turn Propensity', yerr='yerr', ylim=(0, 1), capsize=5)
ax.invert_xaxis()
#ax.set_xticklabels(xticksH.xticksH)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/LHydrophobicBendTurnbar.png', bbox_inches='tight')

ax = mean.plot.bar(stacked = True, x='Resnum', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                   title='L-Valine Substituted Peptide Secondary Structure Propensity', capsize=1.5, color=my_colors)
ax.invert_xaxis()
#ax.set_xticklabels(xticksH.xticksH)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/LHydrophobicAllbar.png', bbox_inches='tight')

HDF_concat = pd.concat((HDF04, HDF05, HDF06))

b = HDF_concat.groupby(HDF_concat.index)
mean = b.mean()
All_err = b.sem()
All_err.loc[:, ['Resnum']] = HDF01.loc[:, ['Resnum']]
bmean = mean[["Beta", "Resnum"]]
berr = All_err["Beta"].values
bmean.insert(2, "yerr", berr, True)
hmean = mean[["Helix", "Resnum"]]
herr = All_err["Helix"].values
hmean.insert(2, "yerr", herr, True)
cmean = mean[["Coil", "Resnum"]]
cerr = All_err["Coil"].values
cmean.insert(2, "yerr", cerr, True)
btmean = mean[["BT", "Resnum"]]
bterr = All_err["BT"].values
btmean.insert(2, "yerr", bterr, True)

All_err = All_err.reindex(index=All_err.index[::-1])
All_err = pd.concat((All_err.loc[0:0, 'Resnum': 'Helix'], All_err))

All_err.reset_index(drop=True, inplace=True)
All_err.drop("Resnum", 1, inplace=True)

ax = bmean.plot.bar(x='Resnum', y='Beta', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                    color='tab:gray', title='Beta Strand Propensity', yerr='yerr', ylim=(0, 1), capsize=5)
ax.invert_xaxis()
#ax.set_xticklabels(xticksN.xticksN)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/LNativeBetabar.png', bbox_inches='tight')

ax = hmean.plot.bar(x='Resnum', y='Helix', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                    color='tab:gray', title='Helix Propensity', yerr='yerr', ylim=(0, 1), capsize=5)
ax.invert_xaxis()
#ax.set_xticklabels(xticksN.xticksN)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/LNativeHelixbar.png', bbox_inches='tight')

ax = cmean.plot.bar(x='Resnum', y='Coil', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                    color='tab:gray', title='Coil Propensity', yerr='yerr', ylim=(0, 1), capsize=5)
ax.invert_xaxis()
#ax.set_xticklabels(xticksN.xticksN)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/LNativeCoilbar.png', bbox_inches='tight')

ax = btmean.plot.bar(x='Resnum', y='BT', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                     color='tab:gray', title='Bend+Turn Propensity', yerr='yerr', ylim=(0, 1), capsize=5)
ax.invert_xaxis()
#ax.set_xticklabels(xticksN.xticksN)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/LNativeBendTurnbar.png', bbox_inches='tight')

ax = mean.plot.bar(stacked = True , x='Resnum', xlabel='Residue Number', ylabel='Propensity', legend=True, edgecolor='k',
                   title='L-Native Peptide Secondary Structure Propensity', capsize=1.5, color=my_colors)
ax.invert_xaxis()
#ax.set_xticklabels(xticksN.xticksN)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/LNativeAllbar.png', bbox_inches='tight')

HDF_concat = pd.concat((HDF07, HDF08, HDF09))

b = HDF_concat.groupby(HDF_concat.index)
mean = b.mean()
All_err = b.sem()
All_err.loc[:, ['Resnum']] = HDF01.loc[:, ['Resnum']]
bmean = mean[["Beta", "Resnum"]]
berr = All_err["Beta"].values
bmean.insert(2, "yerr", berr, True)
hmean = mean[["Helix", "Resnum"]]
herr = All_err["Helix"].values
hmean.insert(2, "yerr", herr, True)
cmean = mean[["Coil", "Resnum"]]
cerr = All_err["Coil"].values
cmean.insert(2, "yerr", cerr, True)
btmean = mean[["BT", "Resnum"]]
bterr = All_err["BT"].values
btmean.insert(2, "yerr", bterr, True)
All_err = All_err.reindex(index=All_err.index[::-1])
All_err = pd.concat((All_err.loc[0:0, 'Coil': 'Helix'], All_err))

All_err.reset_index(drop=True, inplace=True)

ax = bmean.plot.bar(x='Resnum', y='Beta', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                    color='tab:gray', title='Beta Strand Propensity', yerr='yerr', ylim=(0, 1), capsize=5)
ax.invert_xaxis()
#ax.set_xticklabels(xticksH.xticksH)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/DHydrophobicBetabar.png', bbox_inches='tight')

ax = hmean.plot.bar(x='Resnum', y='Helix', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                    color='tab:gray', title='Helix Propensity', yerr='yerr', ylim=(0, 1), capsize=5)
ax.invert_xaxis()
#ax.set_xticklabels(xticksH.xticksH)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/DHydrophobicHelixbar.png', bbox_inches='tight')

ax = cmean.plot.bar(x='Resnum', y='Coil', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                    color='tab:gray', title='Coil Propensity', yerr='yerr', ylim=(0, 1), capsize=5)
ax.invert_xaxis()
#ax.set_xticklabels(xticksH.xticksH)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/DHydrophobicCoilbar.png', bbox_inches='tight')

ax = btmean.plot.bar(x='Resnum', y='BT', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                     color='tab:gray', title='Bend+Turn Propensity', yerr='yerr', ylim=(0, 1), capsize=5)
ax.invert_xaxis()
#ax.set_xticklabels(xticksH.xticksH)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/DHydrophobicBendTurnbar.png', bbox_inches='tight')

ax = mean.plot.bar(stacked = True, x='Resnum', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                   title='D-Valine Substituted Peptide Secondary Structure Propensity', capsize=1.5, color=my_colors)
ax.invert_xaxis()
#ax.set_xticklabels(xticksH.xticksH)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/DHydrophobicAllbar.png', bbox_inches='tight')

HDF_concat = pd.concat((HDF10, HDF11, HDF12))

b = HDF_concat.groupby(HDF_concat.index)
mean = b.mean()
All_err = b.sem()
All_err.loc[:, ['Resnum']] = HDF01.loc[:, ['Resnum']]
# All_err.loc[-1] = [0,0.5,0.5,0.5,0.7]
# All_err.loc[9:9, 'Coil': 'Helix'] = [0.1,0.1,0.1,0.1]

bmean = mean[["Beta", "Resnum"]]
berr = All_err["Beta"].values
bmean.insert(2, "yerr", berr, True)
hmean = mean[["Helix", "Resnum"]]
herr = All_err["Helix"].values
hmean.insert(2, "yerr", herr, True)
cmean = mean[["Coil", "Resnum"]]
cerr = All_err["Coil"].values
cmean.insert(2, "yerr", cerr, True)
btmean = mean[["BT", "Resnum"]]
bterr = All_err["BT"].values
btmean.insert(2, "yerr", bterr, True)
All_err = All_err.reindex(index=All_err.index[::-1])
All_err = pd.concat((All_err.loc[0:0, 'Coil': 'Helix'], All_err))

All_err.reset_index(drop=True, inplace=True)
All_err.drop("Resnum", 1, inplace=True)

# test = All_err.pivot(index='Resnum')

ax = bmean.plot.bar(x='Resnum', y='Beta', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                    color='tab:gray', title='Beta Strand Propensity', yerr='yerr', ylim=(0, 1), capsize=5)
ax.invert_xaxis()
##ax.set_xticklabels(xticksN.xticksN)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/DNativeBetabar.png', bbox_inches='tight')

ax = hmean.plot.bar(x='Resnum', y='Helix', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                    color='tab:gray', title='Helix Propensity', yerr='yerr', ylim=(0, 1), capsize=5)
ax.invert_xaxis()
#ax.set_xticklabels(xticksN.xticksN)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/DNativeHelixbar.png', bbox_inches='tight')

ax = cmean.plot.bar(x='Resnum', y='Coil', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                    color='tab:gray', title='Coil Propensity', yerr='yerr', ylim=(0, 1), capsize=5)
ax.invert_xaxis()
#ax.set_xticklabels(xticksN.xticksN)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/DNativeCoilbar.png', bbox_inches='tight')

ax = btmean.plot.bar(x='Resnum', y='BT', xlabel='Residue Number', ylabel='Propensity', legend=False, edgecolor='k',
                     color='tab:gray', title='Bend+Turn Propensity', yerr='yerr', ylim=(0, 1), capsize=5)
ax.invert_xaxis()
#ax.set_xticklabels(xticksN.xticksN)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/DNativeBendTurnbar.png', bbox_inches='tight')

ax = mean.plot.bar(stacked= True, x='Resnum', xlabel='Residue Number', ylabel='Propensity', legend=True, edgecolor='k',
                   title='D-Native Peptide Secondary Structure Propensity', ylim=(0, 1), capsize=1.5, color=my_colors)
ax.legend( ncol=4)
ax.invert_xaxis()
#ax.set_xticklabels(xticksN.xticksN)
fig = ax.get_figure()
fig.savefig('/home/nav/Documents/MultichainStats/DNativeAllbar.png', bbox_inches='tight')
# f= np.loadtxt('/home/nav/Documents/MultichainStats/Documents/MultichainStats/dssp_fract_NoHeader.txt', unpack='False')
# bins = [ .1,.2,.3,.4,.5,.6,.7,.8,.9,1.0 ]
# plt.hist(f, histtype='bar', bins = bins)
# plt.xlabel('Residues')
# plt.ylabel('Percentage of time')
# plt.title('beta sheet')
# plt.legend()
# plt.show()
