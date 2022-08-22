import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# Load data from CSV
dat1 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgLH01')
dat2 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgLH02')
dat3 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgLH03')
dat4 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgLH04')
dat5 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgLH05')
dat6 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgLN01')
dat7 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgLN02')
dat8 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgLN03')
dat9 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgLN04')
dat10 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgLN05')
dat11 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgDH01')
dat12 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgDH02')
dat13 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgDH03')
dat14 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgDH04')
dat15 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgDH05')
dat16 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgDN01')
dat17 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgDN02')
dat18 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgDN03')
dat19 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgDN04')
dat20 = np.genfromtxt('/home/nav/Documents/LinearStats/RMSF.xvgDN05')
X = dat1[:,0]
Ym1 = dat1[:,1]
Ym2 = dat2[:,1]
Ym3 = dat3[:,1]
Ym4 = dat4[:,1]
Ym5 = dat5[:,1]
Ym6 = dat6[:,1]
Ym7 = dat7[:,1]
Ym8 = dat8[:,1]
Ym9 = dat9[:,1]
Ym10 = dat10[:,1]
Ym11 = dat11[:,1]
Ym12 = dat12[:,1]
Ym13 = dat13[:,1]
Ym14 = dat14[:,1]
Ym15 = dat15[:,1]
Ym16 = dat16[:,1]
Ym17 = dat17[:,1]
Ym18 = dat18[:,1]
Ym19 = dat19[:,1]
Ym20 = dat20[:,1]



# compute the standard error of the mean
avg1 = np.mean([Ym1, Ym2, Ym3, Ym4, Ym5 ], axis=0)
error1 = stats.sem((Ym1, Ym2, Ym3, Ym4, Ym5 ), axis=0, ddof=1)
avg2 = np.mean([Ym6, Ym7, Ym8, Ym9, Ym10], axis=0)
error2 = stats.sem((Ym6, Ym7, Ym8, Ym9, Ym10), axis=0, ddof=1)
avg3 = np.mean([Ym11, Ym12, Ym13, Ym14, Ym15], axis=0)
error3 = stats.sem((Ym11, Ym12, Ym13, Ym14, Ym15), axis=0, ddof=1)
avg4 = np.mean([Ym16, Ym17, Ym18, Ym19, Ym20], axis=0)
error4 = stats.sem((Ym16, Ym17, Ym18, Ym19, Ym20), axis=0, ddof=1)
# Create the plot

plt.plot(X, avg1, color='#E11845', linewidth=0.85)
plt.fill_between(X, avg1-error1, avg1+error1, color='#E11845', alpha=0.3)
plt.plot(X, avg2, color='#0057E9', linewidth=0.85)
plt.fill_between(X, avg2-error2, avg2+error2, color='#0057E9', alpha=0.3)
plt.plot(X, avg3, color='#8931ef', linewidth=0.85)
plt.fill_between(X, avg3-error3, avg3+error3, color='#8931ef', alpha=0.3)
plt.plot(X, avg4, color='#f2ca19', linewidth=0.85)
plt.fill_between(X, avg4-error4, avg4+error4, color='#f2ca19', alpha=0.3)
plt.xticks(ticks=[5 , 15, 39, 59, 81, 101, 125, 141, 157, 173], labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.xlim(5,173)
plt.legend(("L-ValSubstituted" , "_", "L-Native", "_", "D-ValSubstituted", "_", "D-Native"))
plt.title('RMSF of Valine Substituted and Native Peptide in D and L forms')
plt.xlabel('Residue Number')
plt.ylabel('RMSF (nm)')
plt.savefig('/home/nav/Documents/LinearStats/graphs/RMSFLinLD.png', dpi=300)
plt.show()


