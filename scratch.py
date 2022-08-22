import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from itertools import cycle, islice
my_colors = list(islice(cycle(['#BBCCEE', '#FFCCCC', '#CCEEFF', '#EEEEBB']), None, 4))
import matplotlib.ticker as tkr

def tickerd(x, pos):
    s = '{}'.format(x/1000000)
    return s

# Load data from CSV
dat1 = np.genfromtxt('/home/nav/stats/Hydrophobic/hbnum_01.xvg')
dat2 = np.genfromtxt('/home/nav/stats/Hydrophobic/hbnum_02.xvg')
dat3 = np.genfromtxt('/home/nav/stats/Hydrophobic/hbnum_03.xvg')
dat4 = np.genfromtxt('/home/nav/stats/Hydrophobic/hbnum_04.xvg')
dat5 = np.genfromtxt('/home/nav/stats/Hydrophobic/hbnum_05.xvg')
dat6 = np.genfromtxt('/home/nav/stats/Native/hbnum_01.xvg')
dat7 = np.genfromtxt('/home/nav/stats/Native/hbnum_02.xvg')
dat8 = np.genfromtxt('/home/nav/stats/Native/hbnum_03.xvg')
dat9 = np.genfromtxt('/home/nav/stats/Native/hbnum_04.xvg')
dat10 = np.genfromtxt('/home/nav/stats/Native/hbnum_05.xvg')
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


# compute the standard error of the mean
avg1 = np.mean([Ym1, Ym2, Ym3, Ym4, Ym5 ], axis=0)
error1 = stats.sem((Ym1, Ym2, Ym3, Ym4, Ym5 ), axis=0, ddof=1)
avg2 = np.mean([Ym6, Ym7, Ym8, Ym9, Ym10], axis=0)
error2 = stats.sem((Ym6, Ym7, Ym8, Ym9, Ym10), axis=0, ddof=1)
# Create the plot

plt.plot(X, avg1, color=my_colors[0], linewidth=0.85)
plt.fill_between(X, avg1-error1, avg1+error1, color='#EF476F', alpha=0.3)
plt.plot(X, avg2, color=my_colors[1], linewidth=0.85)
plt.fill_between(X, avg2-error2, avg2+error2, color='#118AB2', alpha=0.3)
plt.xticks(rotation=90)
plt.legend(('Native', '_','Valine Substituted'))
#plt.legend(('WT', 'N642H', 'Y665F'))
plt.title('Evolution of Hydrogen Bonds over Time')
plt.xlabel('Time ms')
plt.ylabel('Internal Hydrogen Bonds')
yfmt = tkr.FuncFormatter(tickerd)
plt.xlim([0, 5000000])
plt.xticks([0, 500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000, 5000000])
plt.gca().xaxis.set_major_formatter(yfmt)
plt.savefig('Hbond.png', dpi=300)
plt.show()


