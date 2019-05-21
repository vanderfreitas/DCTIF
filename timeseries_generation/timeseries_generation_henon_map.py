import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import numpy


# Henon map
def f_henon(x, y, a, b):
    x_n = 1.0 - a*(x**2.0) + y
    y_n = b*x

    return x_n,y_n



mt = 2000
m = 4096

amin = 0.0
amax = 1.4
paramIntervals = 1000;
aa = np.linspace(amin, amax, paramIntervals)   # Henon parameter




f = open("henon.txt", "w")

b = 0.3


bifurcation = []

series_max = 1.42857142857 + 0.001
series_min = -1.28367328352 - 0.001
den = series_max - series_min

for a in aa:
	# Initial condition
	x = 0.5
	y = -0.1

	# Transient
	for i in xrange(mt):
     		x,y = f_henon(x, y, a, b)

	

	for i in xrange(m):
		x,y = f_henon(x, y, a, b)

		value = (x - series_min) / den

		f.write(str(a) + str('\t') + str(i+mt) + str('\t') + str(value) + str('\n'))

		bifurcation.append([a,x])

f.close()




################ Bifurcation #########################
tick_locs = [];
tick_lbls = [];
tick_original = [];

for i in xrange(paramIntervals):
	if i % 128 == 0:
		tick_locs.append(i);
		tick_lbls.append("{:1.2f}".format(aa[i]));
		tick_original.append(aa[i]);


fig = plt.figure(figsize=(10, 3.5), facecolor='white')
axs = []
axs.append(fig.add_subplot(1,1,1))

bifurcation = np.array(bifurcation)
axs[0].plot(bifurcation[:,0], bifurcation[:,1],'k.',markersize=0.3)
axs[0].set_title('Bifurcation diagram', fontweight='bold')
axs[0].set_ylabel('Values')
axs[0].set_xlabel('a', fontweight='bold')


#x axis for the bottom image
plt.sca(axs[0]);
axs[0].set_xlim([amin, amax])
plt.xticks(tick_original, tick_lbls)


plt.tight_layout(rect=[0, 0.0, 0.92, 1])
savefig("Henon_bifurcation_diagram.png",dpi=200,facecolor='white')
