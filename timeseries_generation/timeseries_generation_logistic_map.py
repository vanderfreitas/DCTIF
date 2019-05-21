# -*- coding: utf-8 -*-

from __future__ import division # To avoid zero division

import numpy as np
import matplotlib.pyplot as plt
from pylab import *


bifurcation = []
amin = 2.9
amax = 4.0
paramIntervals = 1000;
aa = np.linspace(amin, amax, paramIntervals)   # Henon parameter



nTrans = 1000
nIt = 4096

rr = np.linspace(2.9,4.0,1000)

x0 = 0.7

x = np.zeros(nIt)

# Keep r values in the first column and x in the second.
mat = np.zeros(shape = (nIt*len(rr),2))

#To keep the values of r
aux_r = np.zeros(nIt)

indice1 = 0
indice2 = nIt
#para todos os valores de r
for i in range(len(rr)):
	
	r = rr[i]

	x0 = 0.7
	#Discarding transient
	for k in range(nTrans):
		x0=r*x0-r*x0*x0

	x[0] = x0
	aux_r[0] = r

	#Logistic map timeseries
	for k in range(1,nIt):
		x[k] = r*x[k-1]-r*x[k-1]*x[k-1]
		aux_r[k] = r

		bifurcation.append([r,x[k]])


	mat[indice1:indice2,0] = aux_r
	mat[indice1:indice2,1] = x
	indice1 = indice2
	indice2 = indice2+nIt

np.savetxt('logistic.txt',mat,delimiter='\t')




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
axs[0].set_xlabel('r', fontweight='bold')


#x axis for the bottom image
plt.sca(axs[0]);
axs[0].set_xlim([amin, amax])
plt.xticks(tick_original, tick_lbls)

plt.tight_layout(rect=[0, 0.0, 0.93, 1])
savefig("Logistic_bifurcation_diagram.png",dpi=200,facecolor='white')


