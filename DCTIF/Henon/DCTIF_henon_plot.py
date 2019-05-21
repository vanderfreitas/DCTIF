import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import *
import numpy
from igraph import *
from itertools import groupby
from mpl_toolkits.axes_grid1 import make_axes_locatable



#### PARAMETERS #####
N = 1000;  #Number of subintervals (number of network nodes)
m = 4096;  #Size of the timeseries for each parameter

paramsSize = 1000
rmin = 0.0
rmax = 1.4

parameter_name = 'a'

params = numpy.linspace(rmin, rmax, paramsSize)

system_name = 'Henon'







#opening files
bifurcation = np.genfromtxt ('bifurcation.txt', delimiter="	")
histVisVert = np.genfromtxt ('histvisvert.txt', delimiter="	")
density = np.genfromtxt ('graphdensity.txt', delimiter="	")
betweenness = np.genfromtxt ('betweenness.txt', delimiter="	")
avgDegree = np.genfromtxt ('avgDegree.txt', delimiter="	")
diameter = np.genfromtxt ('diameter.txt', delimiter="	")
nodes_degree = np.genfromtxt ('nodes_degree.txt', delimiter="	")




colored_nodes_degree = np.zeros((N,len(params)))



for j in xrange(len(nodes_degree)):
	colored_nodes_degree[nodes_degree[j,0]][nodes_degree[j,1]] = nodes_degree[j,2]




minAux = 0
maxAux = 0


############################ COLORED GRAPH WITH NODES DEGREE #########################################################
tick_locs = [];
tick_lbls = [];
tick_original = [];

for i in xrange(paramsSize):
	if i % 128 == 0:
		tick_locs.append(i);
		tick_lbls.append("{:1.2f}".format(params[i]));
		tick_original.append(params[i]);

#Ultimos dois valores
tick_locs.append(paramsSize-1);
tick_lbls.append(str(params[paramsSize-1]));
tick_original.append(params[paramsSize-1]);


fig = plt.figure(figsize=(10, 3.5), facecolor='white')
axs = []
axs.append(fig.add_subplot(1,1,1))
#axs.append(fig.add_subplot(2,1,2))


# Imagem colorida
p = axs[0].imshow(colored_nodes_degree, interpolation='none', aspect='auto', extent = [0,paramsSize,0,N], cmap=plt.get_cmap('gist_stern'))

titlee = str(r'Nodes degree influenced by different ') + system_name + str(r' parameters')
axs[0].set_title(titlee, fontweight='bold')
axs[0].set_ylabel('Nodes')
axs[0].set_xlabel(parameter_name, fontweight='bold')

# placing the colorbar side by side with the graph
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)

cb = fig.colorbar(p, ax=axs[0], cax=cax, orientation='vertical')

#Eixo x da imagem superior
plt.sca(axs[0]);
plt.xticks(tick_locs, tick_lbls)

plt.tight_layout()

#Salva Figura
savefig("Henon_nodes_degree.png",dpi=300,facecolor='white')






################ Bifurcation #########################
fig = plt.figure(figsize=(10, 3), facecolor='white')
axs = []
axs.append(fig.add_subplot(1,1,1))

bifurcation = numpy.array(bifurcation)
axs[0].plot(bifurcation[::10,0], bifurcation[::10,1],'-',markersize=0.3)
axs[0].set_title('Bifurcation diagram', fontweight='bold')
axs[0].set_ylabel('Values')
axs[0].set_xlabel(parameter_name, fontweight='bold')


#x axis for the bottom image
plt.sca(axs[0]);
axs[0].set_xlim([rmin, rmax])
plt.xticks(tick_original, tick_lbls)

plt.tight_layout()
savefig("Henon_bifurcation_diagram.pdf",dpi=100,facecolor='white')








################################ STATISTICS ######################################################


f, axarr = plt.subplots(4, figsize=(13, 7), facecolor='white', sharex=True)
axarr[0].set_title('Number of nodes and network density', fontweight='bold')
axarr[0].plot(histVisVert[:,0], histVisVert[:,1], 'b')
axarr[0].set_ylabel('Number of nodes', color='b')
for tl in axarr[0].get_yticklabels():
    tl.set_color('b')

ax2 = axarr[0].twinx()
ax2.plot(density[:,0], density[:,1], 'r')
ax2.set_ylabel('network density', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')

axarr[0].set_xlim([rmin, rmax])

minAux = 0.9*min(histVisVert[:,1])
maxAux = 1.1*max(histVisVert[:,1])
if minAux == 0.0:
	minAux = -0.1

axarr[0].set_ylim([minAux, maxAux])

minAux = 0.9*min(density[:,1])
maxAux = 1.1*max(density[:,1])
if minAux == 0.0:
	minAux = -0.1

ax2.set_ylim([minAux, maxAux])









minAux = 0.9*min(diameter[:,1])
maxAux = 1.1*max(diameter[:,1])
if minAux == 0.0:
	minAux = -0.1

axarr[1].plot(diameter[:,0], diameter[:,1],'k')
axarr[1].set_title('Diameter', fontweight='bold')
axarr[1].set_ylabel('diameter')
axarr[1].set_xlim([rmin, rmax])
axarr[1].set_ylim([minAux, maxAux])






minAux = 0.9*min(avgDegree[:,1])
maxAux = 1.1*max(avgDegree[:,1])
if minAux == 0.0:
	minAux = -0.1

axarr[2].plot(avgDegree[:,0], avgDegree[:,1],'k')
axarr[2].set_title('Average degree', fontweight='bold')
axarr[2].set_ylabel('<k>')
axarr[2].set_xlim([rmin, rmax])
axarr[2].set_ylim([minAux, maxAux])




minAux = 0.9*min(betweenness[:,1])
maxAux = 1.1*max(betweenness[:,1])
if minAux == 0.0:
	minAux = -0.1

axarr[3].plot(betweenness[:,0], betweenness[:,1],'k')
axarr[3].set_title('Average betweenness', fontweight='bold')
axarr[3].set_ylabel('betweenness')
axarr[3].set_xlim([rmin, rmax])
axarr[3].set_xlabel(parameter_name, fontweight='bold')
axarr[3].set_ylim([minAux, maxAux])



plt.tight_layout()
savefig("Henon_statistics.pdf",dpi=100,facecolor='white')

#plt.show()





############################################################################################################
