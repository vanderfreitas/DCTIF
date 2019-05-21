import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import *
import numpy
from itertools import groupby
from mpl_toolkits.axes_grid1 import make_axes_locatable




#### PARAMETERS #####
N = 1024;  #Number of subintervals (number of network nodes)
m = 4096;  #Size of the timeseries for each parameter

paramsSize = 1000
rmin = 2.9
rmax = 4.0

parameter_name = 'r'

params = numpy.linspace(rmin, rmax, paramsSize)

system_name = 'Logistic'



#opening files
#parameter = np.genfromtxt ('bifurc_aa.txt', delimiter="\n")
#nodeNumber = np.genfromtxt ('bifurc_numero_do_no.txt', delimiter="\n")
nodeDegree = np.genfromtxt ('txt/bifurc_grau_do_no.txt', delimiter="\n")



# Initiate matrix
colored_nodes_degree = np.zeros((N,len(params)))

#for j in xrange(len(nodeDegree)):
#	print nodeDegree[


# Filling the matrix with zeros
#for j in xrange(N):
#	colored_nodes_degree[nodes_degree[j,0]][nodes_degree[j,1]] = nodes_degree[j,2]



# Populating the colored graph with the networks degrees
# This is a matrix, with each position (row,col) represents a point (Node,parameter).
# The value of each position is the node degree for a given parameter

col = -1
for j in xrange(len(nodeDegree)):
	if j%1024 == 0:
		col = col+1
	colored_nodes_degree[N-1-j % 1024][col] = nodeDegree[j]
	


############################ COLORED GRAPH WITH NODES DEGREE #########################################################
tick_locs = [];
tick_lbls = [];
tick_original = [];

for i in xrange(N):
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


# Imagem colorida
p = axs[0].imshow(colored_nodes_degree, interpolation='none', aspect='auto', extent = [0,paramsSize,0,N], cmap=plt.get_cmap('gist_stern'))

titlee = str(r'Nodes degree influenced by different ') + system_name + str(r' parameters')
axs[0].set_title(titlee, fontweight='bold')
axs[0].set_ylabel('Nodes')
axs[0].set_xlabel(parameter_name, fontweight='bold')


axs[0].set_xlabel(r'$r$')

# placing the colorbar side by side with the graph
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)

cb = fig.colorbar(p, ax=axs[0], cax=cax, orientation='vertical')

#Eixo x da imagem superior
plt.sca(axs[0]);
plt.xticks(tick_locs, tick_lbls)


fig.tight_layout()


#Salva Figura
savefig("Logistic_nodes_degree.png",dpi=300,facecolor='white')


############################################################################################################
