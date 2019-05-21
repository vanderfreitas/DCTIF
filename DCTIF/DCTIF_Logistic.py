import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import *
import numpy
from igraph import *
from itertools import groupby
from mpl_toolkits.axes_grid1 import make_axes_locatable



# Converts values from the series into integers in the interval [0,N]
def integralFunction(num, N):
	result = int();
	if num*N == 0:
		result = int(1)
	elif round(num*N) - num*N >= 0.0:
		result = int(round(num*N));
	else: 
		result = int(round(num*N) + 1.0);
	return result



def numVisitedVertices(g):
	num = 0;
	for v in g.vs:
		if v.degree() > 0:
			num = num + 1;
	return num;



def averagePathLenght(g):
	avgNode = []

	lista = graphAux.shortest_paths_dijkstra(source=None,target=None, weights=None, mode=ALL);
	for i in lista:
		avgNode.append(mean(i))
	
	return mean(avgNode)	


def addEdge(g, v1, v2):
	# caso esta aresta ainda nao exista
	if g.get_eid(v1, v2, directed=False, error=False) == -1:
		g.add_edge(v1,v2)






# Opening timeseries file
csv_file = np.genfromtxt ('../timeseries_generation/logistic.txt', delimiter="\t")



N = 1024;  #Number of subintervals (number of network nodes)
m = 4096;  #Size of the timeseries for each parameter

paramsSize = 1000
rmin = 2.9
rmax = 4.0
params = numpy.linspace(rmin, rmax, paramsSize)   # Parameters used in the timeseries

g = Graph(); # Network
g.add_vertices(N) # Adding nodes. The number of nodes equals to the number of subintervals
for i in range(N):
	g.vs[i]["label"] = i+1


# Time series
colored_nodes_degree = np.zeros((N,len(params)))
bifurcation = []
lyapunov = []

# Statistics
histVisVert = []
avgDegree = []
diameter = []
betweenness = []
density = []
nodes_degree = []


histVisVert_std = []
avgDegree_std = []
diameter_std = []
betweenness_std = []




# Denominator for the calculus of network density
density_denominator = float((N*(N-1))/2.0)



for indexr in xrange(len(params)):
	param = params[indexr]

	print ('param=', param)

	x = csv_file[indexr*m,1]


	index = int(integralFunction(x, N))


	# Creating the network from the timeseries
	for i in xrange(m):
	     	x = csv_file[indexr*m+i,1]
	     	indexAnterior = index		
	     	index = int(integralFunction(x, N))

		addEdge(g, indexAnterior-1, index-1)
		bifurcation.append([param, x])

	

	##### Auxiliar graph, with the visited nodes only ######
	### The network has usualy less than N nodes ###
	graphAux = Graph();
	numVertices = 0;
	labels = []

	for i in g.vs:
		if i.degree() > 0:
			numVertices = numVertices + 1;
			labels.append(i["label"])

	graphAux.add_vertices(numVertices);
	for i in xrange(numVertices):
		graphAux.vs[i]["label"] = labels[i];

	for i in xrange(len(g.es)):
		src = g.vs[g.es[i].source]["label"]
		dst = g.vs[g.es[i].target]["label"]

		src_ = 0;
		dst_ = 0;
		for j in graphAux.vs:
			if j["label"] == src:
				src_ = j.index;				
		 	if j["label"] == dst:
				dst_ = j.index;
		graphAux.add_edge(src_,dst_);


	#layout = graphAux.layout("rt")
	#plot(graphAux, bbox = (300, 300), margin = 40, layout = layout);


	############### NETWORK STATISTICS ##############################
	# Histograma de vertices visitados
	histVisVert.append([param,len(graphAux.vs)])

	#Grau medio
	avgDegree.append([param,mean(graphAux.degree())])

	#Betweenness medio
	betAux = graphAux.betweenness(vertices=None, directed=False, cutoff=None)
	min_bet = min(betAux)
	max_bet = max(betAux)
	for iii in xrange(len(betAux)):
		if (max_bet - min_bet) > 0:
			betAux[iii] = (betAux[iii] - min_bet) / (max_bet - min_bet)
		else:
			betAux[iii] = 0	
	#betweenness.append([param,mean(graphAux.betweenness(vertices=None, directed=False, cutoff=None))])
	betweenness.append([param,mean(betAux)])

	#Coeficiente de clusterizacao medio
	#clusteringCoeff.append([param,graphAux.transitivity_undirected(mode="nan")])

	# Diameter
	diameter.append([param,graphAux.diameter()])

	#Caminho medio
	#avg_path.append([param,averagePathLenght(graphAux)])


	#Densidade grafo
	nn = len(graphAux.vs)
	density_denominator = float((nn*(nn-1))/2.0)
	if density_denominator == 0.0:
		density.append([param, 0.0])
	else:
		density.append([param, float(len(graphAux.es))/density_denominator])


	#Numero arestas
	#numeroArestas.append([param, len(graphAux.es)])




	# Verifica qual o maior grau da rede
	#for j in xrange(numVertices):
	#	if maiorG < graphAux.degree()[j]:
	#		maiorG = graphAux.degree()[j]


	#janelaPeriodica = True
	#for j in xrange(len(graphAux.degree())):
	#	if graphAux.degree()[j] != maiorG:
	#		janelaPeriodica = False

	#if janelaPeriodica == True:
	#	maiorGrau.append([param,maiorG])


	#layout = graphAux.layout("rt")
	#plot(graphAux, bbox = (250, 250), margin = 20, layout = layout);


	# Populating the colored graph with the networks degrees
	# This is a matrix, with each position (row,col) represents a point (Node,parameter).
	# The value of each position is the node degree for a given parameter
	for j in xrange(N):
		colored_nodes_degree[N-j-1][indexr] = g.degree()[j]
		nodes_degree.append([N-j-1, indexr, g.degree()[j]])



	#Deletar todas as arestas da rede
	num_edges = len(g.es)
	for i in range(num_edges):
		g.delete_edges(0)





# Normalizing the betweenness
betweenness = numpy.array(betweenness)
'''min_b = min(betweenness[:,1])
max_b = max(betweenness[:,1])

for i in xrange(len(betweenness)):
	betweenness[i,1] = (betweenness[i,1] - min_b) / (max_b - min_b)'''




############################## EXPORTING DATA ################################
bifurcation = numpy.array(bifurcation)
histVisVert = numpy.array(histVisVert)
density = numpy.array(density)
avgDegree = numpy.array(avgDegree)
diameter = numpy.array(diameter)



f_bifurc = open("Logistic/bifurcation.txt", "w")
f_histvisvert = open("Logistic/histvisvert.txt", "w")
f_graphdensity = open("Logistic/graphdensity.txt", "w")
f_betweenness = open("Logistic/betweenness.txt", "w")
f_avgDegree = open("Logistic/avgDegree.txt", "w")
f_diameter = open("Logistic/diameter.txt", "w")
f_nodes_degree = open("Logistic/nodes_degree.txt", "w")

for i in xrange(len(bifurcation)):
	f_bifurc.write(str(bifurcation[i,0]) + str('\t') + str(bifurcation[i,1]) + str('\n'))

for i in xrange(len(histVisVert)):
	f_histvisvert.write(str(histVisVert[i,0]) + str('\t') + str(histVisVert[i,1]) + str('\n'))
	f_graphdensity.write(str(density[i,0]) + str('\t') + str(density[i,1]) + str('\n'))
	f_betweenness.write(str(betweenness[i,0]) + str('\t') + str(betweenness[i,1]) + str('\n'))
	f_avgDegree.write(str(avgDegree[i,0]) + str('\t') + str(avgDegree[i,1]) + str('\n'))
	f_diameter.write(str(diameter[i,0]) + str('\t') + str(diameter[i,1]) + str('\n'))

for i in xrange(len(nodes_degree)):
	f_nodes_degree.write(str(nodes_degree[i][0]) + str('\t') + str(nodes_degree[i][1]) + str('\t') + str(nodes_degree[i][2]) + str('\n'))


f_bifurc.close()
f_histvisvert.close()
f_graphdensity.close()
f_betweenness.close()
f_avgDegree.close()
f_diameter.close()
f_nodes_degree.close()





