# -*- coding: utf-8 -*-

from __future__ import division # pra divisao nao dar zero e nao precisar ficar colocando ponto nos numeros



import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
matplotlib.rcParams.update({'font.size': 12})

def plot_network_measures(fig_title,diam,avg_degree,num_vert_conect,density,betw,aa):

	fig = plt.figure(figsize=(12,8), dpi = 100)

	ax = fig.add_subplot(4, 1, 1)
	ax.plot(aa,num_vert_conect,'blue')
	ax.set_xlim(min(aa),max(aa))
	ax.set_ylim(-0.1,1.1*max(num_vert_conect))
	ax.set_title('Number of nodes and network density', fontweight='bold')
	ax.set_ylabel('Number of nodes',color='blue')
	ax.tick_params(axis='y', colors='blue')
		
	ax2 = ax.twinx()
	ax2.plot(aa,density,'red')
	ax2.set_xlim(min(aa),max(aa))	
	ax2.set_ylim(-0.1,1.1*max(density))
	ax2.set_ylabel('Network density',color='red')
	ax2.tick_params(axis='y', colors='red')	

	ax = fig.add_subplot(4, 1, 2)
	ax.plot(aa,diam,'black')
	ax.set_ylabel('Diameter')
	ax.set_xlim(min(aa),max(aa))
	ax.set_ylim(-0.1,1.1*max(diam))
	ax.set_title('Diameter',fontweight='bold')


	ax = fig.add_subplot(4, 1, 3)
	ax.plot(aa,avg_degree,'black')
	ax.set_ylabel('<k>')
	ax.set_xlim(min(aa),max(aa))
	ax.set_ylim(-0.1,1.1*max(avg_degree))
	ax.set_title('Average degree',fontweight='bold')


	ax = fig.add_subplot(4, 1, 4)
	ax.plot(aa,betw,'black')
	ax.set_ylabel('Betweenness')
	ax.set_xlim(min(aa),max(aa))
	ax.set_ylim(-0.1,1.1*max(betw))
	ax.set_title('Average betweenness',fontweight='bold')
	ax.set_xlabel('a')

	ax.set_xlabel(r'$a$', fontweight='bold')

	fig.tight_layout()
	fig.savefig(fig_title, dpi=fig.dpi)
	#plt.show()


	return()
	
	
fig_title = 'figs/network_measures.pdf' 
nome='_meio_0.5'
diam = np.loadtxt('txt/diametro'+nome+'.txt') 
avg_degree =  np.loadtxt('txt/grau_medio'+nome+'.txt')
num_vert_conect = np.loadtxt('txt/num_vert_conect'+nome+'.txt')
density = np.loadtxt('txt/densidade'+nome+'.txt')
betw = np.loadtxt('txt/betw'+nome+'.txt')
rr = np.loadtxt('txt/aa'+nome+'.txt')


plot_network_measures(fig_title,diam,avg_degree,num_vert_conect,density,betw,rr)










