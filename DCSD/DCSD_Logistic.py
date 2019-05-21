# -*- coding: utf-8 -*-

from __future__ import division # pra divisao nao dar zero e nao precisar ficar colocando ponto nos numeros



import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from scipy.integrate import odeint


def rossler(vec, time, a, b, c):
    x = vec[0]
    y = vec[1]
    z = vec[2]
    
    #3 eq. dif. de rossler
    xd = -y -z
    yd = x + a*y
    zd = b + (x-c)*z
    
    return(xd, yd, zd)


def calcula_vetor_binario(x,meio):
	#recebe um vetor x e transforma ele em um vetor binario. aqui x vai de -1 ate 1. 
	#se x >= 0 vou dar valor de 1 se x <0 vou dar valor zero
	#retorna vetor binario com 0 e 1 de tamanho len(x)
	

	vet_binario = np.zeros(len(x))
	
	for i in range(len(x)):
		if x[i] >=meio:
			vet_binario[i] = 1
		else:
			vet_binario[i] = 0
	
	return(vet_binario.astype(int))



def calcula_vetor_quaternario(x):
	#recebe um vetor x e transforma ele em um vetor quaternario. aqui x vai de -1 ate 1. 
	#se -1<=x<-0.5 recebe 0, -0.5<=x<0 recebe 1, 0<=x<0.5 recebe 2, 0.5<=x<=1 recebe 3
	#retorna vetor binario com 0 e 1 de tamanho len(x)
	
	vet_quat = np.zeros(len(x))
	
	for i in range(len(x)):
		if x[i] >=-1 and x[i]<-0.5:
			vet_quat[i] = 0
		elif x[i] >=-0.5 and x[i]<0:
			vet_quat[i] = 1
		elif x[i] >=0 and x[i]<0.5:
			vet_quat[i] = 2
		elif x[i] >=0.5 and x[i]<=1:
			vet_quat[i] = 3
			
			
	return(vet_quat.astype(int))


def vetor_bi2decimal(vet_bi,n):
	#recebe vetor binario e vai pegar numero de tamanho n pra transformar pra decimal. anda de um em um
	#n = tamanho da palavra
	#retorna vetor decimal 
	tam = len(vet_bi)
	
	vet_dec = np.zeros(0)

	
	for i in range(tam-n+1) :#precisa ir ate tam-n 
		
		v = vet_bi[i:i+n]
		string_v = ''
		for j in range(len(v)): #transforma o vetor v em string
			string_v = string_v + str(v[j])
					
		vet_dec = np.append(vet_dec,int(string_v,2))
	

	
	return(vet_dec.astype(int))
	

def vetor_quat2decimal(vet_quat,n):
	#recebe vetor quaternario e vai pegar numero de tamanho n pra transformar pra decimal. anda de um em um
	#n = tamanho da palavra
	#retorna vetor decimal 
	tam = len(vet_quat)
	
	vet_dec = np.zeros(0)

	
	for i in range(tam-n+1) :#precisa ir ate tam-n 
		
		v = vet_quat[i:i+n]
		string_v = ''
		for j in range(len(v)): #transforma o vetor v em string
			string_v = string_v + str(v[j])
					
		vet_dec = np.append(vet_dec,int(string_v,4))
	

	
	return(vet_dec.astype(int))
	
	
	
def mat_adjacencia(vet_dec, n, base):
	#tenho no maximo N = 2^n vertices no grafo. n = tamanho da palavra escolhida na hora da conversao
	#vet_dec(i) se conecta com seu vizinho vet_dec(i+1)
	tam_mat = base**n
	
	mat_adj = np.zeros(shape=(tam_mat,tam_mat))

	for i in range(len(vet_dec)-1):
		j1 = vet_dec[i]#posicao do no inicial na mat_adj
		j2 = vet_dec[i+1]
		
		mat_adj[j1,j2] = 1
		mat_adj[j2,j1] = 1

		#marca esse campo da mat adjacencia como 1. matriz simetrica
		#if j1 != j2: #nao vai permitir autoconexao
		#	mat_adj[j1,j2] = 1
		#	mat_adj[j2,j1] = 1
	
	return(mat_adj.astype(int))	



def grau_total_grafo(mat):
	#calcula o grau medio do grafo com mat adj mat
	tam = len(mat[0,:])
	
	soma = 0
	for i in range(tam):
		soma = soma + sum(mat[i,:])
	
	return soma



def num_vert_conect_grafo(mat):

	tam = len(mat[0,:])
	
	num_vert = 0
	for i in range(tam):
		if sum(mat[i,:]) != 0:
			num_vert = num_vert+1
	
	
	return(num_vert)




def calcula_matriz_adj_soh_dos_nohs_conectados(mat):
	#recebe matriz de adjacencia e calcula diametro da rede.
	#primeiro tem que remover todas as linhas e colunas nulas da matriz se nao da erro na funcao
	
	tam = len(mat[0,:])
	
	indice = 0 #indice auxiliar pra matriz
	tam_novo = tam

	for i in range(tam):
		if indice < tam_novo:
			if sum(mat[indice,:]) == 0: #remove a linha e a coluna i dessa matriz. (se a linha eh toda zero, a coluna tb eh)
				mat_aux = np.delete(mat,indice,0) #deleta a linha i
				mat = np.delete(mat_aux,indice,1) #deleta a coluna i
				if tam_novo - indice > 1:	#quando a diferenca entre eles eh 1 ja ta no final			
					tam_novo = len(mat[indice,:])
			else:
				indice = indice + 1 #soh anda o indice se nao deletar nenhuma linha/coluna
		else:
			break

			

	return(mat)

def calcula_media_betw_normalizada_rede(betw_nohs):
	
	min_betw = np.min(betw_nohs)
	max_betw = np.max(betw_nohs)
	
	betw_norm = np.zeros(len(betw_nohs))
	
	if (max_betw != min_betw):
		denominador = max_betw - min_betw
	else:
		denominador = 1
		
	for i in range(len(betw_nohs)):
		betw_norm[i] = (betw_nohs[i] - min_betw)/denominador

	media_betw = np.mean(betw_norm)

	
	return(media_betw)


	
def plota_diagrama_de_bifurcacao(titulo_fig,bifurc_aa, bifurc_numero_do_no, bifurc_grau_do_no):

    
    fig = plt.figure(figsize=(12,8), dpi = 100)

    ax = fig.add_subplot(1, 1, 1)

    im = ax.scatter(bifurc_aa,bifurc_numero_do_no, c = bifurc_grau_do_no, s = 10, marker=',', cmap=plt.cm.jet ) #plt.cm.afmhot_r  #plt.cm.coolwarm #plt.cm.jet 
    ax.set_xlabel('r')
    ax.set_ylabel('Nodes')
    ax.set_title('Node`s degree influenced by different logistic parameters', fontweight='bold')
    #ax.axis('tight')
    ax.set_ylim(min(bifurc_numero_do_no),max(bifurc_numero_do_no))
    ax.set_xlim(min(bifurc_aa),max(bifurc_aa))
    # Add a colorbar
    fig.colorbar(im, ax=ax)
    
    # set the color limits - not necessary here, but good to know how.
    #im.set_clim(min(r), max(r))   
    im.set_clim(min(bifurc_grau_do_no), max(bifurc_grau_do_no))
    fig.savefig(titulo_fig, dpi=fig.dpi)
    #plt.show()
    
    return()   
	


def plota_medidas_da_rede(titulo_fig,diam,grau_medio,num_vert_conect,densidade,betw,aa):

	fig = plt.figure(figsize=(12,8), dpi = 100)

	ax = fig.add_subplot(4, 1, 1)
	ax.plot(aa,num_vert_conect,'blue')
	ax.set_xlim(min(aa),max(aa))
	ax.set_ylim(-0.1,1.1*max(num_vert_conect))
	ax.set_title('Number of nodes and network density', fontweight='bold')
	ax.set_ylabel('Number of nodes',color='blue')
	ax.tick_params(axis='y', colors='blue')
		
	ax2 = ax.twinx()
	ax2.plot(aa,densidade,'red')
	ax2.set_xlim(min(aa),max(aa))	
	ax2.set_ylim(-0.1,1.1*max(densidade))
	ax2.set_ylabel('Network density',color='red')
	ax2.tick_params(axis='y', colors='red')	

	ax = fig.add_subplot(4, 1, 2)
	ax.plot(aa,diam,'black')
	ax.set_ylabel('Diameter')
	ax.set_xlim(min(aa),max(aa))
	ax.set_ylim(-0.1,1.1*max(diam))
	ax.set_title('Diameter',fontweight='bold')


	ax = fig.add_subplot(4, 1, 3)
	ax.plot(aa,grau_medio,'black')
	ax.set_ylabel('<k>')
	ax.set_xlim(min(aa),max(aa))
	ax.set_ylim(-0.1,1.1*max(grau_medio))
	ax.set_title('Average degree',fontweight='bold')


	ax = fig.add_subplot(4, 1, 4)
	ax.plot(aa,betw,'black')
	ax.set_ylabel('Betweenness')
	ax.set_xlim(min(aa),max(aa))
	ax.set_ylim(-0.1,1.1*max(betw))
	ax.set_title('Average betweenness',fontweight='bold')
	ax.set_xlabel('r')

	fig.tight_layout()
	fig.savefig(titulo_fig, dpi=fig.dpi)
	#plt.show()



	return()

def salva_todas_as_medidas(diam,grau_medio,num_vert_conect,densidade,betw,bifurc_rr, bifurc_numero_do_no, bifurc_grau_do_no,rr):	

	np.savetxt('Logistic/txt/diametro.txt',diam,fmt='%1.2f')
	np.savetxt('Logistic/txt/grau_medio.txt',grau_medio,fmt='%1.4f')
	np.savetxt('Logistic/txt/num_vert_conect.txt',num_vert_conect,fmt='%1.2f')
	np.savetxt('Logistic/txt/densidade.txt',densidade,fmt='%1.5f')
	np.savetxt('Logistic/txt/betw.txt',betw,fmt='%1.5f')
	np.savetxt('Logistic/txt/rr.txt',rr,fmt='%1.8f')
	np.savetxt('Logistic/txt/bifurc_rr.txt',bifurc_rr,fmt='%1.8f')
	np.savetxt('Logistic/txt/bifurc_numero_do_no.txt',bifurc_numero_do_no,fmt='%1.2f')
	np.savetxt('Logistic/txt/bifurc_grau_do_no.txt',bifurc_grau_do_no,fmt='%1.2f')


	return()

	
	
	
	
	
	
	
#------------------------------------------------------------------------------
#COMECOU O PROGRAMA
 

nIt = 4096 
nn = 10
base = 2 #vai ter no meximo 2**nn vertices o grafo

meio = 0.5 #meio da serie temporal. utilizada pra calc o vetor binario
#constantes de rossler
rr = np.linspace(2.9,4.0,1000)

#metricas das redes
diam = np.zeros(len(rr))
grau_medio = np.zeros(len(rr))
num_vert_conect = np.zeros(len(rr))
densidade = np.zeros(len(rr))
betw = np.zeros(len(rr))

#para diagrama de bifurcacao
tam = (base**nn)*len(rr)

bifurc_numero_do_no = np.zeros(tam)
bifurc_aa = np.zeros(tam)
bifurc_grau_do_no = np.zeros(tam)
cont = 0 #contador auxiliar pra ir guardando as coisas daqui

#carrega a serie temporal do mapa logistico
mat = np.loadtxt('../timeseries_generation/logistic.txt')
indice1 = 0
indice2 = nIt
for i in range(len(rr)):
	r = rr[i]
	print 'r=', r

	vet_x = mat[indice1:indice2,1]
	indice1 = indice2
	indice2 = indice2 + nIt

	vetor_binario = calcula_vetor_binario(vet_x,meio)
	
	vet_decimal = vetor_bi2decimal(vetor_binario,nn)
	
	mat_adj = mat_adjacencia(vet_decimal,nn,base)#com autoconexao
	
	mat_adj_nova = calcula_matriz_adj_soh_dos_nohs_conectados(mat_adj)
	
	#np.savetxt('mat_adj.txt',mat_adj)
	#np.savetxt('mat_adj_nova.txt',mat_adj_nova)
	
	num_vert_conect[i] = num_vert_conect_grafo(mat_adj_nova)
	
	grafo = nx.from_numpy_matrix(mat_adj_nova)
	
	'''
	grau_total = grau_total_grafo(mat_adj_nova)	
	
	if num_vert_conect[i] == 0:
		grau_medio[i] = 0
	#elif num_vert_conect[i] == 1 and grau_total == 1: #tem soh 1 vertice com auto conexao
		
	else:	
		grau_medio[i] = grau_total/num_vert_conect[i] 
	'''
	
	aux_grau = nx.degree(grafo)
	#vetor estranho, tenho que fazer o for pra colocar num vetor normal
	grau_nohs = np.zeros(len(aux_grau))
						 
	for jj in range(len(aux_grau)):
		grau_nohs[jj] = aux_grau[jj]	
	
	grau_medio[i] = np.mean(grau_nohs)
	
	if (num_vert_conect[i] > 1) : #te pelo menos um par de vertices conectados

		densidade[i] = nx.density(grafo)
		diam[i] = nx.diameter(grafo)
		#NN = num_vert_conect[i]
		#densidade[i] = nx.number_of_edges(grafo)/((NN*(NN-1))/2)

	else: #tem apenas um noh na rede, quando eh periodica com periodo 1
		densidade[i] = 0
		diam[i] = 0
	
		
		
	#vetor estranho, tenho que somar item a item pra pegar a media
	aux_1 = nx.betweenness_centrality(grafo)
	betw_nohs = np.zeros(len(aux_1))
	for jj in range(len(aux_1)):
		betw_nohs[jj] = aux_1[jj]
		
	
	#vai calcular agora a media da betw normalizada da rede
	betw[i] = calcula_media_betw_normalizada_rede(betw_nohs)
	
	
	
	#calcula as coisas pra plotar o diagrama de bifurcacao. 
	for jj in range(base**nn):
		bifurc_aa[cont] = r
		bifurc_numero_do_no[cont] = jj
		bifurc_grau_do_no[cont] = np.sum(mat_adj[:,jj])
		cont = cont+1
	
	
#end for i #len(aa)



titulo_fig = 'Logistic/figs/diagrama_bifurc.pdf' 
plota_diagrama_de_bifurcacao(titulo_fig,bifurc_aa, bifurc_numero_do_no, bifurc_grau_do_no)

titulo_fig = 'Logistic/figs/medidas_da_rede.pdf' 
plota_medidas_da_rede(titulo_fig,diam,grau_medio,num_vert_conect,densidade,betw,rr)

salva_todas_as_medidas(diam,grau_medio,num_vert_conect,densidade,betw,bifurc_aa, bifurc_numero_do_no, bifurc_grau_do_no,rr)	

