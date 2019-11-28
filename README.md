# COMPLEX NETWORKS APPROACH FOR DYNAMICAL CHARACTERIZATION OF NONLINEAR SYSTEMS

**Abstract**: Bifurcation diagrams and Lyapunov exponents are the main tools for dynamical systems characterization. However, they are often computationally expensive and complex to calculate. We present two approaches for dynamical characterization of nonlinear systems via generation of an undirected complex network that is built from their time series. Periodic windows and chaos can be detected by analyzing network statistics like average degree, density and betweenness centrality. Results are assessed in two discrete time nonlinear maps.


**Collaborators**: Vander L. S. Freitas, Juliana C. Lacerda and Elbert E. N. Macau.


We propose two algorithms for dynamical characterization of nonlinear systems using complex networks:
- Dynamical Characterization with the Top Integral Function (DCTIF)
- Dynamical Characterization with Symbolic Dynamics (DCSD) 



Dependencies (tested on python 2.7):
* matplotlib
* igraph
* networkx



In order to run all the simulations from our paper, open a terminal and type the following:

```
bash reproduce_all_results.sh
```


If you use this code, please cite the paper:

**FREITAS, V. L. S.; LACERDA, J. C.; MACAU, E. E. N. Complex networks approach for dynamical characterization of nonlinear systems. International Journal of Bifurcation and Chaos, v. 29, n. 13, p. 1950188, 2019**
