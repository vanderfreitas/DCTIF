import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


from matplotlib import rc

# Latex font --------------------
#rc('text', usetex=True)
font = {'family' : 'normal',
         'size'   : 10}

rc('font', **font)
params = {'legend.fontsize': 10}
plt.rcParams.update(params)
# -------------------------------


def f_logistic(x, r):
    """Discrete logistic equation with parameter r"""
    return r*x*(1-x)




mt = 1000
m = 4096

rmin = 2.9
rmax = 4.0
paramIntervals = 1000;
rs = np.linspace(rmin, rmax, paramIntervals)   # Logistic parameter


lyapunov = []


for r in rs:
	# Initial condition
	x = 0.2

	# lyapunov exponent
	sumx = 0.0

	# Transient
	for i in xrange(mt):
     		x = f_logistic(x, r)

	# Iterating map
	for i in xrange(m):
     		x = f_logistic(x, r)

		df = r*(1.0-2.0*x);
    		sumx = sumx + np.log(abs(df));


	# Lyapunov exponent
	lyapunov.append([r,sumx/m])



### PLOT ###
lyapunov = np.array(lyapunov)
fig = plt.figure(figsize=(10, 2.5), facecolor='white')
axs = []
axs.append(fig.add_subplot(1,1,1))

zeros = [0] * paramIntervals;
axs[0].plot(lyapunov[:,0], lyapunov[:,1], 'k')
axs[0].plot(lyapunov[:,0], zeros, 'k--')
axs[0].set_title('Lyapunov exponents', fontweight='bold')
axs[0].set_xlim([rmin, rmax])
axs[0].set_ylabel('$ \lambda $')
axs[0].set_xlabel('r', fontweight='bold')


trans = mtransforms.blended_transform_factory(axs[0].transData, axs[0].transAxes)
axs[0].fill_between(lyapunov[:,0], 0, 1, where=lyapunov[:,1] > 0, facecolor='red', interpolate=True, alpha=0.3, transform=trans)
axs[0].fill_between(lyapunov[:,0], 0, 1, where=lyapunov[:,1] < 0, facecolor='green', interpolate=True, alpha=0.3, transform=trans)


# 1 recuo a esquerda
# 2 comprimento em y (baixo pra cima)
# 3 comprimento em x (direita pra esquerda)
plt.tight_layout(rect=[0, 0.0, 0.96, 1])

plt.savefig("Logistic_lyapunov_exponents.pdf",dpi=100,facecolor='white')

#plt.show()


