import numpy as np
import numpy
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import *
from numpy import linalg as LA
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



def f_henon(x, y, a, b):
    """Discrete henon equation with parameter a"""
    x_n = 1.0 - a*(x**2.0) + y
    y_n = b*x

    return x_n,y_n


def henon_jacobian(x,y,a,b):
    return np.array([[-2.0*a*x, 1.0],[b,0.0]])



mt = 2000
m = 4096

amin = 0.0
amax = 1.4
paramIntervals = 1000;
aa = np.linspace(amin, amax, paramIntervals)   # Henon parameter


b = 0.3


direction = np.array([[np.cos(np.pi / 7.0)], [np.sin(np.pi / 7.0)]])



lyapunov = []



for a in aa:
	# Initial condition
	x = 0.5
	y = -0.1

	# lyapunov exponent
	sumx = 0.0

	# Transient
	for i in xrange(mt):
     		x,y = f_henon(x, y, a, b)

	# Iterating map
	for i in xrange(m):
     		x,y = f_henon(x, y, a, b)
		J = henon_jacobian(x,y,a,b)

		df = np.dot(J,direction)
    		sumx = sumx + np.log(LA.norm(df));

		direction = df/LA.norm(df)


	# Lyapunov exponent
	lyapunov.append([a,sumx/m])



### PLOT ###
lyapunov = np.array(lyapunov)
fig = plt.figure(figsize=(10, 2.5), facecolor='white')
axs = []
axs.append(fig.add_subplot(1,1,1))

zeros = [0] * paramIntervals;
axs[0].plot(lyapunov[:,0], lyapunov[:,1], 'k')
axs[0].plot(lyapunov[:,0], zeros, 'k--')
axs[0].set_title('Lyapunov exponents', fontweight='bold')
axs[0].set_xlim([amin, amax])
axs[0].set_ylabel('$ \lambda $')
axs[0].set_xlabel('a', fontweight='bold')


trans = mtransforms.blended_transform_factory(axs[0].transData, axs[0].transAxes)
axs[0].fill_between(lyapunov[:,0], 0, 1, where=lyapunov[:,1] > 0, facecolor='red', interpolate=True, alpha=0.3, transform=trans)
axs[0].fill_between(lyapunov[:,0], 0, 1, where=lyapunov[:,1] <= 0, facecolor='green', interpolate=True, alpha=0.3, transform=trans)


plt.tight_layout(rect=[0, 0.0, 0.96, 1])
plt.savefig("Henon_lyapunov_exponents.pdf",dpi=100,facecolor='white')

#plt.show()




