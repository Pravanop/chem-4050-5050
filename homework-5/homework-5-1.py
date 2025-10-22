import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import expon
def psi_2p_z(x, y, z, a0=1.0):
    
    r = np.sqrt(x*x + y*y + z*z)
    cos_theta = z / r
    pref = 1.0 / (4.0 * np.sqrt(2.0*np.pi) * (a0**1.5))
    return pref * (r/a0) * cos_theta * np.exp(-r/(2.0*a0))

def overlap_integrand(x, y, z, R, a0=1.0):
    
    return psi_2p_z(x, y, z + 0.5*R, a0=a0) * psi_2p_z(x, y, z - 0.5*R, a0=a0)

def monte_carlo_overlap_uniform(R=2.0, L=20.0, N=10, a0=1.0
                                ):
	
    np.random.seed(42)
    pts = np.random.uniform(-L, L, size=(N, 3))
    vals = overlap_integrand(pts[:,0], pts[:,1], pts[:,2], R, a0=a0)
    V = (2.0*L)**3
    mean = np.mean(vals)
    est = V * mean
    return est

def monte_carlo_overlap_importance_exp(R=2.0, N=100_000, a0=1.0):
    
    x = expon.rvs(size=N, scale=1)
    y = expon.rvs(size=N, scale=1)
    z = expon.rvs(size=N, scale=1)
    num = psi_2p_z(x, y, z + 0.5*R, a0) * psi_2p_z(x, y, z - 0.5*R, a0)
    g = expon.pdf(x) * expon.pdf(y) * expon.pdf(z)
    w = num / g
    est = np.mean(w)
    return est


    