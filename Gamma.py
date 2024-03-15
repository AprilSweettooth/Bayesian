import numpy as np
from scipy import integrate
import scipy.special as sc
import matplotlib.pyplot as plt

def Gaussian(x,s,sig):
    return 1/(2*np.pi*sig**2)**0.5 * np.exp(-(x-s)**2/(2*sig**2))

def Dirichlet(x,a):
    return x**a[0]*(1-x)**a[1]/(sc.beta(a[0], a[1])+0.00001)
# x = np.linspace(0,1,100)
# plt.plot(x,Dirichlet(x,np.array([2.99289485, 3.45774506])))
# plt.show()
def Gamma0(dist,density):
    G = np.zeros((4,4))
    func0 = lambda s1,s2,p : Gaussian(s1,dist[0][0],dist[0][1])*Gaussian(s2,dist[1][0],dist[1][1])*Dirichlet(p,dist[2])
    n = integrate.nquad(func0, [[-1,1], [-1,1], [0,1]], full_output=True)[0]
    for i in range(4):
        for j in range(4):
            func = lambda s1,s2,p : Gaussian(s1,dist[0][0],dist[0][1])*Gaussian(s2,dist[1][0],dist[1][1])*Dirichlet(p,dist[2])*density(s1,s2,p)[i,j]
            G[i,j] = integrate.nquad(func, [[-1,1], [-1,1], [0,1]], full_output=True)[0]/n
    return G

def Gamma_s1(dist,density):
    G = np.zeros((4,4))
    func0 = lambda s1,s2,p : Gaussian(s1,dist[0][0],dist[0][1])*Gaussian(s2,dist[1][0],dist[1][1])*Dirichlet(p,dist[2])
    n = integrate.nquad(func0, [[-1,1], [-1,1], [0,1]], full_output=True)[0]
    for i in range(4):
        for j in range(4):
            func = lambda s1,s2,p : s1*Gaussian(s1,dist[0][0],dist[0][1])*Gaussian(s2,dist[1][0],dist[1][1])*Dirichlet(p,dist[2])*density(s1,s2,p)[i,j]
            G[i,j] = integrate.nquad(func, [[-1,1], [-1,1], [0,1]], full_output=True)[0]/n
    return G

def Gamma_s2(dist,density):
    G = np.zeros((4,4))
    func0 = lambda s1,s2,p : Gaussian(s1,dist[0][0],dist[0][1])*Gaussian(s2,dist[1][0],dist[1][1])*Dirichlet(p,dist[2])
    n = integrate.nquad(func0, [[-1,1], [-1,1], [0,1]], full_output=True)[0]
    for i in range(4):
        for j in range(4):
            func = lambda s1,s2,p : s2*Gaussian(s1,dist[0][0],dist[0][1])*Gaussian(s2,dist[1][0],dist[1][1])*Dirichlet(p,dist[2])*density(s1,s2,p)[i,j]
            G[i,j] = integrate.nquad(func, [[-1,1], [-1,1], [0,1]], full_output=True)[0]/n
    return G

def Gamma_p(dist,density):
    G = np.zeros((4,4))
    func0 = lambda s1,s2,p : Gaussian(s1,dist[0][0],dist[0][1])*Gaussian(s2,dist[1][0],dist[1][1])*Dirichlet(p,dist[2])
    n = integrate.nquad(func0, [[-1,1], [-1,1], [0,1]], full_output=True)[0]
    for i in range(4):
        for j in range(4):
            func = lambda s1,s2,p : p*Gaussian(s1,dist[0][0],dist[0][1])*Gaussian(s2,dist[1][0],dist[1][1])*Dirichlet(p,dist[2])*density(s1,s2,p)[i,j]
            G[i,j] = integrate.nquad(func, [[-1,1], [-1,1], [0,1]], full_output=True)[0]/n
    return G