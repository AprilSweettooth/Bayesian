import numpy as np
from Gamma import Gaussian, Dirichlet
from scipy import integrate
from numpy import linalg as LA

def Sigma(dist,m,G0):
    func11 = lambda s1 : s1*s1*Gaussian(s1,dist[0][0],dist[0][1])
    func22 = lambda s2 : s2*s2*Gaussian(s2,dist[1][0],dist[1][1])
    funcbb = lambda p : p*p*Dirichlet(p,dist[2])
    func12 = lambda s1,s2 : s1*s2*Gaussian(s1,dist[0][0],dist[0][1])*Gaussian(s2,dist[1][0],dist[1][1])
    func1b = lambda s1,p : s1*p*Gaussian(s1,dist[0][0],dist[0][1])*Dirichlet(p,dist[2])
    func2b = lambda s2,p : s2*p*Gaussian(s2,dist[1][0],dist[1][1])*Dirichlet(p,dist[2])
    s11 = integrate.nquad(func11, [[-1,1]], full_output=True)[0]
    s22 = integrate.nquad(func22, [[-1,1]], full_output=True)[0]
    sbb = integrate.nquad(funcbb, [[0,1]], full_output=True)[0]
    s12 = integrate.nquad(func12, [[-1,1], [-1,1]], full_output=True)[0]
    s1b = integrate.nquad(func1b, [[-1,1], [0,1]], full_output=True)[0]
    s2b = integrate.nquad(func2b, [[-1,1], [0,1]], full_output=True)[0]
    G11 = np.trace(np.matmul(G0,m[0]))
    G22 = np.trace(np.matmul(G0,m[1]))
    G33 = np.trace(np.matmul(G0,m[2]))
    G12 = np.trace(np.matmul(G0,(np.matmul(m[0],m[1])+np.matmul(m[1],m[0]))/2))
    G13 = np.trace(np.matmul(G0,(np.matmul(m[0],m[2])+np.matmul(m[2],m[0]))/2))
    G23 = np.trace(np.matmul(G0,(np.matmul(m[2],m[1])+np.matmul(m[1],m[2]))/2))
    return np.array([[s11-G11,s12-G12,s1b-G13],[s12-G12,s22-G22,s2b-G23],[s1b-G13,s2b-G23,sbb-G33]])

def H(sigma):
    eigenvalues, eigenvectors = LA.eig(sigma)
    # eigenvalues = np.real(eigenvalues)
    return eigenvectors[list(eigenvalues).index(min(eigenvalues))]