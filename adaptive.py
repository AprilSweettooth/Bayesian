import numpy as np
import matplotlib.pyplot as plt
from fit import fit
import matplotlib.pyplot as plt
from Gamma import *
from sigma import *
from sylvster import *
from init_expt import *
import json


est = np.zeros((50,3))
# with open('output.txt', 'w') as filehandle:
for idx in range(50):
    # print('estimated value:',estimation)
    # json.dump(list(estimation), filehandle)
    # fileObject = open('output.txt', 'a')
    # np.savetxt(fileObject, list(estimation), delimiter = ',', newline = ' ')
    # fileObject.write('\n')
    # fileObject.close()
    est[idx,] = estimation
    s1_var = np.random.uniform(-0.1,0.1,photon)
    s2_var = np.random.uniform(-0.1,0.1,photon)
    b = np.random.dirichlet((p,1-p), photon)
    dataset = np.zeros((3,1000))
    for m in range(len(M)):
        for data in range(1000):
            dataset[m][data] = pdf(s1+np.array(s1_var)[data],s2+np.array(s2_var)[data],np.array(b)[data][0],M[m])

    n, bins = np.histogram(dataset[2], 999)
    posterior_param = [fit(dataset[0],init_a=posterior_param[0],init=False),fit(dataset[1],init_a=posterior_param[1],init=False,s='right'),fit(dataset[2],dirichlet=True,x=bins,init_a=a)]
    # print(posterior_param)
    # estimation = np.array([update_param(posterior_param[0],s_range),update_param(posterior_param[1],s_range),update_param(posterior_param[2],p_range,dirichlet=True)])
    estimation = np.array([posterior_param[0][0],posterior_param[1][0],update_param(posterior_param[2],p_range,dirichlet=True)])
    # print(estimation)

    a = estimation[-1]*(a-2)+1

    G0 = Gamma0(posterior_param,rho)
    # print(G0)
    G1 = Gamma_s1(posterior_param,rho)
    # print(G1)
    G2 = Gamma_s2(posterior_param,rho)
    # print(G2)
    Gb = Gamma_p(posterior_param,rho)
    # print(Gb)
    sig = Sigma(posterior_param,measurement,G0)
    # print(sig)
    h = H(sig)
    # print(h)
    M = np.array([Solve_sylvester(G0,h[0]*G1),Solve_sylvester(G0,h[1]*G2),Solve_sylvester(G0,h[2]*Gb)])
    # print(M)
