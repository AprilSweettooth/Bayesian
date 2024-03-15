import numpy as np
from scipy.stats import norm
import random
import matplotlib.pyplot as plt

photon = 10**3

def PSF(x,s,sig):
    return 1/(2*np.pi*sig**2)**0.25 * np.exp(-(x-s)**2/(4*sig**2))

def pdf(x,s1,s2,sig):
    p = 0.3
    return p*PSF(x,s1,sig)*PSF(x,s1,sig)+(1-p)*PSF(x,s2,sig)*PSF(x,s2,sig)

x = np.linspace(-1,1,photon)
s1 = 0.5
s2 = -0.2
sig = 1
y = pdf(x,s1,s2,sig)
s = random.choices(x, weights = y, k = photon)
data = np.array(s)
N = data.shape[0] # number of data points
K=2 # two components GMM
tot_iterations = 5000 # stopping criterium

# Step-1 (Init)
s = np.random.uniform(low=-1, high=1, size=K) # mean
sigma = np.random.uniform(low=0.1, high=2, size=K) # standard deviaiton
pi = np.ones(K) * (1.0/K) # mixing coefficients
r = np.zeros([K,N]) # responsibilities
nll_list = list() # used to store the neg log-likelihood (nll)

for iteration in range(tot_iterations):
    # Step-2 (E-Step)
    for k in range(K):
        r[k,:] = pi[k] * norm.pdf(x=data, loc=s[k], scale=sigma[k])
    r = r / np.sum(r, axis=0) #[K,N] -> [N]
        
    # Step-3 (M-Step)
    N_k = np.sum(r, axis=1) #[K,N] -> [K]
    for k in range(K):
        s[k] = np.sum(r[k,:] * data) / N_k[k] # update mean
        numerator = r[k] * (data - s[k])**2
    pi = N_k/N # update mixing coefficient
        
    # Estimate likelihood and print info
    likelihood = 0.0
    for k in range(K):
        likelihood += pi[k] * norm.pdf(x=data, loc=s[k], scale=sigma[k])
    nll_list.append(-np.sum(np.log(likelihood)))
    # print("Iteration: "+str(iteration)+"; NLL: "+str(nll_list[-1]))
    # print("Mean "+str(s)+"\nStd "+ str(sigma)+"\nWeights "+ str(pi)+"\n")
    # Step-4 (Check)
    if(iteration==tot_iterations-1): 
        print("Final Mean "+str(s)+"\nStd "+ str(sigma)+"\nWeights "+ str(pi))
        break
# plt.semilogx(ratio, [(e-r)/r for e,r in zip(evolution,ratio)], label=r'$\mu=$%1.3f'% m, color=color[coherence.index(m)])
# plt.xlabel(r'$\theta/\sigma$')
# plt.ylabel(r'$\check{\theta}$ fractional error')
# plt.title('Fractional MLE Error')
# plt.legend()
# plt.show()