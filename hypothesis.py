import numpy as np
from state import rho
import random
import matplotlib.pyplot as plt
from fit import *
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as sts
from Gamma import *
from sigma import *
from sylvster import *

photon = 10**4
s_min = 0.5
# def pdf(s1,s2,p,m):
#     state = rho(s1,s2,p)
#     return np.trace(np.matmul(state,m))
def PSF(x,s):
    return 1/(2*np.pi)**0.25 * np.exp(-(x-s)**2/(4))

def pdf(x,s1,s2,sm):
    p = 0.3
    return p*PSF(x,sm)*PSF(x,s1)+(1-p)*PSF(x,sm)*PSF(x,s2)

s1 = 0.5
s2 = -0.2
x1 = np.linspace(-1+s1,1+s1,photon)
x2 = np.linspace(-1+s2,1+s2,photon)
p = 0.3
yl = pdf(x1,s1,s2,s_min/2)
yr = pdf(x2,s1,s2,-s_min/2)
sl = random.choices(x1, weights = yl, k = photon)
sr = random.choices(x2, weights = yr, k = photon)

# print(sl)
nl, binsl,patchesl = plt.hist(sl, 999, color='red',weights=np.ones(len(sl)) / len(sl),label='Left')
nr, binsr,patchesr = plt.hist(sr, 999, color='blue',weights=np.ones(len(sr)) / len(sr),label='Right')
(mu1, sigma1) = norm.fit(sl)
y = norm.pdf( binsl, mu1, sigma1)/300
l = plt.plot(binsl, y, 'r--', linewidth=2)
(mu2, sigma2) = norm.fit(sr)
y2 = norm.pdf( binsr, mu2, sigma2)/300
l = plt.plot(binsr, y2, 'b--', linewidth=2)
plt.legend()
plt.ylabel('Probability')
plt.xlabel('Position')
plt.title(r'$\mathrm{Left:}\ \mu=%.3f,\ \sigma=%.3f, \mathrm{Right:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu1, sigma1, mu2, sigma2))
plt.grid(True)
print(np.abs(mu1-mu2))
plt.show()