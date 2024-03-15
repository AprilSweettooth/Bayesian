import numpy as np 
import scipy.special as sc
from scipy.stats import norm
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from Gamma import Gaussian

def Dirichlet(x,a,b):
    return x**a*(1-x)**b/(sc.beta(a, b)+0.00001)

def Combine_Dirichlet(prior,post):
    xdata = np.linspace(0, 1, 100)
    y = Dirichlet(xdata, prior[0],prior[1])*Dirichlet(xdata,post[0],post[1])
    popt, pcov = curve_fit(Dirichlet, xdata, y)
    return popt

def Combine_Gaussian(prior,post,s=None):
    xdata = np.linspace(-1, 1, 100)
    y = Gaussian(xdata,post[0],post[1])
    delta = np.abs(s-prior[0])
    popt, pcov = curve_fit(Gaussian, xdata, y,maxfev=1000,bounds=([min(0,s),0],[max(0,s),np.inf]))
    return popt

def fit(data,dirichlet=False,x=None,init_a=None,init=True,s='left'):
	# best fit of data
    if dirichlet:
        fitfunc  = lambda a, x: x**a[0]*(1-x)**a[1]/sc.beta(a[0], a[1])
        errfunc  = lambda a, x, y: (y - fitfunc(a, x))     
        out   = leastsq( errfunc, init_a, args=(x, data))
        a = out[0]
        post_a = Combine_Dirichlet(init_a,a)
        return post_a
    else:
        (mu, sigma) = norm.fit(data)
        if init:
            return mu, sigma
        else:
            if s=='left':
                return Combine_Gaussian(init_a,[mu,sigma],0.5)
            else:
                return Combine_Gaussian(init_a,[mu,sigma],-0.2)
        