import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
def exp(t, A, lbda):
    r"""y(t) = A \cdot \exp(-\lambda t)"""
    return A * np.exp(-lbda * t)

def select_data(data,boundary):
    new_data = []
    for d in range(len(data)):
        if np.abs(data[d]) > boundary[0] and np.abs(data[d]) < boundary[1]:
            new_data.append(data[d])
    return new_data

def find_peaks(x, y):
    peak_x = []
    peak_vals = []
    for i in range(len(y)):
        if i == 0:
            continue
        if i == len(y) - 1:
            continue
        if (y[i-1] < y[i]) and (y[i+1] < y[i]):
            peak_x.append(x[i])
            peak_vals.append(y[i])
    return np.array(peak_x), np.array(peak_vals)

data = np.loadtxt("output2.txt")
fdata = np.loadtxt("f_output2.txt")
s1 = select_data(list(data[:,0]),[0.35,0.8])
# plt.plot(np.arange(len(s1)),s1,label='s1',c='red')
# plt.plot(np.arange(len(s1)),[np.mean(s1[1:]) for _ in range(len(s1))],label=r'mean=%.3f' %np.mean(s1[1:]),linestyle='--',c='red')
fs1 = select_data(list(fdata[:,0]),[0.35,0.8])
error1 = [np.abs(i-0.5) for i in s1]
ferror1 = [np.abs(i-0.5) for i in fs1][:len(error1)]
# plt.hist(error1,bins=int(len(error1)/2),color='red')
# plt.hist(ferror1,bins=int(len(ferror1)/2),color='blue')
# plt.xlim([0, 0.05])

s2 = select_data(list(data[:,1]),[0.1,0.5])
fs2 = select_data(list(fdata[:,1]),[0.1,0.5])
error2 = [np.abs(i+0.2) for i in s2]
ferror2 = [np.abs(i+0.2) for i in fs2][:len(error2)]
# print(len(error2),len(ferror2))
error = (error1 + error2)
error = [2*i for i in error]
ferror = ferror1 + ferror2

plt.hist(error,bins=int(len(error)),color='red',weights=np.ones(len(error)) / len(error),label='Fisher')
plt.hist(ferror,bins=2*int(len(ferror)),color='blue',weights=np.ones(len(ferror)) / len(ferror),label='Personick')
plt.xlim([0, 0.05])
plt.ylim([0, 1])
plt.legend()
plt.ylabel('Probability')
plt.xlabel('Position error')
plt.grid(True)

# p = select_data(list(data[:,2]),[0.1,0.5])
# fp = select_data(list(fdata[:,2]),[0.1,0.5])
# errorp = [np.abs(i-0.3) for i in p]
# ferrorp = [np.abs(i-0.3) for i in fp][:len(errorp)]
# n1, bins1, patches1 = plt.hist(errorp,bins=int(len(errorp)/2),color='red',weights=np.ones(len(errorp)) / len(errorp),label='Fisher')
# n2, bins2, patches2 = plt.hist(ferrorp,bins=2*int(len(ferrorp)/2),color='blue',weights=np.ones(len(ferrorp)) / len(ferrorp),label='Personick')
# # n1, bins1, patches1 = plt.hist(errorp,bins=int(len(errorp)/2),color='red',normed=1,label='Fisher')
# # n2, bins2, patches2 = plt.hist(ferrorp,bins=2*int(len(ferrorp)/2),color='blue',normed=1,label='Personick')
# (mu1, sigma1) = norm.fit(errorp)
# y = norm.pdf( bins1, mu1, sigma1)/200
# l = plt.plot(bins1, y, 'r--', linewidth=2)
# (mu2, sigma2) = norm.fit(ferrorp)
# y2 = norm.pdf( bins2, mu2, sigma2)/300
# l = plt.plot(bins2, y2, 'b--', linewidth=2)
# plt.legend()
# plt.ylabel('Probability')
# plt.xlabel('Position error')
# plt.title(r'$\mathrm{Fisher:}\ \mu=%.3f,\ \sigma=%.3f$,$\mathrm{Personick:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu1, sigma1,mu2, sigma2))
# plt.grid(True)

# plt.show()
# s2 = select_data(list(data[:,1]),[0.1,0.5])
# plt.plot(np.arange(len(s2)),s2,label='s2',c='blue')
# plt.plot(np.arange(len(s2)),[np.mean(s2[1:]) for _ in range(len(s2))],label=r'mean=%.3f' %np.mean(s2[1:]),linestyle='--',c='blue')
# p = select_data(list(data[:,2]),[0.1,0.5])
# plt.plot(np.arange(len(p)),p,label='p',c='black')
# plt.plot(np.arange(len(p)),[np.mean(p[1:]) for _ in range(len(p))],label=r'mean=%.3f' %np.mean(p[1:]),linestyle='--',c='black')

# noisy_peak_x, noisy_peak_y = find_peaks(np.arange(len(s1)-1),s1[1:])
# popt, pcov = curve_fit(exp, noisy_peak_x, noisy_peak_y)
# print(*[f"{val:.2f}+/-{err:.2f}" for val, err in zip(popt, np.sqrt(np.diag(pcov)))])

# plt.legend()
# plt.xlabel('expt')
# plt.ylabel('estimation')
plt.show()