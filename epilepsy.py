from approximation import Variational_Approximation, global_correction
from utils.plot_utils import generate_plots
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from scipy.stats import spearmanr
import copy
import time

def log_h_i(theta_g,theta_l):
    beta,zeta = theta_g[:-3],theta_g[-3:]
    likelihood = torch.distributions.poisson.Poisson(rate=torch.exp(x@beta+z@theta_l)).log_prob(y).sum(axis=1)
    l = torch.zeros((2,2))
    l[0,0] = zeta[0].exp()
    l[1,0] = zeta[1]
    l[1,1] = zeta[2].exp()
    prior = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2),scale_tril=l).log_prob(theta_l.squeeze())[:,None]
    return prior + likelihood

def log_prior_g(theta_g):
    return torch.distributions.normal.Normal(torch.tensor([0.]),torch.tensor([10.])).log_prob(theta_g).sum()

# load data
df = pd.read_csv(r'data/epilepsy.csv')
df['id'] =  np.asarray([[t for _ in range(4)] for t in range(59)]).flatten()

x,y,z = [],[],[]
for i in df['id'].unique():
    x.append(torch.from_numpy(np.asarray(df[df['id']==i][['intercept','base','treatment','base x treatment','age','visit']])))
    z.append(torch.from_numpy(np.asarray(df[df['id']==i][['intercept','visit']])))
    y.append(torch.from_numpy(np.asarray(df[df['id']==i]['y'])))
y = torch.vstack(y).float()[:,:,None]
x = torch.stack(x).float()
z = torch.stack(z).float()

n = len(y)
dim_global = 9
dim_local = 2

approximations = []
runtime = []
samples_global = []
samples_local = []
labels = []

num_samples=25000
max_epochs=150000
window_size=1000
# G-VA and G-VA^+
start = time.time()
vb = Variational_Approximation(log_h_i=log_h_i,log_prior_g=log_prior_g,y=y,dim_global=dim_global,dim_local=dim_local,skewness=False,variance=False)
vb.train(max_epochs=max_epochs,window_size=window_size)
runtime.append(time.time()-start)
approximations.append(copy.deepcopy(vb))
sample_g, sample_l = vb.sample(num_samples)
samples_global.append(sample_g)
samples_local.append(sample_l)
labels.append(r'G-VA')
mean_global = vb.mu_g
mean_local = vb.m_i
sample_g, sample_l = global_correction(sample_g,sample_l,log_h_i,log_prior_g,vb.mu_g,vb.m_i)
samples_global.append(sample_g)
samples_local.append(sample_l)
labels.append(r'G-VA$^+$')
# CSG-VA and GLOSS-VA^-
start = time.time()
vb = Variational_Approximation(log_h_i=log_h_i,log_prior_g=log_prior_g,y=y,dim_global=dim_global,dim_local=dim_local,skewness=False,variance=True)
vb.train(max_epochs=max_epochs,window_size=window_size)
runtime.append(time.time()-start)
approximations.append(copy.deepcopy(vb))
sample_g, sample_l = vb.sample(num_samples)
samples_global.append(sample_g)
samples_local.append(sample_l)
labels.append(r'CSG-VA')
vb.skewness = True
sample_g, sample_l = vb.sample(num_samples)
samples_global.append(sample_g)
samples_local.append(sample_l)
labels.append(r'GLOSS-VA$^-$')
# GLOSS-VA
start = time.time()
vb = Variational_Approximation(log_h_i=log_h_i,log_prior_g=log_prior_g,y=y,dim_global=dim_global,dim_local=dim_local,skewness=True,variance=True)
vb.train(max_epochs=max_epochs,window_size=window_size)
runtime.append(time.time()-start)
approximations.append(copy.deepcopy(vb))
sample_g, sample_l = vb.sample(num_samples)
samples_global.append(sample_g)
samples_local.append(sample_l)
labels.append(r'GLOSS-VA')
# MCMC
df = pd.read_csv("results/EpilepsyMCMC.csv")
samples_mcmc_global = np.asarray(df[['beta.'+str(t) for t in range(1,7)]+['zeta.'+str(t) for t in range(1,4)]])
samples_mcmc_local = np.stack([np.asarray(df[['b.'+str(j)+'.'+str(i) for i in range(1,n+1)]]) for j in range(1,dim_local+1)],axis=-1)


# make plots
generate_plots(samples_global,samples_local,samples_mcmc_global,samples_mcmc_local,labels,'epilepsy')

# example specific plots
var1_mcmc = []
var2_mcmc = []
corr_mcmc = []
for t in range(len(samples_mcmc_global)):
    c = np.asarray([[np.exp(samples_mcmc_global[t,-3]),0],[samples_mcmc_global[t,-2],np.exp(samples_mcmc_global[t,-1])]])
    sigma = np.matmul(c,c.T)
    var1_mcmc.append(sigma[0,0])
    var2_mcmc.append(sigma[1,1])
    corr_mcmc.append(sigma[0,1]/np.sqrt(sigma[0,0]*sigma[1,1]))

anz_methods = len(samples_global)

var1, var2, corr = [[] for _ in range(anz_methods)],[[] for _ in range(anz_methods)],[[] for _ in range(anz_methods)]
for j in range(anz_methods):
    for t in range(num_samples):
        c = np.asarray([[np.exp(samples_global[j][t,-3]),0],[samples_global[j][t,-2],np.exp(samples_global[j][t,-1])]])
        sigma = np.matmul(c,c.T)
        var1[j].append(sigma[0,0])
        var2[j].append(sigma[1,1])
        corr[j].append(sigma[0,1]/np.sqrt(sigma[0,0]*sigma[1,1]))

cmap = plt.get_cmap('rainbow',anz_methods)
c_mcmc = 'black'

fig_width = 8
size_labels = 10
size_titles = 10
fig, axs = plt.subplots(1,3,dpi=300,figsize=(fig_width,.5*((5**.5 - 1) / 2)*fig_width))
axs[0].hist(var1_mcmc,bins=70,density=True,color=c_mcmc,alpha=0.1)
kde = gaussian_kde(var1_mcmc)
pos = np.linspace(np.quantile(var1_mcmc,q=0),max(var1_mcmc),500)
axs[0].plot(pos,kde(pos),color=c_mcmc,alpha=0.8)
for j_method in range(anz_methods):
    axs[0].hist(var1[j_method],bins=70,density=True,color=cmap(j_method),alpha=0.1)
    kde = gaussian_kde(var1[j_method])
    pos = np.linspace(min(var1[j_method]),max(var1[j_method]),500)
    axs[0].plot(pos,kde(pos),color=cmap(j_method),alpha=0.8)
axs[0].set_title(r'$\sigma^2_0$',size=size_titles)
#
axs[1].hist(var2_mcmc,bins=70,density=True,color=c_mcmc,alpha=0.1)
kde = gaussian_kde(var2_mcmc)
pos = np.linspace(np.quantile(var2_mcmc,q=0),max(var2_mcmc),500)
axs[1].plot(pos,kde(pos),color=c_mcmc,alpha=0.8)
for j_method in range(anz_methods):
    axs[1].hist(var2[j_method],bins=70,density=True,color=cmap(j_method),alpha=0.1)
    kde = gaussian_kde(var2[j_method])
    pos = np.linspace(min(var2[j_method]),max(var2[j_method]),500)
    axs[1].plot(pos,kde(pos),color=cmap(j_method),alpha=0.8)
axs[1].set_title(r'$\sigma^2_{0,visit}$',size=size_titles)
#
axs[2].hist(corr_mcmc,bins=70,density=True,color=c_mcmc,alpha=0.1)
kde = gaussian_kde(corr_mcmc)
pos = np.linspace(np.quantile(corr_mcmc,q=0),max(var1_mcmc),500)
axs[2].plot(pos,kde(pos),color=c_mcmc,alpha=0.8)
for j_method in range(anz_methods):
    axs[2].hist(corr[j_method],bins=70,density=True,color=cmap(j_method),alpha=0.1)
    kde = gaussian_kde(corr[j_method])
    pos = np.linspace(min(corr[j_method]),max(corr[j_method]),500)
    axs[2].plot(pos,kde(pos),color=cmap(j_method),alpha=0.8)
axs[2].set_title(r'$\rho_{0,visit}$',size=size_titles)
#
for n_ax, ax in enumerate(axs.flatten()):
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=7)
    ax.text(-0., 1.1, string.ascii_uppercase[n_ax], transform=ax.transAxes, weight='bold')
axs[1].set_xlim((axs[1].get_xlim()[0],2))
axs[0].set_xlim((axs[0].get_xlim()[0],0.63))
fig.tight_layout()
plt.legend(handles=[mpatches.Patch(color=c_mcmc,alpha=0.8,label='MCMC')]+[mpatches.Patch(color=cmap(j_method),alpha=0.8,label=labels[j_method]) for j_method in range(anz_methods)], loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),bbox_transform = plt.gcf().transFigure,ncol=6,fontsize=size_labels)
plt.savefig("results/plots/epilepsy_sigma.pdf", format="pdf", bbox_inches="tight")
plt.show()  

print(np.asarray(runtime)/60)