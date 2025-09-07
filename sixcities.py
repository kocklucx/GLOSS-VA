from approximation import Variational_Approximation, global_correction, GVA_global_learned
from utils.plot_utils import generate_plots
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import copy
import time


def log_h_i(theta_g,theta_l):
    prior_l = torch.distributions.normal.Normal(loc=torch.tensor([0.]),scale=(-theta_g[-1]).exp()).log_prob(theta_l).sum(axis=1)
    likelihood = torch.distributions.bernoulli.Bernoulli(logits=x@theta_g[:-1]+theta_l).log_prob(y).sum(axis=1)
    return prior_l + likelihood

def log_prior_g(theta_g):
    return torch.distributions.normal.Normal(torch.tensor([0.]),torch.tensor([10.])).log_prob(theta_g).sum()

# load data
df = pd.read_csv(r'data/ohio.csv')
df['intercept'] = 1
df['smokeage'] = df['smoke']*df['age']

x,y = [],[]
for i in df['id'].unique():
    x.append(torch.from_numpy(np.asarray(df[df['id']==i][['intercept','smoke','age','smokeage']])))
    y.append(torch.from_numpy(np.asarray(df[df['id']==i]['resp'])))
    df[df['id']==i][['smoke']]
y = torch.vstack(y).float()[:,:,None]
x = torch.stack(x).float()

n = len(y)
dim_global = 5
dim_local = 1

approximations = []
runtime = []
samples_global = []
samples_local = []
labels = []

num_samples=25000
max_epochs=150000
window_size=1000

# Approximations based on the G-VA
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
labels.append(r'G-VA$^{G-}$')
vb.skewness = True
sample_g, sample_l = vb.sample(num_samples)
samples_global.append(sample_g)
samples_local.append(sample_l)
labels.append(r'G-VA$^{H-}$')
vb = GVA_global_learned(log_h_i=log_h_i,log_prior_g=log_prior_g,y=y,dim_global=dim_global,dim_local=dim_local)
vb.train(max_epochs=max_epochs,window_size=window_size)
approximations.append(copy.deepcopy(vb))
sample_g, sample_l = vb.sample(num_samples)
samples_global.append(sample_g)
samples_local.append(sample_l)
labels.append(r'G-VA$^{G+}$')

# Approximations based on CSG-VA
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
labels.append(r'CSG-VA$^{H-}$')

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
df = pd.read_csv("results/SixcitiesMCMC.csv")
samples_mcmc_global = np.asarray(df[['beta.'+str(t) for t in range(1,5)]+['zeta']])
samples_mcmc_local = np.asarray(df[['b.'+str(i) for i in range(1,n+1)]])[:,:,None]

# make plots
generate_plots(samples_global,samples_local,samples_mcmc_global,samples_mcmc_local,labels,'sixcities')

# example specific plots
dim_global = len(samples_mcmc_global[0])
dim_local = len(samples_mcmc_local[0,0])
anz_methods = len(samples_global)
cmap = plt.get_cmap('rainbow',anz_methods)
c_mcmc = 'black'#'gray'
fig_width = 8
size_labels = 10
size_titles = 10
fig, axs = plt.subplots(2,3,dpi=600,figsize=(fig_width,((5**.5 - 1) / 2)*fig_width))
for j in range(dim_global):
    ax = axs.flatten()[j]
    ax.set_title([r'$\beta_{0}$',r'$\beta_{smoke}$',r'$\beta_{age}$',r'$\beta_{smoke\times age}$',r'$\eta$'][j],size=size_titles)
    ax.hist(samples_mcmc_global[:,j],bins=70,density=True,color=c_mcmc,alpha=0.1)
    kde = gaussian_kde(samples_mcmc_global[:,j])
    pos = np.linspace(min(samples_mcmc_global[:,j]),max(samples_mcmc_global[:,j]),500)
    ax.plot(pos,kde(pos),color=c_mcmc,alpha=0.8)
    for j_method in range(anz_methods):
        sample = samples_global[j_method]
        ax.hist(sample[:,j],bins=70,density=True,color=cmap(j_method),alpha=0.1)
        kde = gaussian_kde(sample[:,j])
        pos = np.linspace(min(sample[:,j]),max(sample[:,j]),500)
        ax.plot(pos,kde(pos),color=cmap(j_method),alpha=0.8)
for j in range(dim_global,len(axs.flatten())):
    fig.delaxes(axs.flatten()[j])
for n_ax, ax in enumerate(axs.flatten()):
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=7)
    ax.text(-0., 1.1, string.ascii_uppercase[n_ax], transform=ax.transAxes, weight='bold')
fig.tight_layout()
plt.legend(handles=[mpatches.Patch(color=c_mcmc,alpha=0.8,label='MCMC')]+[mpatches.Patch(color=cmap(j_method),alpha=0.8,label=labels[j_method]) for j_method in range(anz_methods)],bbox_to_anchor = (-0.1, -0.6, 1, 1),bbox_transform = plt.gcf().transFigure, ncol=1,fontsize=size_labels)
plt.savefig("results/plots/sixcities.pdf", format="pdf", bbox_inches="tight")
plt.show()  

fig, axs = plt.subplots(2,3,dpi=600,figsize=(fig_width,((5**.5 - 1) / 2)*fig_width))
pos = np.linspace(min(samples_mcmc_global[:,0]),max(samples_mcmc_global[:,0]),100)
# mcmc
ax = axs.flatten()[0]
ax.scatter(samples_mcmc_global[:,0],samples_mcmc_global[:,-1],color='gray',alpha=0.1,s=3)
ax.set_title('MCMC',size=size_titles)
for j in range(len(labels)):
    ax = axs.flatten()[j+1]
    ax.scatter(samples_global[j][:,0],samples_global[j][:,-1],color='gray',alpha=0.1,s=3)
    ax.set_title(labels[j],size=size_titles)
for n_ax, ax in enumerate(axs.flatten()):
    ax.set_xlim((min(samples_mcmc_global[:,0]),max(samples_mcmc_global[:,0])))
    ax.set_ylim((min(samples_mcmc_global[:,-1]),max(samples_mcmc_global[:,-1])))
    ax.grid(alpha=0.3)
    ax.set_xlabel(r'$\beta_{0}$')
    ax.set_ylabel(r'$\eta$')
    ax.tick_params(axis='both', labelsize=7)
fig.tight_layout()
plt.savefig("results/plots/sixcities_bivariate.pdf", format="pdf", bbox_inches="tight")
plt.show()  

print(np.asarray(runtime)/60)
