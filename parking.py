from approximation import Variational_Approximation, global_correction, GVA_global_learned
from utils.plot_utils import generate_plots
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import pickle
import copy
import time

def log_h_i(theta_g,theta_l):
    beta, eta = theta_g[:5],theta_g[5:-3]#,theta_g[-3:]
    likelihood = torch.distributions.multinomial.Multinomial(logits=((x@beta+z@theta_l).squeeze()).movedim(0,-1)).log_prob(y).sum(axis=1)
    c = torch.zeros(3,3)
    c[0,0] = eta[0].exp()
    c[1,0] = eta[1]
    c[1,1] = eta[2].exp()
    c[2,0] = eta[3]
    c[2,1] = eta[4]
    c[2,2] = eta[5].exp()
    prior_b = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3),precision_matrix=c@c.T).log_prob(theta_l.squeeze())
    return (prior_b + likelihood)[:,None]
    
def log_prior_g(theta_g):
    a_hyper, nu_hyper = 10**3,2
    beta, eta, a_log = theta_g[:5],theta_g[5:-3],theta_g[-3:]
    prior_beta = torch.distributions.normal.Normal(torch.tensor([0.]),torch.tensor([10.**3])).log_prob(beta).sum()
    c = torch.zeros(3,3)
    c[0,0] = eta[0].exp()
    c[1,0] = eta[1]
    c[1,1] = eta[2].exp()
    c[2,0] = eta[3]
    c[2,1] = eta[4]
    c[2,2] = eta[5].exp()
    elimination_matrix = torch.block_diag(torch.eye(3),torch.hstack([torch.zeros(2,3-2),torch.eye(2)]),torch.hstack([torch.zeros(1,3-1),torch.eye(1)])) #L_k
    commutation_matrix = torch.eye(3**2)[torch.arange(3**2).reshape(3,3).T.flatten(),:] #K_{k,k}
    prior_eta = torch.distributions.wishart.Wishart(nu_hyper+3-1,precision_matrix=2*nu_hyper*torch.diag(a_log.exp().flatten())).log_prob(c@c.T)
    prior_eta += torch.linalg.det(elimination_matrix@torch.matmul((torch.eye(3**2)+commutation_matrix),torch.kron(c,torch.eye(3)))@elimination_matrix.T).log()+eta[[0,2,5]].sum()
    prior_a = (torch.distributions.gamma.Gamma(.5,1/a_hyper**2).log_prob(a_log.exp())+a_log).sum()
    return prior_beta+prior_eta+prior_a


# load data
df = pd.read_table('data/parking.dat')
df = df[['ID','CHOICE','AT1', 'AT2', 'AT3', 'TD1', 'TD2', 'TD3','FEE1', 'FEE2', 'FEE3','LI','RESIDENT']]
df['INTERCEPT'] = 1
df['ZERO'] = 0
df['LI x FEE1'] = df['LI']*df['FEE1']
df['LI x FEE2'] = df['LI']*df['FEE2']
df['LI x FEE3'] = df['LI']*df['FEE3']
df['RESIDENT x FEE1'] = df['RESIDENT']*df['FEE1']
df['RESIDENT x FEE2'] = df['RESIDENT']*df['FEE2']
df['RESIDENT x FEE3'] = df['RESIDENT']*df['FEE3']
df[['AT1', 'AT2', 'AT3']] = df[['AT1', 'AT2', 'AT3']]/5-2
df[['TD1', 'TD2', 'TD3']] = df[['TD1', 'TD2', 'TD3']]/5-2

for j in range(3):
    df['C'+str(j+1)] = 1*(df['CHOICE']==(j+1))

x1,x2,x3,y = [],[],[],[]
z1,z2,z3 = [],[],[]
for i in df['ID'].unique():
    x1.append(torch.from_numpy(np.asarray(df[df['ID']==i][['AT1','TD1','FEE1','LI x FEE1','RESIDENT x FEE1']])))
    x2.append(torch.from_numpy(np.asarray(df[df['ID']==i][['AT2','TD2','FEE2','LI x FEE2','RESIDENT x FEE2']])))
    x3.append(torch.from_numpy(np.asarray(df[df['ID']==i][['AT3','TD3','FEE3','LI x FEE3','RESIDENT x FEE3']])))
    z1.append(torch.from_numpy(np.asarray(df[df['ID']==i][['AT1','TD1','FEE1']])))
    z2.append(torch.from_numpy(np.asarray(df[df['ID']==i][['AT2','TD2','FEE2']])))
    z3.append(torch.from_numpy(np.asarray(df[df['ID']==i][['AT3','TD3','FEE3']])))
    y.append(torch.from_numpy(np.asarray(df[df['ID']==i][['C1','C2','C3']])))
y = torch.stack(y).float()
x1 = torch.stack(x1).float()
x2 = torch.stack(x2).float()
x3 = torch.stack(x3).float()
x = torch.stack([x1,x2,x3])
z1 = torch.stack(z1).float()
z2 = torch.stack(z2).float()
z3 = torch.stack(z3).float()
z = torch.stack([z1,z2,z3])

n = 197
dim_global = 5+6+3
dim_local = 3

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
samples_mcmc_global, samples_mcmc_local =  pickle.load(open('results/ParkingMCMC.p', 'rb'))

# make plots
generate_plots(samples_global,samples_local,samples_mcmc_global,samples_mcmc_local,labels,'parking')

# example specific plots
sigma_mcmc = []
for t in range(len(samples_mcmc_global)):
    eta = samples_mcmc_global[t][5:-3]
    c = torch.zeros(3,3)
    c[0,0] = eta[0].exp()
    c[1,0] = eta[1]
    c[1,1] = eta[2].exp()
    c[2,0] = eta[3]
    c[2,1] = eta[4]
    c[2,2] = eta[5].exp()
    sigma_mcmc.append(torch.linalg.inv(c@c.T))
sigma_mcmc = torch.stack(sigma_mcmc)

anz_methods = len(samples_global)

sigma_vb = [[] for _ in range(anz_methods)]
for j in range(anz_methods):
    for t in range(len(samples_global[j])):
        eta = samples_global[j][t][5:-3]
        c = torch.zeros(3,3)
        c[0,0] = eta[0].exp()
        c[1,0] = eta[1]
        c[1,1] = eta[2].exp()
        c[2,0] = eta[3]
        c[2,1] = eta[4]
        c[2,2] = eta[5].exp()
        sigma_vb[j].append(torch.linalg.inv(c@c.T))
    sigma_vb[j]=torch.stack(sigma_vb[j])

cmap = plt.get_cmap('rainbow',anz_methods)
c_mcmc = 'black'

fig_width = 8
size_labels = 10
size_titles = 10
fig, axs = plt.subplots(3,3,dpi=300,figsize=(fig_width,((5**.5 - 1) / 2)*fig_width))
for j1 in range(3):
    for j2 in range(3):
        ax = axs[j1,j2]
        if j2<=j1:
            ax.hist(sigma_mcmc[:,j1,j2],bins=70,density=True,color=c_mcmc,alpha=0.1)
            kde = gaussian_kde(sigma_mcmc[:,j1,j2])
            pos = np.linspace(min(sigma_mcmc[:,j1,j2]),max(sigma_mcmc[:,j1,j2]),500)
            ax.plot(pos,kde(pos),color=c_mcmc,alpha=0.8)
            for j_method in range(anz_methods):
                ax.hist(sigma_vb[j_method][:,j1,j2],bins=70,density=True,color=cmap(j_method),alpha=0.1)
                kde = gaussian_kde(sigma_vb[j_method][:,j1,j2])
                pos = np.linspace(min(sigma_vb[j_method][:,j1,j2]),max(sigma_vb[j_method][:,j1,j2]),500)
                ax.plot(pos,kde(pos),color=cmap(j_method),alpha=0.8)
            #ax.set_xlim((min(sigma_mcmc[:,j1,j2]),max(sigma_mcmc[:,j1,j2])))
        else:
            rho_mcmc = sigma_mcmc[:,j1,j2]/(sigma_mcmc[:,j1,j1].sqrt()*sigma_mcmc[:,j2,j2].sqrt())
            ax.hist(rho_mcmc,bins=70,density=True,color=c_mcmc,alpha=0.1)
            kde = gaussian_kde(rho_mcmc)
            pos = np.linspace(min(rho_mcmc),max(rho_mcmc),500)
            ax.plot(pos,kde(pos),color=c_mcmc,alpha=0.8)
            for j_method in range(anz_methods):
                rho_vb = sigma_vb[j_method][:,j1,j2]/(sigma_vb[j_method][:,j1,j1].sqrt()*sigma_vb[j_method][:,j2,j2].sqrt())
                ax.hist(rho_vb ,bins=70,density=True,color=cmap(j_method),alpha=0.1)
                kde = gaussian_kde(rho_vb)
                pos = np.linspace(min(rho_vb),max(rho_vb),500)
                ax.plot(pos,kde(pos),color=cmap(j_method),alpha=0.8)
            #ax.set_xlim((min(rho_mcmc),max(rho_mcmc)))
        ax.set_title([[r'$\sigma_{at}^2$',r'$\rho_{at,td}$',r'$\rho_{at,fee}$'],[r'$\sigma_{td,at}$',r'$\sigma_{td}^2$',r'$\rho_{td,fee}$'],[r'$\sigma_{fee,at}$',r'$\sigma_{fee,td}$',r'$\sigma_{fee}^2$']][j1][j2],size=size_titles)
        ax.set_xlim(([[1.,0.4,-1.],[0.,0.,-1.],[-8.1,-6,0.]][j1][j2],[[15.,1.,-0.63],[6.,4.,-0.2],[-2.,0.,15.]][j1][j2]))
for n_ax, ax in enumerate(axs.flatten()):
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=7)
    ax.text(-0., 1.1, string.ascii_uppercase[n_ax], transform=ax.transAxes, weight='bold')
fig.tight_layout()
plt.legend(handles=[mpatches.Patch(color=c_mcmc,alpha=0.8,label='MCMC')]+[mpatches.Patch(color=cmap(j_method),alpha=0.8,label=labels[j_method]) for j_method in range(anz_methods)], loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),bbox_transform = plt.gcf().transFigure,ncol=6,fontsize=size_labels)
plt.savefig("results/plots/parking_sigma.pdf", format="pdf", bbox_inches="tight")
plt.show()  

print(np.asarray(runtime)/60)
