import torch
import pandas as pd
import numpy as np
import os
import pickle

# The MCMC sampling approach follows the strategies outlined in
# "Bayesian estimation of mixed multinomial logit models:
# Advances and simulation-based evaluations" by Prateek Bansala,
# Rico Krueger, Michel Bierlairec, Ricardo A. Dazianoa and Taha H. Rashidi

def mcmc(num_samples):
    a_hyper, nu_hyper = 10**3,2
    theta_l = torch.randn((n,3,1))
    beta = torch.randn((5,1))
    a = torch.randn(3)
    omega = torch.linalg.inv(torch.cov(theta_l.squeeze().T))
    
    rho_b = 0.1*torch.ones(1)
    rho_beta = 0.01*torch.ones(1)
    rej_global = []
    rej_local = []
    prior_g = torch.distributions.normal.Normal(torch.tensor([0.]),torch.tensor([10.**3]))
    sample_g = []
    sample_l = []
    chain_length = int(2*num_samples)
    for step in range(chain_length):
        # gibbs steps
        a = torch.distributions.gamma.Gamma(torch.tensor([(nu_hyper+3)/2]),1/a_hyper**2+nu_hyper*torch.diag(omega)).sample() 
        omega = torch.distributions.wishart.Wishart(df=nu_hyper+n+2, precision_matrix=torch.diag(1/(2*nu_hyper*a))+torch.einsum('bik,bjk->bijk', theta_l, theta_l).squeeze().sum(axis=0)).sample()
        # update theta_l 
        prior_l = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3),precision_matrix=omega)
        rej_local_local = []
        for i in range(n):
            bi = theta_l[i]
            bi_proposal = theta_l[i] + rho_b.sqrt()*prior_l.sample().reshape((3,1))
            r_log = torch.distributions.multinomial.Multinomial(logits=((x[:,i,:,:]@beta+z[:,i,:,:]@bi_proposal).squeeze()).movedim(0,-1)).log_prob(y[i]).sum()
            r_log += prior_l.log_prob(bi_proposal.flatten()).sum()
            r_log -= torch.distributions.multinomial.Multinomial(logits=((x[:,i,:,:]@beta+z[:,i,:,:]@bi).squeeze()).movedim(0,-1)).log_prob(y[i]).sum()
            r_log -= prior_l.log_prob(bi.flatten()).sum()
            if torch.rand(1).log()<= r_log:
                theta_l[i] = bi_proposal.clone()
                rej_local_local.append(0)
            else:
                rej_local_local.append(1)
        rej_local.append(np.mean(rej_local_local))
        # update beta
        beta_proposal = beta+rho_beta.sqrt()*torch.randn(beta.shape)#prior_g.sample(beta.shape).squeeze(-1)
        r_log = torch.distributions.multinomial.Multinomial(logits=((x@beta_proposal+z@theta_l).squeeze()).movedim(0,-1)).log_prob(y).sum()
        r_log += prior_g.log_prob(beta_proposal.flatten()).sum()
        r_log -= torch.distributions.multinomial.Multinomial(logits=((x@beta+z@theta_l).squeeze()).movedim(0,-1)).log_prob(y).sum()
        r_log -= prior_g.log_prob(beta.flatten()).sum()
        if torch.rand(1).log()<= r_log:
            beta = beta_proposal.clone()
            rej_global.append(0)
        else:
            rej_global.append(1)
        #save
        sample_l.append(theta_l.clone())
        eta = torch.zeros(6)
        c = torch.linalg.cholesky(omega)
        eta[0] = c[0,0].log()
        eta[1] = c[1,0]
        eta[2] = c[1,1].log()
        eta[3] = c[2,0]
        eta[4] = c[2,1]
        eta[5] = c[2,2].log()
        sample_g.append(torch.hstack([beta.clone().flatten(),eta.clone(),a.clone().log()]))
        if step > 50 and step<(0.9*num_samples):
            if np.mean(rej_local)>0.3:
                rho_b -= 0.001
            else:
                rho_b += 0.001
            rho_b.clip_(1e-5)
#            if np.mean(rej_global)>0.3:
#                rho_beta -= 0.001
#            else:
#                rho_beta += 0.001
#            rho_beta.clip_(1e-5)
        if step%1000==0:
            print(int(100*step/chain_length)/100)
    return torch.stack(sample_g).squeeze()[-num_samples:], torch.stack(sample_l).squeeze()[-num_samples:]


os.chdir('/Users/kocklucas/Desktop/skew_vb/final_code')

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


samples_mcmc_global, samples_mcmc_local = mcmc(num_samples=1000000)

pickle.dump((samples_mcmc_global, samples_mcmc_local),open('results/ParkingMCMC.p','wb'))

#import matplotlib.pyplot as plt
#for t in range(dim_global):
#    fig, ax = plt.subplots(dpi=300)
#    ax.plot(samples_mcmc_global[:,t])
#    ax.set_title(r'$\theta_{'+str(t+1)+'}$')
#    plt.show()
#    
#for _ in range(dim_global):
#    i = np.random.randint(n)
#    j = np.random.randint(3)
#    fig, ax = plt.subplots(dpi=300)
#    ax.plot(samples_mcmc_local[:,i,j])
#    ax.set_title(r'$b_{'+str(i+1)+','+str(j+1)+'}$')
#    plt.show()