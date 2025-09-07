from approximation import Variational_Approximation
import torch
import matplotlib.pyplot as plt

# specifiy the likelihood and the prior
def log_h_i(theta_g,theta_l):
    prior_l = torch.distributions.normal.Normal(loc=torch.tensor([0.]),scale=(-theta_g[1]).exp()).log_prob(theta_l).sum(axis=1)
    likelihood = torch.distributions.bernoulli.Bernoulli(logits=theta_g[0]+theta_l).log_prob(y).sum(axis=1)
    return prior_l + likelihood

def log_prior_g(theta_g):
    return torch.distributions.normal.Normal(torch.tensor([0.]),torch.tensor([10.])).log_prob(theta_g).sum()

# draw data
n = 80
theta_g = torch.tensor([[1.],[2.]]) # fixed effect, log-precision of the random intercept
theta_l = torch.distributions.normal.Normal(loc=torch.tensor([0.]),scale=(-theta_g[1]).exp()).sample((n,)) # simulate random intercepts
y = torch.zeros((n,5,1))
for i in range(n):
    y[i] = torch.distributions.bernoulli.Bernoulli(logits=theta_g[0]+theta_l[i]).sample((5,))

dim_global = 2
dim_local = 1

# learn GLOSS-VA
vb = Variational_Approximation(log_h_i=log_h_i,log_prior_g=log_prior_g,y=y,dim_global=dim_global,dim_local=dim_local,skewness=True,variance=True)
vb.train(max_epochs=150000,window_size=1000)

# simulate from the variational approximation
sample_g, sample_l = vb.sample(5000) # samples for theta_g and theta_l

# plot the results
fig, axs = plt.subplots(1,2)
for j in range(2):
    axs[j].hist(sample_g[:,j],bins=50,density=True,alpha=0.8)
    axs[j].axvline(theta_g[j],color='black')
fig.tight_layout()
plt.show()
