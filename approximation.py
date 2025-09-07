import torch

class Variational_Approximation:
    """ 
    Variational approximations for Bayesian hierarchical models. The method can derive a G_VA, 
    CSG-VA, and GLOSS-VA
    
    Inputs:
        - log_h_i: A vectorized method to calculate the kernel of p(b_i\vert \theta_G,y). The method takes two inputs: 
            a torch array for the local and a torch array for the global parameters
        - log_prior_g: A method to calculate log(p(\theta_G)) takes a torch array for the global parameters as input
        - dim_global: The dimension of the global parameters
        - dim_local: the dimension d_L=d_i for i=1,\dots,n for the local parameters
        - skewness: A binary indicator for the conditional skewness correction
        - variance: A binary indicator for the conditional variance correction
    
    The variational parameters are optimized with the method train()
    A sample from the variational approximation can be generated with sample()
    
    """
    def __init__(self,log_h_i,log_prior_g,y,dim_global,dim_local,skewness=False,variance=False):
        super().__init__()
        self.n = len(y)
        self.dim_global = dim_global 
        self.dim_local = dim_local 
        self.log_h_i = log_h_i
        self.log_prior_g = log_prior_g
        self.y = y
        self.elbo = []
        self.variance = variance
        self.skewness = skewness
        # initalize lambda
        self.mu_g = torch.zeros(self.dim_global,1)
        self.t_g_lt = torch.zeros(int(self.dim_global*(self.dim_global-1)/2)) # lower triangular
        self.t_g_diag = torch.zeros(self.dim_global) # log values of diagonal
        self.m_i =  torch.zeros((self.n,self.dim_local,1))
        self.t_gi = torch.zeros((self.n,self.dim_global,self.dim_local))
        self.f_i_lt = torch.zeros((self.n,int(self.dim_local*(self.dim_local-1)/2))) #lower triangular
        self.f_i_diag = torch.zeros((self.n,self.dim_local)) # effect for log-diagonal
        self.b_i_lt = torch.zeros((self.n,int(self.dim_local*(self.dim_local-1)/2),self.dim_global))
        self.b_i_diag = torch.zeros((self.n,self.dim_local,self.dim_global))
        #
        self.ix_global = torch.tril_indices(self.dim_global,self.dim_global,offset=-1)
        self.ix_local = torch.tril_indices(self.dim_local,self.dim_local,offset=-1)
        #
        #set clever starting value
        self.mu_g.requires_grad=True
        self.m_i.requires_grad=True
        optimizer = torch.optim.Adam([self.mu_g,self.m_i],maximize=True,lr=0.05)
        for _ in range(1000):
            optimizer.zero_grad()
            ll = self.log_h_full(self.mu_g,self.m_i,self.m_i).sum()
            ll.backward()
            optimizer.step()
        self.mu_g.requires_grad=False
        self.m_i.requires_grad=False
        self.t_g_diag += 2.
        self.f_i_diag += 2.
            
    def single_gaussian_sample(self,return_mu_i=False):
        eps_g = torch.randn((self.dim_global,1))
        eps_i = torch.randn((self.n,self.dim_local,1))
        t_g = torch.diag(self.t_g_diag.exp())
        t_g[self.ix_global[0],self.ix_global[1]] = self.t_g_lt
        theta_g = self.mu_g + torch.linalg.solve_triangular(t_g.T,eps_g,upper=True)
        t_i = torch.zeros((self.n,self.dim_local,self.dim_local))
        t_i[:,torch.arange(self.dim_local),torch.arange(self.dim_local)] = (self.f_i_diag + (self.b_i_diag @ theta_g).squeeze(2)).exp()
        t_i[:,self.ix_local[0],self.ix_local[1]] = self.f_i_lt + (self.b_i_lt @ theta_g).squeeze(2)
        mu_i = self.m_i-torch.matmul(torch.linalg.solve_triangular(t_i.transpose(1,2),self.t_gi.transpose(1,2),upper=True),theta_g-self.mu_g)
        theta_l = mu_i+torch.linalg.solve_triangular(t_i.transpose(1,2),eps_i,upper=True)
        if return_mu_i:
            return theta_g, theta_l, mu_i
        else:
            return theta_g, theta_l
        
    def log_w_g(self,theta_g):
        l0 = self.log_h_tilde(theta_g)
        l1 = self.log_h_tilde(2*self.mu_g-theta_g)
        return l0-torch.logsumexp(torch.hstack([l0,l1]),dim=0)
    
    def log_h_tilde(self,theta_g):
        t_i = torch.zeros((self.n,self.dim_local,self.dim_local))
        t_i[:,torch.arange(self.dim_local),torch.arange(self.dim_local)] = (self.f_i_diag + (self.b_i_diag @ theta_g).squeeze(2)).exp()
        t_i[:,self.ix_local[0],self.ix_local[1]] = self.f_i_lt + (self.b_i_lt @ theta_g).squeeze(2)
        mu_i = self.m_i-torch.matmul(torch.linalg.solve_triangular(t_i.transpose(1,2),self.t_gi.transpose(1,2),upper=True),theta_g-self.mu_g)
        h = torch.tensor([2.]).log()
        h += self.log_h_i(theta_g,mu_i).sum()
        h -= self.n*(-self.dim_local/2*torch.tensor(2*torch.pi).log())+(self.f_i_diag + (self.b_i_diag @ theta_g).squeeze(2)).sum()
        return h 
    
    def log_h_full(self,theta_g,theta_l,mu_i):
        if self.skewness:
            w = self.log_w_i(theta_g,theta_l,mu_i).exp()
            return (self.log_h_i(theta_g,theta_l)*w).sum()+(self.log_h_i(theta_g,2*mu_i-theta_l)*(1-w)).sum()+self.log_prior_g(theta_g).sum()
        else:
            return self.log_h_i(theta_g,theta_l).sum()+self.log_prior_g(theta_g).sum()
        
    def log_w_i(self,theta_g,theta_l,mu_i):
        l0 = self.log_h_i(theta_g,theta_l)
        l1 = self.log_h_i(theta_g,2*mu_i-theta_l)
        return (l0-torch.logsumexp(torch.hstack([l0,l1]),dim=1).reshape(l0.shape))
    
    def q_lambda(self,theta_g,theta_l,mu_i=None):
        t_g = torch.diag(self.t_g_diag.exp())
        t_g[self.ix_global[0],self.ix_global[1]] = self.t_g_lt
        t_i = torch.zeros((self.n,self.dim_local,self.dim_local))
        t_i[:,torch.arange(self.dim_local),torch.arange(self.dim_local)] = (self.f_i_diag + (self.b_i_diag @ theta_g).squeeze(2)).exp()
        t_i[:,self.ix_local[0],self.ix_local[1]] = self.f_i_lt + (self.b_i_lt @ theta_g).squeeze(2)
        if mu_i == None:
            mu_i = self.m_i-torch.matmul(torch.linalg.solve_triangular(t_i.transpose(1,2),self.t_gi.transpose(1,2),upper=True),theta_g-self.mu_g)
        q_g = -self.dim_global/2*torch.tensor(2*torch.pi).log()+self.t_g_diag.sum()-0.5*torch.matmul(t_g.T,theta_g-self.mu_g).pow(2).sum()
        q_i = self.n*(-self.dim_local/2*torch.tensor(2*torch.pi).log())+(self.f_i_diag + (self.b_i_diag @ theta_g).squeeze(2)).sum()-0.5*torch.matmul(t_i.transpose(1,2),(theta_l)-mu_i).pow(2).sum()
        if self.skewness:
            w = self.log_w_i(theta_g, theta_l, mu_i).exp()
            q_i += (w*self.log_w_i(theta_g, theta_l, mu_i)+(1-w)*self.log_w_i(theta_g, 2*mu_i-theta_l, mu_i)).sum()
        return q_g+q_i
    
    def train(self,max_epochs=10000,window_size=500):
        params = [self.mu_g,self.t_g_lt,self.t_g_diag,self.m_i,self.t_gi,self.f_i_lt,self.f_i_diag]
        if self.variance==True:
            params += [self.b_i_lt,self.b_i_diag]
        for parameter in params:
            parameter.requires_grad = True   

        optimizer = torch.optim.Adam(params,maximize=True,lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',factor=0.5,patience=1,min_lr=1e-4)
        
        elbo_track = 0
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            if self.skewness:
                eps_g = torch.randn((self.dim_global,1))
                eps_i = torch.randn((self.n,self.dim_local,1))
                t_g = torch.diag(self.t_g_diag.exp())
                t_g[self.ix_global[0],self.ix_global[1]] = self.t_g_lt
                theta_g = self.mu_g + torch.linalg.solve_triangular(t_g.T,eps_g,upper=True)
                t_i = torch.zeros((self.n,self.dim_local,self.dim_local))
                t_i[:,torch.arange(self.dim_local),torch.arange(self.dim_local)] = (self.f_i_diag + (self.b_i_diag @ theta_g).squeeze(2)).exp()
                t_i[:,self.ix_local[0],self.ix_local[1]] = self.f_i_lt + (self.b_i_lt @ theta_g).squeeze(2)
                mu_i = self.m_i-torch.matmul(torch.linalg.solve_triangular(t_i.transpose(1,2),self.t_gi.transpose(1,2),upper=True),theta_g-self.mu_g)
                theta_l = mu_i+torch.linalg.solve_triangular(t_i.transpose(1,2),eps_i,upper=True)
                w_g = self.log_w_g(theta_g).exp()
                elbo = w_g*(self.log_h_full(theta_g,theta_l,mu_i)-self.q_lambda(theta_g,theta_l,mu_i))
                theta_g = 2*self.mu_g-theta_g
                t_i = torch.zeros((self.n,self.dim_local,self.dim_local))
                t_i[:,torch.arange(self.dim_local),torch.arange(self.dim_local)] = (self.f_i_diag + (self.b_i_diag @ theta_g).squeeze(2)).exp()
                t_i[:,self.ix_local[0],self.ix_local[1]] = self.f_i_lt + (self.b_i_lt @ theta_g).squeeze(2)
                mu_i = self.m_i-torch.matmul(torch.linalg.solve_triangular(t_i.transpose(1,2),self.t_gi.transpose(1,2),upper=True),theta_g-self.mu_g)
                theta_l = mu_i+torch.linalg.solve_triangular(t_i.transpose(1,2),eps_i,upper=True)
                elbo += (1-w_g)*(self.log_h_full(theta_g,theta_l,mu_i)-self.q_lambda(theta_g,theta_l,mu_i))
            else:
                # draw_theta
                theta_g, theta_l, mu_i = self.single_gaussian_sample(return_mu_i=True)
                # estimate elbo
                elbo = self.log_h_full(theta_g,theta_l,mu_i)-self.q_lambda(theta_g,theta_l,mu_i)
            # update parameters
            elbo.backward()
            optimizer.step()
            self.elbo.append(elbo.item())
            elbo_track += 1/window_size*elbo.item()
            # print 
            if epoch%window_size==0 and epoch>0:
                print(int(100*epoch/max_epochs),'%',elbo_track)
                scheduler.step(elbo_track)
                elbo_track = 0
        for parameter in params:
            parameter.requires_grad = False
            
    def sample_gaussian(self,num_samples):
        theta_g_draws,theta_l_draws = [],[]
        for _ in range(num_samples):
            theta_g, theta_l = self.single_gaussian_sample()
            theta_g_draws.append(theta_g)
            theta_l_draws.append(theta_l)
        theta_g_draws,theta_l_draws = torch.stack(theta_g_draws).detach_(),torch.stack(theta_l_draws).detach_()
        return theta_g_draws.squeeze(),theta_l_draws.squeeze()
    
    def sample(self,num_samples):
        if self.skewness:
            theta_g_draws,theta_l_draws = [],[]
            for _ in range(num_samples):
                theta_g, theta_l, mu_i = self.single_gaussian_sample(return_mu_i=True)
                w = self.log_w_i(theta_g,theta_l,mu_i)
                u = torch.rand(w.shape).log()
                theta_l += (1*(u>w)).reshape((self.n,1,1))*(2*mu_i-2*theta_l)
                if torch.rand(1).log() > self.log_w_g(theta_g):
                    theta_g = 2*self.mu_g-theta_g
                theta_g_draws.append(theta_g)
                theta_l_draws.append(theta_l)
            theta_g_draws,theta_l_draws = torch.stack(theta_g_draws).detach_(),torch.stack(theta_l_draws).detach_()
            return theta_g_draws.squeeze(),theta_l_draws.squeeze()
        
        else: 
            return self.sample_gaussian(num_samples)

def global_correction(samples_global,samples_local,log_h_i,log_prior_g,mean_global,mean_local):
    samples_global,samples_local = samples_global.clone(),samples_local.clone()
    for j in range(len(samples_global)):
        theta_g = samples_global[j].reshape(mean_global.shape)
        theta_l = samples_local[j].reshape(mean_local.shape)
        l0 = log_h_i(theta_g,theta_l).sum()+log_prior_g(theta_g)
        l1 = log_h_i(2*mean_global-theta_g,2*mean_local-theta_l).sum()+log_prior_g(2*mean_global-theta_g)
        w = (l0-torch.logsumexp(torch.hstack([l0,l1]),dim=0)).exp()
        if torch.rand(1)>w:
            samples_global[j] = (2*mean_global-theta_g).reshape(samples_global[j].shape)
            samples_local[j] = (2*mean_local-theta_l).reshape(samples_local[j].shape)
    return samples_global, samples_local

class GVA_global_learned:
    """ 
    Variational approximations for G-VA$^{G+}$
    
    Inputs:
        - log_h_i: A vectorized method to calculate the kernel of p(b_i\vert \theta_G,y). The method takes two inputs: 
            a torch array for the local and a torch array for the global parameters
        - log_prior_g: A method to calculate log(p(\theta_G)) takes a torch array for the global parameters as input
        - dim_global: The dimension of the global parameters
        - dim_local: the dimension d_L=d_i for i=1,\dots,n for the local parameters
    
    The variational parameters are optimized with the method train()
    A sample from the variational approximation can be generated with sample()
    
    """
    def __init__(self,log_h_i,log_prior_g,y,dim_global,dim_local):
        super().__init__()
        self.n = len(y)
        self.dim_global = dim_global 
        self.dim_local = dim_local 
        self.log_h_i = log_h_i
        self.log_prior_g = log_prior_g
        self.y = y
        self.elbo = []
        # initalize lambda
        self.mu_g = torch.zeros(self.dim_global,1)
        self.t_g_lt = torch.zeros(int(self.dim_global*(self.dim_global-1)/2)) # lower triangular
        self.t_g_diag = torch.zeros(self.dim_global) # log values of diagonal
        self.m_i =  torch.zeros((self.n,self.dim_local,1))
        self.t_gi = torch.zeros((self.n,self.dim_global,self.dim_local))
        self.f_i_lt = torch.zeros((self.n,int(self.dim_local*(self.dim_local-1)/2))) #lower triangular
        self.f_i_diag = torch.zeros((self.n,self.dim_local)) # effect for log-diagonal
        #
        self.ix_global = torch.tril_indices(self.dim_global,self.dim_global,offset=-1)
        self.ix_local = torch.tril_indices(self.dim_local,self.dim_local,offset=-1)
        #
        #set clever starting value
        self.mu_g.requires_grad=True
        self.m_i.requires_grad=True
        optimizer = torch.optim.Adam([self.mu_g,self.m_i],maximize=True,lr=0.05)
        for _ in range(1000):
            optimizer.zero_grad()
            ll = self.log_h_full(self.mu_g,self.m_i).sum()
            ll.backward()
            optimizer.step()
        self.mu_g.requires_grad=False
        self.m_i.requires_grad=False
        self.t_g_diag += 2.
        self.f_i_diag += 2.
            
    def single_gaussian_sample(self,return_mu_i=False):
        eps_g = torch.randn((self.dim_global,1))
        eps_i = torch.randn((self.n,self.dim_local,1))
        t_g = torch.diag(self.t_g_diag.exp())
        t_g[self.ix_global[0],self.ix_global[1]] = self.t_g_lt
        theta_g = self.mu_g + torch.linalg.solve_triangular(t_g.T,eps_g,upper=True)
        t_i = torch.zeros((self.n,self.dim_local,self.dim_local))
        t_i[:,torch.arange(self.dim_local),torch.arange(self.dim_local)] = (self.f_i_diag).exp()
        t_i[:,self.ix_local[0],self.ix_local[1]] = self.f_i_lt
        mu_i = self.m_i-torch.matmul(torch.linalg.solve_triangular(t_i.transpose(1,2),self.t_gi.transpose(1,2),upper=True),theta_g-self.mu_g)
        theta_l = mu_i+torch.linalg.solve_triangular(t_i.transpose(1,2),eps_i,upper=True)
        if return_mu_i:
            return theta_g, theta_l, mu_i
        else:
            return theta_g, theta_l
     
    def log_w_pozza(self,theta_g,theta_l):
        l0 = self.log_h_i(theta_g,theta_l).sum()+self.log_prior_g(theta_g)
        l1 = self.log_h_i(2*self.mu_g-theta_g,2*self.m_i-theta_l).sum()+self.log_prior_g(2*self.mu_g-theta_g)
        log_w = (l0-torch.logsumexp(torch.hstack([l0,l1]),dim=0))   
        return log_w
     
    def log_h_full(self,theta_g,theta_l):
        return self.log_h_i(theta_g,theta_l).sum()+self.log_prior_g(theta_g).sum()
        
    def q_lambda(self,theta_g,theta_l,mu_i=None):
        t_g = torch.diag(self.t_g_diag.exp())
        t_g[self.ix_global[0],self.ix_global[1]] = self.t_g_lt
        t_i = torch.zeros((self.n,self.dim_local,self.dim_local))
        t_i[:,torch.arange(self.dim_local),torch.arange(self.dim_local)] = (self.f_i_diag).exp()
        t_i[:,self.ix_local[0],self.ix_local[1]] = self.f_i_lt
        if mu_i == None:
            mu_i = self.m_i-torch.matmul(torch.linalg.solve_triangular(t_i.transpose(1,2),self.t_gi.transpose(1,2),upper=True),theta_g-self.mu_g)
        q_g = -self.dim_global/2*torch.tensor(2*torch.pi).log()+self.t_g_diag.sum()-0.5*torch.matmul(t_g.T,theta_g-self.mu_g).pow(2).sum()
        q_i = self.n*(-self.dim_local/2*torch.tensor(2*torch.pi).log())+(self.f_i_diag).sum()-0.5*torch.matmul(t_i.transpose(1,2),(theta_l)-mu_i).pow(2).sum()
        return q_g+q_i
    
    def train(self,max_epochs=10000,window_size=500):
        params = [self.mu_g,self.t_g_lt,self.t_g_diag,self.m_i,self.t_gi,self.f_i_lt,self.f_i_diag]
        for parameter in params:
            parameter.requires_grad = True   

        optimizer = torch.optim.Adam(params,maximize=True,lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',factor=0.5,patience=1,min_lr=1e-4)
        
        elbo_track = 0
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            # draw_theta
            theta_g, theta_l, mu_i = self.single_gaussian_sample(return_mu_i=True)
            # estimate elbo
            w_log = self.log_w_pozza(theta_g, theta_l)
            elbo = w_log.exp()*(self.log_h_full(theta_g,theta_l)-self.q_lambda(theta_g,theta_l,mu_i)-w_log)
            elbo += (1-w_log.exp())*(self.log_h_full(2*self.mu_g-theta_g,2*self.m_i-theta_l)-self.q_lambda(2*self.mu_g-theta_g,2*self.m_i-theta_l)-self.log_w_pozza(2*self.mu_g-theta_g,2*self.m_i-theta_l))
            # update parameters
            elbo.backward()
            optimizer.step()
            self.elbo.append(elbo.item())
            elbo_track += 1/window_size*elbo.item()
            # print 
            if epoch%window_size==0 and epoch>0:
                print(int(100*epoch/max_epochs),'%',elbo_track)
                scheduler.step(elbo_track)
                elbo_track = 0
        for parameter in params:
            parameter.requires_grad = False
            
    def sample(self,num_samples):
        theta_g_draws,theta_l_draws = [],[]
        for _ in range(num_samples):
            theta_g, theta_l = self.single_gaussian_sample(return_mu_i=False)
            log_w = self.log_w_pozza(theta_g, theta_l)
            if torch.rand(1) > log_w.exp():
                theta_g = 2*self.mu_g-theta_g
                theta_l = 2*self.m_i-theta_l
            theta_g_draws.append(theta_g)
            theta_l_draws.append(theta_l)
        theta_g_draws,theta_l_draws = torch.stack(theta_g_draws).detach_(),torch.stack(theta_l_draws).detach_()
        return theta_g_draws.squeeze(),theta_l_draws.squeeze()
