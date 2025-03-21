import matplotlib.pyplot as plt
import string
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import scipy
import numpy as np

def generate_plots(samples_global,samples_local,samples_mcmc_global,samples_mcmc_local,labels,name):
    dim_global = len(samples_mcmc_global[0])
    dim_local = len(samples_mcmc_local[0,0])
    anz_methods = len(samples_global)
    cmap = plt.get_cmap('rainbow',anz_methods)
    c_mcmc = 'black'#'gray'
    fig_width = 8
    size_labels = 10
    size_titles = 10
    # global effects
    anz_x, anz_y = int(np.sqrt(dim_global)),int(np.sqrt(dim_global))
    while anz_x*anz_y<dim_global:
        if anz_y<=anz_x:
            anz_y+=1
        else:
            anz_x+=1
    fig, axs = plt.subplots(anz_x,anz_y,dpi=600,figsize=(fig_width,((5**.5 - 1) / 2)*fig_width))
    for j in range(dim_global):
        ax = axs.flatten()[j]
        ax.set_title(r'$\theta_{G,'+str(j+1)+'}$',size=size_titles)
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
    plt.legend(handles=[mpatches.Patch(color=c_mcmc,alpha=0.8,label='MCMC')]+[mpatches.Patch(color=cmap(j_method),alpha=0.8,label=labels[j_method]) for j_method in range(anz_methods)], loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),bbox_transform = plt.gcf().transFigure,ncol=8,fontsize=size_labels)
    plt.savefig("results/plots/"+name+"_global.pdf", format="pdf", bbox_inches="tight")
    plt.show()  

    # local effects
    for j_local in range(dim_local):
        fig, axs = plt.subplots(anz_methods,3,dpi=600,figsize=(fig_width,((5**.5 - 1) / 2)*fig_width))
        for j_method in range(anz_methods):
            if dim_local > 1:
                sample = samples_local[j_method][:,:,j_local]
            else:
                sample = samples_local[j_method]
            axs[j_method,0].scatter(samples_mcmc_local[:,:,j_local].mean(axis=0),sample.mean(axis=0),s=5,alpha=0.8,color='gray')
            axs[j_method,1].scatter(samples_mcmc_local[:,:,j_local].std(axis=0),sample.std(axis=0),s=5,alpha=0.8,color='gray')
            axs[j_method,2].scatter(scipy.stats.skew(samples_mcmc_local[:,:,j_local],0),scipy.stats.skew(sample,0),s=5,alpha=0.8,color='gray')
            axs[j_method,0].set_ylabel(labels[j_method])
        for n_ax, ax in enumerate(axs.flatten()):
            ax.grid(alpha=0.3)
            lims = [np.min([ax.get_xlim(), ax.get_ylim()]),np.max([ax.get_xlim(), ax.get_ylim()])]
            ax.plot(lims, lims, c='gray', zorder=0,alpha=0.3)
            ax.tick_params(axis='both', labelsize=7)
            #ax.text(-0., 1.1, string.ascii_uppercase[n_ax], transform=ax.transAxes, weight='bold')
        axs[0,0].set_title('mean')
        axs[0,1].set_title('std')
        axs[0,2].set_title('skewness')
        fig.tight_layout()
        plt.savefig("results/plots/"+name+"_local"+str(j_local+1)+".pdf", format="pdf", bbox_inches="tight")
        plt.show()