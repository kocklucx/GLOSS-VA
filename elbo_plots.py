import matplotlib.pyplot as plt
import pickle
import string
import numpy as np
from approximation import Variational_Approximation
import matplotlib.patches as mpatches

def log_h_i(theta_g,theta_l):
    return 0

def log_prior_g(theta_g):
    return 0

max_epochs=150000

cmap = plt.get_cmap('rainbow',5)
c_mcmc = 'black'#'gray'
fig_width = 8
size_labels = 10
size_titles = 10

pos = np.arange(max_epochs)
fig, axs = plt.subplots(3,1,dpi=600,figsize=(fig_width,1.5*((5**.5 - 1) / 2)*fig_width))
# logit
approximations = pickle.load(open('results/sixcities.p', 'rb'))['approximations']
axins = axs[0].inset_axes([0.5, 0.15, 0.45, 0.4])
for j in range(3):
    axs[0].plot(pos,approximations[j].elbo,color=cmap([0,2,4][j]))
    axins.plot(pos[-int(0.05*max_epochs):],approximations[j].elbo[-int(0.05*max_epochs):],color=cmap([0,2,4][j]))
axins.grid(alpha=0.3)
axins.tick_params(axis='both', labelsize=7)
axs[0].set_ylim((-59289.40039901734,-100))
axs[0].indicate_inset_zoom(axins)
axs[0].set_title(r'Random intercept logistic regression',size=size_titles)
#
approximations = pickle.load(open('results/epilepsy.p', 'rb'))['approximations']
axins1 = axs[1].inset_axes([0.5, 0.15, 0.45, 0.4])
for j in range(3):
    axs[1].plot(pos,approximations[j].elbo,color=cmap([0,2,4][j]))
    axins1.plot(pos[-int(0.05*max_epochs):],approximations[j].elbo[-int(0.05*max_epochs):],color=cmap([0,2,4][j]))
axins1.grid(alpha=0.3)
axins1.tick_params(axis='both', labelsize=7)
axs[1].indicate_inset_zoom(axins1)
axs[1].set_title(r'Mixed effects Poisson regression',size=size_titles)
#
approximations = pickle.load(open('results/parking.p', 'rb'))['approximations']
axins2 = axs[2].inset_axes([0.5, 0.15, 0.45, 0.4])
for j in range(3):
    axs[2].plot(pos,approximations[j].elbo,color=cmap([0,2,4][j]))
    axins2.plot(pos[-int(0.05*max_epochs):],approximations[j].elbo[-int(0.05*max_epochs):],color=cmap([0,2,4][j]))
axins2.grid(alpha=0.3)
axins2.tick_params(axis='both', labelsize=7)
axs[2].indicate_inset_zoom(axins2)
axs[2].set_title(r'Discrete choice model',size=size_titles)
axs[2].set_ylim(( -661626,-600))
#
for n_ax, ax in enumerate(axs.flatten()):
    ax.grid(alpha=0.3)
    ax.set_yscale("symlog")
    ax.set_xscale("log")
    ax.set_xlabel(r'epoch',size=size_titles)
    ax.set_ylabel(r'ELBO',size=size_titles)
    ax.tick_params(axis='both', labelsize=7)
    ax.text(-0., 1.1, string.ascii_uppercase[n_ax], transform=ax.transAxes, weight='bold')
    ax.set_xlim((10,max_epochs))
fig.tight_layout()
plt.legend(handles=[mpatches.Patch(color=cmap(0),alpha=0.8,label="G-VA"),mpatches.Patch(color=cmap(2),alpha=0.8,label="CSG-VA"),mpatches.Patch(color=cmap(4),alpha=0.8,label="GLOSS-VA")], loc = 'lower center', bbox_to_anchor = (0, -0.05, 1, 1),bbox_transform = plt.gcf().transFigure,ncol=8,fontsize=size_labels)
plt.savefig("results/plots/elbos.pdf", format="pdf", bbox_inches="tight")
plt.show()  