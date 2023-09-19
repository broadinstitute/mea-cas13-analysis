# This script collects all the random search files for the multi-target detection objective, determines the best hyperparameters, and plots the random search results
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(font="Helvetica", style='ticks')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

from matplotlib import font_manager
font_manager.fontManager.addfont('/home/ubuntu/Helvetica.ttf')

plt.rcParams['font.size'] = 20
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams["legend.frameon"] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.top'] = False

plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['font.family'] = 'Helvetica'

cmap = plt.cm.get_cmap('viridis')
gcmap = plt.cm.get_cmap('gray')
base_col = "#bbbbbb"
adapt_col =  "#555555" 
evo_col = "#55ad70"
wgan_col =  (.35, .09, .35) 

fig_dir ='./figures/detection/random_search/'


# Collecting results from the evolutionary algorithm
method = 'evolutionary'
curr_dir = './detection/gen_results/final-random-search/'

df_list = []
for file in [x for x in os.listdir(curr_dir) if 'pkl' in x and method in x]:
    df_list.append(pd.read_pickle(curr_dir + file))
    
results = pd.concat(df_list)

graph = results.groupby(by = ['seed', 'species']).mean()

# Plotting the different parameters
pltvars = ['S', 'j', 'beta', 'gamma']
fig_mis, ax = plt.subplots(nrows = len(pltvars), ncols = 1, figsize = (8, 5*(len(pltvars))))

rn = {'S': "$S$", 'j': "$j$", 'beta': "β", 'gamma': "$\gamma$"}

graph = results.groupby(by = ['seed']).mean()
i = 0
for var in pltvars:
    
    ax[i].scatter(graph[var].values, graph.fitness, c = evo_col)
    ax[i].set_ylabel('Mean probe fitness')
    print(rn[var])
    ax[i].set_xlabel(rn[var])
    
    if(var == 'beta'):
        ax[i].set_xscale('log')

    i += 1
    
plt.tight_layout()
plt.savefig(fig_dir + 'evolutionary_random_search_supp.pdf')
plt.close('all')

print(results.groupby('seed').mean().iloc[results.groupby('seed').mean().fitness.idxmax()])


# ### Final parameters for the evolutionary algorithm:
# 
# population_size                  87.000000
# j                                 0.794996
# beta                              0.077373
# mutation_rate                     0.003362
# model_queries                  1500.000000
# nearest_hd                        2.330000
# fitness                          -1.576848
# baseline_fitness                 -1.866707
# delta_fitness                     0.289860




method = 'wgan'
curr_dir = './detection/gen_results/final-random-search/'

df_list = []
for file in [x for x in os.listdir(curr_dir) if 'pkl' in x and method in x]:
    df_list.append(pd.read_pickle(curr_dir + file))
    
results = pd.concat(df_list)
results

pltvars = ['learning-rate', 'outer_rounds', 'inner_rounds']
fig_mis, ax = plt.subplots(nrows = len(pltvars), ncols = 1, figsize = (8, 5*(len(pltvars))))

cr = {'learning-rate': "α", 'outer_rounds': "$r_{outer}$", 'inner_rounds': "$r_{inner}$"}

graph = results.groupby(by = ['seed']).mean()
i = 0
for var in pltvars:
    
    ax[i].scatter(graph[var].values, graph.fitness, c = wgan_col)
    ax[i].set_ylabel('Mean probe fitness')
    ax[i].set_xlabel(cr[var])
    
    if(var == 'learning-rate'):
            ax[i].set_xscale('log')
            
    i += 1
    
plt.tight_layout()
plt.savefig(fig_dir + 'wgan_random_search_supp.pdf')
plt.close('all')
print(results.groupby('seed').mean().iloc[results.groupby('seed').mean().fitness.idxmax()])


# ### Final parameters for the WGAN-AM algorithm:
# 
# learning-rate         1.540127
# outer_rounds         28.000000
# inner_rounds        275.000000
# nearest_hd            0.850000
# fitness              -1.681419
# baseline_fitness     -1.865434
# delta_fitness         0.184015

