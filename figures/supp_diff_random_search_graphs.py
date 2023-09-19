# This script collects all the random search files for the differential identification objective, determines the best hyperparameters, and plots the random search results

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
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams["legend.frameon"] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.top'] = False
# ax.spines['right'].set_color('red')
# ax.spines['left'].set_color('red')
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['font.family'] = 'Helvetica'

cmap = plt.cm.get_cmap('viridis')
gcmap = plt.cm.get_cmap('gray')
base_col = "#bbbbbb"
adapt_col =  "#555555" 
evolutionary_col = "#55ad70"
wgan_col =  (.35, .09, .35) 

fig_dir ='./figures/discrimination/random_search/'

curr_dir = './discrimination/gen_results/parallel-final-random-search/'

files_list = [x for x in os.listdir(curr_dir) if '.pkl' in x]
results = pd.DataFrame()
for file in files_list:
    try:
        results = results.append(pd.read_pickle(curr_dir + file))
    except Exception:
        continue

# The WGAN results have the optimizer specicifed, while the evolutionary does not
wresults = results[results.optimizer == 'adam']
wresults.seed.unique()

gresults = results[results.optimizer != 'adam']
gresults.seed.unique()


# Evolutionary algorithm results
# Need to make sure that the jobs were successfully run across the targets for each of the parameter sets
valid_seeds = gresults.groupby(by = 'seed').size().where(gresults.groupby(by = 'seed').size() == 700).dropna().index.values
p1 = gresults[gresults['seed'].isin(valid_seeds)]

p2 = p1.groupby(['seed', 'species']).apply(lambda group: group.nlargest(1, columns='fitness')).reset_index(drop=True)
p3 = p2.groupby(by = 'seed').quantile(0.1, interpolation = 'linear')

# As described in methods, we only considered parameter sets which have a mean on-target activity greater than 1.65
valid_seeds = [index for index, row in p3.iterrows() if row['t1_act'] > -1.65]

mean_df = p2.groupby(by = 'seed').mean().reset_index()
mean_df['target_diff'] = mean_df.t1_act - mean_df.t2_act
mean_df['target_div'] = mean_df.t1_act/mean_df.t2_act

mean_df_valid = mean_df[mean_df['seed'].isin(valid_seeds)]
idxmax = mean_df_valid.target_diff.idxmax()

# Extract the best parameter set
print(mean_df_valid.loc[idxmax])

# ### Final evolutionary algorithm parameters
# c                                 1.000000
# a                                 5.897292
# k                                -2.857755
# o                                -2.510856
# t2w                               1.736507
# runtime                           0.340310
# nearest_hd                        2.400000
# fitness                           1.423454
# baseline_fitness                  0.756044
# t1_act                           -1.475120
# t2_act                           -2.898574
# baseline_t1_act                  -0.921333
# baseline_t2_act                  -1.677377
# delta_fitness                     0.667410
# S                               119.000000
# j                                 0.893401
# beta                              2.201796
# mutation_rate                     0.029049
# target_diff                       1.423454
# target_div                        0.508912




pltvars = ['S', 'j', ,'beta', 'gamma', 'a', 'k', 'o', 't2w']
fig_mis, ax = plt.subplots(nrows = len(pltvars), ncols = 1, figsize = (8, 5*(len(pltvars))))

rn = {'S': "$S$", 'j': "$j$", 'beta': "β", 
      'gamma': "$\gamma$", 'a': "$a$", 'k': "$k$", 'o': "$o$", 't2w': "$r$"}

graph = results.groupby(by = ['seed']).mean()
i = 0
for var in pltvars:
    
    ax[i].scatter(mean_df[var].values, mean_df.t1_act - mean_df.t2_act, c =  evolutionary_col)
    ax[i].set_ylabel('Mean(on-target - off-target activity)')
    ax[i].set_xlabel(rn[var])
        
    if(var == 'beta'):
            ax[i].set_xscale('log')
            
    i += 1
    
plt.tight_layout()
plt.savefig(fig_dir + 'snp_evolutionary_random_search_supp.pdf')
plt.close('all')


# Compiling results for WGAN-AM algorithm
valid_seeds = wresults.groupby(by = 'seed').size().where(wresults.groupby(by = 'seed').size() == 700).dropna().index.values
p1 = wresults[wresults['seed'].isin(valid_seeds)]

p2 = p1.groupby(['seed', 'species']).apply(lambda group: group.nlargest(1, columns='fitness')).reset_index(drop=True)

p3 = p2.groupby(by = 'seed').quantile(0.1, interpolation = 'linear')

# As described in methods, we only considered parameter sets which have a mean on-target activity greater than 1.65
valid_seeds = [index for index, row in p3.iterrows() if row['t1_act'] > -1.65]

mean_df = p2.groupby(by = 'seed').mean().reset_index()
mean_df['target_diff'] = mean_df.t1_act - mean_df.t2_act
mean_df['target_div'] = mean_df.t1_act/mean_df.t2_act

mean_df_valid = mean_df[mean_df['seed'].isin(valid_seeds)]
idxmax = mean_df_valid.target_diff.idxmax()

#best results
print(mean_df_valid.loc[idxmax])

pltvars = ['learning-rate', 'outer_rounds', 'inner_rounds', 'a', 'k', 'o', 't2w']

#pltvars = ['mutation_rate', 'children_proportion', 'beta', 'model_queries']
fig_mis, ax = plt.subplots(nrows = len(pltvars), ncols = 1, figsize = (8, 5*(len(pltvars))))
cr = {'learning-rate': "α", 'outer_rounds': "$r_{outer}$", 'inner_rounds': "$r_{inner}$", 
      'mutation_rate': "$\gamma$", 'a': "$a$", 'k': "$k$", 'o': "$o$", 't2w': "$r$"}

graph = results.groupby(by = ['seed']).mean()
i = 0
for var in pltvars:
    
    ax[i].scatter(mean_df[var].values, mean_df.t1_act - mean_df.t2_act, c =  wgan_col)
    ax[i].set_ylabel('Mean(on-target - off-target activity)', fontsize=10)
    ax[i].set_xlabel(cr[var])
    
    if(var == 'learning-rate'):
            ax[i].set_xscale('log')
            
    i += 1
    
plt.tight_layout()
plt.savefig(fig_dir + 'snp_wgan_random_search_supp.pdf')
plt.close('all')


# ### WGAN-AM algorithm results:
# learning-rate                    0.632998
# outer_rounds                     8.000000 
# inner_rounds                   144.000000
# c                                1.000000
# a                                3.769183
# k                               -3.833902
# o                               -2.134395
# t2w                              2.973052
# runtime                          2.039900
# nearest_hd                       1.600000
# fitness                          1.273881
# baseline_fitness                 0.789633
# t1_act                          -1.469775
# t2_act                          -2.743656
# baseline_t1_act                 -0.994261
# baseline_t2_act                 -1.783895
# delta_fitness                    0.484248
# target_diff                      1.273881
# target_div                       0.535700
