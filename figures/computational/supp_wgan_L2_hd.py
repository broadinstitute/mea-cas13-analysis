
# This script generates a figure that characterizes the relationship between the L2 norm of the WGAN's latent space variable and the hamming distance of the resulting guide.
# 100 site are taken from the random search dataset, and 500 guides are generated using randomly sampled latent variables for each site.

import os
import tensorflow as tf 
import pandas as pd
import pandas as pd
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
plt.rcParams["xtick.labelsize"] = 16
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
evolutionary_col = "#55ad70"
wgan_col =  (.35, .09, .35) 

from mea.utils import gan
from mea.utils import prepare_sequences as prep_seqs

absolute_path = os.path.dirname(__file__)
relative_path = '../../model-guided-exploration-algorithms/' 
full_path = os.path.join(absolute_path, relative_path)

global gen_model
gen_model, _ = gan.load_models(full_path + 'utils/gan_data')

target_sets = pd.read_pickle('../../detection/processed_sites/100random_sites_withG/seqs_df.pkl')
seqs = [preq_seqs.consensus_seq(x, nt = True) for x in target_sets.target_set]

output = pd.DataFrame()

for i, seq in enumerate(seqs):
    print(i)
    for sample in range(500):

        z = tf.Variable(initial_value=tf.random.normal([1, 10], stddev = 1.0),
                    trainable=True,
                    name='latent')
        
        gen_guide = gen_model([z, [prep_seqs.one_hot_encode(seq)]], training=False,
                        pad_to_target_length=False)
 
        gen_guide_nt = prep_seqs.convert_to_nucleotides(gen_guide.numpy()[0])
        
        output = output.append(pd.DataFrame({'target_nt': [seq], 'tested_guide_nt': [gen_guide_nt], 'norm_z': [tf.norm(z)], 'hd': [prep_seqs.hamming_dist(seq[10:-10], gen_guide_nt)]}))
        
output = output.reset_index()
output.norm_z = [x.numpy() for x in output.norm_z]

fig, ax = plt.subplots(figsize = (14, 7))

graph_o = output[output.hd < 11]
sns.violinplot(x = 'hd', y = 'norm_z', data = graph_o, palette = 'turbo')
plt.xlabel('Hamming distance between consensus of target set and WGAN-AM generated guide')
plt.ylabel('Euclidean norm of latent variable $z$')
fig.tight_layout()
fig.savefig('wgan_l2norm_vs_hd_supp.pdf', bbox_inches='tight')
plt.close('all')

