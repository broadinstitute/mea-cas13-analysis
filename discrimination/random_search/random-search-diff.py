"""
This script randomly samples a set of hyperparameters for the evolutionary and WGAN explorers and runs them on a set of 50 sequences.

This enables us to benchmark the performance of many hyperparameter sets and choose the one with the best performance.
"""

# Importing standard packages
from scipy.stats import uniform
import traceback
import timeit
import argparse
import pandas as pd
import multiprocessing
from multiprocessing import set_start_method
import os.path as path
from collections import Counter 
from operator import itemgetter 
import time
import numpy as np
import os
import random


# Importing MEA packages
from mea.models.cas13_mult import Cas13Mult
from mea.models.cas13_diff import Cas13Diff

from mea.utils.cas13_landscape import Cas13Landscape
from mea.utils import prepare_sequences as prep_seqs
from mea.explorers.evolutionary import EvolutionaryExplorer
from mea.explorers.wgan_am import WGANExplorer
from mea.utils import import_fasta

def wgan_run(results_dir, alphabet, target_set1, target_set2, site_pos, seq_id, seed):
    # Run the WGAN-AM algorithm with randomly sampled hyperparameters
    
    f = open(results_dir + "job-tracker.txt", "a")
    f.write("\nwgan_seed{}_{}".format(seed, seq_id)) 
    f.close()
    start = timeit.default_timer()

    try:
        np.random.seed(seed)

        curr_grid = {'c': 1.0, 'a' : float(uniform.rvs(.1, 9, size=1)[0]), 'k': float(uniform.rvs(-4, 3, size=1)[0]), 
        'o': float(uniform.rvs(-4.5, 3.0, size=1)[0]), 't2w' : float(uniform.rvs(.1, 10, size=1)[0])}

        learning_rate = 10**uniform.rvs(-2, 4, size=1)[0]

        outer_rounds = int(uniform.rvs(5, 10, size=1)[0])
        inner_rounds = int(uniform.rvs(50, 180, size=1)[0])

        hyperparams = pd.DataFrame({'optimizer': ['adam'], 'learning-rate': [learning_rate], 'seed': [seed], 'seq_id': [seq_id], 'species': [seq_id.split('.')[0]],
        'outer_rounds': [outer_rounds], 'inner_rounds': [inner_rounds],
        'c': [curr_grid['c']], 'a': [curr_grid['a']], 'k': [curr_grid['k']], 'o': [curr_grid['o']], 't2w': [curr_grid['t2w']]})

        cas13landscape = Cas13Landscape()

        cas13diff = Cas13Diff(cas13landscape, target_set1, target_set2, curr_grid)

        starting_sequence = prep_seqs.consensus_seq(cas13diff.target_set1_nt, nt = True)
        baseline_fitness = cas13diff.get_fitness([starting_sequence[10:-10]])
        
        wgan_explorer = WGANExplorer(
            cas13diff,
            rounds=outer_rounds,
            starting_sequence=starting_sequence,
            inner_iterations=inner_rounds,
            adam_lr = learning_rate,
            optimizer = optimizer,
            sequences_batch_size=100,
            model_queries_per_batch=2000,
            )

        wgan_sequences, metadata = wgan_explorer.run(cas13diff, seq_id)
        
        save_rand_results(wgan_sequences, hyperparams, 'wgan', results_dir, start, target_set1 + target_set2, starting_sequence[10:-10], cas13diff, seq_id)

    except Exception:
        f = open(results_dir + "job-tracker.txt", "a")
        print(traceback.format_exc())
        f.write("\nERROR")
        f.write(traceback.format_exc())
        f.close()

def evolutionary_run(results_dir, alphabet, target_set1, target_set2, site_pos, seq_id, seed):
    # Run the evolutionary algorithm with randomly sampled hyperparameters

    f = open(results_dir + "job-tracker.txt", "a")
    f.write("\nevolutionary_seed{}_{}".format(seed, seq_id))
    f.close()
    start = timeit.default_timer()

    try:
        np.random.seed(seed)

        curr_grid = {'c': 1.0, 'a' : float(uniform.rvs(.1, 9, size=1)[0]), 'k': float(uniform.rvs(-4, 3, size=1)[0]), 
        'o': float(uniform.rvs(-4.5, 3.0, size=1)[0]), 't2w' : float(uniform.rvs(.1, 10, size=1)[0])}

        model_queries_per_batch = 1500 # Could try sampling, but 1500 is standard
        S = int(uniform.rvs(50, 200, size=1)[0])
        j = uniform.rvs(0.1, 0.8, size=1)[0] 
        
        beta = 10**uniform.rvs(-2, 4, size=1)[0]
        
        gamma = float(uniform.rvs(0, 2, size=1)[0]/28) 

        hyperparams = pd.DataFrame({'seed': [seed], 'seq_id': [seq_id], 'species': [seq_id.split('.')[0]], 'S': [S], 'j': [j],
        'beta': [beta], 'gamma': [gamma], 'model_queries': [model_queries_per_batch],
        'c': [curr_grid['c']], 'a': [curr_grid['a']], 'k': [curr_grid['k']], 'o': [curr_grid['o']], 't2w': [curr_grid['t2w']]
        })

        cas13landscape = Cas13Landscape()

        cas13diff = Cas13Diff(cas13landscape, target_set1, target_set2, curr_grid)
        
        starting_sequence = prep_seqs.consensus_seq(cas13diff.target_set1_nt, nt = True)
        baseline_fitness = cas13diff.get_fitness([starting_sequence[10:-10]])

        evolutionary_explorer = EvolutionaryExplorer(
            cas13diff,
            S=S,
            beta=beta,
            j=j,
            rounds=1,
            starting_sequence= starting_sequence,
            gamma = gamma,
            sequences_batch_size=50,
            model_queries_per_batch=model_queries_per_batch,
            alphabet=alphabet
            )

        evolutionary_seqs, metadata = evolutionary_explorer.run(cas13diff, seq_id) 

        save_rand_results(evolutionary_seqs, hyperparams, 'evolutionary', results_dir, start, target_set1 + target_set2, starting_sequence[10:-10], cas13diff, seq_id)
    
    except Exception:
        f = open(results_dir + "job-tracker.txt", "a")
        print(traceback.format_exc())
        f.write("\nERROR")
        f.write(traceback.format_exc())
        f.close()


def save_rand_results(df, hyperparams, algo, results_dir, start, target_set, baseline_guide, cas13diff, seq_id):
    # Save the results of the random search to a csv file

    stop = timeit.default_timer()

    print('saving results...')

    hyperparams['runtime'] = (stop - start)/60

    start_df = df.iloc[0]
    guide_df = df.iloc[1:]
    best_guide_df = guide_df[guide_df.model_score == guide_df.model_score.max()].iloc[0]
    gen_guide_nt = best_guide_df.sequence 
    score = best_guide_df.model_score
 
    hyperparams['nearest_hd'] = min([prep_seqs.amming_dist(gen_guide_nt, target[10:-10]) for target in target_set])
    hyperparams['fitness'] = cas13diff._fitness_function(gen_guide_nt, output_type='eval')[0][2]
    hyperparams['baseline_fitness'] = cas13diff._fitness_function(baseline_guide, output_type='eval')[0][2]

    hyperparams['t1_act'] = cas13diff._fitness_function(gen_guide_nt, output_type='eval')[0][0]
    hyperparams['t2_act'] = cas13diff._fitness_function(gen_guide_nt, output_type='eval')[0][1]

    hyperparams['baseline_t1_act'] = cas13diff._fitness_function(baseline_guide, output_type='eval')[0][0]
    hyperparams['baseline_t2_act'] = cas13diff._fitness_function(baseline_guide, output_type='eval')[0][1]


    hyperparams['delta_fitness'] = hyperparams['fitness'] - hyperparams['baseline_fitness']

    hyperparams['sequence'] = gen_guide_nt
    hyperparams['baseline_guide'] = baseline_guide

    results_path = results_dir + '{}_rand-search_{}.pkl'.format(algo, random.random())
    hyperparams.to_pickle(results_path)

    # if(path.exists(results_path)):
    #     time.sleep(np.random.choice(np.arange(10, 60)))
    #     old_results = pd.read_pickle(results_path)
    #     upd_results = old_results.append(hyperparams)
    #     upd_results.to_pickle(results_path)
    #     upd_results.to_csv(results_dir + '{}_rand-search_{}.pkl'.format(algo, random.random()))
    # else:
    #     hyperparams.to_pickle(results_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id',
        help=("number of the seq to run"),
        type = int) 

    args = parser.parse_args()

    set_start_method("spawn")
    num_cpu = 70

    data_dir = './figures/discrimination/50seqs_synmismatch.pkl'
    site_df = pd.read_pickle(data_dir).reset_index(drop=True)
 
    results_name = './discrimination/gen_results/parallel-final-random-search'
    results_dir = results_name + '/'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    alphabet = 'ACGT'

    cmd = []

    for explorer in ['evolutionary_run','wgan_run']: 
            for seed in range(100): 
                for idx, site in site_df.iterrows():          
                    cmd.append([explorer, results_dir, alphabet, site.target_set1_nt, site.target_set2_nt , site.seq_id, site.target1_name + '_vs_' + site.target2_name, seed])

    print(len(cmd))
    cmd = cmd[args.run_id]

    explorer = cmd[0]
    results_dir = cmd[1]
    alphabet = cmd[2]
    target_set1_nt = cmd[3]
    target_set2_nt = cmd[4]
    site_id = cmd[5] 
    seq_id = cmd[6] 
    seed = cmd[7]
    
    exec(f'{explorer}(results_dir, alphabet, target_set1_nt, target_set2_nt , site_id, seq_id, seed)')


    
        
    

