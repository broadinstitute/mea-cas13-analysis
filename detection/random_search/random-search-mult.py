"""
This script randomly samples a set of hyperparameters for the evolutionary and WGAN explorers and runs them on a set of 100 sequences.

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

# Importing MEA packages
from mea.models.cas13_mult import Cas13Mult
from mea.models.cas13_diff import Cas13Diff

from mea.utils.cas13_landscape import Cas13Landscape
from mea.utils import prepare_sequences as prep_seqs
from mea.explorers.evolutionary import EvolutionaryExplorer
from mea.explorers.wgan_am import WGANExplorer
from mea.utils import import_fasta


def wgan_run(results_dir, alphabet, target_set, seq_id, seed):
    # Run the WGAN-AM algorithm with randomly sampled hyperparameters
    
    f = open(results_dir + "job-tracker.txt", "a")
    f.write("\nwgan_seed{}_{}".format(seed, seq_id))
    f.close()
    start = timeit.default_timer()

    try:
        np.random.seed(seed)


        learning_rate = 10**uniform.rvs(-2, 4, size=1)[0]
        outer_rounds = int(uniform.rvs(5, 30, size=1)[0])
        inner_rounds = int(uniform.rvs(50, 275, size=1)[0])

        optimizer = np.random.choice(['adam', 'sgd', 'rmsprop'])

        hyperparams = pd.DataFrame({'optimizer': [optimizer], 'learning-rate': [learning_rate], 'seed': [seed], 'seq_id': [seq_id], 'species': [seq_id.split('.')[0]],
        'outer_rounds': [outer_rounds], 'inner_rounds': [inner_rounds]})

        cas13landscape = Cas13Landscape()
        Cas13Mult = Cas13Mult(cas13landscape, target_set)
        baseline_fitness = Cas13Mult.get_fitness([Cas13Mult.target_cons_no_context_nt])

        wgan_explorer = WGANExplorer(
        Cas13Mult,
        rounds=outer_rounds,
        starting_sequence=Cas13Mult.target_cons_nt,
        inner_iterations=inner_rounds,
        adam_lr = learning_rate,
        optimizer = optimizer,
        sequences_batch_size=100,
        model_queries_per_batch=2000,
        )

        wgan_sequences, metadata = wgan_explorer.run(Cas13Mult, seq_id)
        
        save_rand_results(wgan_sequences, hyperparams, 'wgan', results_dir, baseline_fitness, start, target_set, Cas13Mult.target_cons_no_context_nt, seq_id)

    except Exception:
        f = open(results_dir + "job-tracker.txt", "a")
        f.write("\nERROR")
        f.write(traceback.format_exc())
        f.close()

def evolutionary_run(results_dir, alphabet, target_set, seq_id, seed):
    # Run the evolutionary algorithm with randomly sampled hyperparameters

    f = open(results_dir + "job-tracker.txt", "a")
    f.write("\nevolutionary_seed{}_{}".format(seed, seq_id)) 
    f.close()
    start = timeit.default_timer()

    try:
        np.random.seed(seed)
 
        model_queries_per_batch = 1500
        S = int(uniform.rvs(50, 250, size=1)[0])
        j = uniform.rvs(0.1, 0.8, size=1)[0]
        beta = 10**uniform.rvs(-1.2, 3, size=1)[0]
        
        gamma = uniform.rvs(0, 10, size=1)[0]/28

        hyperparams = pd.DataFrame({'seed': [seed], 'seq_id': [seq_id], 'species': [seq_id.split('.')[0]], 'S': [S],
        'j': [j],'beta': [beta], 'gamma': [gamma], 'model_queries': [model_queries_per_batch]})

        cas13landscape = Cas13Landscape()
        Cas13Mult = Cas13Mult(cas13landscape, target_set)
        starting_sequence = Cas13Mult.target_cons_no_context_nt
        baseline_fitness = Cas13Mult.get_fitness([Cas13Mult.target_cons_no_context_nt])

        evolutionary_explorer = EvolutionaryExplorer(
            Cas13Mult,
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

        evolutionary_seqs, metadata = evolutionary_explorer.run(Cas13Mult, seq_id)      

        save_rand_results(evolutionary_seqs, hyperparams, 'evolutionary', results_dir, baseline_fitness, start, target_set, Cas13Mult.target_cons_no_context_nt, seq_id)
    
    except Exception:
        f = open(results_dir + "job-tracker.txt", "a")
        f.write("\nERROR")
        f.write(traceback.format_exc())
        f.close()

def save_rand_results(df, hyperparams, algo, results_dir, baseline_fitness, start, target_set, baseline_guide, seq_id):
    stop = timeit.default_timer()
    # Save the results of the random search to a csv file

    hyperparams['runtime'] = (stop - start)/60

    start_df = df.iloc[0]
    guide_df = df.iloc[1:]
    best_guide_df = guide_df[guide_df.model_score == guide_df.model_score.max()].iloc[0]
    gen_guide_nt = best_guide_df.sequence 
    score = best_guide_df.model_score

    hyperparams['nearest_hd'] = min([prep_seqs.hamming_dist(gen_guide_nt, target[10:-10]) for target in target_set])
    hyperparams['fitness'] = score
    hyperparams['baseline_fitness'] = baseline_fitness

    hyperparams['delta_fitness'] = score - baseline_fitness

    hyperparams['sequence'] = gen_guide_nt
    hyperparams['baseline_guide'] = baseline_guide

    results_path = results_dir + 'rand-search_{}_{}.pkl'.format(algo, seq_id)

    if(path.exists(results_path)):
        time.sleep(np.random.choice(np.arange(10, 60)))
        old_results = pd.read_pickle(results_path)
        upd_results = old_results.append(hyperparams)
        upd_results.to_pickle(results_path)
        upd_results.to_csv(results_dir + 'rand-search_{}.csv'.format(algo))
    else:
        hyperparams.to_pickle(results_path)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('run_id',
        help=("number of the seq to run"),
        type = int)
    
    args = parser.parse_args()

    try:

        data_dir = '../processed_sites/100random_sites_withG/seqs_df.pkl'
        data_file = pd.read_pickle(data_dir)

        target_sets_nt = data_file['target_set'].values
        seq_ids = data_file['seq_id'].values

        results_name = '../gen_results/final-random-search'
        results_dir = results_name + '/'

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        if not os.path.exists(results_dir + "job-tracker.txt"):
            f = open(results_dir + "job-tracker.txt", "x")
            f.close()
        else:
            f = open(results_dir + "job-tracker.txt", "a")
            f.close()

        alphabet = 'ACGT'

        jobs = []
        all_jobs = []

        cmd = []
      
        # Randomly sample 100 sets of hyperparameters and design guides for each target set using this set of hyperparameters
        for seed in range(100):
            for i, target_set_nt in enumerate(target_sets_nt):
                for explorer in ['evolutionary_run', 'wgan_run']: 
                    cmd.append([explorer, results_dir, alphabet, target_set_nt , seq_ids[i], seed])

        print(len(cmd))  

        cmd = cmd[args.run_id]
        explorer = cmd[0]
        results_dir = cmd[1]
        alphabet = cmd[2]
        target_set_nt = cmd[3]
        seq_id = cmd[4]
        seed = cmd[5] 

        print(explorer)
        exec(f'{explorer}(results_dir, alphabet, target_set_nt , seq_id, seed)')

    except Exception:
        f = open(results_dir + "job-tracker.txt", "a")
        f.write("\nERROR")
        f.write(f'\nRUN ID {args.run_id}' )
        f.write(traceback.format_exc())
        f.close()
    
        
    

