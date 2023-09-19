#!/bin/sh

# Repeatedly run the model-guided exploration algorithms to benchmark the consistency of th eguide generation

parallel --progress -j 85 --delay .1 'design_guides.py mult both $file ./path_to_pickle ./gen_results/mult_runs_benchmarking/run{1} --num_cpu 80 --verbose_results --processed_sites_path --save_pickled_results --output_to_single_directory' ::: $(seq 0 100)