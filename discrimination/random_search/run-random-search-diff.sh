#!/bin/sh

# This bash file runs the random search script, which randomly samples hyperparameters, designs guides, and evaluates them.
parallel --progress -j 85 --delay .1 'python random-search-diff.py {1}' ::: $(seq 0 5000)