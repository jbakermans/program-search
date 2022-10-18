#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:06:01 2022

@author: jbakermans
"""

import pickle
import os
import json
import tetris
import numpy as np
import dsl

def get_in_solution(solution, target):
    # Find which objects are included in the solution
    in_solution = []
    for i, (pos, shape) in enumerate(zip(solution, dsl.PRIMITIVES)):
        if (pos[0] + shape.shape[1] > 0 and pos[0] < len(target[0])
            and pos[1] + shape.shape[0] > 0 and pos[1] < len(target)):
            in_solution.append(i)
    # Return list of objects in solution, in order of object id
    return in_solution
        
def get_final_moves(in_solution, events):
    # Collect final move for all objects that are part of solution
    final_move = []
    for i in in_solution:
        for t, move in enumerate(events[::-1]):
            if move['id'] == i:
                # Swap x and y 
                final_move.append({'t': len(events) - t,
                                  'x': move['release']['x'], 
                                  'y': move['release']['y'],
                                  'id': i})
                break
    # Return final moves in order of appearance
    return sorted(final_move, key=lambda d: d['t'])

def get_model_order(solution, target):
    # Select object with lowest loss
    best_object = solution.program.objects()[
        np.argmax([target.IoU(p) for p in solution.program.objects()])]
    # Walk through solution nodes, and include them if they are in the best object
    return [n.index for n in solution.program.nodes 
            if n in best_object and isinstance(n, tetris.Primitive)]

# Specify all timestamps of model data to load
model_paths = ['2022-10-17T11:04:49']
# Load all of them
model_data = []
for curr_path in model_paths:
    # Load the pickle file
    result_file = pickle.load(open(os.path.join('./experimentOutputs', curr_path, 'testResults.pickle'), 'rb'))
    # Unpack it: pickle file contains names of optimisers and corresponding results
    names, results = zip(*result_file)
    # Only keep the SMC one
    results = [r for n, r in zip(names, results) if n == 'SMC_value'][0]
    # Then select only the final search result for each shape: the best solution
    results = [r[-1] for r in results]
    # And add to data list
    model_data.append(results)

# List files in participant data directory
human_files = sorted([i for i in os.listdir('../tetris-program-synthesis/data') if '.txt' in i])
# Get full path for each participant
human_paths = [os.path.join('../tetris-program-synthesis/data', p) for p in human_files]
# Load all of them
human_data = []
for filename in human_paths:
    # Open data file
    f = open(filename)
    # Parse string to dictionary
    data = json.load(f)
    # Close file
    f.close()
    # An append data
    human_data.append(data)

# Load and parse the shapes that participants & models had to solve
task = tetris.loadShapes('data/task.txt')

# Get number of model runs and number of participants
n_models = len(model_data)
n_participants = len(human_data)
# Get number of trials
n_trials = len(task)

# Initialise data matrices. Correct solution or wrong
human_correct = np.zeros((n_participants, n_trials))
# Time to finish trial
human_time = np.zeros((n_participants, n_trials))
# Number of moves human needed
human_moves = np.zeros((n_participants, n_trials))
# Number of primitives in solution
human_primitives = np.zeros((n_participants, n_trials))
# Order of primitives in solution
human_order = [[[] for _ in range(n_trials)] for _ in range(n_participants)]

# And the same for the model. Correct solution or wrong
model_correct = np.zeros((n_models, n_trials))
# Time to finish trial
model_time = np.zeros((n_models, n_trials))
# Number of moves human needed
model_moves = np.zeros((n_models, n_trials))
# Number of primitives in solution
model_primitives = np.zeros((n_models, n_trials))
# Order of primitives in solution
model_order = [[[] for _ in range(n_trials)] for _ in range(n_models)]

# Now collect all the entries for the human data
for d_i, data in enumerate(human_data):
    for trial in range(n_trials):
        # Print progress
        print(f"Processing sub {d_i+1} / {len(human_data)}, trial {trial+1} / {data['n_trials']}...")  
        # Find which primitives ended up in solution
        in_solution = get_in_solution(data['trial_solution'][trial], data['trial_target'][trial])
        # Find final moves: the shapes in the solution dragged to their final position
        final_moves = get_final_moves(in_solution, data['trial_events'][trial])
        # Get the current target id, which is what data matrices are ordered by
        target_id = data['trial_target_ids'][trial]
        # Collect values
        human_correct[d_i, target_id] = data['trial_correct'][trial]
        human_time[d_i, target_id] = data['trial_time'][trial]/1000
        human_primitives[d_i, target_id] = len(in_solution) if human_correct[d_i, trial] else np.nan
        human_moves[d_i, target_id] = len(data['trial_events'][trial])
        # Store order
        human_order[d_i][target_id] = [m['id'] for m in final_moves]
        
# And then for the model
for d_i, data in enumerate(model_data):
    for trial in range(n_trials):
        # Print progress
        print(f"Processing mod {d_i+1} / {len(data)}, trial {trial+1} / {n_trials}...")  
        # Get the primitives of this solution in order
        primitive_order = get_model_order(data[trial], task[trial])
        # Collect values
        model_correct[d_i, trial] = (data[trial].loss == 0)
        model_time[d_i, trial] = data[trial].time
        model_primitives[d_i, trial] = len(primitive_order) if model_correct[d_i, trial] else np.nan
        model_moves[d_i, trial] = data[trial].evaluations
        # Store order
        model_order[d_i][trial] = primitive_order