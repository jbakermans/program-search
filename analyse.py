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
import random
from matplotlib import pyplot as plt
from scipy.stats import kendalltau

def get_order_human(solution):
    # Get all primitives that are part of submitted solution
    submitted = [s['id'] for s in 
                 [e for e in solution if 'submit' in e.keys()][-1]['submit']['self']]
    # If the submission was just a primitive without any combination: return immediately
    if len(submitted) == 1: return submitted
    # Find last reset, to only include steps that build towards solution
    last_reset = [e for e in solution if e['type'] == 'reset'][-1]['time']
    # Then collect all combine events leading towards solution after last reset
    combine = [e for e in solution if e['time'] > last_reset and 'combine' in e.keys()]
    # Walk through all combine events, and add primitives as they are added to solution
    primitives_order = []
    for e in combine:
        for s in e['combine']['new']:
            if s['id'] in submitted and s['id'] not in primitives_order:
                primitives_order.append(s['id'])
    # If the submission was just a primitive without any combination: use submitted
    return primitives_order

def plot_trace_human(solution):
    # Get partial programs at each solution step
    pp = []
    for event in solution:
        # If this is a reset event (except for the initialisation): clear the pp
        if event['type'] == 'reset' and len(pp) > 0:
            pp.append([])
        if 'combine' in event.keys():
            # If the combined objects aren't in pp already: need to recruit them first
            if len(pp) == 0 or (event['combine']['self'] not in pp[-1] 
                                or event['combine']['other'] not in pp[-1]):
                pp.append([event['combine']['self'], event['combine']['other']] 
                          + [p for p in (pp[-1] if len(pp) > 0 else []) 
                             if p != event['combine']['self'] 
                             and p != event['combine']['other']])
            # Then add the current combination, and copy over any previous nodes            
            pp.append([event['combine']['new']] + [p for p in pp[-1]
                                                   if p != event['combine']['self'] and
                                                   p != event['combine']['other']])            
        if 'submit' in event.keys():
            # Add the final submitted solution to the partial programs, if it isn't there already
            if len(pp) == 0 or pp[-1] != [event['submit']['self']]:
                pp.append([event['submit']['self']])
    # Render all partial programs at each step
    pp_rendered = [[dsl.combine([dsl.PRIMITIVES[prim['id']] * (prim['id'] + 1) for prim in p], 
                               [[prim['y'], prim['x']] for prim in p])
                    for p in pps] if len(pps) > 0 else [] 
                   for pps in pp]
    # Return a list of canvases with all partial programs plotted together at each step
    return plot_partial_programs_combined(pp_rendered)
    

def get_order_model(solution, target):
    # Select object with lowest loss
    best_object = solution.program.objects()[
        np.argmax([target.IoU(p) for p in solution.program.objects()])]
    # Walk through solution nodes, and include them if they are in the best object
    order = []
    for n in solution.program.nodes:
        if n in best_object and isinstance(n, tetris.Primitive) and not n.index in order:
            order.append(n.index)
    return order

def plot_trace_model(solution):
    # Get partial programs at each solution step
    pp = []
    for node in solution.program.nodes:
        # Add the current solution node, and copy over any previous nodes if not in current
        pp.append([node] + [p for p in (pp[-1] if len(pp) > 0 else []) if p not in node])
    # Render all partial programs at each step
    pp_rendered = [[p.render() for p in pps] for pps in pp]
    # Return a list of canvases with all partial programs plotted together at each step
    return plot_partial_programs_combined(pp_rendered)

def plot_partial_programs_combined(pp_rendered):
    # Get ceil sqrt of max nr of partial programs at any time for canvas rows, cols
    canvas_dim = int(np.ceil(np.sqrt(np.max([len(step) for step in pp_rendered]))))
    # Output plots all partial programs at each step on one canvas
    plots = []
    for pps in pp_rendered:
        # Only set positions if partial program set is not empty
        if len(pps) > 0:
            # Get number of rows in current plot
            rows = int(np.ceil(len(pps) / canvas_dim))
            # Find height of each row in the current plot: max nr of rows across pps
            h = [max([p.shape[0] for p in 
                      pps[(curr_row * canvas_dim):((curr_row + 1 ) * canvas_dim)]]) + 1
                 for curr_row in range(rows)]
            # To use height as coordinate, start at 0
            h = [0] + h
            # Then calculate position for each: row given by heights, col by previous
            pos = []
            for p_i, prev_p in enumerate([-1] + pps[:-1]):
                # If this is the first column: col value is 0
                if p_i % canvas_dim == 0:
                    col = 0
                else:
                    # Col value given by previous col plus previous width + 1
                    col = pos[-1][0] + prev_p.shape[1] + 1
                # The row value is given by the heights
                row = h[int(p_i / canvas_dim)]
                # Add row, col coordinate to position lists
                pos.append([row, col])
            # Add final list of positions to pp positions
            plots.append(dsl.combine([p for p in pps], pos))
        else:
            # Empty positions for empty partial program set
            plots.append(np.array([[0]]))
    return plots

def match_sequence(orders):
    # Get number of shapes and number of participants
    n_participants = len(orders)
    n_trials = len(orders[0])
    # Create a big matrix of N * (N-1) / 2 pairs of kendall tau correlations for each shape
    corrs = np.full((int(n_participants * (n_participants - 1) / 2), n_trials), np.nan)
    # For each shape: run through all correlations
    for curr_shape in range(n_trials):
        c = -1
        for i in range(n_participants):
            for j in range(i + 1, n_participants):
                c += 1
                # Collect sequences to compare
                seq1 = orders[i][curr_shape]
                seq2 = orders[j][curr_shape]
                # Remove anything that they don't have both
                seq1 = [s for s in seq1 if s in seq2]
                seq2 = [s for s in seq2 if s in seq1]
                # Set sequence 1 to the index of appearance in 2
                seq1 = [seq2.index(s) for s in seq1]
                # And then sequence 2 is simply in order
                seq2 = [i for i in range(len(seq2))]
                # Then calculate kendall tau rank correlation
                tau, _ = kendalltau(seq1, seq2)
                # Only add if finite
                if np.isfinite(tau): corrs[c, curr_shape] = tau
    return corrs

def match_shapes(orders):
    # Get number of shapes and number of participants
    n_participants = len(orders)
    n_trials = len(orders[0])
    # Create a big matrix of N * (N-1) / 2 pairs of Szymkiewicz–Simpson for each shape
    overlap = np.full((int(n_participants * (n_participants - 1) / 2), n_trials), np.nan)
    # For each shape: run through all correlations
    for curr_shape in range(n_trials):
        c = -1
        for i in range(n_participants):
            for j in range(i + 1, n_participants):
                c += 1
                # Collect sequences to compare
                seq1 = orders[i][curr_shape]
                seq2 = orders[j][curr_shape]
                # Find intersection: shapes in both of the sequences
                intersection = set(seq1).intersection(set(seq2))
                # Calculate https://en.wikipedia.org/wiki/Overlap_coefficient
                curr_ss = len(intersection) / min(len(seq1), len(seq2))
                # Only add if finite
                if np.isfinite(curr_ss): overlap[c, curr_shape] = curr_ss
    return overlap    

# Specify all timestamps of model data to load
model_paths = sorted([i for i in os.listdir('experimentOutputs') if '2022-12-07' in i])
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
human_files = sorted([i for i in os.listdir('data/human') if '.txt' in i])
# Get full path for each participant
human_paths = [os.path.join('data/human', p) for p in human_files]
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
# Human traces: shape at each step towards the solution
human_trace = [[[] for _ in range(n_trials)] for _ in range(n_participants)]

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
# Model traces: shape at each step towards the solution
model_trace = [[[] for _ in range(n_trials)] for _ in range(n_models)]

# Now collect all the entries for the human data
for d_i, data in enumerate(human_data):
    for trial in range(n_trials):
        # Print progress
        print(f"Processing sub {d_i+1} / {len(human_data)}, trial {trial+1} / {data['n_trials']}...")  
        # Get the primitives of this solution in order
        primitive_order = get_order_human(data['trial_events'][trial])
        # Get the current target id, which is what data matrices are ordered by
        target_id = data['trial_target_ids'][trial]
        # Collect values
        human_correct[d_i, target_id] = data['trial_correct'][trial]
        human_time[d_i, target_id] = data['trial_time'][trial]/1000
        human_primitives[d_i, target_id] = len(primitive_order) if human_correct[d_i, trial] else np.nan
        human_moves[d_i, target_id] = len(data['trial_events'][trial])
        # Store order
        human_order[d_i][target_id] = primitive_order
        # Extract traces
        human_trace[d_i][target_id] = plot_trace_human(data['trial_events'][trial])
# Calculate shape and sequence overlaps between all pairs of participants
human_match_shape = match_shapes(human_order)
human_match_sequence = match_sequence(human_order)
        
# And then for the model
for d_i, data in enumerate(model_data):
    for trial in range(n_trials):
        # Print progress
        print(f"Processing mod {d_i+1} / {len(model_data)}, trial {trial+1} / {n_trials}...")  
        # Get the primitives of this solution in order
        primitive_order = get_order_model(data[trial], task[trial])
        # Collect values
        model_correct[d_i, trial] = (1 - data[trial].loss)
        model_time[d_i, trial] = data[trial].time
        model_primitives[d_i, trial] = len(primitive_order) if model_correct[d_i, trial] else np.nan
        model_moves[d_i, trial] = data[trial].evaluations
        # Store order
        model_order[d_i][trial] = primitive_order
        # Extract traces
        model_trace[d_i][trial] = plot_trace_model(data[trial])
# Calculate shape and sequence overlaps between all pairs of models
model_match_shape = match_shapes(model_order)
model_match_sequence = match_sequence(model_order)

# Plot basic performance results
plt.figure()
for r_i, (dat, name) in enumerate(zip(
        [[human_correct, human_time, human_primitives, human_moves],
         [model_correct, model_time, model_primitives, model_moves]],
        ['Human', 'Model'])):
    for c_i, (y, y_label, y_lim) in enumerate(
            zip(dat, ['Correct (T/F)', 'Time (seconds)', '# Primitives (1)', '# Moves (1)'],
                [[0, 1], [], [2, 6], [2, 20]])):
        plt.subplot(2, 4, r_i * 4 + c_i + 1)
        if c_i == 2: 
            plt.plot([sum([p() in t for p in tetris.Primitives]) for t in task], 'r.')
        plt.plot(np.arange(y.shape[1]), y.transpose(), color=(0.8, 0.8, 0.8))
        plt.errorbar(np.arange(y.shape[1]), np.nanmean(y, axis=0), np.nanstd(y, axis=0)/np.sqrt(y.shape[0]), color=(0,0,0))
        plt.ylabel(y_label)
        if len(y_lim) > 0: plt.ylim(y_lim)
        plt.title(name)
        if r_i == 1:
            plt.xlabel('Stimuli')  
            
# Plot population similarity results
plt.figure()
for r_i, (dat, name) in enumerate(zip(
        [[model_match_shape, model_match_sequence],
         [model_match_shape, model_match_sequence]],
        ['Human', 'Model'])):
    for c_i, (y, y_label, y_lim) in enumerate(
            zip(dat, ['Shape match', 'Sequence match'],
                [[0, 1], [-1, 1]])):
        plt.subplot(2, 2, r_i * 2 + c_i + 1)        
        plt.violinplot(dataset=[y[np.isfinite(y[:, i]), i] for i in range(n_trials)],
                       positions=range(n_trials),
                       widths=0.9)
        plt.ylabel(y_label)
        if len(y_lim) > 0: plt.ylim(y_lim)
        plt.title(name)
        if r_i == 1:
            plt.xlabel('Stimuli')          
          
# Plot some human and model traces for each trial
for d_i in range(0):
    # Create colormap that shows how silhouettes are built from blocks
    from matplotlib import colors, cm
    # Get tab10 color map: discrete series of 10 colours
    tab10 = cm.get_cmap('tab10')
    # Then create a new colormap that starts with white
    cmap = colors.ListedColormap(['w'] + [tab10(i) for i in np.arange(10)])
    # Collect data for this trial
    curr_humans = random.sample(list(range(n_participants)), 1)
    curr_models = random.sample(list(range(n_models)), 3)
    human_dat = [human_trace[h_i][d_i] for h_i in curr_humans]
    model_dat = [model_trace[m_i][d_i] for m_i in curr_models]
    # After selecting data we know nr of rows (first row has spec)
    rows = len(curr_humans) + len(curr_models) + 1
    plt.figure()
    # First row: spec
    ax = plt.subplot(rows, 1, 1)
    ax.imshow(task[d_i].execute(), cmap='Greys', vmin=0, vmax=1)
    ax.axis('off')
    ax.set_title('Spec ' + str(d_i))
    # Second and third row: human and model trace
    for row, (dat, name) in enumerate(zip(
            human_dat + model_dat, 
            ['Human ' + str(h_i) + (' (correct)' if human_correct[h_i][d_i] == 1 
                                    else ' (incorrect)')for h_i in curr_humans] +
            ['Model ' + str(m_i) + (' (correct)' if model_correct[m_i][d_i] == 1
                                    else ' (incorrect)') for m_i in curr_models])):
        # Get maximum width & height across data for plotting
        h, w = [max([p.shape[0] for p in dat]), max([p.shape[1] for p in dat])]
        for i, curr_plot in enumerate(dat):
            # Create suplot
            ax = plt.subplot(rows, len(dat), len(dat) * (row + 1) + 1 + i)
            # Plot rendered shape
            ax.imshow(tetris.padToFit(curr_plot, w=w, h=h), cmap=cmap, vmin=0, vmax=10)
            # Create clean canvas
            ax.axis('off')
            # Set title of first column
            if i == 0:
                ax.set_title(name)
    # Optional: save each figure, then close
    if False:
        plt.show()
        plt.savefig(f'/Users/jbakermans/Google Drive/DPhil/Presentations/Own/BMM2022/Traces/trace_{d_i:03}.png')
        plt.close()