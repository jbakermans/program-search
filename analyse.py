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

def get_in_solution_human(solution, target):
    # Find which objects are included in the solution
    in_solution = []
    for i, (pos, shape) in enumerate(zip(solution, dsl.PRIMITIVES)):
        if (pos[0] + shape.shape[1] > 0 and pos[0] < len(target[0])
            and pos[1] + shape.shape[0] > 0 and pos[1] < len(target)):
            in_solution.append(i)
    # Return list of objects in solution, in order of object id
    return in_solution
        
def get_final_moves_human(in_solution, events):
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

def plot_trace_human(events, target):
    # Initialise all object positions at None, so they will be ignored
    pos = [None for _ in range(9)]
    # Each plot is a step towards the solution by the participant
    plots = []    
    # Extract steps towards solution: anything that comes or leaves overlap with target
    for curr_event in events:
        # Get shape, start, and stop
        shape_id = curr_event['id']
        start = [curr_event['press']['y'], curr_event['press']['x']]
        stop = [curr_event['release']['y'], curr_event['release']['x']]
        # Skip this event if nothing changed
        if start == stop:
            continue
        # If this moved to overlap position
        if dsl.do_overlap([target, dsl.PRIMITIVES[shape_id]], [[0,0], stop]):
            # Update position after event
            pos[shape_id] = stop
            # Then plot
            plots.append(plot_partial_programs_human(pos))
        else:
            # If this didn't move to overlap position: see if it came from overlap pos
            if dsl.do_overlap([target, dsl.PRIMITIVES[shape_id]], [[0,0], start]):
                # In that case update position to have moved out
                pos[shape_id] = None
                # And plot
                plots.append(plot_partial_programs_human(pos))
    return plots

def plot_partial_programs_human(pos):
    return dsl.combine([dsl.PRIMITIVES[i] * (i + 1) for i, p in enumerate(pos)
                              if p is not None], 
                             [p for p in pos
                              if p is not None]) \
        if any([p is not None for p in pos]) else np.array([[0]])

def get_order_model(solution, target):
    # Select object with lowest loss
    best_object = solution.program.objects()[
        np.argmax([target.IoU(p) for p in solution.program.objects()])]
    # Walk through solution nodes, and include them if they are in the best object
    return [n.index for n in solution.program.nodes 
            if n in best_object and isinstance(n, tetris.Primitive)]

def plot_trace_model(solution):
    # Get partial programs at each solution step
    pp = []
    for node in solution.program.nodes:
        # Add the current solution node, and copy over any previous nodes if not in current
        pp.append([node] + [p for p in (pp[-1] if len(pp) > 0 else []) if p not in node])
    # Get ceil sqrt of max nr of partial programs at any time for canvas rows, cols
    canvas_dim = int(np.ceil(np.sqrt(np.max([len(step) for step in pp]))))
    # Then build the canvas for each step by adding in each of the partial programs
    plots = []
    for pps in pp:
        # Get number of rows in current plot
        rows = int(np.ceil(len(pps) / canvas_dim))
        # Find height of each row in the current plot: max nr of rows across pps
        h = [max([p.render().shape[0] for p in 
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
                col = pos[-1][0] + prev_p.render().shape[1] + 1
            # The row value is given by the heights
            row = h[int(p_i / canvas_dim)]
            # Add row, col coordinate to position lists
            pos.append([row, col])
        # Now that all partial programs have positions, I can build the plots
        plots.append(dsl.combine([p.render() for p in pps], pos))    
    return plots

# Specify all timestamps of model data to load
model_paths = ['2022-10-18T16:40:42','2022-10-18T17:08:10']
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
        # Find which primitives ended up in solution
        in_solution = get_in_solution_human(data['trial_solution'][trial], data['trial_target'][trial])
        # Find final moves: the shapes in the solution dragged to their final position
        final_moves = get_final_moves_human(in_solution, data['trial_events'][trial])
        # Get the current target id, which is what data matrices are ordered by
        target_id = data['trial_target_ids'][trial]
        # Collect values
        human_correct[d_i, target_id] = data['trial_correct'][trial]
        human_time[d_i, target_id] = data['trial_time'][trial]/1000
        human_primitives[d_i, target_id] = len(in_solution) if human_correct[d_i, trial] else np.nan
        human_moves[d_i, target_id] = len(data['trial_events'][trial])
        # Store order
        human_order[d_i][target_id] = [m['id'] for m in final_moves]
        # Extract traces
        human_trace[d_i][target_id] = plot_trace_human(data['trial_events'][trial], 
                                                       np.array(data['trial_target'][trial]))
        
# And then for the model
for d_i, data in enumerate(model_data):
    for trial in range(n_trials):
        # Print progress
        print(f"Processing mod {d_i+1} / {len(data)}, trial {trial+1} / {n_trials}...")  
        # Get the primitives of this solution in order
        primitive_order = get_order_model(data[trial], task[trial])
        # Collect values
        model_correct[d_i, trial] = (data[trial].loss == 0)
        model_time[d_i, trial] = data[trial].time
        model_primitives[d_i, trial] = len(primitive_order) if model_correct[d_i, trial] else np.nan
        model_moves[d_i, trial] = data[trial].evaluations
        # Store order
        model_order[d_i][trial] = primitive_order     
        # Extract traces
        model_trace[d_i][trial] = plot_trace_model(data[trial])

# Plot results
plt.figure()
for r_i, (dat, name) in enumerate(zip(
        [[human_correct, human_time, human_primitives, human_moves],
         [model_correct, model_time, model_primitives, model_moves]],
        ['Human', 'Model'])):
    for c_i, (y, y_label) in enumerate(
            zip(dat, ['Correct (T/F)', 'Time (seconds)', '# Primitives (1)', '# Moves (1)'])):
        plt.subplot(2, 4, r_i * 4 + c_i + 1)
        plt.plot(np.arange(y.shape[1]), y.transpose(), color=(0.8, 0.8, 0.8))
        plt.errorbar(np.arange(y.shape[1]), np.nanmean(y, axis=0), np.nanstd(y, axis=0)/np.sqrt(y.shape[1]), color=(0,0,0))
        plt.ylabel(y_label)
        plt.title(name)
        #plt.title(names)
        if r_i == 1:
            plt.xlabel('Stimuli')  
          
# Plot some human and model traces for each trial
for d_i in range(n_trials):
    # Create colormap that shows how silhouettes are built from blocks
    from matplotlib import colors, cm
    # Get tab10 color map: discrete series of 10 colours
    tab10 = cm.get_cmap('tab10')
    # Then create a new colormap that starts with white
    cmap = colors.ListedColormap(['w'] + [tab10(i) for i in np.arange(10)])
    # Collect data for this trial
    curr_humans = random.sample(list(range(n_participants)), 2)
    curr_models = list(range(n_models))
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
            ['Human ' + str(h_i) + (' (correct)' if human_correct[h_i][d_i] 
                                    else ' (incorrect)')for h_i in curr_humans] +
            ['Model ' + str(m_i) + (' (correct)' if model_correct[m_i][d_i] 
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
    if True:
        plt.show()
        plt.savefig(f'/Users/jbakermans/Google Drive/DPhil/Presentations/Own/BMM2022/Traces/trace_{d_i:03}.png')
        plt.close()