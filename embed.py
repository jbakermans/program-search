#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:49:14 2023

@author: jbakermans
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.io import savemat

import tetris

def get_shape_patches(loc, unit, shape, shape_col=(0,0,0)):
    patches = []
    origin = loc - unit * (np.array(shape.shape[::-1]) / 2)
    for col, border in zip([(1,1,1), shape_col], [0.1,0]):
        # Run through rows in opposite order because of flipped y-axis
        for r_i, r in enumerate(shape[::-1]):
            for c_i, c in enumerate(r):
                if c:
                    patches.append(Rectangle(
                        [origin[0] + (c_i - border)*unit,
                         origin[1] + (r_i - border)*unit],
                        (1 + 2 * border)*unit,
                        (1 + 2 * border)*unit,
                        color=col))
    return patches
                  
# Load embedding file
filename = '/Users/jbakermans/Documents/Repos/program-search/experimentOutputs/20230403204724_6931/all.npy'
embeddings = np.load(filename, allow_pickle=True)
embeddings = embeddings[()]
embeddings = {key: val[54:] for key, val in embeddings.items()}

# Calculate correlation matrix
corrs = np.corrcoef(embeddings['embed'])
# Plot each spec and the shape similarities
f = plt.figure(figsize=(6,6))
# Create grid, where first row and first column will be shapes
n_rows=5
gs = f.add_gridspec(n_rows, n_rows)
# Find canvas size: biggest shape in both dimensions
res = max([max(e.shape) for e in embeddings['shape']])

# Finally: plot correlation matrix
ax = f.add_subplot(gs[1:,1:])
plt.imshow(corrs, extent=[1,len(embeddings['shape']), 1, len(embeddings['shape'])])
# First plot: all shapes above each other
ax_h = f.add_subplot(gs[1:,0], sharey=ax)
plt.imshow(np.concatenate(
    [tetris.padToFit(e, w=res, h=res) for e in embeddings['shape']], axis=0),
           cmap='Greys', vmin=0, vmax=1, extent=[0,1,1,len(embeddings['shape'])])
plt.xticks([])
plt.yticks([])   
# Second plot: all shapes next to each other
ax_v = f.add_subplot(gs[0,1:], sharex=ax)
plt.imshow(np.concatenate(
    [tetris.padToFit(e, w=res, h=res) for e in embeddings['shape']], axis=1),
           cmap='Greys', vmin=0, vmax=1, extent=[1,len(embeddings['shape']),0,1])
plt.xticks([])
plt.yticks([])   

# Do PCA
y = embeddings['embed'] - np.mean(embeddings['embed'],axis=0)
cov = np.matmul(y.transpose(), y)
w, v = np.linalg.eig(cov)
x = np.real(np.matmul(y, v[:,:2]))
x = x / np.max(np.abs(x))

# Plot each shape at coordinate - in reverse order, so primitives come on top
plt.figure()
ax = plt.axes()
for i, (coord, shape) in enumerate(
        zip(x[::-1], [tetris.padToFit(e, w=res, h=res) for e in embeddings['shape'][::-1]])):
    patches = get_shape_patches(coord, 0.01, shape, shape_col=(0,0,0))
                                #shape_col=(1,0,0) if i >= len(embeddings['shape'])-9 else (0,0,0))
    for p in patches:
        ax.add_patch(p)
ax.set_xlim([np.min(x[:,0])-0.05, np.max(x[:,0])+0.05])
ax.set_ylim([np.min(x[:,1])-0.05, np.max(x[:,1])+0.05])
ax.set_aspect('equal', adjustable='box')

# Calculate the three RDMs that I want to fit
RDM = {}
# 1. Negative embedding correlation, scaled between 0 and 1
RDM['model'] = 1 - (corrs - np.min(corrs)) / (np.max(corrs) - np.min(corrs))
# 2. Total absolute height/width difference
RDM['size'] = np.zeros((len(embeddings['shape']), len(embeddings['shape'])))
for i, s_i in enumerate(embeddings['shape']):
    for j, s_j in enumerate(embeddings['shape']):
        RDM['size'][i,j] = np.abs(s_i.shape[0] - s_j.shape[0]) \
            + np.abs(s_i.shape[1] - s_j.shape[1])
RDM['size'] = (RDM['size'] - np.min(RDM['size'])) / (np.max(RDM['size']) - np.min(RDM['size']))
# 3. IoU between each pair of shapes
RDM['pixel'] = np.zeros((len(embeddings['spec']), len(embeddings['spec'])))
objects = [tetris.parseString(s) for s in embeddings['spec']]
for i, o_i in enumerate(objects):
    print('finished ' + str(i))
    for j, o_j in enumerate(objects):
        RDM['pixel'][i,j] = o_i.IoU(o_j)
RDM['pixel'] = 1 - (RDM['pixel'] - np.min(RDM['pixel'])) / (np.max(RDM['pixel']) - np.min(RDM['pixel']))
# Plot all three RDMs, and also their correlation
plt.figure();
for i, (key, val) in enumerate(RDM.items()):
    plt.subplot(1, len(RDM.keys()) + 1, i + 1)
    plt.imshow(val)
    plt.title(key)
plt.subplot(1, len(RDM.keys()) + 1, len(RDM.keys()) + 1)
RDM_vals = np.stack([m[np.triu_indices(len(embeddings['shape'])-1)] for m in RDM.values()])
plt.imshow(np.corrcoef(RDM_vals), vmin=0, vmax=1)
plt.colorbar();
# Save as .mat file for loading in matlab 
savemat(filename.replace('all.npy','RDM.mat'), RDM)