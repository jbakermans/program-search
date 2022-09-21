#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:07:21 2022

@author: jbakermans
"""

import numpy as np
from lark import Lark

# Create primitives as global
PRIMITIVES = (
    np.array([[1,1],[1,0]]),
    np.array([[1,1],[0,1]]),
    np.array([[1,0],[1,1]]),
    np.array([[0,1],[1,1]]),
    np.array([[1]]),
    np.array([[1],[1]]),
    np.array([[1,1]]),
    np.array([[1],[1],[1]]),
    np.array([[1,1,1]])
    )

# Define language grammar in lark (https://www.lark-parser.org/ide/)
GRAMMAR = """
start: program | shape
shape: S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8
program: shape | horizontal | vertical

S0: "s0"
S1: "s1"
S2: "s2"
S3: "s3"
S4: "s4"
S5: "s5"
S6: "s6"
S7: "s7"
S8: "s8"

horizontal: "hor(" program "," program ";" SIGNED_INT ")"
vertical: "vert(" program "," program ";" SIGNED_INT ")"

%import common.SIGNED_INT
"""

# Parse syntax
PARSER = Lark(GRAMMAR)

# Concatenate two objects horizontally
def hor(a, b, shift=0):
    # If the shift is so big that b will miss to a: break immediately
    if a is None or b is None or shift <= -b.shape[0] or shift >= a.shape[0]:
        return None
    # Place object a at origin
    pos_a = [0, 0]
    # Squeeze object b towards object a until they overlap
    squeeze = 0
    while not do_overlap([a, b], [pos_a, [shift, a.shape[1] - (squeeze + 1)]]):
        squeeze += 1
    # Place object b next to object a, shifted vertically, squeezed horizontally
    pos_b = [shift, a.shape[1] - squeeze]
    # Return shape that combines the two
    return combine([a, b], [pos_a, pos_b])

# Concatenate two objects vertically
def vert(a, b, shift=0):
    # If the shift is so big that b will miss a: break immediately
    if a is None or b is None or shift <= -b.shape[1] or shift >= a.shape[1]:
        return None
    # Place object a at origin
    pos_a = [0, 0]
    # Squeeze object b towards object a until they overlap
    squeeze = 0
    while not do_overlap([a, b], [pos_a, [a.shape[0] - (squeeze + 1), shift]]):
        squeeze += 1
    # Place object b next to object a, shifted vertically, squeezed horizontally
    pos_b = [a.shape[0] - squeeze, shift]
    # Return shape that combines the two
    return combine([a, b], [pos_a, pos_b])
    
# Combine two shapes at given positions to new shape
def combine(shapes, positions):
    # Find coordinates of canvas that tightly fits the combined shape
    canvas_r = min([p[0] for p in positions])
    canvas_c = min([p[1] for p in positions])
    canvas_h = max([p[0] - canvas_r + s.shape[0] for s, p in zip(shapes, positions)])
    canvas_w = max([p[1] - canvas_c + s.shape[1] for s, p in zip(shapes, positions)])
    # Create empty canvas
    canvas = np.zeros([canvas_h, canvas_w])
    # Add shapes to canvas
    for s, p in zip(shapes, positions):
        # Find canvas locations to draw: positions, with canvas offset added
        r, c = p[0] - canvas_r, p[1] - canvas_c
        # Select part of canvas draw on
        draw_area = canvas[r:(r + s.shape[0]), c:(c + s.shape[1])]
        # And colour in all values that are 1 in the shape
        draw_area[s == 1] = 1
    # Return canvas with input shapes on it
    return canvas

# Find if given shapes at given positions have any overlap
def do_overlap(shapes, positions):
    # Calculate areas of separate and combined shapes
    separate_area = sum([np.sum(s) for s in shapes])
    combined_area = np.sum(combine(shapes, positions))
    # There is overlap if the separate area is larger than the combined area
    return combined_area < separate_area

# Create 'generative model' that enumerates all possible two-shape combinations
def generate(primitives=PRIMITIVES):
    # Build a long list of function-shape pairs
    all_shapes = []
    # First add primitives to list
    for i, s in enumerate(primitives):
        all_shapes.append(['s:' + str(i), s])
    # Then add all pairs of primitives
    for i1, s1 in enumerate(primitives):
        for i2, s2 in enumerate(primitives):
            for nf, f in enumerate([hor, vert]):
                for shift in range(-2, 3):
                    all_shapes.append([
                        'f:' + str(nf) + '_s1:' + str(i1) + '_s2:' + str(i2) + '_v:' + str(shift),
                        f(s1, s2, shift=shift)])
            print('Finished object pair (' + str(i1) + ', ' + str(i2) + ')')
    return all_shapes
    
# Take program sentence, parse to tree according to grammar, and execute
def run_program(program):
    parse_tree = PARSER.parse(program)
    return run_node(parse_tree.children[0])
    
# Specify what to do at each node of the parse tree
# The actual functions used here for primitives, hor, and vert can be provided!
# The only requirement is that primitives is a list, and hor and vert are functions
def run_node(t, primitives=PRIMITIVES, hor=hor, vert=vert):
    # If this is a shape: return primitive for that shape
    if t.data == 'shape':
        # Return primitive shape found in child node
        return primitives[int(t.children[0].value[-1])]
    # If this is vertical function: call vertical function with children as input
    elif t.data == 'vertical': 
        # Return vertical concatenation; final child node is shift argument
        return vert(run_node(t.children[0], primitives, hor, vert), 
                    run_node(t.children[1], primitives, hor, vert), 
                    shift=int(t.children[2].value))
    # If this is horizontal function: call vertical function with children as input
    elif t.data == 'horizontal': 
        # Return horizontal concatenation; final child node is shift argument
        return hor(run_node(t.children[0], primitives, hor, vert), 
                   run_node(t.children[1], primitives, hor, vert), 
                   shift=int(t.children[2].value))
    # In all other cases (like start or program nodes), simply continue down
    else:
        return run_node(t.children[0], primitives, hor, vert)