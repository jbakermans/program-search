#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:49:04 2022

@author: jbakermans
"""

import os
import pickle
import numpy as np
import traceback
import sys
import os
import time
import random
import time
import random
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from API import *
from a2c import *
from randomSolver import *
from pointerNetwork import *
from programGraph import *
from SMC import *
from ForwardSample import *
from MCTS import MCTS
from beamSearch import *
from CNN import *

import dsl

RESOLUTION = 16

import torch
import torch.nn as nn

class Tetris(Program):
    lexicon = ['h', 'v'] \
        + [str(i) for i in range(9)] \
            + [i - int(RESOLUTION / 2) for i in range(RESOLUTION + 1)]

    def __init__(self):
        self._rendering = None

    def __repr__(self):
        return str(self)

    def __ne__(self, o): 
        return not (self == o)

    def execute(self):
        # Only render if this has not been rendered before. Pad to RESOLUTION
        if self._rendering is None: self._rendering = padToFit(1.0 * (self.render() > 0))
        return self._rendering

    def clearRendering(self):
        # Clear all existing renders - e.g. when storing programs as training data
        self._rendering = None
        for c in self.children(): c.clearRendering()

    def IoU(self, other):
        # Calculate 'Intersection over Union' between self and other object
        if isinstance(other, Tetris): other = other.execute()
        # If one of the shapes can't be rendered: return 0 IoU
        if other is None or self.execute() is None: 
            return 0
        else:
            # If any of the two shapes is just zeros everywhere: return 0 IoU
            if np.all(other == 0) or np.all(self.execute() == 0):
                return 0
        # Pad other by own size so they can be convolved
        other = np.pad(other,[[other.shape[0], other.shape[0]],
                              [other.shape[1], other.shape[1]]])
        # Make list of shapes; mirror other for convolution
        shapes = [self.execute(), np.flipud(np.fliplr(other))]
        # Find number of pixels in both self AND other for all shifts
        both = convolve2d(shapes[0], shapes[1])
        # Find number of pixels in either self OR (= 1-AND) other for all shifts
        either = max([np.prod(s.shape) for s in shapes]) \
            - convolve2d(1 - shapes[0], 1 - shapes[1], fillvalue=1)
        # Return the maximum ratio of both over either pixels
        return np.max(both[either>0] / either[either>0])
    
    def render(self):
        # Render function will depend on operation; return None by default
        return None    

# The type of CSG's
tTetris = BaseType(Tetris)

class Primitive(Tetris):
    # Token needs to be set in shape-specific class
    token=None
    type = tTetris
    
    def __init__(self):
        super().__init__()
        self.index = int(self.token)
        self.shape = dsl.PRIMITIVES[self.index]
        
    def toTrace(self): return [self]

    def __str__(self):
        return 's' + self.token

    def children(self): return []

    def __eq__(self, o):
        return isinstance(o, Primitive) and o.index == self.index

    def __hash__(self):
        return hash((self.token))
    
    def __contains__(self, p):
        # Use contains to find if queried program occurs anywhere in full program
        return p == self

    def serialize(self):
        return (self.token)
    
    def render(self):
        return self.shape * (self.index + 1)
    
# Make all the primitives. Would be nice to do this in a function, then loop
# (class factory pattern) but then I can't pickle, because the class is local
class S0(Primitive):
    token = '0'
class S1(Primitive):
    token = '1'
class S2(Primitive):
    token = '2'
class S3(Primitive):
    token = '3'
class S4(Primitive):
    token = '4'
class S5(Primitive):
    token = '5'
class S6(Primitive):
    token = '6'
class S7(Primitive):
    token = '7'
class S8(Primitive):
    token = '8'

# Make a list of all primitive classes for easy access
Primitives = [S0, S1, S2, S3, S4, S5, S6, S7, S8]

class Hor(Tetris):
    token = 'h'
    type = arrow(tTetris, tTetris, 
                 integer(-int(RESOLUTION / 2), RESOLUTION - int(RESOLUTION / 2)),
                 tTetris)
    
    def __init__(self, a, b, shift=0):
        super(Hor, self).__init__()
        self.elements = [a,b]
        self.shift = shift

    def toTrace(self):
        return self.elements[0].toTrace() + self.elements[1].toTrace() + [self]

    def __str__(self):
        return f"hor({str(self.elements[0])},{str(self.elements[1])};{str(self.shift)})"

    def children(self): return self.elements

    def serialize(self):
        return (self.token, self.elements[0], self.elements[1], self.shift)

    def __eq__(self, o):
        return isinstance(o, Hor) and tuple(o.elements) == tuple(self.elements) \
            and o.shift == self.shift

    def __contains__(self, p):
        # Use contains to find if queried program occurs anywhere in full program
        if p == self: 
            return True
        else: 
            return any(p in e for e in self.elements)

    def __hash__(self):
        return hash((self.token, tuple(self.elements)))
    
    def render(self):
        return None if any(s() in self.elements[0] and s() in self.elements[1] 
                           for s in Primitives) \
            else dsl.hor(self.elements[0].render(), 
                         self.elements[1].render(), 
                         shift=self.shift)
            
class Vert(Tetris):
    token = 'v'
    type = arrow(tTetris, tTetris, 
                 integer(-int(RESOLUTION / 2), RESOLUTION - int(RESOLUTION / 2)),
                 tTetris)
    
    def __init__(self, a, b, shift=0):
        super(Vert, self).__init__()
        self.elements = [a,b]
        self.shift = shift

    def toTrace(self):
        return self.elements[0].toTrace() + self.elements[1].toTrace() + [self]

    def __str__(self):
        return f"vert({str(self.elements[0])},{str(self.elements[1])};{str(self.shift)})"

    def children(self): return self.elements

    def serialize(self):
        return (self.token, self.elements[0], self.elements[1], self.shift)

    def __eq__(self, o):
        return isinstance(o, Vert) and tuple(o.elements) == tuple(self.elements) \
            and o.shift == self.shift
            
    def __contains__(self, p):
        # Use contains to find if queried program occurs anywhere in full program
        if p == self: 
            return True
        else: 
            return any(p in e for e in self.elements)            

    def __hash__(self):
        return hash((self.token, tuple(self.elements)))
    
    def render(self):
        return None if any(s() in self.elements[0] and s() in self.elements[1] 
                           for s in Primitives) \
            else dsl.vert(self.elements[0].render(), 
                          self.elements[1].render(), 
                          shift=self.shift)

# To specify a DSL, I will need to create a list of operators, one for each primitives
tDSL = DSL([Hor, Vert] + Primitives,
          lexicon=Tetris.lexicon)

""" Loading shapes from file """
# Load all shapes from a text file, with one line on each path
def loadShapes(path):
    # Path is allowed to be list of paths; if it's a single path, convert to list
    path = path if isinstance(path, list) else [path]
    # Load shapes from each line in each file on the path
    shapes = []
    for file_path in path:
        # Open file at path
        with open(file_path, "r") as handle:
            source = handle.readlines()
        # Run through each line one by one, trying to parse it
        for line, program in enumerate(source):
            try:
                shapes.append(parseString(program.strip()))
            except:
                print(f"Can't parse {file_path}, line {str(line)}: {program.strip()}")
    return shapes

# Parse string from grammar to tetris object
def parseString(program):
    # Parse string into tree using grammar definition in dsl
    parse_tree = dsl.PARSER.parse(program)
    # Parse tree into object
    return dsl.run_node(parse_tree.children[0], 
                        primitives=[p() for p in Primitives], hor=Hor, vert=Vert)

"""Small utility functions"""
def padToFit(dat, val=0, w=RESOLUTION, h=RESOLUTION, center=True):
    # If input is None: output is empty canvas
    if dat is None: return np.zeros((h, w))
    # Pad input array with zeros to achieve input shape
    h_add = max([h - dat.shape[0], 0])
    h_add_0 = int(h_add / 2) if center else h_add
    h_add_1 = h_add - h_add_0
    w_add = max([w - dat.shape[1], 0])
    w_add_0 = int(w_add / 2) if center else w_add
    w_add_1 = w_add - w_add_0
    # The resulting padded array has the input roughly in its center
    return np.pad(dat, [[h_add_0, h_add_1],[w_add_0, w_add_1]], 
                  mode='constant', constant_values=val)

def plotShape(s, filename=None):
    # Plot rendered shape
    plt.imshow(s.execute(), cmap='Greys', vmin=0, vmax=1)
    # Create clean canvas
    plt.xticks([])
    plt.yticks([])       
    # Set title to program that generates shape
    plt.title(str(s))
    # If filename is provided: export figure
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
        
def plotSolution(s, filename=None):
    from matplotlib import colors, cm
    # Get tab10 color map: discrete series of 10 colours
    tab10 = cm.get_cmap('tab10')
    # Then create a new colormap that starts with white
    cmap = colors.ListedColormap(['w'] + [tab10(i) for i in np.arange(10)])
    # Plot rendered shape
    plt.imshow(padToFit(s.render()), cmap=cmap, vmin=0, vmax=10)
    # Create clean canvas
    plt.xticks([])
    plt.yticks([])       
    # Set title to program that generates shape
    plt.title(str(s))
    # If filename is provided: export figure
    if filename:
        plt.savefig(filename)
    else:
        plt.show()        

"""Neural networks"""
class ObjectEncoder(CNN):
    """Encodes a 2d object."""
    def __init__(self):
        super(ObjectEncoder, self).__init__(channels=2,
                                            inputImageDimension=RESOLUTION,
                                            filterSizes=[3,3,3,3],
                                            poolSizes=[2,2,1,1],
                                            numberOfFilters=[32,32,32,16])    
                                        
    def forward(self, spec, obj):
        if isinstance(spec, list):
            # batching both along specs and objects
            assert isinstance(obj, list)
            B = len(spec)
            assert len(obj) == B
            return super(ObjectEncoder, self).forward(np.stack([np.stack([s,o]) for s,o in zip(spec, obj)]))
        elif isinstance(obj, list): # batched - expect a single spec and multiple objects
            spec = np.repeat(spec[np.newaxis,:,:],len(obj),axis=0)
            obj = np.stack(obj)
            return super(ObjectEncoder, self).forward(np.stack([spec, obj],1))
        else: # not batched
            return super(ObjectEncoder, self).forward(np.stack([spec, obj]))

class SpecEncoder(CNN):
    """Encodes a 2d spec."""
    def __init__(self):
        super(SpecEncoder, self).__init__(channels=1,
                                          inputImageDimension=RESOLUTION,
                                          filterSizes=[3,3,3,3],
                                          poolSizes=[2,2,1,1],
                                          numberOfFilters=[32,32,32,16])


"""Training"""
def randomScene(maxShapes=5, minShapes=1, verbose=False, export=None):
    # Choose number of shapes to include
    desiredShapes = np.random.randint(minShapes, maxShapes + 1)
    # Randomly sample from primitives
    currPrimitives = random.sample([p for p in Primitives], 
                                  min(desiredShapes, len(Primitives)))
    # Generate initial shape
    s=currPrimitives[0]()
    for currPrimitive in currPrimitives[1:]:
        # Add next primitive to list of arguments for current step
        o = [s, currPrimitive()]
        # Shuffle objects randomly, so which goes where is random
        np.random.shuffle(o)
        # Get shapes of rendered version of both, because that will constrain shift
        d = [i.render().shape for i in o]        
        # Randomly choose between horizontal and vertical concatenation
        if np.random.rand() > 0.5:
            # Horizontal concetanation, with random shift depending on heights
            s = Hor(o[0], o[1], shift=np.random.randint(
                max(-d[1][0] + 1, -int(RESOLUTION / 2)), 
                min(d[0][0], RESOLUTION - int(RESOLUTION / 2))))
        else:
            # Vertical concetanation, with random shift depending on widths
            s = Vert(o[0], o[1], shift=np.random.randint(
                max(-d[1][1] + 1, -int(RESOLUTION / 2)),
                min(d[0][1], RESOLUTION - int(RESOLUTION / 2))))
    if verbose:
        print(s)
        print(ProgramGraph.fromRoot(s, oneParent=True).prettyPrint())
        plotShape(s)
    if export:
        plotShape(s, export)
    return s

def trainCSG(m, getProgram, trainTime=None, checkpoint=None):
    print("cuda?",m.use_cuda)
    assert checkpoint is not None, "must provide a checkpoint path to export to"
    sys.stdout.flush()
    
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    
    startTime = time.time()
    reportingFrequency = 100
    totalLosses = []
    movedLosses = []
    iteration = 0

    B = 16

    while trainTime is None or time.time() - startTime < trainTime:
        sys.stdout.flush()
        ss = [getProgram() for _ in range(B)]
        ls = m.gradientStepTraceBatched(optimizer, [(s, s.toTrace())
                                                    for s in ss])
        for l in ls:
            totalLosses.append(sum(l))
            movedLosses.append(sum(l)/len(l))
        iteration += 1
        if iteration%reportingFrequency == 1:
            print(f"\n\nAfter {iteration*B} training examples...\n\tTrace loss {sum(totalLosses)/len(totalLosses)}\t\tMove loss {sum(movedLosses)/len(movedLosses)}\n{iteration*B/(time.time() - startTime)} examples/sec\n{iteration/(time.time() - startTime)} grad steps/sec")
            totalLosses = []
            movedLosses = []
            torch.save(m, checkpoint)

def testCSG(m, getProgram, timeout, timestamp, solvers, solverSeed=0, n_test=30):
    random.seed(0)
    oneParent = m.oneParent
    print(f"One parent restriction?  {oneParent}")
    _solvers = {"SMC": lambda : SMC(m),
                "bm": lambda : BeamSearch(m),
                "fs": lambda : ForwardSample(m),
                "bmv": lambda : BeamSearch(m,criticCoefficient=1.),
                "noExecution": lambda : ForwardSample_noExecution(m),
                "noExecution_bm": lambda : Beam_noExecution(m)}
    solvers = [_solvers[s]() for s in solvers]
    loss = lambda spec, program: 1-max( o.IoU(spec) for o in program.objects() ) if len(program) > 0 else 1.

    testResults = [[] for _ in solvers]

    outputDirectory = f"experimentOutputs/{timestamp}"
    os.system(f"mkdir  -p {outputDirectory}")

    if isinstance(getProgram, list):
        specs = getProgram
    else:
        specs = [getProgram() for _ in range(n_test) ]

    random.seed(solverSeed)
    torch.manual_seed(solverSeed + 1)
    np.random.seed(solverSeed + 2)
    
    for ti,spec in enumerate(specs):
        #instrumentSMC(str(ti))
        print("Trying to explain the program:")
        print(ProgramGraph.fromRoot(spec, oneParent=oneParent).prettyPrint())
        print()
        
        plotShape(spec, "%s/%03d.png"%(outputDirectory,ti))
        with open("%s/%03d_spec.pickle"%(outputDirectory,ti),"wb") as handle:
            pickle.dump(spec, handle)
        for n, solver in enumerate(solvers):
            print(f"Running solver {solver.name}")
            solver.maximumLength = len(ProgramGraph.fromRoot(spec).nodes) + 1
            testSequence = solver.infer(spec, loss, timeout)
            if len(testSequence) == 0:
                testSequence = [SearchResult(ProgramGraph([]), 1., 0., 1)]
            testResults[n].append(testSequence)
            for result in testSequence:
                print(f"After time {result.time}, achieved loss {result.loss} w/")
                print(result.program.prettyPrint())
                print()
            if len(testSequence) > 0:
                obs = testSequence[-1].program.objects()
                if len(obs) == 0:
                    bestProgram = None
                else:
                    bestProgram = max(obs, key=lambda bp: bp.IoU(spec))
                with open("%s/%03d_%s.pickle"%(outputDirectory,ti,solver.name),"wb") as handle:
                    pickle.dump(bestProgram, handle)
                plotShape(bestProgram,
                              "%s/%03d_%s.png"%(outputDirectory,ti,solver.name))                

    names = [s.name for s in solvers]
    with open(f"{outputDirectory}/testResults.pickle","wb") as handle:
        pickle.dump(list(zip(names, testResults)),handle)
    print(f"Exported to:\n{outputDirectory}/testResults.pickle")
    plotTestResults(testResults, timeout,
                    defaultLoss=1.,
                    names=names,
                    export=f"{outputDirectory}/curve.png")

def plotTestResults(testResults, timeout, defaultLoss=None,
                    names=None, export=None):
    def averageLoss(n, predicate):
        results = testResults[n] # list of list of results, one for each test case
        # Filter out results that occurred after time T
        results = [ [r for r in rs if predicate(r)]
                    for rs in results ]
        losses = [ min([defaultLoss] + [r.loss for r in rs]) for rs in results ]
        return sum(losses)/len(losses)

    plot.figure()
    plot.xlabel('Time')
    plot.ylabel('Average Loss')
    plot.ylim(bottom=0.)
    for n in range(len(testResults)):
        xs = list(np.arange(0,timeout,0.1))
        plot.plot(xs, [averageLoss(n,lambda r: r.time < x) for x in xs],
                  label=names[n])
    plot.legend()
    if export:
        plot.savefig(export)
    else:
        plot.show()
    plot.figure()
    plot.xlabel('Evaluations')
    plot.ylabel('Average Loss')
    plot.ylim(bottom=0.)
    for n in range(len(testResults)):
        xs = list(range(max(r.evaluations for tr in testResults[n] for r in tr )))
        plot.plot(xs, [averageLoss(n,lambda r: r.evaluations < x) for x in xs],
                  label=names[n])
    plot.legend()
    if export:
        plot.savefig(f"{export}_evaluations.png")
    else:
        plot.show()
        
def makeTrainingData():
    data = {} # Map from image to (size, {programs})
    lastUpdate = time.time()
    n_samples = 0
    maximumSize = 0
    startTime = time.time()
    while time.time() < startTime + 3600*5:
        n_samples += 1
        # Sample new random program
        program = randomScene(maxShapes=len(Primitives), minShapes=3)
        # Get the size: steps to build this program
        size = len(program.toTrace())
        # Execute program to obtain rendered shape
        im = program.execute()
        # Convert rendered shape to tuple, so it can be used as dictionary key
        im = tuple( bool(im[x,y])
                    for x in range(RESOLUTION)
                    for y in range(RESOLUTION) )
        # Clear the rendering so it the program can be neatly stored away
        program.clearRendering()
        if im in data:
            # If the rendered shape was found before: only keep shortest programs
            oldSize, oldPrograms = data[im]
            if oldSize < size:
                # New program is longer: ignore new program
                pass
            elif oldSize == size:
                # New program is same length: add to existing programs
                data[im] = (size, {program}|oldPrograms)
            elif oldSize > size:
                # New program is shorter: only keep new program
                data[im] = (size, {program})
            else:
                assert False
        else:
            # If the rendered shape is completely new: add program to data 
            data[im] = (size, {program})
            maximumSize = max(maximumSize,size)

        if time.time() > lastUpdate + 10:
            print(f"After {n_samples} samples; {n_samples/(time.time() - startTime)} samples per second")
            for sz in range(maximumSize):
                n = sum(size == sz
                        for (size,_) in data.values() )
                print(f"{n} images w/ programs of size {sz}")
            print()
            lastUpdate = time.time()

    with open('data/tetris_data.p','wb') as handle:
        pickle.dump(data, handle)        

def getTrainingData(path):
    import copy
    
    with open(path,'rb') as handle:
        data = pickle.load(handle)
    print(f"Loaded {len(data)} images from {path}")
    print(f"Contains {sum(len(ps) for _,ps in data.values() )} programs")
    data = [list(ps) for _,ps in data.values()]

    def getData():
        programs = random.choice(data)
        # make a deep copy because we are caching the renders, and we want these to be garbage collected
        p = copy.deepcopy(random.choice(programs))
        return p

    return getData