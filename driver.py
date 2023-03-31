from tetris import *
import matplotlib.pyplot as plot

from SMC import instrumentSMC

import numpy as np
import torch
import random

from datetime import datetime

def get_arguments(mode=None, options=None):
    # Bit hacky but I want to be able to run this from a interactive python terminal,
    # so instead of just command line arguments I want to create arguments in function
    import argparse
    parser = argparse.ArgumentParser(description = "")
    # If mode is not provided: get it from input
    if mode is None:
        parser.add_argument("mode", choices=["imitation","exit","test","demo","makeData","heatMap",
                                             "critic","render","embed"])
    else:
        parser.add_argument("--mode", default=mode)
    # All other arguments optional
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--maxShapes", default=7,
                            type=int)
    parser.add_argument("--2d", default=True, action='store_true', dest='td')
    parser.add_argument("--viewpoints", default=False, action='store_true', dest='viewpoints')
    parser.add_argument("--trainTime", default=None, type=float,
                        help="Time in hours to train the network")
    parser.add_argument("--attention", default=0, type=int,
                        help="Number of rounds of self attention to perform upon objects in scope")
    parser.add_argument("--heads", default=2, type=int,
                        help="Number of attention heads")
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed")
    parser.add_argument("--ntest", default=30, type=int,
                        help="size of testing set")
    parser.add_argument("--hidden", "-H", type=int, default=512,
                        help="Size of hidden layers")
    parser.add_argument("--timeout", default=5, type=float,
                        help="Test time maximum timeout")
    parser.add_argument("--nudge", default=False, action='store_true')
    parser.add_argument("--oneParent", default=True, action='store_true')
    parser.add_argument("--noTranslate", default=True, action='store_true')
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--render", default=[],type=str,nargs='+')
    parser.add_argument("--task", default=False, action='store_true')
    parser.add_argument("--noExecution", default=False, action='store_true')
    parser.add_argument("--rotate", default=False, action='store_true')
    parser.add_argument("--solvers",default=["fs"],nargs='+')
    # Parse to get arguments
    arguments = parser.parse_args()
    arguments.translate = not arguments.noTranslate    
    # Collect all variables in arguments in a dictionary
    arg_dict = vars(arguments)
    # Update according to options, if provided
    if options:
        for k, v in options.items():
            if k in arg_dict.keys():
                arg_dict[k] = v
    # Return arguments
    return arguments

def run(arguments):
    timestamp = datetime.now().strftime('%FT%T')
    if arguments.render:
        for path in arguments.render:
            if path.endswith(".pickle") or path.endswith(".p"):
                with open(path,"rb") as handle:
                    program = pickle.load(handle)
                    print(f"LOADED {path}")
                    print(ProgramGraph.fromRoot(program).prettyPrint(True))
                    plotShape(program, "tmp/render.png")
            if path.endswith(".txt"):
                    programs = loadShapes(path)
                    for i, program in enumerate(programs):
                        plotShape(program, f"tmp/render_{str(i)}.png")
        import sys
        sys.exit(0)                        

    if arguments.mode == "demo":
        os.system("mkdir demo")
        rs = lambda : randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes)
        
        startTime = time.time()
        ns = 50
        for _ in range(ns):
            rs().execute()
        print(f"{ns/(time.time() - startTime)} (renders + samples)/second")
        if arguments.task:
            # Render all scenes that participants have to solve in task
            rs = loadShapes('data/task.txt')
        else:
            rs = [rs() for _ in range(10) ]
            
        for n,s in enumerate(rs):
            plotShape(s, f"demo/tetris_{n}_hr.png")
            print(s)

        import sys
        sys.exit(0)
        
            
    if arguments.checkpoint is None:
        arguments.checkpoint = f"checkpoints/tetris_{arguments.mode}"
        if arguments.noExecution:
            arguments.checkpoint += "_noExecution"
        if arguments.viewpoints:
            arguments.checkpoint += "_viewpoints"
        if arguments.attention > 0:
            arguments.checkpoint += f"_attention{arguments.attention}_{arguments.heads}"
        if not arguments.td:
            if not arguments.rotate:
                arguments.checkpoint += "_noRotate"
        arguments.checkpoint += f"_{timestamp}.pickle"
        print(f"Setting checkpointpath to {arguments.checkpoint}")
    if arguments.mode == "imitation":
        dsl = tDSL
        oe = ObjectEncoder()
        se = SpecEncoder()
        training = lambda : randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes)

        print(f"CNN output dimensionalitys are {oe.outputDimensionality} & {se.outputDimensionality}")

        if arguments.resume:
            m = torch.load(arguments.resume)
            print(f"Resuming checkpoint {arguments.resume}")
        else:
            if arguments.noExecution:
                m = NoExecution(se,dsl)
            else:
                m = ProgramPointerNetwork(oe, se, dsl,
                                          oneParent=arguments.oneParent,
                                          attentionRounds=arguments.attention,
                                          heads=arguments.heads,
                                          H=arguments.hidden)
        trainCSG(m, training,
                 trainTime=arguments.trainTime*60*60 if arguments.trainTime else None,
                 checkpoint=arguments.checkpoint)
    elif arguments.mode == "critic":
        assert arguments.resume is not None, "You need to specify a checkpoint with --resume, which bootstraps the policy"
        m = torch.load(arguments.resume)
        critic = A2C(m)
        # JB: I want to only reward completely correct solutions
        def R(spec, program):
            if len(program) == 0 or len(program) > len(spec.toTrace()): return False
            for o in program.objects():
                if o.IoU(spec) == 1: return True
            return False
        if arguments.td:
            training = lambda: randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes)
        critic.train(arguments.checkpoint,
                     training,
                     R)
        
    elif arguments.mode == "heatMap":
        print('Heatmap not implemented')
    elif arguments.mode == "makeData":
        makeTrainingData()
    elif arguments.mode == "exit":
        with open(arguments.checkpoint,"rb") as handle:
            m = pickle.load(handle)
        searchAlgorithm = BeamSearch(m, maximumLength=arguments.maxShapes*3 + 1)
        loss = lambda spec, program: 1-max( o.IoU(spec) for o in program.objects() ) if len(program) > 0 else 1.
        searchAlgorithm.train(getTrainingData('data/CSG_data.p'),
                              loss=loss,
                              policyOracle=lambda spec: spec.toTrace(),
                              timeout=1,
                              exitIterations=-1)
    elif arguments.mode == "test":
        m = load_checkpoint(arguments.checkpoint)
        if arguments.task:
            # Test on shapes in participant task
            dataGenerator = loadShapes('data/task.txt')
        else:
            dataGenerator = lambda: randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes)
        testCSG(m,
                dataGenerator,
                arguments.timeout,
                solvers=arguments.solvers,
                timestamp=timestamp,
                solverSeed=arguments.seed,
                n_test=arguments.ntest)    
    elif arguments.mode == "embed":
        m = load_checkpoint(arguments.checkpoint)
        if arguments.task:
            # Test on shapes in participant task
            dataGenerator = loadShapes('data/task.txt')
        else:
            dataGenerator = lambda: randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes)
        embedStims(m,
                dataGenerator,
                timestamp=timestamp,
                n_test=arguments.ntest)    

if __name__ == "__main__":
    arguments = get_arguments()  
    
    
    #print(f"Invoking @ {timestamp} as:\n\tpython {' '.join(sys.argv)}")

    run(arguments)