from time import time
duration = time()

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import pickle
from new_hedonic import Game, Node

print('imports', time() - duration)
duration = time()

game = Game(network = 'conference',
            alpha = .95,
            init = { 'mode': 's', 'param': '' },
            ops = { 'verbose': False, 'export': True })

print('instancia', time() - duration)
duration = time()

with open('initial_conf.pickle', 'rb') as f:
    nodes_inside = pickle.load(f)

print('import initial', time() - duration)
duration = time()

with open('moved_conf.pickle', 'rb') as f:
    nodes_moved = pickle.load(f)

print('import moved', time() - duration)
duration = time()

game.replay(nodes_inside, nodes_moved)

print('replay', time() - duration)
