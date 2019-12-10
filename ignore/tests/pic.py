import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import pickle
from new_hedonic import Game, Node

with open('experiments/conference_190910055449.pickle', 'rb') as f:
    game = pickle.load(f)

print(game.results)

verts_in = game.initial['verts_in']
verts_no = len(game.nodes) - verts_in

initial = []
if verts_in < verts_no:
    for node, cont in game.nodes.items():
        if cont.initial_group:
            initial.append(node)
else:
    for node, cont in game.nodes.items():
        if not cont.initial_group:
            initial.append(node)

with open('initial_conf.pickle', 'wb') as output:
    pickle.dump(initial, output, pickle.HIGHEST_PROTOCOL)

with open('moved_conf.pickle', 'wb') as output:
    pickle.dump(game.iterations['node_moved'], output, pickle.HIGHEST_PROTOCOL)
