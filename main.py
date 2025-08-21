# import numpy as np
from igraph import Graph
from hedonic import Game

###################################

# 1) 50M edges ER graph (undirected, simple)
g = Graph.Erdos_Renyi(n=1_000, m=5_000)

# 2) Scale-free with ~m edges added per node
# g = Graph.Barabasi(n=10_000_000, m=2, power=1, directed=False)

# # 3) Small-world ring lattice rewired with prob p
# g = Graph.Watts_Strogatz(dim=1, size=2_000_000, nei=5, p=0.01)

# # 4) SBM with 3 blocks
# sizes = [500_000, 300_000, 200_000]
# P = np.array([[0.001, 0.0001, 0.00005],
#               [0.0001, 0.002, 0.0001 ],
#               [0.00005,0.0001, 0.0015]])
# g = Graph.SBM(sum(sizes), P, sizes, directed=False)

# # 5) Configuration model from degree sequence
# deg = [10]*1_000_000
# g = Graph.Degree_Sequence(deg, method="configuration")

# # 6) Random geometric on torus
# g = Graph.GRG(n=5_000_000, radius=0.0008, torus=True)

###################################

g = Game(g)
gamma = g.density()

p = g.community_leiden(n_iterations=-1, resolution=g.density(), allow_isolation=False, only_local_moving=True)

p1 = g.community_leiden(n_iterations=1, resolution=gamma)
g.initialize_game(p1.membership)
print(f'p1 in equilibrium: {g.in_equibrium(gamma)}')

p2 = g.community_leiden(n_iterations=-1, resolution=gamma)
g.initialize_game(p2.membership)
print(f'p2 in equilibrium: {g.in_equibrium(gamma)}')
