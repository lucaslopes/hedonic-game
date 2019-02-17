# Hedonic Game
# Finding Clusters with Cooperative Game Theory
# A research experiment of Daniel Sadoc^ and Lucas Lopes^
# ^Federal University of Rio de Janeiro
# February, 2019

##############################################################

# ----------- Datasets ------------
# sample_1 | terrorists | football
# sample_2 | conference | math_net
# sample_3 |   karate   | dolphins

from libs.players import *

params = {
    'graph'  : 'karate', # Choose one above
    'player' :  greedy,  # greedy | partial | stochastic | doNothing
    'alpha'  :  0.95,    # Fragmentation Factor | 0 <= alpha <= 1
    'gamma'  :  0,       # Discount Factor for create a new Cluster
    'limit'  :  2,       # Maximum number of Clusters
    'errors' :  1/4      # Will be multiply by number of verts of graph
}

##############################################################

from libs.Game import Game

game = Game(params)
game.start()
