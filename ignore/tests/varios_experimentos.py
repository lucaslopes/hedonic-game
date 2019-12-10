# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')

## Do an Experiment ############################################################

# -------------------------------- Networks ------------------------------------
# sample_1 |   karate   |  football  |   human   |  1k  |
# sample_2 | terrorists | conference |    dag    |  10k |
#  square  |  dolphins  | 2triangles | 2trig_mid | 100k |

game = Game(network = 'terrorists', # choose one above
            alpha   = .95,    # 0 > alpha < 1
            verbose = True,   # if True will print each move
            export  = True,   # if True will export the game history for future replay
            init_mode  = 'r', # 'r' for random classification and 's' for select nodes
            init_param = .5)  # if 'r' is between 0 and 1; elif 's' is a list of str(nodes)
game.play(sequential) # sequential, stochastic, greedy, pique_pega
