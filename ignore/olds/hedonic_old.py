# Hedonic Games for Clustering
# A research experiment of Giovanni^, Kostya^, Daniel' and Lucas' - ack Eduardo'
# ^INRIA | 'Federal University of Rio de Janeiro (UFRJ)
# August 2019

## Import Dependecies ##########################################################

# from hedonic import Game
from players import sequential, stochastic, greedy, pique_pega
from datetime import datetime
from pathlib import Path
from random import random
from copy import deepcopy
import csv
import os
# import json

## Parameters ##################################################################

# -------------------------------- Networks ------------------------------------
# sample_1 |   karate   |  football  |   human   |  1k  |  |  |  |  |  |
# sample_2 | terrorists | conference |    dag    | 10k  |  |  |  |  |  |
#  square  |  dolphins  | 2triangles | 2trig_mid | 100k |  |  |  |  |  |

p = {
    'graph' : '1k', # Choose one above
    'alpha' : 0.95, # Fragmentation Factor | 0 <= alpha <= 1
    'player': sequential, # sequential | stochastic | greedy
    'init'  : {
        'mode': 'r', # Initial Modes:
            # - Random (r): Nodes will be randomly selected to start inside cluster
            # - Select (s): Chose which nodes will start inside cluster
            # - Any other: Start with an empty cluster
        'param': 0.5 },
            # - Random (r): Number of nodes - is between 0 and 1 will be multiply by quantity of nodes
            # - Select (s): List of selected nodes - [node indice, ..., node indice]
    'print' : False, # Print each iteration in the terminal?
    'save'  : False, # Print each iteration in the terminal?
    'note'  : 'None' } # Free space to comment about the experiment

## A Hedonic Game ##############################################################

class Game:

    def __init__(self, network='sample_2', alpha=.95, init={'mode':'','param':''},
                 verbose=True, export=False, comments='None'):
        self.graph = network
        self.alpha = alpha
        self.ops   = {'ver': verbose, 'exp': export, 'com': comments}
        self.stats = {} # init_potential, finished_at, duration, import_duration, export_duration, init_config and current_performance
        self.hist  = {  # append at each iteration
            'nodes_moved' : [],
            'inst_gain'   : [],
            'groups'      : [],
            'performance' : []}
        self.load_network(f'networks/{network}.csv')
        self.set_ground_truth(f'networks/ground_truth/{network}.csv')
        self.set_initial_state(init['mode'], init['param'])

    ## Load the Network ########################################################

    def load_network(self, file):
        nodes, duration = {}, datetime.now()
        def insert(a, b):
            if a not in nodes:
                nodes[a] = { 'friends': [b] } # create a new node
            elif b not in nodes[a]['friends']:
                nodes[a]['friends'].append(b)
        with open(file, 'r') as f:
            table = csv.reader(f)
            for row in table:
                a = row[0]
                b = row[1]
                insert(a, b)
                insert(b, a)
        self.stats['import_duration'] = datetime.now() - duration
        self.nodes = nodes

    ## Setters #################################################################

    def set_ground_truth(self, file):
        g_t, duration = {}, datetime.now()
        if Path(file).is_file():
            with open(file, newline='') as f:
                table = csv.reader(f)
                row = next(table)
                nodes = []
                for node in row:
                    nodes.append(node)
                row = next(table)
                clusters = []
                for cluster in row:
                    clusters.append(cluster)
                for i in range(len(nodes)):
                    g_t[nodes[i]] = clusters[i]
        else:
            for node in self.nodes:
                g_t[node] = 'none'
        self.stats['import_duration'] += datetime.now() - duration
        self.g_truth = g_t

    def set_initial_state(self, mode = '', param = ''):
        for node in self.nodes:
            self.nodes[node]['my_group'] = 'no' # new node prop
        if mode.lower() == 'r':
            for node in self.nodes:
                if random() <= param:
                    self.nodes[node]['my_group'] = 'yes'
        if mode.lower() == 's':
            for n in param:
                self.nodes[n]['my_group'] = 'yes'
        self.groups = self.count_verts_edges()
        self.initial_state = []
        verts = {'yes':self.groups['yes']['verts'],'no':self.groups['no']['verts']}
        smaller = min(verts, key=verts.get)
        for node in self.nodes:
            if self.nodes[node]['my_group'] == smaller:
                self.initial_state.append(node)
        self.stats['init_potential'] = self.global_potential()
        self.stats['init_config'] = {'mode': mode, 'param': param}
        self.stats['current_performance'] = {
            'profits_consulted': 0,
            'profitable_ones':   0 }

    def set_accuracy(self):
        clusters = self.groups.keys()
        g_truths = {}
        for node, group in self.g_truth.items():
            g_truths[group] = None
        g_truths = g_truths.keys()
        acc = {}
        for c in clusters:
            for gt in g_truths:
                tp = tn = fp = fn = 0
                for node, props in self.nodes.items():
                    if props['my_group'] == c and self.g_truth[node] == gt:
                        tp += 1
                        continue
                    if props['my_group'] != c and self.g_truth[node] != gt:
                        tn += 1
                        continue
                    if props['my_group'] == c and self.g_truth[node] != gt:
                        fp += 1
                        continue
                    if props['my_group'] != c and self.g_truth[node] == gt:
                        fn += 1
                        continue
                s = '{} x {}'.format(c, gt)
                acc[s] = {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
        self.accuracy = acc

    ## Counting ################################################################

    def check_friends(self, node):
        us = them = 0
        for friend in self.nodes[node]['friends']:
            if self.nodes[node]['my_group'] == self.nodes[friend]['my_group']:
                us += 1
            else: them += 1
        self.nodes[node]['my_placar'] = {'us': us, 'them': them} # new node prop
        return us, them

    def count_verts_edges(self):
        yes = {'verts': 0, 'edges': 0}
        no  = {'verts': 0, 'edges': 0}
        def increment(group, node):
            group['verts'] += 1
            group['edges'] += self.check_friends(node)[0]
        for node in self.nodes:
            if self.nodes[node]['my_group'] == 'yes':
                increment(yes, node)
            else: increment(no, node)
        yes['edges'] = int(yes['edges'] / 2)
        no['edges']  = int(no['edges']  / 2)
        return {'yes': yes, 'no': no}

    ## Hedonic Functions #######################################################

    def hedonic(self, friends, all):
        return friends - all * self.alpha

    def local_potential(self, verts, edges):
        return edges - self.alpha * verts * (verts - 1) / 2

    def global_potential(self):
        pot = 0
        for group, v in self.groups.items():
            pot += self.local_potential(v['verts'], v['edges'])
        return pot

    def profit(self, node):
        my_group, there_group = 'yes', 'no'
        if self.nodes[node]['my_group'] == 'no':
            my_group, there_group = 'no', 'yes' # invert if it's from other cluster
        us   = self.nodes[node]['my_placar']['us']
        them = self.nodes[node]['my_placar']['them']
        all_here  = self.groups[my_group]['verts']
        all_there = self.groups[there_group]['verts']
        here  = self.hedonic(us,   all_here - 1) # excluding me
        there = self.hedonic(them, all_there)
        if there - here > 0:
            self.stats['current_performance']['profitable_ones'] += 1
        self.stats['current_performance']['profits_consulted'] += 1
        return there - here

    ## Operations ##############################################################

    def play(self, player):
        duration = datetime.now()
        self.player = player.__name__
        if self.ops['ver']: self.print_parameters()
        player(self)
        self.stats['finished_at'] = datetime.now()
        self.stats['duration'] = self.stats['finished_at'] - duration
        if self.ops['ver']: print('Done!', self.stats['duration'])
        if self.ops['exp']: self.export_results()

    def reach_equilibrium(self):
        for node in self.nodes:
            if self.profit(node) > 0:
                return False #, node
        return True

    def move(self, node):
        increased = self.profit(node)
        self.hist['nodes_moved'].append(node)
        self.hist['groups'].append(deepcopy(self.groups))
        self.hist['performance'].append(deepcopy(self.stats['current_performance']))
        self.hist['inst_gain'].append(increased)
        old_us =       self.nodes[node]['my_placar']['us']
        new_us =     self.nodes[node]['my_placar']['them']
        self.groups[self.nodes[node]['my_group']]['verts'] -= 1 # update group that it was
        self.groups[self.nodes[node]['my_group']]['edges'] -= old_us
        if self.nodes[node]['my_group']   == 'no': # change its group
            self.nodes[node]['my_group']   = 'yes'
        else: self.nodes[node]['my_group'] = 'no'
        self.groups[self.nodes[node]['my_group']]['verts'] += 1 # update group that it moved to
        self.groups[self.nodes[node]['my_group']]['edges'] += new_us
        for friend in self.nodes[node]['friends']: # comunicate change to its friends
            if self.nodes[node]['my_group'] == self.nodes[friend]['my_group']:
                self.nodes[friend]['my_placar']['us']   += 1
                self.nodes[friend]['my_placar']['them'] -= 1
            else:
                self.nodes[friend]['my_placar']['us']   -= 1
                self.nodes[friend]['my_placar']['them'] += 1
        self.nodes[node]['my_placar']['us']              = new_us
        self.nodes[node]['my_placar']['them']            = old_us
        self.stats['current_performance']['profits_consulted'] = 0
        self.stats['current_performance']['profitable_ones']   = 0
        if self.ops['ver']: self.timestamp(node, increased)

    ## Export Results ##########################################################

    def define_game_path(self):
        now = str(datetime.now())[2:-7]
        for _ in ': -': now = now.replace(_, '')
        return f'experiments/{self.graph}_{now}/'

    def print_parameters(self):
        print('# Parameters:')
        print(f'- Graph:          {self.graph}')
        print(f'- Alpha:          {self.alpha}')
        print(f"- Initial Config: {self.stats['init_config']}")
        print(f"- Verbose Mode:   {self.ops['ver']}")
        print(f"- Export Results: {self.ops['exp']}")
        print(f"- Commentaries:   {self.ops['com']}")
        print('Go!!!')

    def timestamp(self, node, increased):
        print(f'{datetime.now()} | move: {node} | increased: {increased:.2f}')

    def create_files(self, path):
        os.makedirs(os.path.dirname(path), exist_ok = True) # create folder
        iterations = open(path + 'iterations.csv', 'w+')
        accuracy   = open(path + 'accuracy.csv', 'w+')
        stats      = open(path + 'stats.csv', 'w+')
        iterations_collumn = 'iteration,node moved,to group,instantaneous gain,\
            accumulated gain,potential,abs gain %,yes verts,no verts,yes edges,\
            no edges,verts %,edges %,potential propor %,consults,performance %\n'
        stats_collumn = 'node,initial group,final group,initial hedonic,final hedonic,\
            % gain hedonic,num of moves,move at,hedonic at move\n'
        self.set_accuracy()
        cols, row = 'iteration,', '0,'
        for pair in self.accuracy.keys():
            cols += f'{pair},,,,'
            row  += f"{self.accuracy[pair]['TP']},{self.accuracy[pair]['TN']},\
                      {self.accuracy[pair]['FP']},{self.accuracy[pair]['FN']},"
        s = 'True Positive,True Negative,False Positive,False Negative,' * len(self.accuracy)
        accuracy_collumn = cols[:-1] + '\n' + f'iteration,{s[:-1]}' + '\n' + row[:-1] + '\n'
        iterations.write(iterations_collumn)
        accuracy.write(accuracy_collumn)
        stats.write(stats_collumn)
        return iterations, accuracy, stats

    def export_results(self):
        self.stats['export_duration'] = datetime.now()
        path = self.define_game_path()
        iterations, accuracy, stats = self.create_files(path)
        init_config = self.stats['init_config'] # maintain first init config
        converged   = self.reach_equilibrium()
        self.set_initial_state('s', self.initial_state) # reset game
        accumulated = 0
        for i in range(len(self.hist['nodes_moved'])):
            node = self.hist['nodes_moved'][i]
            if self.nodes[node]['my_group'] == 'yes':
                self.nodes[node]['my_group'] = 'no'
            else: self.nodes[node]['my_group'] = 'yes'
            gain = self.hist['inst_gain'][i]
            accumulated += gain
            init_pot  = self.stats['init_potential']
            checks    = self.hist['performance'][i]['profits_consulted']
            profits   = self.hist['performance'][i]['profitable_ones']
            verts_yes = self.hist['groups'][i]['yes']['verts']
            verts_no  = self.hist['groups'][i]['no']['verts']
            edges_yes = self.hist['groups'][i]['yes']['edges']
            edges_no  = self.hist['groups'][i]['no']['edges']
            abs_gain  = gain / abs((self.stats['init_potential'] + accumulated)) * 100
            verts_pct = verts_yes / (verts_yes + verts_no) * 100
            edges_pct = edges_yes / (edges_yes + edges_no) * 100
            to_group  = self.nodes[node]['my_group']
            # iterations.write('{},{},{},{:.2f},{:.2f},{:.2f}\n'.format(i+1,node,to_group,gain,accumulated,init_pot+accumulated))
            s = '{},{},{},{:.2f},{:.2f},{:.2f},+{:.2f}%,{},{},{},{},{:.2f}%,\
                {:.2f}%,{:.2f}%,{},{:.2f}%\n'.format(
                i+1,node,to_group,gain,accumulated,init_pot+accumulated,abs_gain,
                verts_yes,verts_no,edges_yes,edges_no,verts_pct,edges_pct,
                self.local_potential(verts_yes,edges_yes)/(init_pot+accumulated)*100,
                checks,profits/checks*100)
            iterations.write(s)
            self.set_accuracy()
            a = f'{i+1},'
            for pair in self.accuracy.keys():
                a += f"{self.accuracy[pair]['TP']},{self.accuracy[pair]['TN']},\
                       {self.accuracy[pair]['FP']},{self.accuracy[pair]['FN']},"
            accuracy.write(a[:-1] + '\n')

            stats.write(',,,,,,,,\n')
        iterations.close()
        accuracy.close()
        stats.close()
        self.export_infos(path, init_config, accumulated, converged)

    def export_infos(self, path, init_config, accumulated, converged):
        init_verts_yes = self.hist['groups'][0]['yes']['verts']
        init_edges_yes = self.hist['groups'][0]['yes']['edges']
        init_edges_no  = self.hist['groups'][0]['no']['edges']
        verts_yes = self.hist['groups'][-1]['yes']['verts']
        edges_yes = self.hist['groups'][-1]['yes']['edges']
        edges_no  = self.hist['groups'][-1]['no']['edges']
        consults  = sum(i['profits_consulted'] for i in self.hist['performance'])
        info = f"\
### PARAMETERS ###\n\
Network,{self.graph}\n\
Alpha,{self.alpha}\n\
Player,{self.player}\n\
¹Initial params,{str(init_config).replace(',',' and')}\n\
Verbose,{self.ops['ver']}\n\
Export,{self.ops['exp']}\n\
Commentaries,{self.ops['com']}\n\n\
### RESULTS ###\n\
Initial potential,{self.stats['init_potential']:.2f}\n\
Final potential,{self.stats['init_potential']+accumulated:.2f}\n\
Accumulated gain,+{accumulated:.2f}\n\
²Abs gain in %,+{accumulated/abs(self.stats['init_potential'])*100:.2f}%\n\
Num of iterations,{len(self.hist['nodes_moved'])}\n\
Finished at,{self.stats['finished_at']}\n\
Import duration,{self.stats['import_duration']}\n\
Run duration,{self.stats['duration']}\n\
Export duration,{datetime.now()-self.stats['export_duration']}\n\
Total duration,{self.stats['import_duration']+self.stats['duration']+datetime.now()-self.stats['export_duration']}\n\
Initial Verts proportion,{init_verts_yes/len(self.nodes)*100:.2f}%\n\
Final Verts proportion,{verts_yes/len(self.nodes)*100:.2f}%\n\
Initial Edges proportion,{init_edges_yes/(init_edges_yes+init_edges_no)*100:.2f}%\n\
Final Edges proportion,{edges_yes/(edges_yes+edges_no)*100:.2f}%\n\
Initial Potential proportion,{self.local_potential(init_verts_yes, init_edges_yes)/self.stats['init_potential']*100:.2f}%\n\
Final Potential proportion,{self.local_potential(verts_yes, edges_yes)/(self.stats['init_potential']+accumulated)*100:.2f}%\n\
Profit consults,{consults}\n\
Performance,{len(self.hist['nodes_moved'])/consults*100:.2f}%\n\
Converged,{converged}\n"
        infos = open(path + 'infos.csv', 'w+')
        infos.write(info)
        infos.close()

## Initiate Experiment #########################################################

# -------------------------------- Networks ------------------------------------
# sample_1 |   karate   |  football  |   human   |  1k  |
# sample_2 | terrorists | conference |    dag    | 10k  |
#  square  |  dolphins  | 2triangles | 2trig_mid | 100k |

game = Game(network = '10k',
            alpha = .95,
            init = { 'mode': 'r', 'param': 0.5 },
            verbose  = True, export = True,
            comments = 'None')
game.play(sequential) # sequential, stochastic, greedy, pique_pega

import pickle
with open('game_10k.pkl', 'wb') as output:
    pickle.dump(game, output, pickle.HIGHEST_PROTOCOL)

## Functions ###################################################################

def get_all_states(nodes): # recebe lista de nodes
	states = []
	for k in range(1, int((2 ** len(nodes)) / 2)):
		num = list(bin(k))[2:]
		num.reverse()
		#print('num', num)
		positions = []
		for pos, value in enumerate(num):
			if int(value) == 1:
				positions.append(nodes[pos])
		#print('pos', positions)
		states.append(positions)
	return states

def check_dist(inside, gt):
    dist, need_move = float('inf'), []
    for cluster in gt:
        corrects = [node for node in inside if node in cluster]
        nodes2move = len(inside) + len(cluster) - 2 * len(corrects)
        if nodes2move < dist:
            dist = nodes2move
            need_move = [node for node in inside+cluster if node not in corrects]
    return need_move

def show_distances(graph, gt):
    result = 'combination,distance,greedy,diffe,non_profit\n'
    p['graph'] = graph
    nodes = list(Game(p).graph)
    combinations = get_all_states(nodes)
    for comb in combinations:
        p['init']['params'] = comb
        greedy = Game(p)
        game = Game(p)
        # inicio da gambiarra
        dist, need_move = float('inf'), []
        for g in gt:
            temp = check_dist(comb, g)
            if len(temp) < dist:
                dist = len(temp)
                need_move = temp
        # fim da gambiarra
        dist = len(need_move)
        non_profit = 0
        while len(need_move) > 0:
            best_profit, best_node = float('-inf'), None
            for node in need_move:
                profit = game.profit(node)
                if profit > best_profit:
                    best_profit = profit
                    best_node = node
            if best_profit < 0: non_profit += 1
            game.move(node)
            need_move.remove(node)
        if game.has_move() is False: print(f'{comb} chegou no sumidouro')
        else: print('terminou mas não convergiu')
        comb_str = str(comb).replace(',','')
        greedy.start()
        g_moves = greedy.iteration
        diff = g_moves - dist
        if dist != greedy.iteration:
            result += f'{comb_str},{dist},{g_moves},{diff},{non_profit}\n'
    return result

def find_stable_states(graph_list):
    stable_states = {}
    for graph in graph_list:
        before = time.time()
        stable_states[graph] = []
        p['graph'] = graph
        p['init']['params'] = []
        nodes = list(Game(p).graph)
        combinations = get_all_states(nodes)
        for comb in combinations:
            p['init']['params'] = comb
            game = Game(p)
            if game.has_move() is False:
                stable_states[graph].append(comb)
        print(f'{graph} duration: {time.time()-before}')
    return stable_states

## Initiate Experiment #########################################################

#game = Game(p)
#game.start()

# before = time.time()
# for _ in range(100):
#     game = Game(p)
#     # game.folder  = game.set_game_path(p['graph'])
#     # game.classes = game.set_classes(p['init'], game.graph.keys())
#     # print(game.has_move())
#     # print('vai começar')
#     game.start()
# print(time.time() - before)


## Graphs Test #####################

# graph_list = ['human']#['sample_1', 'sample_2', 'square', '2triangles', '2triangles_mid', 'dag']
# stable_states = find_stable_states(graph_list)
# with open('output/human.txt', 'w') as f:
#     json.dump(stable_states, f)

# graph = 'square' # no-diff
# gt = [[['1', '2'], ['3', '4']], [['1', '3'], ['2', '4']]]

# graph = '2triangles' # no-diff
# gt = [[['A1', 'A2', 'A3'], ['B1', 'B2', 'B3']]]

# graph = 'sample_1' # no-diff
# gt = [[['0', '1', '2'], ['5', '4', '3']]]

# graph = 'sample_2'
# gt = [[['1', '2', '3', '4'],      ['5', '6', '7', '8']],
#       [['1', '3', '4', '5', '6'], ['2', '7', '8']],
#       [['2', '3', '4', '5', '6'], ['1', '7', '8']],
#       [['1', '3', '5', '7'],      ['2', '4', '6', '8']],
#       [['2', '4', '5', '7'],      ['1', '3', '6', '8']]]

# graph = '2triangles_mid'
# gt = [[['A1', 'A2', 'A3'],      ['C', 'B1', 'B2', 'B3']],
#       [['A1', 'A2', 'A3', 'C'], ['B1', 'B2', 'B3']],
#       [['A1', 'C', 'B1'],       ['A2', 'A3', 'B2', 'B3']]]

# graph = 'dag'
# gt = [[['1', '5', '6', '9', '10'],     ['2', '3', '4', '7', '8']],
#       [['2', '5', '6', '9', '10'],     ['1', '3', '4', '7', '8']],
#       [['3', '5', '6', '9', '10'],     ['1', '2', '4', '7', '8']],
#       [['1', '2', '5', '6', '7'],      ['3', '4', '8', '9', '10']],
#       [['1', '3', '5', '6', '7'],      ['2', '4', '8', '9', '10']],
#       [['2', '3', '5', '6', '7', '9'], ['1', '4', '8', '10']],
#       [['2', '3', '7', '9', '10'],     ['1', '4', '5', '6', '8']]]

# graph = 'human'
# gt = []
#
# result = show_distances(graph, gt)
# with open(f'output/{graph}_dist.csv','w') as resultFile:
#     resultFile.write(result)
