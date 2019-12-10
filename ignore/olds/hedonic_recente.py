# Hedonic Games for Clustering
# A research experiment of Giovanni^, Kostya^, Daniel' and Lucas' - ack Eduardo'
# ^INRIA | 'Federal University of Rio de Janeiro (UFRJ)
# August 2019

## Import Dependecies ##########################################################

from players import sequential, stochastic, greedy, pique_pega
from datetime import datetime
from pathlib import Path
from random import random
from copy import deepcopy
import csv
import os
import pickle
# import json

## Node ########################################################################

class Node:

    def __init__(self, friend, initial_group = False):
        self.friends = { friend : True } # weight avaiable
        self.initial_group = initial_group
        self.with_me = 0
        self.moved_at = []
        self.ground_truth = None

    def add(self, friend):
        self.friends[friend] = True

    def group(self):
        if len(self.moved_at) % 2 == 0: return self.initial_group
        else: return not self.initial_group

    def binary_search(arr, element):
        margin_left, margin_right = 0, len(arr) - 1
        while (margin_left <= margin_right):
            pointer = int((margin_left + margin_right) / 2)
            if element > arr[pointer]:
                margin_left = pointer + 1
            elif element < arr[pointer]:
                margin_right = pointer - 1
            elif element == arr[pointer]:
                return pointer
        if arr[pointer] > element:
            return pointer - 1
        else:
            return pointer

## A Hedonic Game ##############################################################

class Game:

    def __init__(self, network='sample_2', alpha=.95,
                 init={'mode':'','param':''}, ops={'verbose':True,'export':False}):
        self.info = {
            'graph'   : network,
            'alpha'   : alpha,
            'options' : ops,
            'init_config' : init,
            'player'  : None }
        self.reset()
        self.load_network(f'networks/csv/{network}.csv')
        self.set_initial_state(init['mode'], init['param'])

    def reset(self):
        self.iterations = { # hist
            'node_moved'  : [0],
            'moved_to'    : [0], # vai no nó que moveu, ve len(até tal iteração)
            'verts_yes_diff' : [0],
            'edges_yes_diff' : [0],
            'edges_no_diff': [0],
            'accumulated' : [0],
            'consults'    : [0],
            'profitables' : [0]}
        self.initial = {
            'potential' : 0,
            'verts_in'  : 0,
            'edges'     : 0,
            'edges_in'  : 0,
            'edges_out' : 0} # init_potential -> local tbm?
        self.results = { # stats
            'start_at' : datetime.now(),
            'import_duration': 0,
            'run_duration': 0,
            'export_duration': 0,
            'performance' : {'consults': 0, 'profitables': 0},
            'iteration': 0}



        # self.node_moved      = [0]
        # self.moved_to        = [0] # vai no nó que moveu, ve len(até tal iteração)
        # self.verts_yes_diff  = [0]
        # self.edges_yes_diff  = [0]
        # self.edges_no_diff   = [0]
        # self.accumulated     = [0]
        # self.consults        = [0]
        # self.profitables     = [0]
        # self.graph           = network
        # self.alpha           = alpha
        # self.options         = ops
        # self.init_config     = init # init_config
        # self.player          = None
        # self.potential       = 0
        # self.verts_in        = 0
        # self.edges_in        = 0
        # self.edges_out       = 0 # init_potential -> local tbm?
        # self.start_at        = datetime.now()
        # self.import_duration = 0
        # self.run_duration    = 0
        # self.export_duration = 0
        # self.performance     = {'consults': 0, 'profitables': 0}
        # self.iteration       = 0

    ## Load the Network ########################################################

    def load_network(self, file): # 42s
        nodes, duration = {}, datetime.now()
        with open(file, 'r') as f:
            table = csv.reader(f)
            for row in table:
                a = row[0]
                b = row[1]
                if nodes.get(a): nodes[a].add(b) # add new friend
                else: nodes[a] = Node(b) # create a new node
                if nodes.get(b): nodes[b].add(a) # add new friend
                else: nodes[b] = Node(a) # create a new node
        self.nodes = nodes
        self.results['import_duration'] = datetime.now() - duration
        self.set_ground_truth(file.replace('networks/csv/','networks/ground_truth/'))
        # with open('CONFERENCE_NODES.pickle', 'wb') as output:
        #     pickle.dump(nodes, output, pickle.HIGHEST_PROTOCOL)

    ## Setters #################################################################

    def set_ground_truth(self, file):
        g_t, duration = {}, datetime.now()
        if Path(file).is_file():
            with open(file, 'r') as f:
                table = csv.reader(f)
                for row in table:
                    node = row[0]
                    g_t  = row[1]
                    self.initial['edges'] += 1 # gambiarra
                    self.nodes[node].ground_truth = g_t
        else:
            for node in self.nodes:
                self.nodes[node].ground_truth = 'none'
        self.results['import_duration'] += datetime.now() - duration

    def set_initial_state(self, mode = '', param = ''):
        self.reset()
        if mode.lower() == 'r':
            for node in self.nodes:
                if random() <= param: self.nodes[node].initial_group = True
                else: self.nodes[node].initial_group = False
                self.nodes[node].moved_at = []
        if mode.lower() == 's':
            for node in self.nodes:
                self.nodes[node].initial_group = False
                self.nodes[node].moved_at = []
            for node in param:
                self.nodes[node].initial_group = True
        verts_edges = self.count_verts_edges() # verts, edges_yes, edges_no
        self.initial['verts_in']  = verts_edges[0]
        self.initial['edges_in']  = verts_edges[1]
        self.initial['edges_out'] = verts_edges[2]
        self.initial['potential'] = self.global_potential()
        self.info['init_config']  = {'mode': mode, 'param': param}, # stats

    def set_accuracy(self): # depois mexo
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
        with_me = 0
        for friend in self.nodes[node].friends:
            if self.nodes[node].group() == self.nodes[friend].group():
                with_me += 1
        self.nodes[node].with_me = with_me
        return with_me

    def count_verts_edges(self):
        verts, edges_yes, edges_no = 0, 0, 0
        for node in self.nodes:
            if self.nodes[node].group():
                verts += 1
                edges_yes += self.check_friends(node)
            else: edges_no += self.check_friends(node)
        return verts, int(edges_yes/2), int(edges_no/2)

    ## Hedonic Functions #######################################################

    def hedonic(self, have, remainder):
        return have - remainder * self.info['alpha'] # alpha

    def local_potential(self, verts, edges):
        return self.hedonic(edges, (verts * (verts - 1) / 2))

    def global_potential(self, verts_in, edges_in, edges_out, sum = True):
        yes = self.local_potential(verts_in, edges_in)
        no  = self.local_potential(len(self.nodes) - verts_in, edges_out)
        if sum: return yes + no
        else:   return yes,  no

    def profit(self, node):
        us   = self.nodes[node].with_me
        them = len(self.nodes[node].friends) - us
        all_yes = self.initial['verts_in'] + self.iterations['verts_yes_diff'][-1]
        all_no  = len(self.nodes) - all_yes
        if self.nodes[node].group(): all_here, all_there = all_yes, all_no
        else: all_here, all_there = all_no, all_yes
        here  = self.hedonic(us, all_here - 1) # excluding me
        there = self.hedonic(them, all_there)
        if there - here > 0: self.results['performance']['profitables'] += 1
        self.results['performance']['consults'] += 1
        return there - here

    ## Operations ##############################################################

    def play(self, player):
        duration = datetime.now()
        self.player = player.__name__ # player name
        if self.info['options']['verbose']: self.print_parameters()
        player(self)
        self.results['run_duration'] = datetime.now() - duration
        if self.info['options']['verbose']: print('Done!', self.results['run_duration'])
        self.export_results()

    def replay(self, nodes_inside, nodes_moded):
        duration = datetime.now()
        self.set_initial_state(mode = 's', param = nodes_inside)
        for node in nodes_moded:
            self.move(str(node))
        print(f'finish in {datetime.now() - duration}')
        if self.info['options']['verbose']: print('Done!', self.results['run_duration'])
        # self.export_results()

    def reach_equilibrium(self):
        for node in self.nodes:
            if self.profit(node) > 0:
                return False #, node
        return True

    def communicate_friends(self, node):
        for friend in self.nodes[node].friends:
            if self.nodes[node].group() == self.nodes[friend].group():
                self.nodes[friend].with_me += 1
            else:
                self.nodes[friend].with_me -= 1

    def move(self, node):
        self.results['iteration'] += 1
        self.iterations['node_moved'].append(node)
        self.iterations['consults'].append(self.results['performance']['consults'])
        self.iterations['profitables'].append(self.results['performance']['profitables'])
        increased = self.profit(node)
        self.results['performance']['consults'] = 0
        self.results['performance']['profitables'] = 0
        self.nodes[node].moved_at.append(self.results['iteration'])
        self.iterations['moved_to'].append(self.nodes[node].group())
        self.communicate_friends(node)
        self.nodes[node].with_me = len(self.nodes[node].friends) - self.nodes[node].with_me # flip
        if self.nodes[node].group():
            self.iterations['verts_yes_diff'].append(self.iterations['verts_yes_diff'][-1]+1)
            self.iterations['edges_yes_diff'].append(self.iterations['edges_yes_diff'][-1]+self.nodes[node].with_me)
        else:
            self.iterations['verts_yes_diff'].append(self.iterations['verts_yes_diff'][-1]-1)
            self.iterations['edges_no_diff'].append(self.iterations['edges_no_diff'][-1]+self.nodes[node].with_me)
        self.iterations['accumulated'].append(self.iterations['accumulated'][-1]+increased)
        if self.info['options']['verbose']: self.timestamp(node, increased)

    ## Export Results ##########################################################

    def define_game_path(self):
        now = str(datetime.now())[2:-7]
        for _ in ': -': now = now.replace(_, '')
        return f"experiments/{self.info['graph']}_{now}"


    self.info = {
        'graph'   : network,
        'alpha'   : alpha,
        'options' : ops,
        'init_config' : init,
        'player'  : None }
    self.iterations = { # hist
        'node_moved'  : [0],
        'moved_to'    : [0], # vai no nó que moveu, ve len(até tal iteração)
        'verts_yes_diff' : [0],
        'edges_yes_diff' : [0],
        'edges_no_diff': [0],
        'accumulated' : [0],
        'consults'    : [0],
        'profitables' : [0]}
    self.initial = {
        'potential' : 0,
        'verts_in'  : 0,
        'edges_in'  : 0,
        'edges_out' : 0} # init_potential -> local tbm?
    self.results = { # stats
        'start_at' : datetime.now(),
        'import_duration': 0,
        'run_duration': 0,
        'export_duration': 0,
        'performance' : {'consults': 0, 'profitables': 0},
        'iteration': 0}

    def append_result(self, converged, found_unique_state, game_path):
        total_duration = self.results['import_duration'] \
                         + self.results['run_duration'] \
                         + self.results['export_duration']

        init_verts_yes = self.hist['groups'][0]['yes']['verts']
        init_edges_yes = self.hist['groups'][0]['yes']['edges']
        init_edges_no  = self.hist['groups'][0]['no']['edges']
        verts_yes = self.hist['groups'][-1]['yes']['verts']
        edges_yes = self.hist['groups'][-1]['yes']['edges']
        edges_no  = self.hist['groups'][-1]['no']['edges']
        consults  = sum(i['profits_consulted'] for i in self.hist['performance'])

        result = '\n'
        result += f'{game_path},' # Nomenclature,
        result += f"{self.info['graph']}," # Network,
        result += f"{self.info['alpha']}," # Alpha,
        result += f"{self.info['player']}," # Player,
        result += f"{self.info['init_config']['mode']}," # Initial mode,
        result += f"{self.info['init_config']['param']}," # Initial parameter,
        result += f"{self.info['options']['verbose']}," # Verbose,
        result += f"{self.iterations['accumulated'][-1]}," # Accumulated gain,
        result += f"{self.iterations['accumulated'][-1] \
                    / abs(self.stats['init_potential'])*100:.2f}," # Absolute gain in %,
        result += f"{self.initial['potential']}," # Initial potential,
        result += f"{self.initial['potential'] \
                    + self.iterations['accumulated'][-1]}," # Final potential,
        result += f'{to_do},' # Initial Potential proportion,
        result += f'{to_do},' # Final Potential proportion,
        result += f"{self.initial['verts_in']/len(self.nodes)*100:.2f}," # Initial Verts proportion,
        result += f"{(self.initial['verts_in'] + self.iterations['verts_yes_diff'][-1]) \
                    / len(self.nodes)*100:.2f}," # Final Verts proportion,
        result += f"{self.initial['edges_in'] \
                    / (self.initial['edges_in'] \
                    + self.initial['edges_out']) * 100:.2f}," # Initial Edges proportion,
        result += f"{(self.initial['edges_in'] + self.iterations['edges_yes_diff'][-1]) \
                    / ((self.initial['edges_in'] + self.iterations['edges_yes_diff'][-1]) \
                    + (self.initial['edges_out'] + self.iterations['edges_no_diff'][-1])) \
                    * 100:.2f}," # Final Edges proportion,
        result += f'{to_do},' # Initial Edges-off proportion,
        result += f'{to_do},' # Final Edges-off proportion,
        result += f"{len(self.iterations['node_moved'])}," # Num of iterations,
        result += f'{to_do},' # Distance to converged state,
        result += f'{to_do},' # Eficiency to converged state
        result += f'{to_do},' # Distance to closest sink,
        result += f'{to_do},' # Eficiency to closest sink
        result += f'{to_do},' # Profit consults,
        result += f'{to_do},' # Moves per consult,
        result += f'{converged},' # Reach equilibrium,
        result += f'{found_unique_state},' # Found new sink,
        result += f"{self.results['import_duration']}," # Import duration,
        result += f"{self.results['run_duration']}," # Run duration,
        result += f"{self.results['export_duration']}," # Export duration,
        result += f"{total_duration}," # Total duration,
        result += f"{self.results['start_at']}," # Started at,
        result += f"{self.results['start_at'] + total_duration}," # Finished at,

        with open('experiments/results.csv', 'a') as fd:
            fd.write(result)


        # Initial Verts proportion,{}%\n\
        # Final Verts proportion,{verts_yes/len(self.nodes)*100:.2f}%\n\
        # Initial Edges proportion,{init_edges_yes/(init_edges_yes+init_edges_no)*100:.2f}%\n\
        # Final Edges proportion,{edges_yes/(edges_yes+edges_no)*100:.2f}%\n\
        # Initial Potential proportion,{self.local_potential(init_verts_yes, init_edges_yes)/self.stats['init_potential']*100:.2f}%\n\
        # Final Potential proportion,{self.local_potential(verts_yes, edges_yes)/(self.stats['init_potential']+accumulated)*100:.2f}%\n\
        # Profit consults,{consults}\n\
        # Performance,{len(self.hist['nodes_moved'])/consults*100:.2f}%\n\

    def append_sink(self):
        file = f"networks/sinks/{self.info['graph']}.csv"
        nodes = ''
        if not Path(file).is_file(): # create new file
            for node in self.nodes:
                nodes += f'{node},'
            with open(file, 'w+') as f:
                f.write(nodes[:-1])
            nodes = nodes[:-1].split(',')
        else:
            with open(file, newline='') as f:
                nodes = next(csv.reader(f))
        unique = True  # verify if state already exist
        with open(file, 'r') as f:
            table = csv.reader(f)
            next(table, None)
            for row in table:
                if unique:
                    for i, group in enumerate(row):
                        node_group = 1 if self.nodes[nodes[i]].group() else 0
                        if group != str(node_group):
                            break
                        elif i == len(row) - 1:
                            unique = False
                else:
                    break
        if unique:
            groups, inverse = '\n', '\n' # add its complement too
            for node in nodes:
                node_group = self.nodes[node].group()
                groups  += f'{1 if node_group else 0},'
                inverse += f'{0 if node_group else 1},'
            with open(file, 'a') as fd:
                fd.write(groups[:-1])
                fd.write(inverse[:-1])
        return unique

    def export_results(self):
        duration = datetime.now()
        game_path = self.define_game_path()
        converged = self.reach_equilibrium()
        found_unique_state = self.append_sink()
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
        data = {
            'moves' : self.iterations['node_moved'],
            'consults' : self.iterations['consults'],
            'profitables' : self.iterations['profitables'],
            'initial' : initial }
        with open(f'{game_path}.pickle', 'wb') as output:
            pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)
        self.append_result(converged, found_unique_state, game_path)
        print(f'Has Converged: {converged}')
        self.results['export_duration'] = datetime.now() - duration

    def print_parameters(self): # to-do: correct
        print('# Parameters:')
        print(f"- Graph:          {self.info['graph']}")
        print(f"- Alpha:          {self.info['alpha']}")
        print(f"- Initial Config: {self.info['init_config']}")
        print(f"- Verbose Mode:   {self.info['options']['verbose']}")
        print(f"- Export Results: {self.info['options']['export']}")
        print('Go!!!')

    def timestamp(self, node, increased):
        print(f'{datetime.now()} | move: {node} | increased: {increased:.2f}')

## Initiate Experiment #########################################################

# -------------------------------- Networks ------------------------------------
# sample_1 |   karate   |  football  |   human   |  1k  |
# sample_2 | terrorists | conference |    dag    | 10k  |
#  square  |  dolphins  | 2triangles | 2trig_mid | 100k |

if __name__ == '__main__':

    game = Game(network = 'terrorists',
                alpha = .95,
                init = { 'mode': 'r', 'param': 0.5 },
                ops = { 'verbose': False, 'export': True })
    game.play(sequential) # sequential, stochastic, greedy, pique_pega
