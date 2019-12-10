# Clustering with Hedonic Games
# Detecting communities in networks with cooperative game theory
# A research experiment in colaboration between *UFRJ and ^INRIA
# *Lucas Lopes, *Daniel Sadoc, ^Kostya and ^Giovanni
# October 2019

## Import Dependecies ##########################################################

from players  import sequential, stochastic, greedy, pique_pega
from datetime import datetime, timedelta
from random import random
from pathlib  import Path
import pickle
import csv
import os

# from copy import deepcopy
# import json

## Node ########################################################################

class Node:

    def __init__(self, friend, initial_group = False, ground_truth = None):
        self.friends = { friend : True } # weight avaiable
        self.set_with_me(0)
        self.set_initial_group(initial_group)
        self.set_ground_truth(ground_truth)

    def add(self, friend):
        self.friends[friend] = True

    def group(self):
        if len(self.moved_at) % 2 == 0: return self.initial_group
        else: return not self.initial_group

    def set_with_me(self, quantity):
        self.with_me = quantity

    def set_ground_truth(self, name):
        self.ground_truth = name

    def set_initial_group(self, initial_group):
        self.initial_group = initial_group
        self.moved_at = []

    def get_gt(self):
        return self.ground_truth

    def update_with_me(self, what_happened):
        if what_happened == 'join' : self.with_me += 1
        if what_happened == 'lost' : self.with_me -= 1
        if what_happened == 'flip' : self.with_me = len(self.friends) - self.with_me

    # def binary_search(arr, element):
    #     margin_left, margin_right = 0, len(arr) - 1
    #     while (margin_left <= margin_right):
    #         pointer = int((margin_left + margin_right) / 2)
    #         if element > arr[pointer]:
    #             margin_left = pointer + 1
    #         elif element < arr[pointer]:
    #             margin_right = pointer - 1
    #         elif element == arr[pointer]:
    #             return pointer
    #     if arr[pointer] > element:
    #         return pointer - 1
    #     else:
    #         return pointer

## A Hedonic Game ##############################################################

class Game:

    def __init__(self, network='dag', alpha=.95, init_mode='s', init_param=[],
                 verbose=True, export=True, gt_col=1):

        self.infos = { 'verbose': verbose, 'export': export, 'player': None }
        self.set_alpha(alpha)
        self.load_network(network, init_mode, init_param, gt_col)

    ## Load the Network ########################################################

    def load_network(self, network, init_mode='s', init_param=[], gt_col=1):
        self.nodes, self.stats, duration = {}, {}, datetime.now()
        self.infos['network'] = network
        file = f'networks/csv/{network}.csv'
        with open(file, 'r') as f:
            table = csv.reader(f)
            for row in table:
                a = row[0]
                b = row[1]
                if self.nodes.get(a): self.nodes[a].add(b) # add new friend
                else: self.nodes[a] = Node(b) # create a new node
                if self.nodes.get(b): self.nodes[b].add(a)
                else: self.nodes[b] = Node(a)
        self.import_ground_truth(file.replace('csv','ground_truth'), gt_col)
        skip = True if init_mode == 's' and len(init_param) == 0 else False
        self.set_initial_state(init_mode, init_param, skip)
        self.stats['import_duration'] = datetime.now() - duration
        # with open('CONFERENCE_NODES.pickle', 'wb') as output:
        #     pickle.dump(nodes, output, pickle.HIGHEST_PROTOCOL)

    def import_ground_truth(self, file, gt_col=1):
        if Path(file).is_file():
            with open(file, 'r') as f:
                table = csv.reader(f)
                for row in table:
                    node = row[0]
                    g_t  = row[gt_col] # multiple g_truths avaiable
                    self.nodes[node].set_ground_truth(g_t)

    ## Setters #################################################################

    def set_alpha(self, alpha):
        self.infos['alpha'] = max(0, min(1, alpha))

    def set_initial_state(self, mode='s', param=[], skip=False):
        if mode.lower() == 's' and not skip:
            for node in self.nodes:
                self.nodes[node].set_initial_group(False)
            for node in param:
                self.nodes[node].set_initial_group(True)
        if mode.lower() == 'r':
            for node in self.nodes:
                if random() <= param: self.nodes[node].set_initial_group(True)
                else: self.nodes[node].set_initial_group(False)
        self.infos['init_mode']  = mode
        self.infos['init_param'] = len(param) if mode == 's' else param
        self.reset_history()

    ## Helpers #################################################################

    def reset_history(self):
        verts_edges = self.count_verts_edges()
        self.hist = {
            'nodes_moved' : [],
            'moved_to'    : [],
            'consults'    : [],
            'profitables' : [],
            'accumulated' : [0],
            'verts_yes'   : [verts_edges[0]],
            'edges_yes'   : [verts_edges[1]],
            'edges_no'    : [verts_edges[2]] }
        self.stats = {
            'start_at'        : datetime.now(),
            'import_duration' : timedelta(0),
            'run_duration'    : timedelta(0),
            'export_duration' : timedelta(0),
            'potential'       : self.global_potential(verts_edges[0],verts_edges[1],verts_edges[2]),
            'iteration'       : 0,
            'profit_consults' : 0,
            'found_profitable': 0 }

    def separate_nodes(self):
        yes, no = [], []
        for key, node in self.nodes.items():
            if node.group():
                yes.append(node)
            else:
                no.append(node)
        return [yes, no]

    ## Computations ############################################################

    def check_friends(self, node):
        with_me = 0
        for friend in self.nodes[node].friends:
            if self.nodes[node].group() == self.nodes[friend].group():
                with_me += 1
        self.nodes[node].set_with_me(with_me)
        return with_me

    def count_verts_edges(self):
        verts_yes, total_edges, edges_yes, edges_no = 0, 0, 0, 0
        for node in self.nodes:
            total_edges += len(self.nodes[node].friends)
            friends_with_node = self.check_friends(node)
            if self.nodes[node].group():
                verts_yes += 1
                edges_yes += friends_with_node
            else: edges_no += friends_with_node
        self.infos['verts'] = len(self.nodes)
        self.infos['edges'] = int(total_edges/2)
        return verts_yes, int(edges_yes/2), int(edges_no/2)

    def calc_accuracy(self): #> call function and timestamp at each iteration
        g_truths, acc = {}, {}
        for key, node in self.nodes.items():
            g_truths[node.ground_truth] = None
        g_truths = g_truths.keys() # All clusters of Ground Truth
        for c in ['Yes', 'No']: # Name of Game's clusters
            for gt in g_truths:
                tp = tn = fp = fn = 0
                for key, node in self.nodes.items():
                    if node.group() == c and node.get_gt() == gt:
                        tp += 1
                        continue
                    if node.group() != c and node.get_gt() != gt:
                        tn += 1
                        continue
                    if node.group() == c and node.get_gt() != gt:
                        fp += 1
                        continue
                    if node.group() != c and node.get_gt() == gt:
                        fn += 1
                        continue
                s = f'{c} x {gt}'
                acc[s] = {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
        self.accuracy = acc

    ## Hedonic Functions #######################################################

    def hedonic(self, have, total, alpha):
        return have - total * alpha

    def local_potential(self, verts, edges):
        max_edges_possible = verts * (verts - 1) / 2
        return self.hedonic(edges, max_edges_possible, self.infos['alpha'])

    def global_potential(self, verts_yes, edges_yes, edges_no, sum=True):
        yes = self.local_potential(verts_yes, edges_yes)
        no  = self.local_potential(len(self.nodes) - verts_yes, edges_no)
        if sum: return yes + no
        else:   return yes,  no

    ## Operations ##############################################################

    def play(self, player):
        duration = datetime.now()
        self.infos['player'] = player.__name__
        self.show_info()
        print(f'\nGo!!! {datetime.now()}\n')
        player(self)
        print(f'\nDone!!! {datetime.now() - duration}')
        self.stats['run_duration'] = datetime.now() - duration
        self.end_game()

    def replay(self, game):
        duration = datetime.now()
        with open(f'experiments/{game}', "rb") as f:
            game = pickle.load(f)
        # print(game['nodes_moved'])
        self.load_network(game['network'], init_param=game['nodes_inside'])
        self.set_alpha(game['alpha'])
        self.infos['verbose'] = False
        for node in game['nodes_moved']:
            self.move(str(node))
        self.hist['consults']    = game['consults']
        self.hist['profitables'] = game['profitables']
        print(f'replay done in {datetime.now() - duration}')
        print(f'reach equilibrium: {self.reach_equilibrium()}')

    def reach_equilibrium(self, inspec=False):
        for node in self.nodes:
            if self.profit(node) > 0:
                return node if inspec else False
        return True

    def communicate_friends(self, node):
        self.nodes[node].update_with_me('flip')
        for friend in self.nodes[node].friends:
            if self.nodes[node].group() == self.nodes[friend].group():
                self.nodes[friend].update_with_me('join') # += 1
            else:
                self.nodes[friend].update_with_me('lost') # -= 1

    def profit(self, node):
        us   = self.nodes[node].with_me
        them = len(self.nodes[node].friends) - us
        all_yes = self.hist['verts_yes'][-1]
        all_no  = len(self.nodes) - all_yes
        if self.nodes[node].group(): all_here, all_there = all_yes, all_no
        else: all_here, all_there = all_no, all_yes
        here  = self.hedonic(us, all_here - 1, self.infos['alpha']) # excluding me
        there = self.hedonic(them, all_there, self.infos['alpha'])
        self.stats['profit_consults'] += 1
        if there - here > 0: self.stats['found_profitable'] += 1
        return there - here

    def move(self, node):
        self.stats['iteration'] += 1
        self.hist['nodes_moved'].append(node)
        self.hist['consults'].append(self.stats['profit_consults'])
        self.hist['profitables'].append(self.stats['found_profitable'])
        increased = self.profit(node)
        self.stats['profit_consults'] = 0
        self.stats['found_profitable'] = 0
        self.hist['moved_to'].append(self.nodes[node].group())
        self.nodes[node].moved_at.append(self.stats['iteration'])
        self.communicate_friends(node)
        if self.nodes[node].group():
            self.hist['verts_yes'].append(self.hist['verts_yes'][-1] + 1)
            self.hist['edges_yes'].append(self.hist['edges_yes'][-1] + self.nodes[node].with_me)
            self.hist['edges_no'].append(self.hist['edges_no'][-1] - len(self.nodes[node].friends) + self.nodes[node].with_me)
        else:
            self.hist['verts_yes'].append(self.hist['verts_yes'][-1] - 1)
            self.hist['edges_yes'].append(self.hist['edges_yes'][-1] - len(self.nodes[node].friends) + self.nodes[node].with_me)
            self.hist['edges_no'].append(self.hist['edges_no'][-1] + self.nodes[node].with_me)
        self.hist['accumulated'].append(self.hist['accumulated'][-1] + increased)
        if self.infos['verbose']: self.timestamp(node, increased)

    ## Functions ###############################################################

    def get_all_states(self, nodes): # receive list of nodes
    	states = []
    	for k in range(1, int(2 ** (len(nodes) - 1))):
    		num = list(bin(k))[2:]
    		num.reverse()
    		positions = []
    		for pos, value in enumerate(num):
    			if int(value) == 1:
    				positions.append(nodes[pos])
    		states.append(positions)
    	return states

    def get_stable_states(self):
        pass

    def bruteforce_stable_states(self, network, alpha=.95):
        duration = datetime.now()
        stable_states = []
        game = Game(network, alpha)
        nodes = list(game.nodes)
        combinations = get_all_states(nodes)
        for comb in combinations:
            game.set_initial_state(param=comb)
            if game.reach_equilibrium():
                stable_states.append(comb)
        print(f'Found stable states of {network} in {datetime.now() - duration}')
        return stable_states

    def check_dist(self, inside, reference, length=False): # reference = [[...],[...]]
        corrects = [node for node in inside if node in reference]
        distance = len(inside) + len(reference) - 2 * len(corrects)
        if length: return distance
        else: return [node for node in inside+reference if node not in corrects]

    def find_short_dist(self, inside, states, length=False): # states = [ [[...],[...]], [[],[]] ]
        short_dist, need_move = float('inf'), []
        for clusters in states:
            nodes2move = check_dist(inside, clusters)
            if len(nodes2move) < short_dist:
                short_dist = len(nodes2move)
                need_move  = nodes2move
        if length: return distance
        else: return need_move

    def show_distances(self, graph, gt):
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

    ## Export ##################################################################

    def timestamp(self, node, increased):
        print(f'{datetime.now()} | move: {node} | to: {self.nodes[node].group()} | increased: {increased:.2f}')

    def define_game_nomenclature(self):
        init_verts_propor  = int(self.hist['verts_yes'][0]  / len(self.nodes) * 100)
        final_verts_propor = int(self.hist['verts_yes'][-1] / len(self.nodes) * 100)
        init_edges_propor  = int(self.hist['edges_yes'][0]  / self.infos['edges'] * 100)
        final_edges_propor = int(self.hist['edges_yes'][-1] / self.infos['edges'] * 100)
        infos = [
            self.infos['network'][:3].upper(), # First 3 letters of network name
            round(self.infos['alpha'], 2), # Value of Alpha (with 2 decimal plates)
            self.infos['player'][:3].upper(), # First 3 letters of player/strategy name
            f"{self.infos['init_mode'].lower()}{self.infos['init_param']}", # Initial configurations
            f"i{len(self.hist['nodes_moved'])}", # Number of Iterations
            f"v{init_verts_propor}.{final_verts_propor}", # Initial and Final Vertices proportion
            f"e{init_edges_propor}.{final_edges_propor}",
            f"abs{round(self.hist['accumulated'][-1] / abs(self.stats['potential']) * 100, 2)}" ] # Initial and Final Edges proportion
        nomenclature = ''
        for info in infos: nomenclature += f'{info}_'
        return nomenclature[:-1]

    def show_info(self):
        print('# Network:')
        print(f"- Network:    {self.infos['network']}")
        print(f"- Verts:      {self.infos['edges']}")
        print(f"- Edges:      {self.infos['verts']}")
        print('# Parameters:')
        print(f"- Alpha:      {self.infos['alpha']}")
        print(f"- Player:     {self.infos['player']}")
        print(f"- Init Mode:  {self.infos['init_mode']}")
        print(f"- Init Param: {self.infos['init_param']}")
        print(f"- Verbose:    {self.infos['verbose']}")
        print(f"- Export:     {self.infos['export']}")

    def show_networks(get=False):
        path = './networks/csv/'
        networks = [n for n in os.listdir(path) if n.endswith('.csv')]
        if get:
            return [n[:-4] for n in networks]
        else:
            for n in networks: print(f'- {n[:-4]}')

    def append_result(self, game_nomenclature, converged, found_unique_state):
        total_duration = self.stats['import_duration'] + self.stats['run_duration'] + self.stats['export_duration']
        total_consults = sum(self.hist['consults'])
        nodes_on_initial = [node for node in self.nodes if self.nodes[node].initial_group]
        cols = [
            game_nomenclature, # Nomenclature
            self.infos['network'], # Network
            self.infos['alpha'], # Alpha
            self.infos['player'], # Player
            self.infos['init_mode'], # Initial mode
            self.infos['init_param'], # Initial parameter
            round(self.hist['accumulated'][-1], 2), # Accumulated gain
            round(self.hist['accumulated'][-1] / abs(self.stats['potential']) * 100, 2), # Absolute gain in %
            round(self.stats['potential'], 2), # Initial potential
            round(self.stats['potential'] + self.hist['accumulated'][-1], 2), # Final potential
            round(self.local_potential(self.hist['verts_yes'][0],  self.hist['edges_yes'][0])  / self.stats['potential'] * 100, 2), # Initial Potential proportion
            round(self.local_potential(self.hist['verts_yes'][-1], self.hist['edges_yes'][-1]) / self.stats['potential'] * 100, 2), # Final   Potential proportion
            round(self.hist['verts_yes'][0]  / len(self.nodes) * 100, 2), # Initial Verts proportion
            round(self.hist['verts_yes'][-1] / len(self.nodes) * 100, 2), # Final   Verts proportion
            round(self.hist['edges_yes'][0]  / self.infos['edges'] * 100, 2), # Initial Edges proportion
            round(self.hist['edges_yes'][-1] / self.infos['edges'] * 100, 2), # Final   Edges proportion
            round((self.infos['edges'] - self.hist['edges_yes'][0]  - self.hist['edges_no'][0])  / self.infos['edges'] * 100, 2), # Initial Edges-off proportion
            round((self.infos['edges'] - self.hist['edges_yes'][-1] - self.hist['edges_no'][-1]) / self.infos['edges'] * 100, 2), # Final   Edges-off proportion
            len(self.hist['nodes_moved']), # Num of iterations
            'TO_DO: dist(init-end)/num_moves', # self.check_dist(nodes_on_initial, self.separate_nodes(), length=True) / len(self.hist['nodes_moved']), #> Eficiency to converged state
            'TO_DO: dist(init-closest_sink)/num_moves', # self.find_short_dist(nodes_on_initial, self.get_stable_states(), length=True) / len(self.hist['nodes_moved']), #> Eficiency to closest sink -> num de movimentos / dist(init,end)
            total_consults, # Profit consults
            0 if total_consults == 0 else round(sum(self.hist['profitables']) / total_consults, 2), # Profitables per consult
            0 if total_consults == 0 else round(len(self.hist['nodes_moved']) / total_consults, 2), # Moves per consult
            str(converged).lower(), # Reach equilibrium
            str(found_unique_state).lower(), # Found new sink
            str(self.infos['verbose']).lower(), # Verbose
            str(self.infos['export']).lower(), # Export
            self.stats['import_duration'], # Import duration
            self.stats['run_duration'], # Run duration
            self.stats['export_duration'], # Export duration
            total_duration, # Total duration
            self.stats['start_at'], # Started at
            self.stats['start_at'] + total_duration ] # Finished at
        result = '\n'
        for col in cols:
            result += f'{col},'
        with open('experiments/results.csv', 'a') as fd:
            fd.write(result[:-1])

    def append_sink(self):
        file = f"networks/sinks/{self.infos['network']}.csv"
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
                else: break
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

    def end_game(self):
        game_nomenclature = self.define_game_nomenclature()
        converged = self.reach_equilibrium()
        found_unique_state = self.append_sink()
        if self.infos['export']: self.export_game(game_nomenclature)
        self.append_result(game_nomenclature, converged, found_unique_state)
        print(f'Has Converged: {converged}\n')

    def export_game(self, game_nomenclature):
        duration = datetime.now()
        nodes_inside = []
        verts_yes = self.hist['verts_yes'][0]
        verts_no  = len(self.nodes) - verts_yes
        smallest = True if verts_yes < verts_no else False
        for node, value in self.nodes.items():
            if value.initial_group == smallest:
                nodes_inside.append(node)
        game = {
            'network'      : self.infos['network'],
            'nodes_inside' : nodes_inside,
            'alpha'        : self.infos['alpha'],
            'nodes_moved'  : self.hist['nodes_moved'],
            'consults'     : self.hist['consults'],
            'profitables'  : self.hist['profitables']}
        with open(f'experiments/{game_nomenclature}.pickle', 'wb') as output:
            pickle.dump(game, output, pickle.HIGHEST_PROTOCOL)
        self.stats['export_duration'] = datetime.now() - duration

## Do an Experiment ############################################################

# -------------------------------- Networks ------------------------------------
# sample_1 |   karate   |  football  |   human   |  1k  |
# sample_2 | terrorists | conference |    dag    |  10k |
#  square  |  dolphins  | 2triangles | 2trig_mid | 100k |

if __name__ == '__main__':

    # kwargs = {"arg3": 3, "arg2": "two","arg1":5}
    # test_args_kwargs(**kwargs)

    # game = Game(network = 'terrorists', # choose one above
    #             alpha   = .95,    # 0 > alpha < 1
    #             verbose = False,   # if True will print each move
    #             export  = True,   # if True will export the game history for future replay
    #             init_mode  = 'r', # 'r' for random classification and 's' for select nodes
    #             init_param = .5)  # if 'r' is between 0 and 1; elif 's' is a list of str(nodes)
    # game.play(sequential) # sequential, stochastic, greedy, pique_pega

    networks = Game.show_networks(get=True)
    players  = [sequential, stochastic, greedy, pique_pega]
    game = Game(verbose=False) # network='dag', alpha=.95, init_mode='s', init_param=[],verbose=True, export=True, gt_col=1
    for net in networks:
        game.load_network(net) # init_mode='r', init_param=.5, gt_col=1
        for player in players:
            game.set_initial_state('r', .5) # mode='s', param=[], skip=False
            game.play(player)

    # set_alpha(self, alpha)
    # replay(self, game)

    # get_all_states(self, nodes)
    # bruteforce_stable_states(self, network, alpha=.95)
    # check_dist(self, inside, reference, length=False)
    # find_short_dist(self, inside, states, length=False)
    # show_distances(self, graph, gt)
