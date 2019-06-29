# Hedonic Games for Clustering
# A research experiment of Giovanni^, Kostya^, Daniel`, Lucas` (ack Eduardo`)
# ^INRIA ∣ `Federal University of Rio de Janeiro
# June 2019

## Parameters ##################################################################

# ----------- Networks -----------
# sample_1 |   karate   | football
# sample_2 | terrorists | conference
# sample_3 |  dolphins  | simple

p = {
    'graph' : 'conference', # Choose one above
    'alpha' : 0.95, # Fragmentation Factor | 0 <= alpha <= 1
    'init'  : {
        'mode': 'r', # Initial Modes:
            # - Random (r): Nodes will be randomly selected to start inside cluster
            # - Select (s): Chose which nodes will start inside cluster
            # - Any other: Start with an empty cluster
        'params': 0.5 },
            # - Random (r): Number of nodes - If is between 0 and 1 will be multiply by quantity of nodes
            # - Select (s): List of selected nodes - [node indice, ..., node indice]

    'print' : True, # Print each iteration in the terminal?
    'freq'  : 1,    # Probability that Accuracy will be computed at each iteration
    'note'  : 'None' } # Free space to comment about the experiment

## Import Dependecies ##########################################################

from datetime import datetime
from pathlib import Path
import random
import json
import csv
import os

## A Hedonic Game ##############################################################

class Game:

    def __init__(self, p):
        self.graph     = load_network('networks/'+p['graph']+'.csv')
        self.folder    = set_game_path(p['graph'])
        self.g_truth   = set_ground_truth('networks/ground_truth/'+p['graph']+'.csv', self.graph)
        self.classes   = set_classes(p['init'], self.graph.keys())
        self.clusters  = count_verts_edges(self.classes, self.graph)
        self.potential = global_potential(self.clusters)
        self.accuracy  = set_accuracy(self.g_truth.items(), self.clusters.keys())
        self.iteration = 0
        self.score     = 0

    def start(self):
        self.begin = datetime.now()
        print_parameters()
        done = False
        while done is False:
            nodes = list(game.graph)
            find = False
            while find is False:
                i = random.randrange(0, len(nodes))
                p = profit(nodes[i])
                if p > 0:
                    find = True
                    move(nodes[i], p)
                else:
                    nodes[i], nodes[-1] = nodes[-1], nodes[i]
                    nodes.pop()
                    if len(nodes) == 0:
                        done = find = True
                        print('Done!')

## Load the Network ############################################################

def load_network(file):
    dict = 'networks/converted/' + p['graph'] + '.txt'
    if Path(dict).is_file():
        return json.load(open(dict))
    else:
        graph = csv2dict(file)
        with open(dict, 'w') as f:
            json.dump(graph, f)
        return graph

def csv2dict(file):
    d = {}
    def insert(d, a, b):
        if a not in d:
            d[a] = [b]
        elif b not in d[a]:
            d[a].append(b)
        return d
    with open(file, 'r') as f:
        table = csv.reader(f)
        for row in table:
            a = row[0]
            b = row[1]
            d = insert(d, a, b)
            d = insert(d, b, a)
    return d

## Setters #####################################################################

def set_game_path(graph):
    now = str(datetime.now())[2:-7]
    for _ in ': -': now = now.replace(_, '')
    return 'experiments/' + graph + '_' + now  + '/'

def set_ground_truth(file, graph):
    g_t = {}
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
        for node in graph.keys():
            g_t[node] = 'none'
    return g_t

def set_classes(param, nodes):
    c = {}
    for n in nodes: c[n] = 'out'
    if param['mode'].lower() == 'r':
        amount = param['params']
        if type(amount) is not int and type(amount) is not float:
            print('2nd parameter of `Random` is wrong.')
        if amount < 0: amount *= -1
        if amount > 0 and amount < 1: amount *= len(nodes)
        if amount > len(nodes): amount = len(nodes)
        amount = int(amount)
        remain = list(c)
        while amount > 0:
            r = random.randrange(0, len(remain))
            c[remain[r]] = 'in'
            remain[r], remain[-1] = remain[-1], remain[r]
            remain.pop()
            amount -= 1
    if param['mode'].lower() == 's':
        for node in param['params']:
            if c[node] == 'out':
                c[node] = 'in'
    return c

def set_accuracy(g_truth, clusters):
    accuracy = {}
    g_truths = {}
    for node, cluster in g_truth:
        g_truths[cluster] = None
    g_truths = g_truths.keys()
    for c in clusters:
        for gt in g_truths:
            s = '{} x {}'.format(c, gt)
            accuracy[s] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    return accuracy

## Counting ####################################################################

def check_friends(node):
    us = them = 0
    for friend in game.graph[node]:
        if game.classes[str(friend)] == game.classes[node]:
            us += 1
        else: them += 1
    return us, them

def count_verts_edges(Class, graph):
    cluster  = {'verts': 0, 'edges': 0}
    remainer = {'verts': 0, 'edges': 0}
    def increment(coalition, node):
        coalition['verts'] += 1
        for friend in graph[node]:
            if Class[str(friend)] == Class[node]: coalition['edges'] += 1
        return coalition
    for node in Class:
        if Class[node] == 'in':  cluster  = increment(cluster, node)
        if Class[node] == 'out': remainer = increment(remainer, node)
    cluster['edges']  = int(cluster['edges'] / 2)
    remainer['edges'] = int(remainer['edges'] / 2)
    return {'in': cluster, 'out': remainer}

## Hedonic Functions ###########################################################

def hedonic(neighbors, strangers, alpha=p['alpha']):
    return (1 - alpha) * neighbors - alpha * strangers

def global_potential(clusters, alpha=p['alpha']):
    pot = 0
    for name, c in clusters.items():
        pot += c['edges'] - alpha * c['verts'] * (c['verts'] - 1) / 2
    p['inital potential'] = pot
    return pot

def profit(node, us=None, them=None, alpha=p['alpha']):
    if us == None: us, them = check_friends(node)
    f, t = 'in', 'out' # from & to
    if game.classes[node] == 'out': f, t = 'out', 'in' # invert if is from other cluster
    us   = hedonic(us,   game.clusters[f]['verts'] - us - 1)
    them = hedonic(them, game.clusters[t]['verts'] - them)
    return them - us

## Accuracy ####################################################################

def accuracy():
    clusters = game.clusters.keys()
    g_truths = {}
    for node, cluster in game.g_truth.items():
        g_truths[cluster] = None
    g_truths = g_truths.keys()
    acc = {}
    for c in clusters:
        for gt in g_truths:
            tp = tn = fp = fn = 0
            for node, Class in game.classes.items():
                if Class == c and game.g_truth[node] == gt:
                    tp += 1
                    continue
                if Class != c and game.g_truth[node] != gt:
                    tn += 1
                    continue
                if Class == c and game.g_truth[node] != gt:
                    fp += 1
                    continue
                if Class != c and game.g_truth[node] == gt:
                    fn += 1
                    continue
            s = '{} x {}'.format(c, gt)
            acc[s] = {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    return acc

## Move a Node #################################################################

def move(node, increased=None):
    past_friends, future_friend = check_friends(node)
    if increased == None:
        increased = profit(node, past_friends, future_friend)
    game.clusters[game.classes[node]]['verts'] -= 1
    game.clusters[game.classes[node]]['edges'] -= past_friends
    if game.classes[node] == 'out': game.classes[node] = 'in'
    else: game.classes[node] = 'out'
    game.clusters[game.classes[node]]['verts'] += 1
    game.clusters[game.classes[node]]['edges'] += future_friend
    game.score     += increased
    game.potential += increased
    game.iteration += 1
    if random.random() <= p['freq']:
        game.accuracy = accuracy()
    timestamp(node, increased)

## Export Results ##############################################################

files = {
    'iters' : None,
    'props' : None,
    'accur' : None,
    'state' : None }

def create_files(path, cluster_combinations):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    files['iters'] = open(path + 'iterations.csv', 'w+')
    files['props'] = open(path + 'properties.csv', 'w+')
    files['state'] = open(path + 'states.csv',    'w+')
    files['accur'] = open(path + 'accuracy.csv',   'w+')
    files['props'].write('Verts In,Edges In,Verts Out,Edges Out')
    files['iters'].write('Node Moved,From Cluster,To Cluster,Instantaneous Gain,Accumulated Gain,Potential')
    c = ''
    for cluster in game.accuracy.keys():
        c += '{},,,,'.format(cluster)
    files['accur'].write(c[:-1])
    s = 'True Positive,True Negative,False Positive,False Negative,' * cluster_combinations
    files['accur'].write('\n' + s[:-1])
    files['state'].write(stringify_nodes())
    files['state'].write(stringify_state())

def stringify_nodes():
    nodes = ''
    for n in game.classes.keys():
        nodes += n + ','
    return nodes[:-1]

def stringify_state():
    s = '\n'
    for node, cluster in game.classes.items():
        s += cluster + ','
    return s[:-1]

def print_parameters():
    print('# Parameters:')
    print('- Graph:                 {}'.format(p['graph']))
    print('- Alpha:                 {}'.format(p['alpha']))
    print('- Initial Configuration: {}'.format(p['init']))
    print('- Verbose Mode:          {}'.format(p['print']))
    print('- Accuracy Frequency:    {}'.format(p['freq']))
    print('Go!!!'.format(p['freq']))

def timestamp(node, increased):
    f, t = 'out', 'in'
    if game.classes[node] == 'out': f, t = 'in', 'out'
    files['props'].write('\n{},{},{},{}'.format(
        game.clusters['in']['verts'],  game.clusters['in']['edges'],
        game.clusters['out']['verts'], game.clusters['out']['edges']))
    files['iters'].write('\n{},{},{},{:.2f},{:.2f},{:.2f}'.format(
        node, f, t, increased, game.score, game.potential))
    s = '\n'
    for pairs in game.accuracy.keys():
        s += '{},{},{},{},'.format(
            game.accuracy[pairs]['TP'], game.accuracy[pairs]['TN'],
            game.accuracy[pairs]['FP'], game.accuracy[pairs]['FN'])
    files['accur'].write(s[:-1])
    files['state'].write(stringify_state())
    if p['print']:
        print('move: {} | from: {} | to: {} | increased: {:.2f}'.format(
            node, f, t, increased))

def results(duration):
    info = open(game.folder + 'infos.md', 'w+')
    info.write('# Parameters\n')
    info.write('- Graph:                  {}\n'.format(p['graph']))
    info.write('- Alpha:                  {}\n'.format(p['alpha']))
    info.write('- ¹Initial Configuration: {}\n'.format(p['init']))
    info.write('- Verbose Mode:           {}\n'.format(p['print']))
    info.write('- Accuracy Frequency:     {}\n'.format(p['freq']))
    info.write('- commentaries:           {}\n'.format(p['note']))
    info.write('\n# Results\n')
    info.write('- Finished at:            {}\n'.format(str(datetime.now())[:-7]))
    info.write('- Converged in:           {}\n'.format(duration))
    info.write('- Initial Potential:      {:.2f}\n'.format(p['inital potential']))
    info.write('- Final Potential:        {:.2f}\n'.format(game.potential))
    info.write('- Accumulated Gain:       {:.2f}\n'.format(game.score))
    info.write('- ²Potential Gain in %:   {:.2f}%\n'.format(game.score / abs(game.potential) * 100))
    info.write('- Proportion Cluster In:  {:.2f}%\n'.format(game.clusters['in']['verts'] / len(game.graph) * 100))
    info.write('- Iterations:             {}\n'.format(game.iteration))
    info.write('\n# Legend\n')
    info.write('- ¹Initial Configuration\n')
    info.write('  - Modes Configurations:\n')
    info.write('    - Random (r): Nodes will be randomly selected to start inside cluster\n')
    info.write('    - Select (s): Chose which nodes will start inside cluster\n')
    info.write('    - Any other:  Start with an empty cluster\n')
    info.write('  - Modes Parameters:\n')
    info.write('    - Random (r): Number of nodes - If it is between 0 and 1 will be multiply by the number of nodes\n')
    info.write('    - Select (s): List of selected nodes. e.g. [node indice, ..., node indice]\n')
    info.write('- ²Potential Gain in %\n')
    info.write('  - Accumulated Gain / Initial Potential * 100\n')
    info.close()

    files['iters'].close()
    files['props'].close()
    files['accur'].close()
    files['state'].close()

## Initiate Experiment #########################################################

game = Game(p)
create_files(game.folder, len(game.accuracy.keys()))
game.start()
results(datetime.now() - game.begin)
