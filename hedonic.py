########################## Parameters ##########################

# Datasets: sample_1  /  sample_2  / sample_3  / terrorists  / conference
dataset = "datasets/terrorists.csv"
limit   = 2
alpha   = 0.95
path    = "output/" + "greedy-global_potential/" # line: 215

########################## Converting ##########################

import csv

def insert(d, a, b):
    if a not in d:
        d[a] = [b]
    elif b not in d[a]:
        d[a].append(b)
    return d

def csv_2_dict(file):
    d = {}
    with open(file, 'r') as f:
        table = csv.reader(f)
        for row in table:
            a = int(row[0])
            b = int(row[1])
            d = insert(d, a, b)
            d = insert(d, b, a)
    return d

########################## Operations ##########################

from copy import deepcopy

def add(vert, original):
    g = deepcopy(original)
    g[vert] = []
    for key in g:
        if vert in network[key]:
            g[key].append(vert)
            g[vert].append(key)
    return g

def remove(vert, original):
    g = deepcopy(original)
    for v in g[vert]:
        g[v].remove(vert)
    g.pop(vert)
    return g

def move(verts, from_g, to_g):
    if type(verts) is list:
        for v in verts:
            from_g = remove(v, from_g)
            to_g   = add(v, to_g)
    else:
        from_g = remove(verts, from_g)
        to_g   = add(verts, to_g)
    return from_g, to_g

########################## Iterations ##########################

import os

def create_folder(path):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    steps = open(path + "iterations.csv","w+")
    steps.write("move, from, to, increased")
    return steps

def save_step(w):
    if w[0]:
        s = "\n{:02d}, {:02d}, {:02d}, {:.2f}".format(w[0], w[1], w[2], w[3])
        steps.write(s)

########################## Perspectives ##########################

# Individual
def hedonic(coalision, node, _):
    if node and not coalision.get(node):
        coalision = add(node, coalision)

    know   = len(coalision[node])
    unkown = len(coalision) - know - 1

    a = (1 - alpha) * know
    b =      alpha  * unkown
    return a - b

# Local
def potential(coalision, node, _):
    if node and not coalision.get(node):
        c = add(node, coalision)

    total = 0
    for key in coalision:
        total += hedonic(coalision, key, _)

    return total / 2

# Global
def global_potential(here, node, there):
    if not here.get(node) and there.get(node):
        there, here = move(node, there, here)

    return potential(here, None, None) + potential(there, None, None)

########################## Algorithms ##########################

import random

# Stochastic
def stochastic(c, p):

    def iteration():
        here = there = random.choice(c)

        if len(c) < limit:
            c.append({})

        while here == there:
            there = random.choice(c)
        node = random.choice(list(here))

        winner = [None, None, None, 0]
        increased = p(there, node, here) - p(here, node, there)
        if increased > winner[3]:
            winner = [node, c.index(here), c.index(there), increased]

        return winner

    i = 0
    while i < 100:
        n = iteration()
        save_step(n)
        if n[3] > 0:
            c[n[1]], c[n[2]] = move(n[0], c[n[1]], c[n[2]])
            i = 0
        else:
            i += 1
        while {} in c:
            c.remove({})

    return c

# Partial
def partial(c, p):

    def iteration():
        here = random.choice(c)
        node = random.choice(list(here))

        if len(c) < limit:
            c.append({})

        winner = [None, None, None, 0]
        for there in c:
            increased = p(there, node, here) - p(here, node, there)
            if increased > winner[3]:
                winner = [node, c.index(here), c.index(there), increased]

        return winner

    i = 0
    while i < 100:
        n = iteration()
        save_step(n)
        if n[3] > 0:
            c[n[1]], c[n[2]] = move(n[0], c[n[1]], c[n[2]])
            i = 0
        else:
            i += 1
        while {} in c:
            c.remove({})

    return c

# Greedy
def greedy(c, p):

    def iteration():
        if len(c) < limit:
            c.append({})

        winner = [None, None, None, 0]
        for here in c:
            for key in here:
                for there in c:
                    increased = p(there, key, here) - p(here, key, there)
                    if increased > winner[3]:
                        winner = [key, c.index(here), c.index(there), increased]

        return winner

    wanna_move = True
    while wanna_move:
        n = iteration()
        save_step(n)
        if n[3] > 0:
            c[n[1]], c[n[2]] = move(n[0], c[n[1]], c[n[2]])
        else:
            wanna_move = False
        while {} in c:
            c.remove({})

    return c

########################## Trainning ##########################

network    = csv_2_dict(dataset)
coalitions = [network]

from datetime import datetime
begin = datetime.now()
steps = create_folder(path)

coalitions = greedy(coalitions, global_potential)
#   Algorithms:  greedy |  partial  |    stochastic
# Perspectives: hedonic | potential | global_potential

########################## Export ##########################

# Infos
done = datetime.now()
p = open(path + "_info.txt","w+")
p.write("Alpha        -> %s\r\n" % alpha)
p.write("Dataset      -> %s\r\n" % dataset)
p.write("Max Clusters -> %s\r\n" % limit)
p.write("Converged in -> %s\r\n" % str(done - begin))
p.close()
steps.close()

# Graphs
g = open(path + "graph.txt","w+")
g.write("%s\r\n" % network)
g.close()

i = 0
for c in coalitions:
    title = 'coalision_' + str(i)
    f = open(path + title + ".txt","w+")
    f.write("%s\r\n" % c)
    f.close()
    i += 1
