import json
from pathlib import Path
import csv
import time

def load_csv(file):
    nodes = {}
    def insert(a, b):
        if a not in nodes:
            nodes[a] = { # create a new node
                'friends'  : [b],
                'my_placar': {'us': 0, 'them': 0},
                'my_group' : 'no' }
        elif b not in nodes[a]['friends']:
            nodes[a]['friends'].append(b)
    with open(file, 'r') as f:
        table = csv.reader(f)
        for row in table:
            a = row[0]
            b = row[1]
            insert(a, b)
            insert(b, a)
    return nodes

def load(net = 'sample_2'):
    dict = f'networks/new_converted/{net}.txt'
    if Path(dict).is_file():
        return json.load(open(dict))
    else:
        nodes = load_csv(f'networks/{net}.csv')
        with open(dict, 'w') as f:
            json.dump(nodes, f)
        return nodes

def test_speed(net = 'sample_2', times = 10000):
    before = time.time()
    for _ in range(times):
        nodes = load_csv(f'networks/{net}.csv')
    print('CSV:', time.time() - before)
    before = time.time()
    for _ in range(times):
        nodes = json.load(open(f'networks/new_converted/{net}.txt'))
    print('DICT:', time.time() - before)

test_speed()
