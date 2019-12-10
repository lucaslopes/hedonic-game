from datetime import datetime
import csv

################################################################################

class Node:
    def __init__(self, friend):
        self.friends = { friend : True }
    def add(self, friend):
        self.friends[friend] = True

################################################################################

def load_network(file): # 52s
    nodes = {}
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
    return nodes

def load_network_dict(file): # 32s
    nodes = {}
    def insert(a, b):
        if nodes.get(a): nodes[a]['friends'][f'{b}'] = True
        else: nodes[a] = { 'friends' : {f'{b}':True} } # create a new node
    with open(file, 'r') as f:
        table = csv.reader(f)
        for row in table:
            a = row[0]
            b = row[1]
            insert(a, b)
            insert(b, a)
    return nodes

def load_network_obj(file): # 42s
    nodes = {}
    def insert(a, b):
        if nodes.get(a): nodes[a].add(b) # add new friend
        else: nodes[a] = Node(b) # create a new node
    with open(file, 'r') as f:
        table = csv.reader(f)
        for row in table:
            a = row[0]
            b = row[1]
            insert(a, b)
            insert(b, a)
    return nodes

################################################################################

duration = datetime.now()
for _ in range(1000):
    nodes = load_network_dict('./networks/1k.csv')
print(datetime.now() - duration)
